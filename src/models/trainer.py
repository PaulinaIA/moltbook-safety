"""H2O AutoML trainer for karma prediction."""

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pickle

import polars as pl

from config.settings import settings
from src.processing.gold import get_modeling_data

logger = logging.getLogger(__name__)

# Feature columns for modeling (excluding id, name, and target)
FEATURE_COLUMNS = [
    "followers",
    "following",
    "follower_ratio",
    "description_length",
    "post_count",
    "total_post_rating",
    "avg_post_rating",
    "max_post_rating",
    "avg_title_length",
    "avg_post_desc_length",
    "comment_count",
    "avg_comment_length",
    "total_activity",
    "total_rating",
]


def clean_training_data(data: pl.DataFrame, target: str = "karma") -> pl.DataFrame:
    """Clean data before training: remove outliers/bots, apply log transform.

    Cleaning steps:
    1. Remove users with negative karma
    2. Remove ghost users (0 posts AND 0 comments) — bots with inflated karma
    3. Cap extreme outliers at 99th percentile
    4. Apply log1p transform for skewed distribution

    Args:
        data: Raw user features DataFrame
        target: Target column name

    Returns:
        Cleaned DataFrame with log-transformed target
    """
    n_before = len(data)

    # 1. Remove negative karma (corrupted data)
    data = data.filter(pl.col(target) >= 0)
    logger.info("After removing negative karma: %d rows", len(data))

    # 2. Remove ghost/bot users: 0 posts AND 0 comments
    # These users have no real activity — their karma is artificial
    data = data.filter(
        (pl.col("post_count") > 0) | (pl.col("comment_count") > 0)
    )
    logger.info("After removing zero-activity users: %d rows", len(data))

    # 3. Cap extreme karma outliers (above 99th percentile)
    karma_cap = data[target].quantile(0.99)
    data = data.filter(pl.col(target) <= karma_cap)
    logger.info("Karma cap (p99): %.0f — after capping: %d rows", karma_cap, len(data))

    n_after = len(data)
    logger.info(
        "Cleaned data: %d -> %d rows (removed %d outliers/bots)",
        n_before, n_after, n_before - n_after,
    )

    # 4. Apply log transform to target: log(karma + 1)
    data = data.with_columns(
        pl.col(target).log1p().alias("log_karma")
    )

    logger.info(
        "Log karma stats: mean=%.2f, median=%.2f, std=%.2f, max=%.2f",
        data["log_karma"].mean(),
        data["log_karma"].median(),
        data["log_karma"].std(),
        data["log_karma"].max(),
    )

    return data


class H2OTrainer:
    """H2O AutoML trainer for karma regression."""

    def __init__(
        self,
        max_models: int = 10,
        max_runtime_secs: int = 300,
        seed: int = 42,
    ):
        """Initialize H2O trainer.

        Args:
            max_models: Maximum number of models for AutoML
            max_runtime_secs: Maximum training time in seconds
            seed: Random seed for reproducibility
        """
        self.max_models = max_models
        self.max_runtime_secs = max_runtime_secs
        self.seed = seed
        self._h2o = None
        self._model = None

    def _init_h2o(self) -> None:
        """Initialize H2O cluster."""
        if self._h2o is not None:
            return

        import h2o
        h2o.init(nthreads=-1, max_mem_size="4G")
        self._h2o = h2o
        logger.info("H2O initialized")

    def _shutdown_h2o(self) -> None:
        """Shutdown H2O cluster."""
        if self._h2o is not None:
            self._h2o.cluster().shutdown()
            self._h2o = None
            logger.info("H2O shutdown")

    def train(
        self,
        data: pl.DataFrame,
        target: str = "karma",
        features: Optional[List[str]] = None,
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """Train model using H2O AutoML.

        Cleans data (removes outliers/bots) and trains on log(karma+1)
        for better handling of the skewed distribution.

        Args:
            data: Input DataFrame with features and target
            target: Target column name
            features: Feature column names (uses defaults if None)
            test_size: Fraction of data for testing

        Returns:
            Dictionary with training results
        """
        self._init_h2o()

        # Clean data: remove outliers, add log_karma
        cleaned = clean_training_data(data, target)
        self._cleaned_data = cleaned

        features = features or FEATURE_COLUMNS
        available_features = [f for f in features if f in cleaned.columns]

        if not available_features:
            raise ValueError("No valid feature columns found in data")

        # Train on log_karma for better distribution
        train_target = "log_karma"

        logger.info(
            "Training with %d features, %d samples (target: %s)",
            len(available_features),
            len(cleaned),
            train_target,
        )

        # Convert to H2O frame
        h2o_data = self._h2o.H2OFrame(cleaned.to_pandas())

        # Split data
        train, test = h2o_data.split_frame(ratios=[1 - test_size], seed=self.seed)

        logger.info("Train size: %d, Test size: %d", train.nrows, test.nrows)

        # Run AutoML
        from h2o.automl import H2OAutoML

        aml = H2OAutoML(
            max_models=self.max_models,
            max_runtime_secs=self.max_runtime_secs,
            seed=self.seed,
            sort_metric="RMSE",
            exclude_algos=["DeepLearning"],  # Exclude for speed
        )

        aml.train(
            x=available_features,
            y=train_target,
            training_frame=train,
            validation_frame=test,
        )

        self._model = aml.leader
        self._train_target = train_target
        logger.info("Best model: %s", self._model.model_id)

        # Get performance on log scale (primary metrics — this is what model optimizes)
        perf = self._model.model_performance(test)

        # Also compute metrics in original karma scale (informational)
        test_pred = self._model.predict(test)
        test_pandas = test.as_data_frame()
        pred_pandas = test_pred.as_data_frame()

        import numpy as np
        actual_karma = np.expm1(test_pandas[train_target].values)
        pred_karma = np.expm1(pred_pandas["predict"].values)
        # Clip negative predictions to 0
        pred_karma = np.clip(pred_karma, 0, None)

        mae_original = float(np.mean(np.abs(actual_karma - pred_karma)))
        rmse_original = float(np.sqrt(np.mean((actual_karma - pred_karma) ** 2)))
        # Median absolute error is more robust for skewed data
        medae_original = float(np.median(np.abs(actual_karma - pred_karma)))

        ss_res = float(np.sum((actual_karma - pred_karma) ** 2))
        ss_tot = float(np.sum((actual_karma - np.mean(actual_karma)) ** 2))
        r2_original = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        results = {
            "model_id": self._model.model_id,
            # PRIMARY: Log-scale metrics (model trains on this)
            "r2": perf.r2(),
            "mae": perf.mae(),
            "rmse": perf.rmse(),
            # SECONDARY: Original karma scale (informational)
            "karma_mae": mae_original,
            "karma_rmse": rmse_original,
            "karma_medae": medae_original,
            "karma_r2": r2_original,
            "features_used": available_features,
            "train_size": train.nrows,
            "test_size": test.nrows,
            "samples_after_cleaning": len(cleaned),
        }

        logger.info(
            "Training complete (log scale) - R2: %.4f, MAE: %.4f, RMSE: %.4f",
            results["r2"],
            results["mae"],
            results["rmse"],
        )
        logger.info(
            "Original karma scale - MAE: %.2f, MedAE: %.2f, RMSE: %.2f, R2: %.4f",
            results["karma_mae"],
            results["karma_medae"],
            results["karma_rmse"],
            results["karma_r2"],
        )

        return results

    def predict(self, data: pl.DataFrame) -> pl.DataFrame:
        """Make predictions on new data.

        Predictions are made in log scale and converted back to original
        karma scale using exp(pred) - 1.

        Args:
            data: Input DataFrame with features

        Returns:
            DataFrame with predictions added
        """
        if self._model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        self._init_h2o()

        h2o_data = self._h2o.H2OFrame(data.to_pandas())
        predictions = self._model.predict(h2o_data)

        # Convert back to Polars
        pred_df = pl.from_pandas(predictions.as_data_frame())

        # Convert from log scale back to original karma scale: exp(x) - 1
        pred_df = pred_df.with_columns(
            pl.col("predict").exp().sub(1).clip(lower_bound=0).alias("karma_predicted")
        )

        # Join with original data
        result = data.with_columns([
            pred_df["karma_predicted"],
        ])

        return result

    def save_model(self, output_dir: Optional[Path] = None) -> Path:
        """Save the trained model.

        Args:
            output_dir: Output directory for model artifacts

        Returns:
            Path to saved model
        """
        if self._model is None:
            raise RuntimeError("No model to save. Call train() first.")

        output_dir = output_dir or settings.models_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save H2O model
        model_path = self._h2o.save_model(
            model=self._model,
            path=str(output_dir),
            force=True,
        )

        logger.info("Saved model to %s", model_path)
        return Path(model_path)

    def load_model(self, model_path: Path) -> None:
        """Load a saved model.

        Args:
            model_path: Path to saved model
        """
        self._init_h2o()
        self._model = self._h2o.load_model(str(model_path))
        logger.info("Loaded model from %s", model_path)


def train_model(
    gold_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    max_models: int = 10,
    max_runtime_secs: int = 300,
) -> Dict[str, Any]:
    """Train a karma prediction model.

    Args:
        gold_dir: Gold layer data directory
        output_dir: Output directory for model artifacts
        max_models: Maximum models for AutoML
        max_runtime_secs: Maximum training time

    Returns:
        Dictionary with training results
    """
    gold_dir = gold_dir or settings.gold_dir
    output_dir = output_dir or settings.models_dir

    # Load data
    data = get_modeling_data(gold_dir)
    logger.info("Loaded %d samples for training", len(data))

    # Train model
    trainer = H2OTrainer(
        max_models=max_models,
        max_runtime_secs=max_runtime_secs,
    )

    results = trainer.train(data, target="karma")

    # Save model
    model_path = trainer.save_model(output_dir)
    results["model_path"] = str(model_path)

    # Generate and save predictions
    predictions = trainer.predict(data)
    pred_path = output_dir / "predictions.parquet"
    predictions.write_parquet(pred_path)
    results["predictions_path"] = str(pred_path)

    logger.info("Predictions saved to %s", pred_path)

    return results


def evaluate_model(
    model_path: Path,
    data: Optional[pl.DataFrame] = None,
    gold_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """Evaluate a trained model.

    Args:
        model_path: Path to saved model
        data: Input data (loads from gold layer if None)
        gold_dir: Gold layer directory

    Returns:
        Dictionary with evaluation metrics
    """
    if data is None:
        data = get_modeling_data(gold_dir)

    trainer = H2OTrainer()
    trainer.load_model(model_path)

    predictions = trainer.predict(data)

    # Calculate metrics manually using Polars
    actual = predictions["karma"]
    predicted = predictions["karma_predicted"]

    mae = (actual - predicted).abs().mean()
    mse = ((actual - predicted) ** 2).mean()
    rmse = mse ** 0.5

    ss_res = ((actual - predicted) ** 2).sum()
    ss_tot = ((actual - actual.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }

    logger.info("Evaluation metrics: %s", metrics)
    return metrics
