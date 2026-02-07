"""H2O AutoML trainer for karma prediction."""

import logging
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
    "has_description",
    "has_human_owner",
    "description_length",
    "post_count",
    "total_post_rating",
    "avg_post_rating",
    "max_post_rating",
    "avg_title_length",
    "avg_post_desc_length",
    "comment_count",
    "total_comment_rating",
    "avg_comment_rating",
    "avg_comment_length",
    "total_activity",
    "total_rating",
]


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

        Args:
            data: Input DataFrame with features and target
            target: Target column name
            features: Feature column names (uses defaults if None)
            test_size: Fraction of data for testing

        Returns:
            Dictionary with training results
        """
        self._init_h2o()

        features = features or FEATURE_COLUMNS
        available_features = [f for f in features if f in data.columns]

        if not available_features:
            raise ValueError("No valid feature columns found in data")

        logger.info(
            "Training with %d features, %d samples",
            len(available_features),
            len(data),
        )

        # Convert to H2O frame
        h2o_data = self._h2o.H2OFrame(data.to_pandas())

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
            y=target,
            training_frame=train,
            validation_frame=test,
        )

        self._model = aml.leader
        logger.info("Best model: %s", self._model.model_id)

        # Get performance metrics
        perf = self._model.model_performance(test)

        results = {
            "model_id": self._model.model_id,
            "mae": perf.mae(),
            "rmse": perf.rmse(),
            "r2": perf.r2(),
            "features_used": available_features,
            "train_size": train.nrows,
            "test_size": test.nrows,
        }

        logger.info(
            "Training complete - MAE: %.4f, RMSE: %.4f, R2: %.4f",
            results["mae"],
            results["rmse"],
            results["r2"],
        )

        return results

    def predict(self, data: pl.DataFrame) -> pl.DataFrame:
        """Make predictions on new data.

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
        pred_df = pred_df.rename({"predict": "karma_predicted"})

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

    # Filter out users with zero karma (if too many)
    karma_stats = data["karma"].describe()
    logger.info("Karma distribution: %s", karma_stats)

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
