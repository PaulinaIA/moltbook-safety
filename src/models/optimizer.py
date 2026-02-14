"""H2O model optimization with Grid Search.

Implements hyperparameter tuning for the best-performing model type
(GBM) to improve prediction accuracy beyond the AutoML baseline.
"""

import logging
from typing import Dict, List, Optional, Any

import polars as pl

from config.settings import settings

logger = logging.getLogger(__name__)

# Default hyperparameter grid for GBM
DEFAULT_GBM_GRID = {
    "max_depth": [3, 5, 7, 10],
    "learn_rate": [0.01, 0.05, 0.1],
    "ntrees": [50, 100, 200],
    "sample_rate": [0.7, 0.8, 1.0],
}


def optimize_gbm(
    train_frame,
    valid_frame,
    target: str = "karma",
    features: Optional[List[str]] = None,
    hyper_params: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Run H2O Grid Search to optimize a GBM model.

    Args:
        train_frame: H2O training frame
        valid_frame: H2O validation frame
        target: Target column name
        features: Feature column names
        hyper_params: Hyperparameter grid (uses defaults if None)

    Returns:
        Dictionary with optimization results including best model
        and comparison metrics
    """
    import h2o
    from h2o.estimators import H2OGradientBoostingEstimator
    from h2o.grid.grid_search import H2OGridSearch

    hyper_params = hyper_params or DEFAULT_GBM_GRID

    # Define the base GBM estimator
    gbm = H2OGradientBoostingEstimator(
        seed=42,
        stopping_metric="MAE",
        stopping_rounds=5,
        stopping_tolerance=0.001,
    )

    # Set up grid search
    grid = H2OGridSearch(
        model=gbm,
        hyper_params=hyper_params,
        grid_id="gbm_optimization_grid",
        search_criteria={
            "strategy": "RandomDiscrete",
            "max_models": 15,
            "max_runtime_secs": 300,
            "seed": 42,
        },
    )

    logger.info("Starting GBM Grid Search with %d max models...",
                grid.search_criteria.get("max_models", "?"))

    # Train grid
    grid.train(
        x=features,
        y=target,
        training_frame=train_frame,
        validation_frame=valid_frame,
    )

    # Get results sorted by MAE
    grid_table = grid.get_grid(sort_by="mae", decreasing=False)

    # Best model
    best_model = grid_table.models[0]
    best_perf_valid = best_model.model_performance(valid_frame)

    results = {
        "best_model": best_model,
        "best_model_id": best_model.model_id,
        "best_params": {
            "max_depth": best_model.params["max_depth"]["actual"],
            "learn_rate": best_model.params["learn_rate"]["actual"],
            "ntrees": best_model.params["ntrees"]["actual"],
            "sample_rate": best_model.params["sample_rate"]["actual"],
        },
        "mae": best_perf_valid.mae(),
        "rmse": best_perf_valid.rmse(),
        "r2": best_perf_valid.r2(),
        "total_models_trained": len(grid_table.models),
        "grid_summary": grid_table,
    }

    logger.info("Grid Search complete — Best MAE: %.4f, Best R²: %.4f",
                results["mae"], results["r2"])
    return results


def compare_models(
    base_results: Dict[str, Any],
    optimized_results: Dict[str, Any],
) -> pl.DataFrame:
    """Compare base AutoML model with optimized model.

    Args:
        base_results: Results from AutoML training
        optimized_results: Results from grid search optimization

    Returns:
        Polars DataFrame with side-by-side comparison
    """
    comparison = pl.DataFrame({
        "Metric": ["MAE", "RMSE", "R²"],
        "AutoML Base": [
            round(base_results.get("mae", 0), 4),
            round(base_results.get("rmse", 0), 4),
            round(base_results.get("r2", 0), 4),
        ],
        "GBM Optimized": [
            round(optimized_results.get("mae", 0), 4),
            round(optimized_results.get("rmse", 0), 4),
            round(optimized_results.get("r2", 0), 4),
        ],
    })

    # Add improvement column
    comparison = comparison.with_columns(
        pl.when(pl.col("Metric") == "R²")
        .then(
            ((pl.col("GBM Optimized") - pl.col("AutoML Base"))
             / pl.col("AutoML Base").abs() * 100)
        )
        .otherwise(
            ((pl.col("AutoML Base") - pl.col("GBM Optimized"))
             / pl.col("AutoML Base").abs() * 100)
        )
        .round(2)
        .alias("Improvement (%)")
    )

    return comparison
