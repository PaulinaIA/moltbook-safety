"""PySpark analysis module — EDA and model evaluation with Spark.

Provides complementary analysis using PySpark's distributed computing
capabilities. Uses local SparkSession for processing Parquet files
from the silver and gold layers.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

from config.settings import settings

logger = logging.getLogger(__name__)


def create_spark_session(app_name: str = "MoltbookKarma"):
    """Create a local SparkSession.

    Args:
        app_name: Name for the Spark application

    Returns:
        Active SparkSession
    """
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    logger.info("SparkSession created: %s", app_name)
    return spark


def spark_eda(
    silver_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run exploratory data analysis using PySpark.

    Loads silver layer Parquet files into Spark DataFrames and computes
    descriptive statistics, null counts, karma distribution, and
    correlations.

    Args:
        silver_dir: Path to silver layer directory

    Returns:
        Dictionary with EDA results
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType

    silver_dir = silver_dir or settings.silver_dir
    spark = create_spark_session("MoltbookEDA")

    results = {}

    # --- Load data ---
    users_path = str(silver_dir / "users.parquet")
    posts_path = str(silver_dir / "posts.parquet")
    comments_path = str(silver_dir / "comments.parquet")

    users_df = spark.read.parquet(users_path)
    posts_df = spark.read.parquet(posts_path)
    comments_df = spark.read.parquet(comments_path)

    results["record_counts"] = {
        "users": users_df.count(),
        "posts": posts_df.count(),
        "comments": comments_df.count(),
    }
    logger.info("Loaded data — Users: %d, Posts: %d, Comments: %d",
                results["record_counts"]["users"],
                results["record_counts"]["posts"],
                results["record_counts"]["comments"])

    # --- Descriptive statistics ---
    results["users_describe"] = users_df.describe().toPandas()
    results["posts_describe"] = posts_df.describe().toPandas()

    # --- Null counts per column ---
    user_nulls = {}
    for col_name in users_df.columns:
        null_count = users_df.filter(F.col(col_name).isNull()).count()
        user_nulls[col_name] = null_count
    results["user_null_counts"] = user_nulls

    # --- Karma distribution by bins ---
    karma_bins = (
        users_df
        .withColumn("karma_bin", F.when(F.col("karma") == 0, "0")
                    .when(F.col("karma") <= 10, "1-10")
                    .when(F.col("karma") <= 100, "11-100")
                    .when(F.col("karma") <= 1000, "101-1K")
                    .when(F.col("karma") <= 10000, "1K-10K")
                    .otherwise(">10K"))
        .groupBy("karma_bin")
        .agg(F.count("*").alias("count"))
        .orderBy("count", ascending=False)
        .toPandas()
    )
    results["karma_distribution"] = karma_bins

    # --- Top 10 users by karma ---
    top_users = (
        users_df
        .select("name", "karma", "followers", "following")
        .orderBy(F.col("karma").desc())
        .limit(10)
        .toPandas()
    )
    results["top_users"] = top_users

    # --- Correlation: karma vs numeric columns ---
    numeric_cols = ["karma", "followers", "following"]
    correlations = {}
    for col_name in numeric_cols:
        if col_name != "karma":
            corr_val = users_df.stat.corr(
                "karma",
                col_name,
            )
            correlations[f"karma_vs_{col_name}"] = round(corr_val, 4)
    results["correlations"] = correlations

    # --- Posts per user stats ---
    posts_per_user = (
        posts_df
        .groupBy("id_user")
        .agg(
            F.count("*").alias("post_count"),
            F.avg("rating").alias("avg_rating"),
            F.sum("rating").alias("total_rating"),
        )
    )
    results["posts_per_user_describe"] = posts_per_user.describe().toPandas()

    logger.info("PySpark EDA complete")
    spark.stop()
    return results


def spark_evaluate_predictions(
    predictions_path: Optional[Path] = None,
    gold_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate model predictions using PySpark ML evaluation.

    Args:
        predictions_path: Path to predictions Parquet file
        gold_dir: Gold layer directory (used to find predictions)

    Returns:
        Dictionary with evaluation metrics
    """
    from pyspark.sql import functions as F
    from pyspark.ml.evaluation import RegressionEvaluator

    gold_dir = gold_dir or settings.gold_dir
    predictions_path = predictions_path or (settings.models_dir / "predictions.parquet")
    spark = create_spark_session("MoltbookEvaluation")

    results = {}

    # Load predictions
    pred_df = spark.read.parquet(str(predictions_path))
    total = pred_df.count()
    results["total_predictions"] = total

    # Rename for evaluator compatibility
    eval_df = (
        pred_df
        .withColumnRenamed("karma", "label")
        .withColumnRenamed("karma_predicted", "prediction")
    )

    # --- Regression metrics ---
    evaluators = {
        "mae": RegressionEvaluator(metricName="mae"),
        "rmse": RegressionEvaluator(metricName="rmse"),
        "r2": RegressionEvaluator(metricName="r2"),
        "mse": RegressionEvaluator(metricName="mse"),
    }

    for metric_name, evaluator in evaluators.items():
        value = evaluator.evaluate(eval_df)
        results[metric_name] = round(value, 4)
        logger.info("PySpark %s: %.4f", metric_name, value)

    # --- Residual analysis ---
    residuals_df = (
        eval_df
        .withColumn("residual", F.col("label") - F.col("prediction"))
        .withColumn("abs_residual", F.abs(F.col("residual")))
    )

    residual_stats = residuals_df.select(
        F.mean("residual").alias("mean_residual"),
        F.stddev("residual").alias("std_residual"),
        F.min("residual").alias("min_residual"),
        F.max("residual").alias("max_residual"),
        F.mean("abs_residual").alias("mean_abs_residual"),
    ).toPandas()

    results["residual_stats"] = residual_stats.to_dict(orient="records")[0]

    # --- Prediction vs actual by quartiles ---
    quartile_analysis = (
        eval_df
        .withColumn(
            "karma_quartile",
            F.when(F.col("label") == 0, "Q0 (zero)")
            .when(F.col("label") <= F.percentile_approx("label", 0.25), "Q1")
            .when(F.col("label") <= F.percentile_approx("label", 0.5), "Q2")
            .when(F.col("label") <= F.percentile_approx("label", 0.75), "Q3")
            .otherwise("Q4 (top)")
        )
        .groupBy("karma_quartile")
        .agg(
            F.count("*").alias("count"),
            F.avg("label").alias("avg_actual"),
            F.avg("prediction").alias("avg_predicted"),
            F.avg(F.abs(F.col("label") - F.col("prediction"))).alias("avg_error"),
        )
        .orderBy("karma_quartile")
        .toPandas()
    )
    results["quartile_analysis"] = quartile_analysis

    logger.info("PySpark evaluation complete")
    spark.stop()
    return results
