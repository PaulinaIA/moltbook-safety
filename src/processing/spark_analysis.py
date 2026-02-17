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
    """Create or reuse a local SparkSession.

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
    return spark


def spark_eda(
    silver_dir: Optional[Path] = None,
    spark=None,
) -> Dict[str, Any]:
    """Run exploratory data analysis using PySpark.

    Loads silver layer Parquet files into Spark DataFrames and computes
    descriptive statistics, null counts, karma distribution, and
    correlations.

    Args:
        silver_dir: Path to silver layer directory
        spark: Optional existing SparkSession (creates one if None)

    Returns:
        Dictionary with EDA results
    """
    from pyspark.sql import functions as F

    silver_dir = silver_dir or settings.silver_dir
    owns_spark = spark is None
    spark = spark or create_spark_session("MoltbookEDA")

    results = {}

    # --- Load data ---
    users_df = spark.read.parquet(str(silver_dir / "users.parquet")).cache()
    posts_df = spark.read.parquet(str(silver_dir / "posts.parquet"))
    comments_df = spark.read.parquet(str(silver_dir / "comments.parquet"))

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

    # --- Null counts per column (single pass) ---
    null_exprs = [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in users_df.columns]
    null_row = users_df.select(null_exprs).toPandas().iloc[0].to_dict()
    results["user_null_counts"] = {k: int(v) for k, v in null_row.items()}

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
    correlations = {}
    for col_name in ["followers", "following"]:
        corr_val = users_df.stat.corr("karma", col_name)
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

    users_df.unpersist()
    logger.info("PySpark EDA complete")
    if owns_spark:
        spark.stop()
    return results


def spark_evaluate_predictions(
    predictions_path: Optional[Path] = None,
    gold_dir: Optional[Path] = None,
    spark=None,
) -> Dict[str, Any]:
    """Evaluate model predictions using PySpark ML evaluation.

    Args:
        predictions_path: Path to predictions Parquet file
        gold_dir: Gold layer directory (used to find predictions)
        spark: Optional existing SparkSession (creates one if None)

    Returns:
        Dictionary with evaluation metrics
    """
    from pyspark.sql import functions as F
    from pyspark.ml.evaluation import RegressionEvaluator

    gold_dir = gold_dir or settings.gold_dir
    predictions_path = predictions_path or (settings.models_dir / "predictions.parquet")
    owns_spark = spark is None
    spark = spark or create_spark_session("MoltbookEvaluation")

    results = {}

    # Load and cache predictions (small dataset, reused multiple times)
    pred_df = spark.read.parquet(str(predictions_path)).cache()
    total = pred_df.count()
    results["total_predictions"] = total

    # PRIMARY: Evaluate on log scale (consistent with H2O training target)
    log_eval_df = (
        pred_df
        .withColumn("label", F.log1p(F.col("karma").cast("double")))
        .withColumn("prediction", F.log1p(F.col("karma_predicted")))
    ).cache()

    evaluators = {
        "mae": RegressionEvaluator(metricName="mae"),
        "rmse": RegressionEvaluator(metricName="rmse"),
        "r2": RegressionEvaluator(metricName="r2"),
    }

    for metric_name, evaluator in evaluators.items():
        value = evaluator.evaluate(log_eval_df)
        results[metric_name] = round(value, 4)
        logger.info("PySpark (log scale) %s: %.4f", metric_name, value)

    # SECONDARY: Evaluate on original karma scale (informational)
    karma_eval_df = (
        pred_df
        .withColumnRenamed("karma", "label")
        .withColumnRenamed("karma_predicted", "prediction")
    )

    for metric_name, evaluator in evaluators.items():
        value = evaluator.evaluate(karma_eval_df)
        results[f"karma_{metric_name}"] = round(value, 4)
        logger.info("PySpark (karma scale) %s: %.4f", metric_name, value)

    # --- Residual analysis (log scale) ---
    residual_stats = log_eval_df.select(
        F.mean(F.col("label") - F.col("prediction")).alias("mean_residual"),
        F.stddev(F.col("label") - F.col("prediction")).alias("std_residual"),
        F.min(F.col("label") - F.col("prediction")).alias("min_residual"),
        F.max(F.col("label") - F.col("prediction")).alias("max_residual"),
        F.mean(F.abs(F.col("label") - F.col("prediction"))).alias("mean_abs_residual"),
    ).toPandas()

    results["residual_stats"] = residual_stats.to_dict(orient="records")[0]

    # --- Prediction vs actual by quartiles (original karma scale) ---
    pcts = karma_eval_df.select(
        F.percentile_approx("label", 0.25).alias("q25"),
        F.percentile_approx("label", 0.50).alias("q50"),
        F.percentile_approx("label", 0.75).alias("q75"),
    ).first()
    q25, q50, q75 = pcts["q25"], pcts["q50"], pcts["q75"]

    quartile_analysis = (
        karma_eval_df
        .withColumn(
            "karma_quartile",
            F.when(F.col("label") == 0, "Q0 (zero)")
            .when(F.col("label") <= q25, "Q1")
            .when(F.col("label") <= q50, "Q2")
            .when(F.col("label") <= q75, "Q3")
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

    log_eval_df.unpersist()
    pred_df.unpersist()
    logger.info("PySpark evaluation complete")
    if owns_spark:
        spark.stop()
    return results
