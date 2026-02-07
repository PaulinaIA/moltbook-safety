"""Gold layer processing - feature engineering with Polars Lazy evaluation.

Gold layer responsibilities:
- Load silver layer Parquet files
- Engineer features for modeling
- Aggregate statistics from related entities
- Output modeling-ready dataset
"""

import logging
from pathlib import Path
from typing import Optional

import polars as pl

from config.settings import settings

logger = logging.getLogger(__name__)


def load_silver_data(silver_dir: Optional[Path] = None) -> dict:
    """Load all silver layer Parquet files.

    Args:
        silver_dir: Silver layer directory

    Returns:
        Dictionary of LazyFrames by entity name
    """
    silver_dir = silver_dir or settings.silver_dir

    data = {}

    for name in ["users", "posts", "comments", "submolts"]:
        path = silver_dir / f"{name}.parquet"
        if path.exists():
            data[name] = pl.scan_parquet(path)
            logger.info("Loaded %s from silver layer", name)
        else:
            logger.warning("Silver layer file not found: %s", path)
            data[name] = pl.LazyFrame()

    return data


def engineer_user_features(
    users_lf: pl.LazyFrame,
    posts_lf: pl.LazyFrame,
    comments_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    """Engineer features for user karma prediction.

    Args:
        users_lf: Users LazyFrame
        posts_lf: Posts LazyFrame
        comments_lf: Comments LazyFrame

    Returns:
        Feature-engineered LazyFrame ready for modeling
    """
    # Aggregate post statistics per user
    post_stats = (
        posts_lf
        .group_by("id_user")
        .agg([
            pl.count().alias("post_count"),
            pl.col("rating").sum().alias("total_post_rating"),
            pl.col("rating").mean().alias("avg_post_rating"),
            pl.col("rating").max().alias("max_post_rating"),
            pl.col("title").str.len_chars().mean().alias("avg_title_length"),
            pl.col("description").str.len_chars().mean().alias("avg_post_desc_length"),
        ])
    )

    # Aggregate comment statistics per user
    comment_stats = (
        comments_lf
        .group_by("id_user")
        .agg([
            pl.count().alias("comment_count"),
            pl.col("rating").sum().alias("total_comment_rating"),
            pl.col("rating").mean().alias("avg_comment_rating"),
            pl.col("description").str.len_chars().mean().alias("avg_comment_length"),
        ])
    )

    # Join features to users
    features = (
        users_lf
        .join(post_stats, on="id_user", how="left")
        .join(comment_stats, on="id_user", how="left")
        # Fill nulls for users without posts/comments
        .with_columns([
            pl.col("post_count").fill_null(0),
            pl.col("total_post_rating").fill_null(0),
            pl.col("avg_post_rating").fill_null(0.0),
            pl.col("max_post_rating").fill_null(0),
            pl.col("avg_title_length").fill_null(0.0),
            pl.col("avg_post_desc_length").fill_null(0.0),
            pl.col("comment_count").fill_null(0),
            pl.col("total_comment_rating").fill_null(0),
            pl.col("avg_comment_rating").fill_null(0.0),
            pl.col("avg_comment_length").fill_null(0.0),
        ])
        # Create derived features
        .with_columns([
            # Engagement ratio (following vs followers)
            (pl.col("followers") / (pl.col("following") + 1)).alias("follower_ratio"),
            # Total engagement
            (pl.col("post_count") + pl.col("comment_count")).alias("total_activity"),
            # Activity quality score
            (
                pl.col("total_post_rating") + pl.col("total_comment_rating")
            ).alias("total_rating"),
            # Has description flag
            (pl.col("description").str.len_chars() > 0).cast(pl.Int32).alias("has_description"),
            # Has human owner flag
            pl.col("human_owner").is_not_null().cast(pl.Int32).alias("has_human_owner"),
            # Description length
            pl.col("description").str.len_chars().alias("description_length"),
        ])
        # Select final feature columns
        .select([
            # ID for joining back
            "id_user",
            "name",
            # Target variable
            "karma",
            # User profile features
            "followers",
            "following",
            "follower_ratio",
            "has_description",
            "has_human_owner",
            "description_length",
            # Post features
            "post_count",
            "total_post_rating",
            "avg_post_rating",
            "max_post_rating",
            "avg_title_length",
            "avg_post_desc_length",
            # Comment features
            "comment_count",
            "total_comment_rating",
            "avg_comment_rating",
            "avg_comment_length",
            # Derived features
            "total_activity",
            "total_rating",
        ])
    )

    return features


def build_gold_layer(
    silver_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> dict:
    """Build the gold (feature-engineered) data layer.

    Args:
        silver_dir: Silver layer input directory
        output_dir: Gold layer output directory

    Returns:
        Dictionary with output statistics
    """
    silver_dir = silver_dir or settings.silver_dir
    output_dir = output_dir or settings.gold_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load silver data
    data = load_silver_data(silver_dir)

    results = {}

    # Build user features for karma prediction
    try:
        features_lf = engineer_user_features(
            users_lf=data["users"],
            posts_lf=data["posts"],
            comments_lf=data["comments"],
        )

        features_df = features_lf.collect()
        features_path = output_dir / "user_features.parquet"
        features_df.write_parquet(features_path)

        results["user_features"] = len(features_df)
        results["feature_columns"] = len(features_df.columns)

        logger.info(
            "Wrote %d user feature records with %d columns to %s",
            len(features_df),
            len(features_df.columns),
            features_path,
        )

        # Log feature summary
        logger.info("Feature columns: %s", features_df.columns)

    except Exception as e:
        logger.error("Failed to build gold layer: %s", e)
        results["user_features"] = 0
        results["feature_columns"] = 0

    logger.info("Gold layer build complete: %s", results)
    return results


def get_modeling_data(gold_dir: Optional[Path] = None) -> pl.DataFrame:
    """Load the gold layer data for modeling.

    Args:
        gold_dir: Gold layer directory

    Returns:
        DataFrame ready for modeling
    """
    gold_dir = gold_dir or settings.gold_dir
    features_path = gold_dir / "user_features.parquet"

    if not features_path.exists():
        raise FileNotFoundError(
            f"Gold layer not found at {features_path}. Run 'python -m app build' first."
        )

    return pl.read_parquet(features_path)
