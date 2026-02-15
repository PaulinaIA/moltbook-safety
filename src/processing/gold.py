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

    # Build post-level features (denormalized, >20 columns)
    try:
        post_features_lf = engineer_post_features(
            posts_lf=data["posts"],
            users_lf=data["users"],
            comments_lf=data["comments"],
            submolts_lf=data["submolts"],
        )

        post_features_df = post_features_lf.collect()
        post_features_path = output_dir / "post_features.parquet"
        post_features_df.write_parquet(post_features_path)

        results["post_features"] = len(post_features_df)
        results["post_feature_columns"] = len(post_features_df.columns)

        logger.info(
            "Wrote %d post feature records with %d columns to %s",
            len(post_features_df),
            len(post_features_df.columns),
            post_features_path,
        )

    except Exception as e:
        logger.error("Failed to build post features: %s", e)
        results["post_features"] = 0
        results["post_feature_columns"] = 0

    # Build comment-level features (maximum row count for 200K+ target)
    try:
        comment_features_lf = engineer_comment_features(
            comments_lf=data["comments"],
            posts_lf=data["posts"],
            users_lf=data["users"],
            submolts_lf=data["submolts"],
        )

        comment_features_df = comment_features_lf.collect()
        comment_features_path = output_dir / "comment_features.parquet"
        comment_features_df.write_parquet(comment_features_path)

        results["comment_features"] = len(comment_features_df)
        results["comment_feature_columns"] = len(comment_features_df.columns)

        logger.info(
            "Wrote %d comment feature records with %d columns to %s",
            len(comment_features_df),
            len(comment_features_df.columns),
            comment_features_path,
        )

    except Exception as e:
        logger.error("Failed to build comment features: %s", e)
        results["comment_features"] = 0
        results["comment_feature_columns"] = 0

    logger.info("Gold layer build complete: %s", results)
    return results


def engineer_post_features(
    posts_lf: pl.LazyFrame,
    users_lf: pl.LazyFrame,
    comments_lf: pl.LazyFrame,
    submolts_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    """Engineer features at POST level for large denormalized dataset.

    Creates a wide dataset (>20 columns) by joining post data with
    user features, submolt features, and engineered metrics.

    Args:
        posts_lf: Posts LazyFrame
        users_lf: Users LazyFrame
        comments_lf: Comments LazyFrame
        submolts_lf: Submolts LazyFrame

    Returns:
        Feature-engineered LazyFrame at post level (~26 columns)
    """
    # User-level aggregates
    user_post_stats = (
        posts_lf
        .group_by("id_user")
        .agg([
            pl.count().alias("user_total_posts"),
            pl.col("rating").mean().alias("user_avg_rating"),
        ])
    )

    user_comment_stats = (
        comments_lf
        .group_by("id_user")
        .agg([
            pl.count().alias("user_comment_count"),
        ])
    )

    # Enrich users with aggregates
    users_enriched = (
        users_lf
        .join(user_post_stats, on="id_user", how="left")
        .join(user_comment_stats, on="id_user", how="left")
        .with_columns([
            pl.col("user_total_posts").fill_null(0),
            pl.col("user_avg_rating").fill_null(0.0),
            pl.col("user_comment_count").fill_null(0),
            (pl.col("followers") / (pl.col("following") + 1)).alias("user_follower_ratio"),
            (pl.col("description").str.len_chars().fill_null(0)).alias("user_description_length"),
            (pl.col("description").str.len_chars() > 0).cast(pl.Int32).alias("user_has_description"),
            pl.col("human_owner").is_not_null().cast(pl.Int32).alias("user_has_human_owner"),
        ])
        .select([
            "id_user",
            pl.col("name").alias("user_name"),
            pl.col("karma").alias("user_karma"),
            pl.col("followers").alias("user_followers"),
            pl.col("following").alias("user_following"),
            "user_follower_ratio",
            "user_has_description",
            "user_has_human_owner",
            "user_description_length",
            "user_total_posts",
            "user_avg_rating",
            "user_comment_count",
        ])
    )

    # Enrich submolts
    submolts_enriched = (
        submolts_lf
        .with_columns([
            pl.col("description").str.len_chars().fill_null(0).alias("submolt_description_length"),
        ])
        .select([
            "id_submolt",
            pl.col("name").alias("submolt_name"),
            "submolt_description_length",
        ])
    )

    # Build post-level features
    post_features = (
        posts_lf
        .join(users_enriched, on="id_user", how="left")
        .join(submolts_enriched, on="id_submolt", how="left")
        .with_columns([
            pl.col("title").str.len_chars().fill_null(0).alias("title_length"),
            pl.col("description").str.len_chars().fill_null(0).alias("description_length"),
            pl.col("title").str.split(" ").list.len().fill_null(0).alias("title_word_count"),
            (pl.col("description").str.len_chars() > 0).cast(pl.Int32).alias("has_description"),
            pl.col("submolt_name").fill_null("unknown"),
            pl.col("submolt_description_length").fill_null(0),
            pl.col("user_name").fill_null("unknown"),
            pl.col("user_karma").fill_null(0),
            pl.col("user_followers").fill_null(0),
            pl.col("user_following").fill_null(0),
            pl.col("user_follower_ratio").fill_null(0.0),
            pl.col("user_has_description").fill_null(0),
            pl.col("user_has_human_owner").fill_null(0),
            pl.col("user_description_length").fill_null(0),
            pl.col("user_total_posts").fill_null(0),
            pl.col("user_avg_rating").fill_null(0.0),
            pl.col("user_comment_count").fill_null(0),
        ])
        .select([
            # Post base (6)
            "id_post", "title", "description", "rating", "date", "scraped_at",
            # Foreign keys (2)
            "id_user", "id_submolt",
            # User features (10)
            "user_name", "user_karma", "user_followers", "user_following",
            "user_follower_ratio", "user_has_description", "user_has_human_owner",
            "user_description_length", "user_total_posts", "user_avg_rating",
            "user_comment_count",
            # Submolt features (2)
            "submolt_name", "submolt_description_length",
            # Post engineered features (4)
            "title_length", "description_length", "title_word_count", "has_description",
        ])
    )

    return post_features


def engineer_comment_features(
    comments_lf: pl.LazyFrame,
    posts_lf: pl.LazyFrame,
    users_lf: pl.LazyFrame,
    submolts_lf: pl.LazyFrame,
) -> pl.LazyFrame:
    """Engineer features at COMMENT level for maximum row count.

    Creates a wide dataset (>25 columns) by joining each comment with
    its post, author, post author, and submolt data.

    Args:
        comments_lf: Comments LazyFrame
        posts_lf: Posts LazyFrame
        users_lf: Users LazyFrame (comment + post authors)
        submolts_lf: Submolts LazyFrame

    Returns:
        Feature-engineered LazyFrame at comment level
    """
    # Post-level comment stats (for context)
    post_comment_stats = (
        comments_lf
        .group_by("id_post")
        .agg([
            pl.count().alias("post_total_comments"),
            pl.col("rating").mean().alias("post_avg_comment_rating"),
            pl.col("rating").max().alias("post_max_comment_rating"),
            pl.col("description").str.len_chars().mean().alias("post_avg_comment_length"),
        ])
    )

    # Enrich posts with comment stats + submolt
    posts_enriched = (
        posts_lf
        .join(post_comment_stats, on="id_post", how="left")
        .join(
            submolts_lf.select([
                "id_submolt",
                pl.col("name").alias("submolt_name"),
            ]),
            on="id_submolt", how="left",
        )
        .with_columns([
            pl.col("post_total_comments").fill_null(0),
            pl.col("post_avg_comment_rating").fill_null(0.0),
            pl.col("post_max_comment_rating").fill_null(0),
            pl.col("post_avg_comment_length").fill_null(0.0),
            pl.col("submolt_name").fill_null("unknown"),
        ])
        .select([
            "id_post",
            pl.col("id_user").alias("post_author_id"),
            pl.col("title").alias("post_title"),
            pl.col("rating").alias("post_rating"),
            pl.col("date").alias("post_date"),
            "id_submolt",
            "submolt_name",
            "post_total_comments",
            "post_avg_comment_rating",
            "post_max_comment_rating",
            "post_avg_comment_length",
            pl.col("title").str.len_chars().fill_null(0).alias("post_title_length"),
            pl.col("description").str.len_chars().fill_null(0).alias("post_description_length"),
        ])
    )

    # Comment author info
    comment_author = (
        users_lf
        .select([
            "id_user",
            pl.col("name").alias("comment_author_name"),
            pl.col("karma").alias("comment_author_karma"),
            pl.col("followers").alias("comment_author_followers"),
            pl.col("following").alias("comment_author_following"),
            (pl.col("followers") / (pl.col("following") + 1)).alias("comment_author_follower_ratio"),
            pl.col("description").str.len_chars().fill_null(0).alias("comment_author_desc_length"),
            pl.col("human_owner").is_not_null().cast(pl.Int32).alias("comment_author_has_human_owner"),
        ])
    )

    # Post author info
    post_author = (
        users_lf
        .select([
            pl.col("id_user").alias("post_author_id"),
            pl.col("name").alias("post_author_name"),
            pl.col("karma").alias("post_author_karma"),
        ])
    )

    # Build comment-level features
    comment_features = (
        comments_lf
        # Join with post info
        .join(posts_enriched, on="id_post", how="left")
        # Join with comment author
        .join(comment_author, on="id_user", how="left")
        # Join with post author
        .join(post_author, on="post_author_id", how="left")
        # Engineer comment-specific features
        .with_columns([
            pl.col("description").str.len_chars().fill_null(0).alias("comment_length"),
            pl.col("description").str.split(" ").list.len().fill_null(0).alias("comment_word_count"),
            (pl.col("description").str.len_chars() > 0).cast(pl.Int32).alias("has_comment_text"),
            # Fill nulls from joins
            pl.col("comment_author_name").fill_null("unknown"),
            pl.col("comment_author_karma").fill_null(0),
            pl.col("comment_author_followers").fill_null(0),
            pl.col("comment_author_following").fill_null(0),
            pl.col("comment_author_follower_ratio").fill_null(0.0),
            pl.col("comment_author_desc_length").fill_null(0),
            pl.col("comment_author_has_human_owner").fill_null(0),
            pl.col("post_title").fill_null(""),
            pl.col("post_rating").fill_null(0),
            pl.col("post_author_name").fill_null("unknown"),
            pl.col("post_author_karma").fill_null(0),
            pl.col("submolt_name").fill_null("unknown"),
        ])
        .select([
            # Comment base (5)
            "id_comment", "id_user", "id_post", "rating", "date",
            # Comment features (3)
            "comment_length", "comment_word_count", "has_comment_text",
            # Comment author (7)
            "comment_author_name", "comment_author_karma",
            "comment_author_followers", "comment_author_following",
            "comment_author_follower_ratio", "comment_author_desc_length",
            "comment_author_has_human_owner",
            # Post context (8)
            "post_title", "post_rating", "post_date",
            "post_title_length", "post_description_length",
            "post_total_comments", "post_avg_comment_rating", "post_max_comment_rating",
            # Post author (2)
            "post_author_name", "post_author_karma",
            # Submolt (2)
            "id_submolt", "submolt_name",
        ])
    )

    return comment_features


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
