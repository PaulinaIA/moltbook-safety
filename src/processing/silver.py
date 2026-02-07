"""Silver layer processing - data cleaning with Polars Lazy evaluation.

Silver layer responsibilities:
- Load raw data from SQLite
- Clean and validate data types
- Handle missing values
- Remove duplicates
- Output to Parquet format
"""

import logging
from pathlib import Path
from typing import Optional

import polars as pl

from config.settings import settings
from src.database.connection import get_connection

logger = logging.getLogger(__name__)


def load_table_to_lazy(table_name: str) -> pl.LazyFrame:
    """Load a database table into a Polars LazyFrame.

    Args:
        table_name: Name of the table to load

    Returns:
        LazyFrame with table contents
    """
    db_path = settings.project_root / settings.db_path

    # Read from SQLite using Polars
    query = f"SELECT * FROM {table_name}"
    df = pl.read_database_uri(
        query=query,
        uri=f"sqlite:///{db_path}",
    )

    logger.info("Loaded %d rows from %s", len(df), table_name)
    return df.lazy()


def clean_users(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Clean users data.

    Args:
        lf: Raw users LazyFrame

    Returns:
        Cleaned users LazyFrame
    """
    return (
        lf
        # Ensure proper types
        .with_columns([
            pl.col("karma").cast(pl.Int64).fill_null(0),
            pl.col("followers").cast(pl.Int64).fill_null(0),
            pl.col("following").cast(pl.Int64).fill_null(0),
            pl.col("name").str.strip_chars(),
            pl.col("description").str.strip_chars().fill_null(""),
            pl.col("human_owner").str.strip_chars(),
            pl.col("joined").str.strip_chars(),
        ])
        # Remove duplicates by id_user
        .unique(subset=["id_user"], keep="last")
        # Filter out invalid entries
        .filter(pl.col("name").str.len_chars() > 0)
    )


def clean_posts(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Clean posts data.

    Args:
        lf: Raw posts LazyFrame

    Returns:
        Cleaned posts LazyFrame
    """
    return (
        lf
        .with_columns([
            pl.col("rating").cast(pl.Int64).fill_null(0),
            pl.col("title").str.strip_chars().fill_null(""),
            pl.col("description").str.strip_chars().fill_null(""),
        ])
        .unique(subset=["id_post"], keep="last")
    )


def clean_comments(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Clean comments data.

    Args:
        lf: Raw comments LazyFrame

    Returns:
        Cleaned comments LazyFrame
    """
    return (
        lf
        .with_columns([
            pl.col("rating").cast(pl.Int64).fill_null(0),
            pl.col("description").str.strip_chars().fill_null(""),
        ])
        .unique(subset=["id_comment"], keep="last")
    )


def clean_submolts(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Clean submolts data.

    Args:
        lf: Raw submolts LazyFrame

    Returns:
        Cleaned submolts LazyFrame
    """
    return (
        lf
        .with_columns([
            pl.col("name").str.strip_chars(),
            pl.col("description").str.strip_chars().fill_null(""),
        ])
        .unique(subset=["id_submolt"], keep="last")
        .filter(pl.col("name").str.len_chars() > 0)
    )


def build_silver_layer(output_dir: Optional[Path] = None) -> dict:
    """Build the silver (cleaned) data layer.

    Args:
        output_dir: Output directory for Parquet files

    Returns:
        Dictionary with row counts per table
    """
    output_dir = output_dir or settings.silver_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Process users
    try:
        users_lf = load_table_to_lazy("users")
        users_clean = clean_users(users_lf)
        users_df = users_clean.collect()
        users_path = output_dir / "users.parquet"
        users_df.write_parquet(users_path)
        results["users"] = len(users_df)
        logger.info("Wrote %d users to %s", len(users_df), users_path)
    except Exception as e:
        logger.error("Failed to process users: %s", e)
        results["users"] = 0

    # Process posts
    try:
        posts_lf = load_table_to_lazy("posts")
        posts_clean = clean_posts(posts_lf)
        posts_df = posts_clean.collect()
        posts_path = output_dir / "posts.parquet"
        posts_df.write_parquet(posts_path)
        results["posts"] = len(posts_df)
        logger.info("Wrote %d posts to %s", len(posts_df), posts_path)
    except Exception as e:
        logger.error("Failed to process posts: %s", e)
        results["posts"] = 0

    # Process comments
    try:
        comments_lf = load_table_to_lazy("comments")
        comments_clean = clean_comments(comments_lf)
        comments_df = comments_clean.collect()
        comments_path = output_dir / "comments.parquet"
        comments_df.write_parquet(comments_path)
        results["comments"] = len(comments_df)
        logger.info("Wrote %d comments to %s", len(comments_df), comments_path)
    except Exception as e:
        logger.error("Failed to process comments: %s", e)
        results["comments"] = 0

    # Process submolts
    try:
        submolts_lf = load_table_to_lazy("sub_molt")
        submolts_clean = clean_submolts(submolts_lf)
        submolts_df = submolts_clean.collect()
        submolts_path = output_dir / "submolts.parquet"
        submolts_df.write_parquet(submolts_path)
        results["submolts"] = len(submolts_df)
        logger.info("Wrote %d submolts to %s", len(submolts_df), submolts_path)
    except Exception as e:
        logger.error("Failed to process submolts: %s", e)
        results["submolts"] = 0

    logger.info("Silver layer build complete: %s", results)
    return results
