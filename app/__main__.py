"""CLI entry point for moltbook-karma pipeline.

Usage:
    python -m app scrape --max-users 100 --max-posts 500
    python -m app build
    python -m app train
"""

import logging
import sys
from pathlib import Path

import click

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the application.

    Args:
        verbose: Enable debug logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("playwright").setLevel(logging.WARNING)
    logging.getLogger("h2o").setLevel(logging.WARNING)


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
def cli(verbose: bool) -> None:
    """Moltbook Karma Data Engineering Pipeline.

    End-to-end pipeline for scraping moltbook.com, processing data,
    and training a karma prediction model.
    """
    setup_logging(verbose)
    settings.ensure_directories()


@cli.command()
@click.option("--max-users", default=100, help="Maximum users to scrape")
@click.option("--max-posts", default=500, help="Maximum posts to scrape")
@click.option("--max-comments", default=1000, help="Maximum comments to scrape")
@click.option("--force", is_flag=True, help="Force refresh cached pages")
@click.option("--headless/--no-headless", default=True, help="Run browser headless")
def scrape(
    max_users: int,
    max_posts: int,
    max_comments: int,
    force: bool,
    headless: bool,
) -> None:
    """Scrape data from moltbook.com.

    Discovers and scrapes user profiles, submolts, posts, and comments.
    Data is stored in SQLite database with incremental updates.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting scrape pipeline")

    from src.database.connection import init_database, check_database_exists
    from src.scraper.scrapers import MoltbookScraper

    # Initialize database if needed
    if not check_database_exists():
        logger.info("Initializing database")
        init_database()

    # Run scraper
    with MoltbookScraper(headless=headless) as scraper:
        results = scraper.scrape_all(
            max_users=max_users,
            max_posts=max_posts,
            max_comments=max_comments,
            force_refresh=force,
        )

    click.echo("\n--- Scrape Results ---")
    click.echo(f"Users:    {results.get('users', 0)}")
    click.echo(f"SubMolts: {results.get('submolts', 0)}")
    click.echo(f"Posts:    {results.get('posts', 0)}")
    click.echo(f"Comments: {results.get('comments', 0)}")


@cli.command()
def build() -> None:
    """Build silver and gold data layers.

    Processes raw data from SQLite into cleaned (silver) and
    feature-engineered (gold) Parquet files.
    """
    logger = logging.getLogger(__name__)
    logger.info("Building data layers")

    from src.processing.silver import build_silver_layer
    from src.processing.gold import build_gold_layer

    # Build silver layer
    click.echo("Building silver layer (cleaning)...")
    silver_results = build_silver_layer()

    click.echo("\n--- Silver Layer ---")
    for table, count in silver_results.items():
        click.echo(f"{table}: {count} records")

    # Build gold layer
    click.echo("\nBuilding gold layer (features)...")
    gold_results = build_gold_layer()

    click.echo("\n--- Gold Layer ---")
    click.echo(f"User features: {gold_results.get('user_features', 0)} records")
    click.echo(f"Feature columns: {gold_results.get('feature_columns', 0)}")


@cli.command()
@click.option("--max-models", default=10, help="Maximum models for AutoML")
@click.option("--max-time", default=300, help="Maximum training time (seconds)")
def train(max_models: int, max_time: int) -> None:
    """Train karma prediction model.

    Uses H2O AutoML to train a regression model predicting user karma
    from engineered features.
    """
    logger = logging.getLogger(__name__)
    logger.info("Training model")

    from src.models.trainer import train_model

    click.echo("Training karma prediction model...")
    click.echo(f"Max models: {max_models}, Max time: {max_time}s")

    results = train_model(
        max_models=max_models,
        max_runtime_secs=max_time,
    )

    click.echo("\n--- Training Results ---")
    click.echo(f"Model: {results.get('model_id', 'N/A')}")
    click.echo(f"MAE:   {results.get('mae', 0):.4f}")
    click.echo(f"RMSE:  {results.get('rmse', 0):.4f}")
    click.echo(f"R2:    {results.get('r2', 0):.4f}")
    click.echo(f"\nModel saved to: {results.get('model_path', 'N/A')}")
    click.echo(f"Predictions saved to: {results.get('predictions_path', 'N/A')}")


@cli.command()
def status() -> None:
    """Show current pipeline status.

    Displays database statistics and file statuses.
    """
    from src.database.connection import check_database_exists
    from src.database.operations import DatabaseOperations
    from src.database.models import User, Post, Comment, SubMolt

    click.echo("--- Pipeline Status ---\n")

    # Database status
    db_path = settings.project_root / settings.db_path
    if check_database_exists():
        click.echo(f"Database: {db_path} (exists)")
        db_ops = DatabaseOperations()
        click.echo(f"  Users:    {db_ops.count(User)}")
        click.echo(f"  Posts:    {db_ops.count(Post)}")
        click.echo(f"  Comments: {db_ops.count(Comment)}")
        click.echo(f"  SubMolts: {db_ops.count(SubMolt)}")
    else:
        click.echo(f"Database: {db_path} (not initialized)")

    # Silver layer status
    click.echo("")
    silver_dir = settings.silver_dir
    if silver_dir.exists():
        parquet_files = list(silver_dir.glob("*.parquet"))
        click.echo(f"Silver layer: {silver_dir} ({len(parquet_files)} files)")
    else:
        click.echo(f"Silver layer: {silver_dir} (not built)")

    # Gold layer status
    gold_dir = settings.gold_dir
    if gold_dir.exists():
        parquet_files = list(gold_dir.glob("*.parquet"))
        click.echo(f"Gold layer: {gold_dir} ({len(parquet_files)} files)")
    else:
        click.echo(f"Gold layer: {gold_dir} (not built)")

    # Model status
    models_dir = settings.models_dir
    if models_dir.exists():
        model_files = list(models_dir.iterdir())
        click.echo(f"Models: {models_dir} ({len(model_files)} artifacts)")
    else:
        click.echo(f"Models: {models_dir} (not trained)")


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
