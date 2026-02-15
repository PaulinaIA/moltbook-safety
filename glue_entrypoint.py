"""
AWS Glue Python Shell entrypoint for Moltbook ETL.

Setup:
  - Subir extra_libs.zip al bucket S3 (ej. s3://moltbook/extra_libs.zip) y configurar
    el job con "Python library path" apuntando a ese objeto.
  - Job parameters (Key = nombre sin --, Value = valor):
    DB_HOST, DB_NAME, DB_USER, DB_PASSWORD
    Opcionales: DB_PORT (default 5432), MAX_USERS, MAX_SUBMOLTS, MAX_POSTS, MAX_COMMENTS, BATCH_SIZE

  - DB_NAME=postgres es válido: es la base por defecto de RDS. El script crea las tablas
    (users, posts, etc.) en esa base si no existen. Cuando crees la base "moltbook",
    cambia el parámetro DB_NAME a moltbook.

Logging configurado para CloudWatch.
"""

import logging
import os
import sys

# Configure logging for CloudWatch (visible in Glue job run logs)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)


def main() -> None:
    # Resolve Glue job parameters (only required for DB); optional args via env
    args = {}
    try:
        from awsglue.utils import getResolvedOptions
        args = getResolvedOptions(
            sys.argv,
            ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"],
        )
    except Exception as e:
        logger.warning("getResolvedOptions failed (not in Glue?): %s. Using env vars.", e)
    # Optional DB_PORT and scrape params from job args (if provided) or env
    for key in ["DB_PORT", "MAX_USERS", "MAX_SUBMOLTS", "MAX_POSTS", "MAX_COMMENTS", "BATCH_SIZE"]:
        if key not in args and os.environ.get(key):
            args[key] = os.environ[key]

    # Map job args to environment so database and scraper use them
    env_map = {
        "DB_HOST": args.get("DB_HOST", os.environ.get("DB_HOST", "")),
        "DB_NAME": args.get("DB_NAME", os.environ.get("DB_NAME", "")),
        "DB_USER": args.get("DB_USER", os.environ.get("DB_USER", "")),
        "DB_PASSWORD": args.get("DB_PASSWORD", os.environ.get("DB_PASSWORD", "")),
        "DB_PORT": args.get("DB_PORT", os.environ.get("DB_PORT", "5432")),
    }
    for k, v in env_map.items():
        if v:
            os.environ[k] = str(v)

    if not os.environ.get("DB_HOST") or not os.environ.get("DB_NAME"):
        logger.error("DB_HOST and DB_NAME (or job args) are required.")
        sys.exit(1)

    # Optional scrape limits (defaults for Glue)
    max_users = int(args.get("MAX_USERS", os.environ.get("MAX_USERS", "100")))
    max_submolts = int(args.get("MAX_SUBMOLTS", os.environ.get("MAX_SUBMOLTS", "50")))
    max_posts = int(args.get("MAX_POSTS", os.environ.get("MAX_POSTS", "500")))
    max_comments = int(args.get("MAX_COMMENTS", os.environ.get("MAX_COMMENTS", "1000")))
    batch_size = int(args.get("BATCH_SIZE", os.environ.get("BATCH_SIZE", "5000")))

    logger.info(
        "Starting Moltbook ETL: host=%s db=%s max_users=%s max_submolts=%s batch_size=%s",
        os.environ.get("DB_HOST"),
        os.environ.get("DB_NAME"),
        max_users,
        max_submolts,
        batch_size,
    )

    # Ensure project root is on path (Glue runs from job script dir)
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.database.operations import DatabaseOperations
    from src.scraper.scrapers import MoltbookScraper

    db_ops = DatabaseOperations(use_postgres=True, chunk_size=batch_size)
    db_ops.ensure_tables()

    with MoltbookScraper(db_ops=db_ops, headless=True) as scraper:
        result = scraper.scrape_all(
            max_users=max_users,
            max_submolts=max_submolts,
            max_posts=max_posts,
            max_comments=max_comments,
            force_refresh=False,
        )

    logger.info("Moltbook ETL finished: %s", result)


if __name__ == "__main__":
    main()
