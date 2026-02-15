"""Database connection management for SQLite and PostgreSQL."""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, Union

from config.settings import settings

logger = logging.getLogger(__name__)


def get_sqlite_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Create a SQLite connection with proper configuration.

    Args:
        db_path: Path to database file. Uses settings default if None.

    Returns:
        Configured SQLite connection
    """
    if db_path is None:
        db_path = settings.project_root / settings.db_path

    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")

    return conn


def get_postgres_connection():
    """Create a PostgreSQL connection.

    Returns:
        psycopg2 connection with RealDictCursor
    """
    import psycopg2
    from psycopg2.extras import RealDictCursor

    conn = psycopg2.connect(settings.postgres_url, cursor_factory=RealDictCursor)
    conn.autocommit = False
    return conn


@contextmanager
def get_connection(
    db_path: Optional[Path] = None,
) -> Generator[Union[sqlite3.Connection, "psycopg2.extensions.connection"], None, None]:
    """Context manager for database connections.

    Automatically selects SQLite or PostgreSQL based on settings.db_type.

    Args:
        db_path: Optional path override for SQLite database

    Yields:
        Database connection
    """
    if settings.db_type == "postgres":
        conn = get_postgres_connection()
    else:
        conn = get_sqlite_connection(db_path)

    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error("Database error: %s", e)
        raise
    finally:
        conn.close()


def init_database(db_path: Optional[Path] = None, schema_path: Optional[Path] = None) -> None:
    """Initialize the database with the schema.

    Args:
        db_path: Path to database file (SQLite only)
        schema_path: Path to SQL schema file
    """
    if settings.db_type == "postgres":
        if schema_path is None:
            schema_path = settings.project_root / "schema_postgres.sql"
    else:
        if schema_path is None:
            schema_path = settings.project_root / "schema.sql"

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    schema_sql = schema_path.read_text(encoding="utf-8")

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        if settings.db_type == "postgres":
            cursor.execute(schema_sql)
        else:
            conn.executescript(schema_sql)
        logger.info("Database initialized successfully (%s)", settings.db_type)


def check_database_exists(db_path: Optional[Path] = None) -> bool:
    """Check if the database exists and has tables.

    Args:
        db_path: Path to database file (SQLite only)

    Returns:
        True if database exists and has tables
    """
    if settings.db_type == "postgres":
        try:
            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'users')"
                )
                row = cursor.fetchone()
                # RealDictCursor returns dict
                if isinstance(row, dict):
                    return list(row.values())[0]
                return row[0]
        except Exception as e:
            logger.warning("PostgreSQL connection check failed: %s", e)
            return False
    else:
        if db_path is None:
            db_path = settings.project_root / settings.db_path

        if not db_path.exists():
            return False

        with get_connection(db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
            )
            return cursor.fetchone() is not None
