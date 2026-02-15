"""Database connection management.

Supports SQLite (local) and PostgreSQL (RDS/Glue).
Credentials for PostgreSQL come from environment variables:
  DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT (optional, default 5432).
"""

import logging
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional, Union

logger = logging.getLogger(__name__)

# Optional: avoid importing settings at module level for Glue (config may not be in path)
def _get_settings():
    try:
        from config.settings import settings
        return settings
    except ImportError:
        return None


def _use_postgres() -> bool:
    """Use PostgreSQL if connection env vars are set (e.g. in Glue)."""
    return bool(os.environ.get("DB_HOST") or os.environ.get("DB_NAME"))


def get_sqlite_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Create a SQLite connection with proper configuration.

    Args:
        db_path: Path to database file. Uses settings default if None.

    Returns:
        Configured SQLite connection
    """
    settings = _get_settings()
    if settings is None:
        raise RuntimeError("config.settings not available; set db_path explicitly.")
    if db_path is None:
        db_path = settings.project_root / settings.db_path

    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")

    return conn


def get_postgres_connection(
    host: Optional[str] = None,
    dbname: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    port: Optional[int] = None,
) -> Any:
    """Create a PostgreSQL connection (RDS).

    Args are read from env (DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT) if not passed.

    Returns:
        psycopg2 connection (or psycopg2.extensions.connection)
    """
    import psycopg2
    host = host or os.environ.get("DB_HOST", "localhost")
    dbname = dbname or os.environ.get("DB_NAME", "moltbook")
    user = user or os.environ.get("DB_USER", "postgres")
    password = password or os.environ.get("DB_PASSWORD", "")
    port = port or int(os.environ.get("DB_PORT", "5432"))
    conn = psycopg2.connect(
        host=host,
        dbname=dbname,
        user=user,
        password=password,
        port=port,
    )
    return conn


@contextmanager
def get_connection(
    db_path: Optional[Path] = None,
    *,
    use_postgres: Optional[bool] = None,
    pg_host: Optional[str] = None,
    pg_dbname: Optional[str] = None,
    pg_user: Optional[str] = None,
    pg_password: Optional[str] = None,
    pg_port: Optional[int] = None,
) -> Generator[Union[sqlite3.Connection, Any], None, None]:
    """Context manager for database connections (SQLite or PostgreSQL).

    If use_postgres is True or env DB_HOST/DB_NAME is set, uses PostgreSQL.

    Yields:
        Database connection (sqlite3 or psycopg2)
    """
    use_pg = use_postgres if use_postgres is not None else _use_postgres()
    if use_pg:
        conn = get_postgres_connection(
            host=pg_host, dbname=pg_dbname, user=pg_user, password=pg_password, port=pg_port
        )
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error("Database error: %s", e)
            raise
        finally:
            conn.close()
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
    """Initialize the database with the schema (SQLite or PostgreSQL)."""
    settings = _get_settings()
    project_root = Path(__file__).resolve().parent.parent.parent if settings is None else settings.project_root

    if _use_postgres():
        schema_path = schema_path or project_root / "schema_postgres.sql"
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        schema_sql = schema_path.read_text(encoding="utf-8")
        with get_connection(use_postgres=True) as conn:
            with conn.cursor() as cur:
                for stmt in schema_sql.split(";"):
                    stmt = stmt.strip()
                    if stmt and not stmt.startswith("--"):
                        cur.execute(stmt)
            conn.commit()
        logger.info("PostgreSQL database initialized successfully")
        return

    if schema_path is None:
        schema_path = project_root / "schema.sql"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    schema_sql = schema_path.read_text(encoding="utf-8")
    with get_connection(db_path) as conn:
        conn.executescript(schema_sql)
    logger.info("SQLite database initialized successfully")


def check_database_exists(db_path: Optional[Path] = None) -> bool:
    """Check if the database exists and has tables."""
    if _use_postgres():
        with get_connection(use_postgres=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM information_schema.tables WHERE table_name = 'users' LIMIT 1"
                )
                return cur.fetchone() is not None
    settings = _get_settings()
    if settings is None:
        return False
    path = db_path or settings.project_root / settings.db_path
    if not path.exists():
        return False
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
        )
        return cursor.fetchone() is not None
