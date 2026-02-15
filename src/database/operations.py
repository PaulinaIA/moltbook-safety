"""Database CRUD operations with bulk upsert and PostgreSQL/RDS support.

- Ensures tables exist on startup (creates from schema if missing).
- Bulk upsert in chunks to avoid memory and connection saturation.
- Compatible with SQLite (local) and PostgreSQL (Glue/RDS).
"""

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from src.database.connection import get_connection, init_database, _use_postgres
from src.database.models import Comment, Post, SubMolt, User, UserSubMolt

logger = logging.getLogger(__name__)

T = TypeVar("T", User, Post, Comment, SubMolt, UserSubMolt)

# Default batch size for bulk upsert (tune for db.t4g.micro)
DEFAULT_CHUNK_SIZE = 5000


class DatabaseOperations:
    """Database operations for all entity types with bulk upsert and table checks."""

    TABLE_MAPPING = {
        User: "users",
        Post: "posts",
        Comment: "comments",
        SubMolt: "sub_molt",
        UserSubMolt: "user_submolt",
    }

    PK_MAPPING = {
        User: "id_user",
        Post: "id_post",
        Comment: "id_comment",
        SubMolt: "id_submolt",
        UserSubMolt: ("id_user", "id_submolt"),
    }

    def __init__(
        self,
        db_path: Optional[Path] = None,
        use_postgres: Optional[bool] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        """Initialize database operations.

        Args:
            db_path: Optional path (SQLite only).
            use_postgres: Force PostgreSQL if True; if None, uses env DB_HOST/DB_NAME.
            chunk_size: Batch size for bulk upsert (default 5000).
        """
        self.db_path = db_path
        self._use_postgres = use_postgres if use_postgres is not None else _use_postgres()
        self.chunk_size = chunk_size

    def _conn_kw(self) -> dict:
        if self._use_postgres:
            return {"use_postgres": True}
        return {"db_path": self.db_path}

    def ensure_tables(self) -> None:
        """Create tables if they do not exist (using existing schema)."""
        if self._use_postgres:
            with get_connection(use_postgres=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT 1 FROM information_schema.tables WHERE table_name = 'users' LIMIT 1"
                    )
                    if cur.fetchone() is not None:
                        logger.debug("PostgreSQL tables already exist")
                        return
            init_database()
            logger.info("PostgreSQL tables created")
        else:
            from src.database.connection import check_database_exists
            if check_database_exists(self.db_path):
                logger.debug("SQLite tables already exist")
                return
            init_database(self.db_path)
            logger.info("SQLite tables created")

    def upsert(self, entity: T) -> bool:
        """Insert or update a single entity."""
        entity_type = type(entity)
        table = self.TABLE_MAPPING.get(entity_type)
        if table is None:
            raise ValueError(f"Unknown entity type: {entity_type}")

        data = entity.to_dict()
        columns = list(data.keys())
        column_names = ", ".join(columns)
        pk = self.PK_MAPPING[entity_type]
        pk_cols = (pk,) if isinstance(pk, str) else pk
        update_cols = [c for c in columns if c not in pk_cols]
        update_clause = ", ".join(f"{col} = excluded.{col}" for col in update_cols)

        if self._use_postgres:
            placeholders = ", ".join("%s" for _ in columns)
            sql = f"""
                INSERT INTO {table} ({column_names})
                VALUES ({placeholders})
                ON CONFLICT ({", ".join(pk_cols)}) DO UPDATE SET {update_clause}
            """
            with get_connection(**self._conn_kw()) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, list(data.values()))
            return True
        else:
            placeholders = ", ".join("?" for _ in columns)
            pk_condition = (
                " AND ".join(f"{col} = excluded.{col}" for col in pk_cols)
                if isinstance(pk, tuple) else f"{pk} = excluded.{pk}"
            )
            sql = f"""
                INSERT INTO {table} ({column_names})
                VALUES ({placeholders})
                ON CONFLICT DO UPDATE SET {update_clause}
            """
            if "DO UPDATE SET" in sql and "WHERE" not in sql:
                sql = f"{sql.rstrip()} WHERE {pk_condition}"
            with get_connection(**self._conn_kw()) as conn:
                conn.execute(sql, list(data.values()))
            return True

    def bulk_upsert(
        self,
        entities: List[T],
        chunk_size: Optional[int] = None,
    ) -> int:
        """Upsert multiple entities in chunks (batch load).

        Uses execute_values (PostgreSQL) or executemany (SQLite) per chunk
        to avoid saturating memory and connection.
        """
        if not entities:
            return 0
        size = chunk_size or self.chunk_size
        entity_type = type(entities[0])
        table = self.TABLE_MAPPING.get(entity_type)
        if table is None:
            raise ValueError(f"Unknown entity type: {entity_type}")

        data_list = [e.to_dict() for e in entities]
        columns = list(data_list[0].keys())
        column_names = ", ".join(columns)
        pk = self.PK_MAPPING[entity_type]
        pk_cols = (pk,) if isinstance(pk, str) else pk
        update_cols = [c for c in columns if c not in pk_cols]
        update_clause = ", ".join(f"{col} = excluded.{col}" for col in update_cols)
        conflict_target = ", ".join(pk_cols)

        total = 0
        for i in range(0, len(data_list), size):
            chunk = data_list[i : i + size]
            rows = [list(d.values()) for d in chunk]

            if self._use_postgres:
                total += self._bulk_upsert_pg(
                    table, columns, conflict_target, update_clause, rows
                )
            else:
                total += self._bulk_upsert_sqlite(
                    table, columns, pk, update_clause, chunk
                )
        logger.info("Bulk upserted %d/%d %s in chunks of %d", total, len(entities), table, size)
        return total

    def _bulk_upsert_pg(
        self,
        table: str,
        columns: List[str],
        conflict_target: str,
        update_clause: str,
        rows: List[List[Any]],
    ) -> int:
        try:
            from psycopg2.extras import execute_values
        except ImportError:
            # Fallback: simple executemany with INSERT ... ON CONFLICT
            sql = (
                f"INSERT INTO {table} ({', '.join(columns)}) VALUES %s "
                f"ON CONFLICT ({conflict_target}) DO UPDATE SET {update_clause}"
            )
            with get_connection(**self._conn_kw()) as conn:
                with conn.cursor() as cur:
                    for row in rows:
                        placeholders = ", ".join("%s" for _ in row)
                        cur.execute(
                            f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders}) "
                            f"ON CONFLICT ({conflict_target}) DO UPDATE SET {update_clause}",
                            row,
                        )
            return len(rows)

        template = "(" + ", ".join("%s" for _ in columns) + ")"
        sql = (
            f"INSERT INTO {table} ({', '.join(columns)}) VALUES %s "
            f"ON CONFLICT ({conflict_target}) DO UPDATE SET {update_clause}"
        )
        with get_connection(**self._conn_kw()) as conn:
            with conn.cursor() as cur:
                execute_values(cur, sql, rows, template=template, page_size=len(rows))
        return len(rows)

    def _bulk_upsert_sqlite(
        self,
        table: str,
        columns: List[str],
        pk: Union[str, tuple],
        update_clause: str,
        data_list: List[Dict[str, Any]],
    ) -> int:
        placeholders = ", ".join("?" for _ in columns)
        column_names = ", ".join(columns)
        pk_cols = (pk,) if isinstance(pk, str) else pk
        pk_condition = (
            " AND ".join(f"{col} = excluded.{col}" for col in pk_cols)
            if isinstance(pk, tuple) else f"{pk} = excluded.{pk}"
        )
        sql = f"""
            INSERT INTO {table} ({column_names})
            VALUES ({placeholders})
            ON CONFLICT DO UPDATE SET {update_clause} WHERE {pk_condition}
        """
        # SQLite ON CONFLICT DO UPDATE SET ... WHERE is not standard; SQLite uses
        # ON CONFLICT (col) DO UPDATE SET ... with no WHERE. Adjust:
        if isinstance(pk, tuple):
            sql = (
                f"INSERT INTO {table} ({column_names}) VALUES ({placeholders}) "
                f"ON CONFLICT ({', '.join(pk_cols)}) DO UPDATE SET {update_clause}"
            )
        else:
            sql = (
                f"INSERT INTO {table} ({column_names}) VALUES ({placeholders}) "
                f"ON CONFLICT ({pk}) DO UPDATE SET {update_clause}"
            )
        with get_connection(**self._conn_kw()) as conn:
            conn.executemany(sql, [list(d.values()) for d in data_list])
        return len(data_list)

    def get_by_id(
        self, entity_type: Type[T], id_value: Union[str, tuple]
    ) -> Optional[Dict[str, Any]]:
        """Get an entity by its primary key."""
        table = self.TABLE_MAPPING.get(entity_type)
        pk = self.PK_MAPPING.get(entity_type)
        if table is None or pk is None:
            raise ValueError(f"Unknown entity type: {entity_type}")

        if isinstance(pk, tuple):
            conditions = " AND ".join(f"{col} = %s" if self._use_postgres else f"{col} = ?" for col in pk)
            values = id_value if isinstance(id_value, tuple) else (id_value,)
        else:
            conditions = f"{pk} = %s" if self._use_postgres else f"{pk} = ?"
            values = (id_value,)

        sql = f"SELECT * FROM {table} WHERE {conditions}"
        if self._use_postgres:
            with get_connection(**self._conn_kw()) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, values)
                    row = cur.fetchone()
                    if row is None:
                        return None
                    colnames = [d[0] for d in cur.description]
                    return dict(zip(colnames, row))
        with get_connection(**self._conn_kw()) as conn:
            cursor = conn.execute(sql, values)
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all(
        self,
        entity_type: Type[T],
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get all entities of a type."""
        table = self.TABLE_MAPPING.get(entity_type)
        if table is None:
            raise ValueError(f"Unknown entity type: {entity_type}")
        sql = f"SELECT * FROM {table}"
        params: List[Any] = []
        if limit is not None:
            sql += " LIMIT %s OFFSET %s" if self._use_postgres else " LIMIT ? OFFSET ?"
            params = [limit, offset]
        if self._use_postgres:
            with get_connection(**self._conn_kw()) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
                    colnames = [d[0] for d in cur.description]
                    return [dict(zip(colnames, r)) for r in rows]
        with get_connection(**self._conn_kw()) as conn:
            cursor = conn.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]

    def count(self, entity_type: Type[T]) -> int:
        """Count entities of a type."""
        table = self.TABLE_MAPPING.get(entity_type)
        if table is None:
            raise ValueError(f"Unknown entity type: {entity_type}")
        sql = f"SELECT COUNT(*) FROM {table}"
        if self._use_postgres:
            with get_connection(**self._conn_kw()) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    return cur.fetchone()[0]
        with get_connection(**self._conn_kw()) as conn:
            cursor = conn.execute(sql)
            return cursor.fetchone()[0]

    def exists(self, entity_type: Type[T], id_value: Union[str, tuple]) -> bool:
        """Check if an entity exists."""
        return self.get_by_id(entity_type, id_value) is not None

    def get_user_ids(self) -> List[str]:
        """Get all user IDs for incremental scraping."""
        if self._use_postgres:
            with get_connection(**self._conn_kw()) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id_user FROM users")
                    return [r[0] for r in cur.fetchall()]
        with get_connection(**self._conn_kw()) as conn:
            cursor = conn.execute("SELECT id_user FROM users")
            return [row[0] for row in cursor.fetchall()]

    def get_user_names(self) -> List[str]:
        """Get all usernames for URL discovery."""
        if self._use_postgres:
            with get_connection(**self._conn_kw()) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT name FROM users")
                    return [r[0] for r in cur.fetchall()]
        with get_connection(**self._conn_kw()) as conn:
            cursor = conn.execute("SELECT name FROM users")
            return [row[0] for row in cursor.fetchall()]

    def get_submolt_names(self) -> List[str]:
        """Get all submolt names for URL discovery."""
        if self._use_postgres:
            with get_connection(**self._conn_kw()) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT name FROM sub_molt")
                    return [r[0] for r in cur.fetchall()]
        with get_connection(**self._conn_kw()) as conn:
            cursor = conn.execute("SELECT name FROM sub_molt")
            return [row[0] for row in cursor.fetchall()]
