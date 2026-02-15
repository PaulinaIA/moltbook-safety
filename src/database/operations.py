"""Database CRUD operations with upsert support for SQLite and PostgreSQL."""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from config.settings import settings
from src.database.connection import get_connection
from src.database.models import Comment, Post, SubMolt, User, UserSubMolt

logger = logging.getLogger(__name__)

T = TypeVar("T", User, Post, Comment, SubMolt, UserSubMolt)


class DatabaseOperations:
    """Database operations for all entity types. Supports SQLite and PostgreSQL."""

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

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize database operations.

        Args:
            db_path: Optional path to database file (SQLite only)
        """
        self.db_path = db_path
        self._is_postgres = settings.db_type == "postgres"

    @property
    def placeholder(self) -> str:
        """Return the correct placeholder for the current DB type."""
        return "%s" if self._is_postgres else "?"

    def _build_upsert_sql(self, entity_type: Type[T], columns: List[str]) -> str:
        """Build upsert SQL for the current DB type."""
        table = self.TABLE_MAPPING[entity_type]
        ph = self.placeholder
        placeholders = ", ".join(ph for _ in columns)
        column_names = ", ".join(columns)

        pk = self.PK_MAPPING[entity_type]
        pk_cols = pk if isinstance(pk, tuple) else (pk,)
        update_cols = [c for c in columns if c not in pk_cols]

        if self._is_postgres:
            conflict_cols = ", ".join(pk_cols)
            update_clause = ", ".join(f"{col} = EXCLUDED.{col}" for col in update_cols)
            sql = f"""
                INSERT INTO {table} ({column_names})
                VALUES ({placeholders})
                ON CONFLICT ({conflict_cols}) DO UPDATE SET {update_clause}
            """
        else:
            pk_condition = " AND ".join(f"{col} = excluded.{col}" for col in pk_cols)
            update_clause = ", ".join(f"{col} = excluded.{col}" for col in update_cols)
            sql = f"""
                INSERT INTO {table} ({column_names})
                VALUES ({placeholders})
                ON CONFLICT DO UPDATE SET {update_clause}
                WHERE {pk_condition}
            """
        return sql

    def upsert(self, entity: T) -> bool:
        """Insert or update a single entity."""
        entity_type = type(entity)
        table = self.TABLE_MAPPING.get(entity_type)
        if table is None:
            raise ValueError(f"Unknown entity type: {entity_type}")

        data = entity.to_dict()
        columns = list(data.keys())
        sql = self._build_upsert_sql(entity_type, columns)

        with get_connection(self.db_path) as conn:
            try:
                cursor = conn.cursor()
                if self._is_postgres:
                    cursor.execute("SAVEPOINT single_upsert")
                cursor.execute(sql, list(data.values()))
                return True
            except Exception as e:
                if self._is_postgres:
                    cursor.execute("ROLLBACK TO SAVEPOINT single_upsert")
                logger.error("Upsert failed for %s: %s", entity_type.__name__, e)
                return False

    def upsert_many(self, entities: List[T]) -> int:
        """Upsert multiple entities in a batch. Uses true batch operations first, falls back to row-by-row."""
        if not entities:
            return 0

        entity_type = type(entities[0])
        table = self.TABLE_MAPPING.get(entity_type)
        if table is None:
            raise ValueError(f"Unknown entity type: {entity_type}")

        data_list = [e.to_dict() for e in entities]
        columns = list(data_list[0].keys())
        sql = self._build_upsert_sql(entity_type, columns)
        values_list = [list(data.values()) for data in data_list]

        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()

            # Try true batch first
            try:
                if self._is_postgres:
                    import psycopg2.extras
                    psycopg2.extras.execute_batch(cursor, sql, values_list, page_size=1000)
                else:
                    cursor.executemany(sql, values_list)
                logger.info("Batch upserted %d %s records", len(entities), table)
                return len(entities)
            except Exception as e:
                logger.warning("Batch upsert failed, falling back to row-by-row: %s", e)
                if self._is_postgres:
                    cursor.execute("ROLLBACK")
                    conn.commit()

            # Fallback: row-by-row with savepoints
            success_count = 0
            cursor = conn.cursor()
            for i, values in enumerate(values_list):
                try:
                    if self._is_postgres:
                        cursor.execute(f"SAVEPOINT sp_{i}")
                    cursor.execute(sql, values)
                    success_count += 1
                except Exception as e:
                    if self._is_postgres:
                        cursor.execute(f"ROLLBACK TO SAVEPOINT sp_{i}")
                    logger.warning("Upsert failed for row: %s", e)

        logger.info("Upserted %d/%d %s records (row-by-row)", success_count, len(entities), table)
        return success_count

    def get_by_id(
        self, entity_type: Type[T], id_value: Union[str, tuple]
    ) -> Optional[Dict[str, Any]]:
        """Get an entity by its primary key."""
        table = self.TABLE_MAPPING.get(entity_type)
        pk = self.PK_MAPPING.get(entity_type)
        ph = self.placeholder

        if table is None or pk is None:
            raise ValueError(f"Unknown entity type: {entity_type}")

        if isinstance(pk, tuple):
            conditions = " AND ".join(f"{col} = {ph}" for col in pk)
            values = id_value if isinstance(id_value, tuple) else (id_value,)
        else:
            conditions = f"{pk} = {ph}"
            values = (id_value,)

        sql = f"SELECT * FROM {table} WHERE {conditions}"

        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, values)
            row = cursor.fetchone()
            if row is None:
                return None
            return dict(row)

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

        ph = self.placeholder
        sql = f"SELECT * FROM {table}"
        params: List[Any] = []

        if limit is not None:
            sql += f" LIMIT {ph} OFFSET {ph}"
            params = [limit, offset]

        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]

    def count(self, entity_type: Type[T]) -> int:
        """Count entities of a type."""
        table = self.TABLE_MAPPING.get(entity_type)
        if table is None:
            raise ValueError(f"Unknown entity type: {entity_type}")

        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            row = cursor.fetchone()
            if isinstance(row, dict):
                return list(row.values())[0]
            return row[0]

    def exists(self, entity_type: Type[T], id_value: Union[str, tuple]) -> bool:
        """Check if an entity exists."""
        return self.get_by_id(entity_type, id_value) is not None

    def get_user_ids(self) -> List[str]:
        """Get all user IDs for incremental scraping."""
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id_user FROM users")
            rows = cursor.fetchall()
            return [dict(row)["id_user"] if isinstance(row, dict) else row[0] for row in rows]

    def get_user_names(self) -> List[str]:
        """Get all usernames for URL discovery."""
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM users")
            rows = cursor.fetchall()
            return [dict(row)["name"] if isinstance(row, dict) else row[0] for row in rows]

    def get_incomplete_user_names(self) -> List[str]:
        """Get usernames of users with incomplete profiles (no description or karma=0)."""
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM users WHERE name IS NOT NULL "
                "AND (description IS NULL OR karma = 0 OR karma IS NULL)"
            )
            rows = cursor.fetchall()
            return [dict(row)["name"] if isinstance(row, dict) else row[0] for row in rows]

    def get_submolt_names(self) -> List[str]:
        """Get all submolt names for URL discovery."""
        with get_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sub_molt")
            rows = cursor.fetchall()
            return [dict(row)["name"] if isinstance(row, dict) else row[0] for row in rows]
