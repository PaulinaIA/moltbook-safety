"""Database CRUD operations with upsert support."""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from src.database.connection import get_connection
from src.database.models import Comment, Post, SubMolt, User, UserSubMolt

logger = logging.getLogger(__name__)

T = TypeVar("T", User, Post, Comment, SubMolt, UserSubMolt)


class DatabaseOperations:
    """Database operations for all entity types."""

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
            db_path: Optional path to database file
        """
        self.db_path = db_path

    def upsert(self, entity: T) -> bool:
        """Insert or update a single entity.

        Args:
            entity: Entity to upsert

        Returns:
            True if operation succeeded
        """
        entity_type = type(entity)
        table = self.TABLE_MAPPING.get(entity_type)
        if table is None:
            raise ValueError(f"Unknown entity type: {entity_type}")

        data = entity.to_dict()
        columns = list(data.keys())
        placeholders = ", ".join("?" for _ in columns)
        column_names = ", ".join(columns)

        # Build upsert statement
        pk = self.PK_MAPPING[entity_type]
        if isinstance(pk, tuple):
            pk_condition = " AND ".join(f"{col} = excluded.{col}" for col in pk)
        else:
            pk_condition = f"{pk} = excluded.{pk}"

        update_cols = [c for c in columns if c not in (pk if isinstance(pk, tuple) else [pk])]
        update_clause = ", ".join(f"{col} = excluded.{col}" for col in update_cols)

        sql = f"""
            INSERT INTO {table} ({column_names})
            VALUES ({placeholders})
            ON CONFLICT DO UPDATE SET {update_clause}
            WHERE {pk_condition}
        """

        with get_connection(self.db_path) as conn:
            try:
                conn.execute(sql, list(data.values()))
                return True
            except sqlite3.Error as e:
                logger.error("Upsert failed for %s: %s", entity_type.__name__, e)
                return False

    def upsert_many(self, entities: List[T]) -> int:
        """Upsert multiple entities in a batch.

        Args:
            entities: List of entities to upsert

        Returns:
            Number of successfully upserted entities
        """
        if not entities:
            return 0

        entity_type = type(entities[0])
        table = self.TABLE_MAPPING.get(entity_type)
        if table is None:
            raise ValueError(f"Unknown entity type: {entity_type}")

        data_list = [e.to_dict() for e in entities]
        columns = list(data_list[0].keys())
        placeholders = ", ".join("?" for _ in columns)
        column_names = ", ".join(columns)

        pk = self.PK_MAPPING[entity_type]
        if isinstance(pk, tuple):
            pk_cols = pk
        else:
            pk_cols = (pk,)

        update_cols = [c for c in columns if c not in pk_cols]
        update_clause = ", ".join(f"{col} = excluded.{col}" for col in update_cols)

        sql = f"""
            INSERT INTO {table} ({column_names})
            VALUES ({placeholders})
            ON CONFLICT DO UPDATE SET {update_clause}
        """

        success_count = 0
        with get_connection(self.db_path) as conn:
            for data in data_list:
                try:
                    conn.execute(sql, list(data.values()))
                    success_count += 1
                except sqlite3.Error as e:
                    logger.warning("Upsert failed for row: %s", e)

        logger.info("Upserted %d/%d %s records", success_count, len(entities), table)
        return success_count

    def get_by_id(
        self, entity_type: Type[T], id_value: Union[str, tuple]
    ) -> Optional[Dict[str, Any]]:
        """Get an entity by its primary key.

        Args:
            entity_type: Type of entity to retrieve
            id_value: Primary key value (string or tuple for composite)

        Returns:
            Entity data as dictionary or None
        """
        table = self.TABLE_MAPPING.get(entity_type)
        pk = self.PK_MAPPING.get(entity_type)

        if table is None or pk is None:
            raise ValueError(f"Unknown entity type: {entity_type}")

        if isinstance(pk, tuple):
            conditions = " AND ".join(f"{col} = ?" for col in pk)
            values = id_value if isinstance(id_value, tuple) else (id_value,)
        else:
            conditions = f"{pk} = ?"
            values = (id_value,)

        sql = f"SELECT * FROM {table} WHERE {conditions}"

        with get_connection(self.db_path) as conn:
            cursor = conn.execute(sql, values)
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all(
        self,
        entity_type: Type[T],
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Get all entities of a type.

        Args:
            entity_type: Type of entities to retrieve
            limit: Maximum number to return
            offset: Number of records to skip

        Returns:
            List of entity dictionaries
        """
        table = self.TABLE_MAPPING.get(entity_type)
        if table is None:
            raise ValueError(f"Unknown entity type: {entity_type}")

        sql = f"SELECT * FROM {table}"
        params: List[Any] = []

        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params = [limit, offset]

        with get_connection(self.db_path) as conn:
            cursor = conn.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]

    def count(self, entity_type: Type[T]) -> int:
        """Count entities of a type.

        Args:
            entity_type: Type of entities to count

        Returns:
            Number of entities
        """
        table = self.TABLE_MAPPING.get(entity_type)
        if table is None:
            raise ValueError(f"Unknown entity type: {entity_type}")

        with get_connection(self.db_path) as conn:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
            return cursor.fetchone()[0]

    def exists(self, entity_type: Type[T], id_value: Union[str, tuple]) -> bool:
        """Check if an entity exists.

        Args:
            entity_type: Type of entity
            id_value: Primary key value

        Returns:
            True if entity exists
        """
        return self.get_by_id(entity_type, id_value) is not None

    def get_user_ids(self) -> List[str]:
        """Get all user IDs for incremental scraping.

        Returns:
            List of user ID strings
        """
        with get_connection(self.db_path) as conn:
            cursor = conn.execute("SELECT id_user FROM users")
            return [row[0] for row in cursor.fetchall()]

    def get_user_names(self) -> List[str]:
        """Get all usernames for URL discovery.

        Returns:
            List of username strings
        """
        with get_connection(self.db_path) as conn:
            cursor = conn.execute("SELECT name FROM users")
            return [row[0] for row in cursor.fetchall()]

    def get_submolt_names(self) -> List[str]:
        """Get all submolt names for URL discovery.

        Returns:
            List of submolt name strings
        """
        with get_connection(self.db_path) as conn:
            cursor = conn.execute("SELECT name FROM sub_molt")
            return [row[0] for row in cursor.fetchall()]
