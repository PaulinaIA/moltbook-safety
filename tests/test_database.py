"""Unit tests for database operations."""

import pytest
import tempfile
from pathlib import Path

from src.database.models import (
    User,
    Post,
    Comment,
    SubMolt,
    UserSubMolt,
    generate_id,
)
from src.database.connection import get_connection, init_database
from src.database.operations import DatabaseOperations


class TestGenerateId:
    """Tests for ID generation function."""

    def test_generate_id_deterministic(self):
        """Same inputs should produce same ID."""
        id1 = generate_id("user", "test_name")
        id2 = generate_id("user", "test_name")
        assert id1 == id2

    def test_generate_id_different_inputs(self):
        """Different inputs should produce different IDs."""
        id1 = generate_id("user", "alice")
        id2 = generate_id("user", "bob")
        assert id1 != id2

    def test_generate_id_prefix(self):
        """ID should start with the prefix."""
        id1 = generate_id("user", "test")
        assert id1.startswith("user_")

        id2 = generate_id("post", "test")
        assert id2.startswith("post_")

    def test_generate_id_length(self):
        """ID should be prefix + 12 char hash."""
        id1 = generate_id("user", "test_user")
        # Format: prefix_hash12
        assert len(id1.split("_")[1]) == 12


class TestUserModel:
    """Tests for User model."""

    def test_from_scraped_data(self):
        """Test creating user from scraped data."""
        user = User.from_scraped_data(
            name="TestAgent",
            karma=1234,
            description="Test description",
            followers=100,
            following=50,
        )

        assert user.name == "TestAgent"
        assert user.karma == 1234
        assert user.followers == 100
        assert user.id_user.startswith("user_")

    def test_to_dict(self):
        """Test converting user to dictionary."""
        user = User.from_scraped_data(name="Test", karma=100)
        data = user.to_dict()

        assert "id_user" in data
        assert data["name"] == "Test"
        assert data["karma"] == 100
        assert "scraped_at" in data

    def test_case_insensitive_id(self):
        """User IDs should be case-insensitive."""
        user1 = User.from_scraped_data(name="TestAgent")
        user2 = User.from_scraped_data(name="testagent")
        assert user1.id_user == user2.id_user


class TestDatabaseOperations:
    """Tests for database CRUD operations."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            schema_path = Path(__file__).parent.parent / "schema.sql"
            init_database(db_path, schema_path)
            yield db_path

    def test_upsert_user(self, temp_db):
        """Test inserting and updating a user."""
        db_ops = DatabaseOperations(temp_db)

        user = User.from_scraped_data(name="test_user", karma=100)
        result = db_ops.upsert(user)
        assert result is True

        # Verify insert
        fetched = db_ops.get_by_id(User, user.id_user)
        assert fetched is not None
        assert fetched["name"] == "test_user"
        assert fetched["karma"] == 100

        # Update karma
        user.karma = 200
        db_ops.upsert(user)

        fetched = db_ops.get_by_id(User, user.id_user)
        assert fetched["karma"] == 200

    def test_upsert_many(self, temp_db):
        """Test batch upsert."""
        db_ops = DatabaseOperations(temp_db)

        users = [
            User.from_scraped_data(name=f"user_{i}", karma=i * 10)
            for i in range(5)
        ]

        count = db_ops.upsert_many(users)
        assert count == 5

        # Verify all inserted
        total = db_ops.count(User)
        assert total == 5

    def test_count(self, temp_db):
        """Test counting entities."""
        db_ops = DatabaseOperations(temp_db)

        # Initially empty
        assert db_ops.count(User) == 0

        # Add some users
        for i in range(3):
            user = User.from_scraped_data(name=f"user_{i}")
            db_ops.upsert(user)

        assert db_ops.count(User) == 3

    def test_exists(self, temp_db):
        """Test existence check."""
        db_ops = DatabaseOperations(temp_db)

        user = User.from_scraped_data(name="test")
        assert db_ops.exists(User, user.id_user) is False

        db_ops.upsert(user)
        assert db_ops.exists(User, user.id_user) is True

    def test_get_user_names(self, temp_db):
        """Test getting all usernames."""
        db_ops = DatabaseOperations(temp_db)

        names = ["alice", "bob", "charlie"]
        for name in names:
            user = User.from_scraped_data(name=name)
            db_ops.upsert(user)

        result = db_ops.get_user_names()
        assert set(result) == set(names)


class TestDeduplication:
    """Tests for deduplication behavior."""

    @pytest.fixture
    def temp_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            schema_path = Path(__file__).parent.parent / "schema.sql"
            init_database(db_path, schema_path)
            yield db_path

    def test_duplicate_user_upsert(self, temp_db):
        """Upserting same user twice should update, not duplicate."""
        db_ops = DatabaseOperations(temp_db)

        user1 = User.from_scraped_data(name="duplicate_test", karma=100)
        db_ops.upsert(user1)

        user2 = User.from_scraped_data(name="duplicate_test", karma=200)
        db_ops.upsert(user2)

        # Should only have one user
        assert db_ops.count(User) == 1

        # Should have updated karma
        fetched = db_ops.get_by_id(User, user1.id_user)
        assert fetched["karma"] == 200

    def test_post_deduplication(self, temp_db):
        """Posts with same URL should be deduplicated."""
        db_ops = DatabaseOperations(temp_db)

        # Create user first
        user = User.from_scraped_data(name="poster")
        db_ops.upsert(user)

        # Create same post twice
        post1 = Post.from_scraped_data(
            id_user=user.id_user,
            title="Same Post",
            url="/post/123",
        )
        db_ops.upsert(post1)

        post2 = Post.from_scraped_data(
            id_user=user.id_user,
            title="Same Post Updated",
            url="/post/123",
        )
        db_ops.upsert(post2)

        # Should have same ID
        assert post1.id_post == post2.id_post

        # Should only have one post
        assert db_ops.count(Post) == 1
