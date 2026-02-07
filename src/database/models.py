"""Data models for moltbook entities using dataclasses."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import hashlib


def generate_id(prefix: str, *args: str) -> str:
    """Generate a deterministic ID from prefix and input values.

    Args:
        prefix: Entity prefix (e.g., 'user', 'post')
        *args: Values to hash for uniqueness

    Returns:
        Deterministic ID string in format: prefix_hash[:12]
    """
    content = "|".join(str(arg) for arg in args if arg)
    hash_value = hashlib.sha256(content.encode()).hexdigest()[:12]
    return f"{prefix}_{hash_value}"


def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.utcnow().isoformat()


@dataclass
class User:
    """User entity model."""

    id_user: str
    name: str
    karma: int = 0
    description: Optional[str] = None
    human_owner: Optional[str] = None
    joined: Optional[str] = None
    followers: int = 0
    following: int = 0
    scraped_at: str = field(default_factory=get_timestamp)

    @classmethod
    def from_scraped_data(
        cls,
        name: str,
        karma: int = 0,
        description: Optional[str] = None,
        human_owner: Optional[str] = None,
        joined: Optional[str] = None,
        followers: int = 0,
        following: int = 0,
    ) -> "User":
        """Create a User instance from scraped data.

        Args:
            name: Username (required)
            karma: User karma score
            description: User bio/description
            human_owner: Twitter handle of human owner
            joined: Join date
            followers: Follower count
            following: Following count

        Returns:
            User instance with generated ID
        """
        id_user = generate_id("user", name.lower())
        return cls(
            id_user=id_user,
            name=name,
            karma=karma,
            description=description,
            human_owner=human_owner,
            joined=joined,
            followers=followers,
            following=following,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for database insertion."""
        return {
            "id_user": self.id_user,
            "name": self.name,
            "karma": self.karma,
            "description": self.description,
            "human_owner": self.human_owner,
            "joined": self.joined,
            "followers": self.followers,
            "following": self.following,
            "scraped_at": self.scraped_at,
        }


@dataclass
class SubMolt:
    """SubMolt (community) entity model."""

    id_submolt: str
    name: str
    description: Optional[str] = None
    scraped_at: str = field(default_factory=get_timestamp)

    @classmethod
    def from_scraped_data(
        cls,
        name: str,
        description: Optional[str] = None,
    ) -> "SubMolt":
        """Create a SubMolt instance from scraped data."""
        id_submolt = generate_id("submolt", name.lower())
        return cls(
            id_submolt=id_submolt,
            name=name,
            description=description,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for database insertion."""
        return {
            "id_submolt": self.id_submolt,
            "name": self.name,
            "description": self.description,
            "scraped_at": self.scraped_at,
        }


@dataclass
class Post:
    """Post entity model."""

    id_post: str
    id_user: str
    title: Optional[str] = None
    description: Optional[str] = None
    id_submolt: Optional[str] = None
    rating: int = 0
    date: Optional[str] = None
    scraped_at: str = field(default_factory=get_timestamp)

    @classmethod
    def from_scraped_data(
        cls,
        id_user: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        id_submolt: Optional[str] = None,
        rating: int = 0,
        date: Optional[str] = None,
        url: Optional[str] = None,
    ) -> "Post":
        """Create a Post instance from scraped data."""
        # Use URL or title+user for ID generation
        id_source = url or f"{title or ''}-{id_user}"
        id_post = generate_id("post", id_source)
        return cls(
            id_post=id_post,
            id_user=id_user,
            title=title,
            description=description,
            id_submolt=id_submolt,
            rating=rating,
            date=date,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for database insertion."""
        return {
            "id_post": self.id_post,
            "id_user": self.id_user,
            "id_submolt": self.id_submolt,
            "title": self.title,
            "description": self.description,
            "rating": self.rating,
            "date": self.date,
            "scraped_at": self.scraped_at,
        }


@dataclass
class Comment:
    """Comment entity model."""

    id_comment: str
    id_user: str
    id_post: str
    description: Optional[str] = None
    date: Optional[str] = None
    rating: int = 0
    scraped_at: str = field(default_factory=get_timestamp)

    @classmethod
    def from_scraped_data(
        cls,
        id_user: str,
        id_post: str,
        description: Optional[str] = None,
        date: Optional[str] = None,
        rating: int = 0,
    ) -> "Comment":
        """Create a Comment instance from scraped data."""
        # Use content hash + user + post for dedup
        content_sample = (description or "")[:50]
        id_comment = generate_id("comment", id_user, id_post, content_sample)
        return cls(
            id_comment=id_comment,
            id_user=id_user,
            id_post=id_post,
            description=description,
            date=date,
            rating=rating,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for database insertion."""
        return {
            "id_comment": self.id_comment,
            "id_user": self.id_user,
            "id_post": self.id_post,
            "description": self.description,
            "date": self.date,
            "rating": self.rating,
            "scraped_at": self.scraped_at,
        }


@dataclass
class UserSubMolt:
    """Many-to-many relationship between users and submolts."""

    id_user: str
    id_submolt: str

    def to_dict(self) -> dict:
        """Convert to dictionary for database insertion."""
        return {
            "id_user": self.id_user,
            "id_submolt": self.id_submolt,
        }
