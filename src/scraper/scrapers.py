"""Main scraper orchestration for moltbook.com.

Uses batch buffer: entities are accumulated and written in chunks (e.g. every 1,000–5,000)
via bulk_upsert to avoid per-record saves and memory spikes. Flush can run in a thread
via asyncio for non-blocking writes.
"""

import asyncio
import logging
from collections import defaultdict
from typing import List, Optional, Set, Type, TypeVar

from src.database.models import Comment, Post, SubMolt, User, UserSubMolt
from src.database.operations import DatabaseOperations
from src.scraper.discovery import URLDiscovery
from src.scraper.parsers import (
    parse_user_profile,
    parse_submolt_page,
    parse_posts_from_page,
    parse_comments,
)

logger = logging.getLogger(__name__)

# Default batch size for buffer (flush every N records per entity type)
DEFAULT_BATCH_SIZE = 1000

T = TypeVar("T", User, Post, Comment, SubMolt, UserSubMolt)


class BatchWriter:
    """Buffers entities by type and flushes in chunks via bulk_upsert (no per-record save)."""

    def __init__(self, db_ops: DatabaseOperations, batch_size: int = DEFAULT_BATCH_SIZE):
        self.db_ops = db_ops
        self.batch_size = batch_size
        self._buffers: dict = defaultdict(list)

    def add(self, entity: T) -> None:
        """Add entity to buffer; flush when buffer size reaches batch_size."""
        t = type(entity)
        self._buffers[t].append(entity)
        if len(self._buffers[t]) >= self.batch_size:
            self._flush_type(t)

    def _flush_type(self, entity_type: Type[T]) -> int:
        """Flush buffer for one entity type (bulk_upsert). Returns count written."""
        buf = self._buffers[entity_type]
        if not buf:
            return 0
        chunk = list(buf)
        self._buffers[entity_type] = []
        try:
            return self._run_bulk_upsert(chunk)
        except Exception as e:
            logger.exception("Bulk upsert failed for %s: %s", entity_type.__name__, e)
            raise

    def _run_bulk_upsert(self, entities: List[T]) -> int:
        """Run bulk_upsert in a thread via asyncio so the scraper thread is not blocked."""
        if not entities:
            return 0
        try:
            return asyncio.run(asyncio.to_thread(self.db_ops.bulk_upsert, entities))
        except RuntimeError:
            return self.db_ops.bulk_upsert(entities)

    def flush_type(self, entity_type: Type[T]) -> int:
        """Flush buffer for given entity type (e.g. User before get_by_id)."""
        return self._flush_type(entity_type)

    def flush_all(self) -> None:
        """Flush all buffers (call at end of scrape_all or __exit__)."""
        for entity_type in list(self._buffers.keys()):
            if self._buffers[entity_type]:
                self._flush_type(entity_type)


class MoltbookScraper:
    """Orchestrates scraping of moltbook.com.

    Handles:
        - URL discovery from listing pages
        - User profile scraping
        - SubMolt scraping
        - Post and comment extraction
        - Database persistence with upserts
    """

    def __init__(
        self,
        db_ops: Optional[DatabaseOperations] = None,
        headless: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        use_playwright: bool = True,
    ):
        """Initialize moltbook scraper.

        Args:
            db_ops: Database operations instance (uses bulk upsert in chunks)
            headless: Run browser in headless mode (solo con use_playwright=True)
            batch_size: Flush buffer every N records (default 1000)
            use_playwright: Si False, usa requests + BeautifulSoup (sin navegador). Para Glue.
        """
        self.db_ops = db_ops or DatabaseOperations()
        self.headless = headless
        self.use_playwright = use_playwright
        self._batch = BatchWriter(
            self.db_ops,
            batch_size=getattr(self.db_ops, "chunk_size", None) or batch_size,
        )
        self._scraper = None
        self._discovery: Optional[URLDiscovery] = None

    def __enter__(self) -> "MoltbookScraper":
        """Enter context manager."""
        self.db_ops.ensure_tables()
        if self.use_playwright:
            from src.scraper.base import BaseScraper
            self._scraper = BaseScraper(headless=self.headless)
            self._scraper.start()
        else:
            from src.scraper.fetcher_requests import RequestsFetcher
            self._scraper = RequestsFetcher(rate_limit=1.0, request_timeout=30)
        self._discovery = URLDiscovery(self._scraper)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager; flush any remaining buffered records."""
        try:
            self._batch.flush_all()
        finally:
            if self._scraper is not None and hasattr(self._scraper, "close"):
                self._scraper.close()

    @property
    def scraper(self):
        """Get scraper instance."""
        if self._scraper is None:
            raise RuntimeError("Scraper not initialized. Use context manager.")
        return self._scraper

    @property
    def discovery(self) -> URLDiscovery:
        """Get URL discovery instance."""
        if self._discovery is None:
            raise RuntimeError("Discovery not initialized. Use context manager.")
        return self._discovery

    def scrape_users(
        self,
        max_users: int = 100,
        force_refresh: bool = False,
    ) -> List[User]:
        """Scrape user profiles.

        Args:
            max_users: Maximum number of users to scrape
            force_refresh: Force re-scrape even if cached

        Returns:
            List of scraped User objects
        """
        # Get known users for incremental scraping
        known_users: Set[str] = set(self.db_ops.get_user_names())
        logger.info("Found %d existing users in database", len(known_users))

        # Discover user URLs
        user_urls = self.discovery.discover_users(
            max_users=max_users,
            known_users=known_users if not force_refresh else None,
        )

        users: List[User] = []
        for url in user_urls:
            try:
                username = url.split("/u/")[-1]
                html = self.scraper.fetch_with_cache(
                    url,
                    force_refresh=force_refresh,
                    wait_selector="h1, h2",
                )

                user_data = parse_user_profile(html, username)
                user = User.from_scraped_data(**user_data)
                users.append(user)
                self._batch.add(user)
                logger.info("Scraped user: %s (karma=%d)", user.name, user.karma)

            except Exception as e:
                logger.error("Failed to scrape user from %s: %s", url, e)
                continue

        logger.info("Scraped %d users total", len(users))
        return users

    def scrape_submolts(
        self,
        max_submolts: int = 50,
        force_refresh: bool = False,
        max_posts: int = 10,
        max_comments: int = 10,
    ) -> List[SubMolt]:
        """Scrape submolt (community) pages.

        Args:
            max_submolts: Maximum number of submolts to scrape
            force_refresh: Force re-scrape even if cached

        Returns:
            List of scraped SubMolt objects
        """
        known_submolts: Set[str] = set(self.db_ops.get_submolt_names())
        logger.info("Found %d existing submolts in database", len(known_submolts))

        submolt_urls = self.discovery.discover_submolts(
            max_submolts=max_submolts,
            known_submolts=known_submolts if not force_refresh else None,
        )

        submolts: List[SubMolt] = []
        for url in submolt_urls:
            try:
                name = url.split("/m/")[-1]
                html = self.scraper.fetch_with_cache(
                    url,
                    force_refresh=force_refresh,
                    wait_selector="h1, h2",
                )

                submolt_data = parse_submolt_page(html, name)
                submolt = SubMolt.from_scraped_data(**submolt_data)
                submolts.append(submolt)
                self._batch.add(submolt)
                logger.info("Scraped submolt: %s", submolt.name)

                # Also scrape posts from this submolt
                self._scrape_posts_from_html(html, submolt.id_submolt, name, max_posts=max_posts, max_comments=max_comments)

            except Exception as e:
                logger.error("Failed to scrape submolt from %s: %s", url, e)
                continue

        logger.info("Scraped %d submolts total", len(submolts))
        return submolts

    def _scrape_posts_from_html(
        self,
        html: str,
        submolt_id: Optional[str],
        submolt_name: Optional[str],
        max_posts: Optional[int] = 10,
        max_comments: Optional[int] = 10,
    ) -> List[Post]:
        """Extract and persist posts from page HTML.

        Args:
            html: Page HTML content
            submolt_id: SubMolt ID if known
            submolt_name: SubMolt name if known

        Returns:
            List of Post objects
        """
        posts_data = parse_posts_from_page(html, submolt_name, max_posts=max_posts)
        posts: List[Post] = []

        self._batch.flush_type(User)
        for post_data in posts_data:
            author_name = post_data.get("author_name")
            if not author_name:
                continue

            user = self.db_ops.get_by_id(User, f"user_{author_name.lower()[:12]}")
            if user:
                user_id = user["id_user"]
            else:
                new_user = User.from_scraped_data(name=author_name)
                self._batch.add(new_user)
                user_id = new_user.id_user

            id_post = post_data.get("post_url").split("/")[-1]
            post = Post.from_scraped_data(
                id=id_post,
                id_user=user_id,
                title=post_data.get("title"),
                description=post_data.get("description"),
                id_submolt=submolt_id,
                rating=post_data.get("rating", 0),
                date=post_data.get("date"),
                url=post_data.get("post_url"),
            )
            posts.append(post)
            self._batch.add(post)


            try:
                url = post_data.get("post_url")
                name = url.split("/post/")[-1]
                html = self.scraper.fetch_with_cache(
                    url,
                    force_refresh=True,
                    wait_selector="div.mt-6",
                )
                if getattr(self.scraper, "use_browser", False):
                    self.scraper.scroll_to_load_all(max_scrolls=5)
                    html = self.scraper.page.content()
                self._scrape_comments_from_html(html=html, post_id=post.id_post, max_comments=max_comments)

            except Exception as e:
                logger.error("Failed to scrape submolt from %s: %s", url, e)
                continue

            

        logger.debug("Extracted %d posts from page", len(posts))
        return posts
    
    def _scrape_comments_from_html(
        self,
        html: str,
        post_id: str,
        max_comments: Optional[int] = 100,
    ) -> List[Comment]:
        """Extract and persist comments from page HTML.

        Args:
            html: Page HTML content
            post_id: Associated Post ID

        Returns:
            List of Comment objects
        """
        comments_data = parse_comments(html, post_id)
        comments: List[Comment] = []
        self._batch.flush_type(User)

        for comment_data in comments_data:
            author_name = comment_data.get("author_name")
            if not author_name:
                continue

            user_id = None
            user = self.db_ops.get_by_id(User, f"user_{author_name.lower()[:12]}")
            if user:
                user_id = user["id_user"]
            else:
                new_user = User.from_scraped_data(name=author_name)
                self._batch.add(new_user)
                user_id = new_user.id_user
            if not user_id:
                logger.warning("Could not determine user ID for comment author '%s'", author_name)
                continue
            comment = Comment.from_scraped_data(
                id_user=user_id,
                id_post=post_id,
                description=comment_data.get("description"),
                rating=comment_data.get("rating", 0),
                date=comment_data.get("date"),
            )
            comments.append(comment)
            self._batch.add(comment)

        logger.debug("Extracted %d comments from page", len(comments))
        return comments

    def scrape_all(
        self,
        max_users: int = 10,
        max_submolts: int = 5,
        max_posts: int = 10,
        max_comments: int = 100,
        force_refresh: bool = False,
    ) -> dict:
        """Run full scraping pipeline.

        Args:
            max_users: Maximum users to scrape
            max_submolts: Maximum submolts to scrape
            max_posts: Maximum posts to scrape (approximate)
            max_comments: Maximum comments to scrape (approximate)
            force_refresh: Force re-scrape cached pages

        Returns:
            Dictionary with counts of scraped entities
        """
        logger.info("Starting full scrape pipeline")

        users = self.scrape_users(max_users=max_users, force_refresh=force_refresh)
        submolts = self.scrape_submolts(max_submolts=max_submolts, force_refresh=force_refresh, max_posts=max_posts, max_comments=max_comments)

        # Count posts and comments from database
        post_count = self.db_ops.count(Post)
        comment_count = self.db_ops.count(Comment)

        results = {
            "users": len(users),
            "submolts": len(submolts),
            "posts": post_count,
            "comments": comment_count,
        }

        logger.info("Scrape complete: %s", results)
        return results


def run_scraper(
    max_users: int = 100,
    max_submolts: int = 50,
    max_posts: int = 500,
    max_comments: int = 1000,
    headless: bool = True,
) -> dict:
    """Convenience function to run the scraper (tables created via ensure_tables in __enter__).

    Args:
        max_users: Maximum users to scrape
        max_submolts: Maximum submolts to scrape
        max_posts: Maximum posts to scrape
        max_comments: Maximum comments to scrape
        headless: Run browser in headless mode

    Returns:
        Dictionary with scrape results
    """
    with MoltbookScraper(headless=headless) as scraper:
        return scraper.scrape_all(
            max_users=max_users,
            max_submolts=max_submolts,
            max_posts=max_posts,
            max_comments=max_comments,
        )
