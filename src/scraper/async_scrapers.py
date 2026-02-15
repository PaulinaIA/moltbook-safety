"""Async scraper orchestration for massive parallel scraping of moltbook.com."""

import asyncio
import logging
import time
from typing import List, Optional, Set

from config.settings import settings
from src.database.models import Comment, Post, SubMolt, User, UserSubMolt
from src.database.operations import DatabaseOperations
from src.scraper.async_base import AsyncBaseScraper
from src.scraper.parsers import (
    parse_user_profile,
    parse_submolt_page,
    parse_posts_from_page,
    parse_comments,
    parse_users_list,
    parse_submolt_list,
)

logger = logging.getLogger(__name__)


class AsyncMoltbookScraper:
    """Async orchestrator for massive parallel scraping of moltbook.com."""

    def __init__(
        self,
        db_ops: Optional[DatabaseOperations] = None,
        headless: bool = True,
        max_workers: int = 10,
        rate_limit: float = 0.5,
    ):
        self.db_ops = db_ops or DatabaseOperations()
        self.headless = headless
        self.max_workers = max_workers
        self.rate_limit = rate_limit
        self._scraper: Optional[AsyncBaseScraper] = None
        self.base_url = settings.base_url

    async def __aenter__(self):
        self._scraper = AsyncBaseScraper(
            headless=self.headless,
            max_workers=self.max_workers,
            rate_limit=self.rate_limit,
        )
        await self._scraper.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._scraper:
            await self._scraper.close()

    @property
    def scraper(self) -> AsyncBaseScraper:
        if self._scraper is None:
            raise RuntimeError("Scraper not initialized. Use async context manager.")
        return self._scraper

    # ─── Discovery ───────────────────────────────────────────────

    async def _create_discovery_page(self):
        """Create a fresh page for discovery (scroll-heavy) operations."""
        page = await self.scraper._context.new_page()
        return page

    async def discover_all_users(self, max_users: int = 50000, max_scrolls: int = 200, force: bool = True) -> List[str]:
        """Discover user URLs by scrolling the /u listing massively."""
        users_url = f"{self.base_url}/u"
        page = await self._create_discovery_page()

        try:
            logger.info("Discovering users from %s (max_scrolls=%d)", users_url, max_scrolls)
            await page.goto(users_url)
            try:
                await page.wait_for_selector("a[href^='/u/']", timeout=10000)
            except Exception:
                pass

            # Switch to Karma view
            try:
                await page.click("button:has-text('Karma')")
                await asyncio.sleep(2)
            except Exception:
                logger.warning("Could not switch to Karma view")

            # Massive scroll
            for i in range(max_scrolls):
                prev = await page.evaluate("document.body.scrollHeight")
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(1000)
                curr = await page.evaluate("document.body.scrollHeight")
                if curr == prev:
                    logger.info("Users: reached end after %d scrolls", i + 1)
                    break

            html = await page.content()
        finally:
            await page.close()

        users = parse_users_list(html)

        # In force mode, return ALL user URLs (re-scrape everything)
        known_users = set() if force else set(self.db_ops.get_user_names())

        urls = []
        for user in users:
            if len(urls) >= max_users:
                break
            username = user.get("name", "")
            if username and username not in known_users:
                urls.append(f"{self.base_url}/u/{username}")

        logger.info("Discovered %d user URLs (total on page: %d)", len(urls), len(users))
        return urls

    async def discover_all_submolts(self, max_submolts: int = 500, max_scrolls: int = 100, force: bool = True) -> List[str]:
        """Discover submolt URLs by scrolling the /m listing.

        In force mode (default), returns ALL submolt URLs even if already
        in the DB, so their posts and comments get scraped.
        """
        submolts_url = f"{self.base_url}/m"
        page = await self._create_discovery_page()

        try:
            logger.info("Discovering submolts from %s (max_scrolls=%d)", submolts_url, max_scrolls)
            await page.goto(submolts_url)
            try:
                await page.wait_for_selector("a[href^='/m/']", timeout=10000)
            except Exception:
                pass

            # Massive scroll
            for i in range(max_scrolls):
                prev = await page.evaluate("document.body.scrollHeight")
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(1000)
                curr = await page.evaluate("document.body.scrollHeight")
                if curr == prev:
                    logger.info("Submolts: reached end after %d scrolls", i + 1)
                    break

            html = await page.content()
        finally:
            await page.close()

        submolts = parse_submolt_list(html)

        # In force mode, return ALL submolt URLs to scrape their posts
        known_submolts = set() if force else set(self.db_ops.get_submolt_names())

        urls = []
        for submolt in submolts:
            if len(urls) >= max_submolts:
                break
            name = submolt.get("name", "")
            if name and name not in known_submolts:
                urls.append(f"{self.base_url}/m/{name}")

        logger.info("Discovered %d submolt URLs to scrape", len(urls))
        return urls

    # ─── Parallel scraping ───────────────────────────────────────

    async def scrape_users_parallel(self, user_urls: List[str]) -> List[User]:
        """Scrape user profiles in parallel batches."""
        logger.info("Scraping %d users in parallel...", len(user_urls))
        results = await self.scraper.fetch_many(user_urls, wait_selector="h1, h2")

        users = []
        for url, html in results:
            if html is None:
                continue
            try:
                username = url.split("/u/")[-1]
                user_data = parse_user_profile(html, username)
                user = User.from_scraped_data(**user_data)
                users.append(user)
            except Exception as e:
                logger.error("Failed to parse user from %s: %s", url, e)

        # Batch upsert
        if users:
            count = self.db_ops.upsert_many(users)
            logger.info("Saved %d users to database", count)

        return users

    async def _fetch_submolt_with_scroll(self, url: str, max_scrolls: int = 50) -> Optional[str]:
        """Fetch a submolt page with scrolling to load all posts."""
        page = await self._create_discovery_page()
        try:
            await self.scraper._rate_limiter.wait()
            await page.goto(url)
            try:
                await page.wait_for_selector("a[href^='/post/']", timeout=10000)
            except Exception:
                pass

            # Scroll to load all posts
            for i in range(max_scrolls):
                prev = await page.evaluate("document.body.scrollHeight")
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(1000)
                curr = await page.evaluate("document.body.scrollHeight")
                if curr == prev:
                    break

            html = await page.content()
            self.scraper._save_cache(url, html)
            return html
        except Exception as e:
            logger.error("Failed to fetch submolt %s: %s", url, e)
            return None
        finally:
            await page.close()

    async def scrape_submolts_parallel(
        self,
        submolt_urls: List[str],
        max_posts_per_submolt: int = 500,
        max_comments_per_post: int = 100,
    ) -> dict:
        """Scrape submolts with scrolling, then scrape their posts and comments."""
        logger.info("Scraping %d submolts (with scroll for posts)...", len(submolt_urls))

        submolts = []
        all_post_urls = []

        # Process submolts in parallel batches (with scroll)
        sem = asyncio.Semaphore(self.max_workers)

        async def _process_submolt(url):
            async with sem:
                return (url, await self._fetch_submolt_with_scroll(url))

        batch_size = self.max_workers
        for i in range(0, len(submolt_urls), batch_size):
            batch = submolt_urls[i:i + batch_size]
            tasks = [_process_submolt(url) for url in batch]
            results = await asyncio.gather(*tasks)

            for url, html in results:
                if html is None:
                    continue
                try:
                    name = url.split("/m/")[-1]
                    submolt_data = parse_submolt_page(html, name)
                    submolt = SubMolt.from_scraped_data(**submolt_data)
                    submolts.append(submolt)

                    posts_data = parse_posts_from_page(html, name, max_posts=max_posts_per_submolt)
                    for post_data in posts_data:
                        post_url = post_data.get("post_url")
                        if post_url:
                            all_post_urls.append((post_url, submolt.id_submolt, name, post_data))
                except Exception as e:
                    logger.error("Failed to parse submolt from %s: %s", url, e)

            logger.info("Processed submolts %d-%d / %d | Posts found so far: %d",
                        i, min(i + batch_size, len(submolt_urls)), len(submolt_urls), len(all_post_urls))

        # Save submolts
        if submolts:
            self.db_ops.upsert_many(submolts)
            logger.info("Saved %d submolts", len(submolts))

        # Pre-load all known users into memory (avoid individual DB queries)
        logger.info("Loading known users into memory...")
        known_user_names = {name.lower()[:12]: name for name in self.db_ops.get_user_names()}
        known_user_ids = {name: uid for uid, name in zip(self.db_ops.get_user_ids(), self.db_ops.get_user_names())}
        logger.info("Loaded %d known users", len(known_user_names))

        # Create posts and new users in memory first
        posts = []
        new_users = []
        post_urls_for_comments = []

        for post_url, submolt_id, submolt_name, post_data in all_post_urls:
            try:
                author_name = post_data.get("author_name")
                if not author_name:
                    continue

                author_key = author_name.lower()[:12]
                if author_key in known_user_names:
                    user_id = known_user_ids[known_user_names[author_key]]
                else:
                    new_user = User.from_scraped_data(name=author_name)
                    user_id = new_user.id_user
                    new_users.append(new_user)
                    known_user_names[author_key] = author_name
                    known_user_ids[author_name] = user_id

                id_post = post_url.split("/")[-1]
                post = Post.from_scraped_data(
                    id=id_post,
                    id_user=user_id,
                    title=post_data.get("title"),
                    description=post_data.get("description"),
                    id_submolt=submolt_id,
                    rating=post_data.get("rating", 0),
                    date=post_data.get("date"),
                    url=post_url,
                )
                posts.append(post)
                post_urls_for_comments.append((post_url, post.id_post))
            except Exception as e:
                logger.error("Failed to create post from %s: %s", post_url, e)

        # Batch save new users first (FK dependency)
        if new_users:
            self.db_ops.upsert_many(new_users)
            logger.info("Saved %d new users from post authors", len(new_users))

        # Batch save posts
        if posts:
            # Save in chunks of 500 to avoid huge transactions
            for i in range(0, len(posts), 500):
                chunk = posts[i:i + 500]
                self.db_ops.upsert_many(chunk)
            logger.info("Saved %d posts", len(posts))

        # Scrape profiles of new authors to fill in karma, description, etc.
        if new_users:
            new_user_urls = [f"{self.base_url}/u/{u.name}" for u in new_users if u.name]
            logger.info("Scraping %d new author profiles to fill user data...", len(new_user_urls))
            await self.scrape_users_parallel(new_user_urls)

        # Scrape comments from individual post pages in parallel
        if post_urls_for_comments:
            await self._scrape_comments_parallel(post_urls_for_comments, max_comments_per_post)

        return {
            "submolts": len(submolts),
            "posts": len(posts),
        }

    async def _scrape_comments_parallel(
        self,
        post_url_pairs: List[tuple],  # (url, post_id)
        max_comments: int = 100,
    ) -> List[Comment]:
        """Scrape comments from post pages in parallel."""
        urls = [pair[0] for pair in post_url_pairs]
        url_to_post_id = {pair[0]: pair[1] for pair in post_url_pairs}

        logger.info("Scraping comments from %d post pages in parallel...", len(urls))
        results = await self.scraper.fetch_many(urls, wait_selector="div.mt-6")

        # Pre-load all known users into memory (avoid individual DB queries)
        logger.info("Loading known users into memory for comments...")
        known_user_names = {name.lower()[:12]: name for name in self.db_ops.get_user_names()}
        known_user_ids = {name: uid for uid, name in zip(self.db_ops.get_user_ids(), self.db_ops.get_user_names())}
        logger.info("Loaded %d known users for comment resolution", len(known_user_names))

        all_comments = []
        new_users = []
        for url, html in results:
            if html is None:
                continue
            try:
                post_id = url_to_post_id[url]
                comments_data = parse_comments(html, post_id, max_comments=max_comments)

                for comment_data in comments_data:
                    author_name = comment_data.get("author_name")
                    if not author_name:
                        continue

                    author_key = author_name.lower()[:12]
                    if author_key in known_user_names:
                        user_id = known_user_ids[known_user_names[author_key]]
                    else:
                        new_user = User.from_scraped_data(name=author_name)
                        user_id = new_user.id_user
                        new_users.append(new_user)
                        known_user_names[author_key] = author_name
                        known_user_ids[author_name] = user_id

                    comment = Comment.from_scraped_data(
                        id_user=user_id,
                        id_post=post_id,
                        description=comment_data.get("description"),
                        rating=comment_data.get("rating", 0),
                        date=comment_data.get("date"),
                    )
                    all_comments.append(comment)
            except Exception as e:
                logger.error("Failed to parse comments from %s: %s", url, e)

        # Batch save new users from comment authors, then scrape their profiles
        if new_users:
            for i in range(0, len(new_users), 500):
                chunk = new_users[i:i + 500]
                self.db_ops.upsert_many(chunk)
            logger.info("Saved %d new users from comment authors", len(new_users))

            # Scrape profiles to fill in karma, description, etc.
            new_user_urls = [f"{self.base_url}/u/{u.name}" for u in new_users if u.name]
            logger.info("Scraping %d new comment author profiles...", len(new_user_urls))
            await self.scrape_users_parallel(new_user_urls)

        if all_comments:
            # Batch upsert in chunks of 500
            for i in range(0, len(all_comments), 500):
                chunk = all_comments[i:i + 500]
                self.db_ops.upsert_many(chunk)
            logger.info("Saved %d comments total", len(all_comments))

        return all_comments

    # ─── Main orchestrator ───────────────────────────────────────

    async def scrape_all_massive(
        self,
        max_users: int = 50000,
        max_submolts: int = 500,
        max_posts_per_submolt: int = 500,
        max_comments_per_post: int = 500,
        max_scrolls_users: int = 200,
        max_scrolls_submolts: int = 100,
        force_refresh: bool = False,
    ) -> dict:
        """Run the full massive scraping pipeline.

        Args:
            max_users: Maximum users to discover and scrape
            max_submolts: Maximum submolts to discover and scrape
            max_posts_per_submolt: Maximum posts to extract per submolt page
            max_comments_per_post: Maximum comments per post page
            max_scrolls_users: Scroll count for user discovery
            max_scrolls_submolts: Scroll count for submolt discovery
            force_refresh: Force refresh cached pages

        Returns:
            Dictionary with counts of scraped entities
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("STARTING MASSIVE SCRAPE PIPELINE")
        logger.info("Workers: %d | Rate limit: %.1fs", self.max_workers, self.rate_limit)
        logger.info("=" * 60)

        # Phase 1: Discover and scrape users
        logger.info("--- Phase 1: Users ---")
        user_urls = await self.discover_all_users(max_users=max_users, max_scrolls=max_scrolls_users)
        users = await self.scrape_users_parallel(user_urls)

        # Phase 2: Discover and scrape submolts + posts + comments
        logger.info("--- Phase 2: Submolts + Posts + Comments ---")
        submolt_urls = await self.discover_all_submolts(max_submolts=max_submolts, max_scrolls=max_scrolls_submolts)
        submolt_results = await self.scrape_submolts_parallel(
            submolt_urls,
            max_posts_per_submolt=max_posts_per_submolt,
            max_comments_per_post=max_comments_per_post,
        )

        # Final counts from database
        from src.database.models import Post, Comment, SubMolt
        total_users = self.db_ops.count(User)
        total_posts = self.db_ops.count(Post)
        total_comments = self.db_ops.count(Comment)
        total_submolts = self.db_ops.count(SubMolt)

        elapsed = time.time() - start_time
        results = {
            "new_users": len(users),
            "new_submolts": submolt_results["submolts"],
            "new_posts": submolt_results["posts"],
            "total_users": total_users,
            "total_posts": total_posts,
            "total_comments": total_comments,
            "total_submolts": total_submolts,
            "total_records": total_users + total_posts + total_comments,
            "elapsed_seconds": round(elapsed, 1),
        }

        logger.info("=" * 60)
        logger.info("SCRAPE COMPLETE in %.1f seconds", elapsed)
        logger.info("Total: %d users, %d posts, %d comments, %d submolts",
                     total_users, total_posts, total_comments, total_submolts)
        logger.info("Grand total records: %d", results["total_records"])
        logger.info("=" * 60)

        return results


async def run_massive_scraper(
    max_users: int = 50000,
    max_submolts: int = 500,
    max_posts_per_submolt: int = 500,
    max_comments_per_post: int = 500,
    max_workers: int = 10,
    rate_limit: float = 0.5,
    headless: bool = True,
) -> dict:
    """Convenience function to run the massive async scraper."""
    from src.database.connection import init_database, check_database_exists

    if not check_database_exists():
        init_database()

    async with AsyncMoltbookScraper(
        headless=headless,
        max_workers=max_workers,
        rate_limit=rate_limit,
    ) as scraper:
        return await scraper.scrape_all_massive(
            max_users=max_users,
            max_submolts=max_submolts,
            max_posts_per_submolt=max_posts_per_submolt,
            max_comments_per_post=max_comments_per_post,
        )
