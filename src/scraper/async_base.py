"""Async base scraper with Playwright, rate limiting, and concurrent page pool."""

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import Optional, List
from urllib.parse import urlparse

from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright

from config.settings import settings

logger = logging.getLogger(__name__)


class AsyncRateLimiter:
    """Async rate limiter to respect website load."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last_request_time: float = 0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self._last_request_time = time.time()


class PageWorker:
    """A single browser page worker for parallel scraping."""

    def __init__(self, context: BrowserContext, worker_id: int):
        self.context = context
        self.worker_id = worker_id
        self.page: Optional[Page] = None

    async def start(self) -> None:
        self.page = await self.context.new_page()

    async def close(self) -> None:
        if self.page:
            await self.page.close()

    async def ensure_page(self) -> None:
        """Ensure the page is open, recreate if closed."""
        if self.page is None or self.page.is_closed():
            logger.info("[Worker %d] Recreating closed page", self.worker_id)
            self.page = await self.context.new_page()

    async def fetch_page(
        self,
        url: str,
        wait_selector: Optional[str] = None,
        wait_time: int = 500,
    ) -> str:
        """Fetch a page with JavaScript rendering."""
        await self.ensure_page()
        logger.debug("[Worker %d] Fetching: %s", self.worker_id, url)
        await self.page.goto(url, wait_until="domcontentloaded")

        if wait_selector:
            try:
                await self.page.wait_for_selector(wait_selector, timeout=3000)
            except Exception:
                pass

        await self.page.wait_for_timeout(wait_time)
        html = await self.page.content()
        logger.debug("[Worker %d] Fetched %d bytes from %s", self.worker_id, len(html), url)
        return html

    async def scroll_to_load_all(self, max_scrolls: int = 200, scroll_delay: int = 1000) -> None:
        """Scroll page to trigger lazy loading."""
        for i in range(max_scrolls):
            previous_height = await self.page.evaluate("document.body.scrollHeight")
            await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await self.page.wait_for_timeout(scroll_delay)
            new_height = await self.page.evaluate("document.body.scrollHeight")

            if new_height == previous_height:
                logger.debug("[Worker %d] Reached end after %d scrolls", self.worker_id, i + 1)
                break


class AsyncBaseScraper:
    """Async base scraper with a pool of browser page workers.

    Features:
        - Multiple concurrent browser pages (workers)
        - Async rate limiting with semaphore
        - HTML caching for incremental scraping
        - Retry logic with exponential backoff
    """

    def __init__(
        self,
        headless: bool = True,
        max_workers: int = 10,
        rate_limit: Optional[float] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.headless = headless
        self.max_workers = max_workers
        self.rate_limit = rate_limit or settings.rate_limit_seconds
        self.cache_dir = cache_dir or settings.raw_dir

        self._rate_limiter = AsyncRateLimiter(self.rate_limit)
        self._semaphore = asyncio.Semaphore(max_workers)
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._workers: List[PageWorker] = []

    async def start(self) -> None:
        """Start Playwright browser and create worker pool."""
        logger.info("Starting async browser (headless=%s, workers=%d)", self.headless, self.max_workers)
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self._context = await self._browser.new_context(
            user_agent=settings.user_agent,
        )

        # Block images, CSS, fonts, media to speed up page loads
        await self._context.route("**/*.{png,jpg,jpeg,gif,svg,ico,woff,woff2,ttf,eot,mp4,webm}", lambda route: route.abort())
        await self._context.route("**/*.css", lambda route: route.abort())

        for i in range(self.max_workers):
            worker = PageWorker(self._context, i)
            await worker.start()
            self._workers.append(worker)

        logger.info("Created %d page workers", len(self._workers))

    async def close(self) -> None:
        """Close all workers and browser."""
        for worker in self._workers:
            await worker.close()
        self._workers.clear()

        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("Async browser closed")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def _get_cache_filename(self, url: str) -> str:
        """Generate cache filename from URL."""
        parsed = urlparse(url)
        path_safe = parsed.path.replace("/", "_").strip("_") or "index"
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        return f"{path_safe}_{url_hash}.html"

    def _load_cached(self, url: str) -> Optional[str]:
        """Load cached HTML if exists."""
        filename = self._get_cache_filename(url)
        filepath = self.cache_dir / filename
        if filepath.exists():
            return filepath.read_text(encoding="utf-8")
        return None

    def _save_cache(self, url: str, html: str) -> None:
        """Save HTML to cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        filename = self._get_cache_filename(url)
        filepath = self.cache_dir / filename
        filepath.write_text(html, encoding="utf-8")

    async def fetch_url(
        self,
        worker: PageWorker,
        url: str,
        wait_selector: Optional[str] = None,
        force_refresh: bool = False,
        wait_time: Optional[int] = None,
    ) -> str:
        """Fetch a URL with rate limiting, caching, and retry."""
        if not force_refresh:
            cached = self._load_cached(url)
            if cached:
                return cached

        max_retries = 3
        for attempt in range(max_retries):
            try:
                await self._rate_limiter.wait()
                html = await worker.fetch_page(url, wait_selector=wait_selector, wait_time=wait_time or 500)
                self._save_cache(url, html)
                return html
            except Exception as e:
                if attempt < max_retries - 1:
                    retry_wait = 2 ** attempt
                    logger.warning("Retry %d for %s: %s (waiting %ds)", attempt + 1, url, e, retry_wait)
                    await asyncio.sleep(retry_wait)
                else:
                    logger.error("Failed after %d retries for %s: %s", max_retries, url, e)
                    raise

    async def fetch_many(
        self,
        urls: List[str],
        wait_selector: Optional[str] = None,
        force_refresh: bool = False,
        wait_time: Optional[int] = None,
    ) -> List[tuple]:
        """Fetch multiple URLs in parallel using the worker pool.

        Returns:
            List of (url, html) tuples. Failed URLs return (url, None).
        """
        async def _fetch_one(url: str, worker: PageWorker):
            async with self._semaphore:
                try:
                    html = await self.fetch_url(worker, url, wait_selector, force_refresh, wait_time)
                    return (url, html)
                except Exception as e:
                    logger.error("Failed to fetch %s: %s", url, e)
                    return (url, None)

        # Launch all tasks — semaphore controls concurrency
        tasks = [
            _fetch_one(url, self._workers[i % len(self._workers)])
            for i, url in enumerate(urls)
        ]
        logger.info("Fetching %d URLs with %d workers...", len(urls), self.max_workers)
        results = await asyncio.gather(*tasks)
        logger.info("Fetched all %d URLs", len(urls))
        return list(results)
