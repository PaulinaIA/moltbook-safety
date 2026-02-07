"""Base scraper with Playwright, rate limiting, and retry logic."""

import logging
import time
from pathlib import Path
from typing import Optional

from playwright.sync_api import Browser, Page, Playwright, sync_playwright
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from config.settings import settings

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter to respect website load."""

    def __init__(self, min_interval: float):
        """Initialize rate limiter.

        Args:
            min_interval: Minimum seconds between requests
        """
        self.min_interval = min_interval
        self._last_request_time: float = 0

    def wait(self) -> None:
        """Wait if needed to respect rate limit."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            logger.debug("Rate limiting: sleeping %.2f seconds", sleep_time)
            time.sleep(sleep_time)
        self._last_request_time = time.time()


class BaseScraper:
    """Base scraper with Playwright browser automation.

    Features:
        - Headless browser for JavaScript rendering
        - Rate limiting to avoid overloading
        - Exponential backoff retry on failures
        - HTML caching for incremental scraping
    """

    def __init__(
        self,
        headless: bool = True,
        rate_limit: Optional[float] = None,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize base scraper.

        Args:
            headless: Run browser in headless mode
            rate_limit: Seconds between requests (uses settings default if None)
            cache_dir: Directory for HTML cache (uses settings default if None)
        """
        self.headless = headless if headless is not None else settings.headless
        self.rate_limit = rate_limit if rate_limit is not None else settings.rate_limit_seconds
        self.cache_dir = cache_dir if cache_dir is not None else settings.raw_dir

        self._rate_limiter = RateLimiter(self.rate_limit)
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None

    def __enter__(self) -> "BaseScraper":
        """Enter context manager - start browser."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager - close browser."""
        self.close()

    def start(self) -> None:
        """Start Playwright browser."""
        if self._browser is not None:
            return

        logger.info("Starting Playwright browser (headless=%s)", self.headless)
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        self._page = self._browser.new_page(
            user_agent=settings.user_agent,
        )
        self._page.set_default_timeout(settings.request_timeout * 1000)

    def close(self) -> None:
        """Close browser and cleanup."""
        if self._page:
            self._page.close()
            self._page = None
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None
        logger.info("Browser closed")

    @property
    def page(self) -> Page:
        """Get current page, starting browser if needed."""
        if self._page is None:
            self.start()
        return self._page  # type: ignore

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((TimeoutError, Exception)),
        before_sleep=lambda retry_state: logger.warning(
            "Retry attempt %d after error", retry_state.attempt_number
        ),
    )
    def fetch_page(
        self,
        url: str,
        wait_selector: Optional[str] = None,
        wait_time: int = 2000,
    ) -> str:
        """Fetch a page with JavaScript rendering.

        Args:
            url: URL to fetch
            wait_selector: CSS selector to wait for before extracting HTML
            wait_time: Additional wait time in milliseconds after page load

        Returns:
            Rendered HTML content
        """
        self._rate_limiter.wait()

        logger.info("Fetching: %s", url)
        self.page.goto(url)

        # Wait for content to load
        if wait_selector:
            try:
                self.page.wait_for_selector(wait_selector, timeout=10000)
            except Exception as e:
                logger.warning("Selector wait failed: %s", e)

        # Additional wait for dynamic content
        self.page.wait_for_timeout(wait_time)

        html = self.page.content()
        logger.debug("Fetched %d bytes from %s", len(html), url)

        return html

    def save_html(self, html: str, filename: str) -> Path:
        """Save HTML to cache directory.

        Args:
            html: HTML content
            filename: Filename to save as

        Returns:
            Path to saved file
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.cache_dir / filename
        filepath.write_text(html, encoding="utf-8")
        logger.debug("Saved HTML to %s", filepath)
        return filepath

    def load_cached_html(self, filename: str) -> Optional[str]:
        """Load HTML from cache if exists.

        Args:
            filename: Filename to load

        Returns:
            HTML content or None if not cached
        """
        filepath = self.cache_dir / filename
        if filepath.exists():
            logger.debug("Loading cached HTML from %s", filepath)
            return filepath.read_text(encoding="utf-8")
        return None

    def get_cache_filename(self, url: str) -> str:
        """Generate cache filename from URL.

        Args:
            url: URL to generate filename for

        Returns:
            Safe filename string
        """
        import hashlib
        from urllib.parse import urlparse

        parsed = urlparse(url)
        path_safe = parsed.path.replace("/", "_").strip("_") or "index"
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        return f"{path_safe}_{url_hash}.html"

    def fetch_with_cache(
        self,
        url: str,
        force_refresh: bool = False,
        wait_selector: Optional[str] = None,
    ) -> str:
        """Fetch page with caching support.

        Args:
            url: URL to fetch
            force_refresh: Force refresh even if cached
            wait_selector: Selector to wait for

        Returns:
            HTML content (from cache or fresh)
        """
        filename = self.get_cache_filename(url)

        if not force_refresh:
            cached = self.load_cached_html(filename)
            if cached:
                return cached

        html = self.fetch_page(url, wait_selector=wait_selector)
        self.save_html(html, filename)
        return html

    def scroll_to_load_all(self, max_scrolls: int = 10, scroll_delay: int = 1000) -> None:
        """Scroll page to trigger lazy loading.

        Args:
            max_scrolls: Maximum number of scroll attempts
            scroll_delay: Delay between scrolls in milliseconds
        """
        for i in range(max_scrolls):
            previous_height = self.page.evaluate("document.body.scrollHeight")
            self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            self.page.wait_for_timeout(scroll_delay)
            new_height = self.page.evaluate("document.body.scrollHeight")

            if new_height == previous_height:
                logger.debug("Reached end of page after %d scrolls", i + 1)
                break
