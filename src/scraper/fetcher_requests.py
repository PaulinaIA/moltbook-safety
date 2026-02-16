"""
Fetcher basado en requests (sin Playwright) para entornos donde el navegador no está
disponible (p. ej. AWS Glue). Usa solo requests + caché en disco; el parseo sigue
haciéndose con BeautifulSoup en parsers.

- Misma interfaz que BaseScraper para fetch_page, fetch_with_cache, etc.
- use_browser = False para que URLDiscovery no intente click/scroll.
- Sin dependencias de Playwright ni pip cache en directorios del sistema.
"""

import hashlib
import logging
import time
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional

logger = logging.getLogger(__name__)


def _get_headers(user_agent: str) -> dict:
    return {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }


class RateLimiter:
    """Rate limiter reutilizable (sin dependencias de Playwright)."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last_request_time: float = 0

    def wait(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.time()


class RequestsFetcher:
    """
    Obtiene HTML vía HTTP GET (requests). Compatible con Glue: sin Playwright ni pip.

    Uso en Glue: MoltbookScraper(..., use_playwright=False) usa este fetcher.
    El contenido que no venga en el HTML inicial (lazy load, JS) no se verá.
    """

    use_browser = False  # Para que URLDiscovery no llame a page.click ni scroll_to_load_all

    def __init__(
        self,
        rate_limit: float = 1.0,
        cache_dir: Optional[Path] = None,
        user_agent: Optional[str] = None,
        request_timeout: int = 30,
    ):
        try:
            from config.settings import settings
            _rate = getattr(settings, "rate_limit_seconds", 1.0)
            _ua = getattr(settings, "user_agent", None)
            _timeout = getattr(settings, "request_timeout", 30)
            _raw = getattr(settings, "raw_dir", None)
        except Exception:
            _rate, _ua, _timeout, _raw = 1.0, None, 30, None
        self.rate_limit = rate_limit if rate_limit else _rate
        self.cache_dir = cache_dir or (_raw if _raw else Path("/tmp/moltbook_cache"))
        self.user_agent = user_agent or _ua or "MoltbookScraper/1.0 (Academic Research Project)"
        self.request_timeout = request_timeout or _timeout
        self._rate_limiter = RateLimiter(self.rate_limit)
        self._session = None

    def __enter__(self) -> "RequestsFetcher":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    @property
    def session(self):
        import requests
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(_get_headers(self.user_agent))
        return self._session

    def fetch_page(
        self,
        url: str,
        wait_selector: Optional[str] = None,
        wait_time: int = 2000,
    ) -> str:
        """Obtiene el HTML de la URL por GET. wait_selector/wait_time se ignoran (sin JS)."""
        self._rate_limiter.wait()
        logger.info("Fetching: %s", url)
        try:
            r = self.session.get(url, timeout=self.request_timeout)
            r.raise_for_status()
            r.encoding = r.apparent_encoding or "utf-8"
            html = r.text
        except Exception as e:
            logger.warning("Request failed for %s: %s", url, e)
            raise
        if wait_time and wait_time > 0:
            time.sleep(min(wait_time / 1000.0, 5.0))  # respeto opcional
        logger.debug("Fetched %d bytes from %s", len(html), url)
        return html

    def get_cache_filename(self, url: str) -> str:
        parsed = urlparse(url)
        path_safe = parsed.path.replace("/", "_").strip("_") or "index"
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        return f"{path_safe}_{url_hash}.html"

    def save_html(self, html: str, filename: str) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.cache_dir / filename
        filepath.write_text(html, encoding="utf-8")
        logger.debug("Saved HTML to %s", filepath)
        return filepath

    def load_cached_html(self, filename: str) -> Optional[str]:
        filepath = self.cache_dir / filename
        if filepath.exists():
            logger.debug("Loading cached HTML from %s", filepath)
            return filepath.read_text(encoding="utf-8")
        return None

    def fetch_with_cache(
        self,
        url: str,
        force_refresh: bool = False,
        wait_selector: Optional[str] = None,
    ) -> str:
        filename = self.get_cache_filename(url)
        if not force_refresh:
            cached = self.load_cached_html(filename)
            if cached:
                return cached
        html = self.fetch_page(url, wait_selector=wait_selector)
        self.save_html(html, filename)
        return html

    def scroll_to_load_all(self, max_scrolls: int = 10, scroll_delay: int = 1000) -> None:
        """No-op: sin navegador no hay scroll. Para compatibilidad con URLDiscovery."""
        logger.debug("scroll_to_load_all is a no-op (requests-only mode)")
