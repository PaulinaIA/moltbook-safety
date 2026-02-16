"""URL discovery for moltbook.com scraping."""
import time
import logging
from typing import List, Set

from config.settings import settings
from src.scraper.base import BaseScraper
from src.scraper.parsers import parse_users_list, parse_submolt_list

logger = logging.getLogger(__name__)


class URLDiscovery:
    """Discover URLs for users, submolts, and posts."""

    def __init__(self, scraper: BaseScraper):
        """Initialize URL discovery.

        Args:
            scraper: BaseScraper instance for fetching pages
        """
        self.scraper = scraper
        self.base_url = settings.base_url

    def discover_users(self, max_users: int = 100, known_users: Set[str] = None) -> List[str]:
        """Discover user profile URLs from the users listing.

        Args:
            max_users: Maximum number of users to discover
            known_users: Set of already known usernames to skip

        Returns:
            List of user profile URLs
        """
        known_users = known_users or set()
        users_url = f"{self.base_url}/u"

        logger.info("Discovering users from %s", users_url)

        html = self.scraper.fetch_page(users_url, wait_selector="a[href^='/u/']")
        if getattr(self.scraper, "use_browser", False):
            logger.info("Switching to Karma view to fetch the most karma users...")
            self.scraper.page.click("button:has-text('Karma')")
            time.sleep(2)
            self.scraper.scroll_to_load_all(max_scrolls=5)
            html = self.scraper.page.content()

        users = parse_users_list(html)
        urls: List[str] = []

        for user in users:
            if len(urls) >= max_users:
                break
            username = user.get("name", "")
            if username and username not in known_users:
                urls.append(f"{self.base_url}/u/{username}")

        logger.info("Discovered %d new user URLs", len(urls))
        return urls

    def discover_submolts(self, max_submolts: int = 50, known_submolts: Set[str] = None) -> List[str]:
        """Discover submolt URLs from the submolts listing.

        Args:
            max_submolts: Maximum number of submolts to discover
            known_submolts: Set of already known submolt names to skip

        Returns:
            List of submolt page URLs
        """
        known_submolts = known_submolts or set()
        submolts_url = f"{self.base_url}/m"

        logger.info("Discovering submolts from %s", submolts_url)

        html = self.scraper.fetch_page(submolts_url, wait_selector="a[href^='/m/']")
        if getattr(self.scraper, "use_browser", False):
            self.scraper.scroll_to_load_all(max_scrolls=3)
            html = self.scraper.page.content()

        submolts = parse_submolt_list(html)
        urls: List[str] = []

        for submolt in submolts:
            if len(urls) >= max_submolts:
                break
            name = submolt.get("name", "")
            if name and name not in known_submolts:
                urls.append(f"{self.base_url}/m/{name}")

        logger.info("Discovered %d new submolt URLs", len(urls))
        return urls

    def get_user_profile_url(self, username: str) -> str:
        """Get profile URL for a username.

        Args:
            username: Username

        Returns:
            Full profile URL
        """
        return f"{self.base_url}/u/{username}"

    def get_submolt_url(self, name: str) -> str:
        """Get page URL for a submolt.

        Args:
            name: SubMolt name

        Returns:
            Full submolt page URL
        """
        return f"{self.base_url}/m/{name}"
