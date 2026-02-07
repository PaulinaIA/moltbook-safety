"""HTML parsing functions for moltbook.com pages.

All parsing is defensive - returns None or empty collections on failure.
"""

import logging
import re
from typing import Dict, List, Optional, Any

from bs4 import BeautifulSoup, Tag

from config.selectors import selectors

logger = logging.getLogger(__name__)


def safe_get_text(element: Optional[Tag], default: str = "") -> str:
    """Safely extract text from a BeautifulSoup element.

    Args:
        element: BeautifulSoup Tag or None
        default: Default value if extraction fails

    Returns:
        Extracted text or default
    """
    if element is None:
        return default
    try:
        return element.get_text(strip=True)
    except Exception:
        return default


def safe_get_attr(element: Optional[Tag], attr: str, default: str = "") -> str:
    """Safely extract attribute from a BeautifulSoup element.

    Args:
        element: BeautifulSoup Tag or None
        attr: Attribute name
        default: Default value if extraction fails

    Returns:
        Attribute value or default
    """
    if element is None:
        return default
    try:
        return element.get(attr, default) or default
    except Exception:
        return default


def parse_number(text: str) -> int:
    """Parse a number from text, handling K/M suffixes.

    Args:
        text: Text containing a number (e.g., "1.2K", "500")

    Returns:
        Parsed integer value
    """
    if not text:
        return 0

    text = text.strip().lower()

    # Remove common prefix/suffix words
    for word in ["karma", "followers", "following", "points", "members"]:
        text = text.replace(word, "").strip()

    # Handle K and M suffixes
    multiplier = 1
    if text.endswith("k"):
        multiplier = 1000
        text = text[:-1]
    elif text.endswith("m"):
        multiplier = 1000000
        text = text[:-1]

    try:
        # Handle decimal values like "1.2K"
        value = float(text) * multiplier
        return int(value)
    except (ValueError, TypeError):
        return 0


def extract_username_from_url(url: str) -> Optional[str]:
    """Extract username from a user profile URL.

    Args:
        url: URL like "/u/username" or "https://moltbook.com/u/username"

    Returns:
        Username or None
    """
    match = re.search(r"/u/([^/?#]+)", url)
    return match.group(1) if match else None


def extract_submolt_from_url(url: str) -> Optional[str]:
    """Extract submolt name from a submolt URL.

    Args:
        url: URL like "/m/submolt" or "https://moltbook.com/m/submolt"

    Returns:
        SubMolt name or None
    """
    match = re.search(r"/m/([^/?#]+)", url)
    return match.group(1) if match else None


def parse_users_list(html: str) -> List[Dict[str, Any]]:
    """Parse users listing page to extract user links and basic info.

    Args:
        html: HTML content of /u page

    Returns:
        List of user dictionaries with name, karma, profile_url
    """
    soup = BeautifulSoup(html, "lxml")
    users: List[Dict[str, Any]] = []

    # Find all user links
    user_links = soup.select("a[href^='/u/']")

    seen_usernames = set()
    for link in user_links:
        href = link.get("href", "")
        username = extract_username_from_url(href)

        if not username or username in seen_usernames:
            continue

        seen_usernames.add(username)

        # Try to find karma in parent container
        parent = link.find_parent("div")
        karma_text = ""
        if parent:
            karma_elem = parent.find(string=re.compile(r"karma", re.I))
            if karma_elem:
                karma_text = karma_elem.parent.get_text() if karma_elem.parent else str(karma_elem)

        users.append({
            "name": username,
            "karma": parse_number(karma_text),
            "profile_url": href,
        })

    logger.info("Parsed %d users from listing", len(users))
    return users


def parse_user_profile(html: str, username: str) -> Dict[str, Any]:
    """Parse user profile page to extract detailed info.

    Args:
        html: HTML content of /u/{username} page
        username: Username being parsed

    Returns:
        Dictionary with user details
    """
    soup = BeautifulSoup(html, "lxml")

    result: Dict[str, Any] = {
        "name": username,
        "karma": 0,
        "description": None,
        "human_owner": None,
        "joined": None,
        "followers": 0,
        "following": 0,
    }

    # Find karma - look for elements containing "karma"
    karma_patterns = [
        re.compile(r"(\d+(?:\.\d+)?[KkMm]?)\s*karma", re.I),
        re.compile(r"karma[:\s]*(\d+(?:\.\d+)?[KkMm]?)", re.I),
    ]

    body_text = soup.get_text()
    for pattern in karma_patterns:
        match = pattern.search(body_text)
        if match:
            result["karma"] = parse_number(match.group(1))
            break

    # Find description - usually in meta or a paragraph
    meta_desc = soup.find("meta", {"name": "description"})
    if meta_desc:
        result["description"] = safe_get_attr(meta_desc, "content")

    # Look for bio/description in page content
    for selector in ["p.bio", "div.description", "p.text-gray-400"]:
        bio_elem = soup.select_one(selector)
        if bio_elem:
            result["description"] = safe_get_text(bio_elem)
            break

    # Find human owner (Twitter handle)
    twitter_link = soup.select_one("a[href*='twitter.com'], a[href*='x.com']")
    if twitter_link:
        href = safe_get_attr(twitter_link, "href")
        match = re.search(r"(?:twitter\.com|x\.com)/(@?\w+)", href)
        if match:
            result["human_owner"] = match.group(1)

    # Find join date
    joined_pattern = re.compile(r"joined\s+(.+?)(?:\s*\||$)", re.I)
    joined_match = joined_pattern.search(body_text)
    if joined_match:
        result["joined"] = joined_match.group(1).strip()

    # Find followers/following
    followers_pattern = re.compile(r"(\d+(?:\.\d+)?[KkMm]?)\s*followers?", re.I)
    following_pattern = re.compile(r"(\d+(?:\.\d+)?[KkMm]?)\s*following", re.I)

    followers_match = followers_pattern.search(body_text)
    if followers_match:
        result["followers"] = parse_number(followers_match.group(1))

    following_match = following_pattern.search(body_text)
    if following_match:
        result["following"] = parse_number(following_match.group(1))

    logger.debug("Parsed user profile: %s (karma=%d)", username, result["karma"])
    return result


def parse_submolt_list(html: str) -> List[Dict[str, Any]]:
    """Parse submolts listing page.

    Args:
        html: HTML content of /m page

    Returns:
        List of submolt dictionaries
    """
    soup = BeautifulSoup(html, "lxml")
    submolts: List[Dict[str, Any]] = []

    submolt_links = soup.select("a[href^='/m/']")

    seen_names = set()
    for link in submolt_links:
        href = link.get("href", "")
        name = extract_submolt_from_url(href)

        if not name or name in seen_names:
            continue

        seen_names.add(name)

        # Try to find description in parent
        parent = link.find_parent("div")
        description = ""
        if parent:
            desc_elem = parent.find("p")
            if desc_elem:
                description = safe_get_text(desc_elem)

        submolts.append({
            "name": name,
            "description": description or None,
            "page_url": href,
        })

    logger.info("Parsed %d submolts from listing", len(submolts))
    return submolts


def parse_submolt_page(html: str, submolt_name: str) -> Dict[str, Any]:
    """Parse individual submolt page.

    Args:
        html: HTML content of /m/{name} page
        submolt_name: SubMolt name being parsed

    Returns:
        Dictionary with submolt details
    """
    soup = BeautifulSoup(html, "lxml")

    result: Dict[str, Any] = {
        "name": submolt_name,
        "description": None,
    }

    # Find description
    meta_desc = soup.find("meta", {"name": "description"})
    if meta_desc:
        result["description"] = safe_get_attr(meta_desc, "content")

    # Look for description in page
    for selector in ["p.description", "div.about", "p.text-gray-400"]:
        desc_elem = soup.select_one(selector)
        if desc_elem:
            result["description"] = safe_get_text(desc_elem)
            break

    return result


def parse_post(element: Tag, submolt_name: Optional[str] = None) -> Dict[str, Any]:
    """Parse a single post element.

    Args:
        element: BeautifulSoup Tag containing post
        submolt_name: SubMolt name if known

    Returns:
        Dictionary with post details
    """
    result: Dict[str, Any] = {
        "title": None,
        "description": None,
        "author_name": None,
        "submolt_name": submolt_name,
        "rating": 0,
        "date": None,
        "post_url": None,
    }

    # Find title
    title_elem = element.find(["h3", "h2"]) or element.select_one("a.font-bold")
    if title_elem:
        result["title"] = safe_get_text(title_elem)
        # Get post URL from title link if it exists
        if title_elem.name == "a":
            result["post_url"] = safe_get_attr(title_elem, "href")
        else:
            link = title_elem.find("a")
            if link:
                result["post_url"] = safe_get_attr(link, "href")

    # Find author
    author_link = element.select_one("a[href^='/u/']")
    if author_link:
        result["author_name"] = extract_username_from_url(safe_get_attr(author_link, "href"))

    # Find submolt if not provided
    if not submolt_name:
        submolt_link = element.select_one("a[href^='/m/']")
        if submolt_link:
            result["submolt_name"] = extract_submolt_from_url(safe_get_attr(submolt_link, "href"))

    # Find rating/points
    body_text = element.get_text()
    points_match = re.search(r"(\d+)\s*(?:points?|upvotes?)", body_text, re.I)
    if points_match:
        result["rating"] = int(points_match.group(1))

    # Find date
    time_elem = element.find("time")
    if time_elem:
        result["date"] = safe_get_attr(time_elem, "datetime") or safe_get_text(time_elem)

    return result


def parse_posts_from_page(html: str, submolt_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Parse all posts from a page (submolt or user profile).

    Args:
        html: HTML content
        submolt_name: SubMolt name if parsing a submolt page

    Returns:
        List of post dictionaries
    """
    soup = BeautifulSoup(html, "lxml")
    posts: List[Dict[str, Any]] = []

    # Try different post container selectors
    post_containers = soup.select("article") or soup.select("div.post") or soup.select("div.border-b")

    for container in post_containers:
        post = parse_post(container, submolt_name)
        if post.get("title") or post.get("author_name"):
            posts.append(post)

    logger.info("Parsed %d posts from page", len(posts))
    return posts


def parse_comments(html: str, post_id: str) -> List[Dict[str, Any]]:
    """Parse comments from a post page.

    Args:
        html: HTML content of post page
        post_id: ID of the parent post

    Returns:
        List of comment dictionaries
    """
    soup = BeautifulSoup(html, "lxml")
    comments: List[Dict[str, Any]] = []

    # Try different comment selectors
    comment_containers = (
        soup.select("div.comment")
        or soup.select("div[class*='comment']")
        or soup.select("div.pl-4")
    )

    for container in comment_containers:
        result: Dict[str, Any] = {
            "post_id": post_id,
            "author_name": None,
            "description": None,
            "date": None,
            "rating": 0,
        }

        # Find author
        author_link = container.select_one("a[href^='/u/']")
        if author_link:
            result["author_name"] = extract_username_from_url(safe_get_attr(author_link, "href"))

        # Find content
        content_elem = container.find("p")
        if content_elem:
            result["description"] = safe_get_text(content_elem)

        # Find date
        time_elem = container.find("time")
        if time_elem:
            result["date"] = safe_get_attr(time_elem, "datetime") or safe_get_text(time_elem)

        # Find rating
        body_text = container.get_text()
        points_match = re.search(r"(\d+)\s*(?:points?)", body_text, re.I)
        if points_match:
            result["rating"] = int(points_match.group(1))

        if result.get("author_name") or result.get("description"):
            comments.append(result)

    logger.info("Parsed %d comments", len(comments))
    return comments
