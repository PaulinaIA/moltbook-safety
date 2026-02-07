"""CSS selectors for moltbook.com scraping.

All selectors are centralized here for easy maintenance.
Modify these values if the website structure changes.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class UserListSelectors:
    """Selectors for the users listing page (/u)."""

    container: str = "div.grid"
    user_card: str = "div.bg-\\[\\#2d2d2e\\]"
    user_link: str = "a[href^='/u/']"
    username: str = "span.font-bold, div.font-bold"
    karma_badge: str = "span:has-text('karma'), div:has-text('karma')"
    avatar: str = "img.rounded-full"
    load_more_button: str = "button:has-text('Load More'), button:has-text('Show More')"
    sort_buttons: str = "button.px-2\\.5"


@dataclass(frozen=True)
class UserProfileSelectors:
    """Selectors for individual user profile pages (/u/{username})."""

    name: str = "h1, h2.text-2xl"
    karma: str = "span:has-text('karma'), div:has-text('karma')"
    description: str = "p.text-\\[\\#818384\\], div.text-gray-400"
    human_owner: str = "a[href*='x.com'], a[href*='twitter.com']"
    joined_date: str = "span:has-text('Joined'), time"
    followers_count: str = "span:has-text('followers'), a:has-text('followers')"
    following_count: str = "span:has-text('following'), a:has-text('following')"
    avatar: str = "img.rounded-full"
    posts_section: str = "section:has-text('Posts'), div:has-text('Posts')"


@dataclass(frozen=True)
class PostSelectors:
    """Selectors for posts on submolt and user pages."""

    post_card: str = "article, div.border-b"
    title: str = "h3, a.font-bold"
    description: str = "p.text-sm, div.text-gray-300"
    author_link: str = "a[href^='/u/']"
    submolt_link: str = "a[href^='/m/']"
    rating: str = "span:has-text('points'), button:has-text('upvote')"
    date: str = "time, span.text-gray-500"
    comments_count: str = "a:has-text('comments'), span:has-text('comments')"
    post_link: str = "a[href*='/post/'], a[href*='/p/']"


@dataclass(frozen=True)
class CommentSelectors:
    """Selectors for comments on post pages."""

    comment_block: str = "div.comment, div.pl-4"
    author_link: str = "a[href^='/u/']"
    content: str = "p, div.text-sm"
    rating: str = "span:has-text('points')"
    date: str = "time, span.text-gray-500"
    reply_button: str = "button:has-text('reply')"


@dataclass(frozen=True)
class SubMoltListSelectors:
    """Selectors for the submolts listing page (/m)."""

    container: str = "div.grid, main"
    submolt_card: str = "div.bg-\\[\\#2d2d2e\\], a[href^='/m/']"
    submolt_link: str = "a[href^='/m/']"
    name: str = "span.font-bold, h3"
    description: str = "p.text-sm, span.text-gray-400"
    member_count: str = "span:has-text('members')"


@dataclass(frozen=True)
class SubMoltPageSelectors:
    """Selectors for individual submolt pages (/m/{name})."""

    name: str = "h1, h2.text-2xl"
    description: str = "p.text-\\[\\#818384\\], div.text-gray-400"
    member_count: str = "span:has-text('members')"
    posts_container: str = "div.space-y-4, main"
    sidebar: str = "aside, div.sticky"


@dataclass(frozen=True)
class NavigationSelectors:
    """Selectors for navigation and pagination."""

    next_page: str = "a:has-text('Next'), button:has-text('Next')"
    prev_page: str = "a:has-text('Previous'), button:has-text('Prev')"
    page_number: str = "span.current-page, button.active"
    load_more: str = "button:has-text('Load'), button:has-text('More')"


@dataclass(frozen=True)
class Selectors:
    """Container for all selector groups."""

    user_list: UserListSelectors = UserListSelectors()
    user_profile: UserProfileSelectors = UserProfileSelectors()
    post: PostSelectors = PostSelectors()
    comment: CommentSelectors = CommentSelectors()
    submolt_list: SubMoltListSelectors = SubMoltListSelectors()
    submolt_page: SubMoltPageSelectors = SubMoltPageSelectors()
    navigation: NavigationSelectors = NavigationSelectors()


# Global selectors instance
selectors = Selectors()


def get_selector(group: str, name: str) -> Optional[str]:
    """Get a selector by group and name.

    Args:
        group: Selector group name (e.g., 'user_list', 'post')
        name: Selector name within the group

    Returns:
        CSS selector string or None if not found
    """
    group_obj = getattr(selectors, group, None)
    if group_obj is None:
        return None
    return getattr(group_obj, name, None)
