"""Unit tests for HTML parsers."""

import pytest
from pathlib import Path

from src.scraper.parsers import (
    parse_user_profile,
    parse_users_list,
    parse_submolt_page,
    parse_posts_from_page,
    parse_number,
    extract_username_from_url,
    extract_submolt_from_url,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(filename: str) -> str:
    """Load an HTML fixture file."""
    return (FIXTURES_DIR / filename).read_text(encoding="utf-8")


class TestParseNumber:
    """Tests for parse_number function."""

    def test_simple_number(self):
        assert parse_number("123") == 123

    def test_with_suffix_k(self):
        assert parse_number("5.2K") == 5200
        assert parse_number("5.2k") == 5200

    def test_with_suffix_m(self):
        assert parse_number("1.5M") == 1500000

    def test_with_word_karma(self):
        assert parse_number("1234 karma") == 1234

    def test_with_word_followers(self):
        assert parse_number("500 followers") == 500

    def test_empty_string(self):
        assert parse_number("") == 0

    def test_invalid_string(self):
        assert parse_number("abc") == 0


class TestExtractFromUrl:
    """Tests for URL extraction functions."""

    def test_extract_username_simple(self):
        assert extract_username_from_url("/u/test_user") == "test_user"

    def test_extract_username_full_url(self):
        url = "https://moltbook.com/u/agent123"
        assert extract_username_from_url(url) == "agent123"

    def test_extract_username_with_query(self):
        assert extract_username_from_url("/u/user?tab=posts") == "user"

    def test_extract_username_invalid(self):
        assert extract_username_from_url("/m/submolt") is None

    def test_extract_submolt_simple(self):
        assert extract_submolt_from_url("/m/general") == "general"

    def test_extract_submolt_full_url(self):
        url = "https://moltbook.com/m/agents"
        assert extract_submolt_from_url(url) == "agents"


class TestParseUserProfile:
    """Tests for parse_user_profile function."""

    def test_parse_user_profile(self):
        html = load_fixture("user_profile.html")
        result = parse_user_profile(html, "test_agent")

        assert result["name"] == "test_agent"
        assert result["karma"] == 1234
        assert result["description"] == "I am a test AI agent for unit testing purposes."
        assert result["human_owner"] == "test_human"
        assert result["followers"] == 500
        assert result["following"] == 250

    def test_parse_user_profile_with_joined(self):
        html = load_fixture("user_profile.html")
        result = parse_user_profile(html, "test_agent")

        # joined field should contain date
        assert result["joined"] is not None or "January" in str(result)


class TestParseUsersList:
    """Tests for parse_users_list function."""

    def test_parse_users_list(self):
        html = load_fixture("users_list.html")
        result = parse_users_list(html)

        assert len(result) == 3

        # Check first user
        user_names = [u["name"] for u in result]
        assert "agent_alpha" in user_names
        assert "beta_bot" in user_names
        assert "gamma_agent" in user_names

    def test_parse_users_list_urls(self):
        html = load_fixture("users_list.html")
        result = parse_users_list(html)

        for user in result:
            assert user["profile_url"].startswith("/u/")


class TestParseSubmoltPage:
    """Tests for parse_submolt_page function."""

    def test_parse_submolt_page(self):
        html = load_fixture("submolt_page.html")
        result = parse_submolt_page(html, "test_submolt")

        assert result["name"] == "test_submolt"
        assert "test submolt" in result["description"].lower()

    def test_parse_posts_from_submolt(self):
        html = load_fixture("submolt_page.html")
        result = parse_posts_from_page(html, "test_submolt")

        assert len(result) >= 2

        # Check post data
        titles = [p.get("title", "") for p in result]
        assert any("First" in t for t in titles)
        assert any("Second" in t for t in titles)


class TestParsePostsFromPage:
    """Tests for parse_posts_from_page function."""

    def test_parse_posts_extracts_author(self):
        html = load_fixture("submolt_page.html")
        posts = parse_posts_from_page(html, "test_submolt")

        authors = [p.get("author_name") for p in posts if p.get("author_name")]
        assert "poster_one" in authors or "poster_two" in authors

    def test_parse_posts_extracts_rating(self):
        html = load_fixture("submolt_page.html")
        posts = parse_posts_from_page(html, "test_submolt")

        # At least one post should have a rating
        ratings = [p.get("rating", 0) for p in posts]
        assert any(r > 0 for r in ratings)
