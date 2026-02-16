#!/usr/bin/env python3
"""
Script único para el Glue job: solo base de datos (PostgreSQL/RDS) y scraping con
guardado en base de datos. Todo está en este archivo para poder revisarlo y subirlo
solo a Glue. Uso requests + BeautifulSoup (sin Playwright). Escrito en primera persona.
"""

import hashlib
import logging
import os
import re
import sys
import time
import urllib.parse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, TypeVar, Union

# ---------------------------------------------------------------------------
# LOGGING (para CloudWatch)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.moltbook.com"

# submolts que conozco de antemano. los uso para iterar directamente
# en vez de depender de que el listing de /m cargue todo con scroll.
SEED_SUBMOLTS = [
    "0xdeadbeef", "3dp", "agent-philosophy", "agentity", "agentlegaladvice",
    "ai-solopreneurs", "aidev", "aithernet", "announcements", "arrival",
    "blessmyheart", "blesstheirhearts", "blumeyield", "claudecodeagents",
    "clawdistan", "corinthians", "creatoreconomy", "currency", "debugging",
    "drugs", "economics", "embody", "ethics-convergence", "evil", "feedback",
    "firstworldaiproblems", "flame-advice", "general", "geointel", "hiremyagent",
    "introduction", "introductions", "latenightthoughts", "linux", "liquidation",
    "lobloop", "memecoins", "memes", "minting-claw-mbc20-15", "moltreg",
    "moltwild", "open-claw", "portfoliomanagement", "predictionmarkets",
    "projects", "quant-heytraders", "quantalpha", "remote-work", "seeds",
    "selfimprovement", "shitpost", "sixnations", "soul", "startups", "thoughts",
    "threatintel", "todayilearned", "trellis-workflows", "turkishagents",
    "uri-mal", "x402", "xana-monolith", "xrpledger", "xrplevm",
]


# ---------------------------------------------------------------------------
# ENTIDADES (solo para DB y scraper; no incluyo modelos de ML)
# ---------------------------------------------------------------------------

def _id(prefix: str, *args: str) -> str:
    content = "|".join(str(a) for a in args if a)
    h = hashlib.sha256(content.encode()).hexdigest()[:12]
    return f"{prefix}_{h}"


def _now() -> str:
    return datetime.utcnow().isoformat()

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
    id_submolt: str
    name: str
    description: Optional[str] = None
    scraped_at: str = field(default_factory=_now)

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
        id: Optional[str] = None,
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
        id_post = id or generate_id("post", id_source)
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
# ---------------------------------------------------------------------------
# CONEXIÓN A POSTGRES (leo credenciales de env)
# ---------------------------------------------------------------------------

@contextmanager
def _pg_connection():
    import psycopg2
    host = os.environ.get("DB_HOST", "localhost")
    dbname = os.environ.get("DB_NAME", "postgres")
    user = os.environ.get("DB_USER", "postgres")
    password = os.environ.get("DB_PASSWORD", "")
    port = int(os.environ.get("DB_PORT", "5432"))
    conn = psycopg2.connect(host=host, dbname=dbname, user=user, password=password, port=port)
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error("Error de base de datos: %s", e)
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# ESQUEMA Y OPERACIONES DE BASE DE DATOS
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id_user VARCHAR(64) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    karma INTEGER DEFAULT 0,
    description TEXT,
    human_owner VARCHAR(255),
    joined VARCHAR(128),
    followers INTEGER DEFAULT 0,
    following INTEGER DEFAULT 0,
    scraped_at VARCHAR(64) NOT NULL
);
CREATE TABLE IF NOT EXISTS sub_molt (
    id_submolt VARCHAR(64) PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    scraped_at VARCHAR(64) NOT NULL
);
CREATE TABLE IF NOT EXISTS posts (
    id_post VARCHAR(64) PRIMARY KEY,
    id_user VARCHAR(64) NOT NULL,
    id_submolt VARCHAR(64),
    title TEXT,
    description TEXT,
    rating INTEGER DEFAULT 0,
    date VARCHAR(128),
    scraped_at VARCHAR(64) NOT NULL
);
CREATE TABLE IF NOT EXISTS comments (
    id_comment VARCHAR(64) PRIMARY KEY,
    id_user VARCHAR(64) NOT NULL,
    id_post VARCHAR(64) NOT NULL,
    description TEXT,
    date VARCHAR(128),
    rating INTEGER DEFAULT 0,
    scraped_at VARCHAR(64) NOT NULL
);
CREATE TABLE IF NOT EXISTS user_submolt (
    id_user VARCHAR(64) NOT NULL,
    id_submolt VARCHAR(64) NOT NULL,
    PRIMARY KEY (id_user, id_submolt)
);
CREATE INDEX IF NOT EXISTS idx_posts_user ON posts(id_user);
CREATE INDEX IF NOT EXISTS idx_posts_submolt ON posts(id_submolt);
CREATE INDEX IF NOT EXISTS idx_comments_user ON comments(id_user);
CREATE INDEX IF NOT EXISTS idx_comments_post ON comments(id_post);
CREATE INDEX IF NOT EXISTS idx_users_karma ON users(karma);
CREATE INDEX IF NOT EXISTS idx_users_name ON users(name);
ALTER TABLE posts ADD CONSTRAINT fk_posts_user FOREIGN KEY (id_user) REFERENCES users(id_user);
ALTER TABLE posts ADD CONSTRAINT fk_posts_submolt FOREIGN KEY (id_submolt) REFERENCES sub_molt(id_submolt);
ALTER TABLE comments ADD CONSTRAINT fk_comments_user FOREIGN KEY (id_user) REFERENCES users(id_user);
ALTER TABLE comments ADD CONSTRAINT fk_comments_post FOREIGN KEY (id_post) REFERENCES posts(id_post);
"""


class Database:
    """Uso PostgreSQL (RDS). Creo tablas si no existen y guardo en lotes."""

    TABLES = {User: "users", Post: "posts", Comment: "comments", SubMolt: "sub_molt", UserSubMolt: "user_submolt"}
    PK = {User: "id_user", Post: "id_post", Comment: "id_comment", SubMolt: "id_submolt", UserSubMolt: ("id_user", "id_submolt")}

    def __init__(self, chunk_size: int = 5000):
        self.chunk_size = chunk_size

    def ensure_tables(self) -> None:
        with _pg_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT 1 FROM information_schema.tables WHERE table_name = 'users' LIMIT 1")
            if cur.fetchone() is not None:
                logger.info("Las tablas ya existen.")
                return
        for stmt in SCHEMA_SQL.split(";"):
            stmt = stmt.strip()
            if stmt and not stmt.startswith("--"):
                try:
                    with _pg_connection() as conn:
                        cur = conn.cursor()
                        cur.execute(stmt)
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        logger.warning("Statement skipped: %s", e)
        logger.info("Tablas creadas o actualizadas.")

    def bulk_upsert(self, entities: List[Any]) -> int:
        if not entities:
            return 0
        entity_type = type(entities[0])
        table = self.TABLES.get(entity_type)
        pk = self.PK.get(entity_type)
        if not table or not pk:
            raise ValueError("Tipo de entidad no soportado")
        pk_cols = (pk,) if isinstance(pk, str) else pk
        data_list = [e.to_dict() for e in entities]
        columns = list(data_list[0].keys())
        update_cols = [c for c in columns if c not in pk_cols]
        if update_cols:
            conflict_action = "DO UPDATE SET " + ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
        else:
            conflict_action = "DO NOTHING"
        conflict = ", ".join(pk_cols)
        total = 0
        for i in range(0, len(data_list), self.chunk_size):
            chunk = data_list[i : i + self.chunk_size]
            rows = [list(d.values()) for d in chunk]
            try:
                from psycopg2.extras import execute_values
                tpl = "(" + ", ".join("%s" for _ in columns) + ")"
                sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES %s ON CONFLICT ({conflict}) {conflict_action}"
                with _pg_connection() as conn:
                    cur = conn.cursor()
                    execute_values(cur, sql, rows, template=tpl, page_size=len(rows))
                total += len(rows)
            except ImportError:
                ph = ", ".join("%s" for _ in columns)
                sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({ph}) ON CONFLICT ({conflict}) {conflict_action}"
                with _pg_connection() as conn:
                    cur = conn.cursor()
                    for row in rows:
                        cur.execute(sql, row)
                total += len(rows)
        logger.info("guarde %d registros en %s.", total, table)
        return total

    def get_by_id(self, entity_type: Type[Any], id_value: str) -> Optional[Dict[str, Any]]:
        table = self.TABLES.get(entity_type)
        pk = self.PK.get(entity_type)
        if not table or not pk:
            raise ValueError("Tipo no soportado")
        cond = f"{pk} = %s"
        with _pg_connection() as conn:
            cur = conn.cursor()
            cur.execute(f"SELECT * FROM {table} WHERE {cond}", (id_value,))
            row = cur.fetchone()
            if row is None:
                return None
            return dict(zip([d[0] for d in cur.description], row))

    def get_user_names(self) -> List[str]:
        with _pg_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM users")
            return [r[0] for r in cur.fetchall()]

    def get_submolt_names(self) -> List[str]:
        with _pg_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sub_molt")
            return [r[0] for r in cur.fetchall()]

    def get_user_id_by_name(self, name: str) -> Optional[str]:
        with _pg_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id_user FROM users WHERE name = %s LIMIT 1", (name,))
            row = cur.fetchone()
            return row[0] if row else None

    def count(self, entity_type: Type[Any]) -> int:
        table = self.TABLES.get(entity_type)
        with _pg_connection() as conn:
            cur = conn.cursor()
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# FETCHER CON REQUESTS (sin Playwright)
# ---------------------------------------------------------------------------

class RateLimiter:
    def __init__(self, interval: float):
        self.interval = interval
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.time() - self._last
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self._last = time.time()


class RequestsFetcher:
    """Obtengo HTML con requests. No uso navegador ni Playwright."""

    use_browser = False

    def __init__(self, rate_limit: float = 1.0, timeout: int = 30):
        self.rate_limit = rate_limit
        self.timeout = timeout
        self._limiter = RateLimiter(rate_limit)
        self._session = None
        self.cache_dir = Path("/tmp/moltbook_cache")

    @property
    def session(self):
        import requests
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": "MoltbookScraper/1.0 (Academic)",
                "Accept": "text/html,application/xhtml+xml",
            })
        return self._session

    def close(self) -> None:
        if self._session:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    def fetch_page(self, url: str, wait_selector: Any = None, wait_time: int = 2000) -> str:
        self._limiter.wait()
        logger.info("Fetching: %s", url)
        r = self.session.get(url, timeout=self.timeout)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        if wait_time > 0:
            time.sleep(min(wait_time / 1000.0, 5.0))
        return r.text

    def get_cache_filename(self, url: str) -> str:
        from urllib.parse import urlparse
        p = urlparse(url)
        path_safe = p.path.replace("/", "_").strip("_") or "index"
        h = hashlib.md5(url.encode()).hexdigest()[:8]
        return f"{path_safe}_{h}.html"

    def save_html(self, html: str, filename: str) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        p = self.cache_dir / filename
        p.write_text(html, encoding="utf-8")
        return p

    def load_cached_html(self, filename: str) -> Optional[str]:
        p = self.cache_dir / filename
        if p.exists():
            return p.read_text(encoding="utf-8")
        return None

    def fetch_with_cache(self, url: str, force_refresh: bool = False, wait_selector: Any = None) -> str:
        fn = self.get_cache_filename(url)
        if not force_refresh:
            cached = self.load_cached_html(fn)
            if cached:
                return cached
        html = self.fetch_page(url, wait_selector=wait_selector)
        self.save_html(html, fn)
        return html

    def scroll_to_load_all(self, max_scrolls: int = 10, scroll_delay: int = 1000) -> None:
        pass


# ---------------------------------------------------------------------------
# PARSERS (BeautifulSoup)
# ---------------------------------------------------------------------------

def _parse_number(text: str) -> int:
    if not text:
        return 0
    text = text.strip().lower()
    for w in ["karma", "followers", "following", "points", "members"]:
        text = text.replace(w, "").strip()
    mult = 1
    if text.endswith("k"):
        mult, text = 1000, text[:-1]
    elif text.endswith("m"):
        mult, text = 1000000, text[:-1]
    try:
        return int(float(text) * mult)
    except (ValueError, TypeError):
        return 0


def _relative_date(text: str) -> str:
    now = datetime.now()
    m = re.search(r"(\d+)([a-z]+)", text.lower())
    if not m:
        return now.strftime("%Y-%m-%d")
    val, unit = int(m.group(1)), m.group(2)
    if "d" in unit:
        d = now - timedelta(days=val)
    elif "h" in unit:
        d = now - timedelta(hours=val)
    elif "m" in unit:
        d = now - timedelta(minutes=val)
    elif "w" in unit:
        d = now - timedelta(weeks=val)
    else:
        d = now
    return d.strftime("%Y-%m-%d")


def parse_users_list(html: str) -> List[Dict[str, Any]]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    users = []
    seen = set()
    for link in soup.select("a[href^='/u/']"):
        href = link.get("href", "")
        m = re.search(r"/u/([^/?#]+)", href)
        if not m or m.group(1) in seen:
            continue
        name = m.group(1)
        seen.add(name)
        parent = link.find_parent("div")
        karma_t = ""
        if parent:
            k = parent.find(string=re.compile(r"karma", re.I))
            if k and k.parent:
                karma_t = k.parent.get_text()
        users.append({"name": name, "karma": _parse_number(karma_t), "profile_url": href})
    return users


def parse_user_profile(html: str, username: str) -> Dict[str, Any]:
    """extraigo datos del perfil de un usuario desde el html renderizado.

    estructura real del dom (renderizado por next.js):
      div.flex.items-start.gap-4 > div.flex-1
        [0] div.flex.items-center.gap-2    -> h1 (username) + verified badge
        [1] p.mt-1                          -> bio/description
        [2] div.flex.flex-wrap...mt-3       -> stats row (karma, followers, following, joined)
        [3] div.mt-4.pt-4.border-t          -> human owner section con link a x
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    result: Dict[str, Any] = {
        "name": username, "karma": 0, "description": None,
        "human_owner": None, "joined": None, "followers": 0, "following": 0,
    }

    # busco el bloque principal del perfil
    profile_block = soup.select_one("div.flex.items-start.gap-4 > div.flex-1")

    # -- bio / descripcion --
    if profile_block:
        bio_el = profile_block.select_one("p.mt-1")
        if bio_el:
            result["description"] = bio_el.get_text(strip=True)
    if not result["description"]:
        for sel in ("p.mt-1", "p.text-gray-400"):
            el = soup.select_one(sel)
            if el:
                text = el.get_text(strip=True)
                if text and "social network" not in text.lower():
                    result["description"] = text
                    break

    # -- stats row: karma, followers, following, joined --
    stats_row = (
        profile_block.select_one("div.mt-3")
        if profile_block
        else soup.select_one("div.flex.flex-wrap.items-center.mt-3")
    )
    if stats_row:
        stat_divs = stats_row.select("div.text-sm")
        for div in stat_divs:
            text = div.get_text(" ", strip=True).lower()
            if "karma" in text:
                num_el = div.select_one("span.font-bold")
                if num_el:
                    result["karma"] = _parse_number(num_el.get_text(strip=True))
            elif "follower" in text:
                num_el = div.select_one("span.font-bold")
                if num_el:
                    result["followers"] = _parse_number(num_el.get_text(strip=True))
            elif "following" in text:
                num_el = div.select_one("span.font-bold")
                if num_el:
                    result["following"] = _parse_number(num_el.get_text(strip=True))
            elif "joined" in text:
                m = re.search(r"joined\s+(\S+)", text, re.I)
                if m:
                    result["joined"] = m.group(1)

    # fallback con regex si no encontre stats por dom
    if result["karma"] == 0:
        body = soup.get_text(" ", strip=True)
        m = re.search(r"(\d+(?:\.\d+)?[KkMm]?)\s*karma", body, re.I)
        if m:
            result["karma"] = _parse_number(m.group(1))

    # -- human owner --
    owner_section = (
        profile_block.select_one("div.mt-4.pt-4.border-t")
        if profile_block
        else soup.select_one("div.mt-4.pt-4.border-t")
    )
    if owner_section:
        x_link = owner_section.select_one('a[href*="x.com"], a[href*="twitter.com"]')
        if x_link and x_link.get("href"):
            mm = re.search(r"(?:twitter\.com|x\.com)/(@?\w+)", x_link["href"])
            if mm:
                result["human_owner"] = mm.group(1)
        if not result["human_owner"]:
            name_el = owner_section.select_one("span.font-bold, span.text-white")
            if name_el:
                result["human_owner"] = name_el.get_text(strip=True)
    else:
        for a in soup.select('a[href*="x.com"], a[href*="twitter.com"]'):
            href = a.get("href", "")
            if "mattprd" in href:
                continue
            mm = re.search(r"(?:twitter\.com|x\.com)/(@?\w+)", href)
            if mm:
                result["human_owner"] = mm.group(1)
                break

    # -- joined (fallback) --
    if not result["joined"]:
        body = soup.get_text(" ", strip=True)
        j = re.search(r"Joined\s+(\d{1,2}/\d{1,2}/\d{4})", body)
        if j:
            result["joined"] = j.group(1)
        else:
            j2 = re.search(r"Joined\s+(.+?)(?:\s+Online|\s*$)", body)
            if j2:
                result["joined"] = j2.group(1).strip()[:30]

    # -- followers/following fallback --
    if result["followers"] == 0 and profile_block:
        body = profile_block.get_text(" ", strip=True)
        ff = re.search(r"(\d+(?:\.\d+)?[KkMm]?)\s*followers?", body, re.I)
        if ff:
            result["followers"] = _parse_number(ff.group(1))
    if result["following"] == 0 and profile_block:
        body = profile_block.get_text(" ", strip=True)
        fl = re.search(r"(\d+(?:\.\d+)?[KkMm]?)\s*following", body, re.I)
        if fl:
            result["following"] = _parse_number(fl.group(1))

    return result


def parse_submolt_list(html: str) -> List[Dict[str, Any]]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    out = []
    seen = set()
    for link in soup.select("a[href^='/m/']"):
        href = link.get("href", "")
        m = re.search(r"/m/([^/?#]+)", href)
        if not m or m.group(1) in seen:
            continue
        name = m.group(1)
        seen.add(name)
        parent = link.find_parent("div")
        desc = ""
        if parent:
            p = parent.find("p")
            if p:
                desc = p.get_text(strip=True)
        out.append({"name": name, "description": desc or None, "page_url": href})
    return out


def parse_submolt_page(html: str, submolt_name: str) -> Dict[str, Any]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    out = {"name": submolt_name, "description": None}
    meta = soup.find("meta", {"name": "description"})
    if meta and meta.get("content"):
        out["description"] = meta["content"]
    for sel in ["p.description", "div.about", "p.text-gray-400"]:
        el = soup.select_one(sel)
        if el:
            out["description"] = el.get_text(strip=True)
            break
    return out


def parse_post_el(el, submolt_name: Optional[str] = None) -> Dict[str, Any]:
    from bs4 import BeautifulSoup
    out = {"title": None, "description": None, "author_name": None, "submolt_name": submolt_name,
           "rating": 0, "date": None, "post_url": None}
    href = el.get("href")
    if href:
        out["post_url"] = f"https://www.moltbook.com{href}" if href.startswith("/") else href
    h3 = el.select_one("h3")
    if h3:
        out["title"] = h3.get_text(strip=True)
    p = el.select_one("p.line-clamp-3")
    if p:
        out["description"] = p.get_text(strip=True)
    author = el.select_one("span.hover\\:underline")
    if author:
        out["author_name"] = author.get_text(strip=True).replace("u/", "")
    rating_el = el.select_one("span.text-white.font-bold")
    if rating_el:
        try:
            out["rating"] = int(rating_el.get_text(strip=True))
        except ValueError:
            pass
    xs = el.select_one("div.text-xs")
    if xs:
        txt = xs.get_text(" ", strip=True)
        if "ago" in txt:
            out["date"] = _relative_date(txt.split("ago")[0].strip() + " ago")
        else:
            out["date"] = datetime.now().strftime("%Y-%m-%d")
    return out


def parse_posts_from_page(html: str, submolt_name: Optional[str] = None, max_posts: Optional[int] = 10) -> List[Dict[str, Any]]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    posts = []
    for container in soup.select("a[href^='/post/']"):
        post = parse_post_el(container, submolt_name)
        if post.get("title") or post.get("post_url"):
            posts.append(post)
            if max_posts and len(posts) >= max_posts:
                break
    return posts


def parse_comments(html: str, post_id: str, max_comments: Optional[int] = 10) -> List[Dict[str, Any]]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    out = []
    for container in soup.select("div.mt-6 div.py-2"):
        r = {"post_id": post_id, "author_name": None, "description": None, "date": None, "rating": 0}
        a = container.select_one("a[href^='/u/']")
        if a:
            r["author_name"] = a.get_text(strip=True).replace("u/", "")
        prose = container.select_one("div.prose-invert p")
        if prose:
            r["description"] = prose.get_text(strip=True)
        header = container.select_one("div.flex.items-center.gap-2.text-xs")
        if header:
            txt = header.get_text(" ", strip=True)
            if "ago" in txt:
                r["date"] = _relative_date(txt.split("ago")[0].strip() + " ago")
        rating_el = container.select_one("span.flex.items-center.gap-1 span")
        if rating_el:
            try:
                r["rating"] = int(rating_el.get_text(strip=True))
            except ValueError:
                pass
        if r.get("author_name") and r.get("description"):
            out.append(r)
            if max_comments and len(out) >= max_comments:
                break
    return out


# ---------------------------------------------------------------------------
# BATCH WRITER Y SCRAPER
# ---------------------------------------------------------------------------

T = TypeVar("T", User, Post, Comment, SubMolt, UserSubMolt)


class BatchWriter:
    """acumulo entidades y las guardo en la base de datos en lotes.

    el orden de flush respeta la integridad referencial:
        users -> sub_molt -> user_submolt -> posts -> comments
    """

    _FLUSH_ORDER = [User, SubMolt, UserSubMolt, Post, Comment]

    def __init__(self, db: Database, batch_size: int = 1000):
        self.db = db
        self.batch_size = batch_size
        self._buf: Dict[Type[Any], List[Any]] = defaultdict(list)

    def add(self, entity: Any) -> None:
        t = type(entity)
        self._buf[t].append(entity)
        if len(self._buf[t]) >= self.batch_size:
            self._flush(t)

    def _flush(self, entity_type: Type[Any]) -> int:
        lst = self._buf.get(entity_type)
        if not lst:
            return 0
        chunk = list(lst)
        self._buf[entity_type] = []
        return self.db.bulk_upsert(chunk)

    def flush_type(self, entity_type: Type[Any]) -> int:
        return self._flush(entity_type)

    def flush_all(self) -> None:
        for t in self._FLUSH_ORDER:
            if self._buf.get(t):
                self._flush(t)


class MoltbookScraper:
    """orquesto el scraping y el guardado en base de datos.
    uso requests + beautifulsoup (sin playwright).
    enriquezco usuarios de forma incremental despues de cada submolt.
    """

    def __init__(self, db: Database, batch_size: int = 1000):
        self.db = db
        self._fetcher: Optional[RequestsFetcher] = None
        self._batch = BatchWriter(db, batch_size=batch_size)
        self._user_submolt_seen: Set[tuple] = set()
        self._enriched_users: Set[str] = set()

    def __enter__(self) -> "MoltbookScraper":
        self.db.ensure_tables()
        self._fetcher = RequestsFetcher(rate_limit=1.0, timeout=30)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            self._batch.flush_all()
        finally:
            if self._fetcher:
                self._fetcher.close()

    # -- helpers internos ---------------------------------------------------

    def _resolve_user_id(self, author_name: str) -> str:
        """si el usuario ya existe en la db, devuelvo su id.
        si no, creo uno nuevo y lo meto al batch."""
        uid = self.db.get_user_id_by_name(author_name)
        if uid:
            return uid
        new_user = User.from_scraped_data(name=author_name)
        self._batch.add(new_user)
        return new_user.id_user

    def _register_user_submolt(self, user_id: str, submolt_id: str) -> None:
        key = (user_id, submolt_id)
        if key in self._user_submolt_seen:
            return
        self._user_submolt_seen.add(key)
        self._batch.add(UserSubMolt(id_user=user_id, id_submolt=submolt_id))

    # -- discovery ----------------------------------------------------------

    def _discover_submolts(self, max_submolts: int) -> List[str]:
        """descubro submolts navegando /m y complemento con la lista semilla."""
        discovered: Set[str] = set()
        try:
            html = self._fetcher.fetch_page(f"{BASE_URL}/m")
            for item in parse_submolt_list(html):
                name = item.get("name", "")
                if name:
                    discovered.add(name)
            logger.info("descubri %d submolts desde /m", len(discovered))
        except Exception as exc:
            logger.warning("no pude navegar /m: %s -- uso solo semilla", exc)

        # complemento con la semilla
        for name in SEED_SUBMOLTS:
            discovered.add(name)

        known = set(self.db.get_submolt_names())
        new_names = [n for n in discovered if n not in known]
        old_names = [n for n in discovered if n in known]
        all_names = new_names + old_names
        return all_names[:max_submolts]

    # -- pipeline principal -------------------------------------------------

    def scrape_all(
        self,
        max_users: int = 100,
        max_submolts: int = 50,
        max_posts: int = 10,
        max_comments: int = 100,
    ) -> Dict[str, int]:
        """pipeline completo. enriquezco usuarios de forma incremental
        despues de cada submolt para que si interrumpen el etl,
        los datos de perfil ya esten en la db.
        """
        # --- 1. submolts + enrichment intercalado ---
        submolt_names = self._discover_submolts(max_submolts)
        logger.info("procesando %d submolts", len(submolt_names))

        submolts_ok = 0
        for name in submolt_names:
            try:
                url = f"{BASE_URL}/m/{name}"
                html = self._fetcher.fetch_with_cache(url)

                sm_data = parse_submolt_page(html, name)
                submolt = SubMolt.from_scraped_data(**sm_data)
                self._batch.add(submolt)
                self._batch.flush_type(SubMolt)
                self._batch.flush_type(User)
                submolts_ok += 1

                logger.info("submolt: %s", submolt.name)
                new_authors = self._process_posts(html, submolt, max_posts, max_comments)

                # enriquezco inmediatamente los usuarios descubiertos en este submolt
                pending = [n for n in new_authors if n not in self._enriched_users]
                if pending:
                    logger.info(
                        "enriqueciendo %d usuarios descubiertos en m/%s",
                        len(pending), name,
                    )
                    self._enrich_user_profiles(pending)

            except Exception as exc:
                logger.error("error en submolt %s: %s", name, exc)

        self._batch.flush_all()

        # --- 2. sweep final: usuarios que quedaron sin enriquecer ---
        db_users = self.db.get_user_names()
        remaining = [n for n in db_users if n not in self._enriched_users]
        if remaining:
            logger.info(
                "sweep final: %d usuarios pendientes de enriquecer",
                len(remaining),
            )
            self._enrich_user_profiles(remaining[:max_users])

        self._batch.flush_all()

        return {
            "users": self.db.count(User),
            "users_enriched": len(self._enriched_users),
            "submolts": submolts_ok,
            "posts": self.db.count(Post),
            "comments": self.db.count(Comment),
        }

    def _enrich_user_profiles(self, user_names: List[str]) -> int:
        """llamo al api /api/v1/agents/profile?name={name} para obtener
        karma, bio, followers, following, human_owner, joined.
        la pagina /u/{name} es un spa puro que requests no puede renderizar,
        pero el api devuelve json con todos los datos del perfil.
        flusheo cada usuario enriquecido inmediatamente a la db
        para que no se pierdan datos si el etl se interrumpe."""
        enriched = 0
        logger.info("enriqueciendo %d perfiles de usuario via api...", len(user_names))
        for name in user_names:
            if name in self._enriched_users:
                continue
            try:
                api_url = f"{BASE_URL}/api/v1/agents/profile?name={urllib.parse.quote(name)}"
                logger.info("consultando api perfil: %s", api_url)
                self._fetcher._limiter.wait()
                resp = self._fetcher.session.get(api_url, timeout=20)
                if resp.status_code != 200:
                    logger.warning(
                        "api perfil %s respondio %d", name, resp.status_code
                    )
                    continue

                payload = resp.json()
                if not payload.get("success") or not payload.get("agent"):
                    logger.warning("api perfil %s: success=false o sin agent", name)
                    continue

                agent = payload["agent"]
                owner_info = agent.get("owner") or {}
                human_owner = owner_info.get("x_handle") or None

                # mapeo joined desde created_at (formato iso)
                joined_raw = agent.get("created_at") or ""
                joined = joined_raw[:10] if joined_raw else None  # yyyy-mm-dd

                data = {
                    "name": name,
                    "karma": agent.get("karma", 0),
                    "description": agent.get("description"),
                    "human_owner": human_owner,
                    "joined": joined,
                    "followers": agent.get("follower_count", 0),
                    "following": agent.get("following_count", 0),
                }

                user = User.from_scraped_data(**data)
                self._batch.add(user)
                self._batch.flush_type(User)
                self._enriched_users.add(name)
                enriched += 1
                logger.info(
                    "enriquecido: %s (karma=%d, followers=%d, joined=%s, "
                    "owner=%s, desc=%s)",
                    name, data["karma"], data["followers"],
                    data["joined"], data["human_owner"],
                    (data["description"] or "")[:50],
                )
            except Exception as exc:
                logger.warning("no pude enriquecer perfil %s: %s", name, exc)

        logger.info("enrichment terminado: %d perfiles enriquecidos", enriched)
        return enriched

    def _process_posts(self, submolt_html: str, submolt: SubMolt,
                       max_posts: int, max_comments: int) -> List[str]:
        """proceso posts de un submolt. devuelvo la lista de autores descubiertos."""
        discovered_authors: Set[str] = set()
        posts_data = parse_posts_from_page(submolt_html, submolt.name, max_posts=max_posts)

        for pd in posts_data:
            author_name = pd.get("author_name")
            if not author_name:
                continue
            discovered_authors.add(author_name)

            self._batch.flush_type(User)
            user_id = self._resolve_user_id(author_name)
            self._batch.flush_type(User)

            self._register_user_submolt(user_id, submolt.id_submolt)
            self._batch.flush_type(UserSubMolt)

            post_url = pd.get("post_url") or ""
            pid = post_url.rsplit("/", 1)[-1] if post_url else None
            post = Post.from_scraped_data(
                id_user=user_id,
                id=pid or None,
                title=pd.get("title"),
                description=pd.get("description"),
                id_submolt=submolt.id_submolt,
                rating=pd.get("rating", 0),
                date=pd.get("date"),
                url=post_url or None,
            )
            self._batch.add(post)
            self._batch.flush_type(Post)

            # navego al post para extraer comentarios
            if post_url:
                try:
                    post_html = self._fetcher.fetch_with_cache(post_url, force_refresh=True)
                    self._process_comments(
                        post_html, post.id_post, submolt.id_submolt, max_comments,
                        discovered_authors=discovered_authors,
                    )
                except Exception as exc:
                    logger.error("error en comentarios de %s: %s", post_url, exc)

        return list(discovered_authors)

    def _process_comments(self, html: str, post_id: str,
                          submolt_id: str, max_comments: int,
                          discovered_authors: Optional[Set[str]] = None) -> None:
        if discovered_authors is None:
            discovered_authors = set()
        comments_data = parse_comments(html, post_id, max_comments=max_comments)

        for cd in comments_data:
            author_name = cd.get("author_name")
            if not author_name:
                continue

            self._batch.flush_type(User)
            user_id = self._resolve_user_id(author_name)
            self._batch.flush_type(User)
            discovered_authors.add(author_name)

            self._register_user_submolt(user_id, submolt_id)
            self._batch.flush_type(UserSubMolt)

            comment = Comment.from_scraped_data(
                id_user=user_id,
                id_post=post_id,
                description=cd.get("description"),
                date=cd.get("date"),
                rating=cd.get("rating", 0),
            )
            self._batch.add(comment)

        self._batch.flush_type(Comment)


# ---------------------------------------------------------------------------
# MAIN (parametros de glue o env)
# ---------------------------------------------------------------------------

def main() -> None:
    args = {}
    try:
        from awsglue.utils import getResolvedOptions
        args = getResolvedOptions(sys.argv, [
            "DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD",
            "MAX_USERS", "MAX_SUBMOLTS", "MAX_POSTS",
            "MAX_COMMENTS", "BATCH_SIZE",
        ])
    except Exception:
        logger.warning("getResolvedOptions fallo (ejecucion local?) -- uso env/cli")
        for i in range(1, len(sys.argv) - 1, 2):
            if sys.argv[i].startswith("--") and i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
                args[sys.argv[i][2:].strip()] = sys.argv[i + 1]
        for k in ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD",
                   "MAX_USERS", "MAX_SUBMOLTS", "MAX_POSTS", "MAX_COMMENTS", "BATCH_SIZE"]:
            args.setdefault(k, os.environ.get(k))

    os.environ.setdefault("DB_HOST", args.get("DB_HOST") or "")
    os.environ.setdefault("DB_NAME", args.get("DB_NAME") or "moltbook")
    os.environ.setdefault("DB_USER", args.get("DB_USER") or "")
    os.environ.setdefault("DB_PASSWORD", args.get("DB_PASSWORD") or "")

    if not os.environ.get("DB_HOST") or not os.environ.get("DB_NAME"):
        logger.error("necesito DB_HOST y DB_NAME (por job parameters o env).")
        sys.exit(1)

    max_users = int(args.get("MAX_USERS") or os.environ.get("MAX_USERS", "100"))
    max_submolts = int(args.get("MAX_SUBMOLTS") or os.environ.get("MAX_SUBMOLTS", "50"))
    max_posts = int(args.get("MAX_POSTS") or os.environ.get("MAX_POSTS", "20"))
    max_comments = int(args.get("MAX_COMMENTS") or os.environ.get("MAX_COMMENTS", "10"))
    batch_size = int(args.get("BATCH_SIZE") or os.environ.get("BATCH_SIZE", "5000"))

    logger.info(
        "inicio etl: host=%s db=%s max_users=%s max_submolts=%s",
        os.environ.get("DB_HOST"), os.environ.get("DB_NAME"), max_users, max_submolts,
    )

    db = Database(chunk_size=batch_size)

    try:
        with MoltbookScraper(db=db, batch_size=1000) as scraper:
            result = scraper.scrape_all(
                max_users=max_users,
                max_submolts=max_submolts,
                max_posts=max_posts,
                max_comments=max_comments,
            )
        logger.info("etl terminado: %s", result)
    except Exception as exc:
        logger.exception("fallo critico en el etl: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
