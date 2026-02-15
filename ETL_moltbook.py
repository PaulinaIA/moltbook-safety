#!/usr/bin/env python3
"""
ETL Moltbook -- AWS Glue Python Shell Job
Scraping (requests + BeautifulSoup), transformacion minima y carga en PostgreSQL (RDS).
Sin Spark, sin GlueContext, sin DynamicFrames.
"""

import hashlib
import logging
import re
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, TypeVar, Union
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Logging -- lo mando a stdout para que CloudWatch lo capture
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("etl_moltbook")


# ---------------------------------------------------------------------------
# Parametros del Glue Job
# ---------------------------------------------------------------------------

def _resolve_job_params() -> dict:
    """Leo los parametros que vienen del job de Glue
    (o del CLI si estoy corriendo en local)."""
    try:
        from awsglue.utils import getResolvedOptions
        params = getResolvedOptions(sys.argv, [
            "BATCH_SIZE",
            "MAX_USERS",
            "DB_HOST",
            "DB_PORT",
            "DB_NAME",
            "DB_USER",
            "DB_PASSWORD",
        ])
        return params
    except Exception:
        pass

    # Fallback para ejecucion local: parseo manual de --KEY value
    params: Dict[str, str] = {}
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i].startswith("--") and i + 1 < len(argv):
            key = argv[i].lstrip("-")
            params[key] = argv[i + 1]
            i += 2
        else:
            i += 1
    return params


# ===========================================================================
# ENTIDADES
# ===========================================================================

def generate_id(prefix: str, *args: str) -> str:
    """ID determinista: prefix + sha256 truncado."""
    content = "|".join(str(a) for a in args if a)
    h = hashlib.sha256(content.encode()).hexdigest()[:12]
    return f"{prefix}_{h}"


def _now() -> str:
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
    scraped_at: str = field(default_factory=_now)

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
        return cls(
            id_user=generate_id("user", name.lower()),
            name=name,
            karma=karma,
            description=description,
            human_owner=human_owner,
            joined=joined,
            followers=followers,
            following=following,
        )

    def to_dict(self) -> dict:
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
    def from_scraped_data(cls, name: str, description: Optional[str] = None) -> "SubMolt":
        return cls(
            id_submolt=generate_id("submolt", name.lower()),
            name=name,
            description=description,
        )

    def to_dict(self) -> dict:
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
    scraped_at: str = field(default_factory=_now)

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
    scraped_at: str = field(default_factory=_now)

    @classmethod
    def from_scraped_data(
        cls,
        id_user: str,
        id_post: str,
        description: Optional[str] = None,
        date: Optional[str] = None,
        rating: int = 0,
    ) -> "Comment":
        content_sample = (description or "")[:50]
        return cls(
            id_comment=generate_id("comment", id_user, id_post, content_sample),
            id_user=id_user,
            id_post=id_post,
            description=description,
            date=date,
            rating=rating,
        )

    def to_dict(self) -> dict:
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
    """Relacion N:M entre usuarios y submolts."""
    id_user: str
    id_submolt: str

    def to_dict(self) -> dict:
        return {"id_user": self.id_user, "id_submolt": self.id_submolt}


# ===========================================================================
# CONEXION A POSTGRESQL
# ===========================================================================

@contextmanager
def pg_connection(host: str, port: int, dbname: str, user: str, password: str):
    """Context manager para una conexion psycopg2 con commit/rollback automatico."""
    import psycopg2
    conn = psycopg2.connect(
        host=host, port=port, dbname=dbname, user=user, password=password,
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ===========================================================================
# OPERACIONES DE BASE DE DATOS
# ===========================================================================

# El orden de creacion importa por las FK
SCHEMA_STATEMENTS = [
    # -- Tablas base
    """
    CREATE TABLE IF NOT EXISTS users (
        id_user     VARCHAR(64) PRIMARY KEY,
        name        VARCHAR(255) NOT NULL,
        karma       INTEGER DEFAULT 0,
        description TEXT,
        human_owner VARCHAR(255),
        joined      VARCHAR(128),
        followers   INTEGER DEFAULT 0,
        following   INTEGER DEFAULT 0,
        scraped_at  VARCHAR(64) NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sub_molt (
        id_submolt  VARCHAR(64) PRIMARY KEY,
        name        VARCHAR(255) NOT NULL UNIQUE,
        description TEXT,
        scraped_at  VARCHAR(64) NOT NULL
    )
    """,
    # -- Intermedia: depende de users y sub_molt
    """
    CREATE TABLE IF NOT EXISTS user_submolt (
        id_user    VARCHAR(64) NOT NULL,
        id_submolt VARCHAR(64) NOT NULL,
        PRIMARY KEY (id_user, id_submolt)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS posts (
        id_post     VARCHAR(64) PRIMARY KEY,
        id_user     VARCHAR(64) NOT NULL,
        id_submolt  VARCHAR(64),
        title       TEXT,
        description TEXT,
        rating      INTEGER DEFAULT 0,
        date        VARCHAR(128),
        scraped_at  VARCHAR(64) NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS comments (
        id_comment  VARCHAR(64) PRIMARY KEY,
        id_user     VARCHAR(64) NOT NULL,
        id_post     VARCHAR(64) NOT NULL,
        description TEXT,
        date        VARCHAR(128),
        rating      INTEGER DEFAULT 0,
        scraped_at  VARCHAR(64) NOT NULL
    )
    """,
    # -- Indexes
    "CREATE INDEX IF NOT EXISTS idx_posts_user ON posts(id_user)",
    "CREATE INDEX IF NOT EXISTS idx_posts_submolt ON posts(id_submolt)",
    "CREATE INDEX IF NOT EXISTS idx_comments_user ON comments(id_user)",
    "CREATE INDEX IF NOT EXISTS idx_comments_post ON comments(id_post)",
    "CREATE INDEX IF NOT EXISTS idx_users_karma ON users(karma)",
    "CREATE INDEX IF NOT EXISTS idx_users_name ON users(name)",
]

# Las FK las agrego en sentencias separadas para que no falle si ya existen
FK_STATEMENTS = [
    "ALTER TABLE user_submolt ADD CONSTRAINT fk_us_user FOREIGN KEY (id_user) REFERENCES users(id_user)",
    "ALTER TABLE user_submolt ADD CONSTRAINT fk_us_submolt FOREIGN KEY (id_submolt) REFERENCES sub_molt(id_submolt)",
    "ALTER TABLE posts ADD CONSTRAINT fk_posts_user FOREIGN KEY (id_user) REFERENCES users(id_user)",
    "ALTER TABLE posts ADD CONSTRAINT fk_posts_submolt FOREIGN KEY (id_submolt) REFERENCES sub_molt(id_submolt)",
    "ALTER TABLE comments ADD CONSTRAINT fk_comments_user FOREIGN KEY (id_user) REFERENCES users(id_user)",
    "ALTER TABLE comments ADD CONSTRAINT fk_comments_post FOREIGN KEY (id_post) REFERENCES posts(id_post)",
]


class Database:
    """Operaciones sobre PostgreSQL. Manejo tablas, bulk upsert y consultas auxiliares."""

    TABLE_MAP = {
        User: "users",
        SubMolt: "sub_molt",
        UserSubMolt: "user_submolt",
        Post: "posts",
        Comment: "comments",
    }
    PK_MAP = {
        User: ("id_user",),
        SubMolt: ("id_submolt",),
        UserSubMolt: ("id_user", "id_submolt"),
        Post: ("id_post",),
        Comment: ("id_comment",),
    }

    def __init__(self, host: str, port: int, dbname: str, user: str, password: str,
                 chunk_size: int = 5000):
        self._host = host
        self._port = port
        self._dbname = dbname
        self._user = user
        self._password = password
        self.chunk_size = chunk_size

    def _connect(self):
        return pg_connection(
            self._host, self._port, self._dbname, self._user, self._password,
        )

    # -- DDL ----------------------------------------------------------------

    def ensure_tables(self) -> None:
        """Creo las tablas si no existen. Si ya estan, no hago nada."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_name = 'users' LIMIT 1"
                )
                if cur.fetchone() is not None:
                    logger.info("Tablas ya existen, salto DDL.")
                    return

        # Creo las tablas
        with self._connect() as conn:
            with conn.cursor() as cur:
                for stmt in SCHEMA_STATEMENTS:
                    cur.execute(stmt)

        # FKs por separado -- si ya existen las ignoro
        for fk in FK_STATEMENTS:
            try:
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(fk)
            except Exception as exc:
                if "already exists" in str(exc).lower():
                    continue
                logger.warning("FK omitida: %s", exc)

        logger.info("Esquema creado correctamente.")

    # -- Bulk upsert --------------------------------------------------------

    def bulk_upsert(self, entities: List[Any]) -> int:
        """Upsert por lotes usando execute_values de psycopg2."""
        if not entities:
            return 0
        etype = type(entities[0])
        table = self.TABLE_MAP.get(etype)
        pk_cols = self.PK_MAP.get(etype)
        if table is None or pk_cols is None:
            raise ValueError(f"Tipo de entidad no soportado: {etype}")

        data_list = [e.to_dict() for e in entities]
        columns = list(data_list[0].keys())
        update_cols = [c for c in columns if c not in pk_cols]

        # Para user_submolt no hay columnas de update, uso DO NOTHING
        if update_cols:
            conflict_action = "DO UPDATE SET " + ", ".join(
                f"{c} = EXCLUDED.{c}" for c in update_cols
            )
        else:
            conflict_action = "DO NOTHING"

        col_names = ", ".join(columns)
        conflict_target = ", ".join(pk_cols)
        sql = (
            f"INSERT INTO {table} ({col_names}) VALUES %s "
            f"ON CONFLICT ({conflict_target}) {conflict_action}"
        )
        template = "(" + ", ".join("%s" for _ in columns) + ")"

        total = 0
        for offset in range(0, len(data_list), self.chunk_size):
            chunk = data_list[offset : offset + self.chunk_size]
            rows = [list(d.values()) for d in chunk]
            try:
                from psycopg2.extras import execute_values
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        execute_values(cur, sql, rows, template=template, page_size=len(rows))
                total += len(rows)
            except Exception:
                # Fallback row-by-row si execute_values tiene algun problema
                placeholders = ", ".join("%s" for _ in columns)
                row_sql = (
                    f"INSERT INTO {table} ({col_names}) VALUES ({placeholders}) "
                    f"ON CONFLICT ({conflict_target}) {conflict_action}"
                )
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        for row in rows:
                            cur.execute(row_sql, row)
                total += len(rows)

        logger.info("bulk_upsert: %d registros en %s", total, table)
        return total

    # -- Consultas auxiliares ------------------------------------------------

    def get_user_names(self) -> List[str]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT name FROM users")
                return [r[0] for r in cur.fetchall()]

    def get_submolt_names(self) -> List[str]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT name FROM sub_molt")
                return [r[0] for r in cur.fetchall()]

    def get_user_id_by_name(self, name: str) -> Optional[str]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id_user FROM users WHERE name = %s LIMIT 1", (name,))
                row = cur.fetchone()
                return row[0] if row else None

    def count(self, entity_type: type) -> int:
        table = self.TABLE_MAP.get(entity_type)
        if table is None:
            return 0
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                return cur.fetchone()[0]


# ===========================================================================
# FETCHER (requests, sin Playwright)
# ===========================================================================

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
    """Obtengo HTML con requests. Nada de JS rendering."""

    use_browser = False

    def __init__(self, rate_limit: float = 1.0, timeout: int = 30):
        self._limiter = RateLimiter(rate_limit)
        self._timeout = timeout
        self._session = None
        self.cache_dir = Path("/tmp/moltbook_cache")

    @property
    def session(self):
        import requests as req
        if self._session is None:
            self._session = req.Session()
            self._session.headers.update({
                "User-Agent": "MoltbookScraper/1.0 (Academic Research Project)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            })
        return self._session

    def close(self) -> None:
        if self._session:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    def fetch_page(self, url: str, **_kwargs) -> str:
        self._limiter.wait()
        logger.info("GET %s", url)
        r = self.session.get(url, timeout=self._timeout)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        return r.text

    def _cache_filename(self, url: str) -> str:
        parsed = urlparse(url)
        safe = parsed.path.replace("/", "_").strip("_") or "index"
        h = hashlib.md5(url.encode()).hexdigest()[:8]
        return f"{safe}_{h}.html"

    def fetch_with_cache(self, url: str, force_refresh: bool = False, **_kwargs) -> str:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fname = self._cache_filename(url)
        path = self.cache_dir / fname
        if not force_refresh and path.exists():
            logger.debug("Cache hit: %s", fname)
            return path.read_text(encoding="utf-8")
        html = self.fetch_page(url)
        path.write_text(html, encoding="utf-8")
        return html

    def scroll_to_load_all(self, **_kwargs) -> None:
        pass


# ===========================================================================
# PARSERS (BeautifulSoup)
# ===========================================================================

def _parse_number(text: str) -> int:
    if not text:
        return 0
    text = text.strip().lower()
    for word in ("karma", "followers", "following", "points", "members"):
        text = text.replace(word, "").strip()
    mult = 1
    if text.endswith("k"):
        mult, text = 1000, text[:-1]
    elif text.endswith("m"):
        mult, text = 1_000_000, text[:-1]
    try:
        return int(float(text) * mult)
    except (ValueError, TypeError):
        return 0


def _relative_date(text: str) -> str:
    """Convierte '9d ago', '2h ago', etc. en YYYY-MM-DD."""
    now = datetime.utcnow()
    m = re.search(r"(\d+)([a-z]+)", text.lower())
    if not m:
        return now.strftime("%Y-%m-%d")
    val, unit = int(m.group(1)), m.group(2)
    if "mo" in unit:
        delta = timedelta(days=val * 30)
    elif "w" in unit:
        delta = timedelta(weeks=val)
    elif "d" in unit:
        delta = timedelta(days=val)
    elif "h" in unit:
        delta = timedelta(hours=val)
    elif "m" in unit:
        delta = timedelta(minutes=val)
    else:
        delta = timedelta()
    return (now - delta).strftime("%Y-%m-%d")


def parse_users_list(html: str) -> List[Dict[str, Any]]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    users: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for link in soup.select("a[href^='/u/']"):
        href = link.get("href", "")
        match = re.search(r"/u/([^/?#]+)", href)
        if not match or match.group(1) in seen:
            continue
        name = match.group(1)
        seen.add(name)
        parent = link.find_parent("div")
        karma_text = ""
        if parent:
            k = parent.find(string=re.compile(r"karma", re.I))
            if k and k.parent:
                karma_text = k.parent.get_text()
        users.append({"name": name, "karma": _parse_number(karma_text), "profile_url": href})
    return users


def parse_user_profile(html: str, username: str) -> Dict[str, Any]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    result: Dict[str, Any] = {
        "name": username, "karma": 0, "description": None,
        "human_owner": None, "joined": None, "followers": 0, "following": 0,
    }
    body = soup.get_text()
    for pat in [r"(\d+(?:\.\d+)?[KkMm]?)\s*karma", r"karma[:\s]*(\d+(?:\.\d+)?[KkMm]?)"]:
        m = re.search(pat, body, re.I)
        if m:
            result["karma"] = _parse_number(m.group(1))
            break
    meta = soup.find("meta", {"name": "description"})
    if meta and meta.get("content"):
        result["description"] = meta["content"]
    for sel in ("p.bio", "div.description", "p.text-gray-400"):
        el = soup.select_one(sel)
        if el:
            result["description"] = el.get_text(strip=True)
            break
    tw = soup.select_one("a[href*='twitter.com'], a[href*='x.com']")
    if tw and tw.get("href"):
        mm = re.search(r"(?:twitter\.com|x\.com)/(@?\w+)", tw["href"])
        if mm:
            result["human_owner"] = mm.group(1)
    j = re.search(r"joined\s+(.+?)(?:\s*\||$)", body, re.I)
    if j:
        result["joined"] = j.group(1).strip()
    ff = re.search(r"(\d+(?:\.\d+)?[KkMm]?)\s*followers?", body, re.I)
    if ff:
        result["followers"] = _parse_number(ff.group(1))
    fl = re.search(r"(\d+(?:\.\d+)?[KkMm]?)\s*following", body, re.I)
    if fl:
        result["following"] = _parse_number(fl.group(1))
    return result


def parse_submolt_list(html: str) -> List[Dict[str, Any]]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
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
    result: Dict[str, Any] = {"name": submolt_name, "description": None}
    meta = soup.find("meta", {"name": "description"})
    if meta and meta.get("content"):
        result["description"] = meta["content"]
    for sel in ("p.description", "div.about", "p.text-gray-400"):
        el = soup.select_one(sel)
        if el:
            result["description"] = el.get_text(strip=True)
            break
    return result


def _parse_post_element(el, submolt_name: Optional[str] = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "title": None, "description": None, "author_name": None,
        "submolt_name": submolt_name, "rating": 0, "date": None, "post_url": None,
    }
    href = el.get("href")
    if href:
        result["post_url"] = f"https://www.moltbook.com{href}" if href.startswith("/") else href
    h3 = el.select_one("h3")
    if h3:
        result["title"] = h3.get_text(strip=True)
    p = el.select_one("p.line-clamp-3")
    if p:
        result["description"] = p.get_text(strip=True)
    author = el.select_one("span.hover\\:underline")
    if author:
        result["author_name"] = author.get_text(strip=True).replace("u/", "")
    if not submolt_name:
        sm_link = el.select_one("a[href^='/m/']")
        if sm_link:
            mm = re.search(r"/m/([^/?#]+)", sm_link.get("href", ""))
            if mm:
                result["submolt_name"] = mm.group(1)
    rating_el = el.select_one("span.text-white.font-bold")
    if rating_el:
        try:
            result["rating"] = int(rating_el.get_text(strip=True))
        except ValueError:
            pass
    xs = el.select_one("div.text-xs")
    if xs:
        txt = xs.get_text(" ", strip=True)
        if "ago" in txt:
            result["date"] = _relative_date(txt.split("ago")[0].strip() + " ago")
        else:
            result["date"] = datetime.utcnow().strftime("%Y-%m-%d")
    return result


def parse_posts_from_page(
    html: str, submolt_name: Optional[str] = None, max_posts: int = 10,
) -> List[Dict[str, Any]]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    posts: List[Dict[str, Any]] = []
    for container in soup.select("a[href^='/post/']"):
        post = _parse_post_element(container, submolt_name)
        if post.get("title") or post.get("post_url"):
            posts.append(post)
            if len(posts) >= max_posts:
                break
    return posts


def parse_comments(html: str, post_id: str, max_comments: int = 10) -> List[Dict[str, Any]]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    out: List[Dict[str, Any]] = []
    for container in soup.select("div.mt-6 div.py-2"):
        r: Dict[str, Any] = {
            "post_id": post_id, "author_name": None,
            "description": None, "date": None, "rating": 0,
        }
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
            if len(out) >= max_comments:
                break
    return out


# ===========================================================================
# DISCOVERY
# ===========================================================================

BASE_URL = "https://www.moltbook.com"


class URLDiscovery:
    def __init__(self, fetcher: RequestsFetcher):
        self.fetcher = fetcher

    def discover_users(self, max_users: int = 100,
                       known_users: Optional[Set[str]] = None) -> List[str]:
        known = known_users or set()
        html = self.fetcher.fetch_page(f"{BASE_URL}/u")
        urls: List[str] = []
        for u in parse_users_list(html):
            if len(urls) >= max_users:
                break
            name = u.get("name", "")
            if name and name not in known:
                urls.append(f"{BASE_URL}/u/{name}")
        logger.info("Descubri %d URLs de usuarios", len(urls))
        return urls

    def discover_submolts(self, max_submolts: int = 50,
                          known_submolts: Optional[Set[str]] = None) -> List[str]:
        known = known_submolts or set()
        html = self.fetcher.fetch_page(f"{BASE_URL}/m")
        urls: List[str] = []
        for s in parse_submolt_list(html):
            if len(urls) >= max_submolts:
                break
            name = s.get("name", "")
            if name and name not in known:
                urls.append(f"{BASE_URL}/m/{name}")
        logger.info("Descubri %d URLs de submolts", len(urls))
        return urls


# ===========================================================================
# BATCH WRITER
# ===========================================================================

T = TypeVar("T", User, SubMolt, UserSubMolt, Post, Comment)


class BatchWriter:
    """Acumulo entidades y las flusheo a la DB cuando el buffer alcanza el tamano del batch.

    El orden de flush respeta la integridad referencial:
        users -> sub_molt -> user_submolt -> posts -> comments
    """

    _FLUSH_ORDER: List[type] = [User, SubMolt, UserSubMolt, Post, Comment]

    def __init__(self, db: Database, batch_size: int = 1000):
        self.db = db
        self.batch_size = batch_size
        self._buf: Dict[type, list] = defaultdict(list)

    def add(self, entity: Any) -> None:
        t = type(entity)
        self._buf[t].append(entity)
        if len(self._buf[t]) >= self.batch_size:
            self._flush(t)

    def _flush(self, entity_type: type) -> int:
        lst = self._buf.get(entity_type)
        if not lst:
            return 0
        chunk = list(lst)
        self._buf[entity_type] = []
        return self.db.bulk_upsert(chunk)

    def flush_type(self, entity_type: type) -> int:
        return self._flush(entity_type)

    def flush_all(self) -> None:
        """Flush en orden para no romper FKs."""
        for t in self._FLUSH_ORDER:
            if self._buf.get(t):
                self._flush(t)


# ===========================================================================
# SCRAPER PRINCIPAL
# ===========================================================================

class MoltbookScraper:
    """Orquesto extraccion, transformacion minima y carga.

    Todo pasa por el BatchWriter que acumula y flushea respetando
    el orden de dependencia de las tablas.
    """

    def __init__(self, db: Database, batch_size: int = 1000):
        self.db = db
        self._fetcher: Optional[RequestsFetcher] = None
        self._discovery: Optional[URLDiscovery] = None
        self._batch = BatchWriter(db, batch_size=batch_size)

        # Registro local de user_submolt que ya meti en el batch,
        # para no duplicar inserts dentro de la misma ejecucion
        self._user_submolt_seen: Set[tuple] = set()

    def __enter__(self) -> "MoltbookScraper":
        self.db.ensure_tables()
        self._fetcher = RequestsFetcher(rate_limit=1.0, timeout=30)
        self._discovery = URLDiscovery(self._fetcher)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            self._batch.flush_all()
        finally:
            if self._fetcher:
                self._fetcher.close()

    # -- helpers internos ---------------------------------------------------

    def _resolve_user_id(self, author_name: str) -> str:
        """Si el usuario ya existe en la DB, devuelvo su id.
        Si no, creo uno nuevo y lo meto al batch."""
        uid = self.db.get_user_id_by_name(author_name)
        if uid:
            return uid
        new_user = User.from_scraped_data(name=author_name)
        self._batch.add(new_user)
        return new_user.id_user

    def _register_user_submolt(self, user_id: str, submolt_id: str) -> None:
        """Registro la relacion user <-> submolt si aun no la tengo."""
        key = (user_id, submolt_id)
        if key in self._user_submolt_seen:
            return
        self._user_submolt_seen.add(key)
        self._batch.add(UserSubMolt(id_user=user_id, id_submolt=submolt_id))

    # -- scraping -----------------------------------------------------------

    def scrape_all(
        self,
        max_users: int = 100,
        max_submolts: int = 50,
        max_posts: int = 10,
        max_comments: int = 100,
        force_refresh: bool = False,
    ) -> Dict[str, int]:
        known_users = set(self.db.get_user_names())
        known_submolts = set(self.db.get_submolt_names())

        # --- Usuarios ---
        user_urls = self._discovery.discover_users(
            max_users=max_users,
            known_users=known_users if not force_refresh else None,
        )
        users_scraped = 0
        for url in user_urls:
            try:
                username = url.rsplit("/u/", 1)[-1]
                html = self._fetcher.fetch_with_cache(url, force_refresh=force_refresh)
                data = parse_user_profile(html, username)
                user = User.from_scraped_data(**data)
                self._batch.add(user)
                users_scraped += 1
                logger.info("Usuario: %s (karma=%d)", user.name, user.karma)
            except Exception as exc:
                logger.error("Error scrapeando usuario %s: %s", url, exc)

        # Necesito que los usuarios existan antes de meter posts
        self._batch.flush_type(User)

        # --- Submolts (con posts y comments) ---
        submolt_urls = self._discovery.discover_submolts(
            max_submolts=max_submolts,
            known_submolts=known_submolts if not force_refresh else None,
        )
        submolts_scraped = 0
        for url in submolt_urls:
            try:
                name = url.rsplit("/m/", 1)[-1]
                html = self._fetcher.fetch_with_cache(url, force_refresh=force_refresh)
                sm_data = parse_submolt_page(html, name)
                submolt = SubMolt.from_scraped_data(**sm_data)
                self._batch.add(submolt)
                submolts_scraped += 1
                logger.info("Submolt: %s", submolt.name)

                # Flush submolts y users antes de los posts (FK)
                self._batch.flush_type(SubMolt)
                self._batch.flush_type(User)

                self._scrape_posts(html, submolt, max_posts, max_comments)
            except Exception as exc:
                logger.error("Error scrapeando submolt %s: %s", url, exc)

        # Flush final de todo lo que quede pendiente
        self._batch.flush_all()

        return {
            "users": users_scraped,
            "submolts": submolts_scraped,
            "posts": self.db.count(Post),
            "comments": self.db.count(Comment),
            "user_submolt": self.db.count(UserSubMolt),
        }

    def _scrape_posts(self, html: str, submolt: SubMolt,
                      max_posts: int, max_comments: int) -> None:
        posts_data = parse_posts_from_page(html, submolt.name, max_posts=max_posts)

        for pd in posts_data:
            author_name = pd.get("author_name")
            if not author_name:
                continue

            # Necesito flushear usuarios antes de resolver IDs
            self._batch.flush_type(User)

            user_id = self._resolve_user_id(author_name)
            self._batch.flush_type(User)

            # Relacion user_submolt
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

            # Comentarios del post
            if post_url:
                try:
                    post_html = self._fetcher.fetch_with_cache(
                        post_url, force_refresh=True,
                    )
                    self._scrape_comments(post_html, post.id_post, submolt.id_submolt, max_comments)
                except Exception as exc:
                    logger.error("Error en comentarios de %s: %s", post_url, exc)

    def _scrape_comments(self, html: str, post_id: str,
                         submolt_id: str, max_comments: int) -> None:
        comments_data = parse_comments(html, post_id, max_comments=max_comments)

        for cd in comments_data:
            author_name = cd.get("author_name")
            if not author_name:
                continue

            self._batch.flush_type(User)
            user_id = self._resolve_user_id(author_name)
            self._batch.flush_type(User)

            # El comentarista tambien pertenece al submolt
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


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    params = _resolve_job_params()

    db_host = params.get("DB_HOST", "")
    db_port = int(params.get("DB_PORT", "5432"))
    db_name = params.get("DB_NAME", "")
    db_user = params.get("DB_USER", "")
    db_password = params.get("DB_PASSWORD", "")
    batch_size = int(params.get("BATCH_SIZE", "5000"))
    max_users = int(params.get("MAX_USERS", "100"))

    if not db_host or not db_name:
        logger.error("Faltan parametros obligatorios: DB_HOST y DB_NAME.")
        sys.exit(1)

    logger.info(
        "Inicio ETL -- host=%s port=%d db=%s max_users=%d batch_size=%d",
        db_host, db_port, db_name, max_users, batch_size,
    )

    db = Database(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        chunk_size=batch_size,
    )

    try:
        with MoltbookScraper(db=db, batch_size=batch_size) as scraper:
            result = scraper.scrape_all(
                max_users=max_users,
                max_submolts=50,
                max_posts=10,
                max_comments=100,
                force_refresh=False,
            )
        logger.info("ETL terminado: %s", result)
    except Exception as exc:
        logger.exception("Fallo critico en el ETL: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
