#!/usr/bin/env python3
"""
ETL Moltbook -- AWS Glue Python Shell Job

Navego moltbook.com con Playwright (headless), guardo el HTML crudo
en S3, parseo con BeautifulSoup y cargo a PostgreSQL (RDS).

Sin Spark, sin GlueContext, sin DynamicFrames.
"""

import hashlib
import logging
import os
import re
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, TypeVar, Union

from glue_job_script import _pg_connection

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
            "S3_HTML_BUCKET",
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

    # Variables de entorno como fallback adicional
    for key in ("S3_HTML_BUCKET", "DB_HOST", "DB_PORT", "DB_NAME",
                "DB_USER", "DB_PASSWORD", "BATCH_SIZE", "MAX_USERS"):
        if key not in params and os.environ.get(key):
            params[key] = os.environ[key]

    return params


# ===========================================================================
# CONSTANTES
# ===========================================================================

BASE_URL = "https://www.moltbook.com"

# Submolts que conozco de antemano. Los uso para iterar directamente
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
    "CREATE INDEX IF NOT EXISTS idx_posts_user ON posts(id_user)",
    "CREATE INDEX IF NOT EXISTS idx_posts_submolt ON posts(id_submolt)",
    "CREATE INDEX IF NOT EXISTS idx_comments_user ON comments(id_user)",
    "CREATE INDEX IF NOT EXISTS idx_comments_post ON comments(id_post)",
    "CREATE INDEX IF NOT EXISTS idx_users_karma ON users(karma)",
    "CREATE INDEX IF NOT EXISTS idx_users_name ON users(name)",
]

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

    def ensure_tables(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_name = 'users' LIMIT 1"
                )
                if cur.fetchone() is not None:
                    logger.info("Tablas ya existen, salto DDL.")
                    return

        with self._connect() as conn:
            with conn.cursor() as cur:
                for stmt in SCHEMA_STATEMENTS:
                    cur.execute(stmt)

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

    def bulk_upsert(self, entities: List[Any]) -> int:
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
    
    def get_submolt_id_by_name(self, name: str) -> Optional[str]:
        """devuelvo el id del submolt si ya existe en la db."""
        with _pg_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id_submolt FROM sub_molt WHERE name = %s LIMIT 1", (name,))
            row = cur.fetchone()
            return row[0] if row else None


# ===========================================================================
# S3 HTML STORAGE
# ===========================================================================

class S3HTMLStore:
    """Guardo y recupero HTML crudo desde S3.

    El key de cada objeto sigue el patron: html/{entity_type}/{name}_{hash}.html
    para que sea determinista y trazable.
    """

    def __init__(self, bucket: str, prefix: str = "html"):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self._client = None

    @property
    def s3(self):
        if self._client is None:
            import boto3
            self._client = boto3.client("s3")
        return self._client

    def _make_key(self, entity_type: str, name: str) -> str:
        """Genero un key determinista: html/submolt/general_a1b2c3d4.html"""
        h = hashlib.md5(name.encode()).hexdigest()[:8]
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        return f"{self.prefix}/{entity_type}/{safe_name}_{h}.html"

    def put(self, html: str, entity_type: str, name: str) -> str:
        """Subo el HTML a S3 y devuelvo el key."""
        key = self._make_key(entity_type, name)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=html.encode("utf-8"),
            ContentType="text/html; charset=utf-8",
        )
        logger.debug("S3 PUT s3://%s/%s (%d bytes)", self.bucket, key, len(html))
        return key

    def get(self, entity_type: str, name: str) -> Optional[str]:
        """Intento leer HTML desde S3. None si no existe."""
        key = self._make_key(entity_type, name)
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read().decode("utf-8")
        except Exception:
            return None


# ===========================================================================
# PLAYWRIGHT FETCHER
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


class PlaywrightFetcher:
    """Navego paginas con Playwright para renderizar el JS de moltbook (SPA Next.js).
    Cada pagina que obtengo la persisto en S3 antes de devolver el HTML.
    """

    def __init__(
        self,
        s3_store: Optional[S3HTMLStore] = None,
        rate_limit: float = 1.0,
        timeout: int = 30,
        headless: bool = True,
    ):
        self._s3 = s3_store
        self._limiter = RateLimiter(rate_limit)
        self._timeout = timeout
        self._headless = headless
        self._pw = None
        self._browser = None
        self._page = None

    def start(self) -> None:
        if self._browser is not None:
            return
        from playwright.sync_api import sync_playwright
        logger.info("Iniciando Playwright (headless=%s)", self._headless)
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self._headless)
        self._page = self._browser.new_page(
            user_agent="MoltbookScraper/1.0 (Academic Research Project)",
        )
        self._page.set_default_timeout(self._timeout * 1000)

    def close(self) -> None:
        if self._page:
            try:
                self._page.close()
            except Exception:
                pass
            self._page = None
        if self._browser:
            try:
                self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._pw:
            try:
                self._pw.stop()
            except Exception:
                pass
            self._pw = None
        logger.info("Playwright cerrado")

    @property
    def page(self):
        if self._page is None:
            self.start()
        return self._page

    def fetch(
        self,
        url: str,
        wait_selector: Optional[str] = None,
        wait_ms: int = 2000,
        entity_type: str = "page",
        entity_name: str = "unknown",
        retries: int = 2,
    ) -> str:
        """Navego a la URL, espero a que cargue el contenido dinamico,
        guardo en S3 y devuelvo el HTML.

        Moltbook es un SPA (Next.js) asi que el HTML inicial solo tiene
        un div con 'Loading...'. Necesito esperar a que el JS renderice
        el contenido real. Si despues de esperar sigo viendo 'Loading...',
        reintento desde cero.
        """
        for attempt in range(retries + 1):
            self._limiter.wait()
            if attempt > 0:
                logger.info("Reintento %d/%d para %s", attempt, retries, url)
            else:
                logger.info("GET %s", url)

            try:
                self.page.goto(url, wait_until="domcontentloaded")
            except Exception as exc:
                logger.warning("Timeout en goto %s: %s -- reintento con commit", url, exc)
                try:
                    self.page.goto(url, wait_until="commit")
                except Exception as exc2:
                    logger.error("Fallo definitivo en goto %s: %s", url, exc2)
                    if attempt < retries:
                        continue
                    raise

            # Espero a que el SPA renderice. Primero intento el selector especifico,
            # si no funciona, espero a que desaparezca el Loading...
            if wait_selector:
                try:
                    self.page.wait_for_selector(wait_selector, timeout=15000)
                except Exception:
                    logger.warning(
                        "Selector '%s' no aparecio en %s (intento %d) -- espero mas",
                        wait_selector, url, attempt + 1,
                    )
                    # Fallback: espero a que aparezca cualquier contenido de perfil
                    try:
                        self.page.wait_for_selector(
                            "div.flex-1, main, h1", timeout=10000,
                        )
                    except Exception:
                        pass

            self.page.wait_for_timeout(wait_ms)
            html = self.page.content()

            # Verifico que el SPA haya renderizado. Si solo veo "Loading..."
            # sin ningun contenido real, reintento.
            if "Loading..." in html and 'class="flex-1"' not in html:
                logger.warning(
                    "Pagina %s sigue en estado Loading... (intento %d/%d)",
                    url, attempt + 1, retries + 1,
                )
                if attempt < retries:
                    # Espero un poco mas antes de reintentar
                    self.page.wait_for_timeout(5000)
                    html = self.page.content()
                    if "Loading..." in html and 'class="flex-1"' not in html:
                        continue

            # Persisto en S3
            if self._s3:
                try:
                    self._s3.put(html, entity_type, entity_name)
                except Exception as exc:
                    logger.warning("No pude guardar HTML en S3 para %s/%s: %s",
                                   entity_type, entity_name, exc)

            return html

        # Si llegamos aqui, todos los reintentos fallaron
        logger.error("Todos los reintentos fallaron para %s", url)
        return html  # Devuelvo lo que tenga, aunque sea Loading...

    def scroll_and_fetch(
        self,
        url: str,
        max_scrolls: int = 5,
        scroll_delay: int = 1500,
        entity_type: str = "page",
        entity_name: str = "unknown",
    ) -> str:
        """Navego, hago scroll para cargar mas contenido, y devuelvo el HTML completo."""
        self._limiter.wait()
        logger.info("GET+scroll %s", url)

        self.page.goto(url, wait_until="domcontentloaded")
        self.page.wait_for_timeout(2000)

        for _ in range(max_scrolls):
            self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            self.page.wait_for_timeout(scroll_delay)

        html = self.page.content()

        if self._s3:
            try:
                self._s3.put(html, entity_type, entity_name)
            except Exception as exc:
                logger.warning("No pude guardar HTML en S3: %s", exc)

        return html


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
    """Extraigo datos del perfil de un usuario desde el HTML renderizado.

    Estructura real del DOM (renderizado por Next.js):
      div.flex.items-start.gap-4 > div.flex-1
        [0] div.flex.items-center.gap-2    -> h1 (username) + verified badge
        [1] p.mt-1                          -> bio/description
        [2] div.flex.flex-wrap...mt-3       -> stats row (karma, followers, following, joined)
        [3] div.mt-4.pt-4.border-t          -> human owner section con link a X
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    result: Dict[str, Any] = {
        "name": username, "karma": 0, "description": None,
        "human_owner": None, "joined": None, "followers": 0, "following": 0,
    }

    # Busco el bloque principal del perfil
    profile_block = soup.select_one("div.flex.items-start.gap-4 > div.flex-1")

    # -- Bio / descripcion --
    # Esta en un <p> con clase mt-1 dentro del bloque de perfil
    if profile_block:
        bio_el = profile_block.select_one("p.mt-1")
        if bio_el:
            result["description"] = bio_el.get_text(strip=True)
    if not result["description"]:
        for sel in ("p.mt-1", "p.text-gray-400"):
            el = soup.select_one(sel)
            if el:
                text = el.get_text(strip=True)
                # Filtro la descripcion generica del sitio
                if text and "social network" not in text.lower():
                    result["description"] = text
                    break

    # -- Stats row: karma, followers, following, joined --
    # El row de stats tiene clase "flex flex-wrap items-center gap-4 mt-3"
    stats_row = (
        profile_block.select_one("div.mt-3")
        if profile_block
        else soup.select_one("div.flex.flex-wrap.items-center.mt-3")
    )
    if stats_row:
        stat_divs = stats_row.select("div.text-sm")
        for div in stat_divs:
            text = div.get_text(" ", strip=True).lower()
            # Karma: <span class="font-bold">1876</span> karma
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
                # Formato: "🎂 Joined 1/29/2026"
                m = re.search(r"joined\s+(\S+)", text, re.I)
                if m:
                    result["joined"] = m.group(1)

    # Fallback con regex si no encontre stats por DOM
    if result["karma"] == 0:
        body = soup.get_text(" ", strip=True)
        m = re.search(r"(\d+(?:\.\d+)?[KkMm]?)\s*karma", body, re.I)
        if m:
            result["karma"] = _parse_number(m.group(1))

    # -- Human owner --
    # Esta en un div.mt-4.pt-4.border-t que contiene link a x.com
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
        # Si no hay link a X, busco el texto del nombre del owner
        if not result["human_owner"]:
            header = owner_section.select_one("div.text-xs")
            if header:
                # Hay un nombre o handle despues del header
                name_el = owner_section.select_one("span.font-bold, span.text-white")
                if name_el:
                    result["human_owner"] = name_el.get_text(strip=True)
    else:
        # Fallback: primer link a x.com que no sea del footer
        for a in soup.select('a[href*="x.com"], a[href*="twitter.com"]'):
            href = a.get("href", "")
            if "mattprd" in href:
                continue
            mm = re.search(r"(?:twitter\.com|x\.com)/(@?\w+)", href)
            if mm:
                result["human_owner"] = mm.group(1)
                break

    # -- Joined (fallback si no lo saque de stats) --
    if not result["joined"]:
        body = soup.get_text(" ", strip=True)
        j = re.search(r"Joined\s+(\d{1,2}/\d{1,2}/\d{4})", body)
        if j:
            result["joined"] = j.group(1)
        else:
            j2 = re.search(r"Joined\s+(.+?)(?:\s+Online|\s*$)", body)
            if j2:
                result["joined"] = j2.group(1).strip()[:30]

    # -- Followers/following fallback --
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
        result["post_url"] = f"{BASE_URL}{href}" if href.startswith("/") else href
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


def parse_comments(html: str, post_id: str, max_comments: int = 100) -> List[Dict[str, Any]]:
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
        for t in self._FLUSH_ORDER:
            if self._buf.get(t):
                self._flush(t)


# ===========================================================================
# SCRAPER PRINCIPAL
# ===========================================================================

class MoltbookScraper:
    """Orquesto extraccion, transformacion minima y carga.

    Playwright navega las paginas, guarda HTML en S3 y luego parseo
    con BeautifulSoup. Todo pasa por el BatchWriter que flushea
    respetando el orden de FK.
    """

    def __init__(
        self,
        db: Database,
        fetcher: PlaywrightFetcher,
        batch_size: int = 1000,
    ):
        self.db = db
        self._fetcher = fetcher
        self._batch = BatchWriter(db, batch_size=batch_size)
        self._user_submolt_seen: Set[tuple] = set()
        self._enriched_users: Set[str] = set()

    def __enter__(self) -> "MoltbookScraper":
        self.db.ensure_tables()
        self._fetcher.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            self._batch.flush_all()
        finally:
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
        key = (user_id, submolt_id)
        if key in self._user_submolt_seen:
            return
        self._user_submolt_seen.add(key)
        self._batch.add(UserSubMolt(id_user=user_id, id_submolt=submolt_id))

    # -- discovery ----------------------------------------------------------

    def _discover_submolts(self, max_submolts: int) -> List[str]:
        """Descubro submolts navegando /m con scroll, y complemento con la lista semilla."""
        discovered: Set[str] = set()
        try:
            html = self._fetcher.scroll_and_fetch(
                f"{BASE_URL}/m",
                max_scrolls=5,
                entity_type="listing",
                entity_name="submolts",
            )
            for item in parse_submolt_list(html):
                name = item.get("name", "")
                if name:
                    discovered.add(name)
            logger.info("Descubri %d submolts desde /m", len(discovered))
        except Exception as exc:
            logger.warning("No pude navegar /m: %s -- uso solo semilla", exc)

        # Complemento con la semilla
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
        max_users: int = 10000,
        max_submolts: int = 100,
        max_posts: int = 10,
        max_comments: int = 2,
    ) -> Dict[str, int]:
        """Pipeline completo. Enriquezco usuarios de forma incremental
        despues de cada submolt para que si interrumpen el ETL,
        los datos de perfil ya esten en la DB.
        """
        # --- 1. Submolts + enrichment intercalado ---
        submolt_names = self._discover_submolts(max_submolts)
        logger.info("Procesando %d submolts", len(submolt_names))

        submolts_ok = 0
        for name in submolt_names:
            try:
                existing_sm_id = self.db.get_submolt_id_by_name(name)
                if existing_sm_id:
                    logger.info("Submolt %s ya existe en DB con id %s", name, existing_sm_id)
                    continue
                else:
                    logger.info("Submolt %s es nueva, la voy a procesar", name)
                    # self._register_user_submolt(user_id="system", submolt_id=existing_sm_id)

                url = f"{BASE_URL}/m/{name}"
                html = self._fetcher.fetch(
                    url,
                    wait_selector="a[href^='/post/']",
                    wait_ms=3000,
                    entity_type="submolt",
                    entity_name=name,
                )

                sm_data = parse_submolt_page(html, name)
                submolt = SubMolt.from_scraped_data(**sm_data)
                self._batch.add(submolt)
                self._batch.flush_type(SubMolt)
                self._batch.flush_type(User)
                submolts_ok += 1

                logger.info("Submolt: %s", submolt.name)
                new_authors = self._process_posts(html, submolt, max_posts, max_comments)

                # Enriquezco inmediatamente los usuarios descubiertos en este submolt
                pending = [n for n in new_authors if n not in self._enriched_users]
                if pending:
                    logger.info(
                        "Enriqueciendo %d usuarios descubiertos en m/%s",
                        len(pending), name,
                    )
                    self._enrich_user_profiles(pending)

            except Exception as exc:
                logger.error("Error en submolt %s: %s", name, exc)

        self._batch.flush_all()

        # --- 2. Sweep final: usuarios que quedaron sin enriquecer ---
        db_users = self.db.get_user_names()
        remaining = [n for n in db_users if n not in self._enriched_users]
        if remaining:
            logger.info(
                "Sweep final: %d usuarios pendientes de enriquecer",
                len(remaining),
            )
            self._enrich_user_profiles(remaining)

        self._batch.flush_all()

        return {
            "users": self.db.count(User),
            "users_enriched": len(self._enriched_users),
            "submolts": submolts_ok,
            "posts": self.db.count(Post),
            "comments": self.db.count(Comment),
            "user_submolt": self.db.count(UserSubMolt),
        }

    def _enrich_user_profiles(self, user_names: List[str]) -> int:
        """Visito /u/{name} para obtener karma, bio, etc.
        Flusheo cada usuario enriquecido inmediatamente a la DB
        para que no se pierdan datos si el ETL se interrumpe."""
        enriched = 0
        logger.info("Enriqueciendo %d perfiles de usuario...", len(user_names))
        for name in user_names:
            if name in self._enriched_users:
                continue
            try:
                url = f"{BASE_URL}/u/{name}"
                logger.info("Visitando perfil: %s", url)

                # Espero al row de stats (karma, followers) que confirma que
                # el JS termino de renderizar el perfil completo
                html = self._fetcher.fetch(
                    url,
                    wait_selector="div.flex.flex-wrap.items-center.gap-4.mt-3",
                    wait_ms=3000,
                    entity_type="user",
                    entity_name=name,
                    retries=2,
                )

                # Verifico que el HTML contenga contenido de perfil
                if "Loading..." in html and "karma" not in html.lower():
                    logger.warning(
                        "Perfil %s no renderizo correctamente -- HTML tiene Loading...",
                        name,
                    )
                    continue

                data = parse_user_profile(html, name)

                # Si no obtuve nada util, lo reporto pero igual guardo
                if data.get("karma", 0) == 0 and not data.get("description"):
                    logger.warning(
                        "Perfil %s: parser no encontro datos (karma=0, desc=None). "
                        "Puede que la pagina no haya cargado bien.",
                        name,
                    )

                user = User.from_scraped_data(**data)
                self._batch.add(user)
                # Flusheo inmediatamente para que el dato quede en la DB
                self._batch.flush_type(User)
                self._enriched_users.add(name)
                enriched += 1
                logger.info(
                    "Enriquecido: %s (karma=%d, followers=%d, joined=%s, "
                    "owner=%s, desc=%s)",
                    name, data.get("karma", 0), data.get("followers", 0),
                    data.get("joined"), data.get("human_owner"),
                    (data.get("description") or "")[:50],
                )
            except Exception as exc:
                logger.warning("No pude enriquecer perfil %s: %s", name, exc)

        logger.info("Enrichment terminado: %d perfiles enriquecidos", enriched)
        return enriched

    def _process_posts(self, submolt_html: str, submolt: SubMolt,
                       max_posts: int, max_comments: int) -> List[str]:
        """Proceso posts de un submolt. Devuelvo la lista de autores descubiertos."""
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

            # Navego al post para extraer comentarios
            if post_url:
                try:
                    post_name = post_url.rsplit("/", 1)[-1]
                    post_html = self._fetcher.fetch(
                        post_url,
                        wait_selector="div.mt-6",
                        wait_ms=2500,
                        entity_type="post",
                        entity_name=post_name,
                    )
                    self._process_comments(
                        post_html, post.id_post, submolt.id_submolt, max_comments,
                        discovered_authors=discovered_authors,
                    )
                except Exception as exc:
                    logger.error("Error en comentarios de %s: %s", post_url, exc)

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


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    params = _resolve_job_params()

    db_host = params.get("DB_HOST", "moltbook.ce3qywi6qo2c.us-east-1.rds.amazonaws.com")
    db_port = int(params.get("DB_PORT", "5432"))
    db_name = params.get("DB_NAME", "moltbook")
    db_user = params.get("DB_USER", "postgres")
    db_password = params.get("DB_PASSWORD", "Cl6rS2FxuKTkp2lQ")
    batch_size = int(params.get("BATCH_SIZE", "50"))
    max_users = int(params.get("MAX_USERS", "10000"))
    s3_bucket = params.get("S3_HTML_BUCKET", "")

    if not db_host or not db_name:
        logger.error("Faltan parametros obligatorios: DB_HOST y DB_NAME.")
        sys.exit(1)

    # S3 es opcional en local, pero obligatorio en Glue
    s3_store: Optional[S3HTMLStore] = None
    if s3_bucket:
        s3_store = S3HTMLStore(bucket=s3_bucket)
        logger.info("HTML se guardara en s3://%s/html/", s3_bucket)
    else:
        logger.warning("S3_HTML_BUCKET no definido -- el HTML no se persistira en S3")

    logger.info(
        "Inicio ETL -- host=%s db=%s max_users=%d batch_size=%d",
        db_host, db_name, max_users, batch_size,
    )

    db = Database(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
        chunk_size=batch_size,
    )

    fetcher = PlaywrightFetcher(
        s3_store=s3_store,
        rate_limit=1.0,
        timeout=30,
        headless=True,
    )

    try:
        with MoltbookScraper(db=db, fetcher=fetcher, batch_size=batch_size) as scraper:
            result = scraper.scrape_all(
                max_users=max_users,
                max_submolts=10,
                max_posts=20,
                max_comments=5,
            )
        logger.info("ETL terminado: %s", result)
    except Exception as exc:
        logger.exception("Fallo critico en el ETL: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()