-- Moltbook Karma Database Schema
-- SQLite DDL with FK enforcement and indexes

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    id_user TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    karma INTEGER DEFAULT 0,
    description TEXT,
    human_owner TEXT,
    joined TEXT,
    followers INTEGER DEFAULT 0,
    following INTEGER DEFAULT 0,
    scraped_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sub_molt (
    id_submolt TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    scraped_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS posts (
    id_post TEXT PRIMARY KEY,
    id_user TEXT NOT NULL,
    id_submolt TEXT,
    title TEXT,
    description TEXT,
    rating INTEGER DEFAULT 0,
    date TEXT,
    scraped_at TEXT NOT NULL,
    FOREIGN KEY (id_user) REFERENCES users(id_user),
    FOREIGN KEY (id_submolt) REFERENCES sub_molt(id_submolt)
);

CREATE TABLE IF NOT EXISTS comments (
    id_comment TEXT PRIMARY KEY,
    id_user TEXT NOT NULL,
    id_post TEXT NOT NULL,
    description TEXT,
    date TEXT,
    rating INTEGER DEFAULT 0,
    scraped_at TEXT NOT NULL,
    FOREIGN KEY (id_user) REFERENCES users(id_user),
    FOREIGN KEY (id_post) REFERENCES posts(id_post)
);

CREATE TABLE IF NOT EXISTS user_submolt (
    id_user TEXT NOT NULL,
    id_submolt TEXT NOT NULL,
    PRIMARY KEY (id_user, id_submolt),
    FOREIGN KEY (id_user) REFERENCES users(id_user),
    FOREIGN KEY (id_submolt) REFERENCES sub_molt(id_submolt)
);

-- Indexes for query performance
CREATE INDEX IF NOT EXISTS idx_posts_user ON posts(id_user);
CREATE INDEX IF NOT EXISTS idx_posts_submolt ON posts(id_submolt);
CREATE INDEX IF NOT EXISTS idx_comments_user ON comments(id_user);
CREATE INDEX IF NOT EXISTS idx_comments_post ON comments(id_post);
CREATE INDEX IF NOT EXISTS idx_users_karma ON users(karma);
CREATE INDEX IF NOT EXISTS idx_users_name ON users(name);
