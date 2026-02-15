-- Moltbook Karma Database Schema - PostgreSQL (RDS)
-- Compatible with db.t4g.micro; use for ensure_tables / init

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

-- Foreign Keys: añadidos al final para integridad referencia
-- La carga ELT debe respetar el orden: users → sub_molt → posts → comments.
ALTER TABLE posts
    ADD CONSTRAINT fk_posts_user FOREIGN KEY (id_user) REFERENCES users(id_user);
ALTER TABLE posts
    ADD CONSTRAINT fk_posts_submolt FOREIGN KEY (id_submolt) REFERENCES sub_molt(id_submolt);
ALTER TABLE comments
    ADD CONSTRAINT fk_comments_user FOREIGN KEY (id_user) REFERENCES users(id_user);
ALTER TABLE comments
    ADD CONSTRAINT fk_comments_post FOREIGN KEY (id_post) REFERENCES posts(id_post);
