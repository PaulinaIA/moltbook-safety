-- ============================================
-- Moltbook Karma - Oracle SQL DDL
-- Para importar en Oracle SQL Developer Data Modeler:
--   File > Import > DDL File
-- ============================================

CREATE TABLE users (
    id_user     VARCHAR2(50)  NOT NULL,
    name        VARCHAR2(255) NOT NULL,
    karma       NUMBER(10)    DEFAULT 0,
    description CLOB,
    human_owner VARCHAR2(255),
    joined      VARCHAR2(100),
    followers   NUMBER(10)    DEFAULT 0,
    following   NUMBER(10)    DEFAULT 0,
    scraped_at  VARCHAR2(100) NOT NULL,
    CONSTRAINT users_pk PRIMARY KEY (id_user)
);

CREATE TABLE sub_molt (
    id_submolt  VARCHAR2(50)  NOT NULL,
    name        VARCHAR2(255) NOT NULL,
    description CLOB,
    scraped_at  VARCHAR2(100) NOT NULL,
    CONSTRAINT sub_molt_pk PRIMARY KEY (id_submolt),
    CONSTRAINT sub_molt_name_uk UNIQUE (name)
);

CREATE TABLE posts (
    id_post     VARCHAR2(50)  NOT NULL,
    id_user     VARCHAR2(50)  NOT NULL,
    id_submolt  VARCHAR2(50),
    title       VARCHAR2(500),
    description CLOB,
    rating      NUMBER(10)    DEFAULT 0,
    date_posted VARCHAR2(100),
    scraped_at  VARCHAR2(100) NOT NULL,
    CONSTRAINT posts_pk PRIMARY KEY (id_post),
    CONSTRAINT posts_users_fk FOREIGN KEY (id_user)
        REFERENCES users (id_user),
    CONSTRAINT posts_submolt_fk FOREIGN KEY (id_submolt)
        REFERENCES sub_molt (id_submolt)
);

CREATE TABLE comments (
    id_comment  VARCHAR2(50)  NOT NULL,
    id_user     VARCHAR2(50)  NOT NULL,
    id_post     VARCHAR2(50)  NOT NULL,
    description CLOB,
    date_posted VARCHAR2(100),
    rating      NUMBER(10)    DEFAULT 0,
    scraped_at  VARCHAR2(100) NOT NULL,
    CONSTRAINT comments_pk PRIMARY KEY (id_comment),
    CONSTRAINT comments_users_fk FOREIGN KEY (id_user)
        REFERENCES users (id_user),
    CONSTRAINT comments_posts_fk FOREIGN KEY (id_post)
        REFERENCES posts (id_post)
);

CREATE TABLE user_submolt (
    id_user    VARCHAR2(50) NOT NULL,
    id_submolt VARCHAR2(50) NOT NULL,
    CONSTRAINT user_submolt_pk PRIMARY KEY (id_user, id_submolt),
    CONSTRAINT user_submolt_users_fk FOREIGN KEY (id_user)
        REFERENCES users (id_user),
    CONSTRAINT user_submolt_submolt_fk FOREIGN KEY (id_submolt)
        REFERENCES sub_molt (id_submolt)
);

-- Indices para rendimiento
CREATE INDEX idx_posts_user ON posts (id_user);
CREATE INDEX idx_posts_submolt ON posts (id_submolt);
CREATE INDEX idx_comments_user ON comments (id_user);
CREATE INDEX idx_comments_post ON comments (id_post);
CREATE INDEX idx_users_karma ON users (karma);
CREATE INDEX idx_users_name ON users (name);
