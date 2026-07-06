import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aira.db")

# If DATABASE_URL is set (e.g. Render PostgreSQL add-on), use Postgres.
# Otherwise fall back to local SQLite file — nothing else in app.py changes.
DATABASE_URL = os.environ.get("DATABASE_URL")
USE_POSTGRES = bool(DATABASE_URL)

if USE_POSTGRES:
    import psycopg2
    # Render sometimes gives "postgres://" — psycopg2 needs "postgresql://"
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)


class _PGCursor:
    """Wraps a psycopg2 cursor so existing app.py code (written for sqlite3,
    using '?' placeholders and cur.lastrowid) keeps working unchanged."""

    def __init__(self, real_cursor):
        self._cur = real_cursor
        self.lastrowid = None

    def execute(self, query, params=()):
        pg_query = query.replace("?", "%s")
        is_insert = pg_query.strip().upper().startswith("INSERT")
        if is_insert and "RETURNING" not in pg_query.upper():
            pg_query += " RETURNING id"
        try:
            self._cur.execute(pg_query, params)
            if is_insert:
                row = self._cur.fetchone()
                self.lastrowid = row[0] if row else None
        except Exception:
            # some inserts hit tables without an "id" column path issues -> retry plain
            if is_insert:
                self._cur.execute(query.replace("?", "%s"), params)
                self.lastrowid = None
            else:
                raise
        return self

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()

    @property
    def rowcount(self):
        return self._cur.rowcount


class _PGConnection:
    def __init__(self, real_conn):
        self._conn = real_conn

    def cursor(self):
        return _PGCursor(self._conn.cursor())

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()

    def execute(self, query, params=()):
        # used once in sqlite path for PRAGMA; harmless no-op for postgres
        pass


def connect_db():
    if USE_POSTGRES:
        raw = psycopg2.connect(DATABASE_URL)
        return _PGConnection(raw)

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Creates tables if they don't exist + inserts default admin (safe to call every startup)."""
    conn = connect_db()
    cur = conn.cursor()

    if USE_POSTGRES:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS institutions (
                id SERIAL PRIMARY KEY,
                name TEXT,
                email TEXT UNIQUE,
                password TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name TEXT,
                email TEXT UNIQUE,
                password TEXT,
                role TEXT,
                institution_id INTEGER
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id SERIAL PRIMARY KEY,
                name TEXT,
                roll TEXT UNIQUE,
                batch TEXT,
                encoding BYTEA,
                institution_id INTEGER DEFAULT 1
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id SERIAL PRIMARY KEY,
                name TEXT,
                roll TEXT,
                date TEXT,
                time TEXT,
                confidence REAL,
                institution_id INTEGER DEFAULT 1
            )
        """)
        cur.execute("""
            INSERT INTO institutions (id, name, email, password)
            VALUES (1, 'Demo Institute', 'admin@aira.com', '1234')
            ON CONFLICT (id) DO NOTHING
        """)
        cur.execute("""
            INSERT INTO users (name, email, password, role, institution_id)
            VALUES ('Admin', 'admin@aira.com', '1234', 'admin', 1)
            ON CONFLICT (email) DO NOTHING
        """)
    else:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS institutions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT UNIQUE,
                password TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT UNIQUE,
                password TEXT,
                role TEXT,
                institution_id INTEGER
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                roll TEXT UNIQUE,
                batch TEXT,
                encoding BLOB,
                institution_id INTEGER DEFAULT 1
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                roll TEXT,
                date TEXT,
                time TEXT,
                confidence REAL,
                institution_id INTEGER DEFAULT 1
            )
        """)
        cur.execute("INSERT OR IGNORE INTO institutions (id, name, email, password) VALUES (1, 'Demo Institute', 'admin@aira.com', '1234')")
        cur.execute("INSERT OR IGNORE INTO users (name, email, password, role, institution_id) VALUES ('Admin', 'admin@aira.com', '1234', 'admin', 1)")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print("✅ Database initialized (Postgres)" if USE_POSTGRES else f"✅ SQLite database initialized at: {DB_PATH}")
