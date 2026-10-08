"""User accounts and sessions, stored in a local SQLite file.

What is stored: username, a salted scrypt hash of the password, creation time, and
active login sessions (only a SHA-256 hash of each session token). Uploaded images are
never passed to this module and never stored anywhere.
"""
import hashlib
import hmac
import re
import secrets
import sqlite3
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
MIN_PASSWORD, MAX_PASSWORD = 8, 256
SESSION_DAYS = 7
SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1, "dklen": 32}  # memory-hard: slows down password guessing

# Failed logins: at most MAX_FAILS within WINDOW seconds per username before a lockout
MAX_FAILS, WINDOW = 5, 300


class AuthError(Exception):
    """An error whose message is safe to show to the user."""


def hash_password(password, salt=None):
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, **SCRYPT)
    return salt.hex() + "$" + digest.hex()


def verify_password(password, stored):
    salt_hex, digest_hex = stored.split("$", 1)
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), **SCRYPT)
    return hmac.compare_digest(digest.hex(), digest_hex)  # constant time


def _token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


# A real hash to check against when the username does not exist, so a failed login
# takes the same time whether or not the account exists.
_DUMMY_HASH = hash_password(secrets.token_hex(8))


class UserStore:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._fails = defaultdict(deque)  # username -> timestamps of recent failed logins
        self._fails_lock = threading.Lock()
        with self._connect() as db:
            db.executescript("""
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS users (
                    id            INTEGER PRIMARY KEY,
                    username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    created_at    REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    expires_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id);
            """)

    @contextmanager
    def _connect(self):
        """One short-lived connection per call (safe with the threaded web server).
        Commits on success, rolls back on error, and always closes the connection."""
        db = sqlite3.connect(self.db_path, timeout=10)
        try:
            db.execute("PRAGMA foreign_keys = ON")
            with db:
                yield db
        finally:
            db.close()

    # ---- accounts -------------------------------------------------------------
    def create_user(self, username, password):
        username = (username or "").strip()
        if not USERNAME_RE.match(username):
            raise AuthError("Username must be 3 to 32 characters: letters, numbers, dot, dash or underscore.")
        if not MIN_PASSWORD <= len(password or "") <= MAX_PASSWORD:
            raise AuthError(f"Password must be at least {MIN_PASSWORD} characters.")
        if password.lower() == username.lower():
            raise AuthError("Password can't be the same as your username.")
        try:
            with self._connect() as db:
                cur = db.execute("INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                                 (username, hash_password(password), time.time()))
                return cur.lastrowid
        except sqlite3.IntegrityError:
            raise AuthError("That username is taken. Try another one.") from None

    def authenticate(self, username, password):
        """Return the user id, or raise AuthError. Locks an account briefly after repeated failures."""
        key = (username or "").strip().lower()
        now = time.time()
        with self._fails_lock:
            fails = self._fails[key]
            while fails and now - fails[0] > WINDOW:
                fails.popleft()
            if len(fails) >= MAX_FAILS:
                wait = int(WINDOW - (now - fails[0])) + 1
                raise AuthError(f"Too many failed attempts. Try again in {wait} seconds.")
        with self._connect() as db:
            row = db.execute("SELECT id, password_hash FROM users WHERE username = ?", (key,)).fetchone()
        ok = verify_password(password or "", row[1] if row else _DUMMY_HASH) and row is not None
        if not ok:
            with self._fails_lock:
                self._fails[key].append(now)
            raise AuthError("Wrong username or password.")
        with self._fails_lock:
            self._fails.pop(key, None)
        return row[0]

    def delete_user(self, user_id):
        with self._connect() as db:
            db.execute("DELETE FROM users WHERE id = ?", (user_id,))  # sessions cascade

    # ---- sessions -------------------------------------------------------------
    def create_session(self, user_id):
        token = secrets.token_urlsafe(32)
        with self._connect() as db:
            db.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))
            db.execute("INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
                       (_token_hash(token), user_id, time.time() + SESSION_DAYS * 86400))
        return token

    def session_user(self, token):
        """Return (user_id, username) for a valid session token, else None."""
        if not token:
            return None
        with self._connect() as db:
            return db.execute(
                "SELECT u.id, u.username FROM sessions s JOIN users u ON u.id = s.user_id "
                "WHERE s.token_hash = ? AND s.expires_at > ?", (_token_hash(token), time.time())).fetchone()

    def end_session(self, token):
        if token:
            with self._connect() as db:
                db.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))
