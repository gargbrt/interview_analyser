"""Local profile ('login') system.

This is a single-machine, offline tool — there's no server, so "login" means
a local profile that scopes your interview history, reports, and trend
tracking to you specifically (useful if more than one person uses the same
Windows account, or you just want a clean named profile). No password is
transmitted anywhere; if set, it's stored only as a local salted hash to
gate the CLI/tray on this machine.
"""
from __future__ import annotations

import getpass
import hashlib
import os
import sqlite3
from dataclasses import dataclass
from typing import Optional

USERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT,
    password_salt TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


@dataclass
class User:
    id: int
    username: str


def ensure_users_table(conn: sqlite3.Connection) -> None:
    conn.execute(USERS_SCHEMA)
    conn.commit()


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000).hex()


def create_user(conn: sqlite3.Connection, username: str, password: Optional[str] = None) -> User:
    salt = os.urandom(16)
    salt_hex = salt.hex()
    pw_hash = _hash_password(password, salt) if password else None
    cur = conn.execute(
        "INSERT INTO users (username, password_hash, password_salt) VALUES (?, ?, ?)",
        (username, pw_hash, salt_hex),
    )
    conn.commit()
    return User(id=cur.lastrowid, username=username)


def get_user_by_username(conn: sqlite3.Connection, username: str) -> Optional[User]:
    row = conn.execute(
        "SELECT id, username FROM users WHERE username = ?", (username,)
    ).fetchone()
    return User(id=row["id"], username=row["username"]) if row else None


def verify_password(conn: sqlite3.Connection, username: str, password: str) -> bool:
    row = conn.execute(
        "SELECT password_hash, password_salt FROM users WHERE username = ?", (username,)
    ).fetchone()
    if row is None or row["password_hash"] is None:
        return False
    salt = bytes.fromhex(row["password_salt"])
    return _hash_password(password, salt) == row["password_hash"]


def login_or_create(conn: sqlite3.Connection, non_interactive_username: Optional[str] = None) -> User:
    """CLI login flow: prompts for a username on first run of a session,
    creates the profile if it doesn't exist yet, otherwise logs in
    (optionally password-checked if that profile was created with one).

    Pass `non_interactive_username` to skip prompts (used by tests / the
    tray app once a profile is already selected).
    """
    ensure_users_table(conn)

    username = non_interactive_username or input("Username: ").strip()
    existing = get_user_by_username(conn, username)

    if existing is None:
        print(f"No profile named '{username}' found — creating a new one.")
        set_pw = non_interactive_username is None and (
            input("Set a local password for this profile? [y/N]: ").strip().lower() == "y"
        )
        password = None
        if set_pw:
            password = getpass.getpass("Password: ")
        return create_user(conn, username, password)

    # existing profile — check password only if one was set
    row = conn.execute(
        "SELECT password_hash FROM users WHERE username = ?", (username,)
    ).fetchone()
    if row["password_hash"] and non_interactive_username is None:
        password = getpass.getpass("Password: ")
        if not verify_password(conn, username, password):
            raise PermissionError("Incorrect password.")
    return existing
