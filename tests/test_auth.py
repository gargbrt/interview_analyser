import sqlite3

from interview_analyzer import auth


def test_create_and_login_user_no_password():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    auth.ensure_users_table(conn)

    user = auth.create_user(conn, "alex")
    assert user.username == "alex"
    assert user.id is not None

    fetched = auth.get_user_by_username(conn, "alex")
    assert fetched.id == user.id


def test_login_or_create_creates_when_missing():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    user = auth.login_or_create(conn, non_interactive_username="newperson")
    assert user.username == "newperson"

    # calling again with the same name logs in to the same profile, doesn't duplicate
    user2 = auth.login_or_create(conn, non_interactive_username="newperson")
    assert user2.id == user.id


def test_password_hash_roundtrip():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    auth.ensure_users_table(conn)
    auth.create_user(conn, "secure_user", password="hunter2")

    assert auth.verify_password(conn, "secure_user", "hunter2") is True
    assert auth.verify_password(conn, "secure_user", "wrong") is False
