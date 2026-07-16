"""Tests for remembered_login.py: the "remember me" storage used by the
login dialog. The password path uses real Windows DPAPI (via win32crypt)
when available -- these tests exercise the real round-trip on this machine
rather than mocking it, since DPAPI is what makes "not plaintext on disk"
actually true, and that's the property worth verifying."""
from __future__ import annotations

import json

from interview_analyzer.config_loader import Config
from interview_analyzer.remembered_login import forget, load, remember


def _cfg(tmp_path) -> Config:
    return Config(raw={"storage": {"db_path": str(tmp_path / "data" / "interviews.db")}})


def test_nothing_remembered_by_default(tmp_path):
    assert load(_cfg(tmp_path)) is None


def test_remember_and_load_username_only(tmp_path):
    cfg = _cfg(tmp_path)
    remember(cfg, "alice", None)

    result = load(cfg)
    assert result.username == "alice"
    assert result.password is None


def test_remember_and_load_username_and_password_round_trips(tmp_path):
    cfg = _cfg(tmp_path)
    remember(cfg, "alice", "hunter2")

    result = load(cfg)
    assert result.username == "alice"
    assert result.password == "hunter2"


def test_password_is_never_stored_in_plaintext_on_disk(tmp_path):
    cfg = _cfg(tmp_path)
    remember(cfg, "alice", "hunter2")

    raw = (tmp_path / "data" / ".remembered_login.json").read_text(encoding="utf-8")
    assert "hunter2" not in raw
    data = json.loads(raw)
    assert data["username"] == "alice"
    assert "password_enc" in data
    assert data["password_enc"] != "hunter2"


def test_remember_overwrites_previous_value(tmp_path):
    cfg = _cfg(tmp_path)
    remember(cfg, "alice", "hunter2")
    remember(cfg, "bob", None)

    result = load(cfg)
    assert result.username == "bob"
    assert result.password is None


def test_forget_clears_remembered_login(tmp_path):
    cfg = _cfg(tmp_path)
    remember(cfg, "alice", "hunter2")
    forget(cfg)

    assert load(cfg) is None


def test_forget_when_nothing_remembered_is_a_no_op(tmp_path):
    forget(_cfg(tmp_path))  # must not raise


def test_load_handles_corrupted_file_gracefully(tmp_path):
    cfg = _cfg(tmp_path)
    path = tmp_path / "data" / ".remembered_login.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not valid json{{{", encoding="utf-8")

    assert load(cfg) is None
