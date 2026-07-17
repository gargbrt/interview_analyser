"""Tests for remembered_login.py: the "remember me" storage used by the
login dialog.

The Windows-path tests below use *real* DPAPI (via win32crypt) rather than
mocking it, since "not plaintext on disk" is the property worth actually
verifying -- same approach as test_api_keys.py. They're Windows-only
(skipped elsewhere): win32crypt doesn't exist on macOS, and more
importantly, remembered_login.py now dispatches to the *real* macOS
Keychain whenever sys.platform is actually "darwin" -- running these on
real macOS CI without pinning the platform would silently exercise (and
leave entries in) the real Keychain instead of testing DPAPI at all.
TestMacOsUsesKeychainNotDpapi below covers the macOS path instead,
platform-independently, via a faked `keyring`.

Tests that never touch a password at all (username-only, corrupted-file
handling) are platform-agnostic and stay as plain module-level tests.
"""
from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

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


def test_forget_when_nothing_remembered_is_a_no_op(tmp_path):
    forget(_cfg(tmp_path))  # must not raise


def test_load_handles_corrupted_file_gracefully(tmp_path):
    cfg = _cfg(tmp_path)
    path = tmp_path / "data" / ".remembered_login.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not valid json{{{", encoding="utf-8")

    assert load(cfg) is None


@pytest.mark.skipif(sys.platform != "win32", reason="exercises real Windows DPAPI")
class TestWindowsDpapiPasswordStorage:
    def test_remember_and_load_username_and_password_round_trips(self, tmp_path):
        cfg = _cfg(tmp_path)
        remember(cfg, "alice", "hunter2")

        result = load(cfg)
        assert result.username == "alice"
        assert result.password == "hunter2"

    def test_password_is_never_stored_in_plaintext_on_disk(self, tmp_path):
        cfg = _cfg(tmp_path)
        remember(cfg, "alice", "hunter2")

        raw = (tmp_path / "data" / ".remembered_login.json").read_text(encoding="utf-8")
        assert "hunter2" not in raw
        data = json.loads(raw)
        assert data["username"] == "alice"
        assert "password_enc" in data
        assert data["password_enc"] != "hunter2"

    def test_remember_overwrites_previous_value(self, tmp_path):
        cfg = _cfg(tmp_path)
        remember(cfg, "alice", "hunter2")
        remember(cfg, "bob", None)

        result = load(cfg)
        assert result.username == "bob"
        assert result.password is None

    def test_forget_clears_remembered_login(self, tmp_path):
        cfg = _cfg(tmp_path)
        remember(cfg, "alice", "hunter2")
        forget(cfg)

        assert load(cfg) is None


class _FakeKeyring:
    """In-memory stand-in for `keyring`'s module API -- exercises the macOS
    Keychain-backed code path from this (Windows) dev/CI machine; real
    behavior is also covered for real by the macos-latest CI job."""

    def __init__(self):
        self._store: dict[tuple[str, str], str] = {}

    def set_password(self, service, username, password):
        self._store[(service, username)] = password

    def get_password(self, service, username):
        return self._store.get((service, username))

    def delete_password(self, service, username):
        if (service, username) not in self._store:
            raise KeyError("not found")
        del self._store[(service, username)]


def _macos_with_fake_keyring(fake=None):
    fake = fake if fake is not None else _FakeKeyring()
    return (
        patch("interview_analyzer.remembered_login.sys.platform", "darwin"),
        patch("interview_analyzer.remembered_login.keyring", fake),
    )


class TestMacOsUsesKeychainNotDpapi:
    def test_username_is_still_plain_json_password_goes_to_keychain(self, tmp_path):
        cfg = _cfg(tmp_path)
        p1, p2 = _macos_with_fake_keyring()
        with p1, p2:
            remember(cfg, "alice", "hunter2")

            raw = json.loads((tmp_path / "data" / ".remembered_login.json").read_text(encoding="utf-8"))
            assert raw["username"] == "alice"
            assert "password_enc" not in raw  # never touches the Windows DPAPI field

            result = load(cfg)
        assert result.username == "alice"
        assert result.password == "hunter2"

    def test_forget_clears_both_the_file_and_the_keychain_entry(self, tmp_path):
        cfg = _cfg(tmp_path)
        fake = _FakeKeyring()
        p1, p2 = _macos_with_fake_keyring(fake)
        with p1, p2:
            remember(cfg, "alice", "hunter2")
            forget(cfg)
            assert load(cfg) is None
        assert fake.get_password("InterviewAnalyzer-RememberedLogin", "alice") is None

    def test_username_only_remember_does_not_touch_keychain(self, tmp_path):
        cfg = _cfg(tmp_path)
        fake = _FakeKeyring()
        p1, p2 = _macos_with_fake_keyring(fake)
        with p1, p2:
            remember(cfg, "alice", None)
            result = load(cfg)
        assert result.username == "alice"
        assert result.password is None

    def test_falls_back_gracefully_when_keyring_package_unavailable(self, tmp_path):
        cfg = _cfg(tmp_path)
        with patch("interview_analyzer.remembered_login.sys.platform", "darwin"), \
             patch("interview_analyzer.remembered_login.keyring", None):
            remember(cfg, "alice", "hunter2")
            result = load(cfg)
        assert result.username == "alice"
        assert result.password is None  # not remembered, but doesn't crash either
