"""Tests for api_keys.py: local, DPAPI-encrypted storage for cloud
analysis-engine API keys entered via the Settings tab. Uses real Windows
DPAPI (via win32crypt) rather than mocking it, since "never plaintext on
disk" is the property worth actually verifying -- same approach as
test_remembered_login.py for the password case.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from interview_analyzer import api_keys


def _isolated_store(tmp_path):
    return patch.object(api_keys, "_STORE_PATH", tmp_path / ".api_keys.json")


def test_no_key_by_default(tmp_path):
    with _isolated_store(tmp_path):
        assert api_keys.load_key("anthropic_api") is None
        assert api_keys.has_key("anthropic_api") is False


def test_save_and_load_round_trips(tmp_path):
    with _isolated_store(tmp_path):
        assert api_keys.save_key("anthropic_api", "sk-ant-abc123") is True
        assert api_keys.load_key("anthropic_api") == "sk-ant-abc123"
        assert api_keys.has_key("anthropic_api") is True


def test_key_is_never_stored_in_plaintext_on_disk(tmp_path):
    store_path = tmp_path / ".api_keys.json"
    with _isolated_store(tmp_path):
        api_keys.save_key("anthropic_api", "sk-ant-abc123")

    raw = store_path.read_text(encoding="utf-8")
    assert "sk-ant-abc123" not in raw
    data = json.loads(raw)
    assert data["anthropic_api"] != "sk-ant-abc123"


def test_different_providers_are_independent(tmp_path):
    with _isolated_store(tmp_path):
        api_keys.save_key("anthropic_api", "sk-ant-abc123")
        api_keys.save_key("openai_api", "sk-openai-xyz789")

        assert api_keys.load_key("anthropic_api") == "sk-ant-abc123"
        assert api_keys.load_key("openai_api") == "sk-openai-xyz789"


def test_saving_overwrites_only_its_own_provider(tmp_path):
    with _isolated_store(tmp_path):
        api_keys.save_key("anthropic_api", "sk-ant-abc123")
        api_keys.save_key("openai_api", "sk-openai-xyz789")
        api_keys.save_key("anthropic_api", "sk-ant-new-key")

        assert api_keys.load_key("anthropic_api") == "sk-ant-new-key"
        assert api_keys.load_key("openai_api") == "sk-openai-xyz789"


def test_clear_key_removes_only_that_provider(tmp_path):
    with _isolated_store(tmp_path):
        api_keys.save_key("anthropic_api", "sk-ant-abc123")
        api_keys.save_key("openai_api", "sk-openai-xyz789")

        api_keys.clear_key("anthropic_api")

        assert api_keys.load_key("anthropic_api") is None
        assert api_keys.load_key("openai_api") == "sk-openai-xyz789"


def test_clear_key_when_nothing_saved_is_a_no_op(tmp_path):
    with _isolated_store(tmp_path):
        api_keys.clear_key("anthropic_api")  # must not raise


def test_load_handles_corrupted_store_gracefully(tmp_path):
    store_path = tmp_path / ".api_keys.json"
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text("not valid json{{{", encoding="utf-8")
    with _isolated_store(tmp_path):
        assert api_keys.load_key("anthropic_api") is None


class TestMasked:
    def test_masks_a_long_key(self):
        result = api_keys.masked("sk-ant-api03-abcdefghijklmnopqrstuvwxyz")
        assert result.startswith("sk-ant")
        assert result.endswith("wxyz")
        assert "abcdefghijklmnopqrstuv" not in result

    def test_masks_a_short_key_entirely(self):
        assert api_keys.masked("short") == "*****"


class _FakeKeyring:
    """In-memory stand-in for the `keyring` package's module-level API,
    used to exercise the macOS Keychain-backed code path from this
    (Windows) dev/CI machine -- real behavior is also covered for real by
    the macos-latest CI job."""

    def __init__(self):
        self._store: dict[tuple[str, str], str] = {}

    def set_password(self, service, username, password):
        self._store[(service, username)] = password

    def get_password(self, service, username):
        return self._store.get((service, username))

    def delete_password(self, service, username):
        if (service, username) not in self._store:
            raise KeyError("not found")  # stand-in for keyring.errors.PasswordDeleteError
        del self._store[(service, username)]


def _macos_with_fake_keyring(fake=None):
    fake = fake if fake is not None else _FakeKeyring()
    return (
        patch.object(api_keys, "sys", MagicMock(platform="darwin")),
        patch.object(api_keys, "keyring", fake),
    )


class TestMacOsUsesKeychainNotDpapi:
    def test_save_and_load_round_trip_via_keyring(self):
        p1, p2 = _macos_with_fake_keyring()
        with p1, p2:
            assert api_keys.save_key("anthropic_api", "sk-ant-abc123") is True
            assert api_keys.load_key("anthropic_api") == "sk-ant-abc123"
            assert api_keys.has_key("anthropic_api") is True

    def test_no_key_by_default(self):
        p1, p2 = _macos_with_fake_keyring()
        with p1, p2:
            assert api_keys.load_key("anthropic_api") is None
            assert api_keys.has_key("anthropic_api") is False

    def test_clear_key_removes_it(self):
        p1, p2 = _macos_with_fake_keyring()
        with p1, p2:
            api_keys.save_key("anthropic_api", "sk-ant-abc123")
            api_keys.clear_key("anthropic_api")
            assert api_keys.load_key("anthropic_api") is None

    def test_clear_key_when_nothing_saved_is_a_no_op(self):
        p1, p2 = _macos_with_fake_keyring()
        with p1, p2:
            api_keys.clear_key("anthropic_api")  # must not raise

    def test_providers_are_independent(self):
        p1, p2 = _macos_with_fake_keyring()
        with p1, p2:
            api_keys.save_key("anthropic_api", "sk-ant-abc123")
            api_keys.save_key("openai_api", "sk-openai-xyz789")
            assert api_keys.load_key("anthropic_api") == "sk-ant-abc123"
            assert api_keys.load_key("openai_api") == "sk-openai-xyz789"

    def test_returns_false_when_keyring_package_unavailable(self):
        with patch.object(api_keys, "sys", MagicMock(platform="darwin")), \
             patch.object(api_keys, "keyring", None):
            assert api_keys.save_key("anthropic_api", "sk-ant-abc123") is False
            assert api_keys.load_key("anthropic_api") is None
            api_keys.clear_key("anthropic_api")  # must not raise
