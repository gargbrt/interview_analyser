"""Local storage for cloud analysis-engine API keys (anthropic_api,
openai_api), entered via the Settings tab instead of only via an
environment variable.

Storage is platform-specific, but the public functions below
(save_key/load_key/clear_key/has_key) behave identically either way:

  - **Windows**: encrypted at rest with DPAPI (`CryptProtectData`), tied to
    the current Windows user account, in a local JSON file -- never written
    in plaintext.
  - **macOS**: stored in the macOS Keychain via the `keyring` package --
    the OS's own credential store, not a file this app manages at all.

An environment variable (the original, still-supported mechanism -- see
docs/using_cloud_apis.md) always takes precedence over a locally-stored
key, so power users/CI setups that already set one are unaffected.

Important: a claude.ai (or ChatGPT) *subscription* login does not grant API
access -- API keys are a separate, separately-billed credential from
console.anthropic.com / platform.openai.com. This module only ever stores
a real API key you paste in; it has no concept of "logging in" with a
consumer subscription, because no such integration exists for either
provider.
"""
from __future__ import annotations

import base64
import json
import logging
import pathlib
import sys
from typing import Optional

from .config_loader import PROJECT_ROOT

logger = logging.getLogger(__name__)

try:
    import win32crypt  # type: ignore
except ImportError:  # pragma: no cover
    win32crypt = None

try:
    import keyring  # macOS Keychain backend (cross-platform package, only used on macOS here)
except ImportError:  # pragma: no cover
    keyring = None

_DESCRIPTION = "Interview Analyzer cloud API key"
_STORE_PATH = PROJECT_ROOT / "data" / ".api_keys.json"
_KEYRING_SERVICE = "InterviewAnalyzer"


def _encrypt(value: str) -> Optional[str]:
    if win32crypt is None:
        return None
    try:
        blob = win32crypt.CryptProtectData(value.encode("utf-8"), _DESCRIPTION, None, None, None, 0)
        return base64.b64encode(blob).decode("ascii")
    except Exception:  # noqa: BLE001
        logger.warning("Couldn't encrypt API key; it won't be saved.", exc_info=True)
        return None


def _decrypt(encoded: str) -> Optional[str]:
    if win32crypt is None:
        return None
    try:
        blob = base64.b64decode(encoded)
        _desc, decrypted = win32crypt.CryptUnprotectData(blob, None, None, None, 0)
        return decrypted.decode("utf-8")
    except Exception:  # noqa: BLE001
        logger.warning("Couldn't decrypt a stored API key; ignoring it.", exc_info=True)
        return None


def _load_all() -> dict:
    if not _STORE_PATH.exists():
        return {}
    try:
        return json.loads(_STORE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Couldn't read stored API keys.", exc_info=True)
        return {}


def save_key(provider: str, key: str) -> bool:
    """Encrypts and saves `key` under `provider` (e.g. "anthropic_api").
    Returns False (and saves nothing) if the platform's secure-storage
    mechanism isn't available -- callers should tell the user to use the
    environment-variable route instead in that case rather than silently
    doing nothing."""
    if sys.platform == "darwin":
        if keyring is None:
            return False
        try:
            keyring.set_password(_KEYRING_SERVICE, provider, key)
            return True
        except Exception:  # noqa: BLE001
            logger.warning("Couldn't save API key for %s to the macOS Keychain.", provider, exc_info=True)
            return False

    encrypted = _encrypt(key)
    if encrypted is None:
        return False
    data = _load_all()
    data[provider] = encrypted
    try:
        _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STORE_PATH.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        logger.warning("Couldn't save API key for %s.", provider, exc_info=True)
        return False
    return True


def load_key(provider: str) -> Optional[str]:
    if sys.platform == "darwin":
        if keyring is None:
            return None
        try:
            return keyring.get_password(_KEYRING_SERVICE, provider)
        except Exception:  # noqa: BLE001
            logger.warning("Couldn't read API key for %s from the macOS Keychain.", provider, exc_info=True)
            return None

    encrypted = _load_all().get(provider)
    if not encrypted:
        return None
    return _decrypt(encrypted)


def clear_key(provider: str) -> None:
    if sys.platform == "darwin":
        if keyring is None:
            return
        try:
            keyring.delete_password(_KEYRING_SERVICE, provider)
        except Exception:  # noqa: BLE001
            # covers both "nothing was saved" (keyring.errors.PasswordDeleteError)
            # and any other backend error -- either way, clearing an unset key
            # should be a no-op, same as the Windows path below
            pass
        return

    data = _load_all()
    if provider in data:
        del data[provider]
        try:
            _STORE_PATH.write_text(json.dumps(data), encoding="utf-8")
        except OSError:
            logger.warning("Couldn't clear API key for %s.", provider, exc_info=True)


def has_key(provider: str) -> bool:
    if sys.platform == "darwin":
        return load_key(provider) is not None
    return provider in _load_all()


def masked(key: str) -> str:
    """A safe-to-display form, e.g. 'sk-ant-...wxyz' -- never logs or shows
    the full key."""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:6]}...{key[-4:]}"
