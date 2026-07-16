""""Remember me" storage for the login dialog: pre-fills the profile name
(and, if opted in, the password) the next time the login dialog opens after
logging out or restarting the app.

The username alone isn't sensitive, so it's stored as plain JSON. A
remembered password is different -- it's encrypted at rest with Windows
DPAPI (`CryptProtectData`), which ties the ciphertext to the current
Windows user account: the file is useless if copied to another machine or
opened under a different Windows login, and nothing is ever written in
plaintext. If pywin32's `win32crypt` isn't available for some reason, the
password simply isn't remembered (falls back to username-only) rather than
falling back to plaintext storage.
"""
from __future__ import annotations

import base64
import json
import logging
import pathlib
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import win32crypt  # type: ignore
except ImportError:  # pragma: no cover
    win32crypt = None

_DESCRIPTION = "Interview Analyzer remembered login"


@dataclass
class RememberedLogin:
    username: str
    password: Optional[str]


def _remember_path(cfg) -> pathlib.Path:
    db_path = cfg.resolve(cfg.storage.get("db_path", "data/interviews.db"))
    return db_path.with_name(".remembered_login.json")


def _encrypt(password: str) -> Optional[str]:
    if win32crypt is None:
        return None
    try:
        blob = win32crypt.CryptProtectData(password.encode("utf-8"), _DESCRIPTION, None, None, None, 0)
        return base64.b64encode(blob).decode("ascii")
    except Exception:  # noqa: BLE001
        logger.warning("Couldn't encrypt remembered password; it won't be remembered.", exc_info=True)
        return None


def _decrypt(encoded: str) -> Optional[str]:
    if win32crypt is None:
        return None
    try:
        blob = base64.b64decode(encoded)
        _desc, decrypted = win32crypt.CryptUnprotectData(blob, None, None, None, 0)
        return decrypted.decode("utf-8")
    except Exception:  # noqa: BLE001
        # e.g. the file was copied from a different machine/Windows account,
        # or DPAPI's key changed -- fail closed (nothing remembered) rather
        # than raise and block login entirely
        logger.warning("Couldn't decrypt remembered password; ignoring it.", exc_info=True)
        return None


def remember(cfg, username: str, password: Optional[str]) -> None:
    """Saves `username` (plaintext) and `password` (DPAPI-encrypted, or
    simply omitted if that's unavailable) for next time. Overwrites
    whatever was previously remembered."""
    path = _remember_path(cfg)
    data: dict = {"username": username}
    if password:
        encrypted = _encrypt(password)
        if encrypted is not None:
            data["password_enc"] = encrypted
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        logger.warning("Couldn't save remembered login.", exc_info=True)


def forget(cfg) -> None:
    """Clears any remembered login -- called when "Remember me" is left
    unchecked on submit, so unchecking it actually stops remembering rather
    than just not updating a stale saved password."""
    try:
        _remember_path(cfg).unlink(missing_ok=True)
    except OSError:
        logger.warning("Couldn't clear remembered login.", exc_info=True)


def load(cfg) -> Optional[RememberedLogin]:
    path = _remember_path(cfg)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Couldn't read remembered login.", exc_info=True)
        return None
    username = data.get("username")
    if not username:
        return None
    password = None
    if data.get("password_enc"):
        password = _decrypt(data["password_enc"])
    return RememberedLogin(username=username, password=password)
