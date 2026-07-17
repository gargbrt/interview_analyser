"""Prevents two copies of the tray+dashboard app from running at once.

Two instances would both poll for meetings, both show consent/control-panel
popups independently, and both write to the same SQLite DB from separate
OS processes -- confusing at best. A simple exclusive file lock (held for
the process's lifetime) is enough to enforce single-instance, on both
Windows (`msvcrt.locking`) and macOS (`fcntl.flock`).
"""
from __future__ import annotations

import logging
import pathlib
import sys

logger = logging.getLogger(__name__)

# kept alive for the process lifetime so the OS lock stays held; released
# automatically when the process exits (normally or otherwise)
_lock_file = None


def acquire_single_instance_lock(lock_path: pathlib.Path) -> bool:
    """Returns True if this process now holds the lock (i.e. it's the only
    running instance), False if another instance already holds it."""
    global _lock_file
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        f = open(lock_path, "w")
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    _lock_file = f
    return True
