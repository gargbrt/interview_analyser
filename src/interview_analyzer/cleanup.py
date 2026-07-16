"""Deletes raw audio files once they pass their configured retention window.

Transcript + analysis rows in the DB are never deleted by this module —
only the (large) audio file on disk is removed, keeping storage cost near
zero while preserving everything needed for trend tracking.
"""
from __future__ import annotations

import logging
import pathlib

from .db import InterviewDB

logger = logging.getLogger(__name__)


def run_cleanup(db: InterviewDB) -> int:
    """Delete expired audio files. Returns count of files deleted."""
    expired = db.list_expired_audio()
    deleted = 0
    for record in expired:
        if not record.audio_path:
            continue
        path = pathlib.Path(record.audio_path)
        try:
            if path.exists():
                path.unlink()
                logger.info("Deleted expired audio: %s", path)
            db.mark_audio_deleted(record.id)
            deleted += 1
        except OSError as e:
            logger.warning("Failed to delete %s: %s", path, e)
    return deleted
