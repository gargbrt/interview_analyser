"""SQLite storage layer.

Design goal: raw audio is transient (deleted after retention_days), but the
transcript and analysis JSON are tiny and kept forever so trend analysis
across many interviews stays cheap.
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS interviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    source_app TEXT,
    audio_path TEXT,
    audio_expires_at TEXT,
    audio_deleted INTEGER DEFAULT 0,
    transcript TEXT,
    analysis_json TEXT,
    report_path TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- One row per interview: the user's own judgment of whether the
-- transcription/analysis for that interview was accurate. Used to
-- calibrate the confidence score shown on later interviews' reports (see
-- confidence.py) and to feed corrective notes back into future analysis
-- prompts -- see analyzer.py/rubric.py's calibration_notes.
-- Column names are a holdover from when this was a boolean Yes/No rating
-- (kept as-is to avoid an ALTER TABLE migration for anyone with existing
-- rows) -- they now hold a 1-10 quality score, NULL meaning "not rated".
-- See FeedbackRecord's transcript_score/analysis_score for the current,
-- accurately-named Python-level names.
CREATE TABLE IF NOT EXISTS feedback (
    interview_id INTEGER PRIMARY KEY,
    user_id INTEGER,
    transcript_correct INTEGER,  -- 1-10/NULL (NULL = not rated)
    analysis_correct INTEGER,    -- 1-10/NULL
    comment TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (interview_id) REFERENCES interviews(id)
);
"""


@dataclass
class InterviewRecord:
    id: int
    user_id: Optional[int]
    started_at: str
    ended_at: Optional[str]
    source_app: Optional[str]
    audio_path: Optional[str]
    audio_expires_at: Optional[str]
    audio_deleted: bool
    transcript: Optional[str]
    analysis_json: Optional[str]
    report_path: Optional[str]

    @property
    def analysis(self) -> Optional[dict[str, Any]]:
        return json.loads(self.analysis_json) if self.analysis_json else None


@dataclass
class FeedbackRecord:
    interview_id: int
    user_id: Optional[int]
    transcript_score: Optional[int]  # 1-10, None = not rated
    analysis_score: Optional[int]    # 1-10, None = not rated
    comment: Optional[str]
    created_at: str


class InterviewDB:
    """A watcher, its tray icon, and its dashboard all share one InterviewDB
    from different threads (the watcher's background loop, the dashboard's
    own Tk thread). sqlite3 connections aren't safe to use across threads
    without `check_same_thread=False` plus our own serialization, since
    Python's sqlite3 module doesn't guarantee thread-safe concurrent access
    to a single connection on its own."""

    def __init__(self, db_path: pathlib.Path | str):
        from .auth import ensure_users_table

        self.db_path = pathlib.Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        ensure_users_table(self._conn)  # users table must exist before interviews FK is used
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def start_interview(
        self, source_app: str, audio_path: str, retention_days: int, user_id: Optional[int] = None
    ) -> int:
        started_at = dt.datetime.now().isoformat()
        expires_at = (dt.datetime.now() + dt.timedelta(days=retention_days)).isoformat()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO interviews (user_id, started_at, source_app, audio_path, audio_expires_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_id, started_at, source_app, audio_path, expires_at),
            )
            self._conn.commit()
            return cur.lastrowid

    def end_interview(self, interview_id: int, ended_at: Optional[str] = None) -> None:
        """Sets ended_at to `ended_at` if given, else now. The explicit
        form is used to back-fill an interview whose recording never
        cleanly finished (see reprocess_interview in watcher.py) with a
        real timestamp computed from the audio's own duration, rather than
        the moment the (much later) reprocess happened to run."""
        with self._lock:
            self._conn.execute(
                "UPDATE interviews SET ended_at = ? WHERE id = ?",
                (ended_at or dt.datetime.now().isoformat(), interview_id),
            )
            self._conn.commit()

    def save_transcript(self, interview_id: int, transcript: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE interviews SET transcript = ? WHERE id = ?", (transcript, interview_id)
            )
            self._conn.commit()

    def save_analysis(self, interview_id: int, analysis: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE interviews SET analysis_json = ? WHERE id = ?",
                (json.dumps(analysis), interview_id),
            )
            self._conn.commit()

    def save_report_path(self, interview_id: int, report_path: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE interviews SET report_path = ? WHERE id = ?", (report_path, interview_id)
            )
            self._conn.commit()

    def update_audio_path(self, interview_id: int, audio_path: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE interviews SET audio_path = ? WHERE id = ?", (audio_path, interview_id)
            )
            self._conn.commit()

    def mark_audio_deleted(self, interview_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE interviews SET audio_deleted = 1 WHERE id = ?", (interview_id,)
            )
            self._conn.commit()

    def delete_interview(self, interview_id: int) -> None:
        """Removes the DB row entirely -- used by the dashboard's Delete
        action for interviews the user wants to clear out (e.g. ones with
        no audio and no report, from a crash or an accidental recording).
        Does not touch any files on disk; callers decide whether to also
        remove the audio/report files first."""
        with self._lock:
            self._conn.execute("DELETE FROM feedback WHERE interview_id = ?", (interview_id,))
            self._conn.execute("DELETE FROM interviews WHERE id = ?", (interview_id,))
            self._conn.commit()

    def get(self, interview_id: int) -> Optional[InterviewRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM interviews WHERE id = ?", (interview_id,)
            ).fetchone()
            return self._row_to_record(row) if row else None

    def list_all(self, user_id: Optional[int] = None) -> list[InterviewRecord]:
        with self._lock:
            if user_id is not None:
                rows = self._conn.execute(
                    "SELECT * FROM interviews WHERE user_id = ? ORDER BY started_at ASC", (user_id,)
                ).fetchall()
            else:
                rows = self._conn.execute("SELECT * FROM interviews ORDER BY started_at ASC").fetchall()
            return [self._row_to_record(r) for r in rows]

    def list_expired_audio(self) -> list[InterviewRecord]:
        now = dt.datetime.now().isoformat()
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM interviews WHERE audio_deleted = 0 AND audio_expires_at IS NOT NULL "
                "AND audio_expires_at <= ? AND audio_path IS NOT NULL",
                (now,),
            ).fetchall()
            return [self._row_to_record(r) for r in rows]

    def save_feedback(
        self,
        interview_id: int,
        user_id: Optional[int],
        transcript_score: Optional[int],
        analysis_score: Optional[int],
        comment: str = "",
    ) -> None:
        """Upserts the single feedback row for this interview -- resubmitting
        (e.g. changing your mind, adding a comment later, or clearing a
        rating back to "not rated" by passing None) replaces the previous
        rating rather than accumulating duplicates. Scores are 1-10 or None
        (not rated) -- validated here since this is the one place every
        write path (UI, tests) funnels through."""
        for score in (transcript_score, analysis_score):
            if score is not None and not (1 <= score <= 10):
                raise ValueError(f"Feedback scores must be 1-10 or None, got {score!r}.")
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO feedback (interview_id, user_id, transcript_correct, analysis_correct, comment, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(interview_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    transcript_correct = excluded.transcript_correct,
                    analysis_correct = excluded.analysis_correct,
                    comment = excluded.comment,
                    created_at = excluded.created_at
                """,
                (
                    interview_id,
                    user_id,
                    transcript_score,
                    analysis_score,
                    comment,
                    dt.datetime.now().isoformat(),
                ),
            )
            self._conn.commit()

    def delete_feedback(self, interview_id: int) -> None:
        """Removes feedback for one interview entirely -- for clearing out
        a feedback entry given by mistake, distinct from save_feedback(...,
        None, None, "") which still leaves a (now-unrated) row behind."""
        with self._lock:
            self._conn.execute("DELETE FROM feedback WHERE interview_id = ?", (interview_id,))
            self._conn.commit()

    def get_feedback(self, interview_id: int) -> Optional[FeedbackRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM feedback WHERE interview_id = ?", (interview_id,)
            ).fetchone()
            return self._row_to_feedback(row) if row else None

    def list_feedback(self, user_id: Optional[int] = None) -> list[FeedbackRecord]:
        with self._lock:
            if user_id is not None:
                rows = self._conn.execute(
                    "SELECT * FROM feedback WHERE user_id = ? ORDER BY created_at ASC", (user_id,)
                ).fetchall()
            else:
                rows = self._conn.execute("SELECT * FROM feedback ORDER BY created_at ASC").fetchall()
            return [self._row_to_feedback(r) for r in rows]

    @staticmethod
    def _row_to_feedback(row: sqlite3.Row) -> FeedbackRecord:
        return FeedbackRecord(
            interview_id=row["interview_id"],
            user_id=row["user_id"],
            transcript_score=row["transcript_correct"],
            analysis_score=row["analysis_correct"],
            comment=row["comment"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> InterviewRecord:
        return InterviewRecord(
            id=row["id"],
            user_id=row["user_id"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            source_app=row["source_app"],
            audio_path=row["audio_path"],
            audio_expires_at=row["audio_expires_at"],
            audio_deleted=bool(row["audio_deleted"]),
            transcript=row["transcript"],
            analysis_json=row["analysis_json"],
            report_path=row["report_path"],
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
