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

-- Named, reusable assessment profiles (role/seniority/industry/company type
-- + which competencies to score -- see profiles.py) saved under one user's
-- own logged-in profile (auth.py's `users` table), so switching between
-- e.g. "SWE Senior FAANG" and "PM Entry Startup" is a dropdown, not
-- re-entering everything each time. `competencies_json` is a JSON list of
-- competency names (see profiles.AssessmentProfile).
-- `is_active`: at most one row per user_id has this set to 1 -- the
-- profile confirm dialog (profile_confirm.py) prefills from whichever
-- template is active, if any (see get_active_profile_template /
-- set_active_profile_template), else profiles.GENERIC_PROFILE.
CREATE TABLE IF NOT EXISTS assessment_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    name TEXT NOT NULL,
    role TEXT,
    seniority TEXT,
    industry TEXT,
    company_type TEXT,
    competencies_json TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id),
    UNIQUE (user_id, name)
);

-- One row per completed analysis (the first run AND every reprocess) for
-- an interview -- a full audit trail so redoing the assessment under a
-- different profile (see watcher.py's reprocess_interview) never silently
-- loses a previous take. Only the most recent
-- MAX_ANALYSIS_HISTORY_PER_INTERVIEW rows per interview are kept (see
-- append_analysis_history) -- this is a convenience history for the
-- History tab's "Previous assessments" section, not a permanent audit
-- log, so an unbounded table isn't worth the storage cost.
CREATE TABLE IF NOT EXISTS analysis_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interview_id INTEGER NOT NULL,
    analysis_json TEXT NOT NULL,
    profile_snapshot_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (interview_id) REFERENCES interviews(id)
);
"""

# At most this many past assessments are kept per interview (see
# append_analysis_history) -- older ones are pruned automatically.
MAX_ANALYSIS_HISTORY_PER_INTERVIEW = 10

# Columns added after the original schema above was already shipped to real
# users -- CREATE TABLE IF NOT EXISTS alone won't add a column to an
# existing table, so each entry here is (column, "ADD COLUMN" SQL fragment)
# and _migrate_interviews_table applies it only if genuinely missing, once,
# idempotently, at connection-open time.
_INTERVIEWS_MIGRATIONS: list[tuple[str, str]] = [
    (
        "profile_snapshot_json",
        "ALTER TABLE interviews ADD COLUMN profile_snapshot_json TEXT",
    ),
]


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
    profile_snapshot_json: Optional[str] = None

    @property
    def analysis(self) -> Optional[dict[str, Any]]:
        return json.loads(self.analysis_json) if self.analysis_json else None

    @property
    def profile(self):
        """The AssessmentProfile this interview was actually analyzed
        against (see profiles.py), or None if it predates that feature --
        callers fall back to profiles.GENERIC_PROFILE in that case."""
        from .profiles import AssessmentProfile

        if not self.profile_snapshot_json:
            return None
        return AssessmentProfile.from_dict(json.loads(self.profile_snapshot_json))


@dataclass
class FeedbackRecord:
    interview_id: int
    user_id: Optional[int]
    transcript_score: Optional[int]  # 1-10, None = not rated
    analysis_score: Optional[int]    # 1-10, None = not rated
    comment: Optional[str]
    created_at: str


@dataclass
class ProfileTemplateRecord:
    id: int
    user_id: Optional[int]
    name: str
    created_at: str
    updated_at: str
    is_active: bool
    profile: "object"  # profiles.AssessmentProfile -- see _row_to_profile_template


@dataclass
class AnalysisHistoryRecord:
    id: int
    interview_id: int
    analysis_json: str
    profile_snapshot_json: Optional[str]
    created_at: str

    @property
    def analysis(self) -> dict[str, Any]:
        return json.loads(self.analysis_json)

    @property
    def profile(self):
        """The AssessmentProfile this past assessment was run with, or
        None if it predates the assessment-profile feature -- same
        contract as InterviewRecord.profile."""
        from .profiles import AssessmentProfile

        if not self.profile_snapshot_json:
            return None
        return AssessmentProfile.from_dict(json.loads(self.profile_snapshot_json))


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
        self._migrate_interviews_table()

    def _migrate_interviews_table(self) -> None:
        """Applies any ALTER TABLE ADD COLUMN migrations the `interviews`
        table needs, idempotently -- CREATE TABLE IF NOT EXISTS (above)
        only creates the table if it's missing entirely, it doesn't add new
        columns to an already-existing one from an older version of this
        app."""
        existing_columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(interviews)").fetchall()
        }
        for column, alter_sql in _INTERVIEWS_MIGRATIONS:
            if column not in existing_columns:
                self._conn.execute(alter_sql)
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

    def save_profile_snapshot(self, interview_id: int, profile) -> None:
        """Records the exact AssessmentProfile (see profiles.py) used for
        this interview's analysis -- a snapshot, not a live reference to a
        saved template, so renaming/deleting a template later never changes
        what a past report meant (same reasoning as source_app being a
        plain string, not a foreign key)."""
        with self._lock:
            self._conn.execute(
                "UPDATE interviews SET profile_snapshot_json = ? WHERE id = ?",
                (json.dumps(profile.to_dict()), interview_id),
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
            self._conn.execute("DELETE FROM analysis_history WHERE interview_id = ?", (interview_id,))
            self._conn.execute("DELETE FROM interviews WHERE id = ?", (interview_id,))
            self._conn.commit()

    def append_analysis_history(self, interview_id: int, analysis: dict[str, Any], profile=None) -> None:
        """Records a snapshot of a just-completed analysis (and the profile
        it was run with) to this interview's history -- called every time
        an analysis finishes, whether the first run or a later reprocess,
        so trying a different profile never silently discards the
        previous take. Prunes down to the most recent
        MAX_ANALYSIS_HISTORY_PER_INTERVIEW entries for this interview."""
        now = dt.datetime.now().isoformat()
        profile_json = json.dumps(profile.to_dict()) if profile is not None else None
        with self._lock:
            self._conn.execute(
                "INSERT INTO analysis_history (interview_id, analysis_json, profile_snapshot_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (interview_id, json.dumps(analysis), profile_json, now),
            )
            self._conn.execute(
                """
                DELETE FROM analysis_history WHERE interview_id = ? AND id NOT IN (
                    SELECT id FROM analysis_history WHERE interview_id = ?
                    ORDER BY created_at DESC, id DESC LIMIT ?
                )
                """,
                (interview_id, interview_id, MAX_ANALYSIS_HISTORY_PER_INTERVIEW),
            )
            self._conn.commit()

    def list_analysis_history(self, interview_id: int) -> list["AnalysisHistoryRecord"]:
        """Most recent first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM analysis_history WHERE interview_id = ? ORDER BY created_at DESC, id DESC",
                (interview_id,),
            ).fetchall()
            return [self._row_to_analysis_history(r) for r in rows]

    @staticmethod
    def _row_to_analysis_history(row: sqlite3.Row) -> "AnalysisHistoryRecord":
        return AnalysisHistoryRecord(
            id=row["id"], interview_id=row["interview_id"], analysis_json=row["analysis_json"],
            profile_snapshot_json=row["profile_snapshot_json"], created_at=row["created_at"],
        )

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
            profile_snapshot_json=row["profile_snapshot_json"],
        )

    # -- assessment profile templates (see profiles.py) --------------------

    def create_profile_template(self, user_id: Optional[int], name: str, profile) -> int:
        """Saves `profile` (an AssessmentProfile) as a named template under
        `user_id`. Re-saving an existing name for the same user overwrites
        it (upsert) rather than erroring or duplicating -- the natural
        expectation for a "Save as template" action reusing a name."""
        now = dt.datetime.now().isoformat()
        competencies_json = json.dumps(profile.competencies)
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO assessment_profiles
                    (user_id, name, role, seniority, industry, company_type, competencies_json,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, name) DO UPDATE SET
                    role = excluded.role,
                    seniority = excluded.seniority,
                    industry = excluded.industry,
                    company_type = excluded.company_type,
                    competencies_json = excluded.competencies_json,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id, name, profile.role, profile.seniority, profile.industry,
                    profile.company_type, competencies_json, now, now,
                ),
            )
            self._conn.commit()
            if cur.lastrowid:
                return cur.lastrowid
            row = self._conn.execute(
                "SELECT id FROM assessment_profiles WHERE user_id IS ? AND name = ?", (user_id, name)
            ).fetchone()
            return row["id"]

    def list_profile_templates(self, user_id: Optional[int] = None) -> list["ProfileTemplateRecord"]:
        with self._lock:
            if user_id is not None:
                rows = self._conn.execute(
                    "SELECT * FROM assessment_profiles WHERE user_id = ? ORDER BY name ASC", (user_id,)
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM assessment_profiles ORDER BY name ASC"
                ).fetchall()
            return [self._row_to_profile_template(r) for r in rows]

    def get_profile_template(self, template_id: int, user_id: Optional[int] = None) -> Optional["ProfileTemplateRecord"]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM assessment_profiles WHERE id = ?", (template_id,)
            ).fetchone()
            if row is None:
                return None
            if user_id is not None and row["user_id"] != user_id:
                return None  # a template's id from a different user's profile isn't visible
            return self._row_to_profile_template(row)

    def delete_profile_template(self, template_id: int, user_id: Optional[int] = None) -> None:
        with self._lock:
            if user_id is not None:
                self._conn.execute(
                    "DELETE FROM assessment_profiles WHERE id = ? AND user_id = ?", (template_id, user_id)
                )
            else:
                self._conn.execute("DELETE FROM assessment_profiles WHERE id = ?", (template_id,))
            self._conn.commit()

    def set_active_profile_template(self, template_id: int, user_id: Optional[int] = None) -> None:
        """Marks `template_id` as this user's active default (see
        profile_confirm.py, which prefills from it) -- at most one template
        per user is ever active, so this clears the flag from any other of
        the user's templates first."""
        with self._lock:
            self._conn.execute("UPDATE assessment_profiles SET is_active = 0 WHERE user_id IS ?", (user_id,))
            self._conn.execute(
                "UPDATE assessment_profiles SET is_active = 1 WHERE id = ? AND user_id IS ?",
                (template_id, user_id),
            )
            self._conn.commit()

    def get_active_profile_template(self, user_id: Optional[int] = None) -> Optional["ProfileTemplateRecord"]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM assessment_profiles WHERE user_id IS ? AND is_active = 1", (user_id,)
            ).fetchone()
            return self._row_to_profile_template(row) if row else None

    @staticmethod
    def _row_to_profile_template(row: sqlite3.Row) -> "ProfileTemplateRecord":
        from .profiles import AssessmentProfile

        profile = AssessmentProfile(
            competencies=json.loads(row["competencies_json"]),
            role=row["role"], seniority=row["seniority"],
            industry=row["industry"], company_type=row["company_type"],
            name=row["name"],
        )
        return ProfileTemplateRecord(
            id=row["id"], user_id=row["user_id"], name=row["name"],
            created_at=row["created_at"], updated_at=row["updated_at"],
            is_active=bool(row["is_active"]), profile=profile,
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
