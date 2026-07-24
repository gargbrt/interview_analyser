import datetime as dt
import sqlite3

import pytest

from interview_analyzer.db import InterviewDB
from interview_analyzer.profiles import GENERIC_PROFILE, AssessmentProfile


def test_start_and_end_interview(tmp_path):
    db = InterviewDB(tmp_path / "test.db")
    iid = db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)
    assert iid is not None

    db.end_interview(iid)
    record = db.get(iid)
    assert record.ended_at is not None
    assert record.source_app == "Zoom"
    assert record.user_id == 1


def test_end_interview_accepts_an_explicit_timestamp(tmp_path):
    """Used to back-fill ended_at for an interview whose original recording
    never cleanly finished (see reprocess_interview in watcher.py) with a
    timestamp computed from the audio's own duration, rather than always
    stamping "now"."""
    db = InterviewDB(tmp_path / "test.db")
    iid = db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)

    explicit = "2026-07-22T20:45:00"
    db.end_interview(iid, ended_at=explicit)

    assert db.get(iid).ended_at == explicit


def test_user_scoping(tmp_path):
    db = InterviewDB(tmp_path / "test.db")
    db.start_interview("Teams", str(tmp_path / "a.wav"), retention_days=3, user_id=1)
    db.start_interview("Meet", str(tmp_path / "b.wav"), retention_days=3, user_id=2)

    user1_interviews = db.list_all(user_id=1)
    user2_interviews = db.list_all(user_id=2)
    all_interviews = db.list_all()

    assert len(user1_interviews) == 1
    assert len(user2_interviews) == 1
    assert len(all_interviews) == 2
    assert user1_interviews[0].source_app == "Teams"


def test_save_transcript_and_analysis(tmp_path):
    db = InterviewDB(tmp_path / "test.db")
    iid = db.start_interview("Webex", str(tmp_path / "a.wav"), retention_days=3)

    db.save_transcript(iid, "[Interviewer] Hi\n[You] Hello")
    db.save_analysis(iid, {"qa_pairs": [], "session_summary": {}})

    record = db.get(iid)
    assert "Hello" in record.transcript
    assert record.analysis == {"qa_pairs": [], "session_summary": {}}


def test_expired_audio_detection(tmp_path):
    db = InterviewDB(tmp_path / "test.db")
    # retention_days=0 with a manual backdated expiry to simulate "already expired"
    iid = db.start_interview("Chime", str(tmp_path / "a.wav"), retention_days=0)
    past = (dt.datetime.now() - dt.timedelta(days=1)).isoformat()
    db._conn.execute("UPDATE interviews SET audio_expires_at = ? WHERE id = ?", (past, iid))
    db._conn.commit()

    expired = db.list_expired_audio()
    assert len(expired) == 1
    assert expired[0].id == iid

    db.mark_audio_deleted(iid)
    assert db.list_expired_audio() == []


def test_delete_interview_removes_the_row(tmp_path):
    db = InterviewDB(tmp_path / "test.db")
    iid = db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)
    other_iid = db.start_interview("Teams", str(tmp_path / "b.wav"), retention_days=3, user_id=1)

    db.delete_interview(iid)

    assert db.get(iid) is None
    assert db.get(other_iid) is not None  # unrelated rows are untouched
    assert [r.id for r in db.list_all(user_id=1)] == [other_iid]


def test_delete_interview_on_unknown_id_is_a_no_op(tmp_path):
    db = InterviewDB(tmp_path / "test.db")
    db.delete_interview(999)  # must not raise


def test_save_and_get_feedback(tmp_path):
    db = InterviewDB(tmp_path / "test.db")
    iid = db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)

    assert db.get_feedback(iid) is None

    db.save_feedback(iid, user_id=1, transcript_score=9, analysis_score=3, comment="missed a question")

    fb = db.get_feedback(iid)
    assert fb.transcript_score == 9
    assert fb.analysis_score == 3
    assert fb.comment == "missed a question"
    assert fb.user_id == 1


def test_save_feedback_upserts_rather_than_duplicates(tmp_path):
    db = InterviewDB(tmp_path / "test.db")
    iid = db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)

    db.save_feedback(iid, user_id=1, transcript_score=2, analysis_score=2, comment="first take")
    db.save_feedback(iid, user_id=1, transcript_score=10, analysis_score=10, comment="changed my mind")

    fb = db.get_feedback(iid)
    assert fb.transcript_score == 10
    assert fb.analysis_score == 10
    assert fb.comment == "changed my mind"
    assert len(db.list_feedback(user_id=1)) == 1


def test_save_feedback_allows_partial_ratings(tmp_path):
    """A user might only judge the transcription, or only the analysis --
    the other field stays NULL (unrated), not coerced to a score."""
    db = InterviewDB(tmp_path / "test.db")
    iid = db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)

    db.save_feedback(iid, user_id=1, transcript_score=7, analysis_score=None, comment="")

    fb = db.get_feedback(iid)
    assert fb.transcript_score == 7
    assert fb.analysis_score is None


def test_save_feedback_with_both_none_clears_ratings_but_keeps_the_row(tmp_path):
    """The "Clear ratings" UI action resets to unrated and saves -- distinct
    from delete_feedback(), which removes the row entirely."""
    db = InterviewDB(tmp_path / "test.db")
    iid = db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)
    db.save_feedback(iid, user_id=1, transcript_score=8, analysis_score=6, comment="notes")

    db.save_feedback(iid, user_id=1, transcript_score=None, analysis_score=None, comment="")

    fb = db.get_feedback(iid)
    assert fb is not None
    assert fb.transcript_score is None
    assert fb.analysis_score is None
    assert fb.comment == ""


def test_save_feedback_rejects_out_of_range_scores(tmp_path):
    db = InterviewDB(tmp_path / "test.db")
    iid = db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)

    with pytest.raises(ValueError):
        db.save_feedback(iid, user_id=1, transcript_score=0, analysis_score=None, comment="")
    with pytest.raises(ValueError):
        db.save_feedback(iid, user_id=1, transcript_score=11, analysis_score=None, comment="")
    with pytest.raises(ValueError):
        db.save_feedback(iid, user_id=1, transcript_score=None, analysis_score=-1, comment="")


def test_delete_feedback_removes_the_row_without_touching_the_interview(tmp_path):
    db = InterviewDB(tmp_path / "test.db")
    iid = db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)
    db.save_feedback(iid, user_id=1, transcript_score=9, analysis_score=8, comment="great")

    db.delete_feedback(iid)

    assert db.get_feedback(iid) is None
    assert db.get(iid) is not None  # the interview itself is untouched


def test_delete_feedback_on_unrated_interview_is_a_no_op(tmp_path):
    db = InterviewDB(tmp_path / "test.db")
    iid = db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)
    db.delete_feedback(iid)  # must not raise


def test_list_feedback_scoped_by_user(tmp_path):
    db = InterviewDB(tmp_path / "test.db")
    iid1 = db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)
    iid2 = db.start_interview("Meet", str(tmp_path / "b.wav"), retention_days=3, user_id=2)
    db.save_feedback(iid1, user_id=1, transcript_score=8, analysis_score=8, comment="")
    db.save_feedback(iid2, user_id=2, transcript_score=8, analysis_score=8, comment="")

    assert len(db.list_feedback(user_id=1)) == 1
    assert len(db.list_feedback(user_id=2)) == 1
    assert len(db.list_feedback()) == 2


def test_delete_interview_also_removes_its_feedback(tmp_path):
    db = InterviewDB(tmp_path / "test.db")
    iid = db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)
    db.save_feedback(iid, user_id=1, transcript_score=8, analysis_score=8, comment="")

    db.delete_interview(iid)

    assert db.get_feedback(iid) is None


class TestProfileSnapshotMigration:
    def test_profile_snapshot_json_column_is_added_to_a_pre_existing_db(self, tmp_path):
        """Regression coverage for a real migration hazard: a DB file
        created by a version of this app before profile_snapshot_json
        existed must not crash (or silently lose the column) when opened by
        the current code -- CREATE TABLE IF NOT EXISTS alone doesn't add
        columns to an already-existing table."""
        db_path = tmp_path / "old.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE interviews (
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
                report_path TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO interviews (user_id, started_at, source_app) VALUES (1, '2026-01-01T00:00:00', 'Zoom')"
        )
        conn.commit()
        conn.close()

        db = InterviewDB(db_path)  # must not raise

        record = db.list_all(user_id=1)[0]
        assert record.profile_snapshot_json is None
        assert record.profile is None  # falls back to GENERIC_PROFILE at the call site

        # and the column is now genuinely writable
        db.save_profile_snapshot(record.id, GENERIC_PROFILE)
        assert db.get(record.id).profile == GENERIC_PROFILE

    def test_migration_is_idempotent_across_repeated_opens(self, tmp_path):
        db_path = tmp_path / "test.db"
        InterviewDB(db_path)
        InterviewDB(db_path)  # must not raise "duplicate column" on a second open


class TestProfileSnapshot:
    def test_save_and_read_back_a_profile_snapshot(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        iid = db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)
        profile = AssessmentProfile(
            competencies=["Technical Expertise", "Leadership"],
            role="Software Engineer", seniority="Senior/Lead",
        )

        db.save_profile_snapshot(iid, profile)

        assert db.get(iid).profile == profile

    def test_no_snapshot_saved_means_profile_is_none(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        iid = db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)
        assert db.get(iid).profile is None


class TestProfileTemplates:
    def test_create_and_list_templates_scoped_by_user(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        profile1 = AssessmentProfile(competencies=["Leadership"], role="Product")
        profile2 = AssessmentProfile(competencies=["Execution"], role="Sales")

        db.create_profile_template(user_id=1, name="PM Senior", profile=profile1)
        db.create_profile_template(user_id=2, name="Sales Entry", profile=profile2)

        user1_templates = db.list_profile_templates(user_id=1)
        assert len(user1_templates) == 1
        assert user1_templates[0].name == "PM Senior"
        assert user1_templates[0].profile.role == "Product"
        assert len(db.list_profile_templates(user_id=2)) == 1

    def test_saving_the_same_name_again_overwrites_rather_than_duplicates(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        db.create_profile_template(user_id=1, name="My Profile", profile=AssessmentProfile(role="Product"))
        db.create_profile_template(user_id=1, name="My Profile", profile=AssessmentProfile(role="Sales"))

        templates = db.list_profile_templates(user_id=1)
        assert len(templates) == 1
        assert templates[0].profile.role == "Sales"

    def test_get_profile_template_by_id(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        template_id = db.create_profile_template(user_id=1, name="My Profile", profile=AssessmentProfile(role="Data"))

        fetched = db.get_profile_template(template_id, user_id=1)

        assert fetched is not None
        assert fetched.profile.role == "Data"

    def test_get_profile_template_is_invisible_to_a_different_user(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        template_id = db.create_profile_template(user_id=1, name="My Profile", profile=AssessmentProfile())

        assert db.get_profile_template(template_id, user_id=2) is None

    def test_delete_profile_template(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        template_id = db.create_profile_template(user_id=1, name="My Profile", profile=AssessmentProfile())

        db.delete_profile_template(template_id, user_id=1)

        assert db.list_profile_templates(user_id=1) == []

    def test_delete_profile_template_scoped_to_the_right_user(self, tmp_path):
        """A user must not be able to delete another user's template even
        if they somehow know its id."""
        db = InterviewDB(tmp_path / "test.db")
        template_id = db.create_profile_template(user_id=1, name="My Profile", profile=AssessmentProfile())

        db.delete_profile_template(template_id, user_id=2)

        assert len(db.list_profile_templates(user_id=1)) == 1

    def test_new_templates_are_not_active_by_default(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        db.create_profile_template(user_id=1, name="My Profile", profile=AssessmentProfile())

        assert db.get_active_profile_template(user_id=1) is None
        assert db.list_profile_templates(user_id=1)[0].is_active is False

    def test_set_active_profile_template(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        template_id = db.create_profile_template(user_id=1, name="My Profile", profile=AssessmentProfile(role="Product"))

        db.set_active_profile_template(template_id, user_id=1)

        active = db.get_active_profile_template(user_id=1)
        assert active is not None
        assert active.profile.role == "Product"

    def test_setting_a_new_active_template_deactivates_the_previous_one(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        first_id = db.create_profile_template(user_id=1, name="First", profile=AssessmentProfile())
        second_id = db.create_profile_template(user_id=1, name="Second", profile=AssessmentProfile())
        db.set_active_profile_template(first_id, user_id=1)

        db.set_active_profile_template(second_id, user_id=1)

        active = db.get_active_profile_template(user_id=1)
        assert active.id == second_id
        templates = {t.id: t for t in db.list_profile_templates(user_id=1)}
        assert templates[first_id].is_active is False

    def test_active_template_is_scoped_by_user(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        template_id = db.create_profile_template(user_id=1, name="My Profile", profile=AssessmentProfile())
        db.set_active_profile_template(template_id, user_id=1)

        assert db.get_active_profile_template(user_id=2) is None


class TestAnalysisHistory:
    def test_append_and_list_history_most_recent_first(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        iid = db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)

        db.append_analysis_history(iid, {"session_summary": {"confidence": 50}}, profile=AssessmentProfile(role="Sales"))
        db.append_analysis_history(iid, {"session_summary": {"confidence": 80}}, profile=AssessmentProfile(role="Data"))

        history = db.list_analysis_history(iid)
        assert len(history) == 2
        assert history[0].analysis["session_summary"]["confidence"] == 80  # most recent first
        assert history[0].profile.role == "Data"
        assert history[1].profile.role == "Sales"

    def test_history_survives_without_a_profile(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        iid = db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)

        db.append_analysis_history(iid, {"session_summary": {}}, profile=None)

        history = db.list_analysis_history(iid)
        assert len(history) == 1
        assert history[0].profile is None

    def test_history_is_pruned_to_the_most_recent_ten(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        iid = db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)

        for i in range(15):
            db.append_analysis_history(iid, {"session_summary": {"confidence": i}}, profile=None)

        history = db.list_analysis_history(iid)
        assert len(history) == 10
        # the 10 most recent (confidence 5..14) survive, oldest 5 pruned
        confidences = sorted(h.analysis["session_summary"]["confidence"] for h in history)
        assert confidences == list(range(5, 15))

    def test_history_is_scoped_to_its_own_interview(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        iid1 = db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)
        iid2 = db.start_interview("Meet", str(tmp_path / "b.wav"), retention_days=3, user_id=1)

        db.append_analysis_history(iid1, {"session_summary": {}}, profile=None)
        db.append_analysis_history(iid2, {"session_summary": {}}, profile=None)
        db.append_analysis_history(iid2, {"session_summary": {}}, profile=None)

        assert len(db.list_analysis_history(iid1)) == 1
        assert len(db.list_analysis_history(iid2)) == 2

    def test_deleting_an_interview_also_removes_its_analysis_history(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        iid = db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)
        db.append_analysis_history(iid, {"session_summary": {}}, profile=None)

        db.delete_interview(iid)

        assert db.list_analysis_history(iid) == []
