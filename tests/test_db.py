import datetime as dt

import pytest

from interview_analyzer.db import InterviewDB


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
