import datetime as dt

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
