import datetime as dt

from interview_analyzer.cleanup import run_cleanup
from interview_analyzer.db import InterviewDB


def test_cleanup_deletes_expired_audio_file(tmp_path):
    db = InterviewDB(tmp_path / "test.db")
    audio_path = tmp_path / "recording.opus"
    audio_path.write_bytes(b"fake audio bytes")

    iid = db.start_interview("Teams", str(audio_path), retention_days=0)
    past = (dt.datetime.now() - dt.timedelta(days=1)).isoformat()
    db._conn.execute("UPDATE interviews SET audio_expires_at = ? WHERE id = ?", (past, iid))
    db._conn.commit()

    assert audio_path.exists()
    deleted_count = run_cleanup(db)

    assert deleted_count == 1
    assert not audio_path.exists()
    assert db.get(iid).audio_deleted is True


def test_cleanup_ignores_non_expired_audio(tmp_path):
    db = InterviewDB(tmp_path / "test.db")
    audio_path = tmp_path / "recording.opus"
    audio_path.write_bytes(b"fake audio bytes")

    db.start_interview("Teams", str(audio_path), retention_days=3)  # expires in future

    deleted_count = run_cleanup(db)
    assert deleted_count == 0
    assert audio_path.exists()


def test_cleanup_handles_already_missing_file_gracefully(tmp_path):
    db = InterviewDB(tmp_path / "test.db")
    missing_path = tmp_path / "already_gone.opus"

    iid = db.start_interview("Teams", str(missing_path), retention_days=0)
    past = (dt.datetime.now() - dt.timedelta(days=1)).isoformat()
    db._conn.execute("UPDATE interviews SET audio_expires_at = ? WHERE id = ?", (past, iid))
    db._conn.commit()

    deleted_count = run_cleanup(db)  # should not raise even though file doesn't exist
    assert deleted_count == 1
    assert db.get(iid).audio_deleted is True
