"""Tests for confidence.py: the "how much should I trust this assessment"
score shown on each report, and the calibration notes fed back into future
analysis prompts from past feedback. Feedback is a 1-10 quality score (10
highest), not a Yes/No -- see db.py's FeedbackRecord."""
from __future__ import annotations

from unittest.mock import MagicMock

from interview_analyzer.confidence import (
    MIN_FEEDBACK_SAMPLES,
    NEGATIVE_SCORE_THRESHOLD,
    calibrated_confidence,
    calibration_notes,
    format_confidence,
)
from interview_analyzer.db import InterviewDB


class TestCalibratedConfidence:
    def test_falls_back_to_model_reported_score_when_too_little_feedback(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")  # empty feedback table
        result = calibrated_confidence(db, user_id=1, model_reported=72)
        assert result == {"score": 72, "source": "model", "sample_size": 0}

    def test_returns_unavailable_when_no_feedback_and_no_model_score(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        result = calibrated_confidence(db, user_id=1, model_reported=None)
        assert result["score"] is None
        assert result["source"] == "unavailable"

    def test_uses_average_feedback_score_once_enough_samples_exist(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        iid_a = db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)
        iid_b = db.start_interview("Zoom", str(tmp_path / "b.wav"), retention_days=3, user_id=1)
        iid_c = db.start_interview("Zoom", str(tmp_path / "c.wav"), retention_days=3, user_id=1)
        db.save_feedback(iid_a, user_id=1, transcript_score=10, analysis_score=10, comment="")
        db.save_feedback(iid_b, user_id=1, transcript_score=10, analysis_score=8, comment="")
        db.save_feedback(iid_c, user_id=1, transcript_score=10, analysis_score=3, comment="")

        assert MIN_FEEDBACK_SAMPLES == 3  # this test assumes exactly the threshold
        result = calibrated_confidence(db, user_id=1, model_reported=99)
        # average analysis_score = (10+8+3)/3 = 7.0 -> 70%
        assert result == {"score": 70, "source": "feedback", "sample_size": 3}

    def test_ignores_ratings_that_only_cover_transcript_not_analysis(self, tmp_path):
        """analysis_score is the signal for analysis confidence -- someone
        who only ever rated transcription accuracy shouldn't count toward
        the analysis-quality sample size."""
        db = InterviewDB(tmp_path / "test.db")
        for i in range(5):
            iid = db.start_interview("Zoom", str(tmp_path / f"{i}.wav"), retention_days=3, user_id=1)
            db.save_feedback(iid, user_id=1, transcript_score=2, analysis_score=None, comment="")

        result = calibrated_confidence(db, user_id=1, model_reported=55)
        assert result == {"score": 55, "source": "model", "sample_size": 0}

    def test_falls_back_gracefully_when_feedback_table_is_unreadable(self):
        broken_db = MagicMock()
        broken_db.list_feedback.side_effect = RuntimeError("db locked")

        result = calibrated_confidence(broken_db, user_id=1, model_reported=80)
        assert result == {"score": 80, "source": "model", "sample_size": 0}

    def test_returns_unavailable_when_feedback_unreadable_and_no_model_score(self):
        broken_db = MagicMock()
        broken_db.list_feedback.side_effect = RuntimeError("db locked")

        result = calibrated_confidence(broken_db, user_id=1, model_reported=None)
        assert result["score"] is None
        assert result["source"] == "unavailable"

    def test_clamps_out_of_range_model_scores(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        assert calibrated_confidence(db, user_id=1, model_reported=150)["score"] == 100
        assert calibrated_confidence(db, user_id=1, model_reported=-10)["score"] == 0

    def test_handles_non_numeric_model_score_gracefully(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        result = calibrated_confidence(db, user_id=1, model_reported="not a number")
        assert result["score"] is None
        assert result["source"] == "unavailable"


class TestFormatConfidence:
    def test_feedback_sourced_score(self):
        text = format_confidence({"score": 82, "source": "feedback", "sample_size": 12})
        assert "82%" in text
        assert "12" in text

    def test_model_sourced_score(self):
        text = format_confidence({"score": 74, "source": "model", "sample_size": 0})
        assert "74%" in text
        assert "self-assessment" in text

    def test_unavailable(self):
        assert format_confidence({"score": None, "source": "unavailable", "sample_size": 0}) == "not available"
        assert format_confidence(None) == "not available"


class TestCalibrationNotes:
    def test_empty_when_no_negative_feedback(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        iid = db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)
        db.save_feedback(iid, user_id=1, transcript_score=9, analysis_score=9, comment="great")

        assert calibration_notes(db, user_id=1) == ""

    def test_includes_comments_from_low_scoring_feedback(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        iid = db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)
        db.save_feedback(
            iid, user_id=1, transcript_score=9, analysis_score=NEGATIVE_SCORE_THRESHOLD,
            comment="missed that I gave a metric in my answer",
        )

        notes = calibration_notes(db, user_id=1)
        assert "missed that I gave a metric in my answer" in notes

    def test_excludes_comments_from_above_threshold_feedback(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        iid = db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)
        db.save_feedback(
            iid, user_id=1, transcript_score=9, analysis_score=NEGATIVE_SCORE_THRESHOLD + 1,
            comment="a decent analysis, minor nitpick",
        )

        assert calibration_notes(db, user_id=1) == ""

    def test_ignores_negative_feedback_without_a_comment(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        iid = db.start_interview("Zoom", str(tmp_path / "a.wav"), retention_days=3, user_id=1)
        db.save_feedback(iid, user_id=1, transcript_score=9, analysis_score=1, comment="")

        assert calibration_notes(db, user_id=1) == ""

    def test_falls_back_to_empty_when_feedback_table_is_unreadable(self):
        broken_db = MagicMock()
        broken_db.list_feedback.side_effect = RuntimeError("db locked")

        assert calibration_notes(broken_db, user_id=1) == ""

    def test_limits_to_the_most_recent_n_comments(self, tmp_path):
        db = InterviewDB(tmp_path / "test.db")
        for i in range(8):
            iid = db.start_interview("Zoom", str(tmp_path / f"{i}.wav"), retention_days=3, user_id=1)
            db.save_feedback(iid, user_id=1, transcript_score=9, analysis_score=1, comment=f"issue {i}")

        notes = calibration_notes(db, user_id=1, limit=3)
        assert notes.count("- issue") == 3
        assert "issue 7" in notes  # most recent kept
        assert "issue 0" not in notes  # oldest dropped
