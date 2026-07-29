"""Tests for the assessment-profile wiring in watcher.py: which profile a
recording gets (the active default, confirmed/edited via profile_confirm.py
right after recording starts), how it flows into analyze_transcript, the new
selection_probability computed alongside confidence_info, and reprocessing
with a different profile."""
from __future__ import annotations

import time
from unittest.mock import patch

from interview_analyzer.config_loader import Config
from interview_analyzer.profiles import CORE_COMPETENCIES, GENERIC_PROFILE, AssessmentProfile
from interview_analyzer.watcher import MeetingWatcher

FAKE_ANALYSIS_WITH_COMPETENCIES = {
    "qa_pairs": [],
    "session_summary": {
        "top_strengths": [], "top_issues": [], "one_thing_to_practice_next": "", "confidence": 80,
        "competency_scores": [{"name": "Leadership", "score": 70, "remark": "Solid."}],
        "hire_recommendation": {"level": "Hire", "rationale": "Strong overall."},
    },
}


def _test_config(tmp_path) -> Config:
    return Config(raw={
        "retention_days": 3,
        "poll_interval_seconds": 0.01,
        "start_debounce_polls": 1,
        "stop_debounce_polls": 1,
        "watched_processes": {"desktop_apps": [], "browser_tab_keywords": [], "browser_processes": []},
        "audio": {"sample_rate": 16000, "channels": 1, "bitrate_kbps": 64,
                   "format": "opus", "raw_dir": str(tmp_path / "audio")},
        "transcription": {"engine": "faster-whisper", "whisper_model": "tiny",
                           "device": "cpu", "diarization": False},
        "analysis": {"engine": "ollama", "llm_model": "llama3.1:8b",
                     "ollama_host": "http://localhost:11434"},
        "storage": {"db_path": str(tmp_path / "interviews.db")},
        "output": {"output_dir": str(tmp_path / "output"), "reports_subdir": "reports",
                    "trends_filename": "trends.md"},
    })


def _seed_interview(watcher, tmp_path, name="orphaned.wav", app="Zoom") -> int:
    audio_path = tmp_path / "audio" / name
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"RIFF....WAVEreal audio bytes")
    iid = watcher.db.start_interview(app, str(audio_path), retention_days=3, user_id=1)
    watcher.db.end_interview(iid)
    return iid


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class TestGetActiveProfile:
    def test_returns_generic_when_no_active_template(self, tmp_path):
        watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
        assert watcher._get_active_profile() == GENERIC_PROFILE

    def test_returns_the_active_templates_profile(self, tmp_path):
        watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
        template_id = watcher.db.create_profile_template(
            user_id=1, name="PM Senior", profile=AssessmentProfile(role="Product", seniority="Senior/Lead"),
        )
        watcher.db.set_active_profile_template(template_id, user_id=1)

        active = watcher._get_active_profile()

        assert active.role == "Product"
        assert active.seniority == "Senior/Lead"

    def test_falls_back_to_generic_if_the_db_read_fails(self, tmp_path):
        watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
        with patch.object(watcher.db, "get_active_profile_template", side_effect=RuntimeError("db locked")):
            assert watcher._get_active_profile() == GENERIC_PROFILE


class TestConfirmAndSaveProfile:
    def test_saves_whatever_confirm_profile_returns(self, tmp_path):
        watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
        iid = _seed_interview(watcher, tmp_path)
        chosen = AssessmentProfile(competencies=["Execution"], role="Sales")

        with patch("interview_analyzer.watcher.confirm_profile", return_value=chosen) as mock_confirm:
            watcher._confirm_and_save_profile(iid)

        mock_confirm.assert_called_once()
        assert watcher.db.get(iid).profile == chosen

    def test_a_failure_leaves_the_interview_without_a_snapshot_rather_than_raising(self, tmp_path):
        watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
        iid = _seed_interview(watcher, tmp_path)

        with patch("interview_analyzer.watcher.confirm_profile", side_effect=RuntimeError("dialog exploded")):
            watcher._confirm_and_save_profile(iid)  # must not raise

        assert watcher.db.get(iid).profile is None

    def test_the_confirmed_profile_becomes_the_new_active_default(self, tmp_path):
        """The whole point: the NEXT recording's confirm dialog is
        prefilled from _get_active_profile(), so confirming a profile for
        THIS interview must update that default too, not just this one
        interview's own snapshot."""
        watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
        iid = _seed_interview(watcher, tmp_path)
        chosen = AssessmentProfile(competencies=["Execution"], role="Product", seniority="Senior/Lead")

        with patch("interview_analyzer.watcher.confirm_profile", return_value=chosen):
            watcher._confirm_and_save_profile(iid)

        active = watcher._get_active_profile()
        assert active.role == "Product"
        assert active.seniority == "Senior/Lead"
        assert set(active.competencies) == {"Execution"}

    def test_confirming_again_updates_the_default_rather_than_duplicating_it(self, tmp_path):
        watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
        first_iid = _seed_interview(watcher, tmp_path, name="a.wav")
        second_iid = _seed_interview(watcher, tmp_path, name="b.wav")
        first_choice = AssessmentProfile(role="Sales")
        second_choice = AssessmentProfile(role="Data", seniority="Entry Level")

        with patch("interview_analyzer.watcher.confirm_profile", return_value=first_choice):
            watcher._confirm_and_save_profile(first_iid)
        with patch("interview_analyzer.watcher.confirm_profile", return_value=second_choice):
            watcher._confirm_and_save_profile(second_iid)

        # only ever one reserved "remembered default" template, not one
        # per confirmation -- confirming again just updates it in place
        templates = watcher.db.list_profile_templates(user_id=1)
        assert len(templates) == 1
        assert watcher._get_active_profile().role == "Data"
        assert watcher._get_active_profile().seniority == "Entry Level"

    def test_failure_to_remember_the_default_does_not_affect_the_interviews_own_snapshot(self, tmp_path):
        watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
        iid = _seed_interview(watcher, tmp_path)
        chosen = AssessmentProfile(role="Consultant")

        with patch("interview_analyzer.watcher.confirm_profile", return_value=chosen), \
             patch.object(watcher.db, "create_profile_template", side_effect=RuntimeError("db locked")):
            watcher._confirm_and_save_profile(iid)  # must not raise

        # the interview's OWN snapshot still succeeded, even though
        # remembering it as the future default failed
        assert watcher.db.get(iid).profile == chosen

    def test_no_prior_confirmation_still_defaults_to_generic(self, tmp_path):
        """Explicit regression guard for the "default values should be
        kept as generic for all fields" requirement -- a fresh install (or
        a user who has never confirmed a profile) must still see the
        generic fallback, not some other hardcoded default."""
        watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
        assert watcher._get_active_profile() == GENERIC_PROFILE


class TestStartRecordingConfirmsProfileInBackground:
    def test_start_recording_spins_off_profile_confirmation_without_blocking(self, tmp_path):
        """_start_recording itself must return quickly (see its own
        docstring) -- the profile confirmation happens on its own thread,
        which we wait for here rather than it blocking the call."""
        cfg = _test_config(tmp_path)
        watcher = MeetingWatcher(cfg, user_id=1)
        chosen = AssessmentProfile(competencies=["Collaboration"], role="Design")

        with patch("interview_analyzer.watcher.SystemAudioRecorder"), \
             patch("interview_analyzer.watcher.RecordingControlPanel"), \
             patch("interview_analyzer.watcher.confirm_profile", return_value=chosen):
            watcher._start_recording("Zoom")
            interview_id = watcher._current_interview_id

            assert _wait_until(lambda: watcher.db.get(interview_id).profile is not None), \
                "profile confirmation never completed"

        assert watcher.db.get(interview_id).profile == chosen


class TestAnalysisPipelineUsesTheStoredProfile:
    def test_uses_the_interviews_stored_profile_and_computes_selection_probability(self, tmp_path):
        watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
        iid = _seed_interview(watcher, tmp_path)
        stored_profile = AssessmentProfile(competencies=["Leadership"], role="Product", seniority="Senior/Lead")
        watcher.db.save_profile_snapshot(iid, stored_profile)

        captured_profile = {}

        def _fake_analyze(transcript, cfg, on_progress=None, profile=None, calibration_notes=""):
            captured_profile["profile"] = profile
            return FAKE_ANALYSIS_WITH_COMPETENCIES

        with patch("interview_analyzer.watcher.transcribe", return_value="[Interviewer] Hi\n[You] Hello"), \
             patch("interview_analyzer.watcher.analyze_transcript", side_effect=_fake_analyze):
            watcher.reprocess_interview(iid)

        assert captured_profile["profile"] == stored_profile
        record = watcher.db.get(iid)
        assert record.analysis["selection_probability"]["percent"] is not None
        assert record.analysis["selection_probability"]["label"] == "Hire"

    def test_falls_back_to_generic_profile_when_none_is_stored(self, tmp_path):
        watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
        iid = _seed_interview(watcher, tmp_path)  # no profile snapshot saved

        captured_profile = {}

        def _fake_analyze(transcript, cfg, on_progress=None, profile=None, calibration_notes=""):
            captured_profile["profile"] = profile
            return FAKE_ANALYSIS_WITH_COMPETENCIES

        with patch("interview_analyzer.watcher.transcribe", return_value="[Interviewer] Hi\n[You] Hello"), \
             patch("interview_analyzer.watcher.analyze_transcript", side_effect=_fake_analyze):
            watcher.reprocess_interview(iid)

        assert captured_profile["profile"] == GENERIC_PROFILE


class TestReprocessWithDifferentProfile:
    def test_reprocessing_with_a_profile_override_replaces_the_stored_snapshot(self, tmp_path):
        watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
        iid = _seed_interview(watcher, tmp_path)
        watcher.db.save_profile_snapshot(iid, AssessmentProfile(role="Sales"))
        new_profile = AssessmentProfile(competencies=["Execution"], role="Data")

        with patch("interview_analyzer.watcher.transcribe", return_value="[Interviewer] Hi\n[You] Hello"), \
             patch("interview_analyzer.watcher.analyze_transcript", return_value=FAKE_ANALYSIS_WITH_COMPETENCIES):
            watcher.reprocess_interview(iid, profile=new_profile)

        assert watcher.db.get(iid).profile == new_profile

    def test_reprocessing_without_an_override_keeps_the_existing_snapshot(self, tmp_path):
        watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
        iid = _seed_interview(watcher, tmp_path)
        original_profile = AssessmentProfile(role="Sales")
        watcher.db.save_profile_snapshot(iid, original_profile)

        with patch("interview_analyzer.watcher.transcribe", return_value="[Interviewer] Hi\n[You] Hello"), \
             patch("interview_analyzer.watcher.analyze_transcript", return_value=FAKE_ANALYSIS_WITH_COMPETENCIES):
            watcher.reprocess_interview(iid)

        assert watcher.db.get(iid).profile == original_profile

    def test_reprocesses_from_a_saved_transcript_even_after_audio_was_purged(self, tmp_path):
        """Regression coverage for a real bug: retention_days auto-deletes
        raw audio, but the transcript is kept permanently -- and a profile
        change only affects the ANALYSIS step, never transcription. This
        used to hard-require a real audio file on disk, permanently
        blocking "reprocess with a different profile" on every interview
        old enough that its audio had already been purged."""
        watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
        iid = _seed_interview(watcher, tmp_path)
        watcher.db.save_transcript(iid, "[Interviewer] Hi\n[You] Hello")
        watcher.db.update_audio_path(iid, str(tmp_path / "audio" / "long_since_deleted.wav"))

        with patch("interview_analyzer.watcher.transcribe") as mock_transcribe, \
             patch("interview_analyzer.watcher.analyze_transcript", return_value=FAKE_ANALYSIS_WITH_COMPETENCIES):
            watcher.reprocess_interview(iid, profile=AssessmentProfile(role="Product"))

        mock_transcribe.assert_not_called()
        assert watcher.db.get(iid).analysis["session_summary"]["hire_recommendation"]["level"] == "Hire"

    def test_raises_when_both_audio_and_transcript_are_missing(self, tmp_path):
        watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
        iid = _seed_interview(watcher, tmp_path)
        watcher.db.update_audio_path(iid, str(tmp_path / "audio" / "long_since_deleted.wav"))

        try:
            watcher.reprocess_interview(iid)
            assert False, "expected a ValueError"
        except ValueError as e:
            assert "Audio file is missing" in str(e)


class TestAnalysisHistoryRecorded:
    """Every completed analysis (first run and every reprocess) is
    snapshotted to analysis_history -- see db.append_analysis_history --
    so trying a different profile never silently discards a previous take."""

    def test_reprocessing_appends_a_history_entry_with_its_profile(self, tmp_path):
        watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
        iid = _seed_interview(watcher, tmp_path)
        profile = AssessmentProfile(competencies=["Leadership"], role="Product")

        with patch("interview_analyzer.watcher.transcribe", return_value="[Interviewer] Hi\n[You] Hello"), \
             patch("interview_analyzer.watcher.analyze_transcript", return_value=FAKE_ANALYSIS_WITH_COMPETENCIES):
            watcher.reprocess_interview(iid, profile=profile)

        history = watcher.db.list_analysis_history(iid)
        assert len(history) == 1
        assert history[0].profile == profile
        assert history[0].analysis["session_summary"]["hire_recommendation"]["level"] == "Hire"

    def test_repeated_reprocessing_accumulates_multiple_history_entries(self, tmp_path):
        watcher = MeetingWatcher(_test_config(tmp_path), user_id=1)
        iid = _seed_interview(watcher, tmp_path)

        with patch("interview_analyzer.watcher.transcribe", return_value="[Interviewer] Hi\n[You] Hello"), \
             patch("interview_analyzer.watcher.analyze_transcript", return_value=FAKE_ANALYSIS_WITH_COMPETENCIES):
            watcher.reprocess_interview(iid, profile=AssessmentProfile(role="Product"))
            watcher.reprocess_interview(iid, profile=AssessmentProfile(role="Sales"))

        history = watcher.db.list_analysis_history(iid)
        assert len(history) == 2
        assert {h.profile.role for h in history} == {"Product", "Sales"}
