"""Tests for transcribe()'s progress callback. The real faster-whisper
model is a manual-verification boundary (see test_end_to_end.py's
docstring) -- WhisperModel itself is faked here with segments/info shaped
like the real library's output, so the progress-fraction math and the
speaker-labeling around it are what's actually under test."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from interview_analyzer.config_loader import Config
from interview_analyzer.transcriber import transcribe


def _fake_segment(start, end, text):
    return SimpleNamespace(start=start, end=end, text=text)


def _config(diarization=False) -> Config:
    return Config(raw={
        "transcription": {"whisper_model": "tiny", "device": "cpu", "diarization": diarization},
    })


def test_progress_callback_reports_fraction_of_total_duration(tmp_path):
    fake_segments = [
        _fake_segment(0.0, 2.5, "Hello"),
        _fake_segment(2.5, 5.0, "there"),
        _fake_segment(5.0, 10.0, "friend"),
    ]
    fake_info = SimpleNamespace(duration=10.0)

    with patch("faster_whisper.WhisperModel") as MockModel:
        MockModel.return_value.transcribe.return_value = (iter(fake_segments), fake_info)
        progress_calls = []
        transcript = transcribe(tmp_path / "call.wav", _config(), on_progress=progress_calls.append)

    assert progress_calls == [0.25, 0.5, 1.0]
    assert "Hello" in transcript
    assert "there" in transcript
    assert "friend" in transcript


def test_progress_callback_is_optional(tmp_path):
    fake_segments = [_fake_segment(0.0, 1.0, "Hi")]
    fake_info = SimpleNamespace(duration=1.0)

    with patch("faster_whisper.WhisperModel") as MockModel:
        MockModel.return_value.transcribe.return_value = (iter(fake_segments), fake_info)
        transcript = transcribe(tmp_path / "call.wav", _config())  # no on_progress passed

    assert "Hi" in transcript


def test_progress_callback_never_exceeds_one_even_if_a_segment_runs_past_duration(tmp_path):
    """Real audio/VAD timing can occasionally put a segment's end slightly
    past the reported total duration -- progress must still cap at 1.0."""
    fake_segments = [_fake_segment(0.0, 10.2, "Hi")]
    fake_info = SimpleNamespace(duration=10.0)

    with patch("faster_whisper.WhisperModel") as MockModel:
        MockModel.return_value.transcribe.return_value = (iter(fake_segments), fake_info)
        progress_calls = []
        transcribe(tmp_path / "call.wav", _config(), on_progress=progress_calls.append)

    assert progress_calls == [1.0]


def test_no_progress_calls_when_duration_is_zero(tmp_path):
    fake_segments = [_fake_segment(0.0, 1.0, "Hi")]
    fake_info = SimpleNamespace(duration=0.0)

    with patch("faster_whisper.WhisperModel") as MockModel:
        MockModel.return_value.transcribe.return_value = (iter(fake_segments), fake_info)
        progress_calls = []
        transcribe(tmp_path / "call.wav", _config(), on_progress=progress_calls.append)

    assert progress_calls == []


class TestVadAndLanguageSettings:
    """Regression coverage for the "speech ~55s in was dropped" bug: without
    vad_filter (and with the hallucination-prone defaults), a short
    utterance right after a long silence could land in a mostly-silent
    ~30s decode window and get silently skipped. These assert the fix's
    parameters are actually the ones passed to faster-whisper, not just that
    *some* transcript comes back."""

    def test_default_config_enables_vad_filter_and_tuned_decode_params(self, tmp_path):
        fake_info = SimpleNamespace(duration=1.0)
        with patch("faster_whisper.WhisperModel") as MockModel:
            MockModel.return_value.transcribe.return_value = (iter([_fake_segment(0, 1, "hi")]), fake_info)
            transcribe(tmp_path / "call.wav", _config())
            _, kwargs = MockModel.return_value.transcribe.call_args

        assert kwargs["vad_filter"] is True
        assert kwargs["vad_parameters"] == dict(min_silence_duration_ms=300, speech_pad_ms=300)
        assert kwargs["condition_on_previous_text"] is False
        assert kwargs["no_speech_threshold"] == 0.4
        assert kwargs["language"] is None  # "auto" (the default) means let Whisper detect

    def test_vad_filter_can_be_disabled_via_config(self, tmp_path):
        cfg = Config(raw={
            "transcription": {"whisper_model": "tiny", "device": "cpu", "diarization": False, "vad_filter": False},
        })
        fake_info = SimpleNamespace(duration=1.0)
        with patch("faster_whisper.WhisperModel") as MockModel:
            MockModel.return_value.transcribe.return_value = (iter([_fake_segment(0, 1, "hi")]), fake_info)
            transcribe(tmp_path / "call.wav", cfg)
            _, kwargs = MockModel.return_value.transcribe.call_args

        assert kwargs["vad_filter"] is False
        assert kwargs["vad_parameters"] is None

    def test_language_setting_is_passed_through_to_whisper(self, tmp_path):
        cfg = Config(raw={
            "transcription": {"whisper_model": "tiny", "device": "cpu", "diarization": False, "language": "en"},
        })
        fake_info = SimpleNamespace(duration=1.0)
        with patch("faster_whisper.WhisperModel") as MockModel:
            MockModel.return_value.transcribe.return_value = (iter([_fake_segment(0, 1, "hi")]), fake_info)
            transcribe(tmp_path / "call.wav", cfg)
            _, kwargs = MockModel.return_value.transcribe.call_args

        assert kwargs["language"] == "en"

    def test_hinglish_language_setting_pins_whisper_to_hindi(self, tmp_path):
        cfg = Config(raw={
            "transcription": {"whisper_model": "tiny", "device": "cpu", "diarization": False, "language": "hinglish"},
        })
        fake_info = SimpleNamespace(duration=1.0)
        with patch("faster_whisper.WhisperModel") as MockModel:
            MockModel.return_value.transcribe.return_value = (iter([_fake_segment(0, 1, "hi")]), fake_info)
            transcribe(tmp_path / "call.wav", cfg)
            _, kwargs = MockModel.return_value.transcribe.call_args

        assert kwargs["language"] == "hi"  # Whisper has no dedicated code-switched code

    def test_hinglish_transliterates_devanagari_output_to_latin_script(self, tmp_path):
        cfg = Config(raw={
            "transcription": {"whisper_model": "tiny", "device": "cpu", "diarization": False, "language": "hinglish"},
        })
        fake_info = SimpleNamespace(duration=1.0)
        with patch("faster_whisper.WhisperModel") as MockModel:
            MockModel.return_value.transcribe.return_value = (
                iter([_fake_segment(0, 1, "नमस्ते")]), fake_info,
            )
            transcript = transcribe(tmp_path / "call.wav", cfg)

        assert "नमस्ते" not in transcript  # romanized, not left in Devanagari
        assert transcript.strip("[Speaker] \n") != ""

    def test_hinglish_falls_back_gracefully_without_indic_transliteration(self, tmp_path):
        cfg = Config(raw={
            "transcription": {"whisper_model": "tiny", "device": "cpu", "diarization": False, "language": "hinglish"},
        })
        fake_info = SimpleNamespace(duration=1.0)
        with patch("faster_whisper.WhisperModel") as MockModel, \
             patch.dict("sys.modules", {"indic_transliteration": None, "indic_transliteration.sanscript": None}):
            MockModel.return_value.transcribe.return_value = (
                iter([_fake_segment(0, 1, "नमस्ते")]), fake_info,
            )
            transcript = transcribe(tmp_path / "call.wav", cfg)

        assert "नमस्ते" in transcript  # left untouched, not dropped or crashed
