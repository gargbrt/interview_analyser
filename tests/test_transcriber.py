"""Tests for transcribe()'s progress callback. The real faster-whisper
model is a manual-verification boundary (see test_end_to_end.py's
docstring) -- WhisperModel itself is faked here with segments/info shaped
like the real library's output, so the progress-fraction math and the
speaker-labeling around it are what's actually under test."""
from __future__ import annotations

import wave
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from interview_analyzer.config_loader import Config
from interview_analyzer.transcriber import _channel_count, transcribe


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


class TestChannelCount:
    """_channel_count() is a real (not mocked) probe against an actual
    file -- it's the one part of the dual-channel path cheap enough to
    exercise directly rather than through faked faster-whisper internals."""

    def _write_wav(self, path, channels: int, n_frames: int = 8):
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * n_frames * channels)

    def test_detects_mono(self, tmp_path):
        path = tmp_path / "mono.wav"
        self._write_wav(path, channels=1)
        assert _channel_count(path) == 1

    def test_detects_stereo(self, tmp_path):
        path = tmp_path / "stereo.wav"
        self._write_wav(path, channels=2)
        assert _channel_count(path) == 2

    def test_returns_one_for_a_missing_file(self, tmp_path):
        assert _channel_count(tmp_path / "does_not_exist.wav") == 1


class TestDualChannelSpeakerLabeling:
    """Speaker labeling via separate mic/loopback channels (see
    recorder.py) -- the default, tried before falling back to acoustic
    diarization for mono recordings. faster_whisper.audio.decode_audio and
    WhisperModel are both faked (real behavior needs a real stereo
    recording, a manual-verification boundary like the rest of this
    module's real-audio dependencies)."""

    def _config(self, diarization=True) -> Config:
        return Config(raw={
            "transcription": {"whisper_model": "tiny", "device": "cpu", "diarization": diarization},
        })

    def test_merges_both_channels_by_start_time_with_correct_labels(self, tmp_path):
        you_segments = [_fake_segment(1.0, 2.0, "Hi there")]
        interviewer_segments = [_fake_segment(0.0, 1.0, "Hello")]
        you_info = SimpleNamespace(duration=2.0)
        interviewer_info = SimpleNamespace(duration=2.0)

        fake_audio = (np.zeros(10, dtype=np.float32), np.zeros(10, dtype=np.float32))

        with patch("interview_analyzer.transcriber._channel_count", return_value=2), \
             patch("faster_whisper.audio.decode_audio", return_value=fake_audio), \
             patch("faster_whisper.WhisperModel") as MockModel:
            MockModel.return_value.feature_extractor.sampling_rate = 16000
            MockModel.return_value.transcribe.side_effect = [
                (iter(you_segments), you_info),
                (iter(interviewer_segments), interviewer_info),
            ]
            transcript = transcribe(tmp_path / "call.wav", self._config())

        # chronological order (interviewer spoke first at t=0), not
        # channel-processing order (you channel is transcribed first)
        assert transcript.splitlines() == ["[Interviewer] Hello", "[You] Hi there"]

    def test_skips_empty_segments(self, tmp_path):
        you_segments = [_fake_segment(0.0, 1.0, "   ")]  # whitespace-only, e.g. a VAD false positive
        interviewer_segments = [_fake_segment(1.0, 2.0, "Real speech")]
        info = SimpleNamespace(duration=2.0)
        fake_audio = (np.zeros(10, dtype=np.float32), np.zeros(10, dtype=np.float32))

        with patch("interview_analyzer.transcriber._channel_count", return_value=2), \
             patch("faster_whisper.audio.decode_audio", return_value=fake_audio), \
             patch("faster_whisper.WhisperModel") as MockModel:
            MockModel.return_value.feature_extractor.sampling_rate = 16000
            MockModel.return_value.transcribe.side_effect = [
                (iter(you_segments), info),
                (iter(interviewer_segments), info),
            ]
            transcript = transcribe(tmp_path / "call.wav", self._config())

        assert transcript.splitlines() == ["[Interviewer] Real speech"]

    def test_progress_split_evenly_across_both_channels(self, tmp_path):
        you_segments = [_fake_segment(0.0, 5.0, "half")]
        interviewer_segments = [_fake_segment(0.0, 10.0, "all")]
        you_info = SimpleNamespace(duration=10.0)
        interviewer_info = SimpleNamespace(duration=10.0)
        fake_audio = (np.zeros(10, dtype=np.float32), np.zeros(10, dtype=np.float32))

        with patch("interview_analyzer.transcriber._channel_count", return_value=2), \
             patch("faster_whisper.audio.decode_audio", return_value=fake_audio), \
             patch("faster_whisper.WhisperModel") as MockModel:
            MockModel.return_value.feature_extractor.sampling_rate = 16000
            MockModel.return_value.transcribe.side_effect = [
                (iter(you_segments), you_info),
                (iter(interviewer_segments), interviewer_info),
            ]
            progress_calls = []
            transcribe(tmp_path / "call.wav", self._config(), on_progress=progress_calls.append)

        # "you" channel: 5/10 duration -> 0.5 fraction -> 0.0 + 0.5*0.5 = 0.25
        # "interviewer" channel: 10/10 -> 1.0 fraction -> 0.5 + 0.5*1.0 = 1.0
        assert progress_calls == [0.25, 1.0]

    def test_stereo_audio_falls_back_to_mono_path_when_diarization_disabled(self, tmp_path):
        """diarization: false means "no speaker labels at all", even for a
        stereo (mic-captured) recording -- must not take the dual-channel
        path just because two channels are technically available."""
        info = SimpleNamespace(duration=1.0)

        with patch("interview_analyzer.transcriber._channel_count", return_value=2), \
             patch("faster_whisper.WhisperModel") as MockModel:
            MockModel.return_value.transcribe.return_value = (iter([_fake_segment(0, 1, "hi")]), info)
            transcript = transcribe(tmp_path / "call.wav", self._config(diarization=False))

        assert transcript == "[Speaker] hi"
        MockModel.return_value.transcribe.assert_called_once()  # single pass, not two

    def test_cancel_event_stops_a_dual_channel_transcription(self, tmp_path):
        import threading

        from interview_analyzer.transcriber import TranscriptionCancelled

        you_segments = [_fake_segment(0.0, 1.0, "hi")]
        info = SimpleNamespace(duration=1.0)
        fake_audio = (np.zeros(10, dtype=np.float32), np.zeros(10, dtype=np.float32))
        cancel_event = threading.Event()
        cancel_event.set()

        with patch("interview_analyzer.transcriber._channel_count", return_value=2), \
             patch("faster_whisper.audio.decode_audio", return_value=fake_audio), \
             patch("faster_whisper.WhisperModel") as MockModel:
            MockModel.return_value.feature_extractor.sampling_rate = 16000
            MockModel.return_value.transcribe.return_value = (iter(you_segments), info)
            try:
                transcribe(tmp_path / "call.wav", self._config(), cancel_event=cancel_event)
                assert False, "expected TranscriptionCancelled"
            except TranscriptionCancelled:
                pass
