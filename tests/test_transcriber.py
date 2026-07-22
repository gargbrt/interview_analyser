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
import pytest

from interview_analyzer.config_loader import Config
from interview_analyzer.transcriber import (
    _GROQ_MAX_CHUNK_SECONDS,
    TranscriptionCancelled,
    _channel_count,
    _filter_mic_bleed,
    _groq_transcribe_array_chunked,
    _groq_transcribe_file,
    _looks_like_mic_bleed,
    _transcribe_via_groq,
    get_audio_duration_seconds,
    load_whisper_model,
    transcribe,
)


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

    def test_initial_prompt_is_passed_through_when_configured(self, tmp_path):
        cfg = Config(raw={
            "transcription": {
                "whisper_model": "tiny", "device": "cpu", "diarization": False,
                "initial_prompt": "Informal spoken interview, possibly Indian-accented English.",
            },
        })
        fake_info = SimpleNamespace(duration=1.0)
        with patch("faster_whisper.WhisperModel") as MockModel:
            MockModel.return_value.transcribe.return_value = (iter([_fake_segment(0, 1, "hi")]), fake_info)
            transcribe(tmp_path / "call.wav", cfg)
            _, kwargs = MockModel.return_value.transcribe.call_args

        assert kwargs["initial_prompt"] == "Informal spoken interview, possibly Indian-accented English."

    def test_initial_prompt_defaults_to_none_when_unset(self, tmp_path):
        fake_info = SimpleNamespace(duration=1.0)
        with patch("faster_whisper.WhisperModel") as MockModel:
            MockModel.return_value.transcribe.return_value = (iter([_fake_segment(0, 1, "hi")]), fake_info)
            transcribe(tmp_path / "call.wav", _config())
            _, kwargs = MockModel.return_value.transcribe.call_args

        assert kwargs["initial_prompt"] is None

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


class TestReusableModel:
    """A caller that already has a loaded WhisperModel (see
    live_transcribe.py, which transcribes one interview in many periodic
    segments rather than one whole-file call) can pass it in via `model=`
    to skip transcribe()'s own loading step entirely."""

    def test_load_whisper_model_uses_configured_settings(self, tmp_path):
        cfg = Config(raw={
            "transcription": {"whisper_model": "medium", "device": "cuda"},
        })
        with patch("faster_whisper.WhisperModel") as MockModel:
            load_whisper_model(cfg)

        MockModel.assert_called_once_with("medium", device="cuda", compute_type="float16")

    def test_load_whisper_model_falls_back_to_local_cache_on_a_network_error(self, tmp_path):
        """A real Hub outage was observed to break loading a model that
        was already fully downloaded and cached -- WhisperModel phones
        home to check for updates by default even then. This is the
        regression guard for the local_files_only=True retry."""
        cfg = Config(raw={"transcription": {"whisper_model": "medium", "device": "cpu"}})
        offline_model = MagicMock()

        with patch("faster_whisper.WhisperModel", side_effect=[ConnectionError("hub unreachable"), offline_model]) as MockModel:
            result = load_whisper_model(cfg)

        assert result is offline_model
        assert MockModel.call_count == 2
        assert MockModel.call_args.kwargs.get("local_files_only") is True

    def test_load_whisper_model_raises_the_original_error_if_not_cached_either(self, tmp_path):
        """First run, model genuinely not downloaded yet: the offline
        retry fails too, and the original (network) error -- not a
        confusing "not found locally" one -- is what should surface."""
        cfg = Config(raw={"transcription": {"whisper_model": "medium", "device": "cpu"}})
        original_error = ConnectionError("hub unreachable")

        with patch("faster_whisper.WhisperModel", side_effect=[original_error, FileNotFoundError("not cached")]):
            try:
                load_whisper_model(cfg)
                assert False, "expected the original ConnectionError to be raised"
            except ConnectionError as e:
                assert e is original_error

    def test_transcribe_uses_the_given_model_instead_of_loading_one(self, tmp_path):
        fake_info = SimpleNamespace(duration=1.0)
        given_model = MagicMock()
        given_model.transcribe.return_value = (iter([_fake_segment(0, 1, "hi")]), fake_info)

        with patch("faster_whisper.WhisperModel") as MockModel:
            transcript = transcribe(tmp_path / "call.wav", _config(), model=given_model)

        MockModel.assert_not_called()
        given_model.transcribe.assert_called_once()
        assert "hi" in transcript


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


class TestGetAudioDurationSeconds:
    """get_audio_duration_seconds() is a real (not mocked) probe against an
    actual file, same spirit as TestChannelCount -- used to back-fill
    ended_at for an interview whose recording never cleanly finished (see
    reprocess_interview in watcher.py)."""

    def _write_wav(self, path, seconds: float, framerate: int = 16000):
        n_frames = int(seconds * framerate)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(framerate)
            wf.writeframes(b"\x00\x00" * n_frames)

    def test_reports_the_real_duration_of_a_wav_file(self, tmp_path):
        path = tmp_path / "three_seconds.wav"
        self._write_wav(path, seconds=3.0)

        duration = get_audio_duration_seconds(path)

        assert duration is not None
        assert duration == pytest.approx(3.0, abs=0.05)

    def test_returns_none_for_a_missing_file(self, tmp_path):
        assert get_audio_duration_seconds(tmp_path / "does_not_exist.wav") is None


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


class TestGroqTranscribeArrayChunked:
    """Regression coverage for a real bug: a single channel of an actual
    ~23-minute interview was ~43MB as raw 16-bit PCM at 16kHz -- Groq
    rejected it outright with HTTP 413 (Payload Too Large, 25MB limit on
    the free tier), even after per-channel splitting. Uploads must be
    chunked to a bounded duration regardless of how long the call was."""

    def test_short_audio_makes_exactly_one_upload(self, tmp_path):
        sample_rate = 16000
        audio = np.zeros(sample_rate * 5, dtype=np.float32)  # 5s, well under one chunk

        with patch("soundfile.write"), \
             patch(
                 "interview_analyzer.transcriber._groq_transcribe_file",
                 return_value=[{"start": 0.0, "end": 5.0, "text": "hello"}],
             ) as mock_file:
            result = _groq_transcribe_array_chunked(
                audio, sample_rate, "whisper-large-v3-turbo", "en", None, "gsk-key", None, None, 0.0, 1.0,
            )

        mock_file.assert_called_once()
        assert result == [{"start": 0.0, "end": 5.0, "text": "hello"}]

    def test_long_audio_is_split_into_multiple_uploads_with_offset_timestamps(self, tmp_path):
        sample_rate = 16000
        # 2.5x the chunk size -- must produce 3 uploads (ceil division)
        audio = np.zeros(int(sample_rate * _GROQ_MAX_CHUNK_SECONDS * 2.5), dtype=np.float32)

        with patch("soundfile.write"), \
             patch(
                 "interview_analyzer.transcriber._groq_transcribe_file",
                 side_effect=[
                     [{"start": 0.0, "end": 1.0, "text": "first"}],
                     [{"start": 0.0, "end": 1.0, "text": "second"}],
                     [{"start": 0.0, "end": 1.0, "text": "third"}],
                 ],
             ) as mock_file:
            result = _groq_transcribe_array_chunked(
                audio, sample_rate, "whisper-large-v3-turbo", None, None, "gsk-key", None, None, 0.0, 1.0,
            )

        assert mock_file.call_count == 3
        # each chunk's local 0.0-1.0 timestamps get offset back to absolute time
        assert result == [
            {"start": 0.0, "end": 1.0, "text": "first"},
            {"start": _GROQ_MAX_CHUNK_SECONDS + 0.0, "end": _GROQ_MAX_CHUNK_SECONDS + 1.0, "text": "second"},
            {"start": 2 * _GROQ_MAX_CHUNK_SECONDS + 0.0, "end": 2 * _GROQ_MAX_CHUNK_SECONDS + 1.0, "text": "third"},
        ]

    def test_each_chunk_stays_under_the_size_budget(self, tmp_path):
        """Every chunk written to disk (before upload) must be at most
        _GROQ_MAX_CHUNK_SECONDS of 16-bit mono PCM -- ~19.2MB, comfortably
        under Groq's 25MB limit."""
        sample_rate = 16000
        audio = np.zeros(int(sample_rate * _GROQ_MAX_CHUNK_SECONDS * 1.5), dtype=np.float32)
        written_lengths = []

        def _fake_write(path, data, rate, **kwargs):
            written_lengths.append(len(data))

        with patch("soundfile.write", side_effect=_fake_write), \
             patch("interview_analyzer.transcriber._groq_transcribe_file", return_value=[]):
            _groq_transcribe_array_chunked(
                audio, sample_rate, "whisper-large-v3-turbo", None, None, "gsk-key", None, None, 0.0, 1.0,
            )

        assert all(n <= sample_rate * _GROQ_MAX_CHUNK_SECONDS for n in written_lengths)

    def test_reports_progress_per_chunk(self, tmp_path):
        sample_rate = 16000
        audio = np.zeros(int(sample_rate * _GROQ_MAX_CHUNK_SECONDS * 2), dtype=np.float32)
        progress_calls = []

        with patch("soundfile.write"), \
             patch("interview_analyzer.transcriber._groq_transcribe_file", return_value=[]):
            _groq_transcribe_array_chunked(
                audio, sample_rate, "whisper-large-v3-turbo", None, None, "gsk-key",
                progress_calls.append, None, 0.0, 1.0,
            )

        assert progress_calls == [0.5, 1.0]

    def test_cancel_event_stops_mid_chunking(self, tmp_path):
        import threading

        sample_rate = 16000
        audio = np.zeros(int(sample_rate * _GROQ_MAX_CHUNK_SECONDS * 3), dtype=np.float32)
        cancel_event = threading.Event()

        call_count = 0

        def _fake_transcribe_file(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                cancel_event.set()
            return []

        with patch("soundfile.write"), \
             patch("interview_analyzer.transcriber._groq_transcribe_file", side_effect=_fake_transcribe_file):
            try:
                _groq_transcribe_array_chunked(
                    audio, sample_rate, "whisper-large-v3-turbo", None, None, "gsk-key",
                    None, cancel_event, 0.0, 1.0,
                )
                assert False, "expected TranscriptionCancelled"
            except TranscriptionCancelled:
                pass

        assert call_count == 2  # stopped before the 3rd chunk


class TestGroqTranscribeFile:
    """_groq_transcribe_file makes one blocking call to Groq's
    /audio/transcriptions endpoint -- the Groq equivalent of a single
    faster-whisper segment stream, just materialized all at once."""

    def test_sends_the_expected_request_shape(self, tmp_path):
        audio_path = tmp_path / "call.wav"
        audio_path.write_bytes(b"fake audio bytes")
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"segments": [{"start": 0.0, "end": 1.0, "text": "hi"}]}
        fake_resp.raise_for_status.return_value = None

        with patch("interview_analyzer.transcriber.requests.post", return_value=fake_resp) as mock_post:
            segments = _groq_transcribe_file(audio_path, "whisper-large-v3-turbo", "en", "a prompt", "gsk-key", None)

        assert segments == [{"start": 0.0, "end": 1.0, "text": "hi"}]
        call = mock_post.call_args
        assert call.args[0] == "https://api.groq.com/openai/v1/audio/transcriptions"
        assert call.kwargs["headers"]["Authorization"] == "Bearer gsk-key"
        assert call.kwargs["data"]["model"] == "whisper-large-v3-turbo"
        assert call.kwargs["data"]["language"] == "en"
        assert call.kwargs["data"]["prompt"] == "a prompt"
        assert call.kwargs["data"]["response_format"] == "verbose_json"

    def test_omits_language_and_prompt_when_not_given(self, tmp_path):
        audio_path = tmp_path / "call.wav"
        audio_path.write_bytes(b"fake audio bytes")
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"segments": []}
        fake_resp.raise_for_status.return_value = None

        with patch("interview_analyzer.transcriber.requests.post", return_value=fake_resp) as mock_post:
            _groq_transcribe_file(audio_path, "whisper-large-v3-turbo", None, None, "gsk-key", None)

        data = mock_post.call_args.kwargs["data"]
        assert "language" not in data
        assert "prompt" not in data

    def test_raises_cancelled_before_making_the_request_if_already_cancelled(self, tmp_path):
        import threading

        audio_path = tmp_path / "call.wav"
        audio_path.write_bytes(b"fake audio bytes")
        cancel_event = threading.Event()
        cancel_event.set()

        with patch("interview_analyzer.transcriber.requests.post") as mock_post:
            try:
                _groq_transcribe_file(audio_path, "whisper-large-v3-turbo", None, None, "gsk-key", cancel_event)
                assert False, "expected TranscriptionCancelled"
            except TranscriptionCancelled:
                pass
        mock_post.assert_not_called()


class TestTranscribeViaGroq:
    def _config(self, diarization=False, **overrides):
        transcription = {"engine": "groq", "diarization": diarization}
        transcription.update(overrides)
        return Config(raw={"transcription": transcription})

    def test_raises_a_clear_error_with_no_api_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("INTERVIEW_ANALYZER_API_KEY", raising=False)
        with patch("interview_analyzer.transcriber.api_keys.load_key", return_value=None):
            try:
                _transcribe_via_groq(tmp_path / "call.wav", self._config(), None, None)
                assert False, "expected RuntimeError"
            except RuntimeError as e:
                assert "console.groq.com" in str(e)

    def test_uses_saved_key_when_no_env_var_set(self, tmp_path, monkeypatch):
        monkeypatch.delenv("INTERVIEW_ANALYZER_API_KEY", raising=False)
        audio_path = tmp_path / "call.wav"
        audio_path.write_bytes(b"fake")
        fake_audio = np.zeros(10, dtype=np.float32)

        with patch("interview_analyzer.transcriber.api_keys.load_key", return_value="gsk-saved"), \
             patch("interview_analyzer.transcriber._channel_count", return_value=1), \
             patch("faster_whisper.audio.decode_audio", return_value=fake_audio), \
             patch("interview_analyzer.transcriber._groq_transcribe_file", return_value=[]) as mock_file:
            _transcribe_via_groq(audio_path, self._config(), None, None)

        assert mock_file.call_args.args[4] == "gsk-saved"

    def test_mono_no_diarization_returns_plain_speaker_labeled_segments(self, tmp_path, monkeypatch):
        monkeypatch.setenv("INTERVIEW_ANALYZER_API_KEY", "gsk-env")
        audio_path = tmp_path / "call.wav"
        audio_path.write_bytes(b"fake")
        fake_audio = np.zeros(10, dtype=np.float32)

        with patch("interview_analyzer.transcriber._channel_count", return_value=1), \
             patch("faster_whisper.audio.decode_audio", return_value=fake_audio), \
             patch(
                 "interview_analyzer.transcriber._groq_transcribe_file",
                 return_value=[{"start": 0.0, "end": 1.0, "text": " hello "}],
             ):
            progress_calls = []
            result = _transcribe_via_groq(audio_path, self._config(diarization=False), progress_calls.append, None)

        assert result == [(0.0, 1.0, "Speaker", "hello")]
        assert progress_calls == [1.0]

    def test_dual_channel_merges_by_start_time_and_reports_progress(self, tmp_path, monkeypatch):
        monkeypatch.setenv("INTERVIEW_ANALYZER_API_KEY", "gsk-env")
        audio_path = tmp_path / "call.wav"
        audio_path.write_bytes(b"fake")
        fake_audio = (np.zeros(10, dtype=np.float32), np.zeros(10, dtype=np.float32))

        you_segments = [{"start": 1.0, "end": 2.0, "text": "Hi there"}]
        interviewer_segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]

        with patch("interview_analyzer.transcriber._channel_count", return_value=2), \
             patch("faster_whisper.audio.decode_audio", return_value=fake_audio), \
             patch("soundfile.write"), \
             patch(
                 "interview_analyzer.transcriber._groq_transcribe_file",
                 side_effect=[you_segments, interviewer_segments],
             ):
            progress_calls = []
            result = _transcribe_via_groq(audio_path, self._config(diarization=True), progress_calls.append, None)

        # chronological order (interviewer spoke first at t=0), not upload order
        assert result == [(0.0, 1.0, "Interviewer", "Hello"), (1.0, 2.0, "You", "Hi there")]
        assert progress_calls == [0.5, 1.0]

    def test_full_transcribe_dispatches_to_groq_when_configured(self, tmp_path, monkeypatch):
        """The top-level transcribe() must route to the Groq path -- and
        must not touch load_whisper_model() at all -- when
        transcription.engine is "groq"."""
        monkeypatch.setenv("INTERVIEW_ANALYZER_API_KEY", "gsk-env")
        audio_path = tmp_path / "call.wav"
        audio_path.write_bytes(b"fake")
        fake_audio = np.zeros(10, dtype=np.float32)

        with patch("interview_analyzer.transcriber._channel_count", return_value=1), \
             patch("faster_whisper.audio.decode_audio", return_value=fake_audio), \
             patch(
                 "interview_analyzer.transcriber._groq_transcribe_file",
                 return_value=[{"start": 0.0, "end": 1.0, "text": "hello"}],
             ), \
             patch("interview_analyzer.transcriber.load_whisper_model") as mock_load_model:
            transcript = transcribe(audio_path, self._config(diarization=False))

        assert transcript == "[Speaker] hello"
        mock_load_model.assert_not_called()


class TestMicBleedFiltering:
    """Regression coverage for a real bug found on an actual interview:
    without headphones, a laptop mic picked up nearly an entire
    interviewer sentence verbatim, which showed up as a duplicated line
    mislabeled "[You]" -- well beyond the "stray word or two" caveat this
    app's docs describe as an expected minor case."""

    def test_looks_like_mic_bleed_true_for_a_near_verbatim_echo(self):
        assert _looks_like_mic_bleed(
            "Today's session is designed to see your analytical process in action.",
            "Today's session is designed to see your analytical",
        ) is True

    def test_looks_like_mic_bleed_false_for_unrelated_text(self):
        assert _looks_like_mic_bleed(
            "Sure, let me walk through an example from my last role.",
            "Can you describe a specific challenge you faced?",
        ) is False

    def test_looks_like_mic_bleed_false_for_short_text_even_if_identical(self):
        """Avoids false-positiving on short, legitimately-overlapping
        replies (e.g. echoing back a word or two of the question)."""
        assert _looks_like_mic_bleed("Yes, exactly.", "Yes, exactly.") is False

    def test_filter_mic_bleed_drops_the_duplicated_you_segment(self):
        """The exact real-world shape of the bug: the interviewer's
        sentence is split across two Interviewer segments, with a bleed
        segment on the You channel landing in between, temporally
        overlapping both."""
        labeled = [
            (0.0, 3.0, "Interviewer", "Welcome to the interview. Today's session is designed to see your analytical"),
            (2.5, 5.5, "You", "Today's session is designed to see your analytical process in action."),
            (3.0, 6.0, "Interviewer", "process in action and reason through a business problem under a time constraint."),
        ]

        result = _filter_mic_bleed(labeled)

        assert result == [
            (0.0, 3.0, "Interviewer", "Welcome to the interview. Today's session is designed to see your analytical"),
            (3.0, 6.0, "Interviewer", "process in action and reason through a business problem under a time constraint."),
        ]

    def test_filter_mic_bleed_keeps_a_genuine_you_answer(self):
        labeled = [
            (0.0, 2.0, "Interviewer", "Can you describe a challenge you faced while managing a product?"),
            (2.5, 10.0, "You", "Sure, there was a project where I redesigned the onboarding flow."),
        ]

        assert _filter_mic_bleed(labeled) == labeled

    def test_filter_mic_bleed_never_drops_an_interviewer_segment(self):
        """Bleed only runs mic-picks-up-speaker-output, not the reverse --
        the interviewer's side has no way to hear (let alone transcribe)
        your microphone."""
        labeled = [
            (0.0, 3.0, "You", "So the answer to that is roughly the same thing you just said."),
            (0.5, 3.5, "Interviewer", "So the answer to that is roughly the same thing you just said."),
        ]

        result = _filter_mic_bleed(labeled)

        assert ("Interviewer" in [spk for _, _, spk, _ in result])

    def test_filter_mic_bleed_ignores_similar_text_far_apart_in_time(self):
        """A recurring phrase used naturally much later in the interview
        shouldn't be treated as bleed just because it's textually similar
        to something the interviewer said long before."""
        labeled = [
            (0.0, 3.0, "Interviewer", "Let's talk about prioritization frameworks for a moment."),
            (600.0, 603.0, "You", "Let's talk about prioritization frameworks for a moment, if that's okay."),
        ]

        assert _filter_mic_bleed(labeled) == labeled
