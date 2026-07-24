import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from interview_analyzer import api_keys
from interview_analyzer.analyzer import AnthropicEngine, GroqEngine, OllamaEngine, OpenAIEngine, analyze_transcript
from interview_analyzer.config_loader import Config
from interview_analyzer.engines import AnalysisEngine, get_engine, register_engine
from interview_analyzer.profiles import AssessmentProfile
from interview_analyzer.rubric import RESULT_JSON_SCHEMA


class FakeGoodEngine(AnalysisEngine):
    """Simulates an analysis engine that returns well-formed JSON, written
    against the pre-on_progress interface -- exercises analyze_transcript's
    TypeError fallback for third-party engines that predate that parameter."""

    def run(self, prompt: str) -> str:
        assert "[Interviewer]" in prompt or "Transcript" in prompt
        return json.dumps({
            "qa_pairs": [
                {
                    "question": "Tell me about a conflict you resolved.",
                    "answer_summary": "Rambled without a clear structure or outcome.",
                    "issues": [{"category": "structure", "detail": "No STAR structure used."}],
                    "suggested_improvement": "Lead with the situation, then action, then result.",
                }
            ],
            "session_summary": {
                "top_strengths": ["Clear technical vocabulary"],
                "top_issues": ["Lack of structure in behavioral answers"],
                "one_thing_to_practice_next": "Practice STAR format on 3 behavioral questions.",
            },
        })


class FakeBrokenEngine(AnalysisEngine):
    """Simulates an engine that fails to produce valid JSON."""

    def run(self, prompt: str) -> str:
        return "this is not json"


class FakeWrongShapeEngine(AnalysisEngine):
    """Simulates a real failure mode reproduced on an actual interview:
    llama3.1:8b returned syntactically valid JSON that completely ignored
    the requested qa_pairs/session_summary schema -- a generic-looking
    {"title": ..., "topics": [...]} object instead. This used to sail
    through as a "successful" analysis and produce a blank report with no
    way to reprocess it."""

    def run(self, prompt: str) -> str:
        return json.dumps({
            "title": "Product Management Interview",
            "speaker": "Interviewee",
            "topics": ["product roadmap", "market analysis"],
            "time": 15,
            "date": "2023-03-16",
        })


def _make_config(engine_name: str) -> Config:
    return Config(raw={"analysis": {"engine": engine_name}})


def test_custom_engine_registration_and_selection():
    register_engine("fake_good", lambda acfg: FakeGoodEngine())
    cfg = _make_config("fake_good")

    result = analyze_transcript("[Interviewer] Hi\n[You] Hello", cfg)

    assert "qa_pairs" in result
    assert result["session_summary"]["top_issues"] == ["Lack of structure in behavioral answers"]


def test_analyzer_handles_non_json_response_gracefully():
    register_engine("fake_broken", lambda acfg: FakeBrokenEngine())
    cfg = _make_config("fake_broken")

    result = analyze_transcript("[Interviewer] Hi\n[You] Hello", cfg)

    assert result["parse_error"] is True
    assert "raw" in result


def test_unknown_engine_raises_helpful_error():
    with pytest.raises(ValueError, match="Unknown analysis engine"):
        get_engine("totally_made_up_engine", {})


def test_analyzer_treats_wrong_shaped_json_the_same_as_a_parse_error():
    register_engine("fake_wrong_shape", lambda acfg: FakeWrongShapeEngine())
    cfg = _make_config("fake_wrong_shape")

    result = analyze_transcript("[Interviewer] Hi\n[You] Hello", cfg)

    assert result["parse_error"] is True
    assert "raw" in result
    assert json.loads(result["raw"])["title"] == "Product Management Interview"


def test_analyzer_accepts_an_empty_qa_pairs_list_as_valid_shape():
    """qa_pairs=[] is legitimate (e.g. a very short call) -- must not be
    mistaken for the wrong-shape case just because it's empty."""
    class _EmptyButValidEngine(AnalysisEngine):
        def run(self, prompt: str) -> str:
            return json.dumps({"qa_pairs": [], "session_summary": {"top_strengths": [], "top_issues": []}})

    register_engine("fake_empty_valid", lambda acfg: _EmptyButValidEngine())
    cfg = _make_config("fake_empty_valid")

    result = analyze_transcript("[Interviewer] Hi\n[You] Hello", cfg)

    assert result.get("parse_error") is not True
    assert result["qa_pairs"] == []


def _ndjson_lines(*objs) -> list[bytes]:
    return [json.dumps(o).encode("utf-8") for o in objs]


class TestOllamaEngineStreamingProgress:
    """Ollama's streaming /api/generate endpoint reports eval_count (tokens
    generated so far) per NDJSON line -- used to estimate an analyzing %
    for the dashboard, the same way transcription reports real progress.

    Every test here patches ensure_ollama_running to True since it's called
    at the top of OllamaEngine.run() -- see TestOllamaEngineAutoStart for
    coverage of ensure_ollama_running itself."""

    def test_calls_on_progress_with_increasing_fractions_and_reaches_1_0(self):
        engine = OllamaEngine({"ollama_host": "http://localhost:11434", "llm_model": "llama3.1:8b"})
        lines = _ndjson_lines(
            {"response": '{"qa', "eval_count": 10, "done": False},
            {"response": '_pairs": []}', "eval_count": 400, "done": False},
            {"response": "", "eval_count": 420, "done": True},
        )
        fake_resp = MagicMock()
        fake_resp.iter_lines.return_value = lines
        fake_resp.raise_for_status.return_value = None

        with patch("interview_analyzer.analyzer.ensure_ollama_running", return_value=True), \
             patch("interview_analyzer.analyzer.requests.post", return_value=fake_resp) as mock_post:
            progress_calls = []
            raw = engine.run("How many questions? ", on_progress=progress_calls.append)

        assert raw == '{"qa_pairs": []}'
        assert progress_calls[-1] == 1.0
        assert progress_calls == sorted(progress_calls)  # monotonically increasing
        assert mock_post.call_args.kwargs["json"]["format"] == RESULT_JSON_SCHEMA
        assert mock_post.call_args.kwargs["json"]["stream"] is True

    def test_without_on_progress_uses_the_plain_non_streaming_call(self):
        engine = OllamaEngine({"ollama_host": "http://localhost:11434", "llm_model": "llama3.1:8b"})
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"response": '{"qa_pairs": []}'}
        fake_resp.raise_for_status.return_value = None

        with patch("interview_analyzer.analyzer.ensure_ollama_running", return_value=True), \
             patch("interview_analyzer.analyzer.requests.post", return_value=fake_resp) as mock_post:
            raw = engine.run("prompt")

        assert raw == '{"qa_pairs": []}'
        assert mock_post.call_args.kwargs["json"]["stream"] is False

    def test_sends_the_rubric_json_schema_not_just_a_bare_json_format_flag(self):
        """Regression coverage for a real bug: asking Ollama for merely
        "format": "json" only guarantees *some* valid JSON, not the
        expected shape -- reproduced on a real interview, where the model
        returned well-formed but completely unrelated JSON. Passing the
        full schema as `format` constrains decoding to actually match it
        (verified empirically against a real Ollama server)."""
        engine = OllamaEngine({"ollama_host": "http://localhost:11434", "llm_model": "llama3.1:8b"})
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"response": '{"qa_pairs": []}'}
        fake_resp.raise_for_status.return_value = None

        with patch("interview_analyzer.analyzer.ensure_ollama_running", return_value=True), \
             patch("interview_analyzer.analyzer.requests.post", return_value=fake_resp) as mock_post:
            engine.run("prompt")

        assert mock_post.call_args.kwargs["json"]["format"] == RESULT_JSON_SCHEMA

    def test_analyze_transcript_passes_progress_callback_through_to_ollama(self):
        cfg = Config(raw={"analysis": {"engine": "ollama"}})
        response_json = '{"qa_pairs": [], "session_summary": {"top_strengths": [], "top_issues": []}}'
        lines = _ndjson_lines({"response": response_json, "eval_count": 100, "done": True})
        fake_resp = MagicMock()
        fake_resp.iter_lines.return_value = lines
        fake_resp.raise_for_status.return_value = None

        with patch("interview_analyzer.analyzer.ensure_ollama_running", return_value=True), \
             patch("interview_analyzer.analyzer.requests.post", return_value=fake_resp):
            progress_calls = []
            result = analyze_transcript(
                "[Interviewer] Hi\n[You] Hello", cfg, on_progress=progress_calls.append
            )

        assert result == {"qa_pairs": [], "session_summary": {"top_strengths": [], "top_issues": []}}
        assert progress_calls[-1] == 1.0


class TestOllamaEngineAutoStart:
    """OllamaEngine.run() checks/starts Ollama before making any request --
    see model_setup.ensure_ollama_running. Ollama doesn't auto-start with
    the OS by default, so without this a machine reboot (or Ollama simply
    not having been launched yet) surfaces as a raw ConnectionError instead
    of the app just handling it."""

    def test_run_starts_ollama_automatically_before_the_request(self):
        engine = OllamaEngine({"ollama_host": "http://localhost:11434", "llm_model": "llama3.1:8b"})
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"response": '{"qa_pairs": []}'}
        fake_resp.raise_for_status.return_value = None

        with patch("interview_analyzer.analyzer.ensure_ollama_running", return_value=True) as mock_ensure, \
             patch("interview_analyzer.analyzer.requests.post", return_value=fake_resp):
            raw = engine.run("prompt")

        assert raw == '{"qa_pairs": []}'
        mock_ensure.assert_called_once_with("http://localhost:11434")

    def test_run_raises_a_clear_error_when_ollama_cannot_be_started(self):
        engine = OllamaEngine({"ollama_host": "http://localhost:11434", "llm_model": "llama3.1:8b"})

        with patch("interview_analyzer.analyzer.ensure_ollama_running", return_value=False), \
             patch("interview_analyzer.analyzer.requests.post") as mock_post:
            with pytest.raises(RuntimeError, match="couldn't be started automatically"):
                engine.run("prompt")

        mock_post.assert_not_called()


class TestCloudEngineApiKeyResolution:
    """A claude.ai/ChatGPT *subscription* does not grant API access -- both
    cloud engines need a real API key, resolved from either the environment
    variable (the original mechanism) or a key saved via the Settings tab's
    "Cloud API key" section (api_keys.py), with the env var taking
    precedence if both are set.

    api_keys.load_key() itself is mocked directly (rather than exercising
    its real Windows-DPAPI/macOS-Keychain storage backend, which
    api_keys_test.py already covers) so these tests only verify engine
    resolution logic and stay identical regardless of which OS runs them."""

    def test_anthropic_engine_uses_env_var_when_set(self, monkeypatch):
        monkeypatch.setenv("INTERVIEW_ANALYZER_API_KEY", "sk-ant-from-env")
        monkeypatch.setattr(api_keys, "load_key", lambda provider: None)
        engine = AnthropicEngine({})
        assert engine.api_key == "sk-ant-from-env"

    def test_anthropic_engine_falls_back_to_saved_key(self, monkeypatch):
        monkeypatch.delenv("INTERVIEW_ANALYZER_API_KEY", raising=False)
        monkeypatch.setattr(api_keys, "load_key", lambda provider: "sk-ant-saved" if provider == "anthropic_api" else None)
        engine = AnthropicEngine({})
        assert engine.api_key == "sk-ant-saved"

    def test_anthropic_engine_env_var_takes_precedence_over_saved_key(self, monkeypatch):
        monkeypatch.setenv("INTERVIEW_ANALYZER_API_KEY", "sk-ant-from-env")
        monkeypatch.setattr(api_keys, "load_key", lambda provider: "sk-ant-saved")
        engine = AnthropicEngine({})
        assert engine.api_key == "sk-ant-from-env"

    def test_anthropic_engine_raises_a_clear_error_with_neither(self, monkeypatch):
        monkeypatch.delenv("INTERVIEW_ANALYZER_API_KEY", raising=False)
        monkeypatch.setattr(api_keys, "load_key", lambda provider: None)
        with pytest.raises(RuntimeError, match="claude.ai subscription"):
            AnthropicEngine({})

    def test_openai_engine_falls_back_to_saved_key(self, monkeypatch):
        monkeypatch.delenv("INTERVIEW_ANALYZER_API_KEY", raising=False)
        monkeypatch.setattr(api_keys, "load_key", lambda provider: "sk-openai-saved" if provider == "openai_api" else None)
        engine = OpenAIEngine({})
        assert engine.api_key == "sk-openai-saved"

    def test_openai_engine_raises_a_clear_error_with_neither(self, monkeypatch):
        monkeypatch.delenv("INTERVIEW_ANALYZER_API_KEY", raising=False)
        monkeypatch.setattr(api_keys, "load_key", lambda provider: None)
        with pytest.raises(RuntimeError, match="ChatGPT subscription"):
            OpenAIEngine({})

    def test_groq_engine_falls_back_to_saved_key(self, monkeypatch):
        monkeypatch.delenv("INTERVIEW_ANALYZER_API_KEY", raising=False)
        monkeypatch.setattr(api_keys, "load_key", lambda provider: "gsk-saved" if provider == "groq" else None)
        engine = GroqEngine({})
        assert engine.api_key == "gsk-saved"

    def test_groq_engine_raises_a_clear_error_with_neither(self, monkeypatch):
        monkeypatch.delenv("INTERVIEW_ANALYZER_API_KEY", raising=False)
        monkeypatch.setattr(api_keys, "load_key", lambda provider: None)
        with pytest.raises(RuntimeError, match="console.groq.com"):
            GroqEngine({})

    def test_groq_engine_defaults_to_llama_3_3_70b_versatile(self, monkeypatch):
        """Regression coverage for a real bug: the previous default,
        meta-llama/llama-4-scout-17b-16e-instruct, was removed from Groq's
        catalog at some point after it was first chosen -- reproduced
        directly, Groq now returns a 404 "not found" for it. Verified
        directly against Groq's own /v1/models listing that
        llama-3.3-70b-versatile is still available and isn't a reasoning
        model (unlike GPT-OSS)."""
        monkeypatch.setenv("INTERVIEW_ANALYZER_API_KEY", "gsk-from-env")
        engine = GroqEngine({})
        assert engine.model == "llama-3.3-70b-versatile"

    def test_groq_engine_exposes_a_chunking_budget_for_long_transcripts(self, monkeypatch):
        """Regression coverage for a real bug: a full ~1-hour interview
        transcript (~64K characters) needed ~23.7K tokens in one request,
        over llama-3.3-70b-versatile's entire 12K tokens/minute free-tier
        budget by itself -- reproduced directly against Groq's real API, a
        413 no retry could ever fix (that big a request can never fit,
        regardless of timing). analyze_transcript()'s chunking path (see
        TestChunkedAnalysis) relies on this attribute existing and being
        sane."""
        monkeypatch.setenv("INTERVIEW_ANALYZER_API_KEY", "gsk-from-env")
        engine = GroqEngine({})
        assert 0 < engine.max_transcript_chars_per_request <= 12000

    def test_groq_engine_chunking_budget_is_configurable(self, monkeypatch):
        monkeypatch.setenv("INTERVIEW_ANALYZER_API_KEY", "gsk-from-env")
        engine = GroqEngine({"groq_chunk_max_chars": 5000})
        assert engine.max_transcript_chars_per_request == 5000


class TestGroqEngineRequest:
    def test_sends_json_object_response_format(self, monkeypatch):
        """Not response_format=json_schema: reproduced directly against a
        real request, Groq now hard-rejects json_schema (400 "This model
        does not support response format json_schema") for every model
        except GPT-OSS. json_object mode is broadly supported (verified
        directly, including against a real transcript); analyze_transcript()'s
        own shape validation is the actual safety net for a non-compliant
        response, same as before."""
        monkeypatch.setenv("INTERVIEW_ANALYZER_API_KEY", "gsk-from-env")
        engine = GroqEngine({})
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"choices": [{"message": {"content": '{"qa_pairs": []}'}}]}
        fake_resp.raise_for_status.return_value = None

        with patch("interview_analyzer.analyzer.requests.post", return_value=fake_resp) as mock_post:
            raw = engine.run("prompt")

        assert raw == '{"qa_pairs": []}'
        sent = mock_post.call_args.kwargs["json"]
        assert sent["model"] == "llama-3.3-70b-versatile"
        assert sent["response_format"] == {"type": "json_object"}
        assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer gsk-from-env"

    def test_a_404_gets_a_clear_hint_about_the_model_name_not_a_bare_httperror(self, monkeypatch):
        """Regression coverage for a real, confusing failure: switching
        analysis.engine to "groq_api" doesn't reset analysis.llm_model
        (shared across all four engines) -- a name left over from Ollama
        (e.g. "llama3.1:8b") isn't a valid Groq model id, and reproduced
        directly, Groq's plain 404 for that gave no hint why."""
        monkeypatch.setenv("INTERVIEW_ANALYZER_API_KEY", "gsk-from-env")
        engine = GroqEngine({"llm_model": "llama3.1:8b"})
        fake_resp = MagicMock()
        fake_resp.status_code = 404

        with patch("interview_analyzer.analyzer.requests.post", return_value=fake_resp):
            with pytest.raises(RuntimeError, match="llama3.1:8b"):
                engine.run("prompt")

    def test_max_tokens_is_generous_for_a_large_prompt(self, monkeypatch):
        """A large, un-chunked transcript (many questions -> a long rubric
        response) still gets a generous ceiling."""
        monkeypatch.setenv("INTERVIEW_ANALYZER_API_KEY", "gsk-from-env")
        engine = GroqEngine({})
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
        fake_resp.raise_for_status.return_value = None

        with patch("interview_analyzer.analyzer.requests.post", return_value=fake_resp) as mock_post:
            engine.run("x" * 30000)

        assert mock_post.call_args.kwargs["json"]["max_tokens"] == 8000

    def test_max_tokens_scales_down_for_a_small_chunk(self, monkeypatch):
        """Regression coverage for a real bug: Groq's admission control
        reserves the *entire* requested max_tokens against the account's
        per-minute budget up front -- reproduced directly, a small
        ~9000-character transcript chunk was rejected with a 429
        specifically because "Requested" included a full 8000-token
        reservation on top of its own ~3500 prompt tokens, even though its
        actual usage afterward was only ~2800 tokens. A small prompt should
        request a correspondingly small reservation instead of always
        maxing out at 8000."""
        monkeypatch.setenv("INTERVIEW_ANALYZER_API_KEY", "gsk-from-env")
        engine = GroqEngine({})
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
        fake_resp.raise_for_status.return_value = None

        with patch("interview_analyzer.analyzer.requests.post", return_value=fake_resp) as mock_post:
            engine.run("x" * 300)

        assert 1500 <= mock_post.call_args.kwargs["json"]["max_tokens"] < 8000

    def test_retries_a_429_after_waiting_the_retry_after_header(self, monkeypatch):
        """Regression coverage for a real bug: firing several chunk
        requests back to back can trip a *transient* rate limit (recent
        usage from earlier chunks, not this request alone, pushes the
        account over budget that minute) -- reproduced directly against
        Groq's real API, with a real `retry-after` response header. This
        is recoverable by waiting and retrying, unlike a single request
        that's permanently too large."""
        monkeypatch.setenv("INTERVIEW_ANALYZER_API_KEY", "gsk-from-env")
        engine = GroqEngine({})
        rate_limited = MagicMock(status_code=429, headers={"retry-after": "2"})
        success = MagicMock(status_code=200)
        success.json.return_value = {"choices": [{"message": {"content": '{"qa_pairs": []}'}}]}
        success.raise_for_status.return_value = None

        with patch("interview_analyzer.analyzer.requests.post", side_effect=[rate_limited, success]), \
             patch("interview_analyzer.analyzer.time.sleep") as mock_sleep:
            raw = engine.run("prompt")

        assert raw == '{"qa_pairs": []}'
        mock_sleep.assert_called_once_with(3.0)

    def test_gives_up_after_repeated_429s(self, monkeypatch):
        monkeypatch.setenv("INTERVIEW_ANALYZER_API_KEY", "gsk-from-env")
        engine = GroqEngine({})
        rate_limited = MagicMock(status_code=429, headers={"retry-after": "1"})
        rate_limited.raise_for_status.side_effect = requests.exceptions.HTTPError("429")

        with patch("interview_analyzer.analyzer.requests.post", return_value=rate_limited), \
             patch("interview_analyzer.analyzer.time.sleep"):
            with pytest.raises(requests.exceptions.HTTPError):
                engine.run("prompt")

    def test_uses_a_custom_model_when_configured(self, monkeypatch):
        monkeypatch.setenv("INTERVIEW_ANALYZER_API_KEY", "gsk-from-env")
        engine = GroqEngine({"llm_model": "llama-3.1-8b-instant"})
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
        fake_resp.raise_for_status.return_value = None

        with patch("interview_analyzer.analyzer.requests.post", return_value=fake_resp) as mock_post:
            engine.run("prompt")

        assert mock_post.call_args.kwargs["json"]["model"] == "llama-3.1-8b-instant"


class _FakeChunkedEngine(AnalysisEngine):
    """Simulates an engine with a small per-request budget, returning one
    queued response per call so tests can verify chunking actually splits
    the transcript into multiple requests, and that the merge step
    combines the per-chunk results correctly."""

    def __init__(self, max_chars, responses):
        self.max_transcript_chars_per_request = max_chars
        self._responses = list(responses)
        self.calls: list[str] = []

    def run(self, prompt: str, on_progress=None) -> str:
        self.calls.append(prompt)
        return self._responses.pop(0)


class TestChunkedAnalysis:
    """Regression coverage for a real bug: a full ~1-hour interview
    transcript needed far more tokens than Groq's free-tier per-minute
    budget allows in a single request (reproduced directly against a real
    interview: ~64K characters needed ~23.7K tokens against a 12K/minute
    cap), causing an unrecoverable 413 rate-limit error. analyze_transcript
    now splits a transcript that exceeds the engine's declared budget into
    multiple smaller requests and merges the results, rather than either
    failing outright or being limited to engines with generous rate
    limits."""

    def _cfg_for(self, engine, name="fake_chunked") -> Config:
        register_engine(name, lambda acfg: engine)
        return Config(raw={"analysis": {"engine": name}})

    def _long_transcript(self, n=20) -> str:
        return "\n".join(f"[Interviewer] Question {i}?\n[You] Answer number {i}." for i in range(n))

    def test_long_transcript_gets_split_into_multiple_requests(self):
        good = json.dumps({
            "qa_pairs": [{"question": "q"}],
            "session_summary": {"top_strengths": ["Clear"], "top_issues": ["Rambling"],
                                 "one_thing_to_practice_next": "Practice", "confidence": 80},
        })
        engine = _FakeChunkedEngine(max_chars=200, responses=[good] * 20)
        cfg = self._cfg_for(engine, "fake_chunked_split")

        result = analyze_transcript(self._long_transcript(), cfg)

        assert len(engine.calls) > 1
        assert result.get("parse_error") is not True

    def test_short_transcript_is_not_chunked(self):
        engine = _FakeChunkedEngine(
            max_chars=1_000_000,
            responses=[json.dumps({
                "qa_pairs": [], "session_summary": {"top_strengths": [], "top_issues": [],
                                                     "one_thing_to_practice_next": "", "confidence": 50},
            })],
        )
        cfg = self._cfg_for(engine, "fake_chunked_short")

        result = analyze_transcript("[Interviewer] Hi\n[You] Hello", cfg)

        assert len(engine.calls) == 1
        assert result["qa_pairs"] == []

    def test_merges_qa_pairs_and_tallies_strengths_and_issues_across_chunks(self):
        chunk1 = json.dumps({
            "qa_pairs": [{"question": "q1"}],
            "session_summary": {"top_strengths": ["Clear communication"], "top_issues": ["Rambling"],
                                 "one_thing_to_practice_next": "Practice A", "confidence": 60},
        })
        chunk2 = json.dumps({
            "qa_pairs": [{"question": "q2"}],
            "session_summary": {"top_strengths": ["Clear communication"], "top_issues": ["Vague examples"],
                                 "one_thing_to_practice_next": "Practice B", "confidence": 80},
        })
        engine = _FakeChunkedEngine(max_chars=120, responses=[chunk1, chunk2] * 10)
        cfg = self._cfg_for(engine, "fake_chunked_merge")

        result = analyze_transcript(self._long_transcript(10), cfg)

        assert len(result["qa_pairs"]) == len(engine.calls)
        assert "Clear communication" in result["session_summary"]["top_strengths"]
        assert result["session_summary"]["one_thing_to_practice_next"] in ("Practice A", "Practice B")
        assert 60 <= result["session_summary"]["confidence"] <= 80

    def test_a_failing_chunk_is_skipped_not_fatal_to_the_whole_interview(self):
        good = json.dumps({
            "qa_pairs": [{"question": "q1"}],
            "session_summary": {"top_strengths": ["Clear"], "top_issues": [],
                                 "one_thing_to_practice_next": "Practice", "confidence": 70},
        })
        engine = _FakeChunkedEngine(max_chars=120, responses=["not json"] + [good] * 20)
        cfg = self._cfg_for(engine, "fake_chunked_partial_fail")

        result = analyze_transcript(self._long_transcript(10), cfg)

        assert result.get("parse_error") is not True
        assert len(result["qa_pairs"]) >= 1

    def test_every_chunk_failing_reports_a_parse_error(self):
        engine = _FakeChunkedEngine(max_chars=120, responses=["not json"] * 20)
        cfg = self._cfg_for(engine, "fake_chunked_all_fail")

        result = analyze_transcript(self._long_transcript(10), cfg)

        assert result["parse_error"] is True

    def test_merges_competency_scores_by_averaging_and_takes_the_last_hire_recommendation(self):
        """Exercises _merge_chunk_analyses directly with a fixed, known list
        of parsed chunks -- going through analyze_transcript's real
        chunking would make the exact chunk count (and therefore the exact
        average) depend on incidental transcript-splitting details
        unrelated to this merge logic."""
        from interview_analyzer.analyzer import _merge_chunk_analyses

        chunk1 = {
            "qa_pairs": [],
            "session_summary": {
                "competency_scores": [
                    {"name": "Leadership", "score": 60, "remark": "Some ownership shown early on."},
                ],
                "hire_recommendation": {"level": "Lean Hire", "rationale": "Early impression."},
            },
        }
        chunk2 = {
            "qa_pairs": [],
            "session_summary": {
                "competency_scores": [
                    {"name": "Leadership", "score": 80, "remark": "Stronger by the end."},
                ],
                "hire_recommendation": {"level": "Hire", "rationale": "Full conversation considered."},
            },
        }

        result = _merge_chunk_analyses([chunk1, chunk2])

        scores = {c["name"]: c for c in result["session_summary"]["competency_scores"]}
        assert scores["Leadership"]["score"] == 70  # average of 60 and 80
        assert "Some ownership" in scores["Leadership"]["remark"]
        assert "Stronger by the end" in scores["Leadership"]["remark"]
        # last chunk's hire_recommendation wins -- it saw the fuller conversation
        assert result["session_summary"]["hire_recommendation"]["level"] == "Hire"

    def test_merges_competency_scores_across_more_than_two_chunks_via_the_real_chunking_path(self):
        """End-to-end version through the real chunking path -- every
        chunk reports the *same* score for a competency, so the merged
        average is unambiguous regardless of exactly how many chunks the
        transcript actually got split into."""
        same_score_chunk = json.dumps({
            "qa_pairs": [],
            "session_summary": {
                "top_strengths": [], "top_issues": [], "one_thing_to_practice_next": "", "confidence": 70,
                "competency_scores": [{"name": "Execution", "score": 75, "remark": "Consistent."}],
                "hire_recommendation": {"level": "Hire", "rationale": "Consistent."},
            },
        })
        engine = _FakeChunkedEngine(max_chars=120, responses=[same_score_chunk] * 20)
        cfg = self._cfg_for(engine, "fake_chunked_uniform_competency")

        result = analyze_transcript(self._long_transcript(10), cfg)

        assert len(engine.calls) > 1  # actually exercised multiple chunks
        scores = {c["name"]: c for c in result["session_summary"]["competency_scores"]}
        assert scores["Execution"]["score"] == 75


class TestAnalyzeTranscriptProfileThreading:
    """Regression coverage for threading an AssessmentProfile through to the
    prompt -- both the direct (un-chunked) and chunked paths must actually
    use the given profile, not silently fall back to generic."""

    def _cfg_for(self, engine, name) -> Config:
        register_engine(name, lambda acfg: engine)
        return Config(raw={"analysis": {"engine": name}})

    def test_direct_path_passes_the_profile_to_the_prompt(self):
        class _CapturingEngine(AnalysisEngine):
            def __init__(self):
                self.prompts: list[str] = []

            def run(self, prompt, on_progress=None):
                self.prompts.append(prompt)
                return json.dumps({
                    "qa_pairs": [],
                    "session_summary": {"top_strengths": [], "top_issues": [], "one_thing_to_practice_next": "", "confidence": 50},
                })

        engine = _CapturingEngine()
        cfg = self._cfg_for(engine, "fake_capturing_direct")
        profile = AssessmentProfile(competencies=["Leadership", "Execution"], role="Sales")

        analyze_transcript("[Interviewer] Hi\n[You] Hello", cfg, profile=profile)

        assert "Leadership" in engine.prompts[0]
        assert "Execution" in engine.prompts[0]
        assert "Sales" in engine.prompts[0]
        assert "Technical Expertise" not in engine.prompts[0]

    def test_chunked_path_passes_the_profile_to_every_chunk(self):
        good = json.dumps({
            "qa_pairs": [],
            "session_summary": {"top_strengths": [], "top_issues": [], "one_thing_to_practice_next": "", "confidence": 50},
        })
        engine = _FakeChunkedEngine(max_chars=120, responses=[good] * 20)
        cfg = self._cfg_for(engine, "fake_capturing_chunked")
        profile = AssessmentProfile(competencies=["Collaboration"], role="Design")

        transcript = "\n".join(f"[Interviewer] Question {i}?\n[You] Answer number {i}." for i in range(10))
        analyze_transcript(transcript, cfg, profile=profile)

        assert len(engine.calls) > 1
        for prompt in engine.calls:
            assert "Collaboration" in prompt
            assert "Design" in prompt
