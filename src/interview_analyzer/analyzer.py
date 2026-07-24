"""Runs the rubric-based analysis over a transcript using a pluggable
AnalysisEngine (see engines.py). Ships with four built-in engines:

  - "ollama"        (default) - free, fully local. Best free-tier quality
                       tradeoff is Llama 3.1 8B for speed, or Qwen2.5 14B
                       for noticeably better reasoning if your machine can
                       run it (see config.yaml comment).
  - "groq_api"       - bring your own (free, no-credit-card) Groq API key.
                       Runs open models on Groq's own fast hardware --
                       much faster than local Ollama on a CPU-only
                       machine, but your transcript leaves your machine.
  - "anthropic_api"  - bring your own Anthropic API key (NOT the same as a
                       claude.ai subscription - this is billed separately
                       per token via console.anthropic.com).
  - "openai_api"     - bring your own OpenAI API key.

To add a different engine entirely, see engines.py / docs/using_cloud_apis.md.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from typing import Callable, Optional

import requests

from . import api_keys
from .config_loader import Config
from .engines import AnalysisEngine, get_engine, register_engine
from .model_setup import ensure_ollama_running
from .profiles import AssessmentProfile, GENERIC_PROFILE
from .report import _stringify
from .rubric import RESULT_JSON_SCHEMA, build_prompt, split_transcript_for_chunked_analysis

logger = logging.getLogger(__name__)


class OllamaEngine(AnalysisEngine):
    def __init__(self, acfg: dict):
        self.host = acfg.get("ollama_host", "http://localhost:11434")
        self.model = acfg.get("llm_model", "llama3.1:8b")

    def run(self, prompt: str, on_progress: Optional[Callable[[float], None]] = None) -> str:
        # Ollama is a separate local background service that doesn't
        # auto-start with the OS by default -- without this, a machine
        # reboot (or Ollama just not being launched yet) turns into a
        # confusing raw ConnectionError here instead of just working.
        if not ensure_ollama_running(self.host):
            raise RuntimeError(
                f"Ollama isn't running and couldn't be started automatically at {self.host}. "
                "Install it from https://ollama.com, or start it manually, then try again."
            )

        # A full JSON Schema (not just the string "json") -- constrains
        # Ollama's decoding to actually match the rubric's shape, not just
        # produce *some* valid JSON. See rubric.py's RESULT_JSON_SCHEMA
        # docstring for why: a plain "json" format let the model return a
        # syntactically valid but completely unrelated object on a real
        # long transcript, producing a blank report with no clear error.
        # Constrained decoding is measurably slower than free-form
        # generation (grammar-checked per token), hence the longer timeout
        # than a plain generation call would need.
        if on_progress is None:
            resp = requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False, "format": RESULT_JSON_SCHEMA},
                timeout=1200,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")

        # Streaming mode: Ollama's rubric response is fixed-shape JSON, so
        # the number of tokens generated so far (eval_count) against a rough
        # expected-size estimate makes a reasonable progress signal -- an
        # estimate, not exact, same spirit as transcription's real
        # segment-timestamp progress but for a single LLM call there's no
        # exact total to report against ahead of time.
        expected_tokens = max(300, prompt.count("?") * 180 + 150)
        chunks = []
        resp = requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": True, "format": RESULT_JSON_SCHEMA},
            timeout=1200,
            stream=True,
        )
        resp.raise_for_status()
        try:
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunks.append(data.get("response", ""))
                eval_count = data.get("eval_count")
                if eval_count is not None:
                    on_progress(min(eval_count / expected_tokens, 0.97))
                if data.get("done"):
                    on_progress(1.0)
        finally:
            resp.close()
        return "".join(chunks)


class AnthropicEngine(AnalysisEngine):
    """Bring-your-own-API-key Anthropic engine. Requires a real API key from
    console.anthropic.com -- a claude.ai Pro/Max subscription does not grant
    API access, they're billed and provisioned separately.
    """

    def __init__(self, acfg: dict):
        env_var = acfg.get("cloud_api_key_env_var", "INTERVIEW_ANALYZER_API_KEY")
        # the environment variable (the original mechanism) always wins if
        # set; otherwise fall back to a key saved via the Settings tab's
        # "Cloud API key" section (see api_keys.py -- encrypted at rest,
        # never plaintext)
        self.api_key = os.environ.get(env_var) or api_keys.load_key("anthropic_api")
        if not self.api_key:
            raise RuntimeError(
                f"No Anthropic API key found. Set it in the Settings tab's \"Cloud API key\" "
                f"section, or set the {env_var} environment variable, to a real API key from "
                "console.anthropic.com (not your claude.ai subscription -- that doesn't grant API access)."
            )
        self.model = acfg.get("llm_model", "claude-sonnet-5")

    def run(self, prompt: str, on_progress: Optional[Callable[[float], None]] = None) -> str:
        # No incremental progress signal for this engine -- a single
        # non-streaming request; on_progress is accepted for interface
        # consistency but left unused, same as any engine that has nothing
        # partial to report.
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 4000,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=600,
        )
        resp.raise_for_status()
        content = resp.json()["content"]
        return "".join(b["text"] for b in content if b.get("type") == "text")


class OpenAIEngine(AnalysisEngine):
    def __init__(self, acfg: dict):
        env_var = acfg.get("cloud_api_key_env_var", "INTERVIEW_ANALYZER_API_KEY")
        self.api_key = os.environ.get(env_var) or api_keys.load_key("openai_api")
        if not self.api_key:
            raise RuntimeError(
                f"No OpenAI API key found. Set it in the Settings tab's \"Cloud API key\" "
                f"section, or set the {env_var} environment variable, to a real API key from "
                "platform.openai.com (not your ChatGPT subscription -- that doesn't grant API access)."
            )
        self.model = acfg.get("llm_model", "gpt-4o-mini")

    def run(self, prompt: str, on_progress: Optional[Callable[[float], None]] = None) -> str:
        # No incremental progress signal for this engine -- see AnthropicEngine.run.
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            },
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class GroqEngine(AnalysisEngine):
    """Bring-your-own-API-key Groq engine. Unlike the other two cloud
    engines, Groq's free tier (console.groq.com/keys, no credit card) is
    genuinely usable, not just a trial -- it runs open models (Llama,
    GPT-OSS, Qwen) on Groq's own fast hardware. The real tradeoff versus
    Ollama isn't cost, it's privacy: your transcript leaves your machine
    and goes to Groq's servers.

    Defaults to llama-3.3-70b-versatile. The previous default,
    meta-llama/llama-4-scout-17b-16e-instruct, was removed from Groq's
    catalog at some point after it was first chosen here -- reproduced
    directly: Groq now returns a 404 for it, which is exactly the "not
    found" error this engine raises a clearer message for below.
    llama-3.3-70b-versatile isn't a "reasoning" model (unlike GPT-OSS, which
    burns real completion tokens on internal chain-of-thought before
    answering), but Groq currently only lets GPT-OSS models use the
    response_format=json_schema structured-output mode at all -- every
    other model, including this one, gets a hard 400 error if you try
    (reproduced directly). So this uses plain response_format=json_object
    (broadly supported, verified directly against a real transcript) plus
    the rubric's own shape validation (_has_the_expected_shape) as the
    safety net, same spirit as the schema-based approach before it.

    llama-3.3-70b-versatile's free tier is also a much tighter 12,000
    tokens/minute (vs. llama-4-scout's 30,000) and a full ~1-hour interview
    transcript needs far more than that in a single request -- reproduced
    directly: a real ~64K-character transcript needed ~23,700 prompt
    tokens, over the model's entire per-minute budget by itself, a 413
    "rate_limit_exceeded" no retry could ever fix (a single request that
    large will never fit, regardless of timing). This engine opts into
    analyze_transcript()'s chunked-analysis path (splitting the transcript,
    analyzing each piece, then merging the results) via
    max_transcript_chars_per_request below, rather than failing outright on
    any interview long enough to exceed the free-tier budget.

    Even chunked, back-to-back requests can still trip a *transient* 429
    (the account's rolling per-minute usage from recent chunks, not this
    request alone, pushes it over budget) -- reproduced directly. Two
    things fix that: max_tokens is sized to the prompt itself rather than a
    flat 8000 (Groq's admission check reserves the *entire* requested
    max_tokens against the budget up front, confirmed directly -- a small
    ~9000-character chunk was rejected specifically because "Requested"
    included a full 8000-token reservation on top of its own ~3500 prompt
    tokens, even though its actual usage afterward was only ~2800 tokens);
    and run() retries a 429 after waiting however long Groq's own
    Retry-After header says, rather than guessing our own pacing.
    """

    def __init__(self, acfg: dict):
        env_var = acfg.get("cloud_api_key_env_var", "INTERVIEW_ANALYZER_API_KEY")
        self.api_key = os.environ.get(env_var) or api_keys.load_key("groq")
        if not self.api_key:
            raise RuntimeError(
                f"No Groq API key found. Get a free one (no credit card) at "
                f"https://console.groq.com/keys, then set it in the Settings tab's "
                f"\"Cloud API key\" section, or set the {env_var} environment variable."
            )
        self.model = acfg.get("llm_model", "llama-3.3-70b-versatile")

        # Conservative per-request input budget (chars) so a long
        # transcript gets chunked instead of failing outright -- well
        # under the free-tier tokens/minute cap even including the
        # completion. Config-overridable (analysis.groq_chunk_max_chars) in
        # case you're on a higher-limit Groq tier or a different model.
        self.max_transcript_chars_per_request = acfg.get("groq_chunk_max_chars", 9000)

    def run(self, prompt: str, on_progress: Optional[Callable[[float], None]] = None) -> str:
        # No incremental progress signal for this engine -- see AnthropicEngine.run.
        #
        # max_tokens scales with the prompt's own size rather than a flat
        # 8000 -- see the class docstring for why an unconditionally
        # generous reservation works against a small chunk. A floor of
        # 1500 still covers a short transcript's full rubric response; the
        # 8000 ceiling preserves the old generous headroom for a large,
        # un-chunked transcript with many questions.
        max_tokens = max(1500, min(8000, len(prompt) // 3))

        max_retries = 4
        for attempt in range(max_retries + 1):
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "content-type": "application/json"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "response_format": {"type": "json_object"},
                },
                timeout=600,
            )
            if resp.status_code == 404:
                # A real, confusing failure mode: switching analysis.engine
                # to "groq_api" doesn't reset analysis.llm_model, which is
                # shared across all four engines -- a model name left over
                # from Ollama (e.g. "llama3.1:8b"), or a Groq model that's
                # since been deprecated (this happened to our own previous
                # default), isn't a valid Groq model id, and Groq's plain
                # 404 for that gives no hint why. Point directly at the
                # likely cause instead of leaving a bare HTTPError.
                raise RuntimeError(
                    f"Groq returned \"not found\" for model \"{self.model}\" -- likely not a valid (or no "
                    f"longer available) Groq model id (e.g. a name like \"llama3.1:8b\" is Ollama's naming, "
                    f"not Groq's). Set analysis.llm_model to a real, current Groq model, e.g. "
                    f"\"llama-3.3-70b-versatile\" (see console.groq.com/docs/models for the full, current list)."
                )
            if resp.status_code == 429 and attempt < max_retries:
                # A transient rolling-window rate limit (recent usage from
                # other requests, not this one alone, pushed the account
                # over budget this minute) -- Groq's own Retry-After header
                # says exactly how long until there's room again, which is
                # more reliable than guessing our own pacing.
                wait_seconds = float(resp.headers.get("retry-after", 10)) + 1
                logger.info(
                    "Groq rate limit hit; waiting %.1fs before retrying (attempt %s/%s).",
                    wait_seconds, attempt + 1, max_retries,
                )
                time.sleep(wait_seconds)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]


register_engine("ollama", lambda acfg: OllamaEngine(acfg))
register_engine("anthropic_api", lambda acfg: AnthropicEngine(acfg))
register_engine("openai_api", lambda acfg: OpenAIEngine(acfg))
register_engine("groq_api", lambda acfg: GroqEngine(acfg))


def _has_the_expected_shape(parsed: dict) -> bool:
    """True if `parsed` actually looks like the rubric's qa_pairs +
    session_summary schema, not just any valid JSON. A real failure mode
    (reproduced on a real interview: a long transcript against
    llama3.1:8b) is the model returning well-formed JSON that ignores the
    requested schema entirely -- e.g. a generic {"title": ..., "topics":
    [...]} object -- which used to sail through as a "successful" analysis
    and silently produce a blank report with no way to reprocess it (see
    dashboard.py's can_reprocess/history_status_label, which key off this
    same parse_error flag)."""
    return isinstance(parsed.get("qa_pairs"), list) and isinstance(parsed.get("session_summary"), dict)


def _run_engine(engine: AnalysisEngine, prompt: str, on_progress: Optional[Callable[[float], None]]) -> str:
    try:
        return engine.run(prompt, on_progress=on_progress)
    except TypeError:
        # a third-party engine registered before on_progress existed, or
        # that simply doesn't implement it -- fall back to the old call
        # shape rather than breaking custom engines (see engines.py)
        return engine.run(prompt)


def _parse_and_validate(raw: str, context: str = "") -> Optional[dict]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Analyzer returned non-JSON output%s.", context)
        return None
    if not isinstance(parsed, dict) or not _has_the_expected_shape(parsed):
        logger.warning(
            "Analyzer returned valid JSON but not the expected qa_pairs/session_summary shape%s.", context
        )
        return None
    return parsed


def _merge_competency_scores(parsed_chunks: list[dict]) -> list[dict]:
    """Averages each competency's score (by name) across every chunk that
    reported it, and joins each chunk's remark into one combined note --
    one entry per competency actually seen, in first-seen order (matches
    the profile's own competency order, since every chunk was asked to
    score the same fixed list -- see rubric.py's build_prompt)."""
    scores_by_name: dict[str, list[int]] = {}
    remarks_by_name: dict[str, list[str]] = {}
    order: list[str] = []

    for parsed in parsed_chunks:
        for entry in (parsed.get("session_summary") or {}).get("competency_scores") or []:
            if not isinstance(entry, dict):
                continue
            name = _stringify(entry.get("name", ""))
            if not name:
                continue
            if name not in scores_by_name:
                order.append(name)
                scores_by_name[name] = []
                remarks_by_name[name] = []
            score = entry.get("score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                scores_by_name[name].append(score)
            remark = entry.get("remark")
            if remark:
                remarks_by_name[name].append(_stringify(remark))

    return [
        {
            "name": name,
            "score": round(sum(scores_by_name[name]) / len(scores_by_name[name])) if scores_by_name[name] else 0,
            "remark": " ".join(remarks_by_name[name]),
        }
        for name in order
    ]


def _merge_chunk_analyses(parsed_chunks: list[dict]) -> dict:
    """Combines each transcript chunk's own qa_pairs/session_summary into a
    single result with the same shape a normal (unchunked) analysis
    returns, so every downstream consumer (confidence calibration, the
    report, the infographic) needs no chunking-awareness at all."""
    all_qa_pairs: list = []
    strengths_counter: Counter = Counter()
    issues_counter: Counter = Counter()
    practice_candidates: list[str] = []
    confidences: list[float] = []
    hire_recommendations: list[dict] = []

    for parsed in parsed_chunks:
        all_qa_pairs.extend(parsed.get("qa_pairs") or [])
        summary = parsed.get("session_summary") or {}
        for strength in summary.get("top_strengths") or []:
            strengths_counter[_stringify(strength)] += 1
        for issue in summary.get("top_issues") or []:
            issues_counter[_stringify(issue)] += 1
        practice = summary.get("one_thing_to_practice_next")
        if practice:
            practice_candidates.append(_stringify(practice))
        confidence = summary.get("confidence")
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            confidences.append(confidence)
        hire_recommendation = summary.get("hire_recommendation")
        if isinstance(hire_recommendation, dict):
            hire_recommendations.append(hire_recommendation)

    return {
        "qa_pairs": all_qa_pairs,
        "session_summary": {
            "top_strengths": [s for s, _ in strengths_counter.most_common(3)],
            "top_issues": [s for s, _ in issues_counter.most_common(3)],
            # the last chunk saw the most of the conversation's arc, so its
            # suggestion is the best-informed one to surface
            "one_thing_to_practice_next": practice_candidates[-1] if practice_candidates else "",
            # averaged rather than min/max -- each chunk's confidence
            # already reflects that chunk's own transcript quality, so the
            # mean is the fairest estimate for the whole interview
            "confidence": round(sum(confidences) / len(confidences)) if confidences else 0,
            "competency_scores": _merge_competency_scores(parsed_chunks),
            # same "last chunk wins" reasoning as one_thing_to_practice_next
            # -- the final chunk had the most complete view of the
            # conversation's arc when it made its hire-recommendation call
            "hire_recommendation": hire_recommendations[-1] if hire_recommendations else {"level": "", "rationale": ""},
        },
    }


def _analyze_in_chunks(
    transcript: str,
    engine: AnalysisEngine,
    max_chars: int,
    on_progress: Optional[Callable[[float], None]],
    profile: AssessmentProfile,
    calibration_notes: str,
) -> dict:
    chunks = split_transcript_for_chunked_analysis(transcript, max_chars)
    parsed_chunks: list[dict] = []

    for i, chunk in enumerate(chunks):
        # no pre-emptive pacing here -- GroqEngine.run() itself retries a
        # transient rate limit using Groq's own Retry-After header, which
        # adapts to real account usage rather than guessing a fixed delay
        raw = _run_engine(engine, build_prompt(chunk, profile=profile, calibration_notes=calibration_notes), None)
        parsed = _parse_and_validate(raw, context=f" for chunk {i + 1}/{len(chunks)}")
        if parsed is not None:
            parsed_chunks.append(parsed)
        if on_progress:
            on_progress((i + 1) / len(chunks))

    if not parsed_chunks:
        return {"raw": "", "parse_error": True}

    return _merge_chunk_analyses(parsed_chunks)


def analyze_transcript(
    transcript: str,
    cfg: Config,
    on_progress: Optional[Callable[[float], None]] = None,
    profile: AssessmentProfile = GENERIC_PROFILE,
    calibration_notes: str = "",
) -> dict:
    acfg = cfg.analysis
    engine_name = acfg.get("engine", "ollama")
    engine = get_engine(engine_name, acfg)

    max_chars = getattr(engine, "max_transcript_chars_per_request", None)
    if max_chars and len(transcript) > max_chars:
        return _analyze_in_chunks(transcript, engine, max_chars, on_progress, profile, calibration_notes)

    raw = _run_engine(engine, build_prompt(transcript, profile=profile, calibration_notes=calibration_notes), on_progress)
    parsed = _parse_and_validate(raw)
    if parsed is None:
        return {"raw": raw, "parse_error": True}
    return parsed
