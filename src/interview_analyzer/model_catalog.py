"""Periodically discovers which models each cloud analysis engine currently
offers, so the Settings tab's model dropdown doesn't go stale.

Covers every engine with a real, queryable model-listing API: Groq
("groq_api"), Anthropic ("anthropic_api"), and OpenAI ("openai_api"). Their
catalogs genuinely change over time -- this app's own Groq default was
silently retired from Groq's lineup once already (see GroqEngine's
docstring in analyzer.py), which is exactly the kind of drift this module
exists to catch automatically instead of requiring a code update every time
it happens.

Ollama is deliberately NOT covered here: model_setup.py's MODEL_CATALOG is a
small, hand-curated list of models this app has been specifically tuned/
tested against for local use, and unlike the three engines above, Ollama has
no equivalent stable, documented API for discovering "what's in the library"
remotely -- only for what's already pulled locally (see
model_setup.py:list_installed_models, a different question). That curated
list stays exactly as it was; nothing here changes it.

The refresh itself only ever runs in the background (watcher.py's
run_forever loop) or on an explicit Settings-tab button click -- never
inline in a request path -- and every failure is caught and logged rather
than raised, since a stale/unreachable model list must never block analysis
or crash the app.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import pathlib
from typing import Callable, Optional

import requests

from . import api_keys

logger = logging.getLogger(__name__)

# How often the background check in watcher.py re-queries each engine. A
# brand new install has no state file at all, which should_check() below
# treats the same as "overdue" -- so the very first check after installing
# the app happens on its first tick, and every ~15 days after that.
CHECK_INTERVAL_DAYS = 15

# Every engine with a real, queryable model-listing API -- see module
# docstring for why Ollama isn't one of these.
ENGINES_WITH_REMOTE_CATALOGS = ("groq_api", "anthropic_api", "openai_api")

# Which api_keys.py provider name backs each engine's key (matches the
# Settings tab's "Cloud API key" provider dropdown exactly -- see
# dashboard.py's _build_api_key_row).
_KEY_PROVIDER = {"groq_api": "groq", "anthropic_api": "anthropic_api", "openai_api": "openai_api"}

# Used only until an engine's very first successful refresh (e.g. offline,
# or no key saved yet) -- keeps the Settings dropdown usable in the
# meantime rather than empty. Matches each engine's own built-in default
# model in analyzer.py, so a never-refreshed dropdown still shows something
# that actually works.
FALLBACK_MODELS = {
    "groq_api": ["llama-3.3-70b-versatile"],
    "anthropic_api": ["claude-sonnet-5"],
    "openai_api": ["gpt-4o-mini"],
}

# Every provider's models-listing endpoint mixes in non-chat model families
# this app's JSON rubric analysis can't use (transcription, TTS, embeddings,
# image generation, moderation, ...) -- filtered out by a name-hint match
# rather than an explicit allowlist, so a brand new chat model shows up
# automatically without a code change. "orpheus"/"canopylabs" and "compound"
# were added after a real refresh against Groq's live API returned
# "canopylabs/orpheus-v1-english" (a TTS voice model with no "tts" in its
# id) and "groq/compound" (an agentic tool-use pipeline, not a plain
# single-shot chat model this app's response_format=json_object prompting
# is built around) alongside genuinely usable chat models.
_EXCLUDED_NAME_HINTS = (
    "whisper", "tts", "guard", "moderation", "embedding", "dall-e", "image",
    "audio", "realtime", "transcribe", "davinci-002", "babbage",
    "orpheus", "canopylabs", "compound",
)

# OpenAI's /v1/models list is dominated by non-chat model families with no
# other reliable signal to filter on -- unlike Groq/Anthropic, where the
# excluded-hints filter alone is enough, OpenAI's chat/reasoning models are
# additionally required to start with one of these prefixes.
_OPENAI_CHAT_PREFIXES = ("gpt-", "chatgpt-", "o1", "o3", "o4")


def _state_path(cfg) -> pathlib.Path:
    db_path = cfg.resolve(cfg.storage.get("db_path", "data/interviews.db"))
    return db_path.with_name(".model_catalog.json")


def load_state(cfg) -> dict:
    """The full, all-engines state dict: {engine: {"models": [...],
    "last_checked": iso, "first_checked": iso}}. Missing/corrupt state
    reads as {} (i.e. every engine reads as "never checked")."""
    path = _state_path(cfg)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Couldn't read the cached model catalog.", exc_info=True)
        return {}


def load_cached_models(cfg, engine: str) -> list[str]:
    """The model list from `engine`'s last successful refresh, or its
    FALLBACK_MODELS entry if there's never been one yet -- what the
    Settings tab's model dropdown should show for this engine."""
    models = load_state(cfg).get(engine, {}).get("models")
    return list(models) if models else list(FALLBACK_MODELS.get(engine, []))


def save_state(cfg, engine: str, models: list[str]) -> None:
    """Persists `models` for `engine` plus a fresh last_checked timestamp,
    preserving that engine's first_checked from any prior state (so it
    keeps recording "the first time this install ever checked THIS
    engine", independent of how many refreshes have happened since, and
    independent of every other engine's own history)."""
    state = load_state(cfg)
    now = dt.datetime.now().isoformat()
    existing_entry = state.get(engine, {})
    state[engine] = {
        "models": models,
        "last_checked": now,
        "first_checked": existing_entry.get("first_checked", now),
    }
    path = _state_path(cfg)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state), encoding="utf-8")
    except OSError:
        logger.warning("Couldn't save the model catalog.", exc_info=True)


def should_check(cfg, engine: str) -> bool:
    """True if a model-catalog refresh is due for `engine` -- either this
    install has never checked it before (no state entry yet, which is
    exactly what's true right after installing the app), or
    CHECK_INTERVAL_DAYS have passed since its last check. Each engine
    tracks its own schedule independently, since e.g. a Groq key might
    exist long before an OpenAI one ever gets added."""
    last_checked = load_state(cfg).get(engine, {}).get("last_checked")
    if not last_checked:
        return True
    try:
        checked_at = dt.datetime.fromisoformat(last_checked)
    except ValueError:
        return True
    return dt.datetime.now() - checked_at >= dt.timedelta(days=CHECK_INTERVAL_DAYS)


def any_check_due(cfg) -> bool:
    """True if AT LEAST ONE engine is due -- the cheap, no-network check
    watcher.py's poll loop calls every tick to decide whether to bother
    spawning a background refresh thread at all."""
    return any(should_check(cfg, engine) for engine in ENGINES_WITH_REMOTE_CATALOGS)


def _filter_chat_models(entries: list[dict], engine: str) -> list[str]:
    models = []
    for entry in entries:
        model_id = entry.get("id")
        if not model_id or entry.get("active") is False:
            continue
        lowered = model_id.lower()
        if any(hint in lowered for hint in _EXCLUDED_NAME_HINTS):
            continue
        if engine == "openai_api" and not lowered.startswith(_OPENAI_CHAT_PREFIXES):
            continue
        models.append(model_id)
    return sorted(models)


def _fetch_groq_models(api_key: str) -> list[str]:
    resp = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=15,
    )
    resp.raise_for_status()
    return _filter_chat_models(resp.json().get("data", []), "groq_api")


def _fetch_anthropic_models(api_key: str) -> list[str]:
    # Same auth headers as AnthropicEngine.run() in analyzer.py --
    # anthropic-version is required on every Anthropic API call, not just
    # this one.
    resp = requests.get(
        "https://api.anthropic.com/v1/models",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        timeout=15,
    )
    resp.raise_for_status()
    return _filter_chat_models(resp.json().get("data", []), "anthropic_api")


def _fetch_openai_models(api_key: str) -> list[str]:
    resp = requests.get(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=15,
    )
    resp.raise_for_status()
    return _filter_chat_models(resp.json().get("data", []), "openai_api")


_FETCHERS: dict[str, Callable[[str], list[str]]] = {
    "groq_api": _fetch_groq_models,
    "anthropic_api": _fetch_anthropic_models,
    "openai_api": _fetch_openai_models,
}


def fetch_latest_models(engine: str, api_key: str) -> list[str]:
    """Queries `engine`'s own model catalog directly and returns the
    chat-capable model ids, sorted. Raises on a network/HTTP failure --
    callers (refresh_one) decide how to handle that, same as everywhere
    else in this app that separates the raw fetch from its error
    handling."""
    return _FETCHERS[engine](api_key)


def _api_key_for_engine(cfg, engine: str) -> Optional[str]:
    """`engine`'s own stored key (see api_keys.py) if there is one;
    otherwise the shared INTERVIEW_ANALYZER_API_KEY-style env var, but ONLY
    when `engine` is the currently *active* analysis.engine -- that env var
    is a single, generic override meant for whichever engine is actually
    selected (see AnthropicEngine/OpenAIEngine/GroqEngine's own __init__ in
    analyzer.py), not a key for every provider at once, so applying it to
    an engine the user isn't even using would be a real correctness bug,
    not just a convenience."""
    provider = _KEY_PROVIDER[engine]
    stored = api_keys.load_key(provider)
    if stored:
        return stored
    if cfg.analysis.get("engine") == engine:
        env_var = cfg.analysis.get("cloud_api_key_env_var", "INTERVIEW_ANALYZER_API_KEY")
        return os.environ.get(env_var)
    return None


def refresh_one(cfg, engine: str) -> Optional[list[str]]:
    """Best-effort refresh for a single engine: returns the freshly
    fetched model list on success, or None if there's no key available for
    it yet or the request failed (logged, never raised). A failed attempt
    still bumps last_checked (re-saving whatever was cached before) so a
    persistent outage doesn't retry on every single watcher tick --
    CHECK_INTERVAL_DAYS naturally paces retries the same as a successful
    check would, and the Settings tab's "Refresh model list" button
    remains available any time for an on-demand retry. A missing key does
    NOT bump last_checked, so adding one later is picked up promptly
    rather than waiting out the full interval."""
    api_key = _api_key_for_engine(cfg, engine)
    if not api_key:
        return None
    try:
        models = fetch_latest_models(engine, api_key)
    except Exception:  # noqa: BLE001
        logger.warning("Couldn't refresh the %s model catalog; keeping the cached list.", engine, exc_info=True)
        save_state(cfg, engine, load_cached_models(cfg, engine))
        return None
    save_state(cfg, engine, models)
    logger.info("Refreshed the %s model catalog: %s model(s) available.", engine, len(models))
    return models


def refresh_due_catalogs(cfg) -> dict[str, Optional[list[str]]]:
    """Refreshes every engine that's actually due (see should_check),
    skipping the rest -- called from watcher.py's background thread once
    any_check_due() said at least one engine needed it. Returns
    {engine: models-or-None} for whichever engines were attempted."""
    results: dict[str, Optional[list[str]]] = {}
    for engine in ENGINES_WITH_REMOTE_CATALOGS:
        if should_check(cfg, engine):
            results[engine] = refresh_one(cfg, engine)
    return results
