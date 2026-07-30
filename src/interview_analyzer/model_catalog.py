"""Periodically discovers which chat models Groq currently offers on its
free tier, so the Settings tab's model dropdown doesn't go stale.

Groq's catalog genuinely changes over time -- this app's own default model
was silently removed from Groq's lineup once already (see GroqEngine's
docstring in analyzer.py), which is exactly the kind of drift this module
exists to catch automatically instead of requiring a code update every time
it happens.

Unlike Ollama's MODEL_CATALOG (model_setup.py), which is a small, hand-
curated list of models this app has been specifically tuned/tested against
for local use, Groq's catalog is queried live from Groq's own API -- there's
no meaningful curation to do (the whole point of Groq's free tier is that
everything in it is usable, just rate-limited), so "latest free models" here
just means "whatever Groq's /models endpoint currently returns", filtered to
drop the non-chat model families (Whisper transcription, TTS, moderation)
mixed into that same endpoint.

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
from typing import Optional

import requests

from . import api_keys

logger = logging.getLogger(__name__)

GROQ_MODELS_ENDPOINT = "https://api.groq.com/openai/v1/models"

# How often the background check in watcher.py re-queries Groq. A brand new
# install has no state file at all, which should_check() below treats the
# same as "overdue" -- so the very first check after installing the app
# happens on its first tick, and every ~15 days after that.
CHECK_INTERVAL_DAYS = 15

# Used only until the very first successful refresh (e.g. offline, or no
# Groq key saved yet) -- keeps the Settings dropdown usable in the
# meantime rather than empty.
FALLBACK_GROQ_MODELS = ["llama-3.3-70b-versatile"]

# Groq's /models endpoint lists every model on the account, including
# non-chat families that can't serve this app's JSON rubric analysis --
# filtered out by a name-hint match rather than an explicit allowlist, so a
# brand new chat model shows up automatically without a code change.
_EXCLUDED_NAME_HINTS = ("whisper", "tts", "guard", "moderation")


def _state_path(cfg) -> pathlib.Path:
    db_path = cfg.resolve(cfg.storage.get("db_path", "data/interviews.db"))
    return db_path.with_name(".groq_model_catalog.json")


def load_state(cfg) -> dict:
    path = _state_path(cfg)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Couldn't read the cached Groq model catalog.", exc_info=True)
        return {}


def load_cached_models(cfg) -> list[str]:
    """The model list from the last successful refresh, or
    FALLBACK_GROQ_MODELS if there's never been one yet -- what the Settings
    tab's model dropdown should show for analysis.engine == "groq_api"."""
    models = load_state(cfg).get("models")
    return list(models) if models else list(FALLBACK_GROQ_MODELS)


def save_state(cfg, models: list[str]) -> None:
    """Persists `models` plus a fresh last_checked timestamp, preserving
    first_checked from any prior state (so it keeps recording "the first
    time this install ever checked", independent of how many refreshes
    have happened since)."""
    existing = load_state(cfg)
    now = dt.datetime.now().isoformat()
    state = {
        "models": models,
        "last_checked": now,
        "first_checked": existing.get("first_checked", now),
    }
    path = _state_path(cfg)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state), encoding="utf-8")
    except OSError:
        logger.warning("Couldn't save the Groq model catalog.", exc_info=True)


def should_check(cfg) -> bool:
    """True if a Groq model-catalog refresh is due -- either this install
    has never checked before (no state file yet, which is exactly what's
    true right after installing the app), or CHECK_INTERVAL_DAYS have
    passed since the last check."""
    last_checked = load_state(cfg).get("last_checked")
    if not last_checked:
        return True
    try:
        checked_at = dt.datetime.fromisoformat(last_checked)
    except ValueError:
        return True
    return dt.datetime.now() - checked_at >= dt.timedelta(days=CHECK_INTERVAL_DAYS)


def fetch_latest_groq_models(api_key: str) -> list[str]:
    """Queries Groq's own model catalog directly (same Bearer-token
    pattern as GroqEngine.run() in analyzer.py) and returns the chat-
    capable model ids, sorted. Raises on a network/HTTP failure -- callers
    (refresh_model_catalog) decide how to handle that, same as everywhere
    else in this app that separates the raw fetch from its error handling."""
    resp = requests.get(
        GROQ_MODELS_ENDPOINT,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=15,
    )
    resp.raise_for_status()
    entries = resp.json().get("data", [])
    models = []
    for entry in entries:
        model_id = entry.get("id")
        if not model_id or entry.get("active") is False:
            continue
        if any(hint in model_id.lower() for hint in _EXCLUDED_NAME_HINTS):
            continue
        models.append(model_id)
    return sorted(models)


def _groq_api_key(cfg) -> Optional[str]:
    env_var = cfg.analysis.get("cloud_api_key_env_var", "INTERVIEW_ANALYZER_API_KEY")
    return os.environ.get(env_var) or api_keys.load_key("groq")


def refresh_model_catalog(cfg) -> Optional[list[str]]:
    """Best-effort refresh: returns the freshly fetched model list on
    success, or None if there's no Groq key saved yet or the request
    failed (logged, never raised). A failed attempt still bumps
    last_checked (re-saving whatever was cached before) so a persistent
    outage doesn't retry on every single watcher tick -- CHECK_INTERVAL_DAYS
    naturally paces retries the same as a successful check would, and the
    Settings tab's "Refresh model list" button remains available any time
    for an on-demand retry. A missing key does NOT bump last_checked, so
    adding one later is picked up promptly rather than waiting out the
    full interval."""
    api_key = _groq_api_key(cfg)
    if not api_key:
        return None
    try:
        models = fetch_latest_groq_models(api_key)
    except Exception:  # noqa: BLE001
        logger.warning("Couldn't refresh the Groq model catalog; keeping the cached list.", exc_info=True)
        save_state(cfg, load_cached_models(cfg))
        return None
    save_state(cfg, models)
    logger.info("Refreshed the Groq model catalog: %s model(s) available.", len(models))
    return models
