"""Tests for model_catalog.py: the periodic per-engine model-catalog
refresh (check every CHECK_INTERVAL_DAYS days, and on the very first tick
after a fresh install since there's no state file yet) for every engine
with a real remote model-listing API (Groq, Anthropic, OpenAI). Network
calls are mocked throughout -- this covers the caching/scheduling/
filtering logic, not each provider's actual API."""
from __future__ import annotations

import datetime as dt
import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from interview_analyzer.config_loader import Config
from interview_analyzer.model_catalog import (
    CHECK_INTERVAL_DAYS,
    ENGINES_WITH_REMOTE_CATALOGS,
    FALLBACK_MODELS,
    any_check_due,
    fetch_latest_models,
    load_cached_models,
    load_state,
    refresh_due_catalogs,
    refresh_one,
    save_state,
    should_check,
)


def _cfg(tmp_path) -> Config:
    return Config(raw={
        "storage": {"db_path": str(tmp_path / "interviews.db")},
        "analysis": {"engine": "groq_api", "cloud_api_key_env_var": "INTERVIEW_ANALYZER_API_KEY"},
    })


def _state_path(tmp_path):
    return (tmp_path / "interviews.db").with_name(".model_catalog.json")


def _write_state(tmp_path, state: dict) -> None:
    path = _state_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state), encoding="utf-8")


def _openai_style_response(ids: list[str], inactive: list[str] = ()) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "data": (
            [{"id": i, "active": True} for i in ids]
            + [{"id": i, "active": False} for i in inactive]
        )
    }
    return resp


class TestShouldCheck:
    @pytest.mark.parametrize("engine", ENGINES_WITH_REMOTE_CATALOGS)
    def test_true_when_never_checked_before(self, tmp_path, engine):
        """Covers both "fresh install" and "no state entry for this engine
        yet" -- the same condition, since a brand new install has never
        written one for ANY engine."""
        assert should_check(_cfg(tmp_path), engine) is True

    def test_false_right_after_a_check(self, tmp_path):
        cfg = _cfg(tmp_path)
        save_state(cfg, "groq_api", ["llama-3.3-70b-versatile"])
        assert should_check(cfg, "groq_api") is False

    def test_each_engine_tracks_its_own_schedule_independently(self, tmp_path):
        cfg = _cfg(tmp_path)
        save_state(cfg, "groq_api", ["llama-3.3-70b-versatile"])
        assert should_check(cfg, "groq_api") is False
        assert should_check(cfg, "anthropic_api") is True
        assert should_check(cfg, "openai_api") is True

    def test_true_once_the_interval_has_elapsed(self, tmp_path):
        cfg = _cfg(tmp_path)
        save_state(cfg, "groq_api", ["llama-3.3-70b-versatile"])
        state = load_state(cfg)
        stale = dt.datetime.now() - dt.timedelta(days=CHECK_INTERVAL_DAYS, hours=1)
        state["groq_api"]["last_checked"] = stale.isoformat()
        _write_state(tmp_path, state)

        assert should_check(cfg, "groq_api") is True

    def test_false_just_under_the_interval(self, tmp_path):
        cfg = _cfg(tmp_path)
        save_state(cfg, "groq_api", ["llama-3.3-70b-versatile"])
        state = load_state(cfg)
        recent = dt.datetime.now() - dt.timedelta(days=CHECK_INTERVAL_DAYS - 1)
        state["groq_api"]["last_checked"] = recent.isoformat()
        _write_state(tmp_path, state)

        assert should_check(cfg, "groq_api") is False

    def test_true_on_a_corrupt_state_file(self, tmp_path):
        cfg = _cfg(tmp_path)
        path = _state_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")

        assert should_check(cfg, "groq_api") is True


class TestAnyCheckDue:
    def test_true_when_nothing_has_ever_been_checked(self, tmp_path):
        assert any_check_due(_cfg(tmp_path)) is True

    def test_false_once_every_engine_has_been_recently_checked(self, tmp_path):
        cfg = _cfg(tmp_path)
        for engine in ENGINES_WITH_REMOTE_CATALOGS:
            save_state(cfg, engine, ["some-model"])

        assert any_check_due(cfg) is False

    def test_true_if_even_one_engine_is_overdue(self, tmp_path):
        cfg = _cfg(tmp_path)
        for engine in ENGINES_WITH_REMOTE_CATALOGS:
            save_state(cfg, engine, ["some-model"])
        state = load_state(cfg)
        stale = dt.datetime.now() - dt.timedelta(days=CHECK_INTERVAL_DAYS, hours=1)
        state["openai_api"]["last_checked"] = stale.isoformat()
        _write_state(tmp_path, state)

        assert any_check_due(cfg) is True


class TestStateRoundTrip:
    @pytest.mark.parametrize("engine", ENGINES_WITH_REMOTE_CATALOGS)
    def test_load_cached_models_falls_back_when_nothing_saved_yet(self, tmp_path, engine):
        assert load_cached_models(_cfg(tmp_path), engine) == FALLBACK_MODELS[engine]

    def test_save_then_load_round_trips(self, tmp_path):
        cfg = _cfg(tmp_path)
        save_state(cfg, "groq_api", ["model-a", "model-b"])
        assert load_cached_models(cfg, "groq_api") == ["model-a", "model-b"]

    def test_engines_are_stored_independently(self, tmp_path):
        cfg = _cfg(tmp_path)
        save_state(cfg, "groq_api", ["groq-model"])
        save_state(cfg, "anthropic_api", ["claude-model"])

        assert load_cached_models(cfg, "groq_api") == ["groq-model"]
        assert load_cached_models(cfg, "anthropic_api") == ["claude-model"]
        assert load_cached_models(cfg, "openai_api") == FALLBACK_MODELS["openai_api"]

    def test_first_checked_is_preserved_across_repeated_saves(self, tmp_path):
        cfg = _cfg(tmp_path)
        save_state(cfg, "groq_api", ["model-a"])
        first_checked = load_state(cfg)["groq_api"]["first_checked"]

        save_state(cfg, "groq_api", ["model-a", "model-b"])

        entry = load_state(cfg)["groq_api"]
        assert entry["first_checked"] == first_checked
        assert entry["models"] == ["model-a", "model-b"]


class TestFetchLatestModels:
    def test_groq_returns_sorted_active_chat_model_ids(self):
        resp = _openai_style_response(["llama-3.3-70b-versatile", "gpt-oss-20b"])
        with patch("interview_analyzer.model_catalog.requests.get", return_value=resp) as mock_get:
            models = fetch_latest_models("groq_api", "fake-key")

        assert models == ["gpt-oss-20b", "llama-3.3-70b-versatile"]
        assert mock_get.call_args.kwargs["headers"]["Authorization"] == "Bearer fake-key"

    def test_groq_excludes_non_chat_model_families(self):
        resp = _openai_style_response([
            "llama-3.3-70b-versatile", "whisper-large-v3", "distil-whisper-large-v3-en",
            "playai-tts", "llama-guard-3-8b",
        ])
        with patch("interview_analyzer.model_catalog.requests.get", return_value=resp):
            models = fetch_latest_models("groq_api", "fake-key")

        assert models == ["llama-3.3-70b-versatile"]

    def test_groq_excludes_orpheus_tts_and_compound_pipelines(self):
        """Regression coverage for a real response from Groq's live API:
        canopylabs/orpheus-* (TTS voice models with no "tts" in their id)
        and groq/compound(-mini) (agentic tool-use pipelines, not a plain
        chat model) both slipped through the original hint list."""
        resp = _openai_style_response([
            "llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b",
            "allam-2-7b", "canopylabs/orpheus-arabic-saudi", "canopylabs/orpheus-v1-english",
            "groq/compound", "groq/compound-mini",
        ])
        with patch("interview_analyzer.model_catalog.requests.get", return_value=resp):
            models = fetch_latest_models("groq_api", "fake-key")

        assert models == [
            "allam-2-7b", "llama-3.1-8b-instant", "llama-3.3-70b-versatile",
            "openai/gpt-oss-120b", "openai/gpt-oss-20b",
        ]

    def test_groq_excludes_inactive_models(self):
        resp = _openai_style_response(["llama-3.3-70b-versatile"], inactive=["some-deprecated-model"])
        with patch("interview_analyzer.model_catalog.requests.get", return_value=resp):
            models = fetch_latest_models("groq_api", "fake-key")

        assert models == ["llama-3.3-70b-versatile"]

    def test_anthropic_uses_x_api_key_and_version_headers(self):
        resp = _openai_style_response(["claude-sonnet-5", "claude-opus-5"])
        with patch("interview_analyzer.model_catalog.requests.get", return_value=resp) as mock_get:
            models = fetch_latest_models("anthropic_api", "fake-key")

        assert models == ["claude-opus-5", "claude-sonnet-5"]
        assert mock_get.call_args.kwargs["headers"]["x-api-key"] == "fake-key"
        assert mock_get.call_args.kwargs["headers"]["anthropic-version"] == "2023-06-01"
        assert "api.anthropic.com" in mock_get.call_args.args[0]

    def test_openai_keeps_only_chat_and_reasoning_prefixed_models(self):
        resp = _openai_style_response([
            "gpt-4o-mini", "gpt-4o", "o3-mini", "text-embedding-3-small",
            "whisper-1", "dall-e-3", "tts-1", "text-moderation-latest",
        ])
        with patch("interview_analyzer.model_catalog.requests.get", return_value=resp) as mock_get:
            models = fetch_latest_models("openai_api", "fake-key")

        assert models == ["gpt-4o", "gpt-4o-mini", "o3-mini"]
        assert mock_get.call_args.kwargs["headers"]["Authorization"] == "Bearer fake-key"
        assert "api.openai.com" in mock_get.call_args.args[0]

    def test_openai_excludes_transcribe_and_tts_variants_despite_gpt_prefix(self):
        resp = _openai_style_response(["gpt-4o-mini", "gpt-4o-transcribe", "gpt-4o-mini-tts"])
        with patch("interview_analyzer.model_catalog.requests.get", return_value=resp):
            models = fetch_latest_models("openai_api", "fake-key")

        assert models == ["gpt-4o-mini"]

    def test_raises_on_an_http_error(self):
        with patch("interview_analyzer.model_catalog.requests.get", side_effect=requests.RequestException("down")):
            with pytest.raises(requests.RequestException):
                fetch_latest_models("groq_api", "fake-key")


class TestRefreshOne:
    def test_returns_none_and_saves_nothing_without_a_key(self, tmp_path):
        cfg = _cfg(tmp_path)
        with patch("interview_analyzer.model_catalog.api_keys.load_key", return_value=None), \
             patch.dict("os.environ", {}, clear=True):
            result = refresh_one(cfg, "groq_api")

        assert result is None
        assert load_state(cfg) == {}

    def test_success_saves_and_returns_the_fetched_models(self, tmp_path):
        cfg = _cfg(tmp_path)
        resp = _openai_style_response(["llama-3.3-70b-versatile", "gpt-oss-20b"])
        with patch("interview_analyzer.model_catalog.api_keys.load_key", return_value="fake-key"), \
             patch("interview_analyzer.model_catalog.requests.get", return_value=resp):
            result = refresh_one(cfg, "groq_api")

        assert result == ["gpt-oss-20b", "llama-3.3-70b-versatile"]
        assert load_cached_models(cfg, "groq_api") == ["gpt-oss-20b", "llama-3.3-70b-versatile"]

    def test_env_var_key_only_applies_to_the_currently_active_engine(self, tmp_path):
        """The shared INTERVIEW_ANALYZER_API_KEY env var is a single
        override for whichever engine is actually selected (see
        AnthropicEngine/OpenAIEngine/GroqEngine's own __init__ in
        analyzer.py) -- applying it to a DIFFERENT engine's background
        refresh would be a real correctness bug, not just a convenience."""
        cfg = _cfg(tmp_path)  # analysis.engine == "groq_api"
        with patch("interview_analyzer.model_catalog.api_keys.load_key", return_value=None), \
             patch.dict("os.environ", {"INTERVIEW_ANALYZER_API_KEY": "env-key"}), \
             patch("interview_analyzer.model_catalog.requests.get") as mock_get:
            result = refresh_one(cfg, "anthropic_api")  # NOT the active engine

        assert result is None
        mock_get.assert_not_called()

    def test_env_var_key_applies_to_the_active_engine(self, tmp_path):
        cfg = _cfg(tmp_path)  # analysis.engine == "groq_api"
        resp = _openai_style_response(["llama-3.3-70b-versatile"])
        with patch("interview_analyzer.model_catalog.api_keys.load_key", return_value=None), \
             patch.dict("os.environ", {"INTERVIEW_ANALYZER_API_KEY": "env-key"}), \
             patch("interview_analyzer.model_catalog.requests.get", return_value=resp) as mock_get:
            refresh_one(cfg, "groq_api")  # the active engine

        assert mock_get.call_args.kwargs["headers"]["Authorization"] == "Bearer env-key"

    def test_a_stored_key_takes_precedence_over_the_env_var(self, tmp_path):
        cfg = _cfg(tmp_path)
        resp = _openai_style_response(["llama-3.3-70b-versatile"])
        with patch("interview_analyzer.model_catalog.api_keys.load_key", return_value="saved-key"), \
             patch.dict("os.environ", {"INTERVIEW_ANALYZER_API_KEY": "env-key"}), \
             patch("interview_analyzer.model_catalog.requests.get", return_value=resp) as mock_get:
            refresh_one(cfg, "groq_api")

        assert mock_get.call_args.kwargs["headers"]["Authorization"] == "Bearer saved-key"

    def test_a_failed_fetch_still_bumps_last_checked_and_keeps_the_old_list(self, tmp_path):
        """A persistent provider outage must not retry on every single
        watcher tick forever -- see refresh_one's docstring."""
        cfg = _cfg(tmp_path)
        save_state(cfg, "groq_api", ["stale-model"])
        state = load_state(cfg)
        stale = dt.datetime.now() - dt.timedelta(days=CHECK_INTERVAL_DAYS, hours=1)
        state["groq_api"]["last_checked"] = stale.isoformat()
        _write_state(tmp_path, state)
        assert should_check(cfg, "groq_api") is True

        with patch("interview_analyzer.model_catalog.api_keys.load_key", return_value="fake-key"), \
             patch("interview_analyzer.model_catalog.requests.get", side_effect=requests.RequestException("down")):
            result = refresh_one(cfg, "groq_api")

        assert result is None
        assert load_cached_models(cfg, "groq_api") == ["stale-model"]  # unchanged
        assert should_check(cfg, "groq_api") is False  # but last_checked was bumped


class TestRefreshDueCatalogs:
    def test_only_refreshes_engines_that_are_due(self, tmp_path):
        cfg = _cfg(tmp_path)
        save_state(cfg, "groq_api", ["already-cached"])  # not due

        resp = _openai_style_response(["claude-sonnet-5"])
        with patch("interview_analyzer.model_catalog.api_keys.load_key", return_value="fake-key"), \
             patch("interview_analyzer.model_catalog.requests.get", return_value=resp):
            results = refresh_due_catalogs(cfg)

        assert "groq_api" not in results
        assert results["anthropic_api"] == ["claude-sonnet-5"]
        assert results["openai_api"] == []  # "claude-sonnet-5" isn't a gpt-/o-prefixed id
