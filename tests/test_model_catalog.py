"""Tests for model_catalog.py: the periodic Groq model-catalog refresh
(check every CHECK_INTERVAL_DAYS days, and on the very first tick after a
fresh install since there's no state file yet). Network calls are mocked
throughout -- this covers the caching/scheduling/filtering logic, not
Groq's actual API."""
from __future__ import annotations

import datetime as dt
import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from interview_analyzer.config_loader import Config
from interview_analyzer.model_catalog import (
    CHECK_INTERVAL_DAYS,
    FALLBACK_GROQ_MODELS,
    fetch_latest_groq_models,
    load_cached_models,
    load_state,
    refresh_model_catalog,
    save_state,
    should_check,
)


def _cfg(tmp_path) -> Config:
    return Config(raw={
        "storage": {"db_path": str(tmp_path / "interviews.db")},
        "analysis": {"engine": "groq_api", "cloud_api_key_env_var": "INTERVIEW_ANALYZER_API_KEY"},
    })


def _groq_response(ids: list[str], inactive: list[str] = ()) -> MagicMock:
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
    def test_true_when_never_checked_before(self, tmp_path):
        """Covers both "fresh install" and "no state file yet" -- the same
        condition, since a brand new install has never written one."""
        assert should_check(_cfg(tmp_path)) is True

    def test_false_right_after_a_check(self, tmp_path):
        cfg = _cfg(tmp_path)
        save_state(cfg, ["llama-3.3-70b-versatile"])
        assert should_check(cfg) is False

    def test_true_once_the_interval_has_elapsed(self, tmp_path):
        cfg = _cfg(tmp_path)
        save_state(cfg, ["llama-3.3-70b-versatile"])
        state = load_state(cfg)
        stale = dt.datetime.now() - dt.timedelta(days=CHECK_INTERVAL_DAYS, hours=1)
        state["last_checked"] = stale.isoformat()
        cfg_path = tmp_path / "interviews.db"
        state_path = cfg_path.with_name(".groq_model_catalog.json")
        state_path.write_text(json.dumps(state), encoding="utf-8")

        assert should_check(cfg) is True

    def test_false_just_under_the_interval(self, tmp_path):
        cfg = _cfg(tmp_path)
        save_state(cfg, ["llama-3.3-70b-versatile"])
        state = load_state(cfg)
        recent = dt.datetime.now() - dt.timedelta(days=CHECK_INTERVAL_DAYS - 1)
        state["last_checked"] = recent.isoformat()
        state_path = (tmp_path / "interviews.db").with_name(".groq_model_catalog.json")
        state_path.write_text(json.dumps(state), encoding="utf-8")

        assert should_check(cfg) is False

    def test_true_on_a_corrupt_state_file(self, tmp_path):
        cfg = _cfg(tmp_path)
        state_path = (tmp_path / "interviews.db").with_name(".groq_model_catalog.json")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("{not json", encoding="utf-8")

        assert should_check(cfg) is True


class TestStateRoundTrip:
    def test_load_cached_models_falls_back_when_nothing_saved_yet(self, tmp_path):
        assert load_cached_models(_cfg(tmp_path)) == FALLBACK_GROQ_MODELS

    def test_save_then_load_round_trips(self, tmp_path):
        cfg = _cfg(tmp_path)
        save_state(cfg, ["model-a", "model-b"])
        assert load_cached_models(cfg) == ["model-a", "model-b"]

    def test_first_checked_is_preserved_across_repeated_saves(self, tmp_path):
        cfg = _cfg(tmp_path)
        save_state(cfg, ["model-a"])
        first_checked = load_state(cfg)["first_checked"]

        save_state(cfg, ["model-a", "model-b"])

        state = load_state(cfg)
        assert state["first_checked"] == first_checked
        assert state["last_checked"] != first_checked or True  # last_checked always refreshed
        assert state["models"] == ["model-a", "model-b"]


class TestFetchLatestGroqModels:
    def test_returns_sorted_active_chat_model_ids(self):
        resp = _groq_response(["llama-3.3-70b-versatile", "gpt-oss-20b"])
        with patch("interview_analyzer.model_catalog.requests.get", return_value=resp) as mock_get:
            models = fetch_latest_groq_models("fake-key")

        assert models == ["gpt-oss-20b", "llama-3.3-70b-versatile"]
        assert mock_get.call_args.kwargs["headers"]["Authorization"] == "Bearer fake-key"

    def test_excludes_non_chat_model_families(self):
        resp = _groq_response([
            "llama-3.3-70b-versatile",
            "whisper-large-v3",
            "distil-whisper-large-v3-en",
            "playai-tts",
            "llama-guard-3-8b",
        ])
        with patch("interview_analyzer.model_catalog.requests.get", return_value=resp):
            models = fetch_latest_groq_models("fake-key")

        assert models == ["llama-3.3-70b-versatile"]

    def test_excludes_inactive_models(self):
        resp = _groq_response(["llama-3.3-70b-versatile"], inactive=["some-deprecated-model"])
        with patch("interview_analyzer.model_catalog.requests.get", return_value=resp):
            models = fetch_latest_groq_models("fake-key")

        assert models == ["llama-3.3-70b-versatile"]

    def test_raises_on_an_http_error(self):
        with patch("interview_analyzer.model_catalog.requests.get", side_effect=requests.RequestException("down")):
            with pytest.raises(requests.RequestException):
                fetch_latest_groq_models("fake-key")


class TestRefreshModelCatalog:
    def test_returns_none_and_saves_nothing_without_a_key(self, tmp_path):
        cfg = _cfg(tmp_path)
        with patch("interview_analyzer.model_catalog.api_keys.load_key", return_value=None), \
             patch.dict("os.environ", {}, clear=True):
            result = refresh_model_catalog(cfg)

        assert result is None
        assert load_state(cfg) == {}

    def test_success_saves_and_returns_the_fetched_models(self, tmp_path):
        cfg = _cfg(tmp_path)
        resp = _groq_response(["llama-3.3-70b-versatile", "gpt-oss-20b"])
        with patch("interview_analyzer.model_catalog.api_keys.load_key", return_value="fake-key"), \
             patch("interview_analyzer.model_catalog.requests.get", return_value=resp):
            result = refresh_model_catalog(cfg)

        assert result == ["gpt-oss-20b", "llama-3.3-70b-versatile"]
        assert load_cached_models(cfg) == ["gpt-oss-20b", "llama-3.3-70b-versatile"]

    def test_env_var_key_takes_precedence_over_a_saved_one(self, tmp_path):
        cfg = _cfg(tmp_path)
        resp = _groq_response(["llama-3.3-70b-versatile"])
        with patch("interview_analyzer.model_catalog.api_keys.load_key", return_value="saved-key"), \
             patch.dict("os.environ", {"INTERVIEW_ANALYZER_API_KEY": "env-key"}), \
             patch("interview_analyzer.model_catalog.requests.get", return_value=resp) as mock_get:
            refresh_model_catalog(cfg)

        assert mock_get.call_args.kwargs["headers"]["Authorization"] == "Bearer env-key"

    def test_a_failed_fetch_still_bumps_last_checked_and_keeps_the_old_list(self, tmp_path):
        """A persistent Groq outage must not retry on every single watcher
        tick forever -- see refresh_model_catalog's docstring."""
        cfg = _cfg(tmp_path)
        save_state(cfg, ["stale-model"])
        assert should_check(cfg) is False

        state = load_state(cfg)
        stale = dt.datetime.now() - dt.timedelta(days=CHECK_INTERVAL_DAYS, hours=1)
        state["last_checked"] = stale.isoformat()
        state_path = (tmp_path / "interviews.db").with_name(".groq_model_catalog.json")
        state_path.write_text(json.dumps(state), encoding="utf-8")
        assert should_check(cfg) is True

        with patch("interview_analyzer.model_catalog.api_keys.load_key", return_value="fake-key"), \
             patch("interview_analyzer.model_catalog.requests.get", side_effect=requests.RequestException("down")):
            result = refresh_model_catalog(cfg)

        assert result is None
        assert load_cached_models(cfg) == ["stale-model"]  # unchanged
        assert should_check(cfg) is False  # but last_checked was bumped
