"""Tests for model_setup.py: the local Ollama model catalog/size display,
install-detection, and the streaming pull-progress parsing. The Tk wizard
UI itself isn't exercised here (same "manual verification boundary" as
other Tkinter popups in this app -- see consent.py/control_panel.py) --
this covers the non-UI logic that decides *when* to ask and what to show."""
from __future__ import annotations

import json
import pathlib
from unittest.mock import MagicMock, patch

import pytest

from interview_analyzer.config_loader import Config
from interview_analyzer.model_setup import (
    MODEL_CATALOG,
    approx_size_gb,
    ensure_ollama_running,
    is_model_installed,
    list_installed_models,
    mark_setup_done,
    ollama_is_reachable,
    pull_model,
    setup_already_done,
    size_label,
)


def test_default_model_has_a_catalog_entry_with_a_size():
    assert "llama3.1:8b" in MODEL_CATALOG
    assert approx_size_gb("llama3.1:8b") == pytest.approx(4.7)
    assert "GB" in size_label("llama3.1:8b")


def test_size_label_is_honest_about_unknown_models():
    assert approx_size_gb("some-random-model:latest") is None
    assert "unknown" in size_label("some-random-model:latest")


def test_ollama_is_reachable_true_and_false():
    with patch("interview_analyzer.model_setup.requests.get", return_value=MagicMock()):
        assert ollama_is_reachable("http://localhost:11434") is True

    import requests

    with patch("interview_analyzer.model_setup.requests.get", side_effect=requests.RequestException("down")):
        assert ollama_is_reachable("http://localhost:11434") is False


def test_list_installed_models_parses_the_tags_response():
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"models": [{"name": "llama3.1:8b"}, {"name": "qwen2.5:14b"}]}
    fake_resp.raise_for_status.return_value = None
    with patch("interview_analyzer.model_setup.requests.get", return_value=fake_resp):
        assert list_installed_models("http://localhost:11434") == ["llama3.1:8b", "qwen2.5:14b"]


def test_is_model_installed_matches_bare_and_latest_suffixed_names():
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"models": [{"name": "llama3.1:8b"}]}
    fake_resp.raise_for_status.return_value = None
    with patch("interview_analyzer.model_setup.requests.get", return_value=fake_resp):
        assert is_model_installed("llama3.1:8b", "http://localhost:11434") is True
        assert is_model_installed("qwen2.5:14b", "http://localhost:11434") is False


def _ndjson_lines(*objs) -> list[bytes]:
    return [json.dumps(o).encode("utf-8") for o in objs]


class TestPullModel:
    def test_reports_real_byte_progress_and_final_status(self):
        lines = _ndjson_lines(
            {"status": "pulling manifest"},
            {"status": "downloading", "completed": 100, "total": 1000},
            {"status": "downloading", "completed": 1000, "total": 1000},
            {"status": "success"},
        )
        fake_resp = MagicMock()
        fake_resp.iter_lines.return_value = lines
        fake_resp.raise_for_status.return_value = None

        with patch("interview_analyzer.model_setup.requests.post", return_value=fake_resp):
            calls = []
            pull_model("llama3.1:8b", "http://localhost:11434", on_progress=lambda f, s: calls.append((f, s)))

        assert calls[0] == (None, "pulling manifest")  # no byte counts yet -- indeterminate
        assert calls[1] == (0.1, "downloading")
        assert calls[2] == (1.0, "downloading")
        assert calls[3] == (None, "success")

    def test_raises_on_error_status_from_ollama(self):
        lines = _ndjson_lines({"error": "model not found"})
        fake_resp = MagicMock()
        fake_resp.iter_lines.return_value = lines
        fake_resp.raise_for_status.return_value = None

        with patch("interview_analyzer.model_setup.requests.post", return_value=fake_resp):
            with pytest.raises(RuntimeError, match="model not found"):
                pull_model("bogus:model", "http://localhost:11434")

    def test_cancel_event_stops_the_download(self):
        import threading

        lines = _ndjson_lines(
            {"status": "downloading", "completed": 1, "total": 1000},
            {"status": "downloading", "completed": 2, "total": 1000},
        )
        fake_resp = MagicMock()
        fake_resp.iter_lines.return_value = lines
        fake_resp.raise_for_status.return_value = None

        cancel_event = threading.Event()
        cancel_event.set()

        with patch("interview_analyzer.model_setup.requests.post", return_value=fake_resp):
            with pytest.raises(InterruptedError):
                pull_model("llama3.1:8b", "http://localhost:11434", cancel_event=cancel_event)


class TestSetupMarker:
    def test_setup_not_done_until_marked(self, tmp_path):
        cfg = Config(raw={"storage": {"db_path": str(tmp_path / "data" / "interviews.db")}})
        assert setup_already_done(cfg) is False
        mark_setup_done(cfg)
        assert setup_already_done(cfg) is True


class TestEnsureOllamaRunning:
    """ensure_ollama_running is what lets analysis (and Reprocess) just
    work even if Ollama wasn't already running, instead of failing outright
    with a raw connection error -- see analyzer.py's OllamaEngine.run."""

    def test_returns_true_immediately_if_already_reachable(self):
        with patch("interview_analyzer.model_setup.ollama_is_reachable", return_value=True), \
             patch("interview_analyzer.model_setup.shutil.which") as mock_which, \
             patch("interview_analyzer.model_setup.subprocess.Popen") as mock_popen:
            assert ensure_ollama_running("http://localhost:11434") is True

        mock_which.assert_not_called()
        mock_popen.assert_not_called()

    def test_starts_ollama_via_path_and_waits_until_reachable(self):
        # not reachable at first, then reachable after the process starts
        reachable_calls = iter([False, False, True])
        with patch(
            "interview_analyzer.model_setup.ollama_is_reachable",
            side_effect=lambda host: next(reachable_calls),
        ), patch("interview_analyzer.model_setup.shutil.which", return_value="/usr/local/bin/ollama"), \
             patch("interview_analyzer.model_setup.subprocess.Popen") as mock_popen, \
             patch("interview_analyzer.model_setup.time.sleep"):
            assert ensure_ollama_running("http://localhost:11434", timeout=5) is True

        mock_popen.assert_called_once()
        assert mock_popen.call_args.args[0] == ["/usr/local/bin/ollama", "serve"]

    def test_falls_back_to_platform_specific_candidate_paths_when_not_on_path(self):
        fake_exe = pathlib.Path("/opt/homebrew/bin/ollama")
        with patch("interview_analyzer.model_setup.ollama_is_reachable", side_effect=[False, True]), \
             patch("interview_analyzer.model_setup.shutil.which", return_value=None), \
             patch(
                 "interview_analyzer.model_setup._ollama_executable_candidates",
                 return_value=[fake_exe],
             ), \
             patch.object(pathlib.Path, "exists", return_value=True), \
             patch("interview_analyzer.model_setup.subprocess.Popen") as mock_popen, \
             patch("interview_analyzer.model_setup.time.sleep"):
            assert ensure_ollama_running("http://localhost:11434", timeout=5) is True

        assert mock_popen.call_args.args[0] == [str(fake_exe), "serve"]

    def test_returns_false_when_ollama_cannot_be_found_anywhere(self):
        with patch("interview_analyzer.model_setup.ollama_is_reachable", return_value=False), \
             patch("interview_analyzer.model_setup.shutil.which", return_value=None), \
             patch("interview_analyzer.model_setup._ollama_executable_candidates", return_value=[]), \
             patch("interview_analyzer.model_setup.subprocess.Popen") as mock_popen:
            assert ensure_ollama_running("http://localhost:11434") is False

        mock_popen.assert_not_called()

    def test_returns_false_if_it_never_becomes_reachable_within_timeout(self):
        with patch("interview_analyzer.model_setup.ollama_is_reachable", return_value=False), \
             patch("interview_analyzer.model_setup.shutil.which", return_value="/usr/local/bin/ollama"), \
             patch("interview_analyzer.model_setup.subprocess.Popen"), \
             patch("interview_analyzer.model_setup.time.sleep"), \
             patch(
                 "interview_analyzer.model_setup.time.monotonic",
                 side_effect=[0, 1, 2, 999],  # jumps past the deadline
             ):
            assert ensure_ollama_running("http://localhost:11434", timeout=5) is False

    def test_returns_false_if_the_executable_is_found_but_fails_to_launch(self):
        with patch("interview_analyzer.model_setup.ollama_is_reachable", return_value=False), \
             patch("interview_analyzer.model_setup.shutil.which", return_value="/usr/local/bin/ollama"), \
             patch(
                 "interview_analyzer.model_setup.subprocess.Popen",
                 side_effect=OSError("permission denied"),
             ):
            assert ensure_ollama_running("http://localhost:11434") is False
