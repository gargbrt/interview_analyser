"""Tests for language_packs.py's non-UI logic: the pack registry,
installed-detection, and pip subprocess wrapping. The Tk progress dialog
itself isn't exercised here -- same manual-verification boundary as other
Tkinter popups in this app."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from interview_analyzer.language_packs import (
    LANGUAGE_PACKS,
    install_pack,
    is_pack_installed,
    uninstall_pack,
)


def test_hindi_hinglish_pack_is_registered():
    assert "hindi_hinglish" in LANGUAGE_PACKS
    assert LANGUAGE_PACKS["hindi_hinglish"]["pip_package"] == "indic-transliteration"


def test_is_pack_installed_reflects_whether_the_module_is_importable():
    with patch("interview_analyzer.language_packs.importlib.util.find_spec", return_value=object()):
        assert is_pack_installed("hindi_hinglish") is True
    with patch("interview_analyzer.language_packs.importlib.util.find_spec", return_value=None):
        assert is_pack_installed("hindi_hinglish") is False


def test_is_pack_installed_false_for_unknown_pack():
    assert is_pack_installed("not_a_real_pack") is False


def _fake_process(returncode: int, stdout_lines: list[str]) -> MagicMock:
    proc = MagicMock()
    proc.stdout = iter(l + "\n" for l in stdout_lines)
    proc.wait.return_value = returncode
    return proc


class TestInstallUninstall:
    def test_install_pack_runs_pip_install_with_the_right_package(self):
        with patch("interview_analyzer.language_packs.subprocess.Popen", return_value=_fake_process(0, ["Successfully installed"])) as mock_popen, \
             patch("interview_analyzer.language_packs.importlib.invalidate_caches"):
            output = []
            install_pack("hindi_hinglish", on_output=output.append)

        args = mock_popen.call_args[0][0]
        assert "install" in args
        assert "indic-transliteration" in args
        assert output == ["Successfully installed"]

    def test_uninstall_pack_runs_pip_uninstall_with_yes_flag(self):
        with patch("interview_analyzer.language_packs.subprocess.Popen", return_value=_fake_process(0, ["Successfully uninstalled"])) as mock_popen, \
             patch("interview_analyzer.language_packs.importlib.invalidate_caches"):
            uninstall_pack("hindi_hinglish")

        args = mock_popen.call_args[0][0]
        assert "uninstall" in args
        assert "-y" in args
        assert "indic-transliteration" in args

    def test_install_pack_raises_on_nonzero_exit(self):
        with patch("interview_analyzer.language_packs.subprocess.Popen", return_value=_fake_process(1, ["ERROR: could not find a version"])):
            with pytest.raises(RuntimeError, match="could not find a version"):
                install_pack("hindi_hinglish")

    def test_unknown_pack_raises_value_error(self):
        with pytest.raises(ValueError):
            install_pack("not_a_real_pack")
        with pytest.raises(ValueError):
            uninstall_pack("not_a_real_pack")
