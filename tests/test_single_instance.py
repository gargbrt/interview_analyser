"""Tests for the single-instance lock that stops two copies of app.py from
running (and both polling/prompting/writing to the same DB) at once."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from interview_analyzer.single_instance import acquire_single_instance_lock


def test_first_acquisition_succeeds(tmp_path):
    assert acquire_single_instance_lock(tmp_path / "app.lock") is True


def test_second_acquisition_in_same_process_fails(tmp_path):
    lock_path = tmp_path / "app.lock"
    assert acquire_single_instance_lock(lock_path) is True
    # msvcrt.locking is per-file-handle; a second open+lock attempt on the
    # same path (simulating a second app instance) must fail while the
    # first handle is still held open
    assert acquire_single_instance_lock(lock_path) is False


def test_different_lock_paths_do_not_conflict(tmp_path):
    assert acquire_single_instance_lock(tmp_path / "a.lock") is True
    assert acquire_single_instance_lock(tmp_path / "b.lock") is True


class TestMacOsUsesFcntlNotMsvcrt:
    """fcntl doesn't exist on Windows, so these fake the module entirely to
    exercise the macOS code path from this (Windows) dev/CI machine --
    real behavior is also covered for real by the macos-latest CI job."""

    def test_calls_flock_on_darwin(self, tmp_path):
        fake_fcntl = MagicMock()
        fake_fcntl.LOCK_EX = 2
        fake_fcntl.LOCK_NB = 4
        with patch("interview_analyzer.single_instance.sys.platform", "darwin"), \
             patch.dict("sys.modules", {"fcntl": fake_fcntl}):
            result = acquire_single_instance_lock(tmp_path / "app.lock")

        assert result is True
        fake_fcntl.flock.assert_called_once()

    def test_returns_false_when_flock_raises(self, tmp_path):
        fake_fcntl = MagicMock()
        fake_fcntl.flock.side_effect = OSError("already locked")
        with patch("interview_analyzer.single_instance.sys.platform", "darwin"), \
             patch.dict("sys.modules", {"fcntl": fake_fcntl}):
            result = acquire_single_instance_lock(tmp_path / "app.lock")

        assert result is False
