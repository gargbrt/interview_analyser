import pathlib
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from interview_analyzer.profiles import GENERIC_PROFILE  # noqa: E402


@pytest.fixture(autouse=True)
def _never_show_a_real_tk_popup_from_a_background_thread():
    """confirm_profile() (profile_confirm.py) and ask_consent()
    (consent.py) both create a real tk.Tk() and call root.mainloop() on a
    background thread whenever no shared ui_root is passed in (see
    watcher.py's _confirm_and_save_profile and its meeting-detection
    tick, respectively). This is a real, reproduced incident, not a
    hypothetical: creating/running a Tk window from a non-main thread is
    undefined behavior on macOS specifically (Cocoa requires GUI work on
    the main thread) -- confirm_profile's version of this hung
    indefinitely rather than erroring, taking an entire CI job down with
    it (confirmed via GitHub Actions logs: the job ran for its full
    timeout with zero further test output after the first test that let a
    recording start for real, with confirm_profile the only one of the
    two left unmocked there). A single missing mock in one test was the
    trigger, but the same landmine exists in every test that constructs a
    real MeetingWatcher and doesn't mock these two calls -- auto-mocked
    here for every test instead, so it can't recur by omission, for either
    function. A test that genuinely wants to exercise one of the real
    flows (see test_watcher_profiles.py) can still nest its own
    patch("interview_analyzer.watcher.confirm_profile"/"ask_consent", ...)
    inside these; nested patches of the same target compose safely."""
    with patch("interview_analyzer.watcher.confirm_profile", return_value=GENERIC_PROFILE), \
         patch("interview_analyzer.watcher.ask_consent", return_value=False):
        yield
