import pathlib
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from interview_analyzer.profiles import GENERIC_PROFILE  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def _never_show_a_real_tk_popup_from_a_background_thread():
    """confirm_profile() (profile_confirm.py) and ask_consent()
    (consent.py) both create a real tk.Tk() and call root.mainloop() on a
    background thread whenever no shared ui_root is passed in (see
    watcher.py's _confirm_and_save_profile and its meeting-detection
    tick, respectively). This is a real, reproduced incident, not a
    hypothetical: creating/running a Tk window from a non-main thread is
    undefined behavior on macOS specifically (Cocoa requires GUI work on
    the main thread) -- confirm_profile's version of this crashed the
    whole interpreter (SIGABRT/NSException) rather than erroring, taking
    an entire CI job down with it.

    Deliberately SESSION-scoped, not the default per-test scope -- a
    first attempt at function-scoped mocking here still let this crash
    through. _confirm_and_save_profile runs on a fire-and-forget daemon
    thread that nothing ever joins/cancels; under a loaded CI machine, an
    orphaned thread from one test can sit unscheduled long enough that it
    doesn't actually reach the confirm_profile(...) call until AFTER that
    test's own (function-scoped) patch had already been torn down by its
    fixture teardown -- reproduced exactly this way in CI logs, with the
    crash's traceback showing _confirm_and_save_profile calling the REAL,
    unpatched confirm_profile. Holding the patch for the whole session
    closes that window entirely: there is never a moment after the first
    test starts where the real functions are reachable.

    A single missing mock in one test was the original trigger, but the
    same landmine exists in every test that constructs a real
    MeetingWatcher and doesn't mock these two calls -- auto-mocked here
    for the whole run instead, so it can't recur by omission or by timing.
    A test that genuinely wants to exercise one of the real flows (see
    test_watcher_profiles.py) can still nest its own
    patch("interview_analyzer.watcher.confirm_profile"/"ask_consent", ...)
    inside its own test body; nested patches of the same target compose
    safely regardless of what the outer (session-level) one already is."""
    with patch("interview_analyzer.watcher.confirm_profile", return_value=GENERIC_PROFILE), \
         patch("interview_analyzer.watcher.ask_consent", return_value=False):
        yield
