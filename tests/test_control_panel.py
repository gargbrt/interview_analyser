"""Tests for RecordingControlPanel's minimize/hover-to-reveal behavior (see
control_panel.py's module docstring). Uses a real (but withdrawn/invisible)
Tk root -- same pattern as test_profile_confirm.py's `_shared_tk_root`
fixture -- since the compact/full switching is real widget pack()/
pack_forget() state that a MagicMock can't meaningfully verify.

Builds the panel by constructing it directly (bypassing __init__'s
threading/after() dance -- there's no mainloop running for _shared_tk_root
in these tests to pump that through) and calling _build() synchronously
against a real Toplevel of the shared root, exactly like
test_profile_confirm.py's TestBuildPopupIntroText does for _build_popup.
Deliberately reuses the ONE shared root via Toplevel() rather than creating
another real tk.Tk() -- see that fixture's own docstring for why a second
Tk() interpreter on another thread isn't safe here.

Pause/Resume/Stop themselves are already exercised indirectly through
watcher.py's own tests (which mock this class at the boundary); these tests
are specifically about the minimize/restore/hover state machine, which had
no coverage before it existed.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from interview_analyzer.control_panel import RecordingControlPanel

_FAST_COLLAPSE_MS = 20


@pytest.fixture(scope="module")
def _shared_tk_root():
    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"No usable Tk display in this environment: {e}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def panel(_shared_tk_root):
    import tkinter as tk

    p = RecordingControlPanel.__new__(RecordingControlPanel)
    p.app_name = "TestApp"
    p._on_pause = MagicMock()
    p._on_resume = MagicMock()
    p._on_stop = MagicMock()
    p._paused = False
    p._stopped = False
    p._root = None
    p._full_frame = None
    p._compact_frame = None
    p._pause_btn = None
    p._status_label = None
    p._timer_label = None
    p._activity_bar = None
    p._compact_dot = None
    p._compact_timer_label = None
    p._recorded_seconds = 0
    p._ready = threading.Event()
    p._compact = False
    p._collapse_after_id = None

    p._build(tk.Toplevel(_shared_tk_root))
    _shared_tk_root.update()  # newly packed widgets need an idle tick to actually become "mapped"

    yield p
    p.close()
    _shared_tk_root.update()


def _pump(root, seconds: float) -> None:
    """Lets pending Tk `after()` callbacks run by sleeping and processing
    the event loop -- real wall-clock time is needed since `after()` is
    scheduled against Tk's own clock, not something a test can fast-forward."""
    time.sleep(seconds)
    root.update()


class TestInitialState:
    def test_starts_showing_full_controls(self, panel):
        assert panel._full_frame.winfo_ismapped()
        assert not panel._compact_frame.winfo_ismapped()


class TestMinimizeAndRestore:
    def test_minimize_collapses_to_the_compact_pill(self, panel, _shared_tk_root):
        panel.minimize()
        _pump(_shared_tk_root, 0.05)

        assert panel._compact_frame.winfo_ismapped()
        assert not panel._full_frame.winfo_ismapped()

    def test_restore_shows_full_controls_again(self, panel, _shared_tk_root):
        panel.minimize()
        _pump(_shared_tk_root, 0.05)

        panel.restore()
        _pump(_shared_tk_root, 0.05)

        assert panel._full_frame.winfo_ismapped()
        assert not panel._compact_frame.winfo_ismapped()

    def test_restore_survives_a_later_hover_leave(self, panel, _shared_tk_root):
        """Once restore() is used, the panel is no longer in "compact
        mode" at all -- a stray hover-leave afterwards must not collapse
        it, unlike while genuinely minimized."""
        panel.minimize()
        _pump(_shared_tk_root, 0.05)
        panel.restore()
        _pump(_shared_tk_root, 0.05)

        with patch("interview_analyzer.control_panel._COLLAPSE_DELAY_MS", _FAST_COLLAPSE_MS):
            panel._on_hover_leave()
            _pump(_shared_tk_root, 0.05)

        assert panel._full_frame.winfo_ismapped()


class TestHoverPeek:
    def test_hover_enter_while_compact_peeks_full_controls(self, panel, _shared_tk_root):
        panel.minimize()
        _pump(_shared_tk_root, 0.05)

        panel._on_hover_enter()
        _shared_tk_root.update()

        assert panel._full_frame.winfo_ismapped()
        assert not panel._compact_frame.winfo_ismapped()

    def test_hover_leave_collapses_back_to_compact_after_the_delay(self, panel, _shared_tk_root):
        panel.minimize()
        _pump(_shared_tk_root, 0.05)
        panel._on_hover_enter()  # peek

        with patch("interview_analyzer.control_panel._COLLAPSE_DELAY_MS", _FAST_COLLAPSE_MS):
            panel._on_hover_leave()
            _pump(_shared_tk_root, 0.05)

        assert panel._compact_frame.winfo_ismapped()
        assert not panel._full_frame.winfo_ismapped()

    def test_hover_enter_before_the_delay_elapses_cancels_the_collapse(self, panel, _shared_tk_root):
        """Regression coverage for the exact race this delay exists to
        absorb -- Tk can fire a spurious <Leave>/<Enter> pair when the
        mouse moves onto a child widget (e.g. the Pause button) inside the
        panel. A quick re-entry must cancel the pending collapse rather
        than letting it fire anyway once the delay is up."""
        panel.minimize()
        _pump(_shared_tk_root, 0.05)
        panel._on_hover_enter()  # peek

        with patch("interview_analyzer.control_panel._COLLAPSE_DELAY_MS", _FAST_COLLAPSE_MS):
            panel._on_hover_leave()
            panel._on_hover_enter()  # re-entered before the delay elapsed
            _pump(_shared_tk_root, 0.05)  # long enough that the original collapse WOULD have fired

        assert panel._full_frame.winfo_ismapped()
        assert not panel._compact_frame.winfo_ismapped()

    def test_hover_enter_while_not_compact_is_a_noop(self, panel, _shared_tk_root):
        """Hovering over the panel while it's already showing full
        controls (never minimized) shouldn't do anything unusual."""
        panel._on_hover_enter()
        _shared_tk_root.update()
        assert panel._full_frame.winfo_ismapped()
        assert not panel._compact_frame.winfo_ismapped()


class TestPauseSyncsCompactStatus:
    def test_pausing_greys_out_the_compact_status_dot(self, panel):
        panel._pause_btn.invoke()
        assert panel._compact_dot.cget("fg") == "gray"

    def test_resuming_reverts_the_compact_status_dot_to_red(self, panel):
        panel._pause_btn.invoke()
        panel._pause_btn.invoke()
        assert panel._compact_dot.cget("fg") == "red"
