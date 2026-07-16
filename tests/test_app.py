"""Tests for app.py's session/login-loop orchestration: logout must tear
everything down and return to the login dialog without restarting the
process, Quit must exit outright, and --username mode (no login UI) must
not try to loop back into a login dialog that doesn't exist.

Dashboard/TrayIcon/MeetingWatcher are all mocked -- the actual Tk/pystray
windows are a manual-verification boundary elsewhere in this app (see
test_tray.py, test_dashboard_history.py); what's under test here is purely
the control flow deciding when to re-login vs. exit.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from interview_analyzer.app import _login, _run_session, main
from interview_analyzer.auth import User
from interview_analyzer.config_loader import Config


def _cfg(tmp_path) -> Config:
    return Config(raw={
        "poll_interval_seconds": 0.01,
        "storage": {"db_path": str(tmp_path / "interviews.db")},
    })


class TestRunSession:
    def _patch_all(self, tray_run_side_effect):
        return patch.multiple(
            "interview_analyzer.app",
            MeetingWatcher=MagicMock(),
            Dashboard=MagicMock(),
            TrayIcon=MagicMock(**{"return_value.run.side_effect": tray_run_side_effect}),
            maybe_run_first_time_setup=MagicMock(),
        )

    def test_returns_quit_by_default_when_tray_run_returns(self, tmp_path):
        with self._patch_all(tray_run_side_effect=lambda: None):
            action = _run_session(_cfg(tmp_path), User(id=1, username="alice"))
        assert action == "quit"

    def test_returns_logout_when_the_logout_callback_fires_before_tray_run_returns(self, tmp_path):
        """Simulates clicking "Log out" in the tray menu -- TrayIcon is
        constructed with an on_logout callback; invoking it (as the real
        menu item's action would) and then returning from run() (as
        icon.stop() would cause) must be reported back as "logout"."""
        import interview_analyzer.app as app_module

        captured = {}

        def fake_tray_icon(watcher, open_dashboard, on_quit, on_logout=None):
            captured["on_logout"] = on_logout
            icon = MagicMock()
            icon.run.side_effect = lambda: on_logout()
            return icon

        with patch.object(app_module, "MeetingWatcher", MagicMock()), \
             patch.object(app_module, "Dashboard", MagicMock()), \
             patch.object(app_module, "TrayIcon", side_effect=fake_tray_icon), \
             patch.object(app_module, "maybe_run_first_time_setup", MagicMock()):
            action = _run_session(_cfg(tmp_path), User(id=1, username="alice"))

        assert action == "logout"
        assert captured["on_logout"] is not None

    def test_shuts_down_watcher_and_closes_dashboard_regardless_of_outcome(self, tmp_path):
        import interview_analyzer.app as app_module

        mock_watcher_cls = MagicMock()
        mock_watcher = mock_watcher_cls.return_value
        mock_dashboard_cls = MagicMock()
        mock_dashboard = mock_dashboard_cls.return_value

        with patch.object(app_module, "MeetingWatcher", mock_watcher_cls), \
             patch.object(app_module, "Dashboard", mock_dashboard_cls), \
             patch.object(app_module, "TrayIcon", MagicMock(**{"return_value.run.side_effect": lambda: None})), \
             patch.object(app_module, "maybe_run_first_time_setup", MagicMock()):
            _run_session(_cfg(tmp_path), User(id=1, username="alice"))

        mock_watcher.shutdown.assert_called()
        mock_dashboard.close.assert_called_once()
        mock_dashboard.wait_until_closed.assert_called_once()


class TestLogin:
    def test_username_flag_creates_a_new_profile_without_gui(self, tmp_path):
        cfg = _cfg(tmp_path)
        args = MagicMock(username="alice")

        user = _login(cfg, args)

        assert user.username == "alice"

    def test_username_flag_reuses_an_existing_profile(self, tmp_path):
        cfg = _cfg(tmp_path)
        args = MagicMock(username="alice")

        first = _login(cfg, args)
        second = _login(cfg, args)

        assert first.id == second.id

    def test_no_username_flag_shows_the_gui_login_dialog(self, tmp_path):
        cfg = _cfg(tmp_path)
        args = MagicMock(username=None)

        with patch("interview_analyzer.app.gui_login_or_create", return_value=User(id=1, username="bob")) as mock_gui:
            user = _login(cfg, args)

        assert user.username == "bob"
        mock_gui.assert_called_once()
        assert mock_gui.call_args.kwargs.get("cfg") is cfg or mock_gui.call_args.args[-1] is cfg


class TestMainLoop:
    def test_logout_returns_to_login_and_a_second_quit_ends_the_process(self, tmp_path):
        """main() must call _login/_run_session again after a logout, and
        stop once a session ends any other way (quit, or login cancelled)."""
        import interview_analyzer.app as app_module

        user1 = User(id=1, username="alice")
        user2 = User(id=1, username="alice")

        with patch.object(app_module, "load_config", return_value=_cfg(tmp_path)), \
             patch.object(app_module, "acquire_single_instance_lock", return_value=True), \
             patch.object(app_module, "_login", side_effect=[user1, user2]) as mock_login, \
             patch.object(app_module, "_run_session", side_effect=["logout", "quit"]) as mock_session, \
             patch("sys.argv", ["app.py"]):
            main()

        assert mock_login.call_count == 2
        assert mock_session.call_count == 2

    def test_login_cancelled_on_first_try_exits_without_running_a_session(self, tmp_path):
        import interview_analyzer.app as app_module

        with patch.object(app_module, "load_config", return_value=_cfg(tmp_path)), \
             patch.object(app_module, "acquire_single_instance_lock", return_value=True), \
             patch.object(app_module, "_login", return_value=None) as mock_login, \
             patch.object(app_module, "_run_session") as mock_session, \
             patch("sys.argv", ["app.py"]):
            main()

        mock_login.assert_called_once()
        mock_session.assert_not_called()

    def test_username_mode_does_not_loop_back_after_logout(self, tmp_path):
        """--username skips the login dialog entirely -- there's nothing to
        log back out *to*, so a logout in this mode should just exit."""
        import interview_analyzer.app as app_module

        with patch.object(app_module, "load_config", return_value=_cfg(tmp_path)), \
             patch.object(app_module, "acquire_single_instance_lock", return_value=True), \
             patch.object(app_module, "_login", return_value=User(id=1, username="alice")) as mock_login, \
             patch.object(app_module, "_run_session", return_value="logout") as mock_session, \
             patch("sys.argv", ["app.py", "--username", "alice"]):
            main()

        mock_login.assert_called_once()
        mock_session.assert_called_once()

    def test_second_instance_exits_immediately(self, tmp_path):
        import interview_analyzer.app as app_module

        with patch.object(app_module, "load_config", return_value=_cfg(tmp_path)), \
             patch.object(app_module, "acquire_single_instance_lock", return_value=False), \
             patch.object(app_module, "_login") as mock_login, \
             patch("sys.argv", ["app.py"]):
            try:
                app_module.main()
                assert False, "expected SystemExit"
            except SystemExit as e:
                assert e.code == 1

        mock_login.assert_not_called()
