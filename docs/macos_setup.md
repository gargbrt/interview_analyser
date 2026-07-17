# macOS Setup

Interview Analyzer supports macOS as well as Windows. The core pipeline
(transcription, analysis, reports, dashboard) is identical on both --
the differences are all in how the app gets at system audio and window
titles, both of which need a bit of one-time setup on macOS that Windows
doesn't.

## 1. Install a virtual audio loopback driver (required)

macOS has no built-in equivalent of Windows' WASAPI loopback mode -- there's
no way to just ask "give me whatever's currently playing." You need a
virtual audio device that system output gets routed through, which this
app then captures from.

**[BlackHole](https://github.com/ExistentialAudio/BlackHole)** is the
free, open-source option (recommended):

1. Install BlackHole 2ch: `brew install blackhole-2ch`, or download the
   installer from the [BlackHole releases page](https://github.com/ExistentialAudio/BlackHole/releases).
2. Open **Audio MIDI Setup** (Spotlight search for it).
3. Click **+** (bottom-left) → **Create Multi-Output Device**.
4. Check both your normal output (e.g. "MacBook Pro Speakers") and
   **BlackHole 2ch** in the new Multi-Output Device.
5. In **System Settings → Sound → Output**, select that Multi-Output
   Device as your output. You'll still hear audio normally (through the
   speakers in the group) while BlackHole simultaneously receives a copy
   Interview Analyzer can record from.

Only needs to be done once. Switch back to your normal output device when
you're not recording, if you'd rather not leave the Multi-Output Device
selected all the time.

(Loopback by Rogue Amoeba or Soundflower work too, if you already have one
installed -- the app looks for a device with "BlackHole", "Loopback", or
"Soundflower" in its name.)

## 2. Grant permissions

macOS will prompt for these the first time each is actually needed;
you can also grant them upfront in **System Settings → Privacy & Security**:

- **Microphone** -- needed to include your own voice in the recording
  (system audio alone only captures the interviewer's side).
- **Screen Recording** -- needed to read other apps'/browsers' window
  titles for meeting detection (e.g. recognizing a Google Meet browser
  tab). Without this, desktop-app detection (Zoom.app, Teams running as a
  real app) still works fine via process names -- only the weaker
  browser-tab-title signal is affected. If you skip this, use the
  dashboard's manual "Start recording" button instead.

## 3. Install and run

```bash
git clone https://github.com/gargbrt/interview_analyser.git
cd interview_analyser
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

```bash
./run_app.command
```

(or `python -m interview_analyzer.app` directly). `run_app.command` is
double-clickable from Finder too -- right-click it once and "Open" the
first time (macOS Gatekeeper warns on unsigned scripts by default), after
which double-clicking works normally.

## What's different from Windows under the hood

- **Audio capture**: `sounddevice` reading from the BlackHole device
  above, instead of WASAPI loopback (`recorder.py`).
- **Meeting detection**: `Quartz` (via `pyobjc`) for window titles instead
  of `win32gui`; `config.yaml`'s `watched_processes.desktop_apps_macos` /
  `browser_processes_macos` for macOS process names instead of the
  Windows `.exe` lists (`watcher.py`).
- **Credential storage** (remembered login password, cloud API keys):
  macOS Keychain via the `keyring` package, instead of Windows DPAPI
  (`remembered_login.py`, `api_keys.py`).
- **Single-instance lock**: `fcntl.flock` instead of `msvcrt.locking`
  (`single_instance.py`).

None of this needs configuring by hand -- the app picks the right backend
automatically based on the OS it's running on. If detection or recording
doesn't work as expected, check the two setup steps above first (missing
virtual audio device, or missing Screen Recording permission) -- those are
the two things Windows doesn't require that macOS does.

## Known limitation

This macOS support has been built and unit-tested (with the audio/window
APIs mocked, verified by CI running the real test suite on macOS runners),
but not yet exercised on a real Mac in a live interview call the way the
Windows path has been (extensively, over many real recordings during this
project's development). If something doesn't work as described here,
please open an issue with what you saw -- the mic/loopback mixing timing
in particular (`_MacAudioRecorder`) is the part most likely to need
real-world tuning.
