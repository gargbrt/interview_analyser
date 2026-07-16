# Running Automatically at Startup (Windows)

You want the watcher running in the background all the time, without having
to remember to launch it. Two supported options:

## Option A — Task Scheduler (recommended)

1. Open **Task Scheduler** → *Create Task* (not "Basic Task", so you get
   the full options).
2. **General tab**: name it `Interview Analyzer`, check "Run only when user
   is logged on" (so the consent popup can display), check "Run with
   highest privileges" only if you hit permission errors (usually not needed).
3. **Triggers tab**: New → "At log on" → your user account.
4. **Actions tab**: New → "Start a program":
   - Program/script: `C:\path\to\interview-analyzer\.venv\Scripts\pythonw.exe`
   - Add arguments: `-m interview_analyzer.watcher --username yourname`
   - Start in: `C:\path\to\interview-analyzer`

   Using `pythonw.exe` (not `python.exe`) avoids a console window popping up
   at every login; the consent dialog will still appear normally since it's
   a separate Tk window.
5. **Conditions/Settings tabs**: uncheck "Stop the task if it runs longer
   than 3 days" (Settings tab) since this is meant to run indefinitely.

The `--username yourname` flag skips the interactive login prompt at
startup and logs straight into that profile — create the profile once
manually first by running `python -m interview_analyzer.watcher` and typing
your username when prompted.

## Option B — Startup folder (simpler, less robust)

1. Create a `.bat` file:
   ```bat
   @echo off
   cd /d C:\path\to\interview-analyzer
   .venv\Scripts\pythonw.exe -m interview_analyzer.watcher --username yourname
   ```
2. Press `Win+R`, type `shell:startup`, press Enter.
3. Drop a shortcut to the `.bat` file in that folder.

This runs the watcher once each time you log in, but (unlike Task
Scheduler) won't restart it if it crashes, and won't run before an
interactive login.

## Stopping it

Open Task Manager and end the `pythonw.exe` process running
`interview_analyzer.watcher`, or (Option A) disable/delete the scheduled
task.

## Verifying it's watching correctly

Check `output/trends_user<id>.md`'s timestamp after a real recorded interview (each profile gets its own trends file), or
watch the log output by running `python -m interview_analyzer.watcher`
directly (not via `pythonw.exe`) to see the console log lines as it detects
meetings, asks for consent, and processes interviews.
