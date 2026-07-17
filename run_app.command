#!/bin/bash
# Portable launcher: works from any clone location, no absolute paths.
# Double-click this in Finder (or run it from a terminal) to start the app.
# First double-click: right-click -> Open once, since macOS Gatekeeper
# warns on unsigned scripts by default -- after that, double-click works.
set -e
cd "$(dirname "$0")"

if [ -x ".venv/bin/python3" ]; then
    PYTHON=".venv/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
else
    echo "python3 not found. Install Python 3.10+ and/or create a .venv"
    echo "virtual environment first -- see README.md's Setup section."
    read -p "Press Enter to close..."
    exit 1
fi

mkdir -p logs
# pythonw has no console; on macOS a plain python3 GUI app similarly has no
# terminal once double-clicked from Finder, so capture output to a log file
# for troubleshooting instead of it vanishing silently.
"$PYTHON" -m interview_analyzer.app > logs/app_launch.log 2>&1 &
disown

echo "Interview Analyzer starting -- check logs/app_launch.log if it doesn't appear."
sleep 2
