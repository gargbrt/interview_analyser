<#
.SYNOPSIS
    Creates an "Interview Analyzer.lnk" launch shortcut in the project's
    root folder, pointing at run_app.bat with the app's icon.

.DESCRIPTION
    Portable -- resolves all paths relative to this script's own location,
    so it works for anyone who clones the repo, regardless of where. The
    resulting .lnk is machine-specific (it embeds an absolute path), so
    it's git-ignored (*.lnk) rather than committed; run this script once
    after cloning to generate your own.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\create_shortcut.ps1
#>

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$batPath = Join-Path $projectRoot "run_app.bat"
$iconPath = Join-Path $projectRoot "assets\icon.ico"
$shortcutPath = Join-Path $projectRoot "Interview Analyzer.lnk"

if (-not (Test-Path $batPath)) {
    throw "run_app.bat not found at $batPath -- run this script from a full clone of the repo."
}

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $batPath
$shortcut.WorkingDirectory = $projectRoot
$shortcut.WindowStyle = 7  # minimized -- the launcher's own console window is brief anyway
$shortcut.Description = "Launch Interview Analyzer"
if (Test-Path $iconPath) {
    $shortcut.IconLocation = $iconPath
}
$shortcut.Save()

Write-Host "Created shortcut: $shortcutPath"
Write-Host "Double-click it (or run_app.bat directly) to launch Interview Analyzer."
