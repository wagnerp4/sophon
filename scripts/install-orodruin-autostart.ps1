param(
    [string]$Preset = "llama2_7b_chat"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath $Root)) {
    $Root = "C:\Software\Python\NLP\Personal\orodruin"
}

$StartScript = Join-Path $Root "scripts\start-orodruin-chat.ps1"
if (-not (Test-Path -LiteralPath $StartScript)) {
    Write-Error "Missing launcher: $StartScript"
}

$StartupDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Startup"
New-Item -ItemType Directory -Force -Path $StartupDir | Out-Null
$ShortcutPath = Join-Path $StartupDir "orodruin-chat.lnk"

$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = "powershell.exe"
$Shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$StartScript`" -Preset `"$Preset`""
$Shortcut.WorkingDirectory = $Root
$Shortcut.WindowStyle = 7
$Shortcut.Description = "Start orodruin dashboard/chat on login"
$Shortcut.Save()

Write-Host "Installed autostart shortcut:"
Write-Host "  $ShortcutPath"
Write-Host "Preset: $Preset"
Write-Host "Log off/on or reboot to open orodruin after login."
Write-Host "To remove: scripts\uninstall-orodruin-autostart.ps1"
