param(
    [string]$Preset = "llama2_7b_chat"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath $Root)) {
    if ($env:SOPHON_WINDOWS_ROOT -and (Test-Path -LiteralPath $env:SOPHON_WINDOWS_ROOT)) {
        $Root = $env:SOPHON_WINDOWS_ROOT
    } else {
        $Root = "C:\Software\Python\NLP\Personal\sophon"
    }
}

$StartScript = Join-Path $Root "scripts\start-sophon-chat.ps1"
if (-not (Test-Path -LiteralPath $StartScript)) {
    Write-Error "Missing launcher: $StartScript"
}

$Pwsh = Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps\pwsh.exe"
if (-not (Test-Path -LiteralPath $Pwsh)) {
    $PwshCmd = Get-Command "pwsh.exe" -ErrorAction SilentlyContinue
    if ($null -eq $PwshCmd) {
        Write-Error "Store PowerShell (pwsh.exe) was not found."
    }
    $Pwsh = $PwshCmd.Source
}

$StartupDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Startup"
New-Item -ItemType Directory -Force -Path $StartupDir | Out-Null
$ShortcutPath = Join-Path $StartupDir "sophon-chat.lnk"

$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $Pwsh
$Shortcut.Arguments = "-NoLogo -NoProfile -ExecutionPolicy Bypass -File `"$StartScript`" -Preset `"$Preset`""
$Shortcut.WorkingDirectory = $Root
$Shortcut.WindowStyle = 7
$Shortcut.Description = "Start sophon dashboard/chat on login"
$Shortcut.Save()

Write-Host "Installed autostart shortcut:"
Write-Host "  $ShortcutPath"
Write-Host "Preset: $Preset"
Write-Host "Launcher: $Pwsh"
Write-Host "Log off/on or reboot to open sophon after login."
Write-Host "To remove: scripts\uninstall-sophon-autostart.ps1"
