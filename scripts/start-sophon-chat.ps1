param(
    [string]$Preset = $env:SOPHON_CHAT_PRESET,
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"

if (-not $Preset -or $Preset.Trim() -eq "") {
    $Preset = "llama2_7b_chat"
}

$DefaultRoot = "C:\Software\Python\NLP\Personal\sophon"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath $Root)) {
    if ($env:SOPHON_WINDOWS_ROOT -and (Test-Path -LiteralPath $env:SOPHON_WINDOWS_ROOT)) {
        $Root = $env:SOPHON_WINDOWS_ROOT
    } else {
        $Root = $DefaultRoot
    }
}

if (-not (Test-Path -LiteralPath $Root)) {
    Write-Error @"
sophon deploy root is missing: $Root

Windows Terminal 0x8007010b means startingDirectory no longer exists.
From the WSL checkout, recreate the tree and rewrite the profile:

  sophon-cli deploy-windows --sync-venv

Or set SOPHON_WINDOWS_ROOT to an existing Windows checkout, then:

  sophon-cli deploy-windows --profile-only
"@
}

$Cli = Join-Path $Root ".venv\Scripts\sophon-cli.exe"
if (-not (Test-Path -LiteralPath $Cli)) {
    Write-Error "sophon-cli.exe not found at $Cli. From WSL run: sophon-cli deploy-windows --sync-venv"
}

Set-Location -LiteralPath $Root
$env:UV_LINK_MODE = "copy"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONUTF8 = "1"
$env:SOPHON_SKIP_TERMINAL_GRAPHICS = "1"
if (-not $env:SOPHON_CHAT_BACKEND -or $env:SOPHON_CHAT_BACKEND.Trim() -eq "") {
    $env:SOPHON_CHAT_BACKEND = "auto"
}

if ($Foreground) {
    Write-Host "Starting sophon in this window (preset=$Preset; backend=$($env:SOPHON_CHAT_BACKEND))..."
    & $Cli chat --preset $Preset --no-spawn-window
    exit $LASTEXITCODE
}

$Wt = Get-Command "wt.exe" -ErrorAction SilentlyContinue
if ($null -ne $Wt) {
    $env:SOPHON_CHAT_PRESET = $Preset
    Start-Process -FilePath $Wt.Source -ArgumentList @(
        "-w", "0",
        "nt",
        "-p", "sophon",
        "-d", $Root
    ) | Out-Null
    Write-Host "Opened sophon in a new Windows Terminal tab (preset=$Preset)."
    Write-Host "This PowerShell prompt stays free. Use -Foreground to run in-place."
    exit 0
}

Start-Process -FilePath $Cli -ArgumentList @("chat", "--preset", $Preset, "--no-spawn-window") -WorkingDirectory $Root | Out-Null
Write-Host "Opened sophon in a new window (preset=$Preset)."
