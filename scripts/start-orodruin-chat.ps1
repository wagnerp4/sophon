param(
    [string]$Preset = $env:ORODRUIN_CHAT_PRESET,
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"

if (-not $Preset -or $Preset.Trim() -eq "") {
    $Preset = "llama2_7b_chat"
}

$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath $Root)) {
    $Root = "C:\Software\Python\NLP\Personal\orodruin"
}

$Cli = Join-Path $Root ".venv\Scripts\orodruin-cli.exe"
if (-not (Test-Path -LiteralPath $Cli)) {
    Write-Error "orodruin-cli.exe not found at $Cli. Run uv sync from PowerShell in the orodruin repo first."
}

Set-Location -LiteralPath $Root
$env:UV_LINK_MODE = "copy"
if (-not $env:ORODRUIN_CHAT_BACKEND -or $env:ORODRUIN_CHAT_BACKEND.Trim() -eq "") {
    $env:ORODRUIN_CHAT_BACKEND = "auto"
}

if ($Foreground) {
    Write-Host "Starting orodruin in this window (preset=$Preset; backend=$($env:ORODRUIN_CHAT_BACKEND))..."
    & $Cli chat --preset $Preset --no-spawn-window
    exit $LASTEXITCODE
}

$Wt = Get-Command "wt.exe" -ErrorAction SilentlyContinue
if ($null -ne $Wt) {
    Start-Process -FilePath $Wt.Source -ArgumentList @(
        "-w", "0",
        "nt",
        "--title", "orodruin",
        "-d", $Root,
        $Cli,
        "chat",
        "--preset", $Preset,
        "--no-spawn-window"
    ) | Out-Null
    Write-Host "Opened orodruin in a new Windows Terminal window (preset=$Preset)."
    Write-Host "This PowerShell prompt stays free. Use -Foreground to run in-place."
    exit 0
}

Start-Process -FilePath $Cli -ArgumentList @("chat", "--preset", $Preset, "--no-spawn-window") -WorkingDirectory $Root | Out-Null
Write-Host "Opened orodruin in a new window (preset=$Preset)."
