param(
    [switch]$SyncVenv,
    [switch]$SkipVenv,
    [switch]$ProfileOnly
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath (Join-Path $Root "pyproject.toml"))) {
    if ($env:SOPHON_WINDOWS_ROOT -and (Test-Path -LiteralPath $env:SOPHON_WINDOWS_ROOT)) {
        $Root = $env:SOPHON_WINDOWS_ROOT
    } else {
        $Root = "C:\Software\Python\NLP\Personal\sophon"
    }
}

Set-Location -LiteralPath $Root
$env:UV_LINK_MODE = "copy"

if (-not $ProfileOnly) {
    $Cli = Join-Path $Root ".venv\Scripts\sophon-cli.exe"
    $NeedSync = $SyncVenv -or (-not $SkipVenv -and -not (Test-Path -LiteralPath $Cli))
    if ($NeedSync) {
        Write-Host "uv sync --extra tui --extra finetune (root=$Root)"
        uv sync --extra tui --extra finetune
        if ($LASTEXITCODE -ne 0) {
            throw "uv sync failed with exit $LASTEXITCODE"
        }
    } else {
        Write-Host "Windows venv already present at $Cli"
    }
}

$Cli = Join-Path $Root ".venv\Scripts\sophon-cli.exe"
if (Test-Path -LiteralPath $Cli) {
    & $Cli tui-profiles install
} else {
    Write-Host "sophon-cli.exe missing. Profile install skipped until uv sync finishes."
}
