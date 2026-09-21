param()

$ErrorActionPreference = "Stop"
$Here = $PSScriptRoot
if (-not $Here) {
    $Here = Split-Path -Parent $MyInvocation.MyCommand.Path
}

$Compose = Join-Path $Here "docker-compose.yml"
if (-not (Test-Path -LiteralPath $Compose)) {
    Write-Error "Missing $Compose"
}

$Docker = Get-Command "docker.exe" -ErrorAction SilentlyContinue
if ($null -eq $Docker) {
    $Fallback = "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
    if (Test-Path -LiteralPath $Fallback) {
        $DockerPath = $Fallback
    } else {
        Write-Error "docker.exe not found. Start Docker Desktop, then rerun this script."
    }
} else {
    $DockerPath = $Docker.Source
}

& $DockerPath compose -f $Compose up -d
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "SearXNG listening on http://127.0.0.1:8888"
Write-Host "Set SOPHON_SEARXNG_URL=http://127.0.0.1:8888 in .env and restart Nexus."
