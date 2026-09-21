param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if ($null -eq $Rest -or $Rest.Count -eq 0) {
    & uv run sophon-system-check
} else {
    & uv run sophon-system-check @Rest
}
