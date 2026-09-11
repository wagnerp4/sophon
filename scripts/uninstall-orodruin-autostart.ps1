$ErrorActionPreference = "Stop"

$StartupDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Startup"
$ShortcutPaths = @(
    (Join-Path $StartupDir "orodruin-chat.lnk"),
    (Join-Path $StartupDir "mithril-chat.lnk")
)

$removed = $false
foreach ($ShortcutPath in $ShortcutPaths) {
    if (Test-Path -LiteralPath $ShortcutPath) {
        Remove-Item -LiteralPath $ShortcutPath -Force
        Write-Host "Removed $ShortcutPath"
        $removed = $true
    }
}
if (-not $removed) {
    Write-Host "No orodruin/mithril autostart shortcut found under $StartupDir"
}
