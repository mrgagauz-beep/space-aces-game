# .githooks/pre-commit.ps1
$ErrorActionPreference = "SilentlyContinue"

function HasPlaceholder($path) {
  if (Test-Path $path) {
    $content = Get-Content $path -Raw -Encoding UTF8
    return ($content -match '<<<|FILL ME|ВСТАВЬ СЮДА')
  }
  return $false
}

$warn = @()

if (HasPlaceholder "docs\CONTEXT_SNAPSHOT.md") { $warn += "Заполни docs\CONTEXT_SNAPSHOT.md" }
if (HasPlaceholder "docs\DECISIONS.md")        { $warn += "Заполни docs\DECISIONS.md" }

if ($warn.Count -gt 0) {
  Write-Host "[pre-commit] Напоминание:" -ForegroundColor Yellow
  $warn | ForEach-Object { Write-Host " - $_" -ForegroundColor Yellow }
  Write-Host "Совет: в конце сессии запускай `python tools/make_handoff.py`." -ForegroundColor Yellow
}

exit 0  # не блокируем коммит
