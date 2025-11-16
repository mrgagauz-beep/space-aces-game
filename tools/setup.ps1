# tools/setup.ps1
param(
  [switch]$Quiet
)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
$root = Resolve-Path "$here\.."

if (-not $Quiet) { Write-Host "Setting up hooksPath → .githooks" -ForegroundColor Cyan }
try {
  git config core.hooksPath ".githooks"
  if (-not $Quiet) { Write-Host "OK: git hooksPath configured." -ForegroundColor Green }
} catch {
  if (-not $Quiet) { Write-Warning "Git не найден или не репозиторий. Пропускаем настройку hooksPath." }
}

# Ничего критичного, просто напоминание
if (-not $Quiet) {
  Write-Host "Готово. Проверь файлы в /docs и запусти: python tools/make_handoff.py" -ForegroundColor Green
}
