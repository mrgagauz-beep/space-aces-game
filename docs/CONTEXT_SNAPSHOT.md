# CONTEXT_SNAPSHOT.md
Last updated: 2025-11-16

## One-liner
Space Aces: бот для фарма/боёв через CV + навигацию по миникарте.

## Текущее состояние
-  test/bot режимы (run.py)
-  MINIMAP: враги/игрок; MAIN: мобы/ящики/лейблы
-  Реальный парсинг HP/Shield (сейчас TODO в FSM._sense)
-  Навигация: ensure src/nav/navigation.py + подход на 80%
-  FLEESAFE(COOLDOWN позже): оформить как политики

## Фокус (Next steps)
1) Закрыть навигацию: navigation.py + стабильный player_mm.
2) HP/Shield парсер из HUD + пороги flee.
3) Engageлогика по лейблам (как в FSM уже заложено).
4) Self-test с метриками (Recall/FPR) на assets/screenshots.
