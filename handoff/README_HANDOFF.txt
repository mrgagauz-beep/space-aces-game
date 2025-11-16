# README — handoff

Что это:
- `/tools/make_handoff.py` собирает компактный контекст (snapshot + 1–2 последние итерации + next steps)
  и сохраняет в `/handoff/context_clipboard.txt`.

Как использовать:
- В конце каждой сессии обнови `ITERATION_LOG.md` и, если нужно, `CONTEXT_SNAPSHOT.md`.
- Запусти: `python tools/make_handoff.py`.
- Содержимое `context_clipboard.txt` — это первое сообщение в новом чате.
