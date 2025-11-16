# HOWTO_NEW_CHAT.md — как стартовать новый чат

1) Открой `/handoff/context_clipboard.txt` (если файла нет — запусти `python tools/make_handoff.py`).
2) Скопируй всё содержимое и вставь как ПЕРВОЕ сообщение в новом чате (внутри правильного Project).
3) Для задач пользуйся `/docs/playbooks/CODEX_PROMPT_TEMPLATE.md`.
4) В конце сессии:
   - Обнови `/docs/ITERATION_LOG.md` (одна запись на сессию).
   - При необходимости обнови `/docs/CONTEXT_SNAPSHOT.md`.
   - Запусти `python tools/make_handoff.py`.
