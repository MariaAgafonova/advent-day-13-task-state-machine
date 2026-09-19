# Advent Day 12 — персонализация ассистента поверх модели памяти

Проект продолжает День 11: сохраняет четыре стратегии управления контекстом и
добавляет изолированный долговременный профиль пользователя. Профиль загружается
перед каждым вызовом LLM и передаётся один раз в system prompt.

## День 12: что добавлено

- `profile.py` — `UserProfile`, JSON-репозиторий, partial update, default-профиль и три demo-профиля;
- `agent.py` — загрузка профиля на каждый запрос, overrides текущего запроса и безопасные request-level логи;
- `web.py` — API профилей и `POST /api/compare` для сравнения двух профилей одним запросом;
- `templates/index.html` — две панели ответа с выбором профиля для side-by-side сравнения;
- `personalization_compare.py` — одинаковый запрос для beginner, developer и manager;
- `tests/test_profile.py` и `tests/test_personalization.py` — persistence, изоляция и порядок контекста.

Профили лежат отдельно от памяти и истории: `data/profiles/{userId}.json`.
В JSON сохраняются только поля `UserProfile`; пароли, токены, API-ключи,
банковские и медицинские данные моделью не поддерживаются и не записываются.

### Архитектура контекста

```text
current userId
     │
     ├── UserProfileRepository ── data/profiles/{userId}.json
     │
     └── ChatAgent.ask()
          ├── 1. базовые инструкции агента
          ├── 2. USER PROFILE (один блок system prompt)
          ├── 3. LONG-TERM MEMORY (постоянные сведения)
          ├── 4. WORKING MEMORY (текущая задача)
          ├── 5. SHORT-TERM MEMORY (текущий диалог)
          ├── 6. история стратегии контекста
          └── 7. текущий user query — последнее сообщение
```

Приоритет: системные правила и безопасность > явное требование текущего запроса
> профиль > long-term memory > настройки по умолчанию. Например, слово
`подробно` или явный `profile_overrides={"responseLength": "detailed"}`
В system prompt также передаётся явное правило: полный финальный ответ должен быть
на языке из профиля, если текущий запрос явно не требует другой язык.
переопределяет профиль с короткими ответами. Полный профиль не копируется в
историю сообщений.

### Модель профиля

```json
{
  "id": "user_1",
  "name": "Maria",
  "language": "ru",
  "expertiseLevel": "middle_android_developer",
  "responseStyle": "clear_and_practical",
  "preferredFormat": "steps_and_code_examples",
  "responseLength": "medium",
  "interests": ["Android", "Kotlin", "AI"],
  "restrictions": ["avoid_unnecessary_theory"],
  "customInstructions": [
    "Use Kotlin for programming examples",
    "Explain unfamiliar AI terms"
  ]
}
```

Встроенные профили для сравнительного прогона: `beginner`, `developer`, `manager`.
Их можно выбрать в веб-интерфейсе или загрузить через `GET /api/profile`.

В блоке **«Сравнить ответы профилей»** выберите профиль левого и правого окна,
введите общий запрос и нажмите **«Сравнить»**. UI отправляет один текст в
`POST /api/compare`; backend выполняет два независимых LLM-вызова с одинаковым
последним user-message и разными profile blocks. История текущего пользователя
не используется для comparison-вызовов, поэтому ответы сравниваются только по
выбранным настройкам профилей.

Сравнение поддерживает многоходовый диалог: повторная отправка формы использует
отдельную short-term историю левого и правого окна. Поэтому follow-up получает
контекст предыдущих сообщений внутри каждого выбранного профиля. Кнопка
`Очистить` вызывает `POST /api/compare/reset` и начинает сравнение заново.

Пример API-запроса:

```json
{
  "question": "Объясни, как добавить память в AI-агента",
  "left_user_id": "beginner",
  "right_user_id": "developer"
}
```

Ответ содержит `left` и `right` с профилем, ответом и request-level метриками.

## Запуск

```powershell
cd D:\AIAdvent\advent-day-12-personalization
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Для локального сравнения без ключа:

```powershell
python -X utf8 personalization_compare.py --offline
```

Для реального DeepSeek-запроса заполните `DEEPSEEK_API_KEY` в `.env` и запустите:

```powershell
python -X utf8 web.py
```

Затем откройте <http://127.0.0.1:5000>. Профиль и память можно проверить в
панелях справа, а request-level логи показывают `user_id`, `profile_id`,
эффективные настройки, overrides и использованные слои памяти. Секреты в эти
логи не попадают.

## Сравнительное тестирование

Запрос для всех профилей один и тот же: **«Объясни, как добавить память в AI-агента»**.
Команда `--offline` использует детерминированный LLM double, поэтому результат
воспроизводим без API-ключа; без `--offline` используется DeepSeek.

| Профиль | Язык и длина | Формат и детализация | Наблюдаемый результат |
|---|---|---|---|
| beginner | русский, detailed | пошагово, простые слова | объясняет short-term, working и long-term по четырём шагам |
| developer | English, short | code-first, без базовой теории | краткая схема `messages` и пример Python-кода |
| manager | русский, medium | summary + bullets, без реализации | польза, этапы, риски и контроль |

Примеры ответов offline-прогона:

**beginner** — «Память помогает агенту не терять важные сведения между сообщениями. 1) Сохраняйте последние сообщения в краткосрочной памяти. 2) Данные текущей задачи держите в working memory. 3) Устойчивые факты пользователя переносите в long-term memory. 4) Перед ответом соберите эти слои и передайте их модели.»

**developer** —

```python
messages = [system, profile, long_term, working, *short_term, user_query]
response = client.chat.completions.create(messages=messages)
```

Persist only durable facts; keep task state and the sliding dialogue window separate.

**manager** — «Память делает ответы последовательными. Польза — персональный
диалог; этапы — определить данные, сроки хранения и подключить контекст; риски —
лишние данные, ошибки изоляции и рост стоимости; контроль — удаление и проверки.»

## Проверки и демонстрационный сценарий

```powershell
python -B -m unittest discover -s tests -v
python -X utf8 personalization_compare.py --offline --output data/personalization_comparison.json
```

Сценарий persistence: сохранить для `user_1` `language=ru` и
`preferredFormat=step_by_step`, перезапустить приложение и отправить обычный
запрос. Новый `ChatAgent` загрузит тот же JSON и снова передаст русский язык и
пошаговый формат в system prompt. Профиль хранится отдельно от short-term,
working и long-term memory; смена userId не переносит настройки другого пользователя.

Оригинальное описание слоёв и стратегий Дня 11 сохранено ниже.

Проект сравнивает четыре стратегии управления контекстом без summary:

- `sliding_window` — в запрос попадают только последние `N` сообщений;
- `facts` — структурированные sticky facts плюс последние `N` сообщений;
- `branching` — checkpoint и независимые ветки диалога;
- `retrieval` — текущий вопрос плюс top-K релевантных фрагментов из истории; последние `N` сообщений не используются.

## Запуск

```powershell
cd D:\AIAdvent\advent-day-11-memory-layers
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Выполнять только если .env ещё не создан:
# Copy-Item .env.example .env
```

Для запуска веб-интерфейса:

```powershell
python web.py
```

В `.env` укажите тот же локальный `DEEPSEEK_API_KEY`, что используется в day 10. Ключ не хранится в исходниках и не должен попадать в Git.

CLI:

```powershell
python -X utf8 main.py --strategy sliding_window --recent 6
python -X utf8 main.py --strategy facts
python -X utf8 main.py --strategy branching
python -X utf8 main.py --strategy retrieval
```

CLI-команды:

```text
/strategy <sliding_window|facts|branching|retrieval>
/checkpoint <name>
/branch create <name> [checkpoint-id]
/branch switch <branch-id>
/branches
/facts
/retrieved
/analytics
/reset
/exit
```

Веб-интерфейс:

```powershell
python -X utf8 web.py
```

Открыть <http://127.0.0.1:5000>.

## Сравнительный прогон

`compare.py` выполняет один и тот же сценарий сбора ТЗ на всех стратегиях и сохраняет реальные ответы и метрики:

```powershell
python -X utf8 compare.py --output data/comparison.json
```

Для Facts учитываются не только основной запрос, но и дополнительный вызов DeepSeek для обновления facts. В итоговую стоимость входят оба типа запросов.

## Архитектура

- `agent.py` — общий агент, вызовы DeepSeek, проверка лимита контекста и метрики;
- `context_strategies.py` — Sliding Window, Facts и Branching;
- `retrieval.py` — локальный TF-IDF retriever и Retrieval strategy;
- `analytics.py` — request-level и strategy-level агрегаты;
- `scenario.py` — единый сценарий на 12 сообщений;
- `compare.py` — запуск сравнения;
- `main.py` — CLI;
- `web.py` и `templates/index.html` — UI/API;
- `tests/` — unit-тесты стратегий и агента.

Локальный retriever не требует отдельного embedding API и делает эксперимент воспроизводимым. Его можно заменить реализацией, совместимой с интерфейсом `Retriever`.

## Тесты

```powershell
python -X utf8 -m unittest discover -s tests -v
```

## Day 11 — Memory Layers

This folder is a separate copy of the previous Flask UI/API. The Day 10 project is
not modified.

### Retention rules

- Every user and assistant message enters `short_term`; it is bounded by a
  sliding window and removed after `MEMORY_SHORT_TERM_TTL_SECONDS`.
- A task step, intermediate calculation, constraint, or selected parameter is
  also stored in `working`. It is removed by **Завершить задачу**.
- An explicit preference, profile fact, recurring pattern, or reusable past
  solution is also stored in `long_term`. It is upserted, ranked by importance
  and lexical relevance, and survives a new agent instance in JSON mode.
- Context is assembled as `working > short_term > long_term`. Long-term data is
  included only when it matches the current question, unless there is no match.

Memory layers are isolated records. When a message is classified as both
short-term and long-term, the long-term layer receives its own copy at write
time; it does not read the value back from short-term. Clearing one layer
therefore does not affect the others. **Очистить всё** is the only operation
that clears all layers together.

For a chat request, the UI separates two choices. **Контекст ответа** selects
the active memory view and its supporting layers:

- Short-term — working + short-term + relevant long-term records;
- Working — working + relevant long-term records, without short-term dialogue;
- Long-term — only long-term records.

**Куда сохранить сообщение** independently selects where the new message gets
an additional semantic record. The complete conversation history is not
appended to scoped prompts. The internal auto mode remains available for
legacy/API compatibility and uses all layers plus the strategy history.

### Explicit retention target

The chat composer provides a **Куда сохранить сообщение** selector with three options:

- Working memory — force a task record;
- Long-term memory — force a durable knowledge record;
- Только short-term — keep the message only in the current dialogue window.

Every message still enters short-term as dialogue. An explicit target only
controls the additional semantic record and has priority over automatic
classification. The UI defaults to short-term-only; explicit records are marked
as user:explicit in the API state and memory panel.

The **Очистить чат** button clears the visible dialogue and all short-term
records. Working and long-term records remain available for subsequent scoped
requests. When the browser page is closed, a `pagehide` request clears Working
memory for that session. Long-term memory is not cleared by either operation.

### Storage choice

The web UI exposes two modes:

- `in_memory` — process-only memory, useful for a private experiment;
- `json_file` — atomic per-conversation JSON files under `data/memory/`; only
  `long_term` is persisted, while `short_term` and `working` stay session-local.

The copied local `.env` keeps the existing `DEEPSEEK_API_KEY` configuration out
of source code. The key is never returned by the API or written to memory.

Run the day 11 UI from this folder:

```powershell
cd D:\AIAdvent\advent-day-11-memory-layers
python -X utf8 web.py
```

The **Память агента** panel shows saved records by layer, importance, expiry,
the last classification, and the exact records selected for the next prompt.
It also provides per-record **Забыть** actions and task completion.

The chat composer also has a microphone button. In Chrome or Edge, allow
microphone access, speak, and stop recording; the browser converts speech to
text and sends it to the same chat endpoint. Audio is not stored by the app.

### Scenario analysis

The regression tests model the required flows: ordinary dialogue goes to
short-term, a selected task parameter goes to working, and an explicit
preference goes to long-term. They also verify priority ordering, TTL, task
cleanup, JSON persistence, and the web endpoints. The implementation reports
memory coverage and selection, but a claim that answers are objectively better
requires an A/B run against a baseline; this is intentionally not guessed from
one response.
