# Evidence-grounded QA: данные и классификатор answerability

Учебное исследование Research Methods in IT: **Improving Local LLM Reliability in
Evidence-Grounded Question Answering with an Answerability Classifier**.

На первом этапе загружаем настоящий `rajpurkar/squad_v2`, сохраняем исходные части
и проверяем данные. Обучение, подключение LLM, собственное разделение на
train/validation/test и балансировка в этот этап не входят.

## Запуск в Windows / PowerShell

Обнаружены Windows (build 26200), PowerShell 7.6.5 и Python 3.12.2.
Дополнительно обнаружен Python 3.12.14; проект использует Python 3.12.2.
Системные команды `python`/`py` не являются надёжными путями к установленному
интерпретатору на этой машине. Активация окружения не требуется.

```powershell
Set-Location C:\Projects\rmt
# Только если .venv ещё не существует:
& 'C:\Users\PC_SLAVA\AppData\Local\Programs\Python\Python312\python.exe' -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
& '.\.venv\Scripts\python.exe' scripts/download_dataset.py
```

Повторный запуск использует сохранённую локальную копию через `load_from_disk`.
Для запуска из любой рабочей папки:

```powershell
& 'C:\Projects\rmt\.venv\Scripts\python.exe' 'C:\Projects\rmt\scripts\download_dataset.py'
```

Первое скачивание требует доступа к Hugging Face. Кеши направлены в `.cache/`
проекта. При ошибке скрипт завершается с кодом 1 и сообщает причину; данные
не подменяются искусственными. Существующая копия не перезаписывается.

## Файлы

- `scripts/download_dataset.py` — загрузка, сохранение, повторное открытие и аудит.
- `data/raw/squad_v2/` — исходный `DatasetDict` в формате `save_to_disk`.
- `data/raw/squad_v2.provenance.json` — источник и время скачивания в UTC.
- `outputs/dataset_summary.json` — результаты последней проверки: UTC, режим
  загрузки, версии Python и datasets, исходные части, поля, классы, пустые строки,
  дубликаты ID, пересечения ID и доступные fingerprints. При повторном запуске
  сохраняется также результат первоначальной проверки записи и чтения всех данных.
- `outputs/dataset_examples.json` — первые три настоящих примера каждого класса
  из каждой исходной части, с `id`, `context`, `question`, `answers`, `answerable`.
- `requirements.txt` — фактически установленные версии прямых и транзитивных
  зависимостей из `pip freeze` внутри `.venv`.

`answerable = len(answers["text"]) > 0`. Пустота `context` и `question`
проверяется через `strip()` без изменения исходного текста. Для повторяющихся ID
отдельно считаются число различных повторённых ID и число лишних строк.

Правильные ответы сохраняются для будущего вычисления меток и оценивания.
**Во вход будущего классификатора должны передаваться только `context` и
`question`; `answers` и метка `answerable` туда не входят.**

Скачанные данные, окружение и кеши исключены из Git. Отчёты в `outputs/`
можно версионировать. Установка зависимостей выполняется только в `.venv`.

## Фактическая проверка 20 октября 2026

Использованы Python 3.12.2 и datasets 5.1.0. `pip check` не выявил конфликтов.

| Исходная часть | Строк | Answerable | Unanswerable |
| --- | ---: | ---: | ---: |
| train | 130319 | 86821 | 43498 |
| validation | 11873 | 5928 | 5945 |

Обе части содержат поля `id`, `title`, `context`, `question`, `answers`.
Пустых `context`/`question`, повторённых ID и пересечений ID между частями — 0.
При первом сохранении подтверждено полное совпадение Arrow-данных, числа строк,
полей и features после `save_to_disk` / `load_from_disk`.
Проверены 12 сохранённых примеров: по 3 каждого класса из каждой части;
все поля совпадают с настоящими строками сохранённого датасета.

Повторный запуск из `C:\Projects` с отключёнными сетевыми запросами Hugging Face
завершился успешно с `load_mode=local`. Команда этой проверки:

```powershell
Set-Location C:\Projects
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
& 'C:\Projects\rmt\.venv\Scripts\python.exe' 'C:\Projects\rmt\scripts\download_dataset.py'
# Вернуть обычный режим в текущем PowerShell после проверки:
Remove-Item Env:HF_HUB_OFFLINE, Env:HF_DATASETS_OFFLINE
```

Первоначальные попытки установки и скачивания в ограниченной среде завершились
ошибкой DNS; после разрешения сетевого доступа обе операции выполнены успешно.
Hugging Face предупредил о запросах без токена и недоступности symlink для кеша
Windows: скачивание работает, но кеш может занимать больше места. Эти
предупреждения не помешали проверкам. Изменения настроек Windows не требовались.

## Этап 2: наши экспериментальные выборки

`data/raw/squad_v2/` содержит неизменённое разделение **авторов SQuAD 2.0**:
исходные `train` (130319 строк) и `validation` (11873 строки). Имена этих частей
не означают их роль в нашем эксперименте.

`data/processed/squad_v2_answerability/` содержит **наши** выборки:

- `train` — обучение отдельного бинарного классификатора answerability.
- `validation` — выбор модели, настроек и порога классификации.
- `test` — только итоговая оценка после фиксации модели, настроек и порога,
  включая последующее сравнение локальной LLM с фильтром и без фильтра.

**Запрещено подбирать модель, признаки, настройки, prompts или порог на test.**
Скрипт выводит только агрегированные характеристики test. Не следует выводить
его отдельные вопросы, тексты или ответы при разработке. Проверки целостности
и состава данных не являются оценкой качества модели.

### Алгоритм

Источник читается только через `load_from_disk`; повторного скачивания нет.
К исходным полям добавляются:

- `label`: целое 1 при непустом `answers["text"]`, иначе 0.
- `source_split`: исходное имя `train` или `validation`.
- `context_group`: SHA-256 от UTF-8 строки
  `" ".join(unicodedata.normalize("NFC", context).split())`.

Нормализация применяется только для группировки. Исходные `context`, `question`,
`answers` и другие поля сохраняются без изменений. Вход будущего классификатора —
**только `question` и `context`**; `answers`, метки, ID и служебные поля во вход
не передаются. Ответы нужны для меток и последующего оценивания.

1. Определяются общие группы исходных train и validation. Все строки таких групп
   исключаются из исходного train; исходный validation имеет приоритет целиком,
   включая строки, которые не войдут в test.
2. Оставшиеся группы сортируются по идентификатору, затем перемешиваются отдельным
   `random.Random(42)`. Первые `max(1, min(n-1, int(n*0.1+0.5)))` групп становятся
   нашей validation, остальные — нашим train. Сохраняется исходный порядок строк
   внутри каждой части. Балансировка или удаление строк ради долей классов не
   применяется. Доля строк может немного отличаться от 90/10.
3. Независимый `random.Random(42)` выбирает без повторов 500 индексов label=1,
   затем 500 индексов label=0 из исходного validation. Кандидаты перечислены в
   исходном порядке. Объединённый список перемешивается тем же генератором test.
   Длина, сложность и ответы моделей не участвуют в отборе. Остальные 10873 строки
   исходного validation в экспериментальные выборки не включены.

Разделение по нормализованным текстам **не гарантирует разделения по статьям или
темам** и не объединяет семантически похожие, но текстуально разные контексты.
Баланс test 50/50 задан экспериментом; он не отражает естественную долю классов
во всех исходных данных.

### Команды PowerShell

```powershell
Set-Location C:\Projects\rmt
& '.\.venv\Scripts\python.exe' scripts/prepare_splits.py
```

Активация `.venv` и изменение Execution Policy не нужны. Из любой рабочей папки:

```powershell
& 'C:\Projects\rmt\.venv\Scripts\python.exe' 'C:\Projects\rmt\scripts\prepare_splits.py'
```

Повторный запуск заново вычисляет ожидаемое разделение, проверяет все сохранённые
поля, ID и их порядок, настройки, версии библиотек, fingerprints и SHA-256 файлов
источника. При совпадении используется существующий DatasetDict без его перезаписи
(`run_mode=reuse`); отчёты актуализируются. Повреждённые или отличающиеся данные,
метаданные и отчёты вызывают ошибку. Отсутствующие отчёты можно восстановить
повторным запуском, если сохранённый DatasetDict и его метаданные проходят проверки.

Для **намеренного пересоздания**:

```powershell
& '.\.venv\Scripts\python.exe' scripts/prepare_splits.py --overwrite
```

При `--overwrite` сначала создаётся и проверяется новая копия. Предыдущая папка
processed сохраняется рядом с суффиксом `.backup-<UTC>`, затем публикуется новая.
Исходные raw-файлы не перезаписываются. Незавершённые папки
`.squad_v2_answerability.incomplete-*` сохраняются для диагностики. Удалять резервные
копии следует отдельно после проверки результата.

### Результаты и файлы этапа 2

| Наша выборка | Строк | Answerable (1) | Unanswerable (0) | Уникальных context_group |
| --- | ---: | ---: | ---: | ---: |
| train | 117256 | 78151 | 39105 | 17126 |
| validation | 13063 | 8670 | 4393 | 1903 |
| test | 1000 | 500 | 500 | 695 |

Между исходными частями найдено 0 общих групп; исключено из исходного train
0 строк и 0 групп. Проверки всех трёх пар наших выборок не обнаружили пересечений
ID или context_group. Пустых context/question и ошибок label нет. Все сохранённые
поля совпадают с исходными и вычисленными полями после чтения с диска. Повторное
построение с seed=42 даёт те же ID в том же порядке. SHA-256 исходных файлов
проверяются до и после работы, подтверждая отсутствие изменений raw.

Фактическое соотношение строк train/validation — 89.976% / 10.024%.
Повторный запуск из другой рабочей папки завершился с `run_mode=reuse`.
На временной копии настоящих данных дополнительно проверены отказ при
несовпадении метаданных, отказ при отчётах без DatasetDict и пересоздание с
резервной копией при `--overwrite`. Проверки не изменили рабочие raw/processed
или их отчёты. Контрольные суммы списков ID проверены независимым вычислением.

- `scripts/prepare_splits.py` — подготовка, проверки и безопасный повторный запуск.
- `data/processed/squad_v2_answerability/` — наш DatasetDict для `load_from_disk`;
  внутри `preparation_metadata.json` хранится идентичность источника и настроек.
- `outputs/split_summary.json` — UTC-время, режим запуска, источник, fingerprints
  и SHA-256 исходных файлов, настройки, версии Python/datasets/pyarrow, размеры,
  классы, группы, исключения, неиспользованные строки и результаты проверок.
- `outputs/split_manifest.json` — ID каждой нашей выборки в сохранённом порядке
  и их SHA-256. Хешируется UTF-8 JSON-массив ID с `ensure_ascii=False` и
  `separators=(",", ":")`, без завершающего перевода строки.

Новые зависимости не устанавливались: используются Python 3.12.2, datasets 5.1.0,
pyarrow 26.0.0 и стандартная библиотека Python. Алгоритм и версии записываются в
метаданные; изменение источника, алгоритма или версий требует явного `--overwrite`.
Обучение моделей и запуск LLM на этом этапе не выполняются.

## Этап 3: первый answerability-классификатор

Дообучен исходный `distilbert/distilbert-base-uncased` с новой бинарной
классификационной головой (`AutoModelForSequenceClassification`, `num_labels=2`).
Готовые SQuAD-модели и генеративные LLM не используются. Метки: 0 — unanswerable,
1 — answerable. Стандартная cross-entropy без class weights и label smoothing.

### Установка в Windows / PowerShell

Все команды выполняются из `C:\Projects\rmt` напрямую через Python `.venv`.
Активация и изменение Execution Policy не нужны.

```powershell
& '.\.venv\Scripts\python.exe' -m pip install 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
# Фактически выполненная первичная установка остальных библиотек:
& '.\.venv\Scripts\python.exe' -m pip install transformers accelerate scikit-learn
& '.\.venv\Scripts\python.exe' -m pip check
```

Для воспроизведения зафиксированных версий после установки PyTorch используйте
`-m pip install -r requirements.txt` вместо команды без версий. CUDA-сборка PyTorch
получена из [официального индекса cu128](https://download.pytorch.org/whl/cu128).
Системный CUDA Toolkit, torchvision и torchaudio не устанавливаются.

Зафиксированы torch 2.11.0+cu128 (CUDA runtime 12.8), transformers 5.20.2,
accelerate 1.15.0, scikit-learn 1.9.1. Остальные версии — в `requirements.txt`.
Скрипт проверяет доступность CUDA, небольшую операцию с `backward()` и mixed
precision; при недоступности GPU останавливается, не переходя к обучению на CPU.

### Запуск

Параметры находятся в `configs/classifier_distilbert.json`; основные параметры
можно явно переопределить через CLI (`--help`). Каждый запуск получает новый
`run_id`; существующие результаты не перезаписываются. Перед новым полным запуском
нужно отдельно перенести уже существующую папку итоговой модели.

```powershell
# Автоматически: диагностика, отдельный smoke run, затем новая модель и полный run.
& '.\.venv\Scripts\python.exe' -u scripts/train_classifier.py --stage all

# Либо отдельно проверить smoke и затем запустить полный run с тем же ID:
& '.\.venv\Scripts\python.exe' -u scripts/train_classifier.py --stage smoke --run-id my_distilbert_run
& '.\.venv\Scripts\python.exe' -u scripts/train_classifier.py --stage full --run-id my_distilbert_run
```

Если полное обучение уже завершено (это подтверждает `full/training_summary.json`),
но требуется повторить проверку экспорта, доступен `--stage finalize --run-id ...`.
Этот режим загружает выбранный checkpoint, заново оценивает validation с тем же
порогом и сохраняет/проверяет модель; обучение не повторяется. Существующую итоговую
модель он также не перезаписывает. Проверка FP32 logits обходит оставшуюся на
обученной модели обёртку autocast Accelerate, чтобы сравнивать одинаковую точность.

Исходный checkpoint и tokenizer фиксируются по одному точному commit SHA из
Hugging Face. В конфигурации закреплён SHA
`12040accade4e8a0f71eabdb258fecc2e7e948be`; для повторения с уже скачанным
checkpoint можно добавить `--local-files-only`. Модель smoke обучается 8 шагов
на 256 строках train с оценкой на 128
строках validation. Её результаты сохраняются отдельно и не являются итоговыми.
Перед полным запуском заново создаётся модель из исходного checkpoint и заново
устанавливается seed=42 для Python, NumPy, PyTorch, Trainer и sampler. Веса smoke
не переносятся. При OOM запуск начинается заново от исходного checkpoint с меньшим
batch size и большим accumulation; номинальный effective batch остаётся 16.

### Вход, обрезка и настройки

Скрипт отдельно загружает только `data/processed/squad_v2_answerability/train/`
и `validation/` через `load_from_disk`, не загружая корневой DatasetDict или test.
Используется `tokenizer(question, context, max_length=512,
truncation="only_second")`, динамический padding и одна последовательность на
пример. В модель поступают только `input_ids`, `attention_mask` и отдельные
`labels` для loss. Для DistilBERT явно задаётся `return_token_type_ids=False`,
поскольку модель не принимает token type IDs. Дальнейшее локальное применение
должно использовать те же параметры токенизации.
`answers`, `title`, `id`, `source_split`, `context_group` и
текстовое представление метки во вход не включаются.

До обучения для train и validation считаются число и доля обрезанных контекстов.
Offsets fast-tokenizer используются только для диагностики: проверяется, остался
ли хотя бы один целиком видимый корректный gold answer span. Дополнительно
учитываются невалидные span (несовпадение с исходной подстрокой). Ответы не влияют
на выбор фрагмента, разметка после обрезки не меняется. Обрезка способна оставить
label=1 у примера, где ответ в видимой части уже отсутствует. Метки сохраняются
независимо от видимости ответа после обрезки; это ограничение первого подхода.

Первый полный запуск: 2 эпохи, learning rate 2e-5, weight decay 0.01, warmup 10%,
train batch=8, accumulation=2, eval batch=8, max_length=512, workers=0,
AdamW, linear scheduler, max_grad_norm=1.0. В API transformers 5.20 доля warmup
записывается как `warmup_steps=0.1`; фактически это `ceil(0.1 * total_steps)`.
Для DistilBERT явно установлено `accepts_loss_kwargs=False`: его стандартная CE
усредняется по микробатчу и не нормализуется самим model.forward через
`num_items_in_batch`. Это заставляет Trainer правильно делить loss при accumulation.
Перед smoke выполняется численное сравнение градиентов общего batch из 4 строк
train и двух микробатчей (dropout отключён только в этой проверке).
Режим mixed precision фиксируется в артефактах после реальной проверки GPU.
Внешние сервисы логирования отключены. Оценка и checkpoint — после каждой эпохи,
сохраняется максимум 2 checkpoint; лучший выбирается по validation macro-F1.

Итоговые метрики лучшего checkpoint вычисляются на всей validation при
**фиксированном пороге `p(answerable) >= 0.5`**. Порог не подбирается.
Сохраняются accuracy, balanced accuracy, precision/recall/F1/support обоих классов,
macro-F1, ROC-AUC и confusion matrix (строки — истинные метки, столбцы — прогноз,
порядок `[0, 1]`). Те же метрики вычисляются для правила «всегда answerable».

### Артефакты и ограничения

В `outputs/classifier/<run_id>/` сохраняются конфигурация, версии, точные команды
установки, commit SHA модели и tokenizer, сведения GPU/precision, SHA-256 файлов
train/validation и диагностика обрезки. Подпапки `smoke/` и `full/` разделяют
проверочный и полный запуск; внутри — история loss/metrics, состояние Trainer,
фактические настройки, длительность и checkpoints. `status.json` явно отличает
завершённый запуск от неудачного или только smoke.

После успешного полного обучения:

- `models/answerability_distilbert/` — лучшая модель, tokenizer и
  `training_provenance.json` с исходным запуском.
- `validation_metrics.json` — модель и правило «всегда answerable».
- `validation_predictions.jsonl` — ID, правильная метка, вероятность и прогноз.
- `validation_errors.json` — до 10 настоящих ошибок validation, по возможности
  оба направления; примеры не отбираются из test.
- `save_load_verification.json` — сравнение logits и классов до/после локальной
  загрузки на 8 строках validation: float32 eval, atol=1e-5, rtol=1e-4.

Веса, временные checkpoint и tokenization cache исключены из Git. Ограничения:
потеря информации при обрезке, естественный дисбаланс train/validation,
единственный seed и разделение по текстам, а не обязательно по темам/статьям.
Validation используется для разработки и выбора checkpoint, поэтому её метрики
**не являются окончательной независимой оценкой**. Это оценка классификатора,
не проверка H1/H2 и не оценка полной системы с генеративной LLM.
**Test остаётся нетронутым и не используется на этапе 3.**

### Фактический результат первого полного обучения

Завершённый run: `20261020T173000Z_distilbert_seed42`.
Обучены все 117256 строк train, оценены все 13063 строки validation.
Завершены 2 эпохи / 14658 optimizer steps за 1220.97 секунды (около 20 минут
21 секунды). BF16, train batch=8, accumulation=2 (номинально 16 примеров на update),
eval batch=8; OOM и изменений batch size не было. В конце эпохи последний неполный
accumulation batch сохраняется, чтобы не исключать строки train.
Проверка CUDA forward/backward и отдельный smoke run из 8 шагов прошли.
Численная проверка accumulation на train дала максимальное различие градиентов
`1.564621925354004e-07`.

Лучший checkpoint — `full/attempt_1/checkpoints/checkpoint-14658` (эпоха 2).
Macro-F1 validation по эпохам: 0.7842476336 → 0.8030656355.
Checkpoint и tokenizer исходной модели имеют SHA
`12040accade4e8a0f71eabdb258fecc2e7e948be`.

| Метрика validation, threshold=0.5 | DistilBERT | Всегда answerable |
| --- | ---: | ---: |
| Accuracy | 0.832045 | 0.663707 |
| Balanced accuracy | 0.791272 | 0.500000 |
| Macro-F1 | 0.803066 | 0.398932 |
| ROC-AUC | 0.907504 | 0.500000 |

| Метод | Класс | Precision | Recall | F1 | Support |
| --- | --- | ---: | ---: | ---: | ---: |
| DistilBERT | 0 (unanswerable) | 0.800492 | 0.666743 | 0.727521 | 4393 |
| DistilBERT | 1 (answerable) | 0.844322 | 0.915802 | 0.878610 | 8670 |
| Всегда answerable | 0 | 0 | 0 | 0 | 4393 |
| Всегда answerable | 1 | 0.663707 | 1.000000 | 0.797865 | 8670 |

Confusion matrix модели: `[[2929, 1464], [730, 7940]]`;
строки — истинные классы `[0,1]`, столбцы — предсказанные `[0,1]`.
1464 unanswerable ошибочно признаны answerable; 730 answerable ошибочно признаны
unanswerable. Для правила «всегда answerable»: `[[0,4393],[0,8670]]`.

| Диагностика обрезки | Train | Validation |
| --- | ---: | ---: |
| Обрезанные context / всего | 165 / 117256 (0.14072%) | 19 / 13063 (0.14545%) |
| Answerable без целиком видимого gold span / всех answerable | 4 / 78151 (0.00512%) | 6 / 8670 (0.06920%) |
| То же / обрезанных answerable | 4 / 100 (4%) | 6 / 14 (42.857%) |
| Невалидные исходные gold span | 0 | 0 |

Модель и tokenizer сохранены в `models/answerability_distilbert/`.
Все ссылки ниже относятся к `outputs/classifier/20261020T173000Z_distilbert_seed42/`:

- `validation_metrics.json`, `validation_predictions.jsonl`: полные метрики и
  13063 прогноза в исходном порядке validation. Метрики дополнительно пересчитаны
  из сохранённых прогнозов и совпали с отчётом.
- `validation_errors.json`: 5 false answerable и 5 false unanswerable с настоящими
  текстами/вопросами validation.
- `full/attempt_1/history.jsonl`, `trainer_history.json`, `training_arguments.json`
  и `full/training_summary.json`: история, параметры, длительность, выбор checkpoint.
- `gpu_check.json`, `versions.json`, `installation_commands.json`,
  `requested_config.json`, `checkpoint_source.json`, `truncation_summary.json`:
  оборудование, среда, установка, настройки, revision и диагностика.
- `code/` и `requirements.txt`: снимки кода обучения/экспорта и среды.
- `save_load_verification.json`: FP32 logits на 8 строках validation после
  сохранения/загрузки совпали точно (max absolute difference=0), классы одинаковы.
- `status.json`: `completed`, исходные train/validation не изменились по SHA-256,
  `test_accessed=false`.

В `models/answerability_distilbert/training_provenance.json` сохранены сведения
об исходном run, config/revision/seed/versions/GPU, идентичность данных и хеши кода.
Точная копия этого файла для Git находится в
`outputs/classifier/20261020T173000Z_distilbert_seed42/training_provenance.json`.
Состояния Trainer полного и smoke-запусков также скопированы из игнорируемых
checkpoint-каталогов в соответствующие `full/attempt_1/trainer_state.json` и
`smoke/attempt_1/trainer_state.json`. Значения исходных файлов сохранены.
Повторное обучение для использования модели не требуется: загрузите модель и
tokenizer через `from_pretrained(..., local_files_only=True)`, вызовите `eval()` и
используйте тот же парный ввод с `return_token_type_ids=False`; вероятность класса
1 получается через `softmax(logits)[:,1]`.

Диагностические попытки: run `20261020T172100Z_distilbert_seed42`
остановился до обучения на проверке входных колонок; run
`20261020T172500Z_distilbert_seed42` отброшен в начале первой эпохи после выявления
некорректного автоматического определения нормализации loss в Trainer.
Они не являются завершёнными экспериментами и не используются в таблицах.
Финальный run полностью обучен заново из base checkpoint с seed=42.

Первоначальная проверка экспорта финального run сравнила вычисления с оставшейся
BF16-обёрткой Accelerate и FP32 после загрузки и остановилась на строгом допуске.
Экспорт исправлен для сравнения в одинаковом FP32. Затем выполнен
`--stage finalize --local-files-only --run-id 20261020T173000Z_distilbert_seed42`:
только повторная validation и экспорт выбранных завершённых весов, без обучения
(16.24 секунды). Старые отчёты сохранены в `finalization_backups/`; незавершённая
первая папка экспорта остаётся в `models/` для диагностики. Итоговая модель прошла
проверку и сохранена в итоговом локальном каталоге. Порог и настройки обучения
не изменялись.

## Хранение в Git

В Git сохраняются `scripts/`, `configs/`, README, `requirements.txt`, небольшие
отчёты `outputs/`, `split_manifest.json`, итоговые метрики, история обучения,
состояния Trainer, сведения о запуске и примеры ошибок. Диагностические попытки
и снимки кода запуска сохраняются вместе с завершённым экспериментом.
`.gitattributes` оставляет исторические файлы `outputs/` без преобразования
окончаний строк, чтобы сохранить их исходные байты и контрольные суммы.

Локально остаются `.venv/`, `data/raw/`, `data/processed/`, `.cache/`, `models/`,
checkpoint и tokenization cache, `__pycache__/`, настройки IDE и временные файлы.
`.env`, файлы токенов, credentials и приватные ключи исключены из Git.
Игнорирование не удаляет файлы с диска.

Полные прогнозы находятся только локально:
`outputs/classifier/20261020T173000Z_distilbert_seed42/validation_predictions.jsonl`.
Все файлы с таким именем, в том числе в `finalization_backups/`, игнорируются Git.
Файл содержит 13063 строки с ID, меткой, вероятностью answerable и прогнозом.
Он формируется `scripts/train_classifier.py` после `Trainer.predict` на всей
validation при завершении полного запуска или в режиме `--stage finalize`.
Фактическая команда, создавшая текущий файл без повторного обучения:

```powershell
& '.\.venv\Scripts\python.exe' -u scripts/train_classifier.py --stage finalize --local-files-only --run-id 20261020T173000Z_distilbert_seed42
```

Эта команда приведена для происхождения и восстановления экспорта, а не для
повторного обучения. Режим `finalize` требует завершённого checkpoint и
отсутствия итогового каталога `models/answerability_distilbert/`; существующую
модель он не перезаписывает, а прежние отчёты сохраняет в `finalization_backups/`.
Текущие прогнозы уже находятся на диске; при оформлении репозитория они не
пересоздавались. Метрики и история доступны в Git независимо от большого файла
прогнозов и локальных весов.

Отчёты о первоначальном окружении являются историческими снимками, а не описанием
среды после установки PyTorch. Пути обнаруженных интерпретаторов и имена процессов
в этих отчётах сохранены дословно. Сведения об окружении завершённого обучения
находятся в `outputs/classifier/20261020T173000Z_distilbert_seed42/versions.json`
и `gpu_check.json`.
