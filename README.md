# Answerability classification

Бинарный классификатор определяет по **question + context**, достаточно ли
предоставленного текста для ответа. `1` — answerable, `0` — unanswerable.
DistilBERT дообучается для этой задачи; генеративная LLM не используется.

## Структура

```text
configs/
  classifier_distilbert.json  # модель и гиперпараметры
  data_split.json             # алгоритм и эталон размеров/ID SHA-256
scripts/
  check_environment.py
  download_dataset.py
  prepare_splits.py
  train_classifier.py
  report_results.py
data/raw/                     # исходный SQuAD 2.0
data/processed/              # наши train/validation/test
models/                       # сохранённые модели
outputs/                      # отчёты, прогнозы и логи
requirements.txt
```

Рабочие папки создаются автоматически. Данные, модели, outputs, кеш `.cache/`,
окружение `.venv/` и секреты исключены из Git. В репозиторий входят исходники,
конфигурации, зависимости и документация.

## Установка: Windows / PowerShell

Нужны Python 3.12 и совместимая NVIDIA GPU для обучения. Команды выполняются
из корня репозитория, без активации окружения и изменения Execution Policy.
Если `.venv` уже существует, создавать её заново не нужно.

```powershell
# Только для нового окружения; Python Launcher должен быть установлен:
py -3.12 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
```

## Подготовка данных

```powershell
& '.\.venv\Scripts\python.exe' scripts/check_environment.py
& '.\.venv\Scripts\python.exe' -u scripts/download_dataset.py --summary-only
& '.\.venv\Scripts\python.exe' -u scripts/prepare_splits.py
```

Скрипт загрузки скачивает [SQuAD 2.0](https://huggingface.co/datasets/rajpurkar/squad_v2)
только при отсутствии `data/raw/squad_v2/`; затем использует локальную копию.
Аудит печатает размеры, классы, пустые входы и дубликаты. Проверка окружения
различает совместимость CUDA у драйвера и фактическую доступность CUDA в PyTorch.

Исходные train/validation авторов отличаются от наших экспериментальных частей.
Наши train и validation создаются из исходного train по группам контекстов;
test — 500 answerable + 500 unanswerable из исходного validation.
`context_group = SHA-256(UTF-8(' '.join(NFC(context).split())))`.
Группы сортируются и перемешиваются с **split seed=42**; около 10% групп
отводится validation. Пересекающиеся с исходным validation группы исключаются
из исходного train. Порядок строк внутри наших train/validation сохраняется.
Полный алгоритм случайного отбора test записан в `configs/data_split.json`.

Этот файл содержит эталонные размеры и SHA-256 **упорядоченных** списков ID.
При подготовке и повторной проверке они сравниваются с фактическим результатом
до публикации данных. При расхождении скрипт сообщает ожидаемые/фактические
значения и возможные причины, завершается с ошибкой и не обновляет эталон.
При совпадении создаются `outputs/split_summary.json`, `outputs/split_manifest.json`
и `data/processed/squad_v2_answerability/`.

Для проверки уже подготовленных выборок без изменения данных и manifest:

```powershell
& '.\.venv\Scripts\python.exe' -u scripts/prepare_splits.py --check-only
```

## Обучение

```powershell
$run42 = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ') + '_distilbert_seed42'
& '.\.venv\Scripts\python.exe' -u scripts/train_classifier.py --seed 42 --run-id $run42
```

При первом запуске скачивается только фиксированный checkpoint
[distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased/tree/12040accade4e8a0f71eabdb258fecc2e7e948be),
revision `12040accade4e8a0f71eabdb258fecc2e7e948be`.
Кеш находится внутри проекта. Для работы без сети с уже заполненным кешем
добавьте `--local-files-only`. Существующий run/model не перезаписывается.

Протокол: 3 эпохи, learning rate=2e-5, weight decay=0.01, warmup=10%,
train batch=8, gradient accumulation=2, eval batch=8, max_length=512,
workers=0, unweighted cross-entropy, threshold=0.5. BF16 используется при
успешной GPU-проверке; фактические параметры записываются в `run.json`.
При OOM batch уменьшается с сохранением effective batch=16, попытка начинается
из свежей seeded инициализации; прерванная попытка не считается результатом.

До первого optimizer step оценивается вся validation:

> pretrained encoder with randomly initialized classification head, before task fine-tuning.

Проверяются неизменность весов и нулевой global step. Перед обучением той же
модели восстанавливается RNG и устанавливается seed. После каждой эпохи
оценивается validation; лучший checkpoint выбирается по macro-F1.
Модель и tokenizer сохраняются в `models/classifier/<run_id>/` и проверяются
после повторной загрузки в FP32 на validation. Во вход передаются только
question/context; answers используются для метки и диагностики, labels — для loss.

## Результаты и сравнение seed

```powershell
& '.\.venv\Scripts\python.exe' scripts/report_results.py show --run "outputs/classifier/$run42"
& '.\.venv\Scripts\python.exe' scripts/report_results.py error --run "outputs/classifier/$run42" --index 0
```

В `outputs/classifier/<run_id>/`: `run.json` содержит конфигурацию, версии,
revision, идентичность данных, проверки и статус; `initial_validation_metrics.json`
— начальную оценку; `epoch_metrics.jsonl` — метрики эпох; `history.jsonl` — loss,
шаги и learning rate; `validation_metrics.json` — итоговые метрики, confusion
matrix и always-answerable baseline. Прогнозы, настоящие ошибки и вывод
сохраняются в `validation_predictions.jsonl`, `validation_errors.json`,
`stdout.log` и `stderr.log`. Просмотр отчётов не запускает обучение.

Планируемые training seeds — **42, 43, 44**, split seed всегда **42**.
Каждый из трёх запусков использует `epochs=3` из общей конфигурации.
Повторите команду обучения для 43 и 44 с новыми run_id, последовательно.
Затем передайте три фактических каталога:

```powershell
& '.\.venv\Scripts\python.exe' scripts/report_results.py compare --runs "outputs/classifier/$run42" "outputs/classifier/$run43" "outputs/classifier/$run44"
```

Сравнение показывает initial / epoch 1 / epoch 2 / epoch 3 / best для каждого seed,
mean и **sample standard deviation, n=3**, и изменения относительно initial.
Учитываются только завершённые совместимые полные запуски. Матрицы ошибок
не суммируются: все прогоны используют одну validation, а SD между seeds
не является доверительным интервалом.

## Ограничения

- Test предназначен только для итоговой оценки после фиксации протокола;
  обучение, настройка и просмотр ошибок используют только train/validation.
- Разделение по одинаковым контекстам не гарантирует разделения по статьям/темам.
- Context обрезается до лимита 512 токенов пары, иногда скрывая правильный ответ;
  исходная метка не меняется. Диагностика сохраняется в run.json.
- Validation участвует в выборе checkpoint, поэтому это development evaluation,
  а не независимая итоговая оценка. Вероятности не проверены на калибровку.
- Фиксированные версии и seeds не гарантируют побитовую идентичность вычислений
  на другом оборудовании; контроль состава выборок выполняется отдельно.
