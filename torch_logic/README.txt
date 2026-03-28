torch_logic: инструкция по запуску

1) Требования
- Запускать команды из корня репозитория.
- Должны существовать файлы:
  - data/train/train_part_1.parquet
  - data/train/train_part_2.parquet
  - data/train/train_part_3.parquet
  - data/train_labels.parquet
  - data/test/test.parquet
- Для GPU нужен рабочий CUDA + PyTorch с поддержкой CUDA.

2) Обучение модели (linear -> LSTM -> linear)

Вход в первый linear слой (на каждую транзакцию):
- mcc_code
- event_descr
- event_type_nm
- operaton_amt

Базовая команда:
PYTHONPATH=. python3 torch_logic/train.py --epochs 1 --device auto

Параметры:
- --epochs            число эпох (по умолчанию 1)
- --lr                learning rate (по умолчанию 1e-3)
- --device            auto | cuda | cpu (по умолчанию auto)
- --linear-dim        размер первого линейного слоя (по умолчанию 32)
- --lstm-hidden-dim   размер hidden состояния LSTM (по умолчанию 64)
- --lstm-layers       число слоёв LSTM (по умолчанию 1)
- --dropout           dropout между слоями LSTM (по умолчанию 0.0)
- --log-every         частота логов по шагам (по умолчанию 20000)
- --seq-chunk-size    сколько подряд идущих транзакций одного customer_id объединять в один forward LSTM (по умолчанию 128)
- --output            путь чекпоинта (по умолчанию output/weights/model_torch.pt)

Пример:
PYTHONPATH=. python3 torch_logic/train.py --epochs 2 --device cuda --linear-dim 64 --lstm-hidden-dim 128

Во время обучения:
- показывается tqdm-бар по строкам train_part*.

После обучения:
- используется time split по event_dttm (новейшие примерно 20% дней идут в val),
- обучение идет только на train-части split,
- выполняется отдельный eval-проход и считается PR-AUC на val
  через sklearn average_precision_score (метрика как в бустинге).

Важно по таргету:
- target = 1, если event_id присутствует в data/train_labels.parquet
- target = 0, если event_id отсутствует в data/train_labels.parquet
- Значение колонки target из train_labels на обучение не влияет.

Важно по состоянию:
- При смене customer_id состояние LSTM сбрасывается (чистый контекст для нового пользователя).

3) Инференс (только test, без pretest)

Базовая команда:
PYTHONPATH=. python3 torch_logic/predict.py --device auto

Параметры:
- --model-path   путь к чекпоинту (по умолчанию output/weights/model_torch.pt)
- --test-path    путь к test parquet (по умолчанию data/test/test.parquet)
- --output       путь к submission (по умолчанию output/submission.csv)
- --device       auto | cuda | cpu (по умолчанию auto)
- --seq-chunk-size  как при обучении (по умолчанию 128)

Пример:
PYTHONPATH=. python3 torch_logic/predict.py --model-path output/weights/model_torch.pt --output output/submission_torch.csv --device cuda

4) Запуск через общий submission-скрипт

Можно запустить через основной модуль:
PYTHONPATH=. python3 submission/main.py --model torch

В этом режиме:
- используется torch_logic,
- читается только data/test/test.parquet,
- pretest не используется.

5) Результат
- Файл предсказаний сохраняется в CSV с колонками:
  - event_id
  - predict
