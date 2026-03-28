Сборка (нужны Arrow и Parquet из conda: arrow-cpp, libparquet или аналог):

  cmake -S training/cpp -B training/cpp/build
  cmake --build training/cpp/build -j

Бинарник: training/cpp/build/parquet_to_catboost_tsv

Запуск только из корня репозитория (cwd = корень):

  training/cpp/build/parquet_to_catboost_tsv [--threads N]

Cutoff по времени считается внутри (как training.main._find_time_cutoff), VAL_RATIO=0.15
должен совпадать с training/config.py.

В full_dataset.parquet колонка event_dttm — UTF-8 строка «YYYY-MM-DD HH:MM:SS»
(как пишет dataset_cpp/build_dataset), не Arrow timestamp; бинарник парсит её как при сборке датасета.

Жёстко зашито:
  читает  output/full_dataset.parquet
  пишет   output/cat_train_tsv_cache/train.tsv, val.tsv

В TSV только MODEL_INPUT_FEATURES + target + sample_weight (как training/main._detect_columns).
Не пишутся: customer_id, event_id, event_dttm (dttm читается из parquet только для train/val split).

Если есть output/cat_train_tsv_cache/catboost_export_features.txt — он должен дословно совпасть
со встроенным порядком FEATURE_NAMES; иначе ошибка. Если файла нет — используется только встроенный список.

Переменная GUARD_CONTEST_PARQUET_TO_CATBOOST_TSV — путь к бинарнику (для Python).

Экспорт: параллельно по row groups parquet (каждый поток заново открывает файл).
Формат TSV и правила сплита совпадают с training/catboost_train._stream_parquet_to_tsv_splits
(веса 2→5, 5→10; train если event_dttm < cutoff_ns; граница дня — floor к UTC-суткам в нс, как в pandas floor('D')).
