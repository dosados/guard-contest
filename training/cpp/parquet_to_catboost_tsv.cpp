#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

// Экспорт output/full_dataset.parquet → output/cat_train_tsv_cache/train.tsv|val.tsv
// (те же пути, что ожидает training/cat_train.py; cwd = корень репозитория).
//
// Входной parquet (build_dataset) содержит и метаданные, и признаки модели. На вход CatBoost / в TSV
// попадают СТРОГО те же колонки, что и в training/main.py через _detect_columns() →
// resolve_model_input_columns() → MODEL_INPUT_FEATURES (shared.features.FEATURE_NAMES):
//   только числовые фичи в фиксированном порядке + target + sample_weight.
// Не пишем в TSV и не подаём в модель: customer_id, event_id, event_dttm.
// event_dttm читается только из parquet для time-split (train/val) при экспорте.
//
// Параллельно: по одному потоку на row group parquet (каждый поток открывает файл сам).
//
// Сборка (из корня репозитория, conda с arrow-cpp / libparquet):
//   cmake -S training/cpp -B training/cpp/build && cmake --build training/cpp/build -j
//
// Запуск (из корня репозитория):
//   training/cpp/build/parquet_to_catboost_tsv [--threads M]
// Cutoff по времени считается внутри (как training.main._find_time_cutoff + training.config.VAL_RATIO).
// event_dttm в full_dataset.parquet — UTF-8 строка "%Y-%m-%d %H:%M:%S" (dataset_cpp/build_dataset.cpp), не TIMESTAMP.

#include <arrow/api.h>
#include <arrow/array/concatenate.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

// Пути относительно cwd (корень репозитория), как shared.config TRAIN_DATASET_PATH и cat_train cache_dir.
static constexpr const char* kDatasetParquet = "output/full_dataset.parquet";
static constexpr const char* kTrainTsv = "output/cat_train_tsv_cache/train.tsv";
static constexpr const char* kValTsv = "output/cat_train_tsv_cache/val.tsv";
static constexpr const char* kFeaturesFile = "output/cat_train_tsv_cache/catboost_export_features.txt";

// training/config.py VAL_RATIO — при смене в Python обновить здесь.
static constexpr double kValRatio = 0.15;

static void log_err(const std::string& s) { std::cerr << "[parquet_to_catboost_tsv] " << s << "\n"; }

static void log_info(const std::string& s) { std::cerr << "[parquet_to_catboost_tsv] " << s << "\n"; }

/** Одна строка прогресса (\r + очистка), как в dataset_cpp/build_dataset.cpp */
static void progress_line(const char* stage, int64_t done, int64_t total) {
  constexpr int kBarW = 40;
  std::cerr << "\r\033[2K[parquet_to_catboost_tsv] " << stage << " [";
  if (total > 0) {
    int filled = static_cast<int>(kBarW * done / total);
    if (filled > kBarW) filled = kBarW;
    int pct = static_cast<int>(100 * done / total);
    if (pct > 100) pct = 100;
    for (int i = 0; i < kBarW; ++i) std::cerr << (i < filled ? '#' : '.');
    std::cerr << "] " << std::setw(3) << pct << "%  " << done << "/" << total;
  } else {
    for (int i = 0; i < kBarW; ++i) std::cerr << '.';
    std::cerr << "     " << done;
  }
  std::cerr << std::flush;
}

static void progress_newline() { std::cerr << "\n"; }

static std::string trim(std::string s) {
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) s.pop_back();
  size_t i = 0;
  while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
  return s.substr(i);
}

static std::vector<std::string> read_feature_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open features file: " + path);
  std::vector<std::string> out;
  std::string line;
  while (std::getline(in, line)) {
    line = trim(line);
    if (!line.empty()) out.push_back(line);
  }
  if (out.empty()) throw std::runtime_error("empty features file: " + path);
  return out;
}

// Единственный набор признаков для строки TSV = на вход модели (как training/main, без id и без dttm в файле).
// Синхронизировать с shared.features.FEATURE_NAMES при изменении датасета.
static const char* kModelFeatureNames[] = {
    "operation_amt",
    "log_1_plus_transactions_seen",
    "amount_zscore",
    "is_amount_high",
    "transactions_last_24h",
    "sum_amount_last_1h",
    "max_amount_last_24h",
    "is_new_device",
    "is_new_mcc",
    "is_new_channel",
    "is_compromised_device",
    "web_rdp_connection",
    "phone_voip_call_state",
    "hour",
    "day_of_week",
    "is_night_transaction",
    "is_weekend",
    "transactions_last_10m",
    "sum_amount_last_24h",
    "time_since_prev_transaction",
    "is_new_browser_language",
    "transactions_in_session",
    "timezone_missing",
    "tr_amount",
    "event_type_nm",
    "event_descr",
    "mcc_code",
    "trend_mean_last_3_to_10",
    "amount_percentile_rank",
    "std_time_deltas",
    "is_new_device_tz_pair",
    "is_device_switch",
    "is_mcc_switch",
    "session_duration",
    "session_mean_amount",
    "device_freq",
    "delta_1",
    "delta_2",
    "delta_3",
    "acceleration_delta_1_over_2",
    "std_delta_last_k",
    "time_since_last_device_change",
    "time_since_last_mcc_change",
    "event_descr_freq_last_1h",
    "event_descr_freq_last_6h",
    "event_descr_freq_last_24h",
    "event_type_nm_freq_last_1h",
    "event_type_nm_freq_last_6h",
    "event_type_nm_freq_last_24h",
    "mcc_freq_last_1h",
    "mcc_freq_last_6h",
    "mcc_freq_last_24h",
    "mcc_event_descr_pair_new",
    "high_amount_ratio_last_24h",
    "amount_relative_to_mcc_median_5_days",
};
static constexpr size_t kNumModelFeatures = sizeof(kModelFeatureNames) / sizeof(kModelFeatureNames[0]);

/** Как shared.config.resolve_model_input_columns: только MODEL_INPUT_FEATURES, порядок как в Python. */
static std::vector<std::string> resolve_features_embedded(const arrow::Schema& schema) {
  std::vector<std::string> out;
  out.reserve(kNumModelFeatures);
  for (size_t i = 0; i < kNumModelFeatures; ++i) {
    const char* name = kModelFeatureNames[i];
    if (schema.GetFieldIndex(name) < 0) {
      throw std::runtime_error(std::string("parquet: нет колонки фичи (как в shared.features): ") + name);
    }
    out.emplace_back(name);
  }
  return out;
}

static constexpr double kW_eps = 1e-5;

static double remap_sample_weight(double w) {
  if (std::fabs(w - 5.0) < kW_eps) return 10.0;
  if (std::fabs(w - 2.0) < kW_eps) return 5.0;
  return w;
}

static void fmt_cell_g(std::string& line, double v) {
  if (std::isnan(v)) return;
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.9g", v);
  line += buf;
}

static void append_tsv_row(std::string& line, const std::vector<double>& feats, int target, double w) {
  line.clear();
  for (size_t i = 0; i < feats.size(); ++i) {
    if (i) line.push_back('\t');
    fmt_cell_g(line, feats[i]);
  }
  line.push_back('\t');
  char tb[32];
  std::snprintf(tb, sizeof(tb), "%d", target);
  line += tb;
  line.push_back('\t');
  fmt_cell_g(line, w);
  line.push_back('\n');
}

/** Как dataset_cpp build_dataset col_get_str: dictionary / string / large_string. */
static std::string utf8_cell_string(const arrow::Array& col, int64_t row) {
  if (col.IsNull(row)) return {};
  if (col.type_id() == arrow::Type::DICTIONARY) {
    const auto& da = static_cast<const arrow::DictionaryArray&>(col);
    if (da.IsNull(row)) return {};
    const int64_t di = da.GetValueIndex(row);
    if (di < 0) return {};
    const auto& dict = *da.dictionary();
    if (di >= dict.length()) return {};
    return utf8_cell_string(dict, di);
  }
  switch (col.type_id()) {
    case arrow::Type::STRING:
      return static_cast<const arrow::StringArray&>(col).GetString(row);
    case arrow::Type::LARGE_STRING:
      return static_cast<const arrow::LargeStringArray&>(col).GetString(row);
    default:
      return {};
  }
}

/** Как dataset_cpp parse_dttm_fields + mktime (наивное локальное время, как при сборке датасета). */
static std::optional<int64_t> parse_event_dttm_utf8_to_epoch_ns(const std::string& s_in) {
  std::string t = trim(s_in);
  if (t.empty()) return std::nullopt;
  if (t.size() > 19 && t[19] == '.') t.resize(19);
  std::tm tm = {};
  std::memset(&tm, 0, sizeof(tm));
  const char* r = strptime(t.c_str(), "%Y-%m-%d %H:%M:%S", &tm);
  if (r == nullptr) return std::nullopt;
  tm.tm_isdst = -1;
  const time_t tt = mktime(&tm);
  if (tt == static_cast<time_t>(-1)) return std::nullopt;
  return static_cast<int64_t>(tt) * 1000000000LL;
}

/** event_dttm: TIMESTAMP или UTF-8 строка (full_dataset), см. build_dataset.cpp. */
static std::optional<int64_t> event_dttm_to_epoch_ns(const arrow::Array& arr, int64_t row) {
  if (arr.IsNull(row)) return std::nullopt;
  switch (arr.type_id()) {
    case arrow::Type::TIMESTAMP: {
      const auto& ta = static_cast<const arrow::TimestampArray&>(arr);
      const auto* type = static_cast<const arrow::TimestampType*>(arr.type().get());
      int64_t v = ta.Value(row);
      switch (type->unit()) {
        case arrow::TimeUnit::NANO: return v;
        case arrow::TimeUnit::MICRO: return v * 1000;
        case arrow::TimeUnit::MILLI: return v * 1000000;
        case arrow::TimeUnit::SECOND: return v * 1000000000LL;
      }
      return v;
    }
    case arrow::Type::STRING:
    case arrow::Type::LARGE_STRING:
    case arrow::Type::DICTIONARY:
      return parse_event_dttm_utf8_to_epoch_ns(utf8_cell_string(arr, row));
    default:
      return std::nullopt;
  }
}

static bool is_train_split(std::optional<int64_t> ts_ns, int64_t cutoff_ns) {
  if (!ts_ns) return false;
  return *ts_ns < cutoff_ns;
}

static double cell_as_double(const arrow::Array& arr, int64_t row) {
  if (arr.IsNull(row)) return std::numeric_limits<double>::quiet_NaN();
  switch (arr.type_id()) {
    case arrow::Type::DOUBLE:
      return static_cast<const arrow::DoubleArray&>(arr).Value(row);
    case arrow::Type::FLOAT:
      return static_cast<double>(static_cast<const arrow::FloatArray&>(arr).Value(row));
    case arrow::Type::INT64:
      return static_cast<double>(static_cast<const arrow::Int64Array&>(arr).Value(row));
    case arrow::Type::INT32:
      return static_cast<double>(static_cast<const arrow::Int32Array&>(arr).Value(row));
    case arrow::Type::UINT32:
      return static_cast<double>(static_cast<const arrow::UInt32Array&>(arr).Value(row));
    case arrow::Type::UINT64:
      return static_cast<double>(static_cast<const arrow::UInt64Array&>(arr).Value(row));
    case arrow::Type::BOOL:
      return static_cast<const arrow::BooleanArray&>(arr).Value(row) ? 1.0 : 0.0;
    default:
      return std::numeric_limits<double>::quiet_NaN();
  }
}

static int cell_as_target_int(const arrow::Array& arr, int64_t row) {
  if (arr.IsNull(row)) return 0;
  switch (arr.type_id()) {
    case arrow::Type::INT32:
      return static_cast<int>(static_cast<const arrow::Int32Array&>(arr).Value(row));
    case arrow::Type::INT64:
      return static_cast<int>(static_cast<const arrow::Int64Array&>(arr).Value(row));
    case arrow::Type::DOUBLE: {
      double v = static_cast<const arrow::DoubleArray&>(arr).Value(row);
      return static_cast<int>(v);
    }
    default:
      return 0;
  }
}

struct ColumnPlan {
  std::vector<int> feat_schema_cols;
  int target_col{-1};
  int weight_col{-1};
  int dttm_col{-1};
};

static ColumnPlan build_plan(const arrow::Schema& schema, const std::vector<std::string>& features,
                             const std::string& dttm_name) {
  ColumnPlan p;
  for (const auto& name : features) {
    int idx = schema.GetFieldIndex(name);
    if (idx < 0) throw std::runtime_error("parquet missing feature column: " + name);
    p.feat_schema_cols.push_back(idx);
  }
  p.target_col = schema.GetFieldIndex("target");
  p.weight_col = schema.GetFieldIndex("sample_weight");
  p.dttm_col = schema.GetFieldIndex(dttm_name);
  if (p.target_col < 0) throw std::runtime_error("parquet missing column: target");
  if (p.weight_col < 0) throw std::runtime_error("parquet missing column: sample_weight");
  if (p.dttm_col < 0) throw std::runtime_error("parquet missing column: " + dttm_name);
  return p;
}

static std::shared_ptr<arrow::Array> single_chunk(const std::shared_ptr<arrow::ChunkedArray>& col) {
  if (!col || col->num_chunks() == 0) {
    throw std::runtime_error("empty column chunks");
  }
  if (col->num_chunks() == 1) return col->chunk(0);
  arrow::MemoryPool* pool = arrow::default_memory_pool();
  arrow::Result<std::shared_ptr<arrow::Array>> merged = arrow::Concatenate(col->chunks(), pool);
  if (!merged.ok()) throw std::runtime_error(merged.status().ToString());
  return std::move(merged).ValueOrDie();
}

static int64_t floor_div_int64(int64_t a, int64_t b) {
  int64_t q = a / b;
  int64_t r = a % b;
  if (r != 0 && ((a < 0) != (b < 0))) --q;
  return q;
}

/** Начало календарного дня в UTC по наносекундам (аналог pandas dt.floor('D') для naive ns). */
static int64_t floor_event_dttm_to_day_ns(int64_t ts_ns) {
  constexpr int64_t kDayNs = 86400000000000LL;
  return floor_div_int64(ts_ns, kDayNs) * kDayNs;
}

/**
 * Как training.main._find_time_cutoff: считаем строки по дням event_dttm, сортируем дни от новых к старым,
 * набираем с хвоста долю kValRatio от total — последний взятый день = cutoff (train: ts < cutoff_ns).
 */
static arrow::Status find_time_cutoff_ns(const std::string& path, const std::string& dttm_name,
                                         int64_t* cutoff_ns_out, int64_t* val_target_out) {
  arrow::MemoryPool* pool = arrow::default_memory_pool();
  ARROW_ASSIGN_OR_RAISE(auto infile, arrow::io::ReadableFile::Open(path));
  ARROW_ASSIGN_OR_RAISE(auto reader, parquet::arrow::OpenFile(infile, pool));
  std::shared_ptr<arrow::Schema> schema;
  ARROW_RETURN_NOT_OK(reader->GetSchema(&schema));
  const int dttm_col = schema->GetFieldIndex(dttm_name);
  if (dttm_col < 0) {
    return arrow::Status::Invalid("parquet: нет колонки ", dttm_name);
  }
  const int num_rg = reader->num_row_groups();
  if (num_rg <= 0) return arrow::Status::Invalid("no row groups");
  std::unordered_map<int64_t, int64_t> by_day;
  int64_t total = 0;
  const std::vector<int> cols = {dttm_col};
  log_info("time split: скан event_dttm по row groups (" + std::to_string(num_rg) + ") …");
  for (int rg = 0; rg < num_rg; ++rg) {
    std::shared_ptr<arrow::Table> table;
    ARROW_RETURN_NOT_OK(reader->ReadRowGroup(rg, cols, &table));
    std::shared_ptr<arrow::Array> arr;
    try {
      arr = single_chunk(table->column(0));
    } catch (const std::exception& e) {
      progress_newline();
      return arrow::Status::Invalid(e.what());
    }
    for (int64_t i = 0; i < arr->length(); ++i) {
      auto ts = event_dttm_to_epoch_ns(*arr, i);
      if (!ts) continue;
      const int64_t day = floor_event_dttm_to_day_ns(*ts);
      by_day[day] += 1;
      ++total;
    }
    progress_line("time_split row_groups", static_cast<int64_t>(rg + 1), static_cast<int64_t>(num_rg));
  }
  progress_newline();
  if (total == 0) return arrow::Status::Invalid("нет строк event_dttm для time split");
  const int64_t val_target = std::max<int64_t>(1, static_cast<int64_t>(static_cast<double>(total) * kValRatio));
  std::vector<std::pair<int64_t, int64_t>> items(by_day.begin(), by_day.end());
  std::sort(items.begin(), items.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
  int64_t acc = 0;
  int64_t cutoff = items.front().first;
  for (const auto& kv : items) {
    acc += kv.second;
    cutoff = kv.first;
    if (acc >= val_target) break;
  }
  *cutoff_ns_out = cutoff;
  *val_target_out = val_target;
  log_info("time split: строк с валидным event_dttm: " + std::to_string(total) +
           ", уникальных дней: " + std::to_string(items.size()));
  return arrow::Status::OK();
}

struct RgResult {
  int rg{-1};
  arrow::Status st;
  int64_t train_rows{0};
  int64_t val_rows{0};
};

static arrow::Status process_row_group(const std::string& parquet_path, int rg_index, const ColumnPlan& plan,
                                       int64_t cutoff_ns, const std::string& train_part, const std::string& val_part,
                                       int64_t* out_train_rows, int64_t* out_val_rows) {
  arrow::MemoryPool* pool = arrow::default_memory_pool();
  ARROW_ASSIGN_OR_RAISE(auto infile, arrow::io::ReadableFile::Open(parquet_path));
  ARROW_ASSIGN_OR_RAISE(auto reader, parquet::arrow::OpenFile(infile, pool));
  std::shared_ptr<arrow::Table> table;
  ARROW_RETURN_NOT_OK(reader->ReadRowGroup(rg_index, &table));

  std::vector<std::shared_ptr<arrow::Array>> feat_arrays;
  feat_arrays.reserve(plan.feat_schema_cols.size());
  for (int c : plan.feat_schema_cols) {
    feat_arrays.push_back(single_chunk(table->column(c)));
  }
  auto target_arr = single_chunk(table->column(plan.target_col));
  auto weight_arr = single_chunk(table->column(plan.weight_col));
  auto dttm_arr = single_chunk(table->column(plan.dttm_col));

  const int64_t n = table->num_rows();
  std::ofstream train_out(train_part, std::ios::binary | std::ios::trunc);
  std::ofstream val_out(val_part, std::ios::binary | std::ios::trunc);
  if (!train_out || !val_out) {
    return arrow::Status::IOError("cannot open part files for rg ", rg_index);
  }

  std::vector<double> feats(plan.feat_schema_cols.size());
  std::string line;
  line.reserve(4096);
  int64_t tr = 0, va = 0;

  for (int64_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < feat_arrays.size(); ++j) {
      feats[j] = cell_as_double(*feat_arrays[j], i);
    }
    int y = cell_as_target_int(*target_arr, i);
    double w0 = cell_as_double(*weight_arr, i);
    double w = remap_sample_weight(std::isnan(w0) ? 1.0 : w0);
    auto ts = event_dttm_to_epoch_ns(*dttm_arr, i);
    // Строка TSV только фичи + target + weight; dttm не пишем (как Python _stream_parquet_to_tsv_splits).
    append_tsv_row(line, feats, y, w);
    if (is_train_split(ts, cutoff_ns)) {
      train_out.write(line.data(), static_cast<std::streamsize>(line.size()));
      ++tr;
    } else {
      val_out.write(line.data(), static_cast<std::streamsize>(line.size()));
      ++va;
    }
  }
  *out_train_rows = tr;
  *out_val_rows = va;
  return arrow::Status::OK();
}

/** Заголовок TSV: только фичи модели + target + sample_weight (без event_dttm / customer_id / event_id). */
static void write_header(const std::string& path, const std::vector<std::string>& features) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot write: " + path);
  for (size_t i = 0; i < features.size(); ++i) {
    if (i) out.put('\t');
    out << features[i];
  }
  out << "\ttarget\tsample_weight\n";
}

static void append_file_to(const std::string& src, std::ofstream& dst) {
  std::ifstream in(src, std::ios::binary);
  if (!in) return;
  dst << in.rdbuf();
}

static int usage(const char* argv0) {
  log_err("usage (cwd = корень репозитория):");
  log_err(std::string("  ") + argv0 + " [--threads N]");
  log_err("  читает: output/full_dataset.parquet; опц. проверка catboost_export_features.txt == MODEL_INPUT_FEATURES");
  log_err("  пишет:  output/cat_train_tsv_cache/train.tsv, val.tsv");
  log_err("  cutoff: внутри, VAL_RATIO=0.15 как training/config.py");
  return 2;
}

int main(int argc, char** argv) {
  int threads = 0;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* name) -> const char* {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return argv[++i];
    };
    if (a == "--threads") {
      threads = std::stoi(need("--threads"));
    } else if (a == "-h" || a == "--help")
      return usage(argv[0]);
    else
      throw std::runtime_error("unknown arg: " + a);
  }

  const std::string in_path = kDatasetParquet;
  const std::string train_out = kTrainTsv;
  const std::string val_out = kValTsv;
  const std::string feat_path = kFeaturesFile;

  try {
    log_info("входной parquet: " + in_path);
    const std::string dttm_name = "event_dttm";
    int64_t cutoff_ns = 0;
    int64_t val_target = 0;
    {
      arrow::Status cst = find_time_cutoff_ns(in_path, dttm_name, &cutoff_ns, &val_target);
      if (!cst.ok()) {
        log_err(cst.ToString());
        return 1;
      }
      log_info("time split cutoff_ns=" + std::to_string(cutoff_ns) + " val_target_rows~=" +
               std::to_string(val_target) + " (VAL_RATIO=0.15, как training/config.py)");
    }

    std::error_code ec;
    fs::create_directories(fs::path(train_out).parent_path(), ec);

    arrow::MemoryPool* pool = arrow::default_memory_pool();
    auto in_res = arrow::io::ReadableFile::Open(in_path);
    if (!in_res.ok()) {
      log_err(in_res.status().ToString());
      return 1;
    }
    auto infile_main = std::move(in_res).ValueOrDie();
    auto reader_res = parquet::arrow::OpenFile(infile_main, pool);
    if (!reader_res.ok()) {
      log_err(reader_res.status().ToString());
      return 1;
    }
    auto preader = std::move(reader_res).ValueOrDie();
    std::shared_ptr<arrow::Schema> schema;
    if (!preader->GetSchema(&schema).ok()) {
      log_err("GetSchema failed");
      return 1;
    }

    std::vector<std::string> features = resolve_features_embedded(*schema);
    if (fs::exists(feat_path)) {
      log_info("проверка " + feat_path + " (должен совпадать с MODEL_INPUT_FEATURES / training.main) …");
      const std::vector<std::string> from_file = read_feature_file(feat_path);
      if (from_file != features) {
        throw std::runtime_error(
            "catboost_export_features.txt не совпадает с MODEL_INPUT_FEATURES (shared.features.FEATURE_NAMES). "
            "В TSV только фичи модели + target + sample_weight; customer_id, event_id, event_dttm не включаются.");
      }
      log_info("файл списка фич совпал со встроенным порядком");
    } else {
      log_info("файла списка фич нет — только встроенный MODEL_INPUT_FEATURES (" +
               std::to_string(features.size()) + " колонок, проверка по схеме parquet)");
    }
    log_info("в TSV: " + std::to_string(features.size()) +
             " фич модели + target + sample_weight (без customer_id / event_id / event_dttm)");

    ColumnPlan plan = build_plan(*schema, features, dttm_name);

    int num_rg = preader->num_row_groups();
    if (num_rg <= 0) {
      log_err("no row groups");
      return 1;
    }

    int nt = threads <= 0 ? static_cast<int>(std::thread::hardware_concurrency()) : threads;
    if (nt < 1) nt = 1;
    if (nt > num_rg) nt = num_rg;

    log_info("экспорт TSV: parquet row_groups=" + std::to_string(num_rg) + " потоков=" + std::to_string(nt));
    log_info("выход: " + std::string(train_out) + " , " + val_out);

    std::string train_base = train_out + ".part.";
    std::string val_base = val_out + ".part.";

    std::vector<std::string> train_parts(static_cast<size_t>(num_rg));
    std::vector<std::string> val_parts(static_cast<size_t>(num_rg));
    for (int i = 0; i < num_rg; ++i) {
      train_parts[static_cast<size_t>(i)] = train_base + std::to_string(i);
      val_parts[static_cast<size_t>(i)] = val_base + std::to_string(i);
    }

    std::atomic<int> next_rg{0};
    std::atomic<int> export_finished{0};
    std::vector<RgResult> results(static_cast<size_t>(num_rg));
    std::mutex log_mtx;

    auto worker = [&]() {
      while (true) {
        int rg = next_rg.fetch_add(1);
        if (rg >= num_rg) break;
        RgResult& rr = results[static_cast<size_t>(rg)];
        rr.rg = rg;
        rr.st = process_row_group(in_path, rg, plan, cutoff_ns, train_parts[static_cast<size_t>(rg)],
                                  val_parts[static_cast<size_t>(rg)], &rr.train_rows, &rr.val_rows);
        if (!rr.st.ok()) {
          std::lock_guard<std::mutex> lk(log_mtx);
          progress_newline();
          log_err("row group " + std::to_string(rg) + ": " + rr.st.ToString());
        } else {
          const int done = export_finished.fetch_add(1) + 1;
          std::lock_guard<std::mutex> lk(log_mtx);
          progress_line("export row_groups", static_cast<int64_t>(done), static_cast<int64_t>(num_rg));
        }
      }
    };

    std::vector<std::thread> tw;
    tw.reserve(static_cast<size_t>(nt));
    for (int t = 0; t < nt; ++t) tw.emplace_back(worker);
    for (auto& th : tw) th.join();
    progress_newline();

    for (const auto& rr : results) {
      if (!rr.st.ok()) return 1;
    }

    log_info("заголовки TSV и склейка частей …");
    write_header(train_out, features);
    write_header(val_out, features);

    std::ofstream train_final(train_out, std::ios::binary | std::ios::app);
    std::ofstream val_final(val_out, std::ios::binary | std::ios::app);
    if (!train_final || !val_final) {
      log_err("cannot open outputs for append");
      return 1;
    }

    int64_t sum_train = 0, sum_val = 0;
    for (const auto& rr : results) {
      sum_train += rr.train_rows;
      sum_val += rr.val_rows;
    }
    for (int i = 0; i < num_rg; ++i) {
      append_file_to(train_parts[static_cast<size_t>(i)], train_final);
      append_file_to(val_parts[static_cast<size_t>(i)], val_final);
      fs::remove(train_parts[static_cast<size_t>(i)]);
      fs::remove(val_parts[static_cast<size_t>(i)]);
      progress_line("merge parts", static_cast<int64_t>(i + 1), static_cast<int64_t>(num_rg));
    }
    progress_newline();
    train_final.close();
    val_final.close();

    if (sum_train <= 0) {
      log_err("no train rows (check cutoff vs event_dttm)");
      return 1;
    }

    log_info("готово: train_rows=" + std::to_string(sum_train) + " val_rows=" + std::to_string(sum_val) +
             " row_groups=" + std::to_string(num_rg) + " threads=" + std::to_string(nt));
    return 0;
  } catch (const std::exception& e) {
    log_err(std::string("fatal: ") + e.what());
    return 1;
  }
}
