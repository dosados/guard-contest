// Сборка обучающего датасета: Arrow + Parquet + OpenSSL (MD5).
// Train: каждая строка с непустым customer_id → строка в датасет → update окна.
// История: deque не длиннее kWindowCap; фичи — по суффиксу длины min(stored, W),
// где W сэмплируется из эмпирического распределения "транзакций на пользователя" из data/test/pretest.parquet.
// target = 1 если event_id в train_labels.parquet, иначе 0.
// Порядок обхода: для k=1,2,3 подряд pretrain_part_k → train_part_k; после train_part_k из map
// удаляются все customer_id, встреченные в этой паре файлов (когорты по номеру части не пересекаются).
// В выходном parquet строки по-прежнему в порядке train_part_1, train_part_2, train_part_3.
// Запуск из корня репозитория: ./build_dataset [repo_root]

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <openssl/evp.h>
#include <openssl/md5.h>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/metadata.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/** Максимум транзакций в deque и верхняя граница активного окна. */
constexpr int kWindowCap = 512;
constexpr int kOutBatch = 131072;
constexpr double kWeightUnlabeled = 1.0;
constexpr double kWeightLabeled0 = 2.0;
constexpr double kWeightLabeled1 = 5.0;
constexpr double kEps = 1e-9;

static void log_msg(const std::string& s) { std::cerr << "[build_dataset] " << s << "\n"; }

static std::string path_basename(const std::string& p) {
  auto pos = p.find_last_of("/\\");
  if (pos == std::string::npos) return p;
  return p.substr(pos + 1);
}

/** Текстовая шкала прогресса по строкам одного parquet (stderr, одна строка: \\r + полная очистка). */
static void render_row_progress(int64_t done, int64_t total, const std::string& file_tag) {
  constexpr int kBarW = 40;
  // \r — в начало строки; \033[2K — стереть всю строку (иначе короткий апдейт оставляет «хвост»).
  std::cerr << "\r\033[2K[build_dataset] [";
  if (total > 0) {
    int filled = static_cast<int>(kBarW * done / total);
    if (filled > kBarW) filled = kBarW;
    int pct = static_cast<int>(100 * done / total);
    if (pct > 100) pct = 100;
    for (int i = 0; i < kBarW; ++i) std::cerr << (i < filled ? '#' : '.');
    std::cerr << "] " << std::setw(3) << pct << "%  " << file_tag << "  rows " << done << "/" << total;
  } else {
    for (int i = 0; i < kBarW; ++i) std::cerr << '.';
    std::cerr << "   " << file_tag << "  rows " << done;
  }
  std::cerr << std::flush;
}

static const char* kFeatureNames[] = {
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
    "amount_ratio_to_window_median",
    "amount_iqr_normalized",
    "amount_cv_in_window",
    "sum_amount_last_10m",
    "transactions_last_1h",
    "unique_mcc_count_suffix",
    "unique_device_key_count_suffix",
    "unique_channel_key_count_suffix",
    "unique_timezone_count_suffix",
    "unique_browser_language_count_suffix",
    "mcc_switch_count_last_20_tx",
    "device_switch_count_last_20_tx",
    "channel_switch_count_last_20_tx",
    "night_transaction_share_last_24h",
    "distinct_hours_active_last_24h",
    "mean_amount_last_3_transactions",
    "amount_ratio_to_min_amount_24h",
    "compromised_history_count_suffix",
    "seconds_since_last_compromised_tx",
    "web_rdp_count_last_24h",
    "voip_flag_count_last_24h",
    "channel_relative_freq",
    "timezone_relative_freq",
    "browser_language_relative_freq",
    "event_type_nm_share_in_suffix",
    "mcc_consecutive_streak_length",
    "transactions_last_5m",
    "mean_gap_seconds_last_5_intervals",
    "suffix_time_span_hours_log1p",
    "transactions_per_span_hour",
    "mcc_amount_std_same_5d",
    "weekend_transaction_share_last_7d",
    "distinct_event_descr_count_last_24h_norm",
    "is_new_timezone",
    "is_timezone_change",
    "currency_iso_cd_cat",
    "pos_cd_cat",
    "accept_language_cat",
    "battery_level",
    "battery_very_low_flag",
    "screen_size_cat",
    "developer_tools_flag",
    "accept_lang_browser_lang_mismatch",
    "battery_very_low_and_night",
    "amount_diff_prev",
    "amount_ratio_prev",
    "amount_change_sign",
    "amount_increase_streak_suffix",
    "amount_decrease_streak_suffix",
    "is_new_session_id",
    "session_switch_count_last_20_tx",
    "seconds_since_session_start_in_window",
};
constexpr int kNumFeatures = sizeof(kFeatureNames) / sizeof(kFeatureNames[0]);

struct Txn {
  std::optional<double> amount;
  std::optional<std::chrono::system_clock::time_point> dttm;
  int hour_val = -1;
  int dow_py = -1;
  std::string os_type;
  std::string dev_ver;
  std::string mcc;
  std::string ch_type;
  std::string ch_sub;
  std::string tz;
  std::string compromised;
  std::string web_rdp;
  std::string voip;
  std::string session_id;
  std::string browser_language;
  std::string event_type_nm;
  std::string event_descr;
};

struct UserWindow {
  std::deque<Txn> dq;
  void push(const Txn& t) {
    dq.push_back(t);
    while (static_cast<int>(dq.size()) > kWindowCap) dq.pop_front();
  }
  int count() const { return static_cast<int>(dq.size()); }
};

static uint64_t mix64(uint64_t z) {
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

static uint64_t hash_event_customer(int64_t event_id, const std::string& customer_id) {
  uint64_t h = mix64(static_cast<uint64_t>(event_id));
  for (unsigned char c : customer_id) h = mix64(h ^ (static_cast<uint64_t>(c) + 0x100000001b3ULL));
  return h;
}

struct WindowTargetSampler {
  std::vector<int> values;
  std::vector<double> cdf;

  bool empty() const { return values.empty() || cdf.empty(); }

  int sample(int64_t event_id, const std::string& customer_id) const {
    if (empty()) return 1;
    uint64_t h = hash_event_customer(event_id, customer_id);
    uint64_t h2 = mix64(h ^ 0xD6E8FEB866B13AD5ULL);
    constexpr uint64_t kDenom = (1ULL << 53) - 1ULL;
    const double u = static_cast<double>(h2 & kDenom) / static_cast<double>(kDenom);
    auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
    if (it == cdf.end()) return values.back();
    const size_t idx = static_cast<size_t>(std::distance(cdf.begin(), it));
    return values[idx];
  }
};

static std::string trim_copy(std::string s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.erase(s.begin());
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.pop_back();
  return s;
}

static double cat_from_utf8(const std::string& s) {
  std::string t = trim_copy(s);
  if (t.empty()) return 0.0;
  unsigned char digest[EVP_MAX_MD_SIZE];
  unsigned int diglen = 0;
  EVP_MD_CTX* ctx = EVP_MD_CTX_new();
  if (!ctx) return 0.0;
  if (EVP_DigestInit_ex(ctx, EVP_md5(), nullptr) != 1 || EVP_DigestUpdate(ctx, t.data(), t.size()) != 1 ||
      EVP_DigestFinal_ex(ctx, digest, &diglen) != 1 || diglen != MD5_DIGEST_LENGTH) {
    EVP_MD_CTX_free(ctx);
    return 0.0;
  }
  EVP_MD_CTX_free(ctx);
  long long rem = 0;
  for (unsigned int i = 0; i < diglen; ++i) {
    rem = (rem * 256 + static_cast<long long>(digest[i])) % 1000000LL;
  }
  return static_cast<double>(rem);
}

static std::optional<double> parse_double_any(const std::string& s) {
  std::string t = trim_copy(s);
  if (t.empty()) return std::nullopt;
  char* end = nullptr;
  double v = std::strtod(t.c_str(), &end);
  if (end == t.c_str()) return std::nullopt;
  if (!std::isfinite(v)) return std::nullopt;
  return v;
}

// Наивное локальное время, как pandas / strptime без таймзоны
static bool parse_dttm_fields(const std::string& s_in, std::tm* out_tm) {
  std::string t = trim_copy(s_in);
  if (t.empty()) return false;
  // Обрезаем доли секунды (как в Python strptime с .%f)
  if (t.size() > 19 && t[19] == '.') t.resize(19);
  std::memset(out_tm, 0, sizeof(std::tm));
  const char* r = strptime(t.c_str(), "%Y-%m-%d %H:%M:%S", out_tm);
  if (r == nullptr) return false;
  out_tm->tm_isdst = -1;
  return true;
}

static std::optional<std::chrono::system_clock::time_point> dttm_from_tm(std::tm tm) {
  time_t tt = mktime(&tm);
  if (tt == static_cast<time_t>(-1)) return std::nullopt;
  return std::chrono::system_clock::from_time_t(tt);
}

static void fill_txn_time(Txn& t, const std::string& dttm_s) {
  std::tm tm = {};
  if (!parse_dttm_fields(dttm_s, &tm)) return;
  t.hour_val = tm.tm_hour;
  t.dow_py = (tm.tm_wday + 6) % 7;
  if (auto tp = dttm_from_tm(tm)) t.dttm = tp;
}

static bool empty_field(const std::string& s) { return trim_copy(s).empty(); }

/** Parquet часто отдаёт dictionary-encoded колонки: схема проверяет тип значений, а в батче id() == DICTIONARY. */
static std::string col_get_str(const arrow::Array& col, int64_t row) {
  if (col.IsNull(row)) return {};
  if (col.type_id() == arrow::Type::DICTIONARY) {
    const auto& da = static_cast<const arrow::DictionaryArray&>(col);
    if (da.IsNull(row)) return {};
    int64_t di = da.GetValueIndex(row);
    if (di < 0) return {};
    const auto& dict = *da.dictionary();
    if (di >= dict.length()) return {};
    return col_get_str(dict, di);
  }
  switch (col.type_id()) {
    case arrow::Type::STRING: {
      auto& a = static_cast<const arrow::StringArray&>(col);
      return a.GetString(row);
    }
    case arrow::Type::LARGE_STRING: {
      auto& a = static_cast<const arrow::LargeStringArray&>(col);
      return a.GetString(row);
    }
    case arrow::Type::INT64: {
      auto& a = static_cast<const arrow::Int64Array&>(col);
      return std::to_string(a.Value(row));
    }
    case arrow::Type::INT32: {
      auto& a = static_cast<const arrow::Int32Array&>(col);
      return std::to_string(a.Value(row));
    }
    default:
      return {};
  }
}

/** Сумма операции: Parquet часто хранит operaton_amt как double; col_get_str для float/double возвращает пусто. */
static std::optional<double> col_get_optional_double(const arrow::Array& col, int64_t row) {
  if (col.IsNull(row)) return std::nullopt;
  if (col.type_id() == arrow::Type::DICTIONARY) {
    const auto& da = static_cast<const arrow::DictionaryArray&>(col);
    if (da.IsNull(row)) return std::nullopt;
    int64_t di = da.GetValueIndex(row);
    if (di < 0) return std::nullopt;
    const auto& dict = *da.dictionary();
    if (di >= dict.length()) return std::nullopt;
    return col_get_optional_double(dict, di);
  }
  switch (col.type_id()) {
    case arrow::Type::DOUBLE: {
      double v = static_cast<const arrow::DoubleArray&>(col).Value(row);
      return std::isfinite(v) ? std::optional<double>(v) : std::nullopt;
    }
    case arrow::Type::FLOAT: {
      double v = static_cast<double>(static_cast<const arrow::FloatArray&>(col).Value(row));
      return std::isfinite(v) ? std::optional<double>(v) : std::nullopt;
    }
    case arrow::Type::INT64:
      return static_cast<double>(static_cast<const arrow::Int64Array&>(col).Value(row));
    case arrow::Type::INT32:
      return static_cast<double>(static_cast<const arrow::Int32Array&>(col).Value(row));
    case arrow::Type::STRING:
    case arrow::Type::LARGE_STRING: {
      auto s = col_get_str(col, row);
      return parse_double_any(s);
    }
    default:
      return std::nullopt;
  }
}

static std::optional<int64_t> col_get_int64(const arrow::Array& col, int64_t row) {
  if (col.IsNull(row)) return std::nullopt;
  if (col.type_id() == arrow::Type::DICTIONARY) {
    const auto& da = static_cast<const arrow::DictionaryArray&>(col);
    if (da.IsNull(row)) return std::nullopt;
    int64_t di = da.GetValueIndex(row);
    if (di < 0) return std::nullopt;
    const auto& dict = *da.dictionary();
    if (di >= dict.length()) return std::nullopt;
    return col_get_int64(dict, di);
  }
  switch (col.type_id()) {
    case arrow::Type::INT64:
      return static_cast<const arrow::Int64Array&>(col).Value(row);
    case arrow::Type::INT32:
      return static_cast<int64_t>(static_cast<const arrow::Int32Array&>(col).Value(row));
    case arrow::Type::INT16:
      return static_cast<int64_t>(static_cast<const arrow::Int16Array&>(col).Value(row));
    case arrow::Type::INT8:
      return static_cast<int64_t>(static_cast<const arrow::Int8Array&>(col).Value(row));
    case arrow::Type::UINT32:
      return static_cast<int64_t>(static_cast<const arrow::UInt32Array&>(col).Value(row));
    case arrow::Type::UINT64: {
      uint64_t v = static_cast<const arrow::UInt64Array&>(col).Value(row);
      if (v > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) return std::nullopt;
      return static_cast<int64_t>(v);
    }
    default:
      break;
  }
  auto s = col_get_str(col, row);
  if (s.empty()) return std::nullopt;
  try {
    return std::stoll(s);
  } catch (...) {
    return std::nullopt;
  }
}

static int col_index(const arrow::Schema& sch, const std::string& name) {
  auto f = sch.GetFieldByName(name);
  if (!f) return -1;
  return sch.GetFieldIndex(name);
}

static bool path_is_readable_file(const std::string& p) {
  std::ifstream f(p);
  return f.good();
}

/** Базовый Arrow-тип: для dictionary — тип значений. */
static arrow::Type::type base_arrow_type_id(const std::shared_ptr<arrow::DataType>& dt) {
  if (dt->id() == arrow::Type::DICTIONARY) {
    return static_cast<const arrow::DictionaryType&>(*dt).value_type()->id();
  }
  return dt->id();
}

enum class ColTypeExpect {
  /** int32/int64 или строка с числом (event_id, target в labels). */
  kIntKey,
  /** operaton_amt: число или строка с числом. */
  kAmount,
  /** Текст и пр.: utf8 / large_utf8 / int как в col_get_str. */
  kText,
};

static bool type_matches_expectation(arrow::Type::type tid, ColTypeExpect ex) {
  switch (ex) {
    case ColTypeExpect::kIntKey:
      return tid == arrow::Type::INT64 || tid == arrow::Type::INT32 || tid == arrow::Type::INT8 ||
             tid == arrow::Type::INT16 || tid == arrow::Type::UINT32 || tid == arrow::Type::UINT64 ||
             tid == arrow::Type::STRING || tid == arrow::Type::LARGE_STRING;
    case ColTypeExpect::kAmount:
      return tid == arrow::Type::DOUBLE || tid == arrow::Type::FLOAT || tid == arrow::Type::INT64 ||
             tid == arrow::Type::INT32 || tid == arrow::Type::STRING || tid == arrow::Type::LARGE_STRING;
    case ColTypeExpect::kText:
      return tid == arrow::Type::STRING || tid == arrow::Type::LARGE_STRING || tid == arrow::Type::INT64 ||
             tid == arrow::Type::INT32;
    default:
      return false;
  }
}

struct SchemaColumnSpec {
  const char* name;
  bool required;
  ColTypeExpect expect;
};

// Колонки train/pretrain: обязательные — без них пайплайн не имеет смысла; остальные — если есть, тип должен быть читаем кодом.
static const SchemaColumnSpec kTrainParquetColumns[] = {
    {"customer_id", true, ColTypeExpect::kText},
    {"event_id", true, ColTypeExpect::kIntKey},
    {"operaton_amt", true, ColTypeExpect::kAmount},
    {"event_dttm", true, ColTypeExpect::kText},
    {"operating_system_type", false, ColTypeExpect::kText},
    {"device_system_version", false, ColTypeExpect::kText},
    {"mcc_code", false, ColTypeExpect::kText},
    {"channel_indicator_type", false, ColTypeExpect::kText},
    {"channel_indicator_sub_type", false, ColTypeExpect::kText},
    {"channel_indicator_subtype", false, ColTypeExpect::kText},
    {"timezone", false, ColTypeExpect::kText},
    {"compromised", false, ColTypeExpect::kText},
    {"web_rdp_connection", false, ColTypeExpect::kText},
    {"phone_voip_call_state", false, ColTypeExpect::kText},
    {"session_id", false, ColTypeExpect::kText},
    {"browser_language", false, ColTypeExpect::kText},
    {"event_type_nm", false, ColTypeExpect::kAmount},
    {"event_descr", false, ColTypeExpect::kText},
    {"event_desc", false, ColTypeExpect::kText},
    {"currency_iso_cd", false, ColTypeExpect::kText},
    {"pos_cd", false, ColTypeExpect::kText},
    {"accept_language", false, ColTypeExpect::kText},
    {"battery", false, ColTypeExpect::kAmount},
    {"screen_size", false, ColTypeExpect::kText},
    {"developer_tools", false, ColTypeExpect::kText},
};

static const SchemaColumnSpec kLabelsParquetColumns[] = {
    {"event_id", true, ColTypeExpect::kIntKey},
    {"target", true, ColTypeExpect::kIntKey},
};

static arrow::Status read_parquet_schema_only(const std::string& path, std::shared_ptr<arrow::Schema>* out_schema) {
  arrow::MemoryPool* pool = arrow::default_memory_pool();
  ARROW_ASSIGN_OR_RAISE(auto infile, arrow::io::ReadableFile::Open(path));
  ARROW_ASSIGN_OR_RAISE(auto reader, parquet::arrow::OpenFile(infile, pool));
  std::shared_ptr<arrow::Schema> schema;
  ARROW_RETURN_NOT_OK(reader->GetSchema(&schema));
  *out_schema = std::move(schema);
  return arrow::Status::OK();
}

static arrow::Status validate_parquet_against_specs(const std::string& path, const char* role_tag,
                                                    const SchemaColumnSpec* specs, size_t n_specs) {
  std::shared_ptr<arrow::Schema> schema;
  ARROW_RETURN_NOT_OK(read_parquet_schema_only(path, &schema));
  for (size_t i = 0; i < n_specs; ++i) {
    const SchemaColumnSpec& sp = specs[i];
    auto f = schema->GetFieldByName(sp.name);
    if (!f) {
      if (sp.required) {
        std::string msg = std::string(role_tag) + " " + path + ": нет обязательной колонки \"" + sp.name + "\"";
        return arrow::Status::Invalid(msg);
      }
      continue;
    }
    arrow::Type::type tid = base_arrow_type_id(f->type());
    if (!type_matches_expectation(tid, sp.expect)) {
      std::string msg = std::string(role_tag) + " " + path + ": колонка \"" + sp.name + "\" имеет тип " +
                        f->type()->ToString() +
                        ", ожидается тип, совместимый с чтением в build_dataset (см. ColTypeExpect)";
      return arrow::Status::Invalid(msg);
    }
  }
  return arrow::Status::OK();
}

static arrow::Status validate_all_input_parquets(const std::string& labels_path,
                                                 const std::vector<std::string>& pre_paths,
                                                 const std::vector<std::string>& train_paths) {
  if (!path_is_readable_file(labels_path)) {
    return arrow::Status::Invalid("нет или недоступен файл train_labels: " + labels_path);
  }
  ARROW_RETURN_NOT_OK(validate_parquet_against_specs(labels_path, "[labels]", kLabelsParquetColumns,
                                                     sizeof(kLabelsParquetColumns) / sizeof(kLabelsParquetColumns[0])));

  for (const auto& p : pre_paths) {
    if (!path_is_readable_file(p)) continue;
    ARROW_RETURN_NOT_OK(validate_parquet_against_specs(p, "[pretrain]", kTrainParquetColumns,
                                                       sizeof(kTrainParquetColumns) / sizeof(kTrainParquetColumns[0])));
  }
  for (const auto& p : train_paths) {
    if (!path_is_readable_file(p)) continue;
    ARROW_RETURN_NOT_OK(validate_parquet_against_specs(p, "[train]", kTrainParquetColumns,
                                                       sizeof(kTrainParquetColumns) / sizeof(kTrainParquetColumns[0])));
  }
  return arrow::Status::OK();
}

static Txn row_to_txn(const arrow::RecordBatch& batch, int64_t i, const arrow::Schema& sch) {
  auto get_s = [&](const char* n) -> std::string {
    int idx = col_index(sch, n);
    if (idx < 0) return {};
    return col_get_str(*batch.column(idx), i);
  };
  Txn t;
  int amt_idx = col_index(sch, "operaton_amt");
  if (amt_idx >= 0) {
    if (auto p = col_get_optional_double(*batch.column(amt_idx), i)) t.amount = p;
  }
  std::string dt = get_s("event_dttm");
  if (!dt.empty()) fill_txn_time(t, dt);
  t.os_type = get_s("operating_system_type");
  t.dev_ver = get_s("device_system_version");
  t.mcc = get_s("mcc_code");
  t.ch_type = get_s("channel_indicator_type");
  t.ch_sub = get_s("channel_indicator_sub_type");
  if (t.ch_sub.empty()) t.ch_sub = get_s("channel_indicator_subtype");
  t.tz = get_s("timezone");
  t.compromised = get_s("compromised");
  t.web_rdp = get_s("web_rdp_connection");
  t.voip = get_s("phone_voip_call_state");
  t.session_id = get_s("session_id");
  t.browser_language = get_s("browser_language");
  t.event_type_nm = get_s("event_type_nm");
  if (t.event_type_nm.empty()) {
    int et_idx = col_index(sch, "event_type_nm");
    if (et_idx >= 0) {
      if (auto p = col_get_optional_double(*batch.column(et_idx), i)) {
        t.event_type_nm = std::to_string(*p);
      }
    }
  }
  t.event_descr = get_s("event_descr");
  if (t.event_descr.empty()) t.event_descr = get_s("event_desc");
  return t;
}

struct FeatureRow {
  std::array<double, kNumFeatures> f{};
  /** Не признак модели — только метаданные строки датасета. */
  std::string customer_id;
  int64_t event_id = 0;
  int32_t target = 0;
  double sample_weight = 1.0;
  std::string event_dttm_raw;
};

static double nan_val() { return std::numeric_limits<double>::quiet_NaN(); }

static FeatureRow compute_features(const UserWindow& w, int effective_target_len, const arrow::RecordBatch& batch,
                                   int64_t i, const arrow::Schema& sch,
                                   const std::unordered_map<int64_t, int>& labels_orig) {
  FeatureRow out;

  int64_t eid = 0;
  int eidx = col_index(sch, "event_id");
  if (eidx >= 0) {
    if (auto p = col_get_int64(*batch.column(eidx), i)) eid = *p;
  }
  out.event_id = eid;

  auto itl = labels_orig.find(eid);
  if (itl != labels_orig.end()) {
    out.target = 1;
    out.sample_weight = (itl->second == 1) ? kWeightLabeled1 : kWeightLabeled0;
  } else {
    out.target = 0;
    out.sample_weight = kWeightUnlabeled;
  }

  const int sz = w.count();
  const int eff = (sz <= 0) ? 0 : std::min(sz, std::max(0, effective_target_len));
  const int start = sz - eff;

  auto get_row_s = [&](const char* n) -> std::string {
    int idx = col_index(sch, n);
    if (idx < 0) return {};
    return col_get_str(*batch.column(idx), i);
  };

  double amount = nan_val();
  int amt_col = col_index(sch, "operaton_amt");
  if (amt_col >= 0) {
    if (auto p = col_get_optional_double(*batch.column(amt_col), i)) amount = *p;
  }

  std::string dttm_s = get_row_s("event_dttm");
  out.event_dttm_raw = dttm_s;
  Txn cur_tmp;
  if (!dttm_s.empty()) fill_txn_time(cur_tmp, dttm_s);
  std::optional<std::chrono::system_clock::time_point> dttm = cur_tmp.dttm;
  int hour_val = cur_tmp.hour_val;
  int dow_val = cur_tmp.dow_py;

  std::vector<double> amounts;
  amounts.reserve(static_cast<size_t>(eff));
  for (int k = start; k < sz; ++k) {
    const auto& t = w.dq[static_cast<size_t>(k)];
    if (t.amount.has_value() && std::isfinite(*t.amount)) amounts.push_back(*t.amount);
  }
  size_t n_amt = amounts.size();
  double median = nan_val();
  double p95 = nan_val();
  double mean = nan_val();
  double stdv = nan_val();
  if (n_amt > 0) {
    std::vector<double> sorted = amounts;
    std::sort(sorted.begin(), sorted.end());
    if (n_amt % 2 == 1) {
      median = sorted[n_amt / 2];
    } else {
      median = (sorted[n_amt / 2 - 1] + sorted[n_amt / 2]) / 2.0;
    }
    size_t idx95 = static_cast<size_t>(std::floor(0.95 * static_cast<double>(n_amt - 1)));
    std::nth_element(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(idx95), sorted.end());
    p95 = sorted[idx95];
    double s = 0.0;
    for (double a : amounts) s += a;
    mean = s / static_cast<double>(n_amt);
    if (n_amt >= 2) {
      double sq = 0.0;
      for (double a : amounts) {
        double d = a - mean;
        sq += d * d;
      }
      stdv = std::sqrt(std::max(0.0, sq / static_cast<double>(n_amt - 1)));
    }
  }

  std::optional<double> last_amount;
  std::string last_tz;
  std::string last_mcc;
  std::string last_dkey;
  if (sz > 0) {
    const auto& lt = w.dq[static_cast<size_t>(sz - 1)];
    if (lt.amount.has_value() && std::isfinite(*lt.amount)) last_amount = *lt.amount;
    last_tz = lt.tz;
    last_mcc = lt.mcc;
    last_dkey = lt.os_type + "\x1f" + lt.dev_ver;
  }

  double mean_last_3 = nan_val();
  double mean_last_10 = nan_val();
  if (!amounts.empty()) {
    size_t from3 = (amounts.size() > 3) ? (amounts.size() - 3) : 0;
    size_t from10 = (amounts.size() > 10) ? (amounts.size() - 10) : 0;
    double s3 = 0.0;
    for (size_t ai = from3; ai < amounts.size(); ++ai) s3 += amounts[ai];
    mean_last_3 = s3 / static_cast<double>(amounts.size() - from3);
    double s10 = 0.0;
    for (size_t ai = from10; ai < amounts.size(); ++ai) s10 += amounts[ai];
    mean_last_10 = s10 / static_cast<double>(amounts.size() - from10);
  }

  using clock = std::chrono::system_clock;
  struct TA {
    clock::time_point t;
    double a;
  };
  std::vector<TA> recent;
  if (dttm.has_value()) {
    auto t0 = *dttm - std::chrono::hours(24);
    for (int k = start; k < sz; ++k) {
      const auto& t = w.dq[static_cast<size_t>(k)];
      if (t.dttm.has_value() && *t.dttm >= t0 && t.amount.has_value() && std::isfinite(*t.amount)) {
        recent.push_back({*t.dttm, *t.amount});
      }
    }
  }

  auto count_since = [&](std::chrono::seconds delta) -> int {
    if (!dttm.has_value()) return 0;
    auto thr = *dttm - delta;
    int c = 0;
    for (const auto& p : recent)
      if (p.t >= thr) ++c;
    return c;
  };
  auto sum_since = [&](std::chrono::seconds delta) -> double {
    if (!dttm.has_value()) return 0.0;
    auto thr = *dttm - delta;
    double s = 0.0;
    for (const auto& p : recent)
      if (p.t >= thr) s += p.a;
    return s;
  };
  double max24 = nan_val();
  if (!recent.empty()) {
    max24 = recent[0].a;
    for (const auto& p : recent) max24 = std::max(max24, p.a);
  }

  std::optional<clock::time_point> last_ev;
  for (int k = sz - 1; k >= start; --k) {
    const auto& t = w.dq[static_cast<size_t>(k)];
    if (t.dttm.has_value()) {
      last_ev = t.dttm;
      break;
    }
  }

  std::unordered_map<std::string, int> dev_cnt, mcc_cnt, ch_cnt, tz_cnt, bl_cnt, dev_tz_cnt;
  std::unordered_map<std::string, int> sess_cnt;
  std::unordered_map<std::string, double> sess_amount_sum;
  std::unordered_map<std::string, clock::time_point> sess_first_dttm;
  std::unordered_map<std::string, clock::time_point> sess_last_dttm;
  for (int k = start; k < sz; ++k) {
    const auto& t = w.dq[static_cast<size_t>(k)];
    std::string dkey = t.os_type + "\x1f" + t.dev_ver;
    std::string ckey = t.ch_type + "\x1f" + t.ch_sub;
    if (!empty_field(t.os_type) || !empty_field(t.dev_ver)) dev_cnt[dkey]++;
    if (!empty_field(t.mcc)) mcc_cnt[t.mcc]++;
    if (!empty_field(t.ch_type) || !empty_field(t.ch_sub)) ch_cnt[ckey]++;
    if (!empty_field(t.tz)) tz_cnt[t.tz]++;
    if ((!empty_field(t.os_type) || !empty_field(t.dev_ver)) && !empty_field(t.tz)) {
      dev_tz_cnt[dkey + "\x1f" + t.tz]++;
    }
    if (!empty_field(t.browser_language)) bl_cnt[t.browser_language]++;
    if (!t.session_id.empty()) {
      sess_cnt[t.session_id]++;
      if (t.amount.has_value() && std::isfinite(*t.amount)) sess_amount_sum[t.session_id] += *t.amount;
      if (t.dttm.has_value()) {
        auto itf = sess_first_dttm.find(t.session_id);
        if (itf == sess_first_dttm.end() || *t.dttm < itf->second) sess_first_dttm[t.session_id] = *t.dttm;
        auto itl = sess_last_dttm.find(t.session_id);
        if (itl == sess_last_dttm.end() || *t.dttm > itl->second) sess_last_dttm[t.session_id] = *t.dttm;
      }
    }
  }

  std::string os_c = get_row_s("operating_system_type");
  std::string dev_c = get_row_s("device_system_version");
  std::string mcc_c = get_row_s("mcc_code");
  std::string ch_t = get_row_s("channel_indicator_type");
  std::string ch_s = get_row_s("channel_indicator_sub_type");
  if (ch_s.empty()) ch_s = get_row_s("channel_indicator_subtype");
  std::string tz_c = get_row_s("timezone");
  std::string bl_c = get_row_s("browser_language");
  std::string sid_c = get_row_s("session_id");
  std::string descr = get_row_s("event_descr");
  if (descr.empty()) descr = get_row_s("event_desc");
  double event_type_nm_feat = nan_val();
  {
    int et_idx = col_index(sch, "event_type_nm");
    if (et_idx >= 0) {
      const auto& et_col = *batch.column(et_idx);
      if (auto p = col_get_optional_double(et_col, i)) {
        if (std::isfinite(*p)) event_type_nm_feat = *p;
      } else if (auto p2 = parse_double_any(trim_copy(col_get_str(et_col, i)))) {
        if (std::isfinite(*p2)) event_type_nm_feat = *p2;
      }
    }
  }
  std::string descr_trim = trim_copy(descr);
  std::string mcc_trim = trim_copy(mcc_c);

  // Frequencies of current event_descr / mcc_code in last 1h / 6h / 24h (relative to current dttm).
  auto freq_descr_last = [&](std::chrono::seconds delta) -> double {
    if (!dttm.has_value()) return 0.0;
    if (descr_trim.empty()) return 0.0;
    auto thr = *dttm - delta;
    int c = 0;
    for (int k = start; k < sz; ++k) {
      const auto& t = w.dq[static_cast<size_t>(k)];
      if (!t.dttm.has_value()) continue;
      if (*t.dttm < thr) continue;
      if (trim_copy(t.event_descr) == descr_trim) ++c;
    }
    return static_cast<double>(c);
  };
  auto freq_mcc_last = [&](std::chrono::seconds delta) -> double {
    if (!dttm.has_value()) return 0.0;
    if (mcc_trim.empty()) return 0.0;
    auto thr = *dttm - delta;
    int c = 0;
    for (int k = start; k < sz; ++k) {
      const auto& t = w.dq[static_cast<size_t>(k)];
      if (!t.dttm.has_value()) continue;
      if (*t.dttm < thr) continue;
      if (trim_copy(t.mcc) == mcc_trim) ++c;
    }
    return static_cast<double>(c);
  };
  auto freq_event_type_nm_last = [&](std::chrono::seconds delta) -> double {
    if (!dttm.has_value()) return 0.0;
    if (!std::isfinite(event_type_nm_feat)) return 0.0;
    const double target = event_type_nm_feat;
    auto thr = *dttm - delta;
    int c = 0;
    for (int k = start; k < sz; ++k) {
      const auto& t = w.dq[static_cast<size_t>(k)];
      if (!t.dttm.has_value()) continue;
      if (*t.dttm < thr) continue;
      if (auto p = parse_double_any(trim_copy(t.event_type_nm))) {
        if (std::isfinite(*p) && *p == target) ++c;
      }
    }
    return static_cast<double>(c);
  };

  double event_descr_freq_last_1h = freq_descr_last(std::chrono::hours(1));
  double event_descr_freq_last_6h = freq_descr_last(std::chrono::hours(6));
  double event_descr_freq_last_24h = freq_descr_last(std::chrono::hours(24));
  double event_type_nm_freq_last_1h = freq_event_type_nm_last(std::chrono::hours(1));
  double event_type_nm_freq_last_6h = freq_event_type_nm_last(std::chrono::hours(6));
  double event_type_nm_freq_last_24h = freq_event_type_nm_last(std::chrono::hours(24));
  double mcc_freq_last_1h = freq_mcc_last(std::chrono::hours(1));
  double mcc_freq_last_6h = freq_mcc_last(std::chrono::hours(6));
  double mcc_freq_last_24h = freq_mcc_last(std::chrono::hours(24));

  // Pair novelty: 1 if (mcc, event_descr) was never seen before in the window suffix.
  double mcc_event_descr_pair_new = 0.0;
  if (!mcc_trim.empty() && !descr_trim.empty()) {
    bool seen = false;
    for (int k = start; k < sz; ++k) {
      const auto& t = w.dq[static_cast<size_t>(k)];
      if (trim_copy(t.mcc) == mcc_trim && trim_copy(t.event_descr) == descr_trim) {
        seen = true;
        break;
      }
    }
    mcc_event_descr_pair_new = seen ? 0.0 : 1.0;
  }

  // Ratio of "high amount" transactions (> p95) among last 24h transactions.
  double high_amount_ratio_last_24h = 0.0;
  if (std::isfinite(p95) && dttm.has_value()) {
    int cnt_recent = 0;
    int cnt_high = 0;
    auto thr = *dttm - std::chrono::hours(24);
    for (int k = start; k < sz; ++k) {
      const auto& t = w.dq[static_cast<size_t>(k)];
      if (!t.dttm.has_value()) continue;
      if (*t.dttm < thr) continue;
      if (!t.amount.has_value() || !std::isfinite(*t.amount)) continue;
      ++cnt_recent;
      if (*t.amount > p95) ++cnt_high;
    }
    if (cnt_recent > 0) high_amount_ratio_last_24h = static_cast<double>(cnt_high) / static_cast<double>(cnt_recent);
  }

  // Current amount relative to median amount for same mcc in last 5 days.
  double amount_relative_to_mcc_median_5_days = nan_val();
  if (dttm.has_value() && std::isfinite(amount) && !mcc_trim.empty()) {
    auto thr = *dttm - std::chrono::hours(24 * 5);
    std::vector<double> mcc_amounts;
    for (int k = start; k < sz; ++k) {
      const auto& t = w.dq[static_cast<size_t>(k)];
      if (!t.dttm.has_value()) continue;
      if (*t.dttm < thr) continue;
      if (!t.amount.has_value() || !std::isfinite(*t.amount)) continue;
      if (trim_copy(t.mcc) != mcc_trim) continue;
      mcc_amounts.push_back(*t.amount);
    }
    if (!mcc_amounts.empty()) {
      std::sort(mcc_amounts.begin(), mcc_amounts.end());
      size_t n = mcc_amounts.size();
      double med = nan_val();
      if (n % 2 == 1) {
        med = mcc_amounts[n / 2];
      } else {
        med = (mcc_amounts[n / 2 - 1] + mcc_amounts[n / 2]) / 2.0;
      }
      if (std::isfinite(med) && med != 0.0) amount_relative_to_mcc_median_5_days = amount / med;
    }
  }

  std::string dkey_c = os_c + "\x1f" + dev_c;
  std::string ckey_c = ch_t + "\x1f" + ch_s;
  std::string dtz_key_c = dkey_c + "\x1f" + tz_c;

  auto get_count_i = [&](const std::unordered_map<std::string, int>& mp, const std::string& key, bool key_ok) -> int {
    if (!key_ok) return 0;
    auto it = mp.find(key);
    return (it == mp.end()) ? 0 : it->second;
  };

  int device_count_i = get_count_i(dev_cnt, dkey_c, (!empty_field(os_c) || !empty_field(dev_c)));
  int mcc_count_i = get_count_i(mcc_cnt, mcc_c, !empty_field(mcc_c));
  int channel_count_i = get_count_i(ch_cnt, ckey_c, (!empty_field(ch_t) || !empty_field(ch_s)));
  int timezone_count_i = get_count_i(tz_cnt, tz_c, !empty_field(tz_c));
  int bl_count_i = get_count_i(bl_cnt, bl_c, !empty_field(bl_c));

  double is_new_device = 0.0;
  if (!empty_field(os_c) || !empty_field(dev_c)) is_new_device = (device_count_i == 0) ? 1.0 : 0.0;
  double is_new_mcc = 0.0;
  if (!empty_field(mcc_c)) is_new_mcc = (mcc_count_i == 0) ? 1.0 : 0.0;
  double is_new_channel = 0.0;
  if (!empty_field(ch_t) || !empty_field(ch_s)) is_new_channel = (channel_count_i == 0) ? 1.0 : 0.0;
  double is_new_timezone = 0.0;
  if (!empty_field(tz_c)) is_new_timezone = (timezone_count_i == 0) ? 1.0 : 0.0;
  double is_new_browser_language = 0.0;
  if (!empty_field(bl_c)) is_new_browser_language = (bl_count_i == 0) ? 1.0 : 0.0;
  double is_new_device_tz_pair = 0.0;
  if ((!empty_field(os_c) || !empty_field(dev_c)) && !empty_field(tz_c)) {
    is_new_device_tz_pair = (dev_tz_cnt.find(dtz_key_c) == dev_tz_cnt.end()) ? 1.0 : 0.0;
  }

  int n_sess = 0;
  if (!sid_c.empty()) {
    auto it = sess_cnt.find(sid_c);
    if (it != sess_cnt.end()) n_sess = it->second;
  }
  double session_mean_amount = nan_val();
  auto it_sc = sess_cnt.find(sid_c);
  if (it_sc != sess_cnt.end() && it_sc->second > 0) {
    double ssum = sess_amount_sum[sid_c];
    session_mean_amount = ssum / static_cast<double>(it_sc->second);
  }
  double session_duration = nan_val();
  auto it_sf = sess_first_dttm.find(sid_c);
  auto it_sl = sess_last_dttm.find(sid_c);
  if (it_sf != sess_first_dttm.end() && it_sl != sess_last_dttm.end()) {
    auto sec = std::chrono::duration_cast<std::chrono::seconds>(it_sl->second - it_sf->second).count();
    session_duration = static_cast<double>(sec);
  }

  double amount_to_median = nan_val();
  if (std::isfinite(median) && median != 0.0 && std::isfinite(amount)) amount_to_median = amount / median;

  double amount_zscore = nan_val();
  if (std::isfinite(stdv) && stdv > 0.0 && std::isfinite(amount)) amount_zscore = (amount - mean) / stdv;

  double is_amount_high = 0.0;
  if (std::isfinite(p95) && std::isfinite(amount) && amount > p95) is_amount_high = 1.0;

  double max_amount_feat = 0.0;
  if (std::isfinite(max24)) max_amount_feat = max24;

  double time_since = -1.0;
  if (dttm.has_value() && last_ev.has_value()) {
    auto sec = std::chrono::duration_cast<std::chrono::seconds>(*dttm - *last_ev).count();
    time_since = static_cast<double>(sec);
  }

  double amount_diff_prev = nan_val();
  double amount_ratio_prev = nan_val();
  if (last_amount.has_value() && std::isfinite(amount)) {
    amount_diff_prev = amount - *last_amount;
    amount_ratio_prev = amount / ((std::abs(*last_amount) > kEps) ? *last_amount : kEps);
  }

  double trend_mean_last_3_to_10 = nan_val();
  if (std::isfinite(mean_last_3) && std::isfinite(mean_last_10)) {
    trend_mean_last_3_to_10 = mean_last_3 / ((std::abs(mean_last_10) > kEps) ? mean_last_10 : kEps);
  }
  double amount_percentile_rank = nan_val();
  if (!amounts.empty() && std::isfinite(amount)) {
    int le = 0;
    for (double a : amounts)
      if (a <= amount) ++le;
    amount_percentile_rank = static_cast<double>(le) / static_cast<double>(amounts.size());
  }

  std::vector<double> deltas;
  std::optional<clock::time_point> prev_dt;
  for (int k = start; k < sz; ++k) {
    const auto& t = w.dq[static_cast<size_t>(k)];
    if (!t.dttm.has_value()) continue;
    if (prev_dt.has_value()) {
      auto sec = std::chrono::duration_cast<std::chrono::seconds>(*t.dttm - *prev_dt).count();
      deltas.push_back(static_cast<double>(sec));
    }
    prev_dt = *t.dttm;
  }
  double std_time_deltas = nan_val();
  if (deltas.size() >= 2) {
    double m = 0.0;
    for (double d : deltas) m += d;
    m /= static_cast<double>(deltas.size());
    double sq = 0.0;
    for (double d : deltas) {
      double z = d - m;
      sq += z * z;
    }
    std_time_deltas = std::sqrt(std::max(0.0, sq / static_cast<double>(deltas.size() - 1)));
  }
  double delta_1 = (deltas.size() >= 1) ? deltas[deltas.size() - 1] : nan_val();
  double delta_2 = (deltas.size() >= 2) ? deltas[deltas.size() - 2] : nan_val();
  double delta_3 = (deltas.size() >= 3) ? deltas[deltas.size() - 3] : nan_val();
  double acceleration = nan_val();
  if (std::isfinite(delta_1) && std::isfinite(delta_2)) {
    acceleration = delta_1 / ((std::abs(delta_2) > kEps) ? delta_2 : kEps);
  }
  double std_delta_last_k = nan_val();
  double cv_delta_last_k = nan_val();
  if (deltas.size() >= 2) {
    size_t from = (deltas.size() > 10) ? (deltas.size() - 10) : 0;
    size_t n = deltas.size() - from;
    if (n >= 2) {
      double mk = 0.0;
      for (size_t di = from; di < deltas.size(); ++di) mk += deltas[di];
      mk /= static_cast<double>(n);
      double sqk = 0.0;
      for (size_t di = from; di < deltas.size(); ++di) {
        double z = deltas[di] - mk;
        sqk += z * z;
      }
      std_delta_last_k = std::sqrt(std::max(0.0, sqk / static_cast<double>(n - 1)));
      cv_delta_last_k = std_delta_last_k / ((std::abs(mk) > kEps) ? mk : kEps);
    }
  }

  double is_timezone_change = 0.0;
  if (!empty_field(tz_c) && !empty_field(last_tz)) is_timezone_change = (tz_c != last_tz) ? 1.0 : 0.0;
  double is_device_switch = 0.0;
  if (!empty_field(os_c) || !empty_field(dev_c)) is_device_switch = (dkey_c != last_dkey) ? 1.0 : 0.0;
  double is_mcc_switch = 0.0;
  if (!empty_field(mcc_c) && !empty_field(last_mcc)) is_mcc_switch = (mcc_c != last_mcc) ? 1.0 : 0.0;

  double time_since_last_device_change = nan_val();
  double time_since_last_mcc_change = nan_val();
  if (dttm.has_value()) {
    for (int k = sz - 1; k >= start; --k) {
      const auto& t = w.dq[static_cast<size_t>(k)];
      if (!t.dttm.has_value()) continue;
      if (!std::isfinite(time_since_last_device_change) && (!empty_field(os_c) || !empty_field(dev_c))) {
        std::string tdkey = t.os_type + "\x1f" + t.dev_ver;
        if (tdkey != dkey_c) {
          auto sec = std::chrono::duration_cast<std::chrono::seconds>(*dttm - *t.dttm).count();
          time_since_last_device_change = static_cast<double>(sec);
        }
      }
      if (!std::isfinite(time_since_last_mcc_change) && !empty_field(mcc_c) && t.mcc != mcc_c) {
        auto sec = std::chrono::duration_cast<std::chrono::seconds>(*dttm - *t.dttm).count();
        time_since_last_mcc_change = static_cast<double>(sec);
      }
      if (std::isfinite(time_since_last_device_change) && std::isfinite(time_since_last_mcc_change)) break;
    }
  }

  double hour_f = dttm.has_value() ? static_cast<double>(hour_val) : nan_val();
  double dow_f = dttm.has_value() ? static_cast<double>(dow_val) : nan_val();
  double is_night = nan_val();
  double is_weekend = nan_val();
  if (dttm.has_value()) {
    is_night = (hour_val >= 22 || hour_val < 6) ? 1.0 : 0.0;
    is_weekend = (dow_val >= 5) ? 1.0 : 0.0;
  }

  const double tr_am = static_cast<double>(eff);
  const double log_1p_tx = std::log1p(std::max(0.0, tr_am));

  std::string compromised_s = trim_copy(get_row_s("compromised"));
  double compromised_flag = 0.0;
  if (!compromised_s.empty()) {
    if (auto pc = parse_double_any(compromised_s)) compromised_flag = (*pc == 1.0) ? 1.0 : 0.0;
  }

  double tx_10m = static_cast<double>(count_since(std::chrono::minutes(10)));
  double tx_1h = static_cast<double>(count_since(std::chrono::hours(1)));
  double tx_24h = static_cast<double>(count_since(std::chrono::hours(24)));
  double sum_1h = sum_since(std::chrono::hours(1));
  double sum_24h = sum_since(std::chrono::hours(24));
  double sum_amount_last_10m = sum_since(std::chrono::minutes(10));
  double tx_5m = static_cast<double>(count_since(std::chrono::minutes(5)));

  double amount_iqr_normalized = nan_val();
  if (n_amt >= 2 && std::isfinite(amount)) {
    std::vector<double> s_iqr = amounts;
    std::sort(s_iqr.begin(), s_iqr.end());
    size_t i25 = (n_amt - 1) / 4;
    size_t i75 = (3 * (n_amt - 1)) / 4;
    double q25 = s_iqr[i25];
    double q75 = s_iqr[i75];
    double iqr = q75 - q25;
    double denom = (std::abs(iqr) < kEps) ? kEps : iqr;
    amount_iqr_normalized = (amount - q25) / denom;
  }

  double amount_cv_in_window = nan_val();
  if (std::isfinite(mean) && std::abs(mean) > kEps && std::isfinite(stdv)) amount_cv_in_window = stdv / mean;

  std::unordered_set<std::string> umcc, udev, uch, utz, ubl;
  for (int k = start; k < sz; ++k) {
    const auto& t = w.dq[static_cast<size_t>(k)];
    std::string mc = trim_copy(t.mcc);
    if (!mc.empty()) umcc.insert(mc);
    std::string dk = t.os_type + "\x1f" + t.dev_ver;
    if (!empty_field(t.os_type) || !empty_field(t.dev_ver)) udev.insert(dk);
    std::string ck = t.ch_type + "\x1f" + t.ch_sub;
    if (!empty_field(t.ch_type) || !empty_field(t.ch_sub)) uch.insert(ck);
    std::string tzs = trim_copy(t.tz);
    if (!tzs.empty()) utz.insert(tzs);
    std::string bls = trim_copy(t.browser_language);
    if (!bls.empty()) ubl.insert(bls);
  }

  int Lsw = std::min(20, eff);
  int mcc_switch_cnt = 0;
  int dev_switch_cnt = 0;
  int ch_switch_cnt = 0;
  if (Lsw >= 2) {
    int from = sz - Lsw;
    for (int i = from; i < sz - 1; ++i) {
      const auto& a = w.dq[static_cast<size_t>(i)];
      const auto& b = w.dq[static_cast<size_t>(i + 1)];
      if (trim_copy(a.mcc) != trim_copy(b.mcc)) ++mcc_switch_cnt;
      if (a.os_type + "\x1f" + a.dev_ver != b.os_type + "\x1f" + b.dev_ver) ++dev_switch_cnt;
      if (a.ch_type + "\x1f" + a.ch_sub != b.ch_type + "\x1f" + b.ch_sub) ++ch_switch_cnt;
    }
  }

  double night_share_24h = 0.0;
  double distinct_hours_24h = 0.0;
  if (dttm.has_value()) {
    auto thr24 = *dttm - std::chrono::hours(24);
    int cnt24 = 0;
    int night24 = 0;
    std::unordered_set<int> hrs;
    for (int k = start; k < sz; ++k) {
      const auto& t = w.dq[static_cast<size_t>(k)];
      if (!t.dttm.has_value() || *t.dttm < thr24) continue;
      ++cnt24;
      int h = t.hour_val;
      if (h >= 0 && h <= 23) hrs.insert(h);
      if (h >= 22 || h < 6) ++night24;
    }
    if (cnt24 > 0) night_share_24h = static_cast<double>(night24) / static_cast<double>(cnt24);
    distinct_hours_24h = static_cast<double>(hrs.size()) / 24.0;
  }

  double amount_ratio_to_min_amount_24h = nan_val();
  if (dttm.has_value() && std::isfinite(amount)) {
    auto thr24 = *dttm - std::chrono::hours(24);
    double amin = nan_val();
    for (int k = start; k < sz; ++k) {
      const auto& t = w.dq[static_cast<size_t>(k)];
      if (!t.dttm.has_value() || *t.dttm < thr24) continue;
      if (!t.amount.has_value() || !std::isfinite(*t.amount)) continue;
      if (!std::isfinite(amin) || *t.amount < amin) amin = *t.amount;
    }
    if (std::isfinite(amin) && amin > kEps) amount_ratio_to_min_amount_24h = amount / amin;
  }

  int compromised_hist = 0;
  double secs_since_compromised = nan_val();
  if (dttm.has_value()) {
    for (int k = start; k < sz; ++k) {
      const auto& t = w.dq[static_cast<size_t>(k)];
      if (auto pc = parse_double_any(trim_copy(t.compromised))) {
        if (*pc == 1.0) ++compromised_hist;
      }
    }
    for (int k = sz - 1; k >= start; --k) {
      const auto& t = w.dq[static_cast<size_t>(k)];
      if (auto pc = parse_double_any(trim_copy(t.compromised))) {
        if (*pc == 1.0 && t.dttm.has_value()) {
          secs_since_compromised =
              static_cast<double>(std::chrono::duration_cast<std::chrono::seconds>(*dttm - *t.dttm).count());
          break;
        }
      }
    }
  }

  int web_rdp_cnt_24h = 0;
  int voip_cnt_24h = 0;
  if (dttm.has_value()) {
    auto thr24 = *dttm - std::chrono::hours(24);
    for (int k = start; k < sz; ++k) {
      const auto& t = w.dq[static_cast<size_t>(k)];
      if (!t.dttm.has_value() || *t.dttm < thr24) continue;
      if (!empty_field(t.web_rdp)) ++web_rdp_cnt_24h;
      if (!empty_field(t.voip)) ++voip_cnt_24h;
    }
  }

  double channel_rel_freq = (eff > 0) ? (static_cast<double>(channel_count_i) / static_cast<double>(eff)) : nan_val();
  double tz_rel_freq = (eff > 0) ? (static_cast<double>(timezone_count_i) / static_cast<double>(eff)) : nan_val();
  double bl_rel_freq = (eff > 0) ? (static_cast<double>(bl_count_i) / static_cast<double>(eff)) : nan_val();

  double event_type_nm_share_suffix = 0.0;
  if (eff > 0 && std::isfinite(event_type_nm_feat)) {
    int match = 0;
    for (int k = start; k < sz; ++k) {
      const auto& t = w.dq[static_cast<size_t>(k)];
      if (auto p = parse_double_any(trim_copy(t.event_type_nm))) {
        if (std::isfinite(*p) && *p == event_type_nm_feat) ++match;
      }
    }
    event_type_nm_share_suffix = static_cast<double>(match) / static_cast<double>(eff);
  }

  double mcc_streak = 0.0;
  if (!mcc_trim.empty()) {
    int st = 0;
    for (int k = sz - 1; k >= start; --k) {
      if (trim_copy(w.dq[static_cast<size_t>(k)].mcc) == mcc_trim)
        ++st;
      else
        break;
    }
    mcc_streak = static_cast<double>(st);
  }

  double mean_gap_last_5 = nan_val();
  if (!deltas.empty()) {
    size_t n5 = std::min<size_t>(5, deltas.size());
    double sg = 0.0;
    for (size_t j = deltas.size() - n5; j < deltas.size(); ++j) sg += deltas[j];
    mean_gap_last_5 = sg / static_cast<double>(n5);
  }

  std::optional<clock::time_point> span_min, span_max;
  for (int k = start; k < sz; ++k) {
    const auto& t = w.dq[static_cast<size_t>(k)];
    if (!t.dttm.has_value()) continue;
    if (!span_min.has_value() || *t.dttm < *span_min) span_min = t.dttm;
    if (!span_max.has_value() || *t.dttm > *span_max) span_max = t.dttm;
  }
  double suffix_span_log1p = nan_val();
  double tx_per_span_hour = nan_val();
  if (span_min.has_value() && span_max.has_value() && eff > 0) {
    double sec_span =
        static_cast<double>(std::chrono::duration_cast<std::chrono::seconds>(*span_max - *span_min).count());
    double hours = sec_span / 3600.0;
    if (hours < kEps) hours = kEps;
    suffix_span_log1p = std::log1p(std::max(0.0, sec_span / 3600.0));
    tx_per_span_hour = static_cast<double>(eff) / hours;
  }

  double mcc_std_5d = nan_val();
  if (dttm.has_value() && !mcc_trim.empty()) {
    auto thr5 = *dttm - std::chrono::hours(24 * 5);
    std::vector<double> mv;
    for (int k = start; k < sz; ++k) {
      const auto& t = w.dq[static_cast<size_t>(k)];
      if (!t.dttm.has_value() || *t.dttm < thr5) continue;
      if (!t.amount.has_value() || !std::isfinite(*t.amount)) continue;
      if (trim_copy(t.mcc) != mcc_trim) continue;
      mv.push_back(*t.amount);
    }
    if (mv.size() >= 2) {
      double sm = 0.0;
      for (double v : mv) sm += v;
      sm /= static_cast<double>(mv.size());
      double sq = 0.0;
      for (double v : mv) {
        double z = v - sm;
        sq += z * z;
      }
      mcc_std_5d = std::sqrt(std::max(0.0, sq / static_cast<double>(mv.size() - 1)));
    }
  }

  double weekend_share_7d = 0.0;
  if (dttm.has_value()) {
    auto thr7 = *dttm - std::chrono::hours(24 * 7);
    int c7 = 0;
    int wk = 0;
    for (int k = start; k < sz; ++k) {
      const auto& t = w.dq[static_cast<size_t>(k)];
      if (!t.dttm.has_value() || *t.dttm < thr7) continue;
      ++c7;
      if (t.dow_py >= 5) ++wk;
    }
    if (c7 > 0) weekend_share_7d = static_cast<double>(wk) / static_cast<double>(c7);
  }

  double descr_div_24h = 0.0;
  if (dttm.has_value()) {
    auto thr24 = *dttm - std::chrono::hours(24);
    std::unordered_set<std::string> dset;
    int cntd = 0;
    for (int k = start; k < sz; ++k) {
      const auto& t = w.dq[static_cast<size_t>(k)];
      if (!t.dttm.has_value() || *t.dttm < thr24) continue;
      ++cntd;
      std::string ds = trim_copy(t.event_descr);
      if (!ds.empty()) dset.insert(ds);
    }
    int denom = std::max(1, cntd);
    descr_div_24h = static_cast<double>(dset.size()) / static_cast<double>(denom);
  }

  std::string cur_currency = trim_copy(get_row_s("currency_iso_cd"));
  std::string cur_pos = trim_copy(get_row_s("pos_cd"));
  std::string cur_accept = trim_copy(get_row_s("accept_language"));
  std::string cur_screen = trim_copy(get_row_s("screen_size"));
  std::string cur_devtools = trim_copy(get_row_s("developer_tools"));

  double battery_level_feat = nan_val();
  int bat_col = col_index(sch, "battery");
  if (bat_col >= 0) {
    if (auto pb = col_get_optional_double(*batch.column(bat_col), i)) battery_level_feat = *pb;
  }
  if (!std::isfinite(battery_level_feat)) {
    if (auto pb2 = parse_double_any(trim_copy(get_row_s("battery")))) battery_level_feat = *pb2;
  }

  double battery_very_low = 0.0;
  if (std::isfinite(battery_level_feat)) {
    if (battery_level_feat >= 0.0 && battery_level_feat <= 1.0 && battery_level_feat < 0.2) battery_very_low = 1.0;
    if (battery_level_feat > 1.0 && battery_level_feat <= 100.0 && battery_level_feat < 20.0) battery_very_low = 1.0;
  }

  double devtools_flag = 0.0;
  if (!empty_field(cur_devtools)) {
    if (auto pd = parse_double_any(cur_devtools)) {
      devtools_flag = (*pd != 0.0 && std::isfinite(*pd)) ? 1.0 : 0.0;
    } else {
      std::string x = cur_devtools;
      for (auto& c : x) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      if (x == "false" || x == "no" || x == "off" || x == "0")
        devtools_flag = 0.0;
      else
        devtools_flag = 1.0;
    }
  }

  std::string bl_trim_row = trim_copy(bl_c);
  double lang_mismatch = 0.0;
  if (!empty_field(cur_accept) && !empty_field(bl_trim_row) && cur_accept != bl_trim_row) lang_mismatch = 1.0;

  double bat_low_x_night = 0.0;
  if (battery_very_low > 0.5 && std::isfinite(is_night) && is_night == 1.0) bat_low_x_night = 1.0;

  double amount_change_sign = nan_val();
  if (std::isfinite(amount_diff_prev)) {
    if (std::abs(amount_diff_prev) < kEps)
      amount_change_sign = 0.0;
    else if (amount_diff_prev > 0.0)
      amount_change_sign = 1.0;
    else
      amount_change_sign = -1.0;
  }

  int amount_inc_streak = 0;
  int amount_dec_streak = 0;
  if (amounts.size() >= 2) {
    int j = static_cast<int>(amounts.size()) - 1;
    while (j >= 1 && amounts[static_cast<size_t>(j)] > amounts[static_cast<size_t>(j - 1)]) {
      ++amount_inc_streak;
      --j;
    }
    j = static_cast<int>(amounts.size()) - 1;
    while (j >= 1 && amounts[static_cast<size_t>(j)] < amounts[static_cast<size_t>(j - 1)]) {
      ++amount_dec_streak;
      --j;
    }
  }

  double is_new_sess_id = 0.0;
  if (!sid_c.empty()) is_new_sess_id = (n_sess == 0) ? 1.0 : 0.0;

  int session_switch_cnt = 0;
  int Lsess = std::min(20, eff);
  if (Lsess >= 2) {
    int from_s = sz - Lsess;
    for (int ii = from_s; ii < sz - 1; ++ii) {
      const auto& ta = w.dq[static_cast<size_t>(ii)];
      const auto& tb = w.dq[static_cast<size_t>(ii + 1)];
      if (ta.session_id != tb.session_id) ++session_switch_cnt;
    }
  }

  double seconds_since_sess_start = nan_val();
  if (dttm.has_value() && !sid_c.empty()) {
    auto it0 = sess_first_dttm.find(sid_c);
    if (it0 != sess_first_dttm.end()) {
      seconds_since_sess_start =
          static_cast<double>(std::chrono::duration_cast<std::chrono::seconds>(*dttm - it0->second).count());
    } else {
      seconds_since_sess_start = 0.0;
    }
  }

  int fi = 0;
  auto put = [&](double v) { out.f[fi++] = v; };
  put(amount);
  put(log_1p_tx);
  put(amount_zscore);
  put(is_amount_high);
  put(tx_24h);
  put(sum_1h);
  put(max_amount_feat);
  put(is_new_device);
  put(is_new_mcc);
  put(is_new_channel);
  put(compromised_flag);
  put(empty_field(get_row_s("web_rdp_connection")) ? 0.0 : 1.0);
  put(empty_field(get_row_s("phone_voip_call_state")) ? 0.0 : 1.0);
  put(hour_f);
  put(dow_f);
  put(is_night);
  put(is_weekend);
  put(tx_10m);
  put(sum_24h);
  put(time_since);
  put(is_new_browser_language);
  put(static_cast<double>(n_sess + 1));
  put(empty_field(tz_c) ? 1.0 : 0.0);
  put(tr_am);
  put(event_type_nm_feat);
  put(cat_from_utf8(descr));
  put(cat_from_utf8(mcc_c));
  put(trend_mean_last_3_to_10);
  put(amount_percentile_rank);
  put(std_time_deltas);
  put(is_new_device_tz_pair);
  put(is_device_switch);
  put(is_mcc_switch);
  put(session_duration);
  put(session_mean_amount);
  put((eff > 0) ? (static_cast<double>(device_count_i) / static_cast<double>(eff)) : nan_val());
  put(delta_1);
  put(delta_2);
  put(delta_3);
  put(acceleration);
  put(std_delta_last_k);
  put(time_since_last_device_change);
  put(time_since_last_mcc_change);
  put(event_descr_freq_last_1h);
  put(event_descr_freq_last_6h);
  put(event_descr_freq_last_24h);
  put(event_type_nm_freq_last_1h);
  put(event_type_nm_freq_last_6h);
  put(event_type_nm_freq_last_24h);
  put(mcc_freq_last_1h);
  put(mcc_freq_last_6h);
  put(mcc_freq_last_24h);
  put(mcc_event_descr_pair_new);
  put(high_amount_ratio_last_24h);
  put(amount_relative_to_mcc_median_5_days);
  put(amount_to_median);
  put(amount_iqr_normalized);
  put(amount_cv_in_window);
  put(sum_amount_last_10m);
  put(tx_1h);
  put(static_cast<double>(umcc.size()));
  put(static_cast<double>(udev.size()));
  put(static_cast<double>(uch.size()));
  put(static_cast<double>(utz.size()));
  put(static_cast<double>(ubl.size()));
  put(static_cast<double>(mcc_switch_cnt));
  put(static_cast<double>(dev_switch_cnt));
  put(static_cast<double>(ch_switch_cnt));
  put(night_share_24h);
  put(distinct_hours_24h);
  put(mean_last_3);
  put(amount_ratio_to_min_amount_24h);
  put(static_cast<double>(compromised_hist));
  put(secs_since_compromised);
  put(static_cast<double>(web_rdp_cnt_24h));
  put(static_cast<double>(voip_cnt_24h));
  put(channel_rel_freq);
  put(tz_rel_freq);
  put(bl_rel_freq);
  put(event_type_nm_share_suffix);
  put(mcc_streak);
  put(tx_5m);
  put(mean_gap_last_5);
  put(suffix_span_log1p);
  put(tx_per_span_hour);
  put(mcc_std_5d);
  put(weekend_share_7d);
  put(descr_div_24h);
  put(is_new_timezone);
  put(is_timezone_change);
  put(cat_from_utf8(cur_currency));
  put(cat_from_utf8(cur_pos));
  put(cat_from_utf8(cur_accept));
  put(battery_level_feat);
  put(battery_very_low);
  put(cat_from_utf8(cur_screen));
  put(devtools_flag);
  put(lang_mismatch);
  put(bat_low_x_night);
  put(amount_diff_prev);
  put(amount_ratio_prev);
  put(amount_change_sign);
  put(static_cast<double>(amount_inc_streak));
  put(static_cast<double>(amount_dec_streak));
  put(is_new_sess_id);
  put(static_cast<double>(session_switch_cnt));
  put(seconds_since_sess_start);
  if (fi != kNumFeatures) {
    throw std::runtime_error("feature vector size mismatch");
  }

  return out;
}

static arrow::Status load_labels(const std::string& path,
                                 std::unordered_map<int64_t, int>* out) {
  arrow::MemoryPool* pool = arrow::default_memory_pool();
  ARROW_ASSIGN_OR_RAISE(auto infile, arrow::io::ReadableFile::Open(path));
  ARROW_ASSIGN_OR_RAISE(auto reader, parquet::arrow::OpenFile(infile, pool));
  std::shared_ptr<arrow::Table> table;
  ARROW_RETURN_NOT_OK(reader->ReadTable(&table));
  auto id_col = table->GetColumnByName("event_id");
  auto tg_col = table->GetColumnByName("target");
  if (!id_col || !tg_col) return arrow::Status::Invalid("train_labels missing columns");
  for (int c = 0; c < id_col->num_chunks(); ++c) {
    auto id_arr = id_col->chunk(c);
    auto tg_arr = tg_col->chunk(c);
    if (id_arr->length() != tg_arr->length()) continue;
    for (int64_t i = 0; i < id_arr->length(); ++i) {
      auto e = col_get_int64(*id_arr, i);
      auto tg = col_get_int64(*tg_arr, i);
      if (e.has_value() && tg.has_value()) (*out)[*e] = static_cast<int>(*tg);
    }
  }
  return arrow::Status::OK();
}

class DatasetWriter {
 public:
  explicit DatasetWriter(const std::string& out_path) : out_path_(out_path) {
    std::vector<std::shared_ptr<arrow::Field>> fields;
    for (int i = 0; i < kNumFeatures; ++i) {
      fields.push_back(arrow::field(kFeatureNames[i], arrow::float64()));
    }
    fields.push_back(arrow::field("customer_id", arrow::utf8()));
    fields.push_back(arrow::field("event_id", arrow::int64()));
    fields.push_back(arrow::field("target", arrow::int32()));
    fields.push_back(arrow::field("sample_weight", arrow::float64()));
    fields.push_back(arrow::field("event_dttm", arrow::utf8()));
    schema_ = arrow::schema(fields);
    reset_builders();
  }

  arrow::Status append(const FeatureRow& r) {
    for (int j = 0; j < kNumFeatures; ++j) {
      ARROW_RETURN_NOT_OK(feat_builders_[j]->Append(r.f[j]));
    }
    ARROW_RETURN_NOT_OK(cid_b_->Append(r.customer_id));
    ARROW_RETURN_NOT_OK(id_b_->Append(r.event_id));
    ARROW_RETURN_NOT_OK(tg_b_->Append(r.target));
    ARROW_RETURN_NOT_OK(w_b_->Append(r.sample_weight));
    ARROW_RETURN_NOT_OK(dt_b_->Append(r.event_dttm_raw));
    ++buf_size_;
    if (buf_size_ >= kOutBatch) return flush();
    return arrow::Status::OK();
  }

  arrow::Status flush() {
    if (buf_size_ == 0) return arrow::Status::OK();
    ARROW_RETURN_NOT_OK(ensure_open());
    std::vector<std::shared_ptr<arrow::Array>> arrs;
    for (int j = 0; j < kNumFeatures; ++j) {
      std::shared_ptr<arrow::Array> a;
      ARROW_RETURN_NOT_OK(feat_builders_[j]->Finish(&a));
      arrs.push_back(a);
    }
    std::shared_ptr<arrow::Array> a_cid, a_id, a_tg, a_w, a_dt;
    ARROW_RETURN_NOT_OK(cid_b_->Finish(&a_cid));
    ARROW_RETURN_NOT_OK(id_b_->Finish(&a_id));
    ARROW_RETURN_NOT_OK(tg_b_->Finish(&a_tg));
    ARROW_RETURN_NOT_OK(w_b_->Finish(&a_w));
    ARROW_RETURN_NOT_OK(dt_b_->Finish(&a_dt));
    arrs.push_back(a_cid);
    arrs.push_back(a_id);
    arrs.push_back(a_tg);
    arrs.push_back(a_w);
    arrs.push_back(a_dt);
    auto table = arrow::Table::Make(schema_, arrs);
    ARROW_RETURN_NOT_OK(writer_->WriteTable(*table, table->num_rows()));
    buf_size_ = 0;
    reset_builders();
    rows_written_ += table->num_rows();
    log_msg("written row group rows=" + std::to_string(table->num_rows()) +
            " total_rows=" + std::to_string(rows_written_));
    return arrow::Status::OK();
  }

  arrow::Status close() {
    ARROW_RETURN_NOT_OK(flush());
    if (writer_) {
      ARROW_RETURN_NOT_OK(writer_->Close());
      writer_.reset();
    }
    if (outfile_) {
      ARROW_RETURN_NOT_OK(outfile_->Close());
      outfile_.reset();
    }
    log_msg("final dataset written: " + out_path_ + " rows=" + std::to_string(rows_written_));
    return arrow::Status::OK();
  }

 private:
  arrow::Status ensure_open() {
    if (writer_) return arrow::Status::OK();
    ARROW_ASSIGN_OR_RAISE(outfile_, arrow::io::FileOutputStream::Open(out_path_));
    parquet::WriterProperties::Builder pb;
    std::shared_ptr<parquet::WriterProperties> props = pb.build();
    ARROW_ASSIGN_OR_RAISE(
        writer_, parquet::arrow::FileWriter::Open(*schema_, arrow::default_memory_pool(), outfile_, props));
    log_msg("opened output parquet: " + out_path_);
    return arrow::Status::OK();
  }

  void reset_builders() {
    feat_builders_.clear();
    for (int i = 0; i < kNumFeatures; ++i) {
      feat_builders_.push_back(std::make_shared<arrow::DoubleBuilder>());
    }
    cid_b_ = std::make_shared<arrow::StringBuilder>();
    id_b_ = std::make_shared<arrow::Int64Builder>();
    tg_b_ = std::make_shared<arrow::Int32Builder>();
    w_b_ = std::make_shared<arrow::DoubleBuilder>();
    dt_b_ = std::make_shared<arrow::StringBuilder>();
  }

  std::string out_path_;
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<std::shared_ptr<arrow::DoubleBuilder>> feat_builders_;
  std::shared_ptr<arrow::StringBuilder> cid_b_;
  std::shared_ptr<arrow::Int64Builder> id_b_;
  std::shared_ptr<arrow::Int32Builder> tg_b_;
  std::shared_ptr<arrow::DoubleBuilder> w_b_;
  std::shared_ptr<arrow::StringBuilder> dt_b_;
  std::shared_ptr<arrow::io::FileOutputStream> outfile_;
  std::unique_ptr<parquet::arrow::FileWriter> writer_;
  int buf_size_ = 0;
  int64_t rows_written_ = 0;
};

static arrow::Result<WindowTargetSampler> build_sampler_from_pretest(const std::string& pretest_path) {
  WindowTargetSampler sampler;
  std::ifstream check(pretest_path);
  if (!check.good()) {
    log_msg("pretest not found, fallback sampler will be used: " + pretest_path);
    sampler.values = {1};
    sampler.cdf = {1.0};
    return sampler;
  }

  arrow::MemoryPool* pool = arrow::default_memory_pool();
  ARROW_ASSIGN_OR_RAISE(auto infile, arrow::io::ReadableFile::Open(pretest_path));
  ARROW_ASSIGN_OR_RAISE(auto reader, parquet::arrow::OpenFile(infile, pool));
  ARROW_ASSIGN_OR_RAISE(auto rb, reader->GetRecordBatchReader());

  std::unordered_map<std::string, int64_t> user_counts;
  while (true) {
    ARROW_ASSIGN_OR_RAISE(auto batch_ptr, rb->Next());
    if (!batch_ptr) break;
    const arrow::RecordBatch& batch = *batch_ptr;
    int cidx = col_index(*batch.schema(), "customer_id");
    if (cidx < 0) {
      return arrow::Status::Invalid("pretest parquet has no customer_id: ", pretest_path);
    }
    const auto& col_c = *batch.column(cidx);
    const int64_t n = batch.num_rows();
    for (int64_t i = 0; i < n; ++i) {
      std::string ck = col_get_str(col_c, i);
      if (ck.empty()) continue;
      user_counts[ck] += 1;
    }
  }

  if (user_counts.empty()) {
    return arrow::Status::Invalid("no non-empty customer_id values in pretest: ", pretest_path);
  }

  std::unordered_map<int, int64_t> hist;
  for (const auto& kv : user_counts) {
    int capped = static_cast<int>(std::min<int64_t>(kv.second, static_cast<int64_t>(kWindowCap)));
    if (capped <= 0) continue;
    hist[capped] += 1;
  }
  if (hist.empty()) {
    return arrow::Status::Invalid("failed to build transaction-count histogram from pretest: ", pretest_path);
  }

  std::vector<std::pair<int, int64_t>> bins(hist.begin(), hist.end());
  std::sort(bins.begin(), bins.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

  int64_t total = 0;
  for (const auto& b : bins) total += b.second;
  if (total <= 0) {
    return arrow::Status::Invalid("invalid histogram total for pretest sampler: ", pretest_path);
  }

  sampler.values.reserve(bins.size());
  sampler.cdf.reserve(bins.size());
  double acc = 0.0;
  for (const auto& b : bins) {
    acc += static_cast<double>(b.second) / static_cast<double>(total);
    sampler.values.push_back(b.first);
    sampler.cdf.push_back(acc);
  }
  sampler.cdf.back() = 1.0;
  log_msg("pretest sampler ready: users=" + std::to_string(user_counts.size()) +
          " bins=" + std::to_string(sampler.values.size()) + " cap=" + std::to_string(kWindowCap));
  return sampler;
}

static arrow::Status process_file(const std::string& path, bool is_train,
                                  std::unordered_map<std::string, UserWindow>* win_map,
                                  const std::unordered_map<int64_t, int>& labels, const WindowTargetSampler& sampler,
                                  DatasetWriter* wr,
                                  std::unordered_set<std::string>* cohort_customer_ids = nullptr) {
  arrow::MemoryPool* pool = arrow::default_memory_pool();
  ARROW_ASSIGN_OR_RAISE(auto infile, arrow::io::ReadableFile::Open(path));
  ARROW_ASSIGN_OR_RAISE(auto reader, parquet::arrow::OpenFile(infile, pool));
  int64_t total_rows = 0;
  if (auto* pqr = reader->parquet_reader()) {
    if (auto meta = pqr->metadata()) total_rows = meta->num_rows();
  }
  const std::string tag = path_basename(path) + (is_train ? " [train]" : " [pretrain]");
  log_msg("reading " + path + " rows_meta=" + std::to_string(total_rows) + (is_train ? " (emit dataset)" : " (window only)"));
  ARROW_ASSIGN_OR_RAISE(auto rb, reader->GetRecordBatchReader());
  int64_t rows_seen = 0;
  while (true) {
    // Parquet не реализует ReadNext() без аргументов (RecordBatchWithMetadata) — используем Next().
    ARROW_ASSIGN_OR_RAISE(auto batch_ptr, rb->Next());
    if (!batch_ptr) break;
    const arrow::RecordBatch& batch = *batch_ptr;
    const auto& sch = *batch.schema();
    int64_t n = batch.num_rows();
    int cidx = col_index(sch, "customer_id");
    if (cidx < 0) {
      rows_seen += n;
      render_row_progress(rows_seen, total_rows, tag);
      continue;
    }
    auto& col_c = *batch.column(cidx);
    for (int64_t i = 0; i < n; ++i) {
      std::string ck = col_get_str(col_c, i);
      if (ck.empty()) continue;
      if (cohort_customer_ids != nullptr) cohort_customer_ids->insert(ck);
      UserWindow& w = (*win_map)[ck];
      if (is_train) {
        int64_t eid_row = 0;
        int eidx = col_index(sch, "event_id");
        if (eidx >= 0) {
          if (auto p = col_get_int64(*batch.column(eidx), i)) eid_row = *p;
        }
        int target_len = sampler.sample(eid_row, ck);
        FeatureRow fr = compute_features(w, target_len, batch, i, sch, labels);
        fr.customer_id = ck;
        ARROW_RETURN_NOT_OK(wr->append(fr));
      }
      w.push(row_to_txn(batch, i, sch));
    }
    rows_seen += n;
    render_row_progress(rows_seen, total_rows, tag);
  }
  std::cerr << "\n";
  log_msg("finished " + path + " batches_rows_seen=" + std::to_string(rows_seen));
  return arrow::Status::OK();
}

int main(int argc, char** argv) {
  std::string root = ".";
  if (argc >= 2) root = argv[1];
  std::string data_train = root + "/data/train/";
  std::string data_test = root + "/data/test/";
  std::string labels_path = root + "/data/train_labels.parquet";
  std::string pretest_path = data_test + "pretest.parquet";
  std::string out_dir = root + "/output/";
  std::string out_path = out_dir + "full_dataset.parquet";

  log_msg("repo_root=" + root + " out_path=" + out_path);

  std::error_code fs_ec;
  std::filesystem::create_directories(out_dir, fs_ec);
  if (fs_ec) {
    std::cerr << "[build_dataset] Cannot create directory: " << out_dir << " — " << fs_ec.message() << "\n";
    return 1;
  }

  std::unordered_map<int64_t, int> labels_orig;
  std::vector<std::string> pre = {data_train + "pretrain_part_1.parquet", data_train + "pretrain_part_2.parquet",
                                  data_train + "pretrain_part_3.parquet"};
  std::vector<std::string> tr = {data_train + "train_part_1.parquet", data_train + "train_part_2.parquet",
                                 data_train + "train_part_3.parquet"};

  {
    arrow::Status vst = validate_all_input_parquets(labels_path, pre, tr);
    if (!vst.ok()) {
      std::cerr << "[build_dataset] Проверка схем parquet: ОШИБКА — " << vst.ToString() << "\n";
      return 1;
    }
    log_msg("проверка схем parquet (labels + существующие pretrain/train): OK");
  }

  if (!load_labels(labels_path, &labels_orig).ok()) {
    std::cerr << "[build_dataset] Failed to read labels: " << labels_path << "\n";
    return 1;
  }
  log_msg("labels event_ids=" + std::to_string(labels_orig.size()) + " path=" + labels_path);

  auto sampler_res = build_sampler_from_pretest(pretest_path);
  if (!sampler_res.ok()) {
    std::cerr << "[build_dataset] Failed to build pretest sampler: " << sampler_res.status().ToString() << "\n";
    return 1;
  }
  WindowTargetSampler sampler = std::move(sampler_res.ValueOrDie());

  DatasetWriter wr(out_path);
  std::unordered_map<std::string, UserWindow> windows;

  log_msg("phase: for each k=1..3: pretrain_part_k (window only) → train_part_k (dataset) → evict cohort");
  for (int part = 0; part < 3; ++part) {
    std::unordered_set<std::string> cohort_ids;
    cohort_ids.reserve(65536);
    const std::string& pre_path = pre[part];
    const std::string& tr_path = tr[part];

    {
      std::ifstream fp(pre_path);
      if (!fp.good()) {
        log_msg("skip missing pretrain: " + pre_path);
      } else {
        auto st = process_file(pre_path, false, &windows, labels_orig, sampler, &wr, &cohort_ids);
        if (!st.ok()) {
          std::cerr << "[build_dataset] " << st.ToString() << "\n";
          return 1;
        }
      }
    }
    {
      std::ifstream ft(tr_path);
      if (!ft.good()) {
        log_msg("skip missing train: " + tr_path);
      } else {
        auto st = process_file(tr_path, true, &windows, labels_orig, sampler, &wr, &cohort_ids);
        if (!st.ok()) {
          std::cerr << "[build_dataset] " << st.ToString() << "\n";
          return 1;
        }
      }
    }
    size_t erased = 0;
    for (const auto& id : cohort_ids) {
      if (windows.erase(id) > 0) ++erased;
    }
    log_msg("part " + std::to_string(part + 1) + " cohort_unique_customers=" + std::to_string(cohort_ids.size()) +
            " erased_from_map=" + std::to_string(erased) + " windows_remaining=" + std::to_string(windows.size()));
  }
  log_msg("final flush/close output parquet …");
  if (!wr.close().ok()) return 1;
  log_msg("Done. unique customers in window map (expect 0): " + std::to_string(windows.size()));
  return 0;
}
