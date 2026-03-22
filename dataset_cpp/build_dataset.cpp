// Сборка обучающего датасета: Arrow + Parquet + OpenSSL (MD5).
// Train: каждая строка с непустым customer_id → строка в датасет → update окна.
// target = 1 если event_id в train_labels.parquet, иначе 0.
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

constexpr int kWindowMax = 150;
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

/** Текстовая шкала прогресса по строкам одного parquet (stderr, одна строка с \\r). */
static void render_row_progress(int64_t done, int64_t total, const std::string& file_tag) {
  constexpr int kBarW = 40;
  std::cerr << "\r[build_dataset] [";
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
  std::cerr << "\033[K" << std::flush;
}

static const char* kFeatureNames[] = {
    "operation_amt",          "amount_to_median",       "amount_zscore",
    "is_amount_high",         "transactions_last_1h",   "transactions_last_24h",
    "sum_amount_last_1h",     "max_amount_last_24h",    "device_freq",
    "device_count",           "time_since_last_device", "mcc_freq",
    "mcc_count",              "time_since_last_mcc",    "channel_freq",
    "channel_count",          "time_since_last_channel","timezone_freq",
    "browser_language_freq",
    "is_compromised_device",  "web_rdp_connection",     "phone_voip_call_state",
    "hour",                   "day_of_week",            "is_night_transaction",
    "is_weekend",             "transactions_last_10m",  "sum_amount_last_24h",
    "time_since_prev_transaction", "transactions_in_session",
    "timezone_missing",       "tr_amount",              "event_descr",
    "mcc_code",               "event_descr_freq",       "event_type_nm_freq",
    "log_tr_amount",          "transactions_last_10m_norm", "transactions_last_1h_norm",
    "transactions_last_24h_norm", "sum_amount_last_1h_norm", "sum_amount_last_24h_norm",
    "transactions_last_10m_to_1h", "transactions_last_1h_to_24h", "sum_1h_to_24h",
    "time_since_2nd_prev_transaction", "mean_time_between_tx", "std_time_between_tx",
    "amount_to_last_amount",  "amount_to_max_24h",      "time_since_prev_to_mean_gap",
    "device_freq_alt",        "mcc_freq_alt"};
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
    while (static_cast<int>(dq.size()) > kWindowMax) dq.pop_front();
  }
  int count() const { return static_cast<int>(dq.size()); }
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

static std::string col_get_str(const arrow::Array& col, int64_t row) {
  if (col.IsNull(row)) return {};
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

static std::optional<int64_t> col_get_int64(const arrow::Array& col, int64_t row) {
  if (col.IsNull(row)) return std::nullopt;
  if (col.type_id() == arrow::Type::INT64) {
    return static_cast<const arrow::Int64Array&>(col).Value(row);
  }
  if (col.type_id() == arrow::Type::INT32) {
    return static_cast<int64_t>(static_cast<const arrow::Int32Array&>(col).Value(row));
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

static Txn row_to_txn(const arrow::RecordBatch& batch, int64_t i, const arrow::Schema& sch) {
  auto get_s = [&](const char* n) -> std::string {
    int idx = col_index(sch, n);
    if (idx < 0) return {};
    return col_get_str(*batch.column(idx), i);
  };
  Txn t;
  std::string amt_s = get_s("operaton_amt");
  if (!amt_s.empty()) t.amount = parse_double_any(amt_s);
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
  t.event_descr = get_s("event_descr");
  if (t.event_descr.empty()) t.event_descr = get_s("event_desc");
  return t;
}

struct FeatureRow {
  std::array<double, kNumFeatures> f{};
  int64_t event_id = 0;
  int32_t target = 0;
  double sample_weight = 1.0;
  std::string event_dttm_raw;
};

static double nan_val() { return std::numeric_limits<double>::quiet_NaN(); }

static FeatureRow compute_features(const UserWindow& w, const arrow::RecordBatch& batch, int64_t i,
                                 const arrow::Schema& sch,
                                 const std::unordered_map<int64_t, int>& labels_orig) {
  FeatureRow out;
  std::vector<double> amounts;
  amounts.reserve(w.dq.size());
  for (const auto& t : w.dq) {
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
    std::nth_element(sorted.begin(), sorted.begin() + idx95, sorted.end());
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

  auto get_row_s = [&](const char* n) -> std::string {
    int idx = col_index(sch, n);
    if (idx < 0) return {};
    return col_get_str(*batch.column(idx), i);
  };

  std::string amt_cur_s = get_row_s("operaton_amt");
  double amount = nan_val();
  if (auto p = parse_double_any(amt_cur_s)) amount = *p;

  std::string dttm_s = get_row_s("event_dttm");
  out.event_dttm_raw = dttm_s;
  Txn cur_tmp;
  if (!dttm_s.empty()) fill_txn_time(cur_tmp, dttm_s);
  std::optional<std::chrono::system_clock::time_point> dttm = cur_tmp.dttm;
  int hour_val = cur_tmp.hour_val;
  int dow_val = cur_tmp.dow_py;

  using clock = std::chrono::system_clock;
  struct TA {
    clock::time_point t;
    double a;
  };
  std::vector<TA> recent;
  if (dttm.has_value()) {
    auto t0 = *dttm - std::chrono::hours(24);
    for (const auto& t : w.dq) {
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
  std::optional<clock::time_point> second_last_ev;
  for (auto it = w.dq.rbegin(); it != w.dq.rend(); ++it) {
    if (it->dttm.has_value()) {
      if (!last_ev.has_value()) {
        last_ev = it->dttm;
      } else {
        second_last_ev = it->dttm;
        break;
      }
    }
  }

  std::unordered_map<std::string, int> dev_cnt, mcc_cnt, ch_cnt, tz_cnt, bl_cnt, descr_cnt, type_cnt;
  std::unordered_map<std::string, clock::time_point> last_seen_dev, last_seen_mcc, last_seen_ch;
  std::unordered_map<std::string, int> sess_cnt;
  std::vector<double> gap_seconds;
  std::optional<clock::time_point> prev_t_for_gap;
  double last_amount_in_window = nan_val();
  for (const auto& t : w.dq) {
    std::string dkey = t.os_type + "\x1f" + t.dev_ver;
    std::string ckey = t.ch_type + "\x1f" + t.ch_sub;
    if (!empty_field(t.os_type) || !empty_field(t.dev_ver)) dev_cnt[dkey]++;
    if (!empty_field(t.mcc)) mcc_cnt[t.mcc]++;
    if (!empty_field(t.ch_type) || !empty_field(t.ch_sub)) ch_cnt[ckey]++;
    if (!empty_field(t.tz)) tz_cnt[t.tz]++;
    if (!empty_field(t.browser_language)) bl_cnt[t.browser_language]++;
    if (!empty_field(t.event_descr)) descr_cnt[t.event_descr]++;
    if (!empty_field(t.event_type_nm)) type_cnt[t.event_type_nm]++;

    if (t.dttm.has_value()) {
      if (!empty_field(t.os_type) || !empty_field(t.dev_ver)) last_seen_dev[dkey] = *t.dttm;
      if (!empty_field(t.mcc)) last_seen_mcc[t.mcc] = *t.dttm;
      if (!empty_field(t.ch_type) || !empty_field(t.ch_sub)) last_seen_ch[ckey] = *t.dttm;
      if (prev_t_for_gap.has_value()) {
        double ds = static_cast<double>(std::chrono::duration_cast<std::chrono::seconds>(*t.dttm - *prev_t_for_gap).count());
        if (std::isfinite(ds) && ds >= 0.0) gap_seconds.push_back(ds);
      }
      prev_t_for_gap = t.dttm;
    }
    if (t.amount.has_value() && std::isfinite(*t.amount)) last_amount_in_window = *t.amount;
    if (!t.session_id.empty()) sess_cnt[t.session_id]++;
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
  std::string etype_c = get_row_s("event_type_nm");
  std::string descr = get_row_s("event_descr");
  if (descr.empty()) descr = get_row_s("event_desc");

  std::string dkey_c = os_c + "\x1f" + dev_c;
  std::string ckey_c = ch_t + "\x1f" + ch_s;
  double tx_total = static_cast<double>(w.count());
  auto safe_ratio = [&](double num, double den) -> double { return num / (std::abs(den) > kEps ? den : kEps); };
  auto get_count = [&](const std::unordered_map<std::string, int>& mp, const std::string& key, bool key_ok) -> double {
    if (!key_ok) return 0.0;
    auto it = mp.find(key);
    return (it == mp.end()) ? 0.0 : static_cast<double>(it->second);
  };

  double device_count = get_count(dev_cnt, dkey_c, (!empty_field(os_c) || !empty_field(dev_c)));
  double mcc_count = get_count(mcc_cnt, mcc_c, !empty_field(mcc_c));
  double channel_count = get_count(ch_cnt, ckey_c, (!empty_field(ch_t) || !empty_field(ch_s)));
  double timezone_count = get_count(tz_cnt, tz_c, !empty_field(tz_c));
  double bl_count = get_count(bl_cnt, bl_c, !empty_field(bl_c));

  double device_freq = (tx_total > 0.0) ? safe_ratio(device_count, tx_total) : 0.0;
  double mcc_freq = (tx_total > 0.0) ? safe_ratio(mcc_count, tx_total) : 0.0;
  double channel_freq = (tx_total > 0.0) ? safe_ratio(channel_count, tx_total) : 0.0;
  double timezone_freq = (tx_total > 0.0) ? safe_ratio(timezone_count, tx_total) : 0.0;
  double browser_language_freq = (tx_total > 0.0) ? safe_ratio(bl_count, tx_total) : 0.0;

  auto since_last = [&](const std::unordered_map<std::string, clock::time_point>& last_map, const std::string& key, bool key_ok) -> double {
    if (!dttm.has_value() || !key_ok) return -1.0;
    auto it = last_map.find(key);
    if (it == last_map.end()) return -1.0;
    return static_cast<double>(std::chrono::duration_cast<std::chrono::seconds>(*dttm - it->second).count());
  };
  double time_since_last_device = since_last(last_seen_dev, dkey_c, (!empty_field(os_c) || !empty_field(dev_c)));
  double time_since_last_mcc = since_last(last_seen_mcc, mcc_c, !empty_field(mcc_c));
  double time_since_last_channel = since_last(last_seen_ch, ckey_c, (!empty_field(ch_t) || !empty_field(ch_s)));

  int n_sess = 0;
  if (!sid_c.empty()) {
    auto it = sess_cnt.find(sid_c);
    if (it != sess_cnt.end()) n_sess = it->second;
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

  double hour_f = dttm.has_value() ? static_cast<double>(hour_val) : nan_val();
  double dow_f = dttm.has_value() ? static_cast<double>(dow_val) : nan_val();
  double is_night = (hour_val >= 22 || hour_val < 6) ? 1.0 : 0.0;
  double is_weekend = (dow_val >= 5) ? 1.0 : 0.0;

  double tr_am = static_cast<double>(std::min(w.count(), kWindowMax));
  double tx_10m = static_cast<double>(count_since(std::chrono::minutes(10)));
  double tx_1h = static_cast<double>(count_since(std::chrono::hours(1)));
  double tx_24h = static_cast<double>(count_since(std::chrono::hours(24)));
  double sum_1h = sum_since(std::chrono::hours(1));
  double sum_24h = sum_since(std::chrono::hours(24));

  double descr_freq = 0.0;
  if (!empty_field(descr) && tx_total > 0.0) {
    auto it = descr_cnt.find(descr);
    descr_freq = safe_ratio((it == descr_cnt.end() ? 0.0 : static_cast<double>(it->second)), tx_total);
  }
  double etype_freq = 0.0;
  if (!empty_field(etype_c) && tx_total > 0.0) {
    auto it = type_cnt.find(etype_c);
    etype_freq = safe_ratio((it == type_cnt.end() ? 0.0 : static_cast<double>(it->second)), tx_total);
  }

  std::string compromised_s = trim_copy(get_row_s("compromised"));
  double compromised_flag = 0.0;
  if (!compromised_s.empty()) {
    if (auto pc = parse_double_any(compromised_s)) compromised_flag = (*pc == 1.0) ? 1.0 : 0.0;
  }

  double log_tr_amount = std::log1p(std::max(0.0, tr_am));
  double log_tr_denom = (log_tr_amount > kEps) ? log_tr_amount : kEps;
  double tx_10m_norm = safe_ratio(tx_10m, log_tr_denom);
  double tx_1h_norm = safe_ratio(tx_1h, log_tr_denom);
  double tx_24h_norm = safe_ratio(tx_24h, log_tr_denom);
  double sum_1h_norm = safe_ratio(sum_1h, log_tr_denom);
  double sum_24h_norm = safe_ratio(sum_24h, log_tr_denom);

  double tx_10m_to_1h = tx_10m / (tx_1h + 1.0);
  double tx_1h_to_24h = tx_1h / (tx_24h + 1.0);
  double sum_1h_to_24h = sum_1h / (sum_24h + 1.0);

  double time_since_2nd_prev = -1.0;
  if (dttm.has_value() && second_last_ev.has_value()) {
    time_since_2nd_prev =
        static_cast<double>(std::chrono::duration_cast<std::chrono::seconds>(*dttm - *second_last_ev).count());
  }

  double mean_gap = -1.0;
  double std_gap = -1.0;
  if (!gap_seconds.empty()) {
    double s = 0.0;
    for (double g : gap_seconds) s += g;
    mean_gap = s / static_cast<double>(gap_seconds.size());
    if (gap_seconds.size() >= 2) {
      double sq = 0.0;
      for (double g : gap_seconds) {
        double d = g - mean_gap;
        sq += d * d;
      }
      std_gap = std::sqrt(std::max(0.0, sq / static_cast<double>(gap_seconds.size() - 1)));
    } else {
      std_gap = 0.0;
    }
  }

  double amount_to_last_amount = nan_val();
  if (std::isfinite(amount) && std::isfinite(last_amount_in_window)) {
    amount_to_last_amount = safe_ratio(amount, last_amount_in_window + kEps);
  }
  double amount_to_max_24h = nan_val();
  if (std::isfinite(amount) && std::isfinite(max_amount_feat)) {
    amount_to_max_24h = safe_ratio(amount, max_amount_feat + kEps);
  }

  double time_since_prev_to_mean_gap = -1.0;
  if (time_since >= 0.0 && mean_gap >= 0.0) {
    time_since_prev_to_mean_gap = safe_ratio(time_since, mean_gap + kEps);
  }

  double device_freq_alt = safe_ratio(device_count, std::isfinite(amount) ? (amount + kEps) : kEps);
  double mcc_freq_alt = safe_ratio(mcc_count, std::isfinite(amount) ? (amount + kEps) : kEps);

  int fi = 0;
  auto put = [&](double v) { out.f[fi++] = v; };
  put(amount);
  put(amount_to_median);
  put(amount_zscore);
  put(is_amount_high);
  put(tx_1h);
  put(tx_24h);
  put(sum_1h);
  put(max_amount_feat);
  put(device_freq);
  put(device_count);
  put(time_since_last_device);
  put(mcc_freq);
  put(mcc_count);
  put(time_since_last_mcc);
  put(channel_freq);
  put(channel_count);
  put(time_since_last_channel);
  put(timezone_freq);
  put(browser_language_freq);
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
  put(static_cast<double>(n_sess + 1));
  put(empty_field(tz_c) ? 1.0 : 0.0);
  put(tr_am);
  put(cat_from_utf8(descr));
  put(cat_from_utf8(mcc_c));
  put(descr_freq);
  put(etype_freq);
  put(log_tr_amount);
  put(tx_10m_norm);
  put(tx_1h_norm);
  put(tx_24h_norm);
  put(sum_1h_norm);
  put(sum_24h_norm);
  put(tx_10m_to_1h);
  put(tx_1h_to_24h);
  put(sum_1h_to_24h);
  put(time_since_2nd_prev);
  put(mean_gap);
  put(std_gap);
  put(amount_to_last_amount);
  put(amount_to_max_24h);
  put(time_since_prev_to_mean_gap);
  put(device_freq_alt);
  put(mcc_freq_alt);
  if (fi != kNumFeatures) {
    throw std::runtime_error("feature vector size mismatch");
  }

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
    std::shared_ptr<arrow::Array> a_id, a_tg, a_w, a_dt;
    ARROW_RETURN_NOT_OK(id_b_->Finish(&a_id));
    ARROW_RETURN_NOT_OK(tg_b_->Finish(&a_tg));
    ARROW_RETURN_NOT_OK(w_b_->Finish(&a_w));
    ARROW_RETURN_NOT_OK(dt_b_->Finish(&a_dt));
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
    id_b_ = std::make_shared<arrow::Int64Builder>();
    tg_b_ = std::make_shared<arrow::Int32Builder>();
    w_b_ = std::make_shared<arrow::DoubleBuilder>();
    dt_b_ = std::make_shared<arrow::StringBuilder>();
  }

  std::string out_path_;
  std::shared_ptr<arrow::Schema> schema_;
  std::vector<std::shared_ptr<arrow::DoubleBuilder>> feat_builders_;
  std::shared_ptr<arrow::Int64Builder> id_b_;
  std::shared_ptr<arrow::Int32Builder> tg_b_;
  std::shared_ptr<arrow::DoubleBuilder> w_b_;
  std::shared_ptr<arrow::StringBuilder> dt_b_;
  std::shared_ptr<arrow::io::FileOutputStream> outfile_;
  std::unique_ptr<parquet::arrow::FileWriter> writer_;
  int buf_size_ = 0;
  int64_t rows_written_ = 0;
};

static arrow::Status process_file(const std::string& path, bool is_train,
                                  std::unordered_map<std::string, UserWindow>* win_map,
                                  const std::unordered_map<int64_t, int>& labels, DatasetWriter* wr) {
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
      UserWindow& w = (*win_map)[ck];
      if (is_train) {
        FeatureRow fr = compute_features(w, batch, i, sch, labels);
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
  std::string labels_path = root + "/data/train_labels.parquet";
  std::string out_dir = root + "/output/";
  std::string out_path = out_dir + "full_dataset";

  log_msg("repo_root=" + root + " out_path=" + out_path);

  std::error_code fs_ec;
  std::filesystem::create_directories(out_dir, fs_ec);
  if (fs_ec) {
    std::cerr << "[build_dataset] Cannot create directory: " << out_dir << " — " << fs_ec.message() << "\n";
    return 1;
  }

  std::unordered_map<int64_t, int> labels_orig;
  if (!load_labels(labels_path, &labels_orig).ok()) {
    std::cerr << "[build_dataset] Failed to read labels: " << labels_path << "\n";
    return 1;
  }
  log_msg("labels event_ids=" + std::to_string(labels_orig.size()) + " path=" + labels_path);

  DatasetWriter wr(out_path);
  std::unordered_map<std::string, UserWindow> windows;

  std::vector<std::string> pre = {data_train + "pretrain_part_1.parquet", data_train + "pretrain_part_2.parquet",
                                  data_train + "pretrain_part_3.parquet"};
  std::vector<std::string> tr = {data_train + "train_part_1.parquet", data_train + "train_part_2.parquet",
                                 data_train + "train_part_3.parquet"};

  log_msg("phase: pretrain files (window fill, no dataset rows)");
  for (const auto& p : pre) {
    std::ifstream f(p);
    if (!f.good()) {
      log_msg("skip missing: " + p);
      continue;
    }
    auto st = process_file(p, false, &windows, labels_orig, &wr);
    if (!st.ok()) {
      std::cerr << "[build_dataset] " << st.ToString() << "\n";
      return 1;
    }
  }
  log_msg("phase: train files (dataset rows + window)");
  for (const auto& p : tr) {
    std::ifstream f(p);
    if (!f.good()) {
      log_msg("skip missing: " + p);
      continue;
    }
    auto st = process_file(p, true, &windows, labels_orig, &wr);
    if (!st.ok()) {
      std::cerr << "[build_dataset] " << st.ToString() << "\n";
      return 1;
    }
  }
  log_msg("final flush/close output parquet …");
  if (!wr.close().ok()) return 1;
  log_msg("Done. unique customers in window map: " + std::to_string(windows.size()));
  return 0;
}
