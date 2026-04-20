// Global category aggregates → output/datasets/global_aggregates/*.parquet (+ population_meta).

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

#include "progress.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "global_category_features.hpp"

namespace {

constexpr double kEps = 1e-9;
constexpr int kReservoirCap = 512;

static void log_msg(const std::string& s) { std::cerr << "[build_global_aggregates] " << s << "\n"; }

static double nan_val() { return std::numeric_limits<double>::quiet_NaN(); }

static std::string trim_copy(std::string s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.erase(s.begin());
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.pop_back();
  return s;
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

static std::optional<int64_t> col_get_int64(const arrow::Array& col, int64_t row);
static std::string col_get_str(const arrow::Array& col, int64_t row);
static std::optional<double> col_get_optional_double(const arrow::Array& col, int64_t row);

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
    case arrow::Type::BOOL:
      return static_cast<const arrow::BooleanArray&>(col).Value(row) ? "1" : "0";
    case arrow::Type::INT8:
      return std::to_string(static_cast<int>(static_cast<const arrow::Int8Array&>(col).Value(row)));
    case arrow::Type::INT16:
      return std::to_string(static_cast<int>(static_cast<const arrow::Int16Array&>(col).Value(row)));
    case arrow::Type::UINT8:
      return std::to_string(static_cast<unsigned>(static_cast<const arrow::UInt8Array&>(col).Value(row)));
    case arrow::Type::UINT16:
      return std::to_string(static_cast<unsigned>(static_cast<const arrow::UInt16Array&>(col).Value(row)));
    case arrow::Type::UINT32:
      return std::to_string(static_cast<const arrow::UInt32Array&>(col).Value(row));
    case arrow::Type::UINT64: {
      uint64_t v = static_cast<const arrow::UInt64Array&>(col).Value(row);
      return std::to_string(v);
    }
    case arrow::Type::FLOAT: {
      char buf[64];
      std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(static_cast<const arrow::FloatArray&>(col).Value(row)));
      return std::string(buf);
    }
    case arrow::Type::DOUBLE: {
      char buf[64];
      std::snprintf(buf, sizeof(buf), "%.17g", static_cast<const arrow::DoubleArray&>(col).Value(row));
      return std::string(buf);
    }
    case arrow::Type::BINARY: {
      auto v = static_cast<const arrow::BinaryArray&>(col).GetView(row);
      return std::string(reinterpret_cast<const char*>(v.data()), v.size());
    }
    case arrow::Type::LARGE_BINARY: {
      auto v = static_cast<const arrow::LargeBinaryArray&>(col).GetView(row);
      return std::string(reinterpret_cast<const char*>(v.data()), v.size());
    }
    default:
      return {};
  }
}

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
    case arrow::Type::LARGE_STRING:
      return parse_double_any(col_get_str(col, row));
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

struct Welford {
  int64_t n = 0;
  double mean = 0.0;
  double M2 = 0.0;
  void add(double x) {
    ++n;
    double d = x - mean;
    mean += d / static_cast<double>(n);
    double d2 = x - mean;
    M2 += d * d2;
  }
  double variance_sample() const {
    if (n < 2) return nan_val();
    return std::max(0.0, M2 / static_cast<double>(n - 1));
  }
  double std_sample() const {
    double v = variance_sample();
    return std::isfinite(v) ? std::sqrt(v) : nan_val();
  }
};

struct Reservoir {
  std::vector<double> buf;
  uint64_t seen = 0;
  std::mt19937_64 rng{0xC0FFEEULL};

  void add(double x) {
    ++seen;
    if (static_cast<int>(buf.size()) < kReservoirCap) {
      buf.push_back(x);
      return;
    }
    std::uniform_int_distribution<uint64_t> dist(1, seen);
    uint64_t j = dist(rng);
    if (j <= static_cast<uint64_t>(kReservoirCap)) buf[static_cast<size_t>(j - 1)] = x;
  }

  void quantiles(double* q25, double* med, double* q75, double* q90, double* q95, double* q99) const {
    *q25 = *med = *q75 = *q90 = *q95 = *q99 = nan_val();
    if (buf.empty()) return;
    std::vector<double> s = buf;
    std::sort(s.begin(), s.end());
    size_t n = s.size();
    auto at_p = [&](double p) -> double {
      if (n == 1) return s[0];
      double pos = p * static_cast<double>(n - 1);
      size_t i = static_cast<size_t>(pos);
      double f = pos - static_cast<double>(i);
      if (i + 1 < n) return s[i] * (1.0 - f) + s[i + 1] * f;
      return s[i];
    };
    *q25 = at_p(0.25);
    *med = at_p(0.50);
    *q75 = at_p(0.75);
    *q90 = at_p(0.90);
    *q95 = at_p(0.95);
    *q99 = at_p(0.99);
  }
};

struct AxisState {
  Welford clean_amt;
  Reservoir res_clean;
  int64_t cnt_all = 0;
  int64_t train_rows = 0;
  int64_t fraud_train = 0;
};

struct PopTotals {
  int64_t pretrain_rows = 0;
  int64_t train_rows = 0;
  int64_t train_fraud_labels = 0;
};

static void axis_observe_pretrain(AxisState& st, double amount, bool amount_ok) {
  ++st.cnt_all;
  if (!amount_ok || !std::isfinite(amount)) return;
  st.clean_amt.add(amount);
  st.res_clean.add(amount);
}

static void axis_observe_train(AxisState& st, double amount, bool amount_ok, bool in_labels) {
  ++st.cnt_all;
  ++st.train_rows;
  if (in_labels) ++st.fraud_train;
  if (in_labels || !amount_ok || !std::isfinite(amount)) return;
  st.clean_amt.add(amount);
  st.res_clean.add(amount);
}

struct FinalizedAxis {
  double mean = nan_val();
  double stdv = nan_val();
  double median = nan_val();
  double q25 = nan_val();
  double q75 = nan_val();
  double q90 = nan_val();
  double q95 = nan_val();
  double q99 = nan_val();
  double cnt = 0.0;
  double cnt_clean = 0.0;
  double cv = nan_val();
  double fraud_rate = nan_val();
  double fraud_count = 0.0;
  double train_total = 0.0;
  double woe = nan_val();
};

static FinalizedAxis finalize_axis(const AxisState& st, double odds_pop) {
  FinalizedAxis o;
  o.cnt = static_cast<double>(st.cnt_all);
  o.cnt_clean = static_cast<double>(st.clean_amt.n);
  o.train_total = static_cast<double>(st.train_rows);
  o.fraud_count = static_cast<double>(st.fraud_train);
  if (st.train_rows > 0) o.fraud_rate = static_cast<double>(st.fraud_train) / static_cast<double>(st.train_rows);
  else
    o.fraud_rate = nan_val();

  if (st.clean_amt.n > 0) {
    o.mean = st.clean_amt.mean;
    o.stdv = st.clean_amt.std_sample();
    st.res_clean.quantiles(&o.q25, &o.median, &o.q75, &o.q90, &o.q95, &o.q99);
    if (std::isfinite(o.mean) && std::abs(o.mean) > kEps && std::isfinite(o.stdv)) o.cv = o.stdv / o.mean;
  }

  double odds_key = (st.fraud_train + 0.5) / (std::max<int64_t>(1, st.train_rows - st.fraud_train) + 0.5);
  if (std::isfinite(odds_pop) && odds_pop > kEps) o.woe = std::log(odds_key / odds_pop);
  else
    o.woe = nan_val();
  return o;
}

static int64_t mcc_sanitize_int64(int64_t v) {
  if (v == global_category::kMccGlobalKey || v == global_category::kMccMissingKey) return global_category::kMccMissingKey;
  return v;
}

static int64_t parse_mcc_from_double(double x) {
  if (!std::isfinite(x)) return global_category::kMccMissingKey;
  double r = std::round(x);
  if (std::abs(x - r) > 1e-6 * (1.0 + std::abs(x))) return global_category::kMccMissingKey;
  if (r < static_cast<double>(std::numeric_limits<int64_t>::min()) ||
      r > static_cast<double>(std::numeric_limits<int64_t>::max()))
    return global_category::kMccMissingKey;
  return mcc_sanitize_int64(static_cast<int64_t>(r));
}

static int64_t parse_mcc_key(const std::string& raw) {
  std::string t = trim_copy(raw);
  if (t.empty()) return global_category::kMccMissingKey;
  try {
    size_t idx = 0;
    long long v = std::stoll(t, &idx, 10);
    if (idx == t.size()) return mcc_sanitize_int64(static_cast<int64_t>(v));
  } catch (...) {}
  if (auto opt = parse_double_any(t)) return parse_mcc_from_double(*opt);
  return global_category::kMccMissingKey;
}

// MCC from Arrow cell (int/string/binary/float).
static int64_t parse_mcc_from_column(const arrow::Array& col, int64_t row) {
  if (col.IsNull(row)) return global_category::kMccMissingKey;
  if (col.type_id() == arrow::Type::DICTIONARY) {
    const auto& da = static_cast<const arrow::DictionaryArray&>(col);
    if (da.IsNull(row)) return global_category::kMccMissingKey;
    int64_t di = da.GetValueIndex(row);
    if (di < 0) return global_category::kMccMissingKey;
    const auto& dict = *da.dictionary();
    if (di >= dict.length()) return global_category::kMccMissingKey;
    return parse_mcc_from_column(dict, di);
  }
  switch (col.type_id()) {
    case arrow::Type::INT64:
      return mcc_sanitize_int64(static_cast<const arrow::Int64Array&>(col).Value(row));
    case arrow::Type::INT32:
      return mcc_sanitize_int64(static_cast<int64_t>(static_cast<const arrow::Int32Array&>(col).Value(row)));
    case arrow::Type::INT16:
      return mcc_sanitize_int64(static_cast<int64_t>(static_cast<const arrow::Int16Array&>(col).Value(row)));
    case arrow::Type::INT8:
      return mcc_sanitize_int64(static_cast<int64_t>(static_cast<const arrow::Int8Array&>(col).Value(row)));
    case arrow::Type::UINT8:
      return mcc_sanitize_int64(static_cast<int64_t>(static_cast<const arrow::UInt8Array&>(col).Value(row)));
    case arrow::Type::UINT16:
      return mcc_sanitize_int64(static_cast<int64_t>(static_cast<const arrow::UInt16Array&>(col).Value(row)));
    case arrow::Type::UINT32: {
      uint32_t u = static_cast<const arrow::UInt32Array&>(col).Value(row);
      if (u > static_cast<uint32_t>(std::numeric_limits<int64_t>::max())) return global_category::kMccMissingKey;
      return mcc_sanitize_int64(static_cast<int64_t>(u));
    }
    case arrow::Type::UINT64: {
      uint64_t u = static_cast<const arrow::UInt64Array&>(col).Value(row);
      if (u > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) return global_category::kMccMissingKey;
      return mcc_sanitize_int64(static_cast<int64_t>(u));
    }
    case arrow::Type::FLOAT:
      return parse_mcc_from_double(static_cast<double>(static_cast<const arrow::FloatArray&>(col).Value(row)));
    case arrow::Type::DOUBLE:
      return parse_mcc_from_double(static_cast<const arrow::DoubleArray&>(col).Value(row));
    case arrow::Type::DECIMAL128: {
      const auto& a = static_cast<const arrow::Decimal128Array&>(col);
      int32_t scale = static_cast<const arrow::Decimal128Type&>(*a.type()).scale();
      arrow::Decimal128 w(a.GetValue(row));
      return parse_mcc_key(w.ToString(scale));
    }
    default:
      return parse_mcc_key(col_get_str(col, row));
  }
}

static std::string channel_map_key(const std::string& type_raw, const std::string& sub_raw) {
  std::string a = trim_copy(type_raw);
  std::string b = trim_copy(sub_raw);
  if (a.empty() && b.empty()) return "__MISSING__";
  return a + "\x1f" + b;
}

static std::string tz_curr_key(const std::string& tz_raw, const std::string& cur_raw) {
  std::string t = trim_copy(tz_raw);
  std::string c = trim_copy(cur_raw);
  if (t.empty()) t = "__MISSING_TZ__";
  if (c.empty()) c = "__MISSING_CCY__";
  return t + "\x1f" + c;
}

static std::string event_curr_key(double et_num, bool et_ok, const std::string& cur_raw) {
  std::string c = trim_copy(cur_raw);
  if (c.empty()) c = "__MISSING_CCY__";
  if (!et_ok || !std::isfinite(et_num)) return std::string("__MISSING_ET__\x1f") + c;
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.17g", et_num);
  return std::string(buf) + "\x1f" + c;
}

static arrow::Status write_population_meta(const std::string& path, const PopTotals& p) {
  arrow::Int64Builder b0, b1, b2;
  ARROW_RETURN_NOT_OK(b0.Append(p.pretrain_rows));
  ARROW_RETURN_NOT_OK(b1.Append(p.train_rows));
  ARROW_RETURN_NOT_OK(b2.Append(p.train_fraud_labels));
  std::shared_ptr<arrow::Array> a0, a1, a2;
  ARROW_RETURN_NOT_OK(b0.Finish(&a0));
  ARROW_RETURN_NOT_OK(b1.Finish(&a1));
  ARROW_RETURN_NOT_OK(b2.Finish(&a2));
  auto schema = arrow::schema({
      arrow::field("total_pretrain_rows", arrow::int64()),
      arrow::field("total_train_rows", arrow::int64()),
      arrow::field("total_train_rows_with_label", arrow::int64()),
  });
  auto table = arrow::Table::Make(schema, {a0, a1, a2});
  ARROW_ASSIGN_OR_RAISE(auto out, arrow::io::FileOutputStream::Open(path));
  parquet::WriterProperties::Builder pb;
  ARROW_ASSIGN_OR_RAISE(auto writer, parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), out, pb.build()));
  ARROW_RETURN_NOT_OK(writer->WriteTable(*table, 1));
  ARROW_RETURN_NOT_OK(writer->Close());
  ARROW_RETURN_NOT_OK(out->Close());
  return arrow::Status::OK();
}

static arrow::Status write_mcc_parquet(const std::string& path, const std::unordered_map<int64_t, AxisState>& m,
                                       double odds_pop) {
  std::vector<int64_t> keys;
  keys.reserve(m.size());
  for (const auto& kv : m) keys.push_back(kv.first);
  std::sort(keys.begin(), keys.end());

  arrow::Int64Builder kb;
  std::vector<arrow::DoubleBuilder> bs(15);
  for (int64_t k : keys) {
    const auto& st = m.at(k);
    FinalizedAxis f = finalize_axis(st, odds_pop);
    ARROW_RETURN_NOT_OK(kb.Append(k));
    ARROW_RETURN_NOT_OK(bs[0].Append(f.mean));
    ARROW_RETURN_NOT_OK(bs[1].Append(f.stdv));
    ARROW_RETURN_NOT_OK(bs[2].Append(f.median));
    ARROW_RETURN_NOT_OK(bs[3].Append(f.q25));
    ARROW_RETURN_NOT_OK(bs[4].Append(f.q75));
    ARROW_RETURN_NOT_OK(bs[5].Append(f.q95));
    ARROW_RETURN_NOT_OK(bs[6].Append(f.cnt));
    ARROW_RETURN_NOT_OK(bs[7].Append(f.cv));
    ARROW_RETURN_NOT_OK(bs[8].Append(f.fraud_rate));
    ARROW_RETURN_NOT_OK(bs[9].Append(f.fraud_count));
    ARROW_RETURN_NOT_OK(bs[10].Append(f.train_total));
    ARROW_RETURN_NOT_OK(bs[11].Append(f.woe));
    ARROW_RETURN_NOT_OK(bs[12].Append(f.cnt_clean));
    ARROW_RETURN_NOT_OK(bs[13].Append(f.q90));
    ARROW_RETURN_NOT_OK(bs[14].Append(f.q99));
  }
  std::shared_ptr<arrow::Array> ak;
  ARROW_RETURN_NOT_OK(kb.Finish(&ak));
  std::vector<std::shared_ptr<arrow::Array>> arrs;
  arrs.push_back(ak);
  for (int j = 0; j < 15; ++j) {
    std::shared_ptr<arrow::Array> aj;
    ARROW_RETURN_NOT_OK(bs[j].Finish(&aj));
    arrs.push_back(aj);
  }
  auto schema = arrow::schema({
      arrow::field("mcc_code", arrow::int64()),
      arrow::field("global_mean_amount_mcc", arrow::float64()),
      arrow::field("global_std_amount_mcc", arrow::float64()),
      arrow::field("global_median_amount_mcc", arrow::float64()),
      arrow::field("global_q25_mcc", arrow::float64()),
      arrow::field("global_q75_mcc", arrow::float64()),
      arrow::field("global_q95_mcc", arrow::float64()),
      arrow::field("global_cnt_mcc", arrow::float64()),
      arrow::field("global_cv_mcc", arrow::float64()),
      arrow::field("fraud_rate_mcc", arrow::float64()),
      arrow::field("fraud_count_mcc", arrow::float64()),
      arrow::field("train_total_count_mcc", arrow::float64()),
      arrow::field("woe_mcc", arrow::float64()),
      arrow::field("global_cnt_clean_mcc", arrow::float64()),
      arrow::field("global_q90_mcc", arrow::float64()),
      arrow::field("global_q99_mcc", arrow::float64()),
  });
  auto table = arrow::Table::Make(schema, arrs);
  ARROW_ASSIGN_OR_RAISE(auto out, arrow::io::FileOutputStream::Open(path));
  parquet::WriterProperties::Builder pb;
  ARROW_ASSIGN_OR_RAISE(auto writer, parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), out, pb.build()));
  ARROW_RETURN_NOT_OK(writer->WriteTable(*table, table->num_rows()));
  ARROW_RETURN_NOT_OK(writer->Close());
  ARROW_RETURN_NOT_OK(out->Close());
  return arrow::Status::OK();
}

static arrow::Status write_channel_parquet(const std::string& path, const std::unordered_map<std::string, AxisState>& m,
                                           double odds_pop) {
  std::vector<std::string> keys;
  keys.reserve(m.size());
  for (const auto& kv : m) keys.push_back(kv.first);
  std::sort(keys.begin(), keys.end());

  arrow::StringBuilder t_b, s_b;
  std::vector<arrow::DoubleBuilder> bs(15);
  for (const std::string& k : keys) {
    std::string type_part;
    std::string sub_part;
    if (k == "__GLOBAL__" || k == "__MISSING__") {
      type_part = k;
      sub_part = "";
    } else {
      auto sep = k.find('\x1f');
      if (sep == std::string::npos) {
        type_part = k;
        sub_part = "";
      } else {
        type_part = k.substr(0, sep);
        sub_part = k.substr(sep + 1);
      }
    }
    const auto& st = m.at(k);
    FinalizedAxis f = finalize_axis(st, odds_pop);
    ARROW_RETURN_NOT_OK(t_b.Append(type_part));
    ARROW_RETURN_NOT_OK(s_b.Append(sub_part));
    ARROW_RETURN_NOT_OK(bs[0].Append(f.mean));
    ARROW_RETURN_NOT_OK(bs[1].Append(f.stdv));
    ARROW_RETURN_NOT_OK(bs[2].Append(f.median));
    ARROW_RETURN_NOT_OK(bs[3].Append(f.q25));
    ARROW_RETURN_NOT_OK(bs[4].Append(f.q75));
    ARROW_RETURN_NOT_OK(bs[5].Append(f.q95));
    ARROW_RETURN_NOT_OK(bs[6].Append(f.cnt));
    ARROW_RETURN_NOT_OK(bs[7].Append(f.cv));
    ARROW_RETURN_NOT_OK(bs[8].Append(f.fraud_rate));
    ARROW_RETURN_NOT_OK(bs[9].Append(f.fraud_count));
    ARROW_RETURN_NOT_OK(bs[10].Append(f.train_total));
    ARROW_RETURN_NOT_OK(bs[11].Append(f.woe));
    ARROW_RETURN_NOT_OK(bs[12].Append(f.cnt_clean));
    ARROW_RETURN_NOT_OK(bs[13].Append(f.q90));
    ARROW_RETURN_NOT_OK(bs[14].Append(f.q99));
  }
  std::shared_ptr<arrow::Array> a_t, a_s;
  ARROW_RETURN_NOT_OK(t_b.Finish(&a_t));
  ARROW_RETURN_NOT_OK(s_b.Finish(&a_s));
  std::vector<std::shared_ptr<arrow::Array>> arrs = {a_t, a_s};
  for (int j = 0; j < 15; ++j) {
    std::shared_ptr<arrow::Array> aj;
    ARROW_RETURN_NOT_OK(bs[j].Finish(&aj));
    arrs.push_back(aj);
  }
  auto schema = arrow::schema({
      arrow::field("channel_indicator_type", arrow::utf8()),
      arrow::field("channel_indicator_subtype", arrow::utf8()),
      arrow::field("global_mean_amount_channel", arrow::float64()),
      arrow::field("global_std_amount_channel", arrow::float64()),
      arrow::field("global_median_amount_channel", arrow::float64()),
      arrow::field("global_q25_channel", arrow::float64()),
      arrow::field("global_q75_channel", arrow::float64()),
      arrow::field("global_q95_channel", arrow::float64()),
      arrow::field("global_cnt_channel", arrow::float64()),
      arrow::field("global_cv_channel", arrow::float64()),
      arrow::field("fraud_rate_channel", arrow::float64()),
      arrow::field("fraud_count_channel", arrow::float64()),
      arrow::field("train_total_count_channel", arrow::float64()),
      arrow::field("woe_channel", arrow::float64()),
      arrow::field("global_cnt_clean_channel", arrow::float64()),
      arrow::field("global_q90_channel", arrow::float64()),
      arrow::field("global_q99_channel", arrow::float64()),
  });
  auto table = arrow::Table::Make(schema, arrs);
  ARROW_ASSIGN_OR_RAISE(auto out, arrow::io::FileOutputStream::Open(path));
  parquet::WriterProperties::Builder pb;
  ARROW_ASSIGN_OR_RAISE(auto writer, parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), out, pb.build()));
  ARROW_RETURN_NOT_OK(writer->WriteTable(*table, table->num_rows()));
  ARROW_RETURN_NOT_OK(writer->Close());
  ARROW_RETURN_NOT_OK(out->Close());
  return arrow::Status::OK();
}

static arrow::Status write_tz_currency_parquet(const std::string& path, const std::unordered_map<std::string, AxisState>& m,
                                                 double odds_pop) {
  std::vector<std::string> keys;
  keys.reserve(m.size());
  for (const auto& kv : m) keys.push_back(kv.first);
  std::sort(keys.begin(), keys.end());

  arrow::StringBuilder tz_b, c_b;
  std::vector<arrow::DoubleBuilder> bs(15);
  for (const std::string& k : keys) {
    std::string tz_p = "__GLOBAL__";
    std::string c_p = "";
    if (k != "__GLOBAL__") {
      auto sep = k.find('\x1f');
      if (sep != std::string::npos) {
        tz_p = k.substr(0, sep);
        c_p = k.substr(sep + 1);
      } else {
        tz_p = k;
        c_p = "__MISSING_CCY__";
      }
    }
    const auto& st = m.at(k);
    FinalizedAxis f = finalize_axis(st, odds_pop);
    ARROW_RETURN_NOT_OK(tz_b.Append(tz_p));
    ARROW_RETURN_NOT_OK(c_b.Append(c_p));
    ARROW_RETURN_NOT_OK(bs[0].Append(f.mean));
    ARROW_RETURN_NOT_OK(bs[1].Append(f.stdv));
    ARROW_RETURN_NOT_OK(bs[2].Append(f.median));
    ARROW_RETURN_NOT_OK(bs[3].Append(f.q25));
    ARROW_RETURN_NOT_OK(bs[4].Append(f.q75));
    ARROW_RETURN_NOT_OK(bs[5].Append(f.q95));
    ARROW_RETURN_NOT_OK(bs[6].Append(f.cnt));
    ARROW_RETURN_NOT_OK(bs[7].Append(f.cv));
    ARROW_RETURN_NOT_OK(bs[8].Append(f.fraud_rate));
    ARROW_RETURN_NOT_OK(bs[9].Append(f.fraud_count));
    ARROW_RETURN_NOT_OK(bs[10].Append(f.train_total));
    ARROW_RETURN_NOT_OK(bs[11].Append(f.woe));
    ARROW_RETURN_NOT_OK(bs[12].Append(f.cnt_clean));
    ARROW_RETURN_NOT_OK(bs[13].Append(f.q90));
    ARROW_RETURN_NOT_OK(bs[14].Append(f.q99));
  }
  std::shared_ptr<arrow::Array> a_t, a_c;
  ARROW_RETURN_NOT_OK(tz_b.Finish(&a_t));
  ARROW_RETURN_NOT_OK(c_b.Finish(&a_c));
  std::vector<std::shared_ptr<arrow::Array>> arrs = {a_t, a_c};
  for (int j = 0; j < 15; ++j) {
    std::shared_ptr<arrow::Array> aj;
    ARROW_RETURN_NOT_OK(bs[j].Finish(&aj));
    arrs.push_back(aj);
  }
  auto schema = arrow::schema({
      arrow::field("timezone", arrow::utf8()),
      arrow::field("currency_iso_cd", arrow::utf8()),
      arrow::field("global_mean_amount_tz_currency", arrow::float64()),
      arrow::field("global_std_amount_tz_currency", arrow::float64()),
      arrow::field("global_median_amount_tz_currency", arrow::float64()),
      arrow::field("global_q25_tz_currency", arrow::float64()),
      arrow::field("global_q75_tz_currency", arrow::float64()),
      arrow::field("global_q95_tz_currency", arrow::float64()),
      arrow::field("global_cnt_tz_currency", arrow::float64()),
      arrow::field("global_cv_tz_currency", arrow::float64()),
      arrow::field("fraud_rate_tz_currency", arrow::float64()),
      arrow::field("fraud_count_tz_currency", arrow::float64()),
      arrow::field("train_total_count_tz_currency", arrow::float64()),
      arrow::field("woe_tz_currency", arrow::float64()),
      arrow::field("global_cnt_clean_tz_currency", arrow::float64()),
      arrow::field("global_q90_tz_currency", arrow::float64()),
      arrow::field("global_q99_tz_currency", arrow::float64()),
  });
  auto table = arrow::Table::Make(schema, arrs);
  ARROW_ASSIGN_OR_RAISE(auto out, arrow::io::FileOutputStream::Open(path));
  parquet::WriterProperties::Builder pb;
  ARROW_ASSIGN_OR_RAISE(auto writer, parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), out, pb.build()));
  ARROW_RETURN_NOT_OK(writer->WriteTable(*table, table->num_rows()));
  ARROW_RETURN_NOT_OK(writer->Close());
  ARROW_RETURN_NOT_OK(out->Close());
  return arrow::Status::OK();
}

static arrow::Status write_event_type_parquet(const std::string& path, const std::unordered_map<std::string, AxisState>& m,
                                              double odds_pop, double total_tx) {
  std::vector<std::string> keys;
  keys.reserve(m.size());
  for (const auto& kv : m) keys.push_back(kv.first);
  std::sort(keys.begin(), keys.end());

  arrow::DoubleBuilder et_b;
  arrow::StringBuilder c_b;
  std::vector<arrow::DoubleBuilder> bs(12);
  arrow::DoubleBuilder cc_b, q90_b, q99_b, freq_b;
  for (const std::string& k : keys) {
    double et_val = nan_val();
    std::string c_p = "__MISSING_CCY__";
    if (k == "__GLOBAL__") {
      et_val = nan_val();
      c_p = "";
    } else {
      auto sep = k.find('\x1f');
      if (sep != std::string::npos) {
        std::string es = k.substr(0, sep);
        c_p = k.substr(sep + 1);
        if (es.rfind("__MISSING_ET__", 0) == 0) {
          et_val = nan_val();
        } else {
          et_val = std::strtod(es.c_str(), nullptr);
        }
      }
    }
    const auto& st = m.at(k);
    FinalizedAxis f = finalize_axis(st, odds_pop);
    double freq_log = nan_val();
    if (std::isfinite(total_tx) && total_tx >= 1.0) {
      freq_log = -std::log((f.cnt + 1.0) / (total_tx + 1.0));
    }
    ARROW_RETURN_NOT_OK(et_b.Append(et_val));
    ARROW_RETURN_NOT_OK(c_b.Append(c_p));
    ARROW_RETURN_NOT_OK(bs[0].Append(f.mean));
    ARROW_RETURN_NOT_OK(bs[1].Append(f.stdv));
    ARROW_RETURN_NOT_OK(bs[2].Append(f.median));
    ARROW_RETURN_NOT_OK(bs[3].Append(f.q25));
    ARROW_RETURN_NOT_OK(bs[4].Append(f.q75));
    ARROW_RETURN_NOT_OK(bs[5].Append(f.q95));
    ARROW_RETURN_NOT_OK(bs[6].Append(f.cnt));
    ARROW_RETURN_NOT_OK(bs[7].Append(f.cv));
    ARROW_RETURN_NOT_OK(bs[8].Append(f.fraud_rate));
    ARROW_RETURN_NOT_OK(bs[9].Append(f.fraud_count));
    ARROW_RETURN_NOT_OK(bs[10].Append(f.train_total));
    ARROW_RETURN_NOT_OK(bs[11].Append(f.woe));
    ARROW_RETURN_NOT_OK(cc_b.Append(f.cnt_clean));
    ARROW_RETURN_NOT_OK(q90_b.Append(f.q90));
    ARROW_RETURN_NOT_OK(q99_b.Append(f.q99));
    ARROW_RETURN_NOT_OK(freq_b.Append(freq_log));
  }
  std::shared_ptr<arrow::Array> a_et, a_c, a_cc, a_q90, a_q99, a_f;
  ARROW_RETURN_NOT_OK(et_b.Finish(&a_et));
  ARROW_RETURN_NOT_OK(c_b.Finish(&a_c));
  ARROW_RETURN_NOT_OK(cc_b.Finish(&a_cc));
  ARROW_RETURN_NOT_OK(q90_b.Finish(&a_q90));
  ARROW_RETURN_NOT_OK(q99_b.Finish(&a_q99));
  ARROW_RETURN_NOT_OK(freq_b.Finish(&a_f));
  std::vector<std::shared_ptr<arrow::Array>> arrs = {a_et, a_c};
  for (int j = 0; j < 12; ++j) {
    std::shared_ptr<arrow::Array> aj;
    ARROW_RETURN_NOT_OK(bs[j].Finish(&aj));
    arrs.push_back(aj);
  }
  arrs.push_back(a_cc);
  arrs.push_back(a_q90);
  arrs.push_back(a_q99);
  arrs.push_back(a_f);
  auto schema = arrow::schema({
      arrow::field("event_type_nm", arrow::float64()),
      arrow::field("currency_iso_cd", arrow::utf8()),
      arrow::field("global_mean_amount_event_type_currency", arrow::float64()),
      arrow::field("global_std_amount_event_type_currency", arrow::float64()),
      arrow::field("global_median_amount_event_type_currency", arrow::float64()),
      arrow::field("global_q25_event_type_currency", arrow::float64()),
      arrow::field("global_q75_event_type_currency", arrow::float64()),
      arrow::field("global_q95_event_type_currency", arrow::float64()),
      arrow::field("global_cnt_event_type_currency", arrow::float64()),
      arrow::field("global_cv_event_type_currency", arrow::float64()),
      arrow::field("fraud_rate_event_type_currency", arrow::float64()),
      arrow::field("fraud_count_event_type_currency", arrow::float64()),
      arrow::field("train_total_count_event_type_currency", arrow::float64()),
      arrow::field("woe_event_type_currency", arrow::float64()),
      arrow::field("global_cnt_clean_event_type_currency", arrow::float64()),
      arrow::field("global_q90_event_type_currency", arrow::float64()),
      arrow::field("global_q99_event_type_currency", arrow::float64()),
      arrow::field("global_type_frequency_log_event_type_currency", arrow::float64()),
  });
  auto table = arrow::Table::Make(schema, arrs);
  ARROW_ASSIGN_OR_RAISE(auto out, arrow::io::FileOutputStream::Open(path));
  parquet::WriterProperties::Builder pb;
  ARROW_ASSIGN_OR_RAISE(auto writer, parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), out, pb.build()));
  ARROW_RETURN_NOT_OK(writer->WriteTable(*table, table->num_rows()));
  ARROW_RETURN_NOT_OK(writer->Close());
  ARROW_RETURN_NOT_OK(out->Close());
  return arrow::Status::OK();
}

// Empty string → __MISSING__ axis key.
static std::string string_axis_key_missing(const std::string& raw) {
  std::string t = trim_copy(raw);
  if (t.empty()) return "__MISSING__";
  return t;
}

// Timezone-only aggregate key.
static std::string tz_alone_key(const std::string& tz_raw) {
  std::string t = trim_copy(tz_raw);
  if (t.empty()) return "__MISSING_TZ_ALONE__";
  return t;
}

// Single utf8 key column + 15 stat columns.
static arrow::Status write_string_axis_15stats(const std::string& path,
                                                const std::unordered_map<std::string, AxisState>& m,
                                                double odds_pop, const char* key_col_name,
                                                const std::string& stat_suffix) {
  std::vector<std::string> keys;
  keys.reserve(m.size());
  for (const auto& kv : m) keys.push_back(kv.first);
  std::sort(keys.begin(), keys.end());

  arrow::StringBuilder k_b;
  std::vector<arrow::DoubleBuilder> bs(15);
  for (const std::string& k : keys) {
    const auto& st = m.at(k);
    FinalizedAxis f = finalize_axis(st, odds_pop);
    std::string cell = k;
    if (k == "__GLOBAL__") {
      cell = "__GLOBAL__";
    } else if (stat_suffix == "tz_alone" && k == "__MISSING_TZ_ALONE__") {
      cell = "__MISSING_TZ_ALONE__";
    } else if (k == "__MISSING__") {
      cell = "__MISSING__";
    }
    ARROW_RETURN_NOT_OK(k_b.Append(cell));
    ARROW_RETURN_NOT_OK(bs[0].Append(f.mean));
    ARROW_RETURN_NOT_OK(bs[1].Append(f.stdv));
    ARROW_RETURN_NOT_OK(bs[2].Append(f.median));
    ARROW_RETURN_NOT_OK(bs[3].Append(f.q25));
    ARROW_RETURN_NOT_OK(bs[4].Append(f.q75));
    ARROW_RETURN_NOT_OK(bs[5].Append(f.q95));
    ARROW_RETURN_NOT_OK(bs[6].Append(f.cnt));
    ARROW_RETURN_NOT_OK(bs[7].Append(f.cv));
    ARROW_RETURN_NOT_OK(bs[8].Append(f.fraud_rate));
    ARROW_RETURN_NOT_OK(bs[9].Append(f.fraud_count));
    ARROW_RETURN_NOT_OK(bs[10].Append(f.train_total));
    ARROW_RETURN_NOT_OK(bs[11].Append(f.woe));
    ARROW_RETURN_NOT_OK(bs[12].Append(f.cnt_clean));
    ARROW_RETURN_NOT_OK(bs[13].Append(f.q90));
    ARROW_RETURN_NOT_OK(bs[14].Append(f.q99));
  }
  std::shared_ptr<arrow::Array> ak;
  ARROW_RETURN_NOT_OK(k_b.Finish(&ak));
  std::vector<std::shared_ptr<arrow::Array>> arrs = {ak};
  for (int j = 0; j < 15; ++j) {
    std::shared_ptr<arrow::Array> aj;
    ARROW_RETURN_NOT_OK(bs[j].Finish(&aj));
    arrs.push_back(aj);
  }
  const std::string sfx = stat_suffix;
  auto schema = arrow::schema({
      arrow::field(key_col_name, arrow::utf8()),
      arrow::field("global_mean_amount_" + sfx, arrow::float64()),
      arrow::field("global_std_amount_" + sfx, arrow::float64()),
      arrow::field("global_median_amount_" + sfx, arrow::float64()),
      arrow::field("global_q25_" + sfx, arrow::float64()),
      arrow::field("global_q75_" + sfx, arrow::float64()),
      arrow::field("global_q95_" + sfx, arrow::float64()),
      arrow::field("global_cnt_" + sfx, arrow::float64()),
      arrow::field("global_cv_" + sfx, arrow::float64()),
      arrow::field("fraud_rate_" + sfx, arrow::float64()),
      arrow::field("fraud_count_" + sfx, arrow::float64()),
      arrow::field("train_total_count_" + sfx, arrow::float64()),
      arrow::field("woe_" + sfx, arrow::float64()),
      arrow::field("global_cnt_clean_" + sfx, arrow::float64()),
      arrow::field("global_q90_" + sfx, arrow::float64()),
      arrow::field("global_q99_" + sfx, arrow::float64()),
  });
  auto table = arrow::Table::Make(schema, arrs);
  ARROW_ASSIGN_OR_RAISE(auto out, arrow::io::FileOutputStream::Open(path));
  parquet::WriterProperties::Builder pb;
  ARROW_ASSIGN_OR_RAISE(auto writer, parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), out, pb.build()));
  ARROW_RETURN_NOT_OK(writer->WriteTable(*table, table->num_rows()));
  ARROW_RETURN_NOT_OK(writer->Close());
  ARROW_RETURN_NOT_OK(out->Close());
  return arrow::Status::OK();
}

struct AllMaps {
  std::unordered_map<int64_t, AxisState> mcc;
  std::unordered_map<std::string, AxisState> channel;
  std::unordered_map<std::string, AxisState> tz_curr;
  std::unordered_map<std::string, AxisState> event_curr;
  std::unordered_map<std::string, AxisState> event_descr_ax;
  std::unordered_map<std::string, AxisState> pos_cd_ax;
  std::unordered_map<std::string, AxisState> tz_alone_ax;
  PopTotals pop;
  std::unordered_map<int64_t, int64_t> mcc_total_rows;
  std::unordered_map<std::string, int64_t> mcc_ch_joint;
  std::unordered_map<std::string, int64_t> mcc_cur_joint;
  std::unordered_map<std::string, int64_t> mcc_tz_joint;
  std::unordered_map<std::string, int64_t> ch_total_rows;
  std::unordered_map<std::string, int64_t> ch_mcc_joint;
};

static std::string mcc_ch_jk(int64_t mcc_k, const std::string& ch_key) {
  return std::to_string(mcc_k) + "\x1f" + ch_key;
}
static std::string mcc_cur_jk(int64_t mcc_k, const std::string& cur_raw) {
  std::string c = trim_copy(cur_raw);
  if (c.empty()) c = "__MISSING_CCY__";
  return std::to_string(mcc_k) + "\x1f" + c;
}
static std::string mcc_tz_jk(int64_t mcc_k, const std::string& tz_raw) {
  std::string t = trim_copy(tz_raw);
  if (t.empty()) t = "__MISSING_TZ__";
  return std::to_string(mcc_k) + "\x1f" + t;
}
static std::string ch_mcc_jk(const std::string& ch_key, int64_t mcc_k) { return ch_key + "\x1f" + std::to_string(mcc_k); }

static void bump_cooccurrence(AllMaps& M, int64_t mcc_k, const std::string& ch_key, const std::string& cur_raw,
                              const std::string& tz_raw) {
  M.mcc_total_rows[mcc_k]++;
  M.mcc_ch_joint[mcc_ch_jk(mcc_k, ch_key)]++;
  M.mcc_cur_joint[mcc_cur_jk(mcc_k, cur_raw)]++;
  M.mcc_tz_joint[mcc_tz_jk(mcc_k, tz_raw)]++;
  M.ch_total_rows[ch_key]++;
  M.ch_mcc_joint[ch_mcc_jk(ch_key, mcc_k)]++;
}

static void touch_global(AllMaps& M) {
  M.mcc[global_category::kMccGlobalKey];
  M.channel["__GLOBAL__"];
  M.tz_curr["__GLOBAL__"];
  M.event_curr["__GLOBAL__"];
  M.event_descr_ax["__GLOBAL__"];
  M.pos_cd_ax["__GLOBAL__"];
  M.tz_alone_ax["__GLOBAL__"];
}

static arrow::Status write_mcc_totals_parquet(const std::string& path, const AllMaps& M) {
  std::vector<int64_t> keys;
  keys.reserve(M.mcc_total_rows.size());
  for (const auto& kv : M.mcc_total_rows) keys.push_back(kv.first);
  std::sort(keys.begin(), keys.end());
  arrow::Int64Builder kb, nb;
  for (int64_t k : keys) {
    ARROW_RETURN_NOT_OK(kb.Append(k));
    ARROW_RETURN_NOT_OK(nb.Append(M.mcc_total_rows.at(k)));
  }
  std::shared_ptr<arrow::Array> ak, an;
  ARROW_RETURN_NOT_OK(kb.Finish(&ak));
  ARROW_RETURN_NOT_OK(nb.Finish(&an));
  auto schema = arrow::schema({arrow::field("mcc_code", arrow::int64()), arrow::field("n_rows", arrow::int64())});
  auto table = arrow::Table::Make(schema, {ak, an});
  ARROW_ASSIGN_OR_RAISE(auto out, arrow::io::FileOutputStream::Open(path));
  parquet::WriterProperties::Builder pb;
  ARROW_ASSIGN_OR_RAISE(auto writer, parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), out, pb.build()));
  ARROW_RETURN_NOT_OK(writer->WriteTable(*table, table->num_rows()));
  ARROW_RETURN_NOT_OK(writer->Close());
  ARROW_RETURN_NOT_OK(out->Close());
  return arrow::Status::OK();
}

static void split_ch_key(const std::string& ch_comb, std::string* tpart, std::string* spart) {
  size_t q = ch_comb.find('\x1f');
  if (q == std::string::npos) {
    *tpart = ch_comb;
    *spart = "";
  } else {
    *tpart = ch_comb.substr(0, q);
    *spart = ch_comb.substr(q + 1);
  }
}

static arrow::Status write_mcc_channel_joint_parquet(const std::string& path, const AllMaps& M) {
  std::vector<std::pair<std::string, int64_t>> items(M.mcc_ch_joint.begin(), M.mcc_ch_joint.end());
  std::sort(items.begin(), items.end());
  arrow::Int64Builder mcc_b;
  arrow::StringBuilder t_b, s_b;
  arrow::Int64Builder cnt_b;
  for (const auto& kv : items) {
    size_t p = kv.first.find('\x1f');
    if (p == std::string::npos) continue;
    int64_t mcc = 0;
    try {
      mcc = std::stoll(kv.first.substr(0, p));
    } catch (...) {
      continue;
    }
    std::string ch_comb = kv.first.substr(p + 1);
    std::string tp, sp;
    split_ch_key(ch_comb, &tp, &sp);
    ARROW_RETURN_NOT_OK(mcc_b.Append(mcc));
    ARROW_RETURN_NOT_OK(t_b.Append(tp));
    ARROW_RETURN_NOT_OK(s_b.Append(sp));
    ARROW_RETURN_NOT_OK(cnt_b.Append(kv.second));
  }
  std::shared_ptr<arrow::Array> am, at, as, ac;
  ARROW_RETURN_NOT_OK(mcc_b.Finish(&am));
  ARROW_RETURN_NOT_OK(t_b.Finish(&at));
  ARROW_RETURN_NOT_OK(s_b.Finish(&as));
  ARROW_RETURN_NOT_OK(cnt_b.Finish(&ac));
  auto schema = arrow::schema({arrow::field("mcc_code", arrow::int64()), arrow::field("channel_indicator_type", arrow::utf8()),
                               arrow::field("channel_indicator_subtype", arrow::utf8()), arrow::field("cnt", arrow::int64())});
  auto table = arrow::Table::Make(schema, {am, at, as, ac});
  ARROW_ASSIGN_OR_RAISE(auto out, arrow::io::FileOutputStream::Open(path));
  parquet::WriterProperties::Builder pb;
  ARROW_ASSIGN_OR_RAISE(auto writer, parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), out, pb.build()));
  ARROW_RETURN_NOT_OK(writer->WriteTable(*table, table->num_rows()));
  ARROW_RETURN_NOT_OK(writer->Close());
  ARROW_RETURN_NOT_OK(out->Close());
  return arrow::Status::OK();
}

static arrow::Status write_mcc_currency_joint_parquet(const std::string& path, const AllMaps& M) {
  std::vector<std::pair<std::string, int64_t>> items(M.mcc_cur_joint.begin(), M.mcc_cur_joint.end());
  std::sort(items.begin(), items.end());
  arrow::Int64Builder mcc_b;
  arrow::StringBuilder c_b;
  arrow::Int64Builder cnt_b;
  for (const auto& kv : items) {
    size_t p = kv.first.find('\x1f');
    if (p == std::string::npos) continue;
    int64_t mcc = 0;
    try {
      mcc = std::stoll(kv.first.substr(0, p));
    } catch (...) {
      continue;
    }
    ARROW_RETURN_NOT_OK(mcc_b.Append(mcc));
    ARROW_RETURN_NOT_OK(c_b.Append(kv.first.substr(p + 1)));
    ARROW_RETURN_NOT_OK(cnt_b.Append(kv.second));
  }
  std::shared_ptr<arrow::Array> am, ac, an;
  ARROW_RETURN_NOT_OK(mcc_b.Finish(&am));
  ARROW_RETURN_NOT_OK(c_b.Finish(&ac));
  ARROW_RETURN_NOT_OK(cnt_b.Finish(&an));
  auto schema =
      arrow::schema({arrow::field("mcc_code", arrow::int64()), arrow::field("currency_iso_cd", arrow::utf8()), arrow::field("cnt", arrow::int64())});
  auto table = arrow::Table::Make(schema, {am, ac, an});
  ARROW_ASSIGN_OR_RAISE(auto out, arrow::io::FileOutputStream::Open(path));
  parquet::WriterProperties::Builder pb;
  ARROW_ASSIGN_OR_RAISE(auto writer, parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), out, pb.build()));
  ARROW_RETURN_NOT_OK(writer->WriteTable(*table, table->num_rows()));
  ARROW_RETURN_NOT_OK(writer->Close());
  ARROW_RETURN_NOT_OK(out->Close());
  return arrow::Status::OK();
}

static arrow::Status write_mcc_tz_joint_parquet(const std::string& path, const AllMaps& M) {
  std::vector<std::pair<std::string, int64_t>> items(M.mcc_tz_joint.begin(), M.mcc_tz_joint.end());
  std::sort(items.begin(), items.end());
  arrow::Int64Builder mcc_b;
  arrow::StringBuilder z_b;
  arrow::Int64Builder cnt_b;
  for (const auto& kv : items) {
    size_t p = kv.first.find('\x1f');
    if (p == std::string::npos) continue;
    int64_t mcc = 0;
    try {
      mcc = std::stoll(kv.first.substr(0, p));
    } catch (...) {
      continue;
    }
    ARROW_RETURN_NOT_OK(mcc_b.Append(mcc));
    ARROW_RETURN_NOT_OK(z_b.Append(kv.first.substr(p + 1)));
    ARROW_RETURN_NOT_OK(cnt_b.Append(kv.second));
  }
  std::shared_ptr<arrow::Array> am, az, an;
  ARROW_RETURN_NOT_OK(mcc_b.Finish(&am));
  ARROW_RETURN_NOT_OK(z_b.Finish(&az));
  ARROW_RETURN_NOT_OK(cnt_b.Finish(&an));
  auto schema =
      arrow::schema({arrow::field("mcc_code", arrow::int64()), arrow::field("timezone", arrow::utf8()), arrow::field("cnt", arrow::int64())});
  auto table = arrow::Table::Make(schema, {am, az, an});
  ARROW_ASSIGN_OR_RAISE(auto out, arrow::io::FileOutputStream::Open(path));
  parquet::WriterProperties::Builder pb;
  ARROW_ASSIGN_OR_RAISE(auto writer, parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), out, pb.build()));
  ARROW_RETURN_NOT_OK(writer->WriteTable(*table, table->num_rows()));
  ARROW_RETURN_NOT_OK(writer->Close());
  ARROW_RETURN_NOT_OK(out->Close());
  return arrow::Status::OK();
}

static arrow::Status write_channel_mcc_top3_parquet(const std::string& path, const AllMaps& M) {
  std::unordered_map<std::string, std::vector<std::pair<int64_t, int64_t>>> by_ch;
  for (const auto& kv : M.ch_mcc_joint) {
    size_t p = kv.first.rfind('\x1f');
    if (p == std::string::npos) continue;
    std::string chk = kv.first.substr(0, p);
    int64_t mcc = 0;
    try {
      mcc = std::stoll(kv.first.substr(p + 1));
    } catch (...) {
      continue;
    }
    by_ch[chk].push_back({mcc, kv.second});
  }
  std::vector<std::string> ch_keys;
  ch_keys.reserve(by_ch.size());
  for (const auto& kv : by_ch) ch_keys.push_back(kv.first);
  std::sort(ch_keys.begin(), ch_keys.end());
  arrow::StringBuilder t_b, s_b;
  arrow::Int64Builder m1b, m2b, m3b, ntb;
  for (const std::string& chk : ch_keys) {
    auto vec = by_ch[chk];
    std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) {
      if (a.second != b.second) return a.second > b.second;
      return a.first < b.first;
    });
    int64_t c1 = vec.size() > 0 ? vec[0].first : global_category::kMccMissingKey;
    int64_t c2 = vec.size() > 1 ? vec[1].first : global_category::kMccMissingKey;
    int64_t c3 = vec.size() > 2 ? vec[2].first : global_category::kMccMissingKey;
    int64_t tot = 0;
    auto itt = M.ch_total_rows.find(chk);
    if (itt != M.ch_total_rows.end()) tot = itt->second;
    std::string tp, sp;
    split_ch_key(chk, &tp, &sp);
    ARROW_RETURN_NOT_OK(t_b.Append(tp));
    ARROW_RETURN_NOT_OK(s_b.Append(sp));
    ARROW_RETURN_NOT_OK(m1b.Append(c1));
    ARROW_RETURN_NOT_OK(m2b.Append(c2));
    ARROW_RETURN_NOT_OK(m3b.Append(c3));
    ARROW_RETURN_NOT_OK(ntb.Append(tot));
  }
  std::shared_ptr<arrow::Array> at, as, a1, a2, a3, ant;
  ARROW_RETURN_NOT_OK(t_b.Finish(&at));
  ARROW_RETURN_NOT_OK(s_b.Finish(&as));
  ARROW_RETURN_NOT_OK(m1b.Finish(&a1));
  ARROW_RETURN_NOT_OK(m2b.Finish(&a2));
  ARROW_RETURN_NOT_OK(m3b.Finish(&a3));
  ARROW_RETURN_NOT_OK(ntb.Finish(&ant));
  auto schema = arrow::schema({arrow::field("channel_indicator_type", arrow::utf8()),
                               arrow::field("channel_indicator_subtype", arrow::utf8()),
                               arrow::field("top1_mcc", arrow::int64()), arrow::field("top2_mcc", arrow::int64()),
                               arrow::field("top3_mcc", arrow::int64()), arrow::field("ch_row_total", arrow::int64())});
  auto table = arrow::Table::Make(schema, {at, as, a1, a2, a3, ant});
  ARROW_ASSIGN_OR_RAISE(auto out, arrow::io::FileOutputStream::Open(path));
  parquet::WriterProperties::Builder pb;
  ARROW_ASSIGN_OR_RAISE(auto writer, parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), out, pb.build()));
  ARROW_RETURN_NOT_OK(writer->WriteTable(*table, table->num_rows()));
  ARROW_RETURN_NOT_OK(writer->Close());
  ARROW_RETURN_NOT_OK(out->Close());
  return arrow::Status::OK();
}

static arrow::Status write_channel_mcc_pair_parquet(const std::string& path, const AllMaps& M) {
  std::vector<std::pair<std::string, int64_t>> items(M.ch_mcc_joint.begin(), M.ch_mcc_joint.end());
  std::sort(items.begin(), items.end());
  arrow::StringBuilder t_b, s_b;
  arrow::Int64Builder mcc_b, cnt_b;
  for (const auto& kv : items) {
    size_t p = kv.first.rfind('\x1f');
    if (p == std::string::npos) continue;
    std::string chk = kv.first.substr(0, p);
    int64_t mcc = 0;
    try {
      mcc = std::stoll(kv.first.substr(p + 1));
    } catch (...) {
      continue;
    }
    std::string tp, sp;
    split_ch_key(chk, &tp, &sp);
    ARROW_RETURN_NOT_OK(t_b.Append(tp));
    ARROW_RETURN_NOT_OK(s_b.Append(sp));
    ARROW_RETURN_NOT_OK(mcc_b.Append(mcc));
    ARROW_RETURN_NOT_OK(cnt_b.Append(kv.second));
  }
  std::shared_ptr<arrow::Array> at, as, am, ac;
  ARROW_RETURN_NOT_OK(t_b.Finish(&at));
  ARROW_RETURN_NOT_OK(s_b.Finish(&as));
  ARROW_RETURN_NOT_OK(mcc_b.Finish(&am));
  ARROW_RETURN_NOT_OK(cnt_b.Finish(&ac));
  auto schema = arrow::schema({arrow::field("channel_indicator_type", arrow::utf8()),
                               arrow::field("channel_indicator_subtype", arrow::utf8()),
                               arrow::field("mcc_code", arrow::int64()), arrow::field("cnt", arrow::int64())});
  auto table = arrow::Table::Make(schema, {at, as, am, ac});
  ARROW_ASSIGN_OR_RAISE(auto out, arrow::io::FileOutputStream::Open(path));
  parquet::WriterProperties::Builder pb;
  ARROW_ASSIGN_OR_RAISE(auto writer, parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), out, pb.build()));
  ARROW_RETURN_NOT_OK(writer->WriteTable(*table, table->num_rows()));
  ARROW_RETURN_NOT_OK(writer->Close());
  ARROW_RETURN_NOT_OK(out->Close());
  return arrow::Status::OK();
}

static void observe_row(AllMaps& M, const arrow::RecordBatch& batch, int64_t i, const arrow::Schema& sch,
                        const std::unordered_set<int64_t>* label_ids, bool is_train) {
  int cidx = col_index(sch, "customer_id");
  if (cidx < 0) return;
  if (col_get_str(*batch.column(cidx), i).empty()) return;

  int64_t eid = 0;
  int eidx = col_index(sch, "event_id");
  if (eidx >= 0) {
    if (auto p = col_get_int64(*batch.column(eidx), i)) eid = *p;
  }
  bool in_labels = label_ids && label_ids->count(eid) > 0;

  std::optional<double> amt;
  int amt_idx = col_index(sch, "operaton_amt");
  if (amt_idx >= 0) amt = col_get_optional_double(*batch.column(amt_idx), i);
  bool amt_ok = amt.has_value() && std::isfinite(*amt);
  double amount = amt_ok ? *amt : nan_val();

  auto get_s = [&](const char* n) -> std::string {
    int idx = col_index(sch, n);
    if (idx < 0) return {};
    return col_get_str(*batch.column(idx), i);
  };

  int64_t mcc_k = global_category::kMccMissingKey;
  int midx = col_index(sch, "mcc_code");
  if (midx >= 0) mcc_k = parse_mcc_from_column(*batch.column(midx), i);
  else
    mcc_k = parse_mcc_key(get_s("mcc_code"));
  std::string ch_t = get_s("channel_indicator_type");
  std::string ch_s = get_s("channel_indicator_sub_type");
  if (ch_s.empty()) ch_s = get_s("channel_indicator_subtype");
  std::string ch_key = channel_map_key(ch_t, ch_s);
  std::string tz_raw = get_s("timezone");
  std::string cur = get_s("currency_iso_cd");
  std::string tz_key = tz_curr_key(tz_raw, cur);

  double et_num = nan_val();
  bool et_ok = false;
  int et_idx = col_index(sch, "event_type_nm");
  if (et_idx >= 0) {
    if (auto p = col_get_optional_double(*batch.column(et_idx), i)) {
      et_num = *p;
      et_ok = std::isfinite(et_num);
    }
  }
  std::string et_key = event_curr_key(et_num, et_ok, cur);

  std::string ed_raw = get_s("event_descr");
  if (ed_raw.empty()) ed_raw = get_s("event_desc");
  const std::string ed_key = string_axis_key_missing(ed_raw);
  const std::string pos_key = string_axis_key_missing(get_s("pos_cd"));
  const std::string tz_ak = tz_alone_key(tz_raw);

  bump_cooccurrence(M, mcc_k, ch_key, cur, tz_raw);

  if (!is_train) {
    ++M.pop.pretrain_rows;
    axis_observe_pretrain(M.mcc[mcc_k], amount, amt_ok);
    axis_observe_pretrain(M.mcc[global_category::kMccGlobalKey], amount, amt_ok);
    axis_observe_pretrain(M.channel[ch_key], amount, amt_ok);
    axis_observe_pretrain(M.channel["__GLOBAL__"], amount, amt_ok);
    axis_observe_pretrain(M.tz_curr[tz_key], amount, amt_ok);
    axis_observe_pretrain(M.tz_curr["__GLOBAL__"], amount, amt_ok);
    axis_observe_pretrain(M.event_curr[et_key], amount, amt_ok);
    axis_observe_pretrain(M.event_curr["__GLOBAL__"], amount, amt_ok);
    axis_observe_pretrain(M.event_descr_ax[ed_key], amount, amt_ok);
    axis_observe_pretrain(M.event_descr_ax["__GLOBAL__"], amount, amt_ok);
    axis_observe_pretrain(M.pos_cd_ax[pos_key], amount, amt_ok);
    axis_observe_pretrain(M.pos_cd_ax["__GLOBAL__"], amount, amt_ok);
    axis_observe_pretrain(M.tz_alone_ax[tz_ak], amount, amt_ok);
    axis_observe_pretrain(M.tz_alone_ax["__GLOBAL__"], amount, amt_ok);
    return;
  }

  ++M.pop.train_rows;
  if (in_labels) ++M.pop.train_fraud_labels;

  auto train_axis = [&](AxisState& st) {
    axis_observe_train(st, amount, amt_ok, in_labels);
  };
  train_axis(M.mcc[mcc_k]);
  train_axis(M.mcc[global_category::kMccGlobalKey]);
  train_axis(M.channel[ch_key]);
  train_axis(M.channel["__GLOBAL__"]);
  train_axis(M.tz_curr[tz_key]);
  train_axis(M.tz_curr["__GLOBAL__"]);
  train_axis(M.event_curr[et_key]);
  train_axis(M.event_curr["__GLOBAL__"]);
  train_axis(M.event_descr_ax[ed_key]);
  train_axis(M.event_descr_ax["__GLOBAL__"]);
  train_axis(M.pos_cd_ax[pos_key]);
  train_axis(M.pos_cd_ax["__GLOBAL__"]);
  train_axis(M.tz_alone_ax[tz_ak]);
  train_axis(M.tz_alone_ax["__GLOBAL__"]);
}


static arrow::Status load_label_event_ids(const std::string& path, std::unordered_set<int64_t>* out) {
  arrow::MemoryPool* pool = arrow::default_memory_pool();
  ARROW_ASSIGN_OR_RAISE(auto infile, arrow::io::ReadableFile::Open(path));
  ARROW_ASSIGN_OR_RAISE(auto reader, parquet::arrow::OpenFile(infile, pool));
  int64_t total_rows = 0;
  if (auto* pqr = reader->parquet_reader()) {
    if (auto meta = pqr->metadata()) total_rows = meta->num_rows();
  }
  ARROW_ASSIGN_OR_RAISE(auto rb_it, reader->GetRecordBatchReader());
  int64_t rows_seen = 0;
  const std::string tag = ds_progress::path_basename(path) + " [labels->event_ids]";
  log_msg("loading train_labels event_ids rows_meta=" + std::to_string(total_rows));
  while (true) {
    ARROW_ASSIGN_OR_RAISE(auto batch, rb_it->Next());
    if (!batch) break;
    const auto& sch = *batch->schema();
    int ix = col_index(sch, "event_id");
    if (ix < 0) continue;
    const auto& col = *batch->column(ix);
    int64_t n = batch->num_rows();
    for (int64_t i = 0; i < n; ++i) {
      if (auto e = col_get_int64(col, i)) out->insert(*e);
    }
    rows_seen += n;
    ds_progress::render_row_progress(rows_seen, total_rows, tag, "[build_global_aggregates] ");
  }
  ds_progress::finish_progress_line();
  return arrow::Status::OK();
}

static arrow::Status scan_paths(const std::vector<std::string>& paths, AllMaps& M,
                                const std::unordered_set<int64_t>* label_ids, bool is_train) {
  arrow::MemoryPool* pool = arrow::default_memory_pool();
  const size_t npaths = paths.size();
  for (size_t pi = 0; pi < npaths; ++pi) {
    const std::string& path = paths[pi];
    std::ifstream chk(path);
    if (!chk.good()) {
      log_msg("skip missing: " + path);
      continue;
    }
    log_msg("scan " + path + (is_train ? " [train]" : " [pretrain]") +
            " file " + std::to_string(pi + 1) + "/" + std::to_string(npaths));
    ARROW_ASSIGN_OR_RAISE(auto infile, arrow::io::ReadableFile::Open(path));
    ARROW_ASSIGN_OR_RAISE(auto reader, parquet::arrow::OpenFile(infile, pool));
    int64_t total_rows = 0;
    if (auto* pqr = reader->parquet_reader()) {
      if (auto meta = pqr->metadata()) total_rows = meta->num_rows();
    }
    ARROW_ASSIGN_OR_RAISE(auto rb_it, reader->GetRecordBatchReader());
    int64_t rows_seen = 0;
    const std::string tag = ds_progress::path_basename(path) + (is_train ? " [train]" : " [pretrain]") + " (" +
                            std::to_string(pi + 1) + "/" + std::to_string(npaths) + ")";
    while (true) {
      ARROW_ASSIGN_OR_RAISE(auto batch, rb_it->Next());
      if (!batch) break;
      const auto& sch = *batch->schema();
      int64_t n = batch->num_rows();
      for (int64_t i = 0; i < n; ++i) observe_row(M, *batch, i, sch, label_ids, is_train);
      rows_seen += n;
      ds_progress::render_row_progress(rows_seen, total_rows, tag, "[build_global_aggregates] ");
    }
    ds_progress::finish_progress_line();
  }
  return arrow::Status::OK();
}

}  // namespace

int main(int argc, char** argv) {
  std::string root = ".";
  if (argc >= 2) root = argv[1];
  std::string data_train = root + "/data/train/";
  std::string labels_path = root + "/data/train_labels.parquet";
  std::string out_dir = root + "/output/datasets/global_aggregates/";

  std::error_code ec;
  std::filesystem::create_directories(out_dir, ec);
  if (ec) {
    std::cerr << "[build_global_aggregates] cannot mkdir " << out_dir << " " << ec.message() << "\n";
    return 1;
  }

  std::vector<std::string> pre = {data_train + "pretrain_part_1.parquet", data_train + "pretrain_part_2.parquet",
                                  data_train + "pretrain_part_3.parquet"};
  std::vector<std::string> tr = {data_train + "train_part_1.parquet", data_train + "train_part_2.parquet",
                                 data_train + "train_part_3.parquet"};

  std::unordered_set<int64_t> label_ids;
  if (!load_label_event_ids(labels_path, &label_ids).ok()) {
    std::cerr << "[build_global_aggregates] failed labels: " << labels_path << "\n";
    return 1;
  }
  log_msg("train_labels unique event_id count=" + std::to_string(label_ids.size()));

  AllMaps M;
  touch_global(M);
  if (!scan_paths(pre, M, nullptr, false).ok()) {
    std::cerr << "[build_global_aggregates] pretrain scan failed\n";
    return 1;
  }
  if (!scan_paths(tr, M, &label_ids, true).ok()) {
    std::cerr << "[build_global_aggregates] train scan failed\n";
    return 1;
  }

  int64_t tr_n = std::max<int64_t>(1, M.pop.train_rows);
  int64_t fr = M.pop.train_fraud_labels;
  double odds_pop = (static_cast<double>(fr) + 0.5) / (static_cast<double>(tr_n - fr) + 0.5);
  double total_tx = static_cast<double>(M.pop.pretrain_rows + M.pop.train_rows);

  log_msg("population pretrain_rows=" + std::to_string(M.pop.pretrain_rows) + " train_rows=" +
          std::to_string(M.pop.train_rows) + " train_label_rows=" + std::to_string(fr) + " odds_pop=" +
          std::to_string(odds_pop));

  constexpr int k_write_steps = 14;
  int wstep = 0;
  auto wphase = [&](const char* name) {
    ++wstep;
    ds_progress::render_phase_progress(wstep, k_write_steps, std::string("write ") + name,
                                       "[build_global_aggregates] ");
  };

  wphase("population_meta.parquet");
  if (!write_population_meta(out_dir + "population_meta.parquet", M.pop).ok()) return 1;
  wphase("mcc.parquet");
  if (!write_mcc_parquet(out_dir + "mcc.parquet", M.mcc, odds_pop).ok()) return 1;
  wphase("channel_subtype.parquet");
  if (!write_channel_parquet(out_dir + "channel_subtype.parquet", M.channel, odds_pop).ok()) return 1;
  wphase("timezone.parquet");
  if (!write_tz_currency_parquet(out_dir + "timezone.parquet", M.tz_curr, odds_pop).ok()) return 1;
  wphase("event_type_nm.parquet");
  if (!write_event_type_parquet(out_dir + "event_type_nm.parquet", M.event_curr, odds_pop, total_tx).ok()) return 1;
  wphase("mcc_totals.parquet");
  if (!write_mcc_totals_parquet(out_dir + "mcc_totals.parquet", M).ok()) return 1;
  wphase("mcc_channel_joint.parquet");
  if (!write_mcc_channel_joint_parquet(out_dir + "mcc_channel_joint.parquet", M).ok()) return 1;
  wphase("mcc_currency_joint.parquet");
  if (!write_mcc_currency_joint_parquet(out_dir + "mcc_currency_joint.parquet", M).ok()) return 1;
  wphase("mcc_tz_joint.parquet");
  if (!write_mcc_tz_joint_parquet(out_dir + "mcc_tz_joint.parquet", M).ok()) return 1;
  wphase("channel_mcc_top3.parquet");
  if (!write_channel_mcc_top3_parquet(out_dir + "channel_mcc_top3.parquet", M).ok()) return 1;
  wphase("channel_mcc_pair.parquet");
  if (!write_channel_mcc_pair_parquet(out_dir + "channel_mcc_pair.parquet", M).ok()) return 1;
  wphase("event_descr.parquet");
  if (!write_string_axis_15stats(out_dir + "event_descr.parquet", M.event_descr_ax, odds_pop, "event_descr", "event_descr")
           .ok())
    return 1;
  wphase("pos_cd.parquet");
  if (!write_string_axis_15stats(out_dir + "pos_cd.parquet", M.pos_cd_ax, odds_pop, "pos_cd", "pos_cd").ok()) return 1;
  wphase("timezone_alone.parquet");
  if (!write_string_axis_15stats(out_dir + "timezone_alone.parquet", M.tz_alone_ax, odds_pop, "timezone", "tz_alone")
           .ok())
    return 1;
  ds_progress::finish_progress_line();

  log_msg("Done. Written to " + out_dir);
  return 0;
}
