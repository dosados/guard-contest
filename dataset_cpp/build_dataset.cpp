#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

#include <openssl/md5.h>

#include <chrono>
#include <cmath>
#include <deque>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;

// -------- Константы и имена колонок (как в Python) -------------------

static const std::string AMOUNT_COLUMN = "operaton_amt";
static const std::string CUSTOMER_ID_COLUMN = "customer_id";
static const std::string EVENT_DTTM_COLUMN = "event_dttm";
static const std::string EVENT_ID_COLUMN = "event_id";
static const std::string OPERATING_SYSTEM_TYPE = "operating_system_type";
static const std::string DEVICE_SYSTEM_VERSION = "device_system_version";
static const std::string MCC_CODE = "mcc_code";
static const std::string CHANNEL_INDICATOR_TYPE = "channel_indicator_type";
static const std::string CHANNEL_INDICATOR_SUB_TYPE = "channel_indicator_sub_type";
static const std::string TIMEZONE_COLUMN = "timezone";
static const std::string COMPROMISED_COLUMN = "compromised";
static const std::string WEB_RDP_CONNECTION = "web_rdp_connection";
static const std::string PHONE_VOIP_CALL_STATE = "phone_voip_call_state";
static const std::string SESSION_ID_COLUMN = "session_id";
static const std::string BROWSER_LANGUAGE_COLUMN = "browser_language";
static const std::string EVENT_DESCR_COLUMN = "event_descr";
static const std::string EVENT_DESC_COLUMN = "event_desc";

static const std::vector<std::string> FEATURE_COLUMNS = {
    CUSTOMER_ID_COLUMN,
    AMOUNT_COLUMN,
    EVENT_DTTM_COLUMN,
    EVENT_DESCR_COLUMN,
    EVENT_DESC_COLUMN,
    OPERATING_SYSTEM_TYPE,
    DEVICE_SYSTEM_VERSION,
    MCC_CODE,
    CHANNEL_INDICATOR_TYPE,
    CHANNEL_INDICATOR_SUB_TYPE,
    TIMEZONE_COLUMN,
    COMPROMISED_COLUMN,
    WEB_RDP_CONNECTION,
    PHONE_VOIP_CALL_STATE,
    SESSION_ID_COLUMN,
    BROWSER_LANGUAGE_COLUMN,
};

static const int WINDOW_TRANSACTIONS = 150;
static const int WINDOWED_BATCH_SIZE = 30000;

static constexpr double WEIGHT_UNLABELED = 1.0;
static constexpr double WEIGHT_LABELED_0 = 2.0;
static constexpr double WEIGHT_LABELED_1 = 3.0;

// FEATURE_NAMES (из shared/features.py)
static const std::vector<std::string> FEATURE_NAMES = {
    "operation_amt",
    "amount_to_median",
    "amount_zscore",
    "is_amount_high",
    "transactions_last_1h",
    "transactions_last_24h",
    "sum_amount_last_1h",
    "max_amount_last_24h",
    "is_new_device",
    "is_new_mcc",
    "is_new_channel",
    "is_new_timezone",
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
    "event_descr",
    "mcc_code",
};

static const std::string TARGET_COLUMN = "target";
static const std::string WEIGHT_COLUMN = "sample_weight";

// Максимальный размер одного батча (примерно, по числу строк датасета),
// чтобы не держать в памяти весь train сразу.
static const int64_t MAX_BATCH_ROWS = 2'000'000;

// ----------------------- Хелперы для прогресса ------------------------

std::string format_seconds(double seconds) {
    if (seconds < 0) seconds = 0;
    auto total = static_cast<long long>(seconds + 0.5);
    long long h = total / 3600;
    long long m = (total % 3600) / 60;
    long long s = total % 60;
    std::ostringstream oss;
    if (h > 0) {
        oss << h << "h " << std::setw(2) << std::setfill('0') << m << "m";
    } else if (m > 0) {
        oss << m << "m " << std::setw(2) << std::setfill('0') << s << "s";
    } else {
        oss << s << "s";
    }
    return oss.str();
}

void print_progress_bar(const std::string& prefix,
                        int64_t current,
                        int64_t total,
                        const std::chrono::steady_clock::time_point& start_time) {
    if (total <= 0 || current <= 0) return;
    double frac = static_cast<double>(current) / static_cast<double>(total);
    if (frac < 0.0) frac = 0.0;
    if (frac > 1.0) frac = 1.0;

    auto now = std::chrono::steady_clock::now();
    double elapsed_sec = std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time).count();
    double eta_sec = elapsed_sec * (1.0 / std::max(frac, 1e-9) - 1.0);

    const int bar_width = 30;
    int pos = static_cast<int>(bar_width * frac);

    std::ostringstream oss;
    oss << prefix << " [";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) oss << '=';
        else if (i == pos) oss << '>';
        else oss << ' ';
    }
    oss << "] " << std::fixed << std::setprecision(1) << (frac * 100.0) << "%";
    oss << " | elapsed " << format_seconds(elapsed_sec)
        << " | ETA " << format_seconds(eta_sec);

    std::cout << "\r" << oss.str() << std::flush;
}

void finish_progress_line() {
    std::cout << std::endl;
}

// ---------- Дата/время и weekday (как datetime.weekday) ---------------

struct DateTime {
    int year, month, day, hour, minute, second;
};

std::optional<DateTime> parse_dttm_str(const std::string& s) {
    if (s.empty()) return std::nullopt;
    DateTime dt{};
    if (std::sscanf(s.c_str(), "%d-%d-%d %d:%d:%d",
                    &dt.year, &dt.month, &dt.day,
                    &dt.hour, &dt.minute, &dt.second) != 6) {
        return std::nullopt;
    }
    return dt;
}

std::optional<std::chrono::system_clock::time_point> to_time_point(const DateTime& dt) {
    std::tm t{};
    t.tm_year = dt.year - 1900;
    t.tm_mon  = dt.month - 1;
    t.tm_mday = dt.day;
    t.tm_hour = dt.hour;
    t.tm_min  = dt.minute;
    t.tm_sec  = dt.second;
    std::time_t tt = std::mktime(&t);
    if (tt == -1) return std::nullopt;
    return std::chrono::system_clock::from_time_t(tt);
}

// Sakamoto's algorithm, Sunday = 0
int weekday_sakamoto(int y, int m, int d) {
    static int t[] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
    if (m < 3) y -= 1;
    return (y + y/4 - y/100 + y/400 + t[m-1] + d) % 7;
}
// Python weekday: Monday=0, Sunday=6
int weekday_python(int y, int m, int d) {
    int w = weekday_sakamoto(y, m, d); // Sun=0
    return (w + 6) % 7;                // Mon=0, Tue=1, ..., Sun=6
}

// --------------------- Агрегаты по пользователю -----------------------

struct UserAggregates {
    double sum_amt = 0.0;
    double sum_sq = 0.0;
    int count = 0;
    std::vector<double> amounts;
    std::deque<std::pair<std::chrono::system_clock::time_point, double>> recent_events;

    std::set<std::pair<std::string, std::string>> seen_devices;
    std::set<std::string> seen_mcc;
    std::set<std::pair<std::string, std::string>> seen_channels;
    std::set<std::string> seen_timezones;
    bool ever_compromised = false;
    bool ever_web_rdp = false;
    bool ever_voip = false;
    std::optional<std::chrono::system_clock::time_point> last_event_dttm;
    std::unordered_map<std::string, int> session_counts;
    std::set<std::string> seen_browser_languages;

    void update(const std::unordered_map<std::string, std::string>& row) {
        auto get = [&](const std::string& k)->std::string{
            auto it = row.find(k);
            return (it == row.end()) ? "" : it->second;
        };

        std::string amt_s = get(AMOUNT_COLUMN);
        if (amt_s.empty()) return;
        double amount;
        try { amount = std::stod(amt_s); } catch (...) { return; }

        sum_amt += amount;
        sum_sq  += amount * amount;
        count   += 1;
        amounts.push_back(amount);

        std::string dttm_s = get(EVENT_DTTM_COLUMN);
        if (!dttm_s.empty()) {
            auto dt_opt = parse_dttm_str(dttm_s);
            if (dt_opt) {
                auto tp_opt = to_time_point(*dt_opt);
                if (tp_opt) {
                    auto tp = *tp_opt;
                    auto cutoff = tp - std::chrono::hours(24);
                    while (!recent_events.empty() && recent_events.front().first < cutoff)
                        recent_events.pop_front();
                    recent_events.emplace_back(tp, amount);
                    last_event_dttm = tp;
                }
            }
        }

        std::string os_type = get(OPERATING_SYSTEM_TYPE);
        std::string dev_ver = get(DEVICE_SYSTEM_VERSION);
        if (!os_type.empty() || !dev_ver.empty())
            seen_devices.emplace(os_type, dev_ver);

        std::string mcc = get(MCC_CODE);
        if (!mcc.empty()) seen_mcc.insert(mcc);

        std::string ch_type = get(CHANNEL_INDICATOR_TYPE);
        std::string ch_sub  = get(CHANNEL_INDICATOR_SUB_TYPE);
        if (!ch_type.empty() || !ch_sub.empty())
            seen_channels.emplace(ch_type, ch_sub);

        std::string tz = get(TIMEZONE_COLUMN);
        if (!tz.empty()) seen_timezones.insert(tz);

        if (!get(COMPROMISED_COLUMN).empty())      ever_compromised = true;
        if (!get(WEB_RDP_CONNECTION).empty())      ever_web_rdp = true;
        if (!get(PHONE_VOIP_CALL_STATE).empty())   ever_voip = true;

        std::string sid = get(SESSION_ID_COLUMN);
        if (!sid.empty()) session_counts[sid] += 1;

        std::string bl = get(BROWSER_LANGUAGE_COLUMN);
        if (!bl.empty()) seen_browser_languages.insert(bl);
    }

    double mean() const {
        if (count == 0) return std::numeric_limits<double>::quiet_NaN();
        return sum_amt / count;
    }

    double std() const {
        if (count < 2) return std::numeric_limits<double>::quiet_NaN();
        double var = (sum_sq - sum_amt * sum_amt / count) / (count - 1);
        if (var < 0) var = 0;
        return std::sqrt(var);
    }

    double median() const {
        if (amounts.empty()) return std::numeric_limits<double>::quiet_NaN();
        std::vector<double> tmp = amounts;
        std::sort(tmp.begin(), tmp.end());
        size_t n = tmp.size();
        if (n % 2 == 1) return tmp[n/2];
        return 0.5 * (tmp[n/2 - 1] + tmp[n/2]);
    }

    double percentile(double q) const {
        if (amounts.empty()) return std::numeric_limits<double>::quiet_NaN();
        std::vector<double> tmp = amounts;
        std::sort(tmp.begin(), tmp.end());
        double pos = (q / 100.0) * (tmp.size() - 1);
        size_t idx = static_cast<size_t>(std::round(pos));
        if (idx >= tmp.size()) idx = tmp.size() - 1;
        return tmp[idx];
    }

    int transactions_last_1h(const std::chrono::system_clock::time_point& now) const {
        auto cutoff = now - std::chrono::hours(1);
        int c = 0;
        for (auto& p : recent_events) if (p.first >= cutoff) ++c;
        return c;
    }

    int transactions_last_24h(const std::chrono::system_clock::time_point& now) const {
        auto cutoff = now - std::chrono::hours(24);
        int c = 0;
        for (auto& p : recent_events) if (p.first >= cutoff) ++c;
        return c;
    }

    double sum_amount_last_1h(const std::chrono::system_clock::time_point& now) const {
        auto cutoff = now - std::chrono::hours(1);
        double s = 0.0;
        for (auto& p : recent_events) if (p.first >= cutoff) s += p.second;
        return s;
    }

    double max_amount_last_24h(const std::chrono::system_clock::time_point& now) const {
        auto cutoff = now - std::chrono::hours(24);
        double m = std::numeric_limits<double>::quiet_NaN();
        bool has = false;
        for (auto& p : recent_events) {
            if (p.first >= cutoff) {
                if (!has || p.second > m) {
                    m = p.second;
                    has = true;
                }
            }
        }
        return has ? m : std::numeric_limits<double>::quiet_NaN();
    }

    int transactions_last_10m(const std::chrono::system_clock::time_point& now) const {
        auto cutoff = now - std::chrono::minutes(10);
        int c = 0;
        for (auto& p : recent_events) if (p.first >= cutoff) ++c;
        return c;
    }

    double sum_amount_last_24h(const std::chrono::system_clock::time_point& now) const {
        auto cutoff = now - std::chrono::hours(24);
        double s = 0.0;
        for (auto& p : recent_events) if (p.first >= cutoff) s += p.second;
        return s;
    }

    int get_session_count(const std::string& sid) const {
        if (sid.empty()) return 0;
        auto it = session_counts.find(sid);
        return it == session_counts.end() ? 0 : it->second;
    }
};

// ------ Оконное хранение последних транзакций по пользователю --------

struct WindowedAggregates {
    std::deque<std::unordered_map<std::string, std::string>> rows;
    int window_size;

    explicit WindowedAggregates(int ws = WINDOW_TRANSACTIONS)
        : window_size(ws) {}

    void add(const std::unordered_map<std::string, std::string>& row) {
        if ((int)rows.size() == window_size) rows.pop_front();
        rows.push_back(row);
    }

    UserAggregates get_aggregates() const {
        UserAggregates agg;
        for (auto& r : rows) agg.update(r);
        return agg;
    }

    size_t size() const { return rows.size(); }
};

// ----------------- Чтение parquet и преобразование строк --------------

std::shared_ptr<arrow::Table> read_parquet_table(const fs::path& path) {
    std::shared_ptr<arrow::io::ReadableFile> infile;
    PARQUET_ASSIGN_OR_THROW(
        infile,
        arrow::io::ReadableFile::Open(path.string())
    );
    auto reader_result = parquet::arrow::OpenFile(infile, arrow::default_memory_pool());
    if (!reader_result.ok()) {
        throw std::runtime_error(reader_result.status().ToString());
    }
    std::unique_ptr<parquet::arrow::FileReader> reader = std::move(reader_result).ValueOrDie();
    std::shared_ptr<arrow::Table> table;
    PARQUET_THROW_NOT_OK(reader->ReadTable(&table));
    PARQUET_ASSIGN_OR_THROW(table, table->CombineChunks());
    return table;
}

std::unordered_map<std::string, std::string> row_to_map(
    const std::shared_ptr<arrow::Table>& table,
    int64_t row_idx,
    const std::vector<std::string>& cols)
{
    std::unordered_map<std::string, std::string> row;
    for (const auto& name : cols) {
        auto col = table->GetColumnByName(name);
        if (!col) continue;
        auto chunked = std::static_pointer_cast<arrow::ChunkedArray>(col);
        if (chunked->num_chunks() == 0) continue;
        auto arr = chunked->chunk(0);
        if (row_idx >= arr->length()) continue;
        if (arr->IsNull(row_idx)) continue;
        auto scalar_res = arr->GetScalar(row_idx);
        if (!scalar_res.ok()) continue;
        auto scalar = *scalar_res;
        row[name] = scalar->ToString();
    }
    return row;
}

// ------------------- Загрузка разметки event_id -> target ------------

std::unordered_map<long long, int> load_train_labels(const fs::path& path) {
    if (!fs::exists(path))
        throw std::runtime_error("Labels not found: " + path.string());
    auto table = read_parquet_table(path);
    auto event_col = table->GetColumnByName("event_id");
    auto target_col = table->GetColumnByName("target");
    if (!event_col || !target_col)
        throw std::runtime_error("Labels parquet must have event_id and target");

    std::unordered_map<long long, int> labels;
    auto n = table->num_rows();
    auto ev_chunked = std::static_pointer_cast<arrow::ChunkedArray>(event_col);
    auto tg_chunked = std::static_pointer_cast<arrow::ChunkedArray>(target_col);
    auto ev_arr = ev_chunked->chunk(0);
    auto tg_arr = tg_chunked->chunk(0);

    for (int64_t i = 0; i < n; ++i) {
        if (ev_arr->IsNull(i) || tg_arr->IsNull(i)) continue;
        auto ev_scalar_res = ev_arr->GetScalar(i);
        auto tg_scalar_res = tg_arr->GetScalar(i);
        if (!ev_scalar_res.ok() || !tg_scalar_res.ok()) continue;
        auto ev_scalar = *ev_scalar_res;
        auto tg_scalar = *tg_scalar_res;
        long long eid = std::stoll(ev_scalar->ToString());
        int tgt = std::stoi(tg_scalar->ToString());
        labels[eid] = tgt;
    }
    return labels;
}

// ----------------- md5-хеш для категориальных признаков --------------

double cat_to_float(const std::string& val) {
    if (val.empty()) return 0.0;
    unsigned char digest[MD5_DIGEST_LENGTH];
    MD5(reinterpret_cast<const unsigned char*>(val.c_str()), val.size(), digest);
    __int128 big = 0;
    for (int i = 0; i < MD5_DIGEST_LENGTH; ++i) {
        big = (big << 8) | digest[i];
    }
    long long mod = static_cast<long long>(big % 1000000);
    if (mod < 0) mod += 1000000;
    return static_cast<double>(mod);
}

// ----------------- compute_features (порт с shared/features.py) -------

std::unordered_map<std::string, double> compute_features_cpp(
    const UserAggregates& agg,
    const std::unordered_map<std::string, std::string>& row,
    int tr_amount)
{
    auto get_double_nan = [&](const std::string& key)->double{
        auto it = row.find(key);
        if (it == row.end() || it->second.empty())
            return std::numeric_limits<double>::quiet_NaN();
        try { return std::stod(it->second); } catch (...) {
            return std::numeric_limits<double>::quiet_NaN();
        }
    };
    auto get_str = [&](const std::string& key)->std::string{
        auto it = row.find(key);
        return (it == row.end()) ? "" : it->second;
    };

    double amount = get_double_nan(AMOUNT_COLUMN);

    double median = agg.median();
    double mean   = agg.mean();
    double stdv   = agg.std();
    auto nan = std::numeric_limits<double>::quiet_NaN();

    double amount_to_median =
        (!std::isnan(median) && median != 0.0) ? (amount / median) : nan;
    double amount_zscore =
        (!std::isnan(stdv) && stdv > 0.0) ? ((amount - mean) / stdv) : nan;

    double p95 = agg.percentile(95.0);
    double is_amount_high = (!std::isnan(p95) && amount > p95) ? 1.0 : 0.0;

    int transactions_last_1h = 0;
    int transactions_last_24h = 0;
    double sum_amount_last_1h = 0.0;
    double max_amount_last_24h = nan;
    int transactions_last_10m = 0;
    double sum_amount_last_24h = 0.0;
    int hour_val = -1;
    int day_of_week_val = -1;
    double time_since_prev = nan;

    std::string dttm_s = get_str(EVENT_DTTM_COLUMN);
    auto dt_opt = parse_dttm_str(dttm_s);
    if (dt_opt) {
        auto tp_opt = to_time_point(*dt_opt);
        if (tp_opt) {
            auto tp = *tp_opt;
            transactions_last_1h = agg.transactions_last_1h(tp);
            transactions_last_24h = agg.transactions_last_24h(tp);
            sum_amount_last_1h = agg.sum_amount_last_1h(tp);
            max_amount_last_24h = agg.max_amount_last_24h(tp);
            transactions_last_10m = agg.transactions_last_10m(tp);
            sum_amount_last_24h = agg.sum_amount_last_24h(tp);
            hour_val = dt_opt->hour;
            day_of_week_val = weekday_python(dt_opt->year, dt_opt->month, dt_opt->day);
            if (agg.last_event_dttm) {
                auto diff = std::chrono::duration_cast<std::chrono::seconds>(
                    tp - *agg.last_event_dttm).count();
                time_since_prev = static_cast<double>(diff);
            }
        }
    }

    std::string os_type = get_str(OPERATING_SYSTEM_TYPE);
    std::string dev_ver = get_str(DEVICE_SYSTEM_VERSION);
    std::pair<std::string,std::string> device_key{os_type, dev_ver};
    double is_new_device =
        (!os_type.empty() || !dev_ver.empty()) &&
        (agg.seen_devices.find(device_key) == agg.seen_devices.end()) ? 1.0 : 0.0;

    std::string mcc = get_str(MCC_CODE);
    double is_new_mcc =
        (!mcc.empty() && agg.seen_mcc.find(mcc) == agg.seen_mcc.end()) ? 1.0 : 0.0;

    std::string ch_type = get_str(CHANNEL_INDICATOR_TYPE);
    std::string ch_sub  = get_str(CHANNEL_INDICATOR_SUB_TYPE);
    std::pair<std::string,std::string> ch_key{ch_type, ch_sub};
    double is_new_channel =
        (!ch_type.empty() || !ch_sub.empty()) &&
        (agg.seen_channels.find(ch_key) == agg.seen_channels.end()) ? 1.0 : 0.0;

    std::string tz = get_str(TIMEZONE_COLUMN);
    double is_new_timezone =
        (!tz.empty() && agg.seen_timezones.find(tz) == agg.seen_timezones.end()) ? 1.0 : 0.0;

    double is_compromised_device =
        (!get_str(COMPROMISED_COLUMN).empty()) ? 1.0 : 0.0;
    double web_rdp_connection =
        (!get_str(WEB_RDP_CONNECTION).empty()) ? 1.0 : 0.0;
    double phone_voip_call_state =
        (!get_str(PHONE_VOIP_CALL_STATE).empty()) ? 1.0 : 0.0;

    double is_night =
        (hour_val >= 22 || hour_val < 6) ? 1.0 : 0.0;
    double is_weekend =
        (day_of_week_val >= 5) ? 1.0 : 0.0;

    std::string bl = get_str(BROWSER_LANGUAGE_COLUMN);
    double is_new_browser_language =
        (!bl.empty() && agg.seen_browser_languages.find(bl) == agg.seen_browser_languages.end())
        ? 1.0 : 0.0;

    std::string sid = get_str(SESSION_ID_COLUMN);
    double transactions_in_session =
        static_cast<double>(agg.get_session_count(sid) + 1);

    std::string tz_val = get_str(TIMEZONE_COLUMN);
    double timezone_missing =
        (tz_val.empty()) ? 1.0 : 0.0;

    std::unordered_map<std::string, double> res;
    res["operation_amt"] = amount;
    res["amount_to_median"] = amount_to_median;
    res["amount_zscore"] = amount_zscore;
    res["is_amount_high"] = is_amount_high;
    res["transactions_last_1h"] = static_cast<double>(transactions_last_1h);
    res["transactions_last_24h"] = static_cast<double>(transactions_last_24h);
    res["sum_amount_last_1h"] = sum_amount_last_1h;
    res["max_amount_last_24h"] =
        std::isnan(max_amount_last_24h) ? 0.0 : max_amount_last_24h;
    res["is_new_device"] = is_new_device;
    res["is_new_mcc"] = is_new_mcc;
    res["is_new_channel"] = is_new_channel;
    res["is_new_timezone"] = is_new_timezone;
    res["is_compromised_device"] = is_compromised_device;
    res["web_rdp_connection"] = web_rdp_connection;
    res["phone_voip_call_state"] = phone_voip_call_state;
    res["hour"] = (hour_val >= 0) ? static_cast<double>(hour_val)
                                  : std::numeric_limits<double>::quiet_NaN();
    res["day_of_week"] = (day_of_week_val >= 0) ? static_cast<double>(day_of_week_val)
                                                : std::numeric_limits<double>::quiet_NaN();
    res["is_night_transaction"] = is_night;
    res["is_weekend"] = is_weekend;
    res["transactions_last_10m"] = static_cast<double>(transactions_last_10m);
    res["sum_amount_last_24h"] = sum_amount_last_24h;
    res["time_since_prev_transaction"] =
        std::isnan(time_since_prev) ? -1.0 : time_since_prev;
    res["is_new_browser_language"] = is_new_browser_language;
    res["transactions_in_session"] = transactions_in_session;
    res["timezone_missing"] = timezone_missing;
    res["tr_amount"] = static_cast<double>(tr_amount);

    std::string ev_descr = get_str(EVENT_DESCR_COLUMN);
    if (ev_descr.empty()) ev_descr = get_str(EVENT_DESC_COLUMN);
    res["event_descr"] = cat_to_float(ev_descr);
    res["mcc_code"] = cat_to_float(get_str(MCC_CODE));

    return res;
}

// ----------------- Заполнение оконных агрегатов из pretrain ----------

void build_windowed_from_pretrain(
    const std::vector<fs::path>& pretrain_paths,
    std::unordered_map<std::string, WindowedAggregates>& win_aggs)
{
    for (const auto& path : pretrain_paths) {
        if (!fs::exists(path)) {
            std::cerr << "Pretrain file not found: " << path << "\n";
            continue;
        }
        std::cout << "Pretrain: " << path << std::endl;
        auto table = read_parquet_table(path);
        auto n = table->num_rows();
        auto start_time = std::chrono::steady_clock::now();
        const int64_t report_every = std::max<int64_t>(n / 200, 10000); // ~200 апдейтов, но не чаще чем каждые 10k строк
        for (int64_t i = 0; i < n; ++i) {
            auto row = row_to_map(table, i, FEATURE_COLUMNS);
            auto it_cid = row.find(CUSTOMER_ID_COLUMN);
            if (it_cid == row.end() || it_cid->second.empty()) continue;
            std::string cid = it_cid->second;
            auto& wagg = win_aggs[cid];
            if (wagg.window_size == 0) wagg.window_size = WINDOW_TRANSACTIONS;
            wagg.add(row);

            if (i % report_every == 0 || i + 1 == n) {
                print_progress_bar("  rows", i + 1, n, start_time);
            }
        }
        finish_progress_line();
    }
}

// ----------------- Основной проход по train (аналог _run_sequential) --

struct Columnar {
    std::unordered_map<std::string, std::vector<double>> num_cols;
    std::vector<int64_t> event_id;
    std::vector<int32_t> target;
    std::vector<double> weight;
    std::vector<std::string> event_dttm;
};

// Объявление функции записи батча в parquet (реализация ниже).
void write_parquet_dataset(const Columnar& col, const fs::path& out_path);

void init_columnar(Columnar& col) {
    col.num_cols.clear();
    for (const auto& name : FEATURE_NAMES) {
        col.num_cols[name] = {};
    }
    col.event_id.clear();
    col.target.clear();
    col.weight.clear();
    col.event_dttm.clear();
}

int64_t process_train_sequential(
    const std::vector<fs::path>& train_paths,
    const std::unordered_map<long long, int>& labels,
    std::unordered_map<std::string, WindowedAggregates>& win_aggs,
    const fs::path& out_dir)
{
    Columnar batch;
    init_columnar(batch);
    int64_t total_rows = 0;
    int part_index = 0;

    for (const auto& path : train_paths) {
        if (!fs::exists(path)) {
            std::cerr << "Train file not found: " << path << "\n";
            continue;
        }
        std::cout << "Train: " << path << std::endl;
        auto table = read_parquet_table(path);
        auto n = table->num_rows();
        auto start_time = std::chrono::steady_clock::now();
        const int64_t report_every = std::max<int64_t>(n / 200, 10000);
        for (int64_t i = 0; i < n; ++i) {
            auto row = row_to_map(table, i, FEATURE_COLUMNS);
            auto it_cid = row.find(CUSTOMER_ID_COLUMN);
            if (it_cid == row.end() || it_cid->second.empty()) continue;
            std::string cid = it_cid->second;

            auto& wagg = win_aggs[cid];
            if (wagg.window_size == 0) wagg.window_size = WINDOW_TRANSACTIONS;
            UserAggregates agg = wagg.get_aggregates();
            int tr_amount = std::min<int>(wagg.size(), WINDOW_TRANSACTIONS);
            auto feats = compute_features_cpp(agg, row, tr_amount);
            wagg.add(row);

            long long eid = 0;
            auto it_eid = row.find(EVENT_ID_COLUMN);
            if (it_eid != row.end() && !it_eid->second.empty()) {
                eid = std::stoll(it_eid->second);
            }
            bool is_labeled = labels.find(eid) != labels.end();
            int tgt = is_labeled ? 1 : 0;
            double w = WEIGHT_UNLABELED;
            if (is_labeled) {
                int lab = labels.at(eid);
                w = (lab == 0) ? WEIGHT_LABELED_0 : WEIGHT_LABELED_1;
            }

            for (const auto& name : FEATURE_NAMES) {
                batch.num_cols[name].push_back(feats.at(name));
            }
            batch.event_id.push_back(eid);
            batch.target.push_back(tgt);
            batch.weight.push_back(w);
            auto it_dttm = row.find(EVENT_DTTM_COLUMN);
            batch.event_dttm.push_back(it_dttm == row.end() ? "" : it_dttm->second);

            ++total_rows;

            // Если батч разросся, записываем его на диск и очищаем.
            if (static_cast<int64_t>(batch.event_id.size()) >= MAX_BATCH_ROWS) {
                fs::path out_path = out_dir / ("train_dataset_part_" + std::to_string(part_index) + ".parquet");
                std::cout << "\nFlushing batch to " << out_path << " (" << batch.event_id.size() << " rows)\n";
                write_parquet_dataset(batch, out_path);
                ++part_index;
                init_columnar(batch);
            }

            if (i % report_every == 0 || i + 1 == n) {
                print_progress_bar("  rows", i + 1, n, start_time);
            }
        }
        finish_progress_line();
    }

    // Финальный батч, если остались строки.
    if (!batch.event_id.empty()) {
        fs::path out_path = out_dir / ("train_dataset_part_" + std::to_string(part_index) + ".parquet");
        std::cout << "Flushing final batch to " << out_path << " (" << batch.event_id.size() << " rows)\n";
        write_parquet_dataset(batch, out_path);
        ++part_index;
    }

    if (total_rows == 0) {
        throw std::runtime_error("No rows collected");
    }

    std::cout << "Total rows written: " << total_rows
              << " in " << part_index << " parquet part(s)\n";

    return total_rows;
}

// ----------------- Запись результата в parquet ------------------------

void write_parquet_dataset(const Columnar& col, const fs::path& out_path) {
    auto pool = arrow::default_memory_pool();
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;

    for (const auto& name : FEATURE_NAMES) {
        arrow::DoubleBuilder builder(pool);
        const auto& v = col.num_cols.at(name);
        PARQUET_THROW_NOT_OK(builder.AppendValues(v));
        std::shared_ptr<arrow::Array> arr;
        PARQUET_THROW_NOT_OK(builder.Finish(&arr));
        fields.push_back(arrow::field(name, arrow::float64()));
        arrays.push_back(arr);
    }

    {
        arrow::Int64Builder b(pool);
        PARQUET_THROW_NOT_OK(b.AppendValues(col.event_id));
        std::shared_ptr<arrow::Array> arr;
        PARQUET_THROW_NOT_OK(b.Finish(&arr));
        fields.push_back(arrow::field("event_id", arrow::int64()));
        arrays.push_back(arr);
    }
    {
        arrow::Int32Builder b(pool);
        PARQUET_THROW_NOT_OK(b.AppendValues(col.target));
        std::shared_ptr<arrow::Array> arr;
        PARQUET_THROW_NOT_OK(b.Finish(&arr));
        fields.push_back(arrow::field("target", arrow::int32()));
        arrays.push_back(arr);
    }
    {
        arrow::DoubleBuilder b(pool);
        PARQUET_THROW_NOT_OK(b.AppendValues(col.weight));
        std::shared_ptr<arrow::Array> arr;
        PARQUET_THROW_NOT_OK(b.Finish(&arr));
        fields.push_back(arrow::field("sample_weight", arrow::float64()));
        arrays.push_back(arr);
    }
    {
        arrow::StringBuilder b(pool);
        for (auto& s : col.event_dttm) {
            PARQUET_THROW_NOT_OK(b.Append(s));
        }
        std::shared_ptr<arrow::Array> arr;
        PARQUET_THROW_NOT_OK(b.Finish(&arr));
        fields.push_back(arrow::field("event_dttm", arrow::utf8()));
        arrays.push_back(arr);
    }

    auto schema = arrow::schema(fields);
    auto table = arrow::Table::Make(schema, arrays);

    fs::create_directories(out_path.parent_path());
    std::shared_ptr<arrow::io::FileOutputStream> outfile;
    PARQUET_ASSIGN_OR_THROW(
        outfile,
        arrow::io::FileOutputStream::Open(out_path.string())
    );
    PARQUET_THROW_NOT_OK(
        parquet::arrow::WriteTable(*table, pool, outfile, 1024 * 1024)
    );
}

// ----------------- main ----------------------------------------------

int main(int argc, char** argv) {
    try {
        // Ожидается, что бинарь запускается из корня репозитория (PROJECT_ROOT)
        fs::path project_root = fs::current_path();
        fs::path data_root = project_root / "data";
        fs::path train_root = data_root / "train";

        std::vector<fs::path> pretrain_paths = {
            train_root / "pretrain_part_1.parquet",
            train_root / "pretrain_part_2.parquet",
            train_root / "pretrain_part_3.parquet",
        };
        std::vector<fs::path> train_paths = {
            train_root / "train_part_1.parquet",
            train_root / "train_part_2.parquet",
            train_root / "train_part_3.parquet",
        };
        fs::path labels_path = data_root / "train_labels.parquet";
        fs::path out_dir = project_root / "output";

        std::cout << "Loading labels from " << labels_path << std::endl;
        auto labels = load_train_labels(labels_path);
        std::cout << "Loaded " << labels.size() << " labeled events\n";

        std::unordered_map<std::string, WindowedAggregates> win_aggs;
        std::cout << "Building windowed aggregates from pretrain...\n";
        build_windowed_from_pretrain(pretrain_paths, win_aggs);
        std::cout << "Pretrain users: " << win_aggs.size() << "\n";

        std::cout << "Processing train files...\n";
        fs::create_directories(out_dir);
        int64_t total_rows = process_train_sequential(train_paths, labels, win_aggs, out_dir);
        std::cout << "Done, total rows: " << total_rows << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << std::endl;
        return 1;
    }
}

