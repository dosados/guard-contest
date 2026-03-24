"""Имена и расчёт фич — по dataset_cpp_module_spec.txt и shared/parquet_batch_aggregates."""

from __future__ import annotations

import hashlib
import logging
import math
import statistics
from datetime import datetime, timedelta
from typing import Any, Mapping

from shared.parquet_batch_aggregates import UserAggregates, WindowTxn, row_to_window_txn

logger = logging.getLogger(__name__)
EPS = 1e-9


def _percentile_95(values: list[float]) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    n = len(xs)
    idx = int(math.floor(0.95 * (n - 1)))
    return float(xs[idx])

FEATURE_NAMES: list[str] = [
    "operation_amt",
    "log_1_plus_transactions_seen",
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
    "amount_diff_prev",
    "amount_ratio_prev",
    "trend_mean_last_3_to_10",
    "amount_percentile_rank",
    "std_time_deltas",
    "is_timezone_change",
    "is_new_device_tz_pair",
    "is_device_switch",
    "is_mcc_switch",
    "session_duration",
    "session_mean_amount",
    "amount_zscore_x_is_new_device",
    "amount_zscore_x_is_new_mcc",
    "is_new_device_x_is_new_timezone",
    "device_freq",
    "delta_1",
    "delta_2",
    "delta_3",
    "acceleration_delta_1_over_2",
    "std_delta_last_k",
    "cv_delta_last_k",
    "time_since_last_device_change",
    "time_since_last_mcc_change",
]


def cat_to_float(s: Any) -> float:
    if s is None:
        return 0.0
    t = str(s).strip()
    if not t:
        return 0.0
    h = hashlib.md5(t.encode("utf-8")).hexdigest()
    return float(int(h, 16) % 1_000_000)


def _channel_sub(row: Mapping[str, Any]) -> Any:
    v = row.get("channel_indicator_sub_type")
    if v is None:
        v = row.get("channel_indicator_subtype")
    return v


def _empty_str(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    return False


def _flag_nonempty(v: Any) -> float:
    return 1.0 if not _empty_str(v) else 0.0


def _safe_div(num: float, den: float) -> float:
    d = den if abs(den) > EPS else EPS
    return num / d


def _compromised_binary(v: Any) -> float:
    if _empty_str(v):
        return 0.0
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(x):
        return 0.0
    return 1.0 if x == 1.0 else 0.0


def compute_features(
    agg: UserAggregates,
    row: Mapping[str, Any],
    tr_amount: float | None = None,
) -> dict[str, float]:
    """
    Фичи по окну до текущей строки (текущая строка ещё не в окне).
    tr_amount — число транзакций в окне; по умолчанию len(agg).
    Для tr_amount и log_1_plus_transactions_seen: если у agg.window_transaction_cap задан лимит,
    значения режутся по нему; если None (инференс без лимита) — используется полный размер окна.
    """
    window: list[WindowTxn] = list(agg._window)  # noqa: SLF001 — осознанный доступ к окну
    if tr_amount is None:
        tr_amount = float(len(window))

    amounts: list[float] = []
    for t in window:
        if t.amount is not None and math.isfinite(t.amount):
            amounts.append(t.amount)
    n_amt = len(amounts)
    if n_amt == 0:
        median = float("nan")
        p95 = float("nan")
        mean = float("nan")
        std = float("nan")
    else:
        median = float(statistics.median(amounts))
        p95 = _percentile_95(amounts)
        mean = float(statistics.mean(amounts))
        if n_amt >= 2:
            std = float(statistics.stdev(amounts))
        else:
            std = float("nan")

    amount_v = row.get("operaton_amt")
    try:
        amount = float(amount_v) if amount_v is not None else float("nan")
        if math.isnan(amount) or math.isinf(amount):
            amount = float("nan")
    except (TypeError, ValueError):
        amount = float("nan")

    dttm = row_to_window_txn(row).dttm
    hour_val = -1
    dow_val = -1
    if dttm is not None:
        hour_val = dttm.hour
        dow_val = dttm.weekday()

    recent: list[tuple[datetime, float]] = []
    if dttm is not None:
        t0 = dttm - timedelta(hours=24)
        for t in window:
            if t.dttm is not None and t.dttm >= t0 and t.amount is not None and math.isfinite(t.amount):
                recent.append((t.dttm, t.amount))

    def count_since(delta: timedelta) -> int:
        if dttm is None:
            return 0
        thr = dttm - delta
        return sum(1 for tt, _ in recent if tt >= thr)

    def sum_since(delta: timedelta) -> float:
        if dttm is None:
            return 0.0
        thr = dttm - delta
        return float(sum(a for tt, a in recent if tt >= thr))

    def max_since_24h() -> float:
        if not recent:
            return float("nan")
        return float(max(a for _, a in recent))

    last_txn: WindowTxn | None = window[-1] if window else None
    last_amount = last_txn.amount if last_txn is not None else None
    last_tz = last_txn.tz if last_txn is not None else None
    last_mcc = last_txn.mcc if last_txn is not None else None
    last_dkey = (last_txn.os_type, last_txn.dev_ver) if last_txn is not None else (None, None)

    mean_last_3 = float("nan")
    if n_amt >= 1:
        mean_last_3 = float(statistics.mean(amounts[-3:]))
    mean_last_10 = float("nan")
    if n_amt >= 1:
        mean_last_10 = float(statistics.mean(amounts[-10:]))

    last_event_dttm: datetime | None = None
    for t in reversed(window):
        if t.dttm is not None:
            last_event_dttm = t.dttm
            break

    seen_devices: dict[tuple[Any, Any], int] = {}
    seen_mcc: dict[Any, int] = {}
    seen_channels: dict[tuple[Any, Any], int] = {}
    seen_tz: dict[Any, int] = {}
    seen_device_tz: dict[tuple[Any, Any, Any], int] = {}
    seen_bl: dict[Any, int] = {}
    session_counts: dict[Any, int] = {}
    session_amount_sums: dict[Any, float] = {}
    session_first_dttm: dict[Any, datetime] = {}
    session_last_dttm: dict[Any, datetime] = {}
    for t in window:
        dkey = (t.os_type, t.dev_ver)
        ckey = (t.ch_type, t.ch_sub)
        if not _empty_str(t.os_type) or not _empty_str(t.dev_ver):
            seen_devices[dkey] = seen_devices.get(dkey, 0) + 1
        if t.mcc is not None and not (isinstance(t.mcc, str) and not str(t.mcc).strip()):
            seen_mcc[t.mcc] = seen_mcc.get(t.mcc, 0) + 1
        if not _empty_str(t.ch_type) or not _empty_str(t.ch_sub):
            seen_channels[ckey] = seen_channels.get(ckey, 0) + 1
        if not _empty_str(t.tz):
            seen_tz[t.tz] = seen_tz.get(t.tz, 0) + 1
        if (not _empty_str(t.os_type) or not _empty_str(t.dev_ver)) and not _empty_str(t.tz):
            dtz_key = (t.os_type, t.dev_ver, t.tz)
            seen_device_tz[dtz_key] = seen_device_tz.get(dtz_key, 0) + 1
        if not _empty_str(t.browser_language):
            seen_bl[t.browser_language] = seen_bl.get(t.browser_language, 0) + 1
        sid = t.session_id
        if sid is not None:
            session_counts[sid] = session_counts.get(sid, 0) + 1
            if t.amount is not None and math.isfinite(t.amount):
                session_amount_sums[sid] = session_amount_sums.get(sid, 0.0) + float(t.amount)
            if t.dttm is not None:
                prev_first = session_first_dttm.get(sid)
                prev_last = session_last_dttm.get(sid)
                if prev_first is None or t.dttm < prev_first:
                    session_first_dttm[sid] = t.dttm
                if prev_last is None or t.dttm > prev_last:
                    session_last_dttm[sid] = t.dttm

    os_c = row.get("operating_system_type")
    dev_c = row.get("device_system_version")
    mcc_c = row.get("mcc_code")
    ch_t = row.get("channel_indicator_type")
    ch_s = _channel_sub(row)
    tz_c = row.get("timezone")
    bl_c = row.get("browser_language")
    sid_c = row.get("session_id")

    dkey_c = (os_c, dev_c)
    ckey_c = (ch_t, ch_s)
    dtz_key_c = (os_c, dev_c, tz_c)
    device_count_i = int(seen_devices.get(dkey_c, 0)) if (not _empty_str(os_c) or not _empty_str(dev_c)) else 0
    mcc_count_i = int(seen_mcc.get(mcc_c, 0)) if not _empty_str(mcc_c) else 0
    channel_count_i = int(seen_channels.get(ckey_c, 0)) if (not _empty_str(ch_t) or not _empty_str(ch_s)) else 0
    timezone_count_i = int(seen_tz.get(tz_c, 0)) if not _empty_str(tz_c) else 0
    bl_count_i = int(seen_bl.get(bl_c, 0)) if not _empty_str(bl_c) else 0

    is_new_device = 1.0 if (not _empty_str(os_c) or not _empty_str(dev_c)) and device_count_i == 0 else 0.0
    is_new_mcc = 1.0 if not _empty_str(mcc_c) and mcc_count_i == 0 else 0.0
    is_new_channel = 1.0 if (not _empty_str(ch_t) or not _empty_str(ch_s)) and channel_count_i == 0 else 0.0
    is_new_timezone = 1.0 if not _empty_str(tz_c) and timezone_count_i == 0 else 0.0
    is_new_browser_language = 1.0 if not _empty_str(bl_c) and bl_count_i == 0 else 0.0
    is_new_device_tz_pair = (
        1.0
        if (not _empty_str(os_c) or not _empty_str(dev_c)) and not _empty_str(tz_c) and seen_device_tz.get(dtz_key_c, 0) == 0
        else 0.0
    )

    n_sess = session_counts.get(sid_c, 0) if sid_c is not None else 0
    if sid_c is not None and sid_c in session_counts:
        sess_cnt = float(session_counts[sid_c])
        sess_sum = float(session_amount_sums.get(sid_c, 0.0))
        session_mean_amount = sess_sum / max(sess_cnt, 1.0)
    else:
        session_mean_amount = float("nan")

    sf = session_first_dttm.get(sid_c) if sid_c is not None else None
    sl = session_last_dttm.get(sid_c) if sid_c is not None else None
    if sf is not None and sl is not None:
        session_duration = float((sl - sf).total_seconds())
    else:
        session_duration = float("nan")

    if not math.isnan(median) and median != 0.0:
        amount_to_median = amount / median
    else:
        amount_to_median = float("nan")

    if not math.isnan(std) and std > 0.0 and not math.isnan(amount):
        amount_zscore = (amount - mean) / std
    else:
        amount_zscore = float("nan")

    is_amount_high = 0.0
    if not math.isnan(p95) and not math.isnan(amount) and amount > p95:
        is_amount_high = 1.0

    max24 = max_since_24h()
    if math.isnan(max24):
        max_amount_last_24h_feat = 0.0
    else:
        max_amount_last_24h_feat = max24

    if dttm is not None and last_event_dttm is not None:
        tsp = (dttm - last_event_dttm).total_seconds()
    else:
        tsp = float("nan")
    if math.isnan(tsp):
        time_since_prev = -1.0
    else:
        time_since_prev = float(tsp)

    if last_amount is not None and math.isfinite(last_amount) and not math.isnan(amount):
        amount_diff_prev = amount - last_amount
        amount_ratio_prev = _safe_div(amount, last_amount)
    else:
        amount_diff_prev = float("nan")
        amount_ratio_prev = float("nan")

    trend_mean_last_3_to_10 = (
        _safe_div(mean_last_3, mean_last_10)
        if (not math.isnan(mean_last_3) and not math.isnan(mean_last_10))
        else float("nan")
    )
    if n_amt > 0 and not math.isnan(amount):
        amount_percentile_rank = float(sum(1 for a in amounts if a <= amount)) / float(n_amt)
    else:
        amount_percentile_rank = float("nan")

    dttms = [t.dttm for t in window if t.dttm is not None]
    deltas: list[float] = []
    for i in range(1, len(dttms)):
        deltas.append(float((dttms[i] - dttms[i - 1]).total_seconds()))
    if len(deltas) >= 2:
        std_time_deltas = float(statistics.stdev(deltas))
    else:
        std_time_deltas = float("nan")

    is_timezone_change = 1.0 if (not _empty_str(tz_c) and not _empty_str(last_tz) and tz_c != last_tz) else 0.0
    is_device_switch = 1.0 if (not _empty_str(os_c) or not _empty_str(dev_c)) and dkey_c != last_dkey else 0.0
    is_mcc_switch = 1.0 if not _empty_str(mcc_c) and not _empty_str(last_mcc) and mcc_c != last_mcc else 0.0

    delta_1 = deltas[-1] if len(deltas) >= 1 else float("nan")
    delta_2 = deltas[-2] if len(deltas) >= 2 else float("nan")
    delta_3 = deltas[-3] if len(deltas) >= 3 else float("nan")
    acceleration = _safe_div(delta_1, delta_2) if (not math.isnan(delta_1) and not math.isnan(delta_2)) else float("nan")

    last_k = deltas[-10:]
    if len(last_k) >= 2:
        std_delta_last_k = float(statistics.stdev(last_k))
        mean_delta_last_k = float(statistics.mean(last_k))
        cv_delta_last_k = _safe_div(std_delta_last_k, mean_delta_last_k)
    else:
        std_delta_last_k = float("nan")
        cv_delta_last_k = float("nan")

    time_since_last_device_change = float("nan")
    time_since_last_mcc_change = float("nan")
    if dttm is not None:
        for t in reversed(window):
            td = t.dttm
            if td is None:
                continue
            if math.isnan(time_since_last_device_change):
                if (not _empty_str(os_c) or not _empty_str(dev_c)) and (t.os_type, t.dev_ver) != dkey_c:
                    time_since_last_device_change = float((dttm - td).total_seconds())
            if math.isnan(time_since_last_mcc_change):
                if not _empty_str(mcc_c) and t.mcc != mcc_c:
                    time_since_last_mcc_change = float((dttm - td).total_seconds())
            if not math.isnan(time_since_last_device_change) and not math.isnan(time_since_last_mcc_change):
                break

    descr = row.get("event_descr")
    if descr is None:
        descr = row.get("event_desc")

    hour_f = float(hour_val) if dttm is not None else float("nan")
    dow_f = float(dow_val) if dttm is not None else float("nan")

    if dttm is not None:
        is_night = 1.0 if (hour_val >= 22 or hour_val < 6) else 0.0
        is_weekend = 1.0 if dow_val >= 5 else 0.0
    else:
        is_night = float("nan")
        is_weekend = float("nan")

    tx_10m = float(count_since(timedelta(minutes=10)))
    tx_1h = float(count_since(timedelta(hours=1)))
    tx_24h = float(count_since(timedelta(hours=24)))
    sum_1h = sum_since(timedelta(hours=1))
    sum_24h = sum_since(timedelta(hours=24))

    cap = agg.window_transaction_cap
    if cap is None:
        tr_cap = float(tr_amount)
    else:
        tr_cap = float(min(tr_amount, float(cap)))
    log_1_plus_transactions_seen = math.log1p(max(0.0, tr_cap))

    return {
        "operation_amt": amount,
        "log_1_plus_transactions_seen": log_1_plus_transactions_seen,
        "amount_to_median": amount_to_median,
        "amount_zscore": amount_zscore,
        "is_amount_high": is_amount_high,
        "transactions_last_1h": tx_1h,
        "transactions_last_24h": tx_24h,
        "sum_amount_last_1h": sum_1h,
        "max_amount_last_24h": max_amount_last_24h_feat,
        "is_new_device": is_new_device,
        "is_new_mcc": is_new_mcc,
        "is_new_channel": is_new_channel,
        "is_new_timezone": is_new_timezone,
        "is_compromised_device": _compromised_binary(row.get("compromised")),
        "web_rdp_connection": _flag_nonempty(row.get("web_rdp_connection")),
        "phone_voip_call_state": _flag_nonempty(row.get("phone_voip_call_state")),
        "hour": hour_f,
        "day_of_week": dow_f,
        "is_night_transaction": is_night,
        "is_weekend": is_weekend,
        "transactions_last_10m": tx_10m,
        "sum_amount_last_24h": sum_24h,
        "time_since_prev_transaction": time_since_prev,
        "is_new_browser_language": is_new_browser_language,
        "transactions_in_session": float(n_sess + 1),
        "timezone_missing": 1.0 if _empty_str(tz_c) else 0.0,
        "tr_amount": tr_cap,
        "event_descr": cat_to_float(descr),
        "mcc_code": cat_to_float(mcc_c),
        "amount_diff_prev": amount_diff_prev,
        "amount_ratio_prev": amount_ratio_prev,
        "trend_mean_last_3_to_10": trend_mean_last_3_to_10,
        "amount_percentile_rank": amount_percentile_rank,
        "std_time_deltas": std_time_deltas,
        "is_timezone_change": is_timezone_change,
        "is_new_device_tz_pair": is_new_device_tz_pair,
        "is_device_switch": is_device_switch,
        "is_mcc_switch": is_mcc_switch,
        "session_duration": session_duration,
        "session_mean_amount": session_mean_amount,
        "amount_zscore_x_is_new_device": amount_zscore * is_new_device if not math.isnan(amount_zscore) else float("nan"),
        "amount_zscore_x_is_new_mcc": amount_zscore * is_new_mcc if not math.isnan(amount_zscore) else float("nan"),
        "is_new_device_x_is_new_timezone": is_new_device * is_new_timezone,
        "device_freq": _safe_div(float(device_count_i), float(len(window))) if len(window) > 0 else float("nan"),
        "delta_1": delta_1,
        "delta_2": delta_2,
        "delta_3": delta_3,
        "acceleration_delta_1_over_2": acceleration,
        "std_delta_last_k": std_delta_last_k,
        "cv_delta_last_k": cv_delta_last_k,
        "time_since_last_device_change": time_since_last_device_change,
        "time_since_last_mcc_change": time_since_last_mcc_change,
    }
