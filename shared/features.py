"""Имена и расчёт фич — по dataset_cpp_module_spec.txt и shared/parquet_batch_aggregates."""

from __future__ import annotations

import hashlib
import logging
import math
import statistics
from datetime import datetime, timedelta
from typing import Any, Mapping

from shared.parquet_batch_aggregates import UserAggregates, WindowTxn, effective_window_size, row_to_window_txn

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
    "amount_to_median",
    "amount_zscore",
    "is_amount_high",
    "transactions_last_1h",
    "transactions_last_24h",
    "sum_amount_last_1h",
    "max_amount_last_24h",
    "device_freq",
    "device_count",
    "time_since_last_device",
    "mcc_freq",
    "mcc_count",
    "time_since_last_mcc",
    "channel_freq",
    "channel_count",
    "time_since_last_channel",
    "timezone_freq",
    "browser_language_freq",
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
    "transactions_in_session",
    "timezone_missing",
    "tr_amount",
    "event_descr",
    "mcc_code",
    "event_descr_freq",
    "event_type_nm_freq",
    "log_tr_amount",
    "transactions_last_10m_norm",
    "transactions_last_1h_norm",
    "transactions_last_24h_norm",
    "sum_amount_last_1h_norm",
    "sum_amount_last_24h_norm",
    "transactions_last_10m_to_1h",
    "transactions_last_1h_to_24h",
    "sum_1h_to_24h",
    "time_since_2nd_prev_transaction",
    "mean_time_between_tx",
    "std_time_between_tx",
    "amount_to_last_amount",
    "amount_to_max_24h",
    "time_since_prev_to_mean_gap",
    "device_freq_alt",
    "mcc_freq_alt",
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

    last_event_dttm: datetime | None = None
    for t in reversed(window):
        if t.dttm is not None:
            last_event_dttm = t.dttm
            break

    seen_devices: dict[tuple[Any, Any], int] = {}
    seen_mcc: dict[Any, int] = {}
    seen_channels: dict[tuple[Any, Any], int] = {}
    seen_tz: dict[Any, int] = {}
    seen_bl: dict[Any, int] = {}
    seen_descr: dict[Any, int] = {}
    seen_type: dict[Any, int] = {}
    last_seen_dev: dict[tuple[Any, Any], datetime] = {}
    last_seen_mcc: dict[Any, datetime] = {}
    last_seen_channel: dict[tuple[Any, Any], datetime] = {}
    session_counts: dict[Any, int] = {}
    gap_seconds: list[float] = []
    prev_gap_t: datetime | None = None
    last_amount: float = float("nan")
    for t in window:
        dkey = (t.os_type, t.dev_ver)
        ckey = (t.ch_type, t.ch_sub)
        if not _empty_str(t.os_type) or not _empty_str(t.dev_ver):
            seen_devices[dkey] = seen_devices.get(dkey, 0) + 1
        if t.mcc is not None and not (isinstance(t.mcc, str) and not str(t.mcc).strip()):
            seen_mcc[t.mcc] = seen_mcc.get(t.mcc, 0) + 1
        if t.ch_type is not None or t.ch_sub is not None:
            seen_channels[ckey] = seen_channels.get(ckey, 0) + 1
        if not _empty_str(t.tz):
            seen_tz[t.tz] = seen_tz.get(t.tz, 0) + 1
        if not _empty_str(t.browser_language):
            seen_bl[t.browser_language] = seen_bl.get(t.browser_language, 0) + 1
        if not _empty_str(t.event_descr):
            seen_descr[t.event_descr] = seen_descr.get(t.event_descr, 0) + 1
        if not _empty_str(t.event_type_nm):
            seen_type[t.event_type_nm] = seen_type.get(t.event_type_nm, 0) + 1
        if t.dttm is not None:
            if not _empty_str(t.os_type) or not _empty_str(t.dev_ver):
                last_seen_dev[dkey] = t.dttm
            if t.mcc is not None and not (isinstance(t.mcc, str) and not str(t.mcc).strip()):
                last_seen_mcc[t.mcc] = t.dttm
            if t.ch_type is not None or t.ch_sub is not None:
                last_seen_channel[ckey] = t.dttm
            if prev_gap_t is not None:
                gs = (t.dttm - prev_gap_t).total_seconds()
                if math.isfinite(gs) and gs >= 0.0:
                    gap_seconds.append(float(gs))
            prev_gap_t = t.dttm
        if t.amount is not None and math.isfinite(t.amount):
            last_amount = t.amount
        sid = t.session_id
        if sid is not None:
            session_counts[sid] = session_counts.get(sid, 0) + 1

    os_c = row.get("operating_system_type")
    dev_c = row.get("device_system_version")
    mcc_c = row.get("mcc_code")
    ch_t = row.get("channel_indicator_type")
    ch_s = _channel_sub(row)
    tz_c = row.get("timezone")
    bl_c = row.get("browser_language")
    sid_c = row.get("session_id")

    tx_total = float(len(window))
    dkey_c = (os_c, dev_c)
    ckey_c = (ch_t, ch_s)
    device_count = float(seen_devices.get(dkey_c, 0)) if (not _empty_str(os_c) or not _empty_str(dev_c)) else 0.0
    mcc_count = float(seen_mcc.get(mcc_c, 0)) if not _empty_str(mcc_c) else 0.0
    channel_count = float(seen_channels.get(ckey_c, 0)) if (ch_t is not None or ch_s is not None) else 0.0
    timezone_count = float(seen_tz.get(tz_c, 0)) if not _empty_str(tz_c) else 0.0
    bl_count = float(seen_bl.get(bl_c, 0)) if not _empty_str(bl_c) else 0.0

    device_freq = _safe_div(device_count, tx_total) if tx_total > 0 else 0.0
    mcc_freq = _safe_div(mcc_count, tx_total) if tx_total > 0 else 0.0
    channel_freq = _safe_div(channel_count, tx_total) if tx_total > 0 else 0.0
    timezone_freq = _safe_div(timezone_count, tx_total) if tx_total > 0 else 0.0
    browser_language_freq = _safe_div(bl_count, tx_total) if tx_total > 0 else 0.0

    n_sess = session_counts.get(sid_c, 0) if sid_c is not None else 0

    def _since_last(last_map: Mapping[Any, datetime], key: Any, key_ok: bool) -> float:
        if dttm is None or not key_ok:
            return -1.0
        dt = last_map.get(key)
        if dt is None:
            return -1.0
        return float((dttm - dt).total_seconds())

    time_since_last_device = _since_last(last_seen_dev, dkey_c, (not _empty_str(os_c) or not _empty_str(dev_c)))
    time_since_last_mcc = _since_last(last_seen_mcc, mcc_c, not _empty_str(mcc_c))
    time_since_last_channel = _since_last(last_seen_channel, ckey_c, (ch_t is not None or ch_s is not None))

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

    second_last_event_dttm: datetime | None = None
    seen_t = 0
    for t in reversed(window):
        if t.dttm is not None:
            seen_t += 1
            if seen_t == 2:
                second_last_event_dttm = t.dttm
                break
    if dttm is not None and second_last_event_dttm is not None:
        time_since_2nd_prev = float((dttm - second_last_event_dttm).total_seconds())
    else:
        time_since_2nd_prev = -1.0

    descr = row.get("event_descr")
    if descr is None:
        descr = row.get("event_desc")
    etype = row.get("event_type_nm")

    hour_f = float(hour_val) if dttm is not None else float("nan")
    dow_f = float(dow_val) if dttm is not None else float("nan")

    is_night = 1.0 if (hour_val >= 22 or hour_val < 6) else 0.0
    is_weekend = 1.0 if dow_val >= 5 else 0.0

    tx_10m = float(count_since(timedelta(minutes=10)))
    tx_1h = float(count_since(timedelta(hours=1)))
    tx_24h = float(count_since(timedelta(hours=24)))
    sum_1h = sum_since(timedelta(hours=1))
    sum_24h = sum_since(timedelta(hours=24))
    descr_freq = _safe_div(float(seen_descr.get(descr, 0)), tx_total) if (tx_total > 0 and not _empty_str(descr)) else 0.0
    etype_freq = _safe_div(float(seen_type.get(etype, 0)), tx_total) if (tx_total > 0 and not _empty_str(etype)) else 0.0

    log_tr_amount = math.log1p(max(0.0, float(tr_amount)))
    log_den = log_tr_amount if log_tr_amount > EPS else EPS
    tx_10m_norm = _safe_div(tx_10m, log_den)
    tx_1h_norm = _safe_div(tx_1h, log_den)
    tx_24h_norm = _safe_div(tx_24h, log_den)
    sum_1h_norm = _safe_div(sum_1h, log_den)
    sum_24h_norm = _safe_div(sum_24h, log_den)

    tx_10m_to_1h = tx_10m / (tx_1h + 1.0)
    tx_1h_to_24h = tx_1h / (tx_24h + 1.0)
    sum_1h_to_24h = sum_1h / (sum_24h + 1.0)

    if gap_seconds:
        mean_gap = float(statistics.mean(gap_seconds))
        std_gap = float(statistics.stdev(gap_seconds)) if len(gap_seconds) >= 2 else 0.0
    else:
        mean_gap = -1.0
        std_gap = -1.0

    amount_to_last_amount = _safe_div(amount, last_amount + EPS) if (math.isfinite(amount) and math.isfinite(last_amount)) else float("nan")
    amount_to_max_24h = _safe_div(amount, max_amount_last_24h_feat + EPS) if math.isfinite(amount) else float("nan")
    time_since_prev_to_mean_gap = _safe_div(time_since_prev, mean_gap + EPS) if (time_since_prev >= 0.0 and mean_gap >= 0.0) else -1.0
    device_freq_alt = _safe_div(device_count, amount + EPS) if math.isfinite(amount) else _safe_div(device_count, EPS)
    mcc_freq_alt = _safe_div(mcc_count, amount + EPS) if math.isfinite(amount) else _safe_div(mcc_count, EPS)

    return {
        "operation_amt": amount,
        "amount_to_median": amount_to_median,
        "amount_zscore": amount_zscore,
        "is_amount_high": is_amount_high,
        "transactions_last_1h": tx_1h,
        "transactions_last_24h": tx_24h,
        "sum_amount_last_1h": sum_1h,
        "max_amount_last_24h": max_amount_last_24h_feat,
        "device_freq": device_freq,
        "device_count": device_count,
        "time_since_last_device": time_since_last_device,
        "mcc_freq": mcc_freq,
        "mcc_count": mcc_count,
        "time_since_last_mcc": time_since_last_mcc,
        "channel_freq": channel_freq,
        "channel_count": channel_count,
        "time_since_last_channel": time_since_last_channel,
        "timezone_freq": timezone_freq,
        "browser_language_freq": browser_language_freq,
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
        "transactions_in_session": float(n_sess + 1),
        "timezone_missing": 1.0 if _empty_str(tz_c) else 0.0,
        "tr_amount": float(min(tr_amount, float(effective_window_size()))),
        "event_descr": cat_to_float(descr),
        "mcc_code": cat_to_float(mcc_c),
        "event_descr_freq": descr_freq,
        "event_type_nm_freq": etype_freq,
        "log_tr_amount": log_tr_amount,
        "transactions_last_10m_norm": tx_10m_norm,
        "transactions_last_1h_norm": tx_1h_norm,
        "transactions_last_24h_norm": tx_24h_norm,
        "sum_amount_last_1h_norm": sum_1h_norm,
        "sum_amount_last_24h_norm": sum_24h_norm,
        "transactions_last_10m_to_1h": tx_10m_to_1h,
        "transactions_last_1h_to_24h": tx_1h_to_24h,
        "sum_1h_to_24h": sum_1h_to_24h,
        "time_since_2nd_prev_transaction": time_since_2nd_prev,
        "mean_time_between_tx": mean_gap,
        "std_time_between_tx": std_gap,
        "amount_to_last_amount": amount_to_last_amount,
        "amount_to_max_24h": amount_to_max_24h,
        "time_since_prev_to_mean_gap": time_since_prev_to_mean_gap,
        "device_freq_alt": device_freq_alt,
        "mcc_freq_alt": mcc_freq_alt,
    }
