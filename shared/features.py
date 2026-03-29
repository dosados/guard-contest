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
]


def cat_to_float(s: Any) -> float:
    if s is None:
        return 0.0
    t = str(s).strip()
    if not t:
        return 0.0
    h = hashlib.md5(t.encode("utf-8")).hexdigest()
    return float(int(h, 16) % 1_000_000)


def _parse_event_type_nm(v: Any) -> float:
    """Код/число из колонки event_type_nm как float; иначе NaN (без хеширования)."""
    if v is None:
        return float("nan")
    if isinstance(v, bool):
        return float("nan")
    if isinstance(v, (int, float)):
        x = float(v)
        return x if math.isfinite(x) else float("nan")
    s = str(v).strip()
    if not s:
        return float("nan")
    try:
        x = float(s)
        return x if math.isfinite(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


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


def _parse_battery_level(v: Any) -> float:
    if v is None:
        return float("nan")
    if isinstance(v, bool):
        return float("nan")
    if isinstance(v, (int, float)):
        x = float(v)
        return x if math.isfinite(x) else float("nan")
    s = str(v).strip()
    if not s:
        return float("nan")
    try:
        x = float(s)
        return x if math.isfinite(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _battery_very_low_flag(level: float) -> float:
    if not math.isfinite(level):
        return 0.0
    if 0.0 <= level <= 1.0 and level < 0.2:
        return 1.0
    if 1.0 < level <= 100.0 and level < 20.0:
        return 1.0
    return 0.0


def _developer_tools_flag(v: Any) -> float:
    if _empty_str(v):
        return 0.0
    if isinstance(v, (int, float)):
        x = float(v)
        return 1.0 if x != 0.0 and math.isfinite(x) else 0.0
    s = str(v).strip().lower()
    if s in ("false", "no", "off", "0", ""):
        return 0.0
    return 1.0


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

    amount_diff_prev = float("nan")
    amount_ratio_prev = float("nan")
    if last_amount is not None and math.isfinite(float(last_amount)) and math.isfinite(amount):
        la = float(last_amount)
        amount_diff_prev = amount - la
        amount_ratio_prev = _safe_div(amount, la)

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

    amount_change_sign = float("nan")
    if math.isfinite(amount_diff_prev):
        if abs(amount_diff_prev) < EPS:
            amount_change_sign = 0.0
        elif amount_diff_prev > 0.0:
            amount_change_sign = 1.0
        else:
            amount_change_sign = -1.0

    amount_inc_streak = amount_dec_streak = 0
    if len(amounts) >= 2:
        j = len(amounts) - 1
        while j >= 1 and amounts[j] > amounts[j - 1]:
            amount_inc_streak += 1
            j -= 1
        j = len(amounts) - 1
        while j >= 1 and amounts[j] < amounts[j - 1]:
            amount_dec_streak += 1
            j -= 1

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
    else:
        std_delta_last_k = float("nan")

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
    event_type_nm = row.get("event_type_nm")

    hour_f = float(hour_val) if dttm is not None else float("nan")
    dow_f = float(dow_val) if dttm is not None else float("nan")

    if dttm is not None:
        is_night = 1.0 if (hour_val >= 22 or hour_val < 6) else 0.0
        is_weekend = 1.0 if dow_val >= 5 else 0.0
    else:
        is_night = float("nan")
        is_weekend = float("nan")

    tx_10m = float(count_since(timedelta(minutes=10)))
    tx_5m = float(count_since(timedelta(minutes=5)))
    tx_1h = float(count_since(timedelta(hours=1)))
    tx_24h = float(count_since(timedelta(hours=24)))
    sum_1h = sum_since(timedelta(hours=1))
    sum_24h = sum_since(timedelta(hours=24))
    sum_amount_last_10m = sum_since(timedelta(minutes=10))

    # Frequencies and pair novelty (depend on exact strings in window).
    descr_c = "" if _empty_str(descr) else str(descr).strip()
    event_type_nm_cur = _parse_event_type_nm(event_type_nm)
    mcc_c_s = "" if _empty_str(mcc_c) else str(mcc_c).strip()
    if dttm is not None:
        thr_1h = dttm - timedelta(hours=1)
        thr_6h = dttm - timedelta(hours=6)
        thr_24h = dttm - timedelta(hours=24)
        thr_5d = dttm - timedelta(days=5)
    else:
        thr_1h = thr_6h = thr_24h = thr_5d = None

    def _freq_last(thr: datetime | None, *, kind: str) -> float:
        if thr is None:
            return 0.0
        if kind == "descr":
            if not descr_c:
                return 0.0
            return float(
                sum(
                    1
                    for t in window
                    if t.dttm is not None and t.dttm >= thr and ("" if _empty_str(t.event_descr) else str(t.event_descr).strip()) == descr_c
                )
            )
        if kind == "event_type_nm":
            if not math.isfinite(event_type_nm_cur):
                return 0.0
            return float(
                sum(
                    1
                    for t in window
                    if t.dttm is not None
                    and t.dttm >= thr
                    and _parse_event_type_nm(t.event_type_nm) == event_type_nm_cur
                )
            )
        if kind == "mcc":
            if not mcc_c_s:
                return 0.0
            return float(
                sum(
                    1
                    for t in window
                    if t.dttm is not None and t.dttm >= thr and ("" if _empty_str(t.mcc) else str(t.mcc).strip()) == mcc_c_s
                )
            )
        raise ValueError("unknown kind")

    event_descr_freq_last_1h = _freq_last(thr_1h, kind="descr")
    event_descr_freq_last_6h = _freq_last(thr_6h, kind="descr")
    event_descr_freq_last_24h = _freq_last(thr_24h, kind="descr")
    event_type_nm_freq_last_1h = _freq_last(thr_1h, kind="event_type_nm")
    event_type_nm_freq_last_6h = _freq_last(thr_6h, kind="event_type_nm")
    event_type_nm_freq_last_24h = _freq_last(thr_24h, kind="event_type_nm")
    mcc_freq_last_1h = _freq_last(thr_1h, kind="mcc")
    mcc_freq_last_6h = _freq_last(thr_6h, kind="mcc")
    mcc_freq_last_24h = _freq_last(thr_24h, kind="mcc")

    if descr_c and mcc_c_s:
        seen_pair = any(
            ("" if _empty_str(t.mcc) else str(t.mcc).strip()) == mcc_c_s
            and ("" if _empty_str(t.event_descr) else str(t.event_descr).strip()) == descr_c
            for t in window
        )
        mcc_event_descr_pair_new = 0.0 if seen_pair else 1.0
    else:
        mcc_event_descr_pair_new = 0.0

    if thr_24h is None or math.isnan(p95):
        high_amount_ratio_last_24h = 0.0
    else:
        recent_24h_amounts = [
            t.amount
            for t in window
            if t.dttm is not None and t.dttm >= thr_24h and t.amount is not None and math.isfinite(t.amount)
        ]
        if not recent_24h_amounts:
            high_amount_ratio_last_24h = 0.0
        else:
            high_amount_ratio_last_24h = float(sum(1 for a in recent_24h_amounts if a > p95)) / float(
                len(recent_24h_amounts)
            )

    mcc_amount_std_same_5d = float("nan")
    if thr_5d is None or not mcc_c_s or math.isnan(amount):
        amount_relative_to_mcc_median_5_days = float("nan")
    else:
        mcc_amounts_5d = [
            t.amount
            for t in window
            if t.dttm is not None
            and t.dttm >= thr_5d
            and t.amount is not None
            and math.isfinite(t.amount)
            and ("" if _empty_str(t.mcc) else str(t.mcc).strip()) == mcc_c_s
        ]
        if not mcc_amounts_5d:
            amount_relative_to_mcc_median_5_days = float("nan")
        else:
            mcc_med_5d = float(statistics.median(mcc_amounts_5d))
            amount_relative_to_mcc_median_5_days = amount / mcc_med_5d if mcc_med_5d != 0.0 else float("nan")
            if len(mcc_amounts_5d) >= 2:
                mcc_amount_std_same_5d = float(statistics.stdev(mcc_amounts_5d))

    n_win = len(window)
    amount_ratio_to_window_median = (
        amount / median if (not math.isnan(median) and median != 0.0 and math.isfinite(amount)) else float("nan")
    )
    amount_iqr_normalized = float("nan")
    if n_amt >= 2 and math.isfinite(amount):
        s_iqr = sorted(amounts)
        i25 = (n_amt - 1) // 4
        i75 = 3 * (n_amt - 1) // 4
        q25, q75 = s_iqr[i25], s_iqr[i75]
        iqr = q75 - q25
        denom = abs(iqr) if abs(iqr) > EPS else EPS
        amount_iqr_normalized = (amount - q25) / denom
    amount_cv_in_window = (
        (std / mean) if (not math.isnan(mean) and abs(mean) > EPS and not math.isnan(std)) else float("nan")
    )

    umcc: set[str] = set()
    udev: set[tuple[Any, Any]] = set()
    uch: set[tuple[Any, Any]] = set()
    utz: set[str] = set()
    ubl: set[str] = set()
    for t in window:
        if not _empty_str(t.mcc):
            umcc.add(str(t.mcc).strip())
        if not _empty_str(t.os_type) or not _empty_str(t.dev_ver):
            udev.add((t.os_type, t.dev_ver))
        if not _empty_str(t.ch_type) or not _empty_str(t.ch_sub):
            uch.add((t.ch_type, t.ch_sub))
        if not _empty_str(t.tz):
            utz.add(str(t.tz).strip())
        if not _empty_str(t.browser_language):
            ubl.add(str(t.browser_language).strip())

    Lsw = min(20, n_win)
    mcc_switch_cnt = dev_switch_cnt = ch_switch_cnt = 0
    if Lsw >= 2:
        tail = window[-Lsw:]
        for i in range(Lsw - 1):
            a, b = tail[i], tail[i + 1]
            if ("" if _empty_str(a.mcc) else str(a.mcc).strip()) != ("" if _empty_str(b.mcc) else str(b.mcc).strip()):
                mcc_switch_cnt += 1
            if (a.os_type, a.dev_ver) != (b.os_type, b.dev_ver):
                dev_switch_cnt += 1
            if (a.ch_type, a.ch_sub) != (b.ch_type, b.ch_sub):
                ch_switch_cnt += 1

    night_share_24h = 0.0
    distinct_hours_24h = 0.0
    if thr_24h is not None:
        hrs24: set[int] = set()
        cnt24 = night24 = 0
        for t in window:
            if t.dttm is None or t.dttm < thr_24h:
                continue
            cnt24 += 1
            h = t.dttm.hour
            hrs24.add(h)
            if h >= 22 or h < 6:
                night24 += 1
        if cnt24 > 0:
            night_share_24h = float(night24) / float(cnt24)
        distinct_hours_24h = float(len(hrs24)) / 24.0

    amount_ratio_to_min_amount_24h = float("nan")
    if thr_24h is not None and math.isfinite(amount):
        mins: list[float] = []
        for t in window:
            if t.dttm is None or t.dttm < thr_24h:
                continue
            if t.amount is not None and math.isfinite(t.amount):
                mins.append(t.amount)
        if mins:
            amin = min(mins)
            if amin > EPS:
                amount_ratio_to_min_amount_24h = amount / amin

    compromised_hist = 0
    secs_since_compromised = float("nan")
    if dttm is not None:
        compromised_hist = sum(1 for t in window if _compromised_binary(t.compromised) == 1.0)
        for t in reversed(window):
            if _compromised_binary(t.compromised) == 1.0 and t.dttm is not None:
                secs_since_compromised = float((dttm - t.dttm).total_seconds())
                break

    web_rdp_cnt_24h = voip_cnt_24h = 0
    if thr_24h is not None:
        for t in window:
            if t.dttm is None or t.dttm < thr_24h:
                continue
            if not _empty_str(t.web_rdp):
                web_rdp_cnt_24h += 1
            if not _empty_str(t.voip):
                voip_cnt_24h += 1

    channel_rel_freq = (
        _safe_div(float(channel_count_i), float(n_win)) if n_win > 0 else float("nan")
    )
    tz_rel_freq = _safe_div(float(timezone_count_i), float(n_win)) if n_win > 0 else float("nan")
    bl_rel_freq = _safe_div(float(bl_count_i), float(n_win)) if n_win > 0 else float("nan")

    event_type_nm_share_suffix = 0.0
    if n_win > 0 and math.isfinite(event_type_nm_cur):
        event_type_nm_share_suffix = float(
            sum(1 for t in window if _parse_event_type_nm(t.event_type_nm) == event_type_nm_cur)
        ) / float(n_win)

    mcc_streak = 0.0
    if mcc_c_s:
        st = 0
        for t in reversed(window):
            if ("" if _empty_str(t.mcc) else str(t.mcc).strip()) == mcc_c_s:
                st += 1
            else:
                break
        mcc_streak = float(st)

    mean_gap_last_5 = float("nan")
    if deltas:
        n5 = min(5, len(deltas))
        mean_gap_last_5 = float(sum(deltas[-n5:]) / n5)

    dttms_sorted = [t.dttm for t in window if t.dttm is not None]
    suffix_span_log1p = float("nan")
    tx_per_span_hour = float("nan")
    if dttms_sorted and n_win > 0:
        span_sec = (max(dttms_sorted) - min(dttms_sorted)).total_seconds()
        hours = span_sec / 3600.0
        if hours < EPS:
            hours = EPS
        suffix_span_log1p = math.log1p(max(0.0, span_sec / 3600.0))
        tx_per_span_hour = float(n_win) / hours

    weekend_share_7d = 0.0
    if dttm is not None:
        thr7 = dttm - timedelta(days=7)
        c7 = wk7 = 0
        for t in window:
            if t.dttm is None or t.dttm < thr7:
                continue
            c7 += 1
            if t.dttm.weekday() >= 5:
                wk7 += 1
        if c7 > 0:
            weekend_share_7d = float(wk7) / float(c7)

    descr_div_24h = 0.0
    if thr_24h is not None:
        dset: set[str] = set()
        cntd = 0
        for t in window:
            if t.dttm is None or t.dttm < thr_24h:
                continue
            cntd += 1
            ed = "" if _empty_str(t.event_descr) else str(t.event_descr).strip()
            if ed:
                dset.add(ed)
        descr_div_24h = float(len(dset)) / float(max(1, cntd))

    is_new_timezone = 1.0 if (not _empty_str(tz_c) and timezone_count_i == 0) else 0.0
    is_timezone_change = (
        1.0
        if (
            not _empty_str(tz_c)
            and last_tz is not None
            and not _empty_str(last_tz)
            and str(tz_c).strip() != str(last_tz).strip()
        )
        else 0.0
    )

    battery_level_v = _parse_battery_level(row.get("battery"))
    battery_very_low_f = _battery_very_low_flag(battery_level_v)
    developer_tools_f = _developer_tools_flag(row.get("developer_tools"))
    acc_s = "" if _empty_str(row.get("accept_language")) else str(row.get("accept_language")).strip()
    bl_cmp = "" if _empty_str(bl_c) else str(bl_c).strip()
    accept_lang_browser_mismatch = 1.0 if (acc_s and bl_cmp and acc_s != bl_cmp) else 0.0
    battery_very_low_and_night = (
        1.0
        if (battery_very_low_f == 1.0 and not math.isnan(is_night) and is_night == 1.0)
        else 0.0
    )

    is_new_session_id = 1.0 if (sid_c is not None and not _empty_str(sid_c) and n_sess == 0) else 0.0
    Lsess = min(20, n_win)
    session_switch_cnt = 0
    if Lsess >= 2:
        tail_s = window[-Lsess:]
        for i in range(Lsess - 1):
            if tail_s[i].session_id != tail_s[i + 1].session_id:
                session_switch_cnt += 1

    sf_sess = session_first_dttm.get(sid_c) if sid_c is not None else None
    seconds_since_session_start_in_window = float("nan")
    if dttm is not None and sid_c is not None and not _empty_str(sid_c):
        if sf_sess is not None:
            seconds_since_session_start_in_window = float((dttm - sf_sess).total_seconds())
        else:
            seconds_since_session_start_in_window = 0.0

    cap = agg.window_transaction_cap
    if cap is None:
        tr_cap = float(tr_amount)
    else:
        tr_cap = float(min(tr_amount, float(cap)))
    log_1_plus_transactions_seen = math.log1p(max(0.0, tr_cap))

    return {
        "operation_amt": amount,
        "log_1_plus_transactions_seen": log_1_plus_transactions_seen,
        "amount_zscore": amount_zscore,
        "is_amount_high": is_amount_high,
        "transactions_last_24h": tx_24h,
        "sum_amount_last_1h": sum_1h,
        "max_amount_last_24h": max_amount_last_24h_feat,
        "is_new_device": is_new_device,
        "is_new_mcc": is_new_mcc,
        "is_new_channel": is_new_channel,
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
        "event_type_nm": event_type_nm_cur,
        "event_descr": cat_to_float(descr),
        "mcc_code": cat_to_float(mcc_c),
        "trend_mean_last_3_to_10": trend_mean_last_3_to_10,
        "amount_percentile_rank": amount_percentile_rank,
        "std_time_deltas": std_time_deltas,
        "is_new_device_tz_pair": is_new_device_tz_pair,
        "is_device_switch": is_device_switch,
        "is_mcc_switch": is_mcc_switch,
        "session_duration": session_duration,
        "session_mean_amount": session_mean_amount,
        "device_freq": _safe_div(float(device_count_i), float(len(window))) if len(window) > 0 else float("nan"),
        "delta_1": delta_1,
        "delta_2": delta_2,
        "delta_3": delta_3,
        "acceleration_delta_1_over_2": acceleration,
        "std_delta_last_k": std_delta_last_k,
        "time_since_last_device_change": time_since_last_device_change,
        "time_since_last_mcc_change": time_since_last_mcc_change,
        "event_descr_freq_last_1h": event_descr_freq_last_1h,
        "event_descr_freq_last_6h": event_descr_freq_last_6h,
        "event_descr_freq_last_24h": event_descr_freq_last_24h,
        "event_type_nm_freq_last_1h": event_type_nm_freq_last_1h,
        "event_type_nm_freq_last_6h": event_type_nm_freq_last_6h,
        "event_type_nm_freq_last_24h": event_type_nm_freq_last_24h,
        "mcc_freq_last_1h": mcc_freq_last_1h,
        "mcc_freq_last_6h": mcc_freq_last_6h,
        "mcc_freq_last_24h": mcc_freq_last_24h,
        "mcc_event_descr_pair_new": mcc_event_descr_pair_new,
        "high_amount_ratio_last_24h": high_amount_ratio_last_24h,
        "amount_relative_to_mcc_median_5_days": amount_relative_to_mcc_median_5_days,
        "amount_ratio_to_window_median": amount_ratio_to_window_median,
        "amount_iqr_normalized": amount_iqr_normalized,
        "amount_cv_in_window": amount_cv_in_window,
        "sum_amount_last_10m": sum_amount_last_10m,
        "transactions_last_1h": tx_1h,
        "unique_mcc_count_suffix": float(len(umcc)),
        "unique_device_key_count_suffix": float(len(udev)),
        "unique_channel_key_count_suffix": float(len(uch)),
        "unique_timezone_count_suffix": float(len(utz)),
        "unique_browser_language_count_suffix": float(len(ubl)),
        "mcc_switch_count_last_20_tx": float(mcc_switch_cnt),
        "device_switch_count_last_20_tx": float(dev_switch_cnt),
        "channel_switch_count_last_20_tx": float(ch_switch_cnt),
        "night_transaction_share_last_24h": night_share_24h,
        "distinct_hours_active_last_24h": distinct_hours_24h,
        "mean_amount_last_3_transactions": mean_last_3,
        "amount_ratio_to_min_amount_24h": amount_ratio_to_min_amount_24h,
        "compromised_history_count_suffix": float(compromised_hist),
        "seconds_since_last_compromised_tx": secs_since_compromised,
        "web_rdp_count_last_24h": float(web_rdp_cnt_24h),
        "voip_flag_count_last_24h": float(voip_cnt_24h),
        "channel_relative_freq": channel_rel_freq,
        "timezone_relative_freq": tz_rel_freq,
        "browser_language_relative_freq": bl_rel_freq,
        "event_type_nm_share_in_suffix": event_type_nm_share_suffix,
        "mcc_consecutive_streak_length": mcc_streak,
        "transactions_last_5m": tx_5m,
        "mean_gap_seconds_last_5_intervals": mean_gap_last_5,
        "suffix_time_span_hours_log1p": suffix_span_log1p,
        "transactions_per_span_hour": tx_per_span_hour,
        "mcc_amount_std_same_5d": mcc_amount_std_same_5d,
        "weekend_transaction_share_last_7d": weekend_share_7d,
        "distinct_event_descr_count_last_24h_norm": descr_div_24h,
        "is_new_timezone": is_new_timezone,
        "is_timezone_change": is_timezone_change,
        "currency_iso_cd_cat": cat_to_float(row.get("currency_iso_cd")),
        "pos_cd_cat": cat_to_float(row.get("pos_cd")),
        "accept_language_cat": cat_to_float(row.get("accept_language")),
        "battery_level": battery_level_v,
        "battery_very_low_flag": battery_very_low_f,
        "screen_size_cat": cat_to_float(row.get("screen_size")),
        "developer_tools_flag": developer_tools_f,
        "accept_lang_browser_lang_mismatch": accept_lang_browser_mismatch,
        "battery_very_low_and_night": battery_very_low_and_night,
        "amount_diff_prev": amount_diff_prev,
        "amount_ratio_prev": amount_ratio_prev,
        "amount_change_sign": amount_change_sign,
        "amount_increase_streak_suffix": float(amount_inc_streak),
        "amount_decrease_streak_suffix": float(amount_dec_streak),
        "is_new_session_id": is_new_session_id,
        "session_switch_count_last_20_tx": float(session_switch_cnt),
        "seconds_since_session_start_in_window": seconds_since_session_start_in_window,
    }
