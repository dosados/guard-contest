"""
Вычисление фичей по агрегатам и текущей строке.
Фичи считаются до обновления агрегатов текущей транзакцией (online).
Список FEATURE_NAMES задаёт порядок и состав фичей для модели — легко расширять.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from shared.parquet_batch_aggregates import (
    AMOUNT_COLUMN,
    BROWSER_LANGUAGE_COLUMN,
    CHANNEL_INDICATOR_SUB_TYPE,
    CHANNEL_INDICATOR_TYPE,
    COMPROMISED_COLUMN,
    DEVICE_SYSTEM_VERSION,
    EVENT_DTTM_COLUMN,
    MCC_CODE,
    OPERATING_SYSTEM_TYPE,
    PHONE_VOIP_CALL_STATE,
    SESSION_ID_COLUMN,
    TIMEZONE_COLUMN,
    WEB_RDP_CONNECTION,
    parse_dttm,
)

if TYPE_CHECKING:
    from shared.parquet_batch_aggregates import UserAggregates

# Базовый список фичей (без transactions_seen)
FEATURE_NAMES = [
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
    # временные и сессионные
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
]

# Имя фичи «сколько транзакций по клиенту уже было в датасете»
TRANSACTIONS_SEEN_FEATURE = "transactions_seen"

# Полный список фичей для режима "full" (базовые + transactions_seen)
FEATURE_NAMES_FULL = FEATURE_NAMES + [TRANSACTIONS_SEEN_FEATURE]


def compute_features(
    agg: "UserAggregates",
    row: dict[str, Any],
    transactions_seen: int | None = None,
) -> dict[str, float]:
    """
    Считает все фичи по текущему состоянию агрегатов и строке.
    Вызывать до agg.update(row).
    transactions_seen: если задано, добавляется фича transactions_seen (режим full).
    """
    amt_val = row.get(AMOUNT_COLUMN)
    try:
        amount = float(amt_val) if amt_val is not None else float("nan")
    except (TypeError, ValueError):
        amount = float("nan")

    median = agg.median()
    mean = agg.mean()
    std = agg.std()
    amount_to_median = (
        amount / median
        if not math.isnan(median) and median != 0
        else float("nan")
    )
    amount_zscore = (
        (amount - mean) / std
        if not math.isnan(std) and std > 0
        else float("nan")
    )
    p95 = agg.percentile(95.0)
    is_amount_high = 1.0 if (not math.isnan(p95) and amount > p95) else 0.0

    dttm = parse_dttm(row.get(EVENT_DTTM_COLUMN))
    if dttm is None:
        transactions_last_1h = 0
        transactions_last_24h = 0
        sum_amount_last_1h = 0.0
        max_amount_last_24h = float("nan")
        transactions_last_10m = 0
        sum_amount_last_24h = 0.0
        hour_val = -1
        day_of_week_val = -1
        time_since_prev = float("nan")
    else:
        transactions_last_1h = agg.transactions_last_1h(dttm)
        transactions_last_24h = agg.transactions_last_24h(dttm)
        sum_amount_last_1h = agg.sum_amount_last_1h(dttm)
        max_amount_last_24h = agg.max_amount_last_24h(dttm)
        transactions_last_10m = agg.transactions_last_10m(dttm)
        sum_amount_last_24h = agg.sum_amount_last_24h(dttm)
        hour_val = dttm.hour
        day_of_week_val = dttm.weekday()  # 0=Monday, 6=Sunday
        if agg.last_event_dttm is not None:
            time_since_prev = (dttm - agg.last_event_dttm).total_seconds()
        else:
            time_since_prev = float("nan")

    os_type = row.get(OPERATING_SYSTEM_TYPE)
    dev_ver = row.get(DEVICE_SYSTEM_VERSION)
    device_key = (os_type, dev_ver)
    is_new_device = 1.0 if (os_type is not None or dev_ver is not None) and device_key not in agg.seen_devices else 0.0
    mcc = row.get(MCC_CODE)
    is_new_mcc = 1.0 if mcc is not None and mcc not in agg.seen_mcc else 0.0
    ch_type = row.get(CHANNEL_INDICATOR_TYPE)
    ch_sub = row.get(CHANNEL_INDICATOR_SUB_TYPE)
    ch_key = (ch_type, ch_sub)
    is_new_channel = 1.0 if (ch_type is not None or ch_sub is not None) and ch_key not in agg.seen_channels else 0.0
    tz = row.get(TIMEZONE_COLUMN)
    is_new_timezone = 1.0 if tz is not None and tz not in agg.seen_timezones else 0.0

    is_compromised_device = 1.0 if row.get(COMPROMISED_COLUMN) not in (None, "") else 0.0
    web_rdp_connection = 1.0 if row.get(WEB_RDP_CONNECTION) not in (None, "") else 0.0
    phone_voip_call_state = 1.0 if row.get(PHONE_VOIP_CALL_STATE) not in (None, "") else 0.0

    # Ночная транзакция: 22:00–05:59
    is_night = 1.0 if hour_val >= 22 or hour_val < 6 else 0.0
    is_weekend = 1.0 if day_of_week_val >= 5 else 0.0  # 5=Sat, 6=Sun
    bl = row.get(BROWSER_LANGUAGE_COLUMN)
    is_new_browser_language = 1.0 if (bl is not None and str(bl).strip() != "" and bl not in agg.seen_browser_languages) else 0.0
    sid = row.get(SESSION_ID_COLUMN)
    transactions_in_session = float(agg.get_session_count(sid) + 1)  # включая текущую
    tz_val = row.get(TIMEZONE_COLUMN)
    timezone_missing = 1.0 if (tz_val is None or (isinstance(tz_val, str) and tz_val.strip() == "")) else 0.0

    result = {
        "operation_amt": amount,
        "amount_to_median": amount_to_median,
        "amount_zscore": amount_zscore,
        "is_amount_high": is_amount_high,
        "transactions_last_1h": float(transactions_last_1h),
        "transactions_last_24h": float(transactions_last_24h),
        "sum_amount_last_1h": sum_amount_last_1h,
        "max_amount_last_24h": max_amount_last_24h if max_amount_last_24h == max_amount_last_24h else 0.0,
        "is_new_device": is_new_device,
        "is_new_mcc": is_new_mcc,
        "is_new_channel": is_new_channel,
        "is_new_timezone": is_new_timezone,
        "is_compromised_device": is_compromised_device,
        "web_rdp_connection": web_rdp_connection,
        "phone_voip_call_state": phone_voip_call_state,
        "hour": float(hour_val) if hour_val >= 0 else float("nan"),
        "day_of_week": float(day_of_week_val) if day_of_week_val >= 0 else float("nan"),
        "is_night_transaction": is_night,
        "is_weekend": is_weekend,
        "transactions_last_10m": float(transactions_last_10m),
        "sum_amount_last_24h": sum_amount_last_24h,
        "time_since_prev_transaction": -1.0 if math.isnan(time_since_prev) else time_since_prev,
        "is_new_browser_language": is_new_browser_language,
        "transactions_in_session": transactions_in_session,
        "timezone_missing": timezone_missing,
    }
    if transactions_seen is not None:
        result[TRANSACTIONS_SEEN_FEATURE] = float(transactions_seen)
    return result
