"""
Вычисление фичей для пайплайна по агрегатам и текущей строке.
Фичи считаются до обновления агрегатов текущей транзакцией (online).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from parquet_batch_aggregates import (
    AMOUNT_COLUMN,
    CHANNEL_INDICATOR_SUB_TYPE,
    CHANNEL_INDICATOR_TYPE,
    COMPROMISED_COLUMN,
    DEVICE_SYSTEM_VERSION,
    EVENT_DTTM_COLUMN,
    MCC_CODE,
    OPERATING_SYSTEM_TYPE,
    PHONE_VOIP_CALL_STATE,
    TIMEZONE_COLUMN,
    WEB_RDP_CONNECTION,
    parse_dttm,
)

if TYPE_CHECKING:
    from parquet_batch_aggregates import UserAggregates

# Порядок и имена фичей для CatBoost (конфигурируемо для добавления новых)
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
]


def compute_features(agg: UserAggregates, row: dict[str, Any]) -> dict[str, float]:
    """
    Считает все фичи по текущему состоянию агрегатов и строке.
    Вызывать до agg.update(row).
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
    else:
        transactions_last_1h = agg.transactions_last_1h(dttm)
        transactions_last_24h = agg.transactions_last_24h(dttm)
        sum_amount_last_1h = agg.sum_amount_last_1h(dttm)
        max_amount_last_24h = agg.max_amount_last_24h(dttm)

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

    return {
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
    }
