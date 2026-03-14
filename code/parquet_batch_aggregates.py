from __future__ import annotations

import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Колонки из глоссария (README)
AMOUNT_COLUMN = "operaton_amt"
CUSTOMER_ID_COLUMN = "customer_id"
EVENT_DTTM_COLUMN = "event_dttm"
OPERATING_SYSTEM_TYPE = "operating_system_type"
DEVICE_SYSTEM_VERSION = "device_system_version"
MCC_CODE = "mcc_code"
CHANNEL_INDICATOR_TYPE = "channel_indicator_type"
CHANNEL_INDICATOR_SUB_TYPE = "channel_indicator_sub_type"
TIMEZONE_COLUMN = "timezone"
COMPROMISED_COLUMN = "compromised"
WEB_RDP_CONNECTION = "web_rdp_connection"
PHONE_VOIP_CALL_STATE = "phone_voip_call_state"

# Окно для скользящих фичей (транзакции за 1ч / 24ч)
RECENT_EVENTS_HOURS = 24

EVENT_ID_COLUMN = "event_id"

# Колонки, нужные для агрегации и фичей (если в файле нет — не читаем)
FEATURE_COLUMNS = [
    CUSTOMER_ID_COLUMN,
    AMOUNT_COLUMN,
    EVENT_DTTM_COLUMN,
    OPERATING_SYSTEM_TYPE,
    DEVICE_SYSTEM_VERSION,
    MCC_CODE,
    CHANNEL_INDICATOR_TYPE,
    CHANNEL_INDICATOR_SUB_TYPE,
    TIMEZONE_COLUMN,
    COMPROMISED_COLUMN,
    WEB_RDP_CONNECTION,
    PHONE_VOIP_CALL_STATE,
]


def parse_dttm(s: str | None) -> datetime | None:
    if s is None:
        return None
    try:
        return datetime.strptime(str(s).strip(), "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return None


class UserAggregates:
    """
    Агрегаты по пользователю для онлайн-вычисления фичей:
    operation_amt, amount_to_median, amount_zscore, is_amount_high,
    transactions_last_1h/24h, sum_amount_last_1h, max_amount_last_24h,
    is_new_device/mcc/channel/timezone, is_compromised_device, web_rdp_connection, phone_voip_call_state.
    """

    __slots__ = (
        "sum_amt",
        "sum_sq",
        "count",
        "amounts",
        "recent_events",
        "seen_devices",
        "seen_mcc",
        "seen_channels",
        "seen_timezones",
        "ever_compromised",
        "ever_web_rdp",
        "ever_voip",
    )

    def __init__(self) -> None:
        self.sum_amt: float = 0.0
        self.sum_sq: float = 0.0
        self.count: int = 0
        self.amounts: list[float] = []
        self.recent_events: deque[tuple[datetime, float]] = deque()
        self.seen_devices: set[tuple[Any, Any]] = set()
        self.seen_mcc: set[Any] = set()
        self.seen_channels: set[tuple[Any, Any]] = set()
        self.seen_timezones: set[Any] = set()
        self.ever_compromised: bool = False
        self.ever_web_rdp: bool = False
        self.ever_voip: bool = False

    def update(self, row: dict[str, Any]) -> None:
        amt_val = row.get(AMOUNT_COLUMN)
        if amt_val is None:
            return
        try:
            amount = float(amt_val)
        except (TypeError, ValueError):
            return
        self.sum_amt += amount
        self.sum_sq += amount * amount
        self.count += 1
        self.amounts.append(amount)

        dttm = parse_dttm(row.get(EVENT_DTTM_COLUMN))
        if dttm is not None:
            cutoff = dttm - timedelta(hours=RECENT_EVENTS_HOURS)
            while self.recent_events and self.recent_events[0][0] < cutoff:
                self.recent_events.popleft()
            self.recent_events.append((dttm, amount))

        os_type = row.get(OPERATING_SYSTEM_TYPE)
        dev_ver = row.get(DEVICE_SYSTEM_VERSION)
        if os_type is not None or dev_ver is not None:
            self.seen_devices.add((os_type, dev_ver))
        mcc = row.get(MCC_CODE)
        if mcc is not None:
            self.seen_mcc.add(mcc)
        ch_type = row.get(CHANNEL_INDICATOR_TYPE)
        ch_sub = row.get(CHANNEL_INDICATOR_SUB_TYPE)
        if ch_type is not None or ch_sub is not None:
            self.seen_channels.add((ch_type, ch_sub))
        tz = row.get(TIMEZONE_COLUMN)
        if tz is not None:
            self.seen_timezones.add(tz)
        if row.get(COMPROMISED_COLUMN) not in (None, ""):
            self.ever_compromised = True
        if row.get(WEB_RDP_CONNECTION) not in (None, ""):
            self.ever_web_rdp = True
        if row.get(PHONE_VOIP_CALL_STATE) not in (None, ""):
            self.ever_voip = True

    def mean(self) -> float:
        if self.count == 0:
            return float("nan")
        return self.sum_amt / self.count

    def std(self) -> float:
        if self.count < 2:
            return float("nan")
        var = (self.sum_sq - self.sum_amt * self.sum_amt / self.count) / (self.count - 1)
        return float(np.sqrt(max(0.0, var)))

    def median(self) -> float:
        if not self.amounts:
            return float("nan")
        return float(np.median(self.amounts))

    def percentile(self, q: float) -> float:
        if not self.amounts:
            return float("nan")
        return float(np.percentile(self.amounts, q))

    def transactions_last_1h(self, current_dttm: datetime) -> int:
        cutoff = current_dttm - timedelta(hours=1)
        return sum(1 for dttm, _ in self.recent_events if dttm >= cutoff)

    def transactions_last_24h(self, current_dttm: datetime) -> int:
        cutoff = current_dttm - timedelta(hours=24)
        return sum(1 for dttm, _ in self.recent_events if dttm >= cutoff)

    def sum_amount_last_1h(self, current_dttm: datetime) -> float:
        cutoff = current_dttm - timedelta(hours=1)
        return sum(amt for dttm, amt in self.recent_events if dttm >= cutoff)

    def max_amount_last_24h(self, current_dttm: datetime) -> float:
        cutoff = current_dttm - timedelta(hours=24)
        amounts = [amt for dttm, amt in self.recent_events if dttm >= cutoff]
        return max(amounts) if amounts else float("nan")

    def to_result(self) -> dict[str, float]:
        return {
            "mean_operaton_amt": self.mean(),
            "median_operaton_amt": self.median(),
            "std_operaton_amt": self.std(),
        }


def build_user_aggregates(
    paths: list[Path],
    batch_size: int = 100_000,
    show_progress: bool = False,
) -> dict[int | str, UserAggregates]:
    """
    Строит агрегаты по пользователям, проходя по одному или нескольким parquet-файлам.
    Возвращает словарь customer_id -> UserAggregates (объекты, а не to_result()).
    """
    aggregates: dict[int | str, UserAggregates] = defaultdict(UserAggregates)
    path_list = [Path(p) for p in paths]
    path_iter = path_list
    if show_progress:
        try:
            from tqdm import tqdm
            path_iter = tqdm(path_list, desc="Files", unit="file")
        except ImportError:
            pass
    for path in path_iter:
        if not path.exists():
            logger.error("Parquet file not found: %s", path)
            raise FileNotFoundError(f"Parquet file not found: {path}")
        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        columns = [c for c in FEATURE_COLUMNS if c in schema.names] if schema else FEATURE_COLUMNS
        for batch in pf.iter_batches(columns=columns, batch_size=batch_size):
            col_lists = {name: batch.column(name).to_pylist() for name in batch.schema.names}
            n = batch.num_rows
            for i in range(n):
                row = {name: col_lists[name][i] for name in batch.schema.names}
                cid = row.get(CUSTOMER_ID_COLUMN)
                if cid is None:
                    continue
                aggregates[cid].update(row)
    return aggregates


def compute_user_aggregates_from_parquet(
    path: str | Path,
    batch_size: int = 100_000,
    amount_column: str = AMOUNT_COLUMN,
    customer_column: str = CUSTOMER_ID_COLUMN,
) -> dict[int | str, dict[str, float]]:

    path = Path(path)
    if not path.exists():
        logger.error("Parquet file not found: %s", path)
        raise FileNotFoundError(f"Parquet file not found: {path}")
    try:
        pf = pq.ParquetFile(path)
    except Exception as e:
        logger.exception("Failed to open parquet %s: %s", path, e)
        raise

    aggregates: dict[int | str, UserAggregates] = defaultdict(UserAggregates)
    schema = pf.schema_arrow
    columns = [c for c in FEATURE_COLUMNS if c in schema.names] if schema else FEATURE_COLUMNS

    for batch in pf.iter_batches(columns=columns, batch_size=batch_size):
        col_lists = {name: batch.column(name).to_pylist() for name in batch.schema.names}
        n = batch.num_rows
        for i in range(n):
            row = {name: col_lists[name][i] for name in batch.schema.names}
            cid = row.get(CUSTOMER_ID_COLUMN)
            if cid is None:
                continue
            aggregates[cid].update(row)

    return {cid: agg.to_result() for cid, agg in aggregates.items()}


def compute_user_aggregates_from_parquet_lazy(
    path: str | Path,
    batch_size: int = 100_000,
    amount_column: str = AMOUNT_COLUMN,
    customer_column: str = CUSTOMER_ID_COLUMN,
) -> Iterator[tuple[pa.RecordBatch, dict[int | str, UserAggregates]]]:
    """
    Ленивый вариант: выдаёт батч и текущее состояние агрегатов после его обработки.
    Позволяет обрабатывать батч (например, для фич) и при этом накапливать агрегаты.
    """
    path = Path(path)
    if not path.exists():
        logger.error("Parquet file not found: %s", path)
        raise FileNotFoundError(f"Parquet file not found: {path}")
    try:
        pf = pq.ParquetFile(path)
    except Exception as e:
        logger.exception("Failed to open parquet %s: %s", path, e)
        raise

    aggregates: dict[int | str, UserAggregates] = defaultdict(UserAggregates)
    schema = pf.schema_arrow
    columns = [c for c in FEATURE_COLUMNS if c in schema.names] if schema else FEATURE_COLUMNS

    for batch in pf.iter_batches(columns=columns, batch_size=batch_size):
        col_lists = {name: batch.column(name).to_pylist() for name in batch.schema.names}
        n = batch.num_rows
        for i in range(n):
            row = {name: col_lists[name][i] for name in batch.schema.names}
            cid = row.get(CUSTOMER_ID_COLUMN)
            if cid is None:
                continue
            aggregates[cid].update(row)

        yield batch, dict(aggregates)


if __name__ == "__main__":
    

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )



    path = "/home/timofey/Documents/own/guard-contest/data/train/train_part_1.parquet"
    batch_size = 100_000

    try:
        print(f"Reading {path} in batches of {batch_size}...")
        result = compute_user_aggregates_from_parquet(path, batch_size=batch_size)
        print(f"Users: {len(result)}")
    except Exception:
        logger.exception("Error processing parquet %s", path)
        raise

    for i, (cid, aggs) in enumerate(list(result.items())[:5]):
        print(f"  {cid}: mean={aggs['mean_operaton_amt']:.2f}, median={aggs['median_operaton_amt']:.2f}")
    if len(result) > 5:
        print("  ...")
