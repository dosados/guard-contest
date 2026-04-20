from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterator, Mapping, MutableMapping

import pyarrow.parquet as pq
from tqdm import tqdm

from shared.dataset_settings import DATASET_MODE, WINDOW_TRANSACTIONS, WINDOW_TRANSACTIONS_MODE

logger = logging.getLogger(__name__)

CUSTOMER_ID_COLUMN = "customer_id"
EVENT_ID_COLUMN = "event_id"

# Columns to read from parquet (minimal set + optional names)
FEATURE_COLUMNS = [
    "customer_id",
    "event_id",
    "event_dttm",
    "operaton_amt",
    "operating_system_type",
    "device_system_version",
    "mcc_code",
    "channel_indicator_type",
    "channel_indicator_subtype",
    "channel_indicator_sub_type",
    "timezone",
    "compromised",
    "web_rdp_connection",
    "phone_voip_call_state",
    "session_id",
    "browser_language",
    "event_type_nm",
    "event_descr",
    "event_desc",
    "currency_iso_cd",
    "pos_cd",
    "accept_language",
    "battery",
    "screen_size",
    "developer_tools",
]


def effective_window_size() -> int:
    if DATASET_MODE == "window_50":
        return WINDOW_TRANSACTIONS_MODE
    return WINDOW_TRANSACTIONS


@dataclass
class WindowTxn:
    amount: float | None
    dttm: datetime | None
    os_type: Any
    dev_ver: Any
    mcc: Any
    ch_type: Any
    ch_sub: Any
    tz: Any
    compromised: Any
    web_rdp: Any
    voip: Any
    session_id: Any
    browser_language: Any
    event_type_nm: Any
    event_descr: Any


def _parse_amount(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except (TypeError, ValueError):
        return None


def _parse_dttm(v: Any) -> datetime | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    s = str(v).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def row_to_window_txn(row: Mapping[str, Any]) -> WindowTxn:
    ch_sub = row.get("channel_indicator_sub_type")
    if ch_sub is None:
        ch_sub = row.get("channel_indicator_subtype")
    return WindowTxn(
        amount=_parse_amount(row.get("operaton_amt")),
        dttm=_parse_dttm(row.get("event_dttm")),
        os_type=row.get("operating_system_type"),
        dev_ver=row.get("device_system_version"),
        mcc=row.get("mcc_code"),
        ch_type=row.get("channel_indicator_type"),
        ch_sub=ch_sub,
        tz=row.get("timezone"),
        compromised=row.get("compromised"),
        web_rdp=row.get("web_rdp_connection"),
        voip=row.get("phone_voip_call_state"),
        session_id=row.get("session_id"),
        browser_language=row.get("browser_language"),
        event_type_nm=row.get("event_type_nm"),
        event_descr=row.get("event_descr") if row.get("event_descr") is not None else row.get("event_desc"),
    )


class UserAggregates:
    __slots__ = ("_window", "_max_len")

    def __init__(self, max_len: int | None = None, *, unlimited: bool = False) -> None:
        if unlimited:
            self._max_len: int | None = None
        elif max_len is not None:
            self._max_len = max_len
        else:
            self._max_len = effective_window_size()
        self._window: deque[WindowTxn] = deque()

    @property
    def window_transaction_cap(self) -> int | None:
        return self._max_len

    def __len__(self) -> int:
        return len(self._window)

    def update(self, row: Mapping[str, Any]) -> None:
        self._window.append(row_to_window_txn(row))
        if self._max_len is not None:
            while len(self._window) > self._max_len:
                self._window.popleft()

    def transactions_before_current_count(self) -> int:
        return len(self._window)


def _existing_columns(path: Any, want: list[str]) -> list[str]:
    schema = pq.read_schema(path)
    names = set(schema.names)
    return [c for c in want if c in names]


def _parquet_total_rows(paths: list[Any]) -> int:
    total = 0
    for path in paths:
        if not path.exists():
            continue
        try:
            total += int(pq.ParquetFile(path).metadata.num_rows)
        except (OSError, ValueError, AttributeError, TypeError) as e:
            logger.debug("Could not read num_rows for %s: %s", path, e)
    return total


def iter_parquet_rows(
    paths: list[Any],
    columns: list[str] | None = None,
    batch_size: int = 65_536,
    *,
    show_progress: bool = False,
    progress_desc: str = "parquet rows",
) -> Iterator[dict[str, Any]]:
    cols_base = columns or FEATURE_COLUMNS
    row_iter = _iter_parquet_rows_impl(paths, cols_base, batch_size)
    if show_progress:
        total = _parquet_total_rows(paths)
        row_iter = tqdm(row_iter, total=total if total > 0 else None, desc=progress_desc, unit="rows")
    yield from row_iter


def _iter_parquet_rows_impl(
    paths: list[Any],
    cols_base: list[str],
    batch_size: int,
) -> Iterator[dict[str, Any]]:
    for path in paths:
        if not path.exists():
            logger.warning("Parquet file missing, skip: %s", path)
            continue
        use_cols = _existing_columns(path, cols_base)
        if CUSTOMER_ID_COLUMN not in use_cols:
            use_cols = [CUSTOMER_ID_COLUMN] + use_cols
        logger.info("Reading parquet: %s (columns: %d)", path, len(use_cols))
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=use_cols, batch_size=batch_size):
            names = batch.schema.names
            col_lists = {n: batch.column(n).to_pylist() for n in names}
            n = batch.num_rows
            for i in range(n):
                yield {name: col_lists[name][i] for name in names}


def build_windowed_aggregates(
    paths: list[Any],
    batch_size: int = 65_536,
    show_progress: bool = False,
    *,
    unlimited_window: bool = False,
) -> dict[Any, UserAggregates]:
    # Scan paths and fill per-customer aggregates (update only)
    aggregates: dict[Any, UserAggregates] = {}
    logger.info(
        "Windowed aggregates: files=%d, batch_size=%d, unlimited_window=%s",
        len(paths),
        batch_size,
        unlimited_window,
    )
    rows_iter = iter_parquet_rows(
        paths,
        batch_size=batch_size,
        show_progress=show_progress,
        progress_desc="windowed aggregates",
    )
    for row in rows_iter:
        cid = row.get(CUSTOMER_ID_COLUMN)
        if cid is None or (isinstance(cid, float) and math.isnan(cid)):
            continue
        if isinstance(cid, str) and not cid.strip():
            continue
        if cid not in aggregates:
            aggregates[cid] = UserAggregates(unlimited=unlimited_window)
        aggregates[cid].update(row)
    logger.info("Windowed aggregates: unique customer_id=%d", len(aggregates))
    return aggregates


def build_user_aggregates(
    paths: list[Any],
    batch_size: int = 65_536,
    show_progress: bool = False,
    *,
    unlimited_window: bool = False,
) -> dict[Any, UserAggregates]:
    return build_windowed_aggregates(
        paths,
        batch_size=batch_size,
        show_progress=show_progress,
        unlimited_window=unlimited_window,
    )


def defaultdict_aggregates() -> MutableMapping[Any, UserAggregates]:
    return defaultdict(UserAggregates)
