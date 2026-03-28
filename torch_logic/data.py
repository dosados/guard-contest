from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
import torch

from shared.config import BATCH_SIZE

CUSTOMER_ID_COL = "customer_id"
EVENT_ID_COL = "event_id"
MCC_COL = "mcc_code"
EVENT_DESCR_COL = "event_descr"
EVENT_DESC_FALLBACK_COL = "event_desc"
EVENT_TYPE_COL = "event_type_nm"
OPERATION_AMT_COL = "operaton_amt"
EVENT_DTTM_COL = "event_dttm"


@dataclass(frozen=True)
class RawRow:
    customer_id: str
    event_id: int
    mcc_code: float
    event_descr: float
    event_type_nm: float
    operaton_amt: float
    event_dttm: datetime | None


def rows_to_tensor(rows: list[RawRow], device: torch.device) -> torch.Tensor:
    """Последовательность шагов одного пользователя: (1, T, 4), batch_first для LSTM."""
    feats = [[r.mcc_code, r.event_descr, r.event_type_nm, r.operaton_amt] for r in rows]
    return torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)


def _to_float(v: object) -> float:
    if v is None:
        return 0.0
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(x):
        return 0.0
    return x


def _to_customer_id(v: object) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    return s


def _to_datetime(v: object) -> datetime | None:
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


def load_positive_event_ids(labels_path: Path) -> set[int]:
    table = pq.read_table(labels_path, columns=[EVENT_ID_COL])
    ids = table.column(EVENT_ID_COL).to_pylist()
    return {int(v) for v in ids if v is not None}


def iter_raw_rows(paths: list[Path], batch_size: int = BATCH_SIZE) -> Iterator[RawRow]:
    for path in paths:
        if not path.exists():
            continue
        pf = pq.ParquetFile(path)
        schema_names = set(pf.schema_arrow.names)
        cols = [CUSTOMER_ID_COL, EVENT_ID_COL, MCC_COL, EVENT_TYPE_COL, OPERATION_AMT_COL, EVENT_DTTM_COL]
        if EVENT_DESCR_COL in schema_names:
            cols.append(EVENT_DESCR_COL)
        elif EVENT_DESC_FALLBACK_COL in schema_names:
            cols.append(EVENT_DESC_FALLBACK_COL)
        else:
            cols.append(EVENT_DESCR_COL)

        for rb in pf.iter_batches(columns=cols, batch_size=batch_size):
            names = rb.schema.names
            data = {n: rb.column(n).to_pylist() for n in names}
            n_rows = rb.num_rows
            event_descr_name = EVENT_DESCR_COL if EVENT_DESCR_COL in names else EVENT_DESC_FALLBACK_COL

            for i in range(n_rows):
                cid = _to_customer_id(data[CUSTOMER_ID_COL][i])
                if cid is None:
                    continue
                event_val = data[EVENT_ID_COL][i]
                if event_val is None:
                    continue
                yield RawRow(
                    customer_id=cid,
                    event_id=int(event_val),
                    mcc_code=_to_float(data[MCC_COL][i]),
                    event_descr=_to_float(data[event_descr_name][i]) if event_descr_name in data else 0.0,
                    event_type_nm=_to_float(data[EVENT_TYPE_COL][i]) if EVENT_TYPE_COL in data else 0.0,
                    operaton_amt=_to_float(data[OPERATION_AMT_COL][i]) if OPERATION_AMT_COL in data else 0.0,
                    event_dttm=_to_datetime(data[EVENT_DTTM_COL][i]) if EVENT_DTTM_COL in data else None,
                )
