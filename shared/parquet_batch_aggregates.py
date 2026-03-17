"""
Агрегаты по пользователям для онлайн-вычисления фичей.
Колонки и FEATURE_COLUMNS можно менять для добавления новых агрегируемых полей.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from multiprocessing import Pool
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
SESSION_ID_COLUMN = "session_id"
BROWSER_LANGUAGE_COLUMN = "browser_language"

RECENT_EVENTS_HOURS = 24
EVENT_ID_COLUMN = "event_id"

EVENT_DESCR_COLUMN = "event_descr"
EVENT_DESC_COLUMN = "event_desc"  # альтернативное имя колонки в данных

# Колонки для агрегации и фичей — расширяемый список
FEATURE_COLUMNS = [
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
    Агрегаты по пользователю для онлайн-вычисления фичей.
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
        "last_event_dttm",
        "session_counts",
        "seen_browser_languages",
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
        self.last_event_dttm: datetime | None = None
        self.session_counts: dict[Any, int] = defaultdict(int)
        self.seen_browser_languages: set[Any] = set()

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
        if dttm is not None:
            self.last_event_dttm = dttm
        sid = row.get(SESSION_ID_COLUMN)
        if sid is not None:
            self.session_counts[sid] = self.session_counts[sid] + 1
        bl = row.get(BROWSER_LANGUAGE_COLUMN)
        if bl is not None and str(bl).strip() != "":
            self.seen_browser_languages.add(bl)

    def remove(self, row: dict[str, Any]) -> None:
        """
        Обратная операция к update() для числовых и временных агрегатов.
        Используется при работе в фиксированном окне по числу транзакций:
        когда самая старая транзакция выходит из окна, её вклад вычитается.

        Для множественных признаков (seen_* и session_counts) мы намеренно
        не выполняем «откат» до точного окна, чтобы не усложнять логику;
        они остаются монотонными по пользователю, что приемлемо для наших фичей.
        """
        amt_val = row.get(AMOUNT_COLUMN)
        if amt_val is None:
            return
        try:
            amount = float(amt_val)
        except (TypeError, ValueError):
            return

        # Обновляем суммарные статистики, если есть что вычитать
        if self.count > 0:
            self.sum_amt -= amount
            self.sum_sq -= amount * amount
            self.count -= 1
            if self.count < 0:
                self.count = 0

        # Удаляем одно вхождение amount из списка amounts (если есть)
        for idx, val in enumerate(self.amounts):
            if val == amount:
                del self.amounts[idx]
                break

        # Удаляем соответствующее событие из recent_events (если попадает)
        dttm = parse_dttm(row.get(EVENT_DTTM_COLUMN))
        if dttm is not None and self.recent_events:
            # Поскольку удаляем самую старую транзакцию, она должна быть
            # ближе к началу очереди; ищем первое совпадение.
            for idx, (evt_dttm, evt_amt) in enumerate(self.recent_events):
                if evt_dttm == dttm and evt_amt == amount:
                    del self.recent_events[idx]
                    break

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

    def transactions_last_10m(self, current_dttm: datetime) -> int:
        cutoff = current_dttm - timedelta(minutes=10)
        return sum(1 for dttm, _ in self.recent_events if dttm >= cutoff)

    def sum_amount_last_24h(self, current_dttm: datetime) -> float:
        cutoff = current_dttm - timedelta(hours=24)
        return sum(amt for dttm, amt in self.recent_events if dttm >= cutoff)

    def get_session_count(self, session_id: Any) -> int:
        """Количество транзакций в данной сессии до текущей (без текущей)."""
        if session_id is None:
            return 0
        return self.session_counts.get(session_id, 0)

    def to_result(self) -> dict[str, float]:
        return {
            "mean_operaton_amt": self.mean(),
            "median_operaton_amt": self.median(),
            "std_operaton_amt": self.std(),
        }


def _break_ref(val: Any) -> Any:
    """Копирует значение, чтобы не держать ссылку на данные батча."""
    if val is None:
        return None
    if isinstance(val, str):
        return str(val)
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return type(val)(val)
    return val


def _row_to_compact(row: dict[str, Any]) -> tuple:
    """
    Извлекает из row только поля, нужные для update(), в виде кортежа.
    Копирует значения, чтобы не держать ссылки на батч (снижение памяти).
    """
    amt = row.get(AMOUNT_COLUMN)
    if amt is None:
        amount = None
    else:
        try:
            amount = float(amt)
        except (TypeError, ValueError):
            amount = float("nan")
    dttm_raw = row.get(EVENT_DTTM_COLUMN)
    dttm_str = str(dttm_raw).strip() if dttm_raw is not None else None
    return (
        amount,
        dttm_str,
        _break_ref(row.get(OPERATING_SYSTEM_TYPE)),
        _break_ref(row.get(DEVICE_SYSTEM_VERSION)),
        _break_ref(row.get(MCC_CODE)),
        _break_ref(row.get(CHANNEL_INDICATOR_TYPE)),
        _break_ref(row.get(CHANNEL_INDICATOR_SUB_TYPE)),
        _break_ref(row.get(TIMEZONE_COLUMN)),
        _break_ref(row.get(COMPROMISED_COLUMN)),
        _break_ref(row.get(WEB_RDP_CONNECTION)),
        _break_ref(row.get(PHONE_VOIP_CALL_STATE)),
        _break_ref(row.get(SESSION_ID_COLUMN)),
        _break_ref(row.get(BROWSER_LANGUAGE_COLUMN)),
    )


def _compact_to_row(compact: tuple) -> dict[str, Any]:
    """Восстанавливает dict-строку для вызова UserAggregates.update()."""
    return {
        AMOUNT_COLUMN: compact[0],
        EVENT_DTTM_COLUMN: compact[1],
        OPERATING_SYSTEM_TYPE: compact[2],
        DEVICE_SYSTEM_VERSION: compact[3],
        MCC_CODE: compact[4],
        CHANNEL_INDICATOR_TYPE: compact[5],
        CHANNEL_INDICATOR_SUB_TYPE: compact[6],
        TIMEZONE_COLUMN: compact[7],
        COMPROMISED_COLUMN: compact[8],
        WEB_RDP_CONNECTION: compact[9],
        PHONE_VOIP_CALL_STATE: compact[10],
        SESSION_ID_COLUMN: compact[11],
        BROWSER_LANGUAGE_COLUMN: compact[12],
    }


class WindowedAggregates:
    """
    Хранит только последние window_size транзакций по пользователю в компактном
    виде (кортежи), а также один «живой» UserAggregates, который инкрементально
    обновляется при сдвиге окна.
    """

    __slots__ = ("_rows", "_window_size", "_agg")

    def __init__(self, window_size: int) -> None:
        self._rows: deque[tuple] = deque(maxlen=window_size)
        self._window_size = window_size
        self._agg: UserAggregates = UserAggregates()

    def add(self, row: dict[str, Any]) -> None:
        """
        Добавляет транзакцию в окно в компактном виде.
        Если окно уже заполнено, самая старая транзакция удаляется, и её вклад
        вычитается из агрегатов (через UserAggregates.remove()).
        """
        compact = _row_to_compact(row)
        if len(self._rows) == self._rows.maxlen:
            old_compact = self._rows.popleft()
            old_row = _compact_to_row(old_compact)
            self._agg.remove(old_row)
        self._rows.append(compact)
        # Обновляем агрегаты новой транзакцией
        self._agg.update(_compact_to_row(compact))

    def get_aggregates(self) -> UserAggregates:
        """
        Возвращает текущие агрегаты по окну.
        ВНИМАНИЕ: возвращается ссылка на внутренний объект, не изменяй его снаружи.
        """
        return self._agg

    def __len__(self) -> int:
        return len(self._rows)


def _build_aggregates_single_file(
    path: Path,
    batch_size: int,
    show_progress: bool,
) -> dict[int | str, UserAggregates]:
    """
    Строит агрегаты по одному parquet-файлу.
    Используется внутри build_user_aggregates (последовательно или в воркерах).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    aggregates: dict[int | str, UserAggregates] = defaultdict(UserAggregates)
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    columns = [c for c in FEATURE_COLUMNS if c in schema.names] if schema else FEATURE_COLUMNS
    num_rows = pf.metadata.num_rows if pf.metadata else None
    total_batches = (num_rows + batch_size - 1) // batch_size if num_rows is not None else None
    batch_iter = pf.iter_batches(columns=columns, batch_size=batch_size)
    if show_progress:
        try:
            from tqdm import tqdm
            batch_iter = tqdm(
                batch_iter,
                desc=path.name,
                total=total_batches,
                unit="batch",
            )
        except ImportError:
            pass
    for batch in batch_iter:
        col_lists = {name: batch.column(name).to_pylist() for name in batch.schema.names}
        n = batch.num_rows
        for i in range(n):
            row = {name: col_lists[name][i] for name in batch.schema.names}
            cid = row.get(CUSTOMER_ID_COLUMN)
            if cid is None:
                continue
            aggregates[cid].update(row)
    return dict(aggregates)


def _worker_build_aggregates(args: tuple) -> dict:
    """Обёртка для multiprocessing: распаковка аргументов."""
    path, batch_size, show_progress = args
    return _build_aggregates_single_file(path, batch_size, show_progress)


def build_user_aggregates(
    paths: list[Path],
    batch_size: int = 100_000,
    show_progress: bool = False,
    n_workers: int = 1,
) -> dict[int | str, UserAggregates]:
    """
    Строит агрегаты по пользователям, проходя по parquet-файлам.
    Возвращает словарь customer_id -> UserAggregates.

    При n_workers=1 (по умолчанию): последовательная обработка.
    При show_progress=True для каждого файла выводится прогресс-бар по батчам.

    При n_workers>1: файлы обрабатываются параллельно в отдельных процессах,
    результаты объединяются (множества клиентов по файлам не пересекаются).
    Прогресс по батчам в параллельном режиме отключён.
    """
    path_list = [Path(p) for p in paths]
    if not path_list:
        return {}

    if n_workers <= 1:
        aggregates = _build_aggregates_single_file(path_list[0], batch_size, show_progress)
        for path in path_list[1:]:
            aggregates.update(_build_aggregates_single_file(path, batch_size, show_progress))
        return aggregates

    # Параллельный режим: в воркерах не показываем прогресс по батчам
    args_list = [(p, batch_size, False) for p in path_list]
    try:
        from tqdm import tqdm
        progress = lambda it: tqdm(it, desc="Pretrain files", unit="file", total=len(path_list))
    except ImportError:
        progress = lambda it: it

    aggregates = {}
    with Pool(processes=min(n_workers, len(path_list))) as pool:
        for part in progress(pool.imap(_worker_build_aggregates, args_list)):
            aggregates.update(part)
    return aggregates


def _build_windowed_single_file(
    path: Path,
    batch_size: int,
    window_size: int,
    aggregates: dict[int | str, WindowedAggregates],
    show_progress: bool,
) -> None:
    """Добавляет строки из одного parquet в оконные агрегаты (модифицирует aggregates)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    columns = [c for c in FEATURE_COLUMNS if c in schema.names] if schema else FEATURE_COLUMNS
    num_rows = pf.metadata.num_rows if pf.metadata else None
    total_batches = (num_rows + batch_size - 1) // batch_size if num_rows is not None else None
    batch_iter = pf.iter_batches(columns=columns, batch_size=batch_size)
    if show_progress:
        try:
            from tqdm import tqdm
            batch_iter = tqdm(
                batch_iter,
                desc=path.name,
                total=total_batches,
                unit="batch",
            )
        except ImportError:
            pass
    for batch in batch_iter:
        col_lists = {name: batch.column(name).to_pylist() for name in batch.schema.names}
        n = batch.num_rows
        for i in range(n):
            row = {name: col_lists[name][i] for name in batch.schema.names}
            cid = row.get(CUSTOMER_ID_COLUMN)
            if cid is None:
                continue
            if cid not in aggregates:
                aggregates[cid] = WindowedAggregates(window_size)
            aggregates[cid].add(row)


def build_windowed_aggregates(
    paths: list[Path],
    window_size: int,
    batch_size: int = 100_000,
    show_progress: bool = False,
) -> dict[int | str, WindowedAggregates]:
    """
    Строит оконные агрегаты по пользователям: для каждого хранятся только
    последние window_size транзакций из переданных файлов (порядок файлов и строк важен).
    Возвращает словарь customer_id -> WindowedAggregates.
    """
    path_list = [Path(p) for p in paths]
    aggregates: dict[int | str, WindowedAggregates] = {}
    for path in path_list:
        _build_windowed_single_file(path, batch_size, window_size, aggregates, show_progress)
    return aggregates
