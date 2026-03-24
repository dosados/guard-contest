"""
Пробный parquet → build_dataset (C++) → проверка output/full_dataset.parquet.

Сборка бинарника (из корня репозитория):
  cd dataset_cpp && cmake -S . -B build && cmake --build build
Бинарник: dataset_cpp/build/build_dataset. При необходимости задайте BUILD_DATASET_BIN.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from shared.config import WEIGHT_LABELED_1, WEIGHT_UNLABELED
from shared.features import FEATURE_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_build_dataset_bin() -> Path | None:
    env = os.environ.get("BUILD_DATASET_BIN", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return p.resolve()
    for rel in ("build_dataset", "dataset_cpp/build/build_dataset"):
        p = (PROJECT_ROOT / rel).resolve()
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None


def _txn_table(rows: list[dict[str, object]], *, operaton_amt_type: pa.DataType = pa.string()) -> pa.Table:
    amt_vals = [r["operaton_amt"] for r in rows]
    cols = {
        "customer_id": pa.array([r["customer_id"] for r in rows], type=pa.string()),
        "event_id": pa.array([r["event_id"] for r in rows], type=pa.int64()),
        "event_dttm": pa.array([r["event_dttm"] for r in rows], type=pa.string()),
        "operaton_amt": pa.array(amt_vals, type=operaton_amt_type),
        "operating_system_type": pa.array([r["operating_system_type"] for r in rows], type=pa.string()),
        "device_system_version": pa.array([r["device_system_version"] for r in rows], type=pa.string()),
        "mcc_code": pa.array([r["mcc_code"] for r in rows], type=pa.string()),
        "channel_indicator_type": pa.array([r["channel_indicator_type"] for r in rows], type=pa.string()),
        "channel_indicator_sub_type": pa.array([r["channel_indicator_sub_type"] for r in rows], type=pa.string()),
        "timezone": pa.array([r["timezone"] for r in rows], type=pa.string()),
        "compromised": pa.array([r["compromised"] for r in rows], type=pa.string()),
        "web_rdp_connection": pa.array([r["web_rdp_connection"] for r in rows], type=pa.string()),
        "phone_voip_call_state": pa.array([r["phone_voip_call_state"] for r in rows], type=pa.string()),
        "session_id": pa.array([r["session_id"] for r in rows], type=pa.string()),
        "browser_language": pa.array([r["browser_language"] for r in rows], type=pa.string()),
        "event_type_nm": pa.array([r["event_type_nm"] for r in rows], type=pa.string()),
        "event_descr": pa.array([r["event_descr"] for r in rows], type=pa.string()),
    }
    return pa.table(cols)


def _row(
    *,
    customer_id: str,
    event_id: int,
    event_dttm: str,
    operaton_amt: str | float,
    session_id: str = "s1",
    mcc: str = "5411",
    os_t: str = "Android",
    dev: str = "12",
) -> dict[str, object]:
    return {
        "customer_id": customer_id,
        "event_id": event_id,
        "event_dttm": event_dttm,
        "operaton_amt": operaton_amt,
        "operating_system_type": os_t,
        "device_system_version": dev,
        "mcc_code": mcc,
        "channel_indicator_type": "mobile",
        "channel_indicator_sub_type": "app",
        "timezone": "Europe/Moscow",
        "compromised": "",
        "web_rdp_connection": "",
        "phone_voip_call_state": "",
        "session_id": session_id,
        "browser_language": "ru",
        "event_type_nm": "pay",
        "event_descr": "test",
    }


@unittest.skipUnless(_resolve_build_dataset_bin() is not None, "нет исполняемого build_dataset")
class TestBuildDatasetCpp(unittest.TestCase):
    def setUp(self) -> None:
        self._bin = _resolve_build_dataset_bin()
        assert self._bin is not None
        self._tmp = Path(tempfile.mkdtemp(prefix="guard_build_dataset_"))

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_probe_parquet_pipeline(self) -> None:
        data_train = self._tmp / "data" / "train"
        data_train.mkdir(parents=True, exist_ok=True)
        pre_path = data_train / "pretrain_part_1.parquet"
        train_path = data_train / "train_part_1.parquet"
        labels_path = self._tmp / "data" / "train_labels.parquet"
        out_dir = self._tmp / "output"
        out_candidates = [out_dir / "full_dataset.parquet", out_dir / "full_dataset"]

        pre_rows = [
            _row(
                customer_id="cust_a",
                event_id=1,
                event_dttm="2020-01-01 10:00:00",
                operaton_amt="100",
                session_id="s_pre",
            ),
            _row(
                customer_id="cust_a",
                event_id=2,
                event_dttm="2020-01-01 11:00:00",
                operaton_amt="200",
                session_id="s_pre",
            ),
        ]
        train_rows = [
            _row(customer_id="cust_a", event_id=101, event_dttm="2020-01-01 12:00:00", operaton_amt="150"),
            _row(customer_id="cust_a", event_id=102, event_dttm="2020-01-01 13:00:00", operaton_amt="160"),
            _row(customer_id="cust_b", event_id=201, event_dttm="2020-01-01 14:00:00", operaton_amt="50"),
            _row(customer_id="", event_id=999, event_dttm="2020-01-01 15:00:00", operaton_amt="1"),
        ]
        pq.write_table(_txn_table(pre_rows), pre_path)
        pq.write_table(_txn_table(train_rows), train_path)
        pq.write_table(
            pa.table({"event_id": pa.array([102], type=pa.int64()), "target": pa.array([1], type=pa.int64())}),
            labels_path,
        )

        proc = subprocess.run(
            [str(self._bin), str(self._tmp.resolve())],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)

        out_path = next((p for p in out_candidates if p.is_file()), None)
        self.assertIsNotNone(
            out_path,
            msg="ожидался output/full_dataset.parquet (или legacy output/full_dataset): " + proc.stderr + proc.stdout,
        )

        table = pq.read_table(out_path)
        names = table.column_names
        self.assertEqual(names[: len(FEATURE_NAMES)], FEATURE_NAMES)
        self.assertEqual(
            names[len(FEATURE_NAMES) :],
            ["customer_id", "event_id", "target", "sample_weight", "event_dttm"],
        )
        cust = table.column("customer_id").to_pylist()
        eid = table.column("event_id").to_pylist()
        by_eid_cust = {int(e): c for e, c in zip(eid, cust)}
        self.assertEqual(by_eid_cust[101], "cust_a")
        self.assertEqual(by_eid_cust[102], "cust_a")
        self.assertEqual(by_eid_cust[201], "cust_b")
        self.assertEqual(table.num_rows, 3, msg="три строки с непустым customer_id в train")

        event_id_col = table.column("event_id").to_pylist()
        self.assertEqual(sorted(event_id_col), [101, 102, 201])

        op_amt = table.column("operation_amt").to_pylist()
        tgt = table.column("target").to_pylist()
        wgt = table.column("sample_weight").to_pylist()
        by_id = {int(i): (o, t, w) for i, o, t, w in zip(event_id_col, op_amt, tgt, wgt)}

        o, t, w = by_id[102]
        self.assertEqual(t, 1)
        self.assertAlmostEqual(w, WEIGHT_LABELED_1)

        o, t, w = by_id[101]
        self.assertEqual(t, 0)
        self.assertAlmostEqual(w, WEIGHT_UNLABELED)

        self.assertAlmostEqual(by_id[101][0], 150.0)
        self.assertAlmostEqual(by_id[102][0], 160.0)
        self.assertAlmostEqual(by_id[201][0], 50.0)

    def test_operaton_amt_parquet_float64(self) -> None:
        """Реальные train parquet хранят operaton_amt как double, не string."""
        data_train = self._tmp / "data" / "train"
        data_train.mkdir(parents=True, exist_ok=True)
        pre_path = data_train / "pretrain_part_1.parquet"
        train_path = data_train / "train_part_1.parquet"
        labels_path = self._tmp / "data" / "train_labels.parquet"
        out_dir = self._tmp / "output"
        out_candidates = [out_dir / "full_dataset.parquet", out_dir / "full_dataset"]

        pre_rows = [
            _row(
                customer_id="cust_a",
                event_id=1,
                event_dttm="2020-01-01 10:00:00",
                operaton_amt=100.0,
                session_id="s_pre",
            ),
            _row(
                customer_id="cust_a",
                event_id=2,
                event_dttm="2020-01-01 11:00:00",
                operaton_amt=200.0,
                session_id="s_pre",
            ),
        ]
        train_rows = [
            _row(customer_id="cust_a", event_id=101, event_dttm="2020-01-01 12:00:00", operaton_amt=150.0),
            _row(customer_id="cust_a", event_id=102, event_dttm="2020-01-01 13:00:00", operaton_amt=160.0),
            _row(customer_id="cust_b", event_id=201, event_dttm="2020-01-01 14:00:00", operaton_amt=50.0),
        ]
        pq.write_table(_txn_table(pre_rows, operaton_amt_type=pa.float64()), pre_path)
        pq.write_table(_txn_table(train_rows, operaton_amt_type=pa.float64()), train_path)
        pq.write_table(
            pa.table({"event_id": pa.array([102], type=pa.int64()), "target": pa.array([1], type=pa.int64())}),
            labels_path,
        )

        proc = subprocess.run(
            [str(self._bin), str(self._tmp.resolve())],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)

        out_path = next((p for p in out_candidates if p.is_file()), None)
        self.assertIsNotNone(out_path, msg=proc.stderr + proc.stdout)

        table = pq.read_table(out_path)
        self.assertEqual(table.num_rows, 3)

        event_id_col = table.column("event_id").to_pylist()
        op_amt = table.column("operation_amt").to_pylist()
        by_id = {int(eid): amt for eid, amt in zip(event_id_col, op_amt)}
        self.assertAlmostEqual(by_id[101], 150.0)
        self.assertAlmostEqual(by_id[102], 160.0)
        self.assertAlmostEqual(by_id[201], 50.0)


if __name__ == "__main__":
    unittest.main()
