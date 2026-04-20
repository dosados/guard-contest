from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

# Keep in sync with global_category_features.hpp / C++
MCC_GLOBAL_KEY: int = -(2**63)
MCC_MISSING_KEY: int = -2
EPS = 1e-9

GLOBAL_CATEGORY_FEATURE_NAMES: tuple[str, ...] = (
    "global_mean_amount_mcc",
    "global_std_amount_mcc",
    "global_median_amount_mcc",
    "global_q25_mcc",
    "global_q75_mcc",
    "global_q95_mcc",
    "global_cnt_mcc",
    "global_cv_mcc",
    "fraud_rate_mcc",
    "fraud_count_mcc",
    "train_total_count_mcc",
    "woe_mcc",
    "amount_ratio_global_mean_mcc",
    "global_zscore_mcc",
    "inv_global_cnt_mcc",
    "global_cnt_clean_mcc",
    "global_q90_mcc",
    "global_q99_mcc",
    "global_zscore_median_iqr_mcc",
    "amount_percentile_in_mcc",
    "global_mean_amount_channel",
    "global_std_amount_channel",
    "global_median_amount_channel",
    "global_q25_channel",
    "global_q75_channel",
    "global_q95_channel",
    "global_cnt_channel",
    "global_cv_channel",
    "fraud_rate_channel",
    "fraud_count_channel",
    "train_total_count_channel",
    "woe_channel",
    "amount_ratio_global_mean_channel",
    "global_zscore_channel",
    "inv_global_cnt_channel",
    "global_cnt_clean_channel",
    "global_q90_channel",
    "global_q99_channel",
    "amount_z_vs_channel_median",
    "amount_percentile_in_channel",
    "global_mean_amount_tz_currency",
    "global_std_amount_tz_currency",
    "global_median_amount_tz_currency",
    "global_q25_tz_currency",
    "global_q75_tz_currency",
    "global_q95_tz_currency",
    "global_cnt_tz_currency",
    "global_cv_tz_currency",
    "fraud_rate_tz_currency",
    "fraud_count_tz_currency",
    "train_total_count_tz_currency",
    "woe_tz_currency",
    "global_cnt_clean_tz_currency",
    "global_q90_tz_currency",
    "global_q99_tz_currency",
    "amount_z_vs_tz_median",
    "amount_percentile_in_tz_currency",
    "inv_global_cnt_tz_currency",
    "global_zscore_median_iqr_tz_currency",
    "global_mean_amount_event_type_currency",
    "global_std_amount_event_type_currency",
    "global_median_amount_event_type_currency",
    "global_q25_event_type_currency",
    "global_q75_event_type_currency",
    "global_q95_event_type_currency",
    "global_cnt_event_type_currency",
    "global_cv_event_type_currency",
    "fraud_rate_event_type_currency",
    "fraud_count_event_type_currency",
    "train_total_count_event_type_currency",
    "woe_event_type_currency",
    "global_cnt_clean_event_type_currency",
    "global_q90_event_type_currency",
    "global_q99_event_type_currency",
    "global_type_frequency_log_event_type_currency",
    "amount_ratio_global_mean_event_type_currency",
    "global_zscore_event_type_currency",
    "inv_global_cnt_event_type_currency",
    "amount_z_vs_event_type_median",
    "amount_percentile_in_event_type_currency",
    "global_zscore_median_iqr_event_type_currency",
    "channel_rarity_neglog_in_mcc",
    "currency_freq_in_mcc",
    "timezone_freq_in_mcc",
    "surprise_mcc_given_channel_neglog",
    "mcc_not_in_channel_top3_flag",
    "global_mean_amount_event_descr",
    "global_std_amount_event_descr",
    "global_median_amount_event_descr",
    "global_q25_event_descr",
    "global_q75_event_descr",
    "global_q95_event_descr",
    "global_cnt_event_descr",
    "global_cv_event_descr",
    "fraud_rate_event_descr",
    "fraud_count_event_descr",
    "train_total_count_event_descr",
    "woe_event_descr",
    "amount_ratio_global_mean_event_descr",
    "global_zscore_event_descr",
    "inv_global_cnt_event_descr",
    "global_cnt_clean_event_descr",
    "global_q90_event_descr",
    "global_q99_event_descr",
    "amount_z_vs_event_descr_median",
    "amount_percentile_in_event_descr",
    "global_mean_amount_pos_cd",
    "global_std_amount_pos_cd",
    "global_median_amount_pos_cd",
    "global_q25_pos_cd",
    "global_q75_pos_cd",
    "global_q95_pos_cd",
    "global_cnt_pos_cd",
    "global_cv_pos_cd",
    "fraud_rate_pos_cd",
    "fraud_count_pos_cd",
    "train_total_count_pos_cd",
    "woe_pos_cd",
    "amount_ratio_global_mean_pos_cd",
    "global_zscore_pos_cd",
    "inv_global_cnt_pos_cd",
    "global_cnt_clean_pos_cd",
    "global_q90_pos_cd",
    "global_q99_pos_cd",
    "amount_z_vs_pos_cd_median",
    "amount_percentile_in_pos_cd",
    "global_mean_amount_tz_alone",
    "global_std_amount_tz_alone",
    "global_median_amount_tz_alone",
    "global_q25_tz_alone",
    "global_q75_tz_alone",
    "global_q95_tz_alone",
    "global_cnt_tz_alone",
    "global_cv_tz_alone",
    "fraud_rate_tz_alone",
    "fraud_count_tz_alone",
    "train_total_count_tz_alone",
    "woe_tz_alone",
    "amount_ratio_global_mean_tz_alone",
    "global_zscore_tz_alone",
    "inv_global_cnt_tz_alone",
    "global_cnt_clean_tz_alone",
    "global_q90_tz_alone",
    "global_q99_tz_alone",
    "amount_z_vs_tz_alone_median",
    "amount_percentile_in_tz_alone",
)

_REQUIRED_PARQUET = (
    "mcc.parquet",
    "channel_subtype.parquet",
    "timezone.parquet",
    "event_type_nm.parquet",
    "mcc_totals.parquet",
    "mcc_channel_joint.parquet",
    "mcc_currency_joint.parquet",
    "mcc_tz_joint.parquet",
    "channel_mcc_top3.parquet",
    "channel_mcc_pair.parquet",
    "event_descr.parquet",
    "pos_cd.parquet",
    "timezone_alone.parquet",
)


def _nan() -> float:
    return float("nan")


def _parse_event_type_nm_val(v: Any) -> float:
    if v is None:
        return _nan()
    if isinstance(v, bool):
        return _nan()
    if isinstance(v, (int, float, np.integer, np.floating)):
        x = float(v)
        return x if math.isfinite(x) else _nan()
    s = str(v).strip()
    if not s:
        return _nan()
    try:
        x = float(s)
        return x if math.isfinite(x) else _nan()
    except (TypeError, ValueError):
        return _nan()


def _parse_mcc_int(s: Any) -> int:
    if s is None:
        return MCC_MISSING_KEY
    if isinstance(s, bool):
        return MCC_MISSING_KEY
    if isinstance(s, (int, np.integer)):
        v = int(s)
        if v == MCC_GLOBAL_KEY or v == MCC_MISSING_KEY:
            return MCC_MISSING_KEY
        return v
    if isinstance(s, (float, np.floating)):
        x = float(s)
        if not math.isfinite(x):
            return MCC_MISSING_KEY
        r = round(x)
        if abs(x - r) > 1e-6 * (1.0 + abs(x)):
            return MCC_MISSING_KEY
        v = int(r)
        if v == MCC_GLOBAL_KEY or v == MCC_MISSING_KEY:
            return MCC_MISSING_KEY
        return v
    t = str(s).strip()
    if not t:
        return MCC_MISSING_KEY
    try:
        v = int(t, 10)
        if v == MCC_GLOBAL_KEY or v == MCC_MISSING_KEY:
            return MCC_MISSING_KEY
        return v
    except ValueError:
        pass
    try:
        x = float(t)
        if not math.isfinite(x):
            return MCC_MISSING_KEY
        r = round(x)
        if abs(x - r) > 1e-6 * (1.0 + abs(x)):
            return MCC_MISSING_KEY
        v = int(r)
        if v == MCC_GLOBAL_KEY or v == MCC_MISSING_KEY:
            return MCC_MISSING_KEY
        return v
    except ValueError:
        return MCC_MISSING_KEY


def _channel_key(ch_t: Any, ch_s: Any) -> str:
    a = (str(ch_t).strip() if ch_t is not None else "") or ""
    b = (str(ch_s).strip() if ch_s is not None else "") or ""
    if not a and not b:
        return "__MISSING__"
    return a + "\x1f" + b


def _channel_key_from_stored_parts(ch_t: Any, ch_s: Any) -> str:
    a = "" if ch_t is None or pd.isna(ch_t) else str(ch_t).strip()
    b = "" if ch_s is None or pd.isna(ch_s) else str(ch_s).strip()
    if a == "__MISSING__" and not b:
        return "__MISSING__"
    if not a and not b:
        return "__MISSING__"
    if a == "__GLOBAL__":
        return "__GLOBAL__"
    return a + "\x1f" + b


def _tz_curr_key(tz: Any, cur: Any) -> str:
    t = (str(tz).strip() if tz is not None else "") or ""
    c = (str(cur).strip() if cur is not None else "") or ""
    if not t:
        t = "__MISSING_TZ__"
    if not c:
        c = "__MISSING_CCY__"
    return t + "\x1f" + c


def _string_axis_key_missing(raw: Any) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "__MISSING__"
    if isinstance(raw, float) and not math.isfinite(float(raw)):
        return "__MISSING__"
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return "__MISSING__"
    return s


def _tz_alone_key(tz_raw: Any) -> str:
    if tz_raw is None or (isinstance(tz_raw, float) and pd.isna(tz_raw)):
        return "__MISSING_TZ_ALONE__"
    if isinstance(tz_raw, float) and not math.isfinite(float(tz_raw)):
        return "__MISSING_TZ_ALONE__"
    t = str(tz_raw).strip()
    if not t or t.lower() == "nan":
        return "__MISSING_TZ_ALONE__"
    return t


def _axis_key_from_parquet_cell(cell: Any, tz_alone_file: bool) -> str:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return "__MISSING_TZ_ALONE__" if tz_alone_file else "__MISSING__"
    a = str(cell).strip()
    if a == "__GLOBAL__":
        return "__GLOBAL__"
    if tz_alone_file:
        return "__MISSING_TZ_ALONE__" if a in ("", "__MISSING_TZ_ALONE__") else a
    return "__MISSING__" if a in ("", "__MISSING__") else a


def _et_curr_key(et: float, et_ok: bool, cur_raw: Any) -> str:
    c = (str(cur_raw).strip() if cur_raw is not None else "") or ""
    if not c:
        c = "__MISSING_CCY__"
    if not et_ok or not math.isfinite(et):
        return "__MISSING_ET__\x1f" + c
    return f"{et:.17g}\x1f{c}"


def _z_median_iqr_amt(amount: float, med: float, q25: float, q75: float) -> float:
    iqr = abs(q75 - q25)
    scale = max(iqr / 1.35, EPS)
    if not all(math.isfinite(x) for x in (amount, med, q25, q75)):
        return _nan()
    return float((amount - med) / scale)


def _amount_percentile_q(amount: float, q25: float, q75: float) -> float:
    spread = abs(q75 - q25)
    if spread < EPS:
        spread = EPS
    if not all(math.isfinite(x) for x in (amount, q25, q75)):
        return _nan()
    pc = (amount - q25) / spread
    return float(max(0.0, min(1.0, pc)))


def _neglog_smooth(num: int, den: int) -> float:
    return float(-math.log((num + 0.5) / (den + 0.5)))


def _mcc_currency_joint_key(mcc_k: int, cur_raw: Any) -> str:
    c = (str(cur_raw).strip() if cur_raw is not None else "") or ""
    if not c:
        c = "__MISSING_CCY__"
    return f"{mcc_k}\x1f{c}"


def _mcc_tz_joint_key(mcc_k: int, tz_raw: Any) -> str:
    t = (str(tz_raw).strip() if tz_raw is not None else "") or ""
    if not t:
        t = "__MISSING_TZ__"
    return f"{mcc_k}\x1f{t}"


def _put_block_mcc(amount: float, s: np.ndarray | None, out: list[float]) -> None:
    if s is None:
        out.extend([_nan()] * 20)
        return
    x = s
    mean, stdv, med, q25, q75, cnt = float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[6])
    out.extend(float(x[i]) for i in range(12))
    out.append(float(amount / mean) if math.isfinite(amount) and math.isfinite(mean) and abs(mean) > EPS else _nan())
    out.append(
        float((amount - mean) / stdv) if math.isfinite(amount) and math.isfinite(stdv) and stdv > EPS else _nan()
    )
    out.append(float(1.0 / (cnt + 1.0)) if math.isfinite(cnt) else _nan())
    out.append(float(x[12]))
    out.append(float(x[13]))
    out.append(float(x[14]))
    out.append(_z_median_iqr_amt(amount, med, q25, q75))
    out.append(_amount_percentile_q(amount, q25, q75))


def _put_block_channel(amount: float, s: np.ndarray | None, out: list[float]) -> None:
    if s is None:
        out.extend([_nan()] * 20)
        return
    x = s
    mean, stdv, med, q25, q75, cnt = float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[6])
    out.extend(float(x[i]) for i in range(12))
    out.append(float(amount / mean) if math.isfinite(amount) and math.isfinite(mean) and abs(mean) > EPS else _nan())
    out.append(
        float((amount - mean) / stdv) if math.isfinite(amount) and math.isfinite(stdv) and stdv > EPS else _nan()
    )
    out.append(float(1.0 / (cnt + 1.0)) if math.isfinite(cnt) else _nan())
    out.append(float(x[12]))
    out.append(float(x[13]))
    out.append(float(x[14]))
    out.append(
        float((amount - med) / stdv)
        if math.isfinite(amount) and math.isfinite(med) and math.isfinite(stdv) and stdv > EPS
        else _nan()
    )
    out.append(_amount_percentile_q(amount, q25, q75))


def _put_block_tz(amount: float, s: np.ndarray | None, out: list[float]) -> None:
    if s is None:
        out.extend([_nan()] * 19)
        return
    x = s
    mean, stdv, med, q25, q75, cnt = float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[6])
    out.extend(float(x[i]) for i in range(12))
    out.append(float(x[12]))
    out.append(float(x[13]))
    out.append(float(x[14]))
    out.append(
        float((amount - med) / stdv)
        if math.isfinite(amount) and math.isfinite(med) and math.isfinite(stdv) and stdv > EPS
        else _nan()
    )
    out.append(_amount_percentile_q(amount, q25, q75))
    out.append(float(1.0 / (cnt + 1.0)) if math.isfinite(cnt) else _nan())
    out.append(_z_median_iqr_amt(amount, med, q25, q75))


def _put_block_ev(amount: float, s: np.ndarray | None, out: list[float]) -> None:
    if s is None:
        out.extend([_nan()] * 22)
        return
    x = s
    mean, stdv, med, q25, q75, cnt = float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[6])
    out.extend(float(x[i]) for i in range(12))
    out.append(float(x[12]))
    out.append(float(x[13]))
    out.append(float(x[14]))
    out.append(float(x[15]))
    out.append(float(amount / mean) if math.isfinite(amount) and math.isfinite(mean) and abs(mean) > EPS else _nan())
    out.append(
        float((amount - mean) / stdv) if math.isfinite(amount) and math.isfinite(stdv) and stdv > EPS else _nan()
    )
    out.append(float(1.0 / (cnt + 1.0)) if math.isfinite(cnt) else _nan())
    out.append(
        float((amount - med) / stdv)
        if math.isfinite(amount) and math.isfinite(med) and math.isfinite(stdv) and stdv > EPS
        else _nan()
    )
    out.append(_amount_percentile_q(amount, q25, q75))
    out.append(_z_median_iqr_amt(amount, med, q25, q75))


class GlobalCategoryLookups:
    def __init__(self, aggregates_dir: Path) -> None:
        self.loaded = False
        self.mcc: dict[int, np.ndarray] = {}
        self.channel: dict[str, np.ndarray] = {}
        self.tz: dict[str, np.ndarray] = {}
        self.event: dict[str, np.ndarray] = {}
        self.axis_event_descr: dict[str, np.ndarray] = {}
        self.axis_pos_cd: dict[str, np.ndarray] = {}
        self.axis_tz_alone: dict[str, np.ndarray] = {}
        self.mcc_totals_n: dict[int, int] = {}
        self.mcc_ch_cnt: dict[str, int] = {}
        self.mcc_cur_cnt: dict[str, int] = {}
        self.mcc_tz_cnt: dict[str, int] = {}
        self.ch_mcc_pair_cnt: dict[str, int] = {}
        self.ch_total_n: dict[str, int] = {}
        self.ch_top3_mcc: dict[str, tuple[int, int, int]] = {}
        self.load(aggregates_dir)

    def load(self, aggregates_dir: Path) -> None:
        d = Path(aggregates_dir)
        missing = [str(d / name) for name in _REQUIRED_PARQUET if not (d / name).is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing required global aggregate parquet files (run build_global_aggregates first): "
                + "; ".join(missing)
            )

        self.loaded = False
        self.mcc.clear()
        self.channel.clear()
        self.tz.clear()
        self.event.clear()
        self.mcc_totals_n.clear()
        self.mcc_ch_cnt.clear()
        self.mcc_cur_cnt.clear()
        self.mcc_tz_cnt.clear()
        self.ch_mcc_pair_cnt.clear()
        self.ch_total_n.clear()
        self.ch_top3_mcc.clear()
        self.axis_event_descr.clear()
        self.axis_pos_cd.clear()
        self.axis_tz_alone.clear()

        df_m = pd.read_parquet(d / "mcc.parquet")
        for _, r in df_m.iterrows():
            k = int(r["mcc_code"])
            self.mcc[k] = np.array(
                [
                    r["global_mean_amount_mcc"],
                    r["global_std_amount_mcc"],
                    r["global_median_amount_mcc"],
                    r["global_q25_mcc"],
                    r["global_q75_mcc"],
                    r["global_q95_mcc"],
                    r["global_cnt_mcc"],
                    r["global_cv_mcc"],
                    r["fraud_rate_mcc"],
                    r["fraud_count_mcc"],
                    r["train_total_count_mcc"],
                    r["woe_mcc"],
                    r["global_cnt_clean_mcc"],
                    r["global_q90_mcc"],
                    r["global_q99_mcc"],
                ],
                dtype=np.float64,
            )

        df_c = pd.read_parquet(d / "channel_subtype.parquet")
        for _, r in df_c.iterrows():
            kt = str(r["channel_indicator_type"]).strip()
            ks = str(r["channel_indicator_subtype"]).strip() if pd.notna(r["channel_indicator_subtype"]) else ""
            key = kt if kt in ("__GLOBAL__", "__MISSING__") else kt + "\x1f" + ks
            self.channel[key] = np.array(
                [
                    r["global_mean_amount_channel"],
                    r["global_std_amount_channel"],
                    r["global_median_amount_channel"],
                    r["global_q25_channel"],
                    r["global_q75_channel"],
                    r["global_q95_channel"],
                    r["global_cnt_channel"],
                    r["global_cv_channel"],
                    r["fraud_rate_channel"],
                    r["fraud_count_channel"],
                    r["train_total_count_channel"],
                    r["woe_channel"],
                    r["global_cnt_clean_channel"],
                    r["global_q90_channel"],
                    r["global_q99_channel"],
                ],
                dtype=np.float64,
            )

        df_z = pd.read_parquet(d / "timezone.parquet")
        for _, r in df_z.iterrows():
            tzp = str(r["timezone"]).strip()
            cp = str(r["currency_iso_cd"]).strip() if pd.notna(r["currency_iso_cd"]) else ""
            key = "__GLOBAL__" if tzp == "__GLOBAL__" else tzp + "\x1f" + cp
            self.tz[key] = np.array(
                [
                    r["global_mean_amount_tz_currency"],
                    r["global_std_amount_tz_currency"],
                    r["global_median_amount_tz_currency"],
                    r["global_q25_tz_currency"],
                    r["global_q75_tz_currency"],
                    r["global_q95_tz_currency"],
                    r["global_cnt_tz_currency"],
                    r["global_cv_tz_currency"],
                    r["fraud_rate_tz_currency"],
                    r["fraud_count_tz_currency"],
                    r["train_total_count_tz_currency"],
                    r["woe_tz_currency"],
                    r["global_cnt_clean_tz_currency"],
                    r["global_q90_tz_currency"],
                    r["global_q99_tz_currency"],
                ],
                dtype=np.float64,
            )

        df_e = pd.read_parquet(d / "event_type_nm.parquet")
        for _, r in df_e.iterrows():
            cp_raw = r["currency_iso_cd"]
            if pd.isna(cp_raw):
                cp = ""
            else:
                cp = str(cp_raw).strip()
            et_raw = r["event_type_nm"]
            if pd.isna(et_raw) or (isinstance(et_raw, float) and not math.isfinite(float(et_raw))):
                et_ok = False
                etv = _nan()
            else:
                etv = float(et_raw)
                et_ok = math.isfinite(etv)
            if not et_ok and not cp:
                key = "__GLOBAL__"
            else:
                key = _et_curr_key(etv, et_ok, cp_raw)
            self.event[key] = np.array(
                [
                    r["global_mean_amount_event_type_currency"],
                    r["global_std_amount_event_type_currency"],
                    r["global_median_amount_event_type_currency"],
                    r["global_q25_event_type_currency"],
                    r["global_q75_event_type_currency"],
                    r["global_q95_event_type_currency"],
                    r["global_cnt_event_type_currency"],
                    r["global_cv_event_type_currency"],
                    r["fraud_rate_event_type_currency"],
                    r["fraud_count_event_type_currency"],
                    r["train_total_count_event_type_currency"],
                    r["woe_event_type_currency"],
                    r["global_cnt_clean_event_type_currency"],
                    r["global_q90_event_type_currency"],
                    r["global_q99_event_type_currency"],
                    r["global_type_frequency_log_event_type_currency"],
                ],
                dtype=np.float64,
            )

        df_tot = pd.read_parquet(d / "mcc_totals.parquet")
        for _, r in df_tot.iterrows():
            self.mcc_totals_n[int(r["mcc_code"])] = int(r["n_rows"])

        df_mch = pd.read_parquet(d / "mcc_channel_joint.parquet")
        for _, r in df_mch.iterrows():
            chk = _channel_key_from_stored_parts(r["channel_indicator_type"], r["channel_indicator_subtype"])
            self.mcc_ch_cnt[f'{int(r["mcc_code"])}\x1f{chk}'] = int(r["cnt"])

        df_mcur = pd.read_parquet(d / "mcc_currency_joint.parquet")
        for _, r in df_mcur.iterrows():
            self.mcc_cur_cnt[_mcc_currency_joint_key(int(r["mcc_code"]), r["currency_iso_cd"])] = int(r["cnt"])

        df_mtz = pd.read_parquet(d / "mcc_tz_joint.parquet")
        for _, r in df_mtz.iterrows():
            self.mcc_tz_cnt[_mcc_tz_joint_key(int(r["mcc_code"]), r["timezone"])] = int(r["cnt"])

        df_top = pd.read_parquet(d / "channel_mcc_top3.parquet")
        for _, r in df_top.iterrows():
            chk = _channel_key_from_stored_parts(r["channel_indicator_type"], r["channel_indicator_subtype"])
            self.ch_top3_mcc[chk] = (int(r["top1_mcc"]), int(r["top2_mcc"]), int(r["top3_mcc"]))
            self.ch_total_n[chk] = int(r["ch_row_total"])

        df_pair = pd.read_parquet(d / "channel_mcc_pair.parquet")
        for _, r in df_pair.iterrows():
            chk = _channel_key_from_stored_parts(r["channel_indicator_type"], r["channel_indicator_subtype"])
            self.ch_mcc_pair_cnt[f'{chk}\x1f{int(r["mcc_code"])}'] = int(r["cnt"])

        df_ed = pd.read_parquet(d / "event_descr.parquet")
        for _, r in df_ed.iterrows():
            key = _axis_key_from_parquet_cell(r["event_descr"], False)
            self.axis_event_descr[key] = np.array(
                [
                    r["global_mean_amount_event_descr"],
                    r["global_std_amount_event_descr"],
                    r["global_median_amount_event_descr"],
                    r["global_q25_event_descr"],
                    r["global_q75_event_descr"],
                    r["global_q95_event_descr"],
                    r["global_cnt_event_descr"],
                    r["global_cv_event_descr"],
                    r["fraud_rate_event_descr"],
                    r["fraud_count_event_descr"],
                    r["train_total_count_event_descr"],
                    r["woe_event_descr"],
                    r["global_cnt_clean_event_descr"],
                    r["global_q90_event_descr"],
                    r["global_q99_event_descr"],
                ],
                dtype=np.float64,
            )

        df_p = pd.read_parquet(d / "pos_cd.parquet")
        for _, r in df_p.iterrows():
            key = _axis_key_from_parquet_cell(r["pos_cd"], False)
            self.axis_pos_cd[key] = np.array(
                [
                    r["global_mean_amount_pos_cd"],
                    r["global_std_amount_pos_cd"],
                    r["global_median_amount_pos_cd"],
                    r["global_q25_pos_cd"],
                    r["global_q75_pos_cd"],
                    r["global_q95_pos_cd"],
                    r["global_cnt_pos_cd"],
                    r["global_cv_pos_cd"],
                    r["fraud_rate_pos_cd"],
                    r["fraud_count_pos_cd"],
                    r["train_total_count_pos_cd"],
                    r["woe_pos_cd"],
                    r["global_cnt_clean_pos_cd"],
                    r["global_q90_pos_cd"],
                    r["global_q99_pos_cd"],
                ],
                dtype=np.float64,
            )

        df_tza = pd.read_parquet(d / "timezone_alone.parquet")
        for _, r in df_tza.iterrows():
            key = _axis_key_from_parquet_cell(r["timezone"], True)
            self.axis_tz_alone[key] = np.array(
                [
                    r["global_mean_amount_tz_alone"],
                    r["global_std_amount_tz_alone"],
                    r["global_median_amount_tz_alone"],
                    r["global_q25_tz_alone"],
                    r["global_q75_tz_alone"],
                    r["global_q95_tz_alone"],
                    r["global_cnt_tz_alone"],
                    r["global_cv_tz_alone"],
                    r["fraud_rate_tz_alone"],
                    r["fraud_count_tz_alone"],
                    r["train_total_count_tz_alone"],
                    r["woe_tz_alone"],
                    r["global_cnt_clean_tz_alone"],
                    r["global_q90_tz_alone"],
                    r["global_q99_tz_alone"],
                ],
                dtype=np.float64,
            )

        if (
            not self.mcc
            or not self.channel
            or not self.tz
            or not self.event
            or not self.axis_event_descr
            or not self.axis_pos_cd
            or not self.axis_tz_alone
        ):
            raise RuntimeError(f"global aggregates: main tables empty after reading {d}")
        if MCC_GLOBAL_KEY not in self.mcc:
            raise RuntimeError(f"global aggregates: missing MCC global fallback row in {d / 'mcc.parquet'}")

        self.loaded = True

    def _mcc_row(self, k: int) -> np.ndarray | None:
        s = self.mcc.get(k)
        if s is not None:
            return s
        return self.mcc.get(MCC_GLOBAL_KEY)

    def _ch_row(self, k: str) -> np.ndarray | None:
        s = self.channel.get(k)
        if s is not None:
            return s
        return self.channel.get("__GLOBAL__")

    def _tz_row(self, k: str) -> np.ndarray | None:
        s = self.tz.get(k)
        if s is not None:
            return s
        return self.tz.get("__GLOBAL__")

    def _ev_row(self, k: str) -> np.ndarray | None:
        s = self.event.get(k)
        if s is not None:
            return s
        return self.event.get("__GLOBAL__")

    def _ed_row(self, k: str) -> np.ndarray | None:
        s = self.axis_event_descr.get(k)
        if s is not None:
            return s
        return self.axis_event_descr.get("__GLOBAL__")

    def _pos_row(self, k: str) -> np.ndarray | None:
        s = self.axis_pos_cd.get(k)
        if s is not None:
            return s
        return self.axis_pos_cd.get("__GLOBAL__")

    def _tz_alone_row(self, k: str) -> np.ndarray | None:
        s = self.axis_tz_alone.get(k)
        if s is not None:
            return s
        return self.axis_tz_alone.get("__GLOBAL__")

    def _put_joint(self, out: list[float], mcc_k: int, chk: str, cur_raw: Any, tz_raw: Any) -> None:
        t = self.mcc_totals_n.get(mcc_k, 0)
        j_ch = self.mcc_ch_cnt.get(f"{mcc_k}\x1f{chk}", 0)
        out.append(_neglog_smooth(j_ch, t))
        j_cur = self.mcc_cur_cnt.get(_mcc_currency_joint_key(mcc_k, cur_raw), 0)
        out.append(float(j_cur / t) if t > 0 else _nan())
        j_tz = self.mcc_tz_cnt.get(_mcc_tz_joint_key(mcc_k, tz_raw), 0)
        out.append(float(j_tz / t) if t > 0 else _nan())
        pair_cnt = self.ch_mcc_pair_cnt.get(f"{chk}\x1f{mcc_k}", 0)
        ch_tot = self.ch_total_n.get(chk, 0)
        out.append(_neglog_smooth(pair_cnt, ch_tot))
        top3 = self.ch_top3_mcc.get(chk)
        if top3 is None:
            out.append(_nan())
        else:
            m1, m2, m3 = top3
            out.append(0.0 if mcc_k in (m1, m2, m3) else 1.0)

    def features_for_row(self, row: Mapping[str, Any], operation_amt: float) -> dict[str, float]:
        if not self.loaded:
            raise RuntimeError("GlobalCategoryLookups: aggregates not loaded")

        amount = float(operation_amt) if math.isfinite(operation_amt) else _nan()
        mcc_k = _parse_mcc_int(row.get("mcc_code"))
        ch_t = row.get("channel_indicator_type")
        ch_s = row.get("channel_indicator_sub_type")
        if ch_s is None:
            ch_s = row.get("channel_indicator_subtype")
        ck = _channel_key(ch_t, ch_s)
        tz_k = _tz_curr_key(row.get("timezone"), row.get("currency_iso_cd"))

        etn = _parse_event_type_nm_val(row.get("event_type_nm"))
        et_ok = math.isfinite(etn)
        ek = _et_curr_key(etn, et_ok, row.get("currency_iso_cd"))

        out_list: list[float] = []
        _put_block_mcc(amount, self._mcc_row(mcc_k), out_list)
        _put_block_channel(amount, self._ch_row(ck), out_list)
        _put_block_tz(amount, self._tz_row(tz_k), out_list)
        _put_block_ev(amount, self._ev_row(ek), out_list)
        self._put_joint(out_list, mcc_k, ck, row.get("currency_iso_cd"), row.get("timezone"))

        ed_raw = row.get("event_descr")
        if ed_raw is None or (isinstance(ed_raw, str) and not str(ed_raw).strip()):
            ed_raw = row.get("event_desc")
        ed_k = _string_axis_key_missing(ed_raw)
        pos_k = _string_axis_key_missing(row.get("pos_cd"))
        tz_ak = _tz_alone_key(row.get("timezone"))

        _put_block_channel(amount, self._ed_row(ed_k), out_list)
        _put_block_channel(amount, self._pos_row(pos_k), out_list)
        _put_block_channel(amount, self._tz_alone_row(tz_ak), out_list)

        if len(out_list) != len(GLOBAL_CATEGORY_FEATURE_NAMES):
            raise RuntimeError(
                f"global category feature count mismatch: got {len(out_list)}, expected {len(GLOBAL_CATEGORY_FEATURE_NAMES)}"
            )
        return dict(zip(GLOBAL_CATEGORY_FEATURE_NAMES, out_list))


def default_aggregates_dir(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parent.parent
    return root / "output" / "datasets" / "global_aggregates"
