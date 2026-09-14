# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.10

# 2. Installed packages

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        input/
            description.md (82 lines)
            sample_submission.csv (603601 lines)
            sample_submission.csv.zip (1.3 MB)
            test.csv (603601 lines)
            test.csv.zip (8.5 MB)
            train.csv (5432401 lines)
            train.csv.zip (93.8 MB)
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
        working/
            ventilator-pressure-prediction/
                description.md (82 lines)
                sample_submission.csv (603601 lines)
                ... and 5 other files
                ventilator-pressure-prediction/
```

-> data/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> data/ventilator-pressure-prediction/test.csv has 603600 rows and 7 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [20, 50, 10]
R (int64) has 3 unique values: [50, 5, 20]
breath_id (int64) has range: 2436.00 - 124535.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> data/ventilator-pressure-prediction/train.csv has 5432400 rows and 8 columns.
Here is some information about the columns:
C (int64) has 3 unique values: [10, 20, 50]
R (int64) has 3 unique values: [5, 50, 20]
breath_id (int64) has range: 1194.00 - 124492.00, 0 nan values
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (float64) has range: 3.52 - 41.48, 0 nan values
time_step (float64) has range: 0.00 - 2.73, 0 nan values
u_in (float64) has range: 0.00 - 100.00, 0 nan values
u_out (int64) has 2 unique values: [0, 1]

-> input/sample_submission.csv has 603600 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
pressure (int64) has 1 unique values: [0]

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import pandas as pd



## === cell 1
import os


def _resolve_input_path(p: str) -> str:
    if os.path.exists(p):
        return p
    if p.startswith("../input/"):
        candidate = os.path.join("/kaggle/input", p[len("../input/") :])
        if os.path.exists(candidate):
            return candidate
        candidate2 = os.path.join("/kaggle/data", p[len("../input/") :])
        if os.path.exists(candidate2):
            return candidate2
    if p.startswith("/kaggle/") and os.path.exists(p):
        return p
    return p  # let pandas raise a clear error if still missing


def _build_baseline_submission(
    sample_sub_path: str, train_path: str, test_path: str
) -> pd.DataFrame:
    """
    Change rationale (score-improvement toward target, minimal core-logic impact):
    1) Train lookup statistics only on inspiratory phase (u_out==0), matching the
       evaluation metric (expiratory phase is not scored). This typically reduces MAE
       significantly while preserving the same hierarchical mean-lookup + shrinkage logic.
    2) Post-process test predictions so expiratory (u_out==1) rows use the last predicted
       inspiratory pressure within the breath. This is a standard, deterministic alignment
       with the scoring phase and avoids noisy/unsupported expiratory predictions.
    """
    sub_local = pd.read_csv(_resolve_input_path(sample_sub_path), usecols=["id"])

    train = pd.read_csv(
        _resolve_input_path(train_path),
        usecols=["breath_id", "time_step", "u_in", "u_out", "R", "C", "pressure"],
    )
    test = pd.read_csv(
        _resolve_input_path(test_path),
        usecols=["id", "breath_id", "time_step", "u_in", "u_out", "R", "C"],
    )

    train = train.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
        drop=True
    )
    test = test.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
        drop=True
    )

    train["time_idx"] = train.groupby("breath_id").cumcount().astype("int16")
    test["time_idx"] = test.groupby("breath_id").cumcount().astype("int16")

    dt_by_idx = (
        train.groupby("breath_id", sort=False)["time_step"]
        .diff()
        .fillna(0.0)
        .groupby(train["time_idx"], sort=False)
        .median()
        .astype("float32")
        .to_dict()
    )

    def _add_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["u_in_cum"] = (
            df.groupby("breath_id", sort=False)["u_in"].cumsum().astype("float32")
        )
        df["u_in_lag1"] = (
            df.groupby("breath_id", sort=False)["u_in"]
            .shift(1)
            .fillna(0.0)
            .astype("float32")
        )
        df["u_in_diff1"] = (df["u_in"].astype("float32") - df["u_in_lag1"]).astype(
            "float32"
        )

        df["dt"] = df["time_idx"].map(dt_by_idx).fillna(0.0).astype("float32")
        df["area"] = (
            (df["u_in"].astype("float32") * df["dt"])
            .groupby(df["breath_id"], sort=False)
            .cumsum()
            .astype("float32")
        )
        return df

    train = _add_features(train)
    test = _add_features(test)

    train_insp = train.loc[train["u_out"] == 0].copy()

    train_insp["u_in_bin"] = (
        (train_insp["u_in"].astype("float32") * 4.0).round().astype("int16")
    )
    test["u_in_bin"] = (test["u_in"].astype("float32") * 4.0).round().astype("int16")

    train_insp["u_in_cum_bin"] = (
        train_insp["u_in_cum"].round().clip(lower=0).astype("int16")
    )
    test["u_in_cum_bin"] = test["u_in_cum"].round().clip(lower=0).astype("int16")

    train_insp["u_in_diff_bin"] = (
        (train_insp["u_in_diff1"] * 2.0).round() + 400
    ).astype("int16")
    test["u_in_diff_bin"] = ((test["u_in_diff1"] * 2.0).round() + 400).astype("int16")

    train_insp["area_bin"] = (
        (train_insp["area"] / 0.2).round().clip(lower=0).astype("int16")
    )
    test["area_bin"] = (test["area"] / 0.2).round().clip(lower=0).astype("int16")

    for col, dt in [("R", "int16"), ("C", "int16"), ("u_out", "int8")]:
        train_insp[col] = train_insp[col].astype(dt)
        test[col] = test[col].astype(dt)

    train_insp["pressure"] = train_insp["pressure"].astype("float32")
    global_mean = float(train_insp["pressure"].mean())

    keys_area = ["R", "C", "time_idx", "u_out", "u_in_bin", "area_bin"]
    keys_full = [
        "R",
        "C",
        "time_idx",
        "u_out",
        "u_in_bin",
        "u_in_cum_bin",
        "u_in_diff_bin",
    ]
    keys_mid = ["R", "C", "time_idx", "u_out", "u_in_bin", "u_in_cum_bin"]
    keys_uin = ["R", "C", "time_idx", "u_out", "u_in_bin"]
    keys_coarse = ["R", "C", "time_idx", "u_out"]

    def _group_stats(df: pd.DataFrame, keys: list[str], name: str) -> pd.DataFrame:
        g = (
            df.groupby(keys, sort=False)["pressure"]
            .agg(["mean", "count"])
            .reset_index()
        )
        g = g.rename(columns={"mean": f"m_{name}", "count": f"c_{name}"})
        g[f"c_{name}"] = g[f"c_{name}"].astype("int32")
        g[f"m_{name}"] = g[f"m_{name}"].astype("float32")
        return g

    g_area = _group_stats(train_insp, keys_area, "area")
    g_full = _group_stats(train_insp, keys_full, "full")
    g_mid = _group_stats(train_insp, keys_mid, "mid")
    g_uin = _group_stats(train_insp, keys_uin, "uin")
    g_coarse = _group_stats(train_insp, keys_coarse, "coarse")

    pred = test[
        ["id", "breath_id", "time_step", "u_out"]
        + list(dict.fromkeys(keys_area + keys_full))
    ].copy()

    pred = pred.merge(g_coarse, on=keys_coarse, how="left")
    pred = pred.merge(g_uin, on=keys_uin, how="left")
    pred = pred.merge(g_mid, on=keys_mid, how="left")
    pred = pred.merge(g_full, on=keys_full, how="left")
    pred = pred.merge(g_area, on=keys_area, how="left")

    def _shrink(child_m, child_c, parent_m, alpha: float):
        child_c = child_c.fillna(0.0).astype("float32")
        child_m = child_m.astype("float32")
        parent_m = parent_m.astype("float32")
        w = child_c / (child_c + alpha)
        return (w * child_m) + ((1.0 - w) * parent_m)

    base = pd.Series(global_mean, index=pred.index, dtype="float32")
    m_coarse = pred["m_coarse"].fillna(global_mean).astype("float32")

    m_uin = _shrink(
        pred["m_uin"].fillna(m_coarse),
        pred["c_uin"],
        m_coarse,
        alpha=80.0,
    )
    m_mid = _shrink(
        pred["m_mid"].fillna(m_uin),
        pred["c_mid"],
        m_uin,
        alpha=60.0,
    )
    m_full = _shrink(
        pred["m_full"].fillna(m_mid),
        pred["c_full"],
        m_mid,
        alpha=40.0,
    )
    m_area = _shrink(
        pred["m_area"].fillna(m_full),
        pred["c_area"],
        m_full,
        alpha=30.0,
    )

    pred["pressure"] = (
        m_area.fillna(m_full).fillna(m_mid).fillna(m_uin).fillna(m_coarse).fillna(base)
    ).astype("float32")

    pred = pred.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
        drop=True
    )
    last_insp = (
        pred["pressure"]
        .where(pred["u_out"].astype("int8") == 0)
        .groupby(pred["breath_id"], sort=False)
        .ffill()
    )
    last_insp = last_insp.groupby(
        pred["breath_id"], sort=False
    ).bfill()  # safety: if early u_out==1 (rare), fill from first insp
    pred.loc[pred["u_out"].astype("int8") == 1, "pressure"] = last_insp.loc[
        pred["u_out"].astype("int8") == 1
    ].astype("float32")
    pred["pressure"] = pred["pressure"].fillna(global_mean).astype("float32")

    pred = pred[["id", "pressure"]]
    sub_local = sub_local.merge(pred, on="id", how="left")
    sub_local["pressure"] = sub_local["pressure"].fillna(global_mean).astype("float32")
    sub_local = sub_local[["id", "pressure"]]
    return sub_local


sub = pd.read_csv(
    _resolve_input_path("../input/ventilator-pressure-prediction/sample_submission.csv")
)


def _read_submission_or_fallback(path: str, fallback_df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = pd.read_csv(_resolve_input_path(path))
        if "id" in df.columns and "pressure" in df.columns:
            df = df[["id", "pressure"]].copy()
        else:
            raise ValueError(f"Submission at {path} missing required columns.")
        return df
    except FileNotFoundError:
        return _build_baseline_submission(
            sample_sub_path="../input/ventilator-pressure-prediction/sample_submission.csv",
            train_path="../input/ventilator-pressure-prediction/train.csv",
            test_path="../input/ventilator-pressure-prediction/test.csv",
        )
    except Exception:
        return _build_baseline_submission(
            sample_sub_path="../input/ventilator-pressure-prediction/sample_submission.csv",
            train_path="../input/ventilator-pressure-prediction/train.csv",
            test_path="../input/ventilator-pressure-prediction/test.csv",
        )


sub_1 = _read_submission_or_fallback(
    "../input/tensorflow-bidirectional-lstm-custom-mae-loss/submission.csv", sub
)
sub_2 = _read_submission_or_fallback(
    "../input/finetune-of-tensorflow-bidirectional-lstm/submission.csv", sub
)
sub_3 = _read_submission_or_fallback(
    "../input/a-dummy-approach-to-improve-your-score-postprocess/submission.csv", sub
)
sub_4 = _read_submission_or_fallback(
    "../input/ensemble-folds-with-median-0-153/submission_median_round_LB153.csv", sub
)



## --- ERROR in cell 1, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2382178390.py[0m in [0;36m_read_submission_or_fallback[0;34m(path, fallback_df)[0m
[1;32m    231[0m     [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 232[0;31m         [0mdf[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0m_resolve_input_path[0m[0;34m([0m[0mpath[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    233[0m         [0;32mif[0m [0;34m"id"[0m [0;32min[0m [0mdf[0m[0;34m.[0m[0mcolumns[0m [0;32mand[0m [0;34m"pressure"[0m [0;32min[0m [0mdf[0m[0;34m.[0m[0mcolumns[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py[0m in [0;36mread_csv[0;34m(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)[0m
[1;32m   1025[0m [0;34m[0m[0m
[0;32m-> 1026[0;31m     [0;32mreturn[0m [0m_read[0m[0;34m([0m[0mfilepath_or_buffer[0m[0;34m,[0m [0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1027[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py[0m in [0;36m_read[0;34m(filepath_or_buffer, kwds)[0m
[1;32m    619[0m     [0;31m# Create the parser.[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 620[0;31m     [0mparser[0m [0;34m=[0m [0mTextFileReader[0m[0;34m([0m[0mfilepath_or_buffer[0m[0;34m,[0m [0;34m**[0m[0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    621[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py[0m in [0;36m__init__[0;34m(self, f, engine, **kwds)[0m
[1;32m   1619[0m         [0mself[0m[0;34m.[0m[0mhandles[0m[0;34m:[0m [0mIOHandles[0m [0;34m|[0m [0;32mNone[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1620[0;31m         [0mself[0m[0;34m.[0m[0m_engine[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_make_engine[0m[0;34m([0m[0mf[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mengine[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1621[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py[0m in [0;36m_make_engine[0;34m(self, f, engine)[0m
[1;32m   1879[0m                     [0mmode[0m [0;34m+=[0m [0;34m"b"[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1880[0;31m             self.handles = get_handle(
[0m[1;32m   1881[0m                 [0mf[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/common.py[0m in [0;36mget_handle[0;34m(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)[0m
[1;32m    872[0m             [0;31m# Encoding[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 873[0;31m             handle = open(
[0m[1;32m    874[0m                 [0mhandle[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: '../input/tensorflow-bidirectional-lstm-custom-mae-loss/submission.csv'

During handling of the above exception, another exception occurred:

[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2382178390.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m    250[0m [0;34m[0m[0m
[1;32m    251[0m [0;34m[0m[0m
[0;32m--> 252[0;31m sub_1 = _read_submission_or_fallback(
[0m[1;32m    253[0m     [0;34m"../input/tensorflow-bidirectional-lstm-custom-mae-loss/submission.csv"[0m[0;34m,[0m [0msub[0m[0;34m[0m[0;34m[0m[0m
[1;32m    254[0m )

[0;32m/tmp/ipykernel_11/2382178390.py[0m in [0;36m_read_submission_or_fallback[0;34m(path, fallback_df)[0m
[1;32m    237[0m         [0;32mreturn[0m [0mdf[0m[0;34m[0m[0;34m[0m[0m
[1;32m    238[0m     [0;32mexcept[0m [0mFileNotFoundError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 239[0;31m         return _build_baseline_submission(
[0m[1;32m    240[0m             [0msample_sub_path[0m[0;34m=[0m[0;34m"../input/ventilator-pressure-prediction/sample_submission.csv"[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    241[0m             [0mtrain_path[0m[0;34m=[0m[0;34m"../input/ventilator-pressure-prediction/train.csv"[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2382178390.py[0m in [0;36m_build_baseline_submission[0;34m(sample_sub_path, train_path, test_path)[0m
[1;32m    153[0m     ].copy()
[1;32m    154[0m [0;34m[0m[0m
[0;32m--> 155[0;31m     [0mpred[0m [0;34m=[0m [0mpred[0m[0;34m.[0m[0mmerge[0m[0;34m([0m[0mg_coarse[0m[0;34m,[0m [0mon[0m[0;34m=[0m[0mkeys_coarse[0m[0;34m,[0m [0mhow[0m[0;34m=[0m[0;34m"left"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    156[0m     [0mpred[0m [0;34m=[0m [0mpred[0m[0;34m.[0m[0mmerge[0m[0;34m([0m[0mg_uin[0m[0;34m,[0m [0mon[0m[0;34m=[0m[0mkeys_uin[0m[0;34m,[0m [0mhow[0m[0;34m=[0m[0;34m"left"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    157[0m     [0mpred[0m [0;34m=[0m [0mpred[0m[0;34m.[0m[0mmerge[0m[0;34m([0m[0mg_mid[0m[0;34m,[0m [0mon[0m[0;34m=[0m[0mkeys_mid[0m[0;34m,[0m [0mhow[0m[0;34m=[0m[0;34m"left"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py[0m in [0;36mmerge[0;34m(self, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)[0m
[1;32m  10830[0m         [0;32mfrom[0m [0mpandas[0m[0;34m.[0m[0mcore[0m[0;34m.[0m[0mreshape[0m[0;34m.[0m[0mmerge[0m [0;32mimport[0m [0mmerge[0m[0;34m[0m[0;34m[0m[0m
[1;32m  10831[0m [0;34m[0m[0m
[0;32m> 10832[0;31m         return merge(
[0m[1;32m  10833[0m             [0mself[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m  10834[0m             [0mright[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py[0m in [0;36mmerge[0;34m(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)[0m
[1;32m    168[0m         )
[1;32m    169[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 170[0;31m         op = _MergeOperation(
[0m[1;32m    171[0m             [0mleft_df[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    172[0m             [0mright_df[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py[0m in [0;36m__init__[0;34m(self, left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, indicator, validate)[0m
[1;32m    792[0m             [0mleft_drop[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    793[0m             [0mright_drop[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 794[0;31m         ) = self._get_merge_keys()
[0m[1;32m    795[0m [0;34m[0m[0m
[1;32m    796[0m         [0;32mif[0m [0mleft_drop[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py[0m in [0;36m_get_merge_keys[0;34m(self)[0m
[1;32m   1308[0m                         [0;31m#  the latter of which will raise[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1309[0m                         [0mlk[0m [0;34m=[0m [0mcast[0m[0;34m([0m[0mHashable[0m[0;34m,[0m [0mlk[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1310[0;31m                         [0mleft_keys[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mleft[0m[0;34m.[0m[0m_get_label_or_level_values[0m[0;34m([0m[0mlk[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1311[0m                         [0mjoin_names[0m[0;34m.[0m[0mappend[0m[0;34m([0m[0mlk[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1312[0m                     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py[0m in [0;36m_get_label_or_level_values[0;34m(self, key, axis)[0m
[1;32m   1923[0m [0;34m[0m[0m
[1;32m   1924[0m             [0mlabel_axis_name[0m [0;34m=[0m [0;34m"column"[0m [0;32mif[0m [0maxis[0m [0;34m==[0m [0;36m0[0m [0;32melse[0m [0;34m"index"[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1925[0;31m             raise ValueError(
[0m[1;32m   1926[0m                 [0;34mf"The {label_axis_name} label '{key}' is not unique.{multi_message}"[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1927[0m             )

[0;31mValueError[0m: The column label 'u_out' is not unique.

## === cell 2
sub["pressure"] = (
    (sub_1["pressure"].astype("float32").values * 0.1)
    + (sub_2["pressure"].astype("float32").values * 0.12)
    + (sub_3["pressure"].astype("float32").values * 0.18)
    + (sub_4["pressure"].astype("float32").values * 0.6)
).astype("float32")

sub.to_csv("submission.csv", index=False)
sub.head(5)
