# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given time series of breaths, predict the airway pressure in the respiratory circuit during the breath, given the time series of control inputs.

The best submissions will take lung attributes compliance and resistance into account.

## Metric
Mean absolute error between the predicted and actual pressures during the inspiratory phase of each breath. The expiratory phase is not scored.

## Submission Format
For each `id` in the test set, you must predict a value for the `pressure` variable. The file should contain a header and have the following format:

```
id,pressure
1,20
2,23
3,24
etc.
```

## Dataset
The ventilator data used in this competition was produced using a modified [open-source ventilator](https://pvp.readthedocs.io/) connected to an [artificial bellows test lung](https://www.ingmarmed.com/product/quicklung/) via a respiratory circuit. The diagram below illustrates the setup, with the two control inputs highlighted in green and the state variable (airway pressure) to predict in blue. The first control input is a continuous variable from 0 to 100 representing the percentage the inspiratory solenoid valve is open to let air into the lung (i.e., 0 is completely closed and no air is let in and 100 is completely open). The second control input is a binary variable representing whether the exploratory valve is open (1) or closed (0) to let air out.

![Ventilator diagram](https://raw.githubusercontent.com/google/deluca-lung/main/assets/2020-10-02%20Ventilator%20diagram.svg)

Each time series represents an approximately 3-second breath. The files are organized such that each row is a time step in a breath and gives the two control signals, the resulting airway pressure, and relevant attributes of the lung, described below.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `id` - globally-unique time step identifier across an entire file
- `breath_id` - globally-unique time step for breaths
- `R` - lung attribute indicating how restricted the airway is (in cmH2O/L/S). Physically, this is the change in pressure per change in flow (air volume per time). Intuitively, one can imagine blowing up a balloon through a straw. We can change `R` by changing the diameter of the straw, with higher `R` being harder to blow.
- `C` - lung attribute indicating how compliant the lung is (in mL/cmH2O). Physically, this is the change in volume per change in pressure. Intuitively, one can imagine the same balloon example. We can change `C` by changing the thickness of the balloon’s latex, with higher `C` having thinner latex and easier to blow.
- `time_step` - the actual time stamp.
- `u_in` - the control input for the inspiratory solenoid valve. Ranges from 0 to 100.
- `u_out` - the control input for the exploratory solenoid valve. Either 0 or 1.
- `pressure` - the airway pressure measured in the respiratory circuit, measured in cmH2O.

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

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

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import copy
import glob
import random
from random import random as rd
import gc



## === cell 1
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SAMPLE_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"

train_usecols = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
train_dtypes = {
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
df_train = pd.read_csv(TRAIN_PATH, usecols=train_usecols, dtype=train_dtypes)

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)


def find_nearest(prediction):
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        return sorted_pressures[-1]
    elif insert_idx == 0:
        return sorted_pressures[0]
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return (
        lower_val
        if abs(lower_val - prediction) < abs(upper_val - prediction)
        else upper_val
    )


def snap_to_nearest_pressure_vec(pred_arr: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred_arr, dtype=np.float64)
    idx = np.searchsorted(sorted_pressures, pred, side="left")
    idx1 = np.clip(idx, 0, total_pressures_len - 1)
    idx0 = np.clip(idx - 1, 0, total_pressures_len - 1)
    p0 = sorted_pressures[idx0]
    p1 = sorted_pressures[idx1]
    choose1 = np.abs(pred - p1) < np.abs(pred - p0)  # tie -> p0 (same as find_nearest)
    return np.where(choose1, p1, p0)


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    """
    Original intent: read one or two submissions and do a simple weighted combo.
    Bugfix: make parsing robust; if the filename does not contain a score token,
    just treat both weights equally (this preserves the blending idea without crashing).
    """
    l = []
    arrs = []
    for i in range(len(input_list)):
        fp = input_list[i]
        try:
            public_lb_score = int(fp.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1
        l.append(public_lb_score)

        dfp = pd.read_csv(fp)
        if "pressure" not in dfp.columns:
            raise ValueError(f"File {fp} does not have 'pressure' column.")
        arrs.append(dfp["pressure"].to_numpy().ravel())

    l_sum = sum(l) if sum(l) != 0 else len(l)

    if len(arrs) == 1:
        return arrs[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        return arrs[0] * weight1 + arrs[1] * weight2


def _safe_list_pred_files(dp):
    """
    Bugfix: in this environment the external dataset path may not exist.
    Only return readable .csv files; ignore directories/non-csv.
    """
    if dp is None:
        return []
    if not os.path.exists(dp):
        return []
    files = []
    for p in glob.iglob(f"{dp}/*"):
        if os.path.isfile(p) and p.lower().endswith(".csv"):
            files.append(p)
    files.sort()
    return files


def _load_pressure_vector(fp, expected_len):
    """
    Load a submission-like CSV and return pressure vector if shape matches.
    Skip invalid files rather than breaking the run.
    """
    try:
        dfp = pd.read_csv(fp, usecols=["pressure"])
        v = dfp["pressure"].to_numpy().ravel()
        if len(v) != expected_len:
            return None
        return v
    except Exception:
        return None


def _add_time_index(df):
    """
    Avoid float-key mismatch on time_step by using the natural per-breath step index (0..79).
    """
    df = df.sort_values(["breath_id", "time_step"], kind="mergesort").copy()
    df["t_idx"] = df.groupby("breath_id", sort=False).cumcount().astype(np.int16)
    return df


def _quantize_to_int(series, scale=10.0, clip_min=-5000, clip_max=50000):
    """
    Replace fragile float merge keys with integer bins.
    scale=10 => 0.1 resolution.
    """
    x = np.rint(series.to_numpy(dtype=np.float64, copy=False) * scale)
    x = np.clip(x, clip_min, clip_max)
    return x.astype(np.int16)


def _add_lag_features_fast(df, lags=(1, 2, 3, 5)):
    df = df.sort_values(["breath_id", "time_step"], kind="mergesort").copy()
    g_uin = df.groupby("breath_id", sort=False)["u_in"]
    g_uout = df.groupby("breath_id", sort=False)["u_out"]
    g_ts = df.groupby("breath_id", sort=False)["time_step"]

    for k in lags:
        df[f"u_in_lag{k}"] = g_uin.shift(k)
        df[f"u_out_lag{k}"] = g_uout.shift(k)
        df[f"dt_lag{k}"] = g_ts.diff(k)

    df["u_in_diff1"] = g_uin.diff(1)
    df["u_in_diff2"] = g_uin.diff(2)
    df["u_in_cumsum"] = g_uin.cumsum()
    return df.fillna(0.0)


def _nn_lookup_1d_by_group(train_df, test_df, group_cols, xcol, ycol, outcol):
    """
    Deterministic "nearest-neighbor in 1D within group" lookup using searchsorted.
    """
    out = test_df.copy()
    out[outcol] = np.nan

    if len(train_df) == 0 or len(test_df) == 0:
        return out

    tr = train_df[group_cols + [xcol, ycol]].copy()
    te = test_df[group_cols + [xcol]].copy()

    for key, te_g in te.groupby(group_cols, sort=False):
        if len(group_cols) == 1:
            tr_g = tr.loc[tr[group_cols[0]].eq(key)]
        else:
            mask = np.ones(len(tr), dtype=bool)
            for col, val in zip(group_cols, key):
                mask &= tr[col].to_numpy() == val
            tr_g = tr.loc[mask]

        if len(tr_g) == 0:
            continue

        x_tr = tr_g[xcol].to_numpy()
        y_tr = tr_g[ycol].to_numpy()

        order = np.argsort(x_tr, kind="mergesort")
        x_tr = x_tr[order]
        y_tr = y_tr[order]

        x_te = te_g[xcol].to_numpy()
        idx = np.searchsorted(x_tr, x_te, side="left")
        idx0 = np.clip(idx - 1, 0, len(x_tr) - 1)
        idx1 = np.clip(idx, 0, len(x_tr) - 1)

        d0 = np.abs(x_te - x_tr[idx0])
        d1 = np.abs(x_te - x_tr[idx1])
        choose1 = d1 < d0  # tie -> idx0

        pick = np.where(choose1, idx1, idx0)
        out.loc[te_g.index, outcol] = y_tr[pick]

    return out


def _fallback_pressure_lookup_submission(output_df, u_in_round=1):
    """
    Deterministic lookup with backoffs, then snap to nearest valid pressure.
    Inspiratory-only train lookups (u_out==0). Test u_out==1 rows set to 0.0.
    """

    test_usecols = ["id", "breath_id", "R", "C", "time_step", "u_out", "u_in"]
    test_dtypes = {
        "id": "int32",
        "breath_id": "int32",
        "R": "int16",
        "C": "int16",
        "time_step": "float32",
        "u_in": "float32",
        "u_out": "int8",
    }
    test = pd.read_csv(TEST_PATH, usecols=test_usecols, dtype=test_dtypes)

    train_insp = df_train.loc[df_train["u_out"].eq(0)].copy()

    train_insp = _add_time_index(train_insp)
    test = _add_time_index(test)

    train_insp = _add_lag_features_fast(train_insp, lags=(1, 2, 3, 5))
    test = _add_lag_features_fast(test, lags=(1, 2, 3, 5))

    q_cols = [
        "u_in",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_lag3",
        "u_in_lag5",
        "u_in_diff1",
        "u_in_diff2",
        "u_in_cumsum",
    ]
    for c in q_cols:
        train_insp[c + "_q"] = _quantize_to_int(train_insp[c], scale=10.0)
        test[c + "_q"] = _quantize_to_int(test[c], scale=10.0)

    global_med = float(train_insp["pressure"].median())

    grp0 = [
        "R",
        "C",
        "t_idx",
        "u_out",
        "u_in_lag1_q",
        "u_in_lag2_q",
        "u_in_lag3_q",
        "u_in_lag5_q",
        "u_in_diff1_q",
    ]
    grp1 = [
        "R",
        "C",
        "t_idx",
        "u_out",
        "u_in_lag1_q",
        "u_in_lag2_q",
        "u_in_diff1_q",
        "u_in_cumsum_q",
        "u_out_lag1",
        "u_out_lag2",
    ]
    grp2 = [
        "R",
        "C",
        "t_idx",
        "u_out",
        "u_in_lag1_q",
        "u_in_lag2_q",
        "u_in_diff1_q",
        "u_in_cumsum_q",
    ]

    m = test.copy()
    m = _nn_lookup_1d_by_group(train_insp, m, grp0, "u_in_q", "pressure", "pq0")
    m = _nn_lookup_1d_by_group(train_insp, m, grp1, "u_in_q", "pressure", "pq1")
    m = _nn_lookup_1d_by_group(train_insp, m, grp2, "u_in_q", "pressure", "pq2")

    key_q = ["R", "C", "t_idx", "u_out", "u_in_q"]
    key_qb = ["R", "C", "t_idx", "u_out", "u_in_q", "u_in_lag1_q", "u_in_lag2_q"]
    key_qc = ["R", "C", "t_idx", "u_out", "u_in_q", "u_in_diff1_q"]
    key4 = ["R", "C", "t_idx", "u_out"]
    key5 = ["R", "C", "t_idx"]

    g_q = (
        train_insp.groupby(key_q, sort=False)["pressure"]
        .median()
        .rename("pq")
        .reset_index()
    )
    g_qb = (
        train_insp.groupby(key_qb, sort=False)["pressure"]
        .median()
        .rename("pqb")
        .reset_index()
    )
    g_qc = (
        train_insp.groupby(key_qc, sort=False)["pressure"]
        .median()
        .rename("pqc")
        .reset_index()
    )
    g4 = (
        train_insp.groupby(key4, sort=False)["pressure"]
        .median()
        .rename("p4")
        .reset_index()
    )
    g5 = (
        train_insp.groupby(key5, sort=False)["pressure"]
        .median()
        .rename("p5")
        .reset_index()
    )

    m = m.merge(g_q, on=key_q, how="left")
    m = m.merge(g_qb, on=key_qb, how="left")
    m = m.merge(g_qc, on=key_qc, how="left")
    m = m.merge(g4, on=key4, how="left")
    m = m.merge(g5, on=key5, how="left")

    pred = m["pq0"]
    pred = pred.fillna(m["pq1"])
    pred = pred.fillna(m["pq2"])
    pred = pred.fillna(m["pq"])
    pred = pred.fillna(m["pqb"])
    pred = pred.fillna(m["pqc"])
    pred = pred.fillna(m["p4"])
    pred = pred.fillna(m["p5"])
    pred = pred.fillna(global_med)

    output_df = output_df.copy()
    out_pred = pred.to_numpy(dtype=np.float64, copy=False)

    out_pred[m["u_out"].to_numpy() == 1] = 0.0

    output_df["pressure"] = snap_to_nearest_pressure_vec(out_pred)
    return output_df


def g(dp):
    """
    Blend if external prediction files exist; otherwise deterministic fallback lookup.
    """
    output = pd.read_csv(SAMPLE_PATH)
    expected_len = len(output)

    l = _safe_list_pred_files(dp)
    if len(l) == 0:
        output2 = _fallback_pressure_lookup_submission(output, u_in_round=1)
        output2.to_csv("submission.csv", index=False)
        return

    file_count = len(l)
    loop_time = 154
    splits = max(1, file_count // 2)

    flist_paths = []
    for i in range(splits):
        start = i * round(len(l) / splits)
        end = None if i == splits - 1 else (i + 1) * round(len(l) / splits)
        chunk = l[start:end]
        if len(chunk) > 0:
            flist_paths.append(chunk)

    flist = []
    for chunk in flist_paths:
        valid_chunk = []
        for fp in chunk:
            v = _load_pressure_vector(fp, expected_len)
            if v is not None:
                valid_chunk.append(fp)
        if len(valid_chunk) == 0:
            continue
        try:
            vec = wc(valid_chunk)
            if len(vec) == expected_len:
                flist.append(vec.astype(np.float64, copy=False))
        except Exception:
            continue

    if len(flist) == 0:
        output2 = _fallback_pressure_lookup_submission(output, u_in_round=1)
        output2.to_csv("submission.csv", index=False)
        return

    n_models = len(flist)
    preds_mat = np.empty((loop_time, expected_len), dtype=np.float64)

    for loop_idx in range(loop_time):
        set_seed(loop_idx)
        w = np.fromiter(
            (rd() for _ in range(n_models)), dtype=np.float64, count=n_models
        )
        ws = w.sum()
        if ws == 0:
            w[:] = 1.0 / n_models
        else:
            w /= ws
        w.sort()
        w = w[::-1]

        temp = np.zeros(expected_len, dtype=np.float64)
        for j in range(n_models):
            temp += flist[j] * w[j]
        preds_mat[loop_idx] = temp

    output_pred = np.median(preds_mat, axis=0)
    output["pressure"] = snap_to_nearest_pressure_vec(output_pred)

    output.to_csv(f"rwb_{loop_time}_loops.csv", index=False)
    output.to_csv("submission.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.7 + b.pressure * 0.3
    a["pressure"] = snap_to_nearest_pressure_vec(
        a["pressure"].to_numpy(dtype=np.float64, copy=False)
    )
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
g("../input/gb-data-blending-recover")
