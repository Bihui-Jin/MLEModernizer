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
df_train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
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
        dfp = pd.read_csv(fp)
        if "pressure" not in dfp.columns:
            return None
        v = dfp["pressure"].to_numpy().ravel()
        if len(v) != expected_len:
            return None
        return v
    except Exception:
        return None


def _add_lag_features(df, lags=(1, 2, 3, 5)):
    """
    Deterministic feature enrichment: add within-breath lag features for controls/time.
    """
    df = df.sort_values(["breath_id", "time_step"], kind="mergesort").copy()
    for k in lags:
        df[f"u_in_lag{k}"] = df.groupby("breath_id", sort=False)["u_in"].shift(k)
        df[f"u_out_lag{k}"] = df.groupby("breath_id", sort=False)["u_out"].shift(k)
        df[f"dt_lag{k}"] = df.groupby("breath_id", sort=False)["time_step"].diff(k)
    df["u_in_diff1"] = df.groupby("breath_id", sort=False)["u_in"].diff(1)
    df["u_in_diff2"] = df.groupby("breath_id", sort=False)["u_in"].diff(2)
    df["u_in_cumsum"] = df.groupby("breath_id", sort=False)["u_in"].cumsum()
    return df.fillna(0.0)


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


def _nn_lookup_1d_by_group(train_df, test_df, group_cols, xcol, ycol, outcol):
    """
    Score-improving + timeout-safe fix:
    Keep the same deterministic "nearest-neighbor in 1D within group" idea,
    but avoid the huge Cartesian merge. For each group, we sort train by xcol
    and use np.searchsorted to pick the nearest x for each test row.

    This preserves core semantics (NN within backoff groups) but greatly increases
    match quality and makes it feasible within 600s.
    """
    out = test_df.copy()
    out[outcol] = np.nan

    if len(train_df) == 0 or len(test_df) == 0:
        return out

    tr = train_df[group_cols + [xcol, ycol]].copy()
    te = test_df[group_cols + [xcol]].copy()

    for key, te_g in te.groupby(group_cols, sort=False):
        tr_g = tr
        if len(group_cols) == 1:
            tr_g = tr_g[tr_g[group_cols[0]].eq(key)]
        else:
            mask = np.ones(len(tr_g), dtype=bool)
            for col, val in zip(group_cols, key):
                mask &= tr_g[col].to_numpy() == val
            tr_g = tr_g.loc[mask]

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
        choose1 = d1 < d0  # if tie, keep idx0 (deterministic)

        pick = np.where(choose1, idx1, idx0)
        pred = y_tr[pick]

        out.loc[te_g.index, outcol] = pred

    return out


def _fallback_pressure_lookup_submission(output_df, u_in_round=1):
    """
    Only used when no blend files exist:
    Deterministic lookup with backoffs, then snap to nearest valid pressure.

    Metric-aligned:
      - Build lookups from inspiratory rows only (u_out==0).
      - For test u_out==1 rows, set a harmless constant (0.0). These rows are not scored.

    Change (directly score-relevant, preserves core logic):
      - Replace the previous merge-based NN with a per-group searchsorted NN
        so the intended nearest-neighbor logic actually works and doesn't timeout/degenerate.
      - Slightly strengthen the first NN grouping by adding u_in_diff1_q which is already computed.
    """
    train = df_train[
        ["breath_id", "R", "C", "time_step", "u_out", "u_in", "pressure"]
    ].copy()
    test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")[
        ["id", "breath_id", "R", "C", "time_step", "u_out", "u_in"]
    ].copy()

    train = _add_time_index(train)
    test = _add_time_index(test)

    train = _add_lag_features(train, lags=(1, 2, 3, 5))
    test = _add_lag_features(test, lags=(1, 2, 3, 5))

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
        train[c + "_q"] = _quantize_to_int(train[c], scale=10.0)
        test[c + "_q"] = _quantize_to_int(test[c], scale=10.0)

    train_insp = train[train["u_out"].eq(0)].copy()
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
        "u_in_diff1_q",  # minimal added context; already computed
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
    output_df["pressure"] = pred.to_numpy(dtype=np.float64, copy=False)

    output_df.loc[m["u_out"].to_numpy() == 1, "pressure"] = 0.0

    output_df["pressure"] = output_df["pressure"].apply(find_nearest)
    return output_df


def g(dp):
    """
    Original logic: read many CSVs from a dataset, do random-weight blending for many loops,
    then take median across loops, snap to nearest known pressure, write CSV.
    Bugfixes:
      - Handle missing/empty dp or no valid files => fallback submission.
      - Ensure vstack is non-empty and vectors have correct length (603600).
      - Ensure output filename ends with .csv.
    """
    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
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
                flist.append(vec)
        except Exception:
            continue

    if len(flist) == 0:
        output2 = _fallback_pressure_lookup_submission(output, u_in_round=1)
        output2.to_csv("submission.csv", index=False)
        return

    pred_list = []
    for loop_idx in range(loop_time):
        weight = []
        set_seed(loop_idx)
        for _ in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        if weight_sum == 0:
            weight = [1.0 / len(weight)] * len(weight)
        else:
            weight = [w / weight_sum for w in weight]

        weight.sort(reverse=True)

        temp = np.zeros(expected_len, dtype=np.float64)
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)

        del temp
        gc.collect()

    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)

    output.to_csv(f"rwb_{loop_time}_loops.csv", index=False)
    output.to_csv("submission.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.7 + b.pressure * 0.3
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
g("../input/gb-data-blending-recover")
