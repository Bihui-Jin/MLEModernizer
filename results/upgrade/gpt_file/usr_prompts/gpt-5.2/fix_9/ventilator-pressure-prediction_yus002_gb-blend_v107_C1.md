# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.1510712250966403

# 6. Current score

0.89626

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.86378) has done: 'The crash is because the notebook tries to read two external blend files that don’t exist in your Kaggle environment. I keep your existing logic (pressure snapping and blending) but make the pipeline self-contained: if those blend inputs are missing, it automatically fall back to a simple baseline submission derived from the training pressure distribution (median), then snap to the nearest allowed pressure. I also fix the cell numbering to start from 1 (your provided script starts at cell 0) and ensure a valid `submission.csv` is always written. These changes are score-safe (won’t error) and should yield a non-trivial score vs the all-zeros sample submission.'
- What this solution (achieved 3.82938) has done: 'Your current score is very far from the target (10.86 vs 0.151, lower is better), so we need a real modeling-based prediction instead of the constant-median fallback. To keep core logic minimal, I’m leaving your pressure “snapping” and submission writing intact, but I’m replacing only the fallback branch with a lightweight, deterministic k-nearest-neighbors regressor using only the provided tabular signals (R, C, time_step, u_in, u_out) to predict pressure. This stays within sklearn, runs quickly, and should dramatically reduce MAE toward your target without changing evaluation semantics. If the external blend files exist, your original blend path is still used unchanged.'
- What this solution (achieved 0.92756) has done: 'The timeout is coming from KNN training/inference on ~5.4M rows with dense one-hot + scaling: brute-force neighbor search is superlinear and won’t finish in 600s. To preserve the exact core logic (same features, same KNN params, same snapping), the main speedups are: (1) avoid pandas/groupby overhead by computing breath-wise lag/cumsum/diff with vectorized NumPy on already-ordered data, (2) avoid `Pipeline`/`ColumnTransformer` overhead by doing the *same* scaling and one-hot encoding once with NumPy, and (3) reduce KNN work by fitting/predicting per (R,C) group (9 groups) which is exactly equivalent because one-hot makes different (R,C) groups far apart, so neighbors never cross groups. This keeps predictions identical up to negligible float differences, but cuts the neighbor search problem size drastically and eliminates large intermediate dense matrices.'
- What this solution (achieved 0.92756) has done: 'Your current MAE (0.92756, lower-is-better) is still far above the target (0.15107), so we should make a small, safe accuracy improvement without changing the overall approach. The biggest score gain you can get while preserving your KNN core logic is to avoid training/predicting on expiratory timesteps (u_out==1), because they are not scored; instead, predict only inspiratory timesteps with KNN and fill expiratory timesteps with a cheap per-(R,C,time_step) median learned from train. This keeps the same features, same KNN model type/params, and same pressure snapping, but reallocates compute to the scored region and typically improves MAE materially. I also keep the per-(R,C) grouping optimization and ensure the submission is aligned to test `id` and always written to `submission.csv`.'
- What this solution (achieved 0.92756) has done: 'We keep your exact KNN approach, features, snapping, and per-(R,C) grouping, but make two minimal changes that usually reduce MAE substantially for this competition: (1) restrict KNN training to the inspiratory phase only (u_out==0) and (2) apply the same inspiratory-only mask when building the (R,C,time_step) median fallback (so expiratory dynamics don’t pollute medians). Additionally, we avoid the slow merge for expiratory filling by using an index-based reindex, which is faster and less error-prone while producing the same semantics. These are small, metric-aligned changes (only affect how we handle unscored expiratory timesteps and reduce noise) and should move your score down toward the 0.151 target without changing the model family or adding new dependencies. The script still always write a valid `submission.csv`.'
- What this solution (achieved 0.89626) has done: 'Your current score (0.92756, lower-is-better) is still far above the target (0.15107), so we should make a small metric-aligned change that improves accuracy without changing your KNN core approach. The biggest safe win is to keep training KNN only on inspiratory rows (already done), and additionally remove expiratory rows from the *feature scaling statistics* (mean/var), since expiratory dynamics are unscored and can skew scaling used for inspiratory predictions. We also add a tiny, competition-specific post-process: enforce physically consistent “hold” behavior by carrying forward the last inspiratory prediction into expiratory timesteps within each breath (instead of the (R,C,time_step) median), which typically reduces error spillover near the inspiratory/expiratory boundary without changing the model family. All changes keep the same features, same KNN parameters, and still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing
import os
import copy
import glob
import random
from random import random as rd
import gc

from sklearn.neighbors import KNeighborsRegressor


def snap_to_allowed_pressures(
    pred: np.ndarray, sorted_pressures: np.ndarray
) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float32)
    sp = np.asarray(sorted_pressures, dtype=np.float32)
    idx = np.searchsorted(sp, pred, side="left")

    out = np.empty_like(pred, dtype=np.float32)

    m0 = idx == 0
    if np.any(m0):
        out[m0] = sp[0]

    m1 = idx == sp.size
    if np.any(m1):
        out[m1] = sp[-1]

    mm = ~(m0 | m1)
    if np.any(mm):
        ii = idx[mm]
        lower = sp[ii - 1]
        upper = sp[ii]
        x = pred[mm]
        out[mm] = np.where(np.abs(lower - x) < np.abs(upper - x), lower, upper)

    return out


def find_nearest(prediction: float) -> float:
    """Snap a continuous prediction to the nearest pressure value seen in train."""
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        return float(sorted_pressures[-1])
    if insert_idx == 0:
        return float(sorted_pressures[0])
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return float(
        lower_val
        if abs(lower_val - prediction) < abs(upper_val - prediction)
        else upper_val
    )


def set_seed(seed: int = 2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    """
    Original helper: loads submissions and performs a simple weighted combination
    based on numeric tokens in filenames.
    """
    l = []
    for i in range(len(input_list)):
        public_lb_score = int(input_list[i].split("/")[-1].split(".")[1].split(" ")[0])
        l.append(public_lb_score)
        input_list[i] = (pd.read_csv(input_list[i]).pressure).ravel()

    l_sum = sum(l)
    if len(input_list) == 1:
        output = input_list[0]
    else:
        weight1 = (l[1] / l_sum) + 0.15
        weight2 = 1 - weight1
        output = input_list[0] * weight1 + input_list[1] * weight2
    return output


def g(dp):
    """
    Original helper: ensemble predictions from files in a directory.
    Not used in this self-contained fallback unless the directory exists.
    """
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)

    file_count = len(l)
    loop_time = 154
    splits = file_count // 2
    l.sort()
    flist = []
    for i in range(splits):
        if i == splits - 1:
            flist.append(l[i * round(len(l) / splits) :])
        else:
            flist.append(
                l[i * round(len(l) / splits) : (i + 1) * round(len(l) / splits)]
            )

    for i in range(len(flist)):
        flist[i] = wc(flist[i])

    pred_list = []
    for it in range(loop_time):
        weight = []
        set_seed(it)
        for j in range(len(flist)):
            weight.append(rd())

        weight_sum = sum(weight)
        for j in range(len(weight)):
            weight[j] /= weight_sum
        weight.sort(reverse=True)

        temp = 0
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output.pressure = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b, out_path="blend.csv"):
    """
    Original blend: average two submissions and snap to nearest allowed pressure.
    """
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.5 + b.pressure * 0.5
    a["pressure"] = snap_to_allowed_pressures(
        a["pressure"].to_numpy(), sorted_pressures
    )
    a.to_csv(out_path, index=False)
    return a




## === cell 1
a = "../input/gb-data-blending-recover/0.149 blend.csv"
b = "../input/gb-data-blending-recover/0.149 blend2.csv"

sub_path = "submission.csv"

if os.path.exists(a) and os.path.exists(b):
    df_train = pd.read_csv(
        "../input/ventilator-pressure-prediction/train.csv", usecols=["pressure"]
    )
    unique_pressures = df_train["pressure"].unique()
    sorted_pressures = np.sort(unique_pressures).astype(np.float32)
    total_pressures_len = len(sorted_pressures)
    del df_train
    gc.collect()

    df_sub = blend(a, b, out_path=sub_path)
else:
    feature_cols_num_base = ["time_step", "u_in", "u_out"]
    feature_cols_cat = ["R", "C"]
    target_col = "pressure"

    usecols_train = (
        ["id", "breath_id"] + feature_cols_num_base + feature_cols_cat + [target_col]
    )
    usecols_test = ["id", "breath_id"] + feature_cols_num_base + feature_cols_cat

    dtypes_train = {
        "id": "int32",
        "breath_id": "int32",
        "R": "int16",
        "C": "int16",
        "time_step": "float32",
        "u_in": "float32",
        "u_out": "int8",
        "pressure": "float32",
    }
    dtypes_test = {
        "id": "int32",
        "breath_id": "int32",
        "R": "int16",
        "C": "int16",
        "time_step": "float32",
        "u_in": "float32",
        "u_out": "int8",
    }

    set_seed(2021)

    train = pd.read_csv(
        "../input/ventilator-pressure-prediction/train.csv",
        usecols=usecols_train,
        dtype=dtypes_train,
    )
    test = pd.read_csv(
        "../input/ventilator-pressure-prediction/test.csv",
        usecols=usecols_test,
        dtype=dtypes_test,
    )

    unique_pressures = train["pressure"].unique()
    sorted_pressures = np.sort(unique_pressures).astype(np.float32)
    total_pressures_len = len(sorted_pressures)

    def add_breath_features_fast(df: pd.DataFrame) -> pd.DataFrame:
        if not (
            df["breath_id"].is_monotonic_increasing
            and df["time_step"].is_monotonic_increasing
        ):
            df = df.sort_values(["breath_id", "time_step"], kind="mergesort").copy()
        else:
            df = df.copy()

        bid = df["breath_id"].to_numpy(copy=False)
        u_in = df["u_in"].to_numpy(dtype=np.float32, copy=False)
        u_out = df["u_out"].to_numpy(dtype=np.float32, copy=False)
        t = df["time_step"].to_numpy(dtype=np.float32, copy=False)

        new_breath = np.empty(bid.shape[0], dtype=bool)
        new_breath[0] = True
        new_breath[1:] = bid[1:] != bid[:-1]

        u_in_lag1 = np.empty_like(u_in)
        u_in_lag1[0] = 0.0
        u_in_lag1[1:] = u_in[:-1]
        u_in_lag1[new_breath] = 0.0

        u_in_lag2 = np.empty_like(u_in)
        u_in_lag2[:2] = 0.0
        u_in_lag2[2:] = u_in[:-2]
        u_in_lag2[new_breath] = 0.0
        nb_idx = np.flatnonzero(new_breath)
        nb2 = nb_idx + 1
        nb2 = nb2[nb2 < u_in_lag2.shape[0]]
        u_in_lag2[nb2] = 0.0

        u_out_lag1 = np.empty_like(u_out)
        u_out_lag1[0] = 0.0
        u_out_lag1[1:] = u_out[:-1]
        u_out_lag1[new_breath] = 0.0

        dt = np.empty_like(t)
        dt[0] = 0.0
        dt[1:] = t[1:] - t[:-1]
        dt[new_breath] = 0.0

        cs = np.cumsum(u_in, dtype=np.float32)
        start = np.flatnonzero(new_breath)
        offset = np.zeros(start.shape[0], dtype=np.float32)
        mask = start > 0
        offset[mask] = cs[start[mask] - 1]
        u_in_cumsum = (
            cs - offset[np.searchsorted(start, np.arange(cs.size), side="right") - 1]
        )

        Rf = df["R"].to_numpy(dtype=np.float32, copy=False)
        Cf = df["C"].to_numpy(dtype=np.float32, copy=False)

        df["u_in_lag1"] = u_in_lag1
        df["u_in_lag2"] = u_in_lag2
        df["u_out_lag1"] = u_out_lag1
        df["u_in_cumsum"] = u_in_cumsum
        df["dt"] = dt
        df["u_in_over_R"] = u_in / Rf
        df["u_in_over_C"] = u_in / Cf
        return df

    train_fe = add_breath_features_fast(train)
    test_fe = add_breath_features_fast(test)

    feature_cols_num = feature_cols_num_base + [
        "u_in_lag1",
        "u_in_lag2",
        "u_out_lag1",
        "u_in_cumsum",
        "dt",
        "u_in_over_R",
        "u_in_over_C",
    ]

    num_train = train_fe[feature_cols_num].to_numpy(dtype=np.float32, copy=False)
    num_test = test_fe[feature_cols_num].to_numpy(dtype=np.float32, copy=False)
    y_train = train_fe[target_col].to_numpy(dtype=np.float32, copy=False)

    uout_train = train_fe["u_out"].to_numpy(copy=False)
    uout_test = test_fe["u_out"].to_numpy(copy=False)
    insp_train_mask = uout_train == 0
    insp_test_mask = uout_test == 0

    num_train_insp = num_train[insp_train_mask]
    mean_ = num_train_insp.mean(axis=0, dtype=np.float64).astype(np.float32)
    var_ = num_train_insp.var(axis=0, dtype=np.float64).astype(np.float32)
    scale_ = np.sqrt(var_, dtype=np.float32)
    scale_[scale_ == 0.0] = 1.0

    num_train_s = (num_train - mean_) / scale_
    num_test_s = (num_test - mean_) / scale_

    R_train = train_fe["R"].to_numpy(copy=False)
    C_train = train_fe["C"].to_numpy(copy=False)
    R_test = test_fe["R"].to_numpy(copy=False)
    C_test = test_fe["C"].to_numpy(copy=False)

    cats_R = np.sort(np.unique(R_train))
    cats_C = np.sort(np.unique(C_train))
    map_R = {v: i for i, v in enumerate(cats_R.tolist())}
    map_C = {v: i for i, v in enumerate(cats_C.tolist())}
    nR = cats_R.size
    nC = cats_C.size

    def one_hot_rc(R_arr, C_arr):
        oh = np.zeros((R_arr.shape[0], nR + nC), dtype=np.float32)
        idxR = np.fromiter(
            (map_R.get(int(v), -1) for v in R_arr), count=R_arr.size, dtype=np.int32
        )
        m = idxR >= 0
        rows = np.nonzero(m)[0]
        oh[rows, idxR[m]] = 1.0
        idxC = np.fromiter(
            (map_C.get(int(v), -1) for v in C_arr), count=C_arr.size, dtype=np.int32
        )
        m = idxC >= 0
        rows = np.nonzero(m)[0]
        oh[rows, nR + idxC[m]] = 1.0
        return oh

    cat_train_oh = one_hot_rc(R_train, C_train)
    cat_test_oh = one_hot_rc(R_test, C_test)

    X_train_full = np.concatenate([num_train_s, cat_train_oh], axis=1)
    X_test_full = np.concatenate([num_test_s, cat_test_oh], axis=1)

    train_insp_for_median = train_fe.loc[
        insp_train_mask, ["R", "C", "time_step", "pressure"]
    ]
    rc_time_median = train_insp_for_median.groupby(["R", "C", "time_step"], sort=False)[
        "pressure"
    ].median()
    global_median = float(train_insp_for_median["pressure"].median())

    pred = np.empty(X_test_full.shape[0], dtype=np.float32)

    X_train_insp = X_train_full[insp_train_mask]
    y_train_insp = y_train[insp_train_mask]
    R_train_insp = R_train[insp_train_mask]
    C_train_insp = C_train[insp_train_mask]

    X_test_insp = X_test_full[insp_test_mask]
    R_test_insp = R_test[insp_test_mask]
    C_test_insp = C_test[insp_test_mask]

    key_train = (R_train_insp.astype(np.int32) << 16) + C_train_insp.astype(np.int32)
    key_test = (R_test_insp.astype(np.int32) << 16) + C_test_insp.astype(np.int32)

    pred_insp = np.empty(X_test_insp.shape[0], dtype=np.float32)

    unique_keys = np.unique(key_test)
    for k in unique_keys:
        tr_mask = key_train == k
        te_mask = key_test == k

        if not np.any(tr_mask):
            tr_mask = slice(None)

        Xtr = X_train_insp[tr_mask]
        ytr = y_train_insp[tr_mask]
        Xte = X_test_insp[te_mask]

        knn = KNeighborsRegressor(
            n_neighbors=35,
            weights="distance",
            metric="minkowski",
            p=2,
            n_jobs=-1,
        )
        knn.fit(Xtr, ytr)
        pred_insp[te_mask] = knn.predict(Xte).astype(np.float32, copy=False)

    pred[insp_test_mask] = pred_insp

    if np.any(~insp_test_mask):
        idx = pd.MultiIndex.from_frame(
            test_fe.loc[~insp_test_mask, ["R", "C", "time_step"]]
        )
        pred_exp = rc_time_median.reindex(idx).to_numpy(dtype=np.float32, copy=False)
        pred_exp = np.where(np.isfinite(pred_exp), pred_exp, np.float32(global_median))
        pred[~insp_test_mask] = pred_exp

        bid = test_fe["breath_id"].to_numpy(copy=False)
        uout = test_fe["u_out"].to_numpy(copy=False)
        new_breath = np.empty(bid.shape[0], dtype=bool)
        new_breath[0] = True
        new_breath[1:] = bid[1:] != bid[:-1]
        for i in range(1, pred.shape[0]):
            if (not new_breath[i]) and (uout[i] == 1):
                pred[i] = pred[i - 1]

    df_sub = pd.DataFrame({"id": test_fe["id"].to_numpy(copy=False), "pressure": pred})
    df_sub["pressure"] = snap_to_allowed_pressures(
        df_sub["pressure"].to_numpy(), sorted_pressures
    )
    df_sub.to_csv(sub_path, index=False)

print(f"Wrote {sub_path} with shape={df_sub.shape} and columns={list(df_sub.columns)}")
print(df_sub.head())
