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

PRESSURE_MIN = np.float32(3.52)
PRESSURE_MAX = np.float32(41.48)
PRESSURE_STEP = np.float32(0.07)
_n_steps = (
    int(np.round((float(PRESSURE_MAX) - float(PRESSURE_MIN)) / float(PRESSURE_STEP)))
    + 1
)
sorted_pressures = PRESSURE_MIN + PRESSURE_STEP * np.arange(_n_steps, dtype=np.float32)
total_pressures_len = len(sorted_pressures)


def find_nearest(prediction):
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        return float(sorted_pressures[-1])
    elif insert_idx == 0:
        return float(sorted_pressures[0])
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return float(
        lower_val
        if abs(lower_val - prediction) < abs(upper_val - prediction)
        else upper_val
    )


def find_nearest_vec(preds: np.ndarray) -> np.ndarray:
    preds = np.asarray(preds, dtype=np.float32)
    idx = np.searchsorted(sorted_pressures, preds, side="left")
    idx0 = np.clip(idx - 1, 0, total_pressures_len - 1)
    idx1 = np.clip(idx, 0, total_pressures_len - 1)
    lower = sorted_pressures[idx0]
    upper = sorted_pressures[idx1]
    choose_lower = np.abs(preds - lower) < np.abs(preds - upper)
    return np.where(choose_lower, lower, upper).astype(np.float32, copy=False)


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    """
    Original intent: weight-combine two submissions based on a number parsed from filename.
    Bugfix: be robust when filename format doesn't contain that token; fall back to equal weights.
    Also ensure returned vector is 1D float.
    """
    preds = []
    scores = []
    for path in input_list:
        try:
            public_lb_score = int(path.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1
        scores.append(public_lb_score)

        df = pd.read_csv(path)
        if "pressure" not in df.columns:
            raise ValueError(f"File {path} does not contain 'pressure' column.")
        preds.append(df["pressure"].to_numpy(dtype=float).ravel())

    if len(preds) == 1:
        return preds[0]

    l_sum = sum(scores) if sum(scores) != 0 else len(scores)
    weight1 = (scores[1] / l_sum) + 0.1
    weight1 = float(np.clip(weight1, 0.0, 1.0))
    weight2 = 1.0 - weight1
    return preds[0] * weight1 + preds[1] * weight2


def _make_seq_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["breath_id", "time_step"], kind="mergesort").copy()
    g = df.groupby("breath_id", sort=False)

    u_in = df["u_in"]
    u_out = df["u_out"]

    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0)
    df["u_in_lag3"] = g["u_in"].shift(3).fillna(0.0)
    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0).astype(np.int8)

    df["du_in"] = (u_in - df["u_in_lag1"]).astype(np.float32, copy=False)
    df["du_in2"] = (df["u_in_lag1"] - df["u_in_lag2"]).astype(np.float32, copy=False)

    df["dt"] = g["time_step"].diff().fillna(0.0).astype(np.float32)
    df["u_in_dt"] = (u_in.astype(np.float32) * df["dt"]).astype(np.float32, copy=False)
    df["u_in_cum"] = g["u_in_dt"].cumsum().astype(np.float32)

    df["u_out_cum"] = g["u_out"].cumsum().astype(np.float32)

    df["t_rel"] = df["time_step"].astype(np.float32)
    return df


def _add_train_pressure_history_features(train_df: pd.DataFrame) -> pd.DataFrame:
    train_df = train_df.sort_values(["breath_id", "time_step"], kind="mergesort").copy()
    g = train_df.groupby("breath_id", sort=False)

    p = train_df["pressure"]
    train_df["p_lag1"] = g["pressure"].shift(1)
    train_df["p_lag2"] = g["pressure"].shift(2)

    train_df["p_lag1"] = g["p_lag1"].bfill().fillna(p)
    train_df["p_lag2"] = g["p_lag2"].bfill().fillna(p)

    train_df["dp_lag1"] = (p - train_df["p_lag1"]).astype(np.float32)
    return train_df


def _fallback_knn_submission():
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    train_path = "../input/ventilator-pressure-prediction/train.csv"
    test_path = "../input/ventilator-pressure-prediction/test.csv"
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"

    usecols_train = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    usecols_test = ["breath_id", "R", "C", "time_step", "u_in", "u_out"]

    dtypes_train = {
        "breath_id": np.int32,
        "R": np.int16,
        "C": np.int16,
        "time_step": np.float32,
        "u_in": np.float32,
        "u_out": np.int8,
        "pressure": np.float32,
    }
    dtypes_test = {
        "breath_id": np.int32,
        "R": np.int16,
        "C": np.int16,
        "time_step": np.float32,
        "u_in": np.float32,
        "u_out": np.int8,
    }

    train_full = pd.read_csv(
        train_path,
        usecols=usecols_train,
        dtype=dtypes_train,
        engine="c",
        low_memory=False,
    )
    test = pd.read_csv(
        test_path, usecols=usecols_test, dtype=dtypes_test, engine="c", low_memory=False
    )
    sub = pd.read_csv(sample_path, engine="c", low_memory=False)

    unique_breaths = train_full["breath_id"].unique()
    rng = np.random.RandomState(2021)
    rng.shuffle(unique_breaths)
    keep_breaths = set(unique_breaths[:60000])

    train_small = train_full[train_full["breath_id"].isin(keep_breaths)].copy()
    del train_full
    gc.collect()

    train_small = _make_seq_features(train_small)
    test_feat = _make_seq_features(test)

    train_small = _add_train_pressure_history_features(train_small)
    test_feat["p_lag1"] = np.float32(0.0)
    test_feat["p_lag2"] = np.float32(0.0)
    test_feat["dp_lag1"] = np.float32(0.0)

    train_small_insp = train_small[train_small["u_out"] == 0].copy()

    feat_cols = [
        "R",
        "C",
        "t_rel",
        "u_in",
        "u_out",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_lag3",
        "du_in",
        "du_in2",
        "u_out_lag1",
        "u_in_cum",
        "u_out_cum",
        "p_lag1",
        "p_lag2",
        "dp_lag1",
    ]

    X_train = train_small_insp[feat_cols].to_numpy(dtype=np.float32, copy=False)
    y_train = train_small_insp["pressure"].to_numpy(dtype=np.float32, copy=False)
    X_test = test_feat[feat_cols].to_numpy(dtype=np.float32, copy=False)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "knn",
                KNeighborsRegressor(
                    n_neighbors=80,
                    weights="distance",
                    p=2,
                    n_jobs=-1,
                    algorithm="brute",
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test).astype(np.float32, copy=False)

    snapped = find_nearest_vec(pred)
    pred = 0.25 * pred + 0.75 * snapped

    sub["pressure"] = find_nearest_vec(pred).astype(float, copy=False)
    sub.to_csv("submission.csv", index=False)


def g(dp):
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    output = pd.read_csv(sample_path, engine="c", low_memory=False)
    n = len(output)

    if not os.path.isdir(dp):
        _fallback_knn_submission()
        return

    files = sorted([p for p in glob.iglob(f"{dp}/*") if p.lower().endswith(".csv")])
    if len(files) == 0:
        _fallback_knn_submission()
        return

    valid_preds = []
    for p in files:
        try:
            dfp = pd.read_csv(
                p,
                usecols=["pressure"],
                dtype={"pressure": np.float32},
                engine="c",
                low_memory=False,
            )
            vec = dfp["pressure"].to_numpy(copy=False).ravel()
            if len(vec) != n:
                continue
            valid_preds.append(np.ascontiguousarray(vec, dtype=np.float32))
        except Exception:
            continue

    if len(valid_preds) == 0:
        _fallback_knn_submission()
        return

    file_count = len(valid_preds)
    loop_time = 150

    splits = max(1, file_count // 2)
    flist = []
    step = round(file_count / splits)
    for i in range(splits):
        start = i * step
        end = None if i == splits - 1 else (i + 1) * step
        group = valid_preds[start:end]
        if len(group) == 0:
            continue
        if len(group) == 1:
            flist.append(group[0])
        else:
            flist.append((group[0] + group[1]) * 0.5)

    if len(flist) == 0:
        _fallback_knn_submission()
        return

    F = np.vstack(flist).astype(np.float32, copy=False)  # (m, n)
    F = np.ascontiguousarray(F)
    m = F.shape[0]

    Pn = np.empty((n, loop_time), dtype=np.float32)

    w = np.empty(m, dtype=np.float32)
    for t in range(loop_time):
        rng = np.random.RandomState(t)
        w[:] = rng.random_sample(m).astype(np.float32, copy=False)
        wsum = float(w.sum())
        if wsum == 0.0:
            w.fill(np.float32(1.0 / m))
        else:
            w /= np.float32(wsum)
        w[::-1].sort()  # descending (kept exactly)
        Pn[:, t] = np.dot(w, F)

    blended = np.median(Pn, axis=1).astype(np.float32, copy=False)

    output["pressure"] = find_nearest_vec(blended).astype(float, copy=False)
    output.to_csv("submission.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a, engine="c", low_memory=False)
    b = pd.read_csv(b, engine="c", low_memory=False)
    a.pressure = a.pressure * 0.6 + b.pressure * 0.4
    a["pressure"] = find_nearest_vec(
        a["pressure"].to_numpy(dtype=np.float32, copy=False)
    ).astype(float, copy=False)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 1
g("../input/gb-blending")
print("Wrote submission.csv")
