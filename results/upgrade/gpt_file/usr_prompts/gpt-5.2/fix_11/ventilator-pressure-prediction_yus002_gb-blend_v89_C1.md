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

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

df_train = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv", usecols=["pressure"]
)
unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)
del df_train
gc.collect()


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


def find_nearest_vec(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    idx = np.searchsorted(sorted_pressures, pred, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)

    upper = sorted_pressures[idx]
    lower_idx = np.maximum(idx - 1, 0)
    lower = sorted_pressures[lower_idx]

    choose_lower = (idx > 0) & (np.abs(lower - pred) < np.abs(upper - pred))
    out = np.where(choose_lower, lower, upper)
    return out.astype(np.float64, copy=False)


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    l = []
    preds = []
    for i in range(len(input_list)):
        fp = input_list[i]
        try:
            public_lb_score = int(fp.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1
        l.append(public_lb_score)

        df = pd.read_csv(fp, usecols=["pressure"])
        preds.append(df["pressure"].to_numpy().ravel())

    l_sum = sum(l) if sum(l) != 0 else 1

    if len(preds) == 1:
        return preds[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        return preds[0] * weight1 + preds[1] * weight2


def _make_features_from_raw(
    R: np.ndarray,
    C: np.ndarray,
    time_step: np.ndarray,
    u_in: np.ndarray,
    u_out: np.ndarray,
) -> np.ndarray:
    n = u_in.shape[0]
    if n % 80 != 0:
        raise ValueError(
            "Expected number of rows to be multiple of 80 (breath length)."
        )

    n_breaths = n // 80
    R2 = R.astype(np.float32, copy=False).reshape(n_breaths, 80)
    C2 = C.astype(np.float32, copy=False).reshape(n_breaths, 80)
    t2 = time_step.astype(np.float32, copy=False).reshape(n_breaths, 80)
    uin = u_in.astype(np.float32, copy=False).reshape(n_breaths, 80)
    uout = u_out.astype(np.float32, copy=False).reshape(n_breaths, 80)

    u_in_lag1 = np.zeros_like(uin)
    u_in_lag1[:, 1:] = uin[:, :-1]

    u_in_lag2 = np.zeros_like(uin)
    u_in_lag2[:, 2:] = uin[:, :-2]

    u_out_lag1 = np.zeros_like(uout)
    u_out_lag1[:, 1:] = uout[:, :-1]

    dt = np.zeros_like(t2)
    dt[:, 1:] = t2[:, 1:] - t2[:, :-1]

    u_in_diff1 = uin - u_in_lag1
    u_in_dt = uin * dt

    t_idx = np.broadcast_to(np.arange(80, dtype=np.float32), (n_breaths, 80)).copy()

    u_in_cumarea = np.cumsum(u_in_dt, axis=1, dtype=np.float32)

    roll_mean3 = np.empty_like(uin)
    roll_max3 = np.empty_like(uin)
    roll_mean3[:, 0] = uin[:, 0]
    roll_max3[:, 0] = uin[:, 0]
    roll_mean3[:, 1] = 0.5 * (uin[:, 0] + uin[:, 1])
    roll_max3[:, 1] = np.maximum(uin[:, 0], uin[:, 1])
    roll_mean3[:, 2:] = (uin[:, 2:] + uin[:, 1:-1] + uin[:, :-2]) / 3.0
    roll_max3[:, 2:] = np.maximum.reduce([uin[:, 2:], uin[:, 1:-1], uin[:, :-2]])

    u_in_over_R = uin / (R2 + 1e-3)
    u_in_over_C = uin / (C2 + 1e-3)
    area_over_C = u_in_cumarea / (C2 + 1e-3)

    X = np.column_stack(
        [
            R2.reshape(-1),
            C2.reshape(-1),
            t2.reshape(-1),
            t_idx.reshape(-1),
            uin.reshape(-1),
            u_in_lag1.reshape(-1),
            u_in_lag2.reshape(-1),
            u_in_diff1.reshape(-1),
            roll_mean3.reshape(-1),
            roll_max3.reshape(-1),
            u_in_cumarea.reshape(-1),
            uout.reshape(-1),
            u_out_lag1.reshape(-1),
            u_in_over_R.reshape(-1),
            u_in_over_C.reshape(-1),
            area_over_C.reshape(-1),
        ]
    ).astype(np.float32, copy=False)

    return np.ascontiguousarray(X)


def _knn_fallback_submission(
    sample_path="../input/ventilator-pressure-prediction/sample_submission.csv",
    train_path="../input/ventilator-pressure-prediction/train.csv",
    test_path="../input/ventilator-pressure-prediction/test.csv",
):
    sample = pd.read_csv(sample_path, usecols=["id"])

    use_train = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    use_test = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "id"]

    train = pd.read_csv(train_path, usecols=use_train)
    test = pd.read_csv(test_path, usecols=use_test)

    train = train.sort_values(
        ["breath_id", "time_step"], kind="mergesort", ignore_index=True
    )
    test = test.sort_values(
        ["breath_id", "time_step"], kind="mergesort", ignore_index=True
    )

    breaths = train["breath_id"].to_numpy()
    uniq = np.unique(breaths)
    rng = np.random.RandomState(2021)
    rng.shuffle(uniq)
    n_fit = int(len(uniq) * 0.8)
    fit_breaths = uniq[:n_fit]
    fit_mask = np.isin(breaths, fit_breaths, assume_unique=False)

    X = _make_features_from_raw(
        R=train["R"].to_numpy(),
        C=train["C"].to_numpy(),
        time_step=train["time_step"].to_numpy(),
        u_in=train["u_in"].to_numpy(),
        u_out=train["u_out"].to_numpy(),
    )
    y = train["pressure"].to_numpy(dtype=np.float32, copy=False)

    X_test = _make_features_from_raw(
        R=test["R"].to_numpy(),
        C=test["C"].to_numpy(),
        time_step=test["time_step"].to_numpy(),
        u_in=test["u_in"].to_numpy(),
        u_out=test["u_out"].to_numpy(),
    )

    scaler = StandardScaler()
    scaler.fit(X[fit_mask])
    Xs = scaler.transform(X)
    Xs_test = scaler.transform(X_test)

    knn = KNeighborsRegressor(
        n_neighbors=50, weights="distance", metric="minkowski", p=2, n_jobs=-1
    )
    knn.fit(Xs[fit_mask], y[fit_mask])

    n_test = Xs_test.shape[0]
    pred = np.empty(n_test, dtype=np.float64)
    chunk = 200_000
    for s in range(0, n_test, chunk):
        e = min(s + chunk, n_test)
        pred[s:e] = knn.predict(Xs_test[s:e]).astype(np.float64, copy=False)

    pred = find_nearest_vec(pred)

    out = pd.DataFrame({"id": test["id"].to_numpy(), "pressure": pred})
    out = out.sort_values("id", kind="mergesort").reset_index(drop=True)

    output = sample.copy()
    output["pressure"] = out["pressure"].to_numpy()

    if output["pressure"].isna().any():
        base = float(np.median(sorted_pressures))
        base = find_nearest(base)
        output["pressure"] = output["pressure"].fillna(base)

    return output


def _median_over_loops_streamed(
    flist: list[np.ndarray], loop_time: int, n_rows: int, seed_base: int = 0
) -> np.ndarray:
    k1 = loop_time // 2 - 1
    k2 = loop_time // 2

    rng = np.random.RandomState(2021)

    P = np.vstack([np.asarray(a, dtype=np.float64).reshape(1, -1) for a in flist])
    m = P.shape[0]

    W = np.empty((loop_time, m), dtype=np.float64)
    for t in range(loop_time):
        w = rng.random(m)
        s = w.sum()
        if s == 0:
            s = 1.0
        w /= s
        w = w[np.argsort(w)[::-1]]
        W[t] = w

    out = np.empty(n_rows, dtype=np.float64)

    block_size = 32768
    for start in range(0, n_rows, block_size):
        end = min(start + block_size, n_rows)
        block = P[:, start:end]  # (m, B)

        samples = W @ block  # (loop_time, B)

        part = np.partition(samples, (k1, k2), axis=0)
        out[start:end] = 0.5 * (part[k1] + part[k2])

    return out


def g(dp):
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    output = pd.read_csv(sample_path, usecols=["id", "pressure"])
    n_rows = len(output)

    files = []
    if dp is not None and os.path.isdir(dp):
        for fp in glob.iglob(f"{dp}/*"):
            if os.path.isfile(fp) and fp.lower().endswith(".csv"):
                files.append(fp)

    files.sort()
    if len(files) == 0:
        sub = _knn_fallback_submission(sample_path=sample_path)
        sub.to_csv("submission.csv", index=False)
        return sub

    file_count = len(files)
    loop_time = 150
    splits = file_count // 2
    if splits < 1:
        splits = 1

    flist = []
    for i in range(splits):
        start = i * round(len(files) / splits)
        end = None if i == splits - 1 else (i + 1) * round(len(files) / splits)
        chunk = files[start:end]
        if len(chunk) == 0:
            continue
        try:
            blended = wc(chunk)
        except Exception:
            continue
        flist.append(blended)

    flist_valid = []
    for arr in flist:
        arr = np.asarray(arr).ravel()
        if arr.shape[0] == n_rows:
            flist_valid.append(arr.astype(np.float64, copy=False))
    flist = flist_valid

    if len(flist) == 0:
        sub = _knn_fallback_submission(sample_path=sample_path)
        sub.to_csv("submission.csv", index=False)
        return sub

    med = _median_over_loops_streamed(flist=flist, loop_time=loop_time, n_rows=n_rows)

    output["pressure"] = med
    output["pressure"] = find_nearest_vec(output["pressure"].to_numpy())

    output.to_csv("submission.csv", index=False)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)
    return output


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.5 + b.pressure * 0.5
    a["pressure"] = find_nearest_vec(a["pressure"].to_numpy())
    a.to_csv("blend.csv", index=False)
    return a




## === cell 1
g("../input/gb-blending")
print("Wrote submission.csv")
sub = pd.read_csv("submission.csv")
print(sub.head())
print(sub.shape)
print(sub.columns.tolist())
