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

df_train = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
    usecols=["pressure"],
    dtype={"pressure": np.float32},
)
unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)
del df_train


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


def find_nearest_vec(pred):
    pred = np.asarray(pred, dtype=np.float32)
    idx = np.searchsorted(sorted_pressures, pred, side="left")
    idx0 = np.clip(idx, 0, total_pressures_len - 1)
    idx1 = np.clip(idx - 1, 0, total_pressures_len - 1)
    upper = sorted_pressures[idx0]
    lower = sorted_pressures[idx1]
    choose_lower = np.abs(lower - pred) < np.abs(upper - pred)
    out = np.where(
        idx == 0,
        sorted_pressures[0],
        np.where(
            idx == total_pressures_len,
            sorted_pressures[-1],
            np.where(choose_lower, lower, upper),
        ),
    )
    return out.astype(np.float32, copy=False)


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    l = []
    for i in range(len(input_list)):
        public_lb_score = int(input_list[i].split("/")[-1].split(".")[1].split(" ")[0])
        l.append(public_lb_score)
        input_list[i] = (pd.read_csv(input_list[i]).pressure).to_numpy().ravel()
    output = 0
    l_sum = sum(l)
    if len(input_list) == 1:
        output = input_list[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def g(dp):
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)
    file_count = len(l)
    loop_time = 150
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
    for loop_idx in range(loop_time):
        weight = []
        set_seed(loop_idx)
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


def blend(a, b):
    a = pd.read_csv(
        a, usecols=["id", "pressure"], dtype={"id": np.int64, "pressure": np.float32}
    )
    b = pd.read_csv(b, usecols=["pressure"], dtype={"pressure": np.float32})
    a.pressure = a.pressure * np.float32(0.55) + b.pressure * np.float32(0.45)
    a.to_csv("blend.csv", index=False)
    return a


def _add_features(df):
    df = df.sort_values(["breath_id", "time_step"], kind="mergesort").copy()

    n = len(df)
    if n == 0:
        return df
    if n % 80 != 0:
        g = df.groupby("breath_id", sort=False)

        df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0)
        df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0)
        df["u_out_lag1"] = g["u_out"].shift(1).fillna(0).astype(np.int64)

        df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
        df["u_in_diff2"] = df["u_in_lag1"] - df["u_in_lag2"]

        df["u_in_cumsum"] = g["u_in"].cumsum()
        df["u_in_cummean"] = df["u_in_cumsum"] / (g.cumcount() + 1)

        df["RC"] = df["R"].astype(np.float32) * df["C"].astype(np.float32)
        df["u_in_x_R"] = df["u_in"] * df["R"]
        df["u_in_x_C"] = df["u_in"] * df["C"]

        df["u_in_roll3_mean"] = (
            g["u_in"]
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .astype(np.float32)
        )

        uout = df["u_out"].to_numpy()
        bid = df["breath_id"].to_numpy()
        n2 = uout.shape[0]
        run = np.zeros(n2, dtype=np.int16)
        r = 0
        prev_bid = bid[0] if n2 else -1
        for i in range(n2):
            if bid[i] != prev_bid:
                r = 0
                prev_bid = bid[i]
            if uout[i] == 1:
                r += 1
            else:
                r = 0
            run[i] = r
        df["u_out_runlen"] = run.astype(np.int16)

        df["u_out_cumsum"] = g["u_out"].cumsum().astype(np.int16)
        df["u_in_ewm_span5"] = (
            g["u_in"]
            .ewm(span=5, adjust=False)
            .mean()
            .reset_index(level=0, drop=True)
            .astype(np.float32)
        )

        df["RC_cat"] = (
            df["R"].astype(np.int16) * 100 + df["C"].astype(np.int16)
        ).astype(np.int16)
        return df

    breaths = n // 80

    u_in = df["u_in"].to_numpy(np.float32, copy=False).reshape(breaths, 80)
    u_out = df["u_out"].to_numpy(np.int8, copy=False).reshape(breaths, 80)
    R = df["R"].to_numpy(np.int16, copy=False).reshape(breaths, 80)
    C = df["C"].to_numpy(np.int16, copy=False).reshape(breaths, 80)

    u_in_lag1 = np.empty_like(u_in)
    u_in_lag1[:, 0] = 0.0
    u_in_lag1[:, 1:] = u_in[:, :-1]

    u_in_lag2 = np.empty_like(u_in)
    u_in_lag2[:, :2] = 0.0
    u_in_lag2[:, 2:] = u_in[:, :-2]

    u_out_lag1 = np.empty((breaths, 80), dtype=np.int64)
    u_out_lag1[:, 0] = 0
    u_out_lag1[:, 1:] = u_out[:, :-1].astype(np.int64, copy=False)

    u_in_diff1 = u_in - u_in_lag1
    u_in_diff2 = u_in_lag1 - u_in_lag2

    u_in_cumsum = np.cumsum(u_in, axis=1, dtype=np.float32)
    counts = (np.arange(80, dtype=np.float32) + 1.0)[None, :]
    u_in_cummean = u_in_cumsum / counts

    R_f = R.astype(np.float32, copy=False)
    C_f = C.astype(np.float32, copy=False)
    RC = R_f * C_f
    u_in_x_R = u_in * R_f
    u_in_x_C = u_in * C_f

    from numpy.lib.stride_tricks import sliding_window_view

    win3 = sliding_window_view(u_in, window_shape=3, axis=1)  # (breaths, 78, 3)
    roll3_core = win3.mean(axis=2, dtype=np.float32)  # (breaths, 78)
    u_in_roll3_mean = np.empty_like(u_in)
    u_in_roll3_mean[:, 0] = u_in[:, 0]
    u_in_roll3_mean[:, 1] = (u_in[:, 0] + u_in[:, 1]) * 0.5
    u_in_roll3_mean[:, 2:] = roll3_core

    run = np.zeros((breaths, 80), dtype=np.int16)
    for t in range(80):
        if t == 0:
            run[:, 0] = (u_out[:, 0] == 1).astype(np.int16)
        else:
            is1 = u_out[:, t] == 1
            run[:, t] = np.where(is1, (run[:, t - 1] + 1).astype(np.int16), 0)

    u_out_cumsum = np.cumsum(u_out, axis=1, dtype=np.int16)

    span = 5.0
    alpha = np.float32(2.0 / (span + 1.0))
    one_m_alpha = np.float32(1.0) - alpha
    u_in_ewm_span5 = np.empty_like(u_in)
    u_in_ewm_span5[:, 0] = u_in[:, 0]
    for t in range(1, 80):
        u_in_ewm_span5[:, t] = (
            alpha * u_in[:, t] + one_m_alpha * u_in_ewm_span5[:, t - 1]
        )

    RC_cat = (
        R.astype(np.int16, copy=False) * 100 + C.astype(np.int16, copy=False)
    ).astype(np.int16, copy=False)

    df["u_in_lag1"] = u_in_lag1.reshape(-1)
    df["u_in_lag2"] = u_in_lag2.reshape(-1)
    df["u_out_lag1"] = u_out_lag1.reshape(-1)

    df["u_in_diff1"] = u_in_diff1.reshape(-1)
    df["u_in_diff2"] = u_in_diff2.reshape(-1)

    df["u_in_cumsum"] = u_in_cumsum.reshape(-1)
    df["u_in_cummean"] = u_in_cummean.reshape(-1)

    df["RC"] = RC.reshape(-1)
    df["u_in_x_R"] = u_in_x_R.reshape(-1)
    df["u_in_x_C"] = u_in_x_C.reshape(-1)

    df["u_in_roll3_mean"] = u_in_roll3_mean.reshape(-1).astype(np.float32, copy=False)

    df["u_out_runlen"] = run.reshape(-1)
    df["u_out_cumsum"] = u_out_cumsum.reshape(-1)

    df["u_in_ewm_span5"] = u_in_ewm_span5.reshape(-1).astype(np.float32, copy=False)

    df["RC_cat"] = RC_cat.reshape(-1)
    return df


_DATA_CACHE = {}


def _get_preprocessed():
    key = "v2_fast_numpy_features"
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]

    usecols_train = [
        "id",
        "breath_id",
        "R",
        "C",
        "time_step",
        "u_in",
        "u_out",
        "pressure",
    ]
    usecols_test = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]

    train = pd.read_csv(
        "../input/ventilator-pressure-prediction/train.csv",
        usecols=usecols_train,
        dtype={
            "id": np.int32,
            "breath_id": np.int32,
            "R": np.int16,
            "C": np.int16,
            "time_step": np.float32,
            "u_in": np.float32,
            "u_out": np.int8,
            "pressure": np.float32,
        },
    )
    test = pd.read_csv(
        "../input/ventilator-pressure-prediction/test.csv",
        usecols=usecols_test,
        dtype={
            "id": np.int32,
            "breath_id": np.int32,
            "R": np.int16,
            "C": np.int16,
            "time_step": np.float32,
            "u_in": np.float32,
            "u_out": np.int8,
        },
    )

    train = _add_features(train)
    test = _add_features(test)

    feats = [
        "R",
        "C",
        "time_step",
        "u_in",
        "u_out",
        "u_in_lag1",
        "u_in_lag2",
        "u_out_lag1",
        "u_in_diff1",
        "u_in_diff2",
        "u_in_cumsum",
        "u_in_cummean",
        "RC",
        "u_in_x_R",
        "u_in_x_C",
        "u_in_roll3_mean",
        "u_out_runlen",
        "u_out_cumsum",
        "u_in_ewm_span5",
        "RC_cat",
    ]

    X_tr_3d = (
        train[feats].to_numpy(dtype=np.float32, copy=False).reshape(-1, 80, len(feats))
    )
    y_tr = train["pressure"].to_numpy(dtype=np.float32, copy=False).reshape(-1, 80)
    uout_tr = train["u_out"].to_numpy(dtype=np.int8, copy=False).reshape(-1, 80)

    X_te_3d = (
        test[feats].to_numpy(dtype=np.float32, copy=False).reshape(-1, 80, len(feats))
    )
    uout_te = test["u_out"].to_numpy(dtype=np.int8, copy=False).reshape(-1, 80)

    X_tr = X_tr_3d.reshape(X_tr_3d.shape[0], -1)
    X_te = X_te_3d.reshape(X_te_3d.shape[0], -1)

    y_tr_adj = y_tr.copy()
    y_tr_adj[uout_tr == 1] = 0.0

    test_id = test["id"].to_numpy(dtype=np.int64, copy=False)
    test_uout_flat = test["u_out"].to_numpy(dtype=np.int8, copy=False)

    del train, test, X_tr_3d, X_te_3d, y_tr, uout_tr
    gc.collect()

    _DATA_CACHE[key] = (X_tr, y_tr_adj, X_te, uout_te, test_id, test_uout_flat)
    return _DATA_CACHE[key]


def _make_hgb_submission(path, max_iter=250, learning_rate=0.08, max_depth=7):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor

    X_tr, y_tr_adj, X_te, uout_te, test_id, test_uout_flat = _get_preprocessed()

    base = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        max_iter=int(max_iter),
        random_state=2021,
    )
    model = MultiOutputRegressor(base, n_jobs=1)

    model.fit(X_tr, y_tr_adj)
    pred = model.predict(X_te).astype(np.float32)  # (n_breaths, 80)
    pred[uout_te == 1] = 0.0

    pred_flat = pred.reshape(-1)

    sub = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv",
        usecols=["id", "pressure"],
        dtype={"id": np.int64, "pressure": np.float32},
    )
    sub["pressure"] = pred_flat.astype(np.float32, copy=False)

    insp = test_uout_flat == 0
    snapped = sub["pressure"].to_numpy(dtype=np.float32, copy=False)
    snapped[insp] = find_nearest_vec(snapped[insp])
    snapped[~insp] = 0.0
    sub["pressure"] = snapped

    sub.to_csv(path, index=False)
    return path




## === cell 1
a = "../input/gb-blending/0.156 seed 33.csv"
b = "../input/gb-blending/0.157 blend.csv"

if not os.path.exists(a) or not os.path.exists(b):
    a = _make_hgb_submission(
        "fallback_a.csv", max_iter=220, learning_rate=0.08, max_depth=7
    )
    b = _make_hgb_submission(
        "fallback_b.csv", max_iter=260, learning_rate=0.06, max_depth=8
    )

blend_df = blend(a, b)

_, _, _, _, test_id, test_uout_flat = _get_preprocessed()

insp = test_uout_flat == 0
p = blend_df["pressure"].to_numpy(dtype=np.float32, copy=False)
p[~insp] = 0.0
p[insp] = find_nearest_vec(p[insp])
blend_df["pressure"] = p
blend_df = blend_df[["id", "pressure"]]

blend_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", blend_df.shape)
print(blend_df.head())
