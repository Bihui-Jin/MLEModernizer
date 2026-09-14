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
import numpy as np
import pandas as pd
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
min_pressure = float(sorted_pressures[0])
max_pressure = float(sorted_pressures[-1])


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


def snap_to_known_pressures(pred_arr: np.ndarray) -> np.ndarray:
    pred_arr = pred_arr.astype(np.float64, copy=False)
    idx = np.searchsorted(sorted_pressures, pred_arr, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)

    lower_idx = np.clip(idx - 1, 0, total_pressures_len - 1)
    upper_idx = idx

    lower = sorted_pressures[lower_idx]
    upper = sorted_pressures[upper_idx]

    choose_lower = np.abs(pred_arr - lower) < np.abs(pred_arr - upper)
    out = np.where(choose_lower, lower, upper)
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
        input_list[i] = (pd.read_csv(input_list[i]).pressure).ravel()
    output = 0
    l_sum = sum(l)
    if len(input_list) == 1:
        output = input_list[0]
    else:
        weight1 = (l[1] / l_sum) + 0.15
        weight2 = 1 - weight1
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def g(dp):
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
    for i in range(loop_time):
        weight = []
        set_seed(i)
        for i in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        for i in range(len(weight)):
            weight[i] /= weight_sum
        weight.sort(reverse=True)
        temp = 0
        for i in range(len(flist)):
            temp += flist[i] * weight[i]
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
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.5 + b.pressure * 0.5
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["breath_id", "time_step"]).copy()

    g = df.groupby("breath_id", sort=False)

    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0)
    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0).astype(np.int64)

    dt = g["time_step"].diff().fillna(0.0)
    df["dt"] = dt

    df["du_in"] = (df["u_in"] - df["u_in_lag1"]).astype(np.float64)
    df["du_in_dt"] = np.where(df["dt"].to_numpy() > 0, df["du_in"] / df["dt"], 0.0)

    df["u_in_cum"] = g["u_in"].cumsum()
    df["u_in_area"] = (df["u_in"] * dt).groupby(df["breath_id"], sort=False).cumsum()

    eff_u_in = df["u_in"] * (1.0 - df["u_out"].astype(np.float64))
    df["eff_u_in_area"] = (eff_u_in * dt).groupby(df["breath_id"], sort=False).cumsum()

    df["t2"] = df["time_step"].astype(np.float64) ** 2

    df["breath_step"] = g.cumcount().astype(np.int16)  # 0..79
    df["breath_step2"] = (df["breath_step"].astype(np.float64) ** 2).astype(np.float64)
    df["breath_step3"] = (df["breath_step"].astype(np.float64) ** 3).astype(np.float64)

    df["u_in_x_time"] = (
        df["u_in"].astype(np.float64) * df["time_step"].astype(np.float64)
    ).astype(np.float64)
    df["u_in_x_step"] = (
        df["u_in"].astype(np.float64) * df["breath_step"].astype(np.float64)
    ).astype(np.float64)

    if "pressure" in df.columns:
        df["pressure_lag1"] = g["pressure"].shift(1).fillna(0.0).astype(np.float64)
    else:
        df["pressure_lag1"] = 0.0

    return df


train_fe = add_time_features(df_train)
test_fe = add_time_features(df_test)

train_insp = train_fe[train_fe["u_out"] == 0].copy()
global_insp_mean = float(train_insp["pressure"].mean())



## === cell 3
FEATURES = [
    "time_step",
    "t2",
    "breath_step",
    "breath_step2",
    "breath_step3",
    "u_in",
    "u_in_lag1",
    "u_in_lag2",
    "du_in",
    "du_in_dt",
    "u_in_cum",
    "u_in_area",
    "eff_u_in_area",
    "u_in_x_time",
    "u_in_x_step",
    "u_out",
    "u_out_lag1",
    "pressure_lag1",
]

feat_mu = train_insp[FEATURES].astype(np.float64).mean(axis=0).to_numpy()
feat_sigma = (
    train_insp[FEATURES].astype(np.float64).std(axis=0).replace(0.0, 1.0).to_numpy()
)


def _build_design(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURES].astype(np.float64).to_numpy()
    X = (X - feat_mu) / feat_sigma
    ones = np.ones((X.shape[0], 1), dtype=np.float64)
    return np.hstack([ones, X])


def fit_ridge_closed_form(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1e-3,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    if sample_weight is None:
        XtX = X.T @ X
        Xty = X.T @ y
    else:
        w = sample_weight.astype(np.float64, copy=False)
        Xw = X * w[:, None]
        XtX = X.T @ Xw
        Xty = X.T @ (y * w)

    reg = np.eye(XtX.shape[0], dtype=np.float64) * alpha
    reg[0, 0] = 0.0
    coef = np.linalg.solve(XtX + reg, Xty)
    return coef


def predict_linear(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return X @ w


step_norm = train_insp["breath_step"].astype(np.float64).to_numpy() / 79.0
global_sw = 0.7 + 0.6 * step_norm  # range ~[0.7, 1.3]

weights = {}
Xg = _build_design(train_insp)
yg = train_insp["pressure"].astype(np.float64).to_numpy()
fallback_w = fit_ridge_closed_form(Xg, yg, alpha=1e-2, sample_weight=global_sw)

for (R, C), df_rc in train_insp.groupby(["R", "C"], sort=False):
    X = _build_design(df_rc)
    y = df_rc["pressure"].astype(np.float64).to_numpy()
    step_norm_rc = df_rc["breath_step"].astype(np.float64).to_numpy() / 79.0
    sw_rc = 0.7 + 0.6 * step_norm_rc
    w = fit_ridge_closed_form(X, y, alpha=5e-2, sample_weight=sw_rc)
    weights[(int(R), int(C))] = w

test_fe = test_fe.sort_values(["breath_id", "time_step"]).reset_index(
    drop=False
)  # keep original index
orig_index = test_fe["index"].to_numpy()

pred = np.empty(len(test_fe), dtype=np.float64)

for breath_id, idx in test_fe.groupby("breath_id", sort=False).groups.items():
    idx_arr = np.asarray(idx, dtype=np.int64)
    df_b = test_fe.loc[idx_arr].copy()

    Rv = int(df_b["R"].iloc[0])
    Cv = int(df_b["C"].iloc[0])
    w = weights.get((Rv, Cv), fallback_w)

    prev_p = global_insp_mean  # stable init; will be overwritten after first step
    for j, ridx in enumerate(idx_arr):
        df_row = test_fe.loc[[ridx], FEATURES].copy()
        if j == 0:
            df_row["pressure_lag1"] = 0.0
        else:
            df_row["pressure_lag1"] = float(prev_p)

        Xrow = _build_design(df_row)
        pj = float(predict_linear(Xrow, w)[0])
        if not np.isfinite(pj):
            pj = global_insp_mean
        pj = float(np.clip(pj, min_pressure, max_pressure))
        pred[ridx] = pj
        prev_p = pj

pred = pred[np.argsort(np.argsort(orig_index))]

pred = np.where(np.isfinite(pred), pred, global_insp_mean)
pred = np.clip(pred, min_pressure, max_pressure)

u_out_arr = df_test["u_out"].to_numpy()
pred_adj = pred.copy()

test_tmp = df_test[["breath_id", "time_step", "u_out"]].copy()
test_tmp["pred"] = pred_adj
insp_mask = test_tmp["u_out"].to_numpy() == 0

insp_last = (
    test_tmp.loc[insp_mask, ["breath_id", "time_step", "pred"]]
    .sort_values(["breath_id", "time_step"])
    .groupby("breath_id", sort=False)["pred"]
    .last()
)

last_insp_for_row = (
    test_tmp["breath_id"]
    .map(insp_last)
    .astype(np.float64)
    .fillna(global_insp_mean)
    .to_numpy()
)
pred_adj[u_out_arr == 1] = last_insp_for_row[u_out_arr == 1]

pred_adj = np.clip(pred_adj, min_pressure, max_pressure)
pred_snap = snap_to_known_pressures(pred_adj)

submission = pd.DataFrame({"id": df_test["id"].astype(np.int64), "pressure": pred_snap})
submission = submission.sort_values("id").reset_index(drop=True)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print(
    "Pred stats (adj):",
    float(np.min(pred_adj)),
    float(np.mean(pred_adj)),
    float(np.max(pred_adj)),
)
print(
    "Pred stats (snapped):",
    float(np.min(pred_snap)),
    float(np.mean(pred_snap)),
    float(np.max(pred_snap)),
)
print("Inspiratory mean (train):", global_insp_mean)
print("Fitted per-(R,C) models:", len(weights))
print("Pressure range:", min_pressure, max_pressure)
