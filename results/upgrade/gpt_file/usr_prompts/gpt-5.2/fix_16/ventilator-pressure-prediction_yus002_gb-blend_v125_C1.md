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


def snap_to_nearest_pressure(pred: np.ndarray) -> np.ndarray:
    p = pred.astype(np.float32, copy=False)
    idx = np.searchsorted(sorted_pressures, p, side="left")

    idx_hi = np.clip(idx, 0, total_pressures_len - 1)
    idx_lo = np.clip(idx - 1, 0, total_pressures_len - 1)

    hi = sorted_pressures[idx_hi]
    lo = sorted_pressures[idx_lo]

    choose_lo = (idx == total_pressures_len) | (
        (idx > 0) & (np.abs(lo - p) < np.abs(hi - p))
    )
    out = np.where(choose_lo, lo, hi).astype(np.float32)
    return out


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    l = []
    for i in range(len(input_list)):
        try:
            public_lb_score = int(
                input_list[i].split("/")[-1].split(".")[1].split(" ")[0]
            )
        except Exception:
            public_lb_score = 0
        l.append(public_lb_score)
        input_list[i] = (pd.read_csv(input_list[i]).pressure).ravel()
    output = 0
    l_sum = sum(l) if sum(l) != 0 else 1
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
    a.pressure = a.pressure * 0.6 + b.pressure * 0.4
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

test_df = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
sample_sub = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)


def add_minimal_ts_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["breath_id", "time_step"]).copy()

    df["u_in_lag1"] = (
        df.groupby("breath_id")["u_in"].shift(1).fillna(0.0).astype(np.float32)
    )

    df["u_in_lag2"] = (
        df.groupby("breath_id")["u_in"].shift(2).fillna(0.0).astype(np.float32)
    )

    df["u_in_roll3"] = (
        df.groupby("breath_id")["u_in"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .astype(np.float32)
    )

    df["u_out_lag1"] = (
        df.groupby("breath_id")["u_out"].shift(1).fillna(0).astype(np.int8)
    )
    df["is_first_exp"] = (
        (df["u_out"].astype(np.int8) == 1) & (df["u_out_lag1"] == 0)
    ).astype(np.int8)

    df["dt"] = (
        df.groupby("breath_id")["time_step"].diff().fillna(0.0).astype(np.float32)
    )
    df["u_in_dt"] = (df["u_in"].astype(np.float32) * df["dt"]).astype(np.float32)
    df["u_in_dt_cum"] = df.groupby("breath_id")["u_in_dt"].cumsum().astype(np.float32)

    df["t_from_start"] = df["time_step"].astype(np.float32) - df.groupby("breath_id")[
        "time_step"
    ].transform("first").astype(np.float32)
    df["u_in_cum"] = df.groupby("breath_id")["u_in"].cumsum().astype(np.float32)
    df["u_in_diff1"] = (
        df.groupby("breath_id")["u_in"].diff().fillna(0.0).astype(np.float32)
    )

    df["u_in_over_R"] = (
        df["u_in"].astype(np.float32) / df["R"].astype(np.float32)
    ).astype(np.float32)
    df["u_in_over_C"] = (
        df["u_in"].astype(np.float32) / df["C"].astype(np.float32)
    ).astype(np.float32)

    df["step"] = df.groupby("breath_id").cumcount().astype(np.int16)
    df["step_frac"] = (df["step"].astype(np.float32) / 79.0).astype(np.float32)

    if "pressure" in df.columns:
        df["pressure_lag1"] = (
            df.groupby("breath_id")["pressure"].shift(1).fillna(0.0).astype(np.float32)
        )
    else:
        df["pressure_lag1"] = np.float32(0.0)

    return df


def carry_last_inspiratory_to_expiratory_by_breath(
    pred: np.ndarray, breath_id: np.ndarray, u_out: np.ndarray
) -> np.ndarray:
    dfp = pd.DataFrame(
        {"breath_id": breath_id, "u_out": u_out, "pred": pred.astype(np.float32)}
    )
    dfp.loc[dfp["u_out"].values == 1, "pred"] = np.nan
    dfp["pred"] = dfp.groupby("breath_id")["pred"].ffill().bfill()
    return dfp["pred"].to_numpy(dtype=np.float32)


train_fe = add_minimal_ts_features(df_train)
test_fe = add_minimal_ts_features(test_df)

test_df = test_df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)
test_fe = test_fe.reset_index(drop=True)

train_fe_insp = train_fe[train_fe["u_out"] == 0].copy()

train_fe_insp = train_fe_insp.sort_values(["breath_id", "time_step"]).copy()
train_fe_insp["pressure_lag1"] = (
    train_fe_insp.groupby("breath_id")["pressure"]
    .shift(1)
    .fillna(0.0)
    .astype(np.float32)
)

features = [
    "R",
    "C",
    "time_step",
    "u_in",
    "u_in_lag1",
    "u_in_lag2",
    "u_in_roll3",
    "u_in_dt_cum",
    "t_from_start",
    "u_in_cum",
    "u_in_diff1",
    "u_in_over_R",
    "u_in_over_C",
    "step",
    "step_frac",
    "u_out_lag1",
    "is_first_exp",
    "pressure_lag1",
]
target = "pressure"

missing_train = [c for c in features if c not in train_fe_insp.columns]
missing_test = [c for c in features if c not in test_fe.columns]
if missing_train or missing_test:
    raise KeyError(
        f"Missing features. train_missing={missing_train}, test_missing={missing_test}"
    )

global_preprocess = ColumnTransformer(
    transformers=[("num", StandardScaler(), features)],
    remainder="drop",
)
global_preprocess.fit(train_fe_insp[features])

model = Pipeline(
    steps=[
        ("prep", global_preprocess),
        (
            "knn",
            KNeighborsRegressor(
                n_neighbors=15,
                weights="distance",
                metric="minkowski",
            ),
        ),
    ]
)

pred = np.full(len(test_fe), np.nan, dtype=np.float32)

train_insp_groups = {k: v for k, v in train_fe_insp.groupby(["R", "C"])}

test_index_by_breath = test_fe.groupby("breath_id").indices

for (r, c), test_idx_all in test_fe.groupby(["R", "C"]).groups.items():
    test_idx_all = np.asarray(list(test_idx_all), dtype=np.int64)

    train_grp_insp = train_insp_groups.get((r, c), None)
    if train_grp_insp is None or len(train_grp_insp) < 200:
        train_grp_insp = train_fe_insp[
            (train_fe_insp["R"] == r) & (train_fe_insp["C"] == c)
        ]

    X_train = train_grp_insp[features]
    y_train = train_grp_insp[target].astype(np.float32)
    model.fit(X_train, y_train)

    breaths_in_group = test_fe.loc[test_idx_all, "breath_id"].unique()
    for bid in breaths_in_group:
        idxs = np.asarray(test_index_by_breath[bid], dtype=np.int64)
        idxs = idxs[
            (test_fe.loc[idxs, "R"].values == r) & (test_fe.loc[idxs, "C"].values == c)
        ]
        if idxs.size == 0:
            continue

        last_p = np.float32(0.0)
        for row_idx in idxs:
            if test_fe.loc[row_idx, "u_out"] == 1:
                continue

            X_row = test_fe.loc[[row_idx], features].copy()
            X_row.loc[:, "pressure_lag1"] = last_p
            p_hat = np.float32(model.predict(X_row)[0])
            pred[row_idx] = p_hat
            last_p = p_hat

pred = carry_last_inspiratory_to_expiratory_by_breath(
    pred, test_fe["breath_id"].values, test_fe["u_out"].values
)

pred = snap_to_nearest_pressure(pred)

test_df_raw = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
test_df_raw["_orig_row"] = np.arange(len(test_df_raw), dtype=np.int64)
test_df_sorted = test_df_raw.sort_values(["breath_id", "time_step"]).reset_index(
    drop=True
)

pred_back = np.empty_like(pred)
pred_back[test_df_sorted["_orig_row"].to_numpy()] = pred

submission = pd.DataFrame(
    {"id": sample_sub["id"].values, "pressure": pred_back.astype(np.float32)}
)

assert list(submission.columns) == ["id", "pressure"]
assert submission.shape[0] == sample_sub.shape[0] == len(test_df_raw)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print(
    f"Wrote {submission_path} with shape={submission.shape} and columns={list(submission.columns)}"
)
print(submission.head())
