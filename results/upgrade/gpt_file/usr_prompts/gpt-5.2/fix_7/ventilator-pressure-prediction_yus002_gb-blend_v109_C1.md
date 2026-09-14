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


def find_nearest_vec(preds: np.ndarray) -> np.ndarray:
    preds = np.asarray(preds, dtype=np.float64)
    idx = np.searchsorted(sorted_pressures, preds, side="left")

    out = np.empty_like(preds, dtype=sorted_pressures.dtype)

    hi_mask = idx >= total_pressures_len
    lo_mask = idx <= 0
    mid_mask = (~hi_mask) & (~lo_mask)

    out[hi_mask] = sorted_pressures[-1]
    out[lo_mask] = sorted_pressures[0]

    if np.any(mid_mask):
        mid_idx = idx[mid_mask]
        lower = sorted_pressures[mid_idx - 1]
        upper = sorted_pressures[mid_idx]
        p = preds[mid_mask]
        choose_lower = np.abs(lower - p) < np.abs(upper - p)
        out[mid_mask] = np.where(choose_lower, lower, upper)

    return out.astype(np.float64, copy=False)


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
    output["pressure"] = find_nearest_vec(output["pressure"].to_numpy())
    output.to_csv("rwb_154_loops.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.5 + b.pressure * 0.5
    a["pressure"] = find_nearest_vec(a["pressure"].to_numpy())
    a.to_csv("blend.csv", index=False)
    return a




## === cell 1
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

a = "../input/gb-data-blending-recover/0.148 blend.csv"
b = "../input/gb-data-blending-recover/0.148 blend2.csv"

sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"

sub = pd.read_csv(sample_path)
test = pd.read_csv(test_path)

if os.path.exists(a) and os.path.exists(b):
    out = blend(a, b)
    out[["id", "pressure"]].to_csv("submission.csv", index=False)
else:

    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        n = len(df)

        bid = df["breath_id"].to_numpy()
        time_step = df["time_step"].to_numpy(dtype=np.float64, copy=False)
        u_in = df["u_in"].to_numpy(dtype=np.float64, copy=False)
        u_out = df["u_out"].to_numpy(dtype=np.int8, copy=False)
        R = df["R"].to_numpy(dtype=np.float64, copy=False)
        C = df["C"].to_numpy(dtype=np.float64, copy=False)

        new_breath = np.empty(n, dtype=bool)
        new_breath[0] = True
        new_breath[1:] = bid[1:] != bid[:-1]

        breath_step = np.empty(n, dtype=np.int16)
        breath_step[0] = 0
        breath_step[1:] = np.where(new_breath[1:], 0, breath_step[:-1] + 1)

        dt = np.empty(n, dtype=np.float64)
        dt[0] = 0.0
        dt[1:] = np.where(new_breath[1:], 0.0, time_step[1:] - time_step[:-1])

        u_in_lag1 = np.empty(n, dtype=np.float64)
        u_in_lag2 = np.empty(n, dtype=np.float64)
        u_out_lag1 = np.empty(n, dtype=np.float64)  # matches original fillna(0.0) float

        u_in_lag1[0] = 0.0
        u_in_lag2[0] = 0.0
        u_out_lag1[0] = 0.0

        u_in_lag1[1:] = np.where(new_breath[1:], 0.0, u_in[:-1])
        u_out_lag1[1:] = np.where(new_breath[1:], 0.0, u_out[:-1].astype(np.float64))

        u_in_lag2[1] = 0.0
        if n > 2:
            same1 = ~new_breath[1:]
            same2 = np.empty(n, dtype=bool)
            same2[:2] = False
            same2[2:] = same1[1:] & same1[:-1]
            u_in_lag2[2:] = np.where(same2[2:], u_in[:-2], 0.0)

        u_in_diff1 = np.empty(n, dtype=np.float64)
        u_in_diff2 = np.empty(n, dtype=np.float64)
        u_in_diff1[0] = 0.0
        u_in_diff2[0] = 0.0
        u_in_diff1[1:] = np.where(new_breath[1:], 0.0, u_in[1:] - u_in[:-1])
        u_in_diff2[1] = 0.0
        if n > 2:
            u_in_diff2[2:] = np.where(same2[2:], u_in[2:] - u_in[:-2], 0.0)

        u_in_dt = u_in * dt

        u_in_cum = np.empty(n, dtype=np.float64)
        dt_cum = np.empty(n, dtype=np.float64)
        u_out_cum = np.empty(n, dtype=np.int16)
        u_in_dt_cum = np.empty(n, dtype=np.float64)

        cu_in = 0.0
        cdt = 0.0
        cu_out = 0
        cu_in_dt = 0.0
        prev_bid = bid[0]
        for i in range(n):
            if bid[i] != prev_bid:
                cu_in = 0.0
                cdt = 0.0
                cu_out = 0
                cu_in_dt = 0.0
                prev_bid = bid[i]
            cu_in += u_in[i]
            cdt += dt[i]
            cu_out += int(u_out[i])
            cu_in_dt += u_in_dt[i]

            u_in_cum[i] = cu_in
            dt_cum[i] = cdt
            u_out_cum[i] = cu_out
            u_in_dt_cum[i] = cu_in_dt

        u_in_over_R = u_in / np.where(R == 0, np.nan, R)
        u_in_over_R = np.nan_to_num(u_in_over_R, nan=0.0)

        df["breath_step"] = breath_step
        df["u_in_cum"] = u_in_cum
        df["dt"] = dt
        df["dt_cum"] = dt_cum
        df["u_in_lag1"] = u_in_lag1
        df["u_in_lag2"] = u_in_lag2
        df["u_out_lag1"] = u_out_lag1
        df["u_in_over_R"] = u_in_over_R
        df["u_in_x_C"] = u_in * C
        df["u_in_x_R"] = u_in * R
        df["u_out_cum"] = u_out_cum
        df["u_in_diff1"] = u_in_diff1
        df["u_in_diff2"] = u_in_diff2
        df["u_in_dt"] = u_in_dt
        df["u_in_dt_cum"] = u_in_dt_cum

        return df

    train_fe = add_features(df_train)
    test_fe = add_features(test)

    tr = train_fe[train_fe["u_out"] == 0].copy()

    if tr.empty:
        base_pred = find_nearest(float(df_train["pressure"].median()))
        sub["pressure"] = base_pred
    else:
        feat_cols = [
            "R",
            "C",
            "time_step",
            "breath_step",
            "u_in",
            "u_out",
            "u_in_cum",
            "dt",
            "dt_cum",
            "u_in_lag1",
            "u_in_lag2",
            "u_out_lag1",
            "u_in_over_R",
            "u_in_x_C",
            "u_in_x_R",
            "u_out_cum",
            "u_in_diff1",
            "u_in_diff2",
            "u_in_dt",
            "u_in_dt_cum",
        ]
        X_tr = tr[feat_cols]
        y_tr = tr["pressure"].astype(float)
        X_te = test_fe[feat_cols]

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "knn",
                    KNeighborsRegressor(
                        n_neighbors=45,
                        weights="distance",
                        metric="minkowski",
                        p=2,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        model.fit(X_tr, y_tr)

        preds = model.predict(X_te)

        preds = find_nearest_vec(preds)

        neutral = float(np.median(y_tr))
        neutral = float(find_nearest(neutral))
        preds = np.where(test_fe["u_out"].to_numpy() == 1, neutral, preds)

        sub["pressure"] = preds

    sub = sub[["id", "pressure"]]
    sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
