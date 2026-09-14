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
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def g(dp):
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)
    file_count = len(l)
    loop_time = file_count**2
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
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
sample_sub = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)

global_median = float(df_train["pressure"].median())

sub_a = sample_sub.copy()
sub_a["pressure"] = global_median
sub_a["pressure"] = sub_a["pressure"].apply(find_nearest)
a_path = "baseline_a.csv"
sub_a.to_csv(a_path, index=False)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["breath_id", "time_step"], kind="mergesort")

    g = df.groupby("breath_id", sort=False)

    df["breath_time_idx"] = g.cumcount().astype(np.int16)

    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0)
    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0.0)

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_in_diff2"] = df["u_in_lag1"] - df["u_in_lag2"]

    dt = g["time_step"].diff().fillna(0.0)
    df["u_in_cum"] = (df["u_in"] * dt).groupby(df["breath_id"], sort=False).cumsum()
    df["u_out_cum"] = df["u_out"].groupby(df["breath_id"], sort=False).cumsum()

    df["u_in_lag3"] = g["u_in"].shift(3).fillna(0.0)
    df["u_in_diff3"] = df["u_in_lag2"] - df["u_in_lag3"]

    return df


df_train_feat = add_features(df_train)
df_test_feat = add_features(df_test)

df_train_feat["RC"] = (df_train_feat["R"] * df_train_feat["C"]).astype(np.float32)
df_test_feat["RC"] = (df_test_feat["R"] * df_test_feat["C"]).astype(np.float32)

df_train_feat["p_lag1"] = (
    df_train_feat.sort_values(["breath_id", "time_step"], kind="mergesort")
    .groupby("breath_id", sort=False)["pressure"]
    .shift(1)
    .fillna(0.0)
    .astype(np.float32)
)

feature_cols = [
    "R",
    "C",
    "RC",
    "breath_time_idx",
    "time_step",
    "u_in",
    "u_out",
    "u_in_lag1",
    "u_in_lag2",
    "u_in_lag3",
    "u_out_lag1",
    "u_in_cum",
    "u_out_cum",
    "u_in_diff1",
    "u_in_diff2",
    "u_in_diff3",
    "p_lag1",
]

df_train_insp = df_train_feat[df_train_feat["u_out"] == 0].copy()

rng = np.random.RandomState(2021)
rows_per_breath = 25  # minimal increase in signal vs earlier tiny samples; still fast

df_train_insp = df_train_insp.sort_values(["breath_id", "time_step"], kind="mergesort")
g_insp = df_train_insp.groupby("breath_id", sort=False)

idx_list = []
for _, grp in g_insp:
    n = len(grp)
    if n <= rows_per_breath:
        idx_list.append(grp.index.values)
    else:
        take = np.linspace(0, n - 1, rows_per_breath).round().astype(int)
        idx_list.append(grp.index.values[take])

train_idx = np.concatenate(idx_list)
df_train_bal = df_train_insp.loc[train_idx].copy()

X_train = df_train_bal[feature_cols].astype(np.float32)
y_train = df_train_bal["pressure"].astype(np.float32)

knn_model = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "knn",
            KNeighborsRegressor(
                n_neighbors=50,
                weights="distance",
                metric="minkowski",
                p=2,
                n_jobs=-1,
            ),
        ),
    ]
)

knn_model.fit(X_train, y_train)

df_test_ordered = df_test_feat.sort_values(
    ["breath_id", "time_step"], kind="mergesort"
).copy()
test_breath_ids = df_test_ordered["breath_id"].values
n_test = len(df_test_ordered)

pred_ordered = np.zeros(n_test, dtype=np.float32)
p_prev = 0.0
prev_breath = -1

X_static = (
    df_test_ordered[[c for c in feature_cols if c != "p_lag1"]]
    .astype(np.float32)
    .values
)

for i in range(n_test):
    b = test_breath_ids[i]
    if b != prev_breath:
        p_prev = 0.0
        prev_breath = b

    x_row = np.concatenate(
        [X_static[i], np.array([p_prev], dtype=np.float32)], axis=0
    ).reshape(1, -1)
    p_hat = float(knn_model.predict(x_row)[0])

    if int(df_test_ordered["u_out"].iat[i]) == 1:
        p_hat = 0.0

    pred_ordered[i] = p_hat
    p_prev = p_hat

sub_b = df_test_ordered[["id"]].copy()
sub_b["pressure"] = pred_ordered
sub_b = sample_sub[["id"]].merge(sub_b, on="id", how="left", validate="one_to_one")
sub_b["pressure"] = sub_b["pressure"].astype(np.float32).fillna(global_median)
sub_b["pressure"] = sub_b["pressure"].apply(find_nearest)

b_path = "baseline_b.csv"
sub_b.to_csv(b_path, index=False)



## === cell 3
blend(a_path, b_path)

final_sub = pd.read_csv("blend.csv")
final_sub = final_sub[["id", "pressure"]]

final_sub = sample_sub[["id"]].merge(
    final_sub, on="id", how="left", validate="one_to_one"
)
final_sub.to_csv("submission.csv", index=False)
print(final_sub.head())
print("Wrote: submission.csv")
