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
import os
import pandas as pd
import numpy as np



## === cell 1
sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")
train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")




## === cell 2
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["breath_id", "time_step"]).copy()

    df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1).fillna(0.0)
    df["u_out_lag1"] = df.groupby("breath_id")["u_out"].shift(1).fillna(0.0)
    df["u_in_cum"] = df.groupby("breath_id")["u_in"].cumsum()
    df["u_in_diff1"] = (df["u_in"] - df["u_in_lag1"]).astype("float32")

    df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2).fillna(0.0)
    df["u_in_diff2"] = (df["u_in_lag1"] - df["u_in_lag2"]).astype("float32")
    df["u_out_cum"] = df.groupby("breath_id")["u_out"].cumsum().astype("float32")

    df["u_in_x_R"] = (df["u_in"] * df["R"]).astype("float32")
    df["u_in_x_C"] = (df["u_in"] * df["C"]).astype("float32")

    df["u_in_lead1"] = (
        df.groupby("breath_id")["u_in"].shift(-1).fillna(0.0).astype("float32")
    )
    df["u_out_lead1"] = (
        df.groupby("breath_id")["u_out"].shift(-1).fillna(0.0).astype("float32")
    )
    df["u_in_lead2"] = (
        df.groupby("breath_id")["u_in"].shift(-2).fillna(0.0).astype("float32")
    )

    return df


train = add_features(train)
test = add_features(test)



## === cell 3
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

base_features = [
    "R",
    "C",
    "time_step",
    "u_in",
    "u_out",
    "u_in_lag1",
    "u_out_lag1",
    "u_in_cum",
    "u_in_diff1",
    "u_in_lag2",
    "u_in_diff2",
    "u_out_cum",
    "u_in_x_R",
    "u_in_x_C",
    "u_in_lead1",
    "u_out_lead1",
    "u_in_lead2",
]

train_insp = train[train["u_out"] == 0].copy()
train_insp = train_insp.sort_values(["breath_id", "time_step"]).copy()

train_insp["pressure_lag1_true"] = (
    train_insp.groupby("breath_id")["pressure"].shift(1).fillna(0.0).astype("float32")
)

features = base_features + ["pressure_lag1_true"]

X_train_tf = train_insp[features]
y_train = train_insp["pressure"]



## === cell 4
models = {}
global_model = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", Ridge(alpha=3.0, random_state=42)),
    ]
)
global_model.fit(X_train_tf, y_train)

for (r, c), grp in train_insp.groupby(["R", "C"], sort=False):
    Xg = grp[features]
    yg = grp["pressure"]
    m = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=3.0, random_state=42)),
        ]
    )
    m.fit(Xg, yg)
    models[(int(r), int(c))] = m

student_models = {}
global_student_model = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", Ridge(alpha=3.0, random_state=42)),
    ]
)

train_insp_student = train_insp.copy()
train_insp_student["pressure_lag1_pred"] = 0.0

for breath_id, idx in train_insp_student.groupby(
    "breath_id", sort=False
).groups.items():
    b = train_insp_student.loc[idx].sort_values("time_step").copy()
    prev_p = 0.0
    lag_pred = np.empty(len(b), dtype=np.float64)
    for i, (_, row) in enumerate(b.iterrows()):
        lag_pred[i] = prev_p
        x = row[base_features].to_frame().T
        x["pressure_lag1_true"] = np.float32(prev_p)
        prev_p = float(global_model.predict(x)[0])
    train_insp_student.loc[b.index, "pressure_lag1_pred"] = lag_pred

student_features = base_features + ["pressure_lag1_pred"]
X_train_student_global = train_insp_student[student_features]
global_student_model.fit(X_train_student_global, y_train)

for (r, c), grp in train_insp_student.groupby(["R", "C"], sort=False):
    Xg = grp[student_features]
    yg = grp["pressure"]
    m = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", Ridge(alpha=3.0, random_state=42)),
        ]
    )
    m.fit(Xg, yg)
    student_models[(int(r), int(c))] = m



## === cell 5
test_sorted = test.sort_values(["breath_id", "time_step"]).copy()
test_sorted["pred"] = 0.0

insp_mean_pressure = float(train_insp["pressure"].mean())

for breath_id, bidx in test_sorted.groupby("breath_id", sort=False).groups.items():
    b = test_sorted.loc[bidx].sort_values("time_step").copy()

    r = int(b["R"].iloc[0])
    c = int(b["C"].iloc[0])
    m = student_models.get((r, c), global_student_model)

    prev_pred = 0.0
    preds = np.empty(len(b), dtype=np.float64)

    for i, (_, row) in enumerate(b.iterrows()):
        x = row[base_features].to_frame().T
        x["pressure_lag1_pred"] = np.float32(prev_pred)
        p = float(m.predict(x)[0])
        preds[i] = p
        if int(row["u_out"]) == 0:
            prev_pred = p  # only advance state during inspiratory phase

    test_sorted.loc[b.index, "pred"] = preds

tmp = test_sorted[["id", "breath_id", "time_step", "u_out", "pred"]].copy()

insp_mask = tmp["u_out"].values == 0
last_insp_pred_by_breath = tmp.loc[insp_mask].groupby("breath_id")["pred"].last()

mapped_last_insp = (
    tmp["breath_id"]
    .map(last_insp_pred_by_breath)
    .fillna(insp_mean_pressure)
    .astype(np.float64)
)

exp_mask = tmp["u_out"].values == 1
tmp.loc[exp_mask, "pred"] = mapped_last_insp.loc[exp_mask].values

test_pred = tmp["pred"].values.astype(np.float64)

pressure_levels = np.sort(train["pressure"].unique().astype(np.float64))
idx = np.searchsorted(pressure_levels, test_pred, side="left")
idx0 = np.clip(idx - 1, 0, len(pressure_levels) - 1)
idx1 = np.clip(idx, 0, len(pressure_levels) - 1)

p0 = pressure_levels[idx0]
p1 = pressure_levels[idx1]
choose_right = np.abs(p1 - test_pred) < np.abs(test_pred - p0)
snapped = np.where(choose_right, p1, p0)

submission = pd.DataFrame({"id": tmp["id"].values, "pressure": snapped})
submission = submission.merge(
    test[["id"]], on="id", how="right", validate="1:1"
).sort_values("id")

assert len(submission) == len(
    test
), "Row count mismatch: submission and test must match 1:1"
assert submission[
    "id"
].is_monotonic_increasing, "Submission ids must be in ascending order"

submission.to_csv("submission.csv", index=False)

submission.head(5)
