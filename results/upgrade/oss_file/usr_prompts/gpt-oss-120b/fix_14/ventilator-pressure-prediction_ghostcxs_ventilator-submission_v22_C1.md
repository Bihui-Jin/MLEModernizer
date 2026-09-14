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

No external packages required in the script and installed.

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
import warnings
import pandas as pd
import numpy as np
import gc

warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor




## === cell 1
dtype_train = {
    "R": "int8",
    "C": "int8",
    "breath_id": "int32",
    "id": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
dtype_test = {
    "R": "int8",
    "C": "int8",
    "breath_id": "int32",
    "id": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}

train_ori = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv", dtype=dtype_train
)
y_train = train_ori["pressure"].values.astype("float32")  # target
train_ori = train_ori.drop(columns=["pressure"])

test_ori = pd.read_csv(
    "../input/ventilator-pressure-prediction/test.csv", dtype=dtype_test
)


def add_features(df):
    df["area"] = (df["time_step"] * df["u_in"]).astype("float32")
    grp = df.groupby("breath_id", sort=False)
    df["area"] = grp["area"].cumsum()
    df["u_in_cumsum"] = grp["u_in"].cumsum()
    df["time_cumsum"] = grp["time_step"].cumsum()
    df["u_out_cumsum"] = grp["u_out"].cumsum()

    for lag in range(1, 5):
        df[f"u_in_lag{lag}"] = grp["u_in"].shift(lag)
        df[f"u_in_lag_back{lag}"] = grp["u_in"].shift(-lag)

    agg_u = grp["u_in"].agg(
        breath_len="size",
        u_in_mean="mean",
        u_in_std="std",
        u_in_max="max",
    )
    agg_t = grp["time_step"].agg(
        time_mean="mean",
        time_std="std",
    )
    agg_u["u_in_std"] = agg_u["u_in_std"].fillna(0)
    agg_t["time_std"] = agg_t["time_std"].fillna(0)

    df["breath_len"] = df["breath_id"].map(agg_u["breath_len"])
    df["u_in_mean"] = df["breath_id"].map(agg_u["u_in_mean"])
    df["u_in_std"] = df["breath_id"].map(agg_u["u_in_std"])
    df["breath_id__u_in__max"] = df["breath_id"].map(agg_u["u_in_max"])
    df["time_mean"] = df["breath_id"].map(agg_t["time_mean"])
    df["time_std"] = df["breath_id"].map(agg_t["time_std"])

    for lag in range(1, 5):
        df[f"u_in_diff{lag}"] = df["u_in"] - df[f"u_in_lag{lag}"]

    df["cross"] = df["u_in"] * df["u_out"]
    df["cross2"] = df["time_step"] * df["u_out"]
    df["area_u_out"] = df["area"] * df["u_out"]

    df["R_num"] = df["R"].astype("float32")
    df["C_num"] = df["C"].astype("float32")

    df.fillna(0, inplace=True)
    return df


train_feat = add_features(train_ori)
test_feat = add_features(test_ori)

drop_cols = ["id", "breath_id"]
train_feat.drop(columns=drop_cols, errors="ignore", inplace=True)
test_feat.drop(columns=drop_cols, errors="ignore", inplace=True)

train_len = len(train_feat)
all_feat = pd.concat([train_feat, test_feat], axis=0, ignore_index=True)

for r_val in sorted(all_feat["R"].unique()):
    all_feat[f"R_{r_val}"] = (all_feat["R"] == r_val).astype(np.uint8)
for c_val in sorted(all_feat["C"].unique()):
    all_feat[f"C_{c_val}"] = (all_feat["C"] == c_val).astype(np.uint8)

for r_val in sorted(all_feat["R"].unique()):
    for c_val in sorted(all_feat["C"].unique()):
        col_name = f"R__C_{r_val}_{c_val}"
        all_feat[col_name] = (
            (all_feat["R"] == r_val) & (all_feat["C"] == c_val)
        ).astype(np.uint8)

all_feat.drop(columns=["R", "C"], inplace=True)

train_feat = all_feat.iloc[:train_len].reset_index(drop=True)
test_feat = all_feat.iloc[train_len:].reset_index(drop=True)

X_train = train_feat.to_numpy(dtype="float32")
X_test = test_feat.to_numpy(dtype="float32")

median = np.median(X_train, axis=0)
q75 = np.percentile(X_train, 75, axis=0)
q25 = np.percentile(X_train, 25, axis=0)
iqr = q75 - q25
scale = np.where(iqr == 0, 1.0, iqr)

X_train = (X_train - median) / scale
X_test = (X_test - median) / scale

del train_feat, test_feat, train_ori, test_ori, all_feat
gc.collect()




## === cell 2
model = HistGradientBoostingRegressor(
    max_depth=10,
    learning_rate=0.01,
    max_iter=2500,
    loss="absolute_error",
    random_state=42,
    validation_fraction=0.1,
    n_iter_no_change=30,
)

model.fit(X_train, y_train)

preds = model.predict(X_test).astype("float32")

del X_train, X_test, y_train
gc.collect()




## === cell 3
pressure_step = 0.07030248641967773
p_min = -1.7551400036622216
p_max = 64.82099173863328

preds = np.clip(preds, p_min, p_max)

submission = pd.DataFrame(
    {
        "id": pd.read_csv(
            "../input/ventilator-pressure-prediction/test.csv", usecols=["id"]
        )["id"].values,
        "pressure": preds,
    }
)

submission.to_csv("./submission.csv", index=False)
print("Submission written to ./submission.csv")
