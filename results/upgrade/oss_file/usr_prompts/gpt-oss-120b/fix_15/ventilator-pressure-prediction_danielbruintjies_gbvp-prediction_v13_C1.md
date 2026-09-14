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
import os, gc, warnings, time, concurrent.futures
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 300)




## === cell 1
def set_seed(seed: int = 42):
    np.random.seed(seed)
    import random, os

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(23)




## === cell 2
DATA_ROOT = "../input/ventilator-pressure-prediction"
train_path = os.path.join(DATA_ROOT, "train.csv")
test_path = os.path.join(DATA_ROOT, "test.csv")
sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")

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

train = pd.read_csv(train_path, dtype=dtype_train)
test = pd.read_csv(test_path, dtype=dtype_test)
submission = pd.read_csv(sub_path)

DEBUG = False
if DEBUG:
    train = train[:80_000]




## === cell 3
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fast feature engineering: compute cumulative area, cumsum of u_in,
    lag/backward lag features and simple diffs/cross terms.
    All operations are in‑place; the groupby object is cached once.
    """
    df["area"] = df["time_step"] * df["u_in"]
    df["area"] = df.groupby("breath_id", sort=False)["area"].cumsum()
    df["u_in_cumsum"] = df.groupby("breath_id", sort=False)["u_in"].cumsum()

    grp = df.groupby("breath_id", sort=False)

    for lag in range(1, 5):
        df[f"u_in_lag{lag}"] = grp["u_in"].shift(lag).fillna(0)
        df[f"u_out_lag{lag}"] = grp["u_out"].shift(lag).fillna(0)
        df[f"u_in_lag_back{lag}"] = grp["u_in"].shift(-lag).fillna(0)
        df[f"u_out_lag_back{lag}"] = grp["u_out"].shift(-lag).fillna(0)

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_out_diff1"] = df["u_out"] - df["u_out_lag1"]
    df["u_in_diff2"] = df["u_in"] - df["u_in_lag2"]
    df["u_out_diff2"] = df["u_out"] - df["u_out_lag2"]

    df["cross"] = df["u_in"] * df["u_out"]
    df["cross2"] = df["time_step"] * df["u_out"]

    return df


def add_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """
    One‑hot encode R and C exactly as original logic.
    """
    df["R_5"] = (df["R"] == 5).astype(np.uint8)
    df["R_20"] = (df["R"] == 20).astype(np.uint8)
    df["R_50"] = (df["R"] == 50).astype(np.uint8)
    df["C_10"] = (df["C"] == 10).astype(np.uint8)
    df["C_20"] = (df["C"] == 20).astype(np.uint8)
    df["C_50"] = (df["C"] == 50).astype(np.uint8)
    df = df.drop(["R", "C"], axis=1)
    return df


train_fe_df = add_features(train)
test_fe_df = add_features(test)

train_fe_df = add_one_hot(train_fe_df)
test_fe_df = add_one_hot(test_fe_df)

del train, test
gc.collect()




## === cell 4
target = train_fe_df["pressure"].values.astype(np.float32)

train_fe_df = train_fe_df.drop(["pressure", "id", "breath_id"], axis=1)
test_fe_df = test_fe_df.drop(["id", "breath_id"], axis=1)

scaler = RobustScaler()
train_fe = scaler.fit_transform(train_fe_df).astype(np.float32)
test_fe = scaler.transform(test_fe_df).astype(np.float32)

del train_fe_df, test_fe_df
gc.collect()




## === cell 5
NUM_FOLDS = 3
lgb_params = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.03,
    "num_leaves": 256,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": 23,
    "n_jobs": 1,  # each fold uses a single thread
    "min_data_in_leaf": 20,
}

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2021)
test_preds = np.zeros(test_fe.shape[0], dtype=np.float32)
oof_preds = np.zeros(train_fe.shape[0], dtype=np.float32)


def train_fold(fold, tr_idx, val_idx):
    X_tr, X_val = train_fe[tr_idx], train_fe[val_idx]
    y_tr, y_val = target[tr_idx], target[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=10000,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    oof_part = model.predict(X_val, num_iteration=model.best_iteration)
    test_part = model.predict(test_fe, num_iteration=model.best_iteration)

    del X_tr, X_val, y_tr, y_val, train_data, valid_data, model
    gc.collect()

    return fold, val_idx, oof_part, test_part


with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_FOLDS) as executor:
    futures = []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_fe)):
        futures.append(executor.submit(train_fold, fold, tr_idx, val_idx))

    for future in concurrent.futures.as_completed(futures):
        fold, val_idx, oof_part, test_part = future.result()
        oof_preds[val_idx] = oof_part
        test_preds += test_part / NUM_FOLDS




## === cell 6
PRESSURE_MIN = target.min()
PRESSURE_MAX = target.max()
all_pressure = np.sort(np.unique(target))
PRESSURE_STEP = all_pressure[1] - all_pressure[0] if len(all_pressure) > 1 else 0.01

submission["pressure"] = test_preds
submission["pressure"] = (
    np.round((submission["pressure"] - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP
    + PRESSURE_MIN
)
submission["pressure"] = np.clip(submission["pressure"], PRESSURE_MIN, PRESSURE_MAX)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
