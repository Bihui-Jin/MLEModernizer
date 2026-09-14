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
import os
import time
import numpy as np
import pandas as pd
import random
import gc
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 300)
gc.enable()


def set_seed(seed_val: int):
    """Set deterministic seeds for reproducibility."""
    random.seed(seed_val)
    np.random.seed(seed_val)
    os.environ["PYTHONHASHSEED"] = str(seed_val)


start_time = time.time()




## === cell 1
def connect_to_tpu(tpu_address: str = None):
    print("TPU connection not required; using CPU only.")
    return None, None


cluster_resolver, strategy = connect_to_tpu()




## === cell 2
DEBUG = False
TRAIN_MODEL = True




## === cell 3
dtype_map = {
    "R": "int8",
    "C": "int8",
    "breath_id": "int32",
    "id": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
DATA_ROOT = "../input/ventilator-pressure-prediction"
train = pd.read_csv(f"{DATA_ROOT}/train.csv", dtype=dtype_map)
test = pd.read_csv(f"{DATA_ROOT}/test.csv", dtype=dtype_map)
submission = pd.read_csv(f"{DATA_ROOT}/sample_submission.csv")

if DEBUG:
    train = train[:80_000]
    test = test[:8_000]
    submission = submission[:8_000]




## === cell 4
print(f"TRAIN rows: {len(train)}, TEST rows: {len(test)}")
print(
    f"Unique breaths – train: {train['breath_id'].nunique()}, test: {test['breath_id'].nunique()}"
)




## === cell 5
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    grp = df.groupby("breath_id")

    df["area"] = (df["time_step"] * df["u_in"]).groupby(df["breath_id"]).cumsum()
    df["u_in_cumsum"] = grp["u_in"].cumsum()

    for lag in [1, 2, 3, 8]:
        df[f"u_in_lag{lag}"] = grp["u_in"].shift(lag)
        df[f"u_out_lag{lag}"] = grp["u_out"].shift(lag)
        df[f"u_in_lead{lag}"] = grp["u_in"].shift(-lag)
        df[f"u_out_lead{lag}"] = grp["u_out"].shift(-lag)

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_out_diff1"] = df["u_out"] - df["u_out_lag1"]

    df["u_in_roll_mean8"] = (
        grp["u_in"]
        .rolling(window=8, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["u_out_roll_mean8"] = (
        grp["u_out"]
        .rolling(window=8, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["mean_RC_u_in"] = df.groupby(["R", "C"])["u_in"].transform("mean")
    df["max_RC_u_in"] = df.groupby(["R", "C"])["u_in"].transform("max")

    df.fillna(0, inplace=True)

    df["R"] = df["R"].astype("category")
    df["C"] = df["C"].astype("category")

    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype(np.float32)

    return df


train_fe = add_features(train)
test_fe = add_features(test)




## === cell 6
TARGET_COL = "pressure"

y = train_fe[TARGET_COL].values.astype(np.float32)

X = train_fe.drop(columns=[TARGET_COL, "id", "breath_id"])
X_test = test_fe.drop(columns=["id", "breath_id"])




## === cell 7
set_seed(23)

from sklearn.model_selection import KFold
import gc

try:
    import lightgbm as lgb

    LGB_AVAILABLE = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingRegressor

    LGB_AVAILABLE = False

NUM_FOLDS = 3
if DEBUG:
    NUM_FOLDS = 2

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2021)

max_threads = max(1, os.cpu_count() or 1)

fold_indices = list(kf.split(X))


def train_one_fold(fold, train_idx, val_idx):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    if LGB_AVAILABLE:
        cat_cols = ["R", "C"]
        lgb_train = lgb.Dataset(
            X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False
        )
        lgb_val = lgb.Dataset(
            X_val,
            label=y_val,
            reference=lgb_train,
            categorical_feature=cat_cols,
            free_raw_data=False,
        )

        params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.01,
            "verbosity": -1,
            "seed": 23,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "num_threads": max_threads,
        }

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=5000,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )
        fold_pred = model.predict(X_test, num_iteration=model.best_iteration)
    else:
        model = HistGradientBoostingRegressor(
            max_iter=1500,
            learning_rate=0.05,
            random_state=23,
        )
        model.fit(X_tr, y_tr)
        fold_pred = model.predict(X_test)

    del X_tr, X_val, y_tr, y_val, model
    gc.collect()
    return fold_pred


fold_preds = []
for fold, (train_idx, val_idx) in enumerate(fold_indices):
    fold_pred = train_one_fold(fold, train_idx, val_idx)
    fold_preds.append(fold_pred)

test_preds = np.mean(np.column_stack(fold_preds), axis=1)




## === cell 8
PRESSURE_MIN = train["pressure"].min()
PRESSURE_MAX = train["pressure"].max()

submission["pressure"] = np.clip(test_preds, PRESSURE_MIN, PRESSURE_MAX)

submission.to_csv("submission.csv", index=False)
print("Submission file written to 'submission.csv'.")
