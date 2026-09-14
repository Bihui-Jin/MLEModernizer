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
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
xgboost==2.0.3

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
import pickle
import numpy as np
import pandas as pd

import lightgbm as lgb  # kept to preserve original imports (not used)
from xgboost import XGBRegressor, DMatrix
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import seaborn as sns

CACHE_DIR = "./__cache__"
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)




## === cell 1
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SAMPLE_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"

train_dtypes = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
test_dtypes = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}

t0 = time.time()
train = pd.read_csv(TRAIN_PATH, dtype=train_dtypes)
test = pd.read_csv(TEST_PATH, dtype=test_dtypes)
sample = pd.read_csv(SAMPLE_PATH, dtype={"id": "int32", "pressure": "float32"})
print(f"Loaded CSVs in {time.time()-t0:.1f}s")
print(train.shape, test.shape, sample.shape)
print(train.columns)




## === cell 2
feature_cols = [c for c in train.columns if c not in ["pressure", "id", "breath_id"]]

t0 = time.time()
X_train = train[feature_cols].to_numpy(dtype=np.float32, copy=False)
y_train = train["pressure"].to_numpy(dtype=np.float32, copy=False)
X_test = test[feature_cols].to_numpy(dtype=np.float32, copy=False)
print("Features:", feature_cols)
print(
    "X_train:",
    X_train.shape,
    "X_test:",
    X_test.shape,
    f"(prepared in {time.time()-t0:.2f}s)",
)




## === cell 3
rf_cache_path = os.path.join(CACHE_DIR, "rf_model.pkl")

t0 = time.time()
if os.path.exists(rf_cache_path):
    with open(rf_cache_path, "rb") as f:
        rf = pickle.load(f)
    print(f"RF loaded from cache in {time.time() - t0:.1f}s")
else:
    rf = RandomForestRegressor(
        n_estimators=120,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
    )
    rf.fit(X_train, y_train)
    with open(rf_cache_path, "wb") as f:
        pickle.dump(rf, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"RF trained and cached in {time.time() - t0:.1f}s")




## === cell 4
rf_pred_cache_path = os.path.join(CACHE_DIR, "rf_test_preds.npy")

t0 = time.time()
if os.path.exists(rf_pred_cache_path):
    rf_preds = np.load(rf_pred_cache_path).astype(np.float32, copy=False)
    print(f"RF preds loaded from cache in {time.time() - t0:.1f}s")
else:
    rf_preds = rf.predict(X_test).astype(np.float32, copy=False)
    np.save(rf_pred_cache_path, rf_preds)
    print(f"RF preds computed and cached in {time.time() - t0:.1f}s")

rf_sub = pd.DataFrame({"id": sample["id"].values, "pressure": rf_preds})
rf_sub.to_csv("./rf_submission.csv", index=False)
print("Wrote ./rf_submission.csv", rf_sub.shape, rf_sub.head())




## === cell 5
xgb_cache_path = os.path.join(CACHE_DIR, "xgb_model.pkl")

dtrain_cache_path = os.path.join(CACHE_DIR, "dtrain.buffer")
t0 = time.time()
if os.path.exists(dtrain_cache_path):
    dtrain = DMatrix(dtrain_cache_path)
else:
    dtrain = DMatrix(X_train, label=y_train)
    dtrain.save_binary(dtrain_cache_path)
print(f"DMatrix train ready in {time.time() - t0:.1f}s")

t0 = time.time()
if os.path.exists(xgb_cache_path):
    with open(xgb_cache_path, "rb") as f:
        xgb = pickle.load(f)
    print(f"XGB loaded from cache in {time.time() - t0:.1f}s")
else:
    xgb = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    with open(xgb_cache_path, "wb") as f:
        pickle.dump(xgb, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"XGB trained and cached in {time.time() - t0:.1f}s")




## === cell 6
xgb_pred_cache_path = os.path.join(CACHE_DIR, "xgb_test_preds.npy")

dtest_cache_path = os.path.join(CACHE_DIR, "dtest.buffer")
t0 = time.time()
if os.path.exists(xgb_pred_cache_path):
    xgb_preds = np.load(xgb_pred_cache_path).astype(np.float32, copy=False)
    print(f"XGB preds loaded from cache in {time.time() - t0:.1f}s")
else:
    if os.path.exists(dtest_cache_path):
        dtest = DMatrix(dtest_cache_path)
    else:
        dtest = DMatrix(X_test)
        dtest.save_binary(dtest_cache_path)
    xgb_preds = xgb.get_booster().predict(dtest).astype(np.float32, copy=False)
    np.save(xgb_pred_cache_path, xgb_preds)
    print(f"XGB preds computed and cached in {time.time() - t0:.1f}s")

xgb_sub = pd.DataFrame({"id": sample["id"].values, "pressure": xgb_preds})
xgb_sub.to_csv("./xgb_submission.csv", index=False)
print("Wrote ./xgb_submission.csv", xgb_sub.shape, xgb_sub.head())

assert list(xgb_sub.columns) == ["id", "pressure"]
assert len(xgb_sub) == len(test)
assert xgb_sub["pressure"].isna().sum() == 0
print("Submission looks valid.")
