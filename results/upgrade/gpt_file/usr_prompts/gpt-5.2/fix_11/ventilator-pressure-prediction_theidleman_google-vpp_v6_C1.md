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

3.9

# 3. Installed packages

catboost==1.2.8
geopandas==0.14.4
lightgbm==4.6.0
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold

import catboost as cat

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

print("imports done!")




## === cell 1
path_candidates = [
    "../input/ventilator-pressure-prediction/",
    "/kaggle/input/ventilator-pressure-prediction/",
    "/kaggle/data/ventilator-pressure-prediction/",
    "/kaggle/input/",
    "/kaggle/data/",
]
path = None
for p in path_candidates:
    if os.path.exists(os.path.join(p, "train.csv")):
        path = p
        break

if path is None:
    raise FileNotFoundError("Could not locate train.csv in known Kaggle input paths.")

data = {
    "train": os.path.join(path, "train.csv"),
    "test": os.path.join(path, "test.csv"),
    "sample": os.path.join(path, "sample_submission.csv"),
}

TRAIN_COLS = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
TEST_COLS = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
DTYPES_TRAIN_CAT = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
DTYPES_TEST = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}

train = pd.read_csv(data["train"], usecols=TRAIN_COLS, dtype=DTYPES_TRAIN_CAT)
test = pd.read_csv(data["test"], usecols=TEST_COLS, dtype=DTYPES_TEST)
sample = pd.read_csv(data["sample"], usecols=["id"], dtype={"id": "int32"})

print("loaded:", train.shape, test.shape, sample.shape)




## === cell 2
y = train["pressure"].to_numpy(dtype=np.float32, copy=False)

feature_names = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]

cat_feature_names = ["R", "C", "u_out"]
for c in cat_feature_names:
    train[c] = train[c].astype("category")
    test[c] = test[c].astype("category")

cat_feature_indices = [feature_names.index(c) for c in cat_feature_names]

X_train_df = train[feature_names]
X_test_df = test[feature_names]

groups = train["breath_id"].to_numpy(copy=False)
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X_train_df, y, groups=groups))

final_mae = np.empty(5, dtype=np.float64)
test_pred_sum = np.zeros(X_test_df.shape[0], dtype=np.float64)

try:
    import multiprocessing as mp

    cpu = mp.cpu_count()
except Exception:
    cpu = 4

thread_cap = int(min(8, max(1, cpu)))

params = {
    "loss_function": "MAE",
    "iterations": 600,
    "learning_rate": 0.08,
    "depth": 8,
    "random_seed": RANDOM_STATE,
    "task_type": "CPU",
    "thread_count": thread_cap,
    "verbose": False,
    "allow_writing_files": False,
}


def fast_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred), dtype=np.float64))


train_pool_full = cat.Pool(
    data=X_train_df,
    label=y,
    feature_names=feature_names,
    cat_features=cat_feature_indices,
)
test_pool = cat.Pool(
    data=X_test_df,
    feature_names=feature_names,
    cat_features=cat_feature_indices,
)

for fold, (trn_idx, val_idx) in enumerate(folds, 1):
    model = cat.CatBoostRegressor(**params)

    train_pool_fold = train_pool_full.slice(trn_idx)
    valid_pool_fold = train_pool_full.slice(val_idx)

    model.fit(
        train_pool_fold,
        eval_set=valid_pool_fold,
        use_best_model=False,
        verbose=False,
    )

    pred_valid = model.predict(valid_pool_fold)
    pred_test = model.predict(test_pool)

    test_pred_sum += pred_test

    mae = fast_mae(y[val_idx], pred_valid.astype(np.float32, copy=False))
    print(f"fold: {fold}, mae: {mae:.6f}")
    final_mae[fold - 1] = mae

print(f"final mae (cv): {final_mae.mean():.6f}")




## === cell 3
final_test_score = (test_pred_sum / 5.0).astype(np.float32, copy=False)

if len(sample) != len(final_test_score):
    raise ValueError("Sample submission length does not match test predictions length.")

output = sample.copy()
output["pressure"] = final_test_score

if output["pressure"].isna().any():
    raise ValueError("Submission contains NaN predictions.")

output.to_csv("./submission.csv", index=False)
print("wrote ./submission.csv with shape:", output.shape)




## === cell 4
output.head()
