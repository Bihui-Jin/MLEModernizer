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

geopandas==0.14.4
lightgbm==4.6.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
import lightgbm as lgb

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(0)

N_THREADS = int(os.environ.get("LGBM_THREADS", "8"))



## === cell 1
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"

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

train = pd.read_csv(TRAIN_PATH, dtype=train_dtypes)
test = pd.read_csv(TEST_PATH, dtype=test_dtypes)

for df in (train, test):
    df["R"] = df["R"].astype("category")
    df["C"] = df["C"].astype("category")



## === cell 2
_ = None



## === cell 3
seed0 = 2021
params0 = {
    "objective": "mae",
    "boosting_type": "gbdt",
    "max_depth": -1,
    "max_bin": 100,
    "min_data_in_leaf": 500,
    "learning_rate": 0.05,
    "subsample": 0.72,
    "subsample_freq": 4,
    "feature_fraction": 0.5,
    "lambda_l1": 0.5,
    "lambda_l2": 1.0,
    "categorical_column": [0],
    "seed": seed0,
    "feature_fraction_seed": seed0,
    "bagging_seed": seed0,
    "drop_seed": seed0,
    "data_random_seed": seed0,
    "n_jobs": -1,
    "verbose": -1,
}

seed1 = 42
params1 = {
    "objective": "mae",
    "metric": "mae",
    "learning_rate": 0.1,
    "lambda_l1": 2,
    "lambda_l2": 7,
    "num_leaves": 800,
    "min_sum_hessian_in_leaf": 20,
    "feature_fraction": 0.8,
    "feature_fraction_bynode": 0.8,
    "bagging_fraction": 0.9,
    "bagging_freq": 42,
    "min_data_in_leaf": 700,
    "max_depth": 4,
    "categorical_column": [0],
    "seed": seed1,
    "feature_fraction_seed": seed1,
    "bagging_seed": seed1,
    "drop_seed": seed1,
    "data_random_seed": seed1,
    "boosting": "gbdt",
    "verbosity": -1,
    "n_jobs": -1,
}


def _with_runtime_params(params: dict) -> dict:
    p = dict(params)
    p["num_threads"] = N_THREADS
    p["n_jobs"] = N_THREADS
    p.setdefault("force_col_wise", True)
    p.setdefault("deterministic", True)
    p.setdefault("verbosity", -1)
    p.setdefault("verbose", -1)
    return p


def train_and_evaluate_lgb(
    train: pd.DataFrame, test: pd.DataFrame, params: dict
) -> np.ndarray:
    features = [col for col in train.columns if col not in {"id", "pressure"}]

    X = np.ascontiguousarray(train[features].to_numpy(dtype=np.float32, copy=False))
    y = np.ascontiguousarray(train["pressure"].to_numpy(dtype=np.float32, copy=False))
    X_test = np.ascontiguousarray(test[features].to_numpy(dtype=np.float32, copy=False))

    oof_predictions = np.zeros(train.shape[0], dtype=np.float64)
    test_predictions = np.zeros(test.shape[0], dtype=np.float64)

    gkf = GroupKFold(n_splits=5)
    groups = train["breath_id"].to_numpy(copy=False)

    run_params = _with_runtime_params(params)

    feature_name = features

    cat_feature = ["R", "C"]

    dtrain_full = lgb.Dataset(
        X,
        label=y,
        feature_name=feature_name,
        categorical_feature=cat_feature,
        free_raw_data=False,
        params=run_params,
    )
    dtrain_full.construct()

    num_boost_round = 1800
    stopping_rounds = 15
    best_iterations = []

    for fold, (trn_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        print(f"Training fold {fold}/5")

        dtrain = dtrain_full.subset(trn_idx)
        dvalid = dtrain_full.subset(val_idx)

        booster = lgb.train(
            params=run_params,
            train_set=dtrain,
            valid_sets=[dvalid],
            valid_names=["valid"],
            num_boost_round=num_boost_round,
            callbacks=[
                lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=True),
                lgb.log_evaluation(period=250),
            ],
        )

        best_iter = booster.best_iteration or num_boost_round
        best_iterations.append(best_iter)
        print(f"Fold {fold} best_iteration: {best_iter}")

        oof_predictions[val_idx] = booster.predict(X[val_idx], num_iteration=best_iter)
        test_predictions += booster.predict(X_test, num_iteration=best_iter) / 5.0

        del booster, dtrain, dvalid

    best_iteration_report = int(np.round(np.mean(best_iterations)))
    print("Best iteration summary from folds (mean):", best_iteration_report)

    mae_oof = np.mean(np.abs(y.astype(np.float64) - oof_predictions))
    print(f"OOF MAE (unofficial, includes expiratory rows): {mae_oof}")

    return test_predictions


predictions_lgb = train_and_evaluate_lgb(train, test, params1)
test["pressure"] = predictions_lgb

test[["id", "pressure"]].to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", test[["id", "pressure"]].shape)
print(test[["id", "pressure"]].head())



## === cell 4
print("Columns in test:", list(test.columns))
print("pressure column exists:", "pressure" in test.columns)
print(pd.read_csv("submission.csv").head())
