# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.1723273895165379

# 6. Current score

4.21666

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.66932) has done: 'The script now converts the feature DataFrames to NumPy arrays before fitting (removing pandas overhead), frees the large DataFrames early, and reduces the GradientBoostingRegressor to 50 trees, which keeps the same model type while cutting training time enough to stay under the 600 s limit. All other logic, including rounding predictions to the nearest training pressure and creating the submission file, remains unchanged.'
- What this solution (achieved 4.68372) has done: 'We fix the `HistGradientBoostingRegressor` initialization by removing the unsupported `subsample` argument, and then round the model’s predictions to the nearest pressure value seen in the training set using the already‑provided `round_to_nearest` function. This resolves the runtime error and aligns predictions with realistic pressure levels, which should improve the MAE and bring the score toward the target while keeping the original modeling approach unchanged.'
- What this solution (achieved 4.23867) has done: 'I increase the model capacity slightly (more trees, deeper leaves and a small L2 regularization) and clip predictions to the range observed in the training data before optional rounding. These tweaks keep the same HistGradientBoostingRegressor‑based pipeline while expectedly lowering the MAE toward the target.'
- What this solution (achieved 4.21666) has done: 'I switch the HistGradientBoostingRegressor to use the `absolute_error` loss (which directly optimises MAE) and give it a few more boosting iterations with a slightly smaller learning rate. This stays within the same model family while aiming to pull the validation MAE much closer to the target. The rest of the pipeline, including feature engineering, clipping, rounding and CSV creation, is left unchanged.'

# 9. Code solution

## === cell 0
import os
import gc
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def set_seed(seed: int = 2021):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def round_to_nearest(preds: np.ndarray, sorted_vals: np.ndarray) -> np.ndarray:
    """
    Vectorized version of find_nearest.
    Maps each float prediction to the closest value present in sorted_vals.
    """
    total_len = sorted_vals.shape[0]
    idx = np.searchsorted(sorted_vals, preds, side="left")
    nearest = np.empty_like(preds)

    mask0 = idx == 0
    maskN = idx == total_len
    nearest[mask0] = sorted_vals[0]
    nearest[maskN] = sorted_vals[-1]

    mask_mid = (~mask0) & (~maskN)
    idx_mid = idx[mask_mid]
    lower = sorted_vals[idx_mid - 1]
    upper = sorted_vals[idx_mid]
    lower_diff = np.abs(preds[mask_mid] - lower)
    upper_diff = np.abs(upper - preds[mask_mid])
    choose_lower = lower_diff <= upper_diff
    nearest[mask_mid] = np.where(choose_lower, lower, upper)

    return nearest




## === cell 1
train_path = os.path.join("..", "input", "ventilator-pressure-prediction", "train.csv")
test_path = os.path.join("..", "input", "ventilator-pressure-prediction", "test.csv")
sample_sub_path = os.path.join(
    "..", "input", "ventilator-pressure-prediction", "sample_submission.csv"
)

feature_cols = ["R", "C", "time_step", "u_in", "u_out"]
usecols_train = feature_cols + ["pressure"]
usecols_test = feature_cols

df_train = pd.read_csv(
    train_path,
    usecols=usecols_train,
    dtype={
        "R": np.int16,
        "C": np.int16,
        "time_step": np.float32,
        "u_in": np.float32,
        "u_out": np.int8,
        "pressure": np.float32,
    },
)
df_test = pd.read_csv(
    test_path,
    usecols=usecols_test,
    dtype={
        "R": np.int16,
        "C": np.int16,
        "time_step": np.float32,
        "u_in": np.float32,
        "u_out": np.int8,
    },
)

y = df_train["pressure"].values.astype(np.float32)
X = df_train[feature_cols].values.astype(np.float32)
X_test = df_test[feature_cols].values.astype(np.float32)

feat1 = X[:, 0] * X[:, 4]  # R * u_out
feat2 = X[:, 0] * X[:, 3]  # R * u_in
feat3 = X[:, 1] * X[:, 3]  # C * u_in
feat4 = X[:, 2] * X[:, 3]  # time_step * u_in
X = np.column_stack([X, feat1, feat2, feat3, feat4])

feat1_test = X_test[:, 0] * X_test[:, 4]
feat2_test = X_test[:, 0] * X_test[:, 3]
feat3_test = X_test[:, 1] * X_test[:, 3]
feat4_test = X_test[:, 2] * X_test[:, 3]
X_test = np.column_stack([X_test, feat1_test, feat2_test, feat3_test, feat4_test])

unique_pressures = np.sort(df_train["pressure"].unique()).astype(np.float32)

del df_train, df_test
gc.collect()




## === cell 2
set_seed(2021)

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.1, random_state=2021, shuffle=True
)

gbr = HistGradientBoostingRegressor(
    loss="absolute_error",  # directly optimises MAE
    max_iter=800,  # more boosting iterations for higher capacity
    learning_rate=0.02,  # smaller step to keep training stable
    max_depth=5,
    max_bins=255,
    l2_regularization=0.1,
    random_state=2021,
)

gbr.fit(X_tr, y_tr)

min_pressure, max_pressure = y_tr.min(), y_tr.max()
val_pred = gbr.predict(X_val)
val_pred = np.clip(val_pred, min_pressure, max_pressure)
val_mae = mean_absolute_error(y_val, val_pred)
print(f"Validation MAE (clipped predictions): {val_mae:.6f}")

test_pred = gbr.predict(X_test)
test_pred = np.clip(test_pred, min_pressure, max_pressure)

test_pred = round_to_nearest(test_pred, unique_pressures)

submission = pd.read_csv(sample_sub_path)  # ensures correct column order and IDs
submission["pressure"] = test_pred
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

del gbr, X, X_test, y, X_tr, X_val, y_tr, y_val
gc.collect()
