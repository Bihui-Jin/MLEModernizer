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
import os
import random
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV I/O
from sklearn.ensemble import HistGradientBoostingRegressor


def set_seed(seed=2021):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def find_nearest(prediction):
    """Scalar nearest‑pressure lookup (kept for blending step)."""
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


def nearest_array(preds):
    """Vectorized nearest‑pressure mapping for large prediction arrays."""
    preds = preds.astype(np.float32)
    idx = np.searchsorted(sorted_pressures, preds, side="left")
    lower_idx = np.clip(idx - 1, 0, total_pressures_len - 1)
    upper_idx = np.clip(idx, 0, total_pressures_len - 1)

    lower = sorted_pressures[lower_idx]
    upper = sorted_pressures[upper_idx]
    choose_lower = np.abs(lower - preds) < np.abs(upper - preds)
    return np.where(choose_lower, lower, upper)


def add_poly_features(X):
    """
    Efficient polynomial feature expansion with cached interaction indices:
    - original features
    - squared features
    - pairwise interaction terms (i < j)
    """
    X = X.astype(np.float32)
    X_sq = X**2  # (n_samples, n_features)

    if not hasattr(add_poly_features, "iu"):
        n_feat = X.shape[1]
        add_poly_features.iu = np.triu_indices(n_feat, k=1)  # (row_idx, col_idx)

    iu = add_poly_features.iu
    interactions = X[:, iu[0]] * X[:, iu[1]]  # (n_samples, n_interactions)
    return np.hstack([X, X_sq, interactions])




## === cell 1
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"

dtypes = {
    "R": "int8",
    "C": "int8",
    "breath_id": "int32",
    "id": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
df_train = pd.read_csv(train_path, dtype=dtypes)
df_test = pd.read_csv(test_path, dtype=dtypes)

group = df_train.groupby("breath_id")
df_train["cum_u_in"] = group["u_in"].transform("cumsum")
df_train["cum_u_out"] = group["u_out"].transform("cumsum")
df_train["u_in_diff"] = group["u_in"].transform("diff").fillna(0)
df_train["u_out_diff"] = group["u_out"].transform("diff").fillna(0)
df_train["u_in_roll3"] = group["u_in"].transform(
    lambda s: s.rolling(3, min_periods=1).mean()
)
df_train["u_out_roll3"] = group["u_out"].transform(
    lambda s: s.rolling(3, min_periods=1).mean()
)

group_test = df_test.groupby("breath_id")
df_test["cum_u_in"] = group_test["u_in"].transform("cumsum")
df_test["cum_u_out"] = group_test["u_out"].transform("cumsum")
df_test["u_in_diff"] = group_test["u_in"].transform("diff").fillna(0)
df_test["u_out_diff"] = group_test["u_out"].transform("diff").fillna(0)
df_test["u_in_roll3"] = group_test["u_in"].transform(
    lambda s: s.rolling(3, min_periods=1).mean()
)
df_test["u_out_roll3"] = group_test["u_out"].transform(
    lambda s: s.rolling(3, min_periods=1).mean()
)

df_train["R_C"] = df_train["R"] * df_train["C"]
df_test["R_C"] = df_test["R"] * df_test["C"]

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures).astype(np.float32)
total_pressures_len = len(sorted_pressures)

set_seed(42)

base_features = [
    "u_in",
    "u_out",
    "R",
    "C",
    "time_step",
    "cum_u_in",
    "cum_u_out",
    "u_in_diff",
    "u_out_diff",
    "R_C",
    "u_in_roll3",
    "u_out_roll3",
]
X_base = df_train[base_features].values.astype(np.float32)
y = df_train["pressure"].values.astype(np.float32)

X = add_poly_features(X_base)

val_mask = np.random.rand(len(df_train)) < 0.2
X_train, y_train = X[~val_mask], y[~val_mask]
X_val, y_val = X[val_mask], y[val_mask]

model = HistGradientBoostingRegressor(
    max_depth=12,
    learning_rate=0.03,
    max_iter=1500,
    random_state=42,
)
model.fit(X_train, y_train)

val_pred = model.predict(X_val)
val_pred_nearest = nearest_array(val_pred)
mae = np.mean(np.abs(val_pred_nearest - y_val))
print(f"Validation MAE (HGBR + nearest mapping): {mae:.6f}")

X_test_base = df_test[base_features].values.astype(np.float32)
X_test = add_poly_features(X_test_base)
test_pred = model.predict(X_test)
test_pred_nearest = nearest_array(test_pred)

submission = pd.DataFrame({"id": df_test["id"], "pressure": test_pred_nearest})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")




## === cell 2
def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.5 + b.pressure * 0.5
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a
