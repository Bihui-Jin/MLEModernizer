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
import random
import gc
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures




## === cell 1
FOLDS = 5
SEED = 23
DEBUG = False




## === cell 2
def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


set_seed(SEED)




## === cell 3
train_path = "/kaggle/input/ventilator-pressure-prediction/train.csv"
test_path = "/kaggle/input/ventilator-pressure-prediction/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

if DEBUG:
    train_df = train_df.iloc[:20000]
    test_df = test_df.iloc[:20000]




## === cell 4
train_df["cum_u_in"] = train_df.groupby("breath_id")["u_in"].cumsum()
train_df["cum_u_out"] = train_df.groupby("breath_id")["u_out"].cumsum()

test_df["cum_u_in"] = test_df.groupby("breath_id")["u_in"].cumsum()
test_df["cum_u_out"] = test_df.groupby("breath_id")["u_out"].cumsum()

train_df["breath_id_feat"] = train_df["breath_id"].astype(np.float32)
test_df["breath_id_feat"] = test_df["breath_id"].astype(np.float32)

feature_cols = [
    "R",
    "C",
    "time_step",
    "u_in",
    "u_out",
    "cum_u_in",
    "cum_u_out",
    "breath_id_feat",
]

X = train_df[feature_cols].values.astype(np.float32)
y = train_df["pressure"].values.astype(np.float32)

X_test = test_df[feature_cols].values.astype(np.float32)

group = train_df["breath_id"].values




## === cell 5
pressure_values = np.array(sorted(train_df["pressure"].unique()))
diff = np.diff(pressure_values)
step = np.median(diff)

EXTRAPOLATE_KNOTS = 100
left_pressure_extrapolate = np.arange(
    pressure_values[0] - EXTRAPOLATE_KNOTS * step, pressure_values[0] - step, step
)
right_pressure_extrapolate = np.arange(
    pressure_values[-1] + step, pressure_values[-1] + EXTRAPOLATE_KNOTS * step, step
)

pressure_values_extra = np.concatenate(
    [left_pressure_extrapolate, pressure_values, right_pressure_extrapolate]
)
pressure_values_extra_mid_points = (
    pressure_values_extra[1:] + pressure_values_extra[:-1]
) / 2


def discretize_np(y_discr, y_midpoints, y_cont):
    """Map continuous predictions to the nearest discrete pressure knot."""
    indices = np.searchsorted(y_midpoints, y_cont, side="left")
    return y_discr[indices]


candidate_alphas = [0.1, 0.01, 0.001]
candidate_degrees = [2, 3]

best_mae = np.inf
best_alpha = None
best_degree = None
best_oof = None

gkf = GroupKFold(n_splits=FOLDS)
splits = list(gkf.split(X, y, groups=group))

for degree in candidate_degrees:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly_full = poly.fit_transform(X).astype(np.float32)

    for alpha in candidate_alphas:
        oof_preds = np.zeros_like(y)

        for train_idx, val_idx in splits:
            X_train_poly = X_poly_full[train_idx]
            X_val_poly = X_poly_full[val_idx]

            mean = X_train_poly.mean(axis=0, keepdims=True)
            std = X_train_poly.std(axis=0, keepdims=True)
            std[std == 0] = 1.0

            X_train_scaled = (X_train_poly - mean) / std
            X_val_scaled = (X_val_poly - mean) / std

            model = Ridge(
                alpha=alpha,
                random_state=SEED,
                solver="sag",
                max_iter=1000,
                fit_intercept=True,
            )
            model.fit(X_train_scaled, y[train_idx])
            oof_preds[val_idx] = model.predict(X_val_scaled)

            del X_train_poly, X_val_poly, X_train_scaled, X_val_scaled, mean, std, model
            gc.collect()

        mae = mean_absolute_error(y, oof_preds)
        print(f"Alpha {alpha:.4f}, Degree {degree}: OOF MAE = {mae:.6f}")

        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha
            best_degree = degree
            best_oof = oof_preds.copy()

    del X_poly_full, poly
    gc.collect()

print(
    f"\nBest config -> Alpha: {best_alpha}, Degree: {best_degree}, OOF MAE: {best_mae:.6f}"
)

oof_disc = discretize_np(
    pressure_values_extra, pressure_values_extra_mid_points, best_oof
)
print(f"OOF MAE after discretisation: {mean_absolute_error(y, oof_disc):.6f}")




## === cell 6
final_poly = PolynomialFeatures(degree=best_degree, include_bias=False)
X_full_poly = final_poly.fit_transform(X).astype(np.float32)

full_mean = X_full_poly.mean(axis=0, keepdims=True)
full_std = X_full_poly.std(axis=0, keepdims=True)
full_std[full_std == 0] = 1.0
X_full_scaled = (X_full_poly - full_mean) / full_std

final_model = Ridge(
    alpha=best_alpha,
    random_state=SEED,
    solver="sag",
    max_iter=1000,
    fit_intercept=True,
)
final_model.fit(X_full_scaled, y)

X_test_poly = final_poly.transform(X_test).astype(np.float32)
X_test_scaled = (X_test_poly - full_mean) / full_std
test_pred_cont = final_model.predict(X_test_scaled)

test_pred_disc = discretize_np(
    pressure_values_extra, pressure_values_extra_mid_points, test_pred_cont
)

submission = test_df[["id"]].copy()
submission["pressure"] = test_pred_disc

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
