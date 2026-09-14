# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.1728020907036779

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 7.54837) has done: 'I guard the loading of the unavailable .npy files and fall back to a simple linear regression model trained on the provided training data. This ensures the script runs without file‑not‑found errors, produces a valid `submission.csv`, and applies the existing rounding helper to keep predictions close to observed pressure values.'
- What this solution (achieved 5.5195) has done: 'I keep the overall structure but improve the fallback linear‑regression by fitting separate models for each lung‑type pair (`R`,`C`).  Adding a few simple polynomial features (`u_in²`, `time_step²`, `u_in*time_step`) gives a richer linear model while staying within the original “linear regression” approach, so the core logic is unchanged.  Predictions are still snapped to the nearest observed pressure via `find_nearest`, and the script writes the required `submission.csv`.'
- What this solution (achieved 5.91611) has done: 'I replace the fallback model with a richer global linear regression that uses many polynomial interaction features (including cubic terms) for all inputs, then fit it once on the full training set using NumPy least‑squares. This retains the linear‑regression spirit while giving the model far more expressive power, which should sharply lower the MAE toward the target. The rest of the script (loading data, rounding predictions with `find_nearest`, and writing `submission.csv`) remains unchanged.'
- What this solution (achieved 5.17095) has done: 'I keep the overall workflow and polynomial‑feature linear‑regression approach, but fit a separate regression for each lung‑type pair (`R`,`C`). Each group gets its own coefficients, so the model can better capture the distinct pressure dynamics of different lungs while still using the same least‑squares solution and the existing rounding helper. If a test row belongs to an unseen pair it falls back to the original global model. The script now writes a valid `submission.csv` and should achieve a lower MAE, moving the score toward the target.'
- What this solution (achieved 5.17095) has done: 'I extend the feature set with additional polynomial and interaction terms (e.g., R², C², R·C, R·u_in, C·u_in, R·time_step, C·time_step) while keeping the same per‑group linear‑regression workflow. The richer linear model should better capture the pressure dynamics of each lung‑type pair, which is expected to lower the MAE and move the score toward the target without altering the core architecture or training loop.'
- What this solution (achieved 5.17095) has done: 'I keep the overall workflow and polynomial‑feature linear‑regression structure, but replace the ordinary least‑squares solve with a tiny ridge regularisation (λ = 1e‑3).  Adding this stabilising term often lowers the validation MAE without altering the model class, so it moves the score toward the target while preserving the original logic.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import random
import gc




## === cell 1
df_train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)


def find_nearest(prediction):
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


def set_seed(seed=2021):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)




## === cell 2
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")




## === cell 3
def _add_poly_features(df):
    """Create a richer set of polynomial and interaction features."""
    df = df.copy()
    df["u_in_sq"] = df["u_in"] ** 2
    df["time_step_sq"] = df["time_step"] ** 2
    df["u_in_time"] = df["u_in"] * df["time_step"]
    df["u_in_cu"] = df["u_in"] ** 3
    df["time_step_cu"] = df["time_step"] ** 3
    df["u_in_sq_time"] = df["u_in_sq"] * df["time_step"]
    df["u_in_time_sq"] = df["u_in"] * df["time_step_sq"]
    df["R_sq"] = df["R"] ** 2
    df["C_sq"] = df["C"] ** 2
    df["R_C"] = df["R"] * df["C"]
    df["R_u_in"] = df["R"] * df["u_in"]
    df["C_u_in"] = df["C"] * df["u_in"]
    df["R_time_step"] = df["R"] * df["time_step"]
    df["C_time_step"] = df["C"] * df["time_step"]
    df["R_u_out"] = df["R"] * df["u_out"]
    df["C_u_out"] = df["C"] * df["u_out"]
    return df


base_features = ["R", "C", "u_in", "u_out", "time_step"]
train_feat = _add_poly_features(df_train[base_features])
test_feat = _add_poly_features(df_test[base_features])

y_train = df_train["pressure"].values


def _ridge_coeffs(X, y, lam=0.0):
    """Return ridge regression coefficients (including intercept).
    lam=0 recovers ordinary least‑squares."""
    n_features = X.shape[1]
    A = X.T @ X + lam * np.eye(n_features)
    b = X.T @ y
    return np.linalg.solve(A, b)


X_train_global = np.hstack([np.ones((train_feat.shape[0], 1)), train_feat.values])
coeffs_global = _ridge_coeffs(X_train_global, y_train, lam=0.0)

group_cols = ["R", "C"]
coeffs_by_group = {}

for (r, c), idx in train_feat.groupby(group_cols).groups.items():
    Xg = np.hstack([np.ones((len(idx), 1)), train_feat.loc[idx].values])
    yg = y_train[idx]
    coeffs_by_group[(r, c)] = _ridge_coeffs(Xg, yg, lam=0.0)


def predict_row(row):
    """Predict pressure for a single row using the appropriate group model."""
    key = (row["R"], row["C"])
    fv = np.concatenate(
        ([1], row[train_feat.columns].values)
    )  # order matches train_feat columns
    coeffs = coeffs_by_group.get(key, coeffs_global)
    return fv @ coeffs


pred_raw = test_feat.apply(predict_row, axis=1).values

pred_df = pd.DataFrame(
    {
        "breath_id": df_test["breath_id"],
        "time_step": df_test["time_step"],
        "pred_raw": pred_raw,
    }
)
pred_df = pred_df.sort_values(["breath_id", "time_step"])

pred_df["pred_smooth"] = pred_df.groupby("breath_id")["pred_raw"].transform(
    lambda x: x.rolling(window=3, min_periods=1, center=True).mean()
)

pred_smooth_ordered = pred_df.sort_index()["pred_smooth"].values

pred = np.vectorize(find_nearest)(pred_smooth_ordered)

df_test["pressure"] = pred
df_test[["id", "pressure"]].to_csv("submission.csv", index=False)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
LinAlgError                               Traceback (most recent call last)
/tmp/ipykernel_11/1060593644.py in <cell line: 0>()
     47     Xg = np.hstack([np.ones((len(idx), 1)), train_feat.loc[idx].values])
     48     yg = y_train[idx]
---> 49     coeffs_by_group[(r, c)] = _ridge_coeffs(Xg, yg, lam=0.0)
     50 
     51 

/tmp/ipykernel_11/1060593644.py in _ridge_coeffs(X, y, lam)
     34     A = X.T @ X + lam * np.eye(n_features)
     35     b = X.T @ y
---> 36     return np.linalg.solve(A, b)
     37 
     38 

/usr/local/lib/python3.11/dist-packages/numpy/linalg/linalg.py in solve(a, b)
    407     signature = 'DD->D' if isComplexType(t) else 'dd->d'
    408     extobj = get_linalg_error_extobj(_raise_linalgerror_singular)
--> 409     r = gufunc(a, b, signature=signature, extobj=extobj)
    410 
    411     return wrap(r.astype(result_t, copy=False))

/usr/local/lib/python3.11/dist-packages/numpy/linalg/linalg.py in _raise_linalgerror_singular(err, flag)
    110 
    111 def _raise_linalgerror_singular(err, flag):
--> 112     raise LinAlgError("Singular matrix")
    113 
    114 def _raise_linalgerror_nonposdef(err, flag):

LinAlgError: Singular matrix
