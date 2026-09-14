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
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from types import SimpleNamespace




## === cell 1
def add_lag_features(df, lags=(1, 2)):
    """
    For each breath_id, create lagged versions of the control signals.
    Missing values (first steps) are filled with 0.
    """
    df = df.copy()
    df.sort_values(["breath_id", "time_step"], inplace=True)
    for lag in lags:
        df[f"u_in_lag{lag}"] = df.groupby("breath_id")["u_in"].shift(lag).fillna(0)
        df[f"u_out_lag{lag}"] = df.groupby("breath_id")["u_out"].shift(lag).fillna(0)
        df[f"time_step_lag{lag}"] = (
            df.groupby("breath_id")["time_step"].shift(lag).fillna(0)
        )
    return df


def train_linear_model(df, target_col="pressure", alpha=0.3, degree=5, batch_size=5000):
    """
    Train a Ridge regression on polynomial features using a two‑pass,
    batch‑wise approach (first fit the scaler, then solve the ridge normal equations).
    The returned object mimics the original model (has .scaler, .degree and .coef_ attributes).
    """
    scaler = StandardScaler()
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    exclude = {"pressure", "id", "breath_id"}
    feature_cols = [
        c
        for c in df.columns
        if c not in exclude and np.issubdtype(df[c].dtype, np.number)
    ]

    n_rows = df.shape[0]
    first_batch = True
    for start in range(0, n_rows, batch_size):
        batch = df.iloc[start : start + batch_size]
        X_base = batch[feature_cols].astype(float).values
        if first_batch:
            X_poly = poly.fit_transform(X_base)  # fit once on first batch
            first_batch = False
        else:
            X_poly = poly.transform(X_base)
        scaler.partial_fit(X_poly)

    n_features = scaler.mean_.shape[0]  # number of polynomial features
    XtX = np.zeros(
        (n_features + 1, n_features + 1), dtype=np.float64
    )  # +1 for constant
    Xty = np.zeros(n_features + 1, dtype=np.float64)

    y = df[target_col].astype(float).values

    for start in range(0, n_rows, batch_size):
        end = start + batch_size
        batch_X = df.iloc[start:end]
        batch_y = y[start:end]

        X_base = batch_X[feature_cols].astype(float).values
        X_poly = poly.transform(X_base)
        X_scaled = scaler.transform(X_poly)
        X_final = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])

        XtX += X_final.T @ X_final
        Xty += X_final.T @ batch_y

    reg_matrix = alpha * np.eye(XtX.shape[0])
    w = np.linalg.solve(XtX + reg_matrix, Xty)

    model = SimpleNamespace()
    model.coef_ = w
    model.scaler = scaler
    model.degree = degree
    model.poly = poly
    model.feature_cols = feature_cols
    return model


def predict_linear_model(df, model, batch_size=5000):
    """
    Predict using the batch‑wise model returned by train_linear_model.
    """
    n_rows = df.shape[0]
    preds = np.empty(n_rows, dtype=np.float64)

    for start in range(0, n_rows, batch_size):
        end = start + batch_size
        batch = df.iloc[start:end]
        X_base = batch[model.feature_cols].astype(float).values
        X_poly = model.poly.transform(X_base)
        X_scaled = model.scaler.transform(X_poly)
        X_final = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])
        preds[start:end] = X_final @ model.coef_

    return preds




## === cell 2
def locate_file(filename):
    possible_paths = [
        filename,
        os.path.join("..", "input", filename),
        os.path.join("..", "..", "input", filename),
        os.path.join("/kaggle", "input", filename),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Cannot locate {filename}")


train_path = locate_file("train.csv")
test_path = locate_file("test.csv")
sample_sub_path = locate_file("sample_submission.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_df = add_lag_features(train_df, lags=(1, 2))
test_df = add_lag_features(test_df, lags=(1, 2))

model = train_linear_model(train_df, alpha=0.3, degree=5, batch_size=5000)

test_pred = predict_linear_model(test_df, model, batch_size=5000)

min_pressure = train_df["pressure"].min()
max_pressure = train_df["pressure"].max()
test_pred = np.clip(test_pred, min_pressure, max_pressure)

submission = pd.DataFrame({"id": test_df["id"], "pressure": test_pred})
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path} with shape {submission.shape}")
