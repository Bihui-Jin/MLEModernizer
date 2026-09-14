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
import os, numpy as np, pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error


def find_file(rel_path):
    for root, _, files in os.walk("/kaggle/input"):
        if rel_path in files:
            return os.path.join(root, rel_path)
    raise FileNotFoundError(f"{rel_path} not found in /kaggle/input")




## === cell 1
train_path = find_file("train.csv")
test_path = find_file("test.csv")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)




## === cell 2
train_df["time_bin"] = (train_df["time_step"] * 100).astype(np.int16)
test_df["time_bin"] = (test_df["time_step"] * 100).astype(np.int16)

rc_time_mean_df = (
    train_df.groupby(["R", "C", "u_out", "time_bin"])["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "rc_time_mean"})
)

train_df = train_df.merge(
    rc_time_mean_df, on=["R", "C", "u_out", "time_bin"], how="left"
)
test_df = test_df.merge(rc_time_mean_df, on=["R", "C", "u_out", "time_bin"], how="left")

rc_mean_df = (
    train_df.groupby(["R", "C", "u_out"])["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "rc_mean"})
)

train_df = train_df.merge(rc_mean_df, on=["R", "C", "u_out"], how="left")
test_df = test_df.merge(rc_mean_df, on=["R", "C", "u_out"], how="left")

global_mean = train_df["pressure"].mean()
train_df["rc_time_mean"].fillna(train_df["rc_mean"], inplace=True)
test_df["rc_time_mean"].fillna(test_df["rc_mean"], inplace=True)
test_df["rc_mean"].fillna(global_mean, inplace=True)

train_df["cum_u_in"] = train_df.groupby("breath_id")["u_in"].cumsum()
test_df["cum_u_in"] = test_df.groupby("breath_id")["u_in"].cumsum()

train_df["time_step_sq"] = train_df["time_step"] ** 2
test_df["time_step_sq"] = test_df["time_step"] ** 2

train_df["u_in_sq"] = train_df["u_in"] ** 2
test_df["u_in_sq"] = test_df["u_in"] ** 2

train_df["u_out_sq"] = train_df["u_out"] ** 2
test_df["u_out_sq"] = test_df["u_out"] ** 2

train_df["u_in_R"] = train_df["u_in"] * train_df["R"]
test_df["u_in_R"] = test_df["u_in"] * test_df["R"]

train_df["u_in_C"] = train_df["u_in"] * train_df["C"]
test_df["u_in_C"] = test_df["u_in"] * test_df["C"]

train_df["time_u_in"] = train_df["time_step"] * train_df["u_in"]
test_df["time_u_in"] = test_df["time_step"] * test_df["u_in"]

train_df["cum_u_in_R"] = train_df["cum_u_in"] * train_df["R"]
test_df["cum_u_in_R"] = test_df["cum_u_in"] * test_df["R"]

train_df["R_C"] = train_df["R"] * train_df["C"]
test_df["R_C"] = test_df["R"] * test_df["C"]

train_df["u_out_R"] = train_df["u_out"] * train_df["R"]
test_df["u_out_R"] = test_df["u_out"] * test_df["R"]

train_df["u_out_C"] = train_df["u_out"] * train_df["C"]
test_df["u_out_C"] = test_df["u_out"] * test_df["C"]

train_df["cum_u_in_sq"] = train_df["cum_u_in"] ** 2
test_df["cum_u_in_sq"] = test_df["cum_u_in"] ** 2

train_df["time_u_in_sq"] = train_df["time_u_in"] ** 2
test_df["time_u_in_sq"] = test_df["time_u_in"] ** 2

feature_cols = [
    "time_step",
    "u_in",
    "u_out",
    "cum_u_in",
    "time_step_sq",
    "u_in_sq",
    "u_out_sq",
    "R",
    "C",
    "u_in_R",
    "u_in_C",
    "time_u_in",
    "cum_u_in_R",
    "R_C",
    "rc_time_mean",
    "rc_mean",
    "time_bin",  # new
    "u_out_R",  # new
    "u_out_C",  # new
    "cum_u_in_sq",  # new
    "time_u_in_sq",  # new
]

y_train = train_df["pressure"].values.astype(np.float64)

scaler = StandardScaler()
X_scaled_full = scaler.fit_transform(train_df[feature_cols].values.astype(np.float64))

rng = np.random.RandomState(42)
indices = np.arange(X_scaled_full.shape[0])
rng.shuffle(indices)
split = int(0.9 * len(indices))
train_idx, val_idx = indices[:split], indices[split:]

X_train_scaled = X_scaled_full[train_idx]
X_val_scaled = X_scaled_full[val_idx]
y_train_split = y_train[train_idx]
y_val = y_train[val_idx]

candidate_degrees = [2, 3, 4]  # extend to degree 4
candidate_lams = [
    1e-2,
    1e-3,
    1e-4,
    1e-5,
    1e-6,
    1e-7,
    1e-8,
]  # explore smaller regularisation

best_mae = np.inf
best_degree = 3
best_lam = 1e-6

for deg in candidate_degrees:
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_val_poly = poly.transform(X_val_scaled)

    X_train_aug = np.concatenate(
        [np.ones((X_train_poly.shape[0], 1)), X_train_poly], axis=1
    )
    X_val_aug = np.concatenate([np.ones((X_val_poly.shape[0], 1)), X_val_poly], axis=1)

    for lam in candidate_lams:
        I = np.eye(X_train_aug.shape[1])
        I[0, 0] = 0  # do not regularise intercept
        A = X_train_aug.T @ X_train_aug + lam * I
        b = X_train_aug.T @ y_train_split
        w = np.linalg.solve(A, b)

        val_pred = X_val_aug @ w
        mae = mean_absolute_error(y_val, val_pred)
        if mae < best_mae:
            best_mae = mae
            best_degree = deg
            best_lam = lam

poly = PolynomialFeatures(degree=best_degree, include_bias=False)
X_train_poly = poly.fit_transform(X_scaled_full)
X_train_aug = np.concatenate(
    [np.ones((X_train_poly.shape[0], 1)), X_train_poly], axis=1
)

I = np.eye(X_train_aug.shape[1])
I[0, 0] = 0
A = X_train_aug.T @ X_train_aug + best_lam * I
b = X_train_aug.T @ y_train
weights = np.linalg.solve(A, b)

X_test_scaled = scaler.transform(test_df[feature_cols].values.astype(np.float64))
X_test_poly = poly.transform(X_test_scaled)
X_test_aug = np.concatenate([np.ones((X_test_poly.shape[0], 1)), X_test_poly], axis=1)
pressure_pred = X_test_aug @ weights

test_pred = pd.DataFrame({"id": test_df["id"], "pressure": pressure_pred})




## === cell 3
test_pred["id"] = test_pred["id"].astype("int32")
test_pred["pressure"] = test_pred["pressure"].astype("float32")
submission_path = "./submission.csv"
test_pred.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
