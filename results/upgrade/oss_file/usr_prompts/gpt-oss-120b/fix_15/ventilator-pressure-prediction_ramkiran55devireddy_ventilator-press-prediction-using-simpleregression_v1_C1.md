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
tqdm==4.67.1
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
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"

dtypes = {
    "id": "int16",
    "breath_id": "int32",
    "R": "int8",
    "C": "int8",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
train = pd.read_csv(train_path, dtype=dtypes)
test = pd.read_csv(test_path, dtype=dtypes)
sample = pd.read_csv(sample_path)



## === cell 1
train["train_test"] = "train"
test["train_test"] = "test"

full = pd.concat([train, test], axis=0, ignore_index=True)


def add_features(df):
    df["breath_id"] = df["breath_id"].astype("category")

    def group_feat(g):
        g = g.copy()

        g["last_value_u_in"] = g["u_in"].iloc[-1]
        g["mean_value_u_in"] = g["u_in"].mean()
        g["diff_u_in"] = g["u_in"].diff().fillna(0)
        g["diff_diff_u_in"] = g["diff_u_in"].diff().fillna(0)
        g["u_in_cumsum"] = g["u_in"].cumsum()
        sum_u_in = g["u_in"].sum()
        g["sum_value_u_in"] = sum_u_in
        g["u_in_cumsum_rate"] = g["u_in_cumsum"] / (
            sum_u_in if sum_u_in != 0 else np.nan
        )
        g["lag_u_in"] = g["u_in"].shift(1).fillna(0)
        g["lag_2_u_in"] = g["u_in"].shift(2).fillna(0)
        g["max_u_in_breathid"] = g["u_in"].max()
        g["area"] = (g["time_step"] * g["u_in"]).cumsum()
        g["u_in_times_time"] = g["u_in"] * g["time_step"]
        g["u_in_squared"] = g["u_in"] ** 2

        g["last_value_u_out"] = g["u_out"].iloc[-1]
        g["mean_value_u_out"] = g["u_out"].mean()
        g["diff_u_out"] = g["u_out"].diff().fillna(0)
        g["diff_diff_u_out"] = g["diff_u_out"].diff().fillna(0)
        g["u_out_cumsum"] = g["u_out"].cumsum()
        sum_u_out = g["u_out"].sum()
        g["sum_value_u_out"] = sum_u_out
        g["lag_u_out"] = g["u_out"].shift(1).fillna(0)
        g["lag_2_u_out"] = g["u_out"].shift(2).fillna(0)

        g["u_in_u_out"] = g["u_in"] * g["u_out"]
        g["time_step_squared"] = g["time_step"] ** 2

        return g

    df = df.groupby("breath_id", group_keys=False).apply(group_feat)

    df["R_C"] = df["R"].astype(str) + "_" + df["C"].astype(str)
    return df


full = add_features(full)



## === cell 2
full = pd.get_dummies(full, columns=["R_C"])

train = full[full["train_test"] == "train"].drop(columns=["train_test"])
test = full[full["train_test"] == "test"].drop(columns=["train_test", "pressure"])



## === cell 3
target = "pressure"
drop_cols = ["id", "breath_id", target]  # identifiers
X_train = train.drop(columns=drop_cols)
y_train = train[target]
X_test = test.drop(columns=["id", "breath_id"])

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_imp)
X_test_poly = poly.transform(X_test_imp)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train_poly)
X_test_std = scaler.transform(X_test_poly)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_std, y_train, test_size=0.2, random_state=42
)

lr_base = Ridge(alpha=0.0, random_state=42, solver="sag")
lr_base.fit(X_tr, y_tr)
val_pred_lr = lr_base.predict(X_val)
mae_lr = mean_absolute_error(y_val, val_pred_lr)

best_ridge_mae = np.inf
best_alpha = None
best_ridge_model = None
for alpha in [0.1, 1.0, 10.0, 100.0]:
    ridge_tmp = Ridge(alpha=alpha, random_state=42, solver="sag")
    ridge_tmp.fit(X_tr, y_tr)
    val_pred = ridge_tmp.predict(X_val)
    mae = mean_absolute_error(y_val, val_pred)
    if mae < best_ridge_mae:
        best_ridge_mae = mae
        best_alpha = alpha
        best_ridge_model = ridge_tmp

if best_ridge_mae < mae_lr:
    chosen_model = best_ridge_model
    chosen_name = f"Ridge(alpha={best_alpha})"
    chosen_mae = best_ridge_mae
else:
    chosen_model = lr_base
    chosen_name = "LinearRegression (Ridge alpha=0)"
    chosen_mae = mae_lr

print(f"Chosen model: {chosen_name} with validation MAE = {chosen_mae:.4f}")

chosen_model.fit(X_train_std, y_train)
train_pred = chosen_model.predict(X_train_std)
print("In‑sample MAE:", mean_absolute_error(y_train, train_pred))



## === cell 4
test_pred = chosen_model.predict(X_test_std)
sample["pressure"] = test_pred
sample.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")
