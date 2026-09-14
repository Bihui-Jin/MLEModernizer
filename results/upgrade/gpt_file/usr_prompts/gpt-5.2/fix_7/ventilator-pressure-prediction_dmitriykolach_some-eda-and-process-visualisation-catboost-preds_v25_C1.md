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

catboost==1.2.8
geopandas==0.14.4
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

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import numpy as np
import pandas as pd

np.random.seed(555)



## === cell 1
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from catboost import CatBoostRegressor, Pool




## === cell 2
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    if n % 80 == 0:
        u_in = df["u_in"].to_numpy(dtype=np.float32, copy=False)
        u_in_2d = u_in.reshape(-1, 80)  # (n_breaths, 80)

        u_in_cumsum = np.cumsum(u_in_2d, axis=1, dtype=np.float32).reshape(-1)

        lag1 = np.zeros_like(u_in_2d, dtype=np.float32)
        lag2 = np.zeros_like(u_in_2d, dtype=np.float32)
        lag1[:, 1:] = u_in_2d[:, :-1]
        lag2[:, 2:] = u_in_2d[:, :-2]
        u_in_lag_1 = lag1.reshape(-1)
        u_in_lag_2 = lag2.reshape(-1)

        u_in_rolling_mean_2d = np.zeros_like(u_in_2d, dtype=np.float32)
        u_in_rolling_mean_2d[:, 2:] = (u_in_2d[:, :-2] + u_in_2d[:, 1:-1]) * 0.5
        u_in_rolling_mean = u_in_rolling_mean_2d.reshape(-1)

        u_in_end = np.repeat(u_in_2d[:, -1], 80).astype(np.float32, copy=False)
        u_in_max = np.repeat(np.max(u_in_2d, axis=1), 80).astype(np.float32, copy=False)
        u_in_median = np.repeat(np.median(u_in_2d, axis=1), 80).astype(
            np.float32, copy=False
        )

        out = df.copy()
        out["u_in_cumsum"] = u_in_cumsum
        out["u_in_lag_1"] = u_in_lag_1
        out["u_in_lag_2"] = u_in_lag_2
        out["u_in_rolling_mean"] = u_in_rolling_mean
        out["u_in_end"] = u_in_end
        out["u_in_max"] = u_in_max
        out["u_in_median"] = u_in_median
        out = out.fillna(0)
        out = out.drop(["breath_id", "u_in", "u_out"], axis=1)
        return out

    g = df.groupby("breath_id", sort=False)
    u_in = df["u_in"]
    out = df.copy()
    out["u_in_cumsum"] = g["u_in"].cumsum()
    out["u_in_lag_1"] = g["u_in"].shift(1)
    out["u_in_lag_2"] = g["u_in"].shift(2)

    out["u_in_rolling_mean"] = g["u_in"].shift(1).rolling(2).mean()

    last_map = g["u_in"].last()
    max_map = g["u_in"].max()
    med_map = g["u_in"].median()
    bid = out["breath_id"]
    out["u_in_end"] = bid.map(last_map)
    out["u_in_max"] = bid.map(max_map)
    out["u_in_median"] = bid.map(med_map)

    out = out.fillna(0)
    out = out.drop(["breath_id", "u_in", "u_out"], axis=1)
    return out




## === cell 3
def train_and_score(model, X_train, y_train, X_valid, y_valid):
    if isinstance(model, CatBoostRegressor):
        train_pool = Pool(X_train.to_numpy(copy=False), y_train.to_numpy(copy=False))
        valid_pool = Pool(X_valid.to_numpy(copy=False), y_valid.to_numpy(copy=False))
        model.fit(train_pool)
        preds = model.predict(valid_pool)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)




## === cell 4
train_path = "/kaggle/input/ventilator-pressure-prediction/train.csv"
test_path = "/kaggle/input/ventilator-pressure-prediction/test.csv"
sub_path = "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv"

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

df_train = pd.read_csv(train_path, dtype=train_dtypes)
df_test = pd.read_csv(test_path, dtype=test_dtypes)
df_sample_submission = pd.read_csv(sub_path, dtype={"id": "int32", "pressure": "int16"})



## === cell 5
pressure_grid = np.sort(df_train["pressure"].unique())



## === cell 6
df_train_nid = df_train.drop("id", axis=1)

X_full = add_features(df_train_nid.drop("pressure", axis=1))
y_full = df_train_nid["pressure"]

insp_mask_train = df_train_nid["u_out"].to_numpy(copy=False) == 0
X = X_full.loc[insp_mask_train].reset_index(drop=True)
y = y_full.loc[insp_mask_train].reset_index(drop=True)



## === cell 7
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=555
)



## === cell 8
linear_model = LinearRegression()
tree_model = DecisionTreeRegressor(max_depth=15, random_state=555)

cb_model = CatBoostRegressor(
    depth=15,
    loss_function="MAE",
    random_seed=555,
    verbose=0,
    thread_count=4,
    boosting_type="Plain",
    allow_writing_files=False,
    score_function="L2",
    leaf_estimation_method="Gradient",
    train_dir="catboost_info",
    task_type="CPU",
)

dummy = DummyRegressor()



## === cell 9
_ = train_and_score(cb_model, X_train, y_train, X_valid, y_valid)



## === cell 10
cb_model.fit(Pool(X.to_numpy(copy=False), y.to_numpy(copy=False)))



## === cell 11
df_test_featured = add_features(df_test)

X_test = df_test_featured.drop("id", axis=1)

preds = cb_model.predict(Pool(X_test.to_numpy(copy=False)))

preds = np.asarray(preds, dtype=np.float64)

u_out_test = df_test["u_out"].to_numpy(copy=False)
preds[u_out_test == 1] = 0.0

insp_mask = u_out_test == 0
if insp_mask.any():
    p = preds[insp_mask]
    idx = np.searchsorted(pressure_grid, p, side="left")
    idx = np.clip(idx, 0, len(pressure_grid) - 1)
    left = np.clip(idx - 1, 0, len(pressure_grid) - 1)
    right = idx
    choose_right = np.abs(pressure_grid[right] - p) <= np.abs(pressure_grid[left] - p)
    snapped = pressure_grid[left].copy()
    snapped[choose_right] = pressure_grid[right][choose_right]
    preds[insp_mask] = snapped

output = pd.DataFrame({"id": df_test["id"].values, "pressure": preds})
output.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", output.shape)
print(output.head())
