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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor

os.environ["OMP_NUM_THREADS"] = "2"




## === cell 1
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_sub_path = "../input/ventilator-pressure-prediction/sample_submission.csv"




## === cell 2
def dropCols(df, cols):
    """Return a copy of df with the specified columns removed."""
    df = df.copy()
    df.drop(columns=cols, inplace=True)
    return df


def add_breath_stats(df, cols):
    """
    Compute mean, std, max, min, last for each column in `cols` per breath_id
    in a single grouped aggregation and merge back to the dataframe.
    """
    grp = df.groupby("breath_id")[cols].agg(["mean", "std", "max", "min", "last"])
    grp.columns = ["_".join([col, stat]) for col, stat in grp.columns]
    rename_map = {
        f"{col}_{stat}": f"{col}_breath_{stat}"
        for col in cols
        for stat in ["mean", "std", "max", "min", "last"]
    }
    grp.rename(columns=rename_map, inplace=True)
    std_cols = [f"{col}_breath_std" for col in cols]
    grp[std_cols] = grp[std_cols].fillna(0)
    df = df.join(grp, on="breath_id")
    return df


def engineer_features(df):
    """Create all features exactly as in the original solution, but with fewer groupby calls."""
    grp = df.groupby("breath_id")

    df["diff_u_in1"] = df["u_in"] - grp["u_in"].shift(1).fillna(0)
    df["diff_u_in2"] = df["u_in"] - grp["u_in"].shift(2).fillna(0)
    df["u_in_cumsum"] = grp["u_in"].cumsum()
    df["u_in_lag1"] = grp["u_in"].shift(1).fillna(0)
    df["u_in_lag2"] = grp["u_in"].shift(2).fillna(0)

    df["u_out_cumsum"] = grp["u_out"].cumsum()
    df["u_out_lag1"] = grp["u_out"].shift(1).fillna(0)

    df["time_step_sq"] = df["time_step"] ** 2
    df["R_C_inter"] = df["R"] * df["C"]
    df["u_in_x_u_out"] = df["u_in"] * df["u_out"]
    df["time_u_in"] = df["time_step"] * df["u_in"]
    df["time_u_out"] = df["time_step"] * df["u_out"]

    df["u_in_rate"] = df["diff_u_in1"] / df["time_step"].replace(0, np.nan)
    df["u_in_rate"].fillna(0, inplace=True)

    df = add_breath_stats(df, ["u_in", "u_out", "time_step"])

    df["breath_len"] = grp["time_step"].transform("max")
    return df




## === cell 3
dtypes = {
    "R": "int8",
    "C": "int8",
    "id": "int16",
    "breath_id": "int32",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
train_data = pd.read_csv(train_path, dtype=dtypes)
train_data = engineer_features(train_data)




## === cell 4
test_data = pd.read_csv(test_path, dtype=dtypes)
test_data = engineer_features(test_data)




## === cell 5
cols_2_drop = ["id", "breath_id"]




## === cell 6
train_df = dropCols(train_data, cols_2_drop)
y = train_df.pop("pressure").values.astype(
    np.float32
)  # ensure contiguous float32 array
X = train_df.values.astype(np.float32)  # numpy array (samples, features) as float32




## === cell 7
test_df = dropCols(test_data, cols_2_drop)
X_test = test_df.values.astype(np.float32)  # same shape as X (samples, features)




## === cell 8
def build_model(random_state):
    """
    Create a HistGradientBoostingRegressor configured for MAE.
    Parameters are kept exactly as in the original solution.
    """
    return HistGradientBoostingRegressor(
        loss="absolute_error",  # MAE‑compatible loss
        max_iter=2500,  # more boosting iterations
        learning_rate=0.02,  # slower learning for better convergence
        max_depth=None,  # allow deep trees
        random_state=random_state,
    )




## === cell 9
from concurrent.futures import ThreadPoolExecutor


def train_predict_rs(rs):
    """Fit a model with the given random_state and return its predictions on X_test."""
    model = build_model(random_state=rs)
    model.fit(X, y)
    return model.predict(X_test)


with ThreadPoolExecutor(max_workers=2) as executor:
    preds = list(executor.map(train_predict_rs, [42, 52]))

test_pred_mean = np.mean(preds, axis=0)




## === cell 10
submission = pd.read_csv(sample_sub_path)
submission["pressure"] = test_pred_mean
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"\nSubmission written to {submission_path} (shape: {submission.shape})")
