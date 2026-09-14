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
import gc
import numpy as np
import pandas as pd
import random
from random import random as rd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor




## === cell 1
df_train = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
    dtype={
        "R": np.int8,
        "C": np.int8,
        "breath_id": np.int32,
        "id": np.int16,
        "time_step": np.float32,
        "u_in": np.float32,
        "u_out": np.int8,
        "pressure": np.float32,
    },
)
unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state




## === cell 2
def add_engineered_features(df):
    """
    Adds cheap but informative features:
    * u_in_R, u_in_C, R_div_C  – already used
    * cum_u_in                – cumulative u_in per breath (captures total volume)
    * delta_u_in              – change of u_in from previous timestep in the same breath
    """
    df["u_in_R"] = (df["u_in"] * df["R"]).astype(np.float32)
    df["u_in_C"] = (df["u_in"] * df["C"]).astype(np.float32)
    df["R_div_C"] = (df["R"] / df["C"]).astype(np.float32)

    df["cum_u_in"] = (
        df.groupby("breath_id", sort=False)["u_in"].cumsum().astype(np.float32)
    )
    df["delta_u_in"] = (
        df.groupby("breath_id", sort=False)["u_in"]
        .diff()
        .fillna(0.0)
        .astype(np.float32)
    )
    return df


def train_and_predict(df_train, df_test, sample_frac=1.0, random_state=2021):
    """
    Train a HistGradientBoostingRegressor with engineered features.
    The model is fit **once** on the (optionally sampled) full training data.
    A hold‑out split is used only for reporting MAE; it does not affect training.
    """
    df_train = add_engineered_features(df_train)
    df_test = add_engineered_features(df_test)

    FEATURES = [
        "R",
        "C",
        "time_step",
        "u_in",
        "u_out",
        "breath_id",
        "u_in_R",
        "u_in_C",
        "R_div_C",
        "cum_u_in",
        "delta_u_in",
    ]

    X = df_train[FEATURES].to_numpy(dtype=np.float32)
    y = df_train["pressure"].to_numpy(dtype=np.float32)

    del df_train
    gc.collect()

    if 0 < sample_frac < 1.0:
        X, _, y, _ = train_test_split(
            X, y, train_size=sample_frac, random_state=random_state
        )

    rs = np.random.RandomState(random_state)
    n_samples = X.shape[0]
    val_mask = rs.rand(n_samples) < 0.05
    val_idx = np.where(val_mask)[0]

    model = HistGradientBoostingRegressor(
        max_iter=5000,
        learning_rate=0.01,
        max_leaf_nodes=31,
        max_bins=63,
        max_depth=10,
        loss="absolute_error",
        random_state=random_state,
        l2_regularization=0.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    model.fit(X, y)

    if len(val_idx) > 0:
        val_pred = model.predict(X[val_idx])
        val_mae = mean_absolute_error(y[val_idx], val_pred)
        print(f"Validation MAE (on small held‑out split): {val_mae:.5f}")

    X_test = df_test[FEATURES].to_numpy(dtype=np.float32)

    del df_test
    gc.collect()

    test_pred = model.predict(X_test)
    test_pred = np.clip(test_pred, 0.0, 50.0)

    del X, y, X_test
    gc.collect()

    return test_pred




## === cell 3
df_test = pd.read_csv(
    "../input/ventilator-pressure-prediction/test.csv",
    dtype={
        "R": np.int8,
        "C": np.int8,
        "breath_id": np.int32,
        "id": np.int16,
        "time_step": np.float32,
        "u_in": np.float32,
        "u_out": np.int8,
    },
)

raw_pred = train_and_predict(df_train, df_test, sample_frac=1.0, random_state=2021)

submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)
submission["pressure"] = raw_pred
submission = submission.sort_values("id").reset_index(drop=True)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")




## === cell 4
for var in ["df_train", "df_test", "raw_pred", "submission"]:
    if var in globals():
        del globals()[var]
gc.collect()
