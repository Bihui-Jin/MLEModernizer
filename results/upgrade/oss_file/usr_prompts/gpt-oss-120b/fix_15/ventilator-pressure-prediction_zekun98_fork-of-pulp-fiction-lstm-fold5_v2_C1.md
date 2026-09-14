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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from concurrent.futures import ThreadPoolExecutor  # parallel model training




## === cell 1
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_sub_path = "../input/ventilator-pressure-prediction/sample_submission.csv"

usecols_train = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
dtype_train = {
    "breath_id": np.int32,
    "R": np.int16,
    "C": np.int16,
    "time_step": np.float32,
    "u_in": np.float32,
    "u_out": np.int8,
    "pressure": np.float32,
}
tr_full = pd.read_csv(train_path, usecols=usecols_train, dtype=dtype_train)

usecols_test = ["breath_id", "R", "C", "time_step", "u_in", "u_out"]
dtype_test = {
    "breath_id": np.int32,
    "R": np.int16,
    "C": np.int16,
    "time_step": np.float32,
    "u_in": np.float32,
    "u_out": np.int8,
}
te_full = pd.read_csv(test_path, usecols=usecols_test, dtype=dtype_test)


def add_features(df):
    g = df.groupby("breath_id", sort=False)

    df["cumsum_u_in"] = g["u_in"].cumsum()
    df["cumsum_u_out"] = g["u_out"].cumsum()

    df["prev_u_in"] = g["u_in"].shift().fillna(0)
    df["prev_u_out"] = g["u_out"].shift().fillna(0)

    df["delta_u_in"] = df["u_in"] - df["prev_u_in"]
    df["delta_u_out"] = df["u_out"] - df["prev_u_out"]

    df["delta_time"] = df["time_step"] - g["time_step"].shift().fillna(0)

    df["u_in_R"] = df["u_in"] * df["R"]
    df["u_in_C"] = df["u_in"] * df["C"]
    df["u_in_delta_time"] = df["u_in"] * df["delta_time"]
    df["u_out_R"] = df["u_out"] * df["R"]
    df["u_out_C"] = df["u_out"] * df["C"]

    df["time_step_sq"] = df["time_step"] ** 2
    df["cumsum_u_in_sq"] = df["cumsum_u_in"] ** 2

    df["breath_len"] = g["time_step"].transform("count")
    df["max_time_step"] = g["time_step"].transform("max")
    df["norm_time_step"] = df["time_step"] / df["max_time_step"]
    return df


tr = add_features(tr_full)
te = add_features(te_full)

submission = pd.read_csv(sample_sub_path)

mask_u0 = tr["u_out"] == 0
max_pressure_u0 = tr.loc[mask_u0, "pressure"].max()
min_pressure_u0 = tr.loc[mask_u0, "pressure"].min()
print(
    f"train pressure range when u_out==0: {min_pressure_u0:.2f} – {max_pressure_u0:.2f}"
)




## === cell 2
class config:
    post_processing = {
        "max_pressure": 70.0,  # upper bound
        "min_pressure": 0.0,  # lower bound
    }


feature_cols = [
    "R",
    "C",
    "time_step",
    "u_in",
    "u_out",
    "cumsum_u_in",
    "cumsum_u_out",
    "prev_u_in",
    "prev_u_out",
    "delta_u_in",
    "delta_u_out",
    "delta_time",
    "u_in_R",
    "u_in_C",
    "u_in_delta_time",
    "u_out_R",
    "u_out_C",
    "time_step_sq",
    "cumsum_u_in_sq",
    "breath_len",
    "max_time_step",
    "norm_time_step",
]

X = tr[feature_cols].astype(np.float32).values
y = tr["pressure"].astype(np.float32).values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

params1 = {
    "max_iter": 4000,
    "learning_rate": 0.005,
    "max_depth": 15,
    "max_bins": 255,
    "random_state": 42,
}
params2 = {
    "max_iter": 4000,
    "learning_rate": 0.005,
    "max_depth": 10,
    "max_bins": 255,
    "random_state": 7,
}


def train_model(params):
    model = HistGradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    return model


with ThreadPoolExecutor(max_workers=2) as executor:
    future1 = executor.submit(train_model, params1)
    future2 = executor.submit(train_model, params2)
    model1 = future1.result()
    model2 = future2.result()

val_pred1 = model1.predict(X_val)
print(f"Validation MAE (model1): {mean_absolute_error(y_val, val_pred1):.5f}")

val_pred2 = model2.predict(X_val)
print(f"Validation MAE (model2): {mean_absolute_error(y_val, val_pred2):.5f}")

val_pred_ens = (val_pred1 + val_pred2) / 2
print(f"Validation MAE (ensemble): {mean_absolute_error(y_val, val_pred_ens):.5f}")




## === cell 3
test_pred1 = model1.predict(te[feature_cols].astype(np.float32).values)
test_pred2 = model2.predict(te[feature_cols].astype(np.float32).values)
test_pred = (test_pred1 + test_pred2) / 2
submission["pressure"] = test_pred




## === cell 4
pp = config.post_processing
submission["pressure"] = np.clip(
    submission["pressure"], pp["min_pressure"], pp["max_pressure"]
)

display(submission.head())




## === cell 5
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
