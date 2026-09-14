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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import random
from random import random as rd
import gc  # explicit memory cleanup


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state




## === cell 1
def blend(a, b):
    """Blend two prediction files. If any file is missing, raise a clear error."""
    a_path, b_path = a, b
    if not os.path.isfile(a_path):
        raise FileNotFoundError(f"Blend source not found: {a_path}")
    if not os.path.isfile(b_path):
        raise FileNotFoundError(f"Blend source not found: {b_path}")
    a_df = pd.read_csv(a_path)
    b_df = pd.read_csv(b_path)
    a_df.pressure = a_df.pressure * 0.6 + b_df.pressure * 0.4
    a_df.to_csv("blend.csv", index=False)
    return a_df




## === cell 2
a_path = "../input/gb-blending/0.365.csv"
b_path = "../input/gb-blending/0.370 blend.csv"

try:
    blend(a_path, b_path)
except FileNotFoundError as e:
    print(f"Blending files not found ({e}). Proceeding to train a fallback model.")




## === cell 3
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def locate_csv(filename):
    possible_paths = [
        filename,
        os.path.join("..", "input", "ventilator-pressure-prediction", filename),
        os.path.join("..", "input", filename),
        os.path.join("data", filename),
        os.path.join("kaggle", "data", filename),
    ]
    for p in possible_paths:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Could not find {filename} in any known location.")


train_path = locate_csv("train.csv")
test_path = locate_csv("test.csv")
sample_sub_path = locate_csv("sample_submission.csv")

dtype_dict = {
    "R": np.int8,
    "C": np.int8,
    "time_step": np.float32,
    "u_in": np.float32,
    "u_out": np.int8,
    "pressure": np.float32,
    "breath_id": np.int32,
    "id": np.int16,
}

train_df = pd.read_csv(
    train_path,
    dtype=dtype_dict,
    usecols=["R", "C", "time_step", "u_in", "u_out", "pressure", "breath_id"],
)
test_df = pd.read_csv(
    test_path,
    dtype=dtype_dict,
    usecols=["R", "C", "time_step", "u_in", "u_out", "breath_id"],
)

train_df["cumsum_u_in"] = train_df.groupby("breath_id")["u_in"].cumsum()
train_df["cumsum_u_out"] = train_df.groupby("breath_id")["u_out"].cumsum()
test_df["cumsum_u_in"] = test_df.groupby("breath_id")["u_in"].cumsum()
test_df["cumsum_u_out"] = test_df.groupby("breath_id")["u_out"].cumsum()

train_df["u_in_time"] = train_df["u_in"] * train_df["time_step"]
train_df["u_out_time"] = train_df["u_out"] * train_df["time_step"]
test_df["u_in_time"] = test_df["u_in"] * test_df["time_step"]
test_df["u_out_time"] = test_df["u_out"] * test_df["time_step"]

train_df["u_in_diff"] = train_df.groupby("breath_id")["u_in"].diff().fillna(0)
train_df["u_out_diff"] = train_df.groupby("breath_id")["u_out"].diff().fillna(0)
test_df["u_in_diff"] = test_df.groupby("breath_id")["u_in"].diff().fillna(0)
test_df["u_out_diff"] = test_df.groupby("breath_id")["u_out"].diff().fillna(0)

train_df["R_C"] = train_df["R"] * train_df["C"]
train_df["u_in_R"] = train_df["u_in"] * train_df["R"]
train_df["u_in_C"] = train_df["u_in"] * train_df["C"]
test_df["R_C"] = test_df["R"] * test_df["C"]
test_df["u_in_R"] = test_df["u_in"] * test_df["R"]
test_df["u_in_C"] = test_df["u_in"] * test_df["C"]

train_df["breath_len"] = train_df.groupby("breath_id")["time_step"].transform("size")
test_df["breath_len"] = test_df.groupby("breath_id")["time_step"].transform("size")
train_df["u_in_mean"] = train_df.groupby("breath_id")["u_in"].transform("mean")
test_df["u_in_mean"] = test_df.groupby("breath_id")["u_in"].transform("mean")
train_df["u_out_mean"] = train_df.groupby("breath_id")["u_out"].transform("mean")
test_df["u_out_mean"] = test_df.groupby("breath_id")["u_out"].transform("mean")

train_df["time_step_norm"] = train_df["time_step"] / train_df["breath_len"]
test_df["time_step_norm"] = test_df["time_step"] / test_df["breath_len"]
train_df["cumsum_u_in_norm"] = train_df["cumsum_u_in"] / train_df["breath_len"]
test_df["cumsum_u_in_norm"] = test_df["cumsum_u_in"] / test_df["breath_len"]

train_df["u_in_sq"] = train_df["u_in"] ** 2
test_df["u_in_sq"] = test_df["u_in"] ** 2
train_df["time_step_sq"] = train_df["time_step"] ** 2
test_df["time_step_sq"] = test_df["time_step"] ** 2
train_df["R_plus_C"] = train_df["R"] + train_df["C"]
test_df["R_plus_C"] = test_df["R"] + test_df["C"]
train_df["u_in_std"] = train_df.groupby("breath_id")["u_in"].transform("std").fillna(0)
test_df["u_in_std"] = test_df.groupby("breath_id")["u_in"].transform("std").fillna(0)

feature_cols = [
    "R",
    "C",
    "time_step",
    "u_in",
    "u_out",
    "breath_id",
    "cumsum_u_in",
    "cumsum_u_out",
    "u_in_time",
    "u_out_time",
    "u_in_diff",
    "u_out_diff",
    "R_C",
    "u_in_R",
    "u_in_C",
    "breath_len",
    "u_in_mean",
    "u_out_mean",
    "time_step_norm",
    "cumsum_u_in_norm",
    "u_in_sq",
    "time_step_sq",
    "R_plus_C",
    "u_in_std",
]

X = train_df[feature_cols]
y = train_df["pressure"]
X_test = test_df[feature_cols]

del train_df, test_df
gc.collect()

set_seed(2021)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=2021, shuffle=True
)

hgb = HistGradientBoostingRegressor(
    loss="absolute_error",
    max_iter=1800,  # more trees for better fit
    learning_rate=0.005,  # smaller step for finer convergence
    max_depth=10,  # deeper trees to capture interactions
    random_state=2021,
    early_stopping=False,  # train on the full training split
)

hgb.fit(X_train, y_train)

val_pred = hgb.predict(X_val)
print("Validation MAE:", mean_absolute_error(y_val, val_pred))

test_pred = hgb.predict(X_test)

submission = pd.read_csv(sample_sub_path)  # contains correct 'id' column ordering
submission["pressure"] = test_pred
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} with shape {submission.shape}")
