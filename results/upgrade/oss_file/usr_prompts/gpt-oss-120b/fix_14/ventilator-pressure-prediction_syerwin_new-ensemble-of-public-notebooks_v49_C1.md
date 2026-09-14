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
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

dtype_dict = {
    "R": "int8",
    "C": "int8",
    "breath_id": "int32",
    "id": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"

train = pd.read_csv(train_path, dtype=dtype_dict)
test = pd.read_csv(test_path, dtype=dtype_dict)




## === cell 1
def add_features(df):
    df["u_in_time"] = df["u_in"] * df["time_step"]
    df["R_C"] = df["R"] * df["C"]
    df["u_in_R"] = df["u_in"] * df["R"]
    df["u_in_C"] = df["u_in"] * df["C"]
    df["time_R"] = df["time_step"] * df["R"]
    df["time_C"] = df["time_step"] * df["C"]
    df["u_in_sq"] = df["u_in"] ** 2
    df["time_step_sq"] = df["time_step"] ** 2
    df["R_div_C"] = df["R"] / df["C"]
    df["C_div_R"] = df["C"] / df["R"]

    grp = df.groupby("breath_id", observed=True, sort=False)

    df["cumsum_u_in"] = grp["u_in"].cumsum()
    df["cumsum_u_out"] = grp["u_out"].cumsum()
    df["delta_u_in"] = grp["u_in"].diff().fillna(0)
    df["delta_u_out"] = grp["u_out"].diff().fillna(0)

    agg = grp.agg(
        {
            "time_step": ["max", "count"],
            "u_in": ["mean", "max"],
            "u_out": ["mean", "max"],
        },
        sort=False,
    )
    agg.columns = [
        "max_time",
        "breath_len",
        "breath_u_in_mean",
        "breath_u_in_max",
        "breath_u_out_mean",
        "breath_u_out_max",
    ]
    df = df.join(agg, on="breath_id")

    df["norm_time"] = df["time_step"] / df["max_time"]
    df["cumsum_u_in_time"] = df["cumsum_u_in"] * df["time_step"]
    df["cumsum_u_out_time"] = df["cumsum_u_out"] * df["time_step"]
    df["norm_time_u_in"] = df["norm_time"] * df["u_in"]
    return df


train = add_features(train)
test = add_features(test)

base_features = ["R", "C", "time_step", "u_in", "u_out"]
feature_cols = base_features + [
    "u_in_time",
    "R_C",
    "u_in_R",
    "u_in_C",
    "time_R",
    "time_C",
    "cumsum_u_in",
    "cumsum_u_out",
    "cumsum_u_in_time",
    "cumsum_u_out_time",
    "delta_u_in",
    "delta_u_out",
    "norm_time",
    "u_in_sq",
    "norm_time_u_in",
    "breath_u_in_mean",
    "breath_u_in_max",
    "breath_u_out_mean",
    "breath_u_out_max",
    "R_div_C",
    "C_div_R",
    "time_step_sq",
    "breath_len",
]

train_insp = train[train["u_out"] == 0]
X = train_insp[feature_cols].to_numpy(dtype=np.float32)  # convert once for speed
y = train_insp["pressure"].to_numpy(dtype=np.float32)

model = HistGradientBoostingRegressor(
    loss="absolute_error",
    max_depth=20,
    learning_rate=0.01,
    max_iter=10000,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
)
model.fit(X, y)




## === cell 2
test_features = test[feature_cols].to_numpy(
    dtype=np.float32
)  # numpy array for fast prediction
preds = model.predict(test_features)

preds = np.clip(preds, a_min=0, a_max=None)

submission = pd.DataFrame({"id": test["id"], "pressure": preds})
submission.to_csv("submission.csv", index=False)

submission.head()
