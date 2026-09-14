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
from sklearn.ensemble import GradientBoostingRegressor




## === cell 1
def load_data():
    """
    Load train and test CSVs from common Kaggle input locations.
    Includes the breath_id column for richer feature engineering.
    """
    train_cols = ["breath_id", "R", "C", "u_out", "u_in", "time_step", "pressure", "id"]
    train_dtypes = {
        "breath_id": np.int32,
        "R": np.int8,
        "C": np.int8,
        "u_out": np.int8,
        "u_in": np.float32,
        "time_step": np.float32,
        "pressure": np.float32,
        "id": np.int32,
    }

    test_cols = ["breath_id", "R", "C", "u_out", "u_in", "time_step", "id"]
    test_dtypes = {
        "breath_id": np.int32,
        "R": np.int8,
        "C": np.int8,
        "u_out": np.int8,
        "u_in": np.float32,
        "time_step": np.float32,
        "id": np.int32,
    }

    possible_paths = [
        "/kaggle/input/ventilator-pressure-prediction/train.csv",
        "../input/ventilator-pressure-prediction/train.csv",
        "ventilator-pressure-prediction/train.csv",
        "train.csv",
    ]
    for p in possible_paths:
        if os.path.exists(p):
            train_path = p
            break
    else:
        raise FileNotFoundError("train.csv not found")

    possible_paths_test = [
        "/kaggle/input/ventilator-pressure-prediction/test.csv",
        "../input/ventilator-pressure-prediction/test.csv",
        "ventilator-pressure-prediction/test.csv",
        "test.csv",
    ]
    for p in possible_paths_test:
        if os.path.exists(p):
            test_path = p
            break
    else:
        raise FileNotFoundError("test.csv not found")

    train = pd.read_csv(train_path, usecols=train_cols, dtype=train_dtypes)
    test = pd.read_csv(test_path, usecols=test_cols, dtype=test_dtypes)
    return train, test




## === cell 2
def engineer_features(df):
    """
    Build modeling features.
    Adds cumulative_u_in (per breath) and u_in_time (flow × time) to capture
    the underlying physics of pressure generation.
    """
    df = df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)

    cols = ["breath_id", "R", "C", "u_out", "u_in", "time_step", "id"]
    if "pressure" in df.columns:
        cols.append("pressure")
    df = df[cols]

    df["cumulative_u_in"] = df.groupby("breath_id")["u_in"].cumsum()
    df["u_in_time"] = df["u_in"] * df["time_step"]
    return df




## === cell 3
def train_model(train_feat):
    """
    Fit a GradientBoostingRegressor on the enriched feature set.
    Slightly more trees and a modest learning_rate increase retain capacity
    while staying quick enough for the full dataset.
    """
    feature_cols = [
        "R",
        "C",
        "u_out",
        "u_in",
        "time_step",
        "cumulative_u_in",
        "u_in_time",
    ]
    X = train_feat[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y = train_feat["pressure"].to_numpy(dtype=np.float32, copy=False)

    model = GradientBoostingRegressor(
        n_estimators=200,  # more trees for better fit
        learning_rate=0.08,  # keeps overall step size reasonable
        max_depth=3,
        subsample=0.7,
        random_state=42,
    )
    model.fit(X, y)
    return model




## === cell 4
def predict(test_feat, model):
    """
    Generate predictions using the trained model.
    """
    feature_cols = [
        "R",
        "C",
        "u_out",
        "u_in",
        "time_step",
        "cumulative_u_in",
        "u_in_time",
    ]
    X_test = test_feat[feature_cols].to_numpy(dtype=np.float32, copy=False)
    preds = model.predict(X_test)
    return pd.DataFrame({"id": test_feat["id"], "pred_pressure": preds})




## === cell 5
def run_pipeline():
    train_df, test_df = load_data()

    train_feat = engineer_features(train_df)
    test_feat = engineer_features(test_df)

    model = train_model(train_feat)

    preds = predict(test_feat, model)

    submission = preds.rename(columns={"pred_pressure": "pressure"})
    submission_path = "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")


run_pipeline()
