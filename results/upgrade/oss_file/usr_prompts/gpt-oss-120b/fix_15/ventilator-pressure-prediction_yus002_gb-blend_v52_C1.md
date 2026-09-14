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
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import glob
import random
import gc
from sklearn.ensemble import RandomForestRegressor  # import once globally

sorted_pressures = None
total_pressures_len = None


def init_pressure_lookup():
    """Load only the pressure column to build the nearest‑pressure vectors."""
    global sorted_pressures, total_pressures_len
    pressure_series = pd.read_csv(
        "../input/ventilator-pressure-prediction/train.csv",
        dtype={"pressure": "float32"},
        usecols=["pressure"],
    )["pressure"]
    unique_pressures = pressure_series.unique()
    del pressure_series
    sorted_pressures = np.sort(unique_pressures)
    total_pressures_len = len(sorted_pressures)


def find_nearest_vec(predictions):
    """Vectorized nearest‑pressure lookup (uses globals initialized lazily)."""
    preds = np.asarray(predictions, dtype=np.float32)
    idx = np.searchsorted(sorted_pressures, preds, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)
    lower_idx = np.maximum(idx - 1, 0)
    upper_idx = np.minimum(idx, total_pressures_len - 1)

    lower_val = sorted_pressures[lower_idx]
    upper_val = sorted_pressures[upper_idx]

    diff_lower = np.abs(preds - lower_val)
    diff_upper = np.abs(preds - upper_val)
    choose_upper = diff_upper < diff_lower
    return np.where(choose_upper, upper_val, lower_val)


def wc(input_list):
    """
    Load prediction CSVs, extract public leaderboard scores from filenames,
    and compute a deterministic weighted combination of the predictions.
    Optimized to avoid stacking all arrays simultaneously.
    """
    if not input_list:
        return np.array([])  # no files

    scores = np.array(
        [int(f.split("/")[-1].split(".")[1].split(" ")[0]) for f in input_list],
        dtype=np.float32,
    )

    score_sum = scores.sum()
    if score_sum == 0:
        weights = np.full_like(scores, 1.0 / len(scores))
    else:
        weights = scores / score_sum
    max_idx = np.argmax(scores)
    weights[max_idx] = min(weights[max_idx] + 0.1, 1.0)
    weights = weights / weights.sum()  # renormalise

    blended = None
    for f, w in zip(input_list, weights):
        arr = pd.read_csv(
            f, usecols=["pressure"], dtype={"pressure": "float32"}
        ).pressure.values.astype(np.float32)
        if blended is None:
            blended = w * arr
        else:
            blended += w * arr

    return blended


def g(dp):
    """
    Main entry point.
    If prediction CSVs exist under `dp`, blend them deterministically;
    otherwise train a RandomForest model on the full training set.
    """
    files = [f for f in glob.iglob(f"{dp}/*") if f.lower().endswith(".csv")]
    if files:
        files.sort()
        init_pressure_lookup()

        blended_pred = wc(files)

        output = pd.read_csv(
            "../input/ventilator-pressure-prediction/sample_submission.csv"
        )
        output.pressure = blended_pred
        output.pressure = find_nearest_vec(output.pressure)
        output.to_csv("blended_submission.csv", index=False)
        print("Blended submission saved as blended_submission.csv")
    else:
        test_path = "../input/ventilator-pressure-prediction/test.csv"
        sample_sub_path = (
            "../input/ventilator-pressure-prediction/sample_submission.csv"
        )
        dtype_feat_test = {
            "R": "int8",
            "C": "int8",
            "time_step": "float32",
            "u_in": "float32",
            "u_out": "int8",
            "breath_id": "int32",
            "id": "int16",
        }

        dtype_feat_full = {
            "R": "int8",
            "C": "int8",
            "time_step": "float32",
            "u_in": "float32",
            "u_out": "int8",
            "pressure": "float32",
        }
        train_df = pd.read_csv(
            "../input/ventilator-pressure-prediction/train.csv",
            dtype=dtype_feat_full,
            usecols=["R", "C", "time_step", "u_in", "u_out", "pressure"],
        )

        global sorted_pressures, total_pressures_len
        unique_pressures = train_df["pressure"].unique()
        sorted_pressures = np.sort(unique_pressures)
        total_pressures_len = len(sorted_pressures)

        test_df = pd.read_csv(
            test_path,
            dtype=dtype_feat_test,
            usecols=["R", "C", "time_step", "u_in", "u_out"],
        )

        features = ["R", "C", "time_step", "u_in", "u_out"]
        X = train_df[features].values
        y = train_df["pressure"].values

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=2021,
            n_jobs=-1,
            max_depth=None,
            min_samples_leaf=1,
        )
        model.fit(X, y)

        test_pred = model.predict(test_df[features].values)
        test_pred = find_nearest_vec(test_pred)  # vectorized mapping

        submission = pd.read_csv(sample_sub_path)
        submission["pressure"] = test_pred
        submission.to_csv("submission.csv", index=False)
        print("Model trained and submission saved as submission.csv")

        del X, y, train_df, test_df, model
        gc.collect()


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.6 + b.pressure * 0.4
    a["pressure"] = find_nearest_vec(a["pressure"].values)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 1
def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    pass




## === cell 2
g("../input/gb-blending")
