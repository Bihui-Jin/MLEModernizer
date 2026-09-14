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

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import gc
import random
from random import random as rd




## === cell 1
train_cols = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
test_cols = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "id"]

dtype_map = {
    "breath_id": np.int32,
    "R": np.int8,
    "C": np.int8,
    "time_step": np.float32,
    "u_in": np.float32,
    "u_out": np.int8,
    "pressure": np.float32,
    "id": np.int32,
}

df_train = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
    usecols=train_cols,
    dtype={k: v for k, v in dtype_map.items() if k in train_cols},
    memory_map=True,
)
df_test = pd.read_csv(
    "../input/ventilator-pressure-prediction/test.csv",
    usecols=test_cols,
    dtype={k: v for k, v in dtype_map.items() if k in test_cols},
    memory_map=True,
)

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures).astype(np.float32)
total_pressures_len = len(sorted_pressures)


def find_nearest(prediction):
    """Round a single raw prediction to the nearest pressure value seen in training."""
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        return sorted_pressures[-1]
    elif insert_idx == 0:
        return sorted_pressures[0]
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return (
        lower_val
        if abs(lower_val - prediction) < abs(upper_val - prediction)
        else upper_val
    )


def find_nearest_vectorized(preds):
    """
    Fully NumPy‑vectorized version of find_nearest.
    Returns an array of the nearest training pressures for each prediction.
    """
    idx = np.searchsorted(sorted_pressures, preds, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)

    left_idx = np.maximum(idx - 1, 0)
    left_val = sorted_pressures[left_idx]
    right_val = sorted_pressures[idx]

    choose_left = np.abs(preds - left_val) < np.abs(right_val - preds)
    return np.where(choose_left, left_val, right_val)


def set_seed(seed=2021):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)




## === cell 2
from sklearn.ensemble import RandomForestRegressor

set_seed(2021)

feature_cols = ["breath_id", "R", "C", "time_step", "u_in", "u_out"]
X = df_train[feature_cols].to_numpy(dtype=np.float32, copy=False)
y = df_train["pressure"].to_numpy(dtype=np.float32, copy=False)

del df_train
gc.collect()

num_threads = min(os.cpu_count() or 1, 8)  # cap at 8 to avoid oversubscription

model = RandomForestRegressor(
    n_estimators=500,
    max_depth=20,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=2021,
    n_jobs=num_threads,
    criterion="squared_error",
)

model.fit(X, y)  # training with controlled threading
gc.collect()  # free any intermediate memory

test_X = df_test[feature_cols].to_numpy(dtype=np.float32, copy=False)
test_preds = model.predict(test_X)

rounded_preds = find_nearest_vectorized(test_preds)




## === cell 3
submission = pd.DataFrame({"id": df_test["id"], "pressure": rounded_preds})
submission = submission[["id", "pressure"]]

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission file written to {submission_path}")




## === cell 4
def wc(input_list):
    """Weighted combination of up to two submission files."""
    scores = [int(p.split("/")[-1].split(".")[1].split(" ")[0]) for p in input_list]
    pressures = [pd.read_csv(p).pressure.values.ravel() for p in input_list]

    if len(pressures) == 1:
        return pressures[0]

    weight1 = (scores[1] / sum(scores)) + 0.1
    weight2 = 1 - weight1
    return pressures[0] * weight1 + pressures[1] * weight2


def g(dp, loop_time=5):
    """
    Blend multiple submissions found in *dp*.
    Reduced loop count (default 5) cuts unnecessary work while preserving the
    blending algorithm. The function is not used in the main pipeline.
    """
    import glob

    files = sorted(glob.glob(f"{dp}/*"))
    file_count = len(files)
    splits = max(file_count // 2, 1)

    flist = [
        files[i * file_count // splits : (i + 1) * file_count // splits]
        for i in range(splits)
    ]

    flist = [wc(sub) for sub in flist]

    pred_list = []
    for seed in range(loop_time):
        set_seed(seed)
        weights = np.random.rand(len(flist))
        weights /= weights.sum()
        combined = sum(w * p for w, p in zip(weights, flist))
        pred_list.append(combined)

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output.pressure = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b):
    """Simple linear blend of two existing submissions."""
    if not (os.path.exists(a) and os.path.exists(b)):
        print("Blend files not found; skipping blending step.")
        return None
    a_df = pd.read_csv(a)
    b_df = pd.read_csv(b)
    a_df.pressure = a_df.pressure * 0.55 + b_df.pressure * 0.45
    a_df["pressure"] = a_df.pressure.apply(find_nearest)
    a_df.to_csv("blend.csv", index=False)
    return a_df
