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
import glob
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor  # fallback model

warnings.filterwarnings("ignore", category=FutureWarning)


def find_nearest_vec(preds):
    """Round each prediction to the nearest pressure value seen in training."""
    preds = np.asarray(preds)
    idx = np.searchsorted(sorted_pressures, preds)
    idx = np.clip(idx, 1, total_pressures_len - 1)
    lower = sorted_pressures[idx - 1]
    upper = sorted_pressures[idx]
    return np.where(np.abs(preds - lower) < np.abs(upper - preds), lower, upper)


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


_SAMPLE_SUBMISSION_PATH = (
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)
_TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
_sample_sub = pd.read_csv(_SAMPLE_SUBMISSION_PATH)
_test_df = pd.read_csv(_TEST_PATH)

_lr_features = ["R", "C", "time_step", "u_in", "u_out"]
_train_full = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
    usecols=_lr_features + ["pressure"],
    dtype=np.float32,
)



## === cell 1
df_pressure = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
    usecols=["pressure"],
    dtype=np.float32,
)
unique_pressures = df_pressure["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)

rf_model = None  # will be created lazily if needed


def _train_fallback_model():
    """Load the full training data (pre‑loaded) and train the ExtraTreesRegressor."""
    global rf_model
    df_train = _train_full

    X_full = df_train[_lr_features].values  # already float32
    intercept = np.ones((X_full.shape[0], 1), dtype=np.float32)
    X_full = np.column_stack([intercept, X_full])
    y_full = df_train["pressure"].values.astype(np.float32)

    set_seed(2021)
    sample_size = min(2_000_000, X_full.shape[0])
    sample_idx = np.random.choice(X_full.shape[0], size=sample_size, replace=False)
    X_train = X_full[sample_idx]
    y_train = y_full[sample_idx]

    rf_model = ExtraTreesRegressor(
        n_estimators=1200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=5,
        random_state=2021,
        bootstrap=False,
    )
    rf_model.fit(X_train, y_train)


def predict_lr(df):
    """Fallback prediction using the trained ExtraTrees model."""
    global rf_model
    if rf_model is None:
        _train_fallback_model()
    X = df[_lr_features].astype(np.float32).values
    intercept = np.ones((X.shape[0], 1), dtype=np.float32)
    X = np.column_stack([intercept, X])
    preds = rf_model.predict(X)
    preds = np.clip(preds, sorted_pressures[0], sorted_pressures[-1])
    return preds




## === cell 2
def wc(input_list):
    """
    Weighted combination of up to two prediction files.
    Reads each CSV once and extracts its pressure column efficiently.
    """
    l = []
    arrs = []
    for path in input_list:
        public_lb_score = int(path.split("/")[-1].split(".")[1].split(" ")[0])
        l.append(public_lb_score)
        arrs.append(
            pd.read_csv(
                path, usecols=["pressure"], dtype=np.float32
            ).pressure.values.ravel()
        )
    if len(arrs) == 1:
        return arrs[0]
    weight1 = (l[1] / sum(l)) + 0.1
    weight2 = 1 - weight1
    return arrs[0] * weight1 + arrs[1] * weight2


def blend(a_path, b_path):
    a = pd.read_csv(a_path)
    b = pd.read_csv(b_path)
    a.pressure = a.pressure * 0.7 + b.pressure * 0.3
    a["pressure"] = find_nearest_vec(a.pressure.values)
    a.to_csv("blend.csv", index=False)
    return a


def g(dp):
    """
    Main blending routine.
    Vectorized weight generation and matrix multiplication replace slower loops.
    """
    if not os.path.isdir(dp):
        pred_array = np.zeros(_sample_sub.shape[0], dtype=np.float32)
    else:
        files = [p for p in glob.iglob(f"{dp}/*") if p.endswith(".csv")]
        if len(files) == 0:
            pred_array = np.zeros(_sample_sub.shape[0], dtype=np.float32)
        else:
            loop_time = 156
            splits = max(len(files) // 2, 1)
            files.sort()
            flist = []
            for i in range(splits):
                start = i * round(len(files) / splits)
                end = None if i == splits - 1 else (i + 1) * round(len(files) / splits)
                flist.append(wc(files[start:end]))
            flist_arr = np.vstack(flist)  # shape: (n_models, n_samples)

            n_models = flist_arr.shape[0]
            weight_matrix = np.empty((loop_time, n_models), dtype=np.float32)
            for i in range(loop_time):
                np.random.seed(i)  # deterministic per original logic
                w = np.random.rand(n_models).astype(np.float32)
                w /= w.sum()
                weight_matrix[i] = np.sort(w)[::-1]  # descending order as before

            pred_stack = weight_matrix @ flist_arr  # shape: (loop_time, n_samples)

            if pred_stack.shape[1] == _sample_sub.shape[0]:
                pred_array = np.median(pred_stack, axis=0)
            else:
                median_val = np.median(pred_stack)
                pred_array = np.full(_sample_sub.shape[0], median_val, dtype=np.float32)

    if np.all(pred_array == 0):
        pred_array = predict_lr(_test_df)

    output = _sample_sub.copy()
    output.pressure = pred_array
    output["pressure"] = find_nearest_vec(output.pressure.values)
    output.to_csv(
        f'rwb {loop_time if "loop_time" in locals() else 0} loops.csv', index=False
    )




## === cell 3
g("../input/gb-data-blending-recover")
