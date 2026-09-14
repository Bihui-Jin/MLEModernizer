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
import glob
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # fallback model




## === cell 1
df_train = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
    dtype={
        "R": np.float32,
        "C": np.float32,
        "u_in": np.float32,
        "u_out": np.float32,
        "time_step": np.float32,
        "pressure": np.float32,
    },
    memory_map=True,  # <-- faster loading, no extra copy
)

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)


def find_nearest_vectorized(predictions):
    """
    Vectorized nearest‑value lookup using the pre‑sorted pressure array.
    """
    preds = np.asarray(predictions, dtype=np.float32)
    idx = np.searchsorted(sorted_pressures, preds, side="left")
    idx_low = np.clip(idx - 1, 0, total_pressures_len - 1)
    idx_high = np.clip(idx, 0, total_pressures_len - 1)

    low_vals = sorted_pressures[idx_low]
    high_vals = sorted_pressures[idx_high]

    choose_low = np.abs(low_vals - preds) < np.abs(high_vals - preds)
    return np.where(choose_low, low_vals, high_vals)


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    """
    Load prediction files, keep those with allowed public scores,
    and combine them. Reads only the 'pressure' column as float32.
    """
    l = []
    allow = [1348, 1358, 1758]
    for idx in range(len(input_list)):
        try:
            public_lb_score = int(
                input_list[idx].split("/")[-1].split(".")[1].split(" ")[0]
            )
        except Exception:
            continue
        if public_lb_score in allow:
            l.append(public_lb_score)
            input_list[idx] = pd.read_csv(
                input_list[idx],
                usecols=["pressure"],
                dtype={"pressure": np.float32},
                memory_map=True,  # <-- faster loading
            ).pressure.values.ravel()
        else:
            continue
    if len(l) == 0:
        if len(input_list) > 0 and isinstance(input_list[0], np.ndarray):
            return np.zeros_like(input_list[0])
        else:
            return np.array([])
    if len(l) == 1:
        return input_list[0]
    l_sum = sum(l)
    weight1 = (l[1] / l_sum) + 0.1
    weight2 = 1 - weight1
    output = input_list[0] * weight1 + input_list[1] * weight2
    return output


def g(dp):
    """
    Ensemble routine that blends high‑score submissions.
    The heavy 125‑iteration Python loop is replaced by a single
    broadcasting operation, dramatically reducing runtime.
    """
    if not os.path.isdir(dp):
        raise FileNotFoundError("Ensemble directory not found.")
    l = []
    allow = [1348, 1358, 1758]
    for i in glob.iglob(f"{dp}/*"):
        try:
            file_lb = int(i.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            continue
        if file_lb in allow:
            l.append(i)
    if not l:
        raise FileNotFoundError("No allowed high‑score submission files found.")
    file_count = len(l)
    loop_time = 125
    splits = 2
    l.sort()
    flist = [
        (
            l[i * round(len(l) / splits) : (i + 1) * round(len(l) / splits)]
            if i < splits - 1
            else l[i * round(len(l) / splits) :]
        )
        for i in range(splits)
    ]
    flist = [wc(part) for part in flist]

    rng = np.random.default_rng()
    weights = rng.random((loop_time, splits))
    weights /= weights.sum(axis=1, keepdims=True)

    weighted = (weights[..., None] * np.stack(flist, axis=0)).sum(axis=1)

    median_pred = np.median(weighted, axis=0)

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output.pressure = median_pred
    output["pressure"] = find_nearest_vectorized(output["pressure"])
    output.to_csv(f"rwb_{loop_time}_loops.csv", index=False)


def fallback_prediction():
    """
    Simple RandomForest model trained on the provided training data.
    Reuses the pre‑loaded df_train and reads only necessary test columns.
    """
    test_path = "../input/ventilator-pressure-prediction/test.csv"
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"

    train = df_train  # already in memory with float32 dtypes

    test = pd.read_csv(
        test_path,
        usecols=["R", "C", "u_in", "u_out", "time_step"],
        dtype={
            "R": np.float32,
            "C": np.float32,
            "u_in": np.float32,
            "u_out": np.float32,
            "time_step": np.float32,
        },
        memory_map=True,  # <-- faster loading
    )

    features = ["R", "C", "u_in", "u_out", "time_step"]
    X_train = train[features].values  # float32, contiguous
    y_train = train["pressure"].values  # float32

    n_jobs = -1

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=n_jobs,
        max_depth=12,
    )
    model.fit(X_train, y_train)

    preds = model.predict(test[features].values)
    submission = pd.read_csv(sample_path)
    submission["pressure"] = find_nearest_vectorized(preds)
    submission.to_csv("submission.csv", index=False)




## === cell 2
high_score_dir = "../input/ventilator-pressure-high-score-submissions"
if os.path.isdir(high_score_dir):
    try:
        g(high_score_dir)
    except Exception as e:
        print(f"Ensemble failed with error: {e}. Using fallback model.")
        fallback_prediction()
else:
    fallback_prediction()
