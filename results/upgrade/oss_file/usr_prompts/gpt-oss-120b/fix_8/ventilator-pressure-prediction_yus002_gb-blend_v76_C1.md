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
from sklearn.ensemble import GradientBoostingRegressor

TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SAMPLE_SUB_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"

TRAIN_COLS = ["u_in", "u_out", "R", "C", "time_step", "pressure"]
TEST_COLS = ["id", "u_in", "u_out", "R", "C", "time_step"]

TRAIN_DTYPE = {
    "u_in": "float32",
    "u_out": "int8",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "pressure": "float32",
}
TEST_DTYPE = {
    "id": "int32",
    "u_in": "float32",
    "u_out": "int8",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
}
df_train_full = pd.read_csv(TRAIN_PATH, usecols=["pressure"])
unique_pressures = df_train_full["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)


def find_nearest(prediction):
    pred_arr = np.asarray(prediction)
    idx = np.searchsorted(sorted_pressures, pred_arr, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)
    lower_idx = np.maximum(idx - 1, 0)

    lower = sorted_pressures[lower_idx]
    upper = sorted_pressures[idx]

    choose_lower = np.abs(lower - pred_arr) < np.abs(upper - pred_arr)
    return np.where(choose_lower, lower, upper)


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    l = []
    for i in range(len(input_list)):
        try:
            public_lb_score = int(
                input_list[i].split("/")[-1].split(".")[1].split(" ")[0]
            )
        except Exception:
            public_lb_score = 0
        l.append(public_lb_score)
        input_list[i] = pd.read_csv(input_list[i]).pressure.values.ravel()
    if len(l) == 0:
        return np.array([])
    if len(l) == 1:
        return input_list[0]
    l_sum = sum(l)
    weight1 = (l[1] / l_sum) + 0.1
    weight2 = 1 - weight1
    output = input_list[0] * weight1 + input_list[1] * weight2
    return output


def g(dp):
    files = [p for p in glob.iglob(f"{dp}/*") if os.path.isfile(p)]
    file_count = len(files)
    if file_count == 0:
        sample = pd.read_csv(SAMPLE_SUB_PATH)
        sample.to_csv("rwb_0_loops.csv", index=False)
        return

    preds_list = []
    if file_count == 1:
        pred = pd.read_csv(files[0]).pressure.values.ravel()
        preds_list.append(pred)
    else:
        splits = file_count // 2 if file_count >= 2 else 1
        files.sort()
        flist = []
        for i in range(splits):
            if i == splits - 1:
                flist.append(files[i * round(len(files) / splits) :])
            else:
                flist.append(
                    files[
                        i
                        * round(len(files) / splits) : (i + 1)
                        * round(len(files) / splits)
                    ]
                )
        for group in flist:
            combined = wc(group)
            preds_list.append(combined)

    if len(preds_list) == 1:
        median_pred = preds_list[0]
    else:
        median_pred = np.median(np.stack(preds_list, axis=0), axis=0)

    median_pred = find_nearest(median_pred)

    output = pd.read_csv(SAMPLE_SUB_PATH)
    output["pressure"] = median_pred
    output.to_csv(f"rwb_150_loops.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.5 + b.pressure * 0.5
    a["pressure"] = find_nearest(a["pressure"])
    a.to_csv("blend.csv", index=False)
    return a


def add_interaction_features(df):
    df["u_in_time"] = df["u_in"] * df["time_step"]
    df["u_out_time"] = df["u_out"] * df["time_step"]
    return df


def generate_baseline_predictions():
    train = pd.read_csv(
        TRAIN_PATH,
        usecols=TRAIN_COLS,
        dtype=TRAIN_DTYPE,
    )
    test = pd.read_csv(
        TEST_PATH,
        usecols=TEST_COLS,
        dtype=TEST_DTYPE,
    )

    feats = ["u_in", "u_out", "R", "C", "time_step", "u_in_time", "u_out_time"]

    train = add_interaction_features(train)
    test = add_interaction_features(test)

    train_sample = train.sample(frac=0.30, random_state=42)
    X_train = train_sample[feats].values
    y_train = train_sample["pressure"].values

    gbr = GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )
    gbr.fit(X_train, y_train)

    X_test = test[feats].values
    preds = gbr.predict(X_test)

    min_press, max_press = train["pressure"].min(), train["pressure"].max()
    preds = np.clip(preds, min_press, max_press)

    pred_df = pd.DataFrame({"id": test["id"], "pressure": preds})

    out_dir = "tmp_preds"
    os.makedirs(out_dir, exist_ok=True)
    pred_df.to_csv(os.path.join(out_dir, "baseline.csv"), index=False)




## === cell 1
generate_baseline_predictions()
g("tmp_preds")
