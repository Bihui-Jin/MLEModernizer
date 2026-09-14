# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.1390536612619201

# 6. Current score

1.73447

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.90025) has done: 'The update switches to Scikit‑Learn’s histogram‑based Gradient Boosting which is orders‑of‑magnitude faster on millions of rows while keeping the same boosting logic and hyper‑parameters. Minor dtype casting and avoiding unnecessary copies further reduce overhead, and the rest of the pipeline (nearest‑pressure rounding, blending, etc.) stays unchanged.'
- What this solution (achieved 4.23117) has done: 'I added a few inexpensive engineered features that capture the interaction between lung attributes (`R*C`) and a non‑linear term for the inspiratory valve (`u_in²`). These columns are created for both train and test so the model can use them without changing the core GBM logic. I also increased the boosting capacity (more trees and deeper depth) to let the model learn the richer feature set, which should lower the MAE and move the score toward the target while keeping the original pipeline intact. The script now follows the required cell numbering and still writes a valid `submission.csv`.'
- What this solution (achieved 1.73447) has done: 'I add a few more informative features (cumulative u_in and previous u_in per breath) and modestly increase the HistGradientBoostingRegressor capacity (more trees, deeper depth, smaller learning‑rate) to let the model exploit the richer signal, then keep the same nearest‑pressure rounding and submission logic. These changes are small, preserve the core pipeline, and are expected to lower the MAE toward the target value.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import os
import random
from random import random as rd
import gc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor  # faster GBM
from sklearn.metrics import mean_absolute_error



## === cell 1
dtypes = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
df_train = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv", dtype=dtypes
)
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv", dtype=dtypes)

df_train["RC"] = df_train["R"].astype(np.float32) * df_train["C"].astype(np.float32)
df_test["RC"] = df_test["R"].astype(np.float32) * df_test["C"].astype(np.float32)

df_train["u_in_sq"] = df_train["u_in"].astype(np.float32) ** 2
df_test["u_in_sq"] = df_test["u_in"].astype(np.float32) ** 2

df_train["u_in_cumsum"] = (
    df_train.groupby("breath_id")["u_in"].transform("cumsum").astype(np.float32)
)
df_test["u_in_cumsum"] = (
    df_test.groupby("breath_id")["u_in"].transform("cumsum").astype(np.float32)
)

df_train["u_in_lag1"] = (
    df_train.groupby("breath_id")["u_in"].shift(1).fillna(0).astype(np.float32)
)
df_test["u_in_lag1"] = (
    df_test.groupby("breath_id")["u_in"].shift(1).fillna(0).astype(np.float32)
)

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures).astype(np.float32)
total_pressures_len = len(sorted_pressures)


def nearest_array(preds):
    """Vectorized nearest‑pressure lookup for a NumPy array."""
    preds = np.asarray(preds, dtype=np.float32)
    idx = np.searchsorted(sorted_pressures, preds, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)

    lower_idx = np.maximum(idx - 1, 0)

    lower_val = sorted_pressures[lower_idx]
    upper_val = sorted_pressures[idx]

    lower_diff = np.abs(preds - lower_val)
    upper_diff = np.abs(upper_val - preds)

    choose_upper = upper_diff < lower_diff
    return np.where(choose_upper, upper_val, lower_val)


def set_seed(seed=2021):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)




## === cell 2
def wc(input_list):
    l = []
    for i in range(len(input_list)):
        public_lb_score = int(input_list[i].split("/")[-1].split(".")[1].split(" ")[0])
        l.append(public_lb_score)
        input_list[i] = (pd.read_csv(input_list[i]).pressure).ravel()
    output = 0
    l_sum = sum(l)
    if len(input_list) == 1:
        output = input_list[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def blend(a_path, b_path):
    """Blend two existing submission files if they exist; otherwise noop."""
    if os.path.exists(a_path) and os.path.exists(b_path):
        a = pd.read_csv(a_path, dtype={"id": "int32", "pressure": "float32"})
        b = pd.read_csv(b_path, dtype={"id": "int32", "pressure": "float32"})
        a.pressure = a.pressure * 0.65 + b.pressure * 0.35
        a["pressure"] = nearest_array(a.pressure.values)
        a.to_csv("blend.csv", index=False)
        return a
    else:
        print("Blend files not found; skipping blending.")
        return None




## === cell 3
def train_and_predict():
    set_seed(2021)

    feature_cols = [c for c in df_train.columns if c not in ["pressure", "id"]]
    X = df_train[feature_cols].astype(np.float32).values
    y = df_train["pressure"].astype(np.float32).values

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    model = HistGradientBoostingRegressor(
        max_iter=2000,
        learning_rate=0.01,
        max_depth=7,
        random_state=42,
    )
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_pred)
    print(f"Validation MAE: {val_mae:.5f}")

    test_pred = model.predict(df_test[feature_cols].astype(np.float32).values)
    test_pred_rounded = nearest_array(test_pred)

    submission = pd.DataFrame({"id": df_test["id"], "pressure": test_pred_rounded})
    submission_path = "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")

    del X, y, X_train, X_val, y_train, y_val, test_pred
    gc.collect()


train_and_predict()




## === cell 4
a_path = "../input/gb-data-blending-recover/0.1371 blend.csv"
b_path = "../input/gb-data-blending-recover/0.1377 blend.csv"
blend(a_path, b_path)
