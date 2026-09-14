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
import numpy as np
import pandas as pd
import os
import gc
import random
from random import random as rd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
import concurrent.futures


def find_nearest(prediction):
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


def find_nearest_vec(predictions):
    """Vectorized version of find_nearest for an array of predictions."""
    insert_idx = np.searchsorted(sorted_pressures, predictions, side="left")
    insert_idx = np.clip(insert_idx, 0, total_pressures_len)
    lower_idx = np.maximum(insert_idx - 1, 0)
    upper_idx = np.minimum(insert_idx, total_pressures_len - 1)
    lower_val = sorted_pressures[lower_idx]
    upper_val = sorted_pressures[upper_idx]
    choose_lower = np.abs(lower_val - predictions) < np.abs(upper_val - predictions)
    return np.where(choose_lower, lower_val, upper_val)


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


train_usecols = ["R", "C", "time_step", "u_in", "u_out", "breath_id", "pressure"]
dtypes_train = {
    "R": "int8",
    "C": "int8",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "breath_id": "int32",
    "pressure": "float32",
}
df_train = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
    usecols=train_usecols,
    dtype=dtypes_train,
)

df_train["RC"] = df_train["R"] * df_train["C"]
df_train["u_in_sq"] = df_train["u_in"] ** 2
df_train["time_step_sq"] = df_train["time_step"] ** 2
df_train["u_in_time"] = df_train["u_in"] * df_train["time_step"]
df_train["u_out_u_in"] = df_train["u_out"] * df_train["u_in"]
df_train["u_out_time"] = df_train["u_out"] * df_train["time_step"]
df_train["C_div_R"] = df_train["C"] / df_train["R"]
df_train["breath_max_time"] = df_train.groupby("breath_id")["time_step"].transform(
    "max"
)
df_train["time_ratio"] = df_train["time_step"] / df_train["breath_max_time"]
df_train["cum_u_in"] = df_train.groupby("breath_id")["u_in"].cumsum()
df_train["u_in_diff"] = df_train.groupby("breath_id")["u_in"].diff().fillna(0)
df_train["u_out_diff"] = df_train.groupby("breath_id")["u_out"].diff().fillna(0)
df_train["time_step_diff"] = df_train.groupby("breath_id")["time_step"].diff().fillna(0)

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)




## === cell 1
feature_cols = [
    "R",
    "C",
    "time_step",
    "u_in",
    "u_out",
    "RC",
    "u_in_sq",
    "time_step_sq",
    "u_in_time",
    "u_out_u_in",
    "u_out_time",
    "C_div_R",
    "breath_max_time",
    "time_ratio",
    "cum_u_in",
    "u_in_diff",
    "u_out_diff",
    "time_step_diff",
]

X = df_train[feature_cols].astype(np.float32).values
y = df_train["pressure"].astype(np.float32).values

model_params = dict(
    max_depth=15,
    learning_rate=0.02,
    max_iter=2000,
    loss="absolute_error",
    random_state=2021,
)


def fit_model(rs):
    params = model_params.copy()
    params["random_state"] = rs
    m = HistGradientBoostingRegressor(**params)
    m.fit(X, y)
    return m


with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    future1 = executor.submit(fit_model, 2021)
    future2 = executor.submit(fit_model, 2022)
    model1 = future1.result()
    model2 = future2.result()

del df_train
gc.collect()

test_usecols = ["R", "C", "time_step", "u_in", "u_out", "breath_id"]
dtypes_test = {
    "R": "int8",
    "C": "int8",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "breath_id": "int32",
}
df_test = pd.read_csv(
    "../input/ventilator-pressure-prediction/test.csv",
    usecols=test_usecols,
    dtype=dtypes_test,
)

df_test["RC"] = df_test["R"] * df_test["C"]
df_test["u_in_sq"] = df_test["u_in"] ** 2
df_test["time_step_sq"] = df_test["time_step"] ** 2
df_test["u_in_time"] = df_test["u_in"] * df_test["time_step"]
df_test["u_out_u_in"] = df_test["u_out"] * df_test["u_in"]
df_test["u_out_time"] = df_test["u_out"] * df_test["time_step"]
df_test["C_div_R"] = df_test["C"] / df_test["R"]
df_test["breath_max_time"] = df_test.groupby("breath_id")["time_step"].transform("max")
df_test["time_ratio"] = df_test["time_step"] / df_test["breath_max_time"]
df_test["cum_u_in"] = df_test.groupby("breath_id")["u_in"].cumsum()
df_test["u_in_diff"] = df_test.groupby("breath_id")["u_in"].diff().fillna(0)
df_test["u_out_diff"] = df_test.groupby("breath_id")["u_out"].diff().fillna(0)
df_test["time_step_diff"] = df_test.groupby("breath_id")["time_step"].diff().fillna(0)

test_pred_raw = (
    model1.predict(df_test[feature_cols].astype(np.float32).values)
    + model2.predict(df_test[feature_cols].astype(np.float32).values)
) / 2
test_pred = find_nearest_vec(test_pred_raw)

submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)
submission["pressure"] = test_pred
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")




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


def g(dp):
    import glob

    l = [i for i in glob.iglob(f"{dp}/*")]
    file_count = len(l)
    loop_time = 150
    splits = file_count // 2
    l.sort()
    flist = []
    for i in range(splits):
        if i == splits - 1:
            flist.append(l[i * round(len(l) / splits) :])
        else:
            flist.append(
                l[i * round(len(l) / splits) : (i + 1) * round(len(l) / splits)]
            )
    for i in range(len(flist)):
        flist[i] = wc(flist[i])
    pred_list = []
    for i in range(loop_time):
        weight = []
        set_seed(i)
        for i in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        for i in range(len(weight)):
            weight[i] /= weight_sum
        weight.sort(reverse=True)
        temp = 0
        for i in range(len(flist)):
            temp += flist[i] * weight[i]
        pred_list.append(temp)
        del temp
        gc.collect()
    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output.pressure = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a_path, b_path):
    if not (os.path.exists(a_path) and os.path.exists(b_path)):
        print("Blend files not found; skipping blending step.")
        return None
    a = pd.read_csv(a_path)
    b = pd.read_csv(b_path)
    a.pressure = a.pressure * 0.55 + b.pressure * 0.45
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    print("Blend completed, saved to blend.csv")
    return a




## === cell 3
a_path = "../input/gb-blending/0.176.csv"
b_path = "../input/gb-blending/0.178 blend.csv"
blend(a_path, b_path)
