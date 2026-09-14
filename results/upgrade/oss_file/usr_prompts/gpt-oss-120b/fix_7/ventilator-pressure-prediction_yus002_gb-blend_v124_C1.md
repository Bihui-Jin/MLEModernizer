# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.1369925684672309

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 8.43315) has done: 'We fix the runtime error caused by an empty prediction list (the directory `gb-data-blending-recover` doesn’t contain any files). The fix adds a simple fallback that uses the overall mean pressure from the training set when no external predictions are available, ensuring the output length matches the submission template and a valid `.csv` file is written.'
- What this solution (achieved 6.45436) has done: 'We add a cheap baseline that predicts the mean pressure for each combination of lung attributes and the rounded inspiratory‑valve value, so when no external blend files exist the script uses this informed estimate instead of the overall training mean. This small change keeps the original logic unchanged but should lower the MAE toward the target.'
- What this solution (achieved 6.85183) has done: 'I add a simple linear‑regression model (using the existing features) that is trained once on the full training data and used in the fallback path when no external blend files exist. The model’s predictions are combined with the previous “group‑by mean” baseline (averaging them) to give a more informed estimate while keeping the original blending logic unchanged. This small, targeted change is expected to lower the MAE substantially and move the score toward the target.'
- What this solution (achieved 6.8519) has done: 'We stop rounding predictions to the nearest training‑pressure value, because that artificial quantisation adds error to the MAE. By keeping the raw model / baseline predictions the score moves toward the lower target while preserving the original blending logic.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import copy
import glob
import random
from random import random as rd
import gc
from sklearn.linear_model import LinearRegression  # existing model
from sklearn.ensemble import GradientBoostingRegressor  # added stronger model
from joblib import dump, load  # for model caching

df_train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
TRAIN_MEAN_PRESSURE = df_train["pressure"].mean()

df_train["u_in_int"] = df_train["u_in"].round().astype(int)
df_train["cum_u_in"] = df_train.groupby("breath_id")["u_in"].cumsum()
df_train["u_in_shift"] = df_train.groupby("breath_id")["u_in"].shift(1).fillna(0)

lr_features = ["u_in", "u_out", "R", "C", "time_step"]
gbr_features = ["u_in", "u_out", "R", "C", "time_step", "cum_u_in", "u_in_shift"]
df_train[lr_features + gbr_features] = df_train[lr_features + gbr_features].astype(
    np.float32
)

BASELINE_MEANS = (
    df_train.groupby(["R", "C", "u_in_int", "u_out"])["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "base_pred"})
)

LR_MODEL_PATH = "linear_model.pkl"
GBR_MODEL_PATH = "gbr_model.pkl"

if os.path.exists(LR_MODEL_PATH) and os.path.exists(GBR_MODEL_PATH):
    linear_model = load(LR_MODEL_PATH)
    gbr_model = load(GBR_MODEL_PATH)
else:
    lr_X = df_train[lr_features]
    lr_y = df_train["pressure"]
    linear_model = LinearRegression()
    linear_model.fit(lr_X, lr_y)

    gbr_X = df_train[gbr_features]
    gbr_y = df_train["pressure"]
    gbr_model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.1, max_depth=5, random_state=2021
    )
    gbr_model.fit(gbr_X, gbr_y)

    dump(linear_model, LR_MODEL_PATH)
    dump(gbr_model, GBR_MODEL_PATH)

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)


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


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


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
    file_paths = [p for p in glob.iglob(f"{dp}/*")]
    file_count = len(file_paths)
    loop_time = 154
    splits = file_count // 2 if file_count >= 2 else 0
    file_paths.sort()
    flist = []
    for i in range(splits):
        if i == splits - 1:
            flist.append(file_paths[i * round(len(file_paths) / splits) :])
        else:
            flist.append(
                file_paths[
                    i
                    * round(len(file_paths) / splits) : (i + 1)
                    * round(len(file_paths) / splits)
                ]
            )
    for i in range(len(flist)):
        flist[i] = wc(flist[i])

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )

    if not flist:
        df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
        df_test["u_in_int"] = df_test["u_in"].round().astype(int)

        df_test["cum_u_in"] = df_test.groupby("breath_id")["u_in"].cumsum()
        df_test["u_in_shift"] = df_test.groupby("breath_id")["u_in"].shift(1).fillna(0)

        df_test[gbr_features] = df_test[gbr_features].astype(np.float32)

        merged = df_test.merge(
            BASELINE_MEANS, on=["R", "C", "u_in_int", "u_out"], how="left"
        )
        baseline_pred = merged["base_pred"].fillna(TRAIN_MEAN_PRESSURE).values

        model_pred = gbr_model.predict(df_test[gbr_features])

        pred = (baseline_pred + model_pred) / 2.0

        output.pressure = pred
        output.to_csv("fallback_submission.csv", index=False)
        return

    pred_list = []
    for i in range(loop_time):
        weight = []
        set_seed(i)
        for _ in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        if weight_sum == 0:
            weight = [1.0 / len(flist)] * len(flist)
        else:
            weight = [w / weight_sum for w in weight]
        weight.sort(reverse=True)

        temp = 0
        for idx in range(len(flist)):
            temp += flist[idx] * weight[idx]
        pred_list.append(temp)
        del temp
        gc.collect()

    median_pred = np.median(np.vstack(pred_list), axis=0)

    output.pressure = median_pred
    output.to_csv(f"rwb_{loop_time}_loops.csv", index=False)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1932534289.py in <cell line: 0>()
     25 lr_features = ["u_in", "u_out", "R", "C", "time_step"]
     26 gbr_features = ["u_in", "u_out", "R", "C", "time_step", "cum_u_in", "u_in_shift"]
---> 27 df_train[lr_features + gbr_features] = df_train[lr_features + gbr_features].astype(
     28     np.float32
     29 )

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4297             self._setitem_frame(key, value)
   4298         elif isinstance(key, (Series, np.ndarray, list, Index)):
-> 4299             self._setitem_array(key, value)
   4300         elif isinstance(value, DataFrame):
   4301             self._set_item_frame_value(key, value)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _setitem_array(self, key, value)
   4341                 check_key_length(self.columns, key, value)
   4342                 for k1, k2 in zip(key, value.columns):
-> 4343                     self[k1] = value[k2]
   4344 
   4345             elif not is_list_like(value):

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4299             self._setitem_array(key, value)
   4300         elif isinstance(value, DataFrame):
-> 4301             self._set_item_frame_value(key, value)
   4302         elif (
   4303             is_list_like(value)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _set_item_frame_value(self, key, value)
   4427             len_cols = 1 if is_scalar(cols) or isinstance(cols, tuple) else len(cols)
   4428             if len_cols != len(value.columns):
-> 4429                 raise ValueError("Columns must be same length as key")
   4430 
   4431             # align right-hand-side columns if self.columns

ValueError: Columns must be same length as key

## === cell 1
g("../input/gb-data-blending-recover")

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3591655586.py in <cell line: 0>()
----> 1 g("../input/gb-data-blending-recover")

NameError: name 'g' is not defined
