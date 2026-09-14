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

0.1471889433807456

# 6. Current score

4.09059

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65244) has done: 'I adjust the blending function so that the median of the generated predictions is always expanded to match the required number of rows (603 600). This prevents the “Length of values (1) does not match length of index” error when the predicted array is a single value. The change is limited to the assignment of `output.pressure` and preserves all existing logic.'
- What this solution (achieved 8.13673) has done: 'The update adds a lightweight baseline predictor that uses the average pressure for each combination of lung attributes (`R`, `C`) and time step from the training data.  
If no external prediction files are found, the script now falls back to this baseline, merges the predictions with the sample‑submission template, snaps them to the nearest observed pressure, and writes a valid CSV. This change keeps the original blending logic untouched while providing much more realistic predictions, moving the MAE dramatically closer to the target score.'
- What this solution (achieved 7.7927) has done: 'We tighten the baseline by grouping on the additional control inputs (`u_in`, `u_out`) and a rounded `time_step` to capture finer dynamics, which should markedly lower MAE while keeping the overall workflow unchanged. The fallback branch now writes the predictions to the required `submission.csv` file, ensuring a valid Kaggle‑ready output. All other logic, including blending and random‑weight ensembles, remains intact.'
- What this solution (achieved 4.22099) has done: 'I add a lightweight Scikit‑Learn model that is trained on a random subset of the training data when no external predictions are found. This replaces the very coarse “global mean per rounded time step” baseline with a more expressive RandomForest that uses the original numeric features. The model’s predictions are still snapped to the nearest observed pressure to keep the original post‑processing logic, and the script continues to write a valid `submission.csv`. This change is minimal, respects the existing workflow, and is expected to lower the MAE substantially toward the target.'
- What this solution (achieved 7.46833) has done: 'I replace the very coarse RandomForest fallback with a deterministic “nearest‑group” baseline that averages the training pressure for each rounded combination of the important features (`R`, `C`, `u_in`, `u_out`, `time_step`). The test rows are mapped to these groups (or to the overall mean if unseen) and then snapped to the nearest observed pressure, exactly as the original script does. This change preserves the overall workflow while dramatically lowering the MAE, moving the score much closer to the target.'
- What this solution (achieved 4.07644) has done: 'I replace the simple deterministic baseline (used when no external prediction files are found) with a lightweight RandomForest model trained on a random subset of the training data. This model uses the original numeric features, produces more accurate pressure estimates, and the predictions are still snapped to the nearest observed pressure to keep the original post‑processing unchanged. The change is minimal, respects the existing workflow, and is expected to lower the MAE toward the target score.'
- What this solution (achieved 7.46833) has done: 'I replace the fallback RandomForest model with the already‑computed deterministic group‑mean lookup (based on rounded `time_step`, `u_in` and the lung attributes `R`, `C`, `u_out`). This uses the same features the script groups on, gives much more accurate pressure estimates than the small RandomForest trained on a subset, and keeps all later post‑processing (nearest‑pressure snapping and CSV writing) unchanged, thereby moving the MAE dramatically closer to the target.'
- What this solution (achieved 3.93255) has done: 'The update adds a lightweight RandomForest model that is trained on a small random subset of the training data and is used as the fallback predictor when no external prediction files are found. This model leverages the key numeric features (R, C, u_in, u_out, time_step) to produce more accurate pressure estimates before snapping them to the nearest observed pressure, thereby moving the MAE much closer to the target while preserving the original workflow.'
- What this solution (achieved 3.91376) has done: 'I slightly increase the amount of training data and the power of the RandomForest used as the fallback model (sample 15 % of the rows, 500 trees, no depth limit). This keeps the original workflow unchanged while giving the model more capacity, which should lower the MAE and move the score toward the target. No other logic is altered.'
- What this solution (achieved 4.09059) has done: 'I replace the RandomForest fallback with a more powerful yet still lightweight HistGradientBoostingRegressor and increase the training sample to 30 % of the data. This change keeps the overall workflow identical while delivering more accurate predictions, which should lower the MAE and move the score toward the target.'

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

from sklearn.ensemble import HistGradientBoostingRegressor  # new model




## === cell 1
df_train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
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


df_train["time_round"] = df_train["time_step"].round(4)
df_train["u_in_round"] = df_train["u_in"].round(2)

group_key = ["R", "C", "u_in_round", "u_out", "time_round"]
pressure_lookup = (
    df_train.groupby(group_key)["pressure"]
    .mean()
    .reset_index()
    .set_index(group_key)["pressure"]
)

global_mean_pressure = df_train["pressure"].mean()

fallback_model = None


def train_fallback_model():
    global fallback_model
    sample_frac = 0.30
    df_sample = df_train.sample(frac=sample_frac, random_state=42)
    X = df_sample[["R", "C", "u_in", "u_out", "time_step"]]
    y = df_sample["pressure"]
    hgb = HistGradientBoostingRegressor(
        max_iter=500,  # number of boosting iterations
        learning_rate=0.05,  # smaller learning rate for stable convergence
        max_depth=None,  # unlimited depth (leaf-wise growth)
        random_state=42,
    )
    hgb.fit(X, y)
    fallback_model = hgb


train_fallback_model()


def g(dp):
    file_list = list(glob.iglob(f"{dp}/*"))
    if len(file_list) == 0:
        test_df = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
        test_df["time_round"] = test_df["time_step"].round(4)
        test_df["u_in_round"] = test_df["u_in"].round(2)

        if fallback_model is not None:
            test_features = test_df[["R", "C", "u_in", "u_out", "time_step"]]
            test_pred = fallback_model.predict(test_features)
        else:
            merged = test_df.merge(
                pressure_lookup.reset_index(),
                on=group_key,
                how="left",
            )
            merged["pressure"] = merged["pressure"].fillna(global_mean_pressure)
            test_pred = merged["pressure"].values

        test_pred = np.vectorize(find_nearest)(test_pred)

        output = pd.read_csv(
            "../input/ventilator-pressure-prediction/sample_submission.csv"
        )
        output["pressure"] = test_pred
        output.to_csv("submission.csv", index=False)
        print(
            "No external predictions found – HistGradientBoosting fallback CSV created."
        )
        return

    l = []
    for i in range(len(file_list)):
        public_lb_score = int(file_list[i].split("/")[-1].split(".")[1].split(" ")[0])
        l.append(public_lb_score)
        file_list[i] = (pd.read_csv(file_list[i]).pressure).ravel()
    output = 0
    l_sum = sum(l)
    if len(file_list) == 1:
        output = file_list[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        output += file_list[0] * weight1 + file_list[1] * weight2

    flist = []
    file_count = len(file_list)
    loop_time = 150
    splits = file_count // 2
    l.sort()
    for i in range(splits):
        if i == splits - 1:
            flist.append(file_list[i * round(len(file_list) / splits) :])
        else:
            flist.append(
                file_list[
                    i
                    * round(len(file_list) / splits) : (i + 1)
                    * round(len(file_list) / splits)
                ]
            )
    for i in range(len(flist)):
        flist[i] = wc(flist[i])
    pred_list = []
    for i in range(loop_time):
        weight = []
        set_seed(i)
        for _ in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        weight = [w / weight_sum for w in weight]
        weight.sort(reverse=True)
        temp = sum(flist[j] * weight[j] for j in range(len(flist)))
        pred_list.append(temp)
        del temp
        gc_collect()
    output_df = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    median_pred = np.median(np.vstack(pred_list), axis=0)
    if median_pred.ndim == 0 or median_pred.shape[0] != len(output_df):
        median_pred = np.full(len(output_df), float(median_pred))
    output_df.pressure = median_pred
    output_df["pressure"] = output_df["pressure"].apply(find_nearest)
    output_df.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.5 + b.pressure * 0.5
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
g("../input/gb-blending")
