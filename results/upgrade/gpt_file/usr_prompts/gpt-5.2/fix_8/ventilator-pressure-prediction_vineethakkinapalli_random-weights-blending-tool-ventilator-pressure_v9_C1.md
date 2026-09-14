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

0.1728020907036779

# 6. Current score

3.77559

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.02327) has done: 'I fix the crash in the prediction fill step by ensuring `pred` is a mutable NumPy array (right now it’s a pandas Index, which can’t be assigned into). I also make the mapping and fallback assignments consistent and robust by converting both primary and fallback mapped outputs to arrays before masking. Finally, I keep the same median-lookup core logic and pressure snapping via `find_nearest`, and ensure a valid `submission.csv` is written with the required `id,pressure` columns.'
- What this solution (achieved 4.00099) has done: 'Your current score is far worse than the target (MAE 6.02 vs 0.173), so we need a meaningful but still “same-core-logic” improvement. The main issue is that your lookup ignores `u_in`, which is the dominant control input; adding `u_in` (rounded) into the same median-mapping approach typically yields a large MAE drop while keeping the same lookup/median/snapping semantics. I keep everything else the same (groupby-median mapping, fallback mapping, global median fallback, and `find_nearest` pressure snapping), but extend the primary key to include `u_in` and add a second fallback that also includes `u_out` so missing rates stay low. This should move the score substantially toward the target without changing the overall method.'
- What this solution (achieved 5.23934) has done: 'Your current MAE (4.00099) is still far above the target (0.1728), so we need a small but meaningful improvement while keeping the same “groupby-median lookup + fallbacks + pressure snapping” core. The main gain comes from reducing key-collisions by using higher-resolution rounding for `time_step` and `u_in` in the primary lookup (still the same semantics: rounded key → median pressure). To keep coverage high, I also add a fallback that drops `u_out` but keeps `u_in` (so missing values can still be filled using the dominant control input). Everything else (median mappings, fallback chain, global median, and `find_nearest` snapping) remains unchanged, and it still writes a valid `submission.csv`.'
- What this solution (achieved 5.23913) has done: 'Your current MAE (5.239) is far worse than the target (0.173), so we should improve the *same lookup/median/snapping* approach by fixing the biggest semantic mismatch: the evaluation ignores expiratory phase (`u_out==1`), but your mapping currently learns/uses pressures from both phases. I keep the exact same groupby-median → fallback chain → global median → `find_nearest` snapping core, but (1) build the mappings only from inspiratory rows (`u_out==0`) and (2) for test rows with `u_out==1`, directly set predictions to a stable value (global inspiratory median) since they are not scored. These are minimal changes that usually reduce noise substantially and move MAE toward the target without changing the fundamental method or adding new modeling. The script still run end-to-end and write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 4.00096) has done: 'Your current lookup uses very fine rounding for `time_step`/`u_in`, which explodes the number of unique keys and causes most test rows to miss the mapping and fall back to the global median—this typically yields MAE around ~5. To move the score toward the target while keeping the exact same “groupby-median lookup → fallback chain → global median → snap-to-known-pressures” core, I coarsen the rounding just enough to greatly increase lookup hit-rate. I also keep the inspiratory-only mapping (correct for the metric) and keep the same expiratory handling (set to global inspiratory median since not scored). These are minimal parameter-level changes and should reduce MAE materially without changing the method.'
- What this solution (achieved 3.77559) has done: 'Your current MAE (4.00096) is still far above the target (0.1728), so we should increase lookup hit-rate while keeping the exact same “groupby-median mapping → fallback chain → global median → snap-to-known-pressures” approach. The minimal change most likely to help is to make the rounded keys *slightly coarser* (especially `u_in`), because your current `u_in` rounding (0.1) still creates many unique keys and forces many rows into weak fallbacks. I also ensure the mappings use the same dtypes/rounding on both train/test to avoid silent key mismatches (a common source of unnecessary misses). Everything else (inspiratory-only mapping, expiratory constant fill, nearest-pressure snapping, and submission writing) remains unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import glob
import random
from random import random as rd
import gc



## === cell 1
df_train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)


def find_nearest(prediction: float) -> float:
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        return float(sorted_pressures[-1])
    elif insert_idx == 0:
        return float(sorted_pressures[0])
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return float(
        lower_val
        if abs(lower_val - prediction) < abs(upper_val - prediction)
        else upper_val
    )


def set_seed(seed: int = 2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    l = []
    allow = [1348, 1359, 1758]
    for i in range(len(input_list)):
        public_lb_score = int(input_list[i].split("/")[-1].split(".")[1].split(" ")[0])
        if public_lb_score in allow:
            print(public_lb_score)
            l.append(public_lb_score)
            input_list[i] = (pd.read_csv(input_list[i]).pressure).ravel()
        else:
            continue
    output = 0
    l_sum = sum(l) if len(l) else 1
    if len(input_list) == 1:
        output = input_list[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def g(dp):
    l = []
    allow = [1348, 1359, 1758]
    for i in glob.iglob(f"{dp}/*"):
        file_lb = int(i.split("/")[-1].split(".")[1].split(" ")[0])
        if file_lb in allow:
            l.append(i)
        else:
            continue
    loop_time = 125
    splits = 2
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
    for j in range(loop_time):
        weight = []
        set_seed(j)
        for k in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        for k in range(len(weight)):
            weight[k] /= weight_sum
        weight.sort(reverse=True)
        temp = 0
        for k in range(len(flist)):
            temp += flist[k] * weight[k]
        pred_list.append(temp)
        del temp
        gc.collect()
    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output.pressure = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)




## === cell 2
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")



## === cell 3
for col in ["R", "C", "u_out"]:
    df_train[col] = df_train[col].astype(np.int16)
    df_test[col] = df_test[col].astype(np.int16)

df_train["time_step_r"] = df_train["time_step"].round(2).astype(np.float32)
df_test["time_step_r"] = df_test["time_step"].round(2).astype(np.float32)

df_train["u_in_r"] = df_train["u_in"].round(0).astype(np.float32)
df_test["u_in_r"] = df_test["u_in"].round(0).astype(np.float32)

df_train_insp = df_train[df_train["u_out"] == 0].copy()

map_primary = df_train_insp.groupby(
    ["R", "C", "time_step_r", "u_in_r", "u_out"], sort=False
)["pressure"].median()

map_fb_uin = df_train_insp.groupby(["R", "C", "time_step_r", "u_in_r"], sort=False)[
    "pressure"
].median()

map_fb_uout = df_train_insp.groupby(["R", "C", "time_step_r", "u_out"], sort=False)[
    "pressure"
].median()

map_fallback = df_train_insp.groupby(["R", "C", "time_step_r"], sort=False)[
    "pressure"
].median()

test_key = df_test.set_index(["R", "C", "time_step_r", "u_in_r", "u_out"], drop=False)

pred_primary = test_key.index.map(map_primary)
pred = np.asarray(pred_primary, dtype=np.float64)  # ensure mutable ndarray

missing = np.isnan(pred)
if missing.any():
    fb_uin_idx = (
        df_test.loc[missing, ["R", "C", "time_step_r", "u_in_r"]]
        .set_index(["R", "C", "time_step_r", "u_in_r"])
        .index
    )
    pred_fb_uin = np.asarray(fb_uin_idx.map(map_fb_uin), dtype=np.float64)
    pred[missing] = pred_fb_uin

missing = np.isnan(pred)
if missing.any():
    fb1_idx = (
        df_test.loc[missing, ["R", "C", "time_step_r", "u_out"]]
        .set_index(["R", "C", "time_step_r", "u_out"])
        .index
    )
    pred_fb1 = np.asarray(fb1_idx.map(map_fb_uout), dtype=np.float64)
    pred[missing] = pred_fb1

missing = np.isnan(pred)
if missing.any():
    fb2_idx = (
        df_test.loc[missing, ["R", "C", "time_step_r"]]
        .set_index(["R", "C", "time_step_r"])
        .index
    )
    pred_fb2 = np.asarray(fb2_idx.map(map_fallback), dtype=np.float64)
    pred[missing] = pred_fb2

global_med_insp = float(df_train_insp["pressure"].median())
pred = np.where(np.isnan(pred), global_med_insp, pred).astype(np.float64)

pred[df_test["u_out"].to_numpy() == 1] = global_med_insp

pred = np.array([find_nearest(x) for x in pred], dtype=np.float32)



## === cell 4
submission = df_test[["id"]].copy()
submission["pressure"] = pred
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
