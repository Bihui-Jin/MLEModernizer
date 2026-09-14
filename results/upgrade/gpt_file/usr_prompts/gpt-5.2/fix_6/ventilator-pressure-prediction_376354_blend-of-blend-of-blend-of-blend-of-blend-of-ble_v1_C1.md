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

0.14918964911965

# 6. Current score

19.55027

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.86378) has done: 'I make the ensemble routine robust to missing Kaggle “input” datasets by (1) safely handling an empty folder (which currently causes the division-by-zero) and (2) removing hard dependencies on external submissions that aren’t present. To still produce a valid submission end-to-end, I add a minimal fallback that uses the training-set median pressure (snapped to the nearest valid pressure) for every test row if no ensemble files are found. I also fix the cell numbering to be sequential so the notebook/script executes cleanly, and ensure the final output is written to `submission.csv` with the required `id,pressure` columns.'
- What this solution (achieved 10.73825) has done: 'The timeout is dominated by the KNN fallback path: it loads the full 5.4M-row train CSV, builds per-(R,C) matrices, and then computes dense pairwise distances between test and train within each group, which is far too expensive. To preserve the intended “use precomputed submissions if present, otherwise fallback” core logic, I keep the same control flow but replace the fallback with a provably equivalent nearest-neighbor lookup in feature space using a hash-map over *exact* feature tuples; for this dataset, test rows have exact matches in train for the same (R,C,time_step,u_in,u_out), so the nearest neighbor is exactly the matching row without any distance computation. I also eliminate per-row `apply(find_nearest)` by vectorizing the nearest-pressure snapping with `np.searchsorted`, which preserves the same discretization semantics but runs in milliseconds. Finally, I reduce CSV read overhead via explicit dtypes and `usecols`, and avoid repeated groupby/iloc overhead by operating on NumPy arrays.'
- What this solution (achieved 15.59117) has done: 'Your current score is far worse than the target (lower-is-better), and the biggest reason is the fallback: it predicts a global median pressure for any test row not exactly seen in train, which yields very high MAE. To move the score sharply toward the target without changing the overall approach (ensemble-if-available, else fallback), I keep the same control flow but make the fallback more faithful to the problem by doing a per-breath forward-fill of the last known matched pressure and defaulting remaining unknowns to a physically reasonable low baseline (snapped to valid pressures). This keeps the same “exact-match lookup” core logic and discretization behavior, but greatly reduces error when only some timesteps match. I also make the key creation faster/more robust by using a pandas MultiIndex join instead of building huge Python `list(zip(...))`, which reduces runtime and memory risk while preserving semantics.'
- What this solution (achieved 19.55027) has done: 'I keep your overall “ensemble if external files exist, otherwise exact-tuple fallback” logic unchanged, but make the fallback much closer to the competition target by (1) including `breath_id` in the lookup key so we only borrow pressures from the same breath (this avoids mixing across different breaths at identical control inputs) and (2) using a within-breath time-step forward/back fill to reduce the number of baseline-filled points. I also ensure the join stays vectorized via a MultiIndex (no Python loops) so it still runs under the time limit. Finally, I keep the same pressure discretization (snap to nearest valid pressure) and keep writing a valid `submission.csv` with `id,pressure`.'

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



## === cell 1
BASE_INPUT = "/kaggle/input/ventilator-pressure-prediction"
train_path = os.path.join(BASE_INPUT, "train.csv")
test_path = os.path.join(BASE_INPUT, "test.csv")
sample_sub_path = os.path.join(BASE_INPUT, "sample_submission.csv")

_train_usecols = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
_train_dtypes = {
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
df_train = pd.read_csv(train_path, usecols=_train_usecols, dtype=_train_dtypes)

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures.astype(np.float32))
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


def find_nearest_vectorized(pred):
    pred = np.asarray(pred, dtype=np.float32)
    idx = np.searchsorted(sorted_pressures, pred, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)

    lower_idx = np.clip(idx - 1, 0, total_pressures_len - 1)
    upper = sorted_pressures[idx]
    lower = sorted_pressures[lower_idx]

    choose_lower = (idx > 0) & (np.abs(lower - pred) < np.abs(upper - pred))
    out = np.where(choose_lower, lower, upper)
    return out.astype(np.float32)


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


def _knn_fallback_submission():
    """
    Score-relevant, minimal change: keep exact-tuple lookup, but include breath_id in the key so we do NOT
    borrow pressures across different breaths. Then fill missing within each breath using ffill+bfill to
    reduce baseline usage. This should sharply reduce MAE vs the current baseline-heavy fallback.
    Runtime-safe: still fully vectorized with a MultiIndex reindex.
    """
    _test_usecols = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
    _test_dtypes = {
        "id": "int32",
        "breath_id": "int32",
        "R": "int16",
        "C": "int16",
        "time_step": "float32",
        "u_in": "float32",
        "u_out": "int8",
    }
    df_test = pd.read_csv(test_path, usecols=_test_usecols, dtype=_test_dtypes)
    sub = pd.read_csv(
        sample_sub_path,
        usecols=["id", "pressure"],
        dtype={"id": "int32", "pressure": "float32"},
    )

    key_cols = ["breath_id", "R", "C", "time_step", "u_in", "u_out"]

    tr_map = (
        df_train.groupby(key_cols, sort=False, observed=True)["pressure"]
        .mean()
        .astype(np.float32)
    )

    te_idx = pd.MultiIndex.from_frame(df_test[key_cols])
    pred = tr_map.reindex(te_idx).to_numpy(dtype=np.float32, copy=False)

    baseline = float(sorted_pressures[0])

    pred_series = pd.Series(pred)
    g = pred_series.groupby(df_test["breath_id"], sort=False)
    pred_series = g.ffill()
    pred_series = pred_series.groupby(df_test["breath_id"], sort=False).bfill()
    pred_series = pred_series.fillna(baseline)
    pred_filled = pred_series.to_numpy(dtype=np.float32, copy=False)

    sub["pressure"] = find_nearest_vectorized(pred_filled)
    sub.to_csv("submission.csv", index=False)
    return "submission.csv"


def g(dp):
    """
    Original intent: read multiple submission files from a Kaggle dataset folder and do random-weight ensembling.
    When dp has zero matching files, use the exact-match NN fallback.
    """
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)

    file_count = len(l)

    if file_count == 0:
        return _knn_fallback_submission()

    loop_time = 500 // file_count
    loop_time = max(loop_time, 1)

    splits = file_count // 2
    splits = max(splits, 1)

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

    output = pd.read_csv(sample_sub_path)
    output.pressure = np.median(np.vstack(pred_list), axis=0)

    output["pressure"] = find_nearest_vectorized(
        output["pressure"].to_numpy(copy=False)
    )

    rwb_path = f"rwb {loop_time} loops.csv"
    output.to_csv(rwb_path, index=False)
    output.to_csv("submission.csv", index=False)
    return "submission.csv"




## === cell 2
submission_path = g("/kaggle/input/gb-rwbt-files")
print("Wrote:", submission_path)



## === cell 3
base_sub_path = "submission.csv"
alt_path = "/kaggle/input/gb-vpp-to-infinity-and-beyond/submission.csv"

if os.path.exists(alt_path) and os.path.exists(base_sub_path):
    df_1 = pd.read_csv(alt_path)
    df_2 = pd.read_csv(base_sub_path)

    df_1 = df_1.sort_values("id").reset_index(drop=True)
    df_2 = df_2.sort_values("id").reset_index(drop=True)

    df_final = df_1.copy()
    df_final["pressure"] = np.mean(
        np.concatenate(
            [
                np.expand_dims(df_1["pressure"].values, axis=1),
                np.expand_dims(df_2["pressure"].values, axis=1),
            ],
            axis=1,
        ),
        axis=1,
    )
    df_final.to_csv("submission.csv", index=False)
else:
    df_check = pd.read_csv(base_sub_path)
    df_check[["id", "pressure"]].to_csv("submission.csv", index=False)

print("Final submission saved to submission.csv")
