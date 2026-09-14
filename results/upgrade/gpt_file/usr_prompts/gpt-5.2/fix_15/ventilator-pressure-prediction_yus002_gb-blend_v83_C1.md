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

0.154215935886166

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'The crash happens because the notebook tries to read two external submission files from a non-existent Kaggle dataset (`../input/gb-blending/...`). To keep the core logic intact while making it run end-to-end, I modify the final cell to instead produce a valid submission using the provided `g(dp)` ensembling function over any CSVs found in a local directory, and fall back safely to `sample_submission.csv` if none are available. I also fix the cell numbering to start at 1 (as required) and write the final output to `submission.csv` with the required `id,pressure` columns. These changes are execution/stability fixes; they don’t change the underlying ensembling/rounding semantics.'
- What this solution (achieved 7.32063) has done: 'Your current score is very far from the target (17.65 vs 0.154, lower is better), and the main reason is that the pipeline does not actually model the ventilator dynamics—it often falls back to `sample_submission` (all zeros), which yields a terrible MAE. To move toward the target with minimal core-logic change, I keep your “snap predictions to the nearest valid pressure” post-processing exactly as-is, but replace the missing external blend inputs with a simple, fast, legitimate baseline model trained from `train.csv`. Specifically, I fit a per-(R,C,time_step) median pressure table on the inspiratory phase (u_out==0) and use it to predict test rows (falling back to a global inspiratory median when a key is unseen), then apply `find_nearest` and write `submission.csv`.'
- What this solution (achieved 7.23024) has done: 'Your current score (7.32063, lower-is-better) is still far above the target (0.1542), so we should improve accuracy while keeping your core approach intact. I keep your existing “snap to nearest valid pressure” post-processing unchanged, but strengthen the baseline mapping with minimal logic change: add a slightly richer lookup key that includes `u_in` binned (captures valve opening effect) and a fallback hierarchy (full-key median → (R,C,time_step) median → (R,C) median → global inspiratory median). This remains a simple median-table model (no new model class/training loop), should materially reduce MAE, and stays within time limits by using vectorized groupbys and merges. The output still be a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 4.07547) has done: 'Your current score (7.23024, lower-is-better) is still far above the target (0.1542), so we should improve accuracy while keeping the same core “median lookup table + nearest-pressure snapping” approach. The biggest issue is that the lookup key uses raw `time_step` floats, which causes massive key mismatches between train and test and forces frequent fallback to coarse medians. I minimally fix this by quantizing `time_step` to a consistent grid (the dataset uses ~0.03s steps) and using that quantized value in the groupby/merge keys, while preserving your exact post-processing (`find_nearest`). This should materially reduce MAE without changing the overall modeling semantics or adding any new model/training loop.'
- What this solution (achieved 4.0055) has done: 'You’re still far above the target (4.075 vs 0.154, lower-is-better), so we should improve accuracy while keeping the same “median lookup table + nearest-pressure snapping” core logic. The main remaining error source is that `u_in` is only coarsely binned, which loses a lot of signal; we can keep the exact same table/merge/fallback approach but make `u_in` binning finer to better match the true dynamics. I also switch the time-step quantization to integer “tick” indices (derived from the same 0.03 grid) to avoid float merge edge-cases, which can silently increase fallback usage. These are minimal feature-key changes only; the modeling approach, inspiratory-only training, fallback hierarchy, and `find_nearest` post-processing remain identical.'
- What this solution (achieved 4.14362) has done: 'Your current score (4.0055, lower-is-better) is still far above the target (0.1542), so we should improve accuracy with the smallest possible change while keeping the same “median lookup table + fallback hierarchy + find_nearest snapping” core logic. The biggest remaining loss is that rounding `u_in` into 0.5 bins still causes many near-miss keys; switching to a finer `u_in` tick (0.1) preserves the exact same approach but reduces fallback usage and better matches the control signal granularity. I also make the time tick explicitly computed once and kept as small ints to avoid merge mismatches (same semantics as you already do), and keep inspiratory-only training unchanged. The output remains a valid `submission.csv` with the required `id,pressure` columns.'
- What this solution (achieved 5.9648) has done: 'Your current MAE (4.14362, lower-is-better) is still far above the target (0.1542), so we should improve accuracy while keeping your exact “median lookup table + fallback hierarchy + find_nearest snapping” approach intact. The most impactful minimal fix is to stop binning `u_in` (which causes many avoidable key misses) and instead use the exact `u_in` values as the join key; in this dataset, train/test share the same discrete `u_in` grid, so this greatly increases full-key matches without changing the modeling method. To keep merges stable and fast, we also quantize `time_step` to an integer tick (as you already do) and ensure keys use consistent dtypes across train/test. Everything else (inspiratory-only training, medians, fallback order, and nearest-pressure snapping) remains unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 4.15159) has done: 'Your current score (5.9648 MAE, lower-is-better) is still far above the target (0.1542), so we should improve accuracy without changing your core “median lookup table + fallback hierarchy + find_nearest snapping” approach. The main weakness is key mismatch from using `u_in` as a float join key (dtype/representation differences can silently break merges and force fallbacks), so I switch to a stable integer `u_in_tick` key derived by scaling `u_in` to a fixed grid. To further reduce fallbacks with minimal semantic change, I add one extra intermediate fallback level `(R,C,u_in_tick)` between full-key and `(R,C,time_tick)`. Everything else (inspiratory-only training, medians, fallback style, snapping, and writing `submission.csv`) remains the same.'
- What this solution (achieved 3.8406) has done: 'Your current MAE (4.15159, lower-is-better) is still far above the target (0.1542), so we should improve accuracy while keeping the exact same “median lookup table + fallback hierarchy + find_nearest snapping” core logic. The biggest remaining limitation is that the mapping does not condition on breath dynamics (history), so many different states collide into the same (R,C,time,u_in) bucket; a minimal way to add dynamics without changing the approach is to add a single extra key feature: the previous inspiratory `u_in` (as an integer tick), and use it only as an additional highest-priority median table. We keep inspiratory-only training, keep all existing tables/fallbacks intact, and just add one new table and merge/fallback step. This typically reduces ambiguity at the same time_tick and improves MAE, while still being a pure groupby-median lookup with deterministic post-processing and a valid `submission.csv`.'
- What this solution (achieved 3.8406) has done: 'Your current score (3.8406 MAE, lower-is-better) is still far above the target (0.1542), so we should improve accuracy while keeping your exact “groupby-median lookup table + fallback hierarchy + find_nearest snapping” core logic intact. The most impactful minimal fix is to make the extra “previous u_in” feature consistent with inspiratory-only dynamics: for test rows where `u_out==1` (expiratory, not scored), we should reset `u_in_tick_prev` to 0 so those rows don’t pollute the merge keys and force unnecessary fallbacks (even though they’re unscored, their predictions still exist in the submission). In addition, we should include `u_out` in the highest-priority lookup tables so inspiratory rows match inspiratory-trained medians while expiratory rows don’t collide with inspiratory patterns at the same (R,C,time,u_in). This preserves the same modeling semantics (median tables + fallbacks + snapping) and should reduce key collisions/mismatches for the scored inspiratory phase without changing architecture/training loops.'
- What this solution (achieved 3.8406) has done: 'Your current MAE (3.8406, lower-is-better) is still far above the target (0.1542), so we should make a small, legitimate accuracy improvement without changing your core “groupby-median lookup + fallback hierarchy + find_nearest snapping” approach. The biggest remaining mismatch is that you train all medians on inspiratory-only rows (`u_out==0`) but then include `u_out` in every merge key; for test expiratory rows (`u_out==1`) this guarantees key misses and forces coarse/global fallbacks, and it can also harm inspiratory predictions via unnecessary table sparsity. I keep `u_out` in the features (so behavior is still aware of phase), but compute inspiratory medians with `u_out` removed from the keys (since it is constant in training) and explicitly force expiratory predictions to a stable baseline (global inspiratory median) because expiratory rows are not scored. This preserves your exact post-processing and fallback semantics for the scored inspiratory phase, while reducing avoidable key-miss noise and should move MAE closer to the target.'
- What this solution (achieved 3.79696) has done: 'Your current MAE (3.8406, lower-is-better) is still far above the target (0.1542), so we should improve accuracy while keeping your exact “groupby-median lookup + fallback hierarchy + find_nearest snapping” core logic intact. The smallest high-impact change is to add one more dynamics-aware key using the previous predicted/observed pressure level (quantized to the known discrete pressure grid), which better captures state without introducing a new model or training loop. Concretely: we create a `p_prev` feature from the previous time-step pressure (train) and from the previous time-step’s baseline prediction (test), build one extra median table keyed on `(R,C,time_tick,u_in_tick,p_prev)`, and insert it at the top of your existing fallback chain. Everything else (inspiratory-only training, existing tables, expiratory handling, and nearest-pressure snapping) remains the same, and it still writes a valid `submission.csv`.'
- What this solution (achieved 3.71816) has done: 'Your current score (3.79696 MAE, lower-is-better) is still far above the target (0.1542), so we should improve accuracy while keeping the exact same “groupby-median lookup + fallback hierarchy + nearest-pressure snapping” approach. The main issue is that the new `p_prev` table is trained using the *true* previous pressure (train) but applied using a *predicted* previous pressure (test), which creates a train/test feature mismatch and can degrade the top-priority lookup rather than help it. I make `p_prev` consistent by defining it from the same baseline lookup (`pressure_base`) in both train and test, and then build the `(R,C,time_tick,u_in_tick,p_prev)` median table on that consistent `p_prev`. This is a minimal change (no new model, no new loss/loop) and should reduce MAE by making the highest-priority key actually match between train and test.'

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


def g(dp):
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)
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


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.5 + b.pressure * 0.5
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")

train_insp = df_train[df_train["u_out"] == 0].copy()

TIME_BIN = 0.03

train_insp["time_tick"] = np.rint(train_insp["time_step"] / TIME_BIN).astype(np.int16)
df_test_feat = df_test.copy()
df_test_feat["time_tick"] = np.rint(df_test_feat["time_step"] / TIME_BIN).astype(
    np.int16
)

train_insp["R"] = train_insp["R"].astype(np.int16)
train_insp["C"] = train_insp["C"].astype(np.int16)
df_test_feat["R"] = df_test_feat["R"].astype(np.int16)
df_test_feat["C"] = df_test_feat["C"].astype(np.int16)

train_insp["u_out"] = train_insp["u_out"].astype(np.int8)
df_test_feat["u_out"] = df_test_feat["u_out"].astype(np.int8)

UIN_SCALE = 10.0
train_insp["u_in_tick"] = np.rint(train_insp["u_in"] * UIN_SCALE).astype(np.int16)
df_test_feat["u_in_tick"] = np.rint(df_test_feat["u_in"] * UIN_SCALE).astype(np.int16)

train_insp = train_insp.sort_values(["breath_id", "time_step"], kind="mergesort")
df_test_feat = df_test_feat.sort_values(["breath_id", "time_step"], kind="mergesort")

train_insp["u_in_tick_prev"] = (
    train_insp.groupby("breath_id", sort=False)["u_in_tick"]
    .shift(1)
    .fillna(0)
    .astype(np.int16)
)
df_test_feat["u_in_tick_prev"] = (
    df_test_feat.groupby("breath_id", sort=False)["u_in_tick"]
    .shift(1)
    .fillna(0)
    .astype(np.int16)
)

df_test_feat.loc[df_test_feat["u_out"] == 1, "u_in_tick_prev"] = np.int16(0)

key_full = ["R", "C", "time_tick", "u_in_tick"]
key_full_prev = ["R", "C", "time_tick", "u_in_tick", "u_in_tick_prev"]
key_rcu = ["R", "C", "u_in_tick"]
key_t = ["R", "C", "time_tick"]
key_rc = ["R", "C"]

median_full_prev = (
    train_insp.groupby(key_full_prev, sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "p_full_prev"})
)
median_full = (
    train_insp.groupby(key_full, sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "p_full"})
)
median_rcu = (
    train_insp.groupby(key_rcu, sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "p_rcu"})
)
median_t = (
    train_insp.groupby(key_t, sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "p_t"})
)
median_rc = (
    train_insp.groupby(key_rc, sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "p_rc"})
)

global_median_insp = float(train_insp["pressure"].median())

train_base0 = train_insp[["breath_id", "time_step"] + key_full].merge(
    median_full, on=key_full, how="left"
)
train_base0 = train_base0.merge(median_rcu, on=key_rcu, how="left")
train_base0 = train_base0.merge(median_t, on=key_t, how="left")
train_base0 = train_base0.merge(median_rc, on=key_rc, how="left")
train_base0["pressure_base0"] = (
    train_base0["p_full"]
    .fillna(train_base0["p_rcu"])
    .fillna(train_base0["p_t"])
    .fillna(train_base0["p_rc"])
    .fillna(global_median_insp)
    .astype(np.float64)
)

train_base0 = train_base0.sort_values(["breath_id", "time_step"], kind="mergesort")
train_base0["p_prev"] = (
    train_base0.groupby("breath_id", sort=False)["pressure_base0"]
    .shift(1)
    .fillna(global_median_insp)
)
train_base0["p_prev"] = train_base0["p_prev"].apply(find_nearest).astype(np.float32)

train_insp = train_insp.merge(
    train_base0[["breath_id", "time_step", "p_prev"]],
    on=["breath_id", "time_step"],
    how="left",
)

key_full_p = ["R", "C", "time_tick", "u_in_tick", "p_prev"]
median_full_p = (
    train_insp.groupby(key_full_p, sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "p_full_p"})
)

base0 = df_test_feat[["id", "breath_id", "time_step", "u_out"] + key_full].merge(
    median_full, on=key_full, how="left"
)
base0 = base0.merge(median_rcu, on=key_rcu, how="left")
base0 = base0.merge(median_t, on=key_t, how="left")
base0 = base0.merge(median_rc, on=key_rc, how="left")
base0["pressure_base0"] = (
    base0["p_full"]
    .fillna(base0["p_rcu"])
    .fillna(base0["p_t"])
    .fillna(base0["p_rc"])
    .fillna(global_median_insp)
    .astype(np.float64)
)
base0.loc[base0["u_out"] == 1, "pressure_base0"] = global_median_insp

base0 = base0.sort_values(["breath_id", "time_step"], kind="mergesort")
base0["p_prev"] = (
    base0.groupby("breath_id", sort=False)["pressure_base0"]
    .shift(1)
    .fillna(global_median_insp)
)
base0["p_prev"] = base0["p_prev"].apply(find_nearest).astype(np.float32)
base0.loc[base0["u_out"] == 1, "p_prev"] = np.float32(find_nearest(global_median_insp))

pred = base0[["id", "breath_id", "time_step", "u_out", "p_prev"] + key_full_prev].merge(
    median_full_prev, on=key_full_prev, how="left"
)
pred = pred.merge(median_full, on=key_full, how="left")
pred = pred.merge(median_rcu, on=key_rcu, how="left")
pred = pred.merge(median_t, on=key_t, how="left")
pred = pred.merge(median_rc, on=key_rc, how="left")
pred = pred.merge(median_full_p, on=key_full_p, how="left")

pred["pressure"] = (
    pred["p_full_p"]
    .fillna(pred["p_full_prev"])
    .fillna(pred["p_full"])
    .fillna(pred["p_rcu"])
    .fillna(pred["p_t"])
    .fillna(pred["p_rc"])
    .fillna(global_median_insp)
    .astype(np.float64)
)

pred.loc[pred["u_out"] == 1, "pressure"] = global_median_insp
pred["pressure"] = pred["pressure"].apply(find_nearest)

sub = pred[["id", "pressure"]].copy()
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2204914929.py in <cell line: 0>()
    148 
    149 # Now compute the main prediction table merges (including full_prev as a fallback signal).
--> 150 pred = base0[["id", "breath_id", "time_step", "u_out", "p_prev"] + key_full_prev].merge(
    151     median_full_prev, on=key_full_prev, how="left"
    152 )

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['u_in_tick_prev'] not in index"
