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

0.1527035104306432

# 6. Current score

4.01564

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65244) has done: 'The crash is because the notebook tries to read two external “gb-blending” submission files that are not present in your environment. To keep the core logic intact while making it run end-to-end, I (1) add robust path detection for the Ventilator dataset, (2) replace the missing external blend inputs with a safe, local baseline submission built from the provided sample_submission (so a valid `.csv` is always produced), and (3) keep the existing `find_nearest` post-processing to preserve evaluation semantics. This yield a valid `submission.csv` file without changing any modeling/training logic (none exists here).'
- What this solution (achieved 6.11692) has done: 'Your current score is far worse than the target (MAE 17.65 vs 0.1527), and the main reason is that you’re effectively submitting all-zeros (then snapped to the minimum pressure), which is catastrophically bad. With minimal change and without introducing any new model/training, we can move much closer to the target by using a simple, legitimate baseline: predict the per-time-step median pressure from the training set grouped by (R, C, time_step index within breath), and fall back to a global median when unseen. We keep your existing `find_nearest` discretization so evaluation semantics stay consistent with your original code. This should drastically reduce MAE while staying within the constraints and still producing a valid `submission.csv` end-to-end.'
- What this solution (achieved 6.10817) has done: 'Your current MAE (6.11692) is still far above the target (0.1527), so we should legitimately improve predictions while keeping the same simple “table-lookup median + find_nearest snapping” core approach. The biggest gap is that the median table ignores the inspiratory/expiratory regime; since the metric scores only inspiratory (u_out==0), using only inspiratory rows to build the lookup table better matches evaluation and should reduce MAE substantially. To keep robustness, we add hierarchical fallbacks: (R,C,ts_idx) → (R,C,ts_idx,u_out) when available, then (R,C,ts_idx) inspiratory-only, then (R,C) inspiratory median, then global inspiratory median. We keep all file paths and the existing discretization (`find_nearest`) unchanged and still write a valid `submission.csv`.'
- What this solution (achieved 4.06947) has done: 'Your current MAE (6.108) is still far above the target (0.1527), so we should improve the same “median lookup table + find_nearest snapping” approach without changing the overall method. The biggest remaining miss is that the lookup doesn’t condition on the actual `u_in` control signal, which is the primary driver of pressure; adding a minimal `u_in`-binned median table (trained on inspiratory only) usually get much closer to the true curve. To keep it safe and minimal, we add a single additional hierarchical level: first try (R,C,ts_idx,u_out,u_in_bin), then fall back to your existing tables exactly as before. We keep paths, no training loops, and still write a valid `submission.csv`.'
- What this solution (achieved 4.95663) has done: 'I fix the submission merge bug that drops/renames the `pressure` column (causing the KeyError) by updating the code to overwrite `pressure` safely after merging with suffix handling. I also add a small integrity check to guarantee the final submission has exactly the required `id,pressure` columns and no missing predictions, without changing your core “median table lookup + find_nearest snapping” logic. Paths and feature engineering remain the same; this is score-neutral except for preventing accidental column mishandling. The script then run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 4.01237) has done: 'Your current MAE (4.9566) is still far above the target (0.1527), so we should legitimately improve predictions while keeping the same “median lookup table + find_nearest snapping” core logic. The minimal high-impact fix is to condition the lookup on the *actual* control signal `u_in` (not its difference `du_in`), because pressure is primarily driven by valve opening magnitude; we add a single `u_in`-binned median table at the top of the same hierarchical fallback chain. To keep runtime/memory safe and avoid changing the approach, we use a modest bin width and keep all your existing fallback tables unchanged. This should reduce MAE substantially while preserving your evaluation semantics and still writing a valid `submission.csv`.'
- What this solution (achieved 4.01237) has done: 'Your current MAE (4.012) is still far above the target (0.1527), so we should legitimately improve predictions while preserving the same median-lookup + nearest-pressure snapping core logic. The smallest high-impact change is to avoid training the top lookup tables on only inspiratory rows while also keying on `u_out` (which is constant 0 there), because that makes the table too sparse and forces many fallbacks; instead, build those `(R,C,ts_idx,u_out,bin)` tables on full training so both inspiratory and expiratory keys are learned. To better match the metric (which scores only inspiratory), we still keep inspiratory-only tables as higher-priority fallbacks for u_out==0 cases, and keep the rest of your fallback chain intact. This should reduce missing matches and lower MAE without changing model type, loss, or any training loop (there is none), and it still writes a valid `submission.csv`.'
- What this solution (achieved 4.01564) has done: 'Your current MAE (4.012) is still far above the target (0.1527), so we should improve predictions while keeping the same “median lookup table + find_nearest snapping” approach. The most direct minimal gain is to stop binning `u_in` and `du_in` by rounding to a coarse width (which collapses distinct control states), and instead key the lookup tables on the exact `u_in` and `du_in` values present in the data (they are already discretized enough in this dataset to match frequently). To avoid increasing misses, we keep your full existing fallback chain unchanged, just adding exact-key tables at the very top and only using them when they match. This preserves the same evaluation semantics and still produces a valid `submission.csv` end-to-end.'

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
def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these paths exist: {paths}")


TRAIN_PATH = _first_existing(
    [
        "../input/ventilator-pressure-prediction/train.csv",
        "/kaggle/input/ventilator-pressure-prediction/train.csv",
        "/kaggle/data/ventilator-pressure-prediction/train.csv",
        "/kaggle/input/train.csv",
        "/kaggle/data/train.csv",
    ]
)

TEST_PATH = _first_existing(
    [
        "../input/ventilator-pressure-prediction/test.csv",
        "/kaggle/input/ventilator-pressure-prediction/test.csv",
        "/kaggle/data/ventilator-pressure-prediction/test.csv",
        "/kaggle/input/test.csv",
        "/kaggle/data/test.csv",
    ]
)

SAMPLE_SUB_PATH = _first_existing(
    [
        "../input/ventilator-pressure-prediction/sample_submission.csv",
        "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv",
        "/kaggle/data/ventilator-pressure-prediction/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
)

df_train = pd.read_csv(TRAIN_PATH)

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
    output = pd.read_csv(SAMPLE_SUB_PATH)
    output.pressure = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b, out_path="blend.csv"):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.5 + b.pressure * 0.5
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv(out_path, index=False)
    return a




## === cell 2
df_test = pd.read_csv(TEST_PATH)

train_idx = df_train.groupby("breath_id").cumcount().astype(np.int16)
test_idx = df_test.groupby("breath_id").cumcount().astype(np.int16)

df_train = df_train.copy()
df_test = df_test.copy()

df_train["u_in_exact"] = df_train["u_in"].astype(np.float32)
df_test["u_in_exact"] = df_test["u_in"].astype(np.float32)

df_train["du_in"] = (
    df_train.groupby("breath_id")["u_in"].diff().fillna(0.0).astype(np.float32)
)
df_test["du_in"] = (
    df_test.groupby("breath_id")["u_in"].diff().fillna(0.0).astype(np.float32)
)
df_train["du_in_exact"] = df_train["du_in"].astype(np.float32)
df_test["du_in_exact"] = df_test["du_in"].astype(np.float32)

UIN_BIN_WIDTH = 1.0
df_train["u_in_bin"] = (
    np.round(df_train["u_in"] / UIN_BIN_WIDTH) * UIN_BIN_WIDTH
).astype(np.float32)
df_test["u_in_bin"] = (
    np.round(df_test["u_in"] / UIN_BIN_WIDTH) * UIN_BIN_WIDTH
).astype(np.float32)

DUIN_BIN_WIDTH = 1.0
df_train["du_in_bin"] = (
    np.round(df_train["du_in"] / DUIN_BIN_WIDTH) * DUIN_BIN_WIDTH
).astype(np.float32)
df_test["du_in_bin"] = (
    np.round(df_test["du_in"] / DUIN_BIN_WIDTH) * DUIN_BIN_WIDTH
).astype(np.float32)

df_train_i = df_train[df_train["u_out"] == 0].copy()
train_idx_i = train_idx[df_train["u_out"].to_numpy() == 0]

med_table_rc_t_u_uinexact = (
    df_train.assign(ts_idx=train_idx)
    .groupby(["R", "C", "ts_idx", "u_out", "u_in_exact"], sort=False)["pressure"]
    .median()
    .reset_index()
)
med_table_rc_t_u_duinexact = (
    df_train.assign(ts_idx=train_idx)
    .groupby(["R", "C", "ts_idx", "u_out", "du_in_exact"], sort=False)["pressure"]
    .median()
    .reset_index()
)
med_table_rc_t_u_uinexact_insp = (
    df_train_i.assign(ts_idx=train_idx_i)
    .groupby(["R", "C", "ts_idx", "u_out", "u_in_exact"], sort=False)["pressure"]
    .median()
    .reset_index()
)
med_table_rc_t_u_duinexact_insp = (
    df_train_i.assign(ts_idx=train_idx_i)
    .groupby(["R", "C", "ts_idx", "u_out", "du_in_exact"], sort=False)["pressure"]
    .median()
    .reset_index()
)

med_table_rc_t_u_uinbin = (
    df_train.assign(ts_idx=train_idx)
    .groupby(["R", "C", "ts_idx", "u_out", "u_in_bin"], sort=False)["pressure"]
    .median()
    .reset_index()
)

med_table_rc_t_u_duinbin = (
    df_train.assign(ts_idx=train_idx)
    .groupby(["R", "C", "ts_idx", "u_out", "du_in_bin"], sort=False)["pressure"]
    .median()
    .reset_index()
)

med_table_rc_t_u_uinbin_insp = (
    df_train_i.assign(ts_idx=train_idx_i)
    .groupby(["R", "C", "ts_idx", "u_out", "u_in_bin"], sort=False)["pressure"]
    .median()
    .reset_index()
)

med_table_rc_t_u_duinbin_insp = (
    df_train_i.assign(ts_idx=train_idx_i)
    .groupby(["R", "C", "ts_idx", "u_out", "du_in_bin"], sort=False)["pressure"]
    .median()
    .reset_index()
)

med_table_rc_t_u = (
    df_train.assign(ts_idx=train_idx)
    .groupby(["R", "C", "ts_idx", "u_out"], sort=False)["pressure"]
    .median()
    .reset_index()
)

med_table_rc_t_insp = (
    df_train_i.assign(ts_idx=train_idx_i)
    .groupby(["R", "C", "ts_idx"], sort=False)["pressure"]
    .median()
    .reset_index()
)

med_table_rc_insp = (
    df_train_i.groupby(["R", "C"], sort=False)["pressure"].median().reset_index()
)

global_insp_median_pressure = float(df_train_i["pressure"].median())

test_key = df_test[
    ["id", "R", "C", "u_out", "u_in_exact", "du_in_exact", "u_in_bin", "du_in_bin"]
].copy()
test_key["ts_idx"] = test_idx.values

pred_df = test_key.merge(
    med_table_rc_t_u_uinexact,
    on=["R", "C", "ts_idx", "u_out", "u_in_exact"],
    how="left",
)

missing = pred_df["pressure"].isna()
if missing.any():
    m0 = missing & (test_key["u_out"].to_numpy() == 0)
    if m0.any():
        fill0_insp_uin_exact = (
            test_key.loc[m0, ["R", "C", "ts_idx", "u_out", "u_in_exact"]]
            .merge(
                med_table_rc_t_u_uinexact_insp,
                on=["R", "C", "ts_idx", "u_out", "u_in_exact"],
                how="left",
            )["pressure"]
            .to_numpy()
        )
        pred_df.loc[m0, "pressure"] = fill0_insp_uin_exact

missing = pred_df["pressure"].isna()
if missing.any():
    fill0_exact_duin = (
        test_key.loc[missing, ["R", "C", "ts_idx", "u_out", "du_in_exact"]]
        .merge(
            med_table_rc_t_u_duinexact,
            on=["R", "C", "ts_idx", "u_out", "du_in_exact"],
            how="left",
        )["pressure"]
        .to_numpy()
    )
    pred_df.loc[missing, "pressure"] = fill0_exact_duin

missing = pred_df["pressure"].isna()
if missing.any():
    m0 = missing & (test_key["u_out"].to_numpy() == 0)
    if m0.any():
        fill0_insp_duin_exact = (
            test_key.loc[m0, ["R", "C", "ts_idx", "u_out", "du_in_exact"]]
            .merge(
                med_table_rc_t_u_duinexact_insp,
                on=["R", "C", "ts_idx", "u_out", "du_in_exact"],
                how="left",
            )["pressure"]
            .to_numpy()
        )
        pred_df.loc[m0, "pressure"] = fill0_insp_duin_exact

missing = pred_df["pressure"].isna()
if missing.any():
    fill_uin_bin = (
        test_key.loc[missing, ["R", "C", "ts_idx", "u_out", "u_in_bin"]]
        .merge(
            med_table_rc_t_u_uinbin,
            on=["R", "C", "ts_idx", "u_out", "u_in_bin"],
            how="left",
        )["pressure"]
        .to_numpy()
    )
    pred_df.loc[missing, "pressure"] = fill_uin_bin

missing = pred_df["pressure"].isna()
if missing.any():
    m0 = missing & (test_key["u_out"].to_numpy() == 0)
    if m0.any():
        fill0_insp_uin = (
            test_key.loc[m0, ["R", "C", "ts_idx", "u_out", "u_in_bin"]]
            .merge(
                med_table_rc_t_u_uinbin_insp,
                on=["R", "C", "ts_idx", "u_out", "u_in_bin"],
                how="left",
            )["pressure"]
            .to_numpy()
        )
        pred_df.loc[m0, "pressure"] = fill0_insp_uin

missing = pred_df["pressure"].isna()
if missing.any():
    fill0b = (
        test_key.loc[missing, ["R", "C", "ts_idx", "u_out", "du_in_bin"]]
        .merge(
            med_table_rc_t_u_duinbin,
            on=["R", "C", "ts_idx", "u_out", "du_in_bin"],
            how="left",
        )["pressure"]
        .to_numpy()
    )
    pred_df.loc[missing, "pressure"] = fill0b

missing = pred_df["pressure"].isna()
if missing.any():
    m0 = missing & (test_key["u_out"].to_numpy() == 0)
    if m0.any():
        fill0_insp_duin = (
            test_key.loc[m0, ["R", "C", "ts_idx", "u_out", "du_in_bin"]]
            .merge(
                med_table_rc_t_u_duinbin_insp,
                on=["R", "C", "ts_idx", "u_out", "du_in_bin"],
                how="left",
            )["pressure"]
            .to_numpy()
        )
        pred_df.loc[m0, "pressure"] = fill0_insp_duin

missing = pred_df["pressure"].isna()
if missing.any():
    fill1b = (
        test_key.loc[missing, ["R", "C", "ts_idx", "u_out"]]
        .merge(med_table_rc_t_u, on=["R", "C", "ts_idx", "u_out"], how="left")[
            "pressure"
        ]
        .to_numpy()
    )
    pred_df.loc[missing, "pressure"] = fill1b

missing = pred_df["pressure"].isna()
if missing.any():
    fill2 = (
        test_key.loc[missing, ["R", "C", "ts_idx"]]
        .merge(med_table_rc_t_insp, on=["R", "C", "ts_idx"], how="left")["pressure"]
        .to_numpy()
    )
    pred_df.loc[missing, "pressure"] = fill2

missing = pred_df["pressure"].isna()
if missing.any():
    fill3 = (
        test_key.loc[missing, ["R", "C"]]
        .merge(med_table_rc_insp, on=["R", "C"], how="left")["pressure"]
        .to_numpy()
    )
    pred_df.loc[missing, "pressure"] = fill3

pred = pred_df["pressure"].to_numpy()
pred = np.where(np.isnan(pred), global_insp_median_pressure, pred)
pred = np.array([find_nearest(x) for x in pred], dtype=np.float32)

sub = pd.read_csv(SAMPLE_SUB_PATH)[["id"]].copy()
sub = sub.merge(pred_df[["id"]].assign(pred_pressure=pred), on="id", how="left")
sub["pressure"] = sub["pred_pressure"].astype(np.float32)
sub = sub[["id", "pressure"]]

if sub["pressure"].isna().any():
    sub["pressure"] = (
        sub["pressure"].fillna(global_insp_median_pressure).astype(np.float32)
    )

sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
print(
    "Pred stats:",
    float(np.min(sub["pressure"])),
    float(np.mean(sub["pressure"])),
    float(np.max(sub["pressure"])),
)
print("Fallback global inspiratory median:", global_insp_median_pressure)
print("UIN_BIN_WIDTH:", UIN_BIN_WIDTH, "DUIN_BIN_WIDTH:", DUIN_BIN_WIDTH)
