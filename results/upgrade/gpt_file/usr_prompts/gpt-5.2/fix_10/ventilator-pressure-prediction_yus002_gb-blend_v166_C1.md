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

0.1366086854463884

# 6. Current score

2.68757

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'The notebook fails because it depends on an external Kaggle dataset (`../input/gb-data-blending-recover/0.1333.csv`) that is not available in your environment. I keep the same blending/median/mean + “snap-to-nearest-known-pressure” logic, but make it robust: it blend only over whatever submission CSVs are present in the provided directory, and it skip the missing “anchor” file instead of crashing. I also fix the cell numbering to start at 1 and ensure the code always writes a valid `submission.csv` with `id,pressure` in the working directory.'
- What this solution (achieved 9.91366) has done: 'Your current score is far worse than the target (lower-is-better), and the root cause is that you’re writing an all-zeros submission when the external blending folder is missing. To move toward the target with minimal change and without altering your blending/snap-to-known-pressures logic, I add a deterministic fallback model that predicts `pressure` from `u_in` using a per-(R,C,time_step,u_out) median learned from train (and a safe global fallback). This keeps the “use only train/test CSVs + nearest-pressure snapping” semantics, produces a valid `submission.csv`, and should massively reduce MAE versus zeros while staying simple and fast. The blending path is kept intact when the external directory exists; only the missing-data behavior changes.'
- What this solution (achieved 3.76974) has done: 'Your current score (9.91366, lower-is-better) is still far from the target (0.1366), so we should improve the fallback predictor because the blending folder is missing in your environment. With minimal change and without altering your overall approach (simple non-ML fallback + snap-to-known-pressures), I make the fallback respect the competition metric by predicting only for the inspiratory phase (`u_out==0`) and forcing expiratory predictions (`u_out==1`) to a safe constant (0), which reduces MAE since expiratory rows are not scored. I also make the lookup more reliable by rounding `time_step` (it has discrete steps) to avoid float-merge mismatches, and I vectorize the snapping step to keep runtime under the 600s limit. The blending path remains untouched and be used if the external folder exists.'
- What this solution (achieved 3.18309) has done: 'Your current MAE (3.76974, lower-is-better) is still far above the target (0.1366), so we should improve only the fallback path (since the external blending folder is missing) while keeping the same “train-median lookup + snap-to-known-pressures” core logic. The biggest safe gain is to align the fallback with how pressure evolves: add minimal lag/cumulative features (previous u_in/u_out and cumulative inspired volume proxy) and condition the median lookup on them, which stays in the same non-ML grouping/median approach. We also keep the metric-aware behavior (don’t care about u_out==1 rows) and keep runtime bounded by using float32 and vectorized operations. Submission writing and schema are preserved exactly (`submission.csv` with `id,pressure`).'
- What this solution (achieved 2.34833) has done: 'We keep your exact fallback approach (train-derived median lookups + backoffs + snap-to-known pressures), but make two minimal, high-impact fixes that should move MAE much closer to the target: (1) don’t force `u_out==1` predictions to 0 (expiratory rows aren’t scored, but Kaggle still evaluates only on `u_out==0`; setting expiratory to 0 can hurt only if any rows are mistakenly included or if your own internal alignment drifts), and (2) add a very small additional backoff keyed by `(R,C,time_step_r,u_out,u_in_bin)` which uses the known strong `u_in->pressure` relationship while still staying in the same “groupby median lookup” logic. We also ensure the merge/join uses consistent dtypes and that the final submission is aligned to `sample_submission` order by `id`. No model/training loop changes are introduced; runtime stays well under the limit.'
- What this solution (achieved 2.33708) has done: 'Your current score (2.34833, lower-is-better) is still far above the target (0.1366), so we should improve only the fallback path (since the blending folder is missing) while keeping the exact same “train groupby median lookups + backoffs + snap-to-known-pressures” approach. The smallest high-impact change is to add one more backoff that captures the strongest structure in this competition: pressure is largely determined by the pair (R, C) and the cumulative inspired volume proxy (integral of u_in over time); we already compute `cum_u_in_bin`, but we only use it in the most specific (and sparse) key. By adding a mid-sparsity lookup keyed by `(R,C,time_step_r,u_out,cum_u_in_bin)` (and using it before the very weak `(R,C,time_step_r,u_out)` backoff), we reduce missing merges and improve inspiratory predictions without changing the overall logic. Everything else (data paths, snapping, submission alignment to sample_submission, runtime) is kept the same.'
- What this solution (achieved 2.33454) has done: 'Your current score is far above the target (lower-is-better), so we should only strengthen the existing fallback path (since the external blending folder is missing) while keeping the exact same “train groupby median lookups + backoffs + snap-to-known-pressures” approach. The smallest high-impact fix is to make the fallback respect the evaluation masking more precisely by learning medians only on inspiratory rows (already done) and also predicting inspiratory rows from those medians while letting expiratory rows fall back to a simple, stable value (global inspiratory median) without affecting scored rows. Additionally, we add one more mid-sparsity backoff keyed by `(R,C,u_out,cum_u_in_bin)` (dropping `time_step_r`) to reduce merge-miss rate caused by float rounding/time alignment while still using the strong cumulative-volume proxy signal. These changes keep runtime low (single-pass groupbys/merges), preserve your core logic, and should move MAE materially toward the target.'
- What this solution (achieved 2.33439) has done: 'Your current MAE (2.33454, lower-is-better) is still far from the target (0.1366), so we should improve only the existing fallback path (since the external blending folder is missing) while keeping the exact same “train groupby medians + hierarchical backoffs + snap-to-known-pressures” approach. The smallest high-impact issue is that the fallback currently predicts expiratory rows (`u_out==1`) using a constant, but Kaggle evaluates only inspiratory rows, so we can safely avoid contaminating the learned medians by training strictly on inspiratory and also enforce that the lookup keys used for prediction always have `u_out==0` (so we never rely on nonexistent `u_out==1` medians). Additionally, we add one more mid-sparsity backoff keyed by `(R,C,u_out,u_in_bin,cum_u_in_bin)` to reduce merge misses while still using the strong `u_in` + cumulative-volume proxy signal, without changing the fundamental method. Finally, we ensure `id` alignment is correct by sorting output by `id` before merging into `sample_submission` (guarding against any breath/time sorting side effects).'
- What this solution (achieved 2.68757) has done: 'Your current score (2.33439, lower-is-better) is still far from the target (0.1366), so we should improve only the fallback path (since the blending folder is missing) while keeping the exact same “train groupby medians + hierarchical backoffs + snap-to-known-pressures” approach. The smallest likely high-impact fix is to add one additional, low-sparsity but still informative backoff keyed by `(R, C, u_out_key, u_in_bin)` so inspiratory predictions don’t collapse to overly generic medians when `time_step`/history/cum bins miss. We also slightly refine bin widths (smaller `u_in` bin) to better match the discrete pressure levels without changing the core logic. Submission schema and `id` alignment remain unchanged, and runtime stays within limits.'

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
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SAMPLE_SUB_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "/kaggle/input/ventilator-pressure-prediction/train.csv"
if not os.path.exists(TEST_PATH):
    TEST_PATH = "/kaggle/input/ventilator-pressure-prediction/test.csv"
if not os.path.exists(SAMPLE_SUB_PATH):
    SAMPLE_SUB_PATH = (
        "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv"
    )

df_train = pd.read_csv(TRAIN_PATH)

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


def snap_to_known_pressures(pred_arr: np.ndarray) -> np.ndarray:
    pred_arr = pred_arr.astype(np.float64, copy=False)
    idx = np.searchsorted(sorted_pressures, pred_arr, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)

    lower_idx = np.clip(idx - 1, 0, total_pressures_len - 1)
    upper_idx = idx

    lower = sorted_pressures[lower_idx]
    upper = sorted_pressures[upper_idx]

    choose_lower = np.abs(pred_arr - lower) < np.abs(upper - pred_arr)
    out = np.where(choose_lower, lower, upper)
    return out.astype(np.float64, copy=False)


def set_seed(seed: int = 2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    """
    Original intent: combine 1-2 files with weights inferred from filename.
    Keep logic but make it robust if filename doesn't match expected pattern.
    """
    l = []
    for i in range(len(input_list)):
        try:
            public_lb_score = int(
                input_list[i].split("/")[-1].split(".")[1].split(" ")[0]
            )
        except Exception:
            public_lb_score = 1  # safe default weight proxy
        l.append(public_lb_score)
        input_list[i] = (pd.read_csv(input_list[i]).pressure).to_numpy().ravel()

    l_sum = sum(l) if sum(l) != 0 else 1

    if len(input_list) == 1:
        output = input_list[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        output = input_list[0] * weight1 + input_list[1] * weight2
    return output


def g(dp):
    """
    Robustified version of the original g():
    - No longer hard-requires ../input/gb-data-blending-recover/0.1333.csv (missing in this env).
    - Uses only CSVs available under dp.
    - Keeps the same median/mean + random-weight ensembling + snapping to nearest pressure.
    - Always writes a valid submission.csv.
    """
    l = [p for p in glob.iglob(f"{dp}/*") if p.lower().endswith(".csv")]
    l.sort()

    output = pd.read_csv(SAMPLE_SUB_PATH)

    if len(l) == 0:
        output["pressure"] = 0.0
        output.to_csv("submission.csv", index=False)
        return output

    file_count = len(l)
    loop_time = 154

    splits = max(1, file_count // 2)
    flist = []
    chunk = round(len(l) / splits) if splits > 0 else len(l)

    for i in range(splits):
        if i == splits - 1:
            flist.append(l[i * chunk :])
        else:
            flist.append(l[i * chunk : (i + 1) * chunk])

    for i in range(len(flist)):
        if len(flist[i]) == 0:
            flist[i] = None
        else:
            flist[i] = wc(flist[i])

    flist = [x for x in flist if x is not None]
    if len(flist) == 0:
        output["pressure"] = 0.0
        output.to_csv("submission.csv", index=False)
        return output

    pred_list = []
    for seed in range(loop_time):
        set_seed(seed)
        weight = [rd() for _ in range(len(flist))]
        weight_sum = sum(weight) if sum(weight) != 0 else 1.0
        weight = [w / weight_sum for w in weight]
        weight.sort(reverse=True)

        temp = 0
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)

        del temp
        gc.collect()

    median_pred = np.median(np.vstack(pred_list), axis=0)
    mean_pred = np.mean(np.vstack(pred_list), axis=0)

    anchor_path = os.path.join(dp, "0.1333.csv")
    if os.path.exists(anchor_path):
        anchor = pd.read_csv(anchor_path)["pressure"].to_numpy()
        output["pressure"] = 0.45 * median_pred + 0.15 * mean_pred + 0.4 * anchor
    else:
        output["pressure"] = 0.75 * median_pred + 0.25 * mean_pred

    output["pressure"] = snap_to_known_pressures(output["pressure"].to_numpy())
    output.to_csv("submission.csv", index=False)
    return output


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.7 + b.pressure * 0.3
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a


def avg(dp):
    input_list = [p for p in glob.iglob(f"{dp}/*") if p.lower().endswith(".csv")]
    output = pd.read_csv(SAMPLE_SUB_PATH)

    if len(input_list) == 0:
        output["pressure"] = 0.0
        output.to_csv("avg.csv", index=False)
        return output

    preds = []
    for p in input_list:
        preds.append(pd.read_csv(p).pressure.to_numpy().ravel())

    output.pressure = np.median(np.vstack(preds), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv("avg.csv", index=False)
    return output


def fallback_pressure_from_train():
    """
    Keep the same core fallback logic (groupby medians from train + merge onto test + snap).

    Changes here are strictly score-directed and minimal:
    - Add one more low-sparsity backoff keyed by (R,C,u_out_key,u_in_bin) to reduce fallback-to-global.
    - Slightly refine u_in bin width to improve median lookup fidelity (still the same binning approach).
    - Keep inspiratory-only learning (u_out==0) and always use u_out_key=0 for prediction keys.
    """
    test = pd.read_csv(
        TEST_PATH, usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
    )

    df_train_local = df_train[
        ["breath_id", "R", "C", "time_step", "u_out", "u_in", "pressure"]
    ].copy()

    for d in (df_train_local, test):
        d["R"] = d["R"].astype(np.int16)
        d["C"] = d["C"].astype(np.int16)
        d["u_out"] = d["u_out"].astype(np.int8)
        d["time_step"] = d["time_step"].astype(np.float32)
        d["u_in"] = d["u_in"].astype(np.float32)
    df_train_local["pressure"] = df_train_local["pressure"].astype(np.float32)

    test_local = test.copy()

    ts_round = 2
    uin_bin_w = 0.25

    df_train_local["time_step_r"] = df_train_local["time_step"].round(ts_round)
    test_local["time_step_r"] = test_local["time_step"].round(ts_round)

    df_train_local["u_in_bin"] = (
        (df_train_local["u_in"] / uin_bin_w).round() * uin_bin_w
    ).astype(np.float32)
    test_local["u_in_bin"] = (
        (test_local["u_in"] / uin_bin_w).round() * uin_bin_w
    ).astype(np.float32)

    df_train_local.sort_values(["breath_id", "time_step"], inplace=True)
    test_local.sort_values(["breath_id", "time_step"], inplace=True)

    for d in (df_train_local, test_local):
        d["u_in_prev1"] = (
            d.groupby("breath_id", sort=False)["u_in"]
            .shift(1)
            .fillna(0.0)
            .astype(np.float32)
        )
        d["u_out_prev1"] = (
            d.groupby("breath_id", sort=False)["u_out"]
            .shift(1)
            .fillna(0)
            .astype(np.int8)
        )
        dt = (
            d.groupby("breath_id", sort=False)["time_step"]
            .diff()
            .fillna(d["time_step"])
            .clip(lower=0)
            .astype(np.float32)
        )
        d["cum_u_in"] = (
            (d["u_in"] * dt)
            .groupby(d["breath_id"], sort=False)
            .cumsum()
            .astype(np.float32)
        )

        d["u_in_prev1_bin"] = (
            (d["u_in_prev1"] / uin_bin_w).round() * uin_bin_w
        ).astype(np.float32)
        cum_bin_w = 0.2
        d["cum_u_in_bin"] = ((d["cum_u_in"] / cum_bin_w).round() * cum_bin_w).astype(
            np.float32
        )

    train_insp = df_train_local[df_train_local["u_out"] == 0].copy()
    global_med_insp = float(train_insp["pressure"].median())

    train_insp["u_out_key"] = np.int8(0)
    test_local["u_out_key"] = np.int8(0)

    grp_cols_full = [
        "R",
        "C",
        "time_step_r",
        "u_out_key",
        "u_in_bin",
        "u_in_prev1_bin",
        "u_out_prev1",
        "cum_u_in_bin",
    ]
    med_full = (
        train_insp.groupby(grp_cols_full, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_full"})
    )

    grp_cols_mid = [
        "R",
        "C",
        "time_step_r",
        "u_out_key",
        "u_in_bin",
        "u_in_prev1_bin",
        "u_out_prev1",
    ]
    med_mid = (
        train_insp.groupby(grp_cols_mid, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_mid"})
    )

    grp_cols_uin = ["R", "C", "time_step_r", "u_out_key", "u_in_bin"]
    med_uin = (
        train_insp.groupby(grp_cols_uin, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_uin"})
    )

    grp_cols_cum = ["R", "C", "time_step_r", "u_out_key", "cum_u_in_bin"]
    med_cum = (
        train_insp.groupby(grp_cols_cum, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_cum"})
    )

    grp_cols_uin_cum = ["R", "C", "u_out_key", "u_in_bin", "cum_u_in_bin"]
    med_uin_cum = (
        train_insp.groupby(grp_cols_uin_cum, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_uin_cum"})
    )

    grp_cols_cum_coarse = ["R", "C", "u_out_key", "cum_u_in_bin"]
    med_cum_coarse = (
        train_insp.groupby(grp_cols_cum_coarse, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_cum_coarse"})
    )

    grp_cols_rc_uin = ["R", "C", "u_out_key", "u_in_bin"]
    med_rc_uin = (
        train_insp.groupby(grp_cols_rc_uin, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_rc_uin"})
    )

    grp_cols_backoff = ["R", "C", "time_step_r", "u_out_key"]
    med_backoff = (
        train_insp.groupby(grp_cols_backoff, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_backoff"})
    )

    pred = test_local.merge(med_full, on=grp_cols_full, how="left")
    pred = pred.merge(med_mid, on=grp_cols_mid, how="left")
    pred = pred.merge(med_uin, on=grp_cols_uin, how="left")
    pred = pred.merge(med_cum, on=grp_cols_cum, how="left")
    pred = pred.merge(med_uin_cum, on=grp_cols_uin_cum, how="left")
    pred = pred.merge(med_cum_coarse, on=grp_cols_cum_coarse, how="left")
    pred = pred.merge(med_rc_uin, on=grp_cols_rc_uin, how="left")
    pred = pred.merge(med_backoff, on=grp_cols_backoff, how="left")

    pred_pressure = (
        pred["p_full"]
        .fillna(pred["p_mid"])
        .fillna(pred["p_uin"])
        .fillna(pred["p_cum"])
        .fillna(pred["p_uin_cum"])
        .fillna(pred["p_cum_coarse"])
        .fillna(pred["p_rc_uin"])
        .fillna(pred["p_backoff"])
        .fillna(global_med_insp)
        .to_numpy(dtype=np.float64, copy=False)
    )

    pred_pressure = snap_to_known_pressures(pred_pressure)

    out = pd.DataFrame({"id": test_local["id"].to_numpy(), "pressure": pred_pressure})
    out.sort_values("id", inplace=True)

    sub = pd.read_csv(SAMPLE_SUB_PATH)
    sub = sub.drop(columns=["pressure"]).merge(out, on="id", how="left", sort=False)
    sub["pressure"] = sub["pressure"].fillna(global_med_insp).astype(float)
    sub.to_csv("submission.csv", index=False)
    return sub




## === cell 2
DP = "../input/gb-data-blending-recover"
if not os.path.exists(DP):
    fallback_pressure_from_train()
else:
    g(DP)
