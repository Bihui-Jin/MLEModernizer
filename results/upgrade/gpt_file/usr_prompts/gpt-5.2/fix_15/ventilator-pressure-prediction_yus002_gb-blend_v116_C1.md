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

0.1460040612797075

# 6. Current score

2.29256

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.46367) has done: 'I remove the dependency on missing external Kaggle datasets (`gb-data-blending-recover`) that causes the `FileNotFoundError`, and instead generate a valid submission directly from the provided `train.csv/test.csv`. To keep the “core logic” intact (pressure discretization via `find_nearest`), I build a simple, deterministic baseline predictor using only information available in the test rows (a per-(R,C,u_out) median pressure learned from train), then snap predictions to the nearest valid pressure value. This run end-to-end in the given environment and always write a correctly formatted `submission.csv` with `id,pressure`. The blending functions are kept but made safe/optional so they won’t crash if the external files aren’t present.'
- What this solution (achieved 3.76971) has done: 'Your current score (7.46367, lower-is-better) is far worse than the target (0.1460), so we should improve accuracy while keeping your core “median-then-snap-to-nearest-pressure” logic intact. The biggest issue is that predicting a single median per (R,C,u_out) ignores time dynamics and u_in, which are strongly predictive; we can keep the same approach but compute medians on a richer, still-deterministic key using only available features. To stay minimal and fast, we add simple groupwise keys: (R,C,u_out,time_step_rounded) and (R,C,u_out,u_in_binned), then back off to your original (R,C,u_out) and finally global median when missing, and still snap to the nearest valid pressure. This should substantially reduce MAE without changing the submission format or introducing any training loop/model.'
- What this solution (achieved 2.51669) has done: 'Your current score (3.76971, lower-is-better) is far above the target (0.1460), so we should improve accuracy while keeping your deterministic “groupwise median → snap to nearest valid pressure” core logic intact. The biggest gap is that the grouping keys still don’t capture breath dynamics well; we can minimally enrich features with simple, leak-free cumulative signals per breath (cumulative u_in and lagged u_in/u_out) and use them only as additional median lookup keys with a safe backoff chain. This keeps the same approach (no model/training loop), but makes the medians much more context-specific and should reduce MAE substantially. We also keep the snapping step identical and still write a valid `submission.csv`.'
- What this solution (achieved 2.51563) has done: 'We keep your deterministic “groupwise median → backoff chain → snap to nearest valid pressure” core logic, but fix the biggest evaluation mismatch: the metric ignores expiratory rows (`u_out==1`), and your current medians likely overfit those regimes and can distort inspiratory predictions. Concretely, we compute all medians using only inspiratory training rows (`u_out==0`), while still producing predictions for all test rows and snapping exactly as before. Additionally, we add one minimal, leak-free temporal signal (`u_in_diff` and its binned version) as an extra highest-priority key to better capture dynamics without changing the approach. These changes should reduce MAE materially from 2.51669 toward the 0.146 target while staying fast and fully deterministic, and still write a valid `submission.csv`.'
- What this solution (achieved 2.51563) has done: 'Your current score (2.51563, lower-is-better) is far above the target (0.1460), so we should improve accuracy while keeping your “groupwise median → backoff → snap to nearest valid pressure” core logic intact. The largest remaining mismatch is that you compute medians only on inspiratory rows but still key/group by `u_out`; since inspiratory rows have `u_out==0`, all `u_out==1` test rows fall back to coarse/global medians and can be very wrong even if they’re not scored, and the added `u_out`-dependent merges can also reduce coverage for `u_out==0` due to lag features. The minimal fix is to (1) build all median tables without `u_out` as a key (still trained only on inspiratory rows), and (2) force predictions for `u_out==1` test rows to a safe constant (global median snapped) so they don’t interfere and keep behavior deterministic. This preserves the same pipeline, improves key coverage for inspiratory rows, and should move MAE down materially toward the target.'
- What this solution (achieved 2.21615) has done: 'Your current MAE (2.51563, lower-is-better) is far above the target (0.1460), so we need a real accuracy gain while keeping your “groupwise median → backoff → snap-to-valid-pressure” core logic unchanged. The biggest issue is that the current keys are so specific (cum/lag/diff + fine bins) that most test rows miss the higher-priority tables and fall back to coarse medians, which caps performance. I keep all your existing features and median tables, but add two very small, deterministic mid-level backoff tables that dramatically increase match coverage: (R,C,ts_r,u_in_b) trained on inspiratory rows only, and (R,C,ts_r,u_in_b,u_in_lag1_b) as an intermediate step. I also align the default fill to the snapped global median (instead of unsnapped) so the final snapping step doesn’t get avoidable tie/edge artifacts.'
- What this solution (achieved 2.21618) has done: 'Your MAE (2.216, lower-is-better) is still far from the target (0.146), so we need a real accuracy gain while keeping your same deterministic “groupwise median → backoff chain → snap-to-valid-pressure” approach. The minimal high-impact issue is that you train medians only on inspiratory rows but your features (cum/lag/diff) are computed on *all* test rows; after the switch to expiration (`u_out==1`), the cum/lag/diff drift and can slightly contaminate the *next* breath’s early inspiratory matching coverage when breath boundaries/ordering aren’t strictly enforced. I therefore compute cum/lag/diff strictly within each breath *after sorting by (breath_id,time_step)* in both train and test, and I also add one very cheap, mid-level backoff table using only stable features `(R,C,u_in_b)` to increase hit-rate when time-based keys miss. These are small, safe changes that should materially reduce MAE without changing your core logic or any modeling/training.'
- What this solution (achieved 2.21618) has done: 'Your current MAE (2.216, lower-is-better) is still far above the target (0.146), so we need a clear accuracy gain while keeping your same deterministic “groupwise median → backoff → snap-to-valid-pressure” approach. The biggest remaining limitation is that your keys don’t include the main dynamic driver: the cumulative “flow” that builds pressure, which is well-approximated by the integral of `u_in` over time; adding a leak-free `u_in * dt` cumulative feature (and a light bin of it) improves matching without changing the modeling approach. I also compute `dt` robustly within each breath (sorted by `time_step`) and add one intermediate backoff table using `(R,C,ts_r,uin_int_cum_b,u_in_b)` to increase hit-rate when the very specific lag/diff tables miss. Everything else (inspiratory-only training rows, backoff chain, snapping to nearest valid pressure, submission writing) stays the same.'
- What this solution (achieved 2.41421) has done: 'Your current MAE (2.216, lower-is-better) is still far above the target (0.146), so we need to improve accuracy while keeping your same deterministic “groupwise median → backoff chain → snap-to-valid-pressure” core logic. The biggest minimal fix is to stop training medians on the full inspiratory phase mixed together: instead, restrict the median tables to the scored portion more precisely by filtering to `u_out==0` *and* excluding the very early timestep where pressure is less stable, and to make the temporal keys less brittle by using `time_step` rounded to 1 decimal (not 2) for the higher-priority tables to increase match coverage. I keep your existing features, backoff chain, snapping, and submission writing, but add one robust intermediate table keyed by `(R,C,ts_r1,u_in_b)` and change the merge order so we use the higher-coverage time key before falling back to coarse tables. These changes are small, deterministic, and aimed at reducing fallbacks (which is currently the main limiter), moving MAE down toward the target band.'
- What this solution (achieved 2.20401) has done: 'Your current MAE (2.414, lower-is-better) is far worse than the target (0.146), so we need a meaningful accuracy gain while keeping your deterministic “groupwise median → backoff chain → snap-to-valid-pressure” approach intact. The biggest low-risk issue is that most of your fine-grained lookup keys are too brittle, causing heavy fallback to coarse medians; to improve match coverage, I add a couple of higher-coverage median tables using coarser time rounding (`ts_r1`) and fewer keys, and I move the integration-based table earlier in the backoff order so it’s used before overly time-specific fallbacks. I also stop excluding `time_step==0.0` from training (still inspiratory-only) because those rows are scored and provide stable initial conditions; this typically reduces early-step error without changing the core method. All changes are deterministic, keep the same “median lookup then snap” semantics, and still write a valid `submission.csv`.'
- What this solution (achieved 2.62404) has done: 'We keep your deterministic “groupwise median → backoff chain → snap-to-valid-pressure” core logic, but fix two key score limiters: (1) your `u_in_cum` currently treats `u_in` as if the sampling interval is constant; we switch it to a leak-free physically closer proxy (cumulative integral `sum(u_in*dt)`), while leaving your existing integral feature intact and just using it more consistently. (2) Your binning for the integrated signal is very tight and likely causes lots of miss/fallback; we add one slightly coarser-bin median table early in the backoff chain to increase hit-rate without changing the approach. This should reduce MAE from 2.204 toward the 0.146 target while staying fully deterministic, fast, and producing a valid `submission.csv`.'
- What this solution (achieved 2.62404) has done: 'Your current MAE (2.624) is still far above the target (0.146), so we should improve accuracy with the smallest possible edits while keeping your same deterministic “groupwise median → backoff chain → snap-to-valid-pressure” approach. The biggest issue is a feature mismatch: all medians are learned only from inspiratory rows (`u_out==0`), but in train you compute lag/diff/cumulative features *after filtering*, which resets history compared to test (where features are computed over the full breath including `u_out==1`). I fix this by computing all temporal features on the full train/test breaths first, then filtering to inspiratory rows only for building median tables—this keeps your core logic identical but makes keys consistent and should reduce fallback and MAE. I also make the `dt` initialization consistent by setting the first `dt` in each breath to 0 (instead of `time_step`), which better matches an integral from the breath start and avoids an inflated first-step integral.'
- What this solution (achieved 2.29255) has done: 'Your current MAE (2.624, lower-is-better) is still far above the target (0.146), so we need a clear accuracy gain with minimal edits while keeping your deterministic “groupwise median → backoff chain → snap-to-valid-pressure” pipeline unchanged. The highest-impact issue in this style of solution is key brittleness: your very fine bins (especially `u_in_cum_b`, `uin_int_cum_b`, `u_in_diff_b`) make most test rows miss high-priority tables and fall back to coarse medians, capping accuracy. I keep all existing features/tables, but add a small set of *coarser-binned* intermediate median tables and insert them early in the backoff order to increase match coverage without changing core semantics. I also ensure prediction for `u_out==1` stays a constant snapped median (unchanged intent) and keep submission formatting identical.'
- What this solution (achieved 2.29256) has done: 'I keep your exact deterministic “groupwise median lookup → backoff chain → snap to nearest valid pressure” core logic, but fix one high-impact key brittleness source that’s currently forcing lots of fallbacks: `time_step` rounding to 2 decimals is too fine and can mismatch between train/test due to floating representation. Concretely, I switch the high-priority `ts_r` key from `round(2)` to `round(1)` (keeping your existing `ts_r1` intact) and add one new medium-priority lookup table keyed by `(R,C,ts_r,u_in_b,u_in_lag1_b)` to increase match coverage without changing semantics. I also ensure `time_step` is cast to float32 before rounding (reduces representation noise) while keeping everything deterministic and still snapping exactly as you do now. This should reduce fallbacks materially and move MAE down from 2.29 toward your 0.146 target without introducing any training/model changes.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
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
    loop_time = 154
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
        for j in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        for j in range(len(weight)):
            weight[j] /= weight_sum
        weight.sort(reverse=True)
        temp = 0
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output.pressure = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b, out_path="blend.csv"):
    if (not os.path.exists(a)) or (not os.path.exists(b)):
        print(f"blend(): file missing, skipping.\n a={a}\n b={b}")
        return None
    a_df = pd.read_csv(a)
    b_df = pd.read_csv(b)
    a_df["pressure"] = a_df["pressure"] * 0.55 + b_df["pressure"] * 0.45
    a_df["pressure"] = a_df["pressure"].apply(find_nearest)
    a_df.to_csv(out_path, index=False)
    return a_df




## === cell 2
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")

train_full = df_train[
    ["breath_id", "R", "C", "u_out", "u_in", "time_step", "pressure"]
].copy()
test_feat = df_test[["id", "breath_id", "R", "C", "u_out", "u_in", "time_step"]].copy()

train_full = train_full.sort_values(
    ["breath_id", "time_step"], kind="mergesort"
).reset_index(drop=True)
test_feat = test_feat.sort_values(
    ["breath_id", "time_step"], kind="mergesort"
).reset_index(drop=True)

train_full["time_step"] = train_full["time_step"].astype(np.float32)
test_feat["time_step"] = test_feat["time_step"].astype(np.float32)

train_full["ts_r"] = train_full["time_step"].round(1)
test_feat["ts_r"] = test_feat["time_step"].round(1)

train_full["ts_r1"] = train_full["time_step"].round(1)
test_feat["ts_r1"] = test_feat["time_step"].round(1)

train_full["u_in_b"] = np.clip(np.rint(train_full["u_in"] * 2.0), 0, 200).astype(
    np.int16
)
test_feat["u_in_b"] = np.clip(np.rint(test_feat["u_in"] * 2.0), 0, 200).astype(np.int16)

train_full["dt"] = (
    train_full.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0)
)
test_feat["dt"] = (
    test_feat.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0)
)
train_full["dt"] = train_full["dt"].clip(lower=0.0, upper=0.1)
test_feat["dt"] = test_feat["dt"].clip(lower=0.0, upper=0.1)

train_full["uin_int_step"] = train_full["u_in"] * train_full["dt"]
test_feat["uin_int_step"] = test_feat["u_in"] * test_feat["dt"]

train_full["uin_int_cum"] = train_full.groupby("breath_id", sort=False)[
    "uin_int_step"
].cumsum()
test_feat["uin_int_cum"] = test_feat.groupby("breath_id", sort=False)[
    "uin_int_step"
].cumsum()

train_full["u_in_cum"] = train_full["uin_int_cum"]
test_feat["u_in_cum"] = test_feat["uin_int_cum"]

train_full["u_in_cum_b"] = np.clip(
    np.rint(train_full["u_in_cum"] * 100.0), 0, 6000
).astype(np.int16)
test_feat["u_in_cum_b"] = np.clip(
    np.rint(test_feat["u_in_cum"] * 100.0), 0, 6000
).astype(np.int16)

train_full["u_in_lag1_b"] = np.clip(
    np.rint(
        train_full.groupby("breath_id", sort=False)["u_in"].shift(1).fillna(0.0) * 2.0
    ),
    0,
    200,
).astype(np.int16)
test_feat["u_in_lag1_b"] = np.clip(
    np.rint(
        test_feat.groupby("breath_id", sort=False)["u_in"].shift(1).fillna(0.0) * 2.0
    ),
    0,
    200,
).astype(np.int16)

train_full["u_in_diff"] = (
    train_full.groupby("breath_id", sort=False)["u_in"].diff().fillna(0.0)
)
test_feat["u_in_diff"] = (
    test_feat.groupby("breath_id", sort=False)["u_in"].diff().fillna(0.0)
)

train_full["u_in_diff_b"] = np.clip(
    np.rint(train_full["u_in_diff"] * 2.0), -200, 200
).astype(np.int16)
test_feat["u_in_diff_b"] = np.clip(
    np.rint(test_feat["u_in_diff"] * 2.0), -200, 200
).astype(np.int16)

train_full["uin_int_cum_b"] = np.clip(
    np.rint(train_full["uin_int_cum"] * 100.0), 0, 6000
).astype(np.int16)
test_feat["uin_int_cum_b"] = np.clip(
    np.rint(test_feat["uin_int_cum"] * 100.0), 0, 6000
).astype(np.int16)

train_full["uin_int_cum_b2"] = np.clip(
    np.rint(train_full["uin_int_cum"] * 50.0), 0, 3000
).astype(np.int16)
test_feat["uin_int_cum_b2"] = np.clip(
    np.rint(test_feat["uin_int_cum"] * 50.0), 0, 3000
).astype(np.int16)

train_full["u_in_cum_b_coarse"] = np.clip(
    np.rint(train_full["u_in_cum"] * 25.0), 0, 1500
).astype(np.int16)
test_feat["u_in_cum_b_coarse"] = np.clip(
    np.rint(test_feat["u_in_cum"] * 25.0), 0, 1500
).astype(np.int16)

train_full["uin_int_cum_b_coarse"] = np.clip(
    np.rint(train_full["uin_int_cum"] * 25.0), 0, 1500
).astype(np.int16)
test_feat["uin_int_cum_b_coarse"] = np.clip(
    np.rint(test_feat["uin_int_cum"] * 25.0), 0, 1500
).astype(np.int16)

train_full["u_in_diff_b_coarse"] = np.clip(
    np.rint(train_full["u_in_diff"] * 1.0), -100, 100
).astype(np.int16)
test_feat["u_in_diff_b_coarse"] = np.clip(
    np.rint(test_feat["u_in_diff"] * 1.0), -100, 100
).astype(np.int16)

train_feat = train_full[train_full["u_out"] == 0].copy()

global_median = float(train_feat["pressure"].median())
global_median_snapped = float(find_nearest(global_median))

grp_rctsuin_cum_lag_diff = (
    train_feat.groupby(
        ["R", "C", "ts_r", "u_in_b", "u_in_cum_b", "u_in_lag1_b", "u_in_diff_b"],
        sort=False,
    )["pressure"]
    .median()
    .rename("p_med_0")
    .reset_index()
)

grp_rctsuin_cum_lag = (
    train_feat.groupby(
        ["R", "C", "ts_r", "u_in_b", "u_in_cum_b", "u_in_lag1_b"], sort=False
    )["pressure"]
    .median()
    .rename("p_med_1")
    .reset_index()
)

grp_rctsuin_cum = (
    train_feat.groupby(["R", "C", "ts_r", "u_in_b", "u_in_cum_b"], sort=False)[
        "pressure"
    ]
    .median()
    .rename("p_med_2")
    .reset_index()
)

grp_rctsuin_lag = (
    train_feat.groupby(["R", "C", "ts_r", "u_in_b", "u_in_lag1_b"], sort=False)[
        "pressure"
    ]
    .median()
    .rename("p_med_3")
    .reset_index()
)

grp_rctsuin = (
    train_feat.groupby(["R", "C", "ts_r", "u_in_b"], sort=False)["pressure"]
    .median()
    .rename("p_med_4")
    .reset_index()
)

grp_rctsuin_lag_med = (
    train_feat.groupby(["R", "C", "ts_r", "u_in_b", "u_in_lag1_b"], sort=False)[
        "pressure"
    ]
    .median()
    .rename("p_med_4lag")
    .reset_index()
)

grp_rcts_uinint_uin = (
    train_feat.groupby(["R", "C", "ts_r", "uin_int_cum_b", "u_in_b"], sort=False)[
        "pressure"
    ]
    .median()
    .rename("p_med_4b")
    .reset_index()
)

grp_rcts_uinint2_uin = (
    train_feat.groupby(["R", "C", "ts_r", "uin_int_cum_b2", "u_in_b"], sort=False)[
        "pressure"
    ]
    .median()
    .rename("p_med_4b2")
    .reset_index()
)

grp_rctsuin_ts1 = (
    train_feat.groupby(["R", "C", "ts_r1", "u_in_b"], sort=False)["pressure"]
    .median()
    .rename("p_med_4c")
    .reset_index()
)

grp_rctsuin_cum_ts1 = (
    train_feat.groupby(["R", "C", "ts_r1", "u_in_b", "u_in_cum_b"], sort=False)[
        "pressure"
    ]
    .median()
    .rename("p_med_2c")
    .reset_index()
)

grp_rcts_uinint_uin_ts1 = (
    train_feat.groupby(["R", "C", "ts_r1", "uin_int_cum_b", "u_in_b"], sort=False)[
        "pressure"
    ]
    .median()
    .rename("p_med_4d")
    .reset_index()
)

grp_rctsuin_cum_lag_diff_coarse = (
    train_feat.groupby(
        [
            "R",
            "C",
            "ts_r",
            "u_in_b",
            "u_in_cum_b_coarse",
            "u_in_lag1_b",
            "u_in_diff_b_coarse",
        ],
        sort=False,
    )["pressure"]
    .median()
    .rename("p_med_0c")
    .reset_index()
)

grp_rctsuin_cum_coarse = (
    train_feat.groupby(
        ["R", "C", "ts_r", "u_in_b", "u_in_cum_b_coarse"],
        sort=False,
    )["pressure"]
    .median()
    .rename("p_med_2c0")
    .reset_index()
)

grp_rcts_uinint_uin_coarse = (
    train_feat.groupby(
        ["R", "C", "ts_r", "uin_int_cum_b_coarse", "u_in_b"],
        sort=False,
    )["pressure"]
    .median()
    .rename("p_med_4bc0")
    .reset_index()
)

grp_ts = (
    train_feat.groupby(["R", "C", "ts_r"], sort=False)["pressure"]
    .median()
    .rename("p_med_5")
    .reset_index()
)

grp_rc_uin = (
    train_feat.groupby(["R", "C", "u_in_b"], sort=False)["pressure"]
    .median()
    .rename("p_med_6b")
    .reset_index()
)

grp_rc = (
    train_feat.groupby(["R", "C"], sort=False)["pressure"]
    .median()
    .rename("p_med_6")
    .reset_index()
)

test_feat = test_feat.merge(
    grp_rctsuin_cum_lag_diff,
    on=["R", "C", "ts_r", "u_in_b", "u_in_cum_b", "u_in_lag1_b", "u_in_diff_b"],
    how="left",
)
test_feat = test_feat.merge(
    grp_rctsuin_cum_lag_diff_coarse,
    on=[
        "R",
        "C",
        "ts_r",
        "u_in_b",
        "u_in_cum_b_coarse",
        "u_in_lag1_b",
        "u_in_diff_b_coarse",
    ],
    how="left",
)

test_feat = test_feat.merge(
    grp_rctsuin_cum_lag,
    on=["R", "C", "ts_r", "u_in_b", "u_in_cum_b", "u_in_lag1_b"],
    how="left",
)
test_feat = test_feat.merge(
    grp_rctsuin_cum,
    on=["R", "C", "ts_r", "u_in_b", "u_in_cum_b"],
    how="left",
)
test_feat = test_feat.merge(
    grp_rctsuin_cum_coarse,
    on=["R", "C", "ts_r", "u_in_b", "u_in_cum_b_coarse"],
    how="left",
)

test_feat = test_feat.merge(
    grp_rctsuin_lag, on=["R", "C", "ts_r", "u_in_b", "u_in_lag1_b"], how="left"
)
test_feat = test_feat.merge(grp_rctsuin, on=["R", "C", "ts_r", "u_in_b"], how="left")

test_feat = test_feat.merge(
    grp_rctsuin_lag_med, on=["R", "C", "ts_r", "u_in_b", "u_in_lag1_b"], how="left"
)

test_feat = test_feat.merge(
    grp_rcts_uinint_uin, on=["R", "C", "ts_r", "uin_int_cum_b", "u_in_b"], how="left"
)
test_feat = test_feat.merge(
    grp_rcts_uinint_uin_coarse,
    on=["R", "C", "ts_r", "uin_int_cum_b_coarse", "u_in_b"],
    how="left",
)
test_feat = test_feat.merge(
    grp_rcts_uinint2_uin, on=["R", "C", "ts_r", "uin_int_cum_b2", "u_in_b"], how="left"
)

test_feat = test_feat.merge(
    grp_rctsuin_cum_ts1, on=["R", "C", "ts_r1", "u_in_b", "u_in_cum_b"], how="left"
)
test_feat = test_feat.merge(
    grp_rcts_uinint_uin_ts1,
    on=["R", "C", "ts_r1", "uin_int_cum_b", "u_in_b"],
    how="left",
)

test_feat = test_feat.merge(
    grp_rctsuin_ts1, on=["R", "C", "ts_r1", "u_in_b"], how="left"
)

test_feat = test_feat.merge(grp_ts, on=["R", "C", "ts_r"], how="left")
test_feat = test_feat.merge(grp_rc_uin, on=["R", "C", "u_in_b"], how="left")
test_feat = test_feat.merge(grp_rc, on=["R", "C"], how="left")

pred = (
    test_feat["p_med_0"]
    .fillna(test_feat["p_med_0c"])
    .fillna(test_feat["p_med_1"])
    .fillna(test_feat["p_med_2"])
    .fillna(test_feat["p_med_2c0"])
    .fillna(test_feat["p_med_3"])
    .fillna(test_feat["p_med_4"])
    .fillna(test_feat["p_med_4lag"])
    .fillna(test_feat["p_med_4b"])
    .fillna(test_feat["p_med_4bc0"])
    .fillna(test_feat["p_med_4b2"])
    .fillna(test_feat["p_med_2c"])
    .fillna(test_feat["p_med_4d"])
    .fillna(test_feat["p_med_4c"])
    .fillna(test_feat["p_med_5"])
    .fillna(test_feat["p_med_6b"])
    .fillna(test_feat["p_med_6"])
    .fillna(global_median_snapped)
    .to_numpy(dtype=np.float64)
)

u_out = test_feat["u_out"].to_numpy()
pred = np.where(u_out == 1, global_median_snapped, pred)

idx = np.searchsorted(sorted_pressures, pred, side="left")
idx = np.clip(idx, 1, total_pressures_len - 1)
lower = sorted_pressures[idx - 1]
upper = sorted_pressures[idx]
snapped = np.where(np.abs(pred - lower) < np.abs(upper - pred), lower, upper)

out_df = test_feat[["id"]].copy()
out_df["pressure"] = snapped.astype(np.float64)
out_df = out_df.sort_values("id", kind="mergesort").reset_index(drop=True)

sub["id"] = out_df["id"].values
sub["pressure"] = out_df["pressure"].values
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print(
    "Coverage key tables:",
    "p_med_0",
    float(test_feat["p_med_0"].notna().mean()),
    "p_med_0c",
    float(test_feat["p_med_0c"].notna().mean()),
    "p_med_1",
    float(test_feat["p_med_1"].notna().mean()),
    "p_med_2",
    float(test_feat["p_med_2"].notna().mean()),
    "p_med_2c0",
    float(test_feat["p_med_2c0"].notna().mean()),
    "p_med_3",
    float(test_feat["p_med_3"].notna().mean()),
    "p_med_4",
    float(test_feat["p_med_4"].notna().mean()),
    "p_med_4lag",
    float(test_feat["p_med_4lag"].notna().mean()),
    "p_med_4b",
    float(test_feat["p_med_4b"].notna().mean()),
    "p_med_4bc0",
    float(test_feat["p_med_4bc0"].notna().mean()),
    "p_med_4b2",
    float(test_feat["p_med_4b2"].notna().mean()),
    "p_med_2c",
    float(test_feat["p_med_2c"].notna().mean()),
    "p_med_4d",
    float(test_feat["p_med_4d"].notna().mean()),
    "p_med_4c",
    float(test_feat["p_med_4c"].notna().mean()),
    "p_med_5",
    float(test_feat["p_med_5"].notna().mean()),
    "p_med_6b",
    float(test_feat["p_med_6b"].notna().mean()),
    "p_med_6",
    float(test_feat["p_med_6"].notna().mean()),
)
print("Expiratory fraction in test (u_out==1):", float((u_out == 1).mean()))
