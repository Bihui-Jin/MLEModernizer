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

0.1375885537614762

# 6. Current score

5.3653

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.86265) has done: 'I remove the dependency on external Kaggle datasets (`gb-data-blending-recover`) that are not present, which currently causes the `FileNotFoundError`. To keep the core “blend/median + nearest-pressure snapping” logic intact, I instead generate two lightweight baseline submissions from the provided train/test data and then blend them with the existing `blend()` function. I also make paths robust to your described filesystem by auto-detecting the available `train/test/sample_submission` location under `/kaggle/...` while keeping the relative `../input/...` fallback. Finally, I ensure a valid `submission.csv` is always written with columns `id,pressure`.'
- What this solution (achieved 5.96648) has done: 'Your current score is far above the target (MAE 10.86 vs 0.1376, lower is better), so we need a real—but still minimal and fast—signal-driven predictor rather than constant medians. Keeping your “nearest pressure snapping” and submission-writing logic intact, I add a lightweight, purely pandas-based exact-lookup model: learn the median pressure for each (R, C, time_step_index, u_in, u_out) pattern from train and apply it to test, with safe backoff to (R, C, time_step_index, u_out) and then (R, C) medians. This preserves evaluation semantics and runs quickly within constraints while typically cutting MAE dramatically toward your target band. The blend() call is kept, but now blends a weak baseline with the stronger lookup baseline to reduce risk.'
- What this solution (achieved 7.79434) has done: 'Your current MAE (5.96648, lower is better) is still far from the target (0.1376), so we need a modestly stronger predictor while keeping your core “median lookup + backoff + nearest-pressure snapping + blend” logic unchanged. The main weakness is the expensive/row-wise `(R,C)` fallback computed with `apply`, and the lack of a slightly richer backoff level between `(R,C,ts_idx,u_out)` and `(R,C)` that can capture much of the signal without changing the approach. I (1) replace the slow `(R,C)` apply with a vectorized merge lookup (same semantics), and (2) add one extra median backoff level `(R,C,ts_idx,u_in)` before falling back to `(R,C,ts_idx,u_out)` and `(R,C)`, then keep the same blending and snapping. These are minimal, safe changes that usually reduce MAE materially toward your target while keeping runtime within limits.'
- What this solution (achieved 10.86378) has done: 'I keep your existing “median lookup + backoff + nearest-pressure snapping + blend” approach, but fix the most likely cause of your score regression: the test predictions are not aligned to the sample_submission `id` order (your current `sub_template` comes from sample_submission but you assign `pred` in `df_test` row order, which can silently mismatch and blow up MAE). The minimal, safe change is to explicitly build `sub_b` using the `id` coming from `df_test`, then merge onto the sample `id` list to guarantee correct ordering and 1:1 alignment. I also make the `blend()` output re-ordered to the sample `id` list before saving, so blending cannot reintroduce misalignment. No model/feature/backoff logic is changed—only id alignment and output safety.'
- What this solution (achieved 10.86378) has done: 'Your current MAE (10.86378, lower is better) is far above the target (0.1376), so we should improve predictive signal while keeping your “median lookup + backoff + nearest-pressure snapping + blend” core intact. The largest remaining gap is that your lookups don’t exploit the strong cumulative-volume dynamics; adding a single extra lookup level keyed on cumulative u_in (within breath) is still the same approach (grouped medians + backoff) but usually yields a big MAE drop. I also ensure predictions are forced to 0 during expiratory phase (u_out==1), which matches the competition scoring mask and typically reduces error without changing evaluation semantics. All changes are implemented as additional backoff tables and a final post-processing step; blending, snapping, and submission alignment remain unchanged.'
- What this solution (achieved 4.34201) has done: 'Your current MAE is far worse than the target (10.86 vs 0.1376, lower is better), and the biggest issue is a hard bug: your `id` values are not globally unique (they repeat 1..80 within each breath), so joining on `id` (e.g., `df_test.set_index("id")`) corrupts `u_out` mapping and can also scramble alignment—this can easily explode MAE. I keep your exact “median lookup + backoff + nearest-pressure snapping + blend” logic, but fix joins/order to use the true row identifier (`breath_id`,`ts_idx`) and ensure the final submission’s `pressure` is taken in the exact same row order as `sample_submission.csv`. I also replace slow `.apply(find_nearest)` with a vectorized nearest-pressure snap (same semantics, negligible floating-point differences) to stay well under the time limit and avoid timeouts. These changes are purely alignment/lookup correctness and post-processing; the predictive approach itself is unchanged.'
- What this solution (achieved 5.47345) has done: 'Your current MAE (4.34201, lower is better) is still far above the target (0.13759), so we should improve predictive signal without changing your overall “median lookup + backoff + snapping + blend” approach. The main minimal upgrade is to add a higher-priority lookup level that captures the strong time-series dynamics: median pressure keyed by `(R, C, ts_idx, u_out, u_in_rounded)` (rounding `u_in` to 0.1 to increase match rate), and use it before the existing `(R,C,ts_idx,u_in,u_out)` exact key and other fallbacks. We keep your expiratory post-processing (`u_out==1 -> 0`), nearest-pressure snapping, and the final `id` alignment to `sample_submission.csv` unchanged. This should reduce MAE materially toward the target while remaining fast and stable.'
- What this solution (achieved 5.47345) has done: 'Your current MAE (5.47345, lower is better) is still far above the target (0.13759), so we need a stronger signal while keeping your exact “grouped-median lookup + backoff + u_out post-process + nearest-pressure snapping + blend” approach intact. The smallest high-impact fix is to correct the forced expiratory pressure: setting `u_out==1` to `0` is incorrect for this competition (expiratory phase is *not scored*, but predictions are still submitted and must remain plausible), and it can severely hurt MAE if Kaggle’s scoring mask differs from this assumption. I remove the `u_out==1 -> 0` post-processing (in both the baseline and final submission) while leaving all lookup tables/backoff order/blending unchanged. This should materially reduce error toward your target without changing the core modeling logic.'
- What this solution (achieved 4.00816) has done: 'Your current MAE (5.47345, lower is better) is still far from the target (0.13759), so we need a modest but higher-signal improvement while keeping your grouped-median lookup + backoff + snapping + blend core intact. The smallest high-impact change is to add a new *top-priority* lookup table keyed on `(R, C, ts_idx, u_out, u_in_cum_bin, u_in_r01)` which better captures the breath dynamics without changing the overall approach. I also slightly increase the precision of the cumulative binning (round to 0.01 instead of 0.1) to improve match rate while staying deterministic and fast. Everything else (fallback order, blending, id alignment, and nearest-pressure snapping) is preserved.'
- What this solution (achieved 5.36736) has done: 'Your current MAE (4.00816, lower is better) is still far above the target (0.13759), so we need more match rate in the same “grouped-median lookup + backoff + snapping + blend” framework rather than changing modeling. The most minimal high-signal fix is to add two additional lookup/backoff levels that reduce key sparsity: `(R,C,ts_idx,u_out,u_in_rounded_1)` and `(R,C,ts_idx,u_out,u_in_rounded_0)` (i.e., rounding `u_in` to 0.1 and 1.0), then use them ahead of your weaker fallbacks. This preserves the exact same core logic (median tables + hierarchical backoff + nearest-pressure snapping + blending) while increasing the probability that each test row finds a close historical match. Everything else (paths, alignment to sample ids, snapping, and submission writing) remains unchanged.'
- What this solution (achieved 5.36736) has done: 'Your MAE is far above the target (5.367 vs 0.1376, lower is better), so we need a small, high-signal improvement while keeping your exact “grouped-median lookup + hierarchical backoff + nearest-pressure snapping + blend” framework unchanged. The biggest minimal upgrade is to add a top-priority lookup on the most informative missing dynamic feature: cumulative volume *since the last u_out=1* (“inspiratory-only cumulative u_in”), which better aligns breaths that share the same inspiratory trajectory even if total-breath cumulative differs. This is implemented as one extra grouped-median table and one extra merge, then placed ahead of the existing cum-based keys in the same backoff chain. Everything else (paths, snapping, blending, and submission id alignment) remains the same.'
- What this solution (achieved 5.36736) has done: 'Your current MAE (5.367) is still far above the target (0.1376, lower is better), so we need to improve match quality while keeping the same grouped-median lookup + hierarchical backoff + snapping + blending framework. The smallest high-signal fix is to correct a key precision bug: you currently build `lvl3` on raw `u_in` but merge it using float `u_in` from test, which often fails to match due to float representation; switching this level to use the already-created rounded `u_in_r01` preserves the same logic but greatly increases hit rate. I also fix a small inconsistency where `u_in_r1` was mistakenly rounded to 0.1 (same as `u_in_r01`); making it truly 1.0 rounding strengthens that backoff level without changing the approach. Everything else (feature set, backoff chain structure, snapping, blending, and id alignment) is kept intact.'
- What this solution (achieved 5.3653) has done: 'Your current MAE (5.367) is far above the target (0.1376, lower is better), and the biggest issue in your lookup chain is that the highest-signal exact-key table `lvl1` almost never matches due to float `u_in` join precision. I keep your exact “grouped-median lookup + hierarchical backoff + nearest-pressure snapping + blend” core intact, but make `lvl1` use the already-created `u_in_r01` (0.1 rounding) consistently on both train and test so it actually hits. I also remove the redundant/incorrect `u_in_r0` feature (currently identical to `u_in_r1`) and instead make `u_in_r0` a coarser 10.0 rounding backoff level, which increases match rate without changing the approach. Everything else (paths, id alignment to sample, snapping, blending, and submission writing) remains unchanged.'

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
    return None


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

if TRAIN_PATH is None or TEST_PATH is None or SAMPLE_SUB_PATH is None:
    raise FileNotFoundError(
        f"Could not locate required CSVs. Found TRAIN_PATH={TRAIN_PATH}, TEST_PATH={TEST_PATH}, SAMPLE_SUB_PATH={SAMPLE_SUB_PATH}"
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


def snap_to_nearest_pressure(arr):
    arr = np.asarray(arr, dtype=np.float64)
    idx = np.searchsorted(sorted_pressures, arr, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)

    upper = sorted_pressures[idx]
    lower = sorted_pressures[np.clip(idx - 1, 0, total_pressures_len - 1)]

    choose_lower = np.abs(lower - arr) < np.abs(upper - arr)
    out = np.where(choose_lower, lower, upper)
    return out


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
    output["pressure"] = snap_to_nearest_pressure(output["pressure"].to_numpy())
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.6 + b.pressure * 0.4

    sample_ids = pd.read_csv(SAMPLE_SUB_PATH)[["id"]]
    a = sample_ids.merge(a[["id", "pressure"]], on="id", how="left")
    b = sample_ids.merge(b[["id", "pressure"]], on="id", how="left")

    a["pressure"] = a["pressure"].to_numpy() * 0.6 + b["pressure"].to_numpy() * 0.4

    med = float(np.nanmedian(a["pressure"].to_numpy()))
    a["pressure"] = np.where(
        np.isnan(a["pressure"].to_numpy()), med, a["pressure"].to_numpy()
    )
    a["pressure"] = snap_to_nearest_pressure(a["pressure"].to_numpy())

    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
df_test = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sample_ids = sample_sub[["id"]].copy()

for _df in (df_train, df_test):
    _df["ts_idx"] = _df.groupby("breath_id").cumcount().astype(np.int16)

for _df in (df_train, df_test):
    _df["u_in_cum"] = _df.groupby("breath_id")["u_in"].cumsum().astype(np.float32)
    _df["u_in_cum_bin"] = np.round(_df["u_in_cum"], 2).astype(np.float32)

for _df in (df_train, df_test):
    _df["u_in_r01"] = np.round(_df["u_in"].astype(np.float32), 1).astype(np.float32)

for _df in (df_train, df_test):
    _df["u_in_r1"] = np.round(_df["u_in"].astype(np.float32), 0).astype(np.float32)
    _df["u_in_r0"] = (
        np.round((_df["u_in"].astype(np.float32) / 10.0), 0).astype(np.float32) * 10.0
    )

for _df in (df_train, df_test):
    reset_grp = _df.groupby("breath_id")["u_out"].cumsum()
    _df["u_in_cum_insp"] = (
        _df["u_in"].groupby([_df["breath_id"], reset_grp]).cumsum().astype(np.float32)
    )
    _df["u_in_cum_insp_bin"] = np.round(_df["u_in_cum_insp"], 2).astype(np.float32)

global_median = float(df_train["pressure"].median())

sub_a = sample_ids.copy()
sub_a["pressure"] = snap_to_nearest_pressure(
    np.full(len(sub_a), global_median, dtype=np.float64)
)
a_path = "baseline_global_median.csv"
sub_a.to_csv(a_path, index=False)

rc_median_df = (
    df_train.groupby(["R", "C"], sort=False)["pressure"]
    .median()
    .rename("prc")
    .reset_index()
)

lvl3 = (
    df_train.groupby(["R", "C", "ts_idx", "u_in_r01"], sort=False)["pressure"]
    .median()
    .rename("p3")
    .reset_index()
)

lvl2 = (
    df_train.groupby(["R", "C", "ts_idx", "u_out"], sort=False)["pressure"]
    .median()
    .rename("p2")
    .reset_index()
)

lvl1 = (
    df_train.groupby(["R", "C", "ts_idx", "u_in_r01", "u_out"], sort=False)["pressure"]
    .median()
    .rename("p1")
    .reset_index()
)

lvl_cum = (
    df_train.groupby(["R", "C", "ts_idx", "u_out", "u_in_cum_bin"], sort=False)[
        "pressure"
    ]
    .median()
    .rename("pcum")
    .reset_index()
)

lvl_uin_r01 = (
    df_train.groupby(["R", "C", "ts_idx", "u_out", "u_in_r01"], sort=False)["pressure"]
    .median()
    .rename("pr01")
    .reset_index()
)

lvl_cum_r01 = (
    df_train.groupby(
        ["R", "C", "ts_idx", "u_out", "u_in_cum_bin", "u_in_r01"], sort=False
    )["pressure"]
    .median()
    .rename("pcumr01")
    .reset_index()
)

lvl_uin_r1 = (
    df_train.groupby(["R", "C", "ts_idx", "u_out", "u_in_r1"], sort=False)["pressure"]
    .median()
    .rename("pr1")
    .reset_index()
)
lvl_uin_r0 = (
    df_train.groupby(["R", "C", "ts_idx", "u_out", "u_in_r0"], sort=False)["pressure"]
    .median()
    .rename("pr0")
    .reset_index()
)

lvl_cum_insp_r01 = (
    df_train.groupby(
        ["R", "C", "ts_idx", "u_out", "u_in_cum_insp_bin", "u_in_r01"], sort=False
    )["pressure"]
    .median()
    .rename("pcum_insp_r01")
    .reset_index()
)

test_keys = df_test[
    [
        "breath_id",
        "ts_idx",
        "id",
        "R",
        "C",
        "u_in",
        "u_out",
        "u_in_cum_bin",
        "u_in_cum_insp_bin",
        "u_in_r01",
        "u_in_r1",
        "u_in_r0",
    ]
].copy()

test_keys = (
    test_keys.merge(
        lvl_cum_insp_r01,
        on=["R", "C", "ts_idx", "u_out", "u_in_cum_insp_bin", "u_in_r01"],
        how="left",
        copy=False,
    )
    .merge(
        lvl_cum_r01,
        on=["R", "C", "ts_idx", "u_out", "u_in_cum_bin", "u_in_r01"],
        how="left",
        copy=False,
    )
    .merge(
        lvl_uin_r01,
        on=["R", "C", "ts_idx", "u_out", "u_in_r01"],
        how="left",
        copy=False,
    )
    .merge(
        lvl_uin_r1,
        on=["R", "C", "ts_idx", "u_out", "u_in_r1"],
        how="left",
        copy=False,
    )
    .merge(
        lvl_uin_r0,
        on=["R", "C", "ts_idx", "u_out", "u_in_r0"],
        how="left",
        copy=False,
    )
    .merge(lvl1, on=["R", "C", "ts_idx", "u_in_r01", "u_out"], how="left", copy=False)
    .merge(
        lvl_cum,
        on=["R", "C", "ts_idx", "u_out", "u_in_cum_bin"],
        how="left",
        copy=False,
    )
    .merge(lvl3, on=["R", "C", "ts_idx", "u_in_r01"], how="left", copy=False)
    .merge(lvl2, on=["R", "C", "ts_idx", "u_out"], how="left", copy=False)
    .merge(rc_median_df, on=["R", "C"], how="left", copy=False)
)

pcum_insp_r01 = test_keys["pcum_insp_r01"].to_numpy()
pcumr01 = test_keys["pcumr01"].to_numpy()
pr01 = test_keys["pr01"].to_numpy()
pr1 = test_keys["pr1"].to_numpy()
pr0 = test_keys["pr0"].to_numpy()
p1 = test_keys["p1"].to_numpy()
pcum = test_keys["pcum"].to_numpy()
p3 = test_keys["p3"].to_numpy()
p2 = test_keys["p2"].to_numpy()
prc = test_keys["prc"].to_numpy()

pred = np.where(
    ~np.isnan(pcum_insp_r01),
    pcum_insp_r01,
    np.where(
        ~np.isnan(pcumr01),
        pcumr01,
        np.where(
            ~np.isnan(pr01),
            pr01,
            np.where(
                ~np.isnan(pr1),
                pr1,
                np.where(
                    ~np.isnan(pr0),
                    pr0,
                    np.where(
                        ~np.isnan(p1),
                        p1,
                        np.where(
                            ~np.isnan(pcum),
                            pcum,
                            np.where(
                                ~np.isnan(p3), p3, np.where(~np.isnan(p2), p2, prc)
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ),
)
pred = np.where(np.isnan(pred), global_median, pred)

sub_b = pd.DataFrame(
    {"id": test_keys["id"].astype(np.int64).to_numpy(), "pressure": pred}
)

sub_b["pressure"] = snap_to_nearest_pressure(sub_b["pressure"].to_numpy())

sub_b = sample_ids.merge(sub_b, on="id", how="left")
sub_b["pressure"] = np.where(
    sub_b["pressure"].isna().to_numpy(),
    global_median,
    sub_b["pressure"].to_numpy(dtype=np.float64),
)
sub_b["pressure"] = snap_to_nearest_pressure(sub_b["pressure"].to_numpy())

b_path = "baseline_lookup_median.csv"
sub_b.to_csv(b_path, index=False)

_ = blend(a_path, b_path)

final_sub = pd.read_csv("blend.csv")[["id", "pressure"]]
final_sub = sample_ids.merge(final_sub, on="id", how="left")

final_sub["pressure"] = np.where(
    final_sub["pressure"].isna().to_numpy(),
    global_median,
    final_sub["pressure"].to_numpy(dtype=np.float64),
)

final_pressure = final_sub["pressure"].to_numpy(dtype=np.float64)
final_pressure = snap_to_nearest_pressure(final_pressure)

final_sub["pressure"] = final_pressure
final_sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", final_sub.shape)
print(final_sub.head())
