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

0.149921233083101

# 6. Current score

1.98478

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.92376) has done: 'Your notebook currently fails because it tries to read external blend files that don’t exist in this Kaggle environment. I keep your existing blending/core logic intact, but add a safe fallback that generates a simple, valid baseline submission directly from `test.csv` when those files are missing. I also fix the cell numbering (start at cell 1) and ensure the script always writes a submission with a `.csv` suffix and the required `id,pressure` columns. This run end-to-end and yield a valid submission file without relying on unavailable datasets.'
- What this solution (achieved 10.67678) has done: 'Your fallback baseline is too coarse because it ignores the most informative control signal (`u_in`) and merges on exact `time_step`, which causes many misses and heavy fallback to the global median—this drives the MAE up. I keep your overall logic (train-derived lookup + nearest-pressure snapping) but make the lookup much more specific by adding a lightly-quantized `u_in` into the grouping keys while keeping `time_step` exact, so most test rows find a reasonable median pressure. This is a minimal change that should substantially reduce error without changing the overall approach, and it still writes a valid `submission.csv`. I also ensure the output ordering by `id` is preserved after the merge.'
- What this solution (achieved 3.80664) has done: 'Your fallback is still too inaccurate because it relies on exact `time_step` matching, which causes many misses and forces a global-median fallback that explodes MAE. I keep the same “train-derived median lookup + nearest-pressure snapping” core logic, but make matching robust by lightly quantizing `time_step` (in addition to the existing `u_in` bin) and adding a two-stage fallback: first drop `u_out` if needed, then fall back to global median. This should move your MAE substantially toward the target without changing the overall approach or any modeling/training loops (there are none). I also fix the cell numbering to start at 1 while preserving the original order and ensure the submission is aligned and sorted by `id`.'
- What this solution (achieved 3.77847) has done: 'Your current fallback misses the key evaluation detail: only inspiratory phase (`u_out==0`) is scored, so predicting with the same lookup for expiratory rows injects avoidable error. I keep your exact “train median lookup + nearest-pressure snapping” logic, but set predictions for `u_out==1` test rows to a safe constant (0), while leaving `u_out==0` rows using your lookup; this should reduce MAE materially toward the target with minimal change. I also make the global-median fallback computed on inspiratory rows only (consistent with the metric) and keep output sorted by `id` with the required columns. The blend path is untouched if the external files exist.'
- What this solution (achieved 3.76974) has done: 'Your current fallback is still leaving a lot of inspiratory (`u_out==0`) rows unmatched/poorly matched, so the prediction collapses to a coarse median too often. I keep the exact same core logic (train median lookup + nearest-pressure snapping + constant for `u_out==1`) but make the lookup denser by adding a third-stage fallback that drops the `u_in` bin (keeping `R,C,time_bin`) before going to the global median. I also compute the expiratory constant as the train median pressure during `u_out==1` (still unscored, but typically closer to reality than hard 0 and won’t hurt MAE because those rows are ignored). These are minimal, safe changes that should reduce MAE materially toward your target without changing the overall approach.'
- What this solution (achieved 2.21969) has done: 'Your current fallback lookup is still too sparse because it bins `u_in` uniformly; most inspiratory pressure variation is driven by *cumulative* delivered volume/flow, so matching on instantaneous `u_in` misses many train–test correspondences and collapses to coarse medians. I keep your exact core approach (train-derived median lookup with staged fallbacks + nearest-pressure snapping + constant for `u_out==1`) but add one additional, lightweight feature: per-breath cumulative `u_in` (and a small binning of it) to make lookups denser and more physically aligned with pressure dynamics. This is a minimal change (no model/training loops added) and should reduce MAE materially toward your target. I also ensure the merge keys remain int-typed and submission remains sorted by `id`.'
- What this solution (achieved 2.21969) has done: 'Your current fallback is still far from the target MAE, so we should make the lookup closer to the metric and to the true pressure dynamics without changing the overall approach (train-derived medians + staged fallbacks + nearest-pressure snapping). The biggest minimal gain is to align medians to the *inspiratory-only scoring* by using a `u_out==0` key in all inspiratory lookups (your `grp_cols_2/3` currently mix phases and dilute medians), and to make the “cumulative delivered air” feature more informative by computing cumulative `u_in` over the full breath (not just inspiratory rows) to match test-time behavior. These changes keep the same logic and are small, but should reduce mismatches and move MAE materially toward your target. The blend path remains unchanged when those external files exist, and the script still always writes a valid `submission.csv`.'
- What this solution (achieved 2.9982) has done: 'Your current lookup still loses a lot of signal because it bins cumulative `u_in` on raw sums, which vary substantially by breath length/shape and makes the median tables sparse; this forces too many fallbacks to coarse medians and keeps MAE high. I keep the exact same “train median lookup + staged fallbacks + nearest-pressure snapping + constant for `u_out==1`” approach, but switch the cumulative feature to a time-normalized cumulative mean (`cum_uin / (t+eps)`) and include it only in the most specific (first) lookup stage, improving match rates without changing the overall method. I also remove a redundant duplicate stage (`grp_cols_2` was identical to `grp_cols_1`) so the intended staged fallback actually becomes progressively less specific. These are minimal, metric-aligned changes that should reduce MAE toward your target while preserving the core logic and still writing a valid `submission.csv`.'
- What this solution (achieved 2.12277) has done: 'Your current score (2.9982 MAE) is far worse than the target (0.1499), so we need a meaningful but still “same-core-logic” improvement: keep the train-median lookup + staged fallbacks + nearest-pressure snapping, but make the lookup keys much closer to true pressure dynamics. The smallest strong fix is to replace the unstable `cum_uin/time` feature with a more physically-aligned feature: per-breath cumulative inspired volume proxy (`cum_uin * dt`) and its bin, used only in the most specific stage so fallbacks remain intact. I also correct a subtle bug: you compute `_cum_uin` on unsliced data but then slice to inspiratory; instead we compute cumulative features consistently on the full breath and only then restrict the median tables to `u_out==0`, improving match density without changing the method. Submission writing, column names, sorting by `id`, and the optional external blend path are preserved.'
- What this solution (achieved 1.88573) has done: 'Your current fallback is still too “lookup-sparse” on inspiratory rows, so it falls back to coarse medians too often and keeps MAE high. I keep the exact same core approach (train median lookup with staged fallbacks + nearest-pressure snapping + constant prediction for `u_out==1`) but make the most-specific key more matchable by using an additional cumulative-volume proxy that is **normalized by breath progress** (so shapes align better) and by using slightly coarser bins to reduce missing merges. I also ensure that the inspiratory lookup tables are built with `u_out` fixed to 0 (so keys are consistent) while keeping `u_out` in the test merge keys to preserve the expiratory constant override. These are minimal, metric-aligned tweaks that should reduce the fallback rate and move MAE downward toward your target without changing the overall method.'
- What this solution (achieved 1.91648) has done: 'Your current score (1.88573 MAE) is still far above the target (0.1499), so we need a small but meaningful accuracy gain without changing the overall “train-median lookup + staged fallbacks + nearest-pressure snapping + constant for `u_out==1`” approach. The main issue is that the lookup keys still don’t capture enough breath-state, so many inspiratory rows fall back to coarse medians; I add one more physically-aligned per-breath feature (normalized cumulative volume progress) and use it only in the most-specific stage to improve match rate while keeping the same staged fallback structure. I also ensure the inspiratory median tables use `u_out=0` consistently (fixed in the tables) while the test merge still includes `u_out` so the expiratory constant override stays intact. All file paths remain unchanged and the script still always write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 1.8682) has done: 'Your current fallback is still too sparse on inspiratory rows, so many test points fall back to coarse medians, keeping MAE far above the target. I keep the exact same core “train-derived median lookup with staged fallbacks + nearest-pressure snapping + constant for `u_out==1`” approach, but make the most-specific stage match better by adding one extra breath-state key: **previous pressure median** via a binned `u_in` lag (computed in both train/test, no label leakage). I also make inspiratory lookup keys consistent by forcing `u_out=0` inside the train median tables (while still merging on test `u_out` so the expiratory-constant override remains unchanged). These are minimal, metric-aligned changes intended to reduce fallback rate and move MAE down toward your target without changing overall semantics.'
- What this solution (achieved 2.04014) has done: 'Your current MAE (1.8682, lower is better) is still far above the target (0.1499), so we need a small change that materially improves accuracy without changing your core “train-median lookup + staged fallbacks + nearest-pressure snapping + constant for `u_out==1`” approach. The biggest remaining avoidable error is that the lookup doesn’t include any notion of breath “memory”: pressure depends strongly on past delivered air and valve history, not just instantaneous bins. I keep your exact staged-merge structure, but add two minimal, non-leaky state keys computed for both train/test: binned cumulative `u_in*dt` lag (previous cumulative volume) and binned cumulative `u_out*dt` (how long exhalation has been open), used only in the most-specific stage to increase match quality while leaving fallbacks intact. This should reduce inspiratory fallback-to-coarse-median frequency and move MAE downward toward the target, while still producing a valid `submission.csv`.'
- What this solution (achieved 1.98478) has done: 'Your current gap to target is still large (MAE 2.04014 vs 0.1499, lower is better), so we need a meaningful-but-minimal improvement without changing your core “train-median lookup with staged fallbacks + nearest-pressure snapping + constant for u_out==1” approach. The biggest remaining issue is that the most-specific lookup is likely too sparse/misaligned because it bins *rate-like* features with unstable denominators and mixes in an unhelpful cumulative u_out duration for inspiratory rows (which is almost always ~0), causing frequent fallback to coarse medians. I keep your staged merges intact, but (1) replace `_cum_flow_norm` with a more stable, physics-aligned `cum_vol / (cum_dt+eps)` (average flow so far), (2) drop `_cum_uout_dt_bin` from the most-specific key (since train_insp has u_out=0, it carries little signal and increases sparsity), and (3) add one extra intermediate fallback stage that uses the new average-flow bin and volume-progress bin without lag bins to recover matches before collapsing to coarse medians. These changes directly target match-rate/median quality on inspiratory rows and should move MAE down toward your target while preserving your overall method and still writing a valid `submission.csv`.'

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
        weight1 = (l[1] / l_sum) + 0.15
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

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output.pressure = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.6 + b.pressure * 0.4
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
a = "../input/gb-data-blending-recover/0.147 blend.csv"
b = "../input/gb-data-blending-recover/0.149 blend.csv"

submission_path = "submission.csv"

if os.path.exists(a) and os.path.exists(b):
    sub = blend(a, b)
    sub[["id", "pressure"]].to_csv(submission_path, index=False)
else:
    df_test = pd.read_csv(
        "../input/ventilator-pressure-prediction/test.csv",
        usecols=["id", "breath_id", "R", "C", "time_step", "u_out", "u_in"],
    )

    time_bin = 0.02
    uin_bin = 1.0

    avg_flow_bin = 0.6

    vol_prog_bin = 0.03  # coarse to avoid sparsity
    uin_lag_bin = 1.0
    cumvol_lag_bin = 0.2  # coarse bin to keep tables dense

    eps = 1e-6

    df_train_sorted = df_train.sort_values(["breath_id", "time_step"]).copy()
    df_test_sorted = df_test.sort_values(["breath_id", "time_step"]).copy()

    def add_dt_and_features(df):
        df["_dt"] = (
            df.groupby("breath_id", observed=True)["time_step"].diff().fillna(0.0)
        )
        df["_dt"] = df["_dt"].clip(lower=0.0)

        df["_u_in_x_dt"] = df["u_in"] * df["_dt"]
        df["_cum_vol"] = df.groupby("breath_id", observed=True)["_u_in_x_dt"].cumsum()

        df["_cum_dt"] = df.groupby("breath_id", observed=True)["_dt"].cumsum()
        df["_avg_flow"] = df["_cum_vol"] / (df["_cum_dt"] + eps)

        df["_max_cum_vol"] = df.groupby("breath_id", observed=True)[
            "_cum_vol"
        ].transform("max")
        df["_vol_progress"] = df["_cum_vol"] / (df["_max_cum_vol"] + eps)

        df["_u_in_lag"] = (
            df.groupby("breath_id", observed=True)["u_in"].shift(1).fillna(0.0)
        )

        df["_cum_vol_lag"] = (
            df.groupby("breath_id", observed=True)["_cum_vol"].shift(1).fillna(0.0)
        )

        return df

    df_train_sorted = add_dt_and_features(df_train_sorted)
    df_test_sorted = add_dt_and_features(df_test_sorted)

    df_train_insp = df_train_sorted[df_train_sorted["u_out"] == 0].copy()
    df_test = df_test_sorted  # keep name used below

    df_train_insp["_t_bin"] = (
        (df_train_insp["time_step"] / time_bin).round().astype(np.int16)
    )
    df_test["_t_bin"] = (df_test["time_step"] / time_bin).round().astype(np.int16)

    df_train_insp["_u_in_bin"] = (
        (df_train_insp["u_in"] / uin_bin).round().astype(np.int16)
    )
    df_test["_u_in_bin"] = (df_test["u_in"] / uin_bin).round().astype(np.int16)

    df_train_insp["_avg_flow_bin"] = (
        (df_train_insp["_avg_flow"] / avg_flow_bin).round().astype(np.int16)
    )
    df_test["_avg_flow_bin"] = (
        (df_test["_avg_flow"] / avg_flow_bin).round().astype(np.int16)
    )

    df_train_insp["_vol_prog_bin"] = (
        (df_train_insp["_vol_progress"] / vol_prog_bin).round().astype(np.int16)
    )
    df_test["_vol_prog_bin"] = (
        (df_test["_vol_progress"] / vol_prog_bin).round().astype(np.int16)
    )

    df_train_insp["_u_in_lag_bin"] = (
        (df_train_insp["_u_in_lag"] / uin_lag_bin).round().astype(np.int16)
    )
    df_test["_u_in_lag_bin"] = (
        (df_test["_u_in_lag"] / uin_lag_bin).round().astype(np.int16)
    )

    df_train_insp["_cumvol_lag_bin"] = (
        (df_train_insp["_cum_vol_lag"] / cumvol_lag_bin).round().astype(np.int16)
    )
    df_test["_cumvol_lag_bin"] = (
        (df_test["_cum_vol_lag"] / cumvol_lag_bin).round().astype(np.int16)
    )

    df_train_insp["u_out"] = 0

    grp_cols_0 = [
        "R",
        "C",
        "_t_bin",
        "u_out",
        "_u_in_bin",
        "_u_in_lag_bin",
        "_avg_flow_bin",
        "_vol_prog_bin",
        "_cumvol_lag_bin",
    ]
    med0 = (
        df_train_insp.groupby(grp_cols_0, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_pred"})
    )
    out = df_test.merge(med0, on=grp_cols_0, how="left")

    grp_cols_1 = ["R", "C", "_t_bin", "u_out", "_u_in_bin", "_u_in_lag_bin"]
    med1 = (
        df_train_insp.groupby(grp_cols_1, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_pred1"})
    )
    out = out.merge(med1, on=grp_cols_1, how="left")
    out["pressure_pred"] = out["pressure_pred"].fillna(out["pressure_pred1"])

    grp_cols_1b = [
        "R",
        "C",
        "_t_bin",
        "u_out",
        "_u_in_bin",
        "_avg_flow_bin",
        "_vol_prog_bin",
    ]
    med1b = (
        df_train_insp.groupby(grp_cols_1b, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_pred1b"})
    )
    out = out.merge(med1b, on=grp_cols_1b, how="left")
    out["pressure_pred"] = out["pressure_pred"].fillna(out["pressure_pred1b"])

    grp_cols_2 = ["R", "C", "_t_bin", "u_out", "_u_in_bin"]
    med2 = (
        df_train_insp.groupby(grp_cols_2, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_pred2"})
    )
    out = out.merge(med2, on=grp_cols_2, how="left")
    out["pressure_pred"] = out["pressure_pred"].fillna(out["pressure_pred2"])

    grp_cols_3 = ["R", "C", "_t_bin", "u_out"]
    med3 = (
        df_train_insp.groupby(grp_cols_3, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "pressure_pred3"})
    )
    out = out.merge(med3, on=grp_cols_3, how="left")
    out["pressure_pred"] = out["pressure_pred"].fillna(out["pressure_pred3"])

    global_med = float(df_train_insp["pressure"].median())
    out["pressure_pred"] = out["pressure_pred"].fillna(global_med)

    df_train_exp = df_train[df_train["u_out"] == 1]
    exp_const = float(df_train_exp["pressure"].median()) if len(df_train_exp) else 0.0
    out.loc[out["u_out"] == 1, "pressure_pred"] = exp_const

    out["pressure"] = out["pressure_pred"].apply(find_nearest)

    submission = out[["id", "pressure"]].sort_values("id").copy()
    submission.to_csv(submission_path, index=False)

print(f"Wrote {submission_path} with {sum(1 for _ in open(submission_path)) - 1} rows.")
