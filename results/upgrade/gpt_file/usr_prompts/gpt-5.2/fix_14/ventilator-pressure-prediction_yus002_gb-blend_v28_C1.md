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

3.9

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

0.1748488413026785

# 6. Current score

3.18088

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.2068) has done: 'I remove the dependency on missing external “gb-blending” files that causes the `FileNotFoundError` and instead generate a valid submission directly from the provided competition data. To keep changes minimal and score moving toward the target (lower is better) without introducing new modeling complexity, I implement a simple deterministic baseline prediction: use the mean inspiratory-phase pressure per (R, C, time_step) from train, and fall back to global means when unseen. This preserves evaluation semantics (predict pressure per row id) and produces a correctly formatted `submission.csv` end-to-end. I also keep the existing blending utilities but make them safe (only run if files exist) so the notebook never crashes.'
- What this solution (achieved 5.9754) has done: 'Your current baseline is hurt mainly by two avoidable issues: (1) it forces all `u_out==1` predictions to 0 even though those rows are not scored (and this can still indirectly hurt MAE if any inspiratory rows are mis-flagged or due to alignment quirks), and (2) grouping by raw float `time_step` is brittle—small representation differences can cause many “misses” and trigger fallback to coarse means. I keep the same core idea (deterministic lookup of mean pressure from train) but make the time key robust by converting `time_step` to an integer index (rounded milliseconds) before grouping/reindexing, which reduces unnecessary NaNs and should move MAE substantially down toward the target. I also stop overriding `u_out==1` to zero (leave the learned mean), since those rows are unscored and this avoids injecting a large arbitrary value shift. The pipeline still runs end-to-end and writes a valid `submission.csv` with the correct schema.'
- What this solution (achieved 4.23645) has done: 'Your current approach is a deterministic lookup baseline; the biggest remaining avoidable error vs the target is that you average pressure across *all* inspiratory time-steps regardless of the actual control input `u_in`, which strongly influences pressure. To move the MAE down toward the target while preserving the same core “mean lookup from train” logic, I extend the grouping key to include a robustly-binned `u_in` (rounded to 0.1) and keep your robust `time_idx` key. I also add a safe hierarchical fallback (R,C,time,u_in_bin) → (R,C,time) → (R,C,u_in_bin) → (R,C) → global to avoid NaNs without changing evaluation semantics. Everything still runs end-to-end and writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 4.08499) has done: 'Your current score is far above the target (lower is better), so we should make a small, safe improvement that preserves your deterministic “mean-lookup” core logic. The biggest remaining mismatch is that pressure depends strongly on recent history (e.g., integrated flow), not just the instantaneous `(R,C,time,u_in)`; we can capture that with a minimal feature addition: cumulative `u_in` within each breath (a proxy for volume) and keep the same hierarchical mean reindex/fallback. This keeps the exact same approach (groupby means + reindex on test keys) and should reduce MAE meaningfully while staying well within Kaggle constraints and runtime. We also keep the submission alignment via `sample_submission.csv` unchanged and still write a valid `submission.csv`.'
- What this solution (achieved 3.93128) has done: 'Your baseline is still a pure mean-lookup, but it likely misses because the test `cum_u_in` is computed as a simple cumsum of `u_in` without accounting for uneven `time_step` spacing; using a time-weighted integral (volume proxy) is a minimal, core-logic-preserving upgrade that should reduce MAE toward the target. I replace `cum_u_in` with `cum_u_in_dt = cumsum(u_in * delta_time)` (per breath) for both train/test, and build the same hierarchical mean tables using a binned version of this integrated feature. I also widen the binning slightly (0.01 resolution) to reduce collisions/over-sparsity while keeping the same lookup-and-fallback structure. The submission alignment and format remain unchanged, and the script still runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 3.87532) has done: 'Your current score (3.931) is far worse than the target (0.175), so we should make a small, safe improvement to the same deterministic “groupby mean lookup + hierarchical fallback” approach. The biggest avoidable error is that we bin `cum_uin_dt` on an absolute scale that differs systematically across breaths and across (R,C), causing many key misses and over-reliance on coarse fallbacks; we can fix this by adding a *within-breath normalized* integrated-volume proxy (`cum_uin_dt / total_uin_dt`) and using it in the main lookup key. This keeps the exact same core logic (mean tables + reindex + fallbacks), but makes the “history/volume” feature comparable across breaths and improves hit-rate on the most specific table, which should move MAE down toward the target. I also keep the rest of your pipeline unchanged (paths, inspiratory-only training aggregation, submission alignment).'
- What this solution (achieved 2.51204) has done: 'Your current lookup keys are still too sparse because `cum_uin_dt_norm_bin` at 0.001 resolution creates many unseen combinations in test, forcing frequent fallbacks and keeping MAE high. To move the score down toward the target with minimal change to your existing “groupby mean lookup + hierarchical fallback” logic, I only (1) coarsen the normalized integrated-volume bin (better hit-rate) and (2) add one extra intermediate fallback that uses this normalized history without the exact `u_in` bin. Everything else (data loading, inspiratory-only training aggregation, deterministic mean tables, reindexing, and submission writing) stays the same and still produces a valid `submission.csv`.'
- What this solution (achieved 2.38941) has done: 'Your score (2.512) is still far above the target (0.175, lower is better), so we should make a small, low-risk improvement within your existing deterministic “groupby mean lookup + hierarchical fallback” framework. The least invasive win is to make the history feature less sparse by (1) slightly coarsening the `uin_bin` resolution so more test keys hit the most specific mean tables, and (2) adding one more intermediate fallback that uses `(R,C,uin_bin,cum_norm_bin)` without `time_idx` to recover misses where timing alignment differs. This preserves the same core semantics (inspiratory-only mean lookup, hierarchical reindex, deterministic predictions) and should reduce MAE by improving hit-rate without changing the overall approach. The script still runs end-to-end and writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 2.2496) has done: 'We keep your deterministic mean-lookup + hierarchical fallback pipeline intact, but reduce “key sparsity” a bit so more test rows hit specific mean tables instead of falling back to coarse averages (which should lower MAE toward the 0.175 target). Concretely, we (1) slightly coarsen the normalized integrated-volume binning (from 1% steps to 2% steps) and (2) add one extra intermediate fallback using `(R,C,time_idx,cum_norm_bin)` before dropping `time_idx`, which helps when `u_in` bin misses but the breath-progress proxy matches. These are minimal, low-risk changes that preserve your core logic and evaluation semantics and should move the score down from ~2.39. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 2.1784) has done: 'We keep your deterministic groupby-mean lookup + hierarchical fallback exactly as-is, but reduce key-miss rate by making the within-breath progress bin slightly coarser (less sparsity) and adding one more “time-aware but u_in-agnostic” fallback that uses `(R,C,time_idx,cum_norm_bin)` earlier/more reliably. This should shift more rows away from coarse `(R,C)`/global fallbacks and move MAE downward toward the 0.175 target without changing the overall method. We also ensure the submission alignment remains strictly by `sample_submission.csv` ids and still write `submission.csv` end-to-end.'
- What this solution (achieved 2.1846) has done: 'Your current score is still far above the target (lower is better), so we should make a small, safe improvement that keeps your deterministic “groupby mean lookup + hierarchical fallback” core logic intact while reducing key-miss rate. The least invasive lever is to make the main history bin (`cum_uin_dt_norm_bin`) a bit coarser so more test rows hit the most specific lookup tables instead of falling back to coarse means. To avoid introducing bias from rounding artifacts, we switch from `.round()` to a consistent floor-based binning for the normalized progress feature (still deterministic and same semantics). Everything else (features, inspiratory-only aggregation, fallback chain, submission alignment/format) remains unchanged and the script still writes a valid `submission.csv`.'
- What this solution (achieved 1.99748) has done: 'Your current deterministic mean-lookup is still dominated by “key misses” that force fallbacks to coarse (R,C) or global means, keeping MAE far from the target. The smallest safe lever is to make the main history/progress key less sparse by coarsening `uin_bin` slightly so more test rows match the most specific tables, without changing the overall approach (groupby mean tables + hierarchical reindex fallbacks). I keep all features, inspiratory-only aggregation, and the fallback chain intact, only adjusting `to_uin_bin()` resolution to improve hit-rate. The script still run end-to-end and write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 3.18088) has done: 'Your current score (1.997, lower is better) is still far above the target (0.175), so we should make a small, low-risk improvement within the exact same “deterministic groupby-mean lookup + hierarchical fallback” framework. The biggest remaining avoidable error is that the integrated-volume progress feature is computed over the entire breath, but the metric only scores inspiratory phase; switching the normalization to be inspiratory-only makes the progress key better aligned with what is scored and should reduce MAE. To keep changes minimal, we only adjust how `cum_uin_dt_norm` is computed (using u_out==0 totals and freezing after expiration) and keep the same binning, grouping keys, fallback chain, paths, and submission alignment. This should increase hit quality in the most specific tables without changing the overall approach or adding any new model/training.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import copy
import glob
import random
from random import random as rd




## === cell 1
def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    """
    Original utility: weighted combine of 1-2 submission files using weights derived from filename.
    Kept for compatibility; not used in the fixed end-to-end pipeline.
    """
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
        weight1 = l[1] / l_sum
        weight2 = 1 - weight1
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def g(dp):
    """
    Original utility: random weight blending across many files in a directory.
    Kept for compatibility; not used in the fixed end-to-end pipeline.
    """
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)
    file_count = len(l)
    loop_time = file_count**7
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
    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output.pressure = 0
    for i in range(loop_time):
        weight = []
        set_seed(i)
        for j in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        for j in range(len(weight)):
            weight[j] /= weight_sum
        weight.sort(reverse=True)
        for j in range(len(flist)):
            output.pressure += flist[j] * weight[j]
    output.pressure /= loop_time
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)




## === cell 2
def blend(a, b, out_path="blend.csv", w_a=0.65, w_b=0.35):
    """
    Safe blender: only blends if both paths exist; otherwise raises a clear error.
    Kept for compatibility; not required for the fixed pipeline.
    """
    if not os.path.exists(a):
        raise FileNotFoundError(f"Missing file for blend(): {a}")
    if not os.path.exists(b):
        raise FileNotFoundError(f"Missing file for blend(): {b}")
    a_df = pd.read_csv(a)
    b_df = pd.read_csv(b)
    a_df["pressure"] = a_df["pressure"] * w_a + b_df["pressure"] * w_b
    a_df.to_csv(out_path, index=False)
    return a_df




## === cell 3
set_seed(2021)

CANDIDATE_ROOTS = [
    "/kaggle/input/ventilator-pressure-prediction",
    "/kaggle/data/ventilator-pressure-prediction",
    "/kaggle/input",
    "/kaggle/data",
]
DATA_ROOT = None
for r in CANDIDATE_ROOTS:
    if os.path.exists(os.path.join(r, "train.csv")) and os.path.exists(
        os.path.join(r, "test.csv")
    ):
        DATA_ROOT = r
        break
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate train.csv/test.csv under expected /kaggle/input or /kaggle/data paths."
    )

train_path = os.path.join(DATA_ROOT, "train.csv")
test_path = os.path.join(DATA_ROOT, "test.csv")
sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")

usecols_train = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
usecols_test = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]

train = pd.read_csv(train_path, usecols=usecols_train)
test = pd.read_csv(test_path, usecols=usecols_test)
sub = pd.read_csv(sub_path, usecols=["id"])


def to_time_idx(ts: pd.Series) -> pd.Series:
    return (ts.astype(np.float64) * 1000.0).round().astype(np.int16)


def to_uin_bin(u: pd.Series) -> pd.Series:
    """
    Keep current setting (0.5 resolution) since it improved hit-rate previously.
    """
    return (u.astype(np.float64) * 2.0).round().astype(np.int16)  # 0.5 resolution


def add_integrated_uin_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["breath_id", "time_step"], kind="mergesort")
    dt = df.groupby("breath_id", sort=False)["time_step"].diff().astype(np.float64)
    dt = dt.fillna(0.0).clip(lower=0.0)
    df["dt"] = dt
    df["uin_dt"] = df["u_in"].astype(np.float64) * df["dt"]
    df["cum_uin_dt"] = df.groupby("breath_id", sort=False)["uin_dt"].cumsum()
    return df


def add_insp_normalized_progress(df: pd.DataFrame) -> pd.DataFrame:
    """
    Change to move score toward target (lower is better), while preserving core logic:
    - Metric scores inspiratory phase only (u_out==0).
    - Normalize integrated-volume progress using inspiratory-only total per breath.
    - Freeze progress after expiration begins so expiratory rows don't distort the key.
    This keeps the same deterministic mean-lookup approach, only improving feature alignment.
    """
    df = df.sort_values(["breath_id", "time_step"], kind="mergesort")

    insp_mask = df["u_out"].to_numpy() == 0
    uin_dt = df["uin_dt"].to_numpy(dtype=np.float64, copy=False)

    uin_dt_insp = np.where(insp_mask, uin_dt, 0.0)
    df["cum_uin_dt_insp"] = (
        pd.Series(uin_dt_insp, index=df.index)
        .groupby(df["breath_id"], sort=False)
        .cumsum()
        .astype(np.float64)
    )

    total_uin_dt_insp = (
        pd.Series(uin_dt_insp, index=df.index)
        .groupby(df["breath_id"], sort=False)
        .transform("sum")
        .astype(np.float64)
    )
    total_uin_dt_insp = total_uin_dt_insp.replace(0.0, np.nan)

    prog = (df["cum_uin_dt_insp"] / total_uin_dt_insp).fillna(0.0).clip(0.0, 1.0)

    prog_frozen = (
        prog.groupby(df["breath_id"], sort=False).ffill().fillna(0.0).clip(0.0, 1.0)
    )
    df["cum_uin_dt_norm"] = prog_frozen.astype(np.float64)
    return df


def to_cum_uin_dt_norm_bin(cum_u_dt_norm: pd.Series) -> pd.Series:
    """
    Keep existing coarse, floor-based progress binning (5% steps).
    """
    x = cum_u_dt_norm.astype(np.float64).clip(0.0, 1.0) * 20.0
    return np.floor(x + 1e-12).astype(np.int16)


train["time_idx"] = to_time_idx(train["time_step"])
test["time_idx"] = to_time_idx(test["time_step"])
train["uin_bin"] = to_uin_bin(train["u_in"])
test["uin_bin"] = to_uin_bin(test["u_in"])

train = add_integrated_uin_feature(train)
test = add_integrated_uin_feature(test)

train = add_insp_normalized_progress(train)
test = add_insp_normalized_progress(test)

train["cum_uin_dt_norm_bin"] = to_cum_uin_dt_norm_bin(train["cum_uin_dt_norm"])
test["cum_uin_dt_norm_bin"] = to_cum_uin_dt_norm_bin(test["cum_uin_dt_norm"])

train_insp = train[train["u_out"] == 0].copy()

grp_main = ["R", "C", "time_idx", "uin_bin", "cum_uin_dt_norm_bin"]
mean_rc_t_u_cum = train_insp.groupby(grp_main, observed=True)["pressure"].mean()

mean_rc_t_cum = train_insp.groupby(
    ["R", "C", "time_idx", "cum_uin_dt_norm_bin"], observed=True
)["pressure"].mean()

mean_rc_t_u = train_insp.groupby(["R", "C", "time_idx", "uin_bin"], observed=True)[
    "pressure"
].mean()

mean_rc_u_cum = train_insp.groupby(
    ["R", "C", "uin_bin", "cum_uin_dt_norm_bin"], observed=True
)["pressure"].mean()

mean_rc_t = train_insp.groupby(["R", "C", "time_idx"], observed=True)["pressure"].mean()
mean_rc_u = train_insp.groupby(["R", "C", "uin_bin"], observed=True)["pressure"].mean()
mean_rc = train_insp.groupby(["R", "C"], observed=True)["pressure"].mean()
global_mean = float(train_insp["pressure"].mean())

test_key_main = pd.MultiIndex.from_frame(test[grp_main])
pred = mean_rc_t_u_cum.reindex(test_key_main).to_numpy()

miss = np.isnan(pred)
if miss.any():
    key_rc_t_cum = pd.MultiIndex.from_frame(
        test.loc[miss, ["R", "C", "time_idx", "cum_uin_dt_norm_bin"]]
    )
    pred[miss] = mean_rc_t_cum.reindex(key_rc_t_cum).to_numpy()

miss = np.isnan(pred)
if miss.any():
    key_rc_t_u = pd.MultiIndex.from_frame(
        test.loc[miss, ["R", "C", "time_idx", "uin_bin"]]
    )
    pred[miss] = mean_rc_t_u.reindex(key_rc_t_u).to_numpy()

miss = np.isnan(pred)
if miss.any():
    key_rc_u_cum = pd.MultiIndex.from_frame(
        test.loc[miss, ["R", "C", "uin_bin", "cum_uin_dt_norm_bin"]]
    )
    pred[miss] = mean_rc_u_cum.reindex(key_rc_u_cum).to_numpy()

miss = np.isnan(pred)
if miss.any():
    key_rc_t = pd.MultiIndex.from_frame(test.loc[miss, ["R", "C", "time_idx"]])
    pred[miss] = mean_rc_t.reindex(key_rc_t).to_numpy()

miss = np.isnan(pred)
if miss.any():
    key_rc_u = pd.MultiIndex.from_frame(test.loc[miss, ["R", "C", "uin_bin"]])
    pred[miss] = mean_rc_u.reindex(key_rc_u).to_numpy()

miss = np.isnan(pred)
if miss.any():
    key_rc = pd.MultiIndex.from_frame(test.loc[miss, ["R", "C"]])
    pred[miss] = mean_rc.reindex(key_rc).to_numpy()

pred = np.where(np.isnan(pred), global_mean, pred)

pred_df = pd.DataFrame(
    {"id": test["id"].to_numpy(), "pressure": pred.astype(np.float32)}
)

pred_df = pred_df.set_index("id").reindex(sub["id"].to_numpy()).reset_index()

assert pred_df.shape[0] == sub.shape[0], "Submission row count mismatch."
assert list(pred_df.columns) == ["id", "pressure"], "Submission columns mismatch."
assert pred_df["pressure"].notna().all(), "Found NaNs in predicted pressure."

out_file = "submission.csv"
pred_df.to_csv(out_file, index=False)
print(f"Wrote {out_file} with shape={pred_df.shape} from DATA_ROOT={DATA_ROOT}")
