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

0.1588010744253287

# 6. Current score

1.80355

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.445) has done: 'I remove the broken dependency on missing `../input/gb-blending/...` files and instead generate a valid submission directly from the provided competition data. To keep changes minimal and score-neutral (since no prior score exists), the submission use a simple, deterministic baseline: predict the mean training pressure for all test rows. I also fix the notebook cell numbering (starting at cell 1) and make file paths robust by reading from `../input/ventilator-pressure-prediction/`. The script always write a valid `submission.csv` with the required `id,pressure` columns.'
- What this solution (achieved 7.53006) has done: 'I fix the `KeyError: 'pressure'` by avoiding a merge that creates `pressure_x/pressure_y` columns (which removes the plain `pressure` name you later index). I instead build the submission deterministically by aligning predictions to the sample submission’s `id` order via an `id -> predicted_pressure` map, guaranteeing the required `id,pressure` columns exist. I also make the input path resolution robust to both `/kaggle/input/...` and `../input/...` layouts while keeping your core “groupby means” baseline unchanged. This run end-to-end and always write a valid `submission.csv`.'
- What this solution (achieved 6.4545) has done: 'Your current baseline predicts pressure only from (R, C, u_out) group means, which ignores the strongest signal (u_in over time) and yields a very large MAE gap vs the target. To move the score substantially toward the target while keeping the approach simple and deterministic, I keep a lightweight “groupby means” model but extend it to condition on discretized `u_in` (rounded) as well. I also fall back through increasingly coarse groupings to avoid NaNs and ensure a valid submission for every test row. This is still the same core logic (lookup table via group means), just with a minimally richer key that better matches the data-generating process.'
- What this solution (achieved 4.22153) has done: 'To reduce MAE toward the 0.1588 target (lower is better) without changing your core “groupby means lookup-table” approach, I make the lookup key slightly more faithful to the scoring rule by learning only from inspiratory rows (`u_out==0`), since expiratory pressure isn’t scored. Then I strengthen the key minimally by adding a coarse `time_step` bin (rounded to 0.05s) alongside your existing rounded `u_in`, and keep the same backoff/fallback chain to avoid NaNs. Finally, I align the training/test rounding logic exactly and keep submission alignment via `id` mapping unchanged so the output stays valid and deterministic.'
- What this solution (achieved 1.84847) has done: 'Your current lookup-table baseline is still too coarse, so to move the MAE down toward the 0.1588 target we keep the exact same “groupby mean + backoff” core logic but strengthen the key with minimal extra signal: add lagged `u_in` (previous timestep within breath) and cumulative inspired volume proxy `u_in_cumsum`. These are deterministic features computed from the provided inputs and commonly correlate strongly with pressure without changing the modeling approach. We compute group means on inspiratory-only rows (`u_out==0`) as before, merge them into test, and extend the fallback chain so every row gets a prediction. Submission writing and `id` alignment remain unchanged to guarantee a valid `submission.csv`.'
- What this solution (achieved 1.84847) has done: 'You’re far above the target MAE (lower is better), so we should make a small, legitimate improvement that keeps your same “groupby mean lookup table + backoff” approach intact. The biggest easy win without changing the modeling paradigm is to better match the evaluation rule: only inspiratory rows are scored, and pressure during expiratory (`u_out==1`) should be near a baseline rather than inferred from inspiratory lookups. I keep your exact lookup tables trained on `u_out==0`, but for test rows with `u_out==1` I override predictions to a robust constant derived from the training expiratory distribution (median), which typically reduces MAE on the unscored phase and prevents harmful arbitrary values. Everything else (features, merges, fallback chain, submission alignment) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 1.79618) has done: 'Your current score (1.84847 MAE, lower is better) is far from the target (0.1588), so we need a legitimate improvement while keeping your same “groupby mean lookup-table + backoff” approach. The smallest high-impact change is to add a discretized “within-breath step index” (the 0..79 time position) to the group keys, because pressure dynamics are strongly phase-dependent and `time_step` rounding alone can mix positions across breaths. We keep all existing features and fallbacks, just insert `step` into the lookup tables and merges, and keep the same inspiratory-only training and expiratory override. This should reduce lookup noise and move MAE down toward the target without changing the overall modeling paradigm or submission semantics.'
- What this solution (achieved 1.73822) has done: 'Your current MAE (1.796) is still far above the target (0.1588, lower is better), so we need a small, legitimate improvement while keeping your same “groupby-mean lookup table + fallback chain” core approach. The biggest missing signal that fits your existing feature style is within-breath *history*: pressure depends strongly on prior control/flow, so I add a minimally-invasive discretized lag-1 `u_out` and a short-window (last 3 steps) rolling-mean of `u_in` (both computed per breath) and use them only in the most specific lookup tables, with the same backoff behavior as before. This keeps your training/inference semantics the same (pure deterministic aggregation from train), but should reduce collisions where different dynamical contexts share the same key. Submission writing, inspiratory-only training, and expiratory override remain unchanged to preserve validity and stability.'
- What this solution (achieved 1.73822) has done: 'Your current MAE (1.73822, lower is better) is still far above the target (0.1588), so we should make a small, legitimate improvement without changing your core “groupby-mean lookup table + fallback chain” approach. The minimal high-impact fix here is to ensure the inspiratory lookup tables do not include `u_out`-context features (`u_out` and `u_out_prev`) that are constant/degenerate in `train_insp` and can cause unnecessary key fragmentation/mismatch; we keep the same feature set but drop those two from the most-specific group keys and merges (still only predicting inspiratory rows). We also replace the current expiratory override constant (median) with the *start-of-expiration* baseline (pressure at the first `u_out==1` timestep per breath), which better reflects what pressures look like during unscored expiration while remaining deterministic and leakage-free. Everything else (feature engineering, mean-lookup paradigm, fallback behavior, submission alignment) stays the same and still write a valid `submission.csv`.'
- What this solution (achieved 1.80355) has done: 'We keep your deterministic “groupby mean lookup + fallback chain” approach unchanged, but make two small, high-signal alignment tweaks that typically reduce MAE a lot for this competition. First, we stop using a coarse time bin (`t_bin`) in the most-specific tables and instead use the exact within-breath `step` (already computed) as the time/phase indicator, because `time_step` binning can cause avoidable key mismatches. Second, we quantize predictions to the known discrete pressure grid from the training data (a legitimate post-processing step consistent with the metric), which usually yields a noticeable MAE drop without changing the model logic. The rest (inspiratory-only training, expiratory override, backoff behavior, and submission alignment by `id`) remains the same and still writes a valid `submission.csv`.'
- What this solution (achieved 1.80355) has done: 'Your current MAE (1.80355, lower is better) is still far above the target (0.1588), so we need a small but meaningful improvement while keeping your exact “groupby-mean lookup + fallback chain” paradigm intact. The most impactful minimal fix is to stop forcing all `u_out==1` predictions to a constant, because even though expiratory rows aren’t scored, Kaggle’s evaluation filters by the *true* inspiratory phase and your constant override can accidentally hurt rows that should be predicted by the lookup. Instead, we keep the override but restrict it to the *obvious late-expiration region* (after the first `u_out==1` in a breath), leaving the transition point and any edge cases to be handled by your learned lookup. As another minimal, legitimate alignment step, we compute the pressure snapping grid from inspiratory rows only (the scored regime), which reduces the chance of snapping to expiratory-only values. Everything else (features, group keys, merges, fallback order, and submission writing) remains unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
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
    Original blending helper retained for compatibility with the original code,
    but not used in this fixed end-to-end solution (since external blend files are unavailable).
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
    Original random-weight blender retained, but not executed in this fixed solution
    (it depends on local blend files and performs an infeasible number of loops).
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
        for i in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        for i in range(len(weight)):
            weight[i] /= weight_sum
        weight.sort(reverse=True)
        for i in range(len(flist)):
            output.pressure += flist[i] * weight[i]
    output.pressure /= loop_time
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)




## === cell 2
def blend(a, b):
    """
    Original blend function retained, but in this fixed solution we won't call it unless files exist.
    """
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.6 + b.pressure * 0.4
    a.to_csv("blend.csv", index=False)
    return a




## === cell 3
CANDIDATE_DATA_DIRS = [
    "../input/ventilator-pressure-prediction",
    "/kaggle/input/ventilator-pressure-prediction",
    "../kaggle/input/ventilator-pressure-prediction",
]
DATA_DIR = None
for d in CANDIDATE_DATA_DIRS:
    if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
        os.path.join(d, "test.csv")
    ):
        DATA_DIR = d
        break
if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find ventilator-pressure-prediction data directory. Tried: "
        + ", ".join(CANDIDATE_DATA_DIRS)
    )

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

train = pd.read_csv(
    train_path,
    usecols=["breath_id", "R", "C", "u_out", "u_in", "time_step", "pressure"],
)
test = pd.read_csv(
    test_path, usecols=["id", "breath_id", "R", "C", "u_out", "u_in", "time_step"]
)
sub = pd.read_csv(sample_path, usecols=["id", "pressure"]).copy()

train = train.sort_values(["breath_id", "time_step"], kind="mergesort")
test = test.sort_values(["breath_id", "time_step"], kind="mergesort")

train["step"] = train.groupby("breath_id", sort=False).cumcount().astype(np.int16)
test["step"] = test.groupby("breath_id", sort=False).cumcount().astype(np.int16)

train["u_in_round"] = train["u_in"].round(0).astype(np.int16)
test["u_in_round"] = test["u_in"].round(0).astype(np.int16)

train["t_bin"] = (train["time_step"] / 0.05).round(0).astype(np.int16)
test["t_bin"] = (test["time_step"] / 0.05).round(0).astype(np.int16)

train["u_in_prev_round"] = (
    train.groupby("breath_id", sort=False)["u_in"]
    .shift(1)
    .fillna(0.0)
    .round(0)
    .astype(np.int16)
)
test["u_in_prev_round"] = (
    test.groupby("breath_id", sort=False)["u_in"]
    .shift(1)
    .fillna(0.0)
    .round(0)
    .astype(np.int16)
)

train["u_in_cumsum_round"] = (
    train.groupby("breath_id", sort=False)["u_in"]
    .cumsum()
    .div(10.0)
    .round(0)
    .astype(np.int16)
)
test["u_in_cumsum_round"] = (
    test.groupby("breath_id", sort=False)["u_in"]
    .cumsum()
    .div(10.0)
    .round(0)
    .astype(np.int16)
)

train["u_out_prev"] = (
    train.groupby("breath_id", sort=False)["u_out"].shift(1).fillna(0).astype(np.int8)
)
test["u_out_prev"] = (
    test.groupby("breath_id", sort=False)["u_out"].shift(1).fillna(0).astype(np.int8)
)

train["u_in_roll3_round"] = (
    train.groupby("breath_id", sort=False)["u_in"]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
    .round(0)
    .astype(np.int16)
)
test["u_in_roll3_round"] = (
    test.groupby("breath_id", sort=False)["u_in"]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
    .round(0)
    .astype(np.int16)
)

train_insp = train[train["u_out"] == 0].copy()
train_exp = train[train["u_out"] == 1].copy()

global_mean = float(train_insp["pressure"].mean())

if len(train_exp):
    exp_start = train[train["u_out"] == 1].groupby("breath_id", sort=False).head(1)
    exp_baseline = (
        float(exp_start["pressure"].median())
        if len(exp_start)
        else float(train_exp["pressure"].median())
    )
else:
    exp_baseline = global_mean

pressure_grid = np.sort(train_insp["pressure"].unique()).astype(np.float32)

rc_step_hist_mean = (
    train_insp.groupby(
        [
            "R",
            "C",
            "u_in_round",
            "u_in_prev_round",
            "u_in_roll3_round",
            "u_in_cumsum_round",
            "step",
        ],
        sort=False,
    )["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rc_step_hist"})
)

rc_step_hist2_mean = (
    train_insp.groupby(
        [
            "R",
            "C",
            "u_in_round",
            "u_in_prev_round",
            "u_in_roll3_round",
            "step",
        ],
        sort=False,
    )["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rc_step_hist2"})
)

rc_step_cumsum_mean = (
    train_insp.groupby(
        [
            "R",
            "C",
            "u_in_round",
            "u_in_prev_round",
            "u_in_cumsum_round",
            "step",
        ],
        sort=False,
    )["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rc_step_cumsum"})
)

rc_step_prev_mean = (
    train_insp.groupby(
        ["R", "C", "u_in_round", "u_in_prev_round", "step"],
        sort=False,
    )["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rc_step_prev"})
)

rc_step_mean = (
    train_insp.groupby(["R", "C", "u_in_round", "step"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rc_step"})
)

rcuutpccs_hist_mean = (
    train_insp.groupby(
        [
            "R",
            "C",
            "u_in_round",
            "u_in_prev_round",
            "u_in_roll3_round",
            "u_in_cumsum_round",
            "t_bin",
            "step",
        ],
        sort=False,
    )["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcuutpccs_hist"})
)

rcuutpcs_hist_mean = (
    train_insp.groupby(
        [
            "R",
            "C",
            "u_in_round",
            "u_in_prev_round",
            "u_in_roll3_round",
            "t_bin",
            "step",
        ],
        sort=False,
    )["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcuutpcs_hist"})
)

rcuutpccs_mean = (
    train_insp.groupby(
        [
            "R",
            "C",
            "u_in_round",
            "u_in_prev_round",
            "u_in_cumsum_round",
            "t_bin",
            "step",
        ],
        sort=False,
    )["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcuutpccs"})
)

rcuutpcs_mean = (
    train_insp.groupby(
        ["R", "C", "u_in_round", "u_in_prev_round", "t_bin", "step"],
        sort=False,
    )["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcuutpcs"})
)

rcuuts_mean = (
    train_insp.groupby(["R", "C", "u_in_round", "t_bin", "step"], sort=False)[
        "pressure"
    ]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcuuts"})
)

rcuutpcc_mean = (
    train_insp.groupby(
        [
            "R",
            "C",
            "u_in_round",
            "u_in_prev_round",
            "u_in_cumsum_round",
            "t_bin",
        ],
        sort=False,
    )["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcuutpcc"})
)

rcuutpc_mean = (
    train_insp.groupby(
        ["R", "C", "u_in_round", "u_in_prev_round", "t_bin"],
        sort=False,
    )["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcuutpc"})
)

rcuut_mean = (
    train_insp.groupby(["R", "C", "u_in_round", "t_bin"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcuut"})
)

rcuu_mean = (
    train_insp.groupby(["R", "C", "u_in_round"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rcuu"})
)

rc_mean = (
    train_insp.groupby(["R", "C"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_rc"})
)

pred = test.merge(
    rc_step_hist_mean,
    on=[
        "R",
        "C",
        "u_in_round",
        "u_in_prev_round",
        "u_in_roll3_round",
        "u_in_cumsum_round",
        "step",
    ],
    how="left",
)
pred = pred.merge(
    rc_step_hist2_mean,
    on=[
        "R",
        "C",
        "u_in_round",
        "u_in_prev_round",
        "u_in_roll3_round",
        "step",
    ],
    how="left",
)
pred = pred.merge(
    rc_step_cumsum_mean,
    on=["R", "C", "u_in_round", "u_in_prev_round", "u_in_cumsum_round", "step"],
    how="left",
)
pred = pred.merge(
    rc_step_prev_mean,
    on=["R", "C", "u_in_round", "u_in_prev_round", "step"],
    how="left",
)
pred = pred.merge(
    rc_step_mean,
    on=["R", "C", "u_in_round", "step"],
    how="left",
)

pred = pred.merge(
    rcuutpccs_hist_mean,
    on=[
        "R",
        "C",
        "u_in_round",
        "u_in_prev_round",
        "u_in_roll3_round",
        "u_in_cumsum_round",
        "t_bin",
        "step",
    ],
    how="left",
)
pred = pred.merge(
    rcuutpcs_hist_mean,
    on=[
        "R",
        "C",
        "u_in_round",
        "u_in_prev_round",
        "u_in_roll3_round",
        "t_bin",
        "step",
    ],
    how="left",
)

pred = pred.merge(
    rcuutpccs_mean,
    on=[
        "R",
        "C",
        "u_in_round",
        "u_in_prev_round",
        "u_in_cumsum_round",
        "t_bin",
        "step",
    ],
    how="left",
)
pred = pred.merge(
    rcuutpcs_mean,
    on=["R", "C", "u_in_round", "u_in_prev_round", "t_bin", "step"],
    how="left",
)
pred = pred.merge(rcuuts_mean, on=["R", "C", "u_in_round", "t_bin", "step"], how="left")

pred = pred.merge(
    rcuutpcc_mean,
    on=[
        "R",
        "C",
        "u_in_round",
        "u_in_prev_round",
        "u_in_cumsum_round",
        "t_bin",
    ],
    how="left",
)
pred = pred.merge(
    rcuutpc_mean,
    on=["R", "C", "u_in_round", "u_in_prev_round", "t_bin"],
    how="left",
)
pred = pred.merge(rcuut_mean, on=["R", "C", "u_in_round", "t_bin"], how="left")
pred = pred.merge(rcuu_mean, on=["R", "C", "u_in_round"], how="left")
pred = pred.merge(rc_mean, on=["R", "C"], how="left")

pred_pressure = pred["p_rc_step_hist"].copy()
pred_pressure = pred_pressure.fillna(pred["p_rc_step_hist2"])
pred_pressure = pred_pressure.fillna(pred["p_rc_step_cumsum"])
pred_pressure = pred_pressure.fillna(pred["p_rc_step_prev"])
pred_pressure = pred_pressure.fillna(pred["p_rc_step"])

pred_pressure = pred_pressure.fillna(pred["p_rcuutpccs_hist"])
pred_pressure = pred_pressure.fillna(pred["p_rcuutpcs_hist"])
pred_pressure = pred_pressure.fillna(pred["p_rcuutpccs"])
pred_pressure = pred_pressure.fillna(pred["p_rcuutpcs"])
pred_pressure = pred_pressure.fillna(pred["p_rcuuts"])
pred_pressure = pred_pressure.fillna(pred["p_rcuutpcc"])
pred_pressure = pred_pressure.fillna(pred["p_rcuutpc"])
pred_pressure = pred_pressure.fillna(pred["p_rcuut"])
pred_pressure = pred_pressure.fillna(pred["p_rcuu"])
pred_pressure = pred_pressure.fillna(pred["p_rc"])
pred_pressure = pred_pressure.fillna(global_mean).astype(float)

exp_started = (
    pred.groupby("breath_id", sort=False)["u_out"].cummax().astype(np.int8).values
)
u_out_now = pred["u_out"].values.astype(np.int8)
override_mask = (
    (u_out_now == 1)
    & (pred["u_out_prev"].values.astype(np.int8) == 1)
    & (exp_started == 1)
)
pred_pressure = pd.Series(
    np.where(override_mask, exp_baseline, pred_pressure.values).astype(float),
    index=pred.index,
)

grid = pressure_grid
idx = np.searchsorted(grid, pred_pressure.values, side="left")
idx = np.clip(idx, 0, len(grid) - 1)
idx_left = np.clip(idx - 1, 0, len(grid) - 1)
grid_right = grid[idx]
grid_left = grid[idx_left]
use_left = np.abs(pred_pressure.values - grid_left) <= np.abs(
    pred_pressure.values - grid_right
)
snapped = np.where(use_left, grid_left, grid_right).astype(np.float32)
pred_pressure = pd.Series(snapped, index=pred.index).astype(float)

id_to_pred = pd.Series(pred_pressure.values, index=pred["id"].values)
sub["pressure"] = sub["id"].map(id_to_pred).astype(float)
sub["pressure"] = sub["pressure"].fillna(global_mean).astype(float)

sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("pressure stats:", sub["pressure"].describe())
print("DATA_DIR used:", DATA_DIR)
print("global_mean (insp only):", global_mean)
print("exp_baseline (u_out==1):", exp_baseline)
print("pressure grid size (insp only):", len(pressure_grid))
