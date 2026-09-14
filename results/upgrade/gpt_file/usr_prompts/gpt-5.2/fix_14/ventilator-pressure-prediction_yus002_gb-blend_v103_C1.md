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

0.151554346941584

# 6. Current score

2.17998

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.81758) has done: 'I remove the dependency on missing external Kaggle datasets (`gb-data-blending-recover`) that causes the `FileNotFoundError` and instead generate a valid submission directly from the provided competition files. To keep changes minimal and score-neutral (given no current score), I use a simple, fast baseline prediction: per-(R,C,time_step,u_in,u_out) median pressure from train, with a fallback to global median, then snap to the nearest valid pressure level (as your code already does). I also fix the cell numbering to start at 1 and ensure the script always writes a `submission.csv` with the required `id,pressure` format. The rest of your helper functions are preserved, but not used unless you later provide local blend files.'
- What this solution (achieved 6.64707) has done: 'Your current score (6.81758 MAE) is far worse than the target (0.1516), so we should improve while keeping the same “train-median lookup → fallback → snap-to-valid-pressure” core logic. The biggest minimal win is to make the lookup key more informative without changing modeling approach: add a few simple lag/accumulation features (within each breath) and use a hierarchical fallback (fine key → coarser keys → global median) instead of a single global fallback. This stays within your current semantics (pure aggregation from train, no learning loop) but greatly increases match rate and reduces error. I also keep the exact submission format and still snap predictions to the nearest valid pressure level.'
- What this solution (achieved 7.55864) has done: 'We keep your existing “train-median lookup → hierarchical fallback → snap-to-valid-pressure” approach, but add one more low-risk hierarchy level that captures a key invariant of this dataset: pressure depends strongly on the cumulative inspired volume proxy (`u_in * dt` cumulative) and is almost independent of the absolute `time_step` once you condition on that. Concretely, we (1) compute a rounded cumulative-volume feature, (2) add a lookup table keyed by `(R, C, u_out, u_in_cum)` as an additional fallback between your current coarse key and the global median, and (3) use a tighter rounding for `u_in_cum` to improve match rate while staying within the same aggregation semantics. This is a minimal change that typically improves MAE substantially from “time-step keyed” baselines without changing any model/training logic. The script still run end-to-end and write `submission.csv` with `id,pressure`.'
- What this solution (achieved 7.55864) has done: 'We keep your exact “median-lookup with hierarchical fallbacks + snap-to-nearest-valid-pressure” core logic, but fix the most likely cause of the huge MAE: the `id` column is not globally unique and is being overwritten from `test.csv`, which can scramble rows and make the submission invalidly aligned. We instead keep the `id` and row order from `sample_submission.csv` and ensure predictions are assigned in that same order (matching the competition’s expected mapping). As a minimal, score-positive tweak within your same aggregation semantics, we also compute `dt` within each breath more robustly and clip negative/odd diffs to 0 to stabilize `u_in_cum` keys. These changes should materially reduce MAE toward the target without changing your modeling approach.'
- What this solution (achieved 3.95423) has done: 'Your current MAE (7.56) is far worse than the target (0.1516), so we should improve while keeping your core “hierarchical median lookup + snap-to-valid-pressure” logic intact. The biggest low-risk gain is to ensure the cumulative inspired-volume proxy is computed consistently with the data’s fixed timestep (80 steps per breath): use a per-breath constant `dt` from the first positive diff (instead of per-row diff that can introduce mismatched keys), and slightly coarsen rounding to increase train↔test key match rate. We also add one additional fallback level keyed by `(R,C,u_out,u_in_r,u_in_cum_r)` to recover cases where `time_step`/lags differ but volume+inputs align, without changing the approach. Submission row alignment is kept strictly by writing predictions into `sample_submission.csv` order.'
- What this solution (achieved 2.76022) has done: 'We keep your exact “hierarchical median lookup → fallback → snap to nearest valid pressure” approach, but make two minimal, score-relevant fixes that usually reduce MAE a lot for this competition. First, compute cumulative inspired volume using the known fixed step count (80) by deriving a per-breath constant `dt = (max(time_step)-min(time_step))/79` (instead of a “first positive diff” heuristic that can produce mismatched keys between train and test). Second, slightly coarsen the rounding for `u_in_cum_r` (and keep other keys unchanged) to increase train↔test key match rate in the lookup tables without changing the model semantics. The output alignment/format is preserved by writing predictions into `sample_submission.csv` order and producing `submission.csv`.'
- What this solution (achieved 2.31861) has done: 'Your current MAE (2.76) is still far worse than the target (0.1516), so we should improve while keeping your exact “hierarchical median lookup + snap-to-nearest-valid-pressure” core approach. The biggest minimal win is to reduce key mismatch between train and test by computing the cumulative inspired-volume proxy (`u_in_cum`) using per-row `dt = time_step diff` within each breath (instead of a per-breath constant), which better reflects the underlying discretization and stabilizes alignment. To avoid introducing new logic, we keep your same grouping levels, but slightly coarsen `u_in_cum_r` rounding to increase match rate and reduce reliance on coarse/global fallbacks. Submission writing remains identical (use `sample_submission.csv` order, write `submission.csv`).'
- What this solution (achieved 2.62764) has done: 'Your current MAE (2.3186) is still far above the target (0.1516), so we should improve while preserving your exact “hierarchical median lookup + snap-to-nearest-valid-pressure” approach. The most impactful minimal fix is to make the cumulative inspired-volume proxy (`u_in_cum`) consistent with ventilator physics by *only accumulating when `u_out==0`* (inspiration), and resetting/holding steady during exhalation; this better aligns keys with the metric (only inspiratory phase is scored). To reduce train↔test key mismatch without changing core logic, we also add a very small additional fallback table keyed by `(R,C,u_out,u_in_cum_r)` with a slightly finer rounding for `u_in_cum_r`. Everything else (groupby-median semantics, fallback chain, snapping, and writing `submission.csv` in sample order) remains the same.'
- What this solution (achieved 2.16079) has done: 'Your current MAE (2.62764, lower-is-better) is still far above the target (0.15155), so we should improve while keeping your exact “feature-engineer → hierarchical train-median lookup → snap to nearest valid pressure → write submission.csv” core logic. The largest low-risk gain within that framework is to reduce train↔test key mismatches by using slightly coarser, more stable rounding for the continuous keys that drive the joins (especially `u_in_cum_r`, `u_in_r`, and `time_step_r`), which increases the hit-rate of the higher-quality lookup tables and reduces reliance on coarse/global fallbacks. I’m also removing a redundant duplicate fallback table (`grp45` is identical to `grp4` in your code and never adds new information), which won’t change semantics but reduces noise/overhead. Everything else (feature set, median aggregation, fallback chain structure, snapping, and submission alignment via `sample_submission.csv`) remains the same.'
- What this solution (achieved 2.58447) has done: 'Your current MAE (2.16079, lower-is-better) is still far above the target (0.15155), so we should improve while keeping your same “feature engineering → hierarchical train-median lookup → snap-to-valid-pressure” core logic. The most likely remaining high-impact issue is that we’re predicting pressures for expiratory rows (`u_out==1`) even though they are not scored; a minimal, metric-aligned tweak is to force expiratory predictions to the global median (then snap), which avoids noisy lookup misses from exhalation dynamics. Additionally, we can slightly refine rounding for `u_in_cum_r` (keep others unchanged) to increase train↔test key match rate on the best lookup levels without altering the approach. Submission alignment/format stays exactly the same (write predictions into `sample_submission.csv` order as `submission.csv`).'
- What this solution (achieved 2.26167) has done: 'Your MAE (2.584) is still far above the target (0.1516), so we should improve while keeping the same core “feature engineering → hierarchical train-median lookup → snap-to-valid-pressure” approach. The biggest remaining mismatch comes from using rounded `time_step` in the highest-priority lookup keys, which tends to miss due to slight discretization differences; we keep all your existing tables but insert a new high-quality lookup keyed on the physically-relevant cumulative inspired volume (`u_in_cum_r`) and lagged inputs *without* `time_step`. To further increase train↔test key match rate (without changing the approach), we make `u_in_cum_r` slightly coarser (round to 1 decimal) and compute `dt` in a more stable way by clipping to a plausible range and filling the first step with the per-breath median positive `dt`. Submission writing and pressure snapping stay identical.'
- What this solution (achieved 2.26146) has done: 'Your current MAE (2.26167, lower-is-better) is still far from the target (0.15155), so we should improve while preserving your exact “feature engineering → hierarchical train-median lookup → snap-to-valid-pressure” approach. The most impactful minimal fix is to stop forcing expiratory (`u_out==1`) predictions to the global median, since those rows are not scored and that rule can hurt continuity/lookup consistency across the breath. We also add one additional *coarser* fallback table keyed by `(R,C,u_out,u_in_r)` to reduce reliance on the global median when cumulative-volume keys miss, without changing the overall aggregation semantics. Everything else (features, lookup-chain style, snapping, and writing `submission.csv`) remains the same.'
- What this solution (achieved 2.17998) has done: 'Your current MAE (2.26146, lower-is-better) is still far above the target (0.15155), so we should improve while keeping your exact “feature engineering → hierarchical train-median lookup → snap-to-valid-pressure” approach. The most impactful minimal change is to add one more physically-relevant lookup layer that removes `time_step` entirely and relies on cumulative inspired volume plus short lags, which reduces train↔test key mismatches while preserving the same groupby-median semantics. To make that new layer (and your existing cumulative-volume tables) hit more often, we add a slightly coarser rounded cumulative feature (`u_in_cum_r2`), but we keep your original `u_in_cum_r` unchanged and keep the rest of the fallback chain intact. The submission writing/alignment stays exactly the same (use `sample_submission.csv` order and write `submission.csv`).'

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

df_train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)


def find_nearest(prediction):
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
    for j in range(loop_time):
        weight = []
        set_seed(j)
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




## === cell 1
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["time_step_r"] = df["time_step"].round(1)
    df["u_in_r"] = df["u_in"].round(1)

    g = df.groupby("breath_id", sort=False)
    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0).round(1)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0).round(1)
    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0).astype(np.int8)

    dt = g["time_step"].diff().astype(np.float64)
    dt = dt.clip(lower=0.0, upper=0.1)
    dt_pos = dt.where(dt > 0)
    dt_fill = (
        dt_pos.groupby(df["breath_id"], sort=False).transform("median").fillna(0.0)
    )
    dt = dt.fillna(dt_fill)

    insp = (df["u_out"].to_numpy() == 0).astype(np.float64)
    flow = df["u_in"].to_numpy() * dt.to_numpy() * insp
    df["u_in_cum"] = (
        pd.Series(flow, index=df.index).groupby(df["breath_id"], sort=False).cumsum()
    )

    df["u_in_cum_r"] = df["u_in_cum"].round(1)
    df["u_in_cum_r2"] = df["u_in_cum"].round(0)

    return df


train_key = add_features(df_train)
test_key = add_features(df_test)

global_median = float(df_train["pressure"].median())

grp1 = [
    "R",
    "C",
    "time_step_r",
    "u_in_r",
    "u_out",
    "u_in_lag1",
    "u_out_lag1",
    "u_in_cum_r",
]
med1 = (
    train_key.groupby(grp1, observed=True)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "p1"})
)
test_key = test_key.merge(med1, on=grp1, how="left")

grp2 = ["R", "C", "time_step_r", "u_in_r", "u_out", "u_in_lag1", "u_in_cum_r"]
med2 = (
    train_key.groupby(grp2, observed=True)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "p2"})
)
test_key = test_key.merge(med2, on=grp2, how="left")

grp3 = ["R", "C", "time_step_r", "u_in_r", "u_out", "u_in_cum_r"]
med3 = (
    train_key.groupby(grp3, observed=True)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "p3"})
)
test_key = test_key.merge(med3, on=grp3, how="left")

grp33 = ["R", "C", "u_out", "u_in_r", "u_in_lag1", "u_out_lag1", "u_in_cum_r2"]
med33 = (
    train_key.groupby(grp33, observed=True)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "p33"})
)
test_key = test_key.merge(med33, on=grp33, how="left")

grp35 = ["R", "C", "u_out", "u_in_r", "u_in_cum_r"]
med35 = (
    train_key.groupby(grp35, observed=True)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "p35"})
)
test_key = test_key.merge(med35, on=grp35, how="left")

grp25 = ["R", "C", "u_out", "u_in_lag1", "u_out_lag1", "u_in_r", "u_in_cum_r"]
med25 = (
    train_key.groupby(grp25, observed=True)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "p25"})
)
test_key = test_key.merge(med25, on=grp25, how="left")

grp4 = ["R", "C", "u_out", "u_in_cum_r"]
med4 = (
    train_key.groupby(grp4, observed=True)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "p4"})
)
test_key = test_key.merge(med4, on=grp4, how="left")

grp5 = ["R", "C", "u_out", "u_in_r"]
med5 = (
    train_key.groupby(grp5, observed=True)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "p5"})
)
test_key = test_key.merge(med5, on=grp5, how="left")

pred = (
    test_key["p1"]
    .fillna(test_key["p2"])
    .fillna(test_key["p3"])
    .fillna(test_key["p25"])
    .fillna(
        test_key["p33"]
    )  # inserted here: higher quality than coarser volume-only tables
    .fillna(test_key["p35"])
    .fillna(test_key["p4"])
    .fillna(test_key["p5"])
    .fillna(global_median)
    .astype(float)
    .to_numpy()
)

pred = np.array([find_nearest(p) for p in pred], dtype=float)

if len(sub) != len(pred):
    raise ValueError(
        f"Row mismatch: submission has {len(sub)} rows but got {len(pred)} predictions."
    )

sub["pressure"] = pred
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
