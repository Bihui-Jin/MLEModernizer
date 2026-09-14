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

0.1751452934177095

# 6. Current score

2.66271

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.13673) has done: 'I remove the dependency on missing external blend files (the `../input/gb-blending/...` paths) that causes the `FileNotFoundError`, and instead generate a valid prediction directly from the provided competition data. To keep core logic intact (this notebook is a blending/rounding script rather than a training pipeline), I implement a simple, deterministic baseline model using only train statistics: predict the mean pressure per `(R, C, time_step)` computed from train and merge onto test. I keep your `find_nearest` rounding to the known discrete pressure grid (important for this competition) and ensure a correctly formatted `submission.csv` is written end-to-end. The output be created in the current working directory with a `.csv` suffix and correct `id,pressure` columns.'
- What this solution (achieved 8.12685) has done: 'You’re currently far worse than the target (MAE 8.13673 vs 0.1751, lower is better), so we should legitimately improve accuracy with minimal, metric-aligned changes while keeping the same “train-statistics + nearest-pressure-grid rounding” core logic. The main issue with the current baseline is that `(R,C,time_step)` mean lookup is too weak; we can keep the same approach but compute the mean on a richer (yet still simple) key that better captures dynamics, using only existing columns. Specifically, we add per-breath cumulative features (`u_in_cum`, `u_out_cum`) computed identically for train and test, then group by `(R,C,time_step,u_out,u_in_cum_rounded)` to get a much tighter mean estimate, with a safe backoff to your original grouping and then global mean. We preserve your `find_nearest` rounding (important for this competition) and still write a valid `submission.csv`.'
- What this solution (achieved 8.12803) has done: 'Your current score (8.12685 MAE, lower is better) is far from the target (0.1751), so we should legitimately improve accuracy with minimal changes while keeping the same “train-statistics lookup + nearest-pressure-grid rounding” core logic. The biggest issue is that the lookup key is still too coarse; we can keep the same mean-aggregation approach but add a few standard ventilator competition features that better capture breath dynamics (lagged `u_in`, short rolling mean, and cumulative volume proxy), then do a richer-group mean with a safe backoff chain. This stays purely deterministic, uses only provided columns, preserves your rounding-to-discrete-pressure step, and still writes a valid `submission.csv`. The changes are localized to feature creation and grouping keys; no model/training loop is introduced.'
- What this solution (achieved 4.27436) has done: 'We’re far worse than the target (8.128 vs 0.175, lower is better), so we should improve accuracy while keeping your existing “train-statistics lookup + nearest-pressure-grid rounding” core logic intact. The main weakness is using raw floating `time_step` as a merge key, which can mismatch between train/test due to float representation; we instead create an integer `ts_bin` (time_step * 100 rounded) and use that consistently for all groupbys/merges. We also add one more minimal dynamic key (`u_in` binned) to the rich/mid aggregations (with safe backoff unchanged) to tighten the conditional mean without introducing any training loop or new model. Finally, we keep your discrete-pressure rounding and ensure the submission rows align exactly with `sample_submission` by merging predictions on `id`.'
- What this solution (achieved 4.31909) has done: 'Your current MAE (4.274) is still far above the target (0.175; lower is better), so we should improve accuracy while keeping the same “train-statistics lookup + discrete-pressure rounding” core logic. The biggest remaining issue is that the fallback chain can still miss breath-specific dynamics; we add one minimal, competition-standard dynamic proxy (`u_in` integral and first differences) but still only use deterministic group-mean lookups with backoff. We also make the aggregation keys slightly more stable by binning `time_step` at 0.01s (as you do) and additionally using `breath_step` (0..79) to reduce any residual float/bin misalignment without changing evaluation semantics. Finally, we keep your nearest-grid rounding and ensure `submission.csv` is written with correct `id,pressure` alignment.'
- What this solution (achieved 4.31935) has done: 'Your current MAE (4.319, lower-is-better) is still far above the target (0.175), so we should legitimately improve accuracy while keeping your same core “train-conditional-mean lookup + backoff + nearest-pressure-grid rounding” approach. The biggest low-risk gain here is to (1) compute group means only on inspiratory rows (`u_out==0`), since expiratory rows are not scored and have different dynamics that can pollute the averages, and (2) add a final backoff keyed on `(R,C,breath_step,ts_bin,u_out)` to better handle missing matches without exploding the key space. I’m also making the rolling feature computation deterministic per-breath by setting the group key as index during rolling to avoid any subtle misalignment, while leaving the features themselves unchanged. The submission writing and `find_nearest` rounding are preserved exactly, and the script still runs end-to-end and produces `submission.csv`.'
- What this solution (achieved 3.16219) has done: 'You’re still far above the target MAE (4.319 vs 0.175; lower is better), so we need a legitimate accuracy gain without changing the overall “train-conditional-mean lookup + backoff + nearest-pressure-grid rounding” logic. The lowest-risk improvement is to stop forcing expiratory (`u_out==1`) test rows through a noisy lookup: since they are not scored, we can set those predictions to a stable per-(R,C) inspiratory mean, reducing the spillover error from poor expiratory dynamics without affecting the scoring phase. We also add one extra, very cheap backoff level keyed on `(R,C,breath_step,u_out,u_in_bin)` to reduce NaNs/mismatches when the rich/mid keys are too sparse, keeping everything deterministic. Submission writing, alignment on `id`, and your discrete-pressure rounding remain exactly as before.'
- What this solution (achieved 3.16219) has done: 'Your current MAE (3.16219, lower-is-better) is still far above the target (0.1751), so we should make a small, legitimate accuracy improvement without changing your core “train conditional-mean lookup + backoff + nearest-pressure-grid rounding” approach. The most impactful minimal fix is to ensure `id` is treated as a globally-unique row identifier (as in the official dataset) by reading it with the correct dtype and then always aligning predictions to `sample_submission` ordering; if `id` is accidentally parsed/reused as 1..2000 repeatedly, it can silently destroy alignment and inflate MAE. I also make the expiratory-row handling slightly safer by computing an inspiratory fallback per-(R,C,breath_step) (instead of only per-(R,C)), which better matches the expected inspiratory trajectory while still respecting that expiratory is unscored. These changes keep all your features, groupby keys, backoff chain, and discrete rounding intact, but reduce the chance of catastrophic misalignment and improve the fallback quality toward the target.'
- What this solution (achieved 3.16219) has done: 'Your current MAE (3.16219, lower-is-better) is still far above the target (0.1751), so we should make a small, legitimate improvement without changing your core “train conditional-mean lookup + backoff + nearest-pressure-grid rounding” approach. The biggest low-risk win is to stop using expiratory (`u_out==1`) rows in *any* of the lookup tables, because those rows are not scored and their different dynamics can pollute the conditional means used for inspiratory prediction. We keep your exact feature set and backoff chain, but rebuild all group-mean tables from inspiratory-only data and use an inspiratory-only fallback for expiratory test rows. This stays fully deterministic, keeps I/O paths the same, and still writes a valid `submission.csv`.'
- What this solution (achieved 3.16245) has done: 'Your current MAE (3.16219; lower is better) is still far above the target (0.1751), so we need a legitimate accuracy gain while keeping the same deterministic “train conditional-mean lookup + backoff + nearest-pressure-grid rounding” core logic. The biggest low-risk issue I see is that `u_in_lag1_bin` is currently created by rounding `u_in_lag1` to an integer (effectively a 1.0 bin), while `u_in_bin` uses a 0.5 bin; this inconsistency can drastically reduce exact key matches for the rich table and force many fallbacks. I make `u_in_lag1_bin` use the same bin width as `u_in_bin` (0.5) for both train/test, which should increase rich/mid hit-rate and reduce MAE without changing the approach. I also keep all inspiratory-only aggregation, backoff chain, rounding, and the same I/O paths, and still write a valid `submission.csv`.'
- What this solution (achieved 2.82739) has done: 'We’re far worse than the target (3.16245 vs 0.17515 MAE; lower is better), so the smallest legitimate improvement without changing your “conditional-mean lookup + backoff + nearest-pressure-grid rounding” core logic is to make the lookup keys less sparse and better aligned with inspiratory dynamics. I keep all your existing features and tables, but add a single new intermediate backoff table that uses the already-computed continuous dynamics features (roll3, lag1, dt_cum) without the very high-cardinality cum/diff terms that can cause frequent misses. I also make the inspiratory-only filtering consistent by computing the mid/rich tables on `u_out==0` (as you already do) while allowing test merges to fall back cleanly for `u_out==1`. This should increase hit-rate on stronger keys and reduce MAE toward your target while preserving evaluation semantics and producing the same `submission.csv`.'
- What this solution (achieved 2.82739) has done: 'Your current MAE (2.82739, lower-is-better) is still far above the target (0.17515), so we should make a small, metric-aligned improvement without changing your core “inspiratory-only conditional-mean lookup + backoff chain + nearest-pressure-grid rounding” approach. The least invasive gain is to better handle `u_out==1` (expiratory) rows by using a *pressure floor* inferred from train: expiratory pressure in this dataset is typically near the minimum discrete pressure, and since expiratory rows are not scored, setting them to a stable low value reduces noise without affecting the scored inspiratory phase. Concretely, we compute the minimum pressure from the known discrete pressure grid and set all test expiratory predictions to that minimum after we compute inspiratory predictions, keeping the rest of your lookup/backoff unchanged. This is deterministic, preserves your feature extraction, aggregations, and rounding semantics, and still writes a valid `submission.csv`.'
- What this solution (achieved 2.82739) has done: 'Your score is much worse than the target (2.827 vs 0.175, lower-is-better), so we need a real accuracy gain while keeping your same deterministic “conditional-mean lookup + backoff + nearest-pressure-grid rounding” core logic. The biggest low-risk flaw is using `u_out` as a merge key even though all lookup tables are built from inspiratory-only rows (`u_out==0`), which guarantees no matches for expiratory rows and forces coarse fallbacks; we fix this by building *two* sets of tables (insp and exp) and using the appropriate one per row. For the expiratory tables we keep the keys/backoff identical but train them on `u_out==1` rows, then we remove the hard override that forces expiratory predictions to the minimum pressure (that override can harm if Kaggle’s scoring mask differs from expectation). Everything else (features, bins, backoff order, rounding, submission alignment) stays the same and still writes `submission.csv`.'
- What this solution (achieved 2.66271) has done: 'We’re still much worse than the target (2.827 vs 0.175 MAE; lower-is-better), so we need a legitimate accuracy gain while keeping the same deterministic “conditional-mean lookup + backoff + nearest-pressure-grid rounding” core logic. The smallest high-impact fix is to align the lookup tables with the evaluation: build all inspiratory (`u_out==0`) tables from inspiratory-only rows, and for expiratory test rows (unscored) avoid using the expiratory-trained tables entirely by filling them with a stable inspiratory fallback (per `(R,C,breath_step)`), reducing noise without changing the scored inspiratory behavior. In addition, we reduce key sparsity slightly (without changing the approach) by loosening only the richest table’s `u_in_cum_bin` granularity from 0.5 to 1.0, which increases exact-match hit rate and should lower MAE. All paths, feature engineering, backoff order, discrete pressure rounding, and submission alignment remain the same, and the script still writes a valid `submission.csv`.'

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
SAMPLE_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"

df_train = pd.read_csv(
    TRAIN_PATH,
    dtype={
        "id": np.int64,
        "breath_id": np.int64,
        "R": np.int16,
        "C": np.int16,
        "u_out": np.int8,
    },
)

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
    for k in range(loop_time):
        weight = []
        set_seed(k)
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
    output = pd.read_csv(SAMPLE_PATH)
    output.pressure = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.55 + b.pressure * 0.45
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
df_test = pd.read_csv(
    TEST_PATH,
    dtype={
        "id": np.int64,
        "breath_id": np.int64,
        "R": np.int16,
        "C": np.int16,
        "u_out": np.int8,
    },
)
sub = pd.read_csv(SAMPLE_PATH, dtype={"id": np.int64})


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    g = df.groupby("breath_id", sort=False)
    df["breath_step"] = g.cumcount().astype(np.int16)

    df["ts_bin"] = np.rint(df["time_step"].to_numpy(dtype=np.float64) * 100.0).astype(
        np.int16
    )

    df["u_in_cum"] = g["u_in"].cumsum()
    df["u_out_cum"] = g["u_out"].cumsum()

    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0)
    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0).astype(np.int8)

    df["u_in_roll3"] = (
        df.set_index(["breath_id", "breath_step"])["u_in"]
        .groupby(level=0, sort=False)
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .to_numpy()
    )

    df["dt"] = g["time_step"].diff().fillna(0.0)

    df["u_in_diff1"] = g["u_in"].diff().fillna(0.0)
    df["u_in_absdiff_cum"] = (
        (df["u_in_diff1"].abs()).groupby(df["breath_id"], sort=False).cumsum()
    )

    df["u_in_dt_cum"] = (
        (df["u_in"] * df["dt"]).groupby(df["breath_id"], sort=False).cumsum()
    )

    return df


train_feat = add_features(df_train)
test_feat = add_features(df_test)

CUM_BIN = 1.0  # was 0.5
DT_CUM_BIN = 0.05
ROLL_BIN = 0.5
UIN_BIN = 0.5
DIFF_BIN = 0.5
ABSDIFF_CUM_BIN = 1.0

train_feat["u_in_cum_bin"] = (train_feat["u_in_cum"] / CUM_BIN).round().astype(np.int32)
test_feat["u_in_cum_bin"] = (test_feat["u_in_cum"] / CUM_BIN).round().astype(np.int32)

train_feat["u_in_dt_cum_bin"] = (
    (train_feat["u_in_dt_cum"] / DT_CUM_BIN).round().astype(np.int32)
)
test_feat["u_in_dt_cum_bin"] = (
    (test_feat["u_in_dt_cum"] / DT_CUM_BIN).round().astype(np.int32)
)

train_feat["u_in_roll3_bin"] = (
    (train_feat["u_in_roll3"] / ROLL_BIN).round().astype(np.int32)
)
test_feat["u_in_roll3_bin"] = (
    (test_feat["u_in_roll3"] / ROLL_BIN).round().astype(np.int32)
)

train_feat["u_in_lag1_bin"] = (
    (train_feat["u_in_lag1"] / UIN_BIN).round().astype(np.int32)
)
test_feat["u_in_lag1_bin"] = (test_feat["u_in_lag1"] / UIN_BIN).round().astype(np.int32)

train_feat["u_in_bin"] = (train_feat["u_in"] / UIN_BIN).round().astype(np.int16)
test_feat["u_in_bin"] = (test_feat["u_in"] / UIN_BIN).round().astype(np.int16)

train_feat["u_in_diff1_bin"] = (
    (train_feat["u_in_diff1"] / DIFF_BIN).round().astype(np.int16)
)
test_feat["u_in_diff1_bin"] = (
    (test_feat["u_in_diff1"] / DIFF_BIN).round().astype(np.int16)
)

train_feat["u_in_absdiff_cum_bin"] = (
    (train_feat["u_in_absdiff_cum"] / ABSDIFF_CUM_BIN).round().astype(np.int32)
)
test_feat["u_in_absdiff_cum_bin"] = (
    (test_feat["u_in_absdiff_cum"] / ABSDIFF_CUM_BIN).round().astype(np.int32)
)


def build_tables(train_part: pd.DataFrame, suffix: str):
    grp_rich = (
        train_part.groupby(
            [
                "R",
                "C",
                "breath_step",
                "ts_bin",
                "u_out",
                "u_in_bin",
                "u_in_cum_bin",
                "u_in_lag1_bin",
                "u_in_roll3_bin",
                "u_in_dt_cum_bin",
                "u_in_diff1_bin",
                "u_in_absdiff_cum_bin",
            ],
            sort=False,
        )["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": f"pred_rich{suffix}"})
    )

    grp_mid2 = (
        train_part.groupby(
            [
                "R",
                "C",
                "breath_step",
                "ts_bin",
                "u_out",
                "u_in_bin",
                "u_in_lag1_bin",
                "u_in_roll3_bin",
                "u_in_dt_cum_bin",
            ],
            sort=False,
        )["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": f"pred_mid2{suffix}"})
    )

    grp_mid = (
        train_part.groupby(
            ["R", "C", "breath_step", "ts_bin", "u_out", "u_in_bin", "u_in_cum_bin"],
            sort=False,
        )["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": f"pred_mid{suffix}"})
    )

    grp_step_uout_uin = (
        train_part.groupby(["R", "C", "breath_step", "u_out", "u_in_bin"], sort=False)[
            "pressure"
        ]
        .mean()
        .reset_index()
        .rename(columns={"pressure": f"pred_step_uout_uin{suffix}"})
    )

    grp_simple_uout = (
        train_part.groupby(["R", "C", "breath_step", "ts_bin", "u_out"], sort=False)[
            "pressure"
        ]
        .mean()
        .reset_index()
        .rename(columns={"pressure": f"pred_simple_uout{suffix}"})
    )

    grp_simple = (
        train_part.groupby(["R", "C", "breath_step", "ts_bin"], sort=False)["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": f"pred_simple{suffix}"})
    )

    rc_step_mean = (
        train_part.groupby(["R", "C", "breath_step"], sort=False)["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": f"rc_step_mean{suffix}"})
    )

    rc_mean = (
        train_part.groupby(["R", "C"], sort=False)["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": f"rc_mean{suffix}"})
    )

    global_mean = float(train_part["pressure"].mean()) if len(train_part) else np.nan

    return {
        "grp_rich": grp_rich,
        "grp_mid2": grp_mid2,
        "grp_mid": grp_mid,
        "grp_step_uout_uin": grp_step_uout_uin,
        "grp_simple_uout": grp_simple_uout,
        "grp_simple": grp_simple,
        "rc_step_mean": rc_step_mean,
        "rc_mean": rc_mean,
        "global_mean": global_mean,
    }


train_insp = train_feat[train_feat["u_out"] == 0].copy()

tables_insp = build_tables(train_insp, suffix="_insp")

test_pred = test_feat.copy()

test_pred = test_pred.merge(
    tables_insp["grp_rich"],
    on=[
        "R",
        "C",
        "breath_step",
        "ts_bin",
        "u_out",
        "u_in_bin",
        "u_in_cum_bin",
        "u_in_lag1_bin",
        "u_in_roll3_bin",
        "u_in_dt_cum_bin",
        "u_in_diff1_bin",
        "u_in_absdiff_cum_bin",
    ],
    how="left",
)
test_pred = test_pred.merge(
    tables_insp["grp_mid2"],
    on=[
        "R",
        "C",
        "breath_step",
        "ts_bin",
        "u_out",
        "u_in_bin",
        "u_in_lag1_bin",
        "u_in_roll3_bin",
        "u_in_dt_cum_bin",
    ],
    how="left",
)
test_pred = test_pred.merge(
    tables_insp["grp_mid"],
    on=["R", "C", "breath_step", "ts_bin", "u_out", "u_in_bin", "u_in_cum_bin"],
    how="left",
)
test_pred = test_pred.merge(
    tables_insp["grp_step_uout_uin"],
    on=["R", "C", "breath_step", "u_out", "u_in_bin"],
    how="left",
)
test_pred = test_pred.merge(
    tables_insp["grp_simple_uout"],
    on=["R", "C", "breath_step", "ts_bin", "u_out"],
    how="left",
)
test_pred = test_pred.merge(
    tables_insp["grp_simple"], on=["R", "C", "breath_step", "ts_bin"], how="left"
)
test_pred = test_pred.merge(
    tables_insp["rc_step_mean"], on=["R", "C", "breath_step"], how="left"
)
test_pred = test_pred.merge(tables_insp["rc_mean"], on=["R", "C"], how="left")

global_mean = float(train_insp["pressure"].mean())

is_exp = test_pred["u_out"].to_numpy(dtype=np.int8) == 1

pred_rich = test_pred["pred_rich_insp"].to_numpy(dtype=np.float64)
pred_mid2 = test_pred["pred_mid2_insp"].to_numpy(dtype=np.float64)
pred_mid = test_pred["pred_mid_insp"].to_numpy(dtype=np.float64)
pred_step_uout_uin = test_pred["pred_step_uout_uin_insp"].to_numpy(dtype=np.float64)
pred_simple_uout = test_pred["pred_simple_uout_insp"].to_numpy(dtype=np.float64)
pred_simple = test_pred["pred_simple_insp"].to_numpy(dtype=np.float64)
pred_rc_step = test_pred["rc_step_mean_insp"].to_numpy(dtype=np.float64)
pred_rc = test_pred["rc_mean_insp"].to_numpy(dtype=np.float64)

pred = pred_rich
pred = np.where(np.isnan(pred), pred_mid2, pred)
pred = np.where(np.isnan(pred), pred_mid, pred)
pred = np.where(np.isnan(pred), pred_step_uout_uin, pred)
pred = np.where(np.isnan(pred), pred_simple_uout, pred)
pred = np.where(np.isnan(pred), pred_simple, pred)
pred = np.where(np.isnan(pred), pred_rc_step, pred)
pred = np.where(np.isnan(pred), pred_rc, pred)
pred = np.where(np.isnan(pred), global_mean, pred)

exp_fallback = pred_rc_step.copy()
exp_fallback = np.where(np.isnan(exp_fallback), pred_rc, exp_fallback)
exp_fallback = np.where(np.isnan(exp_fallback), global_mean, exp_fallback)
pred = np.where(is_exp, exp_fallback, pred)

pred = np.array([find_nearest(p) for p in pred], dtype=np.float64)

pred_df = pd.DataFrame(
    {"id": test_pred["id"].to_numpy(dtype=np.int64), "pressure": pred}
)

sub_out = sub[["id"]].merge(pred_df, on="id", how="left")
sub_out["pressure"] = sub_out["pressure"].fillna(global_mean).map(find_nearest)

sub_out.to_csv("submission.csv", index=False)

print(sub_out.head())
print("Wrote submission.csv with shape:", sub_out.shape)
print(
    "pressure min/max:",
    float(sub_out["pressure"].min()),
    float(sub_out["pressure"].max()),
)
print("Any NA pressure:", bool(sub_out["pressure"].isna().any()))
