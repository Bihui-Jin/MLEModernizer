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

0.136615405700708

# 6. Current score

2.51108

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.86378) has done: 'The crash happens because `avg('../input/gb-data-blending-recover')` points to a dataset directory that doesn’t exist in your provided environment, so `glob` finds zero files and `np.vstack` fails. I keep your blending/rounding-to-nearest-pressure logic intact, but add a safe fallback: if no blend files are found, create a baseline submission by predicting the global median train pressure for all rows (then snapping to nearest valid pressure). I also make path resolution robust by automatically selecting an existing competition data root (`/kaggle/input/...`), so reading `train.csv`/`sample_submission.csv` works in this environment. The script always write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 9.92097) has done: 'Your current score (10.86378 MAE) is far worse than the target (0.1366), and the main reason is that the code falls back to predicting a constant median pressure because the blend directory doesn’t exist in this environment. To move the score sharply toward the target without changing the overall “make predictions then snap to nearest valid pressure and write submission.csv” semantics, I keep your rounding-to-nearest-pressure logic but replace the fallback with a simple, fast, fully local baseline model: per-(R,C,time_step) median pressure computed from train and joined onto test. For any unmatched rows, it safely fall back to the global median, then still apply your `find_nearest` snapping. This stays within Kaggle constraints, runs quickly on CPU, and should drastically reduce MAE compared with a constant prediction.'
- What this solution (achieved 1.62713) has done: 'Your score is still far from the target because the fallback model (median by `(R,C,time_step)`) ignores the strong within-breath dynamics driven by `u_in/u_out` and accumulated flow; it effectively predicts a coarse lookup table. I keep your overall “predict → snap to nearest valid pressure → write submission.csv” semantics, but improve the fallback to a stronger (still simple/fast) per-time-step KNN regression using only local features (`R,C,time_step,u_in,u_out` plus engineered `u_in` cumulative sums within each breath). This is a minimal change contained entirely inside `avg()`’s “no blend files found” path, so blending logic remains intact. The result should move MAE sharply down toward the target band while still finishing within the time limit and producing a valid `submission.csv`.'
- What this solution (achieved 1.25419) has done: 'Your current MAE (1.627) is still far above the target (0.1366), so we should improve the fallback model (used when no blend files exist) while keeping your overall “predict → snap to nearest valid pressure → write submission.csv” semantics unchanged. The main issue is that KNN is being fit on all 5.4M rows with unscaled features, which both hurts performance and (on Kaggle) is often too slow/memory-heavy; we can instead use a compact, breath-structured representation (80 steps per breath) and standardize features before KNN to improve neighbor quality. Concretely: train KNN on per-timestep features + lag features on a sampled set of breaths (still deterministic, no approximations to training loop since there isn’t one) and then predict test in chunks, finally applying your existing nearest-pressure snapping. These are minimal, localized changes inside the fallback path only and should move MAE substantially down toward the target band while staying within time limits and producing a valid submission.'
- What this solution (achieved 5.65838) has done: 'Your current MAE (1.254) is still far above the target (0.1366), so we should improve the fallback model that’s actually generating your submission (since the blend directory is absent). Keeping the same overall semantics (make predictions → snap to nearest valid pressure → write `submission.csv`), the smallest high-impact change is to use a much stronger deterministic lookup: per-(R,C,step,u_in,u_out,u_in_cumsum) median pressure computed from train, with safe backoffs. This leverages the competition’s discrete pressure levels and the strong dependence on cumulative inspiratory flow without changing architectures/training loops (there are none). We also ensure correct `time_step` alignment by using the within-breath step index (0..79), avoiding floating-point join issues that can degrade the table match rate.'
- What this solution (achieved 3.97101) has done: 'Your current score (5.65838 MAE, lower-is-better) is far above the target (0.1366), and the main cause is that the fallback lookup is matching poorly because it keys on raw floating `u_in` values (nearly all unique), leading to lots of NaNs and coarse backoff predictions. To move the score closer to target while preserving your existing “lookup → backoff → snap to nearest valid pressure → write submission.csv” core semantics, I only change the fallback to quantize `u_in` (and keep `u_in_cumsum` quantized) so the highest-granularity table actually matches test rows. I also incorporate `u_in_cumsum` into the mid-level backoff keys (still pure median lookups) to better capture within-breath dynamics without introducing any new model/training loop. Everything else (data paths, blending behavior, `find_nearest`, output schema) remains unchanged and it still always write a valid `submission.csv`.'
- What this solution (achieved 2.66468) has done: 'Your current MAE (3.97, lower-is-better) is still far above the target (0.1366), so we should improve the fallback (since the blend directory is absent) without changing your overall “lookup → backoff → snap to nearest valid pressure → write submission.csv” approach. The largest issue is that your “fine” lookup keys are too strict and also duplicated (key1==key2), so most rows miss and fall back to coarse medians; we (1) remove the duplicate merge and (2) add a more reliable mid-level key that uses both quantized `u_in` and quantized `u_in_cumsum`, plus slightly coarser quantization to increase match rate. We also make sure the merge output is re-aligned to `sample_submission` by `id` before writing, to avoid any accidental ordering mismatch. These changes are localized inside `_lookup_fallback()` only and keep the same prediction semantics and nearest-pressure snapping.'
- What this solution (achieved 2.78496) has done: 'Your current score (2.66468 MAE; lower is better) is still far above the target (0.1366), and the bottleneck is the fallback lookup missing most rows because it keys on near-unique `u_in` and `u_in_cumsum` values even after rounding to integers. I keep your exact core approach (multi-level median lookup → backoff → snap to nearest valid pressure → write submission.csv) but change the fallback feature quantization to create much higher match rates: quantize `u_in` to 0.5 steps and `u_in_cumsum` to 1.0 steps, and add a robust mid-level table keyed on both quantized values. I also add a tiny, safe improvement by computing cumulative sums on the quantized `u_in` (more consistent between train/test than using raw float then rounding), without changing any modeling paradigm. These are localized changes inside `_lookup_fallback()` only, keep I/O paths unchanged, and still finish quickly.'
- What this solution (achieved 2.35482) has done: 'Your current MAE (2.78496) is still far above the target (0.1366), so we should improve the fallback path (since your blend directory likely has no usable CSVs) while keeping the same overall “multi-level median lookup → backoff → snap to nearest valid pressure → write submission.csv” semantics. The biggest issue is the lookup keys still miss too often because `u_in_q` at 0.5 steps and `u_in_cumsum_q1` at 1.0 steps remain too granular and slightly inconsistent across breaths; we increase match rate by using coarser, deterministic quantization and adding a robust intermediate backoff keyed on both quantized `u_in` and quantized cumulative sum. We also keep your existing merge-to-sample-by-id to guarantee correct submission row alignment and continue snapping to the discrete pressure grid. These changes are localized inside `_lookup_fallback()` only and preserve the rest of your pipeline unchanged.'
- What this solution (achieved 2.35482) has done: 'Your current MAE (2.35482, lower-is-better) is still far above the target (0.1366), and the main limitation is the fallback lookup missing many rows because its keys are not aligned with the actual scoring rule (only inspiratory phase where `u_out==0`). I keep your exact core approach (multi-level median lookup → backoff → snap to nearest valid pressure → write `submission.csv`) but compute the lookup tables using only `u_out==0` training rows (the scored regime), which should make medians much closer to what the metric evaluates. I also add a final backoff table specifically for inspiratory rows keyed on `(R,C,step,u_in_q)` learned from `u_out==0`, used only when the row to predict has `u_out==0`, preserving your existing hierarchy while improving match quality where it matters. All I/O paths, blending behavior, and nearest-pressure snapping remain unchanged, and it still always produces a valid `submission.csv`.'
- What this solution (achieved 2.30554) has done: 'Your current score (2.35482 MAE) is still far above the target (0.1366), so we need a stronger fallback prediction while keeping your existing “multi-level median lookup → backoff → snap to nearest valid pressure → write submission.csv” logic intact. The main reason the lookup underperforms is that it treats `u_in_cumsum` as continuous and only coarsely buckets it, but the true pressure is very tightly tied to breath dynamics; we can greatly increase match quality by using an additional state-like feature (`u_in` lag / delta) and building one extra median table keyed on it. This is a minimal, localized change inside `_lookup_fallback()` only: we add `u_in_q_lag1` and `du_in_q` (both deterministic) and insert one higher-priority table and one mid-level backoff that use these features, while preserving all existing tables and backoffs. This should reduce MAE substantially toward the target without changing the overall approach or output semantics.'
- What this solution (achieved 2.30554) has done: 'Your current MAE (2.30554) is still far above the target (0.1366), so we should strengthen the existing fallback lookup (used because the blend directory has no CSVs) while keeping the same overall “median-lookup → backoff → snap to nearest valid pressure → write submission.csv” semantics. The biggest win with minimal change is to make the lookup keys match more often by (1) quantizing `du_in_q` (right now it’s still too unique) and (2) using *separately trained* lookup tables for inspiratory (`u_out==0`, scored) vs expiratory (`u_out==1`, unscored) rows, so each regime uses medians from the same regime. We keep all current tables/backoffs, but route inspiratory test rows through inspiratory-trained tables and expiratory rows through all-rows tables, improving accuracy where the metric cares without changing the pipeline shape. Submission writing and snapping remain unchanged.'
- What this solution (achieved 2.30554) has done: 'Your current score (2.30554 MAE) is still far above the target (0.1366), so we should improve the fallback path that’s actually producing the submission (since the blend directory has no files) while keeping the same “median lookup → backoff → snap to nearest pressure grid → write submission.csv” approach. The biggest issue is that your lookup tables are trained on *all* inspiratory rows, including breaths that are incomplete in the provided train file, which injects noisy/shifted `step` dynamics and hurts matching quality; we can filter to only full 80-step breaths (and use the same filtering for test) without changing the modeling paradigm. Additionally, using `time_step`-derived steps can be fragile; you already use `cumcount()` steps, so we just ensure train/test both only use consistent 0..79 steps and we add one extra very-light backoff keyed on `(R,C,step,u_in_cumsum_q2)` for inspiratory rows to better capture state when `u_in` quantization misses. These changes stay fully within your existing fallback logic, preserve output semantics (including `find_nearest` snapping), and should move the score materially closer to the target.'
- What this solution (achieved 2.51108) has done: 'Your current score (2.30554 MAE; lower is better) is still far above the target (0.1366), so we should improve the fallback lookup predictions that are being used when no blend files are found, while keeping your existing “median lookup → backoff → snap to nearest pressure grid → write submission.csv” pipeline intact. The smallest high-impact change is to make the lookup keys better align with ventilator dynamics by adding one more deterministic state feature (`u_in` exponentially weighted moving average within each breath) and using it only in the top-priority inspiratory/all tables so match quality improves without changing the overall approach. We also slightly adjust quantization to 0.5 for `u_in_q` (and derive the other features from that) to increase exact key matches between train and test while remaining deterministic and fast. All I/O paths, blending behavior, and the final `find_nearest` snapping and submission format remain unchanged.'
- What this solution (achieved 2.51108) has done: 'Your current score is far above the target (lower-is-better), so we should improve the fallback path that actually generates predictions when no blend CSVs are found. The biggest issue is that the fallback currently drops all non-80-step breaths, which (on this dataset) removes a large portion of test rows and forces them to be predicted as NaN→filled with medians, severely hurting MAE. I keep your exact multi-level median-lookup + backoff + nearest-pressure snapping approach, but change the preprocessing to retain all test rows and only use full 80-step breaths for training tables. I also ensure the final prediction dataframe covers every test `id` (including breaths shorter than 80) by preparing features for all rows and using the lookup tables wherever keys match.'

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
def _pick_data_root():
    candidates = [
        "../input/ventilator-pressure-prediction",
        "/kaggle/input/ventilator-pressure-prediction",
        "/kaggle/data/ventilator-pressure-prediction",
        "/kaggle/input",
        "/kaggle/data",
    ]
    for c in candidates:
        if os.path.exists(c):
            if os.path.isdir(c) and os.path.exists(os.path.join(c, "train.csv")):
                return c
            if os.path.isdir(c) and os.path.exists(
                os.path.join(c, "ventilator-pressure-prediction", "train.csv")
            ):
                return os.path.join(c, "ventilator-pressure-prediction")
    return "."


DATA_ROOT = _pick_data_root()
TRAIN_PATH = os.path.join(DATA_ROOT, "train.csv")
TEST_PATH = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

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
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.7 + b.pressure * 0.3
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a


def avg(dp, out_path="submission.csv"):
    input_list = []
    for i in glob.iglob(f"{dp}/*"):
        if os.path.isfile(i) and i.lower().endswith(".csv"):
            input_list.append(i)

    output = pd.read_csv(SAMPLE_SUB_PATH)

    def _lookup_fallback():
        df_test = pd.read_csv(TEST_PATH)

        def _prep(df, is_train: bool, keep_only_full_breaths: bool):
            df = df.copy()
            g0 = df.groupby("breath_id", sort=False)

            if keep_only_full_breaths:
                breath_sizes = g0.size()
                full_breath_ids = breath_sizes.index[breath_sizes == 80]
                df = df[df["breath_id"].isin(full_breath_ids)].copy()

            g = df.groupby("breath_id", sort=False)
            df["step"] = g.cumcount().astype(np.int16)

            df["u_in"] = df["u_in"].astype(np.float32)

            u_in_q = (np.round(df["u_in"] / 0.5) * 0.5).astype(np.float32)
            df["u_in_q"] = u_in_q

            df["u_in_q_lag1"] = g["u_in_q"].shift(1).fillna(0.0).astype(np.float32)
            df["du_in_q"] = (df["u_in_q"] - df["u_in_q_lag1"]).astype(np.float32)
            df["du_in_q2"] = (np.round(df["du_in_q"] / 1.0) * 1.0).astype(np.float32)

            df["u_in_cumsum_q"] = g["u_in_q"].cumsum().astype(np.float32)
            df["u_in_cumsum_q2"] = (np.round(df["u_in_cumsum_q"] / 2.0) * 2.0).astype(
                np.float32
            )

            u_in_ewm = (
                g["u_in_q"].apply(lambda s: s.ewm(alpha=0.3, adjust=False).mean())
            ).reset_index(level=0, drop=True)
            df["u_in_ewm"] = u_in_ewm.astype(np.float32)
            df["u_in_ewm_q"] = (np.round(df["u_in_ewm"] / 0.5) * 0.5).astype(np.float32)

            if is_train:
                df["pressure"] = df["pressure"].astype(np.float32)

            df["R"] = df["R"].astype(np.int16)
            df["C"] = df["C"].astype(np.int16)
            df["u_out"] = df["u_out"].astype(np.int8)
            return df

        tr = _prep(
            df_train[["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]],
            is_train=True,
            keep_only_full_breaths=True,
        )
        te = _prep(
            df_test[["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]],
            is_train=False,
            keep_only_full_breaths=False,
        )

        tr_insp = tr[tr["u_out"] == 0]

        def _make_tables(tr_src: pd.DataFrame, prefix: str):
            key0 = [
                "R",
                "C",
                "step",
                "u_out",
                "u_in_q",
                "u_in_ewm_q",
                "u_in_q_lag1",
                "du_in_q2",
                "u_in_cumsum_q2",
            ]
            tab0 = (
                tr_src.groupby(key0, sort=False)["pressure"]
                .median()
                .reset_index()
                .rename(columns={"pressure": f"{prefix}p0"})
            )

            key1 = ["R", "C", "step", "u_out", "u_in_q", "u_in_cumsum_q2"]
            tab1 = (
                tr_src.groupby(key1, sort=False)["pressure"]
                .median()
                .reset_index()
                .rename(columns={"pressure": f"{prefix}p1"})
            )

            key2 = ["R", "C", "step", "u_out", "u_in_q"]
            tab2 = (
                tr_src.groupby(key2, sort=False)["pressure"]
                .median()
                .reset_index()
                .rename(columns={"pressure": f"{prefix}p2"})
            )

            key2c = ["R", "C", "step", "u_out", "u_in_q", "du_in_q2"]
            tab2c = (
                tr_src.groupby(key2c, sort=False)["pressure"]
                .median()
                .reset_index()
                .rename(columns={"pressure": f"{prefix}p2c"})
            )

            key2b = ["R", "C", "step", "u_out", "u_in_cumsum_q2"]
            tab2b = (
                tr_src.groupby(key2b, sort=False)["pressure"]
                .median()
                .reset_index()
                .rename(columns={"pressure": f"{prefix}p2b"})
            )

            key2s = ["R", "C", "step", "u_in_cumsum_q2"]
            tab2s = (
                tr_src.groupby(key2s, sort=False)["pressure"]
                .median()
                .reset_index()
                .rename(columns={"pressure": f"{prefix}p2s"})
            )

            key3 = ["R", "C", "step", "u_out"]
            tab3 = (
                tr_src.groupby(key3, sort=False)["pressure"]
                .median()
                .reset_index()
                .rename(columns={"pressure": f"{prefix}p3"})
            )

            key4 = ["R", "C", "step"]
            tab4 = (
                tr_src.groupby(key4, sort=False)["pressure"]
                .median()
                .reset_index()
                .rename(columns={"pressure": f"{prefix}p4"})
            )

            return {
                "key0": key0,
                "tab0": tab0,
                "key1": key1,
                "tab1": tab1,
                "key2": key2,
                "tab2": tab2,
                "key2c": key2c,
                "tab2c": tab2c,
                "key2b": key2b,
                "tab2b": tab2b,
                "key2s": key2s,
                "tab2s": tab2s,
                "key3": key3,
                "tab3": tab3,
                "key4": key4,
                "tab4": tab4,
            }

        tabs_insp = _make_tables(tr_insp, prefix="i_")
        tabs_all = _make_tables(tr, prefix="a_")

        key_insp = ["R", "C", "step", "u_in_q"]
        tab_insp = (
            tr_insp.groupby(key_insp, sort=False)["pressure"]
            .median()
            .reset_index()
            .rename(columns={"pressure": "p_insp"})
        )

        global_med_all = float(tr["pressure"].median())
        global_med_insp = float(tr_insp["pressure"].median())

        m = te

        m = m.merge(tabs_insp["tab0"], on=tabs_insp["key0"], how="left")
        m = m.merge(tabs_insp["tab1"], on=tabs_insp["key1"], how="left")
        m = m.merge(tabs_insp["tab2"], on=tabs_insp["key2"], how="left")
        m = m.merge(tabs_insp["tab2c"], on=tabs_insp["key2c"], how="left")
        m = m.merge(tabs_insp["tab2b"], on=tabs_insp["key2b"], how="left")
        m = m.merge(tabs_insp["tab2s"], on=tabs_insp["key2s"], how="left")
        m = m.merge(tabs_insp["tab3"], on=tabs_insp["key3"], how="left")
        m = m.merge(tabs_insp["tab4"], on=tabs_insp["key4"], how="left")

        m = m.merge(tabs_all["tab0"], on=tabs_all["key0"], how="left")
        m = m.merge(tabs_all["tab1"], on=tabs_all["key1"], how="left")
        m = m.merge(tabs_all["tab2"], on=tabs_all["key2"], how="left")
        m = m.merge(tabs_all["tab2c"], on=tabs_all["key2c"], how="left")
        m = m.merge(tabs_all["tab2b"], on=tabs_all["key2b"], how="left")
        m = m.merge(tabs_all["tab2s"], on=tabs_all["key2s"], how="left")
        m = m.merge(tabs_all["tab3"], on=tabs_all["key3"], how="left")
        m = m.merge(tabs_all["tab4"], on=tabs_all["key4"], how="left")

        m = m.merge(tab_insp, on=key_insp, how="left")

        is_insp = m["u_out"].to_numpy() == 0

        base_pred_insp = (
            m["i_p0"]
            .fillna(m["i_p1"])
            .fillna(m["i_p2"])
            .fillna(m["i_p2c"])
            .fillna(m["i_p2b"])
            .fillna(m["i_p2s"])
            .fillna(m["i_p3"])
            .fillna(m["i_p4"])
        ).to_numpy(dtype=np.float32, copy=False)

        base_pred_all = (
            m["a_p0"]
            .fillna(m["a_p1"])
            .fillna(m["a_p2"])
            .fillna(m["a_p2c"])
            .fillna(m["a_p2b"])
            .fillna(m["a_p2s"])
            .fillna(m["a_p3"])
            .fillna(m["a_p4"])
        ).to_numpy(dtype=np.float32, copy=False)

        pred = np.where(is_insp, base_pred_insp, base_pred_all).astype(
            np.float32, copy=False
        )

        pinsp = m["p_insp"].to_numpy(dtype=np.float32, copy=False)
        nan_mask = np.isnan(pred)
        if nan_mask.any():
            fill_vals = np.where(is_insp, pinsp, np.nan).astype(np.float32, copy=False)
            pred[nan_mask] = fill_vals[nan_mask]

        nan_mask = np.isnan(pred)
        if nan_mask.any():
            pred[nan_mask] = np.where(
                is_insp[nan_mask], global_med_insp, global_med_all
            ).astype(np.float32, copy=False)

        pred_df = pd.DataFrame({"id": m["id"].to_numpy(), "pressure": pred})
        out = output.merge(pred_df, on="id", how="left", suffixes=("", "_pred"))
        out["pressure"] = out["pressure_pred"].astype(np.float32)
        out = out.drop(columns=["pressure_pred"])

        out["pressure"] = out["pressure"].apply(find_nearest)
        out.to_csv(out_path, index=False)
        return out

    if len(input_list) == 0:
        return _lookup_fallback()

    preds = []
    for fp in input_list:
        dfp = pd.read_csv(fp)
        if "pressure" not in dfp.columns:
            continue
        p = dfp["pressure"].to_numpy().ravel()
        if len(p) != len(output):
            continue
        preds.append(p)

    if len(preds) == 0:
        return _lookup_fallback()

    output["pressure"] = np.median(np.vstack(preds), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(out_path, index=False)
    return output




## === cell 2
avg("../input/gb-data-blending-recover", out_path="submission.csv")
sub = pd.read_csv("submission.csv")
print("Wrote submission.csv with columns:", list(sub.columns), "and rows:", len(sub))
print(sub.head())
