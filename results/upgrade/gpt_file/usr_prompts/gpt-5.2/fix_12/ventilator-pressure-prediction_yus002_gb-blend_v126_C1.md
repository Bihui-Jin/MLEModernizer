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

0.1412693199059811

# 6. Current score

3.86762

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 14.70637) has done: 'Your notebook fails because it tries to read external blend CSVs that don’t exist in this Kaggle environment, so we remove that dependency and instead generate a single valid submission from the provided `sample_submission.csv`. To keep core semantics intact, we reuse your existing “snap predictions to nearest known pressure” logic, but we create a simple, deterministic baseline prediction derived only from `u_in` mapped onto the training pressure range (then snapped to the nearest discrete training pressure). This guarantees end-to-end execution, produces a correctly formatted `submission.csv`, and should yield a non-trivial MAE (better than all-zeros) without introducing new modeling/training code. We also make the input path robust to both `../input/...` and the provided `/kaggle/input/...` layout.'
- What this solution (achieved 11.88896) has done: 'Your current score (14.70637 MAE) is far worse than the target (0.1413), so we should improve predictions while keeping your “map to pressure then snap to nearest known pressure” core logic intact. The biggest issue is that your baseline ignores the time-series nature and the lung attributes (R, C) that strongly affect pressure, so we add a minimal, deterministic calibration using only training-set averages. Concretely, we (1) compute mean pressure per (R,C,time_step) from train, (2) use that as the primary prediction for each test row, and (3) fall back to your existing u_in-based mapping when a (R,C,time_step) bucket is missing, then snap to the nearest discrete training pressure as you already do. This preserves evaluation semantics, runs fast (groupby + merge), and should move the MAE much closer to the target without changing to a new model/training loop.'
- What this solution (achieved 6.1959) has done: 'Your current MAE (11.88896) is far worse than the target (0.1413), so we should improve predictions while keeping your existing “calibration via train averages + fallback + snap-to-nearest-discrete-pressure” core logic. The main weakness is averaging pressure by raw floating `time_step`, which can create fragile/overfitted keys; instead we calibrate by `(R, C, time_step_index_within_breath)` which is the true aligned sequence position (0–79) shared across breaths. We also improve the fallback slightly (still deterministic and simple) by using a precomputed mean pressure curve per `(R,C,step)` first, then a global mean curve per `step`, and only then your `u_in` min-max mapping. This keeps evaluation semantics identical (predict pressure per row; snap to valid discrete pressures) and should substantially reduce MAE without changing to a new model/training loop.'
- What this solution (achieved 6.19023) has done: 'Your current MAE (6.1959, lower is better) is far worse than the target (0.1413), so we should improve predictions while preserving your existing “train-average calibration + fallback + snap-to-nearest discrete pressure” core logic. The biggest remaining gap is that the prediction ignores the fact that the evaluation scores only the inspiratory phase (u_out==0), and pressure dynamics depend strongly on the control trajectory; we can incorporate u_out by forcing a safer, low-pressure prediction on expiratory rows and by conditioning the calibration on u_out. Concretely, we (1) build mean pressure tables by (R,C,step,u_out) and (step,u_out), (2) fall back exactly as before, and (3) override predictions where u_out==1 with the global mean pressure for u_out==1 (then snap), which should reduce error from spurious high predictions during expiration while keeping the approach deterministic and lightweight. These changes are small (extra groupbys/merge + a simple override) and should move MAE substantially toward the target without changing the overall method.'
- What this solution (achieved 4.75512) has done: 'We keep your existing “train-average calibration + step alignment + fallback + snap-to-nearest pressure” approach, but make two minimal changes that typically reduce MAE a lot for this competition: (1) condition the calibration on a coarse `u_in` bin so the mean tables capture different inspiratory control intensities, and (2) only force the expiratory (`u_out==1`) override after a short early-expiration grace window (a few steps) because pressure often remains elevated briefly after `u_out` flips. Both changes preserve your semantics (still deterministic, still table-lookup means with fallbacks, still snapping to discrete pressures) and should move the score down toward the target without introducing any new model/training loop. The rest of the pipeline, file paths, and submission writing remain unchanged.'
- What this solution (achieved 4.17026) has done: 'Your current MAE (4.75512, lower is better) is still far above the target (0.14127), so we should improve predictions while preserving your existing “train-mean lookup tables + fallbacks + snap-to-nearest discrete pressure” core logic. The main issue is that conditioning on coarse `u_in_bin` loses too much information; we can keep the same approach but replace the coarse bin with a finer, quantile-based `u_in` bin computed from train and applied to test, which typically reduces MAE without changing the method. Second, we keep your expiration override logic but compute a step-dependent expiratory mean curve (per step) instead of a single global mean, which is still the same calibration-table idea and reduces systematic error during expiration. All paths and the submission writing remain unchanged, and the script still deterministically writes a valid `submission.csv`.'
- What this solution (achieved 4.20784) has done: 'We keep your exact lookup-table + fallback + expiratory-override + snap-to-discrete-pressure core logic, but make the conditioning on `u_in` more informative without changing the approach. Concretely, we (1) replace quantile binning with an exact join on a rounded `u_in` grid (0.0–100.0 at 0.1 resolution) so the table captures much more of the control signal, and (2) add one extra minimal fallback table keyed by `(R,C,step,u_out,u_in_round)` that’s coarser than 0.1 but still tighter than bins (rounded to 1.0) to improve coverage with little risk. This should reduce MAE substantially versus bins while remaining purely deterministic groupby/merge calibration and keeping evaluation semantics identical. The rest (paths, step alignment, expiration override behavior, snapping, CSV writing) stays the same.'
- What this solution (achieved 4.17799) has done: 'Your MAE (4.20784, lower is better) is still far above the target (0.14127), so we should improve predictions while keeping your deterministic lookup-table + fallback + expiratory-override + snap-to-discrete-pressure core logic intact. The biggest remaining issue is that the current tables ignore the strong dependence of pressure on recent control history within a breath, so we add minimal lag features (previous u_in and cumulative u_in) and build additional mean lookup tables keyed on those lags (rounded) to better match dynamics. We keep all existing tables and fallbacks, simply inserting the new lag-conditioned tables at the top of the fallback chain (more specific first), and we keep the same expiratory override and nearest-pressure snapping. This is still just groupby/merge calibration (no new model/training loop), but it typically moves MAE substantially downward on this competition.'
- What this solution (achieved 4.16561) has done: 'Your current MAE (4.17799, lower is better) is still far above the target (0.14127), so we should improve predictions while keeping your exact “train-mean lookup tables + fallback chain + expiratory override + snap-to-nearest discrete pressure” core logic intact. The biggest low-risk gain available within this same approach is to condition the lookup tables on a tiny bit more recent history without changing to a new model: add `u_out` lag, `u_in` lag2, and a short-window rolling mean of `u_in`, then add these as *additional* more-specific mean tables at the very top of the fallback chain. This stays in the same deterministic groupby/merge paradigm (no new training loop), but typically reduces systematic error because pressure depends on whether the valve just opened and on recent `u_in` trajectory. Finally, we keep your submission writing unchanged and still snap predictions to valid discrete pressures.'
- What this solution (achieved 4.16561) has done: 'Your current MAE (4.16561, lower is better) is still far above the target (0.14127), so we should improve predictions while keeping your exact deterministic lookup-table + fallback chain + expiratory override + snap-to-nearest-discrete-pressure core logic intact. The main minimal gain left in this paradigm is to condition the calibration tables on lung attributes without exploding sparsity: add a small “breath progress” feature (normalized step) and use it only for expiratory override and one mid-specific fallback, which improves generalization vs. high-cardinality history keys. We also fix a critical submission-alignment risk: `id` is globally unique already, but your `df_test` is re-sorted and later `out` is sorted by `id`; we ensure predictions are carried strictly by `id` merge to avoid any accidental misalignment. Finally, we replace the slow `np.vectorize(find_nearest)` snapping with a fully vectorized `searchsorted` snap that preserves identical semantics but reduces runtime and variance risk, keeping everything within the same approach.'
- What this solution (achieved 3.86762) has done: 'Your MAE (4.16561, lower is better) is still far above the target (0.14127), so we should improve predictions while keeping your deterministic lookup-table + fallback chain + expiratory override + snap-to-nearest-known-pressure core logic intact. The main low-risk gain within this same paradigm is to incorporate a tiny amount of additional history that is known to matter for pressure: the change in u_in (derivative) and the accumulated “area” under u_in (integral proxy), both computed per breath and then rounded for stable joins. We add two new mean calibration tables keyed on these new history features (plus your existing keys) and place them at the very top of the fallback chain (most specific first), without removing any existing fallbacks. This should reduce systematic error (especially around ramps/plateaus) while preserving your evaluation semantics, keeping runtime under control, and still writing a valid `submission.csv`.'

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
DATASET_DIR_CANDIDATES = [
    "../input/ventilator-pressure-prediction",
    "/kaggle/input/ventilator-pressure-prediction",
    "/kaggle/data/ventilator-pressure-prediction",
    "/kaggle/input",  # fallback if files are directly under input
    "/kaggle/data",
]
DATASET_DIR = None
for d in DATASET_DIR_CANDIDATES:
    if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
        os.path.join(d, "test.csv")
    ):
        DATASET_DIR = d
        break
if DATASET_DIR is None:
    if os.path.exists("/kaggle/data/train.csv") and os.path.exists(
        "/kaggle/data/test.csv"
    ):
        DATASET_DIR = "/kaggle/data"
    elif os.path.exists("/kaggle/input/train.csv") and os.path.exists(
        "/kaggle/input/test.csv"
    ):
        DATASET_DIR = "/kaggle/input"
    else:
        raise FileNotFoundError(
            "Could not locate train.csv/test.csv in expected Kaggle input locations."
        )

train_path = os.path.join(DATASET_DIR, "train.csv")
test_path = os.path.join(DATASET_DIR, "test.csv")
sample_sub_path = os.path.join(DATASET_DIR, "sample_submission.csv")

df_train = pd.read_csv(train_path)

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


def snap_to_known_pressures(pred_array: np.ndarray) -> np.ndarray:
    arr = np.asarray(pred_array, dtype=np.float64)
    idx = np.searchsorted(sorted_pressures, arr, side="left")

    idx0 = np.clip(idx - 1, 0, total_pressures_len - 1)
    idx1 = np.clip(idx, 0, total_pressures_len - 1)

    p0 = sorted_pressures[idx0]
    p1 = sorted_pressures[idx1]

    choose_lower = (idx > 0) & (
        (idx == total_pressures_len) | (np.abs(arr - p0) < np.abs(p1 - arr))
    )
    out = np.where(choose_lower, p0, p1)
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
    output = pd.read_csv(sample_sub_path)
    output.pressure = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.58 + b.pressure * 0.42
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
df_test = pd.read_csv(test_path)
sub = pd.read_csv(sample_sub_path)

df_train = df_train.sort_values(
    ["breath_id", "time_step"], kind="mergesort"
).reset_index(drop=True)
df_test = df_test.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
    drop=True
)

df_train["step"] = df_train.groupby("breath_id", sort=False).cumcount().astype(np.int16)
df_test["step"] = df_test.groupby("breath_id", sort=False).cumcount().astype(np.int16)

MAX_STEP = int(df_train["step"].max())  # should be 79
df_train["step_frac_r2"] = np.round(
    df_train["step"].to_numpy(dtype=np.float64) / MAX_STEP, 2
).astype(np.float32)
df_test["step_frac_r2"] = np.round(
    df_test["step"].to_numpy(dtype=np.float64) / MAX_STEP, 2
).astype(np.float32)

UIN_ROUND_DECIMALS_FINE = 1  # 0.1 resolution
UIN_ROUND_DECIMALS_COARSE = 0  # 1.0 resolution fallback

df_train["u_in_r1"] = np.round(
    df_train["u_in"].to_numpy(dtype=np.float64), UIN_ROUND_DECIMALS_FINE
).astype(np.float32)
df_test["u_in_r1"] = np.round(
    df_test["u_in"].to_numpy(dtype=np.float64), UIN_ROUND_DECIMALS_FINE
).astype(np.float32)

df_train["u_in_r0"] = np.round(
    df_train["u_in"].to_numpy(dtype=np.float64), UIN_ROUND_DECIMALS_COARSE
).astype(np.float32)
df_test["u_in_r0"] = np.round(
    df_test["u_in"].to_numpy(dtype=np.float64), UIN_ROUND_DECIMALS_COARSE
).astype(np.float32)

df_train["u_in_lag1"] = (
    df_train.groupby("breath_id", sort=False)["u_in"]
    .shift(1)
    .fillna(0.0)
    .astype(np.float32)
)
df_test["u_in_lag1"] = (
    df_test.groupby("breath_id", sort=False)["u_in"]
    .shift(1)
    .fillna(0.0)
    .astype(np.float32)
)

df_train["u_in_lag1_r1"] = np.round(
    df_train["u_in_lag1"].to_numpy(dtype=np.float64), 1
).astype(np.float32)
df_test["u_in_lag1_r1"] = np.round(
    df_test["u_in_lag1"].to_numpy(dtype=np.float64), 1
).astype(np.float32)

df_train["u_in_cum"] = (
    df_train.groupby("breath_id", sort=False)["u_in"].cumsum().astype(np.float32)
)
df_test["u_in_cum"] = (
    df_test.groupby("breath_id", sort=False)["u_in"].cumsum().astype(np.float32)
)

df_train["u_in_cum_r0"] = np.round(
    df_train["u_in_cum"].to_numpy(dtype=np.float64), 0
).astype(np.float32)
df_test["u_in_cum_r0"] = np.round(
    df_test["u_in_cum"].to_numpy(dtype=np.float64), 0
).astype(np.float32)

df_train["u_in_lag2"] = (
    df_train.groupby("breath_id", sort=False)["u_in"]
    .shift(2)
    .fillna(0.0)
    .astype(np.float32)
)
df_test["u_in_lag2"] = (
    df_test.groupby("breath_id", sort=False)["u_in"]
    .shift(2)
    .fillna(0.0)
    .astype(np.float32)
)
df_train["u_in_lag2_r1"] = np.round(
    df_train["u_in_lag2"].to_numpy(dtype=np.float64), 1
).astype(np.float32)
df_test["u_in_lag2_r1"] = np.round(
    df_test["u_in_lag2"].to_numpy(dtype=np.float64), 1
).astype(np.float32)

df_train["u_out_lag1"] = (
    df_train.groupby("breath_id", sort=False)["u_out"]
    .shift(1)
    .fillna(0)
    .astype(np.int8)
)
df_test["u_out_lag1"] = (
    df_test.groupby("breath_id", sort=False)["u_out"].shift(1).fillna(0).astype(np.int8)
)

df_train["u_in_rm3"] = (
    df_train.groupby("breath_id", sort=False)["u_in"]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
    .astype(np.float32)
)
df_test["u_in_rm3"] = (
    df_test.groupby("breath_id", sort=False)["u_in"]
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
    .astype(np.float32)
)
df_train["u_in_rm3_r1"] = np.round(
    df_train["u_in_rm3"].to_numpy(dtype=np.float64), 1
).astype(np.float32)
df_test["u_in_rm3_r1"] = np.round(
    df_test["u_in_rm3"].to_numpy(dtype=np.float64), 1
).astype(np.float32)

df_train["u_in_diff1"] = (df_train["u_in"] - df_train["u_in_lag1"]).astype(np.float32)
df_test["u_in_diff1"] = (df_test["u_in"] - df_test["u_in_lag1"]).astype(np.float32)
df_train["u_in_diff1_r1"] = np.round(
    df_train["u_in_diff1"].to_numpy(dtype=np.float64), 1
).astype(np.float32)
df_test["u_in_diff1_r1"] = np.round(
    df_test["u_in_diff1"].to_numpy(dtype=np.float64), 1
).astype(np.float32)

df_train["u_in_area"] = (
    df_train.groupby("breath_id", sort=False)
    .apply(lambda g: ((g["u_in"] + g["u_in"].shift(1).fillna(0.0)) * 0.5).cumsum())
    .reset_index(level=0, drop=True)
    .astype(np.float32)
)
df_test["u_in_area"] = (
    df_test.groupby("breath_id", sort=False)
    .apply(lambda g: ((g["u_in"] + g["u_in"].shift(1).fillna(0.0)) * 0.5).cumsum())
    .reset_index(level=0, drop=True)
    .astype(np.float32)
)
df_train["u_in_area_r0"] = np.round(
    df_train["u_in_area"].to_numpy(dtype=np.float64), 0
).astype(np.float32)
df_test["u_in_area_r0"] = np.round(
    df_test["u_in_area"].to_numpy(dtype=np.float64), 0
).astype(np.float32)

calib_cols_fine = ["R", "C", "step", "u_out", "u_in_r1"]
calib_rc_step_uout_uinr1 = (
    df_train.groupby(calib_cols_fine, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_mean_rc_step_uout_uinr1"})
)

calib_cols_coarse = ["R", "C", "step", "u_out", "u_in_r0"]
calib_rc_step_uout_uinr0 = (
    df_train.groupby(calib_cols_coarse, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_mean_rc_step_uout_uinr0"})
)

calib_step_uout_uinr1 = (
    df_train.groupby(["step", "u_out", "u_in_r1"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_mean_step_uout_uinr1"})
)

calib_step_uout_uinr0 = (
    df_train.groupby(["step", "u_out", "u_in_r0"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_mean_step_uout_uinr0"})
)

calib_rc_step_uout = (
    df_train.groupby(["R", "C", "step", "u_out"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_mean_rc_step_uout"})
)

calib_step_uout = (
    df_train.groupby(["step", "u_out"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_mean_step_uout"})
)

calib_cols_lag = ["R", "C", "step", "u_out", "u_in_r1", "u_in_lag1_r1"]
calib_rc_step_uout_uin_lag = (
    df_train.groupby(calib_cols_lag, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_mean_rc_step_uout_uin_lag"})
)

calib_cols_cum = ["R", "C", "step", "u_out", "u_in_r1", "u_in_cum_r0"]
calib_rc_step_uout_uin_cum = (
    df_train.groupby(calib_cols_cum, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_mean_rc_step_uout_uin_cum"})
)

calib_cols_step_lag = ["step", "u_out", "u_in_r1", "u_in_lag1_r1"]
calib_step_uout_uin_lag = (
    df_train.groupby(calib_cols_step_lag, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_mean_step_uout_uin_lag"})
)

calib_cols_hist_full = [
    "R",
    "C",
    "step",
    "u_out",
    "u_out_lag1",
    "u_in_r1",
    "u_in_lag1_r1",
    "u_in_lag2_r1",
    "u_in_rm3_r1",
]
calib_rc_step_uout_hist = (
    df_train.groupby(calib_cols_hist_full, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_mean_rc_step_uout_hist"})
)

calib_cols_hist_step = [
    "step",
    "u_out",
    "u_out_lag1",
    "u_in_r1",
    "u_in_lag1_r1",
    "u_in_lag2_r1",
    "u_in_rm3_r1",
]
calib_step_uout_hist = (
    df_train.groupby(calib_cols_hist_step, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_mean_step_uout_hist"})
)

calib_cols_rc_frac_uout_uin = ["R", "C", "step_frac_r2", "u_out", "u_in_r1"]
calib_rc_frac_uout_uin = (
    df_train.groupby(calib_cols_rc_frac_uout_uin, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_mean_rc_frac_uout_uin"})
)

calib_cols_frac_uout_uin = ["step_frac_r2", "u_out", "u_in_r1"]
calib_frac_uout_uin = (
    df_train.groupby(calib_cols_frac_uout_uin, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_mean_frac_uout_uin"})
)

calib_step_exp = (
    df_train[df_train["u_out"] == 1]
    .groupby(["step"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_mean_step_exp"})
)

calib_frac_exp = (
    df_train[df_train["u_out"] == 1]
    .groupby(["step_frac_r2"], sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_mean_frac_exp"})
)

p_mean_uout = df_train.groupby(["u_out"], sort=False)["pressure"].mean().to_dict()
p_exp_global = float(p_mean_uout.get(1, float(df_train["pressure"].min())))

calib_cols_hist_diff_area = calib_cols_hist_full + ["u_in_diff1_r1", "u_in_area_r0"]
calib_rc_step_uout_hist_diff_area = (
    df_train.groupby(calib_cols_hist_diff_area, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_mean_rc_step_uout_hist_diff_area"})
)

calib_cols_step_hist_diff_area = calib_cols_hist_step + [
    "u_in_diff1_r1",
    "u_in_area_r0",
]
calib_step_uout_hist_diff_area = (
    df_train.groupby(calib_cols_step_hist_diff_area, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "p_mean_step_uout_hist_diff_area"})
)

df_pred = df_test.merge(
    calib_rc_step_uout_hist_diff_area,
    on=calib_cols_hist_diff_area,
    how="left",
    copy=False,
)
df_pred = df_pred.merge(
    calib_step_uout_hist_diff_area,
    on=calib_cols_step_hist_diff_area,
    how="left",
    copy=False,
)

df_pred = df_pred.merge(
    calib_rc_step_uout_hist, on=calib_cols_hist_full, how="left", copy=False
)
df_pred = df_pred.merge(
    calib_step_uout_hist, on=calib_cols_hist_step, how="left", copy=False
)

df_pred = df_pred.merge(
    calib_rc_step_uout_uin_lag, on=calib_cols_lag, how="left", copy=False
)
df_pred = df_pred.merge(
    calib_rc_step_uout_uin_cum, on=calib_cols_cum, how="left", copy=False
)
df_pred = df_pred.merge(
    calib_step_uout_uin_lag, on=calib_cols_step_lag, how="left", copy=False
)

df_pred = df_pred.merge(
    calib_rc_step_uout_uinr1, on=calib_cols_fine, how="left", copy=False
)
df_pred = df_pred.merge(
    calib_step_uout_uinr1, on=["step", "u_out", "u_in_r1"], how="left", copy=False
)
df_pred = df_pred.merge(
    calib_rc_step_uout_uinr0, on=calib_cols_coarse, how="left", copy=False
)
df_pred = df_pred.merge(
    calib_step_uout_uinr0, on=["step", "u_out", "u_in_r0"], how="left", copy=False
)
df_pred = df_pred.merge(
    calib_rc_step_uout, on=["R", "C", "step", "u_out"], how="left", copy=False
)
df_pred = df_pred.merge(calib_step_uout, on=["step", "u_out"], how="left", copy=False)

df_pred = df_pred.merge(
    calib_rc_frac_uout_uin, on=calib_cols_rc_frac_uout_uin, how="left", copy=False
)
df_pred = df_pred.merge(
    calib_frac_uout_uin, on=calib_cols_frac_uout_uin, how="left", copy=False
)

df_pred = df_pred.merge(calib_step_exp, on=["step"], how="left", copy=False)
df_pred = df_pred.merge(calib_frac_exp, on=["step_frac_r2"], how="left", copy=False)

p_min = float(df_train["pressure"].min())
p_max = float(df_train["pressure"].max())
u = df_pred["u_in"].to_numpy(dtype=np.float64)
u_scaled = u / 100.0  # u_in is in [0, 100]
pred_uin = p_min + u_scaled * (p_max - p_min)

pred = df_pred["p_mean_rc_step_uout_hist_diff_area"].to_numpy(dtype=np.float64)
missing = np.isnan(pred)

if missing.any():
    pred_da_step = df_pred["p_mean_step_uout_hist_diff_area"].to_numpy(dtype=np.float64)
    use_da_step = missing & (~np.isnan(pred_da_step))
    pred[use_da_step] = pred_da_step[use_da_step]
    missing = np.isnan(pred)

if missing.any():
    pred_hist1 = df_pred["p_mean_rc_step_uout_hist"].to_numpy(dtype=np.float64)
    use_hist1 = missing & (~np.isnan(pred_hist1))
    pred[use_hist1] = pred_hist1[use_hist1]
    missing = np.isnan(pred)

if missing.any():
    pred_hist2 = df_pred["p_mean_step_uout_hist"].to_numpy(dtype=np.float64)
    use_hist2 = missing & (~np.isnan(pred_hist2))
    pred[use_hist2] = pred_hist2[use_hist2]
    missing = np.isnan(pred)

if missing.any():
    pred0 = df_pred["p_mean_rc_step_uout_uin_lag"].to_numpy(dtype=np.float64)
    use0 = missing & (~np.isnan(pred0))
    pred[use0] = pred0[use0]
    missing = np.isnan(pred)

if missing.any():
    pred0b = df_pred["p_mean_rc_step_uout_uin_cum"].to_numpy(dtype=np.float64)
    use0b = missing & (~np.isnan(pred0b))
    pred[use0b] = pred0b[use0b]
    missing = np.isnan(pred)

if missing.any():
    pred0c = df_pred["p_mean_step_uout_uin_lag"].to_numpy(dtype=np.float64)
    use0c = missing & (~np.isnan(pred0c))
    pred[use0c] = pred0c[use0c]
    missing = np.isnan(pred)

if missing.any():
    pred1 = df_pred["p_mean_rc_step_uout_uinr1"].to_numpy(dtype=np.float64)
    use1 = missing & (~np.isnan(pred1))
    pred[use1] = pred1[use1]
    missing = np.isnan(pred)

if missing.any():
    pred2 = df_pred["p_mean_step_uout_uinr1"].to_numpy(dtype=np.float64)
    use2 = missing & (~np.isnan(pred2))
    pred[use2] = pred2[use2]
    missing = np.isnan(pred)

if missing.any():
    pred3 = df_pred["p_mean_rc_step_uout_uinr0"].to_numpy(dtype=np.float64)
    use3 = missing & (~np.isnan(pred3))
    pred[use3] = pred3[use3]
    missing = np.isnan(pred)

if missing.any():
    pred4 = df_pred["p_mean_step_uout_uinr0"].to_numpy(dtype=np.float64)
    use4 = missing & (~np.isnan(pred4))
    pred[use4] = pred4[use4]
    missing = np.isnan(pred)

if missing.any():
    pred5 = df_pred["p_mean_rc_step_uout"].to_numpy(dtype=np.float64)
    use5 = missing & (~np.isnan(pred5))
    pred[use5] = pred5[use5]
    missing = np.isnan(pred)

if missing.any():
    pred_mid1 = df_pred["p_mean_rc_frac_uout_uin"].to_numpy(dtype=np.float64)
    use_mid1 = missing & (~np.isnan(pred_mid1))
    pred[use_mid1] = pred_mid1[use_mid1]
    missing = np.isnan(pred)

if missing.any():
    pred_mid2 = df_pred["p_mean_frac_uout_uin"].to_numpy(dtype=np.float64)
    use_mid2 = missing & (~np.isnan(pred_mid2))
    pred[use_mid2] = pred_mid2[use_mid2]
    missing = np.isnan(pred)

if missing.any():
    pred6 = df_pred["p_mean_step_uout"].to_numpy(dtype=np.float64)
    use6 = missing & (~np.isnan(pred6))
    pred[use6] = pred6[use6]
    missing = np.isnan(pred)

if missing.any():
    pred[missing] = pred_uin[missing]

u_out_arr = df_test["u_out"].to_numpy()
step_arr = df_test["step"].to_numpy()
EXP_OVERRIDE_START_STEP = 4  # keep as-is to preserve your existing behavior

override_mask = (u_out_arr == 1) & (step_arr >= EXP_OVERRIDE_START_STEP)
p_exp_step = df_pred["p_mean_step_exp"].to_numpy(dtype=np.float64)
p_exp_frac = df_pred["p_mean_frac_exp"].to_numpy(dtype=np.float64)

pred[override_mask] = np.where(
    ~np.isnan(p_exp_step[override_mask]),
    p_exp_step[override_mask],
    np.where(
        ~np.isnan(p_exp_frac[override_mask]), p_exp_frac[override_mask], p_exp_global
    ),
)

pred = snap_to_known_pressures(pred)

pred_by_id = pd.DataFrame(
    {"id": df_test["id"].to_numpy(), "pressure": pred.astype(np.float64)}
)
out = sub[["id"]].merge(pred_by_id, on="id", how="left", validate="one_to_one")
out["pressure"] = out["pressure"].astype(np.float64)
out.to_csv("submission.csv", index=False)

print(out.head())
print("Wrote submission.csv with shape:", out.shape)
print(
    "Pred stats:",
    float(np.min(out["pressure"])),
    float(np.mean(out["pressure"])),
    float(np.max(out["pressure"])),
)
print("Expiratory global fallback pressure (before snapping):", p_exp_global)
print("Expiration override starts at step >=", EXP_OVERRIDE_START_STEP)
print(
    "u_in rounding fine decimals:",
    UIN_ROUND_DECIMALS_FINE,
    "| coarse decimals:",
    UIN_ROUND_DECIMALS_COARSE,
)
print(
    "Added smooth breath-progress fallbacks: step_frac_r2 for mid-specific lookup and expiratory override"
)
print(
    "Added extra dynamic history keys: u_in_diff1_r1 and u_in_area_r0 (used only for top-level lookups)"
)
