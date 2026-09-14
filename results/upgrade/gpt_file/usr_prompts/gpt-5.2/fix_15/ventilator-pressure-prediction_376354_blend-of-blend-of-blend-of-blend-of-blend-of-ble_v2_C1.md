# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.1463992680053852

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 9.92094) has done: 'I fix the immediate runtime errors by (1) making the code robust to the missing external “gb-…” datasets that caused `file_count=0` and missing CSVs, and (2) ensuring a valid `submission.csv` is always written with the required `id,pressure` columns. To preserve your core ensembling logic, the script still do the same randomized weighted-median blending *when those input prediction files exist*; otherwise it fall back to a simple, legitimate baseline model (groupwise median pressure by `(R,C,time_step,u_in,u_out)`) derived only from `train.csv`. I also correct the cell numbering and paths to match the provided environment. These changes are score-improving versus a zero/empty submission while keeping the original approach intact when the intended external files are present.'
- What this solution (achieved 7.73744) has done: 'Your current score (MAE 9.92094; lower is better) is far from the target (~0.1464), and the main reason is that the fallback baseline is effectively “static lookup medians” that almost never match in test due to continuous `u_in` and `time_step`, causing massive NaN fallback and poor predictions. To move toward the target while keeping your overall “blend if external files exist, else fallback” core logic, I minimally improve only the fallback by adding simple, legitimate time-series features (cumulative sum of `u_in` and time deltas per `breath_id`) and training a lightweight sklearn Ridge regression on inspiratory rows (`u_out==0`), then predicting for test and snapping to the nearest allowed pressure values (same post-processing as you already do). If your external ensemble files exist, your existing behavior is preserved; the Ridge fallback is only used when external predictions are missing. This change keeps runtime under the limit and produces a valid `submission.csv`.'
- What this solution (achieved 8.00013) has done: 'Your current score (7.737 MAE; lower is better) is still far from the target (0.146), so we should improve the fallback model while keeping your ensemble logic untouched. The biggest issue is that the Ridge fallback is trained only on inspiratory rows but then predicts arbitrary values for expiratory rows; since Kaggle ignores expiratory rows in scoring, we can safely set expiratory predictions to a neutral constant (e.g., 0) to reduce model noise without changing evaluation semantics. Additionally, the Ridge model benefits from standardizing features (still the same “Ridge regression” core logic) and using a slightly larger regularization to stabilize. Finally, we keep your “snap to nearest allowed pressure” post-processing for inspiratory predictions and ensure submission alignment by id.'
- What this solution (achieved 8.03922) has done: 'Your current score (MAE 8.00013; lower is better) is still far above the target (0.1464), so we should improve only the fallback path (used when the external ensemble files are missing) while leaving your external blending logic untouched. The biggest gain with minimal semantic change is to align the fallback training loss with the competition metric by training on inspiratory rows only (you already do) and *evaluating/learning* as MAE via `SGDRegressor(loss="epsilon_insensitive", epsilon=0.0)` (L1-style) instead of L2 Ridge, while keeping the same feature set and the same “snap to nearest allowed pressure” post-processing. We also keep expiratory predictions as 0 (not scored) to avoid introducing noise. Finally, we add a tiny, deterministic “per-(R,C) bias correction” computed on a small validation split to correct systematic under/over prediction without changing the overall approach.'
- What this solution (achieved 7.95069) has done: 'Your current score (8.03922 MAE; lower is better) is far above the target (0.1464), so we should improve only the fallback path (used when the external ensemble folders/files aren’t available) while keeping your ensemble/blend logic untouched. The main minimal fix is to align training with the competition metric by using a true L1 regression objective (`SGDRegressor(loss="squared_epsilon_insensitive", epsilon=0.0)`, which corresponds to MAE) instead of the current L2-like objective, and to make the SGD optimization stable via `average=True` (standard, deterministic variance reduction) without changing the model class or feature set. We also compute the (R,C) bias on a slightly larger but still small validation breath split to reduce systematic offsets (keeps the same bias-correction idea, just less noisy). Everything else—including feature engineering, inspiratory-only fitting, snapping to nearest allowed pressures, and writing `submission.csv`—stays the same.'
- What this solution (achieved 7.9185) has done: 'Your gap to target is still very large (7.95069 vs 0.1464 MAE; lower is better), so we should improve only the fallback path (used when the external ensemble files aren’t available) while leaving your external blending logic untouched. The biggest minimal win for this competition is to enforce the known physics constraint: when `u_out==1` (exhalation), airway pressure is (in the simulated data) essentially fixed at the PEEP level for that breath (the first pressure at `time_step==0`), so we can set test expiratory predictions to that per-breath baseline instead of 0. Additionally, we can remove avoidable underfitting by using the true MAE objective for SGD (`loss="epsilon_insensitive", epsilon=0.0`) while keeping the same SGDRegressor + StandardScaler pipeline and the same features and snapping-to-allowed-pressures post-processing. These are small, metric-aligned changes that should move the score substantially toward the target without changing your ensemble logic or overall approach.'
- What this solution (achieved 7.9546) has done: 'Your current MAE (7.9185; lower is better) is far above the target (0.1464), so we should improve only the fallback path (used when external ensemble CSVs aren’t available) while keeping your external blending logic untouched. The main issue in the fallback is that expiratory (`u_out==1`) pressures in test are being set from a coarse (R,C) median PEEP, which is often wrong per-breath; we can instead estimate a per-breath PEEP directly from each test breath using the first timestep (`time_step==0`) features, predicted via the same trained model, then snap to valid pressures. Additionally, we align train/test feature construction by adding a `breath_time_idx` (0..79) to remove dependence on float `time_step` quirks, and we keep all other training settings, snapping, and submission writing unchanged. These are minimal, metric-aligned changes that should move the score materially toward the target without changing the overall approach.'
- What this solution (achieved 7.80933) has done: 'Your current score is far worse than the target (MAE 7.95 vs 0.146; lower is better), so the smallest meaningful improvement is to fix the fallback model’s biggest metric mismatch: Kaggle scores only inspiratory timesteps (`u_out==0`), but your model is trained without telling it which rows are inspiratory except via `u_out`, and you also don’t exploit the very strong “pressure is quantized” property at prediction time beyond simple nearest snapping. I keep your ensemble/blending path unchanged, and only strengthen the fallback by (1) adding two minimal, high-signal, competition-standard engineered features (`u_in`/`u_out` rolling-like dynamics via lags of cumulative sum and a simple “area” term `u_in_cumsum*dt`), and (2) training a fast, deterministic multi-output-free model but with a slightly better-conditioned solver for MAE-like behavior: still `SGDRegressor(loss="epsilon_insensitive")`, same training loop semantics, but with a small feature tweak and a safer per-(R,C,time_idx) bias table (more granular than just (R,C)) computed on validation breaths. These changes stay within sklearn/pandas/numpy, keep runtime under the limit, and should move the MAE materially toward your target while preserving your core logic and submission format.'
- What this solution (achieved 7.816) has done: 'I keep your external-ensemble path unchanged and only improve the fallback (used when those external CSVs aren’t present), because your current MAE (7.81) is far from the target (0.146) and the gap suggests the fallback model is underpowered. The minimal, metric-aligned improvement is to add two high-signal, competition-standard dynamics features (a per-breath “u_in integral” and per-step delta-u_in) without changing the model class (still `SGDRegressor` in a `StandardScaler` pipeline) or the snapping-to-allowed-pressures post-processing. I also ensure the validation bias table keys match train/test consistently and apply it via a fast merge (less error-prone than a Python dict loop) while preserving the same bias-correction idea. This should move the score down materially toward the target while keeping runtime reasonable and still writing a valid `submission.csv`.'
- What this solution (achieved 7.95339) has done: 'Your current score (7.816 MAE; lower is better) is still far above the target (0.1464), so we should improve only the fallback path (used when the external ensemble files aren’t available) while leaving your external blending/averaging logic unchanged. The smallest high-impact fix is to add two very standard ventilator features—per-breath cumulative exhalation time (`u_out` time accumulator) and a short `u_in` lag (t-2)—without changing the model class (still `SGDRegressor` inside a `StandardScaler` pipeline) or the snapping-to-allowed-pressures post-processing. We also make the per-breath PEEP estimate for expiratory rows use the model’s prediction at the first timestep *after* snapping (as you do) but computed more directly and safely aligned by `breath_id`. These changes preserve your overall training approach and keep runtime within the limit while moving the fallback predictions in a more physically-consistent direction.'
- What this solution (achieved 7.96119) has done: 'Your score is far above the target, so we should improve only the fallback path (used when the external ensemble files aren’t present) while keeping your external blending logic intact. The biggest minimal gain for this competition is to align training/inference with the scoring rule by training and predicting only on inspiratory rows (`u_out==0`) and not letting expiratory handling inject noise. We also add a very small, high-signal change to the existing feature set: encode `R` and `C` as one-hot (still linear SGDRegressor, same training loop semantics) because these discrete lung attributes interact nonlinearly with pressure and a linear numeric encoding is suboptimal. Finally, we make the bias-correction split deterministic and non-leaky by splitting by `breath_id` using a fixed hash-like rule (no shuffling variability), keeping the same bias-table idea but making it more stable.'

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
DATA_DIR = "/kaggle/input/ventilator-pressure-prediction"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

df_train = pd.read_csv(TRAIN_PATH, usecols=["pressure"])

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)
del df_train
gc.collect()


def find_nearest_vec(pred):
    pred = np.asarray(pred)
    ins = np.searchsorted(sorted_pressures, pred, side="left")
    ins = ins.astype(np.int64, copy=False)

    out = np.empty_like(pred, dtype=sorted_pressures.dtype)

    mask_hi = ins >= total_pressures_len
    mask_lo = ins <= 0
    mask_mid = ~(mask_hi | mask_lo)

    if mask_hi.any():
        out[mask_hi] = sorted_pressures[-1]
    if mask_lo.any():
        out[mask_lo] = sorted_pressures[0]
    if mask_mid.any():
        i = ins[mask_mid]
        lower = sorted_pressures[i - 1]
        upper = sorted_pressures[i]
        p = pred[mask_mid]
        choose_lower = np.abs(lower - p) < np.abs(upper - p)
        out_mid = np.where(choose_lower, lower, upper)
        out[mask_mid] = out_mid
    return out


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
    """
    Weighted combine of 2 submissions based on a score parsed from filename.
    If filenames don't match expected format, fall back to equal weighting.
    """
    l = []
    preds = []
    for p in input_list:
        try:
            public_lb_score = int(p.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1
        l.append(public_lb_score)
        preds.append(
            pd.read_csv(p, usecols=["pressure"])["pressure"].to_numpy().ravel()
        )

    if len(preds) == 1:
        return preds[0]

    l_sum = sum(l) if sum(l) != 0 else 1
    weight1 = (l[1] / l_sum) + 0.1
    weight2 = 1 - weight1
    return preds[0] * weight1 + preds[1] * weight2


def g(dp, out_name=None):
    """
    Optimization: keep identical randomized weighting + median aggregation semantics,
    but reduce overhead by avoiding repeated list/len calls and minimizing GC.
    """
    l = [i for i in glob.iglob(f"{dp}/*") if os.path.isfile(i)]
    file_count = len(l)
    if file_count == 0:
        return None

    loop_time = 500 // file_count
    loop_time = max(loop_time, 1)

    splits = max(file_count // 2, 1)
    l.sort()

    flist = []
    n_l = len(l)
    step = round(n_l / splits) if splits > 0 else n_l
    for i in range(splits):
        start = i * step
        end = n_l if i == splits - 1 else (i + 1) * step
        chunk = l[start:end]
        if chunk:
            flist.append(chunk)

    for i in range(len(flist)):
        flist[i] = wc(flist[i])

    n = flist[0].shape[0]
    pred_mat = np.empty((loop_time, n), dtype=np.float64)

    k = len(flist)
    for it in range(loop_time):
        set_seed(it)
        weight = [rd() for _ in range(k)]
        weight_sum = sum(weight) if sum(weight) != 0 else 1.0
        inv = 1.0 / weight_sum
        for j in range(k):
            weight[j] *= inv
        weight.sort(reverse=True)

        temp = np.zeros(n, dtype=np.float64)
        for j in range(k):
            temp += flist[j] * weight[j]
        pred_mat[it] = temp

    output = pd.read_csv(SAMPLE_SUB_PATH)
    med = np.median(pred_mat, axis=0)
    output["pressure"] = find_nearest_vec(med)
    if out_name is None:
        out_name = f"rwb {loop_time} loops.csv"
    output.to_csv(out_name, index=False)

    del pred_mat, med, output
    gc.collect()
    return out_name




## === cell 2
ensemble_dir = "/kaggle/input/gb-rwbt-files"
ensemble_csv = g(ensemble_dir)



## === cell 3
test_df = pd.read_csv(TEST_PATH)
sub = pd.read_csv(SAMPLE_SUB_PATH)


def build_baseline_predictions(train_df, test_df):
    from sklearn.linear_model import SGDRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.base import clone

    def add_features(df):
        df = df.copy()
        df.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")

        bid = df["breath_id"].to_numpy()
        n = bid.size

        change = np.empty(n, dtype=bool)
        change[0] = True
        change[1:] = bid[1:] != bid[:-1]
        starts = np.flatnonzero(change)
        ends = np.r_[starts[1:], n]
        lens = ends - starts

        idx = np.arange(n, dtype=np.int16)
        start_per_row = np.repeat(starts, lens)
        breath_time_idx = (idx - start_per_row).astype(np.int16, copy=False)
        df["breath_time_idx"] = breath_time_idx

        ts = df["time_step"].to_numpy(dtype=np.float32, copy=False)
        dt = np.empty(n, dtype=np.float32)
        dt[starts] = 0.0
        dt[~change] = (ts[1:] - ts[:-1]).astype(np.float32, copy=False)
        df["dt"] = dt

        u_in = df["u_in"].to_numpy(dtype=np.float32, copy=False)
        u_out = df["u_out"].to_numpy(dtype=np.int8, copy=False)

        u_in_lag1 = np.empty(n, dtype=np.float32)
        u_in_lag1[starts] = 0.0
        u_in_lag1[~change] = u_in[:-1]
        df["u_in_lag1"] = u_in_lag1

        u_in_lag2 = np.empty(n, dtype=np.float32)
        u_in_lag2[starts] = 0.0
        sec = starts + 1
        sec = sec[sec < n]
        u_in_lag2[sec] = 0.0
        mask_other = np.ones(n, dtype=bool)
        mask_other[starts] = False
        mask_other[sec] = False
        u_in_lag2[mask_other] = u_in[:-2]
        df["u_in_lag2"] = u_in_lag2

        df["u_in_diff"] = (u_in - u_in_lag1).astype(np.float32, copy=False)

        u_out_lag1 = np.empty(n, dtype=np.int8)
        u_out_lag1[starts] = 0
        u_out_lag1[~change] = u_out[:-1]
        df["u_out_lag1"] = u_out_lag1

        u_in_cumsum = np.empty(n, dtype=np.float32)
        u_in_integral_cumsum = np.empty(n, dtype=np.float32)
        u_out_dt_cumsum = np.empty(n, dtype=np.float32)

        u_in_dt = (u_in * dt).astype(np.float32, copy=False)
        u_out_dt = (u_out.astype(np.float32) * dt).astype(np.float32, copy=False)

        for s, e in zip(starts, ends):
            u_in_cumsum[s:e] = np.cumsum(u_in[s:e], dtype=np.float32)
            u_in_integral_cumsum[s:e] = np.cumsum(u_in_dt[s:e], dtype=np.float32)
            u_out_dt_cumsum[s:e] = np.cumsum(u_out_dt[s:e], dtype=np.float32)

        df["u_in_cumsum"] = u_in_cumsum

        u_in_cumsum_lag1 = np.empty(n, dtype=np.float32)
        u_in_cumsum_lag1[starts] = 0.0
        u_in_cumsum_lag1[~change] = u_in_cumsum[:-1]
        df["u_in_cumsum_lag1"] = u_in_cumsum_lag1
        df["u_in_cumsum_diff1"] = (u_in_cumsum - u_in_cumsum_lag1).astype(
            np.float32, copy=False
        )

        df["u_in_integral"] = u_in_integral_cumsum
        df["u_out_dt_cumsum"] = u_out_dt_cumsum

        R = df["R"].to_numpy(dtype=np.int16, copy=False)
        C = df["C"].to_numpy(dtype=np.int16, copy=False)
        df["RC"] = (R * C).astype(np.float32, copy=False)
        df["u_in_x_R"] = (u_in * R).astype(np.float32, copy=False)
        df["u_in_x_C"] = (u_in * C).astype(np.float32, copy=False)

        df["R_5"] = (R == 5).astype(np.int8, copy=False)
        df["R_20"] = (R == 20).astype(np.int8, copy=False)
        df["R_50"] = (R == 50).astype(np.int8, copy=False)
        df["C_10"] = (C == 10).astype(np.int8, copy=False)
        df["C_20"] = (C == 20).astype(np.int8, copy=False)
        df["C_50"] = (C == 50).astype(np.int8, copy=False)

        return df

    def to_breath_matrix_fixed(df_feat, feature_cols):
        breath_ids = df_feat["breath_id"].to_numpy()
        n_breaths = breath_ids.size // 80
        X2 = df_feat[feature_cols].to_numpy(dtype=np.float32, copy=False)
        X_breath = X2.reshape(n_breaths, 80 * len(feature_cols))
        bid = breath_ids[::80].copy()
        return X_breath, bid

    def to_breath_targets_fixed(df_feat, target_col):
        breath_ids = df_feat["breath_id"].to_numpy()
        n_breaths = breath_ids.size // 80
        y = df_feat[target_col].to_numpy(dtype=np.float32, copy=False)
        y_breath = y.reshape(n_breaths, 80)
        bid = breath_ids[::80].copy()
        return y_breath, bid

    tr = add_features(train_df)
    te = add_features(test_df)

    feature_cols = [
        "RC",
        "time_step",
        "breath_time_idx",
        "dt",
        "u_in",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_diff",
        "u_in_cumsum",
        "u_in_cumsum_diff1",
        "u_in_integral",
        "u_out",
        "u_out_lag1",
        "u_out_dt_cumsum",
        "u_in_x_R",
        "u_in_x_C",
        "R_5",
        "R_20",
        "R_50",
        "C_10",
        "C_20",
        "C_50",
    ]

    y_breath, tr_breath_ids = to_breath_targets_fixed(tr, "pressure")
    X_breath, _ = to_breath_matrix_fixed(tr, feature_cols)

    val_mask_b = (tr_breath_ids % 25) == 0
    train_mask_b = ~val_mask_b

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "sgd",
                SGDRegressor(
                    loss="epsilon_insensitive",
                    epsilon=0.0,
                    alpha=1.5e-4,
                    fit_intercept=True,
                    max_iter=4000,
                    tol=1e-5,
                    random_state=2021,
                    learning_rate="invscaling",
                    eta0=0.01,
                    power_t=0.25,
                    average=True,
                ),
            ),
        ]
    )

    models = []
    X_tr = X_breath[train_mask_b]
    for t in range(80):
        m = clone(model)
        m.fit(X_tr, y_breath[train_mask_b, t])
        models.append(m)

    X_te_breath, te_breath_ids = to_breath_matrix_fixed(te, feature_cols)

    pred_te_breath = np.zeros((X_te_breath.shape[0], 80), dtype=np.float64)
    for t in range(80):
        pred_te_breath[:, t] = (
            models[t].predict(X_te_breath).astype(np.float64, copy=False)
        )

    if val_mask_b.any():
        tr_sorted = tr  # already sorted by breath_id,time_step and has breath_time_idx
        val_pos = np.nonzero(val_mask_b)[0]
        row_idx = (
            val_pos[:, None] * 80 + np.arange(80, dtype=np.int32)[None, :]
        ).reshape(-1)

        R_meta = tr_sorted["R"].to_numpy(dtype=np.int16, copy=False)[row_idx]
        C_meta = tr_sorted["C"].to_numpy(dtype=np.int16, copy=False)[row_idx]
        t_meta = tr_sorted["breath_time_idx"].to_numpy(dtype=np.int16, copy=False)[
            row_idx
        ]

        val_pred = np.zeros((len(val_pos), 80), dtype=np.float64)
        X_val_b = X_breath[val_mask_b]
        for t in range(80):
            val_pred[:, t] = models[t].predict(X_val_b).astype(np.float64, copy=False)

        y_val = y_breath[val_mask_b].astype(np.float64, copy=False)
        err = (y_val - val_pred).reshape(-1)

        val_meta = pd.DataFrame(
            {"R": R_meta, "C": C_meta, "breath_time_idx": t_meta, "err": err}
        )
        bias_tbl = (
            val_meta.groupby(["R", "C", "breath_time_idx"], as_index=False, sort=False)[
                "err"
            ]
            .median()
            .rename(columns={"err": "bias"})
        )

        te_sorted = te  # already sorted
        te_meta = te_sorted[["R", "C", "breath_time_idx"]].copy()
        merged = te_meta.merge(
            bias_tbl, on=["R", "C", "breath_time_idx"], how="left", sort=False
        )
        bias = (
            merged["bias"]
            .fillna(0.0)
            .to_numpy(dtype=np.float64, copy=False)
            .reshape(-1, 80)
        )
        pred_te_breath = pred_te_breath + bias

    te_sorted_full = te  # already sorted
    u_out_mat = (
        te_sorted_full["u_out"]
        .to_numpy(dtype=np.int8, copy=False)
        .reshape(len(te_breath_ids), 80)
    )

    peep = find_nearest_vec(pred_te_breath[:, 0]).astype(np.float64, copy=False)

    pred_flat = pred_te_breath.reshape(-1)
    insp_mask_flat = u_out_mat.reshape(-1) == 0
    pred_flat_insp = pred_flat[insp_mask_flat]
    pred_flat[insp_mask_flat] = find_nearest_vec(pred_flat_insp)

    exp_mask = ~insp_mask_flat
    if exp_mask.any():
        peep_rep = np.repeat(peep, 80)
        pred_flat[exp_mask] = peep_rep[exp_mask]

    pred_all = np.empty(len(test_df), dtype=np.float64)
    pred_all[te_sorted_full.index.to_numpy()] = pred_flat.astype(np.float64, copy=False)
    return pred_all


external_df1_path = "/kaggle/input/gb-vpp-whoppity-dub-dub/median_submission.csv"
df_1 = None
if os.path.exists(external_df1_path):
    df_1 = pd.read_csv(external_df1_path)

df_2 = None
if ensemble_csv is not None and os.path.exists(ensemble_csv):
    df_2 = pd.read_csv(ensemble_csv)

if df_1 is not None and df_2 is not None:
    df_final = df_1.copy()
    p1 = df_1["pressure"].to_numpy(dtype=np.float64, copy=False)
    p2 = df_2["pressure"].to_numpy(dtype=np.float64, copy=False)
    df_final["pressure"] = find_nearest_vec((p1 + p2) * 0.5)
    df_final[["id", "pressure"]].to_csv("submission.csv", index=False)
elif df_2 is not None:
    out = df_2.copy()
    out["pressure"] = find_nearest_vec(
        out["pressure"].to_numpy(dtype=np.float64, copy=False)
    )
    out[["id", "pressure"]].to_csv("submission.csv", index=False)
else:
    sub = sub.sort_values("id").reset_index(drop=True)
    test_df = test_df.sort_values("id").reset_index(drop=True)

    train_full = pd.read_csv(TRAIN_PATH)
    sub["pressure"] = build_baseline_predictions(train_full, test_df)
    sub[["id", "pressure"]].to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", pd.read_csv("submission.csv").shape)
print(pd.read_csv("submission.csv").head())

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3713204791.py in <cell line: 0>()
    296     # I/O unchanged: still reads full train when needed.
    297     train_full = pd.read_csv(TRAIN_PATH)
--> 298     sub["pressure"] = build_baseline_predictions(train_full, test_df)
    299     sub[["id", "pressure"]].to_csv("submission.csv", index=False)
    300 

/tmp/ipykernel_11/3713204791.py in build_baseline_predictions(train_df, test_df)
    125         return y_breath, bid
    126 
--> 127     tr = add_features(train_df)
    128     te = add_features(test_df)
    129 

/tmp/ipykernel_11/3713204791.py in add_features(df)
     37         dt = np.empty(n, dtype=np.float32)
     38         dt[starts] = 0.0
---> 39         dt[~change] = (ts[1:] - ts[:-1]).astype(np.float32, copy=False)
     40         df["dt"] = dt
     41 

ValueError: NumPy boolean array indexing assignment cannot assign 5432399 input values to the 5364495 output values where the mask is true
