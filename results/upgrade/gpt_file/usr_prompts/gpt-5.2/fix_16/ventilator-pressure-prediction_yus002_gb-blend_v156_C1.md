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

0.1397380225033982

# 6. Current score

8.32531

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.86378) has done: 'I fix the crash by making the blending code robust when the expected external dataset (`../input/gb-data-blending-recover`) is missing or contains no usable prediction files, which currently leads to an invalid `(1,)` median vector. To keep the core “blend many CSV predictions then median + nearest pressure grid” logic intact, I (1) validate and load only correctly-shaped prediction files, (2) fall back to a simple baseline submission (median training pressure snapped to the nearest valid pressure) if no files are found, and (3) ensure the output filename ends with `.csv` and matches the required `id,pressure` format. These changes are score-neutral relative to the intended blending (they only prevent failure) and always produce a valid submission file end-to-end in this environment.'
- What this solution (achieved 9.91418) has done: 'Your current score is far worse than the target (lower-is-better), and the main reason is that in this environment there are no external prediction files to blend, so the code falls back to a constant “median pressure” baseline which scores very poorly. To move the score toward the target with minimal change and without altering the overall “produce a submission from available data” semantics, I keep your blending logic intact but add a deterministic, lightweight fallback model trained from `train.csv` when no blend files exist. Specifically, the fallback predicts per-row inspiratory pressure using a groupwise median lookup keyed by `(R, C, time_step, u_out, u_in_rounded)` and snaps to the nearest valid pressure grid, which typically yields a much better MAE than a constant baseline. The script still writes a valid `submission.csv` with `id,pressure` and continue to use blending if the external dataset is present.'
- What this solution (achieved 10.23528) has done: 'Your current score (9.91418, lower-is-better) is far from the target (0.1397), and the main reason is that the notebook is almost certainly falling back to the weak constant/lookup baseline because there are no external blend files available. To move the score sharply toward the target while keeping your overall “no NN, no heavy training loop; produce a submission from available data” logic intact, I upgrade only the fallback path to a stronger deterministic per-time-step lookup using the full (breath_id, time_step) context via lag features (u_in/u_out/time_step shifted within each breath) and groupwise median mappings trained on `train.csv`. The blending path is left unchanged and still be used if valid external prediction CSVs exist. The final output remains snapped to the known pressure grid and writes a valid `submission.csv`.'
- What this solution (achieved 7.99659) has done: 'Your score is still extremely far from the target (lower-is-better), which strongly suggests you are always hitting the “no external blend files exist” fallback path; so the only meaningful way to move toward the target is to strengthen that fallback while keeping the overall “deterministic lookup → snap to valid pressure grid → write submission.csv” core logic. The biggest issue in the current fallback is that it uses `time_step` as a raw float key, which causes massive key-mismatch between train/test and collapses predictions to global medians (very bad MAE). I make the smallest effective fix: discretize `time_step` to the dataset’s natural grid index within each breath (0..79) and use that in the lookup keys (keeping your lag-feature lookup/backoff structure unchanged otherwise). This should substantially reduce fallback MAE (often into the sub-1 range) without introducing any new modeling approach, training loop, or changing the blending path.'
- What this solution (achieved 8.25533) has done: 'Your current score is far worse than the target (lower-is-better), and given the environment likely has no external blend files, nearly all score comes from the fallback lookup. I keep your exact “deterministic lookup with backoffs → snap to nearest pressure grid → write submission.csv” core logic, but make two minimal fixes that materially reduce MAE: (1) ensure the lookup is built only on the inspiratory phase (`u_out==0`), matching the evaluation (expiratory is not scored), and (2) ensure test predictions for expiratory rows (`u_out==1`) are set to a safe value (0) so they don’t affect score and don’t introduce noise. This preserves your blending path unchanged and only improves the fallback path where your run is currently spending all its quality budget.'
- What this solution (achieved 8.39477) has done: 'Your current score (8.25533, lower-is-better) is still far from the target (0.1397), and since the external blend dataset path is likely missing here, almost all performance depends on the fallback lookup model. The smallest high-impact fix is to make the fallback mapping substantially less “blurry” by (1) using a more precise rounding for `u_in`-related keys (0.1 instead of 1.0) and (2) adding a lightweight extra backoff keyed by `(R,C,t_idx,u_out,u_in_r,u_in_diff1_r)` which often matches dynamics better than `u_in` alone. This keeps your exact core approach (deterministic group-median lookup with backoffs → snap to known pressure grid → write submission.csv) and only strengthens the fallback path used when no blend files exist. The blending path and output format remain unchanged.'
- What this solution (achieved 8.24837) has done: 'Your current score is far above the target (lower-is-better), and in this environment it’s very likely you’re always hitting the fallback path (no external blend files), so the only meaningful way to move toward the target is to make the fallback lookup much less prone to “no matches → global median.” The smallest high-impact fix is to add a deterministic “pressure mapping by (R,C,t_idx,u_out,u_in)” built from exact `u_in` values (no rounding) since `u_in` takes a small discrete set and matches perfectly between train/test, then keep your existing richer lag-based keys as secondary/backoff. This preserves the core logic (deterministic group-median lookup with backoffs → snap to known pressure grid → write submission.csv) while materially increasing match rate and lowering MAE. The blending path and output format remain unchanged.'
- What this solution (achieved 8.24837) has done: 'Your current score (8.248, lower-is-better) is still far from the target (0.1397), and the code is almost certainly using the fallback lookup path because the external blend dataset isn’t present. The biggest remaining issue in the fallback is still low match-rate (many test rows fall through to coarse backoffs/global median), so I make a minimal, metric-aligned strengthening: add a more “exact” key that includes multiple exact (non-rounded) control/lag values on the natural `t_idx` grid, and insert it early in the backoff chain to reduce fallbacks. I also keep inspiratory-only training for the lookup (matching the metric) and continue setting `u_out==1` predictions to 0 (not scored). Core logic remains the same: deterministic group-median lookup with backoffs → snap to known pressure grid → write `submission.csv`.'
- What this solution (achieved 8.24972) has done: 'Your current MAE (8.248, lower-is-better) is still extremely far from the target (0.1397), and the main bottleneck is the fallback path: many test rows likely miss the lookup keys and fall back to coarse medians. I keep your exact “deterministic group-median lookup with backoffs → snap to known pressure grid → write submission.csv” core logic, but make one minimal, high-impact change: add a pure “memorization” mapping from the full control/state key `(R, C, t_idx, u_out, u_in, u_in_lag1, u_in_lag2, u_in_lag3, u_in_lag5)` using exact float32 values (no rounding) and use it as the first backoff. This increases exact-match rate without changing the modeling approach, and should move the score substantially toward the target while staying deterministic and within Kaggle constraints. The blending path remains unchanged and is still used if valid external prediction CSVs exist.'
- What this solution (achieved 8.11211) has done: 'Your current score is far worse than the target (lower-is-better), and the run is almost certainly always using the fallback lookup path (no external blend files). The smallest change that should materially improve MAE without changing your overall “deterministic lookup with backoffs → snap to pressure grid → write submission.csv” core logic is to use a better key for the fallback: replace float lag values (which don’t match well due to tiny representation differences) with integer-quantized keys derived from the known pressure grid step. Concretely, we convert `u_in` and its lag/diff/cumsum features into int16 “bin indices” on a fixed 0.1 resolution (and cap ranges), then group/merge on these integer keys so train/test match rates rise sharply. Everything else (inspiratory-only lookup, expiratory set to 0, nearest-pressure snapping, blending path) stays the same and the script still produces `submission.csv`.'
- What this solution (achieved 8.20882) has done: 'Your current MAE (8.11, lower-is-better) is still extremely far from the target (0.1397), and in this environment you’re almost certainly always using the fallback path (no external blend files). The biggest minimal, metric-aligned gain available without changing the overall “deterministic lookup + backoffs + snap-to-pressure-grid” logic is to stop the fallback from “blurring” by taking medians and instead use a deterministic nearest-neighbor join: for each test row, find the closest `u_in` (in quantized space) within the same `(R,C,t_idx, u_out, lag context)` group and use that row’s pressure. This preserves the same feature extraction and backoff chain, but replaces the aggregation with a per-row lookup that typically matches the simulator’s near-deterministic mapping much better. I keep inspiratory-only fitting (matches the metric), keep setting `u_out==1` predictions to 0 (not scored), and still snap to the known pressure grid and write a valid `submission.csv`.'
- What this solution (achieved 8.16537) has done: 'The timeout is dominated by the fallback path, which loads the full 5.4M-row train set and then performs multiple full-data groupby+merge operations plus per-group nearest-neighbor scans; this is far more work than needed for a deterministic lookup submission. I keep the exact same fallback logic/semantics, but make it run in time by (1) converting repeated full-frame merges into fast index-based joins, (2) eliminating the extremely expensive per-group Python loop in `_nn_lookup_1d_by_group` by using pre-sorted NumPy arrays and vectorized searchsorted on grouped slices, and (3) reducing repeated sorting/groupby passes by computing the per-breath index and lag features once with stable ordering. All changes are equivalence-preserving (same keys, same tie-break rules, same snapping), and do not change model logic (there is no training here) or evaluation semantics.'
- What this solution (achieved 8.32531) has done: 'Your current MAE (8.165, lower-is-better) is still far from the target (0.1397), and with no external blend files available the submission quality is dominated by the fallback lookup. The main minimal change to move the score down is to make the fallback key-match rate much higher by avoiding float-quantization mismatches: (1) build integer “step index” features (`u_in` and lag/diff/cumsum) on a stable 0.1 grid with safer dtypes, and (2) add a first, very specific nearest-neighbor lookup group that includes the current and recent control context so fewer rows fall back to coarse medians. This preserves your core logic (deterministic lookup with backoffs → snap to known pressure grid → write submission.csv) and keeps the blending path unchanged. It should improve MAE materially while staying within runtime constraints.'

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

train_usecols = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
train_dtypes = {
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
df_train = pd.read_csv(TRAIN_PATH, usecols=train_usecols, dtype=train_dtypes)

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


def snap_to_nearest_pressure_vec(pred_arr: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred_arr, dtype=np.float64)
    idx = np.searchsorted(sorted_pressures, pred, side="left")
    idx1 = np.clip(idx, 0, total_pressures_len - 1)
    idx0 = np.clip(idx - 1, 0, total_pressures_len - 1)
    p0 = sorted_pressures[idx0]
    p1 = sorted_pressures[idx1]
    choose1 = np.abs(pred - p1) < np.abs(pred - p0)  # tie -> p0 (same as find_nearest)
    return np.where(choose1, p1, p0)


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    """
    Original intent: read one or two submissions and do a simple weighted combo.
    Bugfix: make parsing robust; if the filename does not contain a score token,
    just treat both weights equally (this preserves the blending idea without crashing).
    """
    l = []
    arrs = []
    for i in range(len(input_list)):
        fp = input_list[i]
        try:
            public_lb_score = int(fp.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1
        l.append(public_lb_score)

        dfp = pd.read_csv(fp)
        if "pressure" not in dfp.columns:
            raise ValueError(f"File {fp} does not have 'pressure' column.")
        arrs.append(dfp["pressure"].to_numpy().ravel())

    l_sum = sum(l) if sum(l) != 0 else len(l)

    if len(arrs) == 1:
        return arrs[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        return arrs[0] * weight1 + arrs[1] * weight2


def _safe_list_pred_files(dp):
    """
    Bugfix: in this environment the external dataset path may not exist.
    Only return readable .csv files; ignore directories/non-csv.
    """
    if dp is None:
        return []
    if not os.path.exists(dp):
        return []
    files = []
    for p in glob.iglob(f"{dp}/*"):
        if os.path.isfile(p) and p.lower().endswith(".csv"):
            files.append(p)
    files.sort()
    return files


def _load_pressure_vector(fp, expected_len):
    """
    Load a submission-like CSV and return pressure vector if shape matches.
    Skip invalid files rather than breaking the run.
    """
    try:
        dfp = pd.read_csv(fp, usecols=["pressure"])
        v = dfp["pressure"].to_numpy().ravel()
        if len(v) != expected_len:
            return None
        return v
    except Exception:
        return None


def _add_time_index_inplace_sorted(df):
    df.sort_values(["breath_id", "time_step"], kind="mergesort", inplace=True)
    df["t_idx"] = df.groupby("breath_id", sort=False).cumcount().astype(np.int16)
    return df


def _quantize_to_int(series, scale=10.0, clip_min=-500000, clip_max=500000):
    """
    Score-critical: increase key match rate by quantizing to integer bins deterministically.
    Use int32 to avoid overflow for cumsum-based features (int16 overflow caused silent key noise).
    scale=10 => 0.1 resolution.
    """
    x = np.rint(series.to_numpy(dtype=np.float64, copy=False) * scale)
    x = np.clip(x, clip_min, clip_max)
    return x.astype(np.int32)


def _add_lag_features_fast_inplace_sorted(df, lags=(1, 2, 3, 5)):
    g = df.groupby("breath_id", sort=False)
    uin = g["u_in"]
    uout = g["u_out"]
    ts = g["time_step"]

    for k in lags:
        df[f"u_in_lag{k}"] = uin.shift(k)
        df[f"u_out_lag{k}"] = uout.shift(k)
        df[f"dt_lag{k}"] = ts.diff(k)

    df["u_in_diff1"] = uin.diff(1)
    df["u_in_diff2"] = uin.diff(2)
    df["u_in_cumsum"] = uin.cumsum()

    df.fillna(0.0, inplace=True)
    return df


def _nn_lookup_1d_by_group_vectorized(
    train_df, test_df, group_cols, xcol, ycol, outcol
):
    out = test_df.copy()
    out[outcol] = np.nan
    if len(train_df) == 0 or len(test_df) == 0:
        return out

    tr_keys = pd.MultiIndex.from_frame(train_df[group_cols])
    te_keys = pd.MultiIndex.from_frame(test_df[group_cols])
    all_keys = tr_keys.append(te_keys)
    key_codes, _ = pd.factorize(all_keys, sort=False)
    tr_code = key_codes[: len(train_df)].astype(np.int32, copy=False)
    te_code = key_codes[len(train_df) :].astype(np.int32, copy=False)

    x_tr = train_df[xcol].to_numpy()
    y_tr = train_df[ycol].to_numpy()
    x_te = test_df[xcol].to_numpy()

    order = np.lexsort((x_tr, tr_code))
    tr_code_s = tr_code[order]
    x_tr_s = x_tr[order]
    y_tr_s = y_tr[order]

    uniq_tr, start_idx, counts = np.unique(
        tr_code_s, return_index=True, return_counts=True
    )
    end_idx = start_idx + counts

    out_arr = np.full(len(test_df), np.nan, dtype=np.float32)

    for gcode, s, e in zip(uniq_tr, start_idx, end_idx):
        te_mask = te_code == gcode
        if not np.any(te_mask):
            continue
        xg = x_tr_s[s:e]
        yg = y_tr_s[s:e]
        xq = x_te[te_mask]

        idx = np.searchsorted(xg, xq, side="left")
        idx0 = np.clip(idx - 1, 0, len(xg) - 1)
        idx1 = np.clip(idx, 0, len(xg) - 1)

        d0 = np.abs(xq - xg[idx0])
        d1 = np.abs(xq - xg[idx1])
        choose1 = d1 < d0  # tie -> idx0

        pick = np.where(choose1, idx1, idx0)
        out_arr[te_mask] = yg[pick].astype(np.float32, copy=False)

    out[outcol] = out_arr
    return out


def _left_join_series_by_keys(df, ref_df, key_cols, value_col, out_col, agg="median"):
    if len(ref_df) == 0:
        df[out_col] = np.nan
        return df
    s = getattr(ref_df.groupby(key_cols, sort=False)[value_col], agg)()
    s = s.rename(out_col)
    df = df.join(s, on=key_cols)
    return df


def _fallback_pressure_lookup_submission(output_df, u_in_round=1):
    """
    Deterministic lookup with backoffs, then snap to nearest valid pressure.
    Inspiratory-only train lookups (u_out==0). Test u_out==1 rows set to 0.0.
    """

    test_usecols = ["id", "breath_id", "R", "C", "time_step", "u_out", "u_in"]
    test_dtypes = {
        "id": "int32",
        "breath_id": "int32",
        "R": "int16",
        "C": "int16",
        "time_step": "float32",
        "u_in": "float32",
        "u_out": "int8",
    }
    test = pd.read_csv(TEST_PATH, usecols=test_usecols, dtype=test_dtypes)

    train_insp = df_train.loc[df_train["u_out"].eq(0)].copy()

    _add_time_index_inplace_sorted(train_insp)
    _add_time_index_inplace_sorted(test)

    _add_lag_features_fast_inplace_sorted(train_insp, lags=(1, 2, 3, 5))
    _add_lag_features_fast_inplace_sorted(test, lags=(1, 2, 3, 5))

    q_cols = [
        "u_in",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_lag3",
        "u_in_lag5",
        "u_in_diff1",
        "u_in_diff2",
        "u_in_cumsum",
    ]
    for c in q_cols:
        train_insp[c + "_q"] = _quantize_to_int(train_insp[c], scale=10.0)
        test[c + "_q"] = _quantize_to_int(test[c], scale=10.0)

    global_med = float(train_insp["pressure"].median())

    grp00 = [
        "R",
        "C",
        "t_idx",
        "u_out",
        "u_in_lag1_q",
        "u_in_lag2_q",
        "u_in_diff1_q",
        "u_in_diff2_q",
    ]
    grp0 = [
        "R",
        "C",
        "t_idx",
        "u_out",
        "u_in_lag1_q",
        "u_in_lag2_q",
        "u_in_lag3_q",
        "u_in_lag5_q",
        "u_in_diff1_q",
    ]
    grp1 = [
        "R",
        "C",
        "t_idx",
        "u_out",
        "u_in_lag1_q",
        "u_in_lag2_q",
        "u_in_diff1_q",
        "u_in_cumsum_q",
        "u_out_lag1",
        "u_out_lag2",
    ]
    grp2 = [
        "R",
        "C",
        "t_idx",
        "u_out",
        "u_in_lag1_q",
        "u_in_lag2_q",
        "u_in_diff1_q",
        "u_in_cumsum_q",
    ]

    m = test.copy()

    m = _nn_lookup_1d_by_group_vectorized(
        train_insp, m, grp00, "u_in_q", "pressure", "pq00"
    )
    m = _nn_lookup_1d_by_group_vectorized(
        train_insp, m, grp0, "u_in_q", "pressure", "pq0"
    )
    m = _nn_lookup_1d_by_group_vectorized(
        train_insp, m, grp1, "u_in_q", "pressure", "pq1"
    )
    m = _nn_lookup_1d_by_group_vectorized(
        train_insp, m, grp2, "u_in_q", "pressure", "pq2"
    )

    key_q = ["R", "C", "t_idx", "u_out", "u_in_q"]
    key_qb = ["R", "C", "t_idx", "u_out", "u_in_q", "u_in_lag1_q", "u_in_lag2_q"]
    key_qc = ["R", "C", "t_idx", "u_out", "u_in_q", "u_in_diff1_q"]
    key4 = ["R", "C", "t_idx", "u_out"]
    key5 = ["R", "C", "t_idx"]

    m = _left_join_series_by_keys(m, train_insp, key_q, "pressure", "pq", agg="median")
    m = _left_join_series_by_keys(
        m, train_insp, key_qb, "pressure", "pqb", agg="median"
    )
    m = _left_join_series_by_keys(
        m, train_insp, key_qc, "pressure", "pqc", agg="median"
    )
    m = _left_join_series_by_keys(m, train_insp, key4, "pressure", "p4", agg="median")
    m = _left_join_series_by_keys(m, train_insp, key5, "pressure", "p5", agg="median")

    pred = m["pq00"]
    pred = pred.fillna(m["pq0"])
    pred = pred.fillna(m["pq1"])
    pred = pred.fillna(m["pq2"])
    pred = pred.fillna(m["pq"])
    pred = pred.fillna(m["pqb"])
    pred = pred.fillna(m["pqc"])
    pred = pred.fillna(m["p4"])
    pred = pred.fillna(m["p5"])
    pred = pred.fillna(global_med)

    output_df = output_df.copy()
    out_pred = pred.to_numpy(dtype=np.float64, copy=False)

    out_pred[m["u_out"].to_numpy() == 1] = 0.0

    output_df["pressure"] = snap_to_nearest_pressure_vec(out_pred)
    return output_df


def g(dp):
    """
    Blend if external prediction files exist; otherwise deterministic fallback lookup.
    """
    output = pd.read_csv(SAMPLE_PATH)
    expected_len = len(output)

    l = _safe_list_pred_files(dp)
    if len(l) == 0:
        output2 = _fallback_pressure_lookup_submission(output, u_in_round=1)
        output2.to_csv("submission.csv", index=False)
        return

    file_count = len(l)
    loop_time = 154
    splits = max(1, file_count // 2)

    flist_paths = []
    for i in range(splits):
        start = i * round(len(l) / splits)
        end = None if i == splits - 1 else (i + 1) * round(len(l) / splits)
        chunk = l[start:end]
        if len(chunk) > 0:
            flist_paths.append(chunk)

    flist = []
    for chunk in flist_paths:
        valid_chunk = []
        for fp in chunk:
            v = _load_pressure_vector(fp, expected_len)
            if v is not None:
                valid_chunk.append(fp)
        if len(valid_chunk) == 0:
            continue
        try:
            vec = wc(valid_chunk)
            if len(vec) == expected_len:
                flist.append(vec.astype(np.float64, copy=False))
        except Exception:
            continue

    if len(flist) == 0:
        output2 = _fallback_pressure_lookup_submission(output, u_in_round=1)
        output2.to_csv("submission.csv", index=False)
        return

    n_models = len(flist)
    preds_mat = np.empty((loop_time, expected_len), dtype=np.float64)

    for loop_idx in range(loop_time):
        set_seed(loop_idx)
        w = np.fromiter(
            (rd() for _ in range(n_models)), dtype=np.float64, count=n_models
        )
        ws = w.sum()
        if ws == 0:
            w[:] = 1.0 / n_models
        else:
            w /= ws
        w.sort()
        w = w[::-1]

        temp = np.zeros(expected_len, dtype=np.float64)
        for j in range(n_models):
            temp += flist[j] * w[j]
        preds_mat[loop_idx] = temp

    output_pred = np.median(preds_mat, axis=0)
    output["pressure"] = snap_to_nearest_pressure_vec(output_pred)

    output.to_csv(f"rwb_{loop_time}_loops.csv", index=False)
    output.to_csv("submission.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.7 + b.pressure * 0.3
    a["pressure"] = snap_to_nearest_pressure_vec(
        a["pressure"].to_numpy(dtype=np.float64, copy=False)
    )
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
g("../input/gb-data-blending-recover")
