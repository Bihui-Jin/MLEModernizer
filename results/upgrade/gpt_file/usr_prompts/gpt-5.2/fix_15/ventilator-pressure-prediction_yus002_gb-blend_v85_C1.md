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

0.1474

# 6. Current score

7.21324

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65244) has done: 'The crash happens because the code tries to blend prediction files from `../input/gb-blending`, but in your environment that folder either doesn’t exist or doesn’t contain valid submissions, so `pred_list` ends up producing a single scalar instead of a 603600-length vector. I (1) make the input-path discovery robust by checking multiple known dataset locations, (2) add strict validation that every blended file has the right columns/length and skip invalid ones, and (3) add a safe fallback that creates a valid submission (all-zero pressures snapped to the nearest known pressure) if no blend inputs are available—so you always get a `.csv` submission end-to-end. These fixes are score-neutral unless blending files actually exist; if they do exist, the intended blending logic run as originally designed.'
- What this solution (achieved 6.5079) has done: 'I fix the runtime crash in the fallback baseline model by changing Ridge’s solver away from `auto` (which selects `sparse_cg` and triggers a SciPy/Sklearn incompatibility around `cg(tol=...)`). This is a minimal, score-neutral change intended only to make the pipeline run end-to-end and always write a valid `submission.csv`. I also keep the existing blending logic intact and only add a small robustness tweak to ensure the output file name is always a valid `.csv`. No changes are made to the overall approach beyond the necessary solver compatibility fix.'
- What this solution (achieved 6.5079) has done: 'Your current score (6.5079 MAE, lower is better) is far from the target (0.1474), so we need a small change that improves correctness with respect to the metric without changing the overall “simple baseline model” approach. The biggest issue is that you predict pressures for expiratory timesteps (u_out=1) even though they are not scored; this hurts MAE because expiratory pressures follow a different regime. I keep the same Ridge pipeline and features, but (1) train only on inspiratory rows (as you already do) and (2) at inference, overwrite predictions for u_out=1 with a neutral/typical value learned from training expiratory pressures (per R,C group) so expiratory errors don’t dominate and predictions remain realistic. This is minimal, preserves the core logic, and should move the MAE substantially toward the target band.'
- What this solution (achieved 5.9959) has done: 'Your current MAE (6.5079, lower is better) is far from the target (0.1474), so the most “minimal but meaningful” improvement is to align training/inference with the competition’s scoring (only inspiratory phase is scored). I keep your exact Ridge + OneHotEncoder pipeline and feature set, but I (1) remove `breath_id` from the categorical features (it is essentially unique and makes the one-hot space explode, harming generalization and stability), and (2) enforce a metric-aligned post-process: for `u_out==1` (expiratory, not scored), set pressure to a reasonable per-(R,C) median from training to avoid dragging predictions into unrealistic values. Everything else (data paths, snapping to nearest known pressure, CSV writing) is preserved so it still runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 5.9959) has done: 'Your current MAE (5.9959, lower is better) is still far from the target (0.1474), so we should make a minimal change that improves *metric alignment* without changing the Ridge baseline’s core logic. The biggest remaining mismatch is that you currently train on inspiratory rows only but still include `u_out` as a feature (constant 0 in training), then at test time you feed both `u_out=0/1`, creating an out-of-distribution feature shift and noisier inspiratory predictions. I keep the same Ridge + OneHot pipeline and the same features otherwise, but drop `u_out` from the model features while keeping your existing post-process for `u_out==1` timesteps. This should improve inspiratory predictions (the only scored rows) and move MAE materially toward the target while staying well within the “minimal change” constraint.'
- What this solution (achieved 5.9959) has done: 'We keep your Ridge + OneHotEncoder baseline exactly as-is, but fix one metric-alignment bug: the competition scores MAE only on inspiratory rows (u_out==0), so we should not “optimize” expiratory (u_out==1) predictions at all. Your current post-process overwrites expiratory predictions with a plausible median pressure, which can indirectly worsen calibrations/snap-to-grid behavior and isn’t needed for scoring; instead we set expiratory predictions to a neutral constant (0, then snapped) so they don’t interfere and keep the scored inspiratory predictions untouched. This is a minimal change (only post-processing), preserves your core training logic, and should move MAE substantially down toward the target. All file paths and CSV writing remain unchanged and a valid `submission.csv` is always produced.'
- What this solution (achieved 5.9959) has done: 'We keep your Ridge + OneHotEncoder baseline unchanged and focus on the biggest remaining metric mismatch: you train only on inspiratory rows but still score against inspiratory-only MAE, so predicting arbitrary values for expiratory rows is irrelevant—but your current approach sets them to 0 then snaps to a valid pressure, which can create many extreme/off-grid values if anything goes wrong with alignment. The minimal, score-improving change is to ensure expiratory rows are set to the per-(R,C) *median expiratory pressure from training* (computed once) while keeping inspiratory predictions exactly as produced by the model; this stabilizes the full series and typically reduces unintended spillover/edge cases without changing core training logic. We also add a tiny robustness check to guarantee `output["id"]` comes from `sample_submission` so row alignment is never broken. This should move MAE down from ~5.99 toward the target without changing architecture/training loops.'
- What this solution (achieved 5.8821) has done: 'Your MAE is far above the target, so we need a small, metric-aligned improvement without changing the Ridge + OHE baseline approach. The main issue is that the pressure values lie on a fixed discrete grid, and rounding each prediction independently to the nearest pressure can be noisy; a minimal improvement is to snap using a fully vectorized nearest-grid method (stable and faster) and then apply a tiny global calibration shift chosen via an internal train/validation split on inspiratory rows only. This preserves the same model/feature set/training approach, but better aligns predictions to the discrete target and slightly corrects systematic bias, which should move MAE materially downward toward the target. The blending logic and file paths stay intact; if no blend files exist, it still write a valid `submission.csv`.'
- What this solution (achieved 5.8821) has done: 'We keep your Ridge+OHE baseline exactly intact and focus on one minimal, metric-aligned correction: the current “shift calibration” randomly splits individual rows, which leaks breath structure across train/valid and produces a misleading shift that can hurt the scored inspiratory MAE. I change that calibration split to be by `breath_id` (on inspiratory rows only), so the chosen global shift is more reliable while preserving the same model, features, training, and snapping logic. Everything else (paths, blending fallback, snapping to discrete pressure grid, and writing `submission.csv`) stays the same and still runs end-to-end within the time limit.'
- What this solution (achieved 7.21324) has done: 'Your current MAE (5.8821, lower is better) is still far from the target (0.1474), so we need a minimal change that improves correctness with respect to the metric without changing your Ridge+OHE core approach. The biggest remaining gap is that the model ignores the strong sequential structure within each breath; we can keep the same model and training loop but add a couple of lightweight lag features (previous `u_in` and previous `u_out`) computed within each `breath_id`, which usually improves inspiratory pressure prediction substantially for this competition. I also apply the same shift-calibration on a breath-level split, now using these new features, preserving your existing snapping-to-grid post-processing. All paths and submission writing remain unchanged and it still produces a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
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
        "/kaggle/data/train.csv",
        "/kaggle/input/train.csv",
    ]
)
TEST_PATH = _first_existing(
    [
        "../input/ventilator-pressure-prediction/test.csv",
        "/kaggle/input/ventilator-pressure-prediction/test.csv",
        "/kaggle/data/test.csv",
        "/kaggle/input/test.csv",
    ]
)
SAMPLE_SUB_PATH = _first_existing(
    [
        "../input/ventilator-pressure-prediction/sample_submission.csv",
        "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
    ]
)

if TRAIN_PATH is None or TEST_PATH is None or SAMPLE_SUB_PATH is None:
    raise FileNotFoundError(
        f"Could not find required files. TRAIN_PATH={TRAIN_PATH}, TEST_PATH={TEST_PATH}, SAMPLE_SUB_PATH={SAMPLE_SUB_PATH}"
    )

df_train = pd.read_csv(TRAIN_PATH)
sorted_pressures = np.sort(df_train["pressure"].unique()).astype(np.float32)
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


def snap_to_pressure_grid(pred_arr: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred_arr, dtype=np.float32)
    idx = np.searchsorted(sorted_pressures, pred, side="left")
    idx0 = np.clip(idx - 1, 0, total_pressures_len - 1)
    idx1 = np.clip(idx, 0, total_pressures_len - 1)
    p0 = sorted_pressures[idx0]
    p1 = sorted_pressures[idx1]
    choose1 = np.abs(p1 - pred) < np.abs(p0 - pred)
    out = np.where(choose1, p1, p0).astype(np.float32)
    return out


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def _read_pred_file(path, expected_len):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "pressure" not in df.columns:
        return None
    arr = df["pressure"].to_numpy().ravel()
    if arr.shape[0] != expected_len:
        return None
    return arr


def wc(input_list, expected_len):
    l = []
    preds = []
    for i in range(len(input_list)):
        try:
            public_lb_score = int(
                input_list[i].split("/")[-1].split(".")[1].split(" ")[0]
            )
        except Exception:
            public_lb_score = 1
        arr = _read_pred_file(input_list[i], expected_len)
        if arr is None:
            continue
        l.append(public_lb_score)
        preds.append(arr)

    if len(preds) == 0:
        return None
    if len(preds) == 1:
        return preds[0]

    l_sum = sum(l)
    weight1 = (l[1] / l_sum) + 0.1
    weight2 = 1 - weight1
    output = preds[0] * weight1 + preds[1] * weight2
    return output


def _baseline_model_submission():
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge

    sample = pd.read_csv(SAMPLE_SUB_PATH)
    expected_len = len(sample)

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    for df in (train, test):
        df.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")
        df["u_in_lag1"] = (
            df.groupby("breath_id")["u_in"].shift(1).fillna(0.0).astype(np.float32)
        )
        df["u_out_lag1"] = (
            df.groupby("breath_id")["u_out"].shift(1).fillna(0).astype(np.int64)
        )

    train_insp = train[train["u_out"] == 0].copy()

    feature_cols_num = ["time_step", "u_in", "u_in_lag1"]
    feature_cols_cat = ["R", "C", "u_out_lag1"]

    X = train_insp[feature_cols_num + feature_cols_cat]
    y = train_insp["pressure"].astype(np.float32)

    X_test = test[feature_cols_num + feature_cols_cat]

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", feature_cols_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_cat),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    model = Pipeline(
        steps=[
            ("pre", pre),
            ("model", Ridge(alpha=1.0, solver="lsqr", random_state=0)),
        ]
    )

    rng = np.random.RandomState(2021)
    breath_ids = train_insp["breath_id"].to_numpy()
    uniq_b = np.unique(breath_ids)
    rng.shuffle(uniq_b)
    split_b = int(len(uniq_b) * 0.9)
    tr_b, va_b = uniq_b[:split_b], uniq_b[split_b:]
    tr_mask = np.isin(breath_ids, tr_b)
    va_mask = ~tr_mask

    X_tr = X.loc[tr_mask]
    y_tr = y.loc[tr_mask]
    X_va = X.loc[va_mask]
    y_va = y.loc[va_mask].to_numpy()

    model.fit(X_tr, y_tr)
    va_pred = model.predict(X_va).astype(np.float32)

    candidate_shifts = np.linspace(-1.0, 1.0, 41, dtype=np.float32)  # step=0.05
    best_shift = np.float32(0.0)
    best_mae = np.inf
    for s in candidate_shifts:
        snapped = snap_to_pressure_grid(va_pred + s)
        mae = np.mean(np.abs(snapped - y_va))
        if mae < best_mae:
            best_mae = mae
            best_shift = s

    model.fit(X, y)
    pred = model.predict(X_test).astype(np.float32)
    pred = (pred + best_shift).astype(np.float32)

    exp_med = (
        train.loc[train["u_out"] == 1, ["R", "C", "pressure"]]
        .groupby(["R", "C"], as_index=False)["pressure"]
        .median()
    )
    exp_med_map = {(int(r), int(c)): float(p) for r, c, p in exp_med.to_numpy()}
    global_exp_median = (
        float(train.loc[train["u_out"] == 1, "pressure"].median())
        if (train["u_out"] == 1).any()
        else float(train["pressure"].median())
    )

    test_u_out = test["u_out"].to_numpy()
    if np.any(test_u_out == 1):
        rc_pairs = list(
            zip(test["R"].astype(int).to_numpy(), test["C"].astype(int).to_numpy())
        )
        exp_fill = np.array(
            [exp_med_map.get(rc, global_exp_median) for rc in rc_pairs],
            dtype=np.float32,
        )
        pred = np.where(test_u_out == 1, exp_fill, pred).astype(np.float32)

    out = sample.copy()
    out["id"] = sample["id"].to_numpy()
    out["pressure"] = snap_to_pressure_grid(pred)

    if len(out) != expected_len:
        raise RuntimeError("Submission length mismatch with sample_submission.")
    if "pressure" not in out.columns or "id" not in out.columns:
        raise RuntimeError("Submission missing required columns.")
    if out["pressure"].isna().any():
        raise RuntimeError("Submission contains NaN pressures.")

    out_path = "submission.csv"
    out.to_csv(out_path, index=False)
    return out_path


def g(dp):
    output = pd.read_csv(SAMPLE_SUB_PATH)
    expected_len = len(output)

    if dp is None or (not os.path.exists(dp)):
        return _baseline_model_submission()

    files = [p for p in glob.iglob(f"{dp}/*") if os.path.isfile(p)]
    files.sort()
    if len(files) == 0:
        return _baseline_model_submission()

    file_count = len(files)
    loop_time = 150
    splits = max(1, file_count // 2)

    flist_paths = []
    for i in range(splits):
        start = i * round(len(files) / splits)
        end = None if i == splits - 1 else (i + 1) * round(len(files) / splits)
        chunk = files[start:end]
        flist_paths.append(chunk)

    flist = []
    for chunk in flist_paths:
        pred = wc(chunk, expected_len=expected_len)
        if pred is not None:
            flist.append(pred)

    if len(flist) == 0:
        return _baseline_model_submission()

    pred_list = []
    for t in range(loop_time):
        set_seed(t)
        weight = [rd() for _ in range(len(flist))]
        weight_sum = sum(weight)
        weight = [w / weight_sum for w in weight]
        weight.sort(reverse=True)

        temp = np.zeros(expected_len, dtype=np.float32)
        for i in range(len(flist)):
            temp += flist[i].astype(np.float32) * weight[i]
        pred_list.append(temp)
        del temp
        gc.collect()

    output["pressure"] = np.median(np.vstack(pred_list), axis=0).astype(np.float32)
    output["pressure"] = snap_to_pressure_grid(output["pressure"].to_numpy())

    out_path = f"rwb_{loop_time}_loops.csv"
    output.to_csv(out_path, index=False)
    return out_path


def blend(a, b):
    sub = pd.read_csv(SAMPLE_SUB_PATH)
    expected_len = len(sub)

    a_df = pd.read_csv(a)
    b_df = pd.read_csv(b)
    if "pressure" not in a_df.columns or "pressure" not in b_df.columns:
        raise ValueError("Both input files must contain a 'pressure' column.")
    if len(a_df) != expected_len or len(b_df) != expected_len:
        raise ValueError("Input prediction files must match sample_submission length.")

    a_df["pressure"] = (
        a_df["pressure"].to_numpy(dtype=np.float32) * 0.5
        + b_df["pressure"].to_numpy(dtype=np.float32) * 0.5
    )
    a_df["pressure"] = snap_to_pressure_grid(a_df["pressure"].to_numpy())
    a_df.to_csv("blend.csv", index=False)
    return a_df




## === cell 2
blend_dp = _first_existing(
    [
        "../input/gb-blending",
        "/kaggle/input/gb-blending",
        "/kaggle/data/gb-blending",
    ]
)

out_path = g(blend_dp)

sample = pd.read_csv(SAMPLE_SUB_PATH)
expected_len = len(sample)

if out_path != "submission.csv":
    df_out = pd.read_csv(out_path)
    if "id" not in df_out.columns:
        df_out["id"] = sample["id"].to_numpy()
    df_out = df_out[["id", "pressure"]]
    if len(df_out) != expected_len:
        raise RuntimeError("Final submission length mismatch.")
    df_out.to_csv("submission.csv", index=False)

final_df = pd.read_csv("submission.csv")
if list(final_df.columns) != ["id", "pressure"]:
    final_df = final_df[["id", "pressure"]]
    final_df.to_csv("submission.csv", index=False)
if len(final_df) != expected_len:
    raise RuntimeError("submission.csv length mismatch with sample_submission.")
if final_df["pressure"].isna().any():
    raise RuntimeError("submission.csv contains NaN pressures.")

print("Wrote:", out_path, "and submission.csv")
print(final_df.head())
print(final_df.shape)
