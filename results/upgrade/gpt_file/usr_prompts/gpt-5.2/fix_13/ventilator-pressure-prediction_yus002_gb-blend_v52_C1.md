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

0.1540316352128262

# 6. Current score

3.63018

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.24999) has done: 'The runtime error comes from trying to ensemble predictions from a directory (`../input/gb-blending`) that does not exist in this Kaggle environment, resulting in `file_count == 0` and a division by zero. To keep the original blending/rounding-to-nearest-pressure logic intact while making the notebook run end-to-end, I add a safe fallback: if no blend files are found, train a fast baseline model (per-time-step median pressure over `(R,C,time_step,u_out)` groups, with sensible fallbacks) and write a valid `submission.csv`. I also keep your `find_nearest` discretization so predictions are snapped to the known pressure grid, which is consistent with common solutions for this competition. Paths are adjusted to use the provided `/kaggle/input/...` location so the code runs reliably.'
- What this solution (achieved 7.86193) has done: 'Your current score (7.24999 MAE) suggests the fallback baseline is producing nearly-constant (or poorly aligned) predictions; we can improve toward the 0.154 target by keeping your fallback idea but making it time-series aware per breath. I keep your pressure-grid snapping (`find_nearest`) and submission writing exactly the same, but change the fallback mapping from a coarse group median to a minimal autoregressive “physics-lite” feature set (lagged `u_in`, cumulative `u_in`, and `R/C`), trained with a simple linear regression using only the inspiratory phase (`u_out==0`) as the metric requires. This preserves the core “fast baseline fallback” logic (no deep nets, no new training loops), but should dramatically reduce MAE and move much closer to your target band. I also ensure predictions for expiratory steps are set to a stable value (0) since they are not scored, which can slightly stabilize post-processing.'
- What this solution (achieved 7.86193) has done: 'Your fallback model is currently trained only on inspiratory rows but then it forces all expiratory (`u_out==1`) predictions to 0, which creates a large discontinuity at the insp/exp boundary and hurts because many inspiratory steps occur right before the valve opens. I keep your exact fallback approach (fast closed-form ridge + the same feature set + snapping to the known pressure grid), but change only the post-processing: for each breath, set expiratory predictions to the last predicted inspiratory pressure (a stable, physically plausible hold) instead of 0. This doesn’t change the scored rows directly, but it improves continuity and typically improves the learned mapping around transition regions, moving MAE down toward your target without altering core logic. I also ensure the output is aligned to `id` order and remains a valid `submission.csv`.'
- What this solution (achieved 7.79897) has done: 'Your current score (7.86 MAE) is far from the target (0.154), so the biggest “minimal” gain comes from fixing a metric-mismatch in the fallback training: you train only on inspiratory rows (`u_out==0`) but still include `u_out` as a feature, which becomes a constant and wastes capacity; instead, we train on all rows and apply sample weights so `u_out==0` dominates while still learning boundary behavior. To better respect the time-series nature without changing the modeling approach, I keep the same closed-form ridge and feature set but add two tiny “state-like” features (`u_in_diff1` and `u_out_lag1`) that are computed per-breath and are cheap. I also fix a subtle indexing bug where predictions were re-ordered incorrectly (using `te_sorted.index` to index into a prediction array that is already in original row order), which can severely degrade accuracy. The output format and the existing `find_nearest` pressure-grid snapping are preserved, and the script still falls back to the baseline when the blend directory is missing.'
- What this solution (achieved 7.87086) has done: 'Your current score (7.80 MAE, lower is better) is far from the 0.154 target, so we need a real accuracy lift while keeping your core fallback approach (closed-form ridge + same feature family + pressure-grid snapping) intact. The biggest minimal fix is to align training with the evaluation: train only on inspiratory rows (`u_out==0`) and drop `u_out` from features (it was constant during training previously and can hurt conditioning), while still predicting for all test rows. To further reduce MAE without changing the model class, we add two tiny, cheap features (`u_in_roll3` and `u_in_ewm`) that are computed per-breath and help the linear model capture local dynamics. Finally, we remove the unnecessary `id`-based reindexing (which can misalign rows if `id` assumptions ever break) and instead write predictions directly in the original test row order, guaranteeing correct alignment with `sample_submission.csv`.'
- What this solution (achieved 7.87086) has done: 'Your score is far above the target (lower is better), so we need a real accuracy lift while keeping your existing core fallback approach (closed-form ridge + your feature family + snapping to the known pressure grid) intact. The largest minimal win here is fixing a subtle but severe indexing bug in the “hold expiratory to last inspiratory” loop: it currently uses positional indices from `groupby().indices` against arrays that are in a different row order, which scrambles per-breath post-processing and destroys accuracy. I rewrite that loop to use per-breath boolean masks on the already-sorted test frame (same logic, correct alignment), then map back to original test order safely. No model/feature/loss changes are made; only the post-processing alignment is corrected so predictions correspond to the right rows.'
- What this solution (achieved 7.87086) has done: 'Your current MAE (7.87) is far from the target (0.154, lower is better), so the best minimal move is to fix a major alignment bug in the fallback post-processing that scrambles per-breath “hold expiratory to last inspiratory” assignments. I keep your exact core approach (closed-form ridge on lightweight engineered features + snapping to the known pressure grid) and only change the way we iterate breaths: we operate on contiguous slices in the already-sorted test frame instead of using `.isin()` with mismatched boolean masks. This preserves evaluation semantics while ensuring each breath’s predictions are modified correctly and then mapped back to the original test row order. The rest of the pipeline, including the blend fallback behavior and CSV writing, remains the same.'
- What this solution (achieved 7.77443) has done: 'Your current score (7.87 MAE) is far from the 0.154 target (lower is better), so we should improve the fallback baseline accuracy while keeping the same core idea (fast closed-form ridge regression + pressure-grid snapping). The biggest minimal win is to make the ridge training consistent with the evaluation: compute MAE only on inspiratory phase, so we still train on inspiratory rows only but also (a) exclude rows after the first `u_out==1` per breath to avoid learning noisy transition behavior, and (b) add a per-breath time index feature to help the linear model fit the sequence shape without changing the model class. We also fix a subtle but important feature bug: `u_in_roll3` and `u_in_ewm` were computed on raw `u_in` rather than `float32`, which can cause dtype upcasting and inconsistent scaling; we keep the same features but make their computation numerically consistent. Finally, we keep your expiratory “hold last inspiratory” post-processing and the `find_nearest` snapping unchanged, and still write `submission.csv` end-to-end when the blend directory is missing.'
- What this solution (achieved 4.45971) has done: 'Your current MAE (7.77, lower is better) is far above the target (0.154), and the biggest likely issue is prediction/row misalignment caused by building `pred_sorted` from `df_test.index` while `pred` is in the feature-sorted order, plus a second misalignment when mapping back. I keep your exact fallback model class (closed-form ridge), feature set, pressure-grid snapping, and “hold expiratory to last inspiratory” idea, but fix the ordering so predictions are generated and post-processed in a single consistent `(breath_id, time_step)` sorted order and then mapped back to original `df_test` order by index. I also make `df_train` read only the needed columns to reduce memory/time without changing semantics. These minimal alignment fixes should move the score substantially toward the target while preserving your core logic and producing a valid `submission.csv`.'
- What this solution (achieved 3.63018) has done: 'Your current score is still far from the target (lower is better), so the most likely remaining issue is that the fallback model is learning the wrong relationship for many breaths because it doesn’t include key “state” information that strongly correlates with pressure (integrated flow/volume and recent valve behavior), even though we must keep the same core ridge-regression approach. I keep your exact fallback structure (feature engineering → closed-form ridge → “hold expiratory to last inspiratory” → snap to pressure grid → write `submission.csv`), but add two minimal, cheap per-breath state-like features (`vol` as cumulative integral of `u_in` and `time_step_diff`, and `u_out_first` as “has exhalation started”) plus one interaction (`R_div_C * vol`) which is commonly important for this competition. I also fix a subtle numerical issue by computing `dt` from sorted `time_step` float32 consistently and reusing it for both `u_in_cum` and `vol` so the integrals are stable and aligned. These changes are directly aimed at reducing MAE on inspiratory steps while keeping runtime well under the limit and preserving your pipeline and submission format.'
- What this solution (achieved 3.69212) has done: 'Your MAE (3.63, lower is better) is still far from the 0.154 target, so we should make a small, metric-aligned improvement without changing your core approach (fallback ridge regression + per-breath post-processing + snapping to the known pressure grid). The competition scores only inspiratory rows (`u_out==0`), so the safest direct gain is to (1) train on *all* rows but apply heavy sample weights to inspiratory rows (so the model still “sees” boundary behavior while optimizing the scored phase), and (2) apply the same inspiratory-only mask when snapping to the pressure grid (snapping expiratory rows doesn’t help the metric and can inject unnecessary distortion into the per-breath “hold” step). These changes keep your exact model class, closed-form solve, features, and “hold expiratory to last inspiratory” logic intact, while making training/post-processing better match the evaluation. The script still writes a valid `submission.csv` and continues to fall back automatically when the blend directory is missing.'
- What this solution (achieved 3.63018) has done: 'We’re far above the target (3.69 vs 0.154 MAE; lower is better), so we need a real accuracy lift while keeping your existing fallback core (closed-form ridge + same feature family + “hold expiratory” + pressure-grid snapping). The biggest minimal gain here is to make training match the evaluation more directly by training only on the truly-scored inspiratory timesteps (before the first `u_out==1` per breath), instead of using weighted full-breath training that still “pulls” the linear fit toward expiratory dynamics. I also apply the same inspiratory-only mask consistently when forming `X_tr/y_tr` (so weights aren’t needed), and keep the rest of your pipeline unchanged: same features, same ridge solve, same hold-last-inspiratory post-process, same snapping behavior for inspiratory rows only, and the same submission writing. This is a small, targeted change that should move MAE substantially down toward your target band without changing the model class or post-processing semantics.'

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
TRAIN_PATH = "/kaggle/input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "/kaggle/input/ventilator-pressure-prediction/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv"

df_train = pd.read_csv(
    TRAIN_PATH,
    usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
)

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)


def find_nearest(prediction: float) -> float:
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


def set_seed(seed: int = 2021):
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


def _make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create lightweight per-breath time-series features (lags + cumulative signals).
    """
    df = df.sort_values(["breath_id", "time_step"], kind="mergesort").copy()

    u_in_f = df["u_in"].astype(np.float32)
    t_f = df["time_step"].astype(np.float32)

    df["R_div_C"] = (df["R"].astype(np.float32) / df["C"].astype(np.float32)).astype(
        np.float32
    )
    df["u_in_sq"] = (u_in_f * u_in_f).astype(np.float32)

    df["u_in_lag1"] = (
        df.groupby("breath_id", sort=False)["u_in"]
        .shift(1)
        .fillna(0.0)
        .astype(np.float32)
    )
    df["u_in_lag2"] = (
        df.groupby("breath_id", sort=False)["u_in"]
        .shift(2)
        .fillna(0.0)
        .astype(np.float32)
    )

    df["u_in_diff1"] = (u_in_f - df["u_in_lag1"]).astype(np.float32)

    df["u_out_lag1"] = (
        df.groupby("breath_id", sort=False)["u_out"]
        .shift(1)
        .fillna(0)
        .astype(np.float32)
    )

    dt = (
        df.groupby("breath_id", sort=False)["time_step"]
        .diff()
        .fillna(0.0)
        .astype(np.float32)
    )
    df["dt"] = dt

    df["u_in_cum"] = (
        (u_in_f * dt).groupby(df["breath_id"], sort=False).cumsum()
    ).astype(np.float32)

    df["t_u_in"] = (t_f * u_in_f).astype(np.float32)

    df["u_in_roll3"] = (
        u_in_f.groupby(df["breath_id"], sort=False)
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .astype(np.float32)
    )
    df["u_in_ewm"] = (
        u_in_f.groupby(df["breath_id"], sort=False)
        .transform(lambda s: s.ewm(span=5, adjust=False).mean())
        .astype(np.float32)
    )

    df["step"] = df.groupby("breath_id", sort=False).cumcount().astype(np.float32)
    df["step2"] = (df["step"] * df["step"]).astype(np.float32)

    vol = ((u_in_f * dt).groupby(df["breath_id"], sort=False).cumsum()).astype(
        np.float32
    )
    df["vol"] = vol

    u_out_i8 = df["u_out"].to_numpy(np.int8)
    u_out_first = (
        pd.Series(u_out_i8)
        .groupby(df["breath_id"], sort=False)
        .cumsum()
        .clip(upper=1)
        .astype(np.float32)
        .to_numpy()
    )
    df["u_out_first"] = u_out_first.astype(np.float32)

    df["RdivC_vol"] = (df["R_div_C"] * df["vol"]).astype(np.float32)

    return df


def _fit_ridge_closed_form(
    X: np.ndarray, y: np.ndarray, alpha: float = 1.0
) -> np.ndarray:
    """
    Stable closed-form ridge regression (no extra packages).
    """
    X = X.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    XtX = X.T @ X
    n_feat = XtX.shape[0]
    XtX.flat[:: n_feat + 1] += alpha
    Xty = X.T @ y
    w = np.linalg.solve(XtX, Xty)
    return w


def _fit_ridge_closed_form_weighted(
    X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray, alpha: float = 1.0
) -> np.ndarray:
    """
    Minimal extension of the same closed-form ridge:
    solves argmin ||sqrt(w)*(Xb - y)||^2 + alpha||b||^2.
    """
    X = X.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    sw = sample_weight.astype(np.float64, copy=False)
    sw = np.clip(sw, 1e-12, None)

    sqrtw = np.sqrt(sw)[:, None]
    Xw = X * sqrtw
    yw = y * sqrtw[:, 0]

    XtX = Xw.T @ Xw
    n_feat = XtX.shape[0]
    XtX.flat[:: n_feat + 1] += alpha
    Xty = Xw.T @ yw
    w = np.linalg.solve(XtX, Xty)
    return w


def _first_insp_only_mask(sorted_df: pd.DataFrame) -> np.ndarray:
    """
    Mask for inspiratory phase as used by the competition metric.
    Restrict to steps before the first u_out==1 in each breath.
    Assumes sorted by (breath_id, time_step).
    """
    u_out = sorted_df["u_out"].to_numpy(np.int8)
    breath = sorted_df["breath_id"].to_numpy()
    n = len(sorted_df)

    change_idx = np.flatnonzero(breath[1:] != breath[:-1]) + 1
    starts = np.r_[0, change_idx]
    ends = np.r_[change_idx, n]

    mask = np.zeros(n, dtype=bool)
    for s, e in zip(starts, ends):
        uo = u_out[s:e]
        ones = np.flatnonzero(uo == 1)
        cut = (s + ones[0]) if len(ones) else e
        if cut > s:
            mask[s:cut] = True
    return mask


def _fallback_baseline_submission(output_path: str = "submission.csv") -> pd.DataFrame:
    """
    If no external blend files are available, create a valid submission by
    learning a simple mapping from train and predicting on test.

    Core semantics preserved:
    - ridge regression (closed-form)
    - same feature family
    - hold expiratory to last inspiratory within each breath
    - snap predictions to known pressure grid (inspiratory rows only)
    """
    df_test = pd.read_csv(
        TEST_PATH, usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
    )
    sub = pd.read_csv(SAMPLE_SUB_PATH)

    tr_fe = _make_features(df_train)
    te_fe = _make_features(df_test)

    feat_cols = [
        "time_step",
        "step",
        "step2",
        "dt",
        "u_in",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_diff1",
        "u_in_cum",
        "vol",
        "u_out_lag1",
        "u_out_first",
        "R",
        "C",
        "R_div_C",
        "RdivC_vol",
        "u_in_sq",
        "t_u_in",
        "u_in_roll3",
        "u_in_ewm",
    ]

    tr_sorted = tr_fe.sort_values(["breath_id", "time_step"], kind="mergesort")

    insp_mask_tr_sorted = _first_insp_only_mask(tr_sorted)
    X_tr = tr_sorted.loc[insp_mask_tr_sorted, feat_cols].to_numpy(dtype=np.float32)
    y_tr = tr_sorted.loc[insp_mask_tr_sorted, "pressure"].to_numpy(dtype=np.float32)

    te_sorted = te_fe.sort_values(["breath_id", "time_step"], kind="mergesort")
    X_te_sorted = te_sorted[feat_cols].to_numpy(dtype=np.float32)

    mu = X_tr.mean(axis=0, keepdims=True)
    sigma = X_tr.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)

    X_trs = (X_tr - mu) / sigma
    X_tes = (X_te_sorted - mu) / sigma

    X_trs = np.hstack([np.ones((X_trs.shape[0], 1), dtype=X_trs.dtype), X_trs])
    X_tes = np.hstack([np.ones((X_tes.shape[0], 1), dtype=X_tes.dtype), X_tes])

    w = _fit_ridge_closed_form(X_trs, y_tr, alpha=1.0)

    pred_sorted = (X_tes @ w).astype(np.float32)

    u_out_sorted = te_sorted["u_out"].to_numpy(dtype=np.int8)
    held_sorted = pred_sorted.copy()

    breath_ids = te_sorted["breath_id"].to_numpy()
    change_idx = np.flatnonzero(breath_ids[1:] != breath_ids[:-1]) + 1
    starts = np.r_[0, change_idx]
    ends = np.r_[change_idx, len(te_sorted)]

    for s, e in zip(starts, ends):
        uo = u_out_sorted[s:e]
        ps = held_sorted[s:e]
        insp_mask = uo == 0
        if np.any(insp_mask):
            last_insp = ps[np.where(insp_mask)[0][-1]]
            ps[~insp_mask] = last_insp
            held_sorted[s:e] = ps

    held = (
        pd.Series(held_sorted, index=te_sorted.index)
        .reindex(df_test.index)
        .to_numpy(dtype=np.float32)
    )

    if len(sub) != len(held):
        raise ValueError(f"Row count mismatch: sub={len(sub)} vs pred={len(held)}")

    insp_mask_test = df_test["u_out"].to_numpy(np.int8) == 0
    out_pressure = held.astype(np.float32, copy=True)

    out_pressure[insp_mask_test] = np.array(
        [find_nearest(v) for v in out_pressure[insp_mask_test]], dtype=np.float32
    )

    sub["pressure"] = out_pressure.astype(float)
    sub.to_csv(output_path, index=False)
    return sub


def g(dp: str):
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)

    file_count = len(l)
    if file_count == 0:
        print(
            f"[INFO] No blend files found in: {dp}. Using fallback baseline to create submission.csv"
        )
        return _fallback_baseline_submission("submission.csv")

    loop_time = 1131 // file_count
    if loop_time <= 0:
        loop_time = 1

    splits = file_count // 2
    if splits <= 0:
        splits = 1

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
    for lp in range(loop_time):
        weight = []
        set_seed(lp)
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

    output = pd.read_csv(SAMPLE_SUB_PATH)
    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv("submission.csv", index=False)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)
    return output


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.6 + b.pressure * 0.4
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
g("/kaggle/input/gb-blending")
print("[INFO] Done. Wrote submission.csv")
