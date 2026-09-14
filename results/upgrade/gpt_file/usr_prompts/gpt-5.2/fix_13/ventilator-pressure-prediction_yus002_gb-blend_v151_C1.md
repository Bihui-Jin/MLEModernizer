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

0.1365913784900642

# 6. Current score

2.55538

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.92097) has done: 'I fix the runtime error by making the blending code robust to (a) missing/empty input directories and (b) wrong-length prediction files, which currently causes the median stack to collapse to a single value and fail assignment. Because your current approach depends on external Kaggle “input datasets” that are not guaranteed to exist, I add a safe fallback that generates a valid submission directly from the provided competition data using a minimal baseline (predicting per-(R,C,time_step) median pressure from train). I keep your nearest-pressure snapping logic unchanged so outputs remain on the discrete pressure grid used in training. The result run end-to-end and always write a valid `submission.csv` with the required `id,pressure` columns.'
- What this solution (achieved 3.51595) has done: 'Your current score (MAE 9.92097; lower is better) is far from the target (0.1366), so the issue is that the fallback baseline is much too weak. I keep your blending logic intact, but strengthen only the fallback to a known strong classical approach for this competition: a per-(R,C) linear mapping from cumulative inspired volume (`cum_u_in`) to pressure, trained on inspiratory-only points (`u_out==0`) and applied sequentially per breath. I also keep your discrete-pressure snapping unchanged and ensure the submission stays aligned by `id` and is always written to a `.csv`. This is a minimal change in spirit (still a deterministic, non-neural baseline) but should move the MAE dramatically toward your target band.'
- What this solution (achieved 3.3391) has done: 'Your current score is far above the target (lower is better), so we should improve the fallback without changing the overall approach (still a deterministic classical baseline + nearest-pressure snapping). The main issue is the current fallback ignores the dominant dynamic driver (flow/volume) and never conditions on `u_out` beyond filtering training, so it predicts non-physical pressures during exhalation and misses key temporal dynamics during inhalation. I keep your blending logic and snapping unchanged, but strengthen the fallback to a minimal “RC+state” linear model trained only on inspiratory rows: predict pressure from (`cum_u_in`, instantaneous `u_in`, `time_step`) per (R,C), which is still simple linear regression with closed-form least squares. This should move MAE substantially toward your target while staying lightweight and within the same core semantics.'
- What this solution (achieved 2.41237) has done: 'We need to move your MAE down from 3.3391 toward 0.1366 (lower is better), so the fallback must become meaningfully stronger while keeping the same “deterministic classical baseline + nearest-pressure snapping” core. The biggest missing piece is breath-history: pressure depends heavily on recent control history, not just instantaneous `u_in/time_step`, so I minimally extend your closed-form per-(R,C) linear regression to include a few lagged and cumulative features (still linear, still closed-form, no training loop). I also keep the inspiratory-only fitting and the `u_out==1 -> 0` handling, but make the prediction vectorized and ensure strict `id` alignment. These are small, metric-aligned changes that should reduce MAE substantially without changing the overall approach.'
- What this solution (achieved 2.35394) has done: 'Your current MAE (2.41237; lower is better) is still far above the target (0.13659), so we should improve the fallback model while keeping the same deterministic per-(R,C) closed-form linear regression + nearest-pressure snapping core. The smallest high-impact fix is to align training with the metric by fitting only on inspiratory rows **and** scoring-like behavior at prediction time: keep u_out==1 set to 0, but also ensure the regression emphasizes the inspiratory dynamics by adding a couple more history features that capture “memory” (short rolling sums/means of u_in) without changing the model class. I also make the feature construction slightly more robust (avoid any NaNs/infs) and keep strict `id` alignment so the submission rows match the sample submission exactly. These changes should move MAE meaningfully downward while preserving your approach and keeping runtime under the timeout.'
- What this solution (achieved 2.35394) has done: 'Your current MAE (2.35394; lower is better) is still far above the target (0.13659), so we should make a small, metric-aligned improvement to the fallback model without changing its core (deterministic per-(R,C) closed-form linear regression + nearest-pressure snapping). The biggest low-risk gain is to better respect the scoring rule (only inspiratory phase is scored): keep predictions during `u_out==1` from contaminating the learned dynamics by training strictly on `u_out==0`, and at inference time carry forward the last predicted inspiratory pressure within each breath during `u_out==1` (instead of forcing 0), which is closer to typical baselines for this competition while not affecting the scoring phase directly. To avoid accidental score regressions from numerical instability, we also standardize features using train statistics (still linear regression, same semantics) and keep everything strictly aligned by `id`. These are minimal changes that should reduce MAE noticeably toward your target band while staying within runtime and preserving your approach.'
- What this solution (achieved 2.35394) has done: 'Your MAE (2.35394, lower is better) is still far above the target (0.13659), so we should strengthen the existing fallback while keeping the same core approach: per-(R,C) closed-form linear regression with the same engineered history features and the same nearest-pressure snapping. The smallest high-impact change for this competition is to make predictions only for inspiratory timesteps (u_out==0) and then apply the competition’s evaluation behavior by setting expiratory timesteps (u_out==1, unscored) to a safe value (0) so they don’t introduce boundary artifacts from the “carry-forward” heuristic. Additionally, we keep your standardization and ridge-stabilized solve intact, but we fit the global fallback coefficients on the same (R,C)-aware design by adding R and C as standardized numeric features only in the global model (used only when a (R,C) group is missing), which improves robustness without changing the main per-(R,C) logic. These changes are minimal, deterministic, and directly metric-aligned, and should reduce MAE toward your target band.'
- What this solution (achieved 2.35307) has done: 'We need to reduce MAE from 2.35394 toward 0.13659 (lower is better), so we should strengthen the existing fallback without changing its core (per-(R,C) closed-form linear regression with engineered breath-history features + nearest-pressure snapping). The smallest high-impact, metric-aligned tweak is to fit the linear regression on inspiratory rows using **sample weights that emphasize late-inspiration** (where pressure dynamics matter most) while keeping the same model class and closed-form solve (weighted ridge normal equations). At inference, we also apply the same “inspiratory-only is scored” alignment by setting `u_out==1` predictions to the **last inspiratory prediction within the breath** (after snapping) to reduce discontinuities right at the transition; this doesn’t affect scored rows directly but typically improves stability around the boundary. These changes are deterministic, keep your architecture/approach intact, and should move MAE downward toward your target.'
- What this solution (achieved 2.35611) has done: 'Your current MAE (2.35307, lower is better) is still far above the target (0.13659), so we should improve the fallback predictions while keeping your core approach intact: per-(R,C) weighted closed-form linear regression with breath-history features and nearest-pressure snapping. The minimal high-impact fix is to stop forcing the model to also explain expiratory points by including `u_out` as a feature and fitting on **all** rows, but with **near-zero weight on u_out==1** so inspiratory dynamics still dominate exactly as the metric. This keeps the same model class/solve, but removes the distribution shift created by training only on u_out==0 while predicting all rows. We also add two tiny, deterministic “memory” features (`pressure_lag1_proxy`, `pressure_lag2_proxy`) based on lagged snapped predictions during inference only (teacher-forced with zeros during training), which preserves the linear model and feature-extraction spirit but helps capture autoregressive behavior with negligible code change.'
- What this solution (achieved 44.98609) has done: 'We need to reduce MAE from 2.35611 toward the 0.1366 target (lower is better), so the fallback must get materially stronger while keeping your same core approach (deterministic per-(R,C) closed-form linear regression + engineered history features + nearest-pressure snapping). The minimal high-impact fix is to **train the per-(R,C) linear model autoregressively in a teacher-forcing way**: build `pressure_lag1_proxy/pressure_lag2_proxy` from the *true* previous pressures during training (shifted within breath), matching what you already do at inference (lagged snapped predictions). To avoid a train/test mismatch at the inspiratory→expiratory transition, we also compute those lags from inspiratory-only pressure (set to 0 when `u_out==1`) and keep your near-zero sample weights for `u_out==1` unchanged. Finally, we keep alignment and output exactly `id,pressure` and always write `submission.csv`.'
- What this solution (achieved 44.98609) has done: 'Your current MAE (44.98609; lower is better) is far worse than the target (0.1366), and the main culprit is a bug in the fallback’s teacher-forcing lag features: it uses a very slow/fragile `groupby.apply` and (more importantly) computes lagged inspiratory pressure incorrectly, which can collapse the autoregressive signal and severely hurt predictions. I replace that with a correct, fully vectorized within-breath shift of an “inspiratory-only pressure” series, preserving your same linear closed-form model and inference-time autoregressive logic. I also keep your weighting, standardization, snapping, and blending logic unchanged, only fixing the lag-feature construction to restore the intended semantics. This is the smallest change that is directly relevant and expected to substantially reduce MAE toward the target.'
- What this solution (achieved 2.55538) has done: 'Your current MAE (44.986, lower is better) strongly suggests the fallback autoregressive lag features are still mis-specified due to a scale mismatch: you standardize `pressure_lag*_proxy` during training, but at inference you feed **raw snapped pressures** into those standardized slots, which can explode the linear model and destroy MAE. I keep your exact core approach (per-(R,C) weighted closed-form linear regression + same engineered features + same inference-time autoregressive loop + same nearest-pressure snapping), but fix this by (1) keeping and reusing the train mean/std for `pressure_lag1_proxy/pressure_lag2_proxy`, and (2) converting the lag values to the standardized scale inside the inference loop. This is a minimal, directly score-relevant change that should restore the intended teacher-forcing semantics and move MAE back down toward your target. The script still runs end-to-end and always writes a valid `submission.csv` with `id,pressure`.'

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
BASE_PATH = "/kaggle/input/ventilator-pressure-prediction"

train_path = os.path.join(BASE_PATH, "train.csv")
test_path = os.path.join(BASE_PATH, "test.csv")
sample_sub_path = os.path.join(BASE_PATH, "sample_submission.csv")

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


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    preds = []
    weights_hint = []

    for fp in input_list:
        try:
            public_lb_score = int(fp.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1  # fallback neutral weight hint

        dfp = pd.read_csv(fp)
        if "pressure" not in dfp.columns:
            continue
        arr = dfp["pressure"].to_numpy().ravel()
        preds.append(arr)
        weights_hint.append(public_lb_score)

    if len(preds) == 0:
        return None

    if len(preds) == 1:
        return preds[0]

    l_sum = sum(weights_hint) if sum(weights_hint) != 0 else 1.0
    weight1 = (weights_hint[1] / l_sum) + 0.15
    weight2 = 1 - weight1
    return preds[0] * weight1 + preds[1] * weight2


def _collect_prediction_files(dp):
    files = []
    for fp in glob.iglob(f"{dp}/*"):
        if fp.lower().endswith(".csv"):
            files.append(fp)
    files.sort()
    return files


def _load_and_validate_pred(fp, expected_len):
    try:
        dfp = pd.read_csv(fp)
    except Exception:
        return None
    if "pressure" not in dfp.columns:
        return None
    arr = dfp["pressure"].to_numpy().ravel()
    if arr.shape[0] != expected_len:
        return None
    return arr


def g(dp):
    """
    Original intent: read multiple submission CSVs from dp, blend with random weights many times,
    then take median and snap to nearest known pressure.
    Fixes:
      - handle missing/empty dp
      - ensure all predictions have correct length
      - avoid vstack of empty / wrong-shaped arrays
      - always write a valid CSV
    """
    output = pd.read_csv(sample_sub_path)
    expected_len = len(output)

    files = _collect_prediction_files(dp)
    if len(files) == 0:
        raise FileNotFoundError(f"No .csv prediction files found under: {dp}")

    file_count = len(files)
    splits = max(1, file_count // 2)
    flist = []
    for i in range(splits):
        start = i * round(len(files) / splits)
        end = None if i == splits - 1 else (i + 1) * round(len(files) / splits)
        chunk = files[start:end]
        blended = wc(chunk)
        if blended is None:
            continue
        if blended.shape[0] != expected_len:
            continue
        flist.append(blended)

    if len(flist) == 0:
        raise ValueError(
            f"All candidate prediction files under {dp} were invalid length != {expected_len}."
        )

    loop_time = 155
    pred_list = []
    for loop_idx in range(loop_time):
        set_seed(loop_idx)
        weight = [rd() for _ in range(len(flist))]
        weight_sum = sum(weight) if sum(weight) != 0 else 1.0
        weight = [w / weight_sum for w in weight]
        weight.sort(reverse=True)

        temp = np.zeros(expected_len, dtype=np.float64)
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()

    stacked = np.vstack(pred_list)  # (loop_time, expected_len)
    output["pressure"] = np.median(stacked, axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    out_name = f"rwb_{loop_time}_loops.csv"
    output.to_csv(out_name, index=False)
    return out_name


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a["pressure"] = a["pressure"] * 0.69 + b["pressure"] * 0.31
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a


def _fit_group_linear_closed_form(X, y, ridge=1e-6):
    """
    Minimal closed-form ridge regression:
      beta = (X^T X + ridge*I)^(-1) X^T y
    Using ridge only for numerical stability; does not change core approach.
    """
    XtX = X.T @ X
    XtX = XtX + ridge * np.eye(XtX.shape[0], dtype=np.float64)
    Xty = X.T @ y
    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta


def _fit_group_linear_closed_form_weighted(X, y, w, ridge=1e-6):
    """
    Score-directed but same core logic: weighted closed-form ridge regression.
    This remains a linear model solved in closed form; we just reweight samples
    to emphasize metric-relevant points.
      beta = (X^T W X + ridge*I)^(-1) X^T W y
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    w = np.clip(w, 1e-10, None)
    Xw = X * w[:, None]
    XtWX = X.T @ Xw
    XtWX = XtWX + ridge * np.eye(XtWX.shape[0], dtype=np.float64)
    XtWy = X.T @ (y * w)
    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(Xw, y * w, rcond=None)[0]
    return beta


def _add_breath_history_features(df):
    """
    Feature set stays in the same spirit (simple deterministic history features).
    """
    g = df.groupby("breath_id", sort=False)

    df["cum_u_in"] = g["u_in"].cumsum()

    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0)

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]

    dt = g["time_step"].diff().fillna(0.0)
    df["dt"] = dt
    df["u_in_dt"] = df["u_in"] * df["dt"]
    df["cum_u_in_dt"] = g["u_in_dt"].cumsum()

    u_in_roll3_sum = g["u_in"].transform(
        lambda s: s.rolling(window=3, min_periods=1).sum()
    )
    u_in_roll5_mean = g["u_in"].transform(
        lambda s: s.rolling(window=5, min_periods=1).mean()
    )
    df["u_in_roll3_sum"] = u_in_roll3_sum.astype(np.float64)
    df["u_in_roll5_mean"] = u_in_roll5_mean.astype(np.float64)

    engineered = [
        "cum_u_in",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_diff1",
        "dt",
        "u_in_dt",
        "cum_u_in_dt",
        "u_in_roll3_sum",
        "u_in_roll5_mean",
    ]
    df[engineered] = df[engineered].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return df


def _standardize_from_train(train_df, test_df, feat_cols):
    """
    Minimal, score-relevant stability improvement: standardize features using train statistics.
    This keeps the same linear model class and closed-form solution but improves conditioning.
    """
    mu = train_df[feat_cols].mean(axis=0)
    sigma = train_df[feat_cols].std(axis=0).replace(0.0, 1.0)

    train_df.loc[:, feat_cols] = (train_df[feat_cols] - mu) / sigma
    test_df.loc[:, feat_cols] = (test_df[feat_cols] - mu) / sigma

    train_df[feat_cols] = (
        train_df[feat_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    )
    test_df[feat_cols] = test_df[feat_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return train_df, test_df, mu, sigma


def make_fallback_submission(out_path="submission.csv"):
    """
    Same core: per-(R,C) weighted closed-form linear regression + nearest-pressure snapping.

    Change to fix score regression (MAE exploded):
    - BUGFIX: keep autoregressive lag features on the SAME scale used during training.
      We standardize pressure_lag* during training; therefore, at inference we must feed
      standardized lag values into Xrow, not raw snapped pressures.
    """
    train = df_train.copy()
    test = pd.read_csv(test_path)

    train = _add_breath_history_features(train)
    test = _add_breath_history_features(test)

    train["u_out_f"] = train["u_out"].astype(np.float64)
    test["u_out_f"] = test["u_out"].astype(np.float64)

    gtr = train.groupby("breath_id", sort=False)
    train["_insp_pressure"] = (
        train["pressure"].where(train["u_out"] == 0, 0.0).astype(np.float64)
    )
    train["pressure_lag1_proxy"] = gtr["_insp_pressure"].shift(1).fillna(0.0)
    train["pressure_lag2_proxy"] = gtr["_insp_pressure"].shift(2).fillna(0.0)

    test["pressure_lag1_proxy"] = 0.0
    test["pressure_lag2_proxy"] = 0.0

    feat_cols = [
        "cum_u_in",
        "cum_u_in_dt",
        "u_in",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_diff1",
        "u_in_roll3_sum",
        "u_in_roll5_mean",
        "time_step",
        "u_out_f",
        "pressure_lag1_proxy",
        "pressure_lag2_proxy",
    ]

    train, test, feat_mu, feat_sigma = _standardize_from_train(train, test, feat_cols)

    rc_mu = train[["R", "C"]].mean(axis=0)
    rc_sigma = train[["R", "C"]].std(axis=0).replace(0.0, 1.0)
    train["_Rz"] = (train["R"] - rc_mu["R"]) / rc_sigma["R"]
    train["_Cz"] = (train["C"] - rc_mu["C"]) / rc_sigma["C"]
    test["_Rz"] = (test["R"] - rc_mu["R"]) / rc_sigma["R"]
    test["_Cz"] = (test["C"] - rc_mu["C"]) / rc_sigma["C"]
    train[["_Rz", "_Cz"]] = (
        train[["_Rz", "_Cz"]].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    )
    test[["_Rz", "_Cz"]] = (
        test[["_Rz", "_Cz"]].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    )

    t = np.clip(train["time_step"].to_numpy(dtype=np.float64), 0.0, 3.0) / 3.0
    w_time = 1.0 + 1.5 * t
    w_phase = np.where(train["u_out"].to_numpy(dtype=np.int64) == 0, 1.0, 1e-3)
    w_global = w_time * w_phase

    Xg = np.column_stack(
        [np.ones(len(train), dtype=np.float64)]
        + [train[c].to_numpy(dtype=np.float64) for c in feat_cols]
        + [
            train["_Rz"].to_numpy(dtype=np.float64),
            train["_Cz"].to_numpy(dtype=np.float64),
        ]
    )
    yg = train["pressure"].to_numpy(dtype=np.float64)
    beta_g = _fit_group_linear_closed_form_weighted(Xg, yg, w_global, ridge=1e-6)

    coefs = []
    for (r, c), gdf in train.groupby(["R", "C"], sort=False):
        X = np.column_stack(
            [np.ones(len(gdf), dtype=np.float64)]
            + [gdf[col].to_numpy(dtype=np.float64) for col in feat_cols]
        )
        y = gdf["pressure"].to_numpy(dtype=np.float64)

        tg = np.clip(gdf["time_step"].to_numpy(dtype=np.float64), 0.0, 3.0) / 3.0
        w_time_g = 1.0 + 1.5 * tg
        w_phase_g = np.where(gdf["u_out"].to_numpy(dtype=np.int64) == 0, 1.0, 1e-3)
        w = w_time_g * w_phase_g

        beta = _fit_group_linear_closed_form_weighted(X, y, w, ridge=1e-6)

        row = [int(r), int(c)] + [float(v) for v in beta.tolist()]
        coefs.append(row)

    coef_cols = ["R", "C", "b0"] + [f"b_{col}" for col in feat_cols]
    coef = pd.DataFrame(coefs, columns=coef_cols)

    test_m = test.merge(coef, on=["R", "C"], how="left")

    test_m["b0"] = test_m["b0"].fillna(
        beta_g[0] + beta_g[-2] * test_m["_Rz"] + beta_g[-1] * test_m["_Cz"]
    )
    for i, col in enumerate(feat_cols, start=1):
        bcol = f"b_{col}"
        test_m[bcol] = test_m[bcol].fillna(beta_g[i])

    test_m = test_m.sort_values(
        ["breath_id", "time_step", "id"], kind="mergesort"
    ).reset_index(drop=True)
    pred = np.zeros(len(test_m), dtype=np.float64)

    idx_l1 = feat_cols.index("pressure_lag1_proxy")
    idx_l2 = feat_cols.index("pressure_lag2_proxy")

    b0 = test_m["b0"].to_numpy(dtype=np.float64)
    B = np.column_stack(
        [test_m[f"b_{c}"].to_numpy(dtype=np.float64) for c in feat_cols]
    )
    Xstatic = np.column_stack([test_m[c].to_numpy(dtype=np.float64) for c in feat_cols])

    breath_ids = test_m["breath_id"].to_numpy(dtype=np.int64)

    lag1_mu = float(feat_mu["pressure_lag1_proxy"])
    lag1_sigma = (
        float(feat_sigma["pressure_lag1_proxy"])
        if float(feat_sigma["pressure_lag1_proxy"]) != 0.0
        else 1.0
    )
    lag2_mu = float(feat_mu["pressure_lag2_proxy"])
    lag2_sigma = (
        float(feat_sigma["pressure_lag2_proxy"])
        if float(feat_sigma["pressure_lag2_proxy"]) != 0.0
        else 1.0
    )

    start = 0
    n = len(test_m)
    while start < n:
        bid = breath_ids[start]
        end = start + 1
        while end < n and breath_ids[end] == bid:
            end += 1

        last1_raw = 0.0
        last2_raw = 0.0
        for i in range(start, end):
            Xrow = Xstatic[i].copy()

            Xrow[idx_l1] = (last1_raw - lag1_mu) / lag1_sigma
            Xrow[idx_l2] = (last2_raw - lag2_mu) / lag2_sigma

            pred_i = b0[i] + float(np.dot(B[i], Xrow))
            pred[i] = pred_i

            snapped = find_nearest(pred_i)
            last2_raw = last1_raw
            last1_raw = snapped

        start = end

    sub = (
        pd.DataFrame(
            {
                "id": test_m["id"].astype(np.int64),
                "pressure": pred.astype(np.float64),
            }
        )
        .sort_values("id", kind="mergesort")
        .reset_index(drop=True)
    )

    sub["pressure"] = sub["pressure"].apply(find_nearest)
    sub.to_csv(out_path, index=False)
    return out_path




## === cell 2
try:
    submission_path = g("../input/gb-data-blending-recover")
except Exception:
    submission_path = make_fallback_submission("submission.csv")

print("Wrote submission file:", submission_path)
df_out = pd.read_csv(submission_path)
print(df_out.head())
print("Rows:", len(df_out))
print("Columns:", df_out.columns.tolist())
print("Pressure stats:", df_out["pressure"].describe())
