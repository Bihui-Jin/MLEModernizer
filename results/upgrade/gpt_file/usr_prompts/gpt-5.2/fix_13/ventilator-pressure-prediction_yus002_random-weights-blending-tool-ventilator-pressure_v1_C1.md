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

1.773636033397321

# 6. Current score

2.521

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.31337) has done: 'The crash happens because `../input/gb-pred-files` doesn’t exist in this environment, so no prediction files are found and `pred_list` stays empty, causing `np.vstack` to fail. I make the code robust by (1) searching for a valid folder that actually contains `.csv` prediction files (if any), and (2) providing a deterministic, simple fallback baseline prediction (group-mean by `R,C,time_step,u_in,u_out`) so a valid submission is always produced. I also ensure the script writes a `.csv` file with the required `id,pressure` columns. These changes keep the “ensemble and snap to nearest pressure” core intent, while preventing runtime errors and yielding a reasonable MAE baseline toward your target.'
- What this solution (achieved 4.67912) has done: 'Your current fallback baseline (group-mean on exact `time_step` + `u_in`) is too sparse, so most test rows miss and default to the global mean, giving a weak MAE (8.31). To move the score down toward your 1.77 target while preserving the same overall “baseline fallback + snap-to-nearest pressure” logic, I keep the structure but improve the fallback features minimally: (1) add standard ventilator feature engineering (cumulative `u_in`, lag of `u_in/u_out`, and breath-wise `area`), and (2) switch the fallback from a sparse group-mean to a simple sklearn `LinearRegression` trained on inspiratory points only (matching the metric) and predicting all timesteps (expiratory isn’t scored anyway). I keep the nearest-pressure snapping exactly as you do now and still write a valid `id,pressure` submission CSV. This is a small, deterministic change confined to the fallback path and should substantially reduce MAE toward the target without changing the ensemble path if prediction files exist.'
- What this solution (achieved 4.86367) has done: 'Your current fallback model is a plain `LinearRegression`, which tends to underfit this task and is the main reason the MAE is still far from the 1.77 target. To move the score downward toward the target while preserving the same overall pipeline (ensemble if pred-files exist, otherwise deterministic fallback + nearest-pressure snapping), I keep your exact feature set and snapping logic but swap only the fallback regressor to a stronger yet still simple and deterministic `Ridge` (same training approach, no early stopping). I also fix a small inefficiency/bug in the `area` feature computation (your current groupby-apply formulation is not the intended cumulative trapezoid-like area and is slower), replacing it with the standard ventilator “area” definition that is consistent and commonly used. Everything else (I/O paths, submission schema, ensemble path) remains unchanged.'
- What this solution (achieved 4.86367) has done: 'Your current gap to the target is large (MAE 4.86 vs 1.77, lower is better), so we should improve the fallback model while keeping the same overall pipeline (use prediction files if present; otherwise train a deterministic sklearn regressor and snap to the nearest known pressure). The minimal high-impact fix is to keep your exact feature set and training on inspiratory points only, but make the Ridge model properly conditioned by standardizing features (Ridge is sensitive to feature scale here, especially with `u_in_cum/area`). This preserves the same core semantics (linear model + snapping), but typically reduces MAE substantially on this competition without changing loops/architecture. I also keep paths and submission schema identical and ensure the output is always `id,pressure`.'
- What this solution (achieved 4.64035) has done: 'We keep your exact pipeline (search for prediction files; otherwise deterministic fallback; then snap to nearest known pressure) and only strengthen the fallback in a minimal way since your current MAE (4.86367) is still far above the target (1.77, lower is better). The smallest high-impact change is to keep the same linear model family but add a couple of well-known, cheap ventilator features (breath-wise `u_in` standardization and a few more lags) and tune Ridge’s alpha slightly; this preserves the same training semantics (single fit, no CV, no early stopping) while typically reducing MAE materially. We also ensure the submission always matches `sample_submission.csv` row order by writing predictions into that template (already done), and keep the ensemble path unchanged. All changes are confined to the fallback path and feature engineering, so if valid external prediction CSVs exist, your original ensemble behavior remains intact.'
- What this solution (achieved 17.65244) has done: 'Your current score (4.64035 MAE; lower is better) is still far from the target (1.7736), so we should improve the fallback path (used when no external prediction CSVs are found) with minimal, metric-aligned changes. I keep the same overall pipeline (ensemble-if-available else deterministic sklearn regressor + snap-to-nearest pressure), but make the fallback model slightly more expressive by adding interaction-only polynomial features on top of your existing engineered features (still a linear model at the end, same single-fit training approach). This tends to reduce MAE on this competition without changing the “core semantics” of the solution (no CV, no early stopping, no new training loops). I also restrict the pred-file search to only CSVs that match the sample submission schema (id+pressure), preventing accidental ingestion of train/test CSVs that would silently break quality.'
- What this solution (achieved 17.65244) has done: 'Your score (17.65 MAE) is far above the target (1.77, lower is better), and the most likely cause is that your ensemble path is accidentally ingesting non-prediction CSVs (or mixing file schemas), producing junk predictions, while the fallback baseline is never used or is bypassed. I make the pred-file discovery stricter and deterministic: only accept CSVs that exactly match the sample submission `id` ordering and numeric `pressure`, and ignore everything else, so the fallback Ridge+features path is used when no genuine pred files exist. I also ensure the final submission always aligns by `id` (merge onto the sample submission template) to prevent silent row-order mismatches that can explode MAE. These are minimal changes that preserve your core logic (ensemble-if-available else fallback + nearest-pressure snapping) while fixing the likely evaluation-killer.'
- What this solution (achieved 2.70804) has done: 'Your current MAE (17.65, lower is better) is far from the target (1.77), and the most likely reason is that the “ensemble” path is still being triggered by accidentally finding unrelated CSVs (or weak/garbage prediction CSVs), which then dominate over the much better fallback model. To move the score down toward the target with minimal change, I make prediction-file discovery *strict and local*: only use the user-provided folder if it exists and contains valid `id,pressure` submissions; otherwise always use the fallback Ridge pipeline. I also simplify the ensemble weighting loop (keeping the same median-ensemble semantics) so it remains deterministic and doesn’t explode runtime, while still snapping to the nearest known pressure and writing a valid `submission.csv`.'
- What this solution (achieved 2.70807) has done: 'Your current score (2.70804 MAE) is still above the target (1.7736, lower is better), so we should improve the fallback path (used when no external prediction files are present) with the smallest, metric-aligned change. The easiest win without changing the overall approach is to (1) train on inspiratory points (as you already do) but (2) **predict only inspiratory points and set expiratory predictions to a safe constant** (since expiratory isn’t scored), and (3) add a tiny, deterministic post-calibration on inspiratory predictions to reduce systematic bias before snapping to the nearest known pressure. This keeps your exact model class (scaled polynomial interactions + Ridge) and keeps the “snap-to-known-pressure” semantics, but typically reduces MAE materially on this competition. The ensemble path is left unchanged.'
- What this solution (achieved 2.68941) has done: 'We keep your pipeline exactly the same (ensemble if valid `id,pressure` files exist; otherwise use the deterministic fallback Ridge + snapping), and only make a small, metric-aligned improvement in the fallback path to move MAE down toward your 1.77 target. Specifically, we (1) add a couple of very cheap, standard ventilator features (`u_in` lag4 and cumulative `u_out`) without changing the model family/training loop, and (2) extend the post-calibration from a single affine mapping to a slightly more flexible quadratic calibration on inspiratory predictions only (still deterministic, no CV). These changes preserve your core semantics (single fit, same regressor type, same snapping), and should reduce systematic residual error enough to shrink the gap from 2.708 toward 1.77. The ensemble path, file discovery, I/O paths, and submission format remain unchanged.'
- What this solution (achieved 2.5317) has done: 'Your current MAE (2.689) is still above the target (1.774), so we should make a small improvement that is very likely to reduce error without changing your overall pipeline (ensemble-if-available else fallback Ridge+features+snapping). The most effective minimal tweak here is to make the fallback model more robust by using `RidgeCV` over a small fixed alpha grid (still a single deterministic fit, no early stopping, same linear family) and to add a tiny set of standard, low-risk extra features (one more time lag and simple interactions like `u_in*u_out`) that often reduce residual bias in this competition. We keep the inspiratory-only training, the inspiratory-only prediction + constant expiratory fill, and the “snap to nearest known pressure” semantics exactly the same. All I/O paths and the submission schema remain unchanged, and the ensemble path is untouched.'
- What this solution (achieved 2.521) has done: 'Your current MAE (2.5317, lower is better) is still above the target (1.7736), so we should make a small improvement that’s very likely to reduce error while keeping your overall pipeline intact (ensemble-if-valid-files-else fallback + snap-to-known-pressure). The minimal, metric-aligned change is to **train the fallback RidgeCV with sample weights that ignore expiratory points and emphasize early inspiratory timesteps**, which better matches the competition MAE (inspiratory only) without changing the model family or adding new training loops. I keep your feature set, polynomial interaction transform, quadratic calibration, and snapping logic the same, and I leave the ensemble path unchanged. This should move the score downward toward the target with low risk and still produce a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import copy
import glob
import random
from random import random as rd



## === cell 1
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SAMPLE_SUB_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"

df_train = pd.read_csv(TRAIN_PATH)
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
    arrs = []
    for path in input_list:
        base = os.path.basename(path)
        try:
            public_lb_score = int(base.split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1
        l.append(public_lb_score)
        arrs.append(pd.read_csv(path)["pressure"].to_numpy().ravel())

    l_sum = sum(l) if sum(l) != 0 else len(l)
    if len(arrs) == 1:
        return arrs[0]

    weight1 = (l[1] / l_sum) + 0.1
    weight2 = (1 - weight1) - 0.1
    return arrs[0] * weight1 + arrs[1] * weight2


def _is_valid_pred_csv(path, sample_df):
    try:
        dfp = pd.read_csv(path)
    except Exception:
        return False

    if list(dfp.columns) != ["id", "pressure"]:
        return False
    if len(dfp) != len(sample_df):
        return False

    try:
        if not np.array_equal(dfp["id"].to_numpy(), sample_df["id"].to_numpy()):
            return False
    except Exception:
        return False

    if not pd.api.types.is_numeric_dtype(dfp["pressure"]):
        return False
    p = pd.to_numeric(dfp["pressure"], errors="coerce").to_numpy()
    if not np.isfinite(p).all():
        return False

    return True


def _find_pred_folder(preferred_dp):
    sample = pd.read_csv(SAMPLE_SUB_PATH)

    if preferred_dp and os.path.isdir(preferred_dp):
        try:
            for p in glob.glob(os.path.join(preferred_dp, "*.csv")):
                if _is_valid_pred_csv(p, sample):
                    return preferred_dp
        except Exception:
            pass
    return None


def _add_features(df):
    df = df.copy()
    df["RC"] = df["R"] * df["C"]

    g = df.groupby("breath_id", sort=False)

    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0)
    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0.0)
    df["u_out_lag2"] = g["u_out"].shift(2).fillna(0.0)

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_in_diff2"] = df["u_in_lag1"] - df["u_in_lag2"]

    df["u_in_cum"] = g["u_in"].cumsum()

    dt = g["time_step"].diff().fillna(0.0)
    df["area"] = (df["u_in"] * dt).groupby(df["breath_id"], sort=False).cumsum()

    u_in_mean = g["u_in"].transform("mean")
    u_in_std = g["u_in"].transform("std").fillna(0.0)
    df["u_in_z"] = (df["u_in"] - u_in_mean) / (u_in_std.replace(0.0, 1.0))

    df["t_idx"] = g.cumcount().astype(np.float32)

    df["u_in_lag3"] = g["u_in"].shift(3).fillna(0.0)
    df["u_out_lag3"] = g["u_out"].shift(3).fillna(0.0)
    df["u_in_diff3"] = df["u_in_lag2"] - df["u_in_lag3"]

    df["u_in_lag4"] = g["u_in"].shift(4).fillna(0.0)
    df["u_out_cum"] = g["u_out"].cumsum().astype(np.float32)

    df["u_in_lag5"] = g["u_in"].shift(5).fillna(0.0)
    df["u_in_u_out"] = (df["u_in"] * df["u_out"]).astype(np.float32)
    df["u_in_over_C"] = (df["u_in"] / df["C"]).astype(np.float32)

    return df


def _fallback_baseline_submission(out_path="submission.csv"):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.linear_model import RidgeCV

    df_test = pd.read_csv(TEST_PATH)
    sub = pd.read_csv(SAMPLE_SUB_PATH)

    tr = _add_features(df_train)
    te = _add_features(df_test)

    feat_cols = [
        "R",
        "C",
        "RC",
        "time_step",
        "t_idx",
        "u_in",
        "u_in_z",
        "u_out",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_lag3",
        "u_in_lag4",
        "u_in_lag5",
        "u_out_lag1",
        "u_out_lag2",
        "u_out_lag3",
        "u_in_diff1",
        "u_in_diff2",
        "u_in_diff3",
        "u_in_cum",
        "u_out_cum",
        "area",
        "u_in_u_out",
        "u_in_over_C",
    ]

    tr_insp = tr[tr["u_out"] == 0].copy()

    X_tr = tr_insp[feat_cols].to_numpy(dtype=np.float32, copy=False)
    y_tr = tr_insp["pressure"].to_numpy(dtype=np.float32, copy=False)

    t_idx = tr_insp["t_idx"].to_numpy(dtype=np.float32, copy=False)
    sample_weight = (1.0 / (1.0 + 0.03 * t_idx)).astype(np.float32)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "poly",
                PolynomialFeatures(degree=2, include_bias=False, interaction_only=True),
            ),
            ("ridge", RidgeCV(alphas=(0.5, 1.0, 2.0, 4.0), cv=3)),
        ]
    )
    model.fit(X_tr, y_tr, ridge__sample_weight=sample_weight)

    te_mask_insp = te["u_out"].to_numpy() == 0
    pred = np.zeros(len(te), dtype=np.float32)

    if te_mask_insp.any():
        X_te_insp = te.loc[te_mask_insp, feat_cols].to_numpy(
            dtype=np.float32, copy=False
        )
        pred_insp = model.predict(X_te_insp).astype(np.float32)

        tr_pred_insp = model.predict(X_tr).astype(np.float32)
        x = tr_pred_insp.astype(np.float64, copy=False)
        y = y_tr.astype(np.float64, copy=False)

        Xcal = np.stack([x * x, x, np.ones_like(x)], axis=1)
        try:
            coef, _, _, _ = np.linalg.lstsq(Xcal, y, rcond=None)
            a, b, c = coef.tolist()
        except Exception:
            a, b, c = 0.0, 1.0, 0.0

        pred_insp = (
            a * (pred_insp.astype(np.float64) ** 2)
            + b * pred_insp.astype(np.float64)
            + c
        ).astype(np.float32)
        pred[te_mask_insp] = pred_insp

    exp_const = np.float32(find_nearest(0.0))
    pred[~te_mask_insp] = exp_const

    pred = np.array([find_nearest(float(x)) for x in pred], dtype=np.float32)

    sub["pressure"] = pred
    sub[["id", "pressure"]].to_csv(out_path, index=False)
    return out_path


def g(dp):
    resolved_dp = _find_pred_folder(dp)

    if resolved_dp is None:
        return _fallback_baseline_submission(out_path="submission.csv")

    sample = pd.read_csv(SAMPLE_SUB_PATH)

    l = [p for p in glob.glob(f"{resolved_dp}/*.csv")]
    l.sort()
    good = [p for p in l if _is_valid_pred_csv(p, sample)]

    if len(good) == 0:
        return _fallback_baseline_submission(out_path="submission.csv")

    l = good
    file_count = len(l)

    loop_time = min(25, max(5, file_count * 2))

    splits = max(1, file_count // 2)

    flist = []
    chunk = round(len(l) / splits) if splits > 0 else len(l)
    for i in range(splits):
        start = i * chunk
        end = None if i == splits - 1 else (i + 1) * chunk
        group = l[start:end]
        if len(group) > 0:
            flist.append(group)

    flist = [wc(group) for group in flist if len(group) > 0]
    if len(flist) == 0:
        return _fallback_baseline_submission(out_path="submission.csv")

    pred_list = []
    for seed in range(loop_time):
        weight = []
        set_seed(seed)
        for _ in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight) if sum(weight) != 0 else len(weight)
        weight = [w / weight_sum for w in weight]
        weight.sort(reverse=True)

        temp = 0.0
        for i in range(len(flist)):
            temp += flist[i] * weight[i]
        pred_list.append(temp)

    output = sample.copy()
    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)

    out_path = "submission.csv"
    output[["id", "pressure"]].to_csv(out_path, index=False)
    return out_path




## === cell 2
submission_path = g("../input/gb-pred-files")
print("Wrote submission to:", submission_path)
print(pd.read_csv(submission_path).head())
