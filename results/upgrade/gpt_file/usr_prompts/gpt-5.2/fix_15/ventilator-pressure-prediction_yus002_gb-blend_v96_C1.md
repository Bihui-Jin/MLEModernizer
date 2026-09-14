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

0.1537099835881526

# 6. Current score

1.85934

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 14.20469) has done: 'Your notebook fails because it tries to read two external submission files (`../input/gb-data-blending-recover/...`) that are not present in this Kaggle environment. To make it run end-to-end and still produce a reasonable score, I replace that dependency with an in-notebook baseline that predicts `pressure ≈ u_in` by mapping `u_in` to the nearest allowed training pressure (this keeps your existing `find_nearest`/pressure-grid core semantics). I also fix path robustness by reading from the available `../input/ventilator-pressure-prediction/` dataset and ensure we always write a valid `submission.csv` with `id,pressure`. These changes are minimal, unblock execution, and should yield a non-trivial MAE (though likely not as good as the missing blended files).'
- What this solution (achieved 17.65486) has done: 'Your current score is far worse than the target (lower is better), so we should improve the predictions while keeping your “map continuous prediction to nearest allowed pressure grid” core logic intact. The minimal legitimate improvement is to replace the `pressure ≈ u_in` guess with a light, per-(R,C,time_step,u_out) calibration learned from train: predict pressure as an affine function of `u_in` (pressure ≈ a*u_in + b) using groupwise least squares, then fall back to global fit when a group is sparse. This keeps the same submission semantics (still snapped via `find_nearest`) and stays within pandas/numpy only, and should dramatically reduce MAE versus the naive baseline without changing any modeling “architecture”. The rest of the script (paths, find_nearest, submission writing) remains the same and still produces `submission.csv`.'
- What this solution (achieved 4.39373) has done: 'The crash comes from merging your predictions into `sample_submission.csv` without handling the duplicate `pressure` column name; pandas creates `pressure_x/pressure_y`, so `sub["pressure"]` no longer exists. I fix this by writing the submission directly from the `pred` dataframe (aligned by `id`) and add a safety fill (global mean snapped to the allowed pressure grid) in case any `id` is missing after merges. I also make the input path resolution robust to both Kaggle directory layouts you listed, without changing your modeling/calibration logic. The rest of the feature engineering, groupwise linear fit, and `find_nearest` snapping stays the same.'
- What this solution (achieved 4.39373) has done: 'Your current score (MAE 4.39373; lower is better) is far from the target 0.1537, so we should improve predictions while keeping your existing “groupwise linear fit on `u_eff` then snap to nearest allowed pressure” core logic intact. The biggest minimal gain here is to match the evaluation by explicitly handling the inspiratory/expiratory phase: predict during inspiration (`u_out==0`) with your fitted model, and during expiration (`u_out==1`, not scored) fall back to a stable per-(R,C) mean pressure learned from train, which also avoids extreme errors if Kaggle’s scoring mask differs across breaths. Second, we make the group fit more faithful by fitting on inspiratory rows only (since the label dynamics differ when `u_out==1`), while leaving the same linear regression approach. Finally, we add a tiny, safe clipping to the training pressure range before snapping (still snapped via `find_nearest`) to reduce out-of-support predictions without changing semantics.'
- What this solution (achieved 4.39373) has done: 'Your current MAE (4.39; lower is better) is still far from the target (0.1537), so the most direct minimal improvement is to better match the evaluation rule: only inspiratory timesteps (u_out==0) are scored, so we should set expiratory predictions (u_out==1) to a safe constant (0) instead of trying to fit them, which can only add noise and worsen MAE if any leakage/mismatch occurs. Next, we keep your exact groupwise linear-on-u_eff core logic but make it more faithful by fitting separate models for each (R,C) and inspiratory time bucket without including u_out in the group key (since we already filter to u_out==0), improving coverage and reducing unnecessary sparsity fallbacks. Finally, we ensure submission alignment stays correct and still snap to the allowed pressure grid exactly as before.'
- What this solution (achieved 4.39373) has done: 'Your current MAE (4.39; lower is better) is far above the target (0.1537), so we should improve predictions while keeping your existing “groupwise linear fit on `u_eff` then snap to nearest allowed pressure grid” logic intact. The biggest minimal gain is to stop forcing expiratory (`u_out==1`) predictions to 0: even though expiration is typically not scored, the competition metric masks by inspiratory phase per breath, and setting all expiratory points to 0 can still hurt if any scoring rows include them; instead we predict them with a stable per-(R,C,ts) mean pressure learned from train, with a per-(R,C) and global fallback. We also align training/estimation with that plan by learning the mean targets from `u_out==1` rows only (since their distribution differs), while leaving the inspiratory linear regression unchanged. Submission writing stays the same and still produces a valid `submission.csv`.'
- What this solution (achieved 4.39373) has done: 'Your current MAE (4.39373; lower is better) is still far above the target, so we need a larger but still core-logic-preserving improvement. I keep your exact “groupwise linear regression on u_eff then snap to nearest allowed pressure” approach, but make it consistent with the competition’s discrete pressure levels by fitting the regression on *snapped* training targets (pressure rounded to the nearest allowed grid) rather than raw float pressures. I also add a minimal per-(R,C) bias correction computed on training inspiratory rows after snapping, and apply that bias at inference before the final snap; this is a small calibration layer that doesn’t change the modeling family. Finally, I keep your expiratory handling but compute expiratory means on snapped pressures too for consistency and slightly improved stability.'
- What this solution (achieved 2.47085) has done: 'Your current MAE (4.39; lower is better) is still far from the target (0.1537), so we need a modest-but-material modeling correction while keeping your exact “groupwise linear regression on `u_eff` + snapping to allowed pressure grid” core logic. The minimal high-impact issue is that the relationship between control input and pressure is strongly breath-state dependent; adding one extra state feature (`u_in` cumulative integral within the breath) and using it as a **small additive correction** (still linear-in-features, still groupwise, still snapped) typically improves a lot without changing the training loop/architecture. Concretely, we fit an additional groupwise linear term `pressure ≈ a*u_eff + b + k*u_in_cum` on inspiratory rows, with the same fine→coarse→global fallback logic and the same bias correction and expiratory mean handling. Finally, we keep all I/O and submission alignment identical and still write `submission.csv` with `id,pressure`.'
- What this solution (achieved 2.14623) has done: 'Your current MAE (2.47085; lower is better) is still far above the target, so we need a small but high-impact fix that stays within your existing “groupwise linear regression + snapping to allowed pressure grid” core logic. The minimal issue is that your `u_in_cum` feature is unscaled and implicitly tied to sampling rate; making it a physically meaningful integral (`u_in * Δt`) stabilizes the linear fit and usually reduces error without changing the model family. I keep all grouping, fallbacks, bias correction, expiratory handling, and final snapping identical, only replacing the cumulative-sum definition in both train and test with a time-weighted integral. This should move the score materially toward the target while preserving your approach and producing the same valid `submission.csv`.'
- What this solution (achieved 2.14623) has done: 'Your current MAE (2.146) is still far above the target (0.154), so we need a small but meaningful accuracy improvement without changing your core “groupwise linear regression + snapping to allowed pressure grid” approach. The main minimal fix here is to make the regression coefficients more numerically stable and better matched to the data by (1) using a tiny ridge term in the normal equations (same linear model, same features, just stabilized solve) and (2) centering `u_eff` and `u_in_cum` within each fitted group to reduce collinearity and improve the fitted intercept. This keeps the exact same model family and inference semantics (still linear, still groupwise with the same fallbacks, still clipped and snapped), but typically reduces error materially. All I/O paths and the submission writing remain unchanged and still produce a valid `submission.csv`.'
- What this solution (achieved 2.17095) has done: 'Your current MAE (2.146) is still far above the target (0.154), so we should improve predictions while keeping your exact “groupwise linear regression on `u_eff` and `u_in_cum` + bias correction + expiratory mean fallback + snapping to allowed pressure grid” logic intact. The smallest high-impact change is to use the *physically consistent integral* for `u_in_cum`: integrate the effective flow proxy `u_eff` over time (not raw `u_in`), which better matches your model’s main predictor and usually reduces residual structure without changing the model family. I keep the same grouping/fallback scheme and ridge-stabilized solve; I only adjust how `u_in_cum` is computed in both train and test (and keep all I/O/submission logic unchanged). This should move the score materially downward toward the target while preserving the approach and producing a valid `submission.csv`.'
- What this solution (achieved 1.92762) has done: 'Your current MAE (2.17095; lower is better) is still far above the target (0.1537), so we should improve prediction accuracy with the smallest changes that keep your exact “groupwise linear regression on `u_eff` and `u_in_cum` + bias correction + expiratory mean fallback + snapping” core logic intact. The most direct minimal gain is to add the standard ventilator state features that are still linear-friendly and don’t change the modeling family: cumulative volume proxy (`area = ∫ u_in dt`) and a simple flow-change term (`delta_u_in = u_in - lag1`). We then fit the same kind of ridge-stabilized, groupwise linear regression but with these extra linear features (no new training loops, no new model class), and keep your same fine→coarse→global fallback, bias correction, expiratory mean handling, clipping, and snapping. This typically reduces MAE materially on this competition because pressure depends strongly on integrated flow/volume and flow transients, and it should move the score closer to the target while staying within your constraints. The script still runs end-to-end and writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 1.85934) has done: 'Your current MAE (1.9276, lower is better) is still far above the target (0.1537), so we should improve accuracy while keeping your exact core approach (groupwise ridge-stabilized linear regression + expiratory mean fallback + clipping + snap-to-pressure-grid). The biggest minimal gain is to make the “state” features more faithful by adding two standard linear-friendly breath-history terms: a lag of `u_out` and an “inspiratory time within breath” accumulator (resets on expiration), then include them as additional linear regressors with the same fitting/fallback logic. This keeps the same training approach (closed-form linear fit), does not change any loops/architecture, and typically reduces residual error on this competition because pressure dynamics depend on inspiratory phase progression and valve transitions. Submission writing and alignment stay identical and still produce a valid `submission.csv`.'

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
CANDIDATE_BASES = [
    "../input/ventilator-pressure-prediction",
    "/kaggle/input/ventilator-pressure-prediction",
    "/kaggle/data/ventilator-pressure-prediction",
    "../data/ventilator-pressure-prediction",
    "/kaggle/input",
    "/kaggle/data",
    "../input",
    "../data",
]


def resolve_path(filename: str) -> str:
    for base in CANDIDATE_BASES:
        p = os.path.join(base, filename)
        if os.path.exists(p):
            return p
    return os.path.join("../input/ventilator-pressure-prediction", filename)


TRAIN_PATH = resolve_path("train.csv")
TEST_PATH = resolve_path("test.csv")
SAMPLE_PATH = resolve_path("sample_submission.csv")

df_train = pd.read_csv(TRAIN_PATH)

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)

P_MIN = float(sorted_pressures[0])
P_MAX = float(sorted_pressures[-1])


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
        weight1 = (l[1] / l_sum) + 0.01
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
df_test = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_PATH)

train = df_train[
    ["breath_id", "R", "C", "time_step", "u_out", "u_in", "pressure"]
].copy()
train.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")

train["u_in_lag1"] = train.groupby("breath_id", sort=False)["u_in"].shift(1).fillna(0.0)

train["u_out_lag1"] = (
    train.groupby("breath_id", sort=False)["u_out"].shift(1).fillna(0).astype(np.int8)
)

train["u_eff"] = train["u_in"].astype(np.float64) + 0.5 * train["u_in_lag1"].astype(
    np.float64
)
train["ts"] = train["time_step"].round(2).astype(np.float32)

train["dt"] = (
    train.groupby("breath_id", sort=False)["time_step"]
    .diff()
    .fillna(0.0)
    .astype(np.float64)
)

train["area"] = (
    (train["u_in"].astype(np.float64) * train["dt"])
    .groupby(train["breath_id"], sort=False)
    .cumsum()
)
train["delta_u_in"] = train["u_in"].astype(np.float64) - train["u_in_lag1"].astype(
    np.float64
)

train["u_in_cum"] = (
    (train["u_eff"].astype(np.float64) * train["dt"])
    .groupby(train["breath_id"], sort=False)
    .cumsum()
)

train["dt_insp"] = train["dt"].where(train["u_out"].eq(0), 0.0).astype(np.float64)
train["t_insp"] = train["dt_insp"].groupby(train["breath_id"], sort=False).cumsum()

train["pressure_snapped"] = train["pressure"].apply(find_nearest).astype(np.float64)


def fit_group_linear_features(
    df,
    feat_cols=("u_eff", "u_in_cum", "area", "delta_u_in", "t_insp", "u_out_lag1"),
    y_col="pressure_snapped",
    ridge=1e-6,
):
    y = df[y_col].to_numpy(dtype=np.float64)
    n = y.size
    p = len(feat_cols)
    if n < (p + 1):
        out = {f"w{i}": np.nan for i in range(p)}
        out.update({"b": np.nan, "n": n})
        for i in range(p):
            out[f"xm{i}"] = np.nan
        return pd.Series(out)

    X_raw = np.vstack([df[c].to_numpy(dtype=np.float64) for c in feat_cols]).T  # (n,p)
    xm = X_raw.mean(axis=0)
    Xc = X_raw - xm

    X = np.hstack([Xc, np.ones((n, 1), dtype=np.float64)])  # (n,p+1)
    XtX = X.T @ X
    reg = np.zeros((p + 1, p + 1), dtype=np.float64)
    reg[np.arange(p), np.arange(p)] = ridge  # ridge on feature weights only
    Xty = X.T @ y

    try:
        beta = np.linalg.solve(XtX + reg, Xty)  # (p+1,)
        w = beta[:p]
        b0 = float(beta[p])
    except np.linalg.LinAlgError:
        x = Xc[:, 0]
        x_mean = x.mean()
        y_mean = y.mean()
        denom = np.sum((x - x_mean) ** 2)
        w = np.zeros(p, dtype=np.float64)
        if denom <= 1e-12:
            w[0] = 0.0
            b0 = float(y_mean)
        else:
            w[0] = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
            b0 = float(y_mean - w[0] * x_mean)

    b = float(b0 - float(np.dot(w, xm)))

    out = {f"w{i}": float(wi) for i, wi in enumerate(w)}
    out.update({"b": b, "n": int(n)})
    for i in range(p):
        out[f"xm{i}"] = float(xm[i])
    return pd.Series(out)


train_insp = train[train["u_out"] == 0].copy()

grp_cols_fine = ["R", "C", "ts"]
coef_fine = (
    train_insp.groupby(grp_cols_fine, sort=False)
    .apply(lambda d: fit_group_linear_features(d))
    .reset_index()
)

grp_cols_coarse = ["R", "C"]
coef_coarse = (
    train_insp.groupby(grp_cols_coarse, sort=False)
    .apply(lambda d: fit_group_linear_features(d))
    .reset_index()
)

global_coef = fit_group_linear_features(train_insp)
global_w = [
    float(global_coef[f"w{i}"]) if pd.notna(global_coef[f"w{i}"]) else 0.0
    for i in range(6)
]
global_b = (
    float(global_coef["b"])
    if pd.notna(global_coef["b"])
    else float(train_insp["pressure_snapped"].mean())
)

global_mean_pressure = float(train["pressure_snapped"].mean())

train_insp_fit = train_insp.merge(
    coef_coarse, on=grp_cols_coarse, how="left", suffixes=("", "_rc")
)
for i in range(6):
    train_insp_fit[f"w{i}"] = train_insp_fit[f"w{i}"].fillna(global_w[i])
train_insp_fit["b"] = train_insp_fit["b"].fillna(global_b)

X_pred = (
    train_insp_fit["w0"] * train_insp_fit["u_eff"].astype(np.float64)
    + train_insp_fit["w1"] * train_insp_fit["u_in_cum"].astype(np.float64)
    + train_insp_fit["w2"] * train_insp_fit["area"].astype(np.float64)
    + train_insp_fit["w3"] * train_insp_fit["delta_u_in"].astype(np.float64)
    + train_insp_fit["w4"] * train_insp_fit["t_insp"].astype(np.float64)
    + train_insp_fit["w5"] * train_insp_fit["u_out_lag1"].astype(np.float64)
    + train_insp_fit["b"].astype(np.float64)
)
train_insp_fit["pred_fit"] = X_pred
train_insp_fit["resid"] = (
    train_insp_fit["pressure_snapped"] - train_insp_fit["pred_fit"]
)

bias_rc = (
    train_insp_fit.groupby(grp_cols_coarse, sort=False)["resid"]
    .mean()
    .reset_index()
    .rename(columns={"resid": "bias_rc"})
)
global_bias = float(train_insp_fit["resid"].mean()) if len(train_insp_fit) else 0.0

train_exp = train[train["u_out"] == 1].copy()
exp_mean_fine = (
    train_exp.groupby(grp_cols_fine, sort=False)["pressure_snapped"]
    .mean()
    .reset_index()
    .rename(columns={"pressure_snapped": "p_exp_mean"})
)
exp_mean_coarse = (
    train_exp.groupby(grp_cols_coarse, sort=False)["pressure_snapped"]
    .mean()
    .reset_index()
    .rename(columns={"pressure_snapped": "p_exp_mean_coarse"})
)
global_exp_mean = (
    float(train_exp["pressure_snapped"].mean())
    if len(train_exp)
    else global_mean_pressure
)

test = df_test[["id", "breath_id", "R", "C", "time_step", "u_out", "u_in"]].copy()
test.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")
test["u_in_lag1"] = test.groupby("breath_id", sort=False)["u_in"].shift(1).fillna(0.0)

test["u_out_lag1"] = (
    test.groupby("breath_id", sort=False)["u_out"].shift(1).fillna(0).astype(np.int8)
)

test["u_eff"] = test["u_in"].astype(np.float64) + 0.5 * test["u_in_lag1"].astype(
    np.float64
)
test["ts"] = test["time_step"].round(2).astype(np.float32)

test["dt"] = (
    test.groupby("breath_id", sort=False)["time_step"]
    .diff()
    .fillna(0.0)
    .astype(np.float64)
)

test["area"] = (
    (test["u_in"].astype(np.float64) * test["dt"])
    .groupby(test["breath_id"], sort=False)
    .cumsum()
)
test["delta_u_in"] = test["u_in"].astype(np.float64) - test["u_in_lag1"].astype(
    np.float64
)

test["u_in_cum"] = (
    (test["u_eff"].astype(np.float64) * test["dt"])
    .groupby(test["breath_id"], sort=False)
    .cumsum()
)

test["dt_insp"] = test["dt"].where(test["u_out"].eq(0), 0.0).astype(np.float64)
test["t_insp"] = test["dt_insp"].groupby(test["breath_id"], sort=False).cumsum()

test = test.merge(coef_fine, on=grp_cols_fine, how="left")

need_coarse = (
    test["b"].isna()
    | test["w0"].isna()
    | test["w1"].isna()
    | test["w2"].isna()
    | test["w3"].isna()
    | test["w4"].isna()
    | test["w5"].isna()
    | (test["n"].fillna(0) < 30)
)
if need_coarse.any():
    tmp_rows = test.loc[need_coarse, ["R", "C"]].copy()
    tmp_rows["__row"] = tmp_rows.index.to_numpy()
    tmp_rows = tmp_rows.merge(
        coef_coarse, on=grp_cols_coarse, how="left", suffixes=("", "_coarse")
    )
    tmp_rows.set_index("__row", inplace=True)
    for col in ["w0", "w1", "w2", "w3", "w4", "w5", "b", "n"]:
        test.loc[need_coarse, col] = tmp_rows[col].to_numpy()

for i in range(6):
    test[f"w{i}"] = test[f"w{i}"].fillna(global_w[i])
test["b"] = test["b"].fillna(global_b)

pred = test[
    [
        "id",
        "R",
        "C",
        "ts",
        "u_out",
        "u_eff",
        "u_in_cum",
        "area",
        "delta_u_in",
        "t_insp",
        "u_out_lag1",
        "w0",
        "w1",
        "w2",
        "w3",
        "w4",
        "w5",
        "b",
    ]
].copy()

pred["pressure"] = (
    pred["w0"].astype(np.float64) * pred["u_eff"].astype(np.float64)
    + pred["w1"].astype(np.float64) * pred["u_in_cum"].astype(np.float64)
    + pred["w2"].astype(np.float64) * pred["area"].astype(np.float64)
    + pred["w3"].astype(np.float64) * pred["delta_u_in"].astype(np.float64)
    + pred["w4"].astype(np.float64) * pred["t_insp"].astype(np.float64)
    + pred["w5"].astype(np.float64) * pred["u_out_lag1"].astype(np.float64)
    + pred["b"].astype(np.float64)
)

pred = pred.merge(bias_rc, on=grp_cols_coarse, how="left")
pred["bias_rc"] = pred["bias_rc"].fillna(global_bias)
pred["pressure"] = pred["pressure"] + pred["bias_rc"].astype(np.float64)

pred = pred.merge(exp_mean_fine, on=grp_cols_fine, how="left")
need_exp_coarse = pred["p_exp_mean"].isna()
if need_exp_coarse.any():
    tmp_exp = pred.loc[need_exp_coarse, ["R", "C"]].copy()
    tmp_exp["__row"] = tmp_exp.index.to_numpy()
    tmp_exp = tmp_exp.merge(exp_mean_coarse, on=grp_cols_coarse, how="left")
    tmp_exp.set_index("__row", inplace=True)
    pred.loc[need_exp_coarse, "p_exp_mean"] = tmp_exp["p_exp_mean_coarse"].to_numpy()

pred["p_exp_mean"] = pred["p_exp_mean"].fillna(global_exp_mean)

exp_mask = pred["u_out"].to_numpy() == 1
if exp_mask.any():
    pred.loc[exp_mask, "pressure"] = pred.loc[exp_mask, "p_exp_mean"].to_numpy()

pred["pressure"] = pred["pressure"].clip(P_MIN, P_MAX)
pred["pressure"] = pred["pressure"].apply(find_nearest)

sub = pred[["id", "pressure"]].copy()
sub = sample_sub[["id"]].merge(sub, on="id", how="left")

default_pressure = find_nearest(
    global_b if np.isfinite(global_b) else global_mean_pressure
)
sub["pressure"] = sub["pressure"].fillna(default_pressure).astype(np.float64)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print(
    "Paths:",
    {"TRAIN_PATH": TRAIN_PATH, "TEST_PATH": TEST_PATH, "SAMPLE_PATH": SAMPLE_PATH},
)
