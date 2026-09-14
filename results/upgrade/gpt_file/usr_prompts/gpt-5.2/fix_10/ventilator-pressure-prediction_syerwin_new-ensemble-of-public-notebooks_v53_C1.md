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

0.1436976763936639

# 6. Current score

3.07625

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'Your notebook fails because it tries to read four external Kaggle Dataset submission files that are not present in this environment, so `sub_1`…`sub_4` never get created and the blend crashes. To keep the original “blend multiple submissions” core logic while making it runnable end-to-end, I (1) load `sample_submission.csv` as the base, (2) attempt to load the four external submissions but gracefully fall back to using the base submission when they’re missing, and (3) robustly align on `id` before blending so the output is always valid. This generate a correct `submission.csv` with the required columns; since the fallback is the all-zero sample submission, the score likely won’t reach your target unless those external files are available.'
- What this solution (achieved 6.10613) has done: 'Your current score is far worse than the target because the ensemble inputs are missing, so you’re effectively submitting the all-zero sample submission. To move the score much closer to the target while preserving the “blend multiple predictions” core idea, I keep your blending code but add a lightweight, local “fallback model” that generates a reasonable pressure estimate from the provided train/test (no external datasets). Specifically, when an external submission file is missing, we replace that missing component with an in-notebook baseline that predicts the mean inspiratory pressure per (R, C, time_step, u_out) bucket, with a global fallback, and then blend exactly as you already do. This is a minimal change that should dramatically reduce MAE versus zeros and head toward your target without changing evaluation semantics or introducing new training loops/models.'
- What this solution (achieved 3.89158) has done: 'Your score is still far above the target because the fallback predictions are too coarse: binning only by `(R, C, u_out, time_step)` ignores the strong dependence on `u_in` (and the competition only scores inspiratory phase where `u_out=0`). To move the MAE much closer to the target while keeping your “blend multiple submissions with fallback when missing” core logic intact, I upgrade the local fallback to a slightly richer, still non-model baseline: mean pressure by `(R, C, u_out, time_step_bin, u_in_bin)` with a safe multi-level backoff when a bin is unseen. I also ensure that for `u_out==1` in test we predict 0 (not scored, and reduces noise), and keep your blending weights unchanged so changes are minimal and focused on improving fallback quality.'
- What this solution (achieved 3.89137) has done: 'Your current score is still far above the target, so we should legitimately improve the fallback predictions while keeping your “blend multiple submissions; if missing, use local fallback” core logic unchanged. The smallest high-impact fix is to make the fallback much closer to the known pressure discretization used in this competition by snapping predictions to the nearest valid pressure value from the training set (this is a standard, legal post-processing step and does not change your modeling approach). I also add one more backoff level that uses `(R,C,ts_bin,uin_bin)` (since `u_out` is constant 0 in inspiratory train used for lookups), which improves coverage without introducing any new model/training loop. Blending weights and file paths remain unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 2.62534) has done: 'Your current score is still far above the target, so we should legitimately improve the fallback predictions while keeping your existing “blend multiple submissions; if missing, use local fallback” approach unchanged. The smallest high-impact change is to make the local fallback use a breath-wise time index (`step` 0–79) instead of rounding `time_step`, and to add simple, standard ventilator features (`u_in_cum`, `u_in_lag1`, `u_out_lag1`) that don’t change the modeling paradigm but greatly sharpen the lookup. I keep the same multi-level backoff merging and the same blend weights, and I still snap inspiratory predictions to the nearest valid training pressure value. This should reduce MAE substantially versus the current coarse binning while remaining fast and within Kaggle constraints.'
- What this solution (achieved 3.0096) has done: 'Your current score is far worse than the target (lower is better), so we need a modest but meaningful improvement while keeping your “blend multiple submissions with a local fallback” logic intact. The biggest issue is that, in this environment, all four external submission files are missing, so you’re effectively blending four identical fallback predictions (no ensemble benefit). To improve the fallback without changing the overall approach, I add a stronger (still lookup-based) backoff level that uses within-breath delta features (`du_in`, `du_in_lag1`) and a slightly finer binning for `u_in_cum`, and I also snap the final blended predictions (not just the fallback) to the discrete training pressure grid, which typically reduces MAE in this competition. These are minimal, fast changes that preserve the same overall pipeline and still produce a valid `submission.csv`.'
- What this solution (achieved 3.07625) has done: 'Your score is still far above the target (lower is better), so we should legitimately improve the fallback predictions while keeping your existing “blend multiple submissions with a local lookup fallback” core logic unchanged. The biggest gain with minimal risk is to (1) avoid predicting 0 for `u_out==1` (even if not scored, Kaggle still includes those rows in the MAE with masking by inspiratory phase; the safe move is to use the same fallback/backoff there too), and (2) make the lookup slightly more faithful by using a stable within-breath `step` plus a small additional backoff keyed on `(R,C,step,u_in)` without the delta features (improves coverage when deltas are noisy/unseen). Finally, keep snapping to the discrete pressure grid but do it after the final blended prediction (already done) and also ensure the fallback uses the *full* train pressure grid (not only inspiratory unique values) to reduce edge cases. These changes preserve your blending weights/structure and stay within the same lookup-and-backoff approach.'
- What this solution (achieved 3.07625) has done: 'Your current score is still far above the target (lower is better), so we need a modest, safe improvement in the fallback predictions while keeping your ensemble/blend structure unchanged. The biggest issue in your fallback is that you only snap predictions to the discrete pressure grid for `u_out==0`, leaving `u_out==1` rows unsnapped (even if not scored, this can still hurt depending on how the mask is applied and it also makes the blend noisier). I also fix the `id` alignment: you’re using `sample_submission.csv` as the master `id` list, but its `id` range info suggests it may not match the full test set in this environment; using `test.csv` `id` as the master avoids silent misalignment and ensures row count correctness. Finally, I keep your exact blend weights and lookup/backoff logic, only adding snapping for all rows and ensuring the output uses the test ids in correct order.'
- What this solution (achieved 3.07625) has done: 'Your current score (3.07625, lower is better) is still far above the target (0.1437), so we need a legitimate accuracy improvement while keeping your “blend multiple submissions with a local lookup fallback” core logic intact. The smallest high-impact fix is to stop “teaching” the fallback from all inspiratory rows globally and instead use only the inspiratory portion up to the per-breath switch to expiration (first `u_out==1`), which better matches the competition’s inspiratory scoring mask. Additionally, the MAE strongly benefits from using the well-known pressure discretization: we keep snapping, but we also clip predictions to the train pressure range before snapping to avoid edge-bin artifacts. Finally, because all external subs are missing here, we keep your blend structure but shift weight toward the improved fallback component (still a blend, same semantics) so the output moves closer to the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
BASE_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"

test_ids = pd.read_csv(TEST_PATH, usecols=["id"]).copy()
sub = test_ids.sort_values("id", kind="mergesort").reset_index(drop=True)

paths = {
    "sub_1": "../input/vpp-a-basic-ensembling-technique/submission_pp.csv",
    "sub_2": "../input/blend-of-blend-of-blend-of-blend-of-blend-of-ble/submission.csv",
    "sub_3": "../input/gb-vpp-whoppity-dub-dub/median_submission.csv",
    "sub_4": "../input/ensemble-without-overfitting-risk/submission_median.csv",
}


def _snap_to_train_pressures(
    pred: np.ndarray, train_pressures: np.ndarray
) -> np.ndarray:
    """
    Keep: pressure values are discrete; snapping reduces MAE without changing evaluation semantics.
    """
    idx = np.searchsorted(train_pressures, pred, side="left")
    idx = np.clip(idx, 0, len(train_pressures) - 1)

    left_idx = np.clip(idx - 1, 0, len(train_pressures) - 1)
    right_idx = idx

    left_val = train_pressures[left_idx]
    right_val = train_pressures[right_idx]

    choose_right = np.abs(right_val - pred) <= np.abs(pred - left_val)
    return np.where(choose_right, right_val, left_val).astype("float32")


def build_local_fallback_submission(train_path: str, test_path: str) -> pd.DataFrame:
    """
    Fallback baseline: binned mean lookup with safe multi-level backoff.

    Change (score-improving, minimal):
    - Build lookups from the *scored inspiratory portion*: rows with u_out==0 only up to the first
      u_out==1 within each breath. This matches the metric mask better than using all u_out==0 rows
      (which can include post-expiration artifacts depending on signals).
    - Clip to train pressure range before snapping to reduce edge-bin artifacts.
    """
    usecols_train = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    usecols_test = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]

    train = pd.read_csv(train_path, usecols=usecols_train)
    test = pd.read_csv(test_path, usecols=usecols_test)

    train = train.sort_values(["breath_id", "time_step"], kind="mergesort")
    test = test.sort_values(["breath_id", "time_step"], kind="mergesort")

    train["step"] = train.groupby("breath_id").cumcount().astype("int16")
    test["step"] = test.groupby("breath_id").cumcount().astype("int16")

    train["u_in_cum"] = train.groupby("breath_id")["u_in"].cumsum().astype("float32")
    test["u_in_cum"] = test.groupby("breath_id")["u_in"].cumsum().astype("float32")

    train["u_in_lag1"] = (
        train.groupby("breath_id")["u_in"].shift(1).fillna(0.0).astype("float32")
    )
    test["u_in_lag1"] = (
        test.groupby("breath_id")["u_in"].shift(1).fillna(0.0).astype("float32")
    )

    train["u_out_lag1"] = (
        train.groupby("breath_id")["u_out"].shift(1).fillna(0).astype("int8")
    )
    test["u_out_lag1"] = (
        test.groupby("breath_id")["u_out"].shift(1).fillna(0).astype("int8")
    )

    train["du_in"] = (train["u_in"] - train["u_in_lag1"]).astype("float32")
    test["du_in"] = (test["u_in"] - test["u_in_lag1"]).astype("float32")

    train["du_in_lag1"] = (
        train.groupby("breath_id")["du_in"].shift(1).fillna(0.0).astype("float32")
    )
    test["du_in_lag1"] = (
        test.groupby("breath_id")["du_in"].shift(1).fillna(0.0).astype("float32")
    )

    train["uin_bin"] = (np.round(train["u_in"].values * 2.0) / 2.0).astype("float32")
    test["uin_bin"] = (np.round(test["u_in"].values * 2.0) / 2.0).astype("float32")

    train["uinc_bin"] = (np.round(train["u_in_cum"].values / 2.5) * 2.5).astype(
        "float32"
    )
    test["uinc_bin"] = (np.round(test["u_in_cum"].values / 2.5) * 2.5).astype("float32")

    train["uinlag_bin"] = (np.round(train["u_in_lag1"].values * 2.0) / 2.0).astype(
        "float32"
    )
    test["uinlag_bin"] = (np.round(test["u_in_lag1"].values * 2.0) / 2.0).astype(
        "float32"
    )

    train["duin_bin"] = (np.round(train["du_in"].values * 2.0) / 2.0).astype("float32")
    test["duin_bin"] = (np.round(test["du_in"].values * 2.0) / 2.0).astype("float32")

    train["duinlag_bin"] = (np.round(train["du_in_lag1"].values * 2.0) / 2.0).astype(
        "float32"
    )
    test["duinlag_bin"] = (np.round(test["du_in_lag1"].values * 2.0) / 2.0).astype(
        "float32"
    )

    first_uout_step = (
        train.loc[train["u_out"].eq(1), ["breath_id", "step"]]
        .groupby("breath_id", observed=True)["step"]
        .min()
    )
    train = train.join(first_uout_step.rename("first_uout_step"), on="breath_id")
    train["first_uout_step"] = train["first_uout_step"].fillna(10_000).astype("int16")
    train_insp = train[
        (train["u_out"] == 0) & (train["step"] < train["first_uout_step"])
    ].copy()

    grp0 = ["R", "C", "step", "uin_bin", "duin_bin", "duinlag_bin"]
    grp0b = ["R", "C", "step", "uin_bin"]

    grp1 = [
        "R",
        "C",
        "u_out",
        "step",
        "uin_bin",
        "uinc_bin",
        "uinlag_bin",
        "u_out_lag1",
    ]
    grp1b = ["R", "C", "step", "uin_bin", "uinc_bin", "uinlag_bin"]
    grp2 = ["R", "C", "u_out", "step", "uin_bin"]
    grp3 = ["R", "C", "u_out", "step"]
    grp4 = ["R", "C", "u_out"]

    lookup0 = (
        train_insp.groupby(grp0, observed=True)["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "p0"})
    )
    lookup0b = (
        train_insp.groupby(grp0b, observed=True)["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "p0b"})
    )
    lookup1 = (
        train_insp.groupby(grp1, observed=True)["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "p1"})
    )
    lookup1b = (
        train_insp.groupby(grp1b, observed=True)["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "p1b"})
    )
    lookup2 = (
        train_insp.groupby(grp2, observed=True)["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "p2"})
    )
    lookup3 = (
        train_insp.groupby(grp3, observed=True)["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "p3"})
    )
    lookup4 = (
        train_insp.groupby(grp4, observed=True)["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "p4"})
    )

    global_mean = float(train_insp["pressure"].mean())

    merged = test.merge(lookup0, on=grp0, how="left")
    merged = merged.merge(lookup0b, on=grp0b, how="left")
    merged = merged.merge(lookup1, on=grp1, how="left")
    merged = merged.merge(lookup1b, on=grp1b, how="left")
    merged = merged.merge(lookup2, on=grp2, how="left")
    merged = merged.merge(lookup3, on=grp3, how="left")
    merged = merged.merge(lookup4, on=grp4, how="left")

    pred = merged["p0"]
    pred = pred.fillna(merged["p0b"])
    pred = pred.fillna(merged["p1"])
    pred = pred.fillna(merged["p1b"])
    pred = pred.fillna(merged["p2"])
    pred = pred.fillna(merged["p3"])
    pred = pred.fillna(merged["p4"])
    pred = pred.fillna(global_mean).astype("float32").values

    pressure_values_full = np.sort(train["pressure"].astype("float32").unique())
    pred = np.clip(pred, pressure_values_full[0], pressure_values_full[-1])
    pred = _snap_to_train_pressures(pred, pressure_values_full)

    out = merged[["id"]].assign(pressure=pred)
    return out


local_fallback = build_local_fallback_submission(TRAIN_PATH, TEST_PATH)


def load_submission_or_fallback(
    path: str, fallback: pd.DataFrame, master_ids: pd.DataFrame
) -> pd.DataFrame:
    """
    Load a submission file if present; otherwise return fallback.
    Ensures alignment to master_ids and correct columns.
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "id" not in df.columns:
            df = df.reset_index()
        if "pressure" not in df.columns:
            raise ValueError(
                f"Loaded file at {path} but did not find 'pressure' column."
            )
        df = df[["id", "pressure"]].copy()
    else:
        df = fallback[["id", "pressure"]].copy()

    df = master_ids.merge(df, on="id", how="left")
    df["pressure"] = df["pressure"].astype("float32").fillna(0.0)
    return df


sub_1 = load_submission_or_fallback(paths["sub_1"], local_fallback, sub[["id"]])
sub_2 = load_submission_or_fallback(paths["sub_2"], local_fallback, sub[["id"]])
sub_3 = load_submission_or_fallback(paths["sub_3"], local_fallback, sub[["id"]])
sub_4 = load_submission_or_fallback(paths["sub_4"], local_fallback, sub[["id"]])

sub_ids = sub[["id"]].copy()
sub_ids = sub_ids.merge(sub_1, on="id", how="left").rename(columns={"pressure": "p1"})
sub_ids = sub_ids.merge(sub_2, on="id", how="left").rename(columns={"pressure": "p2"})
sub_ids = sub_ids.merge(sub_3, on="id", how="left").rename(columns={"pressure": "p3"})
sub_ids = sub_ids.merge(sub_4, on="id", how="left").rename(columns={"pressure": "p4"})

for c in ["p1", "p2", "p3", "p4"]:
    sub_ids[c] = sub_ids[c].astype("float32").fillna(0.0)

sub = sub_ids



## === cell 2
w1, w2, w3, w4 = 0.70, 0.00, 0.10, 0.20

sub["pressure"] = (
    (sub["p1"].values * w1)
    + (sub["p2"].values * w2)
    + (sub["p3"].values * w3)
    + (sub["p4"].values * w4)
).astype("float32")

train_pressures = np.sort(
    pd.read_csv(TRAIN_PATH, usecols=["pressure"])["pressure"].astype("float32").unique()
)

sub["pressure"] = np.clip(
    sub["pressure"].values, train_pressures[0], train_pressures[-1]
).astype("float32")
sub["pressure"] = _snap_to_train_pressures(sub["pressure"].values, train_pressures)

submission = sub[["id", "pressure"]].copy()

submission = test_ids.merge(submission, on="id", how="left")
submission["pressure"] = submission["pressure"].astype("float32").fillna(0.0)

submission.to_csv("submission.csv", index=False)
print(submission.head(5))
print("submission.csv written with rows:", len(submission))
