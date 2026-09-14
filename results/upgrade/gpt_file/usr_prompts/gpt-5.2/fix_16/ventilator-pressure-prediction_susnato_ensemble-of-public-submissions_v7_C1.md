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

3.9

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

0.1992052343164609

# 6. Current score

4.04577

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.27368) has done: 'I remove the dependency on missing external Kaggle Dataset paths (the `../input/.../submission.csv` files) by switching to an in-notebook baseline that trains on `train.csv` and predicts `pressure` for `test.csv`. This fixes the FileNotFound/NameError chain and guarantees an end-to-end run that produces a valid `submission.csv` with the required `id,pressure` columns. To keep the core logic minimal and stable, I use a fast group-based mean target encoding by `(R, C, time_step)` (computed on inspiratory rows where `u_out==0`, matching the metric focus) with a safe global fallback. The output is aligned by `id` and written with a `.csv` suffix.'
- What this solution (achieved 6.10613) has done: 'You’re already generating a valid submission, but the score is very high because the model ignores the strongest baseline signal: when the expiratory valve is open (`u_out==1`), pressure quickly drops toward a near-constant low value and those rows are not scored, so predicting the inspiratory mean everywhere hurts. To move toward the target with minimal logic change, I keep the same group-mean approach but (1) compute means separately for inspiratory (`u_out==0`) and expiratory (`u_out==1`) and use the appropriate mapping at prediction time, with safe global fallbacks. Additionally, I make `time_step` matching robust by rounding to 2 decimals in both train/test before grouping/merging, preventing floating-point mismatches that create many NaNs and force the weak global fallback. These are small, direct adjustments that typically reduce MAE substantially without changing the overall “target-encoding by grouped means” core idea.'
- What this solution (achieved 5.95901) has done: 'We keep your “group-mean target encoding” core logic, but make it more aligned with the metric by computing the mean map only on inspiratory rows (`u_out==0`), since expiratory rows are not scored and including them can bias the mapping. We also make `time_step` matching more precise/robust by rounding to 3 decimals (the data is effectively at ~0.03s resolution, and 2-decimal rounding can collapse distinct steps and add noise). Finally, we add a very small, leakage-free calibration step: compute a single additive offset on a held-out set (grouped by `breath_id` to avoid within-breath leakage) and apply it to test predictions, which often reduces MAE without changing the model form. The submission format and paths remain unchanged, and the script still runs end-to-end under the time limit.'
- What this solution (achieved 5.95906) has done: 'Your current score (MAE 5.959) is far from the target (0.199), so the biggest “minimal” gain comes from aligning predictions with the known discrete pressure levels used in this competition. We keep your exact group-mean-by-(R,C,time_step) core logic and the same leakage-free offset calibration, but add a final post-processing step that snaps predictions to the nearest pressure value observed in the training set (a standard, metric-aligned fix for this dataset). This typically reduces MAE substantially without changing the modeling approach. I also keep your rounding and fallbacks unchanged to preserve stability and ensure a valid `submission.csv` is always produced.'
- What this solution (achieved 4.39644) has done: 'Your current MAE (5.959) is far above the target (0.199), so we should improve accuracy with the smallest changes that keep your same “group mean by (R,C,time_step)” core logic. The biggest issue is that your mapping ignores the strongest control signal `u_in`, so different pressures get averaged together within each `(R,C,time_step)` bucket. I keep the same rounding, leakage-safe breath-level calibration, clipping, and pressure-level snapping, but add `u_in` (rounded) to the grouping keys used for the mean map and calibration map to better condition the average on the input setting. This is a minimal, metric-aligned change that typically reduces MAE substantially while preserving your overall approach and still writing a valid `submission.csv`.'
- What this solution (achieved 5.70747) has done: 'Your current MAE (4.396) is still far above the target (0.199), so we should improve accuracy with the smallest possible change that preserves your “group mean target encoding + breath-level calibration + snapping” core logic. The main missing signal is short-term history: pressure depends strongly on recent `u_in` and flow dynamics, so using only the current step can still average incompatible states. I keep the same approach but extend the grouping keys with a lagged `u_in` (previous time step within the breath, rounded the same way) to better condition the mean without changing the model class. Everything else (rounding, inspiratory-only training, leakage-safe calibration, clipping, snapping, and submission writing) remains intact.'
- What this solution (achieved 6.22314) has done: 'Your MAE is still far above the target (lower is better), so we should improve accuracy with the smallest change that preserves your existing “group-mean target encoding + leakage-safe calibration + snapping” approach. The strongest missing signal you can add without changing the model class is an additional short-term history term: a 2-step lag of `u_in` within each breath (rounded the same way), which better conditions the mean on recent control dynamics. We keep all other logic identical (inspiratory-only training, rounding, offset calibration split by `breath_id`, clipping, snapping, and submission formatting), just extend the grouping keys and the calibration map accordingly. This should move the score downward (better) toward the target without introducing new training loops or architecture changes.'
- What this solution (achieved 5.70747) has done: 'I fix the main reason your score got worse after adding more lags: the extra lag keys make the `(R,C,time_step,u_in,lag1,lag2)` groups too sparse, so most test rows fall back to the global mean and MAE increases. To move the score down toward the target with minimal change, I keep your exact pipeline (inspiratory-only mean map, breath-level calibration, clipping, and pressure-level snapping) but remove `u_in_lag2_r` from the grouping/calibration keys to reduce sparsity while still using one-step history. I also make the merge more robust by ensuring the rounded lag columns have the same dtype in train/test before grouping/merging, avoiding silent key mismatches. Everything else (paths, submission schema, and post-processing) stays the same.'
- What this solution (achieved 6.78427) has done: 'Your current MAE is far above the target (lower is better), so we should improve accuracy with the smallest change that preserves your existing “group mean target encoding + breath-level calibration + snapping” approach. The key missing signal you can add without changing the modeling class is the cumulative injected volume surrogate `u_in_cumsum` (within-breath integral of `u_in` over time), which strongly correlates with pressure dynamics and is a common baseline feature for this competition. I add a rounded `u_in_cumsum` as an extra grouping key (and keep all existing keys, rounding, inspiratory-only training, leakage-safe calibration split, clipping, snapping, and submission writing unchanged). This should reduce averaging across incompatible states and move the MAE down toward the target while keeping the overall logic intact.'
- What this solution (achieved 5.88315) has done: 'Your current MAE is far above the target, so we should make a small, metric-aligned change that improves accuracy without changing your overall “group-mean target encoding + calibration offset + snapping” pipeline. The main fix is to compute `u_in_cumsum` as a true within-breath integral (`sum(u_in * delta_time)`) rather than a simple cumsum of `u_in`, because pressure dynamics depend on delivered volume over time and the dataset’s `time_step` spacing is not perfectly uniform. This keeps the same feature family (cumulative injected volume surrogate) but makes it physically consistent and usually less noisy, which should reduce grouping collisions and improve the mean map hit-rate. Everything else (rounding, inspiratory-only mapping, breath-level split calibration, clipping, pressure-level snapping, and submission writing) remains unchanged.'
- What this solution (achieved 5.31874) has done: 'We keep your exact “group-mean target encoding + breath-level calibration offset + clipping + pressure-level snapping” pipeline, but make the grouping slightly less sparse so fewer test rows fall back to the global mean. Concretely, we keep `u_in_cumsum` as the physically-correct integral, but round it a bit more coarsely (0 decimals instead of 1) to improve train/test key hit-rate while preserving the same feature and core logic. We also compute the calibration map using the exact same (updated) grouping keys to stay consistent and avoid calibration mismatches. This is a minimal, metric-aligned change that should reduce MAE (lower is better) toward your target without changing the overall approach.'
- What this solution (achieved 6.56546) has done: 'Your MAE (5.3187) is still far above the target (0.199, lower is better), so we need a small change that legitimately improves accuracy without changing the overall “group-mean target encoding + breath-level calibration offset + snapping” pipeline. The biggest issue left is key sparsity/mismatch: `u_in_cumsum_r` is currently rounded on a breath-duration scale that can differ enough between train/test to cause many unseen groups and global-mean fallback. I keep the same feature but normalize it within each breath (divide by the final cumulative delivered “volume”), then round that normalized value as an extra grouping key; this preserves core logic while improving hit-rate. I also ensure the integral uses `dt` computed after sorting (already done) and add a safe epsilon to avoid divide-by-zero for early steps.'
- What this solution (achieved 6.33126) has done: 'I keep your exact “group-mean target encoding + breath-level offset calibration + clipping + snap-to-known-levels” pipeline, but fix the main reason the score is still very poor: the mapping is trained only on inspiratory rows while your group keys include features that evolve during expiration (especially `u_in_cumsum_norm_r`), causing many inspiratory test rows to miss and fall back to a weak global mean. To reduce sparsity with minimal change, I (1) compute the cumulative/integral features only over inspiratory portions (reset to 0 once `u_out` flips to 1) so keys are comparable between train/test for scored timesteps, and (2) cap the influence of `u_in_cumsum_norm_r` by coarsening its rounding slightly to increase train/test hit-rate. Everything else (keys, calibration, snapping, file paths, and submission writing) stays the same and the script still runs end-to-end producing `submission.csv`.'
- What this solution (achieved 4.04577) has done: 'Your MAE is still far above the target (lower is better), so we should improve accuracy with the smallest possible change that preserves your current “group-mean target encoding + breath-level calibration offset + clipping + snap-to-known-levels” pipeline. The main remaining weakness is group sparsity/mismatch: `u_in_cumsum_norm_r` makes keys too brittle, so many test rows miss the map and fall back to a weak global mean. I keep all your existing features and post-processing, but add a minimal hierarchical fallback: if the full key is missing, back off to a slightly simpler mean-map that drops only `u_in_cumsum_norm_r`, then to a map that drops `u_in_lag1_r`, then finally to the global inspiratory mean. This usually reduces fallback error significantly while keeping the same core logic and evaluation semantics, and it still produces a valid `submission.csv`.'
- What this solution (achieved 4.04577) has done: 'Your current MAE (4.04577) is still far above the target (0.199, lower is better), so we should improve accuracy with the smallest change that preserves your existing “group-mean target encoding + breath-level calibration offset + clipping + snap-to-known-levels” pipeline. The main issue is that your mean maps are trained only on inspiratory rows but you apply them to all test rows; for expiratory rows (`u_out==1`) this forces inappropriate inspiratory-derived pressures (and the later snapping can make this worse). I keep the exact same mapping/calibration/snapping core logic, but add a minimal expiratory fallback: learn a simple mean map for expiration keyed only by `(R,C,time_step_r)` (plus a global expiratory mean) and use it only for `u_out==1` rows. This reduces unrealistic predictions on expiration without affecting inspiratory logic, and should move MAE downward toward the target while still producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
DATA_DIR_CANDIDATES = [
    "/kaggle/input/ventilator-pressure-prediction",
    "/kaggle/data/ventilator-pressure-prediction",
    "/kaggle/input",
    "/kaggle/data",
]


def find_file(filename: str) -> str:
    for d in DATA_DIR_CANDIDATES:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    for d in DATA_DIR_CANDIDATES:
        p = os.path.join(d, "ventilator-pressure-prediction", filename)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find {filename} in known Kaggle directories.")


train_path = find_file("train.csv")
test_path = find_file("test.csv")
sample_path = find_file("sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sub = pd.read_csv(sample_path)

required_train_cols = {"breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"}
required_test_cols = {"id", "breath_id", "R", "C", "time_step", "u_in", "u_out"}
if not required_train_cols.issubset(train.columns):
    raise ValueError(
        f"train.csv missing columns: {required_train_cols - set(train.columns)}"
    )
if not required_test_cols.issubset(test.columns):
    raise ValueError(
        f"test.csv missing columns: {required_test_cols - set(test.columns)}"
    )
if not {"id", "pressure"}.issubset(sub.columns):
    raise ValueError("sample_submission.csv must contain ['id','pressure'].")

train = train.copy()
test = test.copy()

train.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")
test.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")

train["time_step_r"] = train["time_step"].round(3)
test["time_step_r"] = test["time_step"].round(3)

train["u_in_r"] = train["u_in"].round(1)
test["u_in_r"] = test["u_in"].round(1)

train["u_in_lag1_r"] = (
    train.groupby("breath_id", sort=False)["u_in_r"].shift(1).fillna(0.0)
)
test["u_in_lag1_r"] = (
    test.groupby("breath_id", sort=False)["u_in_r"].shift(1).fillna(0.0)
)

train["_dt"] = train.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0)
test["_dt"] = test.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0)




## === cell 2
def add_insp_integral_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    insp = (df["u_out"].to_numpy() == 0).astype(np.int8)
    exp_started = (
        pd.Series(insp == 0, index=df.index)
        .groupby(df["breath_id"], sort=False)
        .cumsum()
        .to_numpy()
        > 0
    )
    insp_mask = ~exp_started  # True until expiration begins

    u_in_insp = df["u_in"].to_numpy(dtype=np.float64) * insp_mask.astype(np.float64)
    dt = df["_dt"].to_numpy(dtype=np.float64)
    contrib = u_in_insp * dt

    df["u_in_cumsum"] = (
        pd.Series(contrib, index=df.index).groupby(df["breath_id"], sort=False).cumsum()
    )

    eps = 1e-6
    final = (
        df.groupby("breath_id", sort=False)["u_in_cumsum"]
        .transform("last")
        .to_numpy(dtype=np.float64)
    )
    df["u_in_cumsum_norm"] = df["u_in_cumsum"].to_numpy(dtype=np.float64) / (
        np.abs(final) + eps
    )

    return df


train = add_insp_integral_features(train)
test = add_insp_integral_features(test)

train.drop(columns=["_dt"], inplace=True)
test.drop(columns=["_dt"], inplace=True)

train["u_in_cumsum_r"] = train["u_in_cumsum"].round(0)
test["u_in_cumsum_r"] = test["u_in_cumsum"].round(0)

train["u_in_cumsum_norm_r"] = train["u_in_cumsum_norm"].round(2)
test["u_in_cumsum_norm_r"] = test["u_in_cumsum_norm"].round(2)

for c in [
    "time_step_r",
    "u_in_r",
    "u_in_lag1_r",
    "u_in_cumsum_r",
    "u_in_cumsum_norm_r",
]:
    train[c] = train[c].astype(np.float32)
    test[c] = test[c].astype(np.float32)

grp_cols = ["R", "C", "time_step_r", "u_in_r", "u_in_lag1_r", "u_in_cumsum_norm_r"]



## === cell 3
train_insp = train.loc[train["u_out"] == 0].copy()

grp_cols_full = grp_cols
grp_cols_drop_norm = ["R", "C", "time_step_r", "u_in_r", "u_in_lag1_r"]
grp_cols_drop_lag = ["R", "C", "time_step_r", "u_in_r"]

mean_map_full = (
    train_insp.groupby(grp_cols_full, observed=True)["pressure"]
    .mean()
    .rename("p_full")
    .reset_index()
)
mean_map_drop_norm = (
    train_insp.groupby(grp_cols_drop_norm, observed=True)["pressure"]
    .mean()
    .rename("p_drop_norm")
    .reset_index()
)
mean_map_drop_lag = (
    train_insp.groupby(grp_cols_drop_lag, observed=True)["pressure"]
    .mean()
    .rename("p_drop_lag")
    .reset_index()
)

global_mean_insp = float(train_insp["pressure"].mean())
global_mean_all = float(train["pressure"].mean())

train_exp = train.loc[train["u_out"] == 1].copy()
exp_grp_cols = ["R", "C", "time_step_r"]
exp_map = (
    train_exp.groupby(exp_grp_cols, observed=True)["pressure"]
    .mean()
    .rename("p_exp")
    .reset_index()
)
global_mean_exp = (
    float(train_exp["pressure"].mean()) if len(train_exp) else global_mean_all
)

test_pred = test.copy()
test_pred = test_pred.merge(mean_map_full, on=grp_cols_full, how="left")
test_pred = test_pred.merge(mean_map_drop_norm, on=grp_cols_drop_norm, how="left")
test_pred = test_pred.merge(mean_map_drop_lag, on=grp_cols_drop_lag, how="left")

test_pred = test_pred.merge(exp_map, on=exp_grp_cols, how="left")

test_pred["pressure"] = test_pred["p_full"]
mask = test_pred["pressure"].isna()
if mask.any():
    test_pred.loc[mask, "pressure"] = test_pred.loc[mask, "p_drop_norm"]
mask = test_pred["pressure"].isna()
if mask.any():
    test_pred.loc[mask, "pressure"] = test_pred.loc[mask, "p_drop_lag"]
mask = test_pred["pressure"].isna()
if mask.any():
    test_pred.loc[mask, "pressure"] = global_mean_insp

exp_mask_test = test_pred["u_out"].to_numpy() == 1
if exp_mask_test.any():
    exp_vals = test_pred.loc[exp_mask_test, "p_exp"]
    test_pred.loc[exp_mask_test, "pressure"] = exp_vals.fillna(
        global_mean_exp
    ).to_numpy()

unique_breaths = train_insp["breath_id"].unique()
rng = np.random.default_rng(2021)
rng.shuffle(unique_breaths)

n_val = max(1, int(0.10 * len(unique_breaths)))
val_breaths = set(unique_breaths[:n_val])

val = train_insp.loc[train_insp["breath_id"].isin(val_breaths)].copy()
cal_train = train_insp.loc[~train_insp["breath_id"].isin(val_breaths)]

cal_map_full = (
    cal_train.groupby(grp_cols_full, observed=True)["pressure"]
    .mean()
    .rename("p_full")
    .reset_index()
)
cal_map_drop_norm = (
    cal_train.groupby(grp_cols_drop_norm, observed=True)["pressure"]
    .mean()
    .rename("p_drop_norm")
    .reset_index()
)
cal_map_drop_lag = (
    cal_train.groupby(grp_cols_drop_lag, observed=True)["pressure"]
    .mean()
    .rename("p_drop_lag")
    .reset_index()
)

val_pred = val.merge(cal_map_full, on=grp_cols_full, how="left")
val_pred = val_pred.merge(cal_map_drop_norm, on=grp_cols_drop_norm, how="left")
val_pred = val_pred.merge(cal_map_drop_lag, on=grp_cols_drop_lag, how="left")

val_pred["pred"] = val_pred["p_full"]
m = val_pred["pred"].isna()
if m.any():
    val_pred.loc[m, "pred"] = val_pred.loc[m, "p_drop_norm"]
m = val_pred["pred"].isna()
if m.any():
    val_pred.loc[m, "pred"] = val_pred.loc[m, "p_drop_lag"]
m = val_pred["pred"].isna()
if m.any():
    val_pred.loc[m, "pred"] = global_mean_insp

offset = float(np.median(val_pred["pressure"].to_numpy() - val_pred["pred"].to_numpy()))
test_pred["pressure"] = test_pred["pressure"] + offset

pmin = float(train["pressure"].min())
pmax = float(train["pressure"].max())
test_pred["pressure"] = test_pred["pressure"].clip(pmin, pmax)

pressure_levels = np.sort(train["pressure"].unique()).astype(np.float64)
pred_vals = test_pred["pressure"].to_numpy(dtype=np.float64)

idx = np.searchsorted(pressure_levels, pred_vals, side="left")
idx = np.clip(idx, 0, len(pressure_levels) - 1)
idx_prev = np.clip(idx - 1, 0, len(pressure_levels) - 1)

cand_hi = pressure_levels[idx]
cand_lo = pressure_levels[idx_prev]
choose_lo = np.abs(pred_vals - cand_lo) <= np.abs(pred_vals - cand_hi)
snapped = np.where(choose_lo, cand_lo, cand_hi)

test_pred["pressure"] = snapped

sub_out = sub[["id"]].merge(test_pred[["id", "pressure"]], on="id", how="left")
if sub_out["pressure"].isna().any():
    sub_out["pressure"] = sub_out["pressure"].fillna(global_mean_all)

sub_out.to_csv("submission.csv", index=False)

print(sub_out.head())
print(sub_out.shape)
print(
    f"Saved submission.csv (offset applied: {offset:.6f}; snapped to {len(pressure_levels)} levels)"
)
