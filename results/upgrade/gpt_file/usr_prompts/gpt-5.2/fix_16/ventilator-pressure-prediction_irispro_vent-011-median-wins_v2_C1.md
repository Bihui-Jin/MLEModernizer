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

0.1451642246313568

# 6. Current score

4.30214

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.14507) has done: 'I fix the `FileNotFoundError` by removing dependencies on external Kaggle datasets (your environment only has the ventilator data) and replacing the ensemble step with a simple, fully-local baseline that uses only `train.csv` and `test.csv`. To keep the core idea “simple aggregation-based prediction” while producing a valid submission, I predict pressures by averaging `pressure` grouped by `(R, C, time_step, u_in, u_out)` from train, and fall back to `(R, C, time_step, u_out)` then global mean when unseen combinations appear. I also make path handling robust by searching the provided local data directories, and ensure the output is exactly `submission.csv` with columns `id,pressure` and the correct row count.'
- What this solution (achieved 3.8916) has done: 'Your current MAE (8.145) is far above the target (0.145), so we should improve accuracy without changing the overall “aggregation-based prediction” core logic. The biggest issue is that grouping on raw float `time_step` and `u_in` creates many unseen combinations in test, causing heavy fallback to coarse averages/global mean. I keep the same hierarchical mean-lookup idea, but make keys more robust by rounding (`time_step`) and binning (`u_in`) consistently in train/test, and I add one extra intermediate fallback level to reduce missing-rate while preserving semantics. This should materially reduce error while staying within the same simple non-ML baseline approach and still writing a valid `submission.csv`.'
- What this solution (achieved 4.08579) has done: 'Your current MAE (3.8916) is still far above the target (0.145), so we should legitimately improve accuracy while keeping the same core “hierarchical aggregation/lookup” logic. The main remaining error source is mismatched keys between train and test due to float binning/rounding and overly-granular keys that still miss often; we can reduce misses by (1) using stable integer keys (scaled/rounded) instead of float columns, and (2) adding one more fallback level that drops `time` before collapsing all the way to global mean. These are minimal changes that preserve the same approach (group means with fallbacks) but should substantially reduce fallback-to-global and thus MAE. The script still runs end-to-end and writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 4.08579) has done: 'Your score is far worse than the target (lower is better), so we should improve accuracy while keeping the same “hierarchical group-mean lookup with fallbacks” core logic. The biggest issue is that you’re averaging across both inspiratory and expiratory phases, but Kaggle only scores inspiratory (u_out==0); mixing expiratory rows corrupts the mean targets for the important phase. I compute group means using only inspiratory rows, and add a final, phase-consistent fallback mean per (R,C,u_out) before global mean to reduce misses without changing the overall approach. Everything else (robust integer keys, multi-level fallbacks, submission writing) stays the same.'
- What this solution (achieved 4.08579) has done: 'Your current MAE (4.08579) is still far above the target (0.14516), so we should improve accuracy while keeping the same “hierarchical group-mean lookup with fallbacks” core logic. The biggest remaining issue is that the first several group-mean tables are computed only on inspiratory rows (u_out==0), but you still try to match test rows with u_out==1 against those tables, guaranteeing lots of misses and forcing bad fallbacks. I keep your exact lookup approach, but make it phase-consistent: build inspiratory-only group tables used only for u_out==0 test rows, and build separate expiratory tables (u_out==1) from expiratory rows for those test rows. This is a minimal, semantics-preserving change that should reduce unnecessary fallback and move the score down toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 5.24124) has done: 'Your MAE (4.08579) is still far above the target (0.14516), so we should improve accuracy while preserving your exact “hierarchical group-mean lookup with fallbacks” approach. The main remaining weakness is the overly-coarse binning of `u_in` (0.1) and `time_step` (0.01): it still creates many unseen key combinations and forces fallbacks that blur the mapping between controls and pressure. I keep the same lookup tables and fallback order, but make the robust keys finer (0.001s for time, 0.01 for u_in) using stable integer keys; this is a minimal change that should reduce miss-rate and improve predictions. I also keep the inspiratory/expiratory separation exactly as you already do and still write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 4.08579) has done: 'Your current score is much worse than the target (lower is better), so we should improve accuracy while keeping the exact same “hierarchical group-mean lookup with fallbacks” core logic. The most likely reason your score got worse when you made keys finer is that overly-fine `time_step`/`u_in` keys increase sparsity and miss-rate, causing more fallbacks to coarse/global means. I keep your phase-separated lookup tables and fallback order unchanged, but switch to slightly coarser, more stable integer keys (0.01s for `time_step` and 0.1 for `u_in`) and ensure train/test use identical quantization. This should reduce mismatches, increase hit-rate in the higher-quality tables, and move MAE down toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 4.30216) has done: 'We keep your exact hierarchical group-mean lookup with phase-separated tables, but adjust the quantization slightly to reduce key-mismatch and sparsity (which currently forces too many coarse fallbacks and inflates MAE). Specifically, we use a slightly finer `u_in` key (0.05 instead of 0.1) while keeping `time_step` at 0.01s, which tends to improve match quality without exploding sparsity like the 0.01 `u_in` attempt did. We also add one minimal extra fallback level that drops `time_i` but keeps `u_in_i` (still aggregation/lookup, same semantics) before falling back to `(R,C,u_out)`/global mean. These are small, targeted changes aimed to move MAE down toward your target while staying within your current non-ML approach and producing the same `submission.csv` format.'
- What this solution (achieved 7.81217) has done: 'Your current MAE (4.30216, lower-is-better) is still far above the target (0.14516), so we should improve accuracy with the smallest possible change while keeping the same “phase-separated hierarchical group-mean lookup with fallbacks” core logic. The biggest remaining weakness is that the first lookup level uses `time_i` and `u_in_i` separately, but the true pressure dynamics depend strongly on *recent history*; a minimal way to capture that without changing the approach is to add one extra top-level table using simple lagged robust keys (`u_in_i_prev`, `u_out_prev`) computed per breath. We then try this lag-aware table first (for both train/test), and if it misses we fall back to your existing exact hierarchy unchanged. This should reduce error materially (especially at identical current keys but different trajectories) while preserving the same aggregation/lookup semantics and still producing a valid `submission.csv`.'
- What this solution (achieved 7.81216) has done: 'Your current score (7.81 MAE) is far worse than the target (0.145, lower-is-better), so we should improve accuracy with the smallest possible change while preserving your exact “phase-separated hierarchical group-mean lookup with fallbacks” approach. The lag-aware top key you added likely made the first table too sparse (especially with `u_out_prev`), increasing misses and pushing many rows into coarser fallbacks, which can worsen MAE. I keep all your existing tables and fallback order, but make the lagged key less sparse by dropping `u_out_prev` from the top-level key (still lag-aware via `u_in_i_prev`) and ensuring lags are computed in a stable order per breath. This should increase top-level hit-rate and move MAE down toward the target without changing the modeling paradigm or output format.'
- What this solution (achieved 4.08579) has done: 'Your current MAE (7.812) is far above the target (0.145, lower-is-better), so we should improve accuracy with the smallest change that reduces fallback/mismatch while keeping your exact “phase-separated hierarchical group-mean lookup with fallbacks” approach. The biggest issue is that the added lagged top-level table (`key0`) is extremely sparse and is likely causing many misses that cascade into poor coarse fallbacks; we keep the same hierarchy but remove the lagged level entirely to increase hit-rate at the strongest non-lag key. Additionally, we make the robust key quantization strictly consistent and slightly less sparse by using 0.1 for `u_in` (instead of 0.05) while keeping `time_step` at 0.01s, which typically improves match rate for lookup baselines. Everything else (phase separation, fallback order, output format) stays the same and still writes `submission.csv`.'
- What this solution (achieved 4.0858) has done: 'Your current MAE is far worse than the target (lower-is-better), so the safest way to move toward the target without changing the “hierarchical group-mean lookup with fallbacks” core logic is to reduce key-mismatch and increase hit-rate in the higher-quality (more specific) tables. I keep the same lookup hierarchy and phase-separated tables, but switch the robust key construction from `rint` to deterministic “half-up” quantization via `floor(x*scale + 0.5)`, which is more stable across float32/float64 representations and reduces train/test binning inconsistencies. I also stop downcasting `time_step`/`u_in` to float32 before keying (keeping float64 for quantization), because float32 rounding near bin boundaries can amplify mismatches; this is a minimal change that preserves semantics but typically improves matching. Everything else (tables, fallbacks, submission writing) stays the same and still produces `submission.csv`.'
- What this solution (achieved 4.0858) has done: 'Your current MAE (4.0858, lower-is-better) is far above the target (0.1452), so we should make a minimal change that improves prediction accuracy without changing the overall “phase-separated hierarchical group-mean lookup with fallbacks” approach. The main issue is that expiratory-phase (`u_out==1`) pressure is largely near-zero/constant in this dataset, and trying to learn it via sparse lookup tables adds noise; Kaggle also doesn’t score expiratory rows, so the safest improvement is to set all `u_out==1` test predictions to a fixed baseline (0.0) and keep your existing lookup only for `u_out==0`. This preserves your core logic for the scored part (inspiratory) while legitimately reducing error contributed by noisier expiratory predictions on the leaderboard. Everything still runs end-to-end and writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 4.30214) has done: 'Your MAE (4.0858) is far above the target (0.1452, lower-is-better), so we should improve accuracy with the smallest change that keeps your exact “hierarchical group-mean lookup with fallbacks” approach. The biggest remaining issue is that your `u_in` is quantized too coarsely (0.1), which blurs distinct control inputs and pushes many rows to fallbacks; we make `u_in` quantization slightly finer (0.05) while keeping `time_step` at 0.01 to avoid exploding sparsity. To compensate for any added sparsity, we also add one minimal extra fallback that drops `time_i` but keeps `(R,C,u_out,u_in_i)` before falling back to `(R,C,u_in_i)` and then `(R,C,u_out)`. Everything else (phase separation, table construction style, no ML, same output schema and filename) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 4.30214) has done: 'Your current MAE is far above the target (lower is better), and the main accuracy issue is that you set all `u_out==1` test predictions to 0.0; while expiratory rows aren’t scored, this can still hurt because the public metric implementation may still include them (and in any case it’s unnecessary risk). I keep your exact hierarchical group-mean lookup core logic, but apply it to both phases by using the already-built expiratory lookup tables (`*_exp`) for `u_out==1` rows instead of hard-coding 0.0. This is a minimal change (no new features, no model changes) and should move the score down toward the target while keeping runtime reasonable and producing the same `submission.csv`. Everything else (paths, key quantization, fallbacks, output schema) stays the same.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np


def find_file(filename: str, root_candidates: list[str]) -> str:
    for root in root_candidates:
        candidate = os.path.join(root, filename)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find {filename} in any of: {root_candidates}")


ROOTS = [
    "/kaggle/input/ventilator-pressure-prediction",
    "/kaggle/input",
    "/kaggle/data/ventilator-pressure-prediction",
    "/kaggle/data",
]

train_path = find_file("train.csv", ROOTS)
test_path = find_file("test.csv", ROOTS)
sample_path = find_file("sample_submission.csv", ROOTS)

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sub = pd.read_csv(sample_path)

required_train = {"R", "C", "time_step", "u_in", "u_out", "pressure", "breath_id"}
required_test = {"id", "R", "C", "time_step", "u_in", "u_out", "breath_id"}
if not required_train.issubset(train.columns):
    raise ValueError(
        f"train.csv missing columns: {sorted(required_train - set(train.columns))}"
    )
if not required_test.issubset(test.columns):
    raise ValueError(
        f"test.csv missing columns: {sorted(required_test - set(test.columns))}"
    )



## === cell 1
if "pressure" in train.columns:
    train["pressure"] = train["pressure"].astype(np.float32)

TIME_SCALE = 100  # 0.01s resolution
UIN_SCALE = 20  # 0.05 u_in resolution


def _half_up_int(x: np.ndarray, scale: int) -> np.ndarray:
    return np.floor(x * scale + 0.5).astype(np.int32)


def add_robust_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    t = df["time_step"].to_numpy(dtype=np.float64)
    u = df["u_in"].to_numpy(dtype=np.float64)

    df["time_i"] = _half_up_int(t, TIME_SCALE)
    df["u_in_i"] = _half_up_int(u, UIN_SCALE)

    return df


train_k = add_robust_keys(train)
test_k = add_robust_keys(test)

train_insp = train_k[train_k["u_out"] == 0]
train_exp = train_k[train_k["u_out"] == 1]

key1 = ["R", "C", "time_i", "u_in_i", "u_out"]
key_mid = ["R", "C", "time_i", "u_out"]
key2 = ["R", "C", "time_i"]

key3 = ["R", "C", "u_out", "u_in_i"]
key3b = ["R", "C", "u_in_i"]
key4 = ["R", "C", "u_out"]

key2b = ["R", "C", "u_out", "u_in_i"]

grp1_insp = train_insp.groupby(key1, sort=False)["pressure"].mean()
grp_mid_insp = train_insp.groupby(key_mid, sort=False)["pressure"].mean()
grp2_insp = train_insp.groupby(key2, sort=False)["pressure"].mean()
grp2b_insp = train_insp.groupby(key2b, sort=False)["pressure"].mean()
grp3_insp = train_insp.groupby(key3, sort=False)["pressure"].mean()
grp3b_insp = train_insp.groupby(key3b, sort=False)["pressure"].mean()

grp1_exp = train_exp.groupby(key1, sort=False)["pressure"].mean()
grp_mid_exp = train_exp.groupby(key_mid, sort=False)["pressure"].mean()
grp2_exp = train_exp.groupby(key2, sort=False)["pressure"].mean()
grp2b_exp = train_exp.groupby(key2b, sort=False)["pressure"].mean()
grp3_exp = train_exp.groupby(key3, sort=False)["pressure"].mean()
grp3b_exp = train_exp.groupby(key3b, sort=False)["pressure"].mean()

grp4 = train_k.groupby(key4, sort=False)["pressure"].mean()
global_mean = float(train_k["pressure"].mean())

pred = np.full(len(test_k), np.nan, dtype=np.float32)


def fill_predictions_for_mask(row_mask, g1, gmid, g2, g2b, g3, g3b):
    p = (
        test_k.loc[row_mask, key1]
        .merge(g1.rename("pred").reset_index(), on=key1, how="left")["pred"]
        .to_numpy()
    )

    m = np.isnan(p)
    if m.any():
        p_mid = (
            test_k.loc[row_mask, key_mid]
            .merge(gmid.rename("pred").reset_index(), on=key_mid, how="left")["pred"]
            .to_numpy()
        )
        p[m] = p_mid[m]

    m = np.isnan(p)
    if m.any():
        p2 = (
            test_k.loc[row_mask, key2]
            .merge(g2.rename("pred").reset_index(), on=key2, how="left")["pred"]
            .to_numpy()
        )
        p[m] = p2[m]

    m = np.isnan(p)
    if m.any():
        p2b = (
            test_k.loc[row_mask, key2b]
            .merge(g2b.rename("pred").reset_index(), on=key2b, how="left")["pred"]
            .to_numpy()
        )
        p[m] = p2b[m]

    m = np.isnan(p)
    if m.any():
        p3 = (
            test_k.loc[row_mask, key3]
            .merge(g3.rename("pred").reset_index(), on=key3, how="left")["pred"]
            .to_numpy()
        )
        p[m] = p3[m]

    m = np.isnan(p)
    if m.any():
        p3b = (
            test_k.loc[row_mask, key3b]
            .merge(g3b.rename("pred").reset_index(), on=key3b, how="left")["pred"]
            .to_numpy()
        )
        p[m] = p3b[m]

    return p


mask_insp = test_k["u_out"].to_numpy() == 0
mask_exp = ~mask_insp

if mask_insp.any():
    pred[mask_insp] = fill_predictions_for_mask(
        mask_insp,
        grp1_insp,
        grp_mid_insp,
        grp2_insp,
        grp2b_insp,
        grp3_insp,
        grp3b_insp,
    ).astype(np.float32)

if mask_exp.any():
    pred[mask_exp] = fill_predictions_for_mask(
        mask_exp,
        grp1_exp,
        grp_mid_exp,
        grp2_exp,
        grp2b_exp,
        grp3_exp,
        grp3b_exp,
    ).astype(np.float32)

mask = np.isnan(pred)
if mask.any():
    pred4 = (
        test_k.loc[mask, key4]
        .merge(grp4.rename("pred").reset_index(), on=key4, how="left")["pred"]
        .to_numpy()
    )
    pred[mask] = pred4

pred = np.where(np.isnan(pred), global_mean, pred).astype(np.float32)



## === cell 2
if len(sub) != len(test):
    sub = sub.merge(test[["id"]], on="id", how="right")

pred_by_id = pd.Series(pred, index=test["id"].values)
sub["pressure"] = sub["id"].map(pred_by_id).astype(np.float32)
sub["pressure"] = sub["pressure"].fillna(global_mean).astype(np.float32)

sub[["id", "pressure"]].to_csv("submission.csv", index=False)

sub.head(5)
