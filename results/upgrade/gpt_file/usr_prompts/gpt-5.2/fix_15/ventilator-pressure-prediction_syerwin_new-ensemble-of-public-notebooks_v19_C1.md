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

0.1548933416344145

# 6. Current score

2.05702

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'Your notebook fails because it tries to read four external “../input/…” submission files that do not exist in this environment, so `sub_1`..`sub_4` are never defined and the blend step crashes. To keep the core “blend submissions” logic intact while making it runnable end-to-end, I add a tiny fallback that (1) loads those submissions if present, otherwise (2) uses the provided `sample_submission.csv` as a safe placeholder for the missing ones. This guarantees a correctly formatted `submission.csv` is always written with the required `id,pressure` columns. The score may be poor if all externals are missing, but the pipeline run and produce a valid submission file.'
- What this solution (achieved 4.00681) has done: 'Your current score is extremely far from the target (17.65 vs 0.155, lower is better), because the notebook is blending four external submissions that don’t exist here and falls back to the all-zero sample submission—so you are effectively submitting near-constant predictions. To move the score strongly toward the target while keeping the core “make a submission by combining predictors” logic intact, I keep the same blending structure but replace missing external submissions with a lightweight, local baseline model trained from `train.csv` and applied to `test.csv`. This baseline predict pressure from (`R`,`C`,`time_step`,`u_in`,`u_out`) using a fast median-by-bin lookup (very stable, no iterative training), which should dramatically reduce MAE. The final CSV format and paths remain the same, and if any external submissions actually exist, they still be used as before.'
- What this solution (achieved 2.30313) has done: 'Your current score is far worse than the target because the “blend” is effectively just the coarse binned-median baseline (the external submissions aren’t present), and that baseline doesn’t respect the competition’s key structure: pressure is highly dependent on breath dynamics and is only scored on inspiratory steps (`u_out==0`). To move the MAE much closer to the target while keeping your core approach (produce a submission and blend multiple predictors) intact, I replace the fallback baseline with a stronger but still lightweight, deterministic per-breath feature baseline: compute cumulative inspired volume (`cum_u_in`) per breath and predict pressure via median lookup keyed by `(R,C,u_out,time_bin,cum_u_in_bin)` with safe backoffs. I also apply a minimal metric-aligned postprocess: force predictions to 0 during expiration (`u_out==1`) and snap to the nearest valid training pressure level (this is a common, legitimate improvement for this competition and doesn’t change evaluation semantics). The blending logic and output format/path remain unchanged, and if the external submission files ever exist they still be used as before.'
- What this solution (achieved 1.67781) has done: 'Your current score is still far above the target, so we need a meaningful (but still minimal) lift in the fallback predictor, since the external blend files are missing and you’re effectively submitting only the fallback. I keep your exact pipeline structure (build fallback submission → optionally load externals → blend → write `submission.csv`) but strengthen the fallback with two small, competition-standard, non-ML feature additions: lagged `u_in` and a “hold” proxy (`u_in` when `u_out==0`) plus their cumulative integrals, then use the same median-lookup-with-backoff approach on slightly richer keys. I also make the “expiration pressure=0” step metric-aligned by applying it only after blending (so all components behave consistently), and I keep snapping to valid pressure levels (still legitimate and usually improves MAE here). These changes should move MAE notably downward toward your target without changing the overall approach or introducing training loops.'
- What this solution (achieved 1.68585) has done: 'Your current score is still far from the target, and since the external blend files aren’t present you’re effectively submitting only the fallback baseline; the biggest safe win (without changing the overall “deterministic lookup + blend + postprocess” structure) is to make the lookup less sparse and more breath-dynamics aware. I minimally extend the fallback with one additional standard dynamic proxy (`u_in_diff` and its binned version) and add a slightly broader, more reliable backoff chain that uses `(time_bin, u_in_bin, u_in_diff_bin, u_out)` before falling back to coarser groupings. I also keep your existing snapping-to-valid-pressure-levels and the metric-aligned “force 0 on u_out==1” postprocess (applied after blending as you already do). These are small, deterministic changes intended to reduce MAE noticeably (move downward toward 0.1549) while preserving the pipeline and submission format.'
- What this solution (achieved 1.68585) has done: 'Your current score is still far above the target (lower-is-better), and since the external submissions are absent you’re effectively submitting only the deterministic lookup baseline. To move MAE down without changing the core “fallback baseline → optional load externals → blend → postprocess → write CSV” structure, I minimally strengthen the fallback by adding one more breath-dynamics proxy (`cum_u_in` in addition to `cum_u_in_insp`) and using it in the highest-priority lookup keys with a safe backoff. I also make the metric-aligned “expiration pressure = 0” step more consistent by applying it inside the fallback as well (so if you ever do have externals, the final postprocess still remains the single authority after blending, but the fallback alone is less noisy). These are small, deterministic tweaks intended to reduce the gap toward the target while preserving your approach and output format.'
- What this solution (achieved 1.68585) has done: 'Your current MAE (1.68585, lower is better) is still far above the target (0.1549), and because the external blend files aren’t available you’re effectively submitting the local fallback. To move the score down substantially without changing the overall “deterministic lookup → optional blend → postprocess → write CSV” approach, I make the fallback less sparse and more dynamics-aware by adding a very small, competition-standard state proxy: cumulative exhalation time (`cum_u_out_dt`) and using it in the highest-priority lookup key with safe backoffs. This helps separate inspiratory vs expiratory trajectory shapes beyond just `u_out`, improving the median lookup quality while keeping the same grouped-median core logic. I also snap pressure levels using *inspiratory-only* pressure levels (since the metric ignores expiration), which typically reduces MAE vs mixing in expiratory zeros. The final submission format/path stays identical and still writes `submission.csv`.'
- What this solution (achieved 1.67781) has done: 'Your current score is far above the target (lower is better), and since the external blend files are not present you are effectively submitting only the fallback baseline. The smallest meaningful improvement that preserves your “deterministic grouped-median lookup + backoff + blend + postprocess” core logic is to make the lookup keys less sparse and more physically aligned by adding a per-breath “volume proxy” (`u_in * dt` cumulative) and a tiny extra backoff that uses it. I also tighten the postprocess to snap to valid inspiratory pressure levels *after* forcing `u_out==1` to 0, which avoids snapping expiratory zeros to nearby inspiratory levels. These changes keep the same overall approach, add no training loops, and should move MAE downward toward your target while still producing a valid `submission.csv`.'
- What this solution (achieved 1.36682) has done: 'Your current MAE (1.67781, lower is better) is still far above the target (0.15489), and since the external blend files are not available you’re effectively submitting only the local fallback. To move the score down while preserving your core “grouped-median lookup + backoff + blend + postprocess” logic, I make the fallback lookup less sparse and more breath-trajectory aware by adding two tiny, deterministic dynamics proxies: `u_in_lag2` (second lag) and `u_in_roll3` (3-step rolling mean), then include them in the most-specific grouping and first backoff. I also slightly refine binning for the cumulative volume proxy (`cum_v`) to better match its scale (smaller bins), which improves median table resolution without changing the approach. The blending and final postprocess (force expiration to 0, snap inspiratory to valid pressure levels) remain the same, and the code still writes a valid `submission.csv`.'
- What this solution (achieved 1.22408) has done: 'Your current MAE (1.36682; lower is better) is still far above the target (0.15489), and since the external blend files are not available you’re effectively submitting only the local grouped-median fallback. The smallest meaningful improvement without changing your core “deterministic grouped-median lookup + backoff + blend + postprocess” approach is to add one more highly-informative, cheap, per-breath dynamics proxy: the cumulative count of inspiratory steps (`step_insp`), then include it (binned) in the most-specific lookup keys and first backoff to reduce table ambiguity. This keeps the same median-lookup logic, avoids any training loop, and should move MAE downward by making time-position within inspiration explicit even when `time_step` aligns imperfectly. All I/O paths and the final submission schema (`id,pressure` written to `submission.csv`) remain unchanged.'
- What this solution (achieved 2.06814) has done: 'I fix the crash by keeping `u_out` available until after the inspiratory-only snapping step (you currently drop it with `sub = sub[["id","pressure"]]` and then try to access it). I also make the merge robust by explicitly controlling suffixes and ensuring `u_out` is present exactly once, which prevents silent column naming issues. These changes are score-neutral (they only correct dataframe logic) and ensure the notebook runs end-to-end and writes a valid `submission.csv` with the required `id,pressure` columns.'
- What this solution (achieved 2.0749) has done: 'Your current MAE is still far above the target, so the smallest score-relevant improvement (without changing your overall “grouped-median fallback → optional blend → postprocess” structure) is to make the fallback’s lookup less sparse and more aligned with breath physics. I add two standard, deterministic proxies: within-breath cumulative “area under u_in” without dt (cum_u_in_raw) and time-from-start (t_since_start), then include their binned versions in the highest-priority grouped-median keys with safe backoffs. This preserves your exact approach (no ML training loop, still median lookup + backoff + snapping) while usually reducing MAE notably for this competition. The blending logic and final postprocess remain unchanged, and it still write a valid `submission.csv`.'
- What this solution (achieved 2.05702) has done: 'Your current score (2.0749 MAE; lower is better) is far above the target (0.1549), and the biggest issue is that your strongest lookup table `g0` depends on `p_lag1_bin` but in test this bin is derived from a first-pass prediction (`pred_a`) rather than from the true previous-step pressure—this makes `g0` effectively noisy and often worse than the simpler backoffs. To move the MAE down while preserving your exact “grouped-median lookup + backoff + blend + postprocess” core logic, I replace the `pred_a`-based `p_lag1_bin` with a deterministic, locally-computable proxy computed from **previous-step controls** (`u_in_lag1` and `u_out`) using a per-step median table from train, then snapping to valid inspiratory pressure levels just like you already do. This is a minimal, deterministic change (no training loops, no architecture change) but it makes the most-specific `g0` lookup much more consistent and should reduce MAE. All I/O paths remain the same and the script still writes a valid `submission.csv` with `id,pressure`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
sub = pd.read_csv(sample_path)

train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"


def build_local_baseline_submission(
    train_csv: str, test_csv: str, sample_sub: pd.DataFrame
) -> pd.DataFrame:
    """
    Deterministic grouped-median fallback (no ML training loop), minimally strengthened.

    Change (score-relevant, minimal, preserves core grouped-median + backoff logic):
    - Compute test-time p_lag1_bin from a deterministic proxy of previous-step pressure based on
      previous-step controls (u_in_lag1 and u_out), rather than from a noisy first-pass full-model pred.
      Why: g0/g1 use p_lag1_bin heavily; using a stable proxy improves key consistency and lowers MAE.
    """
    usecols_train = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    usecols_test = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]

    train = pd.read_csv(train_csv, usecols=usecols_train)
    test = pd.read_csv(test_csv, usecols=usecols_test)

    train = train.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
        drop=True
    )
    test = test.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
        drop=True
    )

    train_dt = (
        train.groupby("breath_id", sort=False)["time_step"]
        .diff()
        .fillna(0.0)
        .astype(np.float32)
    )
    test_dt = (
        test.groupby("breath_id", sort=False)["time_step"]
        .diff()
        .fillna(0.0)
        .astype(np.float32)
    )

    train_t0 = (
        train.groupby("breath_id", sort=False)["time_step"]
        .transform("first")
        .astype(np.float32)
    )
    test_t0 = (
        test.groupby("breath_id", sort=False)["time_step"]
        .transform("first")
        .astype(np.float32)
    )
    train["t_since_start"] = (train["time_step"].astype(np.float32) - train_t0).astype(
        np.float32
    )
    test["t_since_start"] = (test["time_step"].astype(np.float32) - test_t0).astype(
        np.float32
    )

    train["u_in_lag1"] = (
        train.groupby("breath_id", sort=False)["u_in"]
        .shift(1)
        .fillna(0.0)
        .astype(np.float32)
    )
    test["u_in_lag1"] = (
        test.groupby("breath_id", sort=False)["u_in"]
        .shift(1)
        .fillna(0.0)
        .astype(np.float32)
    )

    train["u_in_lag2"] = (
        train.groupby("breath_id", sort=False)["u_in"]
        .shift(2)
        .fillna(0.0)
        .astype(np.float32)
    )
    test["u_in_lag2"] = (
        test.groupby("breath_id", sort=False)["u_in"]
        .shift(2)
        .fillna(0.0)
        .astype(np.float32)
    )

    train["u_in_roll3"] = (
        train.groupby("breath_id", sort=False)["u_in"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .astype(np.float32)
    )
    test["u_in_roll3"] = (
        test.groupby("breath_id", sort=False)["u_in"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .astype(np.float32)
    )

    train["u_in_diff"] = train["u_in"].astype(np.float32) - train["u_in_lag1"].astype(
        np.float32
    )
    test["u_in_diff"] = test["u_in"].astype(np.float32) - test["u_in_lag1"].astype(
        np.float32
    )

    train_u_in = train["u_in"].astype(np.float32)
    test_u_in = test["u_in"].astype(np.float32)

    train_uout = train["u_out"].values.astype(np.int8)
    test_uout = test["u_out"].values.astype(np.int8)

    train["u_in_insp"] = np.where(train_uout == 0, train_u_in, 0.0).astype(np.float32)
    test["u_in_insp"] = np.where(test_uout == 0, test_u_in, 0.0).astype(np.float32)

    train["cum_u_in_raw"] = (
        train_u_in.groupby(train["breath_id"], sort=False).cumsum().astype(np.float32)
    )
    test["cum_u_in_raw"] = (
        test_u_in.groupby(test["breath_id"], sort=False).cumsum().astype(np.float32)
    )

    train["cum_u_in"] = (
        (train_u_in * train_dt).groupby(train["breath_id"], sort=False).cumsum()
    )
    test["cum_u_in"] = (
        (test_u_in * test_dt).groupby(test["breath_id"], sort=False).cumsum()
    )

    train["cum_u_in_insp"] = (
        (train["u_in_insp"].values.astype(np.float32) * train_dt)
        .groupby(train["breath_id"], sort=False)
        .cumsum()
    )
    test["cum_u_in_insp"] = (
        (test["u_in_insp"].values.astype(np.float32) * test_dt)
        .groupby(test["breath_id"], sort=False)
        .cumsum()
    )

    train["v"] = (train_u_in * train_dt).astype(np.float32)
    test["v"] = (test_u_in * test_dt).astype(np.float32)
    train["cum_v"] = train["v"].groupby(train["breath_id"], sort=False).cumsum()
    test["cum_v"] = test["v"].groupby(test["breath_id"], sort=False).cumsum()

    train["cum_u_out_dt"] = (
        (train_dt * (train_uout == 1)).groupby(train["breath_id"], sort=False).cumsum()
    ).astype(np.float32)
    test["cum_u_out_dt"] = (
        (test_dt * (test_uout == 1)).groupby(test["breath_id"], sort=False).cumsum()
    ).astype(np.float32)

    train["step_insp"] = (
        pd.Series((train_uout == 0).astype(np.int16))
        .groupby(train["breath_id"], sort=False)
        .cumsum()
        .astype(np.int16)
    )
    test["step_insp"] = (
        pd.Series((test_uout == 0).astype(np.int16))
        .groupby(test["breath_id"], sort=False)
        .cumsum()
        .astype(np.int16)
    )

    train_p_lag1 = (
        train.groupby("breath_id", sort=False)["pressure"]
        .shift(1)
        .fillna(train["pressure"].median())
        .astype(np.float32)
        .values
    )
    train["p_lag1_insp"] = np.where(train_uout == 0, train_p_lag1, 0.0).astype(
        np.float32
    )

    time_bins = np.linspace(0.0, 2.73, 81)  # 80 bins
    trel_bins = np.linspace(0.0, 2.73, 81)  # 80 bins
    cumraw_bins = np.linspace(0.0, 8000.0, 161)  # 160 bins

    cum_bins = np.linspace(0.0, 300.0, 151)  # 150 bins
    cumv_bins = np.linspace(0.0, 300.0, 241)  # 240 bins
    uin_bins = np.linspace(0.0, 100.0, 51)  # 50 bins
    udiff_bins = np.linspace(-100.0, 100.0, 81)  # 80 bins
    outdt_bins = np.linspace(0.0, 2.73, 61)  # 60 bins
    step_bins = np.arange(0, 81, 1)  # 0..80 (covers full breath length)

    train["time_bin"] = np.digitize(
        train["time_step"].values, time_bins, right=True
    ).astype(np.int16)
    test["time_bin"] = np.digitize(
        test["time_step"].values, time_bins, right=True
    ).astype(np.int16)

    train["trel_bin"] = np.digitize(
        train["t_since_start"].values, trel_bins, right=True
    ).astype(np.int16)
    test["trel_bin"] = np.digitize(
        test["t_since_start"].values, trel_bins, right=True
    ).astype(np.int16)

    train["cum_raw_bin"] = np.digitize(
        train["cum_u_in_raw"].values, cumraw_bins, right=True
    ).astype(np.int16)
    test["cum_raw_bin"] = np.digitize(
        test["cum_u_in_raw"].values, cumraw_bins, right=True
    ).astype(np.int16)

    train["cum_insp_bin"] = np.digitize(
        train["cum_u_in_insp"].values, cum_bins, right=True
    ).astype(np.int16)
    test["cum_insp_bin"] = np.digitize(
        test["cum_u_in_insp"].values, cum_bins, right=True
    ).astype(np.int16)

    train["cum_bin"] = np.digitize(
        train["cum_u_in"].values, cum_bins, right=True
    ).astype(np.int16)
    test["cum_bin"] = np.digitize(test["cum_u_in"].values, cum_bins, right=True).astype(
        np.int16
    )

    train["cum_v_bin"] = np.digitize(
        train["cum_v"].values, cumv_bins, right=True
    ).astype(np.int16)
    test["cum_v_bin"] = np.digitize(test["cum_v"].values, cumv_bins, right=True).astype(
        np.int16
    )

    train["uin_bin"] = np.digitize(train["u_in"].values, uin_bins, right=True).astype(
        np.int16
    )
    test["uin_bin"] = np.digitize(test["u_in"].values, uin_bins, right=True).astype(
        np.int16
    )

    train["uin_lag1_bin"] = np.digitize(
        train["u_in_lag1"].values, uin_bins, right=True
    ).astype(np.int16)
    test["uin_lag1_bin"] = np.digitize(
        test["u_in_lag1"].values, uin_bins, right=True
    ).astype(np.int16)

    train["uin_lag2_bin"] = np.digitize(
        train["u_in_lag2"].values, uin_bins, right=True
    ).astype(np.int16)
    test["uin_lag2_bin"] = np.digitize(
        test["u_in_lag2"].values, uin_bins, right=True
    ).astype(np.int16)

    train["uin_roll3_bin"] = np.digitize(
        train["u_in_roll3"].values, uin_bins, right=True
    ).astype(np.int16)
    test["uin_roll3_bin"] = np.digitize(
        test["u_in_roll3"].values, uin_bins, right=True
    ).astype(np.int16)

    train["udiff_bin"] = np.digitize(
        train["u_in_diff"].values, udiff_bins, right=True
    ).astype(np.int16)
    test["udiff_bin"] = np.digitize(
        test["u_in_diff"].values, udiff_bins, right=True
    ).astype(np.int16)

    train["outdt_bin"] = np.digitize(
        train["cum_u_out_dt"].values, outdt_bins, right=True
    ).astype(np.int16)
    test["outdt_bin"] = np.digitize(
        test["cum_u_out_dt"].values, outdt_bins, right=True
    ).astype(np.int16)

    train["step_insp_bin"] = np.digitize(
        train["step_insp"].values, step_bins, right=True
    ).astype(np.int16)
    test["step_insp_bin"] = np.digitize(
        test["step_insp"].values, step_bins, right=True
    ).astype(np.int16)

    insp_levels = np.sort(train.loc[train_uout == 0, "pressure"].unique()).astype(
        np.float32
    )
    if insp_levels.size == 0:
        insp_levels = np.sort(train["pressure"].unique()).astype(np.float32)

    idx = np.searchsorted(insp_levels, train["p_lag1_insp"].values, side="left")
    idx = np.clip(idx, 0, len(insp_levels) - 1)
    idx0 = np.clip(idx - 1, 0, len(insp_levels) - 1)
    choose_left = np.abs(train["p_lag1_insp"].values - insp_levels[idx0]) <= np.abs(
        train["p_lag1_insp"].values - insp_levels[idx]
    )
    train["p_lag1_bin"] = np.where(choose_left, idx0, idx).astype(np.int16)

    global_median = float(train["pressure"].median())

    g0 = train.groupby(
        [
            "R",
            "C",
            "u_out",
            "time_bin",
            "trel_bin",
            "step_insp_bin",
            "cum_raw_bin",
            "cum_insp_bin",
            "cum_bin",
            "outdt_bin",
            "uin_lag1_bin",
            "uin_lag2_bin",
            "uin_roll3_bin",
            "p_lag1_bin",
        ],
        sort=False,
    )["pressure"].median()

    g1 = train.groupby(
        [
            "R",
            "C",
            "u_out",
            "time_bin",
            "trel_bin",
            "step_insp_bin",
            "cum_raw_bin",
            "cum_insp_bin",
            "outdt_bin",
            "uin_lag1_bin",
            "uin_lag2_bin",
            "uin_roll3_bin",
            "p_lag1_bin",
        ],
        sort=False,
    )["pressure"].median()

    g1v = train.groupby(
        ["R", "C", "u_out", "time_bin", "cum_v_bin", "cum_insp_bin"],
        sort=False,
    )["pressure"].median()

    g1b = train.groupby(
        ["R", "C", "u_out", "time_bin", "uin_bin", "udiff_bin"], sort=False
    )["pressure"].median()
    g2 = train.groupby(
        ["R", "C", "u_out", "time_bin", "cum_insp_bin", "outdt_bin"], sort=False
    )["pressure"].median()
    g2b = train.groupby(["R", "C", "u_out", "time_bin", "uin_bin"], sort=False)[
        "pressure"
    ].median()
    g3 = train.groupby(["R", "C", "u_out", "time_bin"], sort=False)["pressure"].median()
    g4 = train.groupby(["R", "C", "u_out"], sort=False)["pressure"].median()
    g5 = train.groupby(["R", "C"], sort=False)["pressure"].median()

    def lookup(group_series, keys_df, key_cols):
        mi = pd.MultiIndex.from_frame(keys_df[key_cols])
        return group_series.reindex(mi).to_numpy()

    g_prev = train.groupby(
        [
            "R",
            "C",
            "u_out",
            "time_bin",
            "uin_lag1_bin",
            "uin_lag2_bin",
            "uin_roll3_bin",
        ],
        sort=False,
    )["pressure"].median()

    prev_proxy = lookup(
        g_prev,
        test,
        [
            "R",
            "C",
            "u_out",
            "time_bin",
            "uin_lag1_bin",
            "uin_lag2_bin",
            "uin_roll3_bin",
        ],
    )
    m = np.isnan(prev_proxy)
    if m.any():
        g_prev_b = train.groupby(
            ["R", "C", "u_out", "time_bin", "uin_lag1_bin"], sort=False
        )["pressure"].median()
        prev_proxy[m] = lookup(
            g_prev_b, test.loc[m], ["R", "C", "u_out", "time_bin", "uin_lag1_bin"]
        )
    m = np.isnan(prev_proxy)
    if m.any():
        prev_proxy[m] = lookup(g3, test.loc[m], ["R", "C", "u_out", "time_bin"])
    m = np.isnan(prev_proxy)
    if m.any():
        prev_proxy[m] = lookup(g4, test.loc[m], ["R", "C", "u_out"])
    m = np.isnan(prev_proxy)
    if m.any():
        prev_proxy[m] = lookup(g5, test.loc[m], ["R", "C"])
    m = np.isnan(prev_proxy)
    if m.any():
        prev_proxy[m] = global_median
    prev_proxy = prev_proxy.astype(np.float32)
    prev_proxy = np.where(test_uout == 1, 0.0, prev_proxy).astype(np.float32)

    idx = np.searchsorted(insp_levels, prev_proxy, side="left")
    idx = np.clip(idx, 0, len(insp_levels) - 1)
    idx0 = np.clip(idx - 1, 0, len(insp_levels) - 1)
    choose_left = np.abs(prev_proxy - insp_levels[idx0]) <= np.abs(
        prev_proxy - insp_levels[idx]
    )
    test["p_lag1_bin"] = np.where(choose_left, idx0, idx).astype(np.int16)

    pred = lookup(
        g0,
        test,
        [
            "R",
            "C",
            "u_out",
            "time_bin",
            "trel_bin",
            "step_insp_bin",
            "cum_raw_bin",
            "cum_insp_bin",
            "cum_bin",
            "outdt_bin",
            "uin_lag1_bin",
            "uin_lag2_bin",
            "uin_roll3_bin",
            "p_lag1_bin",
        ],
    )
    mask = np.isnan(pred)
    if mask.any():
        pred[mask] = lookup(
            g1,
            test.loc[mask],
            [
                "R",
                "C",
                "u_out",
                "time_bin",
                "trel_bin",
                "step_insp_bin",
                "cum_raw_bin",
                "cum_insp_bin",
                "outdt_bin",
                "uin_lag1_bin",
                "uin_lag2_bin",
                "uin_roll3_bin",
                "p_lag1_bin",
            ],
        )
    mask = np.isnan(pred)
    if mask.any():
        pred[mask] = lookup(
            g1v,
            test.loc[mask],
            ["R", "C", "u_out", "time_bin", "cum_v_bin", "cum_insp_bin"],
        )
    mask = np.isnan(pred)
    if mask.any():
        pred[mask] = lookup(
            g1b, test.loc[mask], ["R", "C", "u_out", "time_bin", "uin_bin", "udiff_bin"]
        )
    mask = np.isnan(pred)
    if mask.any():
        pred[mask] = lookup(
            g2,
            test.loc[mask],
            ["R", "C", "u_out", "time_bin", "cum_insp_bin", "outdt_bin"],
        )
    mask = np.isnan(pred)
    if mask.any():
        pred[mask] = lookup(
            g2b, test.loc[mask], ["R", "C", "u_out", "time_bin", "uin_bin"]
        )
    mask = np.isnan(pred)
    if mask.any():
        pred[mask] = lookup(g3, test.loc[mask], ["R", "C", "u_out", "time_bin"])
    mask = np.isnan(pred)
    if mask.any():
        pred[mask] = lookup(g4, test.loc[mask], ["R", "C", "u_out"])
    mask = np.isnan(pred)
    if mask.any():
        pred[mask] = lookup(g5, test.loc[mask], ["R", "C"])
    mask = np.isnan(pred)
    if mask.any():
        pred[mask] = global_median

    pred = pred.astype(np.float32)
    pred = np.where(test_uout == 1, 0.0, pred).astype(np.float32)

    snap_mask = test_uout == 0
    pred_insp = pred[snap_mask]
    idx = np.searchsorted(insp_levels, pred_insp, side="left")
    idx = np.clip(idx, 0, len(insp_levels) - 1)
    idx0 = np.clip(idx - 1, 0, len(insp_levels) - 1)
    choose_left = np.abs(pred_insp - insp_levels[idx0]) <= np.abs(
        pred_insp - insp_levels[idx]
    )
    pred_insp = np.where(choose_left, insp_levels[idx0], insp_levels[idx]).astype(
        np.float32
    )
    pred[snap_mask] = pred_insp

    out = sample_sub[["id"]].copy()
    tmp = pd.DataFrame({"id": test["id"].values, "pressure": pred})
    out = out.merge(tmp, on="id", how="left", validate="one_to_one")
    out["pressure"] = out["pressure"].astype(np.float32).fillna(global_median)
    return out[["id", "pressure"]]


def load_submission_or_fallback(path: str, fallback: pd.DataFrame) -> pd.DataFrame:
    """
    Load a submission CSV with columns ['id','pressure'].
    If missing/unreadable, return the provided fallback (already correct length).
    """
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "pressure" not in df.columns:
                raise ValueError(f"'pressure' column not found in {path}")
            if "id" not in df.columns:
                df = df.copy()
                df["id"] = fallback["id"].values
            if len(df) != len(fallback):
                raise ValueError(
                    f"Row count mismatch for {path}: {len(df)} vs {len(fallback)}"
                )
            df = df[["id", "pressure"]]
            return df
    except Exception:
        pass
    return fallback[["id", "pressure"]].copy()


baseline_sub = build_local_baseline_submission(train_path, test_path, sub)

sub_1 = load_submission_or_fallback(
    "../input/rescaling-layer-for-discrete-output-in-tensorflow/submission.csv",
    baseline_sub,
)
sub_2 = load_submission_or_fallback(
    "../input/ventilator-pressure-prediction-lstm-gpu-infer/submission_median_round.csv",
    baseline_sub,
)
sub_3 = load_submission_or_fallback(
    "../input/a-dummy-approach-to-improve-your-score-postprocess/submission.csv",
    baseline_sub,
)
sub_4 = load_submission_or_fallback(
    "../input/ventilator-pressure-prediction-lstm-gpu-infer/submission_median.csv",
    baseline_sub,
)



## === cell 2
sub["pressure"] = (
    (sub_1["pressure"].values * 0.1)
    + (sub_2["pressure"].values * 0.4)
    + (sub_3["pressure"].values * 0.2)
    + (sub_4["pressure"].values * 0.3)
)

sub["pressure"] = (
    pd.to_numeric(sub["pressure"], errors="coerce").fillna(0.0).astype(np.float32)
)

test_uout = pd.read_csv(test_path, usecols=["id", "u_out"])
sub = sub.merge(
    test_uout, on="id", how="left", validate="one_to_one", suffixes=("", "_y")
)
if "u_out_y" in sub.columns:
    sub.drop(columns=["u_out_y"], inplace=True)

sub.loc[sub["u_out"].values == 1, "pressure"] = 0.0

train_uout_pressure = pd.read_csv(train_path, usecols=["u_out", "pressure"])
insp_levels = np.sort(
    train_uout_pressure.loc[
        train_uout_pressure["u_out"].values == 0, "pressure"
    ].unique()
).astype(np.float32)
if insp_levels.size == 0:
    insp_levels = np.sort(train_uout_pressure["pressure"].unique()).astype(np.float32)

mask_insp = sub["u_out"].values == 0
pred_insp = sub.loc[mask_insp, "pressure"].values.astype(np.float32)
idx = np.searchsorted(insp_levels, pred_insp, side="left")
idx = np.clip(idx, 0, len(insp_levels) - 1)
idx0 = np.clip(idx - 1, 0, len(insp_levels) - 1)
choose_left = np.abs(pred_insp - insp_levels[idx0]) <= np.abs(
    pred_insp - insp_levels[idx]
)
pred_insp = np.where(choose_left, insp_levels[idx0], insp_levels[idx]).astype(
    np.float32
)
sub.loc[mask_insp, "pressure"] = pred_insp

sub = sub[["id", "pressure"]]
sub.to_csv("submission.csv", index=False)

sub.head(5)
