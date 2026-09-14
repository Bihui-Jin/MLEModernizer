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

0.1410670881270352

# 6. Current score

6.37989

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.43394) has done: 'I remove the hard dependency on four external Kaggle Dataset submissions that aren’t present in your environment (the cause of the FileNotFoundError) and replace it with a local, self-contained baseline that reads only `train.csv`/`test.csv` from the provided competition dataset. To keep changes minimal while producing a non-trivial score (better than all-zeros), I use a simple groupwise median pressure prior by `(R, C, time_step)` computed from train and merged onto test, with a safe fallback to the global median if any combination is missing. This preserves evaluation semantics (predict pressure per timestep) and guarantees an end-to-end run that writes a valid `submission.csv` with `id,pressure`. I also make the input path robust by checking both common locations shown in your file tree.'
- What this solution (achieved 6.27399) has done: 'Your current approach (median pressure prior by `(R,C,time_step)`) is leaving a lot of error because pressure depends strongly on the control inputs and recent history, not just the timestamp. To move the MAE down toward your target while preserving the same “groupwise-median prior” core logic, I minimally enrich the grouping keys to include `u_out` and a discretized/rounded `u_in` (so we still use a fast lookup table from train). I also add a simple hierarchical fallback (full key → without `u_in` → without controls → global median) to avoid NaNs and reduce brittleness. These are small, local changes that usually yield a large MAE drop on this competition without changing the overall pipeline structure.'
- What this solution (achieved 6.20794) has done: 'Your current lookup-table baseline is still missing most of the signal because pressure depends heavily on the recent history within each breath. To move the MAE down toward your target while preserving the same “groupwise median prior + hierarchical fallback” core logic, I add a few minimal, fast, history-derived keys (lagged `u_in`, cumulative `u_in`, and lagged `u_out`) computed per `breath_id` for both train and test. I then build medians on an augmented key set using these features (plus `R,C,time_step,u_out,u_in_r`) and keep a strict fallback chain to your existing priors and global median to avoid NaNs. This remains a pure aggregation/merge solution (no model training loops) and still writes a valid `submission.csv`.'
- What this solution (achieved 6.24398) has done: 'Your current lookup-table baseline is still far from the target MAE, so we need a small but meaningful signal increase without changing the overall “groupwise median prior + hierarchical fallback” approach. The minimal improvement here is to add one more short-history key (`u_in_lag2`) and replace the very coarse `u_in_cumsum` binning with a similarly cheap but more relevant within-breath state key: `u_in_diff1` (first difference), both discretized to keep the same aggregation/merge logic. We keep your existing fallback chain intact, only inserting a slightly richer top-priority table so predictions fall back safely when keys are missing. This should reduce error materially while still being fast and fully deterministic, and it continue to write a valid `submission.csv`.'
- What this solution (achieved 6.20781) has done: 'Your current median-lookup baseline is still far from the target MAE, so we need a small signal boost while keeping the exact same “groupwise median prior + hierarchical fallback” core logic. The most leverage with minimal change is to (1) add a slightly longer short-history state (`u_in_lag3`) and (2) add a within-breath “state” proxy (`u_in_cumsum`) but discretized, so it remains a cheap lookup-table feature rather than a model. To avoid sparsity explosions (which can worsen MAE), I also add a new intermediate fallback table that uses these extra keys but drops the noisiest ones, keeping your existing fallback chain intact. This stays fully deterministic, fast, and still writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 6.20769) has done: 'Your current lookup-table median prior is already very stable, but it’s still far from the target because it doesn’t enforce two key competition-specific constraints: pressure takes values on a fixed discrete grid, and expiratory-phase timesteps (u_out=1) are not scored. With minimal disruption to your core “groupwise median + hierarchical fallback” logic, I (1) snap predictions to the nearest allowed pressure level learned from train (usually a large MAE reduction on this competition) and (2) overwrite u_out==1 predictions with a simple, safe within-breath carry-forward from the last inspiratory prediction (helps avoid arbitrary errors in the unscored region without changing the model). These are pure post-processing steps and keep all your aggregation/merge logic intact while moving MAE materially closer to your target. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 6.10646) has done: 'Your current MAE is far above the target, so we need a meaningful (but still minimal) improvement. We keep the exact same “groupwise-median lookup + hierarchical fallback + snap-to-grid + u_out handling” pipeline, but make one high-impact adjustment: compute medians on *only inspiratory rows (u_out==0)*, because expiratory pressures are irrelevant to the metric and mixing phases degrades the lookup tables. Then, for u_out==1 in test, we keep your existing within-breath forward-fill post-processing. This is a small, localized change that typically moves this competition’s MAE down a lot while preserving your core logic and runtime.'
- What this solution (achieved 6.10646) has done: 'Your current lookup-table baseline is still far from the target (MAE needs to go down a lot), and the most impactful minimal fix that preserves your exact “groupwise median lookup + hierarchical fallback + snap-to-grid + u_out handling” logic is to ensure the lookup tables are built from *only inspiratory rows* and that the join keys are consistent with that (i.e., drop `u_out` from the inspiratory priors, since it’s constant in train_insp). This reduces table sparsity/mismatch and prevents accidental key conflicts that can force unnecessary fallbacks to weaker priors. I also add a single extra (very safe) fallback prior keyed by `(R,C,time_step_r,u_in_r)` (without u_out) to capture control signal without phase, which should improve MAE while keeping the same aggregation/merge approach. All paths, output schema, and the snap-to-grid + forward-fill behavior remain unchanged, and it still writes a valid `submission.csv`.'
- What this solution (achieved 5.68952) has done: 'We need to move your MAE down (lower is better) from 6.10646 toward 0.141, so we must materially improve while keeping the same lookup-table median + fallback + snapping core logic. The biggest issue is that several “priors” are duplicates (e.g., `p_full` and `p_uin_only`, `p_no_uin` and `p_base`), which wastes fallback stages and reduces effective coverage. I minimally fix this by replacing the redundant priors with two genuinely different, low-risk tables: one keyed by `(R,C,u_in_r,time_step_r)` (strong) and one keyed by `(R,C,u_in_r)` (weak but useful), keeping the same hierarchical fill strategy. Everything else (history features, inspiratory-only training for priors, snap-to-grid, u_out forward-fill, and submission writing) stays the same.'
- What this solution (achieved 5.64927) has done: 'Your current score (5.6895 MAE) is far worse than the target (0.1411), so we need a meaningful improvement, but still within your existing “groupwise median lookup + hierarchical fallback + snap-to-grid + u_out forward-fill” core logic. The biggest low-risk gain is to make the lookup tables much less sparse by discretizing the high-cardinality history keys more aggressively (especially `u_in_cumsum_r` and the lag/diff roundings), so fewer test rows fall back to weak priors. I also add one extra intermediate prior keyed by `(R,C,time_step_r,u_in_r,u_in_lag1_r)` (still the same median-lookup approach) to improve coverage between your strong and weak tables. These are local changes to the aggregation keys and fallback chain only; the training data used (inspiratory-only), snapping to allowed pressure levels, and expiratory forward-fill behavior all remain intact.'
- What this solution (achieved 5.60647) has done: 'We need to reduce MAE (lower is better) from 5.649 toward 0.141, so the smallest meaningful improvement while preserving your exact “median lookup table + hierarchical fallback + snap-to-grid + u_out forward-fill” logic is to add a slightly stronger, still-low-sparsity prior keyed by `(R, C, time_step_r, u_in_r, u_in_cumsum_r)` and use a more stable definition of `u_in_cumsum_r` based on *delivered volume proxy* `u_in*(1-u_out)` so expiratory timesteps don’t distort the within-breath state. This keeps the same feature extraction style (simple history-derived keys) and the same aggregation/merge mechanics, but improves coverage/consistency so fewer rows fall back to weak priors. I also compute allowed pressure levels from inspiratory rows only (same grid, but avoids any edge-case contamination) and keep your existing post-processing unchanged. All paths and the submission schema remain the same, and it still writes `submission.csv`.'
- What this solution (achieved 5.57227) has done: 'Your current MAE (5.606) is far above the target (0.141, lower is better), so we should improve accuracy materially while keeping the exact same “median lookup tables + hierarchical fallback + snap-to-grid + u_out forward-fill” core logic. The biggest low-risk gain within this framework is to make the lookup keys better match the competition’s physical dynamics by adding a discretized “delivered volume” proxy (`u_in_cumsum_r`) *and* a discretized “instantaneous flow” proxy (`u_in_delta_t_r = u_in_eff * delta_time`) to the strongest priors; this improves coverage and reduces fallbacks without introducing a new model. We also compute `delta_time` per breath (from `time_step`) and keep all existing priors/fallbacks intact, only adding 1–2 stronger priors near the top of the chain. All file paths, inspiratory-only priors, snapping to allowed pressure levels, and submission writing remain unchanged.'
- What this solution (achieved 5.77459) has done: 'We need to reduce MAE (lower is better) from 5.57 toward 0.141, so the smallest meaningful step within your existing “median lookup tables + hierarchical fallback + snap-to-grid + u_out forward-fill” logic is to (1) fix the discretization so keys match the true pressure grid (the dataset’s `u_in` resolution is typically 1/50 = 0.02, so rounding `u_in` to 0.1 makes many different states collide), and (2) reduce sparsity in the strongest priors by slightly coarsening the *history* bins (lags/diff) while keeping `u_in` itself fine. This keeps your exact aggregation/merge approach, but improves lookup fidelity and coverage so fewer rows fall back to weak priors. I’m also keeping inspiratory-only priors and your existing snapping + forward-fill behavior unchanged.'
- What this solution (achieved 6.37989) has done: 'Your current MAE is far above the target (lower is better), so we should make a small but meaningful improvement without changing your core “median lookup tables + hierarchical fallback + snap-to-grid + u_out forward-fill” approach. The biggest likely bug hurting score is that your `id` ranges look wrong (showing 1–2000), which indicates your loaded CSVs may be truncated/incorrectly read; fixing the input directory resolution to always prefer the real competition folder should immediately improve score. Then, keeping everything else the same, we make `u_in` discretization data-driven (infer the actual step from train) instead of assuming 0.02, reducing key collisions/mismatches and improving lookup fidelity. All outputs/paths remain the same and the script still writes a valid `submission.csv` with `id,pressure`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd




## === cell 1
def find_comp_dir():
    candidates = [
        "/kaggle/input/ventilator-pressure-prediction",
        "/kaggle/data/ventilator-pressure-prediction",
        "../input/ventilator-pressure-prediction",
        "../data/ventilator-pressure-prediction",
        "/kaggle/input",
        "/kaggle/data",
        "../input",
        "../data",
    ]

    def has_csvs(d):
        return (
            os.path.isfile(os.path.join(d, "train.csv"))
            and os.path.isfile(os.path.join(d, "test.csv"))
            and os.path.isfile(os.path.join(d, "sample_submission.csv"))
        )

    for c in candidates:
        if os.path.exists(c):
            if has_csvs(c):
                return c
            d = os.path.join(c, "ventilator-pressure-prediction")
            if os.path.isdir(d) and has_csvs(d):
                return d

    raise FileNotFoundError(
        "Could not locate competition directory containing train.csv/test.csv."
    )


comp_dir = find_comp_dir()
train_path = os.path.join(comp_dir, "train.csv")
test_path = os.path.join(comp_dir, "test.csv")
sample_path = os.path.join(comp_dir, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sub = pd.read_csv(sample_path)

assert "id" in sub.columns and "pressure" in sub.columns
assert len(test) == len(sub), "Sample submission rows must match test rows."

if len(train) < 1_000_000 or len(test) < 100_000:
    raise RuntimeError(
        f"Loaded unexpectedly small files (train={len(train):,}, test={len(test):,}). "
        f"Resolved comp_dir={comp_dir}. Please verify the dataset paths."
    )
if train["id"].nunique() != len(train) or test["id"].nunique() != len(test):
    raise RuntimeError(
        "IDs must be unique per row; loaded data seems corrupted/truncated."
    )



## === cell 2
train_ts = train.copy()
test_ts = test.copy()


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["breath_id", "time_step"], kind="mergesort").copy()
    g = df.groupby("breath_id", sort=False)

    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0)
    df["u_in_lag3"] = g["u_in"].shift(3).fillna(0.0)

    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0).astype(int)

    df["u_in_diff1"] = g["u_in"].diff(1).fillna(0.0)

    df["delta_time"] = g["time_step"].diff(1).fillna(df["time_step"])
    df["delta_time"] = df["delta_time"].clip(lower=0.0)

    df["u_in_eff"] = df["u_in"] * (1.0 - df["u_out"].astype(float))
    df["u_in_cumsum"] = g["u_in_eff"].cumsum()

    df["u_in_delta_t"] = df["u_in_eff"] * df["delta_time"]
    df["u_in_delta_t_cumsum"] = g["u_in_delta_t"].cumsum()

    df["u_in_lag1_r"] = (df["u_in_lag1"] / 1.0).round(0) * 1.0
    df["u_in_lag2_r"] = (df["u_in_lag2"] / 1.0).round(0) * 1.0
    df["u_in_lag3_r"] = (df["u_in_lag3"] / 1.0).round(0) * 1.0
    df["u_in_diff1_r"] = (df["u_in_diff1"] / 1.0).round(0) * 1.0

    df["u_in_cumsum_r"] = (df["u_in_cumsum"] / 20.0).round(0) * 20.0

    df["u_in_delta_t_r"] = (df["u_in_delta_t"] / 0.5).round(0) * 0.5
    df["u_in_delta_t_cumsum_r"] = (df["u_in_delta_t_cumsum"] / 10.0).round(0) * 10.0

    return df


train_ts = add_history_features(train_ts)
test_ts = add_history_features(test_ts)

train_ts["time_step_r"] = train_ts["time_step"].round(5)
test_ts["time_step_r"] = test_ts["time_step"].round(5)

uin_unique = np.sort(train_ts["u_in"].unique())
uin_diffs = np.diff(uin_unique)
uin_diffs = uin_diffs[uin_diffs > 1e-12]
uin_step = float(np.quantile(uin_diffs, 0.05)) if len(uin_diffs) else 0.02
uin_step = max(1e-3, min(uin_step, 1.0))  # keep sane bounds

train_ts["u_in_r"] = (train_ts["u_in"] / uin_step).round(0) * uin_step
test_ts["u_in_r"] = (test_ts["u_in"] / uin_step).round(0) * uin_step

train_insp = train_ts[train_ts["u_out"].eq(0)].copy()
global_median = float(train_insp["pressure"].median())

group_hist = [
    "R",
    "C",
    "time_step_r",
    "u_in_r",
    "u_in_lag1_r",
    "u_in_lag2_r",
    "u_in_lag3_r",
    "u_in_diff1_r",
    "u_out_lag1",
    "u_in_cumsum_r",
]
prior_hist = (
    train_insp.groupby(group_hist, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p_hist"})
)

group_hist_dt = group_hist + ["u_in_delta_t_r", "u_in_delta_t_cumsum_r"]
prior_hist_dt = (
    train_insp.groupby(group_hist_dt, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p_hist_dt"})
)

group_hist_mid = [
    "R",
    "C",
    "time_step_r",
    "u_in_r",
    "u_in_lag1_r",
    "u_in_diff1_r",
    "u_in_cumsum_r",
]
prior_hist_mid = (
    train_insp.groupby(group_hist_mid, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p_hist_mid"})
)

group_hist_mid_dt = group_hist_mid + ["u_in_delta_t_cumsum_r"]
prior_hist_mid_dt = (
    train_insp.groupby(group_hist_mid_dt, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p_hist_mid_dt"})
)

group_rc_t_uin_l1 = ["R", "C", "time_step_r", "u_in_r", "u_in_lag1_r"]
prior_rc_t_uin_l1 = (
    train_insp.groupby(group_rc_t_uin_l1, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p_rc_t_uin_l1"})
)

group_rc_t_uin_cum = ["R", "C", "time_step_r", "u_in_r", "u_in_cumsum_r"]
prior_rc_t_uin_cum = (
    train_insp.groupby(group_rc_t_uin_cum, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p_rc_t_uin_cum"})
)

group_rc_t_uin_cumdt = ["R", "C", "time_step_r", "u_in_r", "u_in_delta_t_cumsum_r"]
prior_rc_t_uin_cumdt = (
    train_insp.groupby(group_rc_t_uin_cumdt, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p_rc_t_uin_cumdt"})
)

group_rc_t_uin = ["R", "C", "time_step_r", "u_in_r"]
prior_rc_t_uin = (
    train_insp.groupby(group_rc_t_uin, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p_rc_t_uin"})
)

group_rc_uin = ["R", "C", "u_in_r"]
prior_rc_uin = (
    train_insp.groupby(group_rc_uin, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p_rc_uin"})
)

group_rc_t = ["R", "C", "time_step_r"]
prior_rc_t = (
    train_insp.groupby(group_rc_t, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p_rc_t"})
)

test_ts = test_ts.merge(prior_hist_dt, on=group_hist_dt, how="left")
test_ts = test_ts.merge(prior_hist, on=group_hist, how="left")
test_ts = test_ts.merge(prior_hist_mid_dt, on=group_hist_mid_dt, how="left")
test_ts = test_ts.merge(prior_hist_mid, on=group_hist_mid, how="left")
test_ts = test_ts.merge(prior_rc_t_uin_l1, on=group_rc_t_uin_l1, how="left")
test_ts = test_ts.merge(prior_rc_t_uin_cumdt, on=group_rc_t_uin_cumdt, how="left")
test_ts = test_ts.merge(prior_rc_t_uin_cum, on=group_rc_t_uin_cum, how="left")
test_ts = test_ts.merge(prior_rc_t_uin, on=group_rc_t_uin, how="left")
test_ts = test_ts.merge(prior_rc_uin, on=group_rc_uin, how="left")
test_ts = test_ts.merge(prior_rc_t, on=group_rc_t, how="left")

test_ts["pressure_pred"] = test_ts["p_hist_dt"]
test_ts["pressure_pred"] = test_ts["pressure_pred"].fillna(test_ts["p_hist"])
test_ts["pressure_pred"] = test_ts["pressure_pred"].fillna(test_ts["p_hist_mid_dt"])
test_ts["pressure_pred"] = test_ts["pressure_pred"].fillna(test_ts["p_hist_mid"])
test_ts["pressure_pred"] = test_ts["pressure_pred"].fillna(test_ts["p_rc_t_uin_l1"])
test_ts["pressure_pred"] = test_ts["pressure_pred"].fillna(test_ts["p_rc_t_uin_cumdt"])
test_ts["pressure_pred"] = test_ts["pressure_pred"].fillna(test_ts["p_rc_t_uin_cum"])
test_ts["pressure_pred"] = test_ts["pressure_pred"].fillna(test_ts["p_rc_t_uin"])
test_ts["pressure_pred"] = test_ts["pressure_pred"].fillna(test_ts["p_rc_uin"])
test_ts["pressure_pred"] = test_ts["pressure_pred"].fillna(test_ts["p_rc_t"])
test_ts["pressure_pred"] = test_ts["pressure_pred"].fillna(global_median).astype(float)

pressure_levels = np.sort(train_insp["pressure"].unique()).astype(np.float64)


def snap_to_levels(pred: np.ndarray, levels: np.ndarray) -> np.ndarray:
    pred = pred.astype(np.float64)
    idx = np.searchsorted(levels, pred, side="left")
    idx0 = np.clip(idx - 1, 0, len(levels) - 1)
    idx1 = np.clip(idx, 0, len(levels) - 1)
    choose1 = np.abs(levels[idx1] - pred) <= np.abs(levels[idx0] - pred)
    out = np.where(choose1, levels[idx1], levels[idx0])
    return out


test_ts["pressure_pred"] = snap_to_levels(
    test_ts["pressure_pred"].to_numpy(), pressure_levels
)

test_ts = test_ts.sort_values(["breath_id", "time_step"], kind="mergesort").copy()
test_ts["pressure_pred_insp"] = test_ts["pressure_pred"].where(
    test_ts["u_out"].eq(0), np.nan
)
test_ts["pressure_pred_ffill"] = test_ts.groupby("breath_id", sort=False)[
    "pressure_pred_insp"
].ffill()
test_ts["pressure_pred"] = test_ts["pressure_pred"].where(
    test_ts["u_out"].eq(0),
    test_ts["pressure_pred_ffill"].fillna(global_median),
)
test_ts["pressure_pred"] = snap_to_levels(
    test_ts["pressure_pred"].to_numpy(), pressure_levels
)

test_ts = test_ts.sort_values("id", kind="mergesort").copy()
sub = sub[["id"]].merge(test_ts[["id", "pressure_pred"]], on="id", how="left")
sub = sub.rename(columns={"pressure_pred": "pressure"})
sub["pressure"] = sub["pressure"].fillna(global_median).astype(float)

sub.to_csv("submission.csv", index=False)
sub.head()
