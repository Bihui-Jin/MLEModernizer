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

0.1640017072750322

# 6. Current score

2.57851

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'I fix the runtime by removing references to external Kaggle Dataset paths that don’t exist in your environment and instead load predictions only from CSVs that are actually present. To preserve the original “average multiple submissions” core logic, the script search common input locations for any `submission.csv` files and mean-ensemble all valid ones; if none are found, it safely fall back to producing the provided sample submission (so you always get a valid `.csv`). I also harden alignment by merging on `id` to avoid silent row-order mismatches and ensure the output has exactly `id,pressure` with the correct row count. This should run end-to-end and create `submission.csv` in the working directory.'
- What this solution (achieved 4.20652) has done: 'I fix the KeyError by preventing `sub.merge(...)` from creating `pressure_x/pressure_y` columns; instead I update `sub["pressure"]` directly using an id-aligned Series. I also make the join/assignment robust to any accidental duplicate ids in intermediate frames and keep the original “ensemble if any submission.csv exists, else fallback model” logic unchanged. Finally, I add a last-resort guard that recreates the `pressure` column if it was renamed/dropped unexpectedly, ensuring a valid `submission.csv` is always produced.'
- What this solution (achieved 4.89315) has done: 'Your current fallback model is leaving a lot of signal unused (especially the “inspiratory only is scored” rule and the strong per-breath structure), which keeps MAE far from the target. I keep your overall structure (ensemble existing `submission.csv` files else fallback) but make the fallback better by (1) building median lookups using only inspiratory-phase rows (`u_out==0`) to match the metric, and (2) adding a lightweight per-breath sequential feature (`u_in` cumulative integral proxy) that is standard for this competition and doesn’t change the overall “groupby-median then hierarchical backoff” approach. These are minimal, fast, purely pandas changes that should move your score substantially down toward the target without changing I/O paths or introducing new packages. The output remains a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 2.11878) has done: 'Your fallback predictor is still far from the target MAE, so I keep the same “ensemble existing `submission.csv` else fallback” structure but make the fallback better aligned to this competition: (1) build median mappings from inspiratory-only rows (`u_out==0`) and (2) snap predictions to the known discrete pressure grid from training, which is a standard minimal post-processing that reduces MAE without changing the modeling approach. I also compute `cum_u_in` using `u_in * delta_time` (a closer integral proxy than plain cumsum) while preserving your existing hierarchical backoff logic and the same I/O paths. These are lightweight pandas-only changes and should move the score substantially down toward your 0.164 target while still finishing quickly and producing a valid `submission.csv`.'
- What this solution (achieved 3.2599) has done: 'We keep your existing “ensemble `submission.csv` files else fallback pandas model” structure, but make the fallback better match the metric by learning only from inspiratory rows (`u_out==0`) *and also forcing test predictions to 0 during expiratory phase (`u_out==1`)*, which directly reduces the scored MAE because expiratory rows are ignored. To reduce mapping sparsity without changing the approach, we add a tiny extra backoff path keyed on `(R, C, time_step_r, u_in_r)` (still median lookups) and slightly refine the rounding for `cum_u_in` so more test points hit trained bins. Finally, we snap to the pressure grid using a fully vectorized nearest-neighbor method (same semantics, less overhead and fewer edge issues), keeping runtime under the limit and output format unchanged.'
- What this solution (achieved 17.65486) has done: 'Your current fallback is being hurt by one change that conflicts with the metric: forcing `pressure=0` on expiratory rows (`u_out==1`) does not help because those rows are *ignored* in scoring, but it does damage the model’s internal consistency and can worsen inspiratory predictions through backoff collisions. I remove that expiratory overwrite and instead build *separate* lookup tables for inspiratory (`u_out==0`) and expiratory (`u_out==1`) so the model can still predict expiratory pressures without contaminating the inspiratory mapping. I also include `u_out` in the “RC+time_step” backoff (previously it used a table trained only on `u_out==0` but keyed with `u_out`, which is inconsistent for `u_out==1` test rows). These are minimal pandas-only changes that preserve your core “hierarchical median lookup + snap to pressure grid” logic and should reduce MAE toward your target.'
- What this solution (achieved 3.2599) has done: 'I fix the `KeyError: 'time_step_r'` by ensuring `time_step_r` (and other rounded helper columns) are created on the full `train` dataframe before any `train.groupby([...,"time_step_r",...])` calls. This is a minimal correctness fix that unblocks execution and keeps your existing hierarchical median-lookup + snapping-to-pressure-grid core logic unchanged. I also remove a redundant duplicate lookup (`med_rc_t_phase`) that was identical to `med_rc_t_uout` to avoid confusion, without changing predictions. The script then run end-to-end and write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 4.22004) has done: 'Your current MAE (3.2599) is far worse than the target (0.164), so we need a real accuracy lift while keeping your “pandas median-lookup + hierarchical backoff + snap-to-pressure-grid” core logic intact. The biggest issue is feature sparsity/mismatch from aggressive rounding and using the raw `time_step` as a key; I make the lookup keys match the known data structure (80 steps per breath) by adding an integer `step` index and using it in the hierarchy before falling back to rounded time. I also tune rounding granularity slightly (finer `u_in`, `cum_u_in`) to increase exact-key hits without changing the approach. Everything remains vectorized pandas/numpy, runs fast, and still writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 4.22004) has done: 'To move MAE down toward your 0.164 target without changing your core “hierarchical median lookup + backoff + snap-to-pressure-grid” approach, I fix the biggest remaining mismatch: your `cum_u_in` feature is computed separately for inspiratory/expiratory subsets, which resets the cumulative integral when `u_out` changes and breaks key consistency versus test (computed on full breath). I compute `dt` and `cum_u_in` once on the full train breath sequence, then split into inspiratory/expiratory for building the same lookup tables as before. This is a minimal change (same features, same backoff order, same snapping) but should materially reduce mapping sparsity and improve inspiratory-phase accuracy. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 4.27522) has done: 'Your current score (4.22004, lower-is-better) is still far from the target (0.1640), so we should make a small but meaningful accuracy improvement without changing the overall “pandas median-lookup + hierarchical backoff + snap-to-pressure-grid” approach. The main fix is to use the known discretization: within each breath there are exactly 80 timesteps, so we can derive a stable `step` directly from `time_step` (instead of cumcount), which avoids off-by-one/ordering issues and improves key hits in your lookup tables. I also add one additional, still-lightweight backoff table keyed on `(R,C,step,u_in_r,u_out)` (median), which increases coverage when `cum_u_in_r` mismatches but the control input at a step matches. Everything else (training data usage, median tables, backoff chain style, snapping to the pressure grid, I/O paths) stays the same and it still writes a valid `submission.csv`.'
- What this solution (achieved 4.22004) has done: 'We’re far worse than the target (MAE 4.275 vs 0.164, lower-is-better), so we need a real accuracy lift while preserving your exact “pandas median-lookup + hierarchical backoff + snap-to-pressure-grid” approach. The biggest low-risk gain is to stop keying lookups on *rounded floats* (`time_step_r`) and instead use the true discrete within-breath position (80 steps), derived via `cumcount()` after sorting; this avoids step misassignment from `time_step/0.03` rounding drift and increases exact-key hits. I keep your existing feature set, tables, backoff chain, and snapping, but (minimally) recompute `step` using per-breath order and build an additional high-coverage backoff keyed on `(R,C,step,u_out)` and `(R,C,step)` (still medians) to reduce NaNs without changing semantics. This should materially reduce MAE toward your target while staying within time/memory and still producing a valid `submission.csv`.'
- What this solution (achieved 2.57851) has done: 'Your current score (4.22004 MAE, lower is better) is still far from the target (0.1640), so we need a meaningful accuracy lift while keeping your exact “pandas median-lookup + hierarchical backoff + snap-to-pressure-grid” approach. The biggest issue is that the lookups are too sparse because they rely on rounded continuous features (`u_in_r`, `cum_u_in_r`) where tiny numeric differences break exact matches; we can improve hit-rate by additionally discretizing `u_in` into stable integer “buckets” and using those buckets in the highest-priority tables. I add `u_in_b` (rounded to nearest int) and `cum_u_in_b` (rounded to 2 decimals) and insert corresponding median tables early in the backoff chain, without changing the overall architecture/semantics. This is a minimal, fast change that should move MAE downward toward your target and still produces a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import pandas as pd



## === cell 1
SAMPLE_PATH_CANDIDATES = [
    "../input/ventilator-pressure-prediction/sample_submission.csv",
    "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv",
    "../kaggle/input/ventilator-pressure-prediction/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "../input/sample_submission.csv",
    "../kaggle/input/sample_submission.csv",
]

TRAIN_PATH_CANDIDATES = [
    "../input/ventilator-pressure-prediction/train.csv",
    "/kaggle/input/ventilator-pressure-prediction/train.csv",
    "../kaggle/input/ventilator-pressure-prediction/train.csv",
    "/kaggle/data/train.csv",
    "/kaggle/input/train.csv",
    "../input/train.csv",
    "../kaggle/input/train.csv",
]

TEST_PATH_CANDIDATES = [
    "../input/ventilator-pressure-prediction/test.csv",
    "/kaggle/input/ventilator-pressure-prediction/test.csv",
    "../kaggle/input/ventilator-pressure-prediction/test.csv",
    "/kaggle/data/test.csv",
    "/kaggle/input/test.csv",
    "../input/test.csv",
    "../kaggle/input/test.csv",
]


def _pick_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


sample_path = _pick_first_existing(SAMPLE_PATH_CANDIDATES)
if sample_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in expected Kaggle input paths."
    )

train_path = _pick_first_existing(TRAIN_PATH_CANDIDATES)
test_path = _pick_first_existing(TEST_PATH_CANDIDATES)

sub = pd.read_csv(sample_path)
assert {"id", "pressure"}.issubset(
    sub.columns
), "sample_submission.csv must contain 'id' and 'pressure'."
sub = sub[["id", "pressure"]].copy()



## === cell 2
search_roots = [
    "/kaggle/input",
    "../input",
    "../kaggle/input",
]

submission_paths = []
for root in search_roots:
    if os.path.exists(root):
        submission_paths.extend(
            glob.glob(os.path.join(root, "**", "submission.csv"), recursive=True)
        )

seen = set()
submission_paths = [p for p in submission_paths if not (p in seen or seen.add(p))]

pred_dfs = []
for p in submission_paths:
    try:
        df = pd.read_csv(p)
        if not {"id", "pressure"}.issubset(df.columns):
            continue
        df = df[["id", "pressure"]].copy()
        if not df["id"].is_unique:
            df = df.drop_duplicates(subset=["id"], keep="first")
        merged = sub[["id"]].merge(df, on="id", how="left", validate="one_to_one")
        if merged["pressure"].isna().any():
            continue
        pred_dfs.append(merged)
    except Exception:
        continue

if len(pred_dfs) > 0:
    pressures = (
        pd.concat([d["pressure"] for d in pred_dfs], axis=1)
        .mean(axis=1)
        .astype("float64")
    )
    sub["pressure"] = pressures.values
else:
    if (train_path is None) or (test_path is None):
        sub["pressure"] = sub["pressure"].astype("float64")
    else:
        import numpy as np

        train = pd.read_csv(
            train_path,
            usecols=["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
        )
        test = pd.read_csv(
            test_path,
            usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"],
        )

        train = train.copy()
        test = test.copy()

        train = train.sort_values(["breath_id", "time_step"], kind="mergesort")
        test = test.sort_values(["breath_id", "time_step"], kind="mergesort")

        train["step"] = train.groupby("breath_id").cumcount().astype("int16")
        test["step"] = test.groupby("breath_id").cumcount().astype("int16")
        train["step"] = train["step"].clip(0, 79)
        test["step"] = test["step"].clip(0, 79)

        train["dt"] = train.groupby("breath_id")["time_step"].diff().fillna(0.0)
        test["dt"] = test.groupby("breath_id")["time_step"].diff().fillna(0.0)

        train["cum_u_in"] = (
            (train["u_in"] * train["dt"]).groupby(train["breath_id"]).cumsum()
        )
        test["cum_u_in"] = (
            (test["u_in"] * test["dt"]).groupby(test["breath_id"]).cumsum()
        )

        train_insp = train[train["u_out"] == 0].copy()
        train_exp = train[train["u_out"] == 1].copy()

        for df in (train, train_insp, train_exp, test):
            df["time_step_r"] = df["time_step"].round(3)
            df["u_in_r"] = df["u_in"].round(2)
            df["cum_u_in_r"] = df["cum_u_in"].round(3)

            df["u_in_b"] = df["u_in"].round(0).astype("int16")
            df["cum_u_in_b"] = df["cum_u_in"].round(2)

        key_full = ["R", "C", "step", "u_in_r", "u_out", "cum_u_in_r"]

        med_full_insp = (
            train_insp.groupby(key_full, sort=False)["pressure"]
            .median()
            .rename("p_full_insp")
            .reset_index()
        )
        med_full_exp = (
            train_exp.groupby(key_full, sort=False)["pressure"]
            .median()
            .rename("p_full_exp")
            .reset_index()
        )

        key_full_b = ["R", "C", "step", "u_in_b", "u_out", "cum_u_in_b"]
        med_fullb_insp = (
            train_insp.groupby(key_full_b, sort=False)["pressure"]
            .median()
            .rename("p_fullb_insp")
            .reset_index()
        )
        med_fullb_exp = (
            train_exp.groupby(key_full_b, sort=False)["pressure"]
            .median()
            .rename("p_fullb_exp")
            .reset_index()
        )

        med_rc_step_cum_insp = (
            train_insp.groupby(["R", "C", "step", "cum_u_in_r"], sort=False)["pressure"]
            .median()
            .rename("p_rc_step_cum_insp")
            .reset_index()
        )
        med_rc_step_cum_exp = (
            train_exp.groupby(["R", "C", "step", "cum_u_in_r"], sort=False)["pressure"]
            .median()
            .rename("p_rc_step_cum_exp")
            .reset_index()
        )

        med_rc_step_cumb_insp = (
            train_insp.groupby(["R", "C", "step", "cum_u_in_b"], sort=False)["pressure"]
            .median()
            .rename("p_rc_step_cumb_insp")
            .reset_index()
        )
        med_rc_step_cumb_exp = (
            train_exp.groupby(["R", "C", "step", "cum_u_in_b"], sort=False)["pressure"]
            .median()
            .rename("p_rc_step_cumb_exp")
            .reset_index()
        )

        med_rc_step_uin_insp = (
            train_insp.groupby(["R", "C", "step", "u_in_r"], sort=False)["pressure"]
            .median()
            .rename("p_rc_step_uin_insp")
            .reset_index()
        )
        med_rc_step_uin_exp = (
            train_exp.groupby(["R", "C", "step", "u_in_r"], sort=False)["pressure"]
            .median()
            .rename("p_rc_step_uin_exp")
            .reset_index()
        )

        med_rc_step_uinb_insp = (
            train_insp.groupby(["R", "C", "step", "u_in_b"], sort=False)["pressure"]
            .median()
            .rename("p_rc_step_uinb_insp")
            .reset_index()
        )
        med_rc_step_uinb_exp = (
            train_exp.groupby(["R", "C", "step", "u_in_b"], sort=False)["pressure"]
            .median()
            .rename("p_rc_step_uinb_exp")
            .reset_index()
        )

        med_rc_step_uin_uout = (
            train.groupby(["R", "C", "step", "u_in_r", "u_out"], sort=False)["pressure"]
            .median()
            .rename("p_rc_step_uin_uout")
            .reset_index()
        )

        med_rc_step_uinb_uout = (
            train.groupby(["R", "C", "step", "u_in_b", "u_out"], sort=False)["pressure"]
            .median()
            .rename("p_rc_step_uinb_uout")
            .reset_index()
        )

        med_rc_step_uout = (
            train.groupby(["R", "C", "step", "u_out"], sort=False)["pressure"]
            .median()
            .rename("p_rc_step_uout")
            .reset_index()
        )

        med_rc_step = (
            train.groupby(["R", "C", "step"], sort=False)["pressure"]
            .median()
            .rename("p_rc_step")
            .reset_index()
        )

        med_rc_t_uout = (
            train.groupby(["R", "C", "time_step_r", "u_out"], sort=False)["pressure"]
            .median()
            .rename("p_rc_t_uout")
            .reset_index()
        )

        med_rc_insp = (
            train_insp.groupby(["R", "C"], sort=False)["pressure"]
            .median()
            .rename("p_rc_insp")
            .reset_index()
        )
        med_rc_exp = (
            train_exp.groupby(["R", "C"], sort=False)["pressure"]
            .median()
            .rename("p_rc_exp")
            .reset_index()
        )

        global_med_insp = float(train_insp["pressure"].median())
        global_med_exp = (
            float(train_exp["pressure"].median()) if len(train_exp) else global_med_insp
        )

        pred = test[
            [
                "id",
                "R",
                "C",
                "step",
                "time_step_r",
                "u_in_r",
                "u_in_b",
                "u_out",
                "cum_u_in_r",
                "cum_u_in_b",
            ]
        ].merge(med_full_insp, on=key_full, how="left")
        pred = pred.merge(med_full_exp, on=key_full, how="left")

        pred = pred.merge(med_fullb_insp, on=key_full_b, how="left")
        pred = pred.merge(med_fullb_exp, on=key_full_b, how="left")

        pred = pred.merge(
            med_rc_step_cum_insp, on=["R", "C", "step", "cum_u_in_r"], how="left"
        )
        pred = pred.merge(
            med_rc_step_cum_exp, on=["R", "C", "step", "cum_u_in_r"], how="left"
        )

        pred = pred.merge(
            med_rc_step_cumb_insp, on=["R", "C", "step", "cum_u_in_b"], how="left"
        )
        pred = pred.merge(
            med_rc_step_cumb_exp, on=["R", "C", "step", "cum_u_in_b"], how="left"
        )

        pred = pred.merge(
            med_rc_step_uin_insp, on=["R", "C", "step", "u_in_r"], how="left"
        )
        pred = pred.merge(
            med_rc_step_uin_exp, on=["R", "C", "step", "u_in_r"], how="left"
        )

        pred = pred.merge(
            med_rc_step_uinb_insp, on=["R", "C", "step", "u_in_b"], how="left"
        )
        pred = pred.merge(
            med_rc_step_uinb_exp, on=["R", "C", "step", "u_in_b"], how="left"
        )

        pred = pred.merge(
            med_rc_step_uin_uout, on=["R", "C", "step", "u_in_r", "u_out"], how="left"
        )
        pred = pred.merge(
            med_rc_step_uinb_uout,
            on=["R", "C", "step", "u_in_b", "u_out"],
            how="left",
        )

        pred = pred.merge(med_rc_step_uout, on=["R", "C", "step", "u_out"], how="left")
        pred = pred.merge(med_rc_step, on=["R", "C", "step"], how="left")
        pred = pred.merge(
            med_rc_t_uout, on=["R", "C", "time_step_r", "u_out"], how="left"
        )

        pred = pred.merge(med_rc_insp, on=["R", "C"], how="left")
        pred = pred.merge(med_rc_exp, on=["R", "C"], how="left")

        uout = pred["u_out"].to_numpy()

        p_insp = pred["p_full_insp"]
        p_insp = p_insp.fillna(pred["p_fullb_insp"])
        p_insp = p_insp.fillna(pred["p_rc_step_cum_insp"])
        p_insp = p_insp.fillna(pred["p_rc_step_cumb_insp"])
        p_insp = p_insp.fillna(pred["p_rc_step_uin_insp"])
        p_insp = p_insp.fillna(pred["p_rc_step_uinb_insp"])
        p_insp = p_insp.fillna(pred["p_rc_step_uin_uout"])
        p_insp = p_insp.fillna(pred["p_rc_step_uinb_uout"])
        p_insp = p_insp.fillna(pred["p_rc_step_uout"])
        p_insp = p_insp.fillna(pred["p_rc_step"])
        p_insp = p_insp.fillna(pred["p_rc_t_uout"])
        p_insp = p_insp.fillna(pred["p_rc_insp"])
        p_insp = p_insp.fillna(global_med_insp).astype("float64")

        p_exp = pred["p_full_exp"]
        p_exp = p_exp.fillna(pred["p_fullb_exp"])
        p_exp = p_exp.fillna(pred["p_rc_step_cum_exp"])
        p_exp = p_exp.fillna(pred["p_rc_step_cumb_exp"])
        p_exp = p_exp.fillna(pred["p_rc_step_uin_exp"])
        p_exp = p_exp.fillna(pred["p_rc_step_uinb_exp"])
        p_exp = p_exp.fillna(pred["p_rc_step_uin_uout"])
        p_exp = p_exp.fillna(pred["p_rc_step_uinb_uout"])
        p_exp = p_exp.fillna(pred["p_rc_step_uout"])
        p_exp = p_exp.fillna(pred["p_rc_step"])
        p_exp = p_exp.fillna(pred["p_rc_t_uout"])
        p_exp = p_exp.fillna(pred["p_rc_exp"])
        p_exp = p_exp.fillna(global_med_exp).astype("float64")

        p = pd.Series(
            np.where(uout == 0, p_insp.to_numpy(), p_exp.to_numpy()), index=pred.index
        ).astype("float64")

        pressure_grid = pd.Series(train["pressure"].unique()).sort_values().to_numpy()
        pvals = p.to_numpy()

        idx = np.searchsorted(pressure_grid, pvals, side="left")
        idx = np.clip(idx, 0, len(pressure_grid) - 1)
        left_idx = np.clip(idx - 1, 0, len(pressure_grid) - 1)

        right_val = pressure_grid[idx]
        left_val = pressure_grid[left_idx]
        choose_left = np.abs(pvals - left_val) <= np.abs(pvals - right_val)
        p_snapped = right_val.copy()
        p_snapped[choose_left] = left_val[choose_left]
        p_snapped = p_snapped.astype("float64")

        pred_id_pressure = pd.DataFrame(
            {"id": pred["id"].values, "pressure": p_snapped}
        )
        if not pred_id_pressure["id"].is_unique:
            pred_id_pressure = pred_id_pressure.drop_duplicates(
                subset=["id"], keep="first"
            )

        aligned = sub[["id"]].merge(
            pred_id_pressure, on="id", how="left", validate="one_to_one"
        )
        assert len(aligned) == len(sub), "Alignment sanity check failed."
        assert aligned["pressure"].notna().all(), "Prediction contained NaNs."
        sub["pressure"] = aligned["pressure"].astype("float64").values



## === cell 3
if "pressure" not in sub.columns:
    pressure_cols = [c for c in sub.columns if c.lower().startswith("pressure")]
    if len(pressure_cols) == 1:
        sub = sub.rename(columns={pressure_cols[0]: "pressure"})
    else:
        sub["pressure"] = 0.0

sub = sub[["id", "pressure"]].copy()
assert sub["id"].is_unique, "Submission 'id' must be unique."
assert len(sub) > 0, "Submission must have rows."
assert sub["pressure"].notna().all(), "Submission 'pressure' must not contain NaNs."

sub.to_csv("submission.csv", index=False)
sub.head(5)
