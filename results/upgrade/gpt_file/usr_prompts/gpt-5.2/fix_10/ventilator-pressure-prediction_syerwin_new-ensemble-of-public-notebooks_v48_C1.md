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

0.1437251731308052

# 6. Current score

1.67421

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.90014) has done: 'I remove the dependency on external Kaggle dataset submissions that aren’t available in your environment (the root cause of the FileNotFoundError), and instead generate predictions using only the provided `train.csv`/`test.csv`. To keep core logic minimal and stable while producing a meaningful score, I build a simple group-based median lookup of `pressure` by `(R, C, time_step, u_out)` from the training data and apply it to the test set, with safe fallbacks to progressively coarser medians when a key is unseen. Finally, I write a valid `submission.csv` with exactly `id,pressure` and the correct row alignment.'
- What this solution (achieved 4.20628) has done: 'Your current lookup is too coarse for this competition because pressure is strongly driven by the `u_in` trajectory; ignoring `u_in` forces many different pressures to collapse to the same median, causing large MAE. To move your score down toward the target with minimal logic change, I keep the same “group-median with fallback” approach but add `u_in` to the finest-grain key (and a rounded `u_in`) so the lookup better matches the control input. I also add a second-level fallback that uses `(R, C, time_step, u_out, u_in_r)` → median before dropping `u_out`, which improves coverage without changing the overall method. The output writing/format stays identical, still producing a valid `submission.csv`.'
- What this solution (achieved 3.58052) has done: 'Your current approach is a group-median lookup with fallbacks, but it still loses a lot of signal because the `u_in` value is continuous and rounding to 0.1 can cause many near-identical control inputs to fall into different bins (sparse keys → more fallback to coarse medians → higher MAE). To move the MAE down toward your target while keeping the exact same core “median lookup + fallback” logic, I (1) slightly coarsen the `u_in` rounding to reduce key sparsity, and (2) add one extra intermediate fallback that uses `u_out` with a coarser `u_in` bin before dropping `u_out`. These are minimal, metric-aligned changes that typically improve coverage of the finest-grain medians without changing the overall method or training semantics. The submission writing and column format remain identical and a valid `submission.csv` is produced.'
- What this solution (achieved 4.01781) has done: 'Your current MAE is far above the target (lower is better), so we should improve accuracy with the smallest change that preserves the same “group-median lookup + fallback” core logic. The biggest remaining issue is sparsity/mismatch in the finest keys due to rounding and float join precision, so we (1) use a stable integer time index for joins (derived from known 0.03s step), and (2) add one extra intermediate `u_in` bin (5.0) to reduce fallback frequency without changing the approach. We also keep your existing 1.0 and 2.0 `u_in` bins, but adjust fallback order to prefer `u_out`-aware lookups first. This should move the score down (better) toward the target while staying within your method and producing the same valid `submission.csv`.'
- What this solution (achieved 4.01774) has done: 'Your current approach is still a “median lookup with fallbacks”, but it likely fails to match many rows because `u_in` is continuous and your bins (1/2/5) are too fine, pushing lots of samples into coarser fallbacks. To move the MAE down toward the target with minimal, metric-aligned change, I keep the exact same lookup+fallback core logic and add one more coarser `u_in` bin (step 10) and corresponding `u_out`-aware and `u_out`-agnostic median tables. This reduces sparsity while still leveraging `u_in` signal, which should improve coverage at higher-priority levels and lower the MAE. I also ensure the merge keys are type-stable by explicitly casting the binned `u_in` columns to float64 in both train/test to avoid rare join mismatches.'
- What this solution (achieved 4.01105) has done: 'Your current MAE (4.01774, lower-is-better) is still far above the target (0.1437), so we should make a small, low-risk improvement that keeps the same “group-median lookup + fallback” core logic. The most impactful missing signal you can add without changing the approach is the within-breath time position, because pressure dynamics are highly repeatable by step index even when `time_step` rounding/joins are stable. I add `step` (0–79 per `breath_id`) as an extra key in the highest-priority median tables (with and without `u_out`), while keeping all existing tables and fallbacks intact. This tends to reduce ambiguity at identical `(R,C,t_idx,u_in_bin,...)` combinations and should move MAE down toward your target without changing evaluation semantics or requiring any ML training.'
- What this solution (achieved 1.68796) has done: 'Your current MAE is far above the target (lower is better), so we should make a small improvement that keeps the same “group-median lookup + fallback” core logic. The biggest low-risk gain left is to use a *within-breath cumulative signal* that better captures the pressure trajectory without changing modeling style: add `u_in_cum` (cumulative sum of `u_in` within each breath) and use a binned version of it in the highest-priority median tables. This retains the same training semantics (pure aggregation on train, merge on test) while providing extra information beyond instantaneous `u_in` that should reduce ambiguity and lower MAE. I also keep all your existing tables and fallbacks intact, simply inserting the new, higher-priority lookups before the current ones.'
- What this solution (achieved 1.6883) has done: 'Your current lookup is already fairly rich, but it still wastes a lot of signal because the competition only scores the inspiratory phase (`u_out==0`) and your medians mix inspiratory and expiratory behavior at many fallback levels. I keep the exact same “group-median with fallbacks” core logic, but compute all median tables **only from inspiratory rows in train** (and if a key is missing, fall back to the existing coarser inspiratory tables). Additionally, because expiratory rows are not scored, I set test predictions to a safe constant (global inspiratory median) whenever `u_out==1`, which avoids noisy/uninformative matches without changing the approach. These are minimal, metric-aligned changes that should reduce MAE from 1.68796 toward your target.'
- What this solution (achieved 1.67421) has done: 'Your current score (1.6883, lower-is-better) is still far above the target (0.1437), so we should improve accuracy with the smallest change that keeps the same “train medians → merge on test → fallback fill” core logic. The biggest remaining issue is that you force all `u_out==1` predictions to a constant even though these rows can still act as *context* for nearby inspiratory dynamics; instead, we keep the inspiratory-trained tables but only apply the constant override as a last-resort fallback (i.e., only when we truly have no match). Additionally, we add one more very cheap, highly-informative key that’s still an aggregation/lookup: previous-step `u_in` (and its binned version) to disambiguate identical current-step inputs, inserted as a higher-priority median table before the existing ones. These changes preserve your architecture (median tables + fallbacks), keep runtime reasonable, and should move MAE down toward the target without altering evaluation semantics or introducing ML training.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
DATA_DIR = "../input/ventilator-pressure-prediction"
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sub = pd.read_csv(sub_path)

assert {"id", "pressure"}.issubset(sub.columns)
assert "id" in test.columns
assert len(sub) == len(
    test
), f"sample_submission rows ({len(sub)}) != test rows ({len(test)})"

train.head(), test.head(), sub.head()



## === cell 2
for df in (train, test):
    df["R"] = df["R"].astype("int64")
    df["C"] = df["C"].astype("int64")
    df["u_out"] = df["u_out"].astype("int64")

    df["step"] = df.groupby("breath_id", sort=False).cumcount().astype("int16")
    df["t_idx"] = (df["time_step"] / 0.03).round().astype("int16")

    df["u_in_r"] = df["u_in"].round(0).astype("float64")  # step 1
    df["u_in_r2"] = ((df["u_in"] / 2.0).round(0) * 2.0).astype("float64")  # step 2
    df["u_in_r5"] = ((df["u_in"] / 5.0).round(0) * 5.0).astype("float64")  # step 5
    df["u_in_r10"] = ((df["u_in"] / 10.0).round(0) * 10.0).astype("float64")  # step 10

    df["u_in_cum"] = (
        df.groupby("breath_id", sort=False)["u_in"].cumsum().astype("float64")
    )
    df["u_in_cum_r20"] = ((df["u_in_cum"] / 20.0).round(0) * 20.0).astype("float64")

    df["u_in_prev"] = (
        df.groupby("breath_id", sort=False)["u_in"]
        .shift(1)
        .fillna(0.0)
        .astype("float64")
    )
    df["u_in_prev_r5"] = ((df["u_in_prev"] / 5.0).round(0) * 5.0).astype("float64")

train_insp = train.loc[train["u_out"] == 0].copy()

med_sc0 = (
    train_insp.groupby(
        ["R", "C", "step", "u_out", "u_in_r5", "u_in_cum_r20"], observed=True
    )["pressure"]
    .median()
    .rename("psc0")
    .reset_index()
)
med_sc1 = (
    train_insp.groupby(["R", "C", "step", "u_in_r5", "u_in_cum_r20"], observed=True)[
        "pressure"
    ]
    .median()
    .rename("psc1")
    .reset_index()
)

med_sp0 = (
    train_insp.groupby(
        ["R", "C", "step", "u_out", "u_in_r5", "u_in_prev_r5"], observed=True
    )["pressure"]
    .median()
    .rename("psp0")
    .reset_index()
)
med_sp1 = (
    train_insp.groupby(["R", "C", "step", "u_in_r5", "u_in_prev_r5"], observed=True)[
        "pressure"
    ]
    .median()
    .rename("psp1")
    .reset_index()
)

med_s0 = (
    train_insp.groupby(["R", "C", "step", "u_out", "u_in_r"], observed=True)["pressure"]
    .median()
    .rename("ps0")
    .reset_index()
)
med_s0b = (
    train_insp.groupby(["R", "C", "step", "u_out", "u_in_r2"], observed=True)[
        "pressure"
    ]
    .median()
    .rename("ps0b")
    .reset_index()
)
med_s0c = (
    train_insp.groupby(["R", "C", "step", "u_out", "u_in_r5"], observed=True)[
        "pressure"
    ]
    .median()
    .rename("ps0c")
    .reset_index()
)
med_s0d = (
    train_insp.groupby(["R", "C", "step", "u_out", "u_in_r10"], observed=True)[
        "pressure"
    ]
    .median()
    .rename("ps0d")
    .reset_index()
)

med_s1 = (
    train_insp.groupby(["R", "C", "step", "u_in_r"], observed=True)["pressure"]
    .median()
    .rename("ps1")
    .reset_index()
)
med_s1b = (
    train_insp.groupby(["R", "C", "step", "u_in_r2"], observed=True)["pressure"]
    .median()
    .rename("ps1b")
    .reset_index()
)
med_s1c = (
    train_insp.groupby(["R", "C", "step", "u_in_r5"], observed=True)["pressure"]
    .median()
    .rename("ps1c")
    .reset_index()
)
med_s1d = (
    train_insp.groupby(["R", "C", "step", "u_in_r10"], observed=True)["pressure"]
    .median()
    .rename("ps1d")
    .reset_index()
)

med_0 = (
    train_insp.groupby(["R", "C", "t_idx", "u_out", "u_in_r"], observed=True)[
        "pressure"
    ]
    .median()
    .rename("p0")
    .reset_index()
)
med_0b = (
    train_insp.groupby(["R", "C", "t_idx", "u_out", "u_in_r2"], observed=True)[
        "pressure"
    ]
    .median()
    .rename("p0b")
    .reset_index()
)
med_0c = (
    train_insp.groupby(["R", "C", "t_idx", "u_out", "u_in_r5"], observed=True)[
        "pressure"
    ]
    .median()
    .rename("p0c")
    .reset_index()
)
med_0d = (
    train_insp.groupby(["R", "C", "t_idx", "u_out", "u_in_r10"], observed=True)[
        "pressure"
    ]
    .median()
    .rename("p0d")
    .reset_index()
)

med_1 = (
    train_insp.groupby(["R", "C", "t_idx", "u_in_r"], observed=True)["pressure"]
    .median()
    .rename("p1")
    .reset_index()
)
med_1b = (
    train_insp.groupby(["R", "C", "t_idx", "u_in_r2"], observed=True)["pressure"]
    .median()
    .rename("p1b")
    .reset_index()
)
med_1c = (
    train_insp.groupby(["R", "C", "t_idx", "u_in_r5"], observed=True)["pressure"]
    .median()
    .rename("p1c")
    .reset_index()
)
med_1d = (
    train_insp.groupby(["R", "C", "t_idx", "u_in_r10"], observed=True)["pressure"]
    .median()
    .rename("p1d")
    .reset_index()
)

med_2 = (
    train_insp.groupby(["R", "C", "t_idx", "u_out"], observed=True)["pressure"]
    .median()
    .rename("p2")
    .reset_index()
)
med_3 = (
    train_insp.groupby(["R", "C", "t_idx"], observed=True)["pressure"]
    .median()
    .rename("p3")
    .reset_index()
)
med_4 = (
    train_insp.groupby(["R", "C"], observed=True)["pressure"]
    .median()
    .rename("p4")
    .reset_index()
)

global_med = float(train_insp["pressure"].median())

pred_df = test[
    [
        "id",
        "R",
        "C",
        "step",
        "t_idx",
        "u_out",
        "u_in_r",
        "u_in_r2",
        "u_in_r5",
        "u_in_r10",
        "u_in_cum_r20",
        "u_in_prev_r5",
    ]
].copy()

pred_df = pred_df.merge(
    med_sc0, how="left", on=["R", "C", "step", "u_out", "u_in_r5", "u_in_cum_r20"]
)
pred_df = pred_df.merge(
    med_sc1, how="left", on=["R", "C", "step", "u_in_r5", "u_in_cum_r20"]
)

pred_df = pred_df.merge(
    med_sp0, how="left", on=["R", "C", "step", "u_out", "u_in_r5", "u_in_prev_r5"]
)
pred_df = pred_df.merge(
    med_sp1, how="left", on=["R", "C", "step", "u_in_r5", "u_in_prev_r5"]
)

pred_df = pred_df.merge(med_s0, how="left", on=["R", "C", "step", "u_out", "u_in_r"])
pred_df = pred_df.merge(med_s0b, how="left", on=["R", "C", "step", "u_out", "u_in_r2"])
pred_df = pred_df.merge(med_s0c, how="left", on=["R", "C", "step", "u_out", "u_in_r5"])
pred_df = pred_df.merge(med_s0d, how="left", on=["R", "C", "step", "u_out", "u_in_r10"])

pred_df = pred_df.merge(med_s1, how="left", on=["R", "C", "step", "u_in_r"])
pred_df = pred_df.merge(med_s1b, how="left", on=["R", "C", "step", "u_in_r2"])
pred_df = pred_df.merge(med_s1c, how="left", on=["R", "C", "step", "u_in_r5"])
pred_df = pred_df.merge(med_s1d, how="left", on=["R", "C", "step", "u_in_r10"])

pred_df = pred_df.merge(med_0, how="left", on=["R", "C", "t_idx", "u_out", "u_in_r"])
pred_df = pred_df.merge(med_0b, how="left", on=["R", "C", "t_idx", "u_out", "u_in_r2"])
pred_df = pred_df.merge(med_0c, how="left", on=["R", "C", "t_idx", "u_out", "u_in_r5"])
pred_df = pred_df.merge(med_0d, how="left", on=["R", "C", "t_idx", "u_out", "u_in_r10"])

pred_df = pred_df.merge(med_1, how="left", on=["R", "C", "t_idx", "u_in_r"])
pred_df = pred_df.merge(med_1b, how="left", on=["R", "C", "t_idx", "u_in_r2"])
pred_df = pred_df.merge(med_1c, how="left", on=["R", "C", "t_idx", "u_in_r5"])
pred_df = pred_df.merge(med_1d, how="left", on=["R", "C", "t_idx", "u_in_r10"])

pred_df = pred_df.merge(med_2, how="left", on=["R", "C", "t_idx", "u_out"])
pred_df = pred_df.merge(med_3, how="left", on=["R", "C", "t_idx"])
pred_df = pred_df.merge(med_4, how="left", on=["R", "C"])

pressure_pred = (
    pred_df["psc0"]
    .fillna(pred_df["psc1"])
    .fillna(pred_df["psp0"])
    .fillna(pred_df["psp1"])
    .fillna(pred_df["ps0"])
    .fillna(pred_df["ps0b"])
    .fillna(pred_df["ps0c"])
    .fillna(pred_df["ps0d"])
    .fillna(pred_df["ps1"])
    .fillna(pred_df["ps1b"])
    .fillna(pred_df["ps1c"])
    .fillna(pred_df["ps1d"])
    .fillna(pred_df["p0"])
    .fillna(pred_df["p0b"])
    .fillna(pred_df["p0c"])
    .fillna(pred_df["p0d"])
    .fillna(pred_df["p1"])
    .fillna(pred_df["p1b"])
    .fillna(pred_df["p1c"])
    .fillna(pred_df["p1d"])
    .fillna(pred_df["p2"])
    .fillna(pred_df["p3"])
    .fillna(pred_df["p4"])
    .fillna(global_med)
    .astype("float64")
)

need_global = pd.isna(
    pred_df["psc0"]
    .fillna(pred_df["psc1"])
    .fillna(pred_df["psp0"])
    .fillna(pred_df["psp1"])
    .fillna(pred_df["ps0"])
    .fillna(pred_df["ps0b"])
    .fillna(pred_df["ps0c"])
    .fillna(pred_df["ps0d"])
    .fillna(pred_df["ps1"])
    .fillna(pred_df["ps1b"])
    .fillna(pred_df["ps1c"])
    .fillna(pred_df["ps1d"])
    .fillna(pred_df["p0"])
    .fillna(pred_df["p0b"])
    .fillna(pred_df["p0c"])
    .fillna(pred_df["p0d"])
    .fillna(pred_df["p1"])
    .fillna(pred_df["p1b"])
    .fillna(pred_df["p1c"])
    .fillna(pred_df["p1d"])
    .fillna(pred_df["p2"])
    .fillna(pred_df["p3"])
    .fillna(pred_df["p4"])
).values

pressure_pred = np.where(
    (pred_df["u_out"].values == 1) & need_global,
    global_med,
    pressure_pred,
).astype("float64")

submission = sub[["id"]].copy()
tmp = pd.DataFrame({"id": pred_df["id"].values, "pressure": pressure_pred})
submission = submission.merge(tmp, how="left", on="id")
submission["pressure"] = submission["pressure"].fillna(global_med).astype("float64")

submission.head()



## === cell 3
out_path = "submission.csv"
submission.to_csv(out_path, index=False)

assert os.path.exists(out_path)
assert list(submission.columns) == ["id", "pressure"]
assert len(submission) == len(test)

print(f"Wrote {out_path} with shape={submission.shape}")
print(submission.head(10))
