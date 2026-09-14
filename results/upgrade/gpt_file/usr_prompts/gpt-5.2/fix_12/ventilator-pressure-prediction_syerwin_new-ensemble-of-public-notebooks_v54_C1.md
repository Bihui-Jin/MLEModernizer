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

0.1437022590987769

# 6. Current score

1.87864

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'Your notebook fails because it tries to ensemble submission files from other Kaggle datasets that are not present in this environment, so `pd.read_csv` raises `FileNotFoundError` and downstream variables are undefined. I keep the same core “blend multiple submissions by fixed weights” logic, but make it robust: only load blend components that actually exist, renormalize weights over the available files, and fall back to a safe baseline (all zeros from `sample_submission`) if none are found. This run end-to-end and always write a valid `submission.csv` with the required `id,pressure` columns. Since your current score is “Not yielded”, the priority is producing a valid submission; any score improvement beyond that is constrained by available external files.'
- What this solution (achieved 8.32522) has done: 'Your current score (17.65486, lower-is-better) is far worse than the target (0.1437), and the main reason is that the script falls back to predicting all zeros because the external blend submission files are not available in this environment. To move the score strongly toward the target while keeping core logic minimal, I keep the “make a submission from available sources” structure but replace the unavailable ensemble inputs with a simple, local, leakage-free baseline model trained on `train.csv` and applied to `test.csv`. Specifically, I compute the mean `pressure` for each `(R, C, time_step, u_in, u_out)` pattern in the training data and use it to predict test rows, with a fallback to the global mean if an exact pattern is unseen. This is a small, fast change that preserves evaluation semantics and drastically improve over all-zeros.'
- What this solution (achieved 3.97014) has done: 'Your baseline is failing mainly because it tries to match on exact floating-point `time_step` and continuous `u_in`, which causes many unseen combinations in test and forces lots of fallbacks to the global mean (hurting MAE). I keep the exact same “groupby-mean lookup then merge then fillna” core logic, but make the key more matchable by (1) rounding `time_step` and `u_in` to a fixed precision before grouping/merging, and (2) adding a second-stage, slightly coarser fallback mean map before finally using the global mean. This should reduce the number of missing merges and move your score substantially toward the target without changing the overall approach. The output still be a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 3.9603) has done: 'Your current score (3.97 MAE, lower-is-better) is far above the target (0.1437), and the main issue is that the lookup baseline is still too “sparse”: many test rows won’t find a match even after rounding, so they fall back to overly-generic means. I keep the exact same core logic (groupby-mean maps → merge → staged fillna) but add two additional, slightly coarser fallback maps that are still local and cheap: one dropping `time_step` (captures pressure dependence on controls/RC), and one using only `(R,C,time_step_r,u_out)` (captures the time trajectory by lung attributes). I also use `float32` for the rounded keys to reduce merge mismatches due to dtype differences and keep runtime/memory stable. The script still run end-to-end and write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 3.39132) has done: 'Your current MAE (3.9603, lower-is-better) is still far above the target (0.1437), and most of the gap is likely because the lookup keys don’t capture the sequential “state” of the breath (pressure depends strongly on recent u_in history and accumulated volume), so many situations share the same coarse keys but have different pressures. Keeping the same core “groupby-mean maps → merge → staged fillna” logic, I add two minimal, cheap state-like features (`u_in` cumulative sum and lag-1 `u_in`) computed per breath using the existing time order (grouped by `(R,C,breath_id)`), then build an additional higher-priority mean map keyed on these features plus `time_step_r` and `u_out`. This should reduce ambiguity and improve matching without changing the overall approach or introducing any training loop/model. The rest of your fallback chain remains intact, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 3.40502) has done: 'To move your MAE down toward the target while preserving your core “groupby-mean maps → merge → staged fillna” logic, I fix the biggest issue in the current state features: you’re grouping lag/cumsum by `["R","C","breath_id"]`, which can mix multiple breaths that share `(R,C,breath_id)` ordering assumptions and is unnecessary—state should be per `breath_id` only. Then I add one more minimal, high-signal state feature (`u_in_diff1`, per-breath first difference) and include it in the highest-priority lookup key, which helps disambiguate similar cumulative/lag values without changing the overall approach. Finally, I keep all your existing fallback maps intact and only adjust dtypes/rounding consistently to reduce merge miss rates.'
- What this solution (achieved 3.59955) has done: 'Your current MAE (3.405) is far above the target (0.1437), so we should make a small change that improves the lookup accuracy without changing the overall “groupby-mean maps → merge → staged fillna” approach. The biggest gap driver is that your keys still don’t capture the strong dependence on *accumulated volume/flow dynamics*; we add one minimal, physically meaningful state feature: `u_in * delta_time` integrated per breath (a proxy for delivered volume), then use a rounded version of that feature in the highest-priority state mean map. This keeps the same logic (just one extra per-breath feature + one extra key column in the top map), and should reduce ambiguity/mismatches while keeping runtime within limits. Everything else (fallback chain, submission writing) stays the same.'
- What this solution (achieved 3.59955) has done: 'Your current MAE (3.59955, lower-is-better) is still far above the target (0.1437), so we need a small but meaningful improvement while preserving your exact “groupby-mean lookup → merge → staged fillna” approach. The biggest easy win is fixing a subtle but severe bug: your `u_in_*` state features on `test` are computed by `groupby(breath_id)` without ensuring rows are sorted by `time_step`, so lag/cumsum/diff/integral can be wrong if the file isn’t perfectly ordered. I add a minimal, deterministic sort by `["breath_id","time_step"]` before computing per-breath state features (and then restore original test row order via the existing `id` alignment), which should improve matching accuracy substantially without changing the modeling logic. I also ensure both train/test compute state features under the same sorted-within-breath assumption to reduce train/test feature skew.'
- What this solution (achieved 1.69041) has done: 'I keep your exact “groupby mean maps → merge → staged fillna” approach, but make the highest-priority (state) lookup substantially less sparse by (1) adding a per-breath `u_out` cumulative count (captures when expiration starts) and (2) using slightly coarser rounding for the most problematic continuous state keys (`u_in_int` and `u_in_cumsum`) to increase train↔test match rate. I also add one more intermediate fallback map keyed on `(R,C,time_step_r,u_in_int_r, u_out)` so the integral feature helps even when lag/diff/cumsum don’t match. These are minimal, deterministic feature/key changes that should reduce the number of fallbacks to generic means and move MAE down toward your target, while preserving your overall logic and producing the same valid `submission.csv`.'
- What this solution (achieved 1.87864) has done: 'Your current MAE (1.69041, lower-is-better) is still far above the target (0.1437), so we should improve accuracy while keeping your exact “groupby mean maps → merge → staged fillna” core logic unchanged. The smallest high-impact change is to make the mean maps more robust by using **count-aware smoothing**: for each lookup key, blend the group mean with a broader prior mean (next fallback level) based on the number of training samples in that group, which reduces noisy/overfit group means and typically lowers MAE. This preserves the same semantics (still a deterministic lookup + fallback chain) but improves predictions especially where your current fine-grained keys match only a few training rows. I also keep ID alignment/submission writing identical and add only the minimal extra columns (`count` and smoothed predictions) needed.'
- What this solution (achieved 1.87864) has done: 'Your current MAE (1.8786, lower-is-better) is still far above the target (0.1437), so we should improve accuracy while preserving your exact “groupby mean maps → merge → staged smoothing + fillna fallback chain” approach. The highest-impact minimal fix is to align your post-processing with the competition metric: predictions during expiratory phase (`u_out==1`) are not scored, so we can safely set them to a strong per-breath baseline (the last predicted inspiratory pressure) to reduce harmful noise without affecting the scored portion. This does not change your feature engineering, mean maps, or smoothing; it only adjusts the final prediction series in a metric-consistent way. I also make sure the per-breath “last inspiratory” value is computed in proper time order and then restored back to `id` alignment for a valid submission.'

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
sub_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"

sub = pd.read_csv(sub_path)

train = pd.read_csv(
    train_path,
    usecols=["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
)
test = pd.read_csv(
    test_path, usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
)



## === cell 2
train["time_step_r"] = train["time_step"].round(2).astype("float32")
test["time_step_r"] = test["time_step"].round(2).astype("float32")

train["u_in_r1"] = train["u_in"].round(1).astype("float32")
test["u_in_r1"] = test["u_in"].round(1).astype("float32")

train["u_in_r0"] = train["u_in"].round(0).astype("float32")
test["u_in_r0"] = test["u_in"].round(0).astype("float32")



## === cell 3
train = train.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
    drop=True
)
test = test.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
    drop=True
)

grp_cols = ["breath_id"]

train["u_in_lag1"] = (
    train.groupby(grp_cols, sort=False)["u_in"].shift(1).fillna(0.0).astype("float32")
)
test["u_in_lag1"] = (
    test.groupby(grp_cols, sort=False)["u_in"].shift(1).fillna(0.0).astype("float32")
)

train["u_in_cumsum"] = (
    train.groupby(grp_cols, sort=False)["u_in"].cumsum().astype("float32")
)
test["u_in_cumsum"] = (
    test.groupby(grp_cols, sort=False)["u_in"].cumsum().astype("float32")
)

train["u_in_diff1"] = (
    train.groupby(grp_cols, sort=False)["u_in"].diff().fillna(0.0).astype("float32")
)
test["u_in_diff1"] = (
    test.groupby(grp_cols, sort=False)["u_in"].diff().fillna(0.0).astype("float32")
)

train["u_in_lag1_r1"] = train["u_in_lag1"].round(1).astype("float32")
test["u_in_lag1_r1"] = test["u_in_lag1"].round(1).astype("float32")

train["u_in_cumsum_r10"] = (train["u_in_cumsum"] / 10.0).round(0).astype("float32")
test["u_in_cumsum_r10"] = (test["u_in_cumsum"] / 10.0).round(0).astype("float32")

train["u_in_diff1_r1"] = train["u_in_diff1"].round(1).astype("float32")
test["u_in_diff1_r1"] = test["u_in_diff1"].round(1).astype("float32")

train["dt"] = (
    train.groupby(grp_cols, sort=False)["time_step"]
    .diff()
    .fillna(train["time_step"])
    .clip(lower=0.0)
    .astype("float32")
)
test["dt"] = (
    test.groupby(grp_cols, sort=False)["time_step"]
    .diff()
    .fillna(test["time_step"])
    .clip(lower=0.0)
    .astype("float32")
)

train["u_in_dt"] = (train["u_in"].astype("float32") * train["dt"]).astype("float32")
test["u_in_dt"] = (test["u_in"].astype("float32") * test["dt"]).astype("float32")

train["u_in_int"] = (
    train.groupby(grp_cols, sort=False)["u_in_dt"].cumsum().astype("float32")
)
test["u_in_int"] = (
    test.groupby(grp_cols, sort=False)["u_in_dt"].cumsum().astype("float32")
)

train["u_in_int_r0"] = train["u_in_int"].round(0).astype("float32")
test["u_in_int_r0"] = test["u_in_int"].round(0).astype("float32")

train["u_out_cum"] = (
    train.groupby(grp_cols, sort=False)["u_out"].cumsum().astype("float32")
)
test["u_out_cum"] = (
    test.groupby(grp_cols, sort=False)["u_out"].cumsum().astype("float32")
)




## === cell 4
def build_mean_count(df, key_cols, target_col="pressure", pred_name="pred"):
    g = (
        df.groupby(key_cols, sort=False)[target_col]
        .agg(["mean", "count"])
        .reset_index()
    )
    g = g.rename(columns={"mean": pred_name, "count": f"{pred_name}__n"})
    return g


def apply_smoothing(df_merged, pred_col, n_col, prior_col, alpha):
    prior = df_merged[prior_col]
    meanv = df_merged[pred_col]
    n = df_merged[n_col].astype("float32")
    sm = (n * meanv + float(alpha) * prior) / (n + float(alpha))
    return sm.where(prior.notna() & meanv.notna(), meanv)




## === cell 5
global_mean = float(train["pressure"].mean())

key_cols_nouin = ["R", "C", "time_step_r", "u_out"]
mean_map_nouin = build_mean_count(
    train, key_cols_nouin, pred_name="pred_pressure_nouin"
)
test_pred = test.merge(mean_map_nouin, on=key_cols_nouin, how="left")

key_cols_notime = ["R", "C", "u_in_r1", "u_out"]
mean_map_notime = build_mean_count(
    train, key_cols_notime, pred_name="pred_pressure_notime"
)
test_pred = test_pred.merge(mean_map_notime, on=key_cols_notime, how="left")

key_cols_0 = ["R", "C", "time_step_r", "u_in_r0", "u_out"]
mean_map_0 = build_mean_count(train, key_cols_0, pred_name="pred_pressure_0")
test_pred = test_pred.merge(mean_map_0, on=key_cols_0, how="left")

key_cols_1 = ["R", "C", "time_step_r", "u_in_r1", "u_out"]
mean_map_1 = build_mean_count(train, key_cols_1, pred_name="pred_pressure_1")
test_pred = test_pred.merge(mean_map_1, on=key_cols_1, how="left")

key_cols_int = ["R", "C", "time_step_r", "u_in_int_r0", "u_out"]
mean_map_int = build_mean_count(train, key_cols_int, pred_name="pred_pressure_int")
test_pred = test_pred.merge(mean_map_int, on=key_cols_int, how="left")

key_cols_state = [
    "R",
    "C",
    "time_step_r",
    "u_in_r1",
    "u_in_lag1_r1",
    "u_in_diff1_r1",
    "u_in_cumsum_r10",
    "u_in_int_r0",
    "u_out",
    "u_out_cum",
]
mean_map_state = build_mean_count(
    train, key_cols_state, pred_name="pred_pressure_state"
)
test_pred = test_pred.merge(mean_map_state, on=key_cols_state, how="left")



## === cell 6
test_pred["pred_pressure_nouin_f"] = test_pred["pred_pressure_nouin"].fillna(
    global_mean
)

test_pred["pred_pressure_notime_f"] = apply_smoothing(
    test_pred,
    pred_col="pred_pressure_notime",
    n_col="pred_pressure_notime__n",
    prior_col="pred_pressure_nouin_f",
    alpha=20.0,
).fillna(test_pred["pred_pressure_nouin_f"])

test_pred["pred_pressure_0_f"] = apply_smoothing(
    test_pred,
    pred_col="pred_pressure_0",
    n_col="pred_pressure_0__n",
    prior_col="pred_pressure_1",
    alpha=15.0,
).fillna(test_pred["pred_pressure_notime_f"])

test_pred["pred_pressure_1_f"] = apply_smoothing(
    test_pred,
    pred_col="pred_pressure_1",
    n_col="pred_pressure_1__n",
    prior_col="pred_pressure_0_f",
    alpha=15.0,
).fillna(test_pred["pred_pressure_notime_f"])

test_pred["pred_pressure_int_f"] = apply_smoothing(
    test_pred,
    pred_col="pred_pressure_int",
    n_col="pred_pressure_int__n",
    prior_col="pred_pressure_1_f",
    alpha=10.0,
).fillna(test_pred["pred_pressure_1_f"])

test_pred["pred_pressure_state_f"] = apply_smoothing(
    test_pred,
    pred_col="pred_pressure_state",
    n_col="pred_pressure_state__n",
    prior_col="pred_pressure_int_f",
    alpha=5.0,
).fillna(test_pred["pred_pressure_int_f"])



## === cell 7
test_pred["pred_pressure"] = test_pred["pred_pressure_state_f"]
test_pred["pred_pressure"] = test_pred["pred_pressure"].fillna(
    test_pred["pred_pressure_int_f"]
)
test_pred["pred_pressure"] = test_pred["pred_pressure"].fillna(
    test_pred["pred_pressure_1_f"]
)
test_pred["pred_pressure"] = test_pred["pred_pressure"].fillna(
    test_pred["pred_pressure_0_f"]
)
test_pred["pred_pressure"] = test_pred["pred_pressure"].fillna(
    test_pred["pred_pressure_notime_f"]
)
test_pred["pred_pressure"] = test_pred["pred_pressure"].fillna(
    test_pred["pred_pressure_nouin_f"]
)
test_pred["pred_pressure"] = test_pred["pred_pressure"].fillna(global_mean)

insp = test_pred["u_out"].astype("int8").eq(0)
last_insp_pred = (
    test_pred.loc[insp, ["breath_id", "pred_pressure"]]
    .groupby("breath_id", sort=False)["pred_pressure"]
    .last()
)
test_pred["last_insp_pred"] = (
    test_pred["breath_id"].map(last_insp_pred).astype("float32")
)
test_pred.loc[~insp, "pred_pressure"] = test_pred.loc[~insp, "last_insp_pred"].values
test_pred["pred_pressure"] = test_pred["pred_pressure"].fillna(global_mean)

pred_by_id = test_pred.set_index("id")["pred_pressure"]

sub = sub.set_index("id")
sub["pressure"] = pred_by_id.reindex(sub.index).values

sub.reset_index()[["id", "pressure"]].to_csv("submission.csv", index=False)

print(sub.reset_index().head())
print("Wrote submission.csv with", len(sub), "rows")
