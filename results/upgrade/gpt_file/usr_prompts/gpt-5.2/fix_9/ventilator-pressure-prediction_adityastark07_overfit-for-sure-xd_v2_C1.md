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

0.1439384146214482

# 6. Current score

2.49723

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.77703) has done: 'I remove the dependency on missing external Kaggle dataset submissions (those `../input/.../submission*.csv` files aren’t available in your environment), which currently prevents the notebook from running and producing any submission. To keep the core “weighted ensemble” logic intact, I instead ensemble multiple lightweight baseline predictors generated from the provided `train.csv` only (group-wise medians/means by `R,C` and by `R,C,time_step`) and blend them with the same 0.28/0.28/0.28/0.16 weights. This run end-to-end within the Kaggle file structure you provided and write a valid `submission.csv` with `id,pressure`. The changes are minimal and focused on fixing runtime errors and creating a legitimate prediction pipeline.'
- What this solution (achieved 9.80657) has done: 'Your current score (MAE ≈ 9.78, lower is better) is far from the target (~0.144), so we need a meaningful but still “core-logic-preserving” improvement. The biggest issue is that the current approach ignores the crucial per-breath dynamics and the inspiratory/expiratory scoring rule; grouping only by `(R,C,time_step)` averages away the dependence on control inputs and breath history. Keeping the same overall “non-ML, aggregation-based predictors + weighted blend” approach, we add two additional baseline predictors that condition on `u_out` and `u_in` (binned), which better matches how pressure depends on the control signals, then blend them into the existing ensemble with minimal changes. We also make the fallback filling fully vectorized (same semantics, faster) to stay within runtime limits.'
- What this solution (achieved 9.82822) has done: 'Your current MAE (~9.81; lower is better) is still extremely far from the target (~0.144), so we need a stronger but still aggregation-based improvement without changing the overall “groupby lookup predictors + weighted blend” core. The main missing piece is exploiting per-breath history: pressure depends heavily on the integrated flow (cumulative `u_in`) and whether the valve is venting (`u_out`), so I add one additional lookup predictor keyed by `(R,C,time_step,u_out,cum_u_in_bin)` and blend it in with a small set of weight changes. To keep semantics identical and avoid leakage, `cum_u_in` is computed within each breath for both train and test and then binned; everything remains pure train-derived aggregation merged onto test. I also make the NA fallback slightly more “local” by falling back through progressively coarser keys before using `(R,C)` and then global median, which typically reduces error without altering the approach.'
- What this solution (achieved 9.85854) has done: 'Your current MAE (~9.83, lower is better) is still very far from the target (~0.144), and the main weakness is that the lookups don’t explicitly encode “inspiratory-only scoring” behavior (u_out==0 is what matters) and the strong dependence on recent control history. Keeping the same core logic (train-derived groupby-median/mean predictors + weighted blending + fallback filling), I add one additional predictor keyed on `(R,C,time_step,u_out,cum_u_in_bin,u_in_bin)` to better capture both accumulated flow and instantaneous valve opening. I also add a tiny adjustment that forces predictions during `u_out==1` to a train-derived median for expiratory points (which are not scored), reducing noise without changing the modeling approach. Finally, I minimally re-tune ensemble weights to put more mass on the more specific predictors and include the new one, while keeping everything deterministic and within runtime.'
- What this solution (achieved 9.87018) has done: 'Your current MAE is far worse than the target, so we need a stronger signal while keeping the same “train-derived groupby lookup predictors + weighted blend + fallback filling” core. The largest remaining gap is that your most-specific keys still don’t capture the strong dependence on *recent* control history (not just cumulative), so I add one additional lookup predictor keyed by `(R,C,time_step,u_out,cum_u_in_bin,du_in_bin)` where `du_in` is the within-breath change in `u_in`. I keep all existing predictors and the same fallback mechanism, then minimally reweight the ensemble to put more mass on the most specific predictors (including the new one) to improve MAE. This stays deterministic, uses only `train.csv` aggregates, and still writes a valid `submission.csv`.'
- What this solution (achieved 9.86488) has done: 'Your current MAE (~9.87, lower is better) is still far from the target, so we need a modest but meaningful improvement while keeping the same “train-derived groupby lookup predictors + weighted blend + fallback filling” approach. The biggest low-risk gain is to use a more physically-aligned history signal: compute within-breath cumulative inspired volume as `cum_u_in_dt = cumsum(u_in * delta_time)` (rather than cumsum of raw `u_in`) and build one additional lookup keyed by `(R,C,time_step,u_out,cum_u_in_dt_bin)`. Then, minimally reweight the ensemble to put some weight on this new predictor while keeping the existing ones unchanged. This preserves evaluation semantics and stays purely aggregation-based, but should reduce error by conditioning on a better proxy for delivered volume.'
- What this solution (achieved 2.49727) has done: 'Your score is far worse than the target (MAE 9.86 vs 0.144; lower is better), so we need a genuine signal improvement while keeping your “train-derived groupby lookup predictors + weighted blend + fallback filling” core intact. The biggest low-risk gain is to stop keying on raw floating `time_step` (merge misses due to float representation), and instead use the known discrete within-breath index (0–79) via `time_idx`, which greatly increases exact match rates for the same semantics. I replace every lookup key that includes `time_step` with `time_idx` (computed per-breath using `cumcount()`), keep the rest of the feature/ensemble structure unchanged, and leave the same u_out expiratory override. This is a minimal change in logic (still pure aggregation lookups), but it should materially reduce MAE by eliminating systematic NaNs and coarse fallbacks.'
- What this solution (achieved 2.49723) has done: 'Your current MAE (2.497) is still far above the target (0.144; lower is better), so we should make a small but meaningful improvement without changing the overall “train-derived groupby lookup predictors + weighted blend + fallback filling” core. The least invasive high-impact fix is to align predictions to the discrete pressure grid used in the dataset by snapping final predictions to the nearest pressure value seen in train, which typically reduces MAE a lot for this competition. This is purely post-processing (doesn’t change features, models, or training) and keeps evaluation semantics intact. I also compute the pressure grid once from train and apply it after the ensemble and expiratory override, then write the same `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from pathlib import Path



## === cell 1
BASE_DIR_CANDIDATES = [
    Path("/kaggle/input/ventilator-pressure-prediction"),
    Path("/kaggle/input"),
    Path("../input/ventilator-pressure-prediction"),
    Path("../input"),
]


def find_file(filename: str) -> Path:
    for base in BASE_DIR_CANDIDATES:
        p = base / filename
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find {filename} in any of: {BASE_DIR_CANDIDATES}"
    )


train_path = find_file("train.csv")
test_path = find_file("test.csv")
sample_sub_path = find_file("sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sub = pd.read_csv(sample_sub_path)

assert {"id", "pressure"}.issubset(sub.columns)
assert "pressure" in train.columns



## === cell 2
for col in ["R", "C"]:
    train[col] = train[col].astype(np.int16)
    test[col] = test[col].astype(np.int16)

train["u_out"] = train["u_out"].astype(np.int8)
test["u_out"] = test["u_out"].astype(np.int8)

train["time_idx"] = train.groupby("breath_id", sort=False).cumcount().astype(np.int16)
test["time_idx"] = test.groupby("breath_id", sort=False).cumcount().astype(np.int16)

train["u_in_bin"] = np.floor(train["u_in"].values * 0.5).astype(np.int16)
test["u_in_bin"] = np.floor(test["u_in"].values * 0.5).astype(np.int16)

train["cum_u_in"] = (
    train.groupby("breath_id", sort=False)["u_in"].cumsum().astype(np.float32)
)
test["cum_u_in"] = (
    test.groupby("breath_id", sort=False)["u_in"].cumsum().astype(np.float32)
)

train["cum_u_in_bin"] = np.floor(train["cum_u_in"].values / 5.0).astype(np.int16)
test["cum_u_in_bin"] = np.floor(test["cum_u_in"].values / 5.0).astype(np.int16)

train_du = (
    train.groupby("breath_id", sort=False)["u_in"].diff().fillna(0.0).astype(np.float32)
)
test_du = (
    test.groupby("breath_id", sort=False)["u_in"].diff().fillna(0.0).astype(np.float32)
)
train["du_in_bin"] = np.clip(np.floor(train_du.values / 2.0), -25, 25).astype(np.int16)
test["du_in_bin"] = np.clip(np.floor(test_du.values / 2.0), -25, 25).astype(np.int16)

train_dt = (
    train.groupby("breath_id", sort=False)["time_step"]
    .diff()
    .fillna(train["time_step"])
    .astype(np.float32)
)
test_dt = (
    test.groupby("breath_id", sort=False)["time_step"]
    .diff()
    .fillna(test["time_step"])
    .astype(np.float32)
)

train["cum_u_in_dt"] = (
    (train["u_in"].astype(np.float32) * train_dt)
    .groupby(train["breath_id"], sort=False)
    .cumsum()
    .astype(np.float32)
)
test["cum_u_in_dt"] = (
    (test["u_in"].astype(np.float32) * test_dt)
    .groupby(test["breath_id"], sort=False)
    .cumsum()
    .astype(np.float32)
)

train["cum_u_in_dt_bin"] = np.floor(train["cum_u_in_dt"].values / 0.5).astype(np.int16)
test["cum_u_in_dt_bin"] = np.floor(test["cum_u_in_dt"].values / 0.5).astype(np.int16)

global_median = float(train["pressure"].median())
exp_median = float(train.loc[train["u_out"] == 1, "pressure"].median())

pressure_grid = np.sort(train["pressure"].unique().astype(np.float32))

rc_key = ["R", "C"]
rc_t_key = ["R", "C", "time_idx"]
rc_t_uo_key = ["R", "C", "time_idx", "u_out"]
rc_t_uo_ui_key = ["R", "C", "time_idx", "u_out", "u_in_bin"]

med_rc = (
    train.groupby(rc_key, sort=False)["pressure"].median().rename("pred").reset_index()
)
rc_median_df = med_rc.rename(columns={"pred": "rc_med"})

med_rc_t = (
    train.groupby(rc_t_key, sort=False)["pressure"]
    .median()
    .rename("pred")
    .reset_index()
)
med_rc_t_uo = (
    train.groupby(rc_t_uo_key, sort=False)["pressure"]
    .median()
    .rename("pred")
    .reset_index()
)
med_rc_t_uo_ui = (
    train.groupby(rc_t_uo_ui_key, sort=False)["pressure"]
    .median()
    .rename("pred")
    .reset_index()
)

sub_1 = (
    test[["id"] + rc_t_key]
    .merge(med_rc_t, on=rc_t_key, how="left")[["id", "pred"]]
    .rename(columns={"pred": "pressure"})
)

mean_rc_t = (
    train.groupby(rc_t_key, sort=False)["pressure"].mean().rename("pred").reset_index()
)
sub_2 = (
    test[["id"] + rc_t_key]
    .merge(mean_rc_t, on=rc_t_key, how="left")[["id", "pred"]]
    .rename(columns={"pred": "pressure"})
)

sub_3 = (
    test[["id"] + rc_key]
    .merge(med_rc, on=rc_key, how="left")[["id", "pred"]]
    .rename(columns={"pred": "pressure"})
)

sub_4 = pd.DataFrame(
    {
        "id": test["id"].values,
        "pressure": np.full(len(test), global_median, dtype=np.float32),
    }
)

sub_5 = (
    test[["id"] + rc_t_uo_key]
    .merge(med_rc_t_uo, on=rc_t_uo_key, how="left")[["id", "pred"]]
    .rename(columns={"pred": "pressure"})
)

sub_6 = (
    test[["id"] + rc_t_uo_ui_key]
    .merge(med_rc_t_uo_ui, on=rc_t_uo_ui_key, how="left")[["id", "pred"]]
    .rename(columns={"pred": "pressure"})
)

rc_t_uo_cui_key = ["R", "C", "time_idx", "u_out", "cum_u_in_bin"]
med_rc_t_uo_cui = (
    train.groupby(rc_t_uo_cui_key, sort=False)["pressure"]
    .median()
    .rename("pred")
    .reset_index()
)
sub_7 = (
    test[["id"] + rc_t_uo_cui_key]
    .merge(med_rc_t_uo_cui, on=rc_t_uo_cui_key, how="left")[["id", "pred"]]
    .rename(columns={"pred": "pressure"})
)

rc_t_uo_cui_ui_key = ["R", "C", "time_idx", "u_out", "cum_u_in_bin", "u_in_bin"]
med_rc_t_uo_cui_ui = (
    train.groupby(rc_t_uo_cui_ui_key, sort=False)["pressure"]
    .median()
    .rename("pred")
    .reset_index()
)
sub_8 = (
    test[["id"] + rc_t_uo_cui_ui_key]
    .merge(med_rc_t_uo_cui_ui, on=rc_t_uo_cui_ui_key, how="left")[["id", "pred"]]
    .rename(columns={"pred": "pressure"})
)

rc_t_uo_cui_du_key = ["R", "C", "time_idx", "u_out", "cum_u_in_bin", "du_in_bin"]
med_rc_t_uo_cui_du = (
    train.groupby(rc_t_uo_cui_du_key, sort=False)["pressure"]
    .median()
    .rename("pred")
    .reset_index()
)
sub_9 = (
    test[["id"] + rc_t_uo_cui_du_key]
    .merge(med_rc_t_uo_cui_du, on=rc_t_uo_cui_du_key, how="left")[["id", "pred"]]
    .rename(columns={"pred": "pressure"})
)

rc_t_uo_cuidt_key = ["R", "C", "time_idx", "u_out", "cum_u_in_dt_bin"]
med_rc_t_uo_cuidt = (
    train.groupby(rc_t_uo_cuidt_key, sort=False)["pressure"]
    .median()
    .rename("pred")
    .reset_index()
)
sub_10 = (
    test[["id"] + rc_t_uo_cuidt_key]
    .merge(med_rc_t_uo_cuidt, on=rc_t_uo_cuidt_key, how="left")[["id", "pred"]]
    .rename(columns={"pred": "pressure"})
)

test_id_rc = test[["id"] + rc_key].copy().sort_values("id").reset_index(drop=True)
test_id_rc_t = test[["id"] + rc_t_key].copy().sort_values("id").reset_index(drop=True)
test_id_rc_t_uo = (
    test[["id"] + rc_t_uo_key].copy().sort_values("id").reset_index(drop=True)
)


def fill_with_fallback(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("id").reset_index(drop=True)
    p = df["pressure"].astype("float32")

    if p.isna().any():
        tmp1 = test_id_rc_t_uo.merge(
            med_rc_t_uo.rename(columns={"pred": "fb1"}), on=rc_t_uo_key, how="left"
        )[["id", "fb1"]]
        p = p.fillna(tmp1["fb1"].astype("float32"))

    if p.isna().any():
        tmp2 = test_id_rc_t.merge(
            med_rc_t.rename(columns={"pred": "fb2"}), on=rc_t_key, how="left"
        )[["id", "fb2"]]
        p = p.fillna(tmp2["fb2"].astype("float32"))

    if p.isna().any():
        tmp3 = test_id_rc.merge(rc_median_df, on=rc_key, how="left")[["id", "rc_med"]]
        p = p.fillna(tmp3["rc_med"].astype("float32"))

    p = p.fillna(global_median).astype("float32")
    df["pressure"] = p
    return df


sub_1 = fill_with_fallback(sub_1)
sub_2 = fill_with_fallback(sub_2)
sub_3 = fill_with_fallback(sub_3)
sub_4 = fill_with_fallback(sub_4)
sub_5 = fill_with_fallback(sub_5)
sub_6 = fill_with_fallback(sub_6)
sub_7 = fill_with_fallback(sub_7)
sub_8 = fill_with_fallback(sub_8)
sub_9 = fill_with_fallback(sub_9)
sub_10 = fill_with_fallback(sub_10)



## === cell 3
sub = sub.sort_values("id").reset_index(drop=True)
sub_1 = sub_1.sort_values("id").reset_index(drop=True)
sub_2 = sub_2.sort_values("id").reset_index(drop=True)
sub_3 = sub_3.sort_values("id").reset_index(drop=True)
sub_4 = sub_4.sort_values("id").reset_index(drop=True)
sub_5 = sub_5.sort_values("id").reset_index(drop=True)
sub_6 = sub_6.sort_values("id").reset_index(drop=True)
sub_7 = sub_7.sort_values("id").reset_index(drop=True)
sub_8 = sub_8.sort_values("id").reset_index(drop=True)
sub_9 = sub_9.sort_values("id").reset_index(drop=True)
sub_10 = sub_10.sort_values("id").reset_index(drop=True)

w1, w2, w3, w4, w5, w6, w7, w8, w9, w10 = (
    0.02,
    0.02,
    0.02,
    0.01,
    0.07,
    0.18,
    0.22,
    0.19,
    0.16,
    0.11,
)
assert abs((w1 + w2 + w3 + w4 + w5 + w6 + w7 + w8 + w9 + w10) - 1.0) < 1e-9

pred = (
    (sub_1["pressure"].values * w1)
    + (sub_2["pressure"].values * w2)
    + (sub_3["pressure"].values * w3)
    + (sub_4["pressure"].values * w4)
    + (sub_5["pressure"].values * w5)
    + (sub_6["pressure"].values * w6)
    + (sub_7["pressure"].values * w7)
    + (sub_8["pressure"].values * w8)
    + (sub_9["pressure"].values * w9)
    + (sub_10["pressure"].values * w10)
).astype("float32")

pred = pred.copy()

pred[test["u_out"].values.astype(bool)] = np.float32(exp_median)

idx = np.searchsorted(pressure_grid, pred, side="left")
idx = np.clip(idx, 0, len(pressure_grid) - 1)

left = pressure_grid[np.clip(idx - 1, 0, len(pressure_grid) - 1)]
right = pressure_grid[idx]
use_right = (idx == 0) | (np.abs(right - pred) <= np.abs(pred - left))
pred_snapped = np.where(use_right, right, left).astype(np.float32)

sub["pressure"] = pred_snapped
sub[["id", "pressure"]].to_csv("submission.csv", index=False)
sub.head(5)
