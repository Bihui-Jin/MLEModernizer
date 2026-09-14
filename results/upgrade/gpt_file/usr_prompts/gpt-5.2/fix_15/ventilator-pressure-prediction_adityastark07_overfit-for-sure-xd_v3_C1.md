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

0.1438694648501155

# 6. Current score

2.19979

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.02329) has done: 'I remove the dependency on external notebook submission files that don’t exist in your environment (the cause of the FileNotFoundError), and replace it with a self-contained baseline that trains from the provided `train.csv` and predicts for `test.csv`. To keep changes minimal while restoring end-to-end functionality, I use a simple, deterministic per-(R,C,time_step,u_out) median-lookup model that aligns with the MAE metric and preserves the competition’s semantics (predict pressure per row). I also ensure the submission is written as `submission.csv` with exactly `id,pressure` and in the correct row order. This should yield a non-trivial score (better than all-zeros) while avoiding heavy dependencies and staying within runtime.'
- What this solution (achieved 3.80669) has done: 'Your current score (6.02329) is far worse than the target (0.1439), so we should improve accuracy while keeping the same “lookup aggregation + merge + fallback” core logic. The main issue is that the lookup key is missing the strongest signal (`u_in`), so predictions collapse to coarse medians; adding a rounded `u_in` to the groupby keys is a minimal change that usually yields a large MAE drop for this competition. To keep it stable and avoid over-fragmenting (which could increase missing merges), we round `u_in` to a modest resolution and keep the same global-median fallback. We also keep the submission format identical (`id,pressure`) and preserve row order.'
- What this solution (achieved 3.75839) has done: 'Your current MAE (3.80669, lower-is-better) is still far above the target (0.14387), so we should improve accuracy while keeping the same “groupby median lookup + merge + global fallback” core logic. The biggest remaining issue is that the lookup table is too sparse and the global fallback is too crude; we can reduce miss-rate and improve fallback quality by (1) adding a second, coarser lookup that ignores `u_out` and uses a slightly coarser `u_in` bin, and (2) using a per-(R,C,time_step_r) fallback median instead of a single global median. These are minimal, fast changes that preserve evaluation semantics and should move the score substantially toward the target without changing the approach. The submission format and row order are kept identical and it still writes `submission.csv`.'
- What this solution (achieved 2.38827) has done: 'Your current MAE (3.758, lower-is-better) is far above the target (0.144), so we should improve accuracy while keeping the same “groupby median lookup + merge + fallback” core approach. The main weakness is that using `time_step` directly (even rounded) misses dynamic information; adding minimal lagged/accumulated features (`u_in` cumulative sum and 1-step lags) keeps the same lookup paradigm but captures much more of the pressure trajectory. We build an additional “enhanced” median lookup on these new keys and use it as the first-choice prediction, then fall back to your existing fine/coarse/phase/global hierarchy. This remains deterministic, fast, and still writes a valid `submission.csv` with `id,pressure` in the correct order.'
- What this solution (achieved 2.58387) has done: 'We keep your same “median lookup + merge + fallback” core logic, but make two minimal changes that typically close a large part of the MAE gap for this competition. First, we add an `u_out`-aware phase fallback (because evaluation only scores inspiratory phase and `u_out` strongly changes the pressure distribution), instead of using a phase median that mixes inspiratory/expiratory. Second, we make your dynamic lookup slightly less sparse by using `u_in_cum` with a finer bin (so fewer misses and a better first-choice prediction), while keeping the same lag/cumsum feature family and the same merge hierarchy. This should move your score down toward the 0.14 target without changing the overall approach or requiring new packages, and it still write a valid `submission.csv`.'
- What this solution (achieved 2.56755) has done: 'Your current MAE (2.58387, lower-is-better) is still far above the target (0.14387), so we should improve accuracy while keeping your same “median lookup + merge + fallback” approach. The main issue is that your keys don’t include the strongest structure in this competition: the discrete pressure levels and the fact that pressure is highly conditioned on cumulative delivered air volume (integral of u_in over time), not just instantaneous u_in. With minimal change, we (1) add a more informative dynamic key using a better-scaled cumulative u_in (“area under curve” proxy) plus a small u_in lag, and (2) snap final predictions to the nearest training pressure level (a standard, lightweight post-processing aligned with the metric). This preserves your architecture (groupby-median tables + merge hierarchy), stays deterministic, and still writes a valid `submission.csv`.'
- What this solution (achieved 2.54063) has done: 'Your score is still far above the target (lower-is-better), so the smallest likely win while preserving your “groupby-median lookup + merge + fallback + snap-to-levels” core logic is to (1) incorporate the key evaluation fact that expiratory phase (`u_out==1`) is not scored by replacing those predictions with a safe constant (0.0 is commonly used), and (2) make your dynamic “delivered volume” proxy less sparse by binning `u_in_area` a bit more coarsely so more test rows hit the high-priority dyn lookup instead of falling through to weaker fallbacks. These changes keep your same feature family, same lookup approach, and same snapping post-process; they just improve coverage where it matters for MAE. The script still runs end-to-end and writes a valid `submission.csv` with `id,pressure` in the correct order.'
- What this solution (achieved 2.54063) has done: 'Your current MAE (2.54063, lower-is-better) is far above the target (0.14387), so we should improve accuracy while keeping your same “groupby-median lookup + merge + fallback + snap-to-levels” approach. The smallest high-impact issue is that your `id` column range indicates it is not globally unique in this environment, so the submission must be aligned to `sample_submission.csv`’s `id` order rather than trusting `test["id"]`; otherwise Kaggle scoring can be severely degraded by misalignment. I keep all your feature engineering and lookup hierarchy identical, but (1) build the submission by merging predictions onto `sub` by `id` and (2) add a safety check that every `id` in `sub` got exactly one prediction. This change is directly score-relevant (correct row alignment) and preserves your model logic.'
- What this solution (achieved 2.54063) has done: 'We keep your exact “groupby-median lookup + merge hierarchy + snap-to-pressure-levels” approach, but fix two score-critical issues that can easily keep MAE high. First, your `id` column in this environment is not globally unique, so `drop_duplicates` loses rows and `merge` can mis-assign predictions; we instead generate predictions in the exact `sub` row order by predicting for `test` in its original order and then directly setting `submission["pressure"]` row-by-row (no merge on non-unique ids). Second, we align with the metric by only forcing `u_out==1` rows to 0.0 after the prediction vector is aligned to the submission order, guaranteeing the inspiratory rows keep your best predictions and expiratory rows don’t affect scoring. These are minimal changes that preserve your model logic and should move MAE down toward the target by eliminating submission misalignment/row loss.'
- What this solution (achieved 2.71422) has done: 'Your score is far above the target (lower is better), so we should make a small but high-impact accuracy improvement while keeping your exact “groupby-median lookup + merge hierarchy + snap-to-pressure-levels” approach. The main weakness is that your dynamic keys don’t use the competition’s most informative “delivered volume” proxy: the integral of flow over time depends on `u_in * dt`, so using *cumulative u_in* alone is a poorer surrogate than using *area under curve*. We minimally swap the `u_in_cum_r` key in `keys_dyn` to an `u_in_area_r2` key (a coarser-binned area to keep coverage high), while leaving your existing `dyn2/fine/coarse/phase/global` fallbacks and snapping unchanged. This should reduce MAE toward the target without changing the overall method, and it still writes a valid `submission.csv` aligned row-for-row with `sample_submission.csv`.'
- What this solution (achieved 2.63179) has done: 'Your current MAE (2.714) is still far above the target (0.144), so we should improve accuracy while keeping the exact same “groupby-median lookup + merge hierarchy + snap-to-pressure-levels (+ u_out==1 set to 0)” core logic. The smallest high-impact adjustment is to make the dynamic lookup keys less sparse and more physically aligned by (a) using the *dt-weighted delivered volume* (`u_in_area`) in both dyn tables (not just one), and (b) adding `u_in_area` and a second lag into the higher-priority dyn table while slightly coarsening its bins to improve hit-rate (coverage) on test. We keep your fine/coarse/phase/global fallbacks unchanged, keep snapping unchanged, and keep row-order alignment unchanged (no merges on `id`). This should reduce fallbacks and move MAE downward toward the target without changing the overall approach.'
- What this solution (achieved 2.50773) has done: 'Your current MAE (2.63179, lower-is-better) is still far above the target (0.14387), so we make the smallest changes that improve accuracy while preserving your exact “groupby-median lookup + merge hierarchy + snap-to-pressure-levels (+ u_out==1 set to 0)” approach. The biggest safe gain without changing the modeling paradigm is to improve your delivered-volume proxy: use trapezoidal integration (based on average of current and previous `u_in`) and add a similarly integrated `u_out` area, then use those in the dynamic lookup keys to reduce collisions and better match the physics. We keep all existing tables and fallbacks, just add one higher-priority dyn table keyed on the improved integrated signals, and keep snapping/row-order handling identical. This should reduce fallbacks and move the score downward toward the target while staying deterministic and within runtime.'
- What this solution (achieved 2.54165) has done: 'Your current MAE (2.50773, lower-is-better) is far above the target (0.14387), so we should improve accuracy with the smallest changes that keep your exact “median lookup tables + merge hierarchy + snap-to-pressure-levels (+ u_out==1 set to 0)” approach. The most impactful missing signal you can add without changing the paradigm is an explicit breath-position index (`step`) and a better-conditioned dynamic key that uses `u_in_area_trap` at a finer-but-not-too-sparse bin, because pressure trajectories are strongly step-dependent and volume-dependent. We add `step`-based lookup tables (highest priority) and a per-(R,C,u_out,step) fallback, then keep all your existing dyn/fine/coarse/phase/global fallbacks unchanged. This keeps runtime reasonable (single extra feature + a few extra groupbys/merges) and still writes a valid `submission.csv` in the correct order.'
- What this solution (achieved 2.19979) has done: 'Your MAE is still far above the target (lower is better), so we should improve accuracy while preserving your exact “median lookup tables + merge hierarchy + snap-to-pressure-levels (+ u_out==1 set to 0)” approach. The biggest score issue left is that your highest-priority dynamic tables are still too sparse (many test rows miss and fall back to weaker maps), so we add one *slightly coarser* step-based dynamic lookup that uses the same feature family but with coarser bins to greatly increase hit-rate. We also add a very small, safe improvement to the expiratory handling by using the **train median pressure for u_out==1** (instead of hard 0.0) to avoid any edge-case scoring/mask mismatch and keep predictions plausible. Everything else (feature engineering style, median groupby modeling, snapping, and submission row-order) is kept the same, and it still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
DATA_CANDIDATES = [
    "/kaggle/input/ventilator-pressure-prediction",
    "/kaggle/data/ventilator-pressure-prediction",
    "/kaggle/input",
    "/kaggle/data",
]


def pick_file(filename: str) -> str:
    for base in DATA_CANDIDATES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find {filename} in any of: {DATA_CANDIDATES}")


train_path = pick_file("train.csv")
test_path = pick_file("test.csv")
sample_path = pick_file("sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sub = pd.read_csv(sample_path)

train.shape, test.shape, sub.shape




## === cell 2
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["step"] = df.groupby("breath_id", sort=False).cumcount().astype(np.int16)

    df["time_step_r"] = (df["time_step"] * 100).round().astype(np.int32)  # 0.01s
    df["u_in_r"] = (df["u_in"] * 2).round().astype(np.int16)  # 0.5 resolution
    df["u_in_r2"] = (df["u_in"] * 1).round().astype(np.int16)  # 1.0 resolution

    g = df.groupby("breath_id", sort=False)

    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0)

    df["u_in_cum"] = g["u_in"].cumsum()

    df["u_in_lag1_r"] = (df["u_in_lag1"] * 2).round().astype(np.int16)  # 0.5
    df["u_in_lag2_r"] = (df["u_in_lag2"] * 2).round().astype(np.int16)  # 0.5

    df["u_in_cum_r"] = (df["u_in_cum"] / 5.0).round().astype(np.int16)  # bin size 5

    df["dt"] = g["time_step"].diff().fillna(0.0)

    df["u_in_trap"] = 0.5 * (df["u_in"] + df["u_in_lag1"])
    df["u_in_area_trap"] = (
        (df["u_in_trap"] * df["dt"]).groupby(df["breath_id"], sort=False).cumsum()
    )

    df["u_in_area"] = (
        (df["u_in"] * df["dt"]).groupby(df["breath_id"], sort=False).cumsum()
    )

    df["u_out_area"] = (
        (df["u_out"].astype(np.float32) * df["dt"])
        .groupby(df["breath_id"], sort=False)
        .cumsum()
    )

    df["u_in_area_r"] = (df["u_in_area"] / 0.04).round().astype(np.int16)
    df["u_in_area_r2"] = (df["u_in_area"] / 0.08).round().astype(np.int16)
    df["u_in_area_r3"] = (df["u_in_area"] / 0.12).round().astype(np.int16)

    df["u_in_area_trap_r2"] = (df["u_in_area_trap"] / 0.08).round().astype(np.int16)
    df["u_in_area_trap_r3"] = (df["u_in_area_trap"] / 0.12).round().astype(np.int16)

    df["u_in_area_trap_r1"] = (df["u_in_area_trap"] / 0.04).round().astype(np.int16)

    df["u_out_area_r"] = (df["u_out_area"] / 0.20).round().astype(np.int16)

    df["u_in_area_trap_r4"] = (df["u_in_area_trap"] / 0.16).round().astype(np.int16)

    df["u_in_lag1_r2"] = (df["u_in_lag1"] * 1).round().astype(np.int16)  # 1.0

    return df


train_f = add_features(train)
test_f = add_features(test)

keys_fine = ["R", "C", "u_out", "time_step_r", "u_in_r"]
median_map_fine = (
    train_f.groupby(keys_fine, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "pressure_pred"})
)

keys_coarse = ["R", "C", "time_step_r", "u_in_r2"]
median_map_coarse = (
    train_f.groupby(keys_coarse, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "pressure_pred2"})
)

keys_phase_uout = ["R", "C", "u_out", "time_step_r"]
phase_map_uout = (
    train_f.groupby(keys_phase_uout, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "pressure_phase_uout"})
)

keys_phase_step_uout = ["R", "C", "u_out", "step"]
phase_step_map_uout = (
    train_f.groupby(keys_phase_step_uout, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "pressure_phase_step_uout"})
)

global_median = float(train_f["pressure"].median())

keys_dyn3 = [
    "R",
    "C",
    "u_out",
    "time_step_r",
    "u_in_r",
    "u_in_lag1_r",
    "u_in_area_trap_r3",
    "u_out_area_r",
]
median_map_dyn3 = (
    train_f.groupby(keys_dyn3, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "pressure_dyn3"})
)

keys_dyn2 = [
    "R",
    "C",
    "u_out",
    "time_step_r",
    "u_in_r",
    "u_in_lag1_r",
    "u_in_lag2_r",
    "u_in_area_r3",
]
median_map_dyn2 = (
    train_f.groupby(keys_dyn2, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "pressure_dyn2"})
)

keys_dyn = ["R", "C", "u_out", "time_step_r", "u_in_r", "u_in_lag1_r", "u_in_area_r2"]
median_map_dyn = (
    train_f.groupby(keys_dyn, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "pressure_dyn"})
)

keys_dyn_step = [
    "R",
    "C",
    "u_out",
    "step",
    "u_in_r",
    "u_in_lag1_r",
    "u_in_area_trap_r1",
]
median_map_dyn_step = (
    train_f.groupby(keys_dyn_step, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "pressure_dyn_step"})
)

keys_dyn_step_coarse = [
    "R",
    "C",
    "u_out",
    "step",
    "u_in_r2",
    "u_in_lag1_r2",
    "u_in_area_trap_r4",
]
median_map_dyn_step_coarse = (
    train_f.groupby(keys_dyn_step_coarse, as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "pressure_dyn_step_coarse"})
)

pred = test_f.merge(median_map_dyn_step, on=keys_dyn_step, how="left")
pred = pred.merge(median_map_dyn_step_coarse, on=keys_dyn_step_coarse, how="left")
pred = pred.merge(median_map_dyn3, on=keys_dyn3, how="left")
pred = pred.merge(median_map_dyn2, on=keys_dyn2, how="left")
pred = pred.merge(median_map_dyn, on=keys_dyn, how="left")

pred["pressure_final"] = (
    pred["pressure_dyn_step"]
    .fillna(pred["pressure_dyn_step_coarse"])
    .fillna(pred["pressure_dyn3"])
    .fillna(pred["pressure_dyn2"])
    .fillna(pred["pressure_dyn"])
)

pred = pred.merge(median_map_fine, on=keys_fine, how="left")
pred["pressure_final"] = pred["pressure_final"].fillna(pred["pressure_pred"])

pred = pred.merge(median_map_coarse, on=keys_coarse, how="left")
pred["pressure_final"] = pred["pressure_final"].fillna(pred["pressure_pred2"])

pred = pred.merge(phase_step_map_uout, on=keys_phase_step_uout, how="left")
pred["pressure_final"] = pred["pressure_final"].fillna(pred["pressure_phase_step_uout"])

pred = pred.merge(phase_map_uout, on=keys_phase_uout, how="left")
pred["pressure_final"] = pred["pressure_final"].fillna(pred["pressure_phase_uout"])

pred["pressure_final"] = pred["pressure_final"].fillna(global_median).astype(np.float32)

pressure_levels = np.sort(train["pressure"].unique()).astype(np.float32)

p = pred["pressure_final"].values.astype(np.float32)
idx = np.searchsorted(pressure_levels, p, side="left")
idx0 = np.clip(idx - 1, 0, len(pressure_levels) - 1)
idx1 = np.clip(idx, 0, len(pressure_levels) - 1)
p0 = pressure_levels[idx0]
p1 = pressure_levels[idx1]
pred_snapped = np.where(np.abs(p - p0) <= np.abs(p - p1), p0, p1).astype(np.float32)

uout1_fill = float(train.loc[train["u_out"] == 1, "pressure"].median())
pred_snapped = pred_snapped.copy()
pred_snapped[test["u_out"].values.astype(np.int8) == 1] = uout1_fill

submission = sub[["id"]].copy()
if len(pred_snapped) != len(submission):
    raise ValueError(
        f"Row count mismatch: got {len(pred_snapped)} predictions but sub has {len(submission)} rows."
    )
submission["pressure"] = pred_snapped.astype(np.float32)

fallback_p = float(pressure_levels[np.argmin(np.abs(pressure_levels - global_median))])
submission["pressure"] = submission["pressure"].fillna(fallback_p).astype(np.float32)

assert submission.shape[0] == sub.shape[0]
assert list(submission.columns) == ["id", "pressure"]

submission.to_csv("submission.csv", index=False)
submission.head()
