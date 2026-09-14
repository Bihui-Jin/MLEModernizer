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

0.3067978092570314

# 6. Current score

4.40915

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.04865) has done: 'The error comes from trying to load three external Kaggle dataset submissions that are not present in your environment, so the notebook stops before writing any `submission.csv`. To keep the core idea (simple ensembling) but make it runnable end-to-end, I replace those missing inputs with a minimal, deterministic baseline model trained from the provided `train.csv`, and then write predictions for `test.csv` in the required `id,pressure` format. This also legitimately improves score from “no submission” to a real (though simple) model without changing the evaluation semantics. Paths are updated to use the available `/kaggle/input/...` competition files.'
- What this solution (achieved 3.99964) has done: 'Your current score (MAE 4.04865; lower is better) is far above the target (~0.3068), so we need a legitimate accuracy improvement while keeping your core approach (lookup-table mean by binned features) intact. The biggest issue is that you overwrite all expiratory-phase (`u_out==1`) predictions with a global mean, which is unnecessary because the metric ignores expiratory rows; instead, we should just predict normally there too (harmless to the metric and can help if Kaggle has any full-file checks). The main accuracy gain (still within your “mean lookup” logic) comes from adding simple time-series derived features that don’t change the modeling approach: cumulative `u_in`, lagged `u_in`, and a “time since inspiration started” proxy via cumulative count of `u_out==0`, then grouping on their binned versions as well. Finally, we fix the subtle alignment bug: you sort `sub`/`test` but never reorder `test_pred` to match; we build predictions in the same sorted order to guarantee correct `id -> pressure` mapping.'
- What this solution (achieved 3.97843) has done: 'Your current MAE (3.99964; lower is better) is far worse than the target (~0.3068), so we should improve accuracy with minimal, core-logic-preserving changes to your lookup-table mean approach. The largest gain available without changing the modeling paradigm is to add a second lookup that conditions on `u_out` and a coarse “phase within breath” (row index), because pressure dynamics differ strongly across the breath and your current features don’t encode step position robustly. We compute a per-breath `step` (0–79) and add an additional fallback map keyed by `(R,C,u_out,step,t_b)`; this increases match rate and reduces reliance on the global mean while keeping the same “groupby-mean then merge with fallbacks” logic. Finally, we keep your id alignment guarantees and still write a valid `submission.csv`.'
- What this solution (achieved 3.97843) has done: 'Your current MAE (3.97843; lower is better) is still far above the target (~0.3068), so we need a legitimate accuracy boost while keeping your core “lookup-table mean with fallbacks” approach unchanged. The biggest gap is that the main lookup is trained only on inspiratory rows (`u_out==0`), so expiratory rows in test almost always miss and fall back to coarse/global means; while expiratory isn’t scored, those misses also happen near the transition and reduce match quality. I (1) build the primary `mean_map` on **all rows** but include `u_out` in the key, so inspiratory dynamics are still separated, and (2) make the phase proxy more consistent by defining `insp_step` as a within-breath count only during inspiration (and 0 during expiration), which improves key consistency without changing the modeling paradigm. Everything else (binning, groupby-mean, merge, fallbacks, submission writing and id alignment) stays the same.'
- What this solution (achieved 3.97843) has done: 'Your current MAE (3.978) is far above the target (0.307), so we should improve accuracy while keeping your core “groupby-mean lookup with fallbacks” approach intact. The biggest win within the same logic is to (1) use an inspiration-specific time index (`insp_step`) for the phase-based fallback (instead of `step` which mixes expiration timing) and (2) make the “global mean” fallback depend on `(R,C,insp_step)` rather than a single scalar, which reduces error when lookups miss. These changes don’t alter the modeling paradigm (still pure lookup-table means + deterministic fallbacks) and should move the score substantially downward toward the target. Everything else (feature extraction, binning, merge-based prediction, submission alignment) remains the same.'
- What this solution (achieved 3.97788) has done: 'Your current MAE (3.97843; lower is better) is still far from the target (~0.3068), so we need a real accuracy gain while keeping your core “groupby-mean lookup with fallbacks” approach intact. The biggest issue is that your most-specific lookup key uses continuous/binned values that still don’t uniquely capture the pressure dynamics; a minimal, high-signal addition is to include lagged `u_in` (binned) and an `u_in`-integral during inspiration (binned), which are classic ventilator features and remain fully within your deterministic feature+lookup paradigm. We keep your existing maps and add one extra, more-specific mean-map keyed on these new features, used before existing fallbacks to reduce misses and large errors. All I/O paths, submission schema, and alignment checks remain unchanged.'
- What this solution (achieved 3.97702) has done: 'Your score (MAE 3.97788; lower is better) is still far above the target (~0.3068), so we need a legitimate accuracy gain while keeping your “groupby-mean lookup with fallbacks” core logic intact. The biggest issue is that your most-specific keys are still too sparse, so many test rows miss and fall back to coarse averages; a minimal, high-signal fix is to add a more stable “within-breath u_in percentile rank” feature (binned) and use it only as an extra, more-specific lookup layer before your existing fallbacks. This preserves the same deterministic feature engineering + lookup-table mean paradigm and doesn’t change any training loop/model/loss (there is none). I also keep your existing alignment checks and submission writing unchanged.'
- What this solution (achieved 3.97696) has done: 'Your current MAE (3.977) is far worse than the target (~0.3068), so we should improve accuracy while keeping your existing deterministic “groupby-mean lookup with fallbacks” core logic unchanged. The biggest low-risk gain here is to constrain predictions to the discrete pressure grid used in this competition by snapping outputs to the nearest observed pressure value from training; this preserves evaluation semantics but reduces error because true labels are quantized. We also make the pressure grid extraction explicit and stable, and apply snapping only at the very end so all your existing lookup layers remain identical. This is a minimal change, should run within the time limit, and still writes a valid `submission.csv`.'
- What this solution (achieved 3.97696) has done: 'We keep your deterministic “groupby-mean lookup with fallbacks” approach intact, but fix one key feature bug that is currently making the phase feature inconsistent: `insp_step` is computed with a groupby-apply that can misalign indices and also counts expiratory steps before being zeroed, which harms lookup hit-rate for inspiratory rows (the only scored rows). We recompute `insp_step` using a fully vectorized, index-aligned within-breath cumsum of `(u_out==0)` so it’s correct and stable. Everything else (binning scheme, lookup tables, fallback order, pressure-grid snapping, file paths, submission writing) stays the same to preserve core logic while legitimately improving MAE toward the target. The resulting script still runs end-to-end and writes `submission.csv` with `id,pressure`.'
- What this solution (achieved 4.04649) has done: 'Your current MAE (3.97696; lower is better) is still far above the target (~0.3068), and the main reason is that a pure mean-lookup approach is producing many mismatches and coarse fallbacks. Keeping your core logic (deterministic feature engineering + groupby-mean lookup with merge-based fallbacks) identical, I add one more *lookup layer* that is still the same paradigm but uses a higher-signal discretization: snap `u_in` and `u_in_lag1` to the *exact discrete values observed in train* (instead of coarse rounding), then use that map first. This typically increases exact-key hit rate without changing the approach, and it’s fast because `u_in` only takes a limited set of values. Everything else (existing maps, fallback order, pressure-grid snapping, paths, submission writing) stays the same.'
- What this solution (achieved 4.40915) has done: 'We need to move your MAE down toward 0.3068 (lower is better), and your current 4.046 suggests the current lookup keys are not matching the underlying pressure dynamics well. Keeping your same core “feature engineering → groupby mean maps → merge + fallback → snap to pressure grid” logic, the smallest high-impact fix is to correct the main high-specificity map: it currently keys on raw `time_step` floats, which almost never match exactly between train/test due to floating-point representation, causing massive miss rates and heavy fallback usage. I replace that key with the already-binned `t_b` (consistent with your other maps) so the grid map actually hits, and I keep everything else (maps, fallback order, snapping, I/O) unchanged. This should materially reduce error while preserving your approach and staying well within the runtime limit.'
- What this solution (achieved 4.40915) has done: 'Your current MAE (4.40915; lower is better) is still far above the target (~0.3068), so we should improve accuracy with the smallest change that increases lookup hit-rate on inspiratory rows (the only scored phase) without changing the overall “feature engineering → groupby-mean maps → merge + fallback → snap to pressure grid” logic. The most impactful low-risk fix is to add a dedicated inspiratory-only high-specificity map (same key structure you already use) and apply it only when `u_out==0`, so we don’t dilute inspiratory dynamics with expiratory behavior. This preserves your architecture and all existing maps/fallbacks, but gives scored rows a much better first-match chance. Everything else (paths, alignment, snapping, submission writing) stays the same.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/input/ventilator-pressure-prediction"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sub = pd.read_csv(sample_path)

assert {"id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"}.issubset(
    train.columns
)
assert {"id", "breath_id", "R", "C", "time_step", "u_in", "u_out"}.issubset(
    test.columns
)
assert {"id", "pressure"}.issubset(sub.columns)

train.shape, test.shape, sub.shape




## === cell 1
def add_ts_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["breath_id", "time_step"]).copy()
    g = df.groupby("breath_id", sort=False)

    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0)
    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_in_cum"] = g["u_in"].cumsum()

    df["step"] = g.cumcount().astype(np.int16)

    df["insp_step"] = (
        g["u_out"]
        .transform(lambda s: (s.to_numpy() == 0).astype(np.int16).cumsum())
        .astype(np.int16)
    )
    df.loc[df["u_out"] == 1, "insp_step"] = 0

    df["u_in_insp_cum"] = (
        df["u_in"]
        .where(df["u_out"].eq(0), 0.0)
        .groupby(df["breath_id"], sort=False)
        .cumsum()
    )

    df["u_in_rank"] = g["u_in"].rank(method="average", pct=True).astype(np.float32)

    return df


train_feat = add_ts_features(train)
test_feat = add_ts_features(test)

u_in_bin = 0.5
t_bin = 0.01
u_in_cum_bin = 2.0
u_in_diff_bin = 0.5
insp_step_bin = 2.0

u_in_lag_bin = 0.5
u_in_insp_cum_bin = 2.0

u_in_rank_bin = 0.02  # 50 bins across [0,1]

train_all = train_feat.copy()

train_all["u_in_b"] = (train_all["u_in"] / u_in_bin).round(0) * u_in_bin
train_all["t_b"] = (train_all["time_step"] / t_bin).round(0) * t_bin
train_all["u_in_cum_b"] = (train_all["u_in_cum"] / u_in_cum_bin).round(0) * u_in_cum_bin
train_all["u_in_diff_b"] = (train_all["u_in_diff1"] / u_in_diff_bin).round(
    0
) * u_in_diff_bin
train_all["insp_step_b"] = (train_all["insp_step"] / insp_step_bin).round(
    0
) * insp_step_bin

train_all["u_in_lag1_b"] = (train_all["u_in_lag1"] / u_in_lag_bin).round(
    0
) * u_in_lag_bin
train_all["u_in_insp_cum_b"] = (train_all["u_in_insp_cum"] / u_in_insp_cum_bin).round(
    0
) * u_in_insp_cum_bin

train_all["u_in_rank_b"] = (train_all["u_in_rank"] / u_in_rank_bin).round(
    0
) * u_in_rank_bin

grp_cols = [
    "R",
    "C",
    "u_out",
    "u_in_b",
    "t_b",
    "u_in_cum_b",
    "u_in_diff_b",
    "insp_step_b",
]

mean_map = (
    train_all.groupby(grp_cols, observed=True)["pressure"]
    .mean()
    .rename("pred")
    .reset_index()
)

grp_cols_v2 = grp_cols + ["u_in_lag1_b", "u_in_insp_cum_b"]
mean_map_v2 = (
    train_all.groupby(grp_cols_v2, observed=True)["pressure"]
    .mean()
    .rename("pred_v2")
    .reset_index()
)

grp_cols_v3 = grp_cols_v2 + ["u_in_rank_b"]
mean_map_v3 = (
    train_all.groupby(grp_cols_v3, observed=True)["pressure"]
    .mean()
    .rename("pred_v3")
    .reset_index()
)

u_in_grid = np.sort(train_all["u_in"].unique()).astype(np.float32)


def snap_to_grid(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    idx = np.searchsorted(grid, values, side="left")
    idx0 = np.clip(idx - 1, 0, len(grid) - 1)
    idx1 = np.clip(idx, 0, len(grid) - 1)
    g0 = grid[idx0]
    g1 = grid[idx1]
    return np.where(np.abs(values - g0) <= np.abs(values - g1), g0, g1).astype(
        np.float32
    )


train_all["u_in_g"] = snap_to_grid(train_all["u_in"].to_numpy(), u_in_grid)
train_all["u_in_lag1_g"] = snap_to_grid(train_all["u_in_lag1"].to_numpy(), u_in_grid)

grp_cols_grid = ["R", "C", "u_out", "t_b", "u_in_g", "u_in_lag1_g"]
mean_map_grid = (
    train_all.groupby(grp_cols_grid, observed=True)["pressure"]
    .mean()
    .rename("pred_grid")
    .reset_index()
)

train_insp_all = train_all[train_all["u_out"] == 0].copy()
grp_cols_grid_insp = ["R", "C", "t_b", "u_in_g", "u_in_lag1_g"]
mean_map_grid_insp = (
    train_insp_all.groupby(grp_cols_grid_insp, observed=True)["pressure"]
    .mean()
    .rename("pred_grid_insp")
    .reset_index()
)

train_insp = train_feat[train_feat["u_out"] == 0].copy()
rc_inspstep_mean_map = (
    train_insp.groupby(["R", "C", "insp_step"], observed=True)["pressure"]
    .mean()
    .rename("pred_rc_inspstep")
    .reset_index()
)

rc_t_map = (
    train_all.groupby(["R", "C", "t_b"], observed=True)["pressure"]
    .mean()
    .rename("pred_rc_t")
    .reset_index()
)

rc_uout_inspstep_t_map = (
    train_all.groupby(["R", "C", "u_out", "insp_step", "t_b"], observed=True)[
        "pressure"
    ]
    .mean()
    .rename("pred_rc_uout_inspstep_t")
    .reset_index()
)

global_mean = float(train_insp["pressure"].mean())

pressure_grid = np.sort(train["pressure"].unique()).astype(np.float32)

(
    global_mean,
    pressure_grid.shape,
    mean_map.shape,
    mean_map_v2.shape,
    mean_map_v3.shape,
    mean_map_grid.shape,
    mean_map_grid_insp.shape,
    rc_t_map.shape,
    rc_uout_inspstep_t_map.shape,
    rc_inspstep_mean_map.shape,
)



## === cell 2
test_sorted = test_feat.sort_values("id").reset_index(drop=True)

test_sorted["u_in_b"] = (test_sorted["u_in"] / u_in_bin).round(0) * u_in_bin
test_sorted["t_b"] = (test_sorted["time_step"] / t_bin).round(0) * t_bin
test_sorted["u_in_cum_b"] = (test_sorted["u_in_cum"] / u_in_cum_bin).round(
    0
) * u_in_cum_bin
test_sorted["u_in_diff_b"] = (test_sorted["u_in_diff1"] / u_in_diff_bin).round(
    0
) * u_in_diff_bin
test_sorted["insp_step_b"] = (test_sorted["insp_step"] / insp_step_bin).round(
    0
) * insp_step_bin

test_sorted["u_in_lag1_b"] = (test_sorted["u_in_lag1"] / u_in_lag_bin).round(
    0
) * u_in_lag_bin
test_sorted["u_in_insp_cum_b"] = (
    test_sorted["u_in_insp_cum"] / u_in_insp_cum_bin
).round(0) * u_in_insp_cum_bin

test_sorted["u_in_rank_b"] = (test_sorted["u_in_rank"] / u_in_rank_bin).round(
    0
) * u_in_rank_bin

test_sorted["u_in_g"] = snap_to_grid(test_sorted["u_in"].to_numpy(), u_in_grid)
test_sorted["u_in_lag1_g"] = snap_to_grid(
    test_sorted["u_in_lag1"].to_numpy(), u_in_grid
)

pred_df = test_sorted.merge(mean_map_grid_insp, on=grp_cols_grid_insp, how="left")

pred_df = pred_df.merge(mean_map_grid, on=grp_cols_grid, how="left")
pred_df = pred_df.merge(mean_map_v3, on=grp_cols_v3, how="left")
pred_df = pred_df.merge(mean_map_v2, on=grp_cols_v2, how="left")
pred_df = pred_df.merge(mean_map, on=grp_cols, how="left")

pred_df = pred_df.merge(rc_t_map, on=["R", "C", "t_b"], how="left")
pred_df = pred_df.merge(
    rc_uout_inspstep_t_map, on=["R", "C", "u_out", "insp_step", "t_b"], how="left"
)
pred_df = pred_df.merge(rc_inspstep_mean_map, on=["R", "C", "insp_step"], how="left")

pred_grid_insp = pred_df["pred_grid_insp"].to_numpy()
pred_grid = pred_df["pred_grid"].to_numpy()
pred_v3 = pred_df["pred_v3"].to_numpy()
pred_v2 = pred_df["pred_v2"].to_numpy()
pred_v1 = pred_df["pred"].to_numpy()
fallback_rc_t = pred_df["pred_rc_t"].to_numpy()
fallback_rc_uout_inspstep_t = pred_df["pred_rc_uout_inspstep_t"].to_numpy()
fallback_rc_inspstep = pred_df["pred_rc_inspstep"].to_numpy()

u_out_arr = pred_df["u_out"].to_numpy()

test_pred = np.full(len(pred_df), np.nan, dtype=np.float32)
test_pred = np.where(
    (u_out_arr == 0) & ~np.isnan(pred_grid_insp), pred_grid_insp, test_pred
)

test_pred = np.where(np.isnan(test_pred), pred_grid, test_pred)
test_pred = np.where(np.isnan(test_pred), pred_v3, test_pred)
test_pred = np.where(np.isnan(test_pred), pred_v2, test_pred)
test_pred = np.where(np.isnan(test_pred), pred_v1, test_pred)
test_pred = np.where(np.isnan(test_pred), fallback_rc_uout_inspstep_t, test_pred)
test_pred = np.where(np.isnan(test_pred), fallback_rc_t, test_pred)
test_pred = np.where(np.isnan(test_pred), fallback_rc_inspstep, test_pred)
test_pred = np.where(np.isnan(test_pred), global_mean, test_pred).astype(np.float32)

idx = np.searchsorted(pressure_grid, test_pred, side="left")
idx0 = np.clip(idx - 1, 0, len(pressure_grid) - 1)
idx1 = np.clip(idx, 0, len(pressure_grid) - 1)
p0 = pressure_grid[idx0]
p1 = pressure_grid[idx1]
test_pred_snapped = np.where(
    np.abs(test_pred - p0) <= np.abs(test_pred - p1), p0, p1
).astype(np.float32)

sub_sorted = sub.sort_values("id").reset_index(drop=True)
assert sub_sorted["id"].equals(
    test_sorted["id"]
), "ID alignment mismatch between sample_submission and test.csv"

sub_sorted["pressure"] = test_pred_snapped
sub_sorted.head()



## === cell 3
out_path = "submission.csv"
sub_sorted.to_csv(out_path, index=False)

check = pd.read_csv(out_path)
assert list(check.columns) == ["id", "pressure"]
assert len(check) == len(test)
out_path
