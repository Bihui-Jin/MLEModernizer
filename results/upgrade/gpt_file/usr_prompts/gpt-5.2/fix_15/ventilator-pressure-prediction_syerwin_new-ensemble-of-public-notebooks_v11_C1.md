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

0.1591277623696617

# 6. Current score

1.99878

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.40922) has done: 'I remove the hard dependency on external “../input/…” submission files that are not present in your environment, which is causing the FileNotFoundError and cascading NameError. Instead, the script train a simple per-(R,C) baseline on the provided `train.csv` and predict `test.csv` using only those available files, ensuring it runs end-to-end. This keeps the core intent (generate pressures for all test ids) while making the pipeline self-contained and producing a valid `submission.csv`. The approach is deterministic and lightweight enough to finish under the time limit, and it yield a non-trivial score (better than the all-zeros sample).'
- What this solution (achieved 6.40922) has done: 'Your current score is far worse than the target (lower-is-better), and the biggest issue in the baseline is forcing `pressure=0` whenever `u_out==1`, which creates huge errors because the metric only ignores expiratory *ground truth* but still expects reasonable predictions across the full sequence. I keep your core approach (per-(R,C,u_in_bin) mean lookup with fallbacks) but remove the `u_out==1 -> 0` override and instead use the same learned lookup for all timesteps, which is the smallest change that should materially reduce MAE. I also make the paths robust to your provided `/kaggle/data/...` layout (without changing I/O semantics) so it runs in this environment and still writes `submission.csv`. No model/loop/architecture changes are introduced—this remains a deterministic aggregation baseline.'
- What this solution (achieved 3.85221) has done: 'Your current MAE (6.40922, lower-is-better) is far from the target (0.1591), so we need a real improvement while keeping your “lookup from train aggregates” core logic. The biggest gain with minimal change is to respect the evaluation: MAE is computed only where `u_out==0`, so for `u_out==1` we can safely output any value without affecting score—setting those predictions to something stable (e.g., the per-(R,C) mean) removes noise without changing scored timesteps. For the inspiratory (`u_out==0`) timesteps, we keep your bin-mean lookup but add one tiny, deterministic enhancement: also condition on the discretized `time_step` (the pressure curve is highly time-dependent), with clean fallbacks to your existing `(R,C,u_in_bin)` then `(R,C)` then global mean. This stays in the same “groupby mean table + merge” approach and should move the score substantially toward the target without changing the overall method.'
- What this solution (achieved 2.11045) has done: 'Your current score (3.85 MAE, lower-is-better) is still far from the target (0.159), so we need a meaningful improvement but with the same “groupby-mean lookup + merge + fallbacks” core logic. The largest low-risk gain is to condition the lookup on more time-history without changing modeling style: add lagged `u_in` features within each breath and a cumulative-integral (`u_in` area) feature, then discretize them and extend the mean tables accordingly. This better captures the pressure trajectory dependence while remaining the same deterministic aggregation baseline (no new model, no training loop, no loss). We keep your existing fallbacks and submission alignment to ensure a valid `submission.csv` is always produced.'
- What this solution (achieved 1.52827) has done: 'Your current MAE (2.11045, lower-is-better) is still far above the target (0.159), so we need a meaningful improvement while keeping your same “groupby mean lookup + merge + fallback” core logic. The biggest low-risk issue is that your discretization is far too fine (especially `area_bin`), causing extreme sparsity and lots of fallback to coarse means; we reduce sparsity by using coarser, deterministic bins for `u_in`, `time_step`, and especially `u_in_area` while keeping the same lookup-table approach and fallback chain. We also make `u_in_area` integration slightly more faithful by using trapezoidal integration (still a cumulative integral feature) to better align with pressure dynamics without changing the method class. These are minimal, metric-relevant changes that should move MAE substantially downward while still producing the same valid `submission.csv`.'
- What this solution (achieved 2.25974) has done: 'Your current MAE (1.528) is still far above the target (0.159, lower-is-better), so we need a modest but meaningful improvement while keeping your same “groupby mean lookup + merge + fallbacks” logic. The least invasive win is to make the time/area discretization better match the data’s true granularity: `time_step` effectively lies on a fixed 80-step grid per breath, so binning by the within-breath step index (0..79) is more stable than rounding floats; similarly, scaling `u_in_area` bins relative to its range reduces sparsity without changing features. I also keep your u_out handling identical in spirit (unscored region uses a stable fallback) and preserve the same fallback chain, just with better-aligned bins to reduce NaNs and improve the scored inspiratory predictions. These changes should move MAE downward (toward the target) without changing the overall method class.'
- What this solution (achieved 1.95742) has done: 'Your current MAE (2.2597, lower-is-better) is still far above the target (0.1591), so we should improve the lookup accuracy while keeping the same “groupby-mean tables + merge + fallback” core logic. The smallest high-impact fix is to make the discretization less lossy but without making it too sparse: keep your existing features, but (1) make `u_in` and lag bins slightly finer (1.0 instead of 2.0) and (2) make `u_in_area` bins data-driven by using quantile-based bins computed from train inspiratory rows (still deterministic, still a discretized lookup key). This reduces systematic averaging error on the inspiratory phase (the only scored region) while preserving the same prediction pipeline and fallback chain. All paths and the submission-writing logic remain unchanged, and it still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 1.95715) has done: 'We keep your deterministic “groupby-mean lookup + merge + fallback” pipeline unchanged in spirit, but make two small metric-relevant fixes to move MAE down toward the target. First, we clamp `area_bin` to the valid quantile-bin range so test values outside the train range don’t become out-of-vocabulary and force unnecessary fallbacks. Second, we add a final lightweight post-processing step that snaps predictions to the nearest allowed pressure value observed in training (the target is effectively quantized), which typically reduces MAE without changing the underlying modeling logic. All file paths, feature construction, merge/fallback chain, and submission writing remain the same.'
- What this solution (achieved 1.95715) has done: 'Your current MAE (1.957) is still far above the target (0.159, lower-is-better), so we should improve the lookup accuracy while keeping the same deterministic “groupby mean tables + merge + fallback + pressure snapping” core logic. The smallest metric-aligned gain is to add one extra, very lightweight history signal that strongly correlates with pressure: a discretized cumulative count of `u_out==1` events within a breath (valve-open history), and condition the highest-resolution mean table on it. This keeps the same feature-engineering style (within-breath history + discretize + groupby mean), the same merge/fallback chain, and still runs fast. Everything else (paths, submission alignment, snapping to training pressure levels, and CSV output) stays unchanged.'
- What this solution (achieved 1.95715) has done: 'Your current MAE (1.957, lower-is-better) is far above the target (0.159), so we need a small but meaningful improvement while keeping your exact “discretize features → groupby mean tables → merge → fallback → snap-to-pressure-levels” pipeline. The highest-leverage minimal fix is to stop using `u_out_cum` in the highest-resolution lookup: in this dataset `u_out` is essentially a phase flag (0 during inspiration, then 1), so cumulative `u_out` is almost always 0 on scored rows and creates unnecessary sparsity/out-of-vocabulary behavior on test, increasing fallback. Instead, we condition the most-detailed table on `u_out` itself (binary) which preserves the intended phase information while being much denser and more stable. Everything else (features, binning, fallback chain, u_out==1 handling, and pressure snapping) is kept the same so the core logic and semantics remain intact and it still writes a valid `submission.csv`.'
- What this solution (achieved 1.95715) has done: 'Your current MAE (1.957, lower-is-better) is still far above the target (0.159), so we need a modest improvement while keeping the exact same “discretize features → groupby mean tables → merge → fallback → snap-to-pressure-levels” approach. The smallest high-impact fix here is that you are training all lookup tables only on inspiratory rows (`u_out==0`) but then also keying the most-detailed table by `u_out_bin`; this makes `u_out_bin` always 0 in training, so `p_mean0` be missing for all `u_out==1` rows in test and forces unnecessary fallback. I keep your u_out handling idea (unscored region uses a stable fallback) but make the most-detailed table *not* depend on `u_out_bin`, eliminating pointless sparsity and slightly improving the fallback behavior overall without changing the modeling style. Everything else (features, quantile area bins with clamping, fallback chain, and pressure snapping) remains the same and it still writes a valid `submission.csv`.'
- What this solution (achieved 1.94289) has done: 'Your current MAE (1.957, lower-is-better) is still far from the target (0.159), so we should make a small but high-impact improvement without changing your core “discretize → groupby mean tables → merge → fallback → snap-to-levels” approach. The biggest issue is that your finest-grain lookup tables are *means*, but MAE is minimized (per key) by the *median*; switching the finest few aggregations from mean→median is a minimal semantic change that should reduce MAE on the scored inspiratory rows. To keep behavior stable and avoid hurting coverage, the coarse fallback tables remain means, and the merge/fallback chain is unchanged. Everything still runs end-to-end and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 1.99056) has done: 'We keep your exact “discretize → groupby lookup → fallback chain → snap-to-pressure-levels” pipeline, but make the aggregation consistent with MAE by using median (instead of mean) for the remaining fallback tables that still use mean. This is a minimal, metric-aligned change: for each key, the median minimizes expected absolute error, and it should reduce errors when you fall back from the finest table. No feature engineering, binning, merge keys, fallback order, u_out handling, or snapping logic is changed, so runtime and semantics stay the same while nudging MAE downward toward the target.'
- What this solution (achieved 1.99878) has done: 'Your current MAE (1.99056, lower-is-better) is still far above the target (0.1591), so we should make a small, metric-aligned improvement while keeping your exact “discretize → groupby lookup → fallback → snap-to-levels” approach. The biggest low-risk gain here is to stop throwing away training data by building lookup tables only on inspiratory rows: we can build the same tables on *all* training rows while keeping scoring semantics unchanged (Kaggle only scores u_out==0, but better calibration across the full trajectory helps the learned medians and fallbacks). To preserve your existing intent for unscored expiratory rows, we keep your `u_out==1` prediction override (stable per-(R,C) fallback), but now that fallback is also learned from all rows. No model/loop/architecture changes are introduced—this remains deterministic groupby-median lookup with the same keys and snapping.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
CANDIDATE_BASES = [
    "/kaggle/data/ventilator-pressure-prediction",
    "/kaggle/input/ventilator-pressure-prediction",
    "../input/ventilator-pressure-prediction",
    "/kaggle/data",
    "/kaggle/input",
    "../input",
]


def _find_file(filename: str) -> str:
    for base in CANDIDATE_BASES:
        p = os.path.join(base, filename)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find {filename} in any of: {CANDIDATE_BASES}")


train_path = _find_file("train.csv")
test_path = _find_file("test.csv")
sub_path = _find_file("sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sub = pd.read_csv(sub_path)


def _add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["breath_id", "time_step"]).copy()

    g = df.groupby("breath_id", sort=False)
    df["step"] = g.cumcount().astype(np.int16)

    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0).astype(np.float32)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0).astype(np.float32)

    dt = g["time_step"].diff().fillna(0.0).astype(np.float32)
    u_in_f = df["u_in"].astype(np.float32)
    u_in_prev = g["u_in"].shift(1).fillna(0.0).astype(np.float32)
    df["u_in_area"] = (
        (((u_in_f + u_in_prev) * 0.5) * dt)
        .groupby(df["breath_id"], sort=False)
        .cumsum()
        .astype(np.float32)
    )

    df["u_out_cum"] = g["u_out"].cumsum().astype(np.int16)

    return df


train_h = _add_history_features(train)
test_h = _add_history_features(test)

train_all = train_h.loc[
    :,
    [
        "R",
        "C",
        "step",
        "time_step",
        "u_in",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_area",
        "u_out",
        "u_out_cum",
        "pressure",
    ],
].copy()

UIN_STEP = 1.0


def _uin_bin(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x / UIN_STEP), 0, int(100 / UIN_STEP)).astype(np.int16)


def _step_bin(x: np.ndarray) -> np.ndarray:
    return x.astype(np.int16)


def _uout_bin(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.int16) != 0).astype(np.int16)


_AREA_Q = 64
_area_edges = np.unique(
    np.quantile(
        train_all["u_in_area"].to_numpy(np.float32), np.linspace(0.0, 1.0, _AREA_Q + 1)
    )
).astype(np.float32)
if _area_edges.size < 3:
    _area_edges = np.array(
        [0.0, float(train_all["u_in_area"].max()) + 1e-6], dtype=np.float32
    )

_AREA_NBINS = int(max(_area_edges.size - 1, 1))


def _area_bin_quantile(x: np.ndarray) -> np.ndarray:
    b = (
        np.searchsorted(_area_edges, x.astype(np.float32), side="right").astype(
            np.int16
        )
        - 1
    )
    return np.clip(b, 0, _AREA_NBINS - 1).astype(np.int16)


train_all["u_in_bin"] = _uin_bin(train_all["u_in"].to_numpy(np.float32))
train_all["u_in_lag1_bin"] = _uin_bin(train_all["u_in_lag1"].to_numpy(np.float32))
train_all["u_in_lag2_bin"] = _uin_bin(train_all["u_in_lag2"].to_numpy(np.float32))
train_all["t_bin"] = _step_bin(train_all["step"].to_numpy(np.int16))
train_all["area_bin"] = _area_bin_quantile(train_all["u_in_area"].to_numpy(np.float32))
train_all["u_out_bin"] = _uout_bin(train_all["u_out"].to_numpy(np.int16))

rc_t_uin_lags_area_mean0 = (
    train_all.groupby(
        [
            "R",
            "C",
            "t_bin",
            "u_in_bin",
            "u_in_lag1_bin",
            "u_in_lag2_bin",
            "area_bin",
        ],
        as_index=False,
    )["pressure"]
    .median()
    .rename(columns={"pressure": "p_mean0"})
)

rc_t_uin_lags_area_mean = (
    train_all.groupby(
        ["R", "C", "t_bin", "u_in_bin", "u_in_lag1_bin", "u_in_lag2_bin", "area_bin"],
        as_index=False,
    )["pressure"]
    .median()
    .rename(columns={"pressure": "p_mean"})
)

rc_t_uin_lags_mean = (
    train_all.groupby(
        ["R", "C", "t_bin", "u_in_bin", "u_in_lag1_bin", "u_in_lag2_bin"],
        as_index=False,
    )["pressure"]
    .median()
    .rename(columns={"pressure": "p_mean2"})
)

rc_t_uin_mean = (
    train_all.groupby(["R", "C", "t_bin", "u_in_bin"], as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p_mean3"})
)

rc_uin_mean = (
    train_all.groupby(["R", "C", "u_in_bin"], as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p_rc_uin_mean"})
)

rc_mean = (
    train_all.groupby(["R", "C"], as_index=False)["pressure"]
    .median()
    .rename(columns={"pressure": "p_rc_mean"})
)

p_global = float(train_all["pressure"].median())

test_pred = test_h.loc[
    :,
    [
        "id",
        "breath_id",
        "R",
        "C",
        "step",
        "time_step",
        "u_in",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_area",
        "u_out",
        "u_out_cum",
    ],
].copy()

test_pred["u_in_bin"] = _uin_bin(test_pred["u_in"].to_numpy(np.float32))
test_pred["u_in_lag1_bin"] = _uin_bin(test_pred["u_in_lag1"].to_numpy(np.float32))
test_pred["u_in_lag2_bin"] = _uin_bin(test_pred["u_in_lag2"].to_numpy(np.float32))
test_pred["t_bin"] = _step_bin(test_pred["step"].to_numpy(np.int16))
test_pred["area_bin"] = _area_bin_quantile(test_pred["u_in_area"].to_numpy(np.float32))
test_pred["u_out_bin"] = _uout_bin(test_pred["u_out"].to_numpy(np.int16))

test_pred = test_pred.merge(
    rc_t_uin_lags_area_mean0,
    on=[
        "R",
        "C",
        "t_bin",
        "u_in_bin",
        "u_in_lag1_bin",
        "u_in_lag2_bin",
        "area_bin",
    ],
    how="left",
)
test_pred = test_pred.merge(
    rc_t_uin_lags_area_mean,
    on=["R", "C", "t_bin", "u_in_bin", "u_in_lag1_bin", "u_in_lag2_bin", "area_bin"],
    how="left",
)
test_pred = test_pred.merge(
    rc_t_uin_lags_mean,
    on=["R", "C", "t_bin", "u_in_bin", "u_in_lag1_bin", "u_in_lag2_bin"],
    how="left",
)
test_pred = test_pred.merge(
    rc_t_uin_mean, on=["R", "C", "t_bin", "u_in_bin"], how="left"
)
test_pred = test_pred.merge(rc_uin_mean, on=["R", "C", "u_in_bin"], how="left")
test_pred = test_pred.merge(rc_mean, on=["R", "C"], how="left")

pred = test_pred["p_mean0"].to_numpy(dtype=np.float32)

mask = np.isnan(pred)
if mask.any():
    pred[mask] = test_pred.loc[mask, "p_mean"].to_numpy(dtype=np.float32)

mask = np.isnan(pred)
if mask.any():
    pred[mask] = test_pred.loc[mask, "p_mean2"].to_numpy(dtype=np.float32)

mask = np.isnan(pred)
if mask.any():
    pred[mask] = test_pred.loc[mask, "p_mean3"].to_numpy(dtype=np.float32)

mask = np.isnan(pred)
if mask.any():
    pred[mask] = test_pred.loc[mask, "p_rc_uin_mean"].to_numpy(dtype=np.float32)

mask = np.isnan(pred)
if mask.any():
    pred[mask] = test_pred.loc[mask, "p_rc_mean"].to_numpy(dtype=np.float32)

mask = np.isnan(pred)
if mask.any():
    pred[mask] = np.float32(p_global)

uout1 = test_pred["u_out"].to_numpy() == 1
if uout1.any():
    rc_fallback = test_pred["p_rc_mean"].to_numpy(dtype=np.float32)
    rc_fallback = np.where(np.isnan(rc_fallback), np.float32(p_global), rc_fallback)
    pred[uout1] = rc_fallback[uout1]

pressure_levels = np.sort(train["pressure"].unique()).astype(np.float32)
idx = np.searchsorted(pressure_levels, pred, side="left")
idx = np.clip(idx, 0, pressure_levels.size - 1)
idx0 = np.clip(idx - 1, 0, pressure_levels.size - 1)
p_hi = pressure_levels[idx]
p_lo = pressure_levels[idx0]
pred = np.where((pred - p_lo) <= (p_hi - pred), p_lo, p_hi).astype(np.float32)

out = pd.DataFrame(
    {"id": test_pred["id"].astype(np.int64), "pressure": pred.astype(np.float32)}
)
out = out.sort_values("id").reset_index(drop=True)

out = sub[["id"]].merge(out, on="id", how="left")
out["pressure"] = out["pressure"].fillna(np.float32(p_global)).astype(np.float32)

out.to_csv("submission.csv", index=False)
out.head()



## === cell 2
assert os.path.exists("submission.csv"), "submission.csv was not created"
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["id", "pressure"], f"Wrong columns: {chk.columns.tolist()}"
assert len(chk) == len(sub), f"Row count mismatch: got {len(chk)}, expected {len(sub)}"
chk.describe(include="all")
