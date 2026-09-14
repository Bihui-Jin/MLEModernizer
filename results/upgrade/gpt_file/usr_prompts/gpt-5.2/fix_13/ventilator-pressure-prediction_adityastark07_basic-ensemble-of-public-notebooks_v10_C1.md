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

0.1439620186632863

# 6. Current score

1.7831

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 5.753) has done: 'The notebook currently fails because it tries to read three external submission files that are not present in your environment; this prevents any `.csv` from being generated. I remove that dependency and replace it with a minimal, fully self-contained baseline model that trains on `train.csv` and predicts `pressure` for `test.csv` using only available packages (numpy/pandas). To keep it stable and within time, the approach fits per-(R,C) linear regression on inspiratory rows (`u_out==0`), which matches the competition’s scoring phase and typically scores far better than a constant/zero baseline. Finally, it write a valid `submission.csv` with the required `id,pressure` columns.'
- What this solution (achieved 2.40411) has done: 'Your current score (MAE 5.753, lower is better) is far from the target (0.144), so we need a legitimate but still minimal upgrade that better matches the competition’s structure: pressure is scored only during inspiration and is highly dependent on within-breath dynamics. I keep your closed-form ridge regression core, but fit it per-(R,C) **and per time_step index within the breath** (0–79), which captures the strong time-dependent relationship without changing the modeling family. I also add two very small, physically-motivated features (`u_in` cumulative sum and lag-1 `u_in`) computed within each breath; these are simple deterministic feature extractions and usually reduce MAE dramatically. The submission writing/alignment logic remains the same and still produces a valid `submission.csv`.'
- What this solution (achieved 2.40412) has done: 'Your current MAE (2.404) is still far above the target (0.144), so we should make a small, legitimate improvement that better matches the competition’s structure without changing your ridge-regression core. The biggest gap is that your model predicts pressures during expiratory rows too, even though those rows are not scored and have different behavior; setting `pressure=0` whenever `u_out==1` in the test predictions is a simple, common, metric-aligned post-processing step. Additionally, within-breath dynamics are strongly tied to how `u_in` changes, so we add one minimal deterministic feature (`u_in_diff` within breath) while keeping the same closed-form ridge fitting per (R,C,t_idx). These changes should reduce MAE substantially while preserving your approach and producing the same valid `submission.csv`.'
- What this solution (achieved 2.40415) has done: 'We keep your closed-form ridge regression and grouping logic intact, but make two metric-aligned tweaks that usually improve MAE a lot for this competition: (1) remove `u_out` from the feature matrix since you already train only on `u_out==0` (it’s constant in train and adds an unnecessary coefficient), and (2) apply the well-known “pressure quantization” post-processing by snapping predictions to the discrete pressure grid seen in `train` (the true target takes only 950 discrete values). These are minimal changes that preserve your approach (same model family, same fitting, same grouping), while better matching the evaluation target distribution. The submission writing and paths remain unchanged and still produce a valid `submission.csv`.'
- What this solution (achieved 1.91802) has done: 'Your current MAE (2.404, lower is better) is still far from the target (0.144), so we should make a metric-aligned improvement while keeping the same ridge closed-form approach and per-(R,C,t_idx) grouping. The largest remaining issue is that the relationship between `u_in` and `pressure` is strongly breath-history dependent; adding a couple more deterministic within-breath state features (cumulative `u_out` / “exhalation started” flag and a short lag window of `u_in`) usually improves this a lot without changing the model family or training loop. I also make ridge numerically/scale-stabler by standardizing features within each training group (still closed-form ridge, same semantics) and reusing those group scalers at inference. Submission writing stays identical and still outputs `submission.csv` with `id,pressure`.'
- What this solution (achieved 1.91796) has done: 'Your current MAE (1.918, lower is better) is still far above the target (0.144), so we should make a small, metric-aligned improvement without changing your ridge-regression-per-(R,C,t_idx) core. The biggest remaining mismatch is that your regression is learning a continuous target even though true `pressure` is discrete; switching the regression target to the **nearest pressure grid value during training** (same model, same loss/fit, just label preprocessing) typically reduces MAE after you already snap predictions to the grid. Additionally, the ventilator dynamics depend heavily on within-breath cumulative volume and the interaction with lung attributes, so we add two minimal deterministic interaction features (`u_in_cum/(R+eps)` and `u_in_cum/(C+eps)`) while keeping the exact same closed-form ridge training/inference. These changes are lightweight, keep the same training loop/approach, and still produce a valid `submission.csv`.'
- What this solution (achieved 1.91796) has done: 'We keep your closed-form ridge-per-(R,C,t_idx) approach intact and focus on one metric-aligned fix that should reduce MAE materially: don’t force `pressure=0` when `u_out==1` in the test set, because Kaggle’s metric ignores expiratory rows and your post-processing can unnecessarily distort predictions around the phase transition (and may also create unintended effects if the scoring mask differs from `u_out`). Instead, we leave expiratory predictions as-is (still clipped + snapped on inspiratory rows only), preserving your existing metric alignment while removing a likely source of error. Everything else (features, grouping, standardization, grid snapping, submission alignment) remains unchanged to keep risk low and runtime within limits.'
- What this solution (achieved 1.78594) has done: 'Your current MAE (1.91796, lower is better) is still far above the target (0.14396), so we should make a small, metric-aligned improvement without changing your ridge-per-(R,C,t_idx) closed-form core. The biggest gain with minimal risk is to add a single breath-history feature that approximates delivered volume: the within-breath integral of flow, i.e., `u_in * delta_time` cumulative sum; this often helps substantially because pressure depends on accumulated air. To keep semantics stable, we compute `delta_time` within breath from `time_step`, add `u_in_dt` and `u_in_dt_cum` to the existing feature matrix, and keep the same training/inference, standardization, and grid snapping. Everything else (grouping, ridge alpha, submission alignment and writing) remains unchanged and still produces `submission.csv`.'
- What this solution (achieved 1.78594) has done: 'We keep your same closed-form ridge regression per-(R,C,t_idx) with standardization and grid snapping, but fix one key mismatch: the regression currently has no explicit bias for the “baseline pressure” that depends strongly on lung settings (R,C) and breath progression (t_idx). I add a single deterministic feature, `rc_t_mean_pressure`, computed on train inspiratory rows as the mean pressure for each (R,C,t_idx) group and merged into both train/test (with a global fallback), which often reduces MAE a lot while preserving your modeling family and loop. To avoid leaking information across breaths improperly, this uses only training data aggregated by (R,C,t_idx), not by breath_id, and is applied identically at inference. Everything else (features, ridge alpha, snapping, submission writing) remains intact and it still produce a valid `submission.csv`.'
- What this solution (achieved 4.87742) has done: 'Your current MAE (1.78594, lower is better) is still far above the target (0.14396), so we should make a small, metric-aligned improvement while keeping your exact ridge-per-(R,C,t_idx) approach. The main missing piece for this competition is enforcing known physical/label constraints: pressure at `u_in==0` during inspiration is almost always the minimum grid value, and predictions benefit from “clamping” those rows accordingly. Additionally, your “rc_t_mean_pressure” feature already encodes a strong baseline; we can safely shrink the ridge penalty slightly (same closed-form ridge, same features, same grouping) to let the model follow that baseline more closely without changing the modeling family. Finally, we keep grid snapping/clipping and submission alignment identical to preserve semantics and stability.'
- What this solution (achieved 1.7831) has done: 'Your current MAE (4.877, lower is better) is still far above the target (0.144), so we should make small, metric-aligned fixes that improve accuracy without changing your ridge-per-(R,C,t_idx) closed-form approach. The biggest issue is likely a schema bug: your `id` range shows only 1–2000, which indicates you are accidentally reading the wrong `train.csv/test.csv` (a truncated copy), which destroy score; we robustly select the correct dataset files from the available paths without changing the modeling logic. Next, we remove the hard clamp `u_in==0 -> min pressure`, which is too aggressive and can severely hurt MAE for many inspiratory rows. Everything else (features, per-group standardization, ridge alpha, snapping to pressure grid, submission writing) remains the same.'
- What this solution (achieved 1.7831) has done: 'We keep your exact ridge-per-(R,C,t_idx) closed-form approach and feature set, but fix one score-relevant mismatch: you currently compute history features (cumsums/lags/diffs) across the entire breath, including after exhalation begins, while you only train on inspiratory rows; this contaminates inspiratory-feature semantics near the phase boundary and hurts generalization. We recompute those history features so they reset once `u_out` turns 1 (i.e., per inspiratory segment within each breath), which better matches the scored phase without changing the model family or training loop. Everything else (grouping, standardization, alpha, grid snapping, submission alignment/writing) stays the same and still produces `submission.csv`. This is a minimal, metric-aligned change that should move MAE down from ~1.78 toward the target.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from pathlib import Path


def _pick_largest_existing(paths):
    candidates = []
    for p in paths:
        pp = Path(p)
        if pp.exists():
            candidates.append(pp)
    if not candidates:
        raise FileNotFoundError(f"No candidate files exist among: {paths}")
    return str(max(candidates, key=lambda x: x.stat().st_size))


TRAIN_PATH = _pick_largest_existing(
    [
        "/kaggle/input/ventilator-pressure-prediction/train.csv",
        "/kaggle/data/ventilator-pressure-prediction/train.csv",
        "/kaggle/input/train.csv",
        "/kaggle/data/train.csv",
        "../input/ventilator-pressure-prediction/train.csv",
    ]
)
TEST_PATH = _pick_largest_existing(
    [
        "/kaggle/input/ventilator-pressure-prediction/test.csv",
        "/kaggle/data/ventilator-pressure-prediction/test.csv",
        "/kaggle/input/test.csv",
        "/kaggle/data/test.csv",
        "../input/ventilator-pressure-prediction/test.csv",
    ]
)
SAMPLE_SUB_PATH = _pick_largest_existing(
    [
        "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv",
        "/kaggle/data/ventilator-pressure-prediction/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
        "../input/ventilator-pressure-prediction/sample_submission.csv",
    ]
)

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sub = pd.read_csv(SAMPLE_SUB_PATH)

assert {"id", "pressure"}.issubset(sub.columns)
assert "pressure" in train.columns
assert "id" in test.columns

assert (
    len(train) > 1_000_000
), f"Train looks too small ({len(train)}). Wrong file: {TRAIN_PATH}"
assert (
    len(test) > 100_000
), f"Test looks too small ({len(test)}). Wrong file: {TEST_PATH}"
assert (
    train["breath_id"].nunique() > 10_000
), "Unexpectedly few breaths; likely wrong dataset file."
assert (
    test["breath_id"].nunique() > 1_000
), "Unexpectedly few breaths; likely wrong dataset file."




## === cell 1
def add_breath_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    g_breath = df.groupby("breath_id", sort=False)
    df["t_idx"] = g_breath.cumcount().astype(np.int16)

    df["u_out_cum"] = g_breath["u_out"].cumsum()
    df["exhale_started"] = (df["u_out_cum"] > 0).astype(np.int8)

    df["insp_seg_id"] = (
        df["u_out"]
        .astype(np.int8)
        .groupby(df["breath_id"], sort=False)
        .cumsum()
        .astype(np.int16)
    )

    g_seg = df.groupby(["breath_id", "insp_seg_id"], sort=False)

    df["u_in_cum"] = g_seg["u_in"].cumsum()
    df["u_in_lag1"] = g_seg["u_in"].shift(1).fillna(0.0)
    df["u_in_diff"] = g_seg["u_in"].diff().fillna(0.0)

    df["u_in_lag2"] = g_seg["u_in"].shift(2).fillna(0.0)
    df["u_in_lag3"] = g_seg["u_in"].shift(3).fillna(0.0)

    eps = 1e-6
    df["u_in_cum_div_R"] = df["u_in_cum"] / (df["R"].astype(np.float64) + eps)
    df["u_in_cum_div_C"] = df["u_in_cum"] / (df["C"].astype(np.float64) + eps)

    dt = g_seg["time_step"].diff().fillna(0.0).astype(np.float64)
    df["delta_time"] = dt
    df["u_in_dt"] = (df["u_in"].astype(np.float64) * df["delta_time"]).astype(
        np.float64
    )
    df["u_in_dt_cum"] = g_seg["u_in_dt"].cumsum().astype(np.float64)

    df = df.drop(columns=["insp_seg_id"])

    return df


train_fe = add_breath_features(train)
test_fe = add_breath_features(test)

TARGET = "pressure"

train_insp_tmp = train_fe[train_fe["u_out"] == 0].copy()
global_insp_mean = float(train_insp_tmp[TARGET].mean())

rc_t_mean = (
    train_insp_tmp.groupby(["R", "C", "t_idx"], sort=False)[TARGET]
    .mean()
    .reset_index()
    .rename(columns={TARGET: "rc_t_mean_pressure"})
)

train_fe = train_fe.merge(rc_t_mean, on=["R", "C", "t_idx"], how="left")
test_fe = test_fe.merge(rc_t_mean, on=["R", "C", "t_idx"], how="left")

train_fe["rc_t_mean_pressure"] = train_fe["rc_t_mean_pressure"].fillna(global_insp_mean)
test_fe["rc_t_mean_pressure"] = test_fe["rc_t_mean_pressure"].fillna(global_insp_mean)

FEATURES = [
    "u_in",
    "time_step",
    "u_in_cum",
    "u_in_lag1",
    "u_in_lag2",
    "u_in_lag3",
    "u_in_diff",
    "exhale_started",
    "u_in_cum_div_R",
    "u_in_cum_div_C",
    "u_in_dt",
    "u_in_dt_cum",
    "rc_t_mean_pressure",
]

train_insp = train_fe[train_fe["u_out"] == 0].copy()


def fit_ridge_closed_form(X, y, alpha=1.0):
    n = X.shape[0]
    if n == 0:
        return None
    Xb = np.c_[np.ones((n, 1), dtype=np.float64), X]
    p = Xb.shape[1]
    A = Xb.T @ Xb
    A[np.diag_indices(p)] += alpha
    b = Xb.T @ y
    return np.linalg.solve(A, b)


def standardize_fit(X: np.ndarray):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    Xs = (X - mu) / sigma
    return Xs, mu, sigma


def standardize_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return (X - mu) / sigma


pressure_grid = np.sort(train[TARGET].unique()).astype(np.float64)


def snap_to_grid(y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if grid.size == 0:
        return y
    idxg = np.searchsorted(grid, y, side="left")
    idxg = np.clip(idxg, 0, grid.size - 1)
    idx0 = np.clip(idxg - 1, 0, grid.size - 1)
    g1 = grid[idxg]
    g0 = grid[idx0]
    choose1 = np.abs(y - g1) <= np.abs(y - g0)
    return np.where(choose1, g1, g0)


RIDGE_ALPHA = 0.3

Xg_raw = train_insp[FEATURES].to_numpy(dtype=np.float64)
yg_cont = train_insp[TARGET].to_numpy(dtype=np.float64)
yg = snap_to_grid(yg_cont, pressure_grid)

Xg, mu_g, sig_g = standardize_fit(Xg_raw)
beta_global = fit_ridge_closed_form(Xg, yg, alpha=RIDGE_ALPHA)

betas = {}
scalers = {}  # (R,C,t_idx) -> (mu, sigma)
for (r, c, t), grp in train_insp.groupby(["R", "C", "t_idx"], sort=False):
    X_raw = grp[FEATURES].to_numpy(dtype=np.float64)
    y_cont = grp[TARGET].to_numpy(dtype=np.float64)
    y = snap_to_grid(y_cont, pressure_grid)
    Xs, mu, sig = standardize_fit(X_raw)
    beta = fit_ridge_closed_form(Xs, y, alpha=RIDGE_ALPHA)
    if beta is not None:
        key = (int(r), int(c), int(t))
        betas[key] = beta
        scalers[key] = (mu, sig)



## === cell 2
X_test_raw = test_fe[FEATURES].to_numpy(dtype=np.float64)
pred = np.empty(len(test_fe), dtype=np.float64)

rc_t = np.c_[
    test_fe["R"].astype(np.int16).to_numpy(),
    test_fe["C"].astype(np.int16).to_numpy(),
    test_fe["t_idx"].astype(np.int16).to_numpy(),
].astype(np.int32, copy=False)

unique_groups, inv = np.unique(rc_t, axis=0, return_inverse=True)

train_mean_insp = float(train_insp[TARGET].mean())

for gi, (r, c, t) in enumerate(unique_groups):
    idx = np.where(inv == gi)[0]
    key = (int(r), int(c), int(t))

    beta = betas.get(key, beta_global)
    if beta is None:
        pred[idx] = train_mean_insp
        continue

    X_i_raw = X_test_raw[idx]
    if key in scalers:
        mu, sig = scalers[key]
        X_i = standardize_apply(X_i_raw, mu, sig)
    else:
        X_i = standardize_apply(X_i_raw, mu_g, sig_g)

    Xb_i = np.c_[np.ones((len(idx), 1), dtype=np.float64), X_i]
    pred[idx] = Xb_i @ beta

pmin = float(train[TARGET].min())
pmax = float(train[TARGET].max())
pred = np.clip(pred, pmin, pmax)

insp_mask = test_fe["u_out"].to_numpy(dtype=np.int8) == 0
if pressure_grid.size > 0:
    vals = pred[insp_mask]
    idxg = np.searchsorted(pressure_grid, vals, side="left")
    idxg = np.clip(idxg, 0, pressure_grid.size - 1)
    idx0 = np.clip(idxg - 1, 0, pressure_grid.size - 1)
    g1 = pressure_grid[idxg]
    g0 = pressure_grid[idx0]
    choose1 = np.abs(vals - g1) <= np.abs(vals - g0)
    pred[insp_mask] = np.where(choose1, g1, g0)



## === cell 3
submission = pd.DataFrame(
    {"id": test["id"].astype(np.int64), "pressure": pred.astype(np.float64)}
)

submission = submission.sort_values("id").reset_index(drop=True)
sub_sorted = sub.sort_values("id").reset_index(drop=True)
if len(submission) == len(sub_sorted) and submission["id"].equals(sub_sorted["id"]):
    submission = submission
else:
    submission = sub_sorted[["id"]].merge(submission, on="id", how="left")
    submission["pressure"] = submission["pressure"].fillna(float(train_mean_insp))

submission.to_csv("submission.csv", index=False)

submission.head()
