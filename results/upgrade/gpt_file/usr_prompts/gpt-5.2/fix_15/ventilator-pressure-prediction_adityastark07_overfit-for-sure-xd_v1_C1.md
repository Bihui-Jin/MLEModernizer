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

0.1441764734364301

# 6. Current score

4.42779

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.16447) has done: 'The notebook fails because it tries to read four external “public notebook” submission files that are not present in your environment, so execution stops before writing any `submission.csv`. To keep the same ensembling core idea (a weighted blend) but make it self-contained, I replace those missing inputs with four lightweight baseline predictors trained from `train.csv` only, then ensemble their predictions with the same weights. This fixes the runtime errors, produces a valid `submission.csv` with the required `id,pressure` columns, and should achieve a reasonable MAE (likely better than the all-zeros sample) without changing the overall approach (simple ensemble of multiple predictors). Paths are adjusted to the provided dataset location and everything runs end-to-end.'
- What this solution (achieved 5.07472) has done: 'I fix the immediate runtime error by ensuring the binned feature columns (`uin_bin_100`, `uin_bin_200`, `tbin_40`, `tbin_80`) are created on the full `train` dataframe (not only on `train_insp`), since predictors 2–4 group over `train`. Then I keep the same four-predictor median-lookup ensemble logic and weights, but make the fallback lookups robust and vectorized-friendly so it completes reliably within the time limit. Finally, I ensure the script always writes `submission.csv` with exactly `id,pressure` and matching row order/length to the provided sample submission.'
- What this solution (achieved 5.07472) has done: 'Your current score (5.07472, lower-is-better) is far worse than the target (0.14418), so we should improve accuracy while keeping your “median-lookup ensemble” core logic unchanged. The biggest issue is that your predictors 2–4 are built on all phases of `train`, but the competition metric ignores expiratory (`u_out==1`) timesteps, so those medians inject label noise and hurt MAE. I keep the same 4 predictors + same weights, but compute all groupby medians using only inspiratory rows (`u_out==0`) and explicitly set predictions to 0 for test rows where `u_out==1` (they are unscored). This is a minimal, metric-aligned change that should move the score much closer to the target band while still producing a valid `submission.csv`.'
- What this solution (achieved 5.07544) has done: 'Your current MAE (5.07, lower-is-better) is still far from the target (0.144), so we should improve accuracy while keeping your same “median-lookup ensemble” logic. The biggest remaining issue is that the fallbacks (global_median and chained pred1→pred2→pred3→pred4) are not conditioned on lung attributes/time, so many unseen keys collapse to a crude median and inflate MAE. I add one more *hierarchical* fallback layer per predictor using slightly coarser groupings (still medians, still lookup-based, no model/loop changes), so missing fine-grain keys fall back to a more appropriate median instead of the global median. I also (safely) remove `u_out` from the groupby keys since you’re training only on inspiratory rows (where `u_out==0`), which reduces sparsity and should materially reduce unseen-key fallbacks.'
- What this solution (achieved 4.93059) has done: 'Your current MAE (5.075, lower-is-better) is far from the target (0.144), so we should materially improve accuracy while keeping your same “median-lookup ensemble” approach. The biggest remaining issue is that continuous `u_in` and `time_step` values are binned, which creates many unseen bin combinations in test and forces frequent fallback to coarse medians—this heavily inflates error. I keep the exact same 4 predictors + weights + median-lookup logic, but switch from equal-width binning to **quantile (equal-frequency) binning based on train inspiratory rows**, which greatly reduces sparsity and improves match rate without changing the model class. I also align the bin ranges to inspiratory data (the only scored phase), while keeping the `u_out==1` test predictions at 0 as you already do.'
- What this solution (achieved 4.93063) has done: 'Your MAE is still far above the target, so we should improve accuracy while keeping your same “median-lookup ensemble with binned u_in/time” core logic intact. The largest remaining avoidable error is that predictions are continuous floats, but the true pressures come from a small discrete grid; snapping predictions to the nearest valid pressure level typically reduces MAE substantially in this competition. I add a minimal post-processing step that maps each predicted pressure to the nearest pressure value observed in inspiratory training data, while keeping your existing u_out==1 masking to 0. The rest of your pipeline (bins, groupby medians, hierarchical fallbacks, weights, output format) stays unchanged.'
- What this solution (achieved 4.58013) has done: 'Your current MAE (4.93063, lower-is-better) is still far above the target (0.14418), so we should improve accuracy while keeping your same “median lookup ensemble with binned u_in/time + hierarchical fallbacks + pressure snapping” core logic. The biggest missing piece is that your predictors ignore the strong sequential structure within each breath (pressure depends heavily on recent control history), which causes many ambiguous states to collapse to coarse medians. With minimal change, we can add a single extra categorical key capturing short history (previous-step binned u_in and previous u_out) into predictors 3 and 4, while keeping everything else (bins, medians, fallbacks, weights, snapping, u_out masking) intact. This should reduce collisions and improve match rates without changing the overall modeling approach. The submission writing and format remain unchanged.'
- What this solution (achieved 4.50802) has done: 'Your current score (4.58013 MAE, lower-is-better) is still far from the target (0.14418), so we should improve accuracy while keeping your median-lookup ensemble core logic intact. The biggest remaining avoidable error is that each timestep is predicted independently, but pressure within a breath is very smooth; adding a tiny “within-breath smoothing” post-process on the inspiratory phase usually reduces MAE without changing the modeling approach. I keep your four predictors, weights, hierarchical fallbacks, u_out masking, and pressure snapping, and only add (1) a per-(breath_id, u_out==0) rolling-median smoothing and (2) re-snap to valid pressure levels after smoothing to preserve the discrete-grid advantage. This is a minimal, metric-aligned change that should move the MAE down toward the target band while still producing a valid `submission.csv`.'
- What this solution (achieved 4.50268) has done: 'We’re still far above the target MAE, so we should improve accuracy while keeping your same median-lookup ensemble + snapping + u_out masking intact. The most leveraged minimal change is to make the within-breath smoothing slightly more informative by blending a rolling median (robust) with a rolling mean (less quantized), and to smooth each breath using only inspiratory timesteps as you already do. This preserves your existing predictors and weights, but reduces timestep noise while better following smooth pressure trajectories, which should move MAE downward. We then re-snap to valid pressure levels exactly as before to retain the discrete-grid advantage and keep submission format unchanged.'
- What this solution (achieved 4.50246) has done: 'Your current MAE (4.50268, lower-is-better) is still far above the target (0.14418), so we should improve accuracy while keeping your exact “median-lookup ensemble + snapping + inspiratory-only smoothing” core logic unchanged. The biggest likely remaining issue is that predictions for expiratory timesteps (`u_out==1`) should not influence smoothing at all, but your smoothing currently reassigns inspiratory predictions using an index-aligned Series that can become misaligned if any ordering changes; we make the smoothing assignment explicitly aligned to the original row positions. Additionally, we add a tiny, metric-consistent post-process to enforce monotonic-ish smoothness within inspiratory phase by applying a second-pass light rolling mean (still within-breath, inspiratory only) before re-snapping—this is minimal and typically reduces timestep noise without changing the model. All file paths and the required `submission.csv` output format remain unchanged.'
- What this solution (achieved 4.50246) has done: 'Your current MAE (4.50, lower-is-better) is still far above the target (0.144), so we should improve accuracy while keeping your median-lookup ensemble + snapping + inspiratory-only smoothing core logic intact. The smallest high-impact fix is to make the smoothing step *metric-aware*: the MAE is only computed for inspiratory timesteps, so we should not smooth across the inspiratory/expiratory boundary within each breath (otherwise early expiratory predictions can “pull” the last inspiratory points). I restrict smoothing to the **continuous inspiratory prefix** of each breath (up to the first `u_out==1`), keep expiratory predictions at 0 as you already do, and keep your ensemble weights, bins, groupby medians, and snapping unchanged. This is a minimal post-process change that typically reduces MAE materially in this competition while preserving your approach and producing the same valid `submission.csv`.'
- What this solution (achieved 4.42779) has done: 'Your current MAE (4.50, lower-is-better) is still far above the target (0.144), so we should improve accuracy while keeping your exact “median-lookup ensemble + snapping + inspiratory-prefix smoothing” approach. The highest-impact minimal fix is that your binning for `time_step` and `u_in` is currently **global across all breaths**, but the signal is strongly **step-index dependent** (the same `time_step`/`u_in` values map to different pressures at different positions in the 80-step sequence). I keep your same four median-lookup predictors and weights, but add one extra discrete key: the within-breath step index (`step`), and include it only in the *most time-sensitive* predictors (p3/p4) plus their fallbacks to reduce collisions while preserving the overall logic. Everything else—hierarchical fallbacks, pressure snapping, inspiratory-prefix-only smoothing, `u_out==1` masking, and `submission.csv` format—remains unchanged.'
- What this solution (achieved 4.42779) has done: 'We keep your exact median-lookup ensemble and post-processing, but fix a major submission-alignment issue that’s likely dominating the MAE: you currently write predictions in the order of `sample_submission.csv`, not in the order of `test.csv` (Kaggle scoring matches by `id`). I rebuild the final `sub` starting from `test[['id']]`, carry predictions in the same row order as `test`, and then (optionally) align by sorting on `id` to be extra safe. This is a minimal change that preserves your modeling logic and should move the score sharply downward toward the target by ensuring each `pressure` is attached to the correct `id`. Everything else (bins, groupby medians, hierarchical fallbacks, snapping, inspiratory-prefix smoothing, and `u_out` masking) remains unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/input/ventilator-pressure-prediction"

train_path = f"{DATA_DIR}/train.csv"
test_path = f"{DATA_DIR}/test.csv"
sample_path = f"{DATA_DIR}/sample_submission.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

assert "id" in test.columns and "id" in sample_sub.columns
assert len(test) == len(sample_sub)

train_insp = train[train["u_out"] == 0].copy()
global_median = float(train_insp["pressure"].median())


def _quantile_edges(x, nbins):
    x = np.asarray(x, dtype=np.float64)
    qs = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.quantile(x, qs)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _bin_by_edges(x, edges):
    x = np.asarray(x, dtype=np.float64)
    return (np.searchsorted(edges, x, side="right") - 1).astype(np.int32)


NBINS_UIN_1 = 200
NBINS_UIN_2 = 100
NBINS_TIME_1 = 80
NBINS_TIME_2 = 40

uin_edges_200 = _quantile_edges(train_insp["u_in"].values, NBINS_UIN_1)
uin_edges_100 = _quantile_edges(train_insp["u_in"].values, NBINS_UIN_2)
t_edges_80 = _quantile_edges(train_insp["time_step"].values, NBINS_TIME_1)
t_edges_40 = _quantile_edges(train_insp["time_step"].values, NBINS_TIME_2)

train = train.copy()
train["uin_bin_200"] = _bin_by_edges(train["u_in"].values, uin_edges_200)
train["uin_bin_100"] = _bin_by_edges(train["u_in"].values, uin_edges_100)
train["tbin_80"] = _bin_by_edges(train["time_step"].values, t_edges_80)
train["tbin_40"] = _bin_by_edges(train["time_step"].values, t_edges_40)

train["step"] = train.groupby("breath_id", sort=False).cumcount().astype(np.int16)

train["uin_bin_200_prev1"] = (
    train.groupby("breath_id", sort=False)["uin_bin_200"]
    .shift(1)
    .fillna(-1)
    .astype(np.int32)
)
train["uin_bin_100_prev1"] = (
    train.groupby("breath_id", sort=False)["uin_bin_100"]
    .shift(1)
    .fillna(-1)
    .astype(np.int32)
)
train["u_out_prev1"] = (
    train.groupby("breath_id", sort=False)["u_out"].shift(1).fillna(0).astype(np.int8)
)

train_insp = train[train["u_out"] == 0].copy()

test_uin_bin_200 = _bin_by_edges(test["u_in"].values, uin_edges_200)
test_uin_bin_100 = _bin_by_edges(test["u_in"].values, uin_edges_100)
test_tbin_80 = _bin_by_edges(test["time_step"].values, t_edges_80)
test_tbin_40 = _bin_by_edges(test["time_step"].values, t_edges_40)

test_uin_bin_200_prev1 = (
    pd.Series(test_uin_bin_200)
    .groupby(test["breath_id"], sort=False)
    .shift(1)
    .fillna(-1)
    .astype(np.int32)
    .values
)
test_uin_bin_100_prev1 = (
    pd.Series(test_uin_bin_100)
    .groupby(test["breath_id"], sort=False)
    .shift(1)
    .fillna(-1)
    .astype(np.int32)
    .values
)
test_u_out_prev1 = (
    test.groupby("breath_id", sort=False)["u_out"]
    .shift(1)
    .fillna(0)
    .astype(np.int8)
    .values
)

test_step = test.groupby("breath_id", sort=False).cumcount().astype(np.int16).values

rc_med = train_insp.groupby(["R", "C"], sort=False)["pressure"].median()
r_med = train_insp.groupby(["R"], sort=False)["pressure"].median()
c_med = train_insp.groupby(["C"], sort=False)["pressure"].median()

p1_med = train_insp.groupby(["R", "C", "uin_bin_200"], sort=False)["pressure"].median()
keys1 = list(zip(test["R"].values, test["C"].values, test_uin_bin_200))
keys_rc = list(zip(test["R"].values, test["C"].values))

pred1 = np.empty(len(keys1), dtype=np.float32)
for i, (k_fine, k_rc) in enumerate(zip(keys1, keys_rc)):
    v = p1_med.get(k_fine, np.nan)
    if v == v:
        pred1[i] = float(v)
        continue
    v = rc_med.get(k_rc, np.nan)
    if v == v:
        pred1[i] = float(v)
        continue
    v = r_med.get((k_rc[0],), np.nan)
    if v == v:
        pred1[i] = float(v)
        continue
    v = c_med.get((k_rc[1],), np.nan)
    pred1[i] = float(v) if (v == v) else float(global_median)

p2_med = train_insp.groupby(["R", "C", "uin_bin_100"], sort=False)["pressure"].median()
keys2 = list(zip(test["R"].values, test["C"].values, test_uin_bin_100))
pred2_arr = np.empty(len(keys2), dtype=np.float32)
for i, (k_fine, k_rc) in enumerate(zip(keys2, keys_rc)):
    v = p2_med.get(k_fine, np.nan)
    if v == v:
        pred2_arr[i] = float(v)
        continue
    base = pred1[i]
    v2 = rc_med.get(k_rc, np.nan)
    if v2 == v2:
        pred2_arr[i] = float(v2)
    else:
        v2 = r_med.get((k_rc[0],), np.nan)
        if v2 == v2:
            pred2_arr[i] = float(v2)
        else:
            v2 = c_med.get((k_rc[1],), np.nan)
            pred2_arr[i] = float(v2) if (v2 == v2) else float(base)

p3_med_hist = train_insp.groupby(
    ["R", "C", "step", "tbin_40", "uin_bin_100", "uin_bin_100_prev1", "u_out_prev1"],
    sort=False,
)["pressure"].median()
p3_med = train_insp.groupby(["R", "C", "step", "tbin_40", "uin_bin_100"], sort=False)[
    "pressure"
].median()
p3_fallback_time = train_insp.groupby(["R", "C", "step", "tbin_40"], sort=False)[
    "pressure"
].median()

keys3_hist = list(
    zip(
        test["R"].values,
        test["C"].values,
        test_step,
        test_tbin_40,
        test_uin_bin_100,
        test_uin_bin_100_prev1,
        test_u_out_prev1,
    )
)
keys3 = list(
    zip(test["R"].values, test["C"].values, test_step, test_tbin_40, test_uin_bin_100)
)
keys3_time = list(zip(test["R"].values, test["C"].values, test_step, test_tbin_40))

pred3 = np.empty(len(keys3), dtype=np.float32)
for i, (k_hist, k_fine, k_time, k_rc) in enumerate(
    zip(keys3_hist, keys3, keys3_time, keys_rc)
):
    v = p3_med_hist.get(k_hist, np.nan)
    if v == v:
        pred3[i] = float(v)
        continue
    v = p3_med.get(k_fine, np.nan)
    if v == v:
        pred3[i] = float(v)
        continue
    v = p3_fallback_time.get(k_time, np.nan)
    if v == v:
        pred3[i] = float(v)
        continue
    base = pred2_arr[i]
    v2 = rc_med.get(k_rc, np.nan)
    if v2 == v2:
        pred3[i] = float(v2)
    else:
        v2 = r_med.get((k_rc[0],), np.nan)
        if v2 == v2:
            pred3[i] = float(v2)
        else:
            v2 = c_med.get((k_rc[1],), np.nan)
            pred3[i] = float(v2) if (v2 == v2) else float(base)

p4_med_hist = train_insp.groupby(
    ["R", "C", "step", "tbin_80", "uin_bin_200", "uin_bin_200_prev1", "u_out_prev1"],
    sort=False,
)["pressure"].median()
p4_med = train_insp.groupby(["R", "C", "step", "tbin_80", "uin_bin_200"], sort=False)[
    "pressure"
].median()
p4_fallback_time = train_insp.groupby(["R", "C", "step", "tbin_80"], sort=False)[
    "pressure"
].median()

keys4_hist = list(
    zip(
        test["R"].values,
        test["C"].values,
        test_step,
        test_tbin_80,
        test_uin_bin_200,
        test_uin_bin_200_prev1,
        test_u_out_prev1,
    )
)
keys4 = list(
    zip(test["R"].values, test["C"].values, test_step, test_tbin_80, test_uin_bin_200)
)
keys4_time = list(zip(test["R"].values, test["C"].values, test_step, test_tbin_80))

pred4 = np.empty(len(keys4), dtype=np.float32)
for i, (k_hist, k_fine, k_time, k_rc) in enumerate(
    zip(keys4_hist, keys4, keys4_time, keys_rc)
):
    v = p4_med_hist.get(k_hist, np.nan)
    if v == v:
        pred4[i] = float(v)
        continue
    v = p4_med.get(k_fine, np.nan)
    if v == v:
        pred4[i] = float(v)
        continue
    v = p4_fallback_time.get(k_time, np.nan)
    if v == v:
        pred4[i] = float(v)
        continue
    base = pred3[i]
    v2 = rc_med.get(k_rc, np.nan)
    if v2 == v2:
        pred4[i] = float(v2)
    else:
        v2 = r_med.get((k_rc[0],), np.nan)
        if v2 == v2:
            pred4[i] = float(v2)
        else:
            v2 = c_med.get((k_rc[1],), np.nan)
            pred4[i] = float(v2) if (v2 == v2) else float(base)

sub = test[["id"]].copy()
sub_1 = sub.copy()
sub_2 = sub.copy()
sub_3 = sub.copy()
sub_4 = sub.copy()

sub_1["pressure"] = pred1
sub_2["pressure"] = pred2_arr
sub_3["pressure"] = pred3
sub_4["pressure"] = pred4



## === cell 1
sub["pressure"] = (
    (sub_1["pressure"].values * 0.28)
    + (sub_2["pressure"].values * 0.28)
    + (sub_3["pressure"].values * 0.28)
    + (sub_4["pressure"].values * 0.16)
).astype(np.float32)

pressure_levels = np.sort(train_insp["pressure"].unique().astype(np.float32))
pred = sub["pressure"].values.astype(np.float32)

idx = np.searchsorted(pressure_levels, pred, side="left")
idx = np.clip(idx, 0, len(pressure_levels) - 1)
idx_prev = np.clip(idx - 1, 0, len(pressure_levels) - 1)

lvl = pressure_levels[idx]
lvl_prev = pressure_levels[idx_prev]
choose_prev = np.abs(pred - lvl_prev) <= np.abs(pred - lvl)
sub["pressure"] = np.where(choose_prev, lvl_prev, lvl).astype(np.float32)

tmp = pd.DataFrame(
    {
        "row": np.arange(len(test), dtype=np.int32),
        "breath_id": test["breath_id"].values,
        "u_out": test["u_out"].values,
        "pred": sub["pressure"].values.astype(np.float32),
    }
)

breath_uout_max = tmp.groupby("breath_id", sort=False)["u_out"].cummax().values
insp_prefix_mask = (tmp["u_out"].values == 0) & (breath_uout_max == 0)

tmp_insp = tmp.loc[insp_prefix_mask, ["row", "breath_id", "pred"]].copy()

g = tmp_insp.groupby("breath_id", sort=False)["pred"]
roll_med = (
    g.rolling(window=5, center=True, min_periods=1)
    .median()
    .reset_index(level=0, drop=True)
    .astype(np.float32)
)
roll_mean = (
    g.rolling(window=5, center=True, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
    .astype(np.float32)
)

pred_smooth_1 = (0.7 * roll_med + 0.3 * roll_mean).astype(np.float32)
tmp_insp["pred_smooth_1"] = pred_smooth_1

g2 = tmp_insp.groupby("breath_id", sort=False)["pred_smooth_1"]
roll_mean2 = (
    g2.rolling(window=3, center=True, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
    .astype(np.float32)
)
tmp_insp["pred_smooth"] = (
    0.85 * tmp_insp["pred_smooth_1"].values + 0.15 * roll_mean2.values
).astype(np.float32)

sub_pred = sub["pressure"].values.astype(np.float32)
sub_pred[tmp_insp["row"].values] = tmp_insp["pred_smooth"].values.astype(np.float32)
sub["pressure"] = sub_pred

pred2 = sub["pressure"].values.astype(np.float32)
idx = np.searchsorted(pressure_levels, pred2, side="left")
idx = np.clip(idx, 0, len(pressure_levels) - 1)
idx_prev = np.clip(idx - 1, 0, len(pressure_levels) - 1)
lvl = pressure_levels[idx]
lvl_prev = pressure_levels[idx_prev]
choose_prev = np.abs(pred2 - lvl_prev) <= np.abs(pred2 - lvl)
sub["pressure"] = np.where(choose_prev, lvl_prev, lvl).astype(np.float32)

sub.loc[test["u_out"].values.astype(bool), "pressure"] = 0.0

sub = sub[["id", "pressure"]]

sub = sub.sort_values("id").reset_index(drop=True)

assert len(sub) == len(sample_sub)
assert list(sub.columns) == ["id", "pressure"]

sub.to_csv("submission.csv", index=False)
print(sub.head(5))
print("Wrote submission.csv with rows:", len(sub))
