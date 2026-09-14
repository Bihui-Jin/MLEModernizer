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

0.2370308705512619

# 6. Current score

1.54367

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.63464) has done: 'The crash comes from trying to read out-of-environment blend files (`../input/gb-blending/...`) that don’t exist; to make this run end-to-end we remove that dependency and instead generate a single valid submission from the provided competition data. To keep changes minimal and score-reasonable without introducing new modeling complexity, we create a deterministic baseline that predicts the mean training pressure during the inspiratory phase (`u_out==0`), which matches the evaluation semantics and guarantees a correctly formatted `submission.csv`. We also fix the cell numbering (starting at 1) and keep the existing blending helpers intact but unused so the notebook runs cleanly. The result reliably produce `submission.csv` with columns `id,pressure`.'
- What this solution (achieved 7.63337) has done: 'We keep your core “single constant prediction” logic, but make it much closer to the competition metric by predicting a physically-valid pressure value from the discrete pressure grid used in the data (so the MAE drops substantially without changing the approach). Concretely, we compute the inspiratory-phase (`u_out==0`) mean pressure as you already do, then snap that mean to the nearest observed pressure level from the training set (a tiny post-processing step that usually improves MAE on this competition). We also avoid an unnecessary merge by directly creating the submission from `test.csv` ids to guarantee perfect row alignment and ordering. The blending helper cells are preserved unchanged (but still unused), and the script still writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 7.53007) has done: 'Your current score (7.63 MAE) is far from the target (0.237), so the constant-pressure baseline needs a small but meaningful metric-aligned upgrade without changing the overall “no ML model, simple deterministic rule” core. We keep the same approach of deriving predictions purely from the training distribution, but instead of one global constant we use a per-(R,C) constant (still just means snapped to the known pressure grid), which better matches the known dependency on lung attributes. We also ensure perfect `id` alignment by building the submission directly from `test.csv` in its original order (no sort), which avoids any accidental mis-ordering risk. All other blending/helper functions remain intact and unused, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 6.59572) has done: 'Your current MAE (7.53) is far above the target (0.237) for a lower-is-better metric, so we need a small but meaningful metric-aligned upgrade while keeping your “no ML, deterministic rule from train distribution” core. The main issue is that a per-(R,C) constant still ignores the strongest within-breath driver `u_in`, which is present in test and highly correlated with pressure during inspiration; adding a simple per-(R,C,u_in_bin) lookup keeps the same core logic (group mean + snap-to-grid) but should substantially reduce error. To keep it stable and avoid overfitting/noise, we discretize `u_in` into a modest number of bins and use a clean fallback chain: (R,C,u_in_bin) -> (R,C) -> global inspiratory mean, all snapped to the known pressure grid. We also keep test row order intact and write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 4.11424) has done: 'Your current MAE (6.59572, lower-is-better) is far above the target (0.237), so we need a modest but meaningful improvement without changing the overall “lookup from train distribution + snap-to-grid” core. The biggest missing signal in your current grouping is time: pressure changes systematically over the inspiratory trajectory even at similar `u_in`, so I add `time_step_bin` into the same mean-lookup table (still just group means + fallback + snap-to-grid). To keep it stable and minimal, we use a small number of time bins and preserve your existing fallback chain, plus we replace slow merges with dictionary lookups keyed by `(R,C,u_in_bin,time_bin)` to stay under the 600s constraint. Output remains a valid `submission.csv` with `id,pressure` in the original test order.'
- What this solution (achieved 4.11424) has done: 'We keep your deterministic “group-mean lookup + snap-to-pressure-grid + fallback chain” approach, but make one minimal, metric-aligned improvement: include `u_out` in the test-time features and force predictions to 0.0 when `u_out==1`, since expiratory-phase rows are not scored and this reduces unnecessary error without changing the inspiratory modeling logic. We also read `u_out` in test and keep row order untouched to avoid alignment issues. Everything else (binning, maps, snapping, and fallbacks) remains the same, so runtime stays within limits and the script still writes a valid `submission.csv`.'
- What this solution (achieved 4.30913) has done: 'We need to move MAE down from 4.114 toward 0.237 (lower is better), so we keep your same deterministic “group-mean lookup + snap-to-pressure-grid + fallback chain” logic but add one missing high-signal feature: the within-breath step index. Time stamps are almost aligned across breaths, but using `breath_id` + per-breath row number (`step`) captures the trajectory more robustly than global time binning and typically improves this competition a lot without changing the approach. Concretely, we replace `t_bin` with `step` (0–79) in the grouping keys and fallbacks while keeping `u_in_bin`, snapping, and the `u_out==1 -> 0.0` rule unchanged. This is a minimal, metric-aligned change that should substantially reduce error while staying fast and producing the same submission format.'
- What this solution (achieved 4.25889) has done: 'Your current MAE (4.309, lower-is-better) is still far above the target (0.237), so we should modestly improve the same deterministic “group-mean lookup + snap-to-pressure-grid + fallback chain” logic without changing the overall approach. The smallest high-signal change is to use the true `time_step` (binned) in addition to within-breath `step`, because `time_step` varies slightly across breaths and captures trajectory alignment better than `step` alone in some regions. To keep runtime under control, we only add a small number of `time_step` bins and keep the same fallback chain (R,C,u_in_bin,step,t_bin) → (R,C,step,t_bin) → (R,C,step) → (R,C) → global, plus keep the `u_out==1 -> 0.0` rule and snapping unchanged. This stays fully deterministic, preserves evaluation semantics, and still writes a valid `submission.csv` in the correct order.'
- What this solution (achieved 4.25889) has done: 'Your current MAE (4.25889, lower-is-better) is still far above the target (0.237), so we should improve predictions while keeping the same deterministic “group-mean lookup + snap-to-pressure-grid + fallback chain” core. The main issue is likely that forcing `u_out==1` predictions to `0.0` can introduce large absolute errors on many expiratory rows (even if they’re not scored, this can still hurt if the platform’s scoring includes them or if there’s any mismatch), so we switch that to a safe metric-aligned choice: predict the nearest-grid global inspiratory mean for expiratory rows as well. To keep changes minimal and stable, we otherwise preserve your exact binning/grouping logic and only adjust this post-processing rule plus add a small speed-safe micro-optimization (iterate over numpy arrays directly) without changing semantics. The script still runs end-to-end and writes a valid `submission.csv` with `id,pressure` in the original test order.'
- What this solution (achieved 2.14543) has done: 'Your MAE (4.25889, lower-is-better) is still far above the target (0.237), so we should improve accuracy while keeping your same deterministic “group-mean lookup + snap-to-pressure-grid + fallback chain” core. The biggest missing signal is cumulative delivered volume, which is strongly tied to pressure; we add `u_in_cum` (cumulative sum of `u_in` within each breath) and bin it, then use it in the highest-resolution lookup while keeping all existing maps and fallbacks intact. This is still the same modeling approach (bin features → group mean on train inspiratory rows → snap to pressure grid → dictionary lookup with fallbacks) and stays fast. We keep test row order unchanged and continue writing a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 1.87444) has done: 'We keep your deterministic “binned feature → group mean on inspiratory rows → snap to known pressure grid → fallback chain” core intact, but tighten it so the lookup better matches the evaluation (inspiratory-only) and uses two strong, still-legit signals already present in the data. Concretely, we (1) compute `u_in_cum` only during inspiration (reset accumulation after `u_out==1`) so the cumulative-volume proxy aligns with the scored phase, and (2) add `u_in_diff` (within-breath first difference of `u_in`) as an additional binned key at the highest-resolution map, with a fallback to your existing maps if unseen. These are minimal feature-engineering extensions that preserve your overall approach (no ML model, just group means + snapping) while typically reducing MAE substantially toward your target. The script still runs end-to-end under the same paths and writes a valid `submission.csv` with `id,pressure` in original test order.'
- What this solution (achieved 1.54367) has done: 'Your current MAE (1.87444, lower-is-better) is still far above the target (0.23703), so we should improve accuracy with the smallest changes that keep your deterministic “binned features → group mean on inspiratory rows → snap-to-pressure-grid → fallback chain” core intact. The main low-risk gain is to make the cumulative-volume proxy closer to the real physics by using `area = ∫ u_in dt` instead of a plain cumulative sum; we compute `u_in_area` (inspiration-only, with reset after `u_out==1`) and use a binned version only in the highest-resolution lookup, keeping all existing maps and fallbacks unchanged. This preserves your training approach (no ML, same grouping/means/snapping) but typically reduces MAE substantially on this competition because `dt` is not perfectly constant. We also apply the exact same `u_in_area` feature to test and keep submission ordering and schema identical.'
- What this solution (achieved 1.54367) has done: 'Your current MAE (1.54367, lower-is-better) is still far above the target (0.23703), so we should improve accuracy with the smallest possible changes while keeping your exact “binned features → train inspiratory group means → snap-to-pressure-grid → fallback chain” core intact. The main issue is that your “inspiration-only reset” for cumulative features is not actually resetting at `u_out==1` (it only zeros the increment but continues the same cumsum), which misaligns `u_in_cum`/`u_in_area` with the scored inspiratory phase and makes the high-resolution lookup noisier. I change only the cumulative-feature construction to *true segment-wise cumsums* that restart after each `u_out==1` within a breath (in both train and test), leaving all binning, grouping keys, snapping, and fallback maps unchanged. This is a metric-aligned fix that typically yields a clear MAE drop without changing the modeling approach or adding complexity, and it still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import copy
import glob
import random
from random import random as rd




## === cell 1
def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    l = []
    for i in range(len(input_list)):
        public_lb_score = int(input_list[i].split("/")[-1].split(".")[1].split(" ")[0])
        l.append(public_lb_score)
        input_list[i] = (pd.read_csv(input_list[i]).pressure).ravel()
    output = 0
    l_sum = sum(l)
    if len(input_list) == 1:
        output = input_list[0]
    else:
        weight1 = 0.8
        weight2 = 0.2
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def g(dp):
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)
    file_count = len(l)
    loop_time = file_count**3
    splits = file_count // 2
    l.sort()
    flist = []
    for i in range(splits):
        if i == splits - 1:
            flist.append(l[i * round(len(l) / splits) :])
        else:
            flist.append(
                l[i * round(len(l) / splits) : (i + 1) * round(len(l) / splits)]
            )
    for i in range(len(flist)):
        flist[i] = wc(flist[i])
    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output.pressure = 0
    for i in range(loop_time):
        weight = []
        set_seed(i)
        for i in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        for i in range(len(weight)):
            weight[i] /= weight_sum
        weight.sort(reverse=True)
        for i in range(len(flist)):
            output.pressure += flist[i] * weight[i]
    output.pressure /= loop_time
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)




## === cell 2
def blend(a, b, c):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    c = pd.read_csv(c)
    a.pressure = a.pressure * 0.6 + b.pressure * 0.25 + c.pressure * 0.15
    a.to_csv("blend.csv", index=False)
    return a




## === cell 3
DATA_DIR = "/kaggle/input/ventilator-pressure-prediction"
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")

train = pd.read_csv(
    train_path,
    usecols=["breath_id", "R", "C", "u_out", "u_in", "time_step", "pressure"],
)
train["pressure"] = train["pressure"].astype(np.float64)

train["step"] = train.groupby("breath_id", sort=False).cumcount().astype(np.int16)

dt_train = (
    train.groupby("breath_id", sort=False)["time_step"]
    .diff()
    .fillna(0.0)
    .astype(np.float32)
)

seg_train = train.groupby("breath_id", sort=False)["u_out"].cumsum().astype(np.int32)
u_in_insp_only = train["u_in"].where(train["u_out"].eq(0), 0.0).astype(np.float32)
dt_insp = dt_train.where(train["u_out"].eq(0), 0.0).astype(np.float32)

train["u_in_area"] = (
    (u_in_insp_only * dt_insp)
    .groupby([train["breath_id"], seg_train], sort=False)
    .cumsum()
    .astype(np.float32)
)

train["u_in_cum"] = (
    u_in_insp_only.groupby([train["breath_id"], seg_train], sort=False)
    .cumsum()
    .astype(np.float32)
)

train["u_in_diff"] = (
    train.groupby("breath_id", sort=False)["u_in"].diff().fillna(0.0).astype(np.float32)
)

insp = train.loc[
    train["u_out"] == 0,
    [
        "R",
        "C",
        "u_in",
        "u_in_diff",
        "u_in_cum",
        "u_in_area",
        "step",
        "time_step",
        "pressure",
    ],
].copy()

pressure_levels = np.sort(train["pressure"].unique())


def snap_to_grid(x, levels):
    x = float(x)
    return float(levels[np.argmin(np.abs(levels - x))])


N_BINS = 50
bin_edges = np.linspace(0.0, 100.0, N_BINS + 1)


def uin_to_bin(u):
    u = float(u)
    idx = int(np.searchsorted(bin_edges, u, side="right") - 1)
    if idx < 0:
        return 0
    if idx >= N_BINS:
        return N_BINS - 1
    return idx


insp["u_in_bin"] = insp["u_in"].map(uin_to_bin).astype(np.int16)

T_BINS = 20
t_edges = np.linspace(0.0, 3.0, T_BINS + 1)


def t_to_bin(t):
    t = float(t)
    idx = int(np.searchsorted(t_edges, t, side="right") - 1)
    if idx < 0:
        return 0
    if idx >= T_BINS:
        return T_BINS - 1
    return idx


insp["t_bin"] = insp["time_step"].map(t_to_bin).astype(np.int16)

UC_BINS = 80
uc_edges = np.linspace(0.0, 300.0, UC_BINS + 1)


def uinc_to_bin(x):
    x = float(x)
    idx = int(np.searchsorted(uc_edges, x, side="right") - 1)
    if idx < 0:
        return 0
    if idx >= UC_BINS:
        return UC_BINS - 1
    return idx


insp["u_in_cum_bin"] = insp["u_in_cum"].map(uinc_to_bin).astype(np.int16)

UD_BINS = 41
ud_edges = np.linspace(-100.0, 100.0, UD_BINS + 1)


def udiff_to_bin(x):
    x = float(x)
    idx = int(np.searchsorted(ud_edges, x, side="right") - 1)
    if idx < 0:
        return 0
    if idx >= UD_BINS:
        return UD_BINS - 1
    return idx


insp["u_in_diff_bin"] = insp["u_in_diff"].map(udiff_to_bin).astype(np.int16)

UA_BINS = 120
ua_edges = np.linspace(0.0, 300.0, UA_BINS + 1)


def uarea_to_bin(x):
    x = float(x)
    idx = int(np.searchsorted(ua_edges, x, side="right") - 1)
    if idx < 0:
        return 0
    if idx >= UA_BINS:
        return UA_BINS - 1
    return idx


insp["u_in_area_bin"] = insp["u_in_area"].map(uarea_to_bin).astype(np.int16)

group_mean_rcucudst = insp.groupby(
    [
        "R",
        "C",
        "u_in_area_bin",
        "u_in_cum_bin",
        "u_in_diff_bin",
        "u_in_bin",
        "step",
        "t_bin",
    ],
    sort=False,
)["pressure"].mean()
map_rcucudst = {
    k: snap_to_grid(v, pressure_levels) for k, v in group_mean_rcucudst.items()
}

group_mean_rcucust = insp.groupby(
    ["R", "C", "u_in_cum_bin", "u_in_bin", "step", "t_bin"], sort=False
)["pressure"].mean()
map_rcucust = {
    k: snap_to_grid(v, pressure_levels) for k, v in group_mean_rcucust.items()
}

group_mean_rcust = insp.groupby(["R", "C", "u_in_bin", "step", "t_bin"], sort=False)[
    "pressure"
].mean()
map_rcust = {k: snap_to_grid(v, pressure_levels) for k, v in group_mean_rcust.items()}

group_mean_rcst = insp.groupby(["R", "C", "step", "t_bin"], sort=False)[
    "pressure"
].mean()
map_rcst = {k: snap_to_grid(v, pressure_levels) for k, v in group_mean_rcst.items()}

group_mean_rcs = insp.groupby(["R", "C", "step"], sort=False)["pressure"].mean()
map_rcs = {k: snap_to_grid(v, pressure_levels) for k, v in group_mean_rcs.items()}

group_mean_rc = insp.groupby(["R", "C"], sort=False)["pressure"].mean()
map_rc = {k: snap_to_grid(v, pressure_levels) for k, v in group_mean_rc.items()}

global_insp_mean = float(insp["pressure"].mean())
global_snapped = snap_to_grid(global_insp_mean, pressure_levels)

test = pd.read_csv(
    test_path, usecols=["id", "breath_id", "R", "C", "u_in", "u_out", "time_step"]
)
test["step"] = test.groupby("breath_id", sort=False).cumcount().astype(np.int16)

dt_test = (
    test.groupby("breath_id", sort=False)["time_step"]
    .diff()
    .fillna(0.0)
    .astype(np.float32)
)

seg_test = test.groupby("breath_id", sort=False)["u_out"].cumsum().astype(np.int32)
u_in_insp_only_t = test["u_in"].where(test["u_out"].eq(0), 0.0).astype(np.float32)
dt_insp_t = dt_test.where(test["u_out"].eq(0), 0.0).astype(np.float32)

test["u_in_area"] = (
    (u_in_insp_only_t * dt_insp_t)
    .groupby([test["breath_id"], seg_test], sort=False)
    .cumsum()
    .astype(np.float32)
)

test["u_in_cum"] = (
    u_in_insp_only_t.groupby([test["breath_id"], seg_test], sort=False)
    .cumsum()
    .astype(np.float32)
)

test["u_in_diff"] = (
    test.groupby("breath_id", sort=False)["u_in"].diff().fillna(0.0).astype(np.float32)
)

test["u_in_bin"] = test["u_in"].map(uin_to_bin).astype(np.int16)
test["t_bin"] = test["time_step"].map(t_to_bin).astype(np.int16)
test["u_in_cum_bin"] = test["u_in_cum"].map(uinc_to_bin).astype(np.int16)
test["u_in_diff_bin"] = test["u_in_diff"].map(udiff_to_bin).astype(np.int16)
test["u_in_area_bin"] = test["u_in_area"].map(uarea_to_bin).astype(np.int16)

R_arr = test["R"].to_numpy(np.int16, copy=False)
C_arr = test["C"].to_numpy(np.int16, copy=False)
uareab_arr = test["u_in_area_bin"].to_numpy(np.int16, copy=False)
uincb_arr = test["u_in_cum_bin"].to_numpy(np.int16, copy=False)
udb_arr = test["u_in_diff_bin"].to_numpy(np.int16, copy=False)
uinb_arr = test["u_in_bin"].to_numpy(np.int16, copy=False)
step_arr = test["step"].to_numpy(np.int16, copy=False)
tbin_arr = test["t_bin"].to_numpy(np.int16, copy=False)
uout_arr = test["u_out"].to_numpy(np.int8, copy=False)

pred = np.empty(len(test), dtype=np.float64)

for i in range(len(pred)):
    if int(uout_arr[i]) == 1:
        pred[i] = global_snapped
        continue

    key_rcucudst = (
        int(R_arr[i]),
        int(C_arr[i]),
        int(uareab_arr[i]),
        int(uincb_arr[i]),
        int(udb_arr[i]),
        int(uinb_arr[i]),
        int(step_arr[i]),
        int(tbin_arr[i]),
    )
    v = map_rcucudst.get(key_rcucudst)
    if v is None:
        key_rcucust = (
            int(R_arr[i]),
            int(C_arr[i]),
            int(uincb_arr[i]),
            int(uinb_arr[i]),
            int(step_arr[i]),
            int(tbin_arr[i]),
        )
        v = map_rcucust.get(key_rcucust)
        if v is None:
            key_rcust = (
                int(R_arr[i]),
                int(C_arr[i]),
                int(uinb_arr[i]),
                int(step_arr[i]),
                int(tbin_arr[i]),
            )
            v = map_rcust.get(key_rcust)
            if v is None:
                key_rcst = (
                    int(R_arr[i]),
                    int(C_arr[i]),
                    int(step_arr[i]),
                    int(tbin_arr[i]),
                )
                v = map_rcst.get(key_rcst)
                if v is None:
                    key_rcs = (int(R_arr[i]), int(C_arr[i]), int(step_arr[i]))
                    v = map_rcs.get(key_rcs)
                    if v is None:
                        key_rc = (int(R_arr[i]), int(C_arr[i]))
                        v = map_rc.get(key_rc, global_snapped)

    pred[i] = v

sub = pd.DataFrame({"id": test["id"].to_numpy(copy=False), "pressure": pred})
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print(f"global_insp_mean={global_insp_mean:.6f} -> global_snapped={global_snapped:.6f}")
print(
    f"N_BINS={N_BINS}, T_BINS={T_BINS}, UC_BINS={UC_BINS}, UD_BINS={UD_BINS}, UA_BINS={UA_BINS}, pressure_levels={len(pressure_levels)}"
)
print(sub.head())
