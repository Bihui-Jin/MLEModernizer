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

0.1629180208249741

# 6. Current score

8.44115

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.88771) has done: 'The crash happens because the notebook tries to read external blend files from `../input/gb-blending/`, which are not present in your environment. To make the solution run end-to-end and still follow the same “blend submissions then snap to nearest train pressure” core idea, I switch to a self-contained baseline that generates two simple model-free predictions from the available `test.csv` (one using per-(R,C,time_step) median pressure, another using per-(R,C,u_in rounded) median pressure), then blends them and applies your `find_nearest` mapping. This produces a valid `submission.csv` with the required columns and avoids introducing new modeling/training logic. I also fix the cell numbering and add safe fallbacks for unseen keys to prevent NaNs in the submission.'
- What this solution (achieved 7.51678) has done: 'Your current score (MAE 7.88771; lower is better) is far from the target (0.1629), so we need a real-but-still-minimal improvement while keeping your core “lookup medians then blend then snap to nearest train pressure” logic intact. The biggest issue is that your lookups ignore the breath dynamics and `u_out`, which the metric effectively emphasizes (only inspiratory phase is scored), so we add breath-local cumulative features and include `u_out` in the groupby keys without introducing any ML model or training loop. We keep your two-predictor blend structure but replace the coarse keys with: (R,C,u_out,time_step,u_in_cumsum) and (R,C,u_out,u_in_round,u_in_cumsum), plus a stable hierarchical fallback to reduce NaNs. Finally, we keep the same `find_nearest` snapping and ensure a valid `submission.csv` is written.'
- What this solution (achieved 7.40784) has done: 'Your current score (7.51678 MAE; lower is better) is still far from the target (0.1629), so we need a meaningful but still “same-idea” improvement without introducing any ML model/training loop. The main issue is that your median lookups are dominated by the expiratory phase, but the metric only scores inspiratory points (`u_out==0`), so I compute all pressure medians using only `u_out==0` rows (while still allowing predictions for all test rows). To better match breath dynamics with minimal change, I also add one more breath-local cumulative feature (`area = cumsum(u_in * dt)`) and use it in the same hierarchical lookup/fallback structure you already use. Finally, I keep your same two-predictor blend and the same `find_nearest` snapping, and still write a valid `submission.csv`.'
- What this solution (achieved 7.66129) has done: 'Your current gap to the target is very large (7.41 vs 0.163 MAE; lower is better), and the biggest “minimal but meaningful” fix while preserving your lookup/blend/snap core idea is to stop forcing expiratory (`u_out==1`) predictions to look like inspiratory pressures. I keep your exact two-branch median-lookup structure and `find_nearest` snapping, but I (1) build separate median tables for inspiratory and expiratory phases, and (2) for `u_out==1` rows predict a stable low-pressure baseline (phase-specific medians) rather than inspiratory-derived medians. This directly aligns with the metric (only inspiratory is scored) and typically reduces harmful errors on expiratory rows without touching any ML/training logic. I also ensure `time_step` keys are merged stably by rounding to a fixed precision to avoid unnecessary NaNs from float mismatch.'
- What this solution (achieved 7.66792) has done: 'Your current MAE (7.66; lower is better) is still far from the target (0.163), so we need a meaningful improvement while keeping your exact “median lookup → blend two branches → snap to nearest train pressure” core logic intact. The biggest correctness issue is that you’re training inspiratory medians on `u_out==0` but then still using `u_out` as a key (which is constant 0), and your expiratory predictions are not explicitly anchored to the known physical behavior that pressure should quickly drop near the PEEP-like baseline when `u_out==1`. I keep your two-branch structure, but (1) remove `u_out` from inspiratory groupby keys to increase match rates, (2) add a minimal, phase-aware rule: predict a stable low baseline for `u_out==1` using (R,C) expiratory medians (with global fallback), and (3) ensure float-key joins are more reliable by rounding `area` a bit coarser to reduce unseen-key NaNs. These are small changes that typically reduce error substantially versus sparse/NaN-heavy joins, without introducing any ML model or changing evaluation semantics.'
- What this solution (achieved 7.80107) has done: 'Your current approach is mostly being hurt by (1) sparse float-based keys causing many unseen-key fallbacks and (2) weaker breath-dynamics alignment in the lookup tables. I keep the exact same core logic (median lookup → blend two branches → snap to nearest train pressure), but make the lookups denser and more stable by (a) using a more join-friendly `time_step` integer index (0–79) instead of rounded floats, and (b) adding two very small, physically meaningful breath-local features (`u_in_lag1` and `u_in_diff1`) into the highest-resolution groupby keys with hierarchical fallback. This should reduce NaNs/over-fallback and move your MAE down materially toward the target without introducing any ML model or changing the evaluation semantics. The submission writing and snapping behavior remain the same.'
- What this solution (achieved 8.59533) has done: 'Your current MAE (7.80) is far above the target (0.163), so we need a meaningful improvement while keeping your exact “median lookup → blend two branches → snap to nearest train pressure” core idea. The largest low-risk gain here is to stop using high-cardinality float-ish cumulative keys (which force heavy fallback to coarse medians) and instead condition the medians on compact, join-stable breath-dynamics proxies that generalize to test: time index plus a discretized “valve-open progression” (`u_in_cumsum` quantile bin) and a discretized “delivered volume proxy” (`area` quantile bin), with the same hierarchical fallback. I keep your two-branch structure and expiratory handling, but replace the sparse highest-resolution keys with these bin features to increase hit-rate and reduce over-fallback. Submission writing, columns, sorting, and snapping to the nearest train pressure level remain unchanged.'
- What this solution (achieved 7.80195) has done: 'I fix the crash by using pandas’ nullable integer dtype correctly (it must be `pd.array(..., dtype="Int64")` rather than NumPy’s `.astype("Int64")`, which causes your TypeError). I also make the bin columns consistently typed on both train and test so merges/groupbys don’t silently fail or upcast, but I won’t change your lookup/blend/snap logic or any feature definitions. Finally, I keep the same output path/name and ensure `submission.csv` is always written with the required `id,pressure` columns and no NaNs.'
- What this solution (achieved 8.1684) has done: 'Your current score (7.80 MAE; lower is better) is still far from the target (0.163), so we need a meaningful but still “same lookup/blend/snap” improvement. The biggest low-risk issue is that the current binning for `u_in_cumsum` and `area` is learned via an approximate (R,C,col)->bin mapping, which is noisy and often mismatched; instead, we can compute bins deterministically from within-breath ranks (percentile position) for both train and test, which is join-stable and better aligns with dynamics without changing your overall approach. I keep your two-branch median tables (A keyed by time index, B keyed by u_in_round) and the same hierarchical fallback and nearest-pressure snapping, but replace the bin construction with a consistent per-(breath_id) rank-based binning (still optionally conditioned by R,C via the medians). This should increase “hit rate” for the higher-resolution groupbys and reduce fallback to coarse medians, moving MAE down toward the target while keeping the core logic intact.'
- What this solution (achieved 8.04676) has done: 'Your current MAE is far worse than the target (8.1684 vs 0.1629; lower is better), so we should improve score substantially while keeping your same “median lookup → blend two branches → snap to nearest train pressure” core logic. The largest issue is that your within-breath rank bins are computed independently in train vs test, so the same physical point in the breath can land in different bins and cause many missed joins/fallbacks; I replace that with deterministic bins derived from the time index (`t_idx`) so train/test keys align perfectly. I also keep your existing keys/features but make the bin columns small-int (not nullable) to avoid merge dtype quirks and reduce unnecessary NaNs. Everything else (two-branch blend weights, expiratory baseline handling, and nearest-pressure snapping) stays the same and a valid `submission.csv` is still written.'
- What this solution (achieved 7.88883) has done: 'Your MAE (8.0468; lower is better) is still far above the target (0.1629), so we should improve substantially while keeping your same median-lookup → blend two branches → snap-to-nearest-pressure core. The biggest low-risk issue is that your high-resolution lookups are too sparse and fall back too often; to fix that without changing the approach, I make the binning a bit coarser (fewer bins) and slightly reduce key cardinality by rounding the lag/diff features less aggressively (more stable matches). I also add one extra hierarchical fallback level that drops the lag/diff terms but keeps (R,C,t_idx,u_in_round) to increase hit-rate before collapsing to coarse medians. Everything else (inspiratory-only medians, separate expiratory baseline, blend weights, and `find_nearest` snapping) stays the same, and it still writes a valid `submission.csv`.'
- What this solution (achieved 7.73914) has done: 'Your score is much worse than the target (7.89 vs 0.163 MAE; lower is better), so we should improve materially while keeping your exact “median lookup → blend two branches → snap to nearest train pressure” approach. The biggest low-risk issue is that your expiratory (`u_out==1`) predictions are evaluated too (since Kaggle computes MAE only where the *true* inspiratory mask holds, but your model should still avoid pathological values), and your current expiratory baseline ignores how quickly pressure drops within a breath; we add a minimal time-index-conditioned expiratory median table and fall back hierarchically (t_idx → (R,C) → global). For inspiratory predictions, we add one additional high-hit-rate fallback level that uses `(R,C,t_idx,u_in_round)` before dropping to `(R,C,t_idx)` in branch A, improving match rate without changing features or introducing ML. Finally, we make `find_nearest` vectorized for speed (same semantics) to stay within time limits while keeping outputs identical up to negligible tie-breaking.'
- What this solution (achieved 7.81486) has done: 'Your current MAE (7.739) is far worse than the target (0.163; lower is better), so we should make a small but meaningful correction that better matches the competition metric while preserving your exact “median lookup → blend two branches → snap to nearest train pressure” core logic. The metric ignores expiratory timesteps, so we can safely set all `u_out==1` predictions to a stable baseline (global inspiratory median snapped to train pressures) to avoid any harmful noise from an expiratory model without changing the inspiratory pipeline. Separately, your `u_in_round` is currently too fine (0.1) which creates sparse keys and forces frequent fallback; rounding to 1.0 increases hit-rate in your existing lookup tables and typically reduces inspiratory MAE without introducing any new modeling approach. Finally, we keep all file paths and the same submission writing, and only touch these two levers to move the score down toward the target.'
- What this solution (achieved 8.44115) has done: 'Your current score is far above the target (7.81 vs 0.163, lower is better), so we need a meaningful improvement while keeping your exact “median lookup → blend two branches → snap to nearest train pressure” core intact. The biggest issue is that your two “bin” features (`u_in_cumsum_bin`, `area_bin`) are currently just re-labeled time bins (derived only from `t_idx`), so they don’t add information and instead create unnecessary key sparsity; we replace them with true within-breath binnings of `u_in_cumsum` and `area` computed deterministically from `t_idx`-aligned quantiles on the training set. This keeps the same lookup/blend structure and same features, but makes the keys meaningfully represent breath progression and improves join hit-rate. Finally, we keep expiratory handling and snapping unchanged, and still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import copy
import glob
import random
from random import random as rd
import gc



## === cell 1
df_train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)


def find_nearest(prediction: float) -> float:
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        return float(sorted_pressures[-1])
    elif insert_idx == 0:
        return float(sorted_pressures[0])
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return (
        float(lower_val)
        if abs(lower_val - prediction) < abs(upper_val - prediction)
        else float(upper_val)
    )


def snap_to_train_pressures(pred: np.ndarray) -> np.ndarray:
    pred = pred.astype(np.float64, copy=False)
    idx = np.searchsorted(sorted_pressures, pred, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)

    upper = sorted_pressures[idx]
    lower = sorted_pressures[np.clip(idx - 1, 0, total_pressures_len - 1)]

    choose_lower = (idx > 0) & (np.abs(pred - lower) < np.abs(upper - pred))
    out = np.where(choose_lower, lower, upper)
    return out.astype(np.float64, copy=False)


def set_seed(seed: int = 2021):
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
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def g(dp):
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)
    file_count = len(l)
    loop_time = 150
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
    pred_list = []
    for k in range(loop_time):
        weight = []
        set_seed(k)
        for j in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        for j in range(len(weight)):
            weight[j] /= weight_sum
        weight.sort(reverse=True)
        temp = 0
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()
    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output.pressure = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.55 + b.pressure * 0.45
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
sample_sub = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)

train = df_train[
    ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
].copy()
test = df_test[["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]].copy()

train = train.sort_values(["breath_id", "time_step"], kind="mergesort")
test = test.sort_values(["breath_id", "time_step"], kind="mergesort")
train["t_idx"] = train.groupby("breath_id", sort=False).cumcount().astype(np.int16)
test["t_idx"] = test.groupby("breath_id", sort=False).cumcount().astype(np.int16)

train["u_in_round"] = train["u_in"].round(0)  # was .round(1)
test["u_in_round"] = test["u_in"].round(0)

train["dt"] = train.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0)
test["dt"] = test.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0)

train["u_in_cumsum"] = train.groupby("breath_id", sort=False)["u_in"].cumsum()
test["u_in_cumsum"] = test.groupby("breath_id", sort=False)["u_in"].cumsum()

train["area"] = (
    (train["u_in"] * train["dt"]).groupby(train["breath_id"], sort=False).cumsum()
)
test["area"] = (
    (test["u_in"] * test["dt"]).groupby(test["breath_id"], sort=False).cumsum()
)

train["u_in_lag1"] = train.groupby("breath_id", sort=False)["u_in"].shift(1).fillna(0.0)
test["u_in_lag1"] = test.groupby("breath_id", sort=False)["u_in"].shift(1).fillna(0.0)
train["u_in_diff1"] = train.groupby("breath_id", sort=False)["u_in"].diff().fillna(0.0)
test["u_in_diff1"] = test.groupby("breath_id", sort=False)["u_in"].diff().fillna(0.0)

train["u_in_lag1_round"] = train["u_in_lag1"].round(-1)  # tens
test["u_in_lag1_round"] = test["u_in_lag1"].round(-1)
train["u_in_diff1_round"] = train["u_in_diff1"].round(1)
test["u_in_diff1_round"] = test["u_in_diff1"].round(1)

train_insp = train[train["u_out"] == 0].copy()
train_exp = train[train["u_out"] == 1].copy()

global_median_insp = float(train_insp["pressure"].median())
global_median_exp = (
    float(train_exp["pressure"].median()) if len(train_exp) else global_median_insp
)


def _add_quantile_bins_by_rc_tidx(
    tr_insp: pd.DataFrame,
    tr_all: pd.DataFrame,
    te_all: pd.DataFrame,
    value_col: str,
    n_bins: int,
    out_col: str,
):
    grp = tr_insp.groupby(["R", "C", "t_idx"], sort=False)[value_col]

    qs = np.linspace(0.0, 1.0, n_bins + 1)

    edges = grp.quantile(qs).unstack(-1).reset_index()  # columns are quantiles
    qcols = [c for c in edges.columns if c not in ["R", "C", "t_idx"]]
    qcols_sorted = sorted(qcols, key=lambda x: float(x))
    edges = edges[["R", "C", "t_idx"] + qcols_sorted].copy()

    global_edges = np.quantile(tr_insp[value_col].to_numpy(np.float64), qs)
    global_edges = np.maximum.accumulate(global_edges)

    def _assign_bins(df: pd.DataFrame) -> np.ndarray:
        d = df.merge(edges, on=["R", "C", "t_idx"], how="left")
        vals = d[value_col].to_numpy(np.float64)

        qmat = d[qcols_sorted].to_numpy(np.float64)
        missing = np.isnan(qmat).any(axis=1)
        if missing.any():
            qmat[missing, :] = global_edges

        qmat = np.maximum.accumulate(qmat, axis=1)

        internal = qmat[:, 1:-1]  # (n_bins-1) cut points
        bins = (vals[:, None] > internal).sum(axis=1).astype(np.int16)
        bins = np.clip(bins, 0, n_bins - 1)
        return bins

    tr_all[out_col] = _assign_bins(tr_all)
    te_all[out_col] = _assign_bins(te_all)
    return tr_all, te_all


train, test = _add_quantile_bins_by_rc_tidx(
    tr_insp=train_insp,
    tr_all=train,
    te_all=test,
    value_col="u_in_cumsum",
    n_bins=15,
    out_col="u_in_cumsum_bin",
)
train, test = _add_quantile_bins_by_rc_tidx(
    tr_insp=train_insp,
    tr_all=train,
    te_all=test,
    value_col="area",
    n_bins=20,
    out_col="area_bin",
)

train_insp = train[train["u_out"] == 0].copy()
train_exp = train[train["u_out"] == 1].copy()

med_rc_exp = (
    train_exp.groupby(["R", "C"], sort=False)["pressure"]
    .median()
    .rename("p_rc_exp")
    .reset_index()
)

med_rc_t_exp = (
    train_exp.groupby(["R", "C", "t_idx"], sort=False)["pressure"]
    .median()
    .rename("p_rc_t_exp")
    .reset_index()
)

med_rc = (
    train_insp.groupby(["R", "C"], sort=False)["pressure"]
    .median()
    .rename("p_rc")
    .reset_index()
)

med_a = (
    train_insp.groupby(
        [
            "R",
            "C",
            "t_idx",
            "u_in_cumsum_bin",
            "area_bin",
            "u_in_lag1_round",
            "u_in_diff1_round",
        ],
        sort=False,
    )["pressure"]
    .median()
    .rename("p_a")
    .reset_index()
)
test_a = test.merge(
    med_a,
    on=[
        "R",
        "C",
        "t_idx",
        "u_in_cumsum_bin",
        "area_bin",
        "u_in_lag1_round",
        "u_in_diff1_round",
    ],
    how="left",
)

med_a2 = (
    train_insp.groupby(["R", "C", "t_idx", "u_in_cumsum_bin", "area_bin"], sort=False)[
        "pressure"
    ]
    .median()
    .rename("p_a2")
    .reset_index()
)
test_a = test_a.merge(
    med_a2, on=["R", "C", "t_idx", "u_in_cumsum_bin", "area_bin"], how="left"
)

med_a3 = (
    train_insp.groupby(["R", "C", "t_idx", "u_in_cumsum_bin"], sort=False)["pressure"]
    .median()
    .rename("p_a3")
    .reset_index()
)
test_a = test_a.merge(med_a3, on=["R", "C", "t_idx", "u_in_cumsum_bin"], how="left")

med_a_u = (
    train_insp.groupby(["R", "C", "t_idx", "u_in_round"], sort=False)["pressure"]
    .median()
    .rename("p_a_u")
    .reset_index()
)
test_a = test_a.merge(med_a_u, on=["R", "C", "t_idx", "u_in_round"], how="left")

med_a4 = (
    train_insp.groupby(["R", "C", "t_idx"], sort=False)["pressure"]
    .median()
    .rename("p_a4")
    .reset_index()
)
test_a = test_a.merge(med_a4, on=["R", "C", "t_idx"], how="left")
test_a = test_a.merge(med_rc, on=["R", "C"], how="left")

pred_a = test_a["p_a"].to_numpy(dtype=np.float64)
pred_a2 = test_a["p_a2"].to_numpy(dtype=np.float64)
pred_a3 = test_a["p_a3"].to_numpy(dtype=np.float64)
pred_a_u = test_a["p_a_u"].to_numpy(dtype=np.float64)
pred_a4 = test_a["p_a4"].to_numpy(dtype=np.float64)
pred_rc = test_a["p_rc"].to_numpy(dtype=np.float64)

pred_a = np.where(~np.isnan(pred_a), pred_a, pred_a2)
pred_a = np.where(~np.isnan(pred_a), pred_a, pred_a3)
pred_a = np.where(~np.isnan(pred_a), pred_a, pred_a_u)
pred_a = np.where(~np.isnan(pred_a), pred_a, pred_a4)
pred_a = np.where(~np.isnan(pred_a), pred_a, pred_rc)
pred_a = np.where(np.isnan(pred_a), global_median_insp, pred_a)

med_b = (
    train_insp.groupby(
        [
            "R",
            "C",
            "u_in_round",
            "u_in_cumsum_bin",
            "area_bin",
            "u_in_lag1_round",
            "u_in_diff1_round",
        ],
        sort=False,
    )["pressure"]
    .median()
    .rename("p_b")
    .reset_index()
)
test_b = test.merge(
    med_b,
    on=[
        "R",
        "C",
        "u_in_round",
        "u_in_cumsum_bin",
        "area_bin",
        "u_in_lag1_round",
        "u_in_diff1_round",
    ],
    how="left",
)

med_b_t = (
    train_insp.groupby(["R", "C", "t_idx", "u_in_round"], sort=False)["pressure"]
    .median()
    .rename("p_b_t")
    .reset_index()
)
test_b = test_b.merge(med_b_t, on=["R", "C", "t_idx", "u_in_round"], how="left")

med_b2 = (
    train_insp.groupby(
        ["R", "C", "u_in_round", "u_in_cumsum_bin", "area_bin"], sort=False
    )["pressure"]
    .median()
    .rename("p_b2")
    .reset_index()
)
test_b = test_b.merge(
    med_b2, on=["R", "C", "u_in_round", "u_in_cumsum_bin", "area_bin"], how="left"
)

med_b3 = (
    train_insp.groupby(["R", "C", "u_in_round", "u_in_cumsum_bin"], sort=False)[
        "pressure"
    ]
    .median()
    .rename("p_b3")
    .reset_index()
)
test_b = test_b.merge(
    med_b3, on=["R", "C", "u_in_round", "u_in_cumsum_bin"], how="left"
)

med_b4 = (
    train_insp.groupby(["R", "C", "u_in_round"], sort=False)["pressure"]
    .median()
    .rename("p_b4")
    .reset_index()
)
test_b = test_b.merge(med_b4, on=["R", "C", "u_in_round"], how="left")
test_b = test_b.merge(med_rc, on=["R", "C"], how="left")

pred_b = test_b["p_b"].to_numpy(dtype=np.float64)
pred_b_t = test_b["p_b_t"].to_numpy(dtype=np.float64)
pred_b2 = test_b["p_b2"].to_numpy(dtype=np.float64)
pred_b3 = test_b["p_b3"].to_numpy(dtype=np.float64)
pred_b4 = test_b["p_b4"].to_numpy(dtype=np.float64)
pred_rc_b = test_b["p_rc"].to_numpy(dtype=np.float64)

pred_b = np.where(~np.isnan(pred_b), pred_b, pred_b_t)
pred_b = np.where(~np.isnan(pred_b), pred_b, pred_b2)
pred_b = np.where(~np.isnan(pred_b), pred_b, pred_b3)
pred_b = np.where(~np.isnan(pred_b), pred_b, pred_b4)
pred_b = np.where(~np.isnan(pred_b), pred_b, pred_rc_b)
pred_b = np.where(np.isnan(pred_b), global_median_insp, pred_b)

pred_insp = 0.55 * pred_a + 0.45 * pred_b

exp_baseline = float(find_nearest(global_median_insp))

u_out_arr = test["u_out"].to_numpy(dtype=np.int64)
pred = np.where(u_out_arr == 0, pred_insp, exp_baseline)

pred = snap_to_train_pressures(pred)

submission = pd.DataFrame({"id": df_test["id"].astype(np.int64), "pressure": pred})
submission = submission.sort_values("id").reset_index(drop=True)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Saved submission.csv with shape:", submission.shape)
print("Pressure stats:", submission["pressure"].describe())
print("Any NaNs in submission?", submission["pressure"].isna().any())
print("u_out distribution in test:", pd.Series(u_out_arr).value_counts().to_dict())
print(
    "global_median_insp:", global_median_insp, "global_median_exp:", global_median_exp
)
print("exp_baseline (snapped):", exp_baseline)
