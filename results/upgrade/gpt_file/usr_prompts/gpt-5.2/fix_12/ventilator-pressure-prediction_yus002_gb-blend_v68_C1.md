# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.172396893516722

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 7.24999) has done: 'Your script fails because it tries to blend two external submission files that don’t exist in this environment (`../input/gb-blending/...`). Since we can’t access those, the minimal fix is to replace that blending step with an end-to-end model that trains on `train.csv` and predicts `test.csv`, then writes a valid `submission.csv` with `id,pressure`. To keep the original “pressure snapping” semantics, the code still map predictions to the nearest valid pressure value observed in the training set. The model is a lightweight per-(R,C,time_step,u_out) median lookup (with safe fallbacks) so it runs within the time limit and yields a nontrivial score instead of no submission.'
- What this solution (achieved 7.54092) has done: 'Your current score (7.24999 MAE) is far from the target (0.1724), so we need a meaningful but still simple improvement while keeping your “lookup + snap-to-valid-pressures” core approach intact. The biggest issue is that your lookup key doesn’t include `u_in` or any breath history, so it can’t track inspiratory pressure changes; adding a minimal set of time-series-derived features (cumulative `u_in` and a short lag) keeps the same median-aggregation logic but makes the mapping much more informative. We build median maps on `(R,C,time_step,u_out,u_in_rounded, cum_u_in_rounded, u_in_lag1_rounded)` with safe fallbacks to coarser groupings, then snap with your existing `find_nearest`. This stays within the installed packages, runs fast (groupby medians), and should move MAE substantially toward the target without changing the overall modeling paradigm.'
- What this solution (achieved 7.54092) has done: 'Your current MAE (7.54) is far above the target (0.172), so we should improve the existing “groupby-median lookup + snap-to-valid-pressures” approach without changing its core logic. The biggest win with minimal disruption is to add a tiny bit more breath history to the lookup keys: include a 2-step lag of `u_in`, plus a very lightweight interaction proxy (`u_in * (1-u_out)`) and its cumulative sum, since only inspiratory (mostly `u_out=0`) matters for scoring. We keep your same median-maps-with-fallbacks design, just adding one more “main” map (most specific) and extending the existing hierarchy so missing-key cases still fall back safely. This should materially reduce error while staying fast (pure pandas groupbys) and still producing a valid `submission.csv`.'
- What this solution (achieved 8.48748) has done: 'Your current MAE (7.54) is far above the target (0.172), so the fastest legitimate way to move toward the target without changing your “groupby median lookup + fallback + snap-to-valid-pressures” core is to reduce key sparsity and add one more very lightweight breath-history signal. I (1) quantize `time_step` to an integer index (0..79) so train/test keys match exactly (your current float merge likely misses a lot), (2) include a tiny additional proxy for dynamics (`delta_u_in` and its rounded form) in the most-specific map, and (3) keep the same fallback hierarchy but base all time-dependent groupings on the quantized time index. This keeps the same semantics (median lookup + fallbacks + snapping), but should dramatically increase hit-rate of the main maps and reduce error toward the target. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 7.9913) has done: 'Your current MAE is far above the target, so we should improve the accuracy of your existing “groupby-median lookup + fallback + snap-to-valid-pressures” approach without changing its core paradigm. The biggest issue is still key mismatch/sparsity: even with `t_idx`, your highly-specific maps miss often, and including raw `u_out` in all keys can hurt because the metric ignores expiratory rows. I (1) add a minimal but very informative breath-history feature (`time_since_insp_start`) and include it in the top lookup (keeps same lookup/median logic), and (2) add an inspiratory-only set of median maps (trained on `u_out==0`) used only when predicting inspiratory rows, with safe fallbacks unchanged for expiratory. This keeps the same evaluation semantics (median lookup then nearest-pressure snapping) but should materially reduce error toward the target by focusing modeling capacity on the scored phase.'
- What this solution (achieved 8.03168) has done: 'Your current MAE (7.99) is far above the target (0.172), so we should improve accuracy while keeping your exact “groupby-median lookup + fallback hierarchy + snap-to-valid-pressures” approach intact. The biggest likely source of error now is (a) merging on float-derived `t_idx`/`t_since_insp_idx` that can still mismatch between train/test, and (b) using `time_since_insp_start` which is mostly redundant when `t_idx` already indexes the 80 steps and can introduce extra sparsity. I make `t_idx` an exact within-breath integer step index (0..79) via `cumcount()` (minimal semantic change, but greatly increases key hit-rate), and I drop `t_since_insp_idx` from the *most specific* maps to reduce sparsity while keeping your existing fallback maps and inspiratory-only logic. Everything else (median maps, fallbacks, snapping, output file) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 7.63348) has done: 'Your score is far worse than the target (lower-is-better), so we should make a small, safe accuracy improvement while keeping your exact “groupby-median lookup + fallback hierarchy + snap-to-valid-pressures” core intact. The biggest likely issue is key mismatch due to `id` not being globally unique in these files (it repeats 1..2000 for every breath), which can scramble row alignment when you rebuild `sub` via merges; we instead write predictions in the exact order of `test.csv` and only use `sample_submission.csv` for the `id` column. Next, to improve hit-rate without changing the paradigm, we replace the float-derived `cum_u_in_r` binning with a within-breath monotonic integer bin (`cum_u_in_bin`) computed from `cum_u_in` rank-scaled per breath, which is more consistent between train/test and reduces sparsity while preserving “history” information. Everything else (feature set idea, median maps, inspiratory-only maps, fillna fallbacks, and nearest-pressure snapping) stays the same, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 8.14573) has done: 'The crash happens because you try to build quantile bin edges from `df_train["cum_u_in"]` and `df_train["cum_u_in_insp"]` before those feature columns exist. I fix this by computing the needed cumulative features once on a minimal copy of `train` inside the bin-edge builder, so the rest of your feature/median-lookup/fallback/snapping logic stays the same. I also keep the test prediction order aligned with `sample_submission.csv` exactly as you already do, and ensure `submission.csv` is written with the required columns. No model/logic changes beyond the bug fix, so scoring behavior should only change insofar as the code now runs end-to-end.'
- What this solution (achieved 7.60617) has done: 'We keep your exact “groupby median lookup + hierarchical fallbacks + snap-to-nearest valid pressure” approach, but make a minimal change to improve hit-rate and reduce sparsity: compute the cumulative bins *within each (R,C) lung setting* instead of globally. This is still the same binning feature (just better calibrated per lung attributes), and it typically increases the usefulness of `cum_u_in_bin` / `cum_u_in_insp_bin` without changing any modeling semantics. We also fix a subtle alignment risk by ensuring the prediction Series is ordered exactly like `sample_submission` via the `id` merge output (rather than relying on incidental index alignment). Everything remains pure pandas/numpy, runs quickly, and still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
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


def find_nearest(prediction):
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        return float(sorted_pressures[-1])
    elif insert_idx == 0:
        return float(sorted_pressures[0])
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return float(
        lower_val
        if abs(lower_val - prediction) < abs(upper_val - prediction)
        else upper_val
    )


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
    for i in range(loop_time):
        weight = []
        set_seed(i)
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


def _make_bins_from_train_by_rc(
    train_df: pd.DataFrame, col: str, n_bins: int
) -> pd.DataFrame:
    if col not in ("cum_u_in", "cum_u_in_insp"):
        raise KeyError(f"Only supports cumulative cols, got '{col}'")

    tmp = train_df[["breath_id", "time_step", "u_in", "u_out", "R", "C"]].copy()
    tmp.sort_values(["breath_id", "time_step"], inplace=True)
    tmp["cum_u_in"] = tmp.groupby("breath_id")["u_in"].cumsum()
    tmp["u_in_insp"] = tmp["u_in"] * (1.0 - tmp["u_out"].astype(float))
    tmp["cum_u_in_insp"] = tmp.groupby("breath_id")["u_in_insp"].cumsum()

    qs = np.linspace(0.0, 1.0, n_bins + 1)

    rows = []
    for (r, c), g in tmp.groupby(["R", "C"], sort=False):
        x = g[col].to_numpy(dtype=np.float64, copy=False)
        x = x[np.isfinite(x)]
        if x.size == 0:
            edges = np.array([0.0, 1.0], dtype=np.float64)
        else:
            edges = np.unique(np.quantile(x, qs))
            if edges.size < 2:
                edges = np.array([float(np.min(x)), float(np.max(x))], dtype=np.float64)
                if edges[0] == edges[1]:
                    edges[1] = edges[0] + 1e-6
        rows.append({"R": int(r), "C": int(c), "edges": edges.astype(np.float64)})

    return pd.DataFrame(rows)


def _apply_bins_by_edges(values: pd.Series, edges: np.ndarray) -> pd.Series:
    out = pd.cut(values, bins=edges, labels=False, include_lowest=True)
    out = out.fillna(len(edges) - 2).astype(np.int16)
    return out


def add_features(
    df: pd.DataFrame, edges_cum_by_rc: pd.DataFrame, edges_cum_insp_by_rc: pd.DataFrame
) -> pd.DataFrame:
    df = df.copy()
    df.sort_values(["breath_id", "time_step"], inplace=True)

    df["t_idx"] = df.groupby("breath_id").cumcount().astype(np.int16)

    df["cum_u_in"] = df.groupby("breath_id")["u_in"].cumsum()
    df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1).fillna(0.0)
    df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2).fillna(0.0)

    df["delta_u_in"] = (df["u_in"] - df["u_in_lag1"]).astype(np.float32)

    df["u_in_insp"] = df["u_in"] * (1.0 - df["u_out"].astype(float))
    df["cum_u_in_insp"] = df.groupby("breath_id")["u_in_insp"].cumsum()

    edges_map_cum = {
        (int(r), int(c)): e
        for r, c, e in edges_cum_by_rc[["R", "C", "edges"]].itertuples(
            index=False, name=None
        )
    }
    edges_map_cum_insp = {
        (int(r), int(c)): e
        for r, c, e in edges_cum_insp_by_rc[["R", "C", "edges"]].itertuples(
            index=False, name=None
        )
    }

    cum_bins = np.empty(len(df), dtype=np.int16)
    cum_insp_bins = np.empty(len(df), dtype=np.int16)

    for (r, c), idx in df.groupby(["R", "C"], sort=False).indices.items():
        e1 = edges_map_cum[(int(r), int(c))]
        e2 = edges_map_cum_insp[(int(r), int(c))]
        cum_bins[idx] = _apply_bins_by_edges(df.loc[idx, "cum_u_in"], e1).to_numpy()
        cum_insp_bins[idx] = _apply_bins_by_edges(
            df.loc[idx, "cum_u_in_insp"], e2
        ).to_numpy()

    df["cum_u_in_bin"] = cum_bins
    df["cum_u_in_insp_bin"] = cum_insp_bins

    df["u_in_r"] = (df["u_in"] * 2).round().astype(np.int16)  # 0.5 resolution
    df["u_in_lag1_r"] = (df["u_in_lag1"] * 2).round().astype(np.int16)
    df["u_in_lag2_r"] = (df["u_in_lag2"] * 2).round().astype(np.int16)
    df["u_in_insp_r"] = (df["u_in_insp"] * 2).round().astype(np.int16)
    df["delta_u_in_r"] = (df["delta_u_in"] * 2).round().astype(np.int16)

    return df


N_BINS_CUM = 48
N_BINS_CUM_INSP = 48

edges_cum_by_rc = _make_bins_from_train_by_rc(df_train, "cum_u_in", N_BINS_CUM)
edges_cum_insp_by_rc = _make_bins_from_train_by_rc(
    df_train, "cum_u_in_insp", N_BINS_CUM_INSP
)

train_fe = add_features(df_train, edges_cum_by_rc, edges_cum_insp_by_rc)
test_fe = add_features(df_test, edges_cum_by_rc, edges_cum_insp_by_rc)

train_insp = train_fe[train_fe["u_out"] == 0].copy()

grp_main3 = [
    "R",
    "C",
    "t_idx",
    "u_out",
    "u_in_r",
    "cum_u_in_bin",
    "u_in_lag1_r",
    "u_in_lag2_r",
    "cum_u_in_insp_bin",
    "delta_u_in_r",
]
median_map_main3 = (
    train_insp.groupby(grp_main3, sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_main3"})
)

grp_main2 = [
    "R",
    "C",
    "t_idx",
    "u_out",
    "u_in_r",
    "cum_u_in_bin",
    "u_in_lag1_r",
    "u_in_lag2_r",
    "cum_u_in_insp_bin",
]
median_map_main2 = (
    train_insp.groupby(grp_main2, sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_main2"})
)

grp_main = [
    "R",
    "C",
    "t_idx",
    "u_out",
    "u_in_r",
    "cum_u_in_bin",
    "u_in_lag1_r",
]
median_map_main = (
    train_insp.groupby(grp_main, sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_main"})
)

grp_mid = ["R", "C", "t_idx", "u_out", "u_in_r", "cum_u_in_bin"]
median_map_mid = (
    train_insp.groupby(grp_mid, sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_mid"})
)

grp_basic = ["R", "C", "t_idx", "u_out", "u_in_r"]
median_map_basic = (
    train_insp.groupby(grp_basic, sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_basic"})
)

median_map_rcu = (
    train_insp.groupby(["R", "C", "u_out"], sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_rcu"})
)

median_map_u = (
    train_insp.groupby(["u_out"], sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_u"})
)

grp_insp3 = [
    "R",
    "C",
    "t_idx",
    "u_in_r",
    "cum_u_in_bin",
    "u_in_lag1_r",
    "u_in_lag2_r",
    "cum_u_in_insp_bin",
    "delta_u_in_r",
]
median_map_insp3 = (
    train_insp.groupby(grp_insp3, sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_insp3"})
)

grp_insp2 = [
    "R",
    "C",
    "t_idx",
    "u_in_r",
    "cum_u_in_bin",
    "u_in_lag1_r",
    "u_in_lag2_r",
    "cum_u_in_insp_bin",
]
median_map_insp2 = (
    train_insp.groupby(grp_insp2, sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_insp2"})
)

grp_insp1 = [
    "R",
    "C",
    "t_idx",
    "u_in_r",
    "cum_u_in_bin",
    "u_in_lag1_r",
]
median_map_insp1 = (
    train_insp.groupby(grp_insp1, sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_insp1"})
)

median_map_insp_rc = (
    train_insp.groupby(["R", "C"], sort=False)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pred_insp_rc"})
)

median_map_insp_global = pd.DataFrame(
    {"pred_insp_global": [float(train_insp["pressure"].median())]}
)

merge_cols = ["id", "breath_id", "t_idx", "u_out"] + grp_main3
test_pred = test_fe[merge_cols].merge(median_map_main3, on=grp_main3, how="left")
test_pred = test_pred.merge(median_map_main2, on=grp_main2, how="left")
test_pred = test_pred.merge(median_map_main, on=grp_main, how="left")
test_pred = test_pred.merge(median_map_mid, on=grp_mid, how="left")
test_pred = test_pred.merge(median_map_basic, on=grp_basic, how="left")
test_pred = test_pred.merge(median_map_rcu, on=["R", "C", "u_out"], how="left")
test_pred = test_pred.merge(median_map_u, on=["u_out"], how="left")

test_pred = test_pred.merge(median_map_insp3, on=grp_insp3, how="left")
test_pred = test_pred.merge(median_map_insp2, on=grp_insp2, how="left")
test_pred = test_pred.merge(median_map_insp1, on=grp_insp1, how="left")
test_pred = test_pred.merge(median_map_insp_rc, on=["R", "C"], how="left")
test_pred["__k"] = 1
median_map_insp_global["__k"] = 1
test_pred = test_pred.merge(median_map_insp_global, on="__k", how="left").drop(
    columns="__k"
)

pred_all = test_pred["pred_main3"]
pred_all = pred_all.fillna(test_pred["pred_main2"])
pred_all = pred_all.fillna(test_pred["pred_main"])
pred_all = pred_all.fillna(test_pred["pred_mid"])
pred_all = pred_all.fillna(test_pred["pred_basic"])
pred_all = pred_all.fillna(test_pred["pred_rcu"])
pred_all = pred_all.fillna(test_pred["pred_u"])
pred_all = pred_all.fillna(float(df_train["pressure"].median()))

pred_insp = test_pred["pred_insp3"]
pred_insp = pred_insp.fillna(test_pred["pred_insp2"])
pred_insp = pred_insp.fillna(test_pred["pred_insp1"])
pred_insp = pred_insp.fillna(test_pred["pred_insp_rc"])
pred_insp = pred_insp.fillna(test_pred["pred_insp_global"])
pred_insp = pred_insp.fillna(float(df_train["pressure"].median()))

is_insp = test_pred["u_out"].to_numpy() == 0
pred = pred_all.astype(float).to_numpy(copy=True)
pred[is_insp] = pred_insp.astype(float).to_numpy()[is_insp]

pred_series = pd.Series(pred, index=test_pred.index, dtype="float64")
pred_series[is_insp] = pred_insp.astype(float).to_numpy()[is_insp]
pred_series = pred_series.groupby(test_pred["breath_id"], sort=False).ffill().bfill()

pred_series = pred_series.map(find_nearest)

sub = sample_sub.copy()
if len(sub) != len(pred_series):
    raise RuntimeError(
        f"Row count mismatch: submission {len(sub)} vs preds {len(pred_series)}"
    )
sub["pressure"] = pred_series.to_numpy()

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3058764394.py in <cell line: 0>()
    256 
    257 merge_cols = ["id", "breath_id", "t_idx", "u_out"] + grp_main3
--> 258 test_pred = test_fe[merge_cols].merge(median_map_main3, on=grp_main3, how="left")
    259 test_pred = test_pred.merge(median_map_main2, on=grp_main2, how="left")
    260 test_pred = test_pred.merge(median_map_main, on=grp_main, how="left")

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in merge(self, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
  10830         from pandas.core.reshape.merge import merge
  10831 
> 10832         return merge(
  10833             self,
  10834             right,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
    168         )
    169     else:
--> 170         op = _MergeOperation(
    171             left_df,
    172             right_df,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in __init__(self, left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, indicator, validate)
    792             left_drop,
    793             right_drop,
--> 794         ) = self._get_merge_keys()
    795 
    796         if left_drop:

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _get_merge_keys(self)
   1308                         #  the latter of which will raise
   1309                         lk = cast(Hashable, lk)
-> 1310                         left_keys.append(left._get_label_or_level_values(lk))
   1311                         join_names.append(lk)
   1312                     else:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _get_label_or_level_values(self, key, axis)
   1923 
   1924             label_axis_name = "column" if axis == 0 else "index"
-> 1925             raise ValueError(
   1926                 f"The {label_axis_name} label '{key}' is not unique.{multi_message}"
   1927             )

ValueError: The column label 't_idx' is not unique.
