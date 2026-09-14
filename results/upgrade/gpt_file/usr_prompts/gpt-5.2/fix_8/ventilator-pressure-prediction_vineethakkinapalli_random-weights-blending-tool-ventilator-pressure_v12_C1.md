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

0.1396788290304115

# 6. Current score

2.00725

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.86378) has done: 'I make the script robust to the missing `ventilator-pressure-high-score-submissions` dataset (it isn’t in your provided paths), which currently causes the `IndexError` because no allowed files are found to blend. I keep the blending logic intact when those files exist, but add safe guards for empty/short file lists and a fallback that still produces a valid `submission.csv` using a simple in-training pressure prior (median pressure) snapped to the nearest known pressure level. I also fix the cell numbering (your notebook starts at cell 0) and ensure the output filename ends with `.csv` and matches the required `id,pressure` format. These changes are execution/stability fixes; they don’t change model architecture/training because the original approach is purely blending.'
- What this solution (achieved 6.11805) has done: 'Your current score is far worse than the target (lower-is-better), and the main issue is that your blend dataset isn’t available, so you always fall back to a constant median-pressure prediction. To move the score toward the target with minimal changes and without introducing a new “model,” I keep your exact blending logic intact when files exist, but upgrade the fallback to a simple lookup prior: for each (R, C, time_step, u_out) bin, predict the median training pressure conditioned on similar control state, then snap to the nearest known pressure level as you already do. This preserves the overall “submission-from-priors/blending” approach while making the fallback much closer to the true signal than a constant. The script still writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 4.08551) has done: 'Your current score (6.11805, lower-is-better) is far from the target (0.1397), and the biggest remaining gap is that the fallback “prior” ignores the key control signal `u_in`. To move toward the target while preserving your “no-model / prior-or-blend” core logic, I minimally upgrade the fallback lookup to condition on `(R, C, u_out, discretized time_step, discretized u_in)` with a safe hierarchical backoff if a fine-grained bin is missing. I keep the same nearest-known-pressure snapping and keep your blending path unchanged when the external high-score submissions exist. I also fix the cell numbering to start at 1 (as required) without changing execution behavior.'
- What this solution (achieved 1.87614) has done: 'Your current MAE (4.08551, lower-is-better) is still far from the target (0.13968), so we should improve the fallback prior because the external blend files are not present in your provided paths. Keeping your “train-lookup prior + nearest-pressure snapping” core logic intact, I minimally strengthen the fallback by conditioning on per-breath history: add a discretized cumulative `u_in` (“area under the inflow curve so far”) and previous `u_in` bin, with a hierarchical backoff to your existing bins when sparse. This remains a deterministic median-lookup (not a new model), but better matches ventilator physics (pressure depends strongly on integrated inflow). The blending path and snapping remain unchanged, and the script still write a valid `submission.csv`.'
- What this solution (achieved 2.01036) has done: 'Your current score is much worse than the target (lower-is-better), and since the external blend dataset isn’t available, almost all performance comes from the fallback prior. To move toward the target without changing the overall “lookup prior + snap-to-known-pressures” core logic, I minimally improve the fallback features to better match the scoring (inspiratory phase) by adding an inspiratory-only training filter (`u_out == 0`) when building medians. I also add a slightly more physics-aligned history proxy (`u_in * delta_time` cumulative volume) while keeping the same hierarchical backoff structure and the same submission-writing behavior. These changes are localized to the fallback and keep the blending path untouched when those files exist.'
- What this solution (achieved 2.01593) has done: 'I keep your core “blend-if-available, otherwise deterministic lookup prior + snap-to-known-pressures” logic unchanged, but strengthen the fallback lookup so it captures the main physics signal that your current bins still miss: breath-level integrated flow and lagged controls. Specifically, I add two minimal history features (lagged cumulative `u_in*dt` and lagged `u_out`) and use a slightly richer hierarchical backoff that still ends in the same global median + snapping behavior. This should reduce MAE (lower-is-better) from your current 2.01036 toward the 0.1397 target without introducing a new model or changing evaluation semantics. The blending path remains identical when the external submission files exist, and the script still writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 2.00725) has done: 'Your current MAE (2.01593, lower-is-better) is still far from the target (0.13968), so we should keep your existing “blend-if-available, otherwise deterministic lookup prior + snap-to-known-pressures” core logic intact and only strengthen the fallback lookup. The minimal high-impact fix is to align training medians with the competition metric by building them from inspiratory timesteps only **and** to avoid mixing inspiratory/expiratory behavior at prediction time by explicitly forcing expiratory (`u_out==1`) predictions to a stable, low-variance prior (global inspiratory median snapped to valid pressure levels). Additionally, we add one more tiny history feature already implied by your approach (lagged `u_in_dt`) to better capture instantaneous flow effects without introducing any new model or training loop. The blending path remains unchanged when external submission files exist, and the script still writes a valid `submission.csv` with `id,pressure`.'

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
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SAMPLE_SUB_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"

df_train = pd.read_csv(TRAIN_PATH)

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)


def find_nearest(prediction: float) -> float:
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx >= total_pressures_len:
        return float(sorted_pressures[-1])
    if insert_idx <= 0:
        return float(sorted_pressures[0])
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return float(
        lower_val
        if abs(lower_val - prediction) < abs(upper_val - prediction)
        else upper_val
    )


def set_seed(seed: int = 2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def _write_fallback_submission():
    """
    Score-improvement fallback (still a 'prior' submission, not a new model).

    Changes made to move MAE toward the target (lower is better) while preserving core logic:
      1) Metric alignment: build lookup medians using inspiratory phase only (u_out==0).
      2) Prediction-time alignment: expiratory phase (u_out==1) is not scored, so force those
         predictions to a stable prior to avoid injecting noise from weakly-informed bins.
      3) Minimal added dynamics feature: include lagged u_in_dt (instantaneous delivered volume),
         keeping the same deterministic median-lookup + hierarchical backoff + snapping.
    """
    df_test = pd.read_csv(TEST_PATH)
    output = pd.read_csv(SAMPLE_SUB_PATH)

    tr = df_train[
        ["breath_id", "time_step", "u_in", "u_out", "R", "C", "pressure"]
    ].copy()
    te = df_test[["breath_id", "time_step", "u_in", "u_out", "R", "C"]].copy()

    tr_insp = tr[tr["u_out"] == 0].copy()
    if len(tr_insp) == 0:
        tr_insp = tr  # safety backstop

    for df_ in (tr_insp, te):
        df_["dt"] = (
            df_.groupby("breath_id", sort=False)["time_step"]
            .diff()
            .fillna(0.0)
            .astype(np.float32)
        )
        df_["u_in_dt"] = (df_["u_in"].astype(np.float32) * df_["dt"]).astype(np.float32)
        df_["u_in_cum"] = (
            df_.groupby("breath_id", sort=False)["u_in_dt"].cumsum().astype(np.float32)
        )
        df_["u_in_prev"] = (
            df_.groupby("breath_id", sort=False)["u_in"]
            .shift(1)
            .fillna(0.0)
            .astype(np.float32)
        )
        df_["u_in_cum_prev"] = (
            df_.groupby("breath_id", sort=False)["u_in_cum"]
            .shift(1)
            .fillna(0.0)
            .astype(np.float32)
        )
        df_["u_out_prev"] = (
            df_.groupby("breath_id", sort=False)["u_out"]
            .shift(1)
            .fillna(df_["u_out"])
            .astype(np.int8)
        )
        df_["u_in_dt_prev"] = (
            df_.groupby("breath_id", sort=False)["u_in_dt"]
            .shift(1)
            .fillna(0.0)
            .astype(np.float32)
        )

    n_t_bins = 80
    t_min = float(min(tr_insp["time_step"].min(), te["time_step"].min()))
    t_max = float(max(tr_insp["time_step"].max(), te["time_step"].max()))
    if t_max <= t_min:
        t_max = t_min + 1.0
    t_edges = np.linspace(t_min, t_max, n_t_bins + 1)

    train_tbin = np.digitize(tr_insp["time_step"].values, t_edges, right=False) - 1
    test_tbin = np.digitize(te["time_step"].values, t_edges, right=False) - 1
    train_tbin = np.clip(train_tbin, 0, n_t_bins - 1).astype(np.int16)
    test_tbin = np.clip(test_tbin, 0, n_t_bins - 1).astype(np.int16)

    n_u_bins = 50
    u_vals = tr_insp["u_in"].values.astype(np.float64)
    q = np.linspace(0.0, 1.0, n_u_bins + 1)
    u_edges = np.quantile(u_vals, q)
    u_edges = np.unique(u_edges)
    if u_edges.size < 3:
        u_edges = np.array([0.0, 50.0, 100.0], dtype=np.float64)

    train_ubin = np.digitize(tr_insp["u_in"].values, u_edges, right=False) - 1
    test_ubin = np.digitize(te["u_in"].values, u_edges, right=False) - 1
    train_ubin = np.clip(train_ubin, 0, len(u_edges) - 2).astype(np.int16)
    test_ubin = np.clip(test_ubin, 0, len(u_edges) - 2).astype(np.int16)

    n_cum_bins = 70
    cum_vals = tr_insp["u_in_cum"].values.astype(np.float64)
    cum_edges = np.quantile(cum_vals, np.linspace(0.0, 1.0, n_cum_bins + 1))
    cum_edges = np.unique(cum_edges)
    if cum_edges.size < 3:
        mx = float(np.max(cum_vals)) if cum_vals.size else 1.0
        cum_edges = np.array([0.0, 0.5 * mx, mx + 1e-6], dtype=np.float64)

    train_cbin = np.digitize(tr_insp["u_in_cum"].values, cum_edges, right=False) - 1
    test_cbin = np.digitize(te["u_in_cum"].values, cum_edges, right=False) - 1
    train_cbin = np.clip(train_cbin, 0, len(cum_edges) - 2).astype(np.int16)
    test_cbin = np.clip(test_cbin, 0, len(cum_edges) - 2).astype(np.int16)

    train_pbin = np.digitize(tr_insp["u_in_prev"].values, u_edges, right=False) - 1
    test_pbin = np.digitize(te["u_in_prev"].values, u_edges, right=False) - 1
    train_pbin = np.clip(train_pbin, 0, len(u_edges) - 2).astype(np.int16)
    test_pbin = np.clip(test_pbin, 0, len(u_edges) - 2).astype(np.int16)

    train_cpbin = (
        np.digitize(tr_insp["u_in_cum_prev"].values, cum_edges, right=False) - 1
    )
    test_cpbin = np.digitize(te["u_in_cum_prev"].values, cum_edges, right=False) - 1
    train_cpbin = np.clip(train_cpbin, 0, len(cum_edges) - 2).astype(np.int16)
    test_cpbin = np.clip(test_cpbin, 0, len(cum_edges) - 2).astype(np.int16)

    n_uidt_bins = 40
    uidt_vals = tr_insp["u_in_dt_prev"].values.astype(np.float64)
    uidt_edges = np.quantile(uidt_vals, np.linspace(0.0, 1.0, n_uidt_bins + 1))
    uidt_edges = np.unique(uidt_edges)
    if uidt_edges.size < 3:
        mx = float(np.max(uidt_vals)) if uidt_vals.size else 1.0
        uidt_edges = np.array([0.0, 0.5 * mx, mx + 1e-6], dtype=np.float64)

    train_uidtbin = (
        np.digitize(tr_insp["u_in_dt_prev"].values, uidt_edges, right=False) - 1
    )
    test_uidtbin = np.digitize(te["u_in_dt_prev"].values, uidt_edges, right=False) - 1
    train_uidtbin = np.clip(train_uidtbin, 0, len(uidt_edges) - 2).astype(np.int16)
    test_uidtbin = np.clip(test_uidtbin, 0, len(uidt_edges) - 2).astype(np.int16)

    tmp_train = tr_insp[["R", "C", "u_out", "pressure", "u_out_prev"]].copy()
    tmp_train["t_bin"] = train_tbin
    tmp_train["u_bin"] = train_ubin
    tmp_train["c_bin"] = train_cbin
    tmp_train["p_bin"] = train_pbin
    tmp_train["cp_bin"] = train_cpbin
    tmp_train["uidt_bin"] = train_uidtbin

    grp_hist3 = (
        tmp_train.groupby(
            [
                "R",
                "C",
                "u_out",
                "u_out_prev",
                "t_bin",
                "u_bin",
                "c_bin",
                "cp_bin",
                "p_bin",
                "uidt_bin",
            ],
            sort=False,
        )["pressure"]
        .median()
        .astype(np.float64)
    )

    grp_hist2 = (
        tmp_train.groupby(
            [
                "R",
                "C",
                "u_out",
                "u_out_prev",
                "t_bin",
                "u_bin",
                "c_bin",
                "cp_bin",
                "p_bin",
            ],
            sort=False,
        )["pressure"]
        .median()
        .astype(np.float64)
    )

    grp_hist = (
        tmp_train.groupby(
            ["R", "C", "u_out", "t_bin", "u_bin", "c_bin", "p_bin"], sort=False
        )["pressure"]
        .median()
        .astype(np.float64)
    )
    grp_fine = (
        tmp_train.groupby(["R", "C", "u_out", "t_bin", "u_bin"], sort=False)["pressure"]
        .median()
        .astype(np.float64)
    )
    grp_time = (
        tmp_train.groupby(["R", "C", "u_out", "t_bin"], sort=False)["pressure"]
        .median()
        .astype(np.float64)
    )
    grp_coarse = (
        tmp_train.groupby(["R", "C", "u_out"], sort=False)["pressure"]
        .median()
        .astype(np.float64)
    )

    global_median = float(np.median(tr_insp["pressure"].values))
    global_median_snapped = find_nearest(global_median)

    tmp_test = te[["R", "C", "u_out", "u_out_prev"]].copy()
    tmp_test["t_bin"] = test_tbin
    tmp_test["u_bin"] = test_ubin
    tmp_test["c_bin"] = test_cbin
    tmp_test["p_bin"] = test_pbin
    tmp_test["cp_bin"] = test_cpbin
    tmp_test["uidt_bin"] = test_uidtbin

    idx_hist3 = pd.MultiIndex.from_frame(
        tmp_test[
            [
                "R",
                "C",
                "u_out",
                "u_out_prev",
                "t_bin",
                "u_bin",
                "c_bin",
                "cp_bin",
                "p_bin",
                "uidt_bin",
            ]
        ]
    )
    pred = grp_hist3.reindex(idx_hist3).to_numpy()

    if np.any(pd.isna(pred)):
        idx_hist2 = pd.MultiIndex.from_frame(
            tmp_test[
                [
                    "R",
                    "C",
                    "u_out",
                    "u_out_prev",
                    "t_bin",
                    "u_bin",
                    "c_bin",
                    "cp_bin",
                    "p_bin",
                ]
            ]
        )
        pred_hist2 = grp_hist2.reindex(idx_hist2).to_numpy()
        pred = np.where(pd.isna(pred), pred_hist2, pred)

    if np.any(pd.isna(pred)):
        idx_hist = pd.MultiIndex.from_frame(
            tmp_test[["R", "C", "u_out", "t_bin", "u_bin", "c_bin", "p_bin"]]
        )
        pred_hist = grp_hist.reindex(idx_hist).to_numpy()
        pred = np.where(pd.isna(pred), pred_hist, pred)

    if np.any(pd.isna(pred)):
        idx_fine = pd.MultiIndex.from_frame(
            tmp_test[["R", "C", "u_out", "t_bin", "u_bin"]]
        )
        pred_fine = grp_fine.reindex(idx_fine).to_numpy()
        pred = np.where(pd.isna(pred), pred_fine, pred)

    if np.any(pd.isna(pred)):
        idx_time = pd.MultiIndex.from_frame(tmp_test[["R", "C", "u_out", "t_bin"]])
        pred_time = grp_time.reindex(idx_time).to_numpy()
        pred = np.where(pd.isna(pred), pred_time, pred)

    if np.any(pd.isna(pred)):
        idx_coarse = pd.MultiIndex.from_frame(tmp_test[["R", "C", "u_out"]])
        pred_coarse = grp_coarse.reindex(idx_coarse).to_numpy()
        pred = np.where(pd.isna(pred), pred_coarse, pred)

    pred = np.where(pd.isna(pred), global_median, pred).astype(np.float64)

    u_out_test = te["u_out"].to_numpy().astype(np.int8)
    pred = np.where(u_out_test == 1, float(global_median_snapped), pred)

    insert_idx = np.searchsorted(sorted_pressures, pred, side="left")
    insert_idx = np.clip(insert_idx, 0, total_pressures_len - 1)
    lower_idx = np.clip(insert_idx - 1, 0, total_pressures_len - 1)
    upper_idx = insert_idx

    lower_val = sorted_pressures[lower_idx]
    upper_val = sorted_pressures[upper_idx]
    choose_lower = np.abs(lower_val - pred) < np.abs(upper_val - pred)
    snapped = np.where(choose_lower, lower_val, upper_val).astype(np.float64)

    output["pressure"] = snapped
    output.to_csv("submission.csv", index=False)


def wc(input_list):
    """
    Original intent: load selected submission files and apply a fixed 2-file weighting heuristic.
    Bugfix: handle cases where 0/1 eligible files are present to avoid IndexError.
    """
    l = []
    allow = [1348, 1579]

    loaded_preds = []
    for i in range(len(input_list)):
        public_lb_score = int(input_list[i].split("/")[-1].split(".")[1].split(" ")[0])
        if public_lb_score in allow:
            l.append(public_lb_score)
            loaded_preds.append(pd.read_csv(input_list[i]).pressure.to_numpy().ravel())
        else:
            continue

    if len(loaded_preds) == 0:
        return None

    if len(loaded_preds) == 1:
        return loaded_preds[0]

    l_sum = sum(l[:2])
    if l_sum == 0:
        return 0.5 * (loaded_preds[0] + loaded_preds[1])

    weight1 = (l[1] / l_sum) + 0.1
    weight2 = 1 - weight1
    return loaded_preds[0] * weight1 + loaded_preds[1] * weight2


def g(dp):
    """
    Original intent: find specific high-score submissions in dp, blend them multiple times,
    then median-aggregate and snap to nearest known pressure.
    Bugfix: if dp doesn't exist / contains no eligible files, fall back to a valid baseline submission.
    Improvement toward target: fallback is a conditional-median lookup aligned to inspiratory scoring.
    """
    allow = [1348, 1579]
    l = []
    for i in glob.iglob(f"{dp}/*"):
        try:
            file_lb = int(i.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            continue
        if file_lb in allow:
            l.append(i)

    if len(l) == 0:
        _write_fallback_submission()
        return

    loop_time = 125
    splits = 2
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

    flist = [x for x in flist if x is not None]

    if len(flist) == 0:
        _write_fallback_submission()
        return

    pred_list = []
    for it in range(loop_time):
        weight = []
        set_seed(it)
        for _ in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        if weight_sum == 0:
            weight = [1.0 / len(flist)] * len(flist)
        else:
            for j in range(len(weight)):
                weight[j] /= weight_sum
        weight.sort(reverse=True)

        temp = 0
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()

    output = pd.read_csv(SAMPLE_SUB_PATH)
    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)

    output.to_csv("submission.csv", index=False)




## === cell 2
g("../input/ventilator-pressure-high-score-submissions")
print("Wrote submission.csv")
