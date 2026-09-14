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

0.1420181219423538

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 10.86378) has done: 'The crash happens because the script expects to find many prediction CSVs in `../input/gb-data-blending-recover`, but in your environment that folder is missing/empty, so the blending code ends up producing a single scalar instead of a 603600-length vector. I fix this by (1) making the blender robust to empty/missing directories and variable file counts, and (2) adding a safe fallback that produces a valid baseline submission by predicting the median training pressure (then snapping to the nearest allowed pressure), so you always get a `.csv` submission. I also update paths to use the provided `/kaggle/input/...` structure and ensure we always write `submission.csv`. Core logic (median of weighted blends + nearest pressure snapping) is preserved; the only “modeling” addition is a baseline fallback to ensure a valid submission when external blend files aren’t available.'
- What this solution (achieved 7.20701) has done: 'Your current score is far worse than the target (lower-is-better), and the main reason is that the blender falls back to a constant median-pressure submission because the external blend directory is missing—this produces a very poor MAE. To move toward the target with minimal change and without changing the “blending + nearest pressure snapping” core, I keep the existing blending logic but add a small, deterministic, physics-inspired baseline generator (using per-(R,C) mapping from training and time-step–aligned mean inspiratory pressure). This baseline is only used when no valid external prediction CSVs are found, and it still snaps predictions to the nearest allowed pressure values as before. The result should drastically reduce MAE versus the constant baseline and move the score much closer to your target.'
- What this solution (achieved 6.10591) has done: 'Your current score is far above the target (lower-is-better), and the main culprit is the fallback baseline: it uses only inspiratory (u_out==0) mean pressure and then applies those pressures to *all* test rows, including expiratory rows (u_out==1), which are typically near-zero pressure—this miscalibration heavily hurts MAE when snapped to discrete pressures. I keep your blending core unchanged, but improve the fallback baseline to be phase-aware by learning separate (R,C,time_step) means for u_out==0 and u_out==1, then mapping test rows using their own u_out value. I also make the time_step join robust to float representation by rounding time_step to 2 decimals in both train/test (the dataset is on a 0.03 grid), which prevents accidental missing merges that currently force weaker fallbacks. These are minimal changes limited to the fallback generator and should move the score substantially toward your target without altering the blending logic.'
- What this solution (achieved 6.19023) has done: 'Your current score is far worse than the target (lower-is-better), and since the external blending directory is missing you’re always using the fallback baseline; the only way to move toward the target is to make that fallback more faithful to the scoring setup. I keep the overall “baseline-from-train-means + snap-to-nearest-pressure + write submission.csv” core intact, but fix two baseline issues that strongly inflate MAE: (1) learn means only on inspiratory rows (u_out==0), because expiratory rows are not scored and mixing them distorts the learned mapping, and (2) align each time step within a breath by using an integer `step` index (0..79) instead of float `time_step` joins, which avoids merge misses and phase misalignment. Everything else (blending behavior when files exist, nearest-pressure snapping, output schema/path) stays the same.'
- What this solution (achieved 4.51352) has done: 'Your score is far above the target (lower-is-better) because you’re always hitting the fallback baseline (the blend directory is missing) and that baseline currently ignores key dynamics from `u_in` and breath progression. I keep the same core fallback structure (learn train means → map onto test → snap to nearest allowed pressure), but make the learned mapping depend on `u_in` as well as `(R,C,step)`, which is a minimal, directly relevant improvement for this competition. To keep it stable and avoid sparse-group NaNs, I use a small number of `u_in` bins and back off through a hierarchy: `(R,C,step,u_in_bin)` → `(R,C,step)` → `(R,C,u_in_bin)` → `(R,C)` → global median. Everything else (blending behavior, snapping, output filename/columns) stays the same.'
- What this solution (achieved 4.51352) has done: 'Your current score (4.51 MAE) is still far above the target (0.142), and since the external blend folder is missing you always use the fallback baseline—so the only way to move toward the target is to improve that fallback while keeping the same “train-means lookup → merge onto test → snap to nearest allowed pressure” core. The biggest remaining gap is that the baseline learns only inspiratory behavior but then fills expiratory rows with an (R,C) average, which can be badly wrong when u_out=1; even though expiratory isn’t scored, wrong pressures there still can’t help and can hurt overall alignment if anything is off. I change the fallback to be explicitly phase-aware by learning separate mappings for u_out==0 and u_out==1 (still using the same step and u_in_bin hierarchy), and then apply the appropriate mapping per test row. This is a minimal, directly score-relevant adjustment that should reduce MAE substantially without changing your blending logic or snapping semantics.'
- What this solution (achieved 3.60745) has done: 'Your current score is far above the target (lower-is-better), and since the external blend folder is missing you always use the fallback baseline—so improving that fallback is the only lever without changing the blending core. The biggest gain with minimal semantic change is to make the fallback mapping more “sequence-aware” by incorporating the previous control/phase state, which strongly influences pressure dynamics in this dataset. I keep the exact same lookup→hierarchical backoff→nearest-pressure snapping structure, but add `u_in_bin_prev` and `u_out_prev` (computed within breath via shift) to the most-specific group, then back off to your existing groups when sparse. This should materially reduce MAE and move the score closer to the 0.142 target while preserving the overall approach.'

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
TRAIN_PATH = "/kaggle/input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "/kaggle/input/ventilator-pressure-prediction/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv"

df_train_pressure = pd.read_csv(TRAIN_PATH, usecols=["pressure"])
unique_pressures = df_train_pressure["pressure"].unique()
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


def _read_pressure_vector(csv_path: str, n_expected: int) -> np.ndarray | None:
    """Read a submission-like CSV and return pressure vector if valid, else None."""
    try:
        df = pd.read_csv(csv_path, usecols=["pressure"])
    except Exception:
        return None
    arr = df["pressure"].to_numpy()
    if arr.shape[0] != n_expected:
        return None
    return arr


def wc(input_list):
    """
    Original intent: take 1 or 2 files, parse an LB-score-like number from filename, then weighted combine.
    Bug fix: make filename score parsing optional; if parsing fails, fall back to equal weights.
    """
    vectors = []
    parsed_scores = []
    for p in input_list:
        score_val = None
        try:
            score_val = int(p.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            score_val = None
        parsed_scores.append(score_val)
        vectors.append(pd.read_csv(p, usecols=["pressure"]).pressure.to_numpy().ravel())

    if len(vectors) == 0:
        return None
    if len(vectors) == 1:
        return vectors[0]

    if parsed_scores[0] is not None and parsed_scores[1] is not None:
        l_sum = parsed_scores[0] + parsed_scores[1]
        if l_sum == 0:
            w1 = 0.5
        else:
            w1 = (parsed_scores[1] / l_sum) + 0.1
        w2 = 1 - w1
    else:
        w1, w2 = 0.5, 0.5

    return vectors[0] * w1 + vectors[1] * w2


def _baseline_from_train_means() -> pd.DataFrame:
    """
    Fallback baseline (used when no blend CSVs exist).

    Change (score-improving, minimal, preserves core semantics):
    - Keep the same "train means lookup -> merge onto test -> hierarchical backoff -> snap to nearest pressure".
    - Add two lightweight physics-proxy features that are strongly predictive in this competition:
        * u_eff = u_in * (1 - u_out)  (effective inspiratory flow command)
        * u_in_cum = cumulative sum of u_eff within breath (proxy for inspired volume)
      Then condition the most-specific mean on binned versions of these features, with backoff to the
      existing (R,C,step,u_in_bin,prev) hierarchy when sparse.
    """
    UIN_BINS = 25
    UEFF_BINS = 25
    UCUM_BINS = 40

    usecols = ["breath_id", "R", "C", "u_out", "u_in", "pressure"]
    tr = pd.read_csv(TRAIN_PATH, usecols=usecols)

    tr["step"] = tr.groupby("breath_id", sort=False).cumcount().astype(np.int16)

    tr["u_in_bin"] = np.floor(
        tr["u_in"].to_numpy(dtype=np.float64) / (100.0 / UIN_BINS)
    ).astype(np.int16)
    tr["u_in_bin"] = tr["u_in_bin"].clip(0, UIN_BINS - 1)

    tr["u_in_bin_prev"] = (
        tr.groupby("breath_id", sort=False)["u_in_bin"].shift(1).fillna(tr["u_in_bin"])
    ).astype(np.int16)
    tr["u_out_prev"] = (
        tr.groupby("breath_id", sort=False)["u_out"].shift(1).fillna(tr["u_out"])
    ).astype(np.int8)

    u_eff = tr["u_in"].to_numpy(dtype=np.float64) * (
        1.0 - tr["u_out"].to_numpy(dtype=np.float64)
    )
    tr["u_eff_bin"] = np.floor(u_eff / (100.0 / UEFF_BINS)).astype(np.int16)
    tr["u_eff_bin"] = tr["u_eff_bin"].clip(0, UEFF_BINS - 1)

    tr["u_in_cum"] = (
        tr.groupby("breath_id", sort=False)
        .apply(
            lambda g: (
                g["u_in"].to_numpy(dtype=np.float64)
                * (1.0 - g["u_out"].to_numpy(dtype=np.float64))
            ).cumsum()
        )
        .reset_index(level=0, drop=True)
        .astype(np.float32)
    )

    cum_edges = np.quantile(
        tr["u_in_cum"].to_numpy(dtype=np.float64), np.linspace(0, 1, UCUM_BINS + 1)
    )
    cum_edges[0] = -1e9
    cum_edges[-1] = 1e9
    tr["u_in_cum_bin"] = (
        np.searchsorted(
            cum_edges, tr["u_in_cum"].to_numpy(dtype=np.float64), side="right"
        )
        - 1
    ).astype(np.int16)
    tr["u_in_cum_bin"] = tr["u_in_cum_bin"].clip(0, UCUM_BINS - 1)

    tr_insp = tr[tr["u_out"] == 0].copy()
    tr_exp = tr[tr["u_out"] == 1].copy()

    mean_insp_rc_step_dyn = (
        tr_insp.groupby(
            [
                "R",
                "C",
                "step",
                "u_in_bin",
                "u_in_bin_prev",
                "u_out_prev",
                "u_eff_bin",
                "u_in_cum_bin",
            ],
            sort=False,
        )["pressure"]
        .mean()
        .rename("p_insp_rc_step_dyn")
        .reset_index()
    )
    mean_exp_rc_step_dyn = (
        tr_exp.groupby(
            [
                "R",
                "C",
                "step",
                "u_in_bin",
                "u_in_bin_prev",
                "u_out_prev",
                "u_eff_bin",
                "u_in_cum_bin",
            ],
            sort=False,
        )["pressure"]
        .mean()
        .rename("p_exp_rc_step_dyn")
        .reset_index()
    )

    mean_insp_rc_step_u_prev = (
        tr_insp.groupby(
            ["R", "C", "step", "u_in_bin", "u_in_bin_prev", "u_out_prev"], sort=False
        )["pressure"]
        .mean()
        .rename("p_insp_rc_step_u_prev")
        .reset_index()
    )
    mean_exp_rc_step_u_prev = (
        tr_exp.groupby(
            ["R", "C", "step", "u_in_bin", "u_in_bin_prev", "u_out_prev"], sort=False
        )["pressure"]
        .mean()
        .rename("p_exp_rc_step_u_prev")
        .reset_index()
    )

    mean_insp_rc_step_u = (
        tr_insp.groupby(["R", "C", "step", "u_in_bin"], sort=False)["pressure"]
        .mean()
        .rename("p_insp_rc_step_u")
        .reset_index()
    )
    mean_insp_rc_step = (
        tr_insp.groupby(["R", "C", "step"], sort=False)["pressure"]
        .mean()
        .rename("p_insp_rc_step")
        .reset_index()
    )
    mean_insp_rc_u = (
        tr_insp.groupby(["R", "C", "u_in_bin"], sort=False)["pressure"]
        .mean()
        .rename("p_insp_rc_u")
        .reset_index()
    )
    mean_insp_rc = (
        tr_insp.groupby(["R", "C"], sort=False)["pressure"]
        .mean()
        .rename("p_insp_rc")
        .reset_index()
    )
    global_med_insp = float(tr_insp["pressure"].median())

    mean_exp_rc_step_u = (
        tr_exp.groupby(["R", "C", "step", "u_in_bin"], sort=False)["pressure"]
        .mean()
        .rename("p_exp_rc_step_u")
        .reset_index()
    )
    mean_exp_rc_step = (
        tr_exp.groupby(["R", "C", "step"], sort=False)["pressure"]
        .mean()
        .rename("p_exp_rc_step")
        .reset_index()
    )
    mean_exp_rc_u = (
        tr_exp.groupby(["R", "C", "u_in_bin"], sort=False)["pressure"]
        .mean()
        .rename("p_exp_rc_u")
        .reset_index()
    )
    mean_exp_rc = (
        tr_exp.groupby(["R", "C"], sort=False)["pressure"]
        .mean()
        .rename("p_exp_rc")
        .reset_index()
    )
    global_med_exp = (
        float(tr_exp["pressure"].median()) if len(tr_exp) else global_med_insp
    )
    if np.isnan(global_med_exp):
        global_med_exp = global_med_insp

    del tr, tr_insp, tr_exp
    gc.collect()

    te = pd.read_csv(TEST_PATH, usecols=["id", "breath_id", "R", "C", "u_out", "u_in"])
    te["step"] = te.groupby("breath_id", sort=False).cumcount().astype(np.int16)

    te["u_in_bin"] = np.floor(
        te["u_in"].to_numpy(dtype=np.float64) / (100.0 / UIN_BINS)
    ).astype(np.int16)
    te["u_in_bin"] = te["u_in_bin"].clip(0, UIN_BINS - 1)

    te["u_in_bin_prev"] = (
        te.groupby("breath_id", sort=False)["u_in_bin"].shift(1).fillna(te["u_in_bin"])
    ).astype(np.int16)
    te["u_out_prev"] = (
        te.groupby("breath_id", sort=False)["u_out"].shift(1).fillna(te["u_out"])
    ).astype(np.int8)

    u_eff_te = te["u_in"].to_numpy(dtype=np.float64) * (
        1.0 - te["u_out"].to_numpy(dtype=np.float64)
    )
    te["u_eff_bin"] = np.floor(u_eff_te / (100.0 / UEFF_BINS)).astype(np.int16)
    te["u_eff_bin"] = te["u_eff_bin"].clip(0, UEFF_BINS - 1)

    te["u_in_cum"] = (
        te.groupby("breath_id", sort=False)
        .apply(
            lambda g: (
                g["u_in"].to_numpy(dtype=np.float64)
                * (1.0 - g["u_out"].to_numpy(dtype=np.float64))
            ).cumsum()
        )
        .reset_index(level=0, drop=True)
        .astype(np.float32)
    )

    te["u_in_cum_bin"] = (
        np.searchsorted(
            cum_edges, te["u_in_cum"].to_numpy(dtype=np.float64), side="right"
        )
        - 1
    ).astype(np.int16)
    te["u_in_cum_bin"] = te["u_in_cum_bin"].clip(0, UCUM_BINS - 1)

    te2 = te.merge(
        mean_insp_rc_step_dyn,
        on=[
            "R",
            "C",
            "step",
            "u_in_bin",
            "u_in_bin_prev",
            "u_out_prev",
            "u_eff_bin",
            "u_in_cum_bin",
        ],
        how="left",
    )
    te2 = te2.merge(
        mean_insp_rc_step_u_prev,
        on=["R", "C", "step", "u_in_bin", "u_in_bin_prev", "u_out_prev"],
        how="left",
    )
    te2 = te2.merge(mean_insp_rc_step_u, on=["R", "C", "step", "u_in_bin"], how="left")
    te2 = te2.merge(mean_insp_rc_step, on=["R", "C", "step"], how="left")
    te2 = te2.merge(mean_insp_rc_u, on=["R", "C", "u_in_bin"], how="left")
    te2 = te2.merge(mean_insp_rc, on=["R", "C"], how="left")

    te2 = te2.merge(
        mean_exp_rc_step_dyn,
        on=[
            "R",
            "C",
            "step",
            "u_in_bin",
            "u_in_bin_prev",
            "u_out_prev",
            "u_eff_bin",
            "u_in_cum_bin",
        ],
        how="left",
    )
    te2 = te2.merge(
        mean_exp_rc_step_u_prev,
        on=["R", "C", "step", "u_in_bin", "u_in_bin_prev", "u_out_prev"],
        how="left",
    )
    te2 = te2.merge(mean_exp_rc_step_u, on=["R", "C", "step", "u_in_bin"], how="left")
    te2 = te2.merge(mean_exp_rc_step, on=["R", "C", "step"], how="left")
    te2 = te2.merge(mean_exp_rc_u, on=["R", "C", "u_in_bin"], how="left")
    te2 = te2.merge(mean_exp_rc, on=["R", "C"], how="left")

    pred = np.empty(len(te2), dtype=np.float64)
    exp_mask = te2["u_out"].to_numpy() == 1
    insp_mask = ~exp_mask

    pred_insp = te2.loc[insp_mask, "p_insp_rc_step_dyn"].to_numpy(dtype=np.float64)
    m = np.isnan(pred_insp)
    if m.any():
        pred_insp[m] = te2.loc[insp_mask, "p_insp_rc_step_u_prev"].to_numpy(
            dtype=np.float64
        )[m]
    m = np.isnan(pred_insp)
    if m.any():
        pred_insp[m] = te2.loc[insp_mask, "p_insp_rc_step_u"].to_numpy(
            dtype=np.float64
        )[m]
    m = np.isnan(pred_insp)
    if m.any():
        pred_insp[m] = te2.loc[insp_mask, "p_insp_rc_step"].to_numpy(dtype=np.float64)[
            m
        ]
    m = np.isnan(pred_insp)
    if m.any():
        pred_insp[m] = te2.loc[insp_mask, "p_insp_rc_u"].to_numpy(dtype=np.float64)[m]
    m = np.isnan(pred_insp)
    if m.any():
        pred_insp[m] = te2.loc[insp_mask, "p_insp_rc"].to_numpy(dtype=np.float64)[m]
    m = np.isnan(pred_insp)
    if m.any():
        pred_insp[m] = global_med_insp

    pred_exp = te2.loc[exp_mask, "p_exp_rc_step_dyn"].to_numpy(dtype=np.float64)
    m = np.isnan(pred_exp)
    if m.any():
        pred_exp[m] = te2.loc[exp_mask, "p_exp_rc_step_u_prev"].to_numpy(
            dtype=np.float64
        )[m]
    m = np.isnan(pred_exp)
    if m.any():
        pred_exp[m] = te2.loc[exp_mask, "p_exp_rc_step_u"].to_numpy(dtype=np.float64)[m]
    m = np.isnan(pred_exp)
    if m.any():
        pred_exp[m] = te2.loc[exp_mask, "p_exp_rc_step"].to_numpy(dtype=np.float64)[m]
    m = np.isnan(pred_exp)
    if m.any():
        pred_exp[m] = te2.loc[exp_mask, "p_exp_rc_u"].to_numpy(dtype=np.float64)[m]
    m = np.isnan(pred_exp)
    if m.any():
        pred_exp[m] = te2.loc[exp_mask, "p_exp_rc"].to_numpy(dtype=np.float64)[m]
    m = np.isnan(pred_exp)
    if m.any():
        pred_exp[m] = global_med_exp

    pred[insp_mask] = pred_insp
    pred[exp_mask] = pred_exp

    out = te[["id"]].copy()
    out["pressure"] = pred
    out["pressure"] = out["pressure"].map(find_nearest).astype(np.float32)
    return out


def g(dp: str):
    output = pd.read_csv(SAMPLE_SUB_PATH)
    n_expected = len(output)

    file_paths = sorted(
        [p for p in glob.iglob(f"{dp}/*") if p.lower().endswith(".csv")]
    )

    if len(file_paths) == 0:
        output = _baseline_from_train_means()
        output.to_csv("submission.csv", index=False)
        return output

    file_count = len(file_paths)
    splits = max(1, file_count // 2)

    flist = []
    chunk_size = int(round(len(file_paths) / splits)) if splits > 0 else len(file_paths)
    for i in range(splits):
        start = i * chunk_size
        end = None if i == splits - 1 else (i + 1) * chunk_size
        chunk = file_paths[start:end]
        if len(chunk) > 0:
            flist.append(chunk)

    vecs = []
    for chunk in flist:
        v = wc(chunk)
        if v is None:
            continue
        if len(v) != n_expected:
            continue
        vecs.append(v)

    if len(vecs) == 0:
        output = _baseline_from_train_means()
        output.to_csv("submission.csv", index=False)
        return output

    loop_time = 154  # keep original
    pred_list = []
    for t in range(loop_time):
        weight = []
        set_seed(t)
        for _ in range(len(vecs)):
            weight.append(rd())
        weight_sum = sum(weight)
        if weight_sum == 0:
            weight = [1.0 / len(vecs)] * len(vecs)
        else:
            weight = [w / weight_sum for w in weight]
        weight.sort(reverse=True)

        temp = np.zeros(n_expected, dtype=np.float64)
        for i in range(len(vecs)):
            temp += vecs[i] * weight[i]
        pred_list.append(temp)
        del temp
        gc.collect()

    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv("submission.csv", index=False)
    return output


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a["pressure"] = a["pressure"] * 0.5 + b["pressure"] * 0.5
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
_ = g("/kaggle/input/gb-data-blending-recover")
print("Wrote submission.csv with rows:", len(pd.read_csv("submission.csv")))
print(pd.read_csv("submission.csv").head())

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
TypeError: only length-1 arrays can be converted to Python scalars

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/85282514.py in <cell line: 0>()
----> 1 _ = g("/kaggle/input/gb-data-blending-recover")
      2 print("Wrote submission.csv with rows:", len(pd.read_csv("submission.csv")))
      3 print(pd.read_csv("submission.csv").head())

/tmp/ipykernel_11/3240409441.py in g(dp)
    423 
    424     if len(file_paths) == 0:
--> 425         output = _baseline_from_train_means()
    426         output.to_csv("submission.csv", index=False)
    427         return output

/tmp/ipykernel_11/3240409441.py in _baseline_from_train_means()
    128         )
    129         .reset_index(level=0, drop=True)
--> 130         .astype(np.float32)
    131     )
    132 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in astype(self, dtype, copy, errors)
   6641         else:
   6642             # else, only a single dtype is given
-> 6643             new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
   6644             res = self._constructor_from_mgr(new_data, axes=new_data.axes)
   6645             return res.__finalize__(self, method="astype")

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in astype(self, dtype, copy, errors)
    428             copy = False
    429 
--> 430         return self.apply(
    431             "astype",
    432             dtype=dtype,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in apply(self, f, align_keys, **kwargs)
    361                 applied = b.apply(f, **kwargs)
    362             else:
--> 363                 applied = getattr(b, f)(**kwargs)
    364             result_blocks = extend_blocks(applied, result_blocks)
    365 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in astype(self, dtype, copy, errors, using_cow, squeeze)
    756             values = values[0, :]  # type: ignore[call-overload]
    757 
--> 758         new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
    759 
    760         new_values = maybe_coerce_values(new_values)

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array_safe(values, dtype, copy, errors)
    235 
    236     try:
--> 237         new_values = astype_array(values, dtype, copy=copy)
    238     except (ValueError, TypeError):
    239         # e.g. _astype_nansafe can fail on object-dtype of strings

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array(values, dtype, copy)
    180 
    181     else:
--> 182         values = _astype_nansafe(values, dtype, copy=copy)
    183 
    184     # in pandas we don't store numpy str dtypes, so convert to object

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_nansafe(arr, dtype, copy, skipna)
    131     if copy or arr.dtype == object or dtype == object:
    132         # Explicit copy, or required since NumPy can't view from / to object.
--> 133         return arr.astype(dtype, copy=True)
    134 
    135     return arr.astype(dtype, copy=copy)

ValueError: setting an array element with a sequence.
