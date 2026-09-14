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

0.1533028098228836

# 6. Current score

3.8815

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.0062) has done: 'The current notebook fails because it expects external prediction CSVs under `../input/gb-blending`, but that dataset is not present, so the blending code ends up producing arrays of length 1 and crashes when assigning into the 603600-row submission. I keep the existing “blend/median + snap-to-nearest-pressure” logic intact, but add a safe fallback that generates a baseline prediction from the provided train/test files when no blend files are found. This baseline uses per-(R,C,time_step,u_out) median pressure from train and otherwise backs off to per-(R,C,time_step) and then global per-time_step medians—so it runs end-to-end and produces a valid `submission.csv`. I also fix the undefined `a, b` usage by making that cell optional and non-failing.'
- What this solution (achieved 9.00291) has done: 'Your current score (9.0062 MAE; lower is better) is far from the target (0.1533), so we should improve meaningfully while keeping your “train/test-only fallback + snap-to-nearest-pressure” core approach intact. The biggest issue in the fallback is that it ignores the sequential nature of a breath; adding simple lag features via previous `u_in/u_out` and `time_step` within each `breath_id` and using a slightly richer group-median lookup (still just medians, no model) typically yields a large MAE drop on this competition. I keep your blending logic unchanged, but upgrade only the fallback baseline to use these breath-wise lag keys with safe backoff, and I make sure the submission remains correctly aligned to `id` and written to `submission.csv`. This stays fast (groupby/merge only), deterministic, and preserves evaluation semantics (still snapping to valid pressure levels).'
- What this solution (achieved 3.7753) has done: 'Your current fallback baseline is still too coarse for this competition (MAE ~9), mainly because it ignores the “inspiratory only” scoring mask (`u_out==0`) and doesn’t exploit strong within-breath structure. To move the score sharply toward the 0.153 target while preserving your core “lookup medians + backoff + snap-to-nearest-pressure” approach (no model/training loop changes), I upgrade the fallback to (1) learn medians only on inspiratory rows and (2) use cumulative-within-breath signals (cumulative `u_in`, time index) plus a couple of lagged `u_in` values for more specific keys, with safe backoff. I also enforce the known physical constraint that expiratory phase isn’t scored by simply setting `pressure=0` when `u_out==1`, which typically reduces MAE substantially without changing evaluation semantics. Blending logic and snapping remain intact, and the script still writes a valid `submission.csv` end-to-end.'
- What this solution (achieved 3.77494) has done: 'Your current MAE (3.7753, lower-is-better) is still far from the target (0.1533), so we need a meaningful improvement while keeping your core “median lookup + backoff + snap-to-nearest-pressure” logic intact. The biggest easy win is to stop forcing `pressure=0` for `u_out==1` (that hurts because Kaggle evaluates only inspiratory rows rather than treating expiratory as zero-error), and instead simply predict normally for all rows while still *training the lookups on inspiratory rows only* to match the metric focus. I also add one extra within-breath feature (`u_in_lag2_r`) into the most specific lookup key to better capture dynamics with minimal additional complexity and runtime. Everything else (blending logic, snapping, I/O paths, submission writing) remains the same.'
- What this solution (achieved 3.77486) has done: 'Your current score (3.77494 MAE; lower is better) is still far above the target (0.1533), so we should improve meaningfully while keeping your core “group-median lookup + backoff + snap-to-nearest-pressure” approach unchanged. The largest remaining gain with minimal logic change is to add an additional, very lightweight signal that better captures breath progression: a rounded delta of `u_in` (first difference) within each breath, and include it only in the most-specific median key (with the same safe backoff chain as before). This preserves the exact same overall method (pure lookups/medians, no model), keeps runtime low, and tends to reduce MAE by making the top-level lookup less ambiguous. Everything else (file handling, blending fallback behavior, snapping, and writing `submission.csv`) stays intact.'
- What this solution (achieved 3.75822) has done: 'Your current MAE (3.77486) is still far above the target (0.1533), so we should improve the fallback baseline while preserving your existing “group-median lookup + backoff + snap-to-nearest-pressure” core logic. The biggest low-risk win is to stop learning/predicting “raw pressure” directly and instead learn the *residual above a simple physical baseline* `u_in / C` (per row) using the same median-lookup/backoff chain; this keeps the same semantics (still medians + backoff + snapping) but typically reduces error a lot. Concretely, we compute `res = pressure - u_in/C`, build the same group-median tables on `res` (inspiratory rows only), then predict `pressure = u_in/C + res_pred` and snap to valid pressure levels. Blending code/path handling stays unchanged, and the script still writes a valid `submission.csv` end-to-end.'
- What this solution (achieved 3.75822) has done: 'We need to move MAE down from 3.758 toward 0.153, so we should improve the existing fallback (median-lookup + backoff + snap) without changing the overall approach. The largest remaining low-risk gain is to ensure we predict only on the inspiratory phase correctly and leverage the known discrete pressure grid more strongly by predicting **pressure class** via residual snapping that is conditioned on (R,C,t_idx) rather than a single global snap. Concretely, we keep your residual-above-`u_in/C` idea, but (1) build per-(R,C,t_idx) residual medians and (2) snap using the closest **valid pressure values observed for that (R,C)** pair (backing off to global grid if unseen). This preserves your core logic (pure group medians + backoff + snapping) and typically improves MAE substantially with minimal extra runtime. The blending path behavior and submission writing remain unchanged and still produce `submission.csv` end-to-end.'
- What this solution (achieved 3.89743) has done: 'We need to reduce MAE from 3.758 toward 0.153 (lower is better), so we should strengthen the existing fallback while preserving your core “median lookup + backoff + snap-to-valid-grid” logic. The biggest minimal win is to make the residual baseline a bit more physically appropriate by adding an `R * flow` term using an approximate flow proxy (`u_in`), then learn/smooth only the remaining residual with the same group-median/backoff chain. We keep all I/O paths, keep inspiratory-only rows for fitting the lookup tables, and keep the same (R,C)-conditioned snapping grid. The change is small (one extra baseline term + medians computed on a smaller-magnitude residual) and should move the score down without introducing new modeling/training machinery.'
- What this solution (achieved 3.8815) has done: 'Your current MAE (3.897) is still far above the target (0.153, lower is better), so we should make a small, low-risk improvement inside the existing fallback (median lookup + backoff + snap) rather than changing the overall approach. The biggest issue is that the added “R*u_in” baseline term is not physically well-aligned and likely harms accuracy; I keep the residual-baseline idea but switch to a breath-wise **volume proxy** (`u_in * dt` cumulative) and use `R * flow_proxy` where `flow_proxy ≈ dV/dt`, then fit its coefficient exactly the same way as your current `k` fit (still just one scalar OLS). Everything else stays the same: inspiratory-only rows for fitting tables, same lookup/backoff chain, same (R,C)-conditioned pressure snapping, and the script still produces `submission.csv` end-to-end.'

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




## === cell 2
def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    """
    Read 1-2 submission files and do a simple weighted combination based on the filename "score" token.
    Bugfix: make parsing robust; if score can't be parsed, fall back to equal weights.
    """
    l = []
    preds = []
    for i in range(len(input_list)):
        base = os.path.basename(input_list[i])
        score = None
        try:
            parts = base.split(".")
            if len(parts) >= 2:
                score_token = parts[1].split(" ")[0]
                score = float(score_token)
        except Exception:
            score = None
        l.append(score)
        preds.append(pd.read_csv(input_list[i]).pressure.to_numpy().ravel())

    if len(preds) == 1:
        return preds[0]

    if (l[0] is None) or (l[1] is None) or (l[0] + l[1] == 0):
        weight1, weight2 = 0.5, 0.5
    else:
        l_sum = l[0] + l[1]
        weight1 = (l[1] / l_sum) + 0.1
        weight1 = min(max(weight1, 0.0), 1.0)
        weight2 = 1 - weight1

    return preds[0] * weight1 + preds[1] * weight2


def _fallback_baseline_submission():
    """
    Core logic preserved: (median lookup + backoff + snap-to-valid-pressure-grid).

    Change (expected MAE improvement toward target):
      - Replace the misaligned baseline term k*(R*u_in) with a more physically meaningful proxy:
        k*(R*flow_proxy), where flow_proxy is approximated from a volume proxy V(t)=cumsum(u_in*dt):
          flow_proxy ≈ dV/dt.
        This keeps the exact same residual-learning approach (single scalar k via OLS) while making the
        baseline closer to RC pressure drop behavior, typically reducing residual variance and MAE.
      - Keep inspiratory-only rows (u_out==0) for fitting lookup tables (metric is inspiratory-phase MAE).
      - Keep (R,C)-conditioned snapping grid with global fallback.
    """
    df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
    sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")

    tr = df_train[
        ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    ].copy()
    te = df_test[["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]].copy()

    tr.sort_values(["breath_id", "time_step"], inplace=True)
    te.sort_values(["breath_id", "time_step"], inplace=True)

    tr["t_idx"] = tr.groupby("breath_id").cumcount().astype(np.int16)
    te["t_idx"] = te.groupby("breath_id").cumcount().astype(np.int16)

    tr["u_in_lag1"] = tr.groupby("breath_id")["u_in"].shift(1)
    tr["u_in_lag2"] = tr.groupby("breath_id")["u_in"].shift(2)
    te["u_in_lag1"] = te.groupby("breath_id")["u_in"].shift(1)
    te["u_in_lag2"] = te.groupby("breath_id")["u_in"].shift(2)

    tr["u_in_cum"] = tr.groupby("breath_id")["u_in"].cumsum()
    te["u_in_cum"] = te.groupby("breath_id")["u_in"].cumsum()

    tr["du_in"] = tr.groupby("breath_id")["u_in"].diff()
    te["du_in"] = te.groupby("breath_id")["u_in"].diff()

    for c in ["u_in", "u_in_lag1", "u_in_lag2", "du_in"]:
        tr[c + "_r"] = tr[c].round(1)
        te[c + "_r"] = te[c].round(1)
    tr["u_in_cum_r"] = tr["u_in_cum"].round(1)
    te["u_in_cum_r"] = te["u_in_cum"].round(1)

    tr["base0"] = tr["u_in"] / tr["C"].astype(np.float64)
    te["base0"] = te["u_in"] / te["C"].astype(np.float64)

    tr["dt"] = (
        tr.groupby("breath_id")["time_step"].diff().fillna(0.0).astype(np.float64)
    )
    te["dt"] = (
        te.groupby("breath_id")["time_step"].diff().fillna(0.0).astype(np.float64)
    )

    tr["V"] = (
        (tr["u_in"].astype(np.float64) * tr["dt"]).groupby(tr["breath_id"]).cumsum()
    )
    te["V"] = (
        (te["u_in"].astype(np.float64) * te["dt"]).groupby(te["breath_id"]).cumsum()
    )

    tr["flow_proxy"] = tr.groupby("breath_id")["V"].diff() / tr["dt"].replace(
        0.0, np.nan
    )
    te["flow_proxy"] = te.groupby("breath_id")["V"].diff() / te["dt"].replace(
        0.0, np.nan
    )
    tr["flow_proxy"] = tr["flow_proxy"].fillna(0.0).astype(np.float64)
    te["flow_proxy"] = te["flow_proxy"].fillna(0.0).astype(np.float64)

    tri_fit = tr[tr["u_out"] == 0].copy()

    x = (
        tri_fit["R"].astype(np.float64) * tri_fit["flow_proxy"].astype(np.float64)
    ).to_numpy()
    y = (
        tri_fit["pressure"].astype(np.float64) - tri_fit["base0"].astype(np.float64)
    ).to_numpy()

    denom = float(np.dot(x, x))
    if denom > 0.0:
        k = float(np.dot(x, y) / denom)
    else:
        k = 0.0

    k = float(np.clip(k, -0.2, 0.2))

    tr["base"] = tr["base0"] + k * (
        tr["R"].astype(np.float64) * tr["flow_proxy"].astype(np.float64)
    )
    te["base"] = te["base0"] + k * (
        te["R"].astype(np.float64) * te["flow_proxy"].astype(np.float64)
    )

    tr["res"] = tr["pressure"] - tr["base"]

    tri = tr[tr["u_out"] == 0].copy()

    g0 = (
        tri.groupby(
            [
                "R",
                "C",
                "t_idx",
                "u_in_r",
                "u_in_lag1_r",
                "u_in_lag2_r",
                "u_in_cum_r",
                "du_in_r",
            ],
            sort=False,
        )["res"]
        .median()
        .rename("r0")
        .reset_index()
    )

    g1 = (
        tri.groupby(["R", "C", "t_idx", "u_in_r", "u_in_cum_r"], sort=False)["res"]
        .median()
        .rename("r1")
        .reset_index()
    )
    g2 = (
        tri.groupby(["R", "C", "t_idx", "u_in_r"], sort=False)["res"]
        .median()
        .rename("r2")
        .reset_index()
    )
    g3 = (
        tri.groupby(["R", "C", "t_idx"], sort=False)["res"]
        .median()
        .rename("r3")
        .reset_index()
    )
    g4 = tri.groupby(["t_idx"], sort=False)["res"].median().rename("r4").reset_index()

    m = (
        te[
            [
                "id",
                "breath_id",
                "R",
                "C",
                "t_idx",
                "u_out",
                "u_in_r",
                "u_in_lag1_r",
                "u_in_lag2_r",
                "u_in_cum_r",
                "du_in_r",
                "base",
            ]
        ]
        .merge(
            g0,
            on=[
                "R",
                "C",
                "t_idx",
                "u_in_r",
                "u_in_lag1_r",
                "u_in_lag2_r",
                "u_in_cum_r",
                "du_in_r",
            ],
            how="left",
        )
        .merge(
            g1,
            on=["R", "C", "t_idx", "u_in_r", "u_in_cum_r"],
            how="left",
        )
        .merge(
            g2,
            on=["R", "C", "t_idx", "u_in_r"],
            how="left",
        )
        .merge(
            g3,
            on=["R", "C", "t_idx"],
            how="left",
        )
        .merge(
            g4,
            on=["t_idx"],
            how="left",
        )
    )

    res_pred = m["r0"]
    res_pred = res_pred.fillna(m["r1"])
    res_pred = res_pred.fillna(m["r2"])
    res_pred = res_pred.fillna(m["r3"])
    res_pred = res_pred.fillna(m["r4"])
    res_pred = res_pred.fillna(tri["res"].median())

    pred = (m["base"] + res_pred).to_numpy(dtype=np.float64)

    rc_to_grid = (
        df_train.groupby(["R", "C"])["pressure"]
        .unique()
        .apply(lambda x: np.sort(np.asarray(x, dtype=np.float64)))
        .to_dict()
    )

    def find_nearest_in_grid(prediction, grid):
        insert_idx = np.searchsorted(grid, prediction)
        if insert_idx == grid.shape[0]:
            return float(grid[-1])
        if insert_idx == 0:
            return float(grid[0])
        lo = grid[insert_idx - 1]
        hi = grid[insert_idx]
        return float(lo if abs(lo - prediction) < abs(hi - prediction) else hi)

    snapped = np.empty_like(pred, dtype=np.float64)
    R_arr = m["R"].to_numpy()
    C_arr = m["C"].to_numpy()
    for i in range(pred.shape[0]):
        grid = rc_to_grid.get((int(R_arr[i]), int(C_arr[i])), sorted_pressures)
        snapped[i] = find_nearest_in_grid(pred[i], grid)

    pred_by_id = pd.DataFrame({"id": m["id"].to_numpy(), "pressure": snapped})
    pred_by_id.sort_values("id", inplace=True)

    sub = sub.merge(pred_by_id, on="id", how="left", suffixes=("", "_pred"))
    sub["pressure"] = sub["pressure_pred"]
    sub.drop(columns=["pressure_pred"], inplace=True)

    sub.to_csv("submission.csv", index=False)
    return sub


def g(dp):
    """
    Original intent: read multiple submission files under dp, ensemble them with random weights,
    take median of ensembles, snap to nearest valid pressure, write CSV.

    Bugfix: handle missing/empty dp and ensure all predictions have correct length before stacking.
    If no valid files exist, fall back to a train/test-only baseline to still yield a submission.
    """
    files = sorted([p for p in glob.iglob(f"{dp}/*") if os.path.isfile(p)])
    if len(files) == 0:
        print(
            f"[g] No files found in {dp}. Creating fallback baseline submission.csv instead."
        )
        return _fallback_baseline_submission()

    file_count = len(files)
    loop_time = 100

    splits = max(1, file_count // 2)
    flist = []
    for i in range(splits):
        start = i * round(len(files) / splits)
        end = None if i == splits - 1 else (i + 1) * round(len(files) / splits)
        chunk = files[start:end]
        if len(chunk) > 0:
            flist.append(chunk)

    flist = [wc(chunk) for chunk in flist]

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    n = len(output)
    good = []
    for arr in flist:
        arr = np.asarray(arr).ravel()
        if arr.shape[0] == n:
            good.append(arr)

    if len(good) == 0:
        print(
            f"[g] No valid prediction files with length {n} in {dp}. Using fallback baseline."
        )
        return _fallback_baseline_submission()

    pred_list = []
    for seed in range(loop_time):
        set_seed(seed)
        weight = [rd() for _ in range(len(good))]
        weight_sum = sum(weight)
        weight = [w / weight_sum for w in weight]
        weight.sort(reverse=True)

        temp = np.zeros(n, dtype=np.float64)
        for j in range(len(good)):
            temp += good[j] * weight[j]
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
    a.pressure = a.pressure * 0.55 + b.pressure * 0.45
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 3
g("../input/gb-blending")



## === cell 4
a = None
b = None
if (
    isinstance(a, str)
    and isinstance(b, str)
    and os.path.exists(a)
    and os.path.exists(b)
):
    blend(a, b)
else:
    print("[blend] Skipped because a/b are not set to existing file paths.")
