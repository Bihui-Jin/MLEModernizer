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

0.1432423129282703

# 6. Current score

2.25585

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.92111) has done: 'Your current code fails because it expects a directory of external “high-score submissions” that is not present in this Kaggle environment, causing empty lists and an `IndexError` during blending. To keep the core “blend multiple submissions then snap to nearest valid pressure” logic intact while making it runnable end-to-end, I add a safe fallback: if no eligible submission files are found, train a simple per-(R,C,time_step,u_out) median lookup model from `train.csv` and predict for `test.csv`. This is score-reasonable (not random), uses only provided data, and still outputs `id,pressure` with valid snapping via your existing `find_nearest`. Finally, I ensure the notebook writes a `.csv` submission file to the working directory.'
- What this solution (achieved 9.88149) has done: 'Your current fallback median-lookup is valid but far too weak for this competition, which is why the MAE is ~9.92 vs your target ~0.143 (lower is better). To move substantially toward the target without changing the “no ML model / pure pandas aggregation + snapping to valid pressures” core approach, I upgrade the fallback to a breath-aware physical proxy: compute cumulative inspired volume (`u_in` integrated over time) per breath and learn a per-(R,C,time_step,u_out) median mapping from that state to pressure, with safe backoffs. This keeps the solution as a deterministic lookup/regression-by-aggregation and retains your existing `find_nearest` snapping and submission writing. The blending path is kept intact; it only run if those external submissions actually exist.'
- What this solution (achieved 1.43592) has done: 'Your current MAE (~9.88) is far worse than the target (~0.143, lower is better), so we need a meaningful but still “pure pandas aggregation + snapping” upgrade in the fallback path (since the external high-score submissions directory isn’t available). I keep your overall logic and snapping intact, but strengthen the fallback by learning the median pressure from a more informative per-breath state: cumulative inspired volume plus short history of `u_in` and `u_out` (lags), which better captures dynamics without introducing any ML training loops. I also make `time_step` joinable by converting it to a stable integer index per breath (avoids float-key mismatches that can silently destroy lookup quality). These are minimal, deterministic changes aimed specifically at reducing MAE while preserving your approach and still writing `submission.csv`.'
- What this solution (achieved 2.25585) has done: 'Main bottlenecks are (a) the Monte-Carlo ensembling loop in `g()` (125 iterations building large `pred_list` and then `vstack`), (b) per-row pandas `reindex` inside nested Python loops in `_fallback_model_predict`, and (c) `output["pressure"].apply(find_nearest)` which is a slow Python loop over 603,600 rows. The optimized version keeps the same core semantics (same blending logic + same deterministic fallback model and nearest-pressure snapping), but replaces slow Python/pandas per-row operations with vectorized NumPy operations and a per-breath array loop that only does O(80) work per breath. It also computes the ensemble median in a streaming way without storing 125 full prediction arrays, and it snaps predictions to the nearest allowed pressure using a vectorized `searchsorted`-based routine (equivalent to `find_nearest` for all values). These changes remove the dominant overhead while preserving outputs up to negligible floating-point differences.'

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

TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SAMPLE_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"

train_dtypes = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
usecols_train = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
df_train = pd.read_csv(TRAIN_PATH, usecols=usecols_train, dtype=train_dtypes)

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)


def find_nearest(prediction):
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        return sorted_pressures[-1]
    elif insert_idx == 0:
        return sorted_pressures[0]
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return (
        lower_val
        if abs(lower_val - prediction) < abs(upper_val - prediction)
        else upper_val
    )


def snap_to_nearest_pressure_vec(preds: np.ndarray) -> np.ndarray:
    preds = np.asarray(preds, dtype=np.float32)
    idx = np.searchsorted(sorted_pressures, preds, side="left")
    idx = idx.astype(np.int32)
    idx0 = np.clip(idx - 1, 0, total_pressures_len - 1)
    idx1 = np.clip(idx, 0, total_pressures_len - 1)
    lower = sorted_pressures[idx0]
    upper = sorted_pressures[idx1]
    choose_lower = np.abs(lower - preds) < np.abs(upper - preds)
    out = np.where(choose_lower, lower, upper).astype(np.float32)
    return out


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    """
    Blend a list of submission filepaths.
    Original code assumed exactly 2 items and specific filename patterns; we keep the intent
    but make it robust to 0/1/N files so it doesn't crash.
    """
    preds = []
    scores = []
    allow = [1348, 1358, 1758]

    for fp in input_list:
        try:
            public_lb_score = int(fp.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            continue

        if public_lb_score in allow:
            scores.append(public_lb_score)
            preds.append(
                pd.read_csv(fp, usecols=["pressure"], dtype={"pressure": "float32"})
                .pressure.to_numpy()
                .ravel()
            )

    if len(preds) == 0:
        return None

    if len(preds) == 1:
        return preds[0]

    order = np.argsort(scores)[::-1]
    preds = [preds[i] for i in order[:2]]
    scores = [scores[i] for i in order[:2]]

    l_sum = sum(scores)
    weight1 = (scores[1] / l_sum) + 0.1
    weight1 = float(np.clip(weight1, 0.0, 1.0))
    weight2 = 1.0 - weight1
    return preds[0] * weight1 + preds[1] * weight2


def _add_engineered_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic feature engineering for lookup-median regression:
    - Integer time index within each breath (robust vs float-key mismatches).
    - Cumulative inspired volume proxy: integral of u_in over time (u_in_dt cumulative sum).
    - Small lag history of controls/state (u_in, u_out, u_in_cum) to capture dynamics.
    """
    out = df.copy()
    out.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")

    out["t_idx"] = out.groupby("breath_id", sort=False).cumcount().astype(np.int16)

    dt = out.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0).to_numpy()
    out["dt"] = dt.astype(np.float32)
    out["u_in_dt"] = (
        out["u_in"].to_numpy(dtype=np.float32) * out["dt"].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    out["u_in_cum"] = (
        out.groupby("breath_id", sort=False)["u_in_dt"].cumsum().astype(np.float32)
    )

    g = out.groupby("breath_id", sort=False)
    out["u_in_l1"] = g["u_in"].shift(1).fillna(0.0).astype(np.float32)
    out["u_in_l2"] = g["u_in"].shift(2).fillna(0.0).astype(np.float32)
    out["u_out_l1"] = g["u_out"].shift(1).fillna(0).astype(np.int8)
    out["u_in_cum_l1"] = g["u_in_cum"].shift(1).fillna(0.0).astype(np.float32)

    out.sort_values("id", inplace=True, kind="mergesort")
    return out


def _fallback_model_predict(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> np.ndarray:
    """
    Fallback when external submission files aren't available.

    Semantics preserved:
    - Same engineered features.
    - Same median-lookup hierarchy (seq -> base -> RC+t -> global).
    - Same inspiratory-only training (u_out==0).
    - Same expiratory handling: carry forward last predicted inspiratory pressure in the breath.
    - Same snapping to nearest allowed pressure for state propagation (used for p_prev_bin).
    """
    tr_full = _add_engineered_state(train_df)
    te = _add_engineered_state(test_df)

    tr = tr_full[tr_full["u_out"].to_numpy() == 0].copy()

    q_cum = np.linspace(0, 1, 81)  # 80 bins for cumulative volume
    edges_cum = np.unique(np.quantile(tr["u_in_cum"].to_numpy(), q_cum))
    if edges_cum.size < 3:
        edges_cum = np.array([tr["u_in_cum"].min(), tr["u_in_cum"].max() + 1e-6])

    q_uin = np.linspace(0, 1, 51)  # 50 bins for instantaneous u_in / lagged u_in
    edges_uin = np.unique(np.quantile(tr["u_in"].to_numpy(), q_uin))
    if edges_uin.size < 3:
        edges_uin = np.array([tr["u_in"].min(), tr["u_in"].max() + 1e-6])

    def pressure_to_bin(p_arr: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(sorted_pressures, p_arr, side="left")
        idx = np.clip(idx, 0, total_pressures_len - 1).astype(np.int16)
        return idx

    def to_bin(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
        return np.digitize(x, edges[1:-1], right=False).astype(np.int16)

    for df in (tr, te):
        df["u_in_cum_bin"] = to_bin(df["u_in_cum"].to_numpy(), edges_cum)
        df["u_in_bin"] = to_bin(df["u_in"].to_numpy(), edges_uin)
        df["u_in_l1_bin"] = to_bin(df["u_in_l1"].to_numpy(), edges_uin)

    tr["p_bin"] = pressure_to_bin(tr["pressure"].to_numpy(dtype=np.float32))
    tr.sort_values(["breath_id", "t_idx"], inplace=True, kind="mergesort")
    tr["p_prev_bin"] = (
        tr.groupby("breath_id", sort=False)["p_bin"]
        .shift(1)
        .fillna(-1)
        .astype(np.int16)
    )

    keys_seq = [
        "R",
        "C",
        "t_idx",
        "u_out",
        "u_in_cum_bin",
        "u_in_bin",
        "u_in_l1_bin",
        "u_out_l1",
        "p_prev_bin",
    ]
    keys_base = [
        "R",
        "C",
        "t_idx",
        "u_out",
        "u_in_cum_bin",
        "u_in_bin",
        "u_in_l1_bin",
        "u_out_l1",
    ]
    keys_rc_t = ["R", "C", "t_idx"]

    med_seq = tr.groupby(keys_seq, sort=False)["pressure"].median()
    med_base = tr.groupby(keys_base, sort=False)["pressure"].median()
    med_rc_t = tr.groupby(keys_rc_t, sort=False)["pressure"].median()
    global_med = float(tr["pressure"].median())

    med_seq_d = med_seq.to_dict()
    med_base_d = med_base.to_dict()
    med_rc_t_d = med_rc_t.to_dict()

    te_sorted = te.sort_values(["breath_id", "t_idx"], kind="mergesort").copy()

    R = te_sorted["R"].to_numpy(dtype=np.int16)
    C = te_sorted["C"].to_numpy(dtype=np.int16)
    t_idx = te_sorted["t_idx"].to_numpy(dtype=np.int16)
    u_out = te_sorted["u_out"].to_numpy(dtype=np.int8)
    u_in_cum_bin = te_sorted["u_in_cum_bin"].to_numpy(dtype=np.int16)
    u_in_bin = te_sorted["u_in_bin"].to_numpy(dtype=np.int16)
    u_in_l1_bin = te_sorted["u_in_l1_bin"].to_numpy(dtype=np.int16)
    u_out_l1 = te_sorted["u_out_l1"].to_numpy(dtype=np.int8)

    pred = np.empty(len(te_sorted), dtype=np.float32)

    breath_id_arr = te_sorted["breath_id"].to_numpy(dtype=np.int32)
    change = np.empty(len(breath_id_arr) + 1, dtype=bool)
    change[0] = True
    change[1:-1] = breath_id_arr[1:] != breath_id_arr[:-1]
    change[-1] = True
    bounds = np.flatnonzero(change)

    for bi in range(len(bounds) - 1):
        start = int(bounds[bi])
        end = int(bounds[bi + 1])
        last_pbin = -1
        for pos in range(start, end):
            k_seq = (
                int(R[pos]),
                int(C[pos]),
                int(t_idx[pos]),
                int(u_out[pos]),
                int(u_in_cum_bin[pos]),
                int(u_in_bin[pos]),
                int(u_in_l1_bin[pos]),
                int(u_out_l1[pos]),
                int(last_pbin),
            )
            p_val = med_seq_d.get(k_seq)
            if p_val is None:
                k_base = (
                    int(R[pos]),
                    int(C[pos]),
                    int(t_idx[pos]),
                    int(u_out[pos]),
                    int(u_in_cum_bin[pos]),
                    int(u_in_bin[pos]),
                    int(u_in_l1_bin[pos]),
                    int(u_out_l1[pos]),
                )
                p_val = med_base_d.get(k_base)
            if p_val is None:
                k_rc_t = (int(R[pos]), int(C[pos]), int(t_idx[pos]))
                p_val = med_rc_t_d.get(k_rc_t)
            if p_val is None:
                p_val = global_med

            if int(u_out[pos]) == 1 and last_pbin != -1:
                p_val = float(sorted_pressures[last_pbin])

            pred[pos] = float(p_val)

            snapped = find_nearest(float(p_val))
            last_pbin = int(np.searchsorted(sorted_pressures, snapped, side="left"))
            if last_pbin < 0:
                last_pbin = 0
            elif last_pbin >= total_pressures_len:
                last_pbin = total_pressures_len - 1

    te_sorted["pred"] = pred
    te_out = te_sorted.sort_values("id", kind="mergesort")
    return te_out["pred"].to_numpy(dtype=np.float32)


def g(dp):
    """
    Original intent: blend a set of existing high-score submissions from dp.
    If dp doesn't exist / contains no eligible files, fall back to deterministic lookup
    so we still generate a valid .csv submission end-to-end.

    NOTE (timeout fix): The original code performed 125 Monte Carlo weightings and stored
    125 full-length arrays, then vstack+median (very slow and memory-heavy).
    We keep identical semantics but compute the median via a fixed-size 2D array
    (preallocated) and vectorized weight generation, avoiding Python list growth + vstack.
    """
    allow = [1348, 1358, 1758]
    l = []

    if dp is not None and os.path.isdir(dp):
        for fp in glob.iglob(f"{dp}/*"):
            try:
                file_lb = int(fp.split("/")[-1].split(".")[1].split(" ")[0])
            except Exception:
                continue
            if file_lb in allow:
                l.append(fp)

    l.sort()

    if len(l) > 0:
        loop_time = 125
        splits = 2

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
            test_dtypes = {
                "id": "int32",
                "breath_id": "int32",
                "R": "int16",
                "C": "int16",
                "time_step": "float32",
                "u_in": "float32",
                "u_out": "int8",
            }
            usecols_test = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
            df_test = pd.read_csv(TEST_PATH, usecols=usecols_test, dtype=test_dtypes)
            preds = _fallback_model_predict(df_train, df_test)
            output = pd.read_csv(
                SAMPLE_PATH, dtype={"id": "int32", "pressure": "float32"}
            )
            output["pressure"] = snap_to_nearest_pressure_vec(preds)
            output.to_csv("submission.csv", index=False)
            return

        n = flist[0].shape[0]
        P = np.empty((loop_time, n), dtype=np.float32)

        m = len(flist)
        for seed in range(loop_time):
            set_seed(seed)
            w = np.random.random(m).astype(np.float32)
            w /= w.sum()
            w[::-1].sort()  # descending

            temp = np.zeros(n, dtype=np.float32)
            for j in range(m):
                temp += flist[j].astype(np.float32, copy=False) * w[j]
            P[seed] = temp

        output = pd.read_csv(SAMPLE_PATH, dtype={"id": "int32", "pressure": "float32"})
        output["pressure"] = snap_to_nearest_pressure_vec(np.median(P, axis=0))
        output.to_csv("submission.csv", index=False)
        return

    test_dtypes = {
        "id": "int32",
        "breath_id": "int32",
        "R": "int16",
        "C": "int16",
        "time_step": "float32",
        "u_in": "float32",
        "u_out": "int8",
    }
    usecols_test = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
    df_test = pd.read_csv(TEST_PATH, usecols=usecols_test, dtype=test_dtypes)
    preds = _fallback_model_predict(df_train, df_test)

    output = pd.read_csv(SAMPLE_PATH, dtype={"id": "int32", "pressure": "float32"})
    output["pressure"] = snap_to_nearest_pressure_vec(preds)
    output.to_csv("submission.csv", index=False)




## === cell 1
g("../input/ventilator-pressure-high-score-submissions")
print("Wrote submission.csv")
sub = pd.read_csv("submission.csv")
print(sub.head())
print(sub.shape)
print(sub.columns.tolist())
print(sub.isna().sum().to_dict())
