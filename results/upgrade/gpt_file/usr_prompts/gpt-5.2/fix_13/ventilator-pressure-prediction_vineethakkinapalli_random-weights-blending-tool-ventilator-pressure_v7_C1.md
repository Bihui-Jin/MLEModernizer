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

3.58413

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.92111) has done: 'Your current code fails because it expects a directory of external “high-score submissions” that is not present in this Kaggle environment, causing empty lists and an `IndexError` during blending. To keep the core “blend multiple submissions then snap to nearest valid pressure” logic intact while making it runnable end-to-end, I add a safe fallback: if no eligible submission files are found, train a simple per-(R,C,time_step,u_out) median lookup model from `train.csv` and predict for `test.csv`. This is score-reasonable (not random), uses only provided data, and still outputs `id,pressure` with valid snapping via your existing `find_nearest`. Finally, I ensure the notebook writes a `.csv` submission file to the working directory.'
- What this solution (achieved 9.88149) has done: 'Your current fallback median-lookup is valid but far too weak for this competition, which is why the MAE is ~9.92 vs your target ~0.143 (lower is better). To move substantially toward the target without changing the “no ML model / pure pandas aggregation + snapping to valid pressures” core approach, I upgrade the fallback to a breath-aware physical proxy: compute cumulative inspired volume (`u_in` integrated over time) per breath and learn a per-(R,C,time_step,u_out) median mapping from that state to pressure, with safe backoffs. This keeps the solution as a deterministic lookup/regression-by-aggregation and retains your existing `find_nearest` snapping and submission writing. The blending path is kept intact; it only run if those external submissions actually exist.'
- What this solution (achieved 1.43592) has done: 'Your current MAE (~9.88) is far worse than the target (~0.143, lower is better), so we need a meaningful but still “pure pandas aggregation + snapping” upgrade in the fallback path (since the external high-score submissions directory isn’t available). I keep your overall logic and snapping intact, but strengthen the fallback by learning the median pressure from a more informative per-breath state: cumulative inspired volume plus short history of `u_in` and `u_out` (lags), which better captures dynamics without introducing any ML training loops. I also make `time_step` joinable by converting it to a stable integer index per breath (avoids float-key mismatches that can silently destroy lookup quality). These are minimal, deterministic changes aimed specifically at reducing MAE while preserving your approach and still writing `submission.csv`.'
- What this solution (achieved 2.25585) has done: 'Main bottlenecks are (a) the Monte-Carlo ensembling loop in `g()` (125 iterations building large `pred_list` and then `vstack`), (b) per-row pandas `reindex` inside nested Python loops in `_fallback_model_predict`, and (c) `output["pressure"].apply(find_nearest)` which is a slow Python loop over 603,600 rows. The optimized version keeps the same core semantics (same blending logic + same deterministic fallback model and nearest-pressure snapping), but replaces slow Python/pandas per-row operations with vectorized NumPy operations and a per-breath array loop that only does O(80) work per breath. It also computes the ensemble median in a streaming way without storing 125 full prediction arrays, and it snaps predictions to the nearest allowed pressure using a vectorized `searchsorted`-based routine (equivalent to `find_nearest` for all values). These changes remove the dominant overhead while preserving outputs up to negligible floating-point differences.'
- What this solution (achieved 2.25657) has done: 'Your current score (2.25585 MAE) is far from the target (0.14324), so we need a real accuracy gain while keeping the same “deterministic pandas aggregation lookup + snapping” core logic. The biggest correctness issue hurting MAE is that the fallback learns medians only on inspiratory rows (`u_out==0`) but then tries to match test expiratory rows (`u_out==1`) in the same lookup keys, which almost always misses and forces weak backoffs; I keep the same hierarchy but build medians for both `u_out` values while still deriving `p_prev_bin` from inspiratory only (preserving your semantics). I also fix the `p_prev_bin` update to use the correct bin index (current code can shift by +1 due to `searchsorted`), which directly affects your sequential lookup hit-rate. Finally, I keep your existing “expiratory carry-forward last inspiratory pressure” behavior and vectorized snapping unchanged, ensuring a valid `submission.csv` is written.'
- What this solution (achieved 3.71245) has done: 'Your current MAE (2.25657, lower-is-better) is far from the target (0.14324), so we need a real accuracy improvement but without changing your core “deterministic lookup/aggregation + expiratory carry-forward + snap-to-valid-pressures” approach. The biggest accuracy limiter is that the fallback lookup keys don’t include enough dynamic state, so many test rows miss the strongest median table and fall back to weak global medians. I add two lightweight engineered state variables that are still pure deterministic feature extraction (no ML loops): per-breath time delta (`dt`) bins and a short-horizon change in control (`du_in = u_in - u_in_l1`) bins, and then include their bins into the existing sequential and base lookup keys (keeping the same backoff hierarchy and expiratory behavior). This should increase hit-rate of the strongest median mapping and reduce MAE materially while preserving your overall semantics and producing the same `submission.csv` format.'
- What this solution (achieved 2.77856) has done: 'Your current score (3.71245 MAE, lower-is-better) is far from the target (0.14324), so we should improve accuracy while keeping your deterministic median-lookup + sequential state + expiratory carry-forward + snapping core logic intact. The biggest likely regression in your last change is that adding `dt_bin`/`du_in_bin` makes the strongest lookup table too sparse, causing frequent misses and falling back to weak medians. I keep those engineered features computed (so logic is preserved) but remove them from the *groupby keys* and instead add a stronger, still-minimal “history” signal by including `u_in_l2_bin` in the lookup keys (you already compute `u_in_l2`). I also fix the `last_pbin` update to be robust and correct by using a direct mapping from snapped pressure to index (avoids off-by-one/float equality issues), which stabilizes the sequential lookup and reduces cascading errors.'
- What this solution (achieved 2.77856) has done: 'Your current MAE (2.77856, lower-is-better) is still far from the target (0.14324), so we need an accuracy gain while keeping your deterministic median-lookup + sequential `p_prev_bin` + expiratory carry-forward + snapping core logic intact. The biggest issue in the fallback is that `p_prev_bin` is learned only from inspiratory rows and then missing for expiratory rows in `tr_full`, making the strongest sequential table effectively sparse and causing frequent backoff; I fix this by propagating the last inspiratory `p_bin` forward within each breath to all timesteps (so expiratory rows also have a meaningful `p_prev_bin`). This preserves your semantics (state depends on last inspiratory pressure; expiratory still carry-forwards prediction), but materially increases hit-rate for `med_seq` keys and should reduce MAE. I also keep all I/O paths and the submission writing unchanged, ensuring `submission.csv` is produced end-to-end.'
- What this solution (achieved 2.77807) has done: 'Your current score (2.77856 MAE; lower is better) is still far above the target (0.14324), so we need a meaningful accuracy lift while preserving your deterministic median-lookup + sequential `p_prev_bin` + expiratory carry-forward + snapping core logic. The biggest low-risk gain is to reduce key sparsity and improve hit-rate of the strongest tables by (1) learning medians on a *coarser time index* (`t_bin`, e.g., 40 bins over 80 steps) while keeping your per-breath sequential loop unchanged, and (2) adding a “u_in vs cumulative volume” interaction via an additional `u_in_cum_l1_bin` key (you already compute `u_in_cum_l1`). These are still pure pandas aggregations + the same sequential prediction semantics, but they make lookups far less brittle than exact `t_idx` matching. I also keep your existing inspiratory-only derivation of pressure bins, the forward-fill of state to expiratory rows, and the same nearest-pressure snapping and submission writing.'
- What this solution (achieved 3.57032) has done: 'Your current MAE (2.77807, lower-is-better) is still far from the target (0.14324), so we need a real accuracy lift with minimal changes while preserving your deterministic lookup + sequential `p_prev_bin` + expiratory carry-forward + snapping approach. The weakest link is that the fallback lookup is too sparse and relies heavily on fragile quantile bin edges and a too-coarse `t_bin`, causing frequent backoffs to weak medians; I make the bins more stable and the strongest tables denser without changing the overall method. Concretely, I (1) replace quantile-based binning for `u_in` and cumulative volume with fixed-width bins aligned to known ranges (0–100 for `u_in`; per-(R,C) robust cap for cumulative volume), and (2) increase time resolution slightly (`t_bin = t_idx` instead of `t_idx//2`) to reduce temporal aliasing while keeping all the same groupby-median tables and sequential inference. These are small, deterministic adjustments that should increase exact-key hit-rate and reduce MAE, and the script still writes a valid `submission.csv` end-to-end.'
- What this solution (achieved 3.58413) has done: 'Your current score (3.57032 MAE; lower is better) is still far above the target (0.14324), so we should improve the fallback’s accuracy with minimal, metric-aligned changes while preserving your deterministic “median-lookup + sequential p_prev_bin + expiratory carry-forward + snap-to-valid-pressures” core logic. The least invasive high-impact fix is to (1) stop using the *test-time predicted* `p_prev_bin` state (which compounds errors) and instead use a *deterministic prior* `p_prev_bin` derived from the training set for each (R,C,t_bin,u_in_bin,u_in_cum_bin,…) key, and (2) only apply the sequential `med_seq` table on inspiratory rows (`u_out==0`) while keeping your expiratory carry-forward behavior unchanged. This keeps the same tables/semantics but removes the main error-amplification loop that is hurting MAE. I also keep your binning/time resolution and snapping intact, and the script still writes a valid `submission.csv` end-to-end.'

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
sorted_pressures = np.sort(unique_pressures).astype(np.float32)
total_pressures_len = len(sorted_pressures)

pressure_to_index = {float(p): i for i, p in enumerate(sorted_pressures)}


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

    out["du_in"] = (out["u_in"] - out["u_in_l1"]).astype(np.float32)

    out.sort_values("id", inplace=True, kind="mergesort")
    return out


def _fallback_model_predict(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> np.ndarray:
    """
    Fallback when external submission files aren't available.

    Deterministic median-lookup + expiratory carry-forward + snapping.

    CHANGE (accuracy, preserves core logic):
      - Remove error-amplifying predicted-state recursion: instead of updating `last_pbin` from our own
        predicted pressure (which compounds mistakes across the breath), compute a deterministic prior
        `p_prev_bin_prior` from training for each row's non-pressure features and use it as the
        `p_prev_bin` used in `med_seq` lookup.
      - Apply the sequential `med_seq` table only on inspiratory rows (u_out==0); expiratory rows
        keep the existing carry-forward last inspiratory pressure behavior.
    """
    tr_full = _add_engineered_state(train_df)
    te = _add_engineered_state(test_df)

    tr_insp = tr_full[tr_full["u_out"].to_numpy() == 0].copy()

    edges_uin = np.linspace(0.0, 100.0, 101, dtype=np.float32)  # 100 bins of width 1

    rc_caps = (
        tr_insp.groupby(["R", "C"], sort=False)["u_in_cum"]
        .quantile(0.997)
        .astype(np.float32)
    )
    rc_caps_d = rc_caps.to_dict()

    def _cap_for_rc(r: int, c: int) -> float:
        v = float(rc_caps_d.get((int(r), int(c)), float(tr_insp["u_in_cum"].max())))
        return max(v, 1e-3)

    def pressure_to_bin_left(p_arr: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(sorted_pressures, p_arr, side="left")
        idx = np.clip(idx, 0, total_pressures_len - 1).astype(np.int16)
        return idx

    def to_bin_fixed(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
        return np.digitize(x, edges[1:-1], right=False).astype(np.int16)

    def to_tbin(t_idx_arr: np.ndarray) -> np.ndarray:
        return t_idx_arr.astype(np.int16)

    for df in (tr_full, te):
        df["u_in_bin"] = to_bin_fixed(df["u_in"].to_numpy(dtype=np.float32), edges_uin)
        df["u_in_l1_bin"] = to_bin_fixed(
            df["u_in_l1"].to_numpy(dtype=np.float32), edges_uin
        )
        df["u_in_l2_bin"] = to_bin_fixed(
            df["u_in_l2"].to_numpy(dtype=np.float32), edges_uin
        )
        df["t_bin"] = to_tbin(df["t_idx"].to_numpy(dtype=np.int16))

    def add_cum_bins(df: pd.DataFrame) -> None:
        r_arr = df["R"].to_numpy(dtype=np.int16)
        c_arr = df["C"].to_numpy(dtype=np.int16)
        cum = df["u_in_cum"].to_numpy(dtype=np.float32)
        cum_l1 = df["u_in_cum_l1"].to_numpy(dtype=np.float32)

        u_in_cum_bin = np.empty(len(df), dtype=np.int16)
        u_in_cum_l1_bin = np.empty(len(df), dtype=np.int16)

        combos = np.unique(np.stack([r_arr, c_arr], axis=1), axis=0)
        for r, c in combos:
            mask = (r_arr == r) & (c_arr == c)
            cap = _cap_for_rc(int(r), int(c))
            edges_cum = np.linspace(0.0, cap, 201, dtype=np.float32)
            u_in_cum_bin[mask] = to_bin_fixed(np.clip(cum[mask], 0.0, cap), edges_cum)
            u_in_cum_l1_bin[mask] = to_bin_fixed(
                np.clip(cum_l1[mask], 0.0, cap), edges_cum
            )

        df["u_in_cum_bin"] = u_in_cum_bin
        df["u_in_cum_l1_bin"] = u_in_cum_l1_bin

    add_cum_bins(tr_full)
    add_cum_bins(te)

    tr_insp = tr_full[tr_full["u_out"].to_numpy() == 0].copy()
    tr_insp["p_bin"] = pressure_to_bin_left(
        tr_insp["pressure"].to_numpy(dtype=np.float32)
    )
    tr_insp.sort_values(["breath_id", "t_idx"], inplace=True, kind="mergesort")
    tr_insp["p_prev_bin"] = (
        tr_insp.groupby("breath_id", sort=False)["p_bin"]
        .shift(1)
        .fillna(-1)
        .astype(np.int16)
    )

    key_cols = ["breath_id", "t_idx"]
    tr_full = tr_full.sort_values(key_cols, kind="mergesort")

    tr_insp_map = tr_insp[key_cols + ["p_prev_bin"]].copy()
    tr_full = tr_full.merge(tr_insp_map, on=key_cols, how="left", sort=False)

    tr_full["p_prev_bin"] = (
        tr_full.sort_values(["breath_id", "t_idx"], kind="mergesort")
        .groupby("breath_id", sort=False)["p_prev_bin"]
        .ffill()
        .fillna(-1)
        .astype(np.int16)
    )

    keys_seq = [
        "R",
        "C",
        "t_bin",
        "u_out",
        "u_in_cum_bin",
        "u_in_cum_l1_bin",
        "u_in_bin",
        "u_in_l1_bin",
        "u_in_l2_bin",
        "u_out_l1",
        "p_prev_bin",
    ]
    keys_base = [
        "R",
        "C",
        "t_bin",
        "u_out",
        "u_in_cum_bin",
        "u_in_cum_l1_bin",
        "u_in_bin",
        "u_in_l1_bin",
        "u_in_l2_bin",
        "u_out_l1",
    ]
    keys_rc_t = ["R", "C", "t_bin"]

    med_seq = tr_full.groupby(keys_seq, sort=False)["pressure"].median()
    med_base = tr_full.groupby(keys_base, sort=False)["pressure"].median()
    med_rc_t = tr_full.groupby(keys_rc_t, sort=False)["pressure"].median()

    global_med = float(tr_insp["pressure"].median())

    med_seq_d = med_seq.to_dict()
    med_base_d = med_base.to_dict()
    med_rc_t_d = med_rc_t.to_dict()

    pprev_prior = tr_full.groupby(keys_base, sort=False)["p_prev_bin"].median()
    pprev_prior_d = pprev_prior.to_dict()

    te_sorted = te.sort_values(["breath_id", "t_idx"], kind="mergesort").copy()

    R = te_sorted["R"].to_numpy(dtype=np.int16)
    C = te_sorted["C"].to_numpy(dtype=np.int16)
    t_bin = te_sorted["t_bin"].to_numpy(dtype=np.int16)
    u_out = te_sorted["u_out"].to_numpy(dtype=np.int8)
    u_in_cum_bin = te_sorted["u_in_cum_bin"].to_numpy(dtype=np.int16)
    u_in_cum_l1_bin = te_sorted["u_in_cum_l1_bin"].to_numpy(dtype=np.int16)
    u_in_bin = te_sorted["u_in_bin"].to_numpy(dtype=np.int16)
    u_in_l1_bin = te_sorted["u_in_l1_bin"].to_numpy(dtype=np.int16)
    u_in_l2_bin = te_sorted["u_in_l2_bin"].to_numpy(dtype=np.int16)
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

        last_insp_pressure = None  # stored as float pressure value
        for pos in range(start, end):
            k_base = (
                int(R[pos]),
                int(C[pos]),
                int(t_bin[pos]),
                int(u_out[pos]),
                int(u_in_cum_bin[pos]),
                int(u_in_cum_l1_bin[pos]),
                int(u_in_bin[pos]),
                int(u_in_l1_bin[pos]),
                int(u_in_l2_bin[pos]),
                int(u_out_l1[pos]),
            )

            if int(u_out[pos]) == 1:
                if last_insp_pressure is not None:
                    p_val = float(last_insp_pressure)
                else:
                    p_val = med_base_d.get(k_base)
                    if p_val is None:
                        p_val = med_rc_t_d.get(
                            (int(R[pos]), int(C[pos]), int(t_bin[pos]))
                        )
                    if p_val is None:
                        p_val = global_med
            else:
                p_prev_bin_prior = pprev_prior_d.get(k_base)
                if p_prev_bin_prior is None:
                    p_prev_bin_prior = -1
                else:
                    p_prev_bin_prior = int(np.round(float(p_prev_bin_prior)))

                k_seq = k_base + (int(p_prev_bin_prior),)
                p_val = med_seq_d.get(k_seq)
                if p_val is None:
                    p_val = med_base_d.get(k_base)
                if p_val is None:
                    p_val = med_rc_t_d.get((int(R[pos]), int(C[pos]), int(t_bin[pos])))
                if p_val is None:
                    p_val = global_med

                snapped = float(find_nearest(float(p_val)))
                last_insp_pressure = snapped

            pred[pos] = float(p_val)

    te_sorted["pred"] = pred
    te_out = te_sorted.sort_values("id", kind="mergesort")
    return te_out["pred"].to_numpy(dtype=np.float32)


def g(dp):
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
