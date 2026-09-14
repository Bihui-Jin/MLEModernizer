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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
seaborn==0.12.2
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

2.893034360980532

# 6. Current score

4.29321

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.16811) has done: 'Your notebook fails because it depends on external Kaggle “../input/...” submissions that are not present in this environment, so none of the `sub*` arrays exist and all later cells error. I replace that broken ensemble-loading with a minimal, fully self-contained baseline that trains from the provided `train.csv` and predicts for `test.csv` using only the competition columns, then writes a valid `submission.csv`. To stay within the installed packages (no TensorFlow/PyTorch here) and keep runtime reasonable, the model be a simple groupwise regression: estimate pressure as a function of `u_in` separately for each `(R, C, u_out)` group, and output 0 during expiration (`u_out==1`) to align with the metric scoring only inspiration. This should produce a valid submission and typically beats the all-zero baseline, moving MAE down toward your target.'
- What this solution (achieved 6.42515) has done: 'Your current baseline is too weak because it ignores the strong per‑breath temporal structure and the fact that pressure takes on a fixed discrete set of values; a linear `u_in` fit leaves large systematic errors. To move the MAE down toward your target while keeping changes minimal and within installed packages, I keep the same “fit on inspiration only, predict 0 on expiration” core idea but replace the per-group linear regression with a per-group 1D interpolation (mapping `u_in -> expected pressure`) learned from the training inspiration rows. Finally, I snap predictions to the nearest valid pressure level observed in training, which is well-aligned with this competition and typically improves MAE without changing evaluation semantics. The script still runs end-to-end and writes `submission.csv` (and keeps your extra CSVs).'
- What this solution (achieved 4.00992) has done: 'Your current method is bottlenecked by per-row Python looping and by fitting a single global `u_in -> pressure` curve that ignores the strong dependence on lung attributes `(R, C)` and time evolution within a breath. Keeping the same core idea (learn an inspiration-only mapping and output 0 for `u_out==1`, then snap to valid pressure levels), I (1) build the mapping by `(R, C)` (not `(R, C, u_out)` since inspiration rows always have `u_out=0`) and (2) include `time_step` as an additional 1D axis via small binning to better capture dynamics without changing the overall approach. I also vectorize prediction by processing each `(R, C)` group at once (no model change, just faster and more stable execution). This should reduce MAE materially from 6.425 toward your target 2.893 while still producing a valid `submission.csv`.'
- What this solution (achieved 5.6494) has done: 'Your current interpolation is already leveraging `time_step` and `(R,C)`, so the smallest likely gain toward your lower-is-better target is to better match the metric: only inspiratory *and* before the first `u_out==1` within each breath is scored. I keep the same bin-mean mapping and nearest-pressure snapping, but change the training inspiration mask and test prediction mask to “inspiration until first exhale” per breath, so we don’t learn/predict on late rows that may be unscored/mismatched. This usually reduces MAE without changing the modeling approach, and it’s a minimal, fully self-contained change. The script still run end-to-end and write `submission.csv` exactly as required.'
- What this solution (achieved 16.06993) has done: 'Your current score (5.6494 MAE, lower is better) is still far from the target (2.8930), so we can make a slightly stronger—but still same-core—mapping without changing the overall approach (bin means + nearest-time-bin + 1D interp + snapping to valid pressure levels). The minimal likely gain is to condition the learned mapping not just on `(R,C,time_step)` but also on a coarse “within-breath position” (step index), because `time_step` alone plus binning can blur early/late dynamics across breaths. We add `step` (0–79 within each breath) as another key in the lookup, still using the same mean-aggregation and `np.interp` prediction mechanics, and keep the same “inspiration until first exhale” masking and pressure-level snapping. This should reduce systematic error toward your target while keeping runtime under the limit and producing the same `submission.csv` outputs.'
- What this solution (achieved 3.84049) has done: 'I fix the KeyError in inference by making the `(t, step)` lookup robust: when an exact key is missing, we fall back to the nearest available `(t, step)` pair within the same `(R,C)` mapping (or global fallback), instead of assuming `(t, min_step)` exists. This is a minimal logic correction that unblocks execution end-to-end and keeps your existing interpolation + snapping approach intact. I also remove an unnecessary object/Series-based grouping (which is slow and memory-heavy) and replace it with a stable NumPy-based grouping to avoid runtime/memory issues without changing predictions. The script then write `submission.csv` (and the extra CSVs) successfully.'
- What this solution (achieved 3.84049) has done: 'Your current gap to the target is still sizable (MAE 3.84 vs 2.89; lower is better), so the safest minimal improvement is to keep the exact same lookup/interp/snapping approach but make the learned mapping more specific to lung attributes. Concretely, we build the pressure mapping keyed by `(R, C, t_bin, step, u_bin)` instead of sharing a single `(R,C)` mapping across all breaths, and we only fall back to the global mapping when an `(R,C)` group truly has no data. This is a small change in conditioning (not a new model), and it usually reduces systematic bias because pressure dynamics differ strongly by `(R,C)`. I also precompute the `(t_bins, s_bins)` arrays per group once during training to avoid recomputing them inside the inference loop (no semantic change, just speed/stability).'
- What this solution (achieved 5.59118) has done: 'We keep your exact “(R,C,t_bin,step,u_bin) mean lookup → nearest (t,step) fallback → 1D interp → snap to valid pressure levels” core, but make two minimal metric-aligned fixes that usually reduce MAE. First, we only force `pressure=0` for rows that are truly unscored (post-first-exhale); for missing/mapped inspiration rows we no longer implicitly bias toward 0 via the global fallback, reducing underprediction. Second, we add a tiny, deterministic calibration step learned on the training inspiratory rows (a single scalar scale+shift on predictions before snapping) to correct systematic bias without changing the modeling approach. These are small, safe changes and keep runtime under the limit while producing the same required `submission.csv`.'
- What this solution (achieved 3.83869) has done: 'Your current MAE (5.59) is still much worse than the target (2.89), so we should improve accuracy but with minimal, metric-aligned edits that keep the same lookup→interp→snap core. The biggest low-risk gain here is to stop forcing expiration predictions to 0 (those rows are unscored anyway) and instead fill them with a reasonable “carry-forward” of the last inspiratory prediction within each breath; this can’t hurt the metric but often stabilizes per-breath dynamics and avoids odd artifacts. Next, we make the learned calibration slightly more metric-consistent by computing it on *snapped* (discretized) predictions, matching what you actually submit. All other logic, including binning, nearest-key fallback, interpolation, and pressure-level snapping, is preserved.'
- What this solution (achieved 3.83869) has done: 'To move your MAE down toward the 2.893 target without changing the core “(R,C,t_bin,step,u_bin) mean lookup → nearest-key fallback → 1D interp → snap-to-levels + linear calibration” logic, I make two minimal, metric-aligned adjustments. First, I stop applying the linear calibration to all rows (including unscored expiration) and apply it only on inspiratory rows, then do the breath-wise carry-forward; this preserves your intended semantics while avoiding calibration-induced distortions before the ffill. Second, because the evaluation ignores expiration entirely, I apply the discrete snapping only on inspiratory rows (after calibration) and leave expiration as the carried-forward continuous values; this keeps inspiratory predictions optimally aligned to the discrete target without introducing extra quantization artifacts into the ffill inputs. These are small post-processing changes (not a model change) and should modestly reduce MAE from 3.83869 toward your target.'
- What this solution (achieved 4.45867) has done: 'Your current score (3.83869 MAE, lower-is-better) is still above the target (2.89303), so we should make a small, metric-aligned improvement that keeps your exact lookup→interp→calibrate→snap core intact. The most impactful minimal change here is to avoid over-quantizing `u_in` during training-map construction: right now you bin `u_in` at 0.5, which collapses distinct control inputs and weakens the interpolation curve; we can keep the same approach but build maps from the raw `u_in` values (still using the same `time_step` bins, `(R,C,t,step)` keys, and mean aggregation). This is not a new model—just a higher-resolution version of the same mapping—and it typically reduces systematic error while remaining stable and fast. Everything else (inspiration-until-first-exhale masking, nearest-key fallback, linear calibration on snapped preds, inspiratory snapping only, and breath-wise ffill for expiration) is preserved, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 4.33383) has done: 'We make a minimal, score-improving change by increasing the resolution of the pressure lookup over time without changing the core lookup→interp→calibrate→snap pipeline. Concretely, we slightly reduce the `time_step` bin size (`BIN_T`) so the `(R,C,t_bin,step,u)` mean maps blur less across different dynamics, which should lower MAE toward your 2.893 target (you’re currently worse than target, and lower is better). Everything else (inspiration-until-first-exhale masking, nearest-key fallback, linear calibration on snapped train preds, inspiratory snapping only, and breath-wise forward-fill for expiration) is preserved. This is a small, safe adjustment that typically improves accuracy while keeping runtime reasonable and still writes a valid `submission.csv`.'
- What this solution (achieved 4.29321) has done: 'We keep your exact lookup → nearest-(t,step) fallback → 1D `np.interp` → linear calibration → inspiratory snapping → breath-wise ffill pipeline intact, and only make a small, metric-aligned refinement to reduce MAE from 4.33 toward 2.89. The minimal change is to slightly increase time resolution again (smaller `BIN_T`) so the learned `(R,C,t_bin,step)` curves blur less across different pressure dynamics, which typically improves this mapping-based approach. Because higher resolution can create sparse keys, we also make the `(t,step)` bins per `(R,C)` robust by ensuring the bin arrays are never empty (fallback to global bins for that group) without changing prediction semantics. Everything still runs end-to-end and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

np.random.seed(42)



## === cell 1
DATA_DIR = "/kaggle/data"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
SAMPLE_PATH = f"{DATA_DIR}/sample_submission.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_PATH)

assert {"id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"}.issubset(
    train.columns
)
assert {"id", "breath_id", "R", "C", "time_step", "u_in", "u_out"}.issubset(
    test.columns
)
assert list(sample.columns) == ["id", "pressure"]



## === cell 2
train_first_exhale_step = train.groupby("breath_id", sort=False)["u_out"].transform(
    "cummax"
)
train_insp = train[train_first_exhale_step == 0].copy()

BIN_U = None  # (was 0.5)
BIN_T = 0.008  # was 0.01

train_insp["step"] = (
    train_insp.groupby("breath_id", sort=False).cumcount().astype(np.int16)
)

if len(train_insp) == 0:
    global_u = np.array([0.0, 100.0], dtype=np.float32)
    global_p = np.array([0.0, 0.0], dtype=np.float32)
    global_tp = {(0.0, 0): (global_u, global_p)}
else:
    t_bin = np.round(train_insp["time_step"].to_numpy(np.float32) / BIN_T) * BIN_T
    step = train_insp["step"].to_numpy(np.int16)

    u_vals = train_insp["u_in"].to_numpy(np.float32)

    tmp = pd.DataFrame(
        {
            "t": t_bin.astype(np.float32),
            "step": step,
            "u": u_vals,
            "p": train_insp["pressure"].to_numpy(np.float32),
        }
    )
    gb = tmp.groupby(["t", "step", "u"], sort=True)["p"].mean().reset_index()

    global_tp = {}
    for (t, s), g in gb.groupby(["t", "step"], sort=True):
        u_arr = g["u"].to_numpy(np.float32)
        p_arr = g["p"].to_numpy(np.float32)
        if len(u_arr) >= 2:
            global_tp[(float(t), int(s))] = (u_arr, p_arr)
    if len(global_tp) == 0:
        global_tp[(0.0, 0)] = (
            np.array([0.0, 100.0], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
        )

maps = {}
rc_bins = (
    {}
)  # precomputed bins per (R,C) to avoid recompute in inference (no semantic change)

if len(train_insp) > 0:
    t_bin = np.round(train_insp["time_step"].to_numpy(np.float32) / BIN_T) * BIN_T
    step = train_insp["step"].to_numpy(np.int16)

    u_vals = train_insp["u_in"].to_numpy(np.float32)

    tmp = pd.DataFrame(
        {
            "R": train_insp["R"].to_numpy(np.int16),
            "C": train_insp["C"].to_numpy(np.int16),
            "t": t_bin.astype(np.float32),
            "step": step,
            "u": u_vals,
            "p": train_insp["pressure"].to_numpy(np.float32),
        }
    )
    gb = tmp.groupby(["R", "C", "t", "step", "u"], sort=True)["p"].mean().reset_index()

    for (R, C), g_rc in gb.groupby(["R", "C"], sort=False):
        ts_dict = {}
        for (t, s), g_ts in g_rc.groupby(["t", "step"], sort=True):
            u_arr = g_ts["u"].to_numpy(np.float32)
            p_arr = g_ts["p"].to_numpy(np.float32)
            if len(u_arr) >= 2:
                ts_dict[(float(t), int(s))] = (u_arr, p_arr)
        if len(ts_dict) > 0:
            key_rc = (int(R), int(C))
            maps[key_rc] = ts_dict
            rc_bins[key_rc] = (
                np.array(sorted({k[0] for k in ts_dict.keys()}), dtype=np.float32),
                np.array(sorted({k[1] for k in ts_dict.keys()}), dtype=np.int16),
            )

pmin = float(train["pressure"].min())
pmax = float(train["pressure"].max())

pressure_levels = np.sort(train["pressure"].round(5).unique().astype(np.float32))

global_t_bins = np.array(sorted({k[0] for k in global_tp.keys()}), dtype=np.float32)
global_s_bins = np.array(sorted({k[1] for k in global_tp.keys()}), dtype=np.int16)


def _nearest_key_fallback(ts_dict, t_val, s_val):
    """
    If exact (t_val,s_val) missing, fall back to nearest available key in ts_dict.
    Uses nearest in (t, step) L1 distance, deterministic.
    """
    key = (float(t_val), int(s_val))
    if key in ts_dict:
        return key
    keys = np.array(list(ts_dict.keys()), dtype=object)
    t_arr = keys[:, 0].astype(np.float32)
    s_arr = keys[:, 1].astype(np.int32)
    dist = np.abs(t_arr - np.float32(t_val)) + 0.001 * np.abs(s_arr - int(s_val))
    j = int(np.argmin(dist))
    return (float(t_arr[j]), int(s_arr[j]))


def _predict_from_maps(df_insp: pd.DataFrame) -> np.ndarray:
    """
    Predict pressures for inspiration rows only, using the same logic as test-time:
    nearest (t, step) within (R,C) map else global, then np.interp on u.
    """
    if len(df_insp) == 0:
        return np.zeros(0, dtype=np.float32)

    t_arr = np.round(df_insp["time_step"].to_numpy(np.float32) / BIN_T) * BIN_T
    u_arr = df_insp["u_in"].to_numpy(np.float32)
    R_arr = df_insp["R"].to_numpy(np.int16)
    C_arr = df_insp["C"].to_numpy(np.int16)
    s_arr = df_insp["step"].to_numpy(np.int16)

    out = np.empty(len(df_insp), dtype=np.float32)

    rc_pairs = np.stack([R_arr.astype(np.int16), C_arr.astype(np.int16)], axis=1)
    uniq_rc, inv_rc = np.unique(rc_pairs, axis=0, return_inverse=True)

    for i in range(len(uniq_rc)):
        R, C = int(uniq_rc[i, 0]), int(uniq_rc[i, 1])
        rows = np.flatnonzero(inv_rc == i).astype(np.int64)

        t_rows = t_arr[rows]
        s_rows = s_arr[rows]
        u_rows = u_arr[rows]

        ts_dict = maps.get((R, C), None)

        if ts_dict is None:
            pos_t = np.searchsorted(global_t_bins, t_rows, side="left")
            pos_t = np.clip(pos_t, 0, len(global_t_bins) - 1)
            pos_tl = np.clip(pos_t - 1, 0, len(global_t_bins) - 1)
            t_right = global_t_bins[pos_t]
            t_left = global_t_bins[pos_tl]
            use_left_t = np.abs(t_rows - t_left) <= np.abs(t_rows - t_right)
            t_near = np.where(use_left_t, t_left, t_right).astype(np.float32)

            pos_s = np.searchsorted(global_s_bins, s_rows, side="left")
            pos_s = np.clip(pos_s, 0, len(global_s_bins) - 1)
            pos_sl = np.clip(pos_s - 1, 0, len(global_s_bins) - 1)
            s_right = global_s_bins[pos_s]
            s_left = global_s_bins[pos_sl]
            use_left_s = np.abs(s_rows - s_left) <= np.abs(s_rows - s_right)
            s_near = np.where(use_left_s, s_left, s_right).astype(np.int16)

            pairs = np.stack([t_near, s_near.astype(np.float32)], axis=1)
            uniq_pairs, inv = np.unique(pairs, axis=0, return_inverse=True)
            for k, (t_val, s_val_f) in enumerate(uniq_pairs):
                sel = inv == k
                key = (float(t_val), int(s_val_f))
                if key not in global_tp:
                    key = _nearest_key_fallback(global_tp, t_val, int(s_val_f))
                u_vals, p_vals = global_tp[key]
                out[rows[sel]] = np.interp(u_rows[sel], u_vals, p_vals).astype(
                    np.float32
                )
        else:
            t_bins_rc, s_bins_rc = rc_bins.get((R, C), (global_t_bins, global_s_bins))
            if len(t_bins_rc) == 0:
                t_bins_rc = global_t_bins
            if len(s_bins_rc) == 0:
                s_bins_rc = global_s_bins

            pos_t = np.searchsorted(t_bins_rc, t_rows, side="left")
            pos_t = np.clip(pos_t, 0, len(t_bins_rc) - 1)
            pos_tl = np.clip(pos_t - 1, 0, len(t_bins_rc) - 1)
            t_right = t_bins_rc[pos_t]
            t_left = t_bins_rc[pos_tl]
            use_left_t = np.abs(t_rows - t_left) <= np.abs(t_rows - t_right)
            t_near = np.where(use_left_t, t_left, t_right).astype(np.float32)

            pos_s = np.searchsorted(s_bins_rc, s_rows, side="left")
            pos_s = np.clip(pos_s, 0, len(s_bins_rc) - 1)
            pos_sl = np.clip(pos_s - 1, 0, len(s_bins_rc) - 1)
            s_right = s_bins_rc[pos_s]
            s_left = s_bins_rc[pos_sl]
            use_left_s = np.abs(s_rows - s_left) <= np.abs(s_rows - s_right)
            s_near = np.where(use_left_s, s_left, s_right).astype(np.int16)

            pairs = np.stack([t_near, s_near.astype(np.float32)], axis=1)
            uniq_pairs, inv = np.unique(pairs, axis=0, return_inverse=True)
            for k, (t_val, s_val_f) in enumerate(uniq_pairs):
                sel = inv == k
                key = _nearest_key_fallback(ts_dict, t_val, int(s_val_f))
                u_vals, p_vals = ts_dict[key]
                out[rows[sel]] = np.interp(u_rows[sel], u_vals, p_vals).astype(
                    np.float32
                )

    return out


def _snap_to_levels(x: np.ndarray) -> np.ndarray:
    """Deterministic nearest-neighbor snap to valid discrete pressure levels."""
    x = np.clip(x.astype(np.float32), np.float32(pmin), np.float32(pmax))
    idx = np.searchsorted(pressure_levels, x, side="left")
    idx = np.clip(idx, 0, len(pressure_levels) - 1)
    idx_left = np.clip(idx - 1, 0, len(pressure_levels) - 1)
    right = pressure_levels[idx]
    left = pressure_levels[idx_left]
    choose_left = np.abs(x - left) <= np.abs(x - right)
    return np.where(choose_left, left, right).astype(np.float32)


if len(train_insp) > 0:
    train_insp_small = train_insp[
        ["breath_id", "R", "C", "time_step", "u_in", "pressure", "step"]
    ].copy()
    y_true = train_insp_small["pressure"].to_numpy(np.float32)

    y_pred_raw = _predict_from_maps(train_insp_small).astype(np.float32)
    y_pred = _snap_to_levels(y_pred_raw)

    x = y_pred
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y_true))
    denom = float(np.mean((x - x_mean) ** 2))
    if denom > 1e-12:
        a = float(np.mean((x - x_mean) * (y_true - y_mean)) / denom)
    else:
        a = 1.0
    b = float(y_mean - a * x_mean)

    a = float(np.clip(a, 0.8, 1.2))
    b = float(np.clip(b, -2.0, 2.0))
else:
    a, b = 1.0, 0.0

(a, b)



## === cell 3
pred = np.zeros(len(test), dtype=np.float32)

test_first_exhale_step = test.groupby("breath_id", sort=False)["u_out"].transform(
    "cummax"
)
insp_mask = (test_first_exhale_step == 0).to_numpy()

test_insp = test.loc[insp_mask, ["breath_id", "R", "C", "time_step", "u_in"]].copy()
if len(test_insp) > 0:
    test_insp["step"] = (
        test_insp.groupby("breath_id", sort=False).cumcount().astype(np.int16)
    )

    t_test = np.round(test_insp["time_step"].to_numpy(np.float32) / BIN_T) * BIN_T
    s_test = test_insp["step"].to_numpy(np.int16)
    u_test = test_insp["u_in"].to_numpy(np.float32)
    R_test = test_insp["R"].to_numpy(np.int16)
    C_test = test_insp["C"].to_numpy(np.int16)

    insp_idx = np.flatnonzero(insp_mask)

    rc_pairs = np.stack([R_test.astype(np.int16), C_test.astype(np.int16)], axis=1)
    uniq_rc, inv_rc = np.unique(rc_pairs, axis=0, return_inverse=True)

    for i in range(len(uniq_rc)):
        R, C = int(uniq_rc[i, 0]), int(uniq_rc[i, 1])
        rows = np.flatnonzero(inv_rc == i).astype(np.int64)

        t_rows = t_test[rows]
        s_rows = s_test[rows]
        u_rows = u_test[rows]

        ts_dict = maps.get((R, C), None)

        if ts_dict is None:
            pos_t = np.searchsorted(global_t_bins, t_rows, side="left")
            pos_t = np.clip(pos_t, 0, len(global_t_bins) - 1)
            pos_tl = np.clip(pos_t - 1, 0, len(global_t_bins) - 1)
            t_right = global_t_bins[pos_t]
            t_left = global_t_bins[pos_tl]
            use_left_t = np.abs(t_rows - t_left) <= np.abs(t_rows - t_right)
            t_near = np.where(use_left_t, t_left, t_right).astype(np.float32)

            pos_s = np.searchsorted(global_s_bins, s_rows, side="left")
            pos_s = np.clip(pos_s, 0, len(global_s_bins) - 1)
            pos_sl = np.clip(pos_s - 1, 0, len(global_s_bins) - 1)
            s_right = global_s_bins[pos_s]
            s_left = global_s_bins[pos_sl]
            use_left_s = np.abs(s_rows - s_left) <= np.abs(s_rows - s_right)
            s_near = np.where(use_left_s, s_left, s_right).astype(np.int16)

            out = np.empty(len(rows), dtype=np.float32)
            pairs = np.stack([t_near, s_near.astype(np.float32)], axis=1)
            uniq_pairs, inv = np.unique(pairs, axis=0, return_inverse=True)
            for k, (t_val, s_val_f) in enumerate(uniq_pairs):
                sel = inv == k
                key = (float(t_val), int(s_val_f))
                if key not in global_tp:
                    key = _nearest_key_fallback(global_tp, t_val, int(s_val_f))
                u_vals, p_vals = global_tp[key]
                out[sel] = np.interp(u_rows[sel], u_vals, p_vals).astype(np.float32)
            pred[insp_idx[rows]] = out
        else:
            t_bins_rc, s_bins_rc = rc_bins.get((R, C), (global_t_bins, global_s_bins))
            if len(t_bins_rc) == 0:
                t_bins_rc = global_t_bins
            if len(s_bins_rc) == 0:
                s_bins_rc = global_s_bins

            pos_t = np.searchsorted(t_bins_rc, t_rows, side="left")
            pos_t = np.clip(pos_t, 0, len(t_bins_rc) - 1)
            pos_tl = np.clip(pos_t - 1, 0, len(t_bins_rc) - 1)
            t_right = t_bins_rc[pos_t]
            t_left = t_bins_rc[pos_tl]
            use_left_t = np.abs(t_rows - t_left) <= np.abs(t_rows - t_right)
            t_near = np.where(use_left_t, t_left, t_right).astype(np.float32)

            pos_s = np.searchsorted(s_bins_rc, s_rows, side="left")
            pos_s = np.clip(pos_s, 0, len(s_bins_rc) - 1)
            pos_sl = np.clip(pos_s - 1, 0, len(s_bins_rc) - 1)
            s_right = s_bins_rc[pos_s]
            s_left = s_bins_rc[pos_sl]
            use_left_s = np.abs(s_rows - s_left) <= np.abs(s_rows - s_right)
            s_near = np.where(use_left_s, s_left, s_right).astype(np.int16)

            out = np.empty(len(rows), dtype=np.float32)
            pairs = np.stack([t_near, s_near.astype(np.float32)], axis=1)
            uniq_pairs, inv = np.unique(pairs, axis=0, return_inverse=True)
            for k, (t_val, s_val_f) in enumerate(uniq_pairs):
                sel = inv == k
                key = _nearest_key_fallback(ts_dict, t_val, int(s_val_f))
                u_vals, p_vals = ts_dict[key]
                out[sel] = np.interp(u_rows[sel], u_vals, p_vals).astype(np.float32)
            pred[insp_idx[rows]] = out

pred[insp_mask] = (a * pred[insp_mask] + b).astype(np.float32)

pred = np.clip(pred, pmin, pmax)

pred[insp_mask] = _snap_to_levels(pred[insp_mask])

pred_s = pd.Series(pred)
pred = (
    pred_s.groupby(test["breath_id"], sort=False)
    .ffill()
    .fillna(pred_s)  # safety for any all-NaN group (shouldn't happen)
    .to_numpy(np.float32)
)



## === cell 4
sub = sample.copy()
assert len(sub) == len(test) == len(pred)

sub["pressure"] = pred.astype(np.float32)
sub.to_csv("submission.csv", index=False)

sub.head()



## === cell 5
submission = pred.copy()
submission_median = pred.copy()  # identical; placeholder to preserve semantics of cells
std_sub = float(np.std(submission))
std_min = float(np.mean(submission) - std_sub)
std_max = float(np.mean(submission) + std_sub)
submission_clip = np.clip(submission, std_min, std_max)

sub2 = sample.copy()
sub2["pressure"] = submission_median.astype(np.float32)
sub2.to_csv("submission_median.csv", index=False)

sub3 = sample.copy()
sub3["pressure"] = submission_clip.astype(np.float32)
sub3.to_csv("submission_clip.csv", index=False)

(std_sub, std_min, std_max, sub.tail())
