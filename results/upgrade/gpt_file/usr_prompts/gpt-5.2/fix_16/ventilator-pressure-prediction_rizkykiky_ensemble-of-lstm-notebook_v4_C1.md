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

0.1521573353916013

# 6. Current score

4.05525

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.02313) has done: 'Your notebook fails because it tries to read six external submission files that do not exist in this Kaggle environment, so none of the `sub*` arrays are created and every later cell errors. I keep the ensemble logic intact, but add a safe fallback: if those external submissions aren’t available, we generate six reasonable baseline prediction arrays from the provided `train.csv` (simple per-(R,C,time_step) median pressure lookup with a global fallback), so the pipeline runs end-to-end and writes a valid `submission.csv`. This is score-improving versus all-zeros while staying lightweight and within the competition’s rules. I also preserve output file names and ensure the submission length matches `sample_submission.csv`.'
- What this solution (achieved 1.66907) has done: 'Your fallback baseline is currently leaving a lot of score on the table because it ignores the most predictive structure in this competition: the within-breath dynamics and the fact that evaluation only scores inspiratory steps (`u_out==0`). I keep your ensemble structure intact (still producing `sub1..sub6` and using your mean/median/robust combiner), but improve the fallback predictions by adding lightweight time-series features that don’t change the modeling approach: cumulative inspired volume (`u_in` integrated over time), lagged `u_in/u_out`, and discretized `(R,C,time_step)` conditioning. I also fit the lookup only on inspiratory rows and set expiratory predictions to a stable baseline, which typically reduces MAE on the scored region without any heavy training. These are minimal changes confined to the fallback block, and the notebook still write the same three submission CSVs.'
- What this solution (achieved 1.96267) has done: 'Your current fallback already runs end-to-end, but it likely scores poorly because it bins `time_step` too coarsely and doesn’t exploit the strongest “cheap” signal: the within-breath cumulative volume (`u_in` integrated over `dt`) paired with `(R,C)` at the exact step. I keep your ensemble structure unchanged (still producing `sub1..sub6` and the same mean/median/robust combiners), and only improve the fallback lookup by (1) using an exact discrete `step` index per breath (0–79) instead of `ts_bin`, and (2) using a slightly finer cumulative-volume bin so keys match better without exploding memory. I also make expiratory predictions use a per-(R,C,step) inspiratory median rather than a global median, which usually reduces leakage of inspiratory distribution shifts into expiratory rows while keeping semantics stable. These are minimal edits confined to the fallback block and should move MAE down toward your target.'
- What this solution (achieved 1.96272) has done: 'Your current fallback baseline is still far from the target MAE, so we make a minimal-but-impactful improvement that stays within your “lookup/ensemble” core logic: snap all predictions to the discrete set of pressure values seen in training (the true target is quantized in this competition). This doesn’t change your approach (still builds `sub1..sub6` from a lightweight median-lookup), but it usually yields a large MAE drop because it removes impossible intermediate values produced by medians/merges. We also make the within-breath `step` creation robust by using integer `time_step*100` ranking (avoids any edge cases from float sorting ties) while keeping the same semantics. The rest of your ensemble/robust combiner and output files remain identical.'
- What this solution (achieved 1.97271) has done: 'Your current fallback is already doing a reasonable “median lookup + snap to pressure grid”, but it’s still far from the target because it doesn’t use the strongest cheap signal available without changing your approach: conditioning on the current `u_in` (and a short history) at each step. I keep the same ensemble structure (`sub1..sub6` with eps jitter) and the same merge/median-lookup logic, but extend the key to include binned `u_in` (and a 2-step lag) with a controlled backoff chain so coverage stays high. This should reduce MAE substantially (toward your target) while preserving your core logic and still writing the same three submission CSVs. I also keep the snapping-to-grid step intact.'
- What this solution (achieved 1.97271) has done: 'Your current MAE (1.97271) is far above the target (0.15216), so we need a meaningful but still “lookup-based” improvement without changing your ensemble/combiner structure. The biggest issue is that the fallback keying is too strict (low coverage) and it doesn’t exploit the known deterministic pressure mapping for `(R, C)` using the provided simulator equation on inspiratory steps; we can still keep your median-lookup as the core, but add a minimal physics-based prediction as a higher-quality backoff before falling to coarse medians. Concretely: compute per-breath integrated volume `u_in_cum`, estimate flow `u_in`, and derive pressure `p = R*flow + vol/C + PEEP`, where `PEEP` is estimated from training as the median initial pressure at `step==0` per `(R,C)`; then snap to the training pressure grid as you already do. This only touches the fallback generation of `sub1..sub6`, keeps the same downstream ensemble outputs/files, and should move MAE substantially toward your target.'
- What this solution (achieved 1.76169) has done: 'Your current fallback is already producing a valid submission, but the score is far from the target, so we need a higher-quality (still lightweight) prediction inside the same lookup/ensemble structure. I keep your downstream ensemble/combiner cells unchanged and only adjust the fallback feature engineering + lookup: (1) compute within-breath `step` deterministically without relying on float quirks, (2) use a more faithful “physics-style” backoff by estimating flow from `u_in` change (not raw `u_in`) and using a per-(R,C) PEEP, and (3) improve key coverage by adding a backoff that includes `u_out` state and a slightly smoother binning for `u_in_cum`. These changes keep the same semantics (median lookups + physics backoff + snap-to-grid) but should substantially reduce MAE toward your target without introducing heavy models or changing the ensemble logic.'
- What this solution (achieved 1.76169) has done: 'Your current fallback prediction is dominated by an inaccurate “physics” backoff and a too-fragmented lookup, which keeps MAE far above the target. I keep your ensemble structure and robust combiner unchanged, but make the smallest effective change: replace the physics backoff with a *data-driven* backoff chain that uses per-(R,C,step) medians and a per-(R,C) PEEP estimate (step==0) as the final fallback. I also ensure expiratory (`u_out==1`) predictions use the stable PEEP (since expiratory isn’t scored, this avoids injecting noise without changing evaluation semantics). Finally, I keep your snapping-to-training-pressure-grid step, which is important for this competition.'
- What this solution (achieved 5.0903) has done: 'Your current fallback is already stable but still too coarse for this competition’s quantized target; the biggest low-risk improvement is to add a final “pressure-grid snapping + (R,C)-conditioned value correction” that only adjusts predictions to the most likely discrete pressures seen for each (R,C,step). I keep your ensemble structure (`sub1..sub6` with eps jitter) and your existing backoff chain intact, and only add a lightweight correction table learned from training that maps any raw predicted pressure to the nearest *valid* pressure for that (R,C,step) context. This preserves your lookup-based core logic and evaluation semantics while typically lowering MAE substantially versus global snapping alone. The rest of the notebook (mean/median/robust combiner and output CSVs) remains unchanged.'
- What this solution (achieved 1.76169) has done: 'Your current score (5.0903 MAE, lower is better) is far worse than the target (0.1522), so we should make a small but high-impact correction inside your existing lookup/ensemble pipeline. The biggest low-risk issue is that the evaluation only scores inspiratory rows (`u_out==0`), but your lookup is trained only on inspiratory while test expiratory rows are forced to PEEP; we can improve inspiratory accuracy by adding a final within-(R,C,step) calibration that maps your already-snapped prediction to the most likely pressure *for that exact (R,C,step)* using a fast nearest-neighbor over the per-step unique pressure set (not just top-8). This keeps your core logic intact (median/backoff lookups + snapping + ensemble combiner), but reduces quantization mismatch errors that can dominate MAE. Changes are confined to the fallback block; submission writing and filenames stay the same.'
- What this solution (achieved 3.97465) has done: 'Your current score is much worse than the target (lower is better), so we should make a small but high-impact improvement while keeping your “lookup/ensemble + snapping” core logic intact. The biggest issue is that the fallback never uses the single most predictive raw signal per step (`u_in` and its recent history) in a *continuous* way; using hard bins can fragment the lookup and cause frequent backoff to coarse medians. I keep your exact backoff chain structure and outputs, but (1) replace the coarse rounding bins with quantile-based bins learned from train (stable coverage), and (2) add a final lightweight per-(R,C,step,u_in_bin) median correction applied only on inspiratory rows before snapping-to-grid (preserves your discrete-output semantics). This should reduce MAE materially without changing model class, training loops, loss, or submission format.'
- What this solution (achieved 4.60026) has done: 'Your current MAE is far above the target (lower is better), and the biggest issue is that the fallback prediction is still a brittle lookup that often backs off to coarse medians. I keep your ensemble/combiner logic exactly the same, but make two minimal, high-impact changes inside the fallback: (1) build a direct per-(R,C,step,u_in) median table using a small integer bin for `u_in` (0–1000) to preserve continuity without exploding keys, and (2) apply a monotone per-(R,C,step) calibration that maps your snapped prediction to the nearest *most frequent* pressure for that (R,C,step) context (stronger than a simple “any-unique” nearest set). These changes keep the same “lookup + snap-to-grid” semantics, don’t add any training loops/models, and should materially reduce MAE toward your target while staying lightweight and deterministic.'
- What this solution (achieved 4.60026) has done: 'Your current score (4.60026 MAE; lower is better) is far from the target (0.15216), so we need a small but meaningful improvement without changing your ensemble/combiner core. The biggest correctness issue is that you are overfitting to absolute `step` and within-breath features but not using the most important deterministic lung-attribute structure: pressure trajectories differ strongly by `(R,C)` and are very consistent per step during inspiration. I keep your existing backoff chain and snapping logic, but add one minimal, high-impact refinement: compute a per-(R,C,step,u_out) *mean* pressure table (not median) and insert it early in the fallback chain to reduce bias and improve coverage where your high-dimensional keys miss. This is a lightweight change confined to the fallback generation, preserves the same “lookup ensemble + robust combine” semantics, and still writes the same three submission CSVs.'
- What this solution (achieved 4.60026) has done: 'I keep your ensemble/combiner cells unchanged and only adjust the fallback prediction block to better match the competition’s scored region (inspiration only) while preserving your lookup-based core. The main minimal fix is to stop forcing expiratory (`u_out==1`) predictions to PEEP (even though not scored) because it disrupts within-breath continuity and can indirectly hurt inspiratory calibration in your later per-step “top pressure set” correction; instead we use the best available per-(R,C,step,u_out) mean/median for expiratory as well. I also make the final “top pressure set” correction apply only on inspiratory rows (where the metric is computed) to avoid any unintended side effects, while keeping your global pressure-grid snapping intact. These are small, localized changes expected to reduce MAE (move down from 4.60026 toward your 0.152 target) without changing the overall pipeline structure or output format.'
- What this solution (achieved 4.05525) has done: 'Your current fallback builds many high-dimensional lookup tables from 5.4M train rows, which is slow and (more importantly) brittle: lots of missing keys cause frequent backoff to coarse medians, keeping MAE high. I keep your overall ensemble/combiner logic identical (still creating `sub1..sub6` and writing the same three submission files), but make two minimal changes inside the fallback: (1) build the lookup tables using only the inspiratory rows and only the columns needed for each table (reduces noise and makes grouping more consistent with the scored region), and (2) replace the slow per-row Python nearest-neighbor “top pressure set” correction with a vectorized, learned per-(R,C,step) quantization mapping from the raw snapped prediction to the most frequent pressure in that context (same intent, much more stable and faster). These changes should materially reduce your MAE from 4.60026 toward the target while staying within your lookup-based core approach and finishing under the time limit. The submission format and file names remain unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)



## === cell 1
SUB_PATHS = [
    "../input/improvement-base-on-tensor-bidirect-lstm-0-173/submission.csv",
    "../input/ventilator-pressure-eda-lstm-0-189/lstm.csv",
    "../input/tensorflow-bidirectional-lstm-custom-mae-loss/submission.csv",
    "../input/rescaling-layer-for-discrete-output-in-tensorflow/submission.csv",
    "../input/dnn-lstm-tpu/submission.csv",
    "../input/dnn-lstm-tpu/median_submission.csv",
]


def _try_load_pressure(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "pressure" not in df.columns:
            raise ValueError(f"'pressure' column not found in {path}")
        return df["pressure"].to_numpy()
    return None


loaded = [_try_load_pressure(p) for p in SUB_PATHS]
all_present = all(x is not None for x in loaded)

sample = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")
n_test = len(sample)

if all_present:
    for i, arr in enumerate(loaded, start=1):
        if len(arr) != n_test:
            raise ValueError(
                f"Submission {i} length {len(arr)} != sample length {n_test}"
            )
    sub1, sub2, sub3, sub4, sub5, sub6 = loaded
else:
    train_path = "../input/ventilator-pressure-prediction/train.csv"
    test_path = "../input/ventilator-pressure-prediction/test.csv"

    usecols_train = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    usecols_test = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "id"]
    train = pd.read_csv(train_path, usecols=usecols_train)
    test = pd.read_csv(test_path, usecols=usecols_test)

    def _add_dyn_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["_ts_key"] = (df["time_step"].astype(np.float32) * 1000.0 + 1e-3).astype(
            np.int16
        )
        df = df.sort_values(["breath_id", "_ts_key"], kind="mergesort")
        df["step"] = df.groupby("breath_id", sort=False).cumcount().astype(np.int16)

        df["dt"] = (
            df.groupby("breath_id", sort=False)["time_step"]
            .diff()
            .fillna(0.0)
            .astype(np.float32)
        )
        df["u_in_dt"] = (df["u_in"].astype(np.float32) * df["dt"]).astype(np.float32)
        df["u_in_cum"] = (
            df.groupby("breath_id", sort=False)["u_in_dt"].cumsum().astype(np.float32)
        )

        df["u_in_lag1"] = (
            df.groupby("breath_id", sort=False)["u_in"]
            .shift(1)
            .fillna(0.0)
            .astype(np.float32)
        )
        df["u_in_lag2"] = (
            df.groupby("breath_id", sort=False)["u_in"]
            .shift(2)
            .fillna(0.0)
            .astype(np.float32)
        )

        df["du_in"] = (df["u_in"].astype(np.float32) - df["u_in_lag1"]).astype(
            np.float32
        )
        df["du_in_lag1"] = (
            df.groupby("breath_id", sort=False)["du_in"]
            .shift(1)
            .fillna(0.0)
            .astype(np.float32)
        )

        df["u_out_lag1"] = (
            df.groupby("breath_id", sort=False)["u_out"]
            .shift(1)
            .fillna(0)
            .astype(np.int16)
        )

        return df.drop(columns=["_ts_key"])

    train = _add_dyn_features(train)
    test = _add_dyn_features(test)

    train_insp = train[train["u_out"] == 0].copy()

    def _make_quantile_bins(values: np.ndarray, n_bins: int):
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(values, qs)
        edges = np.unique(edges.astype(np.float32))
        if edges.size < 2:
            edges = np.array([values.min(), values.max() + 1e-3], dtype=np.float32)
        return edges

    def _to_bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
        return np.clip(
            np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2
        ).astype(np.int16)

    u_in_edges = _make_quantile_bins(train_insp["u_in"].to_numpy(np.float32), n_bins=64)
    u_in_l1_edges = _make_quantile_bins(
        train_insp["u_in_lag1"].to_numpy(np.float32), n_bins=64
    )
    u_in_l2_edges = _make_quantile_bins(
        train_insp["u_in_lag2"].to_numpy(np.float32), n_bins=64
    )
    du_in_edges = _make_quantile_bins(
        train_insp["du_in"].to_numpy(np.float32), n_bins=64
    )
    u_in_cum_edges = _make_quantile_bins(
        train_insp["u_in_cum"].to_numpy(np.float32), n_bins=128
    )

    for df in (train, test):
        df["u_in_bin"] = _to_bin(df["u_in"].to_numpy(np.float32), u_in_edges)
        df["u_in_lag1_bin"] = _to_bin(
            df["u_in_lag1"].to_numpy(np.float32), u_in_l1_edges
        )
        df["u_in_lag2_bin"] = _to_bin(
            df["u_in_lag2"].to_numpy(np.float32), u_in_l2_edges
        )
        df["du_in_bin"] = _to_bin(df["du_in"].to_numpy(np.float32), du_in_edges)
        df["du_in_lag1_bin"] = _to_bin(
            df["du_in_lag1"].to_numpy(np.float32), du_in_edges
        )
        df["u_in_cum_bin"] = _to_bin(
            df["u_in_cum"].to_numpy(np.float32), u_in_cum_edges
        ).astype(np.int32)

        df["u_in_01"] = np.clip(
            np.rint(df["u_in"].to_numpy(np.float32) * 10.0), 0.0, 1000.0
        ).astype(np.int16)

    train_insp = train[train["u_out"] == 0].copy()

    key_cols = [
        "R",
        "C",
        "step",
        "u_in_bin",
        "u_in_lag1_bin",
        "u_in_lag2_bin",
        "u_in_cum_bin",
    ]
    med_by_key = (
        train_insp.groupby(key_cols, sort=False)["pressure"]
        .median()
        .rename("p_med")
        .reset_index()
    )

    key_du = ["R", "C", "step", "u_in_bin", "du_in_bin", "u_in_cum_bin"]
    med_by_key_du = (
        train_insp.groupby(key_du, sort=False)["pressure"]
        .median()
        .rename("p_med_du")
        .reset_index()
    )

    med_by_key_b1 = (
        train_insp.groupby(
            ["R", "C", "step", "u_in_bin", "u_in_lag1_bin", "u_in_cum_bin"], sort=False
        )["pressure"]
        .median()
        .rename("p_med_b1")
        .reset_index()
    )

    med_by_key_b2 = (
        train_insp.groupby(["R", "C", "step", "u_in_bin", "u_in_cum_bin"], sort=False)[
            "pressure"
        ]
        .median()
        .rename("p_med_b2")
        .reset_index()
    )

    med_by_key2 = (
        train_insp.groupby(["R", "C", "step", "u_in_cum_bin"], sort=False)["pressure"]
        .median()
        .rename("p_med2")
        .reset_index()
    )

    med_by_key3 = (
        train_insp.groupby(["R", "C", "step"], sort=False)["pressure"]
        .median()
        .rename("p_med3")
        .reset_index()
    )

    med_by_rc = (
        train_insp.groupby(["R", "C"], sort=False)["pressure"]
        .median()
        .rename("p_med_rc")
        .reset_index()
    )
    med_by_step = (
        train_insp.groupby(["step"], sort=False)["pressure"]
        .median()
        .rename("p_med_step")
        .reset_index()
    )

    global_median_insp = float(train_insp["pressure"].median())

    test = test.merge(med_by_key, on=key_cols, how="left")
    test = test.merge(med_by_key_du, on=key_du, how="left")
    test = test.merge(
        med_by_key_b1,
        on=["R", "C", "step", "u_in_bin", "u_in_lag1_bin", "u_in_cum_bin"],
        how="left",
    )
    test = test.merge(
        med_by_key_b2, on=["R", "C", "step", "u_in_bin", "u_in_cum_bin"], how="left"
    )
    test = test.merge(med_by_key2, on=["R", "C", "step", "u_in_cum_bin"], how="left")
    test = test.merge(med_by_key3, on=["R", "C", "step"], how="left")

    test = test.merge(med_by_rc, on=["R", "C"], how="left")
    test = test.merge(med_by_step, on=["step"], how="left")

    peep_by_rc = (
        train_insp.loc[train_insp["step"] == 0]
        .groupby(["R", "C"], sort=False)["pressure"]
        .median()
        .rename("peep_rc")
        .reset_index()
    )
    test = test.merge(peep_by_rc, on=["R", "C"], how="left")
    peep_global = float(train_insp.loc[train_insp["step"] == 0, "pressure"].median())

    mean_rc_step_uout = (
        train.groupby(["R", "C", "step", "u_out"], sort=False)["pressure"]
        .mean()
        .rename("p_mean_rc_step_uout")
        .reset_index()
    )
    test = test.merge(mean_rc_step_uout, on=["R", "C", "step", "u_out"], how="left")

    med_rc_step_uout = (
        train.groupby(["R", "C", "step", "u_out"], sort=False)["pressure"]
        .median()
        .rename("p_med_rc_step_uout")
        .reset_index()
    )
    test = test.merge(med_rc_step_uout, on=["R", "C", "step", "u_out"], how="left")

    base_pred = (
        test["p_med"]
        .fillna(test["p_med_du"])
        .fillna(test["p_med_b1"])
        .fillna(test["p_med_b2"])
        .fillna(test["p_med2"])
        .fillna(test["p_med3"])
        .fillna(test["p_med_rc_step_uout"])
        .fillna(test["p_mean_rc_step_uout"])
        .fillna(test["p_med_rc"])
        .fillna(test["p_med_step"])
        .fillna(test["peep_rc"])
        .fillna(peep_global)
        .fillna(global_median_insp)
        .to_numpy(dtype=np.float32)
    )

    corr_tbl_fine = (
        train_insp.groupby(["R", "C", "step", "u_in_01"], sort=False)["pressure"]
        .median()
        .rename("p_corr_uin_fine")
        .reset_index()
    )
    test = test.merge(corr_tbl_fine, on=["R", "C", "step", "u_in_01"], how="left")
    insp_mask = test["u_out"].to_numpy() == 0
    p_corr_fine = test["p_corr_uin_fine"].to_numpy(np.float32)
    use_corr_fine = insp_mask & ~np.isnan(p_corr_fine)
    base_pred[use_corr_fine] = p_corr_fine[use_corr_fine]

    corr_tbl = (
        train_insp.groupby(["R", "C", "step", "u_in_bin"], sort=False)["pressure"]
        .median()
        .rename("p_corr_uin")
        .reset_index()
    )
    test = test.merge(corr_tbl, on=["R", "C", "step", "u_in_bin"], how="left")
    p_corr = test["p_corr_uin"].to_numpy(np.float32)
    use_corr = insp_mask & ~np.isnan(p_corr) & ~use_corr_fine
    base_pred[use_corr] = p_corr[use_corr]

    pressure_grid = np.sort(train["pressure"].unique()).astype(np.float32)
    idx = np.searchsorted(pressure_grid, base_pred, side="left")
    idx = np.clip(idx, 0, len(pressure_grid) - 1)
    prev_idx = np.clip(idx - 1, 0, len(pressure_grid) - 1)
    choose_prev = np.abs(base_pred - pressure_grid[prev_idx]) <= np.abs(
        base_pred - pressure_grid[idx]
    )
    base_pred = np.where(
        choose_prev, pressure_grid[prev_idx], pressure_grid[idx]
    ).astype(np.float32)

    train_insp_snap = train_insp[["R", "C", "step", "pressure"]].copy()
    p = train_insp_snap["pressure"].to_numpy(np.float32)
    j = np.searchsorted(pressure_grid, p, side="left")
    j = np.clip(j, 0, len(pressure_grid) - 1)
    jprev = np.clip(j - 1, 0, len(pressure_grid) - 1)
    choose_prev_p = np.abs(p - pressure_grid[jprev]) <= np.abs(p - pressure_grid[j])
    p_snap = np.where(choose_prev_p, pressure_grid[jprev], pressure_grid[j]).astype(
        np.float32
    )
    train_insp_snap["p_snap"] = p_snap

    map_tbl = (
        train_insp_snap.groupby(["R", "C", "step", "p_snap", "pressure"], sort=False)
        .size()
        .rename("cnt")
        .reset_index()
        .sort_values(
            ["R", "C", "step", "p_snap", "cnt"],
            ascending=[True, True, True, True, False],
        )
    )
    map_tbl = map_tbl.drop_duplicates(["R", "C", "step", "p_snap"], keep="first")[
        ["R", "C", "step", "p_snap", "pressure"]
    ].rename(columns={"pressure": "p_map"})
    test_map = test[["R", "C", "step"]].copy()
    test_map["p_snap"] = base_pred
    test_map = test_map.merge(map_tbl, on=["R", "C", "step", "p_snap"], how="left")
    p_map = test_map["p_map"].to_numpy(np.float32)
    use_map = insp_mask & ~np.isnan(p_map)
    base_pred[use_map] = p_map[use_map].astype(np.float32)

    eps = 1e-6
    sub1 = base_pred + 0 * eps
    sub2 = base_pred + 1 * eps
    sub3 = base_pred - 1 * eps
    sub4 = base_pred + 2 * eps
    sub5 = base_pred - 2 * eps
    sub6 = base_pred + 3 * eps

    if not (test["id"].to_numpy() == sample["id"].to_numpy()).all():
        order = pd.Series(np.arange(len(test)), index=test["id"].to_numpy())
        idx = order.loc[sample["id"].to_numpy()].to_numpy()
        sub1, sub2, sub3, sub4, sub5, sub6 = [
            arr[idx] for arr in (sub1, sub2, sub3, sub4, sub5, sub6)
        ]




## === cell 2
def better_than_median(inputs, axis):
    """Compute the mean of the predictions if there are no outliers,
    or the median if there are outliers.

    Parameter: inputs = ndarray of shape (n_samples, n_folds)"""
    spread = inputs.max(axis=axis) - inputs.min(axis=axis)
    spread_lim = 0.45
    print(f"Inliers:  {(spread < spread_lim).sum():7} -> compute mean")
    print(f"Outliers: {(spread >= spread_lim).sum():7} -> compute median")
    print(f"Total:    {len(inputs):7}")
    return np.where(
        spread < spread_lim, np.mean(inputs, axis=axis), np.median(inputs, axis=axis)
    )




## === cell 3
submission = np.mean([sub1, sub2, sub3, sub4, sub5, sub6], axis=0)
submission



## === cell 4
sample = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")
sample



## === cell 5
sample["pressure"] = submission
sample.to_csv("submission_mean.csv", index=False)
sample



## === cell 6
submission_median = np.median([sub1, sub2, sub3, sub4, sub5, sub6], axis=0)
submission_median



## === cell 7
sample["pressure"] = submission_median
sample.to_csv("submission_median.csv", index=False)
sample



## === cell 8
new = better_than_median(np.column_stack((sub1, sub2, sub3, sub4, sub5, sub6)), axis=1)
print(new)



## === cell 9
sample["pressure"] = new
sample.to_csv("submission.csv", index=False)

print(sample.head())
print("Wrote submission.csv with shape:", sample.shape)
print("Columns:", list(sample.columns))
print("pressure stats:", pd.Series(sample["pressure"]).describe())
