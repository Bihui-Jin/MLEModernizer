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

1.76169

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

        df["u_in_cum_bin"] = (df["u_in_cum"] * 10.0).round().astype(np.int32)  # ~0.1

        df["u_in_bin"] = (df["u_in"].astype(np.float32) * 2.0).round().astype(np.int16)
        df["u_in_lag1_bin"] = (df["u_in_lag1"] * 2.0).round().astype(np.int16)
        df["u_in_lag2_bin"] = (df["u_in_lag2"] * 2.0).round().astype(np.int16)

        df["du_in_bin"] = (df["du_in"] * 2.0).round().astype(np.int16)
        df["du_in_lag1_bin"] = (df["du_in_lag1"] * 2.0).round().astype(np.int16)

        return df.drop(columns=["_ts_key"])

    train = _add_dyn_features(train)
    test = _add_dyn_features(test)

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

    base_pred = (
        test["p_med"]
        .fillna(test["p_med_du"])
        .fillna(test["p_med_b1"])
        .fillna(test["p_med_b2"])
        .fillna(test["p_med2"])
        .fillna(test["p_med3"])
        .fillna(test["p_med_rc"])
        .fillna(test["p_med_step"])
        .fillna(test["peep_rc"])
        .fillna(peep_global)
        .fillna(global_median_insp)
        .to_numpy(dtype=np.float32)
    )

    exp_mask = test["u_out"].to_numpy() == 1
    if exp_mask.any():
        exp_fill = (
            test.loc[exp_mask, "peep_rc"].fillna(peep_global).to_numpy(dtype=np.float32)
        )
        base_pred[exp_mask] = exp_fill

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

    train_rcs = train_insp[["R", "C", "step", "pressure"]].drop_duplicates()
    pset = (
        train_rcs.groupby(["R", "C", "step"], sort=False)["pressure"]
        .apply(lambda s: np.sort(s.to_numpy(dtype=np.float32)))
        .rename("p_set")
        .reset_index()
    )
    test_pset = test[["R", "C", "step"]].merge(pset, on=["R", "C", "step"], how="left")

    def _nearest_from_sorted_set(val: float, arr):
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            return val
        j = int(np.searchsorted(arr, val, side="left"))
        if j <= 0:
            return float(arr[0])
        if j >= arr.size:
            return float(arr[-1])
        prev_v = arr[j - 1]
        next_v = arr[j]
        return float(prev_v if abs(val - prev_v) <= abs(val - next_v) else next_v)

    corrected = np.fromiter(
        (
            _nearest_from_sorted_set(v, a)
            for v, a in zip(base_pred, test_pset["p_set"].values)
        ),
        dtype=np.float32,
        count=len(base_pred),
    )
    if exp_mask.any():
        corrected[exp_mask] = base_pred[exp_mask]
    base_pred = corrected

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
