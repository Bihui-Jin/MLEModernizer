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

0.137018805076577

# 6. Current score

8.46542

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.92097) has done: 'The crash happens because `g()` assumes there are prediction files in `../input/gb-data-blending-recover`, but in your environment that directory doesn’t exist (so it silently builds empty arrays and ends up producing a single scalar median). I add strict checks so `g()` fails early with a clear message if no valid prediction CSVs are found, and I implement a safe fallback that still produces a valid submission by generating a baseline prediction from the provided `test.csv` using the median training pressure per `(R, C, time_step)` (and then snapping to the nearest allowed pressure, preserving your `find_nearest` core calibration step). This keeps the overall semantics (predict pressures and quantize to known pressure levels) and ensures a `.csv` submission is always written end-to-end. I also keep paths compatible with Kaggle’s `/kaggle/input/...` layout while not changing your existing relative paths.'
- What this solution (achieved 9.5126) has done: 'Your current score (9.92097 MAE) is far from the target (0.1370), so the issue is that you’re effectively submitting a very weak fallback instead of a meaningful model/blend. To move the score sharply toward the target while preserving your core “predict then snap to nearest allowed pressure” logic, I keep `g()` and the quantization exactly as-is but upgrade the fallback to a stronger, still-simple lookup: median pressure by `(R, C, u_out, time_step)` plus a cumulative-`u_in` correction per `(R, C, u_out)` computed from train. This remains a deterministic train-derived baseline (no new model/training loop) and should substantially reduce MAE versus the current coarse median-by-time_step only. The script still write `submission.csv` end-to-end even when the external blend directory is missing.'
- What this solution (achieved 9.90136) has done: 'Your current MAE (9.5126, lower-is-better) is still very far from the target (0.1370), which strongly suggests the code is nearly always falling back (no external blend files) and the fallback predictor is too weak. To move sharply toward the target while preserving your “lookup-based prediction + snap to nearest allowed pressure” core semantics, I keep `g()`/`wc()`/`find_nearest()` intact and only strengthen the fallback using a deterministic, train-derived KNN-by-key lookup that matches the metric better. Concretely, I replace the fallback with a group-median keyed on `(R,C,u_out,u_in_bin,time_step)` (binning `u_in` is a minimal, non-model change) and keep the final quantization step unchanged. This should materially reduce MAE without introducing any new model architecture/training loop and still always write a valid `submission.csv`.'
- What this solution (achieved 9.90852) has done: 'Your current score is far worse than the target (lower-is-better), and the main reason is that your blend directory is missing so you always fall back to a weak lookup. I keep your overall “train-derived lookup prediction + snap to nearest allowed pressure” semantics intact, but strengthen the fallback to use breath-dynamics features that match the metric better: cumulative `u_in`, lagged `u_in`, and lagged pressure as additional lookup keys, with a safe backoff ladder to coarser keys. This is still a deterministic, non-model, group-median-based predictor (no new training loop/architecture), but it captures how pressure evolves within each breath much more accurately than static binning alone. The rest of your blending/quantization logic stays the same, and the script still always write a valid `submission.csv` end-to-end.'
- What this solution (achieved 7.30803) has done: 'Your score is far worse than the target (lower-is-better), and the main cause is that the external blending directory is missing so you always fall back to the lookup baseline. Keeping your core logic (lookup → predict → snap to nearest allowed pressure) intact, the smallest high-impact improvement is to compute the lookup only on the inspiratory phase (`u_out==0`), because that is the only phase scored, and expiratory dynamics otherwise pollute the medians. I also add a final safety merge so the submission `id` ordering always matches `sample_submission.csv`, preventing any silent misalignment. These changes keep your overall approach identical (deterministic medians + quantization) but should materially reduce MAE toward the target.'
- What this solution (achieved 7.57944) has done: 'Your score is far worse than the target (lower-is-better), and your code is still falling back to the deterministic lookup baseline because `../input/gb-data-blending-recover` isn’t available. The current fallback is polluted by training on inspiratory-only but still predicting expiratory timesteps with inspiratory medians; we can move the MAE toward the target by forcing physically-correct behavior on expiratory phase (set predictions to 0 when `u_out==1`), since those timesteps are not scored and pressure rapidly drops there. I keep your core logic (lookup medians + backoff ladder + `find_nearest` quantization) and only add this minimal post-processing based on `u_out`, plus a small safety alignment to ensure test `id` ordering is correct. This should materially reduce error without changing model/training approach or introducing any new modeling.'
- What this solution (achieved 7.31289) has done: 'Your current MAE (7.57944, lower-is-better) is still far from the target (0.1370), and since the external blend directory is missing you’re always using the fallback; the fastest way to move the score toward the target without changing the overall “lookup → predict → snap-to-grid” core logic is to make the fallback more faithful to the underlying discrete pressure dynamics. I keep your lookup/backoff ladder and `find_nearest()` quantization, but change the fallback’s expiratory handling to be physically consistent (hold last inspiratory pressure during `u_out==1` rather than forcing to 0), and I add a tiny, still-deterministic calibration step that maps the coarse median prediction to the nearest *mode* pressure observed in train for the same `(R,C,time_step,u_in_bin)` key (reduces MAE vs raw median in this discrete-label task). I also keep the strict id alignment with `sample_submission.csv` to avoid any silent row-order issues. These are minimal changes that should substantially reduce error from the fallback while preserving the same overall approach and runtime constraints.'
- What this solution (achieved 2.13782) has done: 'Your current MAE is far above the target, and the biggest controllable issue in this notebook is that `g()` almost always falls back because `../input/gb-data-blending-recover` is missing; so the score is dominated by the fallback baseline. To move the score materially toward the target while preserving your core “lookup-based prediction → snap to nearest allowed pressure” logic, I strengthen the fallback in a minimal way by adding one more physics-aligned feature: a binned within-breath time index (`t_idx`), which avoids float merge mismatches on `time_step` and better captures sequence position. I also make the inspiratory-only LUT construction explicit and ensure that expiratory predictions are always forward-filled from the last inspiratory prediction (not scored, but stabilizes transitions) while keeping your final `find_nearest` quantization unchanged. All paths and outputs are kept the same, and the script still writes a valid `submission.csv` end-to-end.'
- What this solution (achieved 8.46538) has done: 'Your current score (2.13782 MAE, lower is better) is still far above the target, so we should improve the fallback path (since `../input/gb-data-blending-recover` is missing and you’re almost certainly using the fallback). Keeping your core “lookup-based prediction → backoff ladder → snap to nearest allowed pressure” logic intact, I make the fallback LUT more faithful to the true sequence by using the fixed 80-step structure explicitly (integer step index from time ordering, not float time), and I build LUTs on inspiratory only while ensuring expiratory predictions are forward-filled within each breath. The key score-improver (still lookup/median-based) is adding a high-impact physics proxy `area` (cumulative u_in * dt) and including it (binned) in the lookup/backoff ladder, which commonly helps this competition without introducing a new model or training loop. I also fix the `id` range issue by always reading/writing in the exact order of `sample_submission.csv`, preventing any silent misalignment.'
- What this solution (achieved 8.46542) has done: 'Your current MAE is still far above the target (lower is better), and the main lever available (without changing your blending core logic) is improving the deterministic fallback that gets used when `../input/gb-data-blending-recover` is missing. I keep your overall “lookup/backoff ladder → expiratory forward-fill → snap to nearest allowed pressure” semantics, but make the LUT significantly more precise by (1) using the exact 80-step structure explicitly and (2) adding an additional train-derived dynamic proxy (`u_in_rate = du_in/dt`, binned) into the high-granularity keys, with a safe backoff when missing. This is still purely a group-median lookup (no new model/training loop/loss) and should reduce the gap toward the target while remaining stable and within runtime. I also ensure strict `id` alignment with `sample_submission.csv` remains intact so submission ordering cannot silently degrade score.'
- What this solution (achieved 8.46542) has done: 'Your current MAE (8.465) is still far worse than the target (0.137, lower-is-better), and since the external blend folder is missing, your score is dominated by the fallback lookup. The biggest issue in the fallback is that it uses the test `id` column (which repeats 1..2000) as a join key; this creates a many-to-many merge that silently corrupts predictions and explodes MAE. I make the minimal fix: keep your lookup/backoff/quantization logic intact, but replace all merges on `id` with merges on a unique per-row key (`row_id` from the dataframe index) and preserve final alignment by using `sample_submission.csv` order. This should dramatically reduce MAE toward your target without changing the overall approach.'

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


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    """
    Reads one or two prediction files and blends them based on their score encoded in filename.
    Original logic preserved; now includes minimal validation and robust parsing.
    """
    l = []
    arrs = []
    for i in range(len(input_list)):
        fn = input_list[i].split("/")[-1]
        try:
            public_lb_score = int(fn.split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1
        l.append(public_lb_score)

        dfp = pd.read_csv(input_list[i])
        if "pressure" not in dfp.columns:
            raise ValueError(f"File {input_list[i]} missing 'pressure' column.")
        arrs.append(dfp["pressure"].to_numpy().ravel())

    base_len = len(arrs[0])
    for j, a in enumerate(arrs):
        if len(a) != base_len:
            raise ValueError(
                f"Prediction length mismatch in {input_list[j]}: {len(a)} vs {base_len}"
            )

    output = 0
    l_sum = sum(l)

    if len(arrs) == 1:
        output = arrs[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        output = arrs[0] * weight1 + arrs[1] * weight2
    return output


def _add_breath_features(df):
    df = df.sort_values(["breath_id", "time_step"], kind="mergesort").copy()

    df["t_idx"] = df.groupby("breath_id", sort=False).cumcount().astype(np.int16)

    df["u_in_cum"] = (
        df.groupby("breath_id", sort=False)["u_in"].cumsum().astype(np.float32)
    )

    df["u_in_lag1"] = (
        df.groupby("breath_id", sort=False)["u_in"]
        .shift(1)
        .fillna(0.0)
        .astype(np.float32)
    )

    df["dt"] = (
        df.groupby("breath_id", sort=False)["time_step"]
        .diff()
        .fillna(0.0)
        .astype(np.float32)
    )

    df["area"] = (
        (df["u_in"] * df["dt"])
        .groupby(df["breath_id"], sort=False)
        .cumsum()
        .astype(np.float32)
    )

    du = (df["u_in"] - df["u_in_lag1"]).astype(np.float32)
    df["u_in_rate"] = np.where(
        df["dt"].to_numpy() > 0, du.to_numpy() / df["dt"].to_numpy(), 0.0
    ).astype(np.float32)

    return df


def _fallback_baseline_submission(out_path="submission.csv"):
    """
    Strong deterministic fallback (still lookup-based + quantization).

    Minimal score-critical fix:
    - The competition's test `id` repeats 1..2000 (not unique across rows). Any merge on `id`
      causes a many-to-many join that corrupts predictions and explodes MAE.
    - We therefore introduce a unique per-row key `row_id` (the dataframe index) and use it for
      internal merges; final output remains keyed by `id` to match sample_submission format.

    All other core logic (lookup/backoff ladder -> expiratory forward-fill -> find_nearest quantization)
    is preserved.
    """
    test_path = "../input/ventilator-pressure-prediction/test.csv"
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"

    df_test = pd.read_csv(test_path)
    sub = pd.read_csv(sample_path)

    df_test = df_test.reset_index(drop=True)
    df_test["row_id"] = df_test.index.astype(np.int32)

    tr_full = _add_breath_features(df_train)
    te = _add_breath_features(df_test)

    te["row_id"] = df_test["row_id"].to_numpy()

    tr = tr_full.loc[tr_full["u_out"] == 0].copy()
    if len(tr) == 0:
        tr = tr_full.copy()

    tr["p_lag1"] = (
        tr.groupby("breath_id", sort=False)["pressure"]
        .shift(1)
        .bfill()
        .fillna(tr["pressure"].median())
        .astype(np.float32)
    )

    BIN_UIN = 1.0
    BIN_UCUM = 5.0
    BIN_PLAG = float(np.diff(sorted_pressures).min())  # pressure step (~0.07)
    BIN_AREA = 0.5
    BIN_URATE = 50.0  # coarse bin for du_in/dt; stable and reduces sparsity

    for df in (tr, te):
        df["u_in_bin"] = (np.rint(df["u_in"].to_numpy() / BIN_UIN) * BIN_UIN).astype(
            np.float32
        )
        df["u_in_lag1_bin"] = (
            np.rint(df["u_in_lag1"].to_numpy() / BIN_UIN) * BIN_UIN
        ).astype(np.float32)
        df["u_in_cum_bin"] = (
            np.rint(df["u_in_cum"].to_numpy() / BIN_UCUM) * BIN_UCUM
        ).astype(np.float32)
        df["area_bin"] = (np.rint(df["area"].to_numpy() / BIN_AREA) * BIN_AREA).astype(
            np.float32
        )
        df["u_in_rate_bin"] = (
            np.rint(df["u_in_rate"].to_numpy() / BIN_URATE) * BIN_URATE
        ).astype(np.float32)

    tr["p_lag1_bin"] = (np.rint(tr["p_lag1"].to_numpy() / BIN_PLAG) * BIN_PLAG).astype(
        np.float32
    )

    global_med = float(tr["pressure"].median())

    key_coarse = ["R", "C", "u_out", "t_idx", "u_in_bin", "area_bin"]

    lut_coarse_med = (
        tr.groupby(key_coarse, sort=False)["pressure"]
        .median()
        .rename("p0_med")
        .reset_index()
    )

    def _mode_smallest(s: pd.Series) -> float:
        vc = s.value_counts()
        if vc.empty:
            return np.nan
        m = vc.max()
        return float(vc[vc == m].index.min())

    lut_coarse_mode = (
        tr.groupby(key_coarse, sort=False)["pressure"]
        .apply(_mode_smallest)
        .rename("p0_mode")
        .reset_index()
    )

    tem0 = te.merge(lut_coarse_med, on=key_coarse, how="left")
    tem0 = tem0.merge(lut_coarse_mode, on=key_coarse, how="left")

    p0 = tem0["p0_med"].to_numpy()
    p0_mode = tem0["p0_mode"].to_numpy()
    use_mode = ~np.isnan(p0_mode)
    p0 = p0.astype(np.float64)
    p0[use_mode] = p0_mode[use_mode]
    p0[np.isnan(p0)] = global_med

    te_pred0 = te[["breath_id", "row_id"]].copy()
    te_pred0["p0"] = p0.astype(np.float32)
    te_pred0 = te_pred0.sort_values(["breath_id", "row_id"], kind="mergesort")
    te_pred0["p0_lag1"] = (
        te_pred0.groupby("breath_id", sort=False)["p0"].shift(1).fillna(te_pred0["p0"])
    )
    te_pred0["p0_lag1_bin"] = (
        np.rint(te_pred0["p0_lag1"].to_numpy() / BIN_PLAG) * BIN_PLAG
    ).astype(np.float32)

    te = te.merge(te_pred0[["row_id", "p0_lag1_bin"]], on="row_id", how="left")

    key9 = [
        "R",
        "C",
        "u_out",
        "t_idx",
        "u_in_bin",
        "u_in_lag1_bin",
        "u_in_cum_bin",
        "area_bin",
        "u_in_rate_bin",
        "p0_lag1_bin",
    ]
    tr_for_key = tr.copy()
    tr_for_key["p0_lag1_bin"] = tr_for_key["p_lag1_bin"]

    lut9 = (
        tr_for_key.groupby(key9, sort=False)["pressure"]
        .median()
        .rename("p9")
        .reset_index()
    )

    key8 = [
        "R",
        "C",
        "u_out",
        "t_idx",
        "u_in_bin",
        "u_in_lag1_bin",
        "u_in_cum_bin",
        "area_bin",
        "p0_lag1_bin",
    ]
    lut8 = (
        tr_for_key.groupby(key8, sort=False)["pressure"]
        .median()
        .rename("p8")
        .reset_index()
    )

    key7 = [
        "R",
        "C",
        "u_out",
        "t_idx",
        "u_in_bin",
        "u_in_cum_bin",
        "area_bin",
        "p0_lag1_bin",
    ]
    lut7 = (
        tr_for_key.groupby(key7, sort=False)["pressure"]
        .median()
        .rename("p7")
        .reset_index()
    )

    key6 = ["R", "C", "u_out", "t_idx", "u_in_bin", "area_bin", "p0_lag1_bin"]
    lut6 = (
        tr_for_key.groupby(key6, sort=False)["pressure"]
        .median()
        .rename("p6")
        .reset_index()
    )

    key5 = ["R", "C", "u_out", "t_idx", "u_in_bin", "area_bin"]
    lut5 = tr.groupby(key5, sort=False)["pressure"].median().rename("p5").reset_index()

    key4 = ["R", "C", "u_out", "t_idx", "u_in_bin"]
    lut4 = tr.groupby(key4, sort=False)["pressure"].median().rename("p4").reset_index()

    key3 = ["R", "C", "u_out", "t_idx"]
    lut3 = tr.groupby(key3, sort=False)["pressure"].median().rename("p3").reset_index()

    key2 = ["R", "C", "t_idx"]
    lut2 = tr.groupby(key2, sort=False)["pressure"].median().rename("p2").reset_index()

    tem = te.merge(lut9, on=key9, how="left")
    tem = tem.merge(lut8, on=key8, how="left")
    tem = tem.merge(lut7, on=key7, how="left")
    tem = tem.merge(lut6, on=key6, how="left")
    tem = tem.merge(lut5, on=key5, how="left")
    tem = tem.merge(lut4, on=key4, how="left")
    tem = tem.merge(lut3, on=key3, how="left")
    tem = tem.merge(lut2, on=key2, how="left")

    pred = tem["p9"].to_numpy()
    for col in ["p8", "p7", "p6", "p5", "p4", "p3", "p2"]:
        mask = np.isnan(pred)
        if mask.any():
            pred[mask] = tem.loc[mask, col].to_numpy()

    mask = np.isnan(pred)
    if mask.any():
        pred[mask] = global_med

    pred_df = pd.DataFrame(
        {
            "breath_id": df_test["breath_id"].to_numpy(),
            "u_out": df_test["u_out"].to_numpy(),
            "pred": pred.astype(np.float64),
            "row_id": df_test["row_id"].to_numpy(),
            "id": df_test["id"].to_numpy(),  # kept for submission output only
        }
    ).sort_values(["breath_id", "row_id"], kind="mergesort")

    pred_df["pred2"] = pred_df["pred"].copy()
    pred_df.loc[pred_df["u_out"] == 1, "pred2"] = np.nan
    pred_df["pred2"] = pred_df.groupby("breath_id", sort=False)["pred2"].ffill()
    pred_df["pred2"] = pred_df["pred2"].fillna(pred_df["pred"])

    pred_vals = pred_df.sort_values("row_id", kind="mergesort")["pred2"].to_numpy()
    if len(pred_vals) != len(sub):
        raise ValueError(
            f"Pred length {len(pred_vals)} != submission length {len(sub)}"
        )

    sub = sub.copy()
    sub["pressure"] = pred_vals
    sub["pressure"] = sub["pressure"].apply(find_nearest)
    sub.to_csv(out_path, index=False)
    return sub


def g(dp):
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)

    if len(l) == 0:
        _fallback_baseline_submission("submission.csv")
        return

    file_count = len(l)
    loop_time = 154
    splits = file_count // 2

    if splits < 1:
        splits = 1

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
        if len(flist[i]) == 0:
            continue
        flist[i] = wc(flist[i])

    if len(flist) == 0:
        _fallback_baseline_submission("submission.csv")
        return

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
    med = np.median(np.vstack(pred_list), axis=0)

    if len(med) != len(output):
        raise ValueError(
            f"Pred length {len(med)} does not match submission length {len(output)}"
        )

    output["pressure"] = med
    output["pressure"] = output["pressure"].apply(find_nearest)

    output.to_csv("submission.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.6 + b.pressure * 0.4
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
g("../input/gb-data-blending-recover")
