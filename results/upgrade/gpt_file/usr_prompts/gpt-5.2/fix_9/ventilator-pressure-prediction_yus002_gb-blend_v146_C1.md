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

0.1358952337889259

# 6. Current score

1.40825

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65244) has done: 'The crash happens because the blending directory is missing in this environment, so your `glob` finds no files and `pred_list` ends up containing a single scalar (median length 1), which can’t be assigned to the 603600-row submission. I keep your ensemble/median logic intact but add robust guards: automatically search common `/kaggle/input/...` locations for prediction CSVs, validate each file has the right `pressure` length, and fall back to a safe baseline submission if none are found so you always get a valid `.csv`. I also make `g()` return the created submission path and ensure the output filename ends with `.csv` (no spaces/illegal characters). These changes are execution/stability fixes; they don’t change model logic beyond preventing empty-input failure.'
- What this solution (achieved 8.43315) has done: 'Your current score is far worse than the target (lower-is-better), and the main reason is that your pipeline isn’t actually generating meaningful predictions—when no valid external prediction CSVs are found, it falls back to all-zeros, which scores terribly. I keep your ensemble/median + nearest-pressure snapping logic intact, but change the fallback behavior to a simple, legitimate baseline that uses the training-set mean pressure (snapped to the nearest allowed pressure), which should drastically reduce MAE while staying minimal. I also prevent accidentally ingesting `train.csv/test.csv/sample_submission.csv` as “prediction files” during the recursive search, which currently makes the ensemble effectively garbage even when files exist. The output still be a valid `submission.csv` with `id,pressure` and run end-to-end.'
- What this solution (achieved 6.19023) has done: 'Your current score (8.43315, lower-is-better) is far from the target, and the biggest issue is that the script often can’t find any valid external prediction files, so it falls back to a constant prediction (train mean), which is still very weak. To move the score much closer to the target while keeping your blending/median/snap-to-nearest logic intact, I change the fallback to a legitimate, much stronger baseline: a per-(R,C,time_step_index,u_out) mean pressure computed from the training set, then applied to test and snapped to the nearest allowed pressure. This does not change your core logic when external prediction CSVs exist; it only improves the “no valid prediction files found” path that is currently dominating your score. I also ensure alignment is correct by merging on (R,C,breath_step,u_out) and filling any missing groups with the global mean.'
- What this solution (achieved 4.5528) has done: 'Your current score (6.19023, lower-is-better) is still far from the target, so the fastest safe gain is to make the fallback baseline stronger while keeping your blending/median/snap core logic unchanged. Right now, the fallback only uses (R,C,step,u_out) mean pressure; adding u_in information (as a binned feature) is a minimal, legitimate extension that better matches the pressure dynamics and should reduce MAE substantially without changing any model/training logic. I keep the same output alignment (sorted by `id`) and the same “snap to nearest allowed pressure” post-processing so evaluation semantics remain consistent. No changes are made to the ensemble path when external prediction CSVs exist; only the “no valid prediction files found” path is improved.'
- What this solution (achieved 3.62256) has done: 'Your current score is far worse than the target (lower-is-better), and the dominant driver is the fallback path when no valid external prediction CSVs are found; improving that fallback is the smallest change that legitimately moves MAE down without touching your ensemble logic. I keep your existing blending/median and snap-to-nearest-pressure post-processing unchanged, but strengthen the fallback from a coarse group mean to a per-(R,C,step,u_out,u_in_bin,du_in_bin) mean that captures local dynamics while remaining a simple train-derived lookup. I also make the fallback respect the metric by explicitly setting expiratory-phase (u_out==1) predictions to 0 (not scored), which avoids unnecessary errors in case Kaggle’s scoring mask differs from expectations. The script still runs end-to-end and always writes a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 1.5837) has done: 'Your current score (3.62256, lower-is-better) is still far from the target, and the main lever available without changing your ensemble logic is improving the fallback path (which is used whenever no valid external prediction CSVs are found). I keep your blending/median/snap-to-nearest-pressure logic intact, but strengthen the fallback lookup to use a more appropriate key that matches the time-series nature of the problem: `(R, C, breath_step, u_out, u_in_bin, du_in_bin, cum_u_in_bin)`, with a clear backoff chain to avoid NaNs. This remains a simple train-derived mean lookup (no new model/training loop) and should reduce MAE materially toward your target. I also keep the submission alignment strictly by `id` and always write a valid `submission.csv`.'
- What this solution (achieved 1.46849) has done: 'Your current score (1.5837, lower-is-better) is still far above the target, and the biggest lever available without changing your ensemble logic is improving the fallback path that runs whenever no valid external prediction CSVs are found. I keep your blending/median/snap logic exactly the same, but strengthen the fallback to a slightly more expressive (still train-derived lookup) key by adding a binned cumulative “volume proxy” and a binned time_step (in addition to the existing step index), with a safe backoff chain so it never produces NaNs. This is a minimal, legitimate change that better matches the evaluation (inspiratory phase dynamics) while preserving semantics (still mean lookup + nearest-pressure snapping). The script still run end-to-end and always write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 1.40825) has done: 'Your current score (1.46849, lower-is-better) is still far above the target, so we should make the smallest legitimate improvement that primarily strengthens the fallback path (used when no external prediction CSVs are found) without changing your ensemble/median/snap core logic. I keep your blending/median + nearest-pressure snapping exactly the same, but improve the fallback lookup by adding one minimal, high-signal grouping key: the previous-step valve opening (`u_in_prev_bin`), with a safe backoff chain so it never produces NaNs. This better captures short-term dynamics with negligible additional complexity and should reduce MAE toward the target while staying within your constraints. I also keep alignment strictly by `id` and still force `u_out==1` to 0 as you already do.'

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

TRAIN_MEAN_PRESSURE = float(df_train["pressure"].mean())


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


def _load_submission_template():
    candidates = [
        "../input/ventilator-pressure-prediction/sample_submission.csv",
        "../input/sample_submission.csv",
        "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected locations."
    )


def _list_prediction_csvs(dp):
    paths = []
    if dp is not None and os.path.exists(dp):
        paths = [p for p in glob.iglob(f"{dp}/*") if os.path.isfile(p)]

    if len(paths) == 0:
        for p in glob.iglob("/kaggle/input/**/**/*.csv", recursive=True):
            if os.path.isfile(p):
                paths.append(p)
        for p in glob.iglob("../input/**/**/*.csv", recursive=True):
            if os.path.isfile(p):
                paths.append(p)

    def _is_obvious_competition_data_csv(p):
        base = os.path.basename(p).lower()
        if base in {"train.csv", "test.csv", "sample_submission.csv"}:
            return True
        if base.endswith(".csv.zip"):
            return True
        return False

    paths = [p for p in paths if not _is_obvious_competition_data_csv(p)]

    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _try_read_pred_file(path, expected_len):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "pressure" not in df.columns:
        return None
    arr = df["pressure"].to_numpy().ravel()
    if arr.shape[0] != expected_len:
        return None
    return arr


def wc(input_list, expected_len):
    """
    Original logic assumed input_list file name encodes a public LB score and always reads pressure.
    We keep the blending semantics but make it robust if filenames don't match that pattern.
    """
    l = []
    arrs = []
    for i in range(len(input_list)):
        fn = input_list[i]
        try:
            public_lb_score = int(fn.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1
        arr = _try_read_pred_file(fn, expected_len)
        if arr is None:
            continue
        l.append(public_lb_score)
        arrs.append(arr)

    if len(arrs) == 0:
        return None

    l_sum = sum(l) if sum(l) != 0 else len(l)
    if len(arrs) == 1:
        return arrs[0]
    else:
        a0, a1 = arrs[0], arrs[1]
        s0, s1 = l[0], l[1]
        weight1 = (s1 / l_sum) + 0.15
        weight2 = 1 - weight1
        return a0 * weight1 + a1 * weight2


def _fallback_pressure_by_group(expected_len):
    """
    ONLY affects the no-external-predictions fallback path to reduce MAE (lower-is-better).

    Change (minimal, directly score-relevant):
    - Add u_in_prev_bin (binned previous-step u_in) as an additional key in the fine lookup.
      This captures short-term control dynamics with negligible complexity and keeps the same
      mean-lookup + safe-backoff semantics as before.
    """
    df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")

    tr = df_train[
        ["breath_id", "R", "C", "u_out", "u_in", "time_step", "pressure"]
    ].copy()
    tr["breath_step"] = tr.groupby("breath_id").cumcount().astype(np.int16)

    te = df_test[["id", "breath_id", "R", "C", "u_out", "u_in", "time_step"]].copy()
    te["breath_step"] = te.groupby("breath_id").cumcount().astype(np.int16)

    tr["cum_u_in"] = tr.groupby("breath_id")["u_in"].cumsum().astype(np.float32)
    te["cum_u_in"] = te.groupby("breath_id")["u_in"].cumsum().astype(np.float32)

    BIN_EDGES = np.linspace(0.0, 100.0, 21)
    tr["u_in_bin"] = np.digitize(tr["u_in"].to_numpy(), BIN_EDGES, right=False).astype(
        np.int16
    )
    te["u_in_bin"] = np.digitize(te["u_in"].to_numpy(), BIN_EDGES, right=False).astype(
        np.int16
    )

    tr["u_in_prev_bin"] = (
        tr.groupby("breath_id")["u_in_bin"]
        .shift(1)
        .fillna(tr["u_in_bin"])
        .astype(np.int16)
    )
    te["u_in_prev_bin"] = (
        te.groupby("breath_id")["u_in_bin"]
        .shift(1)
        .fillna(te["u_in_bin"])
        .astype(np.int16)
    )

    tr["du_in"] = tr.groupby("breath_id")["u_in"].diff().fillna(0.0).astype(np.float32)
    te["du_in"] = te.groupby("breath_id")["u_in"].diff().fillna(0.0).astype(np.float32)

    DU_EDGES = np.array(
        [-1000.0, -20.0, -5.0, -1.0, 1.0, 5.0, 20.0, 1000.0], dtype=np.float32
    )
    tr["du_in_bin"] = np.digitize(tr["du_in"].to_numpy(), DU_EDGES, right=False).astype(
        np.int16
    )
    te["du_in_bin"] = np.digitize(te["du_in"].to_numpy(), DU_EDGES, right=False).astype(
        np.int16
    )

    q = np.linspace(0.0, 1.0, 21)
    cum_edges = np.quantile(tr["cum_u_in"].to_numpy(), q).astype(np.float32)
    cum_edges = np.unique(cum_edges)
    if cum_edges.shape[0] < 3:
        cum_edges = np.array(
            [tr["cum_u_in"].min(), tr["cum_u_in"].max()], dtype=np.float32
        )

    tr["cum_u_in_bin"] = np.digitize(
        tr["cum_u_in"].to_numpy(), cum_edges, right=False
    ).astype(np.int16)
    te["cum_u_in_bin"] = np.digitize(
        te["cum_u_in"].to_numpy(), cum_edges, right=False
    ).astype(np.int16)

    TIME_EDGES = np.linspace(0.0, 2.8, 81, dtype=np.float32)
    tr["time_bin"] = np.digitize(
        tr["time_step"].to_numpy(), TIME_EDGES, right=False
    ).astype(np.int16)
    te["time_bin"] = np.digitize(
        te["time_step"].to_numpy(), TIME_EDGES, right=False
    ).astype(np.int16)

    grp_fine = (
        tr.groupby(
            [
                "R",
                "C",
                "breath_step",
                "time_bin",
                "u_out",
                "u_in_bin",
                "u_in_prev_bin",
                "du_in_bin",
                "cum_u_in_bin",
            ],
            sort=False,
        )["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "pred_fine"})
    )

    grp_drop_prev = (
        tr.groupby(
            [
                "R",
                "C",
                "breath_step",
                "time_bin",
                "u_out",
                "u_in_bin",
                "du_in_bin",
                "cum_u_in_bin",
            ],
            sort=False,
        )["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "pred_drop_prev"})
    )

    grp_dropcum = (
        tr.groupby(
            ["R", "C", "breath_step", "time_bin", "u_out", "u_in_bin", "du_in_bin"],
            sort=False,
        )["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "pred_dropcum"})
    )

    grp_mid = (
        tr.groupby(
            ["R", "C", "breath_step", "time_bin", "u_out", "u_in_bin"],
            sort=False,
        )["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "pred_mid"})
    )

    grp_dropuin = (
        tr.groupby(
            ["R", "C", "breath_step", "time_bin", "u_out"],
            sort=False,
        )["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "pred_dropuin"})
    )

    grp_coarse = (
        tr.groupby(["R", "C", "breath_step", "u_out"], sort=False)["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "pred_coarse"})
    )

    te = te.merge(
        grp_fine,
        on=[
            "R",
            "C",
            "breath_step",
            "time_bin",
            "u_out",
            "u_in_bin",
            "u_in_prev_bin",
            "du_in_bin",
            "cum_u_in_bin",
        ],
        how="left",
    )
    te = te.merge(
        grp_drop_prev,
        on=[
            "R",
            "C",
            "breath_step",
            "time_bin",
            "u_out",
            "u_in_bin",
            "du_in_bin",
            "cum_u_in_bin",
        ],
        how="left",
    )
    te = te.merge(
        grp_dropcum,
        on=["R", "C", "breath_step", "time_bin", "u_out", "u_in_bin", "du_in_bin"],
        how="left",
    )
    te = te.merge(
        grp_mid,
        on=["R", "C", "breath_step", "time_bin", "u_out", "u_in_bin"],
        how="left",
    )
    te = te.merge(
        grp_dropuin,
        on=["R", "C", "breath_step", "time_bin", "u_out"],
        how="left",
    )
    te = te.merge(grp_coarse, on=["R", "C", "breath_step", "u_out"], how="left")

    te["pred"] = te["pred_fine"]
    te["pred"] = te["pred"].fillna(te["pred_drop_prev"])
    te["pred"] = te["pred"].fillna(te["pred_dropcum"])
    te["pred"] = te["pred"].fillna(te["pred_mid"])
    te["pred"] = te["pred"].fillna(te["pred_dropuin"])
    te["pred"] = te["pred"].fillna(te["pred_coarse"])
    te["pred"] = te["pred"].fillna(TRAIN_MEAN_PRESSURE)

    te.loc[te["u_out"] == 1, "pred"] = 0.0

    te = te.sort_values("id", kind="mergesort")
    arr = te["pred"].to_numpy()
    if arr.shape[0] != expected_len:
        arr = np.full(expected_len, TRAIN_MEAN_PRESSURE, dtype=float)

    arr = np.array([find_nearest(x) for x in arr], dtype=float)
    return arr


def g(dp):
    """
    Robust version of original g():
    - If no prediction files exist, create a valid baseline submission.
    - Validate each loaded prediction has correct length (603600) before using it.
    - Keep original median-over-random-weights ensemble logic unchanged.
    """
    output = _load_submission_template()
    expected_len = len(output)

    l = _list_prediction_csvs(dp)
    l.sort()

    valid_files = []
    for p in l:
        arr = _try_read_pred_file(p, expected_len)
        if arr is not None:
            valid_files.append(p)

    file_count = len(valid_files)
    loop_time = 154

    if file_count == 0:
        output["pressure"] = _fallback_pressure_by_group(expected_len)
        out_path = "submission.csv"
        output.to_csv(out_path, index=False)
        return out_path

    splits = max(1, file_count // 2)
    flist = []
    for i in range(splits):
        if i == splits - 1:
            chunk = valid_files[i * round(len(valid_files) / splits) :]
        else:
            chunk = valid_files[
                i
                * round(len(valid_files) / splits) : (i + 1)
                * round(len(valid_files) / splits)
            ]
        flist.append(chunk)

    new_flist = []
    for i in range(len(flist)):
        arr = wc(flist[i], expected_len)
        if arr is not None:
            new_flist.append(arr)
    flist = new_flist

    if len(flist) == 0:
        output["pressure"] = _fallback_pressure_by_group(expected_len)
        out_path = "submission.csv"
        output.to_csv(out_path, index=False)
        return out_path

    pred_list = []
    for i in range(loop_time):
        weight = []
        set_seed(i)
        for j in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight) if sum(weight) != 0 else 1.0
        for j in range(len(weight)):
            weight[j] /= weight_sum
        weight.sort(reverse=True)
        temp = 0
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()

    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)

    out_path = f"rwb_{loop_time}_loops.csv"
    output.to_csv(out_path, index=False)
    return out_path


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.55 + b.pressure * 0.45
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
sub_path = g("../input/gb-data-blending-recover")
print("Wrote submission to:", sub_path)

if sub_path != "submission.csv":
    pd.read_csv(sub_path).to_csv("submission.csv", index=False)
print("Also wrote submission.csv")
