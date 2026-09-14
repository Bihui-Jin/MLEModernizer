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

0.143744859069948

# 6. Current score

10.86378

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.86378) has done: 'The crash happens because `../input/gb-data-blending-recover` doesn’t exist in your environment, so `glob` finds zero files, `pred_list` becomes a list of scalars, and `np.median(np.vstack(pred_list), axis=0)` returns a length-1 array that can’t be assigned to 603600 rows. I fix `g()` to (1) search for the blending folder in the actual provided paths and (2) validate that found prediction files have the expected length. If no external prediction files are available, the script fall back to a simple deterministic baseline (median pressure from train) so it always produces a valid `submission.csv`. These changes are only to make the pipeline run end-to-end and generate a correct-format submission.'
- What this solution (achieved 10.86378) has done: 'Your current score is extremely far from the target (MAE 10.86 vs 0.1437), and the reason is that the code is effectively producing a near-constant “median pressure” fallback because no valid external prediction files exist in this environment. To move the score sharply toward the target while preserving your core “blend multiple CSV submissions then snap to nearest allowed pressure” logic, I generate those missing per-row prediction CSVs from the provided `train.csv/test.csv` using a deterministic, lightweight lookup baseline keyed by `(R, C, time_step, u_in, u_out)` (and a safe fallback for unseen keys). Then your existing `g()` blender read and combine them as intended, producing a much more accurate submission without changing the blending approach. I also ensure the generated prediction files have the exact required length/alignment with `sample_submission.csv` so the output is always valid.'
- What this solution (achieved 10.86378) has done: 'Your current MAE (10.86) is far from the target (0.1437) because the script ends up blending only locally-generated constant/weak lookup predictions; the biggest win while preserving the same “blend CSVs then snap to allowed pressures” core is to make those locally-generated blend inputs much stronger. I keep your blending loop intact and only improve `_make_local_blend_inputs()` by (1) using `breath_id` + `time_step` keyed medians (much more informative than global time_step medians) and (2) creating two diverse-but-reasonable variants so the blender has meaningful signals to combine. I also ensure perfect row alignment with `sample_submission` by building predictions in `id` order and writing the generated CSVs into the same folder your `g()` already reads.'
- What this solution (achieved 10.86378) has done: 'Your score is far above the target (MAE 10.86 vs 0.1437; lower is better), so the blended inputs you generate are still too weak; the smallest big gain without changing your blending logic is to generate much stronger local prediction CSVs for `g()` to blend. I keep your whole blender (`wc()`, random-weight ensembling, median over iterations, and snapping to nearest allowed pressure) intact, and only upgrade `_make_local_blend_inputs()` to use a no-ML “nearest-neighbor in (R,C,u_out,time_step,u_in)” lookup from train plus a smooth fallback that uses the previous timestep within the same breath (this matches the sequential nature and helps a lot). I also ensure strict `id` alignment by merging predictions back onto `sample_submission` by `id` before writing each local CSV, so `g()` always reads correctly ordered files. This should move the MAE sharply downward toward the target while preserving your approach and keeping runtime under the limit.'
- What this solution (achieved 10.86378) has done: 'Your current score (10.86 MAE) is far from the target (0.1437; lower is better) because the locally-generated blend inputs are still too weak and also have an `id` alignment bug (you merge `sample_submission` with `test[["id"]]` but don’t actually order predictions by `id`). I keep your blending logic (`wc()` + random-weight ensembling + median + snapping) intact and only strengthen `_make_local_blend_inputs()` using a deterministic train-to-test nearest-neighbor lookup keyed by `(R,C,u_out,time_step,u_in)` (with quantization for speed) and strict `id`-based alignment before writing the local CSVs. I also make `wc()` robust to any file ordering by reindexing each read submission to `sample_submission`’s `id` order, preventing silent row misalignment that destroys MAE. These minimal, score-relevant fixes should move MAE sharply down toward the target while preserving your overall approach and producing a valid `submission.csv` end-to-end.'
- What this solution (achieved 10.86378) has done: 'Your current MAE (10.86) is far from the target (0.1437; lower is better), so we need a real signal instead of the weak/mostly-constant fallback. I keep your existing blending pipeline (`wc()` + random-weight ensembling + median + snapping) intact, and only strengthen `_make_local_blend_inputs()` by generating a much better per-row prediction using a deterministic “same-breath time-step shape” lookup: for each `(R,C,time_index)` we learn the median pressure from train and reuse it for test, then apply your existing inspiratory-only smoothing. I also fix a score-killing mismatch: your files currently load from `../input/...` but your environment paths are `/kaggle/input/...`, which makes the code silently fall back; I add a tiny path resolver so the intended data is actually read. These minimal changes should sharply decrease MAE toward the target while preserving your overall approach and producing a valid `submission.csv`.'
- What this solution (achieved 10.86378) has done: 'Your MAE is far from the target because the “shape-only” `(R,C,time_index)` lookup ignores the control inputs, so predictions are almost generic trajectories and can’t match per-row pressure well. To move sharply toward the target while preserving your core pipeline (generate local blend CSVs → `wc()` blending → random-weight ensemble → median → snap-to-allowed-pressures), I only strengthen `_make_local_blend_inputs()` by learning a deterministic train lookup keyed on `(R,C,u_out,t_idx,u_in_bin)` (and a secondary `(R,C,u_out,t_idx)` fallback), which injects the most important signal (`u_in`, `u_out`) without any new model/training loop. I also write the generated files into the same folder that `g()` actually resolves, so you don’t accidentally blend from an empty directory and fall back to weak baselines. Everything else (including blending semantics and snapping) stays the same.'
- What this solution (achieved 10.86378) has done: 'Your current MAE (10.86) is far from the target (0.1437; lower is better) because the blend inputs are still too weak and the blender’s random-weight ensemble is accidentally re-weighting predictors inconsistently (weights are sorted but predictions aren’t), which can destroy signal. I keep the same overall pipeline (generate local CSVs → `wc()` blending → random-weight ensemble → median → snap to allowed pressures), but strengthen the locally-generated blend inputs using a per-breath “relative pressure change” (Δpressure) lookup keyed on `(R,C,u_out,t_idx,u_in_bin)` so test breaths get realistic dynamics, then integrate Δp to get absolute pressure. I also fix the weight sorting bug by sorting weights and predictions together (same semantics, just correct), and keep strict `id` alignment when reading every CSV. These minimal, score-relevant changes should move MAE sharply downward toward the target without changing the fundamental approach.'
- What this solution (achieved 10.86378) has done: 'Your current score is far from the target because the blender is effectively working with weak/misaligned locally-generated inputs, so we strengthen only the local CSV generation while keeping your blending approach, snapping, and submission semantics unchanged. Concretely, we generate better per-row predictions by learning (from train) a per-step *delta pressure* conditioned on `(R, C, u_out, t_idx, u_in_bin)` and also a “stepwise level” fallback, then integrate within each breath—this preserves your no-ML lookup style but injects the key control signal. We also make sure the generated predictions respect inspiratory-only scoring by forcing predictions to a simple, stable behavior on expiratory steps (`u_out==1`) rather than letting integration drift. Finally, we keep strict `id` alignment both when writing local blend files and when reading any blended files, which avoids silent row-order errors that can catastrophically worsen MAE.'
- What this solution (achieved 10.86378) has done: 'Your current MAE (10.86) is far from the target (0.1437), so we need a big real-signal gain while keeping your existing blending pipeline intact. The main score-killer in your local inputs is that the generated dp/level lookups are built against `t_idx` and `u_in_bin`, which are too coarse/misaligned to the continuous dynamics; we can preserve the same “train-lookup → per-row prediction CSVs → blend → snap” logic but key the lookups on the actual `time_step` (quantized) and use a finer `u_in` bin to better match trajectories. I also add one more locally-generated input variant that uses a robust within-breath “p(t)=p(t-1)+dp” integration but anchors to a level lookup each step (reduces drift) without changing your blender. Finally, I keep strict `id` alignment and write the new local CSVs into the same blend directory so `g()` actually blends stronger signals instead of falling back.'
- What this solution (achieved 10.86378) has done: 'Your score is far from the target because the current run is still effectively producing weak/misaligned “local blend inputs”, so the blender is averaging low-signal predictors. I keep your blending pipeline intact (local CSV generation → `wc()` → random-weight ensemble → median → snap-to-allowed-pressures) and only strengthen `_make_local_blend_inputs()` by using a more informative, still-deterministic lookup: within each (R,C,u_out,time_step_quant) group, fit a robust linear mapping from `u_in` to `pressure` (plus a simple mean fallback) and use that to predict per-row pressures. This preserves your no-ML/lookup nature but injects the key control signal continuously instead of via coarse bins, which should sharply reduce MAE toward the target. I also keep strict `id` alignment when writing/reading every generated CSV so row-order bugs can’t inflate MAE.'
- What this solution (achieved 10.86378) has done: 'Your current score is far worse than the target, so we need a real signal improvement while keeping your existing pipeline (generate local CSVs → `wc()` blending → random-weight ensemble → median → snap). The smallest high-impact fix is to make the locally-generated blend inputs much more faithful to ventilator dynamics by leveraging strong train-time structure: for each (R,C) learn a median inspiratory pressure trajectory by time index, and then condition it on control input using a per-(R,C,u_out,t_idx) linear mapping from `u_in` to `pressure` (fallback safely when groups are sparse). I’m keeping `g()`/`wc()`/random weighting/median/snap semantics unchanged, only upgrading `_make_local_blend_inputs()` to use `t_idx` (exact step number) instead of time quantization and to fit the `u_in→pressure` relation with proper group moments (mean/cov/var) for stability. This should sharply reduce MAE toward your target while preserving the same overall approach and still writing a valid `submission.csv` end-to-end.'

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
def _kaggle_path(*parts):
    p = os.path.join(*parts)
    if os.path.exists(p):
        return p
    alt1 = os.path.join("/kaggle", *[x.strip("/").replace("../", "") for x in parts])
    if os.path.exists(alt1):
        return alt1
    alt2 = os.path.join("/kaggle/input", parts[-1].split("/")[-1]) if parts else p
    return p


train_path = _kaggle_path("/kaggle/input/ventilator-pressure-prediction/train.csv")
if not os.path.exists(train_path):
    train_path = _kaggle_path("../input/ventilator-pressure-prediction/train.csv")

df_train = pd.read_csv(train_path)

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


def wc(input_list, id_order=None):
    """
    Keep core blending logic, but ensure every file is aligned by 'id' to sample_submission order;
    otherwise row order mismatches can explode MAE even if predictions are good.
    """
    l = []
    arrs = []
    for i in range(len(input_list)):
        fn = input_list[i]
        base = fn.split("/")[-1]
        public_lb_score = None
        try:
            public_lb_score = int(base.split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1
        l.append(public_lb_score)

        df = pd.read_csv(fn)
        if "pressure" not in df.columns:
            raise ValueError(f"File {fn} does not contain 'pressure' column.")
        if "id" in df.columns and id_order is not None:
            df = df.set_index("id").reindex(id_order)
            arr = df["pressure"].to_numpy(dtype=np.float64).ravel()
        else:
            arr = df["pressure"].to_numpy(dtype=np.float64).ravel()
        arrs.append(arr)

    l_sum = sum(l) if sum(l) != 0 else 1

    if len(arrs) == 1:
        return arrs[0]
    else:
        weight1 = (l[1] / l_sum) + 0.15
        weight2 = 1 - weight1
        return arrs[0] * weight1 + arrs[1] * weight2


def _resolve_blend_dir(dp):
    candidates = [
        dp,
        "../input/gb-data-blending-recover",
        "/kaggle/input/gb-data-blending-recover",
        "../kaggle/input/gb-data-blending-recover",
        "../input/ventilator-pressure-prediction",
        "/kaggle/input/ventilator-pressure-prediction",
        "../kaggle/input/ventilator-pressure-prediction",
    ]
    for c in candidates:
        if c is not None and os.path.isdir(c):
            return c
    return None


def _make_local_blend_inputs(write_dir=None):
    """
    Score-relevant minimal improvement (toward much lower MAE) while preserving the same pipeline:
    - Still generates several local prediction CSVs that g()/wc() will blend (core logic unchanged).
    - Replace fragile time_step quantization with exact per-breath step index (t_idx), which matches
      the dataset's fixed 80-step structure and yields much cleaner train->test lookups.
    - Fit deterministic per-(R,C,u_out,t_idx) linear mapping pressure ≈ a + b*u_in using group
      moments (mean/var/cov) for stability, with robust fallbacks for sparse groups.
    - Keep strict 'id' alignment when writing files to avoid silent row-order MAE blow-ups.
    """
    base_dir = (
        write_dir
        or _resolve_blend_dir("../input/gb-data-blending-recover")
        or "../input/gb-data-blending-recover"
    )
    os.makedirs(base_dir, exist_ok=True)

    sample_path = _kaggle_path(
        "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv"
    )
    if not os.path.exists(sample_path):
        sample_path = _kaggle_path(
            "../input/ventilator-pressure-prediction/sample_submission.csv"
        )

    test_path = _kaggle_path("/kaggle/input/ventilator-pressure-prediction/test.csv")
    if not os.path.exists(test_path):
        test_path = _kaggle_path("../input/ventilator-pressure-prediction/test.csv")

    sub = pd.read_csv(sample_path)
    test = pd.read_csv(
        test_path, usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
    )

    sub = sub.sort_values("id").reset_index(drop=True)
    test = test.sort_values("id").reset_index(drop=True)
    if len(sub) != len(test) or not np.array_equal(
        sub["id"].to_numpy(), test["id"].to_numpy()
    ):
        test = (
            test.merge(sub[["id"]], on="id", how="right", validate="one_to_one")
            .sort_values("id")
            .reset_index(drop=True)
        )

    test["t_idx"] = test.groupby("breath_id").cumcount().astype(np.int16)

    tr = df_train[
        ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    ].copy()
    tr["t_idx"] = tr.groupby("breath_id").cumcount().astype(np.int16)

    for col, dt in [("R", np.int16), ("C", np.int16), ("u_out", np.int8)]:
        tr[col] = tr[col].astype(dt)
        test[col] = test[col].astype(dt)

    grp_cols = ["R", "C", "u_out", "t_idx"]

    g = tr.groupby(grp_cols, sort=False)
    stats = g.agg(
        u_mean=("u_in", "mean"),
        p_mean=("pressure", "mean"),
        u2_mean=(
            "u_in",
            lambda x: float(np.mean(np.square(x.to_numpy(dtype=np.float64)))),
        ),
        p_median=("pressure", "median"),
        n=("pressure", "size"),
    ).reset_index()

    tr_tmp = tr[grp_cols + ["u_in", "pressure"]].copy()
    tr_tmp["up"] = tr_tmp["u_in"].to_numpy(dtype=np.float64) * tr_tmp[
        "pressure"
    ].to_numpy(dtype=np.float64)
    up = (
        tr_tmp.groupby(grp_cols, sort=False)["up"]
        .mean()
        .reset_index()
        .rename(columns={"up": "up_mean"})
    )
    stats = stats.merge(up, on=grp_cols, how="left")

    var_u = stats["u2_mean"].to_numpy(dtype=np.float64) - np.square(
        stats["u_mean"].to_numpy(dtype=np.float64)
    )
    cov_up = stats["up_mean"].to_numpy(dtype=np.float64) - (
        stats["u_mean"].to_numpy(dtype=np.float64)
        * stats["p_mean"].to_numpy(dtype=np.float64)
    )

    eps = 1e-6
    b = cov_up / np.maximum(var_u, eps)
    b = np.clip(b, -2.0, 2.0)
    a = stats["p_mean"].to_numpy(dtype=np.float64) - b * stats["u_mean"].to_numpy(
        dtype=np.float64
    )

    stats["a"] = a.astype(np.float32)
    stats["b"] = b.astype(np.float32)

    global_p_median = float(tr["pressure"].median())
    global_p_mean = float(tr["pressure"].mean())

    p_rc_t = (
        tr.groupby(["R", "C", "t_idx"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_rc_t"})
    )
    p_t = (
        tr.groupby(["t_idx"], sort=False)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "p_t"})
    )

    tmp = test.merge(
        stats[grp_cols + ["a", "b", "p_median", "n"]], on=grp_cols, how="left"
    )
    tmp = tmp.merge(p_rc_t, on=["R", "C", "t_idx"], how="left")
    tmp = tmp.merge(p_t, on=["t_idx"], how="left")

    u = tmp["u_in"].to_numpy(dtype=np.float64)
    a = tmp["a"].to_numpy(dtype=np.float64)
    b = tmp["b"].to_numpy(dtype=np.float64)

    p_lin = a + b * u
    p_med = tmp["p_median"].to_numpy(dtype=np.float64)
    p_rc = tmp["p_rc_t"].to_numpy(dtype=np.float64)
    p_tt = tmp["p_t"].to_numpy(dtype=np.float64)

    pred = np.where(np.isnan(p_lin), p_med, p_lin)
    pred = np.where(np.isnan(pred), p_rc, pred)
    pred = np.where(np.isnan(pred), p_tt, pred)
    pred = np.where(np.isnan(pred), global_p_median, pred).astype(np.float64)

    breath_id = test["breath_id"].to_numpy()
    u_out = test["u_out"].to_numpy(dtype=np.int8)

    pred_hold = pred.copy()
    for i in range(1, len(pred_hold)):
        if breath_id[i] == breath_id[i - 1] and u_out[i] == 1:
            pred_hold[i] = pred_hold[i - 1]

    pred_smooth = pred_hold.copy()
    for i in range(1, len(pred_smooth)):
        if breath_id[i] == breath_id[i - 1] and u_out[i] == 0 and u_out[i - 1] == 0:
            pred_smooth[i] = 0.80 * pred_smooth[i] + 0.20 * pred_smooth[i - 1]
        elif breath_id[i] == breath_id[i - 1] and u_out[i] == 1:
            pred_smooth[i] = pred_smooth[i - 1]

    n = tmp["n"].to_numpy(dtype=np.float64)
    shrink = np.clip((n - 10.0) / 40.0, 0.0, 1.0)
    pred_shrunk = shrink * pred_hold + (1.0 - shrink) * np.where(
        np.isnan(p_med), global_p_mean, p_med
    )

    def _snap(arr):
        return np.array([find_nearest(float(x)) for x in arr], dtype=np.float64)

    out_a = pd.DataFrame({"id": sub["id"].to_numpy(), "pressure": _snap(pred_hold)})
    out_a.to_csv(os.path.join(base_dir, "local.1 a.csv"), index=False)

    out_b = pd.DataFrame({"id": sub["id"].to_numpy(), "pressure": _snap(pred_smooth)})
    out_b.to_csv(os.path.join(base_dir, "local.2 b.csv"), index=False)

    out_c = pd.DataFrame({"id": sub["id"].to_numpy(), "pressure": _snap(pred_shrunk)})
    out_c.to_csv(os.path.join(base_dir, "local.3 c.csv"), index=False)

    out_d = pd.DataFrame(
        {
            "id": sub["id"].to_numpy(),
            "pressure": _snap(0.5 * pred_smooth + 0.5 * pred_shrunk),
        }
    )
    out_d.to_csv(os.path.join(base_dir, "local.4 d.csv"), index=False)


def g(dp):
    sample_path = _kaggle_path(
        "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv"
    )
    if not os.path.exists(sample_path):
        sample_path = _kaggle_path(
            "../input/ventilator-pressure-prediction/sample_submission.csv"
        )

    output = pd.read_csv(sample_path).sort_values("id").reset_index(drop=True)
    n = len(output)
    id_order = output["id"].to_numpy()

    dp_resolved = _resolve_blend_dir(dp)
    if dp_resolved is None:
        dp_resolved = dp

    files = []
    if dp_resolved is not None and os.path.isdir(dp_resolved):
        for f in glob.iglob(f"{dp_resolved}/*"):
            if f.lower().endswith(".csv"):
                files.append(f)
    files.sort()

    if len(files) == 0:
        _make_local_blend_inputs(write_dir=dp_resolved)
        dp_resolved = _resolve_blend_dir(dp) or dp_resolved or dp
        files = []
        if dp_resolved is not None and os.path.isdir(dp_resolved):
            for f in glob.iglob(f"{dp_resolved}/*"):
                if f.lower().endswith(".csv"):
                    files.append(f)
        files.sort()

    if len(files) == 0:
        baseline = float(df_train["pressure"].median())
        output["pressure"] = baseline
        output["pressure"] = output["pressure"].apply(find_nearest)
        output.to_csv("submission.csv", index=False)
        return

    file_count = len(files)
    loop_time = 154
    splits = max(1, file_count // 2)

    flist = []
    for i in range(splits):
        start = i * round(len(files) / splits)
        end = None if i == splits - 1 else (i + 1) * round(len(files) / splits)
        chunk = files[start:end]
        if len(chunk) == 0:
            continue
        flist.append(chunk)

    preds = []
    for chunk in flist:
        try:
            arr = wc(chunk, id_order=id_order)
            if len(arr) != n:
                continue
            preds.append(arr)
        except Exception:
            continue

    if len(preds) == 0:
        baseline = float(df_train["pressure"].median())
        output["pressure"] = baseline
        output["pressure"] = output["pressure"].apply(find_nearest)
        output.to_csv("submission.csv", index=False)
        return

    pred_list = []
    for t in range(loop_time):
        set_seed(t)
        weights = np.array([rd() for _ in range(len(preds))], dtype=np.float64)
        wsum = float(weights.sum()) if float(weights.sum()) != 0.0 else 1.0
        weights = weights / wsum

        order = np.argsort(-weights)  # descending
        weights = weights[order]
        preds_ordered = [preds[i] for i in order]

        temp = np.zeros(n, dtype=np.float64)
        for i in range(len(preds_ordered)):
            temp += preds_ordered[i] * float(weights[i])
        pred_list.append(temp)
        del temp
        gc.collect()

    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)

    output.to_csv("submission.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.5 + b.pressure * 0.5
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
g("../input/gb-data-blending-recover")
print("Wrote submission.csv")
