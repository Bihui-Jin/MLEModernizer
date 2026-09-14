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

0.1394353448349509

# 6. Current score

4.89414

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.92375) has done: 'I fix the pipeline so it runs end-to-end in this Kaggle environment by removing dependencies on missing external input submissions and instead generating predictions from the provided `train.csv`/`test.csv`. To keep the “core logic” intact (simple ensembling), I replace the unavailable ensemble members with a lightweight, deterministic median-by-(R,C,time_step,u_out) lookup built from the training data, with a global median fallback for unseen combinations. This produces a correctly formatted `submission.csv` with `id,pressure` and avoids runtime errors. This should also yield a non-trivial MAE (far better than all-zeros) and move toward the target score.'
- What this solution (achieved 4.12056) has done: 'Your current score (9.92375 MAE) is far worse than the target (0.1394), so we should improve accuracy while keeping the same “lookup from train medians → merge onto test → global fallback” core logic. The main issue is that using raw `time_step` as a float key makes the median table sparse/misaligned due to float representation, causing excessive fallback to the global median and high error. I keep the same approach but make the join key robust by converting `time_step` to an integer timestep index within each breath (0–79), and I include `u_in` (rounded) in the lookup key since pressure is strongly driven by `u_in`. Finally, I add a slightly less coarse fallback hierarchy (drop `u_in`, then drop `u_out`) to reduce fallback error without changing the overall method.'
- What this solution (achieved 4.99193) has done: 'Your current MAE (4.12056) is far above the target (0.1394), so we should improve accuracy while keeping the same median-lookup core logic. The biggest easy win without changing modeling approach is to stop merging three separate times (which can misalign rows if anything shifts) and instead do a single left-join cascade on the same test frame, filling from most-specific to least-specific keys. Next, we make the `u_in` binning slightly finer (0.05 instead of 0.1) so the most-specific table matches more often without changing the method. Finally, we add one extra intermediate fallback (drop only `u_out` but keep `u_in_r`) to reduce fallback-to-global error on inspiratory segments.'
- What this solution (achieved 4.55331) has done: 'Your current MAE (4.99193, lower is better) is still far from the target (0.1394), so we should improve accuracy while keeping the same core “median lookup from train → merge onto test → fallback hierarchy” logic. The biggest gap is that the lookup key is too sparse/misaligned: using rounded `u_in` still misses many cases, and medians alone ignore the strong near-linear relationship between pressure and `u_in` within each (R,C,timestep,u_out) context. I keep the same exact pipeline structure but add a minimal, deterministic “linear calibration” per (R,C,t_idx,u_out): fit `pressure ≈ a*u_in + b` on train and use it only when the most-specific median is missing, before falling back to coarser medians/global. This is still a pure train-derived lookup/join approach (no new model architecture/training loops) and should significantly reduce fallback error toward the target.'
- What this solution (achieved 4.55331) has done: 'We keep your exact “train-derived lookup/join with fallback hierarchy” core logic, but make the calibration step actually correct and stronger without introducing a new model/training loop. The current `mean_up` computation is wrong because it multiplies `u_in` by the full-train `pressure` via `train.loc[x.index,...]` inside a groupby lambda, which can misalign and is extremely slow; we replace it with a deterministic, vectorized per-group OLS fit using precomputed `u_in^2` and `u_in*pressure`. We also add one minimal additional fallback calibration that drops `u_out` (still linear-in-`u_in` per (R,C,t_idx)) to reduce misses when `u_out` differs, then keep your existing median fallbacks unchanged. This should materially reduce MAE (move toward the much lower target) while preserving your pipeline semantics and producing the same submission format.'
- What this solution (achieved 4.55321) has done: 'Your MAE (4.55331, lower is better) is still far above the target (0.1394), so we should increase accuracy while keeping the same “train-derived lookup/join with fallback hierarchy + linear per-group calibration” core logic. The biggest low-risk improvement is to align predictions with the discrete pressure grid used in this competition: most strong solutions snap predictions to the nearest valid pressure value from the training set, which reduces MAE without changing the underlying model. We therefore compute the sorted unique pressure values from `train`, and after the existing fallback prediction is produced, replace each prediction by its nearest neighbor on that grid (vectorized, fast). This is a minimal post-processing step, preserves the full pipeline, and should move the score substantially toward the target.'
- What this solution (achieved 4.5526) has done: 'Your current MAE (4.55321, lower is better) is still far above the target (0.1394), so we should improve accuracy while keeping your same “train-derived lookup/join + per-group linear calibration + pressure-grid snapping” core logic. The main remaining issue is that the median tables and calibration are built across *all* rows (including expiratory phase where `u_out=1`), but Kaggle only scores the inspiratory phase; mixing phases harms the learned mappings and increases error. I therefore build all lookup/calibration tables using only inspiratory rows (`u_out==0`) and then apply them to all test rows (expiratory rows aren’t scored, but still must be predicted), which is a minimal semantic-aligned change. Additionally, I add a tiny, deterministic improvement by including `u_out` in the final reindexing source (`t["id"]`) to ensure the prediction ordering is strictly tied to the merged frame (no logic change, just alignment safety).'
- What this solution (achieved 5.07358) has done: 'I fix the `merge_asof` runtime error by ensuring both left/right frames are sorted by the exact columns Pandas requires: primarily the `on` column (`u_in`) globally, and secondarily the `by` columns; your current sort order (`by` first) triggers “left keys must be sorted”. I keep the same lookup/calibration/fallback logic and only adjust the sorting and dtype alignment of the merge keys to make the join deterministic and valid. After the fix, the pipeline run end-to-end and write a valid `submission.csv` with the required `id,pressure` columns so cell 2 can load and inspect it. No score-tuning changes beyond this correctness fix are introduced.'
- What this solution (achieved 5.07365) has done: 'I fix the `merge_asof` crash by making sure all `by` merge keys (`R,C,t_idx,u_out`) have exactly matching dtypes on both left (test) and right (lookup tables), since your current code casts `u_out` to `int8` only on the left while the right tables keep `int64`. This is a pure runtime/correctness fix and does not change the modeling/lookup logic. I also add a small defensive cast for `R,C,t_idx` to keep them consistent across all intermediate tables, preventing similar dtype mismatches. Once the join works, the script proceed to write `submission.csv`, unblocking cell 2.'
- What this solution (achieved 4.02572) has done: 'Your current MAE (5.07, lower is better) is far worse than the target (0.139), and the main reason is that the `merge_asof` lookup by `u_in` within `(R,C,t_idx,u_out)` groups is extremely sparse: the exact `u_in` values rarely repeat at a fixed timestep, so “nearest” in a tiny group is effectively noise and dominates your fallback chain. To keep the same core lookup/fallback logic but make the most-specific signal meaningful, I switch the asof-join to use a lightly binned `u_in` (same data, same join, just a more robust key) and I also require a minimum per-group support before trusting the asof median; otherwise we fall through to your existing `u_in_r` median and linear calibration steps. This is a minimal, metric-aligned adjustment (still train-derived joins + fallbacks + snapping) that should materially reduce error toward the target without changing the overall approach. The submission format and ordering logic remain unchanged.'
- What this solution (achieved 4.92265) has done: 'Your current MAE (4.02572, lower is better) is far above the target (0.1394), so we should improve accuracy while keeping your same core “train-derived lookup/join with fallback hierarchy + linear calibration + pressure-grid snapping” logic. The main low-risk gain is to make the most-specific lookup less noisy by (1) using the *exact* per-(R,C,t_idx,u_out,u_in_b) median via a normal merge first, and only then (2) using `merge_asof` within-group as a fallback for the remaining unmatched rows. Additionally, we tighten `u_in` binning modestly (0.2 instead of 0.5) to increase specificity without exploding sparsity, and we slightly lower the minimum support threshold for trusting the calibration (so it helps more often) while keeping the same linear form and fallback order. These are minimal, deterministic changes that keep your approach intact and should move the score materially toward the much lower target while still producing a valid `submission.csv`.'
- What this solution (achieved 4.89414) has done: 'We keep your exact lookup/fallback + linear calibration + snapping pipeline, but make two minimal, metric-aligned fixes that should improve MAE toward the much lower target. First, we build the pressure snapping grid from the full training set (not only inspiratory rows) so predictions for any phase are snapped to valid competition pressures (your current grid is unnecessarily restrictive and can add error). Second, we add one additional, still “same-core-logic” fallback median keyed on `(R,C,t_idx,u_out,u_in_b)` (same binning as the asof table) to reduce reliance on noisier asof-nearest when exact binned matches exist. These are deterministic, small changes and keep the model semantics unchanged while improving accuracy.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os



## === cell 1
BASE_PATHS = [
    "/kaggle/input/ventilator-pressure-prediction",
    "/kaggle/data/ventilator-pressure-prediction",
    "/kaggle/input",
    "/kaggle/data",
]


def _find_file(filename: str) -> str:
    for bp in BASE_PATHS:
        cand = os.path.join(bp, filename)
        if os.path.exists(cand):
            return cand
        cand2 = os.path.join(bp, "ventilator-pressure-prediction", filename)
        if os.path.exists(cand2):
            return cand2
    raise FileNotFoundError(
        f"Could not find {filename} in known Kaggle paths: {BASE_PATHS}"
    )


train_path = _find_file("train.csv")
test_path = _find_file("test.csv")
sample_path = _find_file("sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sub = pd.read_csv(sample_path)

req_train = {"breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"}
req_test = {"id", "breath_id", "R", "C", "time_step", "u_in", "u_out"}
if not req_train.issubset(train.columns):
    missing = req_train - set(train.columns)
    raise ValueError(f"train.csv missing columns: {missing}")
if not req_test.issubset(test.columns):
    missing = req_test - set(test.columns)
    raise ValueError(f"test.csv missing columns: {missing}")

train = train.sort_values(["breath_id", "time_step"], kind="mergesort")
test = test.sort_values(["breath_id", "time_step"], kind="mergesort")

train["t_idx"] = train.groupby("breath_id").cumcount().astype("int16")
test["t_idx"] = test.groupby("breath_id").cumcount().astype("int16")

train["u_in_r"] = np.round(train["u_in"].astype("float32"), 2).astype("float32")
test["u_in_r"] = np.round(test["u_in"].astype("float32"), 2).astype("float32")

for df in (train, test):
    df["R"] = df["R"].astype("int16")
    df["C"] = df["C"].astype("int16")
    df["t_idx"] = df["t_idx"].astype("int16")
    df["u_out"] = df["u_out"].astype("int8")
    df["u_in"] = df["u_in"].astype("float32")

train_insp = train[train["u_out"] == 0].copy()
train_full = train.copy()

global_median_insp = float(train_insp["pressure"].median())
global_median_full = float(train_full["pressure"].median())

UIN_BIN = 0.2
train_insp["u_in_b"] = (
    np.round(train_insp["u_in"].astype("float32") / UIN_BIN) * UIN_BIN
).astype("float32")
train_full["u_in_b"] = (
    np.round(train_full["u_in"].astype("float32") / UIN_BIN) * UIN_BIN
).astype("float32")
test["u_in_b"] = (np.round(test["u_in"].astype("float32") / UIN_BIN) * UIN_BIN).astype(
    "float32"
)

key_asof = ["R", "C", "t_idx", "u_out"]

median_map_asof_insp = (
    train_insp.groupby(key_asof + ["u_in_b"], sort=False)["pressure"]
    .agg(p_k_asof="median", cnt="size")
    .reset_index()
)
median_map_asof_insp["u_in_b"] = median_map_asof_insp["u_in_b"].astype("float32")
median_map_asof_insp["R"] = median_map_asof_insp["R"].astype("int16")
median_map_asof_insp["C"] = median_map_asof_insp["C"].astype("int16")
median_map_asof_insp["t_idx"] = median_map_asof_insp["t_idx"].astype("int16")
median_map_asof_insp["u_out"] = median_map_asof_insp["u_out"].astype("int8")
median_map_asof_insp["cnt"] = median_map_asof_insp["cnt"].astype("int16")

median_map_asof_full = (
    train_full.groupby(key_asof + ["u_in_b"], sort=False)["pressure"]
    .agg(p_k_asof="median", cnt="size")
    .reset_index()
)
median_map_asof_full["u_in_b"] = median_map_asof_full["u_in_b"].astype("float32")
median_map_asof_full["R"] = median_map_asof_full["R"].astype("int16")
median_map_asof_full["C"] = median_map_asof_full["C"].astype("int16")
median_map_asof_full["t_idx"] = median_map_asof_full["t_idx"].astype("int16")
median_map_asof_full["u_out"] = median_map_asof_full["u_out"].astype("int8")
median_map_asof_full["cnt"] = median_map_asof_full["cnt"].astype("int16")

key_asof_bin = key_asof + ["u_in_b"]
median_map_asofbin_insp = (
    train_insp.groupby(key_asof_bin, sort=False)["pressure"].median().reset_index()
).rename(columns={"pressure": "p_k_asofbin"})
median_map_asofbin_full = (
    train_full.groupby(key_asof_bin, sort=False)["pressure"].median().reset_index()
).rename(columns={"pressure": "p_k_asofbin"})
for mm in (median_map_asofbin_insp, median_map_asofbin_full):
    mm["R"] = mm["R"].astype("int16")
    mm["C"] = mm["C"].astype("int16")
    mm["t_idx"] = mm["t_idx"].astype("int16")
    mm["u_out"] = mm["u_out"].astype("int8")
    mm["u_in_b"] = mm["u_in_b"].astype("float32")

key1b = ["R", "C", "t_idx", "u_in_r"]
median_map_1b_insp = (
    train_insp.groupby(key1b, sort=False)["pressure"].median().reset_index()
)
median_map_1b_full = (
    train_full.groupby(key1b, sort=False)["pressure"].median().reset_index()
)

key2 = ["R", "C", "t_idx", "u_out"]
median_map_2_insp = (
    train_insp.groupby(key2, sort=False)["pressure"].median().reset_index()
)
median_map_2_full = (
    train_full.groupby(key2, sort=False)["pressure"].median().reset_index()
)
for mm in (median_map_2_insp, median_map_2_full):
    mm["R"] = mm["R"].astype("int16")
    mm["C"] = mm["C"].astype("int16")
    mm["t_idx"] = mm["t_idx"].astype("int16")
    mm["u_out"] = mm["u_out"].astype("int8")

key3 = ["R", "C", "t_idx"]
median_map_3_insp = (
    train_insp.groupby(key3, sort=False)["pressure"].median().reset_index()
)
median_map_3_full = (
    train_full.groupby(key3, sort=False)["pressure"].median().reset_index()
)
for mm in (median_map_3_insp, median_map_3_full):
    mm["R"] = mm["R"].astype("int16")
    mm["C"] = mm["C"].astype("int16")
    mm["t_idx"] = mm["t_idx"].astype("int16")


def _make_calib(df: pd.DataFrame, g_keys: list[str]) -> pd.DataFrame:
    u = df["u_in"].astype("float64")
    p = df["pressure"].astype("float64")
    tmp = df.copy()
    tmp["_u2"] = (u * u).astype("float64")
    tmp["_up"] = (u * p).astype("float64")
    calib = (
        tmp.groupby(g_keys, sort=False)
        .agg(
            mean_u=("u_in", "mean"),
            mean_p=("pressure", "mean"),
            mean_u2=("_u2", "mean"),
            mean_up=("_up", "mean"),
            n=("pressure", "size"),
        )
        .reset_index()
    )
    ridge = 1e-6
    var_u = calib["mean_u2"].astype("float64") - np.square(
        calib["mean_u"].astype("float64")
    )
    cov_up = calib["mean_up"].astype("float64") - (
        calib["mean_u"].astype("float64") * calib["mean_p"].astype("float64")
    )
    calib["a"] = (cov_up / (var_u + ridge)).astype("float32")
    calib["b"] = (
        calib["mean_p"].astype("float64")
        - calib["a"].astype("float64") * calib["mean_u"].astype("float64")
    ).astype("float32")
    return calib[g_keys + ["a", "b", "n"]]


g1_keys = ["R", "C", "t_idx", "u_out"]
calib1_insp = _make_calib(train_insp, g1_keys)
calib1_full = _make_calib(train_full, g1_keys)
for cb in (calib1_insp, calib1_full):
    cb["R"] = cb["R"].astype("int16")
    cb["C"] = cb["C"].astype("int16")
    cb["t_idx"] = cb["t_idx"].astype("int16")
    cb["u_out"] = cb["u_out"].astype("int8")

g2_keys = ["R", "C", "t_idx"]
calib2_insp = _make_calib(train_insp, g2_keys)
calib2_full = _make_calib(train_full, g2_keys)
for cb in (calib2_insp, calib2_full):
    cb["R"] = cb["R"].astype("int16")
    cb["C"] = cb["C"].astype("int16")
    cb["t_idx"] = cb["t_idx"].astype("int16")

t = test[["id", "R", "C", "t_idx", "u_out", "u_in", "u_in_r", "u_in_b"]].copy()
t["R"] = t["R"].astype("int16")
t["C"] = t["C"].astype("int16")
t["t_idx"] = t["t_idx"].astype("int16")
t["u_in"] = t["u_in"].astype("float32")
t["u_in_b"] = t["u_in_b"].astype("float32")
t["u_out"] = t["u_out"].astype("int8")

need_cols_left = key_asof + ["u_in_b"]
if t[need_cols_left].isna().any().any():
    raise ValueError("Nulls found in left merge_asof keys")


def _asof_join_only_missing(base: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if right[key_asof + ["u_in_b"]].isna().any().any():
        raise ValueError("Nulls found in right merge_asof keys")

    right2 = right.copy()
    for k in key_asof:
        right2[k] = right2[k].astype(base[k].dtype, copy=False)
    right2["u_in_b"] = right2["u_in_b"].astype(base["u_in_b"].dtype, copy=False)

    joined = base.merge(
        right2,
        on=key_asof + ["u_in_b"],
        how="left",
        suffixes=("", "_r"),
        copy=False,
    )

    miss_mask = joined["p_k_asof"].isna()
    if not bool(miss_mask.any()):
        return joined

    base_miss = joined.loc[miss_mask, base.columns].copy()

    left_sorted = base_miss.sort_values(
        ["u_in_b"] + key_asof, kind="mergesort"
    ).reset_index(drop=True)
    right_sorted = right2.sort_values(
        ["u_in_b"] + key_asof, kind="mergesort"
    ).reset_index(drop=True)

    filled = pd.merge_asof(
        left_sorted,
        right_sorted,
        on="u_in_b",
        by=key_asof,
        direction="nearest",
        allow_exact_matches=True,
    )

    joined.loc[miss_mask, "p_k_asof"] = filled["p_k_asof"].to_numpy()
    joined.loc[miss_mask, "cnt"] = filled["cnt"].to_numpy()
    return joined


t0 = t[t["u_out"] == 0].copy()
t1 = t[t["u_out"] == 1].copy()

t0 = t0.merge(median_map_asofbin_insp, on=key_asof_bin, how="left")
t1 = t1.merge(median_map_asofbin_full, on=key_asof_bin, how="left")

t0 = _asof_join_only_missing(t0, median_map_asof_insp)
t1 = _asof_join_only_missing(t1, median_map_asof_full)

t0 = t0.merge(
    median_map_1b_insp.rename(columns={"pressure": "p_k1b"}), on=key1b, how="left"
)
t1 = t1.merge(
    median_map_1b_full.rename(columns={"pressure": "p_k1b"}), on=key1b, how="left"
)

t0 = t0.merge(
    median_map_2_insp.rename(columns={"pressure": "p_k2"}), on=key2, how="left"
)
t1 = t1.merge(
    median_map_2_full.rename(columns={"pressure": "p_k2"}), on=key2, how="left"
)

t0 = t0.merge(
    median_map_3_insp.rename(columns={"pressure": "p_k3"}), on=key3, how="left"
)
t1 = t1.merge(
    median_map_3_full.rename(columns={"pressure": "p_k3"}), on=key3, how="left"
)

t0 = t0.merge(calib1_insp, on=g1_keys, how="left")
t1 = t1.merge(calib1_full, on=g1_keys, how="left")

CALIB1_MIN_N = 10
t0["p_calib"] = np.where(
    (t0["n"].fillna(0).astype("int32") >= CALIB1_MIN_N)
    & t0["a"].notna()
    & t0["b"].notna(),
    (
        t0["a"].astype("float32") * t0["u_in"].astype("float32")
        + t0["b"].astype("float32")
    ).astype("float32"),
    np.nan,
).astype("float32")
t1["p_calib"] = np.where(
    (t1["n"].fillna(0).astype("int32") >= CALIB1_MIN_N)
    & t1["a"].notna()
    & t1["b"].notna(),
    (
        t1["a"].astype("float32") * t1["u_in"].astype("float32")
        + t1["b"].astype("float32")
    ).astype("float32"),
    np.nan,
).astype("float32")

t0 = t0.merge(calib2_insp, on=g2_keys, how="left", suffixes=("", "_tidx"))
t1 = t1.merge(calib2_full, on=g2_keys, how="left", suffixes=("", "_tidx"))

t0["p_calib2"] = np.where(
    (t0["n_tidx"].fillna(0).astype("int32") >= 50)
    & t0["a_tidx"].notna()
    & t0["b_tidx"].notna(),
    (
        t0["a_tidx"].astype("float32") * t0["u_in"].astype("float32")
        + t0["b_tidx"].astype("float32")
    ).astype("float32"),
    np.nan,
).astype("float32")
t1["p_calib2"] = np.where(
    (t1["n_tidx"].fillna(0).astype("int32") >= 50)
    & t1["a_tidx"].notna()
    & t1["b_tidx"].notna(),
    (
        t1["a_tidx"].astype("float32") * t1["u_in"].astype("float32")
        + t1["b_tidx"].astype("float32")
    ).astype("float32"),
    np.nan,
).astype("float32")


def _predict(df: pd.DataFrame, global_median: float) -> np.ndarray:
    MIN_ASOF_CNT = 5
    p_asof = df["p_k_asof"].astype("float32")
    p_asof = p_asof.mask(df["cnt"].fillna(0).astype("int32") < MIN_ASOF_CNT)

    pred_local = df["p_k_asofbin"].astype("float32")
    pred_local = pred_local.fillna(p_asof)
    pred_local = pred_local.fillna(df["p_k1b"].astype("float32"))
    pred_local = pred_local.fillna(df["p_calib"].astype("float32"))
    pred_local = pred_local.fillna(df["p_calib2"].astype("float32"))
    pred_local = pred_local.fillna(df["p_k2"].astype("float32"))
    pred_local = pred_local.fillna(df["p_k3"].astype("float32"))
    pred_local = pred_local.fillna(global_median).astype("float32").values
    return pred_local


pred0 = _predict(t0, global_median_insp)
pred1 = _predict(t1, global_median_full)

t0["pred"] = pred0
t1["pred"] = pred1

t_all = pd.concat([t0[["id", "pred"]], t1[["id", "pred"]]], axis=0, ignore_index=True)

pressure_grid = np.sort(train_full["pressure"].unique()).astype("float32")

pred = t_all["pred"].astype("float32").values
idx = np.searchsorted(pressure_grid, pred, side="left")
idx = np.clip(idx, 0, len(pressure_grid) - 1)
idx_left = np.clip(idx - 1, 0, len(pressure_grid) - 1)
right = pressure_grid[idx]
left = pressure_grid[idx_left]
choose_left = np.abs(pred - left) <= np.abs(right - pred)
pred = np.where(choose_left, left, right).astype("float32")

if "id" not in sub.columns:
    raise ValueError("sample_submission.csv must contain 'id' column")

out = pd.DataFrame({"id": t_all["id"].values, "pressure": pred})
out = out.set_index("id").reindex(sub["id"].values).reset_index()

if out.shape[0] != sub.shape[0]:
    raise ValueError(
        f"Submission row count mismatch: got {out.shape[0]}, expected {sub.shape[0]}"
    )
if out["pressure"].isna().any():
    raise ValueError("NaNs found in predicted pressure")

out.to_csv("submission.csv", index=False)
out.head()



## === cell 2
check = pd.read_csv("submission.csv")
print(check.head())
print(check.shape)
print(check.columns.tolist())
print(
    "pressure stats:",
    check["pressure"].min(),
    check["pressure"].max(),
    check["pressure"].mean(),
)
