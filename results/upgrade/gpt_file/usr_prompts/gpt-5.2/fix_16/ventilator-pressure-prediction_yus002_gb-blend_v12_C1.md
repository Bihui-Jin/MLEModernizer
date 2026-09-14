# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.9

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

0.5066636070438698

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 7.46367) has done: 'I fix the pipeline so it no longer depends on missing external Kaggle datasets (`../input/gb-blending/...`) and instead produces a valid submission using only the provided competition files. The current notebook is purely a blending script; since there are no blend source CSVs available, the minimal correct fallback is to generate a baseline prediction from `train.csv` and align it to `test.csv` ids. I implement a simple, deterministic `R,C`-grouped median pressure baseline computed on inspiratory phase (`u_out==0`), which is score-improving versus all-zeros while keeping the change minimal and robust. Finally, it write `submission.csv` with exactly `id,pressure`.'
- What this solution (achieved 6.10818) has done: 'Your current baseline ignores the time-series control inputs, so it can’t get close to the target MAE; we keep the same “train-derived deterministic baseline” core idea but make it time-step aware using only allowed columns. Specifically, compute an inspiratory-only median pressure lookup keyed by `(R, C, time_step_index)` (position within breath), and use that to predict each test row; this preserves the simple median-aggregation logic while aligning better with the per-timestep evaluation. We also keep a safe fallback hierarchy (RC+idx → RC → global) to avoid missing keys and ensure every row gets a prediction. The output still be a valid `submission.csv` with exactly `id,pressure`, aligned to the test ids.'
- What this solution (achieved 4.1668) has done: 'Your current lookup uses only `(R, C, position_in_breath)` and ignores the key control signal `u_in`, which drives most of the pressure variation; adding `u_in` to the median lookup is a minimal extension of the same “train-derived deterministic median baseline” core logic and should reduce MAE toward your target. To keep it robust and avoid missing-key holes, we use a small fallback hierarchy: `(R,C,t_idx,u_in_bin)` → `(R,C,t_idx)` → `(R,C)` → global median, so every test row still gets a prediction. We bin `u_in` (rather than using the raw float) to keep the lookup dense and stable without changing the overall approach. The submission writing is kept identical (`submission.csv` with `id,pressure`) and aligned to test `id`s.'
- What this solution (achieved 4.04274) has done: 'We keep your current “train-derived median lookup with fallbacks” core logic, but make one minimal change to better match the MAE metric by reducing discretization error from `u_in` binning. Specifically, we tighten `UIN_BIN_WIDTH` a bit (still dense enough to avoid excessive missing keys thanks to your fallback chain), which should move the score down (lower is better) toward the 0.5067 target. We also add a tiny safety clip of predictions to the observed training pressure range to avoid rare out-of-range medians due to missing-key fallbacks (this usually improves MAE slightly without changing the modeling approach). Submission writing and schema stay identical (`submission.csv` with `id,pressure`).'
- What this solution (achieved 4.01358) has done: 'Your current approach is a deterministic median-lookup baseline; to move the MAE down toward the target we should make it slightly less “blocky” without changing the overall logic. I keep the exact same lookup hierarchy and aggregations, but switch from hard floor-binning of `u_in` to nearest-bin rounding (same bin width) to reduce systematic bias around bin edges. I also ensure the submission `id` order exactly matches `test.csv` by directly using `test[['id']]` (avoids any potential merge/order issues) while keeping the same output schema and filename. These are minimal changes that should improve score modestly and safely within constraints.'
- What this solution (achieved 4.0642) has done: 'We keep your deterministic median-lookup baseline and fallback hierarchy exactly the same, but make the lookup slightly more expressive in a minimal way by adding a coarse `u_in` residual bin inside each `(R,C,t_idx,u_in_bin)` group. This preserves the same aggregation approach (groupby median) while reducing systematic error when two samples share the same rounded `u_in_bin` but have meaningfully different actual `u_in`. To avoid sparse-key issues, we keep the same fallback chain and only apply the residual-binned lookup when available; otherwise we fall back exactly as before. This should move MAE down (lower is better) from ~4.01 toward your target without changing the overall training/prediction semantics or requiring any new packages.'
- What this solution (achieved 4.0153) has done: 'Your current lookup is still too sparse at the deepest key level, so many test rows likely fall back to coarser medians and you don’t actually gain from the added residual bins. I make the smallest change that increases effective match rate: reduce `UIN_RESID_BINS` from 10 to 4 so the `(R,C,t_idx,u_in_bin,u_in_rbin)` groups are denser and used more often, which should lower MAE toward the target (lower is better). I also keep everything else (same median-lookup core logic, same fallback chain, same clipping, same submission writing) identical to minimize risk. This should be a safe incremental improvement without changing the modeling approach.'
- What this solution (achieved 4.01358) has done: 'Your current median-lookup baseline is already using the right signals, but it likely loses accuracy because many test keys fall back to coarser medians (sparsity) and because inspiratory-only `t_idx` in train can misalign with test `t_idx` when `u_out==1` appears (test `t_idx` counts all rows). I make two minimal, score-relevant fixes: compute `t_idx` on the full train (then filter to inspiratory for medians) so indexing matches test, and slightly reduce the sparsity of the deepest key by removing the residual-bin level (keep `(R,C,t_idx,u_in_bin)` as the finest key). This preserves the same “deterministic groupby-median with fallbacks” core logic and should move MAE down toward the target without changing the modeling approach. The script still write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 4.05902) has done: 'Your current MAE (4.01358) is far above the target (0.50666), so we should improve accuracy while keeping the same “train-derived deterministic median lookup with fallbacks” core logic. The biggest missing scored-signal is `u_out`: the competition scores only inspiratory (`u_out==0`), so predicting something sensible during expiratory steps (where pressure quickly drops) helps because those rows are effectively ignored by the metric but still must be present in the submission; however, many methods still benefit from explicitly handling `u_out==1` to avoid polluting fallbacks and to stabilize predictions. With minimal change, we (1) load `u_out` for test, (2) set predictions for `u_out==1` to a simple stable value (global inspiratory median), and (3) for `u_out==0` keep your exact lookup hierarchy but add one extra deepest-key option using raw `u_in` rounded to 0.1 (a denser “micro-bin”) before falling back to the 1.0 bin—this reduces discretization error without changing the overall approach. Submission writing remains `submission.csv` with `id,pressure` aligned to `test.csv` order.'
- What this solution (achieved 4.05902) has done: 'Your current score (4.059) is much worse than the target (0.5067), so we should improve accuracy while keeping the exact same “train-derived deterministic median lookup with fallbacks” approach. The biggest minimal gain left is to make `u_out==1` predictions more realistic: setting them to the inspiratory global median is arbitrary; instead we can learn an expiratory baseline from `train` (`u_out==1`) keyed by `(R,C,t_idx)` with the same fallback chain, without changing the model family. For inspiratory (`u_out==0`) we keep your existing hierarchy unchanged, but we also ensure the `t_idx` alignment remains identical (already fixed) and keep the same binning/rounding/clip behavior. This is a small, directly score-relevant change that should reduce MAE by better matching typical low pressures during expiration while preserving your core logic.'
- What this solution (achieved 4.0155) has done: 'Your current median-lookup core logic is fine but it’s leaving MAE on the table because the finest keys are too sparse and because expiratory (`u_out==1`) dynamics depend strongly on how much air was pushed in earlier in the breath. I keep the same deterministic “groupby median with fallback hierarchy” approach, but (1) slightly relax the inspiratory micro-binning (0.1→0.2) to increase match rate so fewer rows fall back to coarse medians, and (2) make expiratory predictions minimally more informed by adding a `u_in`-based expiratory lookup (still just medians) with the same fallbacks. These are small, directly score-relevant changes that preserve evaluation semantics and still write a valid `submission.csv` aligned to `test.csv` ids.'
- What this solution (achieved 3.98992) has done: 'We keep your existing deterministic “groupby median lookup with fallbacks” approach intact, but make the finest-grain lookup slightly denser by increasing the micro-bin width from 0.2 to 0.5 so fewer test rows fall back to coarser medians (which should reduce MAE from ~4.02 toward the 0.51 target). We also speed up and stabilize predictions (without changing semantics) by replacing the Python row loop with vectorized `MultiIndex.reindex` lookups that implement the same fallback hierarchy, reducing overhead and avoiding any subtle per-row branching inconsistencies. The expiratory (`u_out==1`) path remains learned from train expiratory medians with the same key structure and fallbacks. The script still runs end-to-end and writes a valid `submission.csv` with `id,pressure` aligned to `test.csv` order.'
- What this solution (achieved 4.0155) has done: 'Your current MAE (3.98992, lower is better) is still far above the target (0.50666), so we should improve accuracy with the smallest possible change inside your existing “median lookup with fallbacks” core logic. The biggest remaining error source is sparsity in the finest `(R,C,t_idx,u_in_mbin)` key: many rows likely miss and fall back to coarse medians. We can make the micro-binning denser (smaller width) so more test rows hit the finest lookup, while keeping the same hierarchy and semantics. I change only `UIN_MICRO_BIN_WIDTH` from `0.5` to `0.2` and keep everything else identical.'
- What this solution (achieved 4.05902) has done: 'To move MAE down toward your target while preserving the same deterministic “groupby median lookup with fallbacks” core logic, I only make the finest-grain key denser so more test rows hit the deepest lookup instead of falling back to coarser medians. Concretely, I reduce `UIN_MICRO_BIN_WIDTH` from `0.2` to `0.1` (keeping the same rounding-based binning, same hierarchy, same clipping, and same inspiratory/expiratory handling). This is the smallest single knob in your current approach that should improve accuracy without changing modeling semantics. Everything else (data paths, features used, aggregation type, fallback chain, and `submission.csv` schema) stays identical.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import copy
import glob
import random
from random import random as rd




## === cell 1
def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    """
    Original ensembling helper (kept for compatibility with the original core logic),
    but not used in the fixed end-to-end pipeline because external blend files are unavailable.
    """
    l = []
    for i in range(len(input_list)):
        public_lb_score = int(input_list[i].split("/")[-1].split(".")[1].split(" ")[0])
        l.append(public_lb_score)
        input_list[i] = (pd.read_csv(input_list[i]).pressure).ravel()
    output = 0
    if len(input_list) == 1:
        output = input_list[0]
    else:
        weight1 = 0.8
        weight2 = 0.2
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def g(dp):
    """
    Original randomized blending loop (kept), but requires existing prediction files in dp.
    """
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)
    file_count = len(l)
    if file_count == 0:
        raise FileNotFoundError(f"No files found to blend in directory: {dp}")
    loop_time = file_count**3
    splits = file_count // 2 if file_count // 2 > 0 else 1
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
        flist[i] = wc(flist[i])

    output = pd.read_csv(
        "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output.pressure = 0.0

    for it in range(loop_time):
        weight = []
        set_seed(it)
        for j in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        for j in range(len(weight)):
            weight[j] /= weight_sum
        weight.sort(reverse=True)
        for j in range(len(flist)):
            output.pressure += flist[j] * weight[j]

    output.pressure /= loop_time
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)
    return output




## === cell 2
def blend(a, b):
    """
    Original blending utility. Left intact, but now used only if both files exist.
    """
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a["pressure"] = a["pressure"] * 0.7 + b["pressure"] * 0.3
    a.to_csv("blend.csv", index=False)
    return a




## === cell 3
TRAIN_PATH = "/kaggle/input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "/kaggle/input/ventilator-pressure-prediction/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv"

UIN_BIN_WIDTH = 1.0

UIN_MICRO_QBINS = 32  # dense but not too sparse; chosen to reduce fallback frequency

train = pd.read_csv(
    TRAIN_PATH,
    usecols=["breath_id", "R", "C", "u_out", "u_in", "pressure"],
)
test = pd.read_csv(
    TEST_PATH,
    usecols=["id", "breath_id", "R", "C", "u_in", "u_out"],
)

train = train.copy()
train["t_idx"] = train.groupby("breath_id").cumcount().astype(np.int16)

test = test.copy()
test["t_idx"] = test.groupby("breath_id").cumcount().astype(np.int16)

uin_test = test["u_in"].to_numpy(dtype=np.float32)
test["u_in_bin"] = np.rint(uin_test / UIN_BIN_WIDTH).astype(np.int16)

train_insp = train[train["u_out"] == 0].copy()
train_exp = train[train["u_out"] == 1].copy()


def _make_edges_qbins(g_uin: pd.Series, qbins: int) -> np.ndarray:
    x = g_uin.to_numpy(dtype=np.float64)
    if x.size < 8:
        return np.array([-np.inf, np.inf], dtype=np.float64)
    qs = np.linspace(0.0, 1.0, qbins + 1)
    edges = np.quantile(x, qs)
    edges = np.unique(edges)
    if edges.size < 2:
        return np.array([-np.inf, np.inf], dtype=np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges.astype(np.float64)


def _assign_qbin_from_edges(u: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return (np.searchsorted(edges, u, side="right") - 1).astype(np.int16)


def _build_edges_map(df_insp: pd.DataFrame, qbins: int) -> dict:
    edges_map = {}
    for (R, C, t_idx), g in df_insp.groupby(["R", "C", "t_idx"], sort=False):
        edges_map[(int(R), int(C), int(t_idx))] = _make_edges_qbins(g["u_in"], qbins)
    return edges_map


def _assign_micro_bins(df: pd.DataFrame, edges_map: dict) -> np.ndarray:
    u = df["u_in"].to_numpy(dtype=np.float64)
    r = df["R"].to_numpy(dtype=np.int64)
    c = df["C"].to_numpy(dtype=np.int64)
    t = df["t_idx"].to_numpy(dtype=np.int64)
    out = np.empty(len(df), dtype=np.int16)
    for i in range(len(df)):
        edges = edges_map.get((int(r[i]), int(c[i]), int(t[i])))
        if edges is None:
            out[i] = np.int16(0)
        else:
            out[i] = _assign_qbin_from_edges(np.array([u[i]], dtype=np.float64), edges)[
                0
            ]
    return out


edges_map = _build_edges_map(train_insp, UIN_MICRO_QBINS)

train_insp["u_in_mbin"] = _assign_micro_bins(train_insp, edges_map)
test["u_in_mbin"] = _assign_micro_bins(test, edges_map)

if len(train_exp) > 0:
    train_exp = train_exp.copy()
    uin_train_exp = train_exp["u_in"].to_numpy(dtype=np.float32)
    train_exp["u_in_bin"] = np.rint(uin_train_exp / UIN_BIN_WIDTH).astype(np.int16)
    train_exp["u_in_mbin"] = _assign_micro_bins(train_exp, edges_map)

rc_t_umicro_median = train_insp.groupby(["R", "C", "t_idx", "u_in_mbin"], sort=False)[
    "pressure"
].median()
rc_t_u_median = train_insp.groupby(["R", "C", "t_idx", "u_in_bin"], sort=False)[
    "pressure"
].median()
rc_t_median = train_insp.groupby(["R", "C", "t_idx"], sort=False)["pressure"].median()
rc_median = train_insp.groupby(["R", "C"], sort=False)["pressure"].median()
global_median = float(train_insp["pressure"].median())

if len(train_exp) > 0:
    rc_t_umicro_exp_median = train_exp.groupby(
        ["R", "C", "t_idx", "u_in_mbin"], sort=False
    )["pressure"].median()
    rc_t_u_exp_median = train_exp.groupby(["R", "C", "t_idx", "u_in_bin"], sort=False)[
        "pressure"
    ].median()
    rc_t_exp_median = train_exp.groupby(["R", "C", "t_idx"], sort=False)[
        "pressure"
    ].median()
    rc_exp_median = train_exp.groupby(["R", "C"], sort=False)["pressure"].median()
    global_exp_median = float(train_exp["pressure"].median())
else:
    rc_t_umicro_exp_median = rc_t_umicro_median
    rc_t_u_exp_median = rc_t_u_median
    rc_t_exp_median = rc_t_median
    rc_exp_median = rc_median
    global_exp_median = global_median

keys = pd.MultiIndex.from_frame(test[["R", "C", "t_idx", "u_in_mbin"]].astype(np.int64))
keys_u = pd.MultiIndex.from_frame(
    test[["R", "C", "t_idx", "u_in_bin"]].astype(np.int64)
)
keys_t = pd.MultiIndex.from_frame(test[["R", "C", "t_idx"]].astype(np.int64))
keys_rc = pd.MultiIndex.from_frame(test[["R", "C"]].astype(np.int64))

pred = np.empty(len(test), dtype=np.float32)

mask_exp = test["u_out"].to_numpy() == 1
mask_insp = ~mask_exp

pred_insp = rc_t_umicro_median.reindex(keys).to_numpy()
miss = pd.isna(pred_insp)
if miss.any():
    tmp = rc_t_u_median.reindex(keys_u).to_numpy()
    pred_insp[miss] = tmp[miss]
    miss = pd.isna(pred_insp)
if miss.any():
    tmp = rc_t_median.reindex(keys_t).to_numpy()
    pred_insp[miss] = tmp[miss]
    miss = pd.isna(pred_insp)
if miss.any():
    tmp = rc_median.reindex(keys_rc).to_numpy()
    pred_insp[miss] = tmp[miss]
    miss = pd.isna(pred_insp)
if miss.any():
    pred_insp[miss] = global_median
pred[mask_insp] = pred_insp[mask_insp].astype(np.float32)

pred_exp = rc_t_umicro_exp_median.reindex(keys).to_numpy()
miss = pd.isna(pred_exp)
if miss.any():
    tmp = rc_t_u_exp_median.reindex(keys_u).to_numpy()
    pred_exp[miss] = tmp[miss]
    miss = pd.isna(pred_exp)
if miss.any():
    tmp = rc_t_exp_median.reindex(keys_t).to_numpy()
    pred_exp[miss] = tmp[miss]
    miss = pd.isna(pred_exp)
if miss.any():
    tmp = rc_exp_median.reindex(keys_rc).to_numpy()
    pred_exp[miss] = tmp[miss]
    miss = pd.isna(pred_exp)
if miss.any():
    pred_exp[miss] = global_exp_median
pred[mask_exp] = pred_exp[mask_exp].astype(np.float32)

pmin = float(train["pressure"].min())
pmax = float(train["pressure"].max())
pred = np.clip(pred, pmin, pmax)

sub = test[["id"]].copy()
sub["pressure"] = pred
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Pred stats:", float(np.min(pred)), float(np.mean(pred)), float(np.max(pred)))
print("u_in_bin width:", UIN_BIN_WIDTH, "| micro quantile bins:", UIN_MICRO_QBINS)
print("Clip range:", pmin, pmax)
print("Test u_out counts:", pd.Series(test["u_out"]).value_counts().to_dict())
print(
    "Train expiratory median:",
    global_exp_median,
    "| inspiratory median:",
    global_median,
)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3240894171.py in <cell line: 0>()
    102     "pressure"
    103 ].median()
--> 104 rc_t_u_median = train_insp.groupby(["R", "C", "t_idx", "u_in_bin"], sort=False)[
    105     "pressure"
    106 ].median()

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in groupby(self, by, axis, level, as_index, sort, group_keys, observed, dropna)
   9181             raise TypeError("You have to supply one of 'by' and 'level'")
   9182 
-> 9183         return DataFrameGroupBy(
   9184             obj=self,
   9185             keys=by,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/groupby.py in __init__(self, obj, keys, axis, level, grouper, exclusions, selection, as_index, sort, group_keys, observed, dropna)
   1327 
   1328         if grouper is None:
-> 1329             grouper, exclusions, obj = get_grouper(
   1330                 obj,
   1331                 keys,

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/grouper.py in get_grouper(obj, key, axis, level, sort, observed, validate, dropna)
   1041                 in_axis, level, gpr = False, gpr, None
   1042             else:
-> 1043                 raise KeyError(gpr)
   1044         elif isinstance(gpr, Grouper) and gpr.key is not None:
   1045             # Add key to exclusions

KeyError: 'u_in_bin'
