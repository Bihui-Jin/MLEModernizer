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

No external packages required in the script and installed.

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

0.2125954295882218

# 6. Current score

4.04396

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 19.55027) has done: 'Your notebook fails early due to (1) Jupyter-only magics (`%matplotlib inline`) and a TensorFlow/protobuf import crash, and later due to missing ensemble submission files, which leaves `test_preds` empty and breaks concatenation. I make the TensorFlow stack optional (since it’s unused here) and remove the notebook magic so imports work in a script environment. Then I robustly load only ensemble files that actually exist; if none exist (as in your environment), I fall back to a simple, valid baseline prediction (all pressures = minimum train pressure, quantized to the known pressure grid) so a correct `.csv` submission is always produced. This keeps the core intent (ensemble + pressure-grid rounding) while ensuring end-to-end execution and a nontrivial score better than an arbitrary constant like 0.'
- What this solution (achieved 7.46367) has done: 'I remove the TensorFlow import block that is crashing due to a protobuf incompatibility, since it is not used anywhere in this ensemble/rounding script. Then I make the ensemble loading more robust (shape/length checks) and keep the exact median/mean + pressure-grid rounding logic unchanged. Finally, if no external ensemble files exist (as in your environment), I fall back to a slightly stronger but still simple baseline than “always PRESSURE_MIN”: predict a per-(R,C,u_out) median pressure from the training data (computed on inspiratory rows only), then apply the same pressure-grid rounding/clipping before writing valid `submission0.csv` and `submission1.csv`.'
- What this solution (achieved 6.02329) has done: 'Your current fallback baseline ignores the strong time-series structure (80 steps per breath) and uses only `(R,C,u_out)` medians, which explains the very high MAE. To move the score much closer to the target with minimal semantic change, I keep your same “fallback when no ensemble files exist” logic and the exact same pressure-grid rounding/clipping, but strengthen the fallback to use a per-time-step median conditioned on `(R,C,time_step,u_out)` (computed on inspiratory rows only). This preserves the core “training-set median lookup + round-to-known-pressure-grid” approach while using information the metric actually depends on (pressure evolution during inspiration). I also harden the `targets.reshape(-1,80,1)` step by sorting by `(breath_id, time_step)` so the pressure grid extraction is consistent.'
- What this solution (achieved 6.10819) has done: 'Your current fallback is still too coarse because `time_step` is being rounded and looked up via slow per-row `.get`, which causes many misses and defaulting to broader medians, inflating MAE. I keep the exact same “train median lookup baseline + pressure-grid rounding/clipping + write two submissions” core logic, but make the lookup exact and vectorized by using the known 80 within-breath step index (derived from sorted order) instead of rounded `time_step`. This preserves evaluation semantics while sharply reducing fallback error by matching the actual per-step pressure evolution patterns. I also ensure test is sorted by `(breath_id, time_step)` for a correct step index, then map predictions back to the original row order so the submission aligns with `id`.'
- What this solution (achieved 4.30314) has done: 'Your current fallback baseline is still far from the target because it ignores the strongest available signal: the instantaneous control input `u_in` (and its accumulated effect over the breath). I keep your exact “fallback-only when no ensemble files exist + median lookup + pressure-grid rounding/clipping + write two submissions” core logic, but strengthen the fallback median mapping to condition on `(R, C, u_out, step, u_in_bin)` and also add cumulative `u_in` (`cum_u_in`) binned by step to better capture pressure dynamics. These are still simple training medians (no model/loop/loss changes) and remain fully deterministic, but should substantially reduce MAE toward your target. I also preserve your sorting/step indexing and the final mapping back to the original test row order so the submission aligns correctly by `id`.'
- What this solution (achieved 3.29715) has done: 'Your current fallback is still missing an important piece of the time-series signal: the *history* of the controls beyond `cum_u_in`, especially short-term dynamics (derivative) and whether you’re in a plateau/transition (lags). I keep your exact “fallback-only median lookup + pressure-grid rounding/clipping + submission write” core logic, but strengthen the fallback mapping by adding small, deterministic features that are cheap and capture dynamics: `u_in_lag1`, `u_in_diff1`, and `cum_u_in` (with light binning) and then using a stricter-to-looser merge cascade. This should reduce MAE toward the target without changing the modeling approach or introducing training loops/models. I also keep the existing sorting/step indexing and mapping back to original test order so the submission aligns correctly by `id`.'
- What this solution (achieved 3.11941) has done: 'Your current score (3.29715 MAE) is far worse than the target (0.2126), so we should improve predictions while keeping your existing “training median lookup + merge cascade + pressure-grid rounding” core logic intact. The biggest minimal win is to make the lookup more consistent with the scoring (inspiratory only) and reduce merge misses by using a tighter, more informative discretization of the key continuous driver (`u_in`) and its cumulative effect, without changing the approach. Concretely, I replace the coarse hand-made `u_in_bins` with per-step quantile bins learned from the inspiratory training data (still deterministic), and I compute `cum_bin` using the same per-step quantile scheme but with more bins; this keeps the same cascade structure and median aggregation but increases hit-rate on the strongest conditioning signals. I also keep everything sorted and mapped back to original order exactly as you already do, and still round to the known pressure grid for submission.'
- What this solution (achieved 3.11971) has done: 'Your current fallback is still missing the single most important scoring rule: rows with `u_out==1` (expiratory phase) are ignored in the MAE, so we can safely force those predictions to a stable constant to reduce noise without affecting inspiratory quality. I keep your exact median-lookup + merge-cascade + pressure-grid rounding core logic, but (1) build medians only from inspiratory rows as you already do, (2) after creating the full-length test prediction, overwrite `u_out==1` predictions with the global inspiratory median (then pressure-grid round/clip as before). I also make the quantile edge creation slightly more robust to duplicated quantiles by uniquing edges before `digitize`, which reduces accidental binning pathologies and lookup misses while preserving the same approach. These are minimal, deterministic changes aimed at reducing MAE toward the target without changing the overall method.'
- What this solution (achieved 3.03841) has done: 'Your current MAE (3.11971) is much worse than the target (0.2126), so we should legitimately improve predictions while preserving your exact “train medians via merge-cascade + pressure-grid rounding” core logic. The biggest minimal gain is to make the discretization consistent and higher-signal: use the within-breath `step` index (already present) and add a *few more deterministic bins* for `u_in`/`cum_u_in`, plus one extra stable dynamic feature (`cum_u_in_lag1`) to reduce lookup misses without changing the approach. I also keep your important rule of overwriting `u_out==1` predictions to a constant (doesn’t affect scoring) and ensure all merges remain left-joins with the same fallback ladder. These changes should reduce inspiratory error by improving median-table hit rate and conditioning, moving the score toward the target.'
- What this solution (achieved 3.01716) has done: 'Your current score (3.03841 MAE) is still far above the target (0.2126), so we should improve predictions while keeping your exact “median lookup via merge-cascade + pressure-grid rounding + overwrite u_out==1” core logic intact. The smallest high-impact change is to make the binning for `u_in` and `cum_u_in` more faithful by learning step-specific quantile edges separately for each `(R,C)` group (since pressure dynamics differ strongly by lung attributes), while keeping the same cascade tables and joins. To avoid excessive misses from over-fragmentation, we keep your existing global step edges as a fallback when an `(R,C,step)` group is too small. Everything else (features, median aggregations, rounding/clipping, submission writing) remains the same, just with more informative bins to reduce inspiratory MAE toward the target.'
- What this solution (achieved 3.01613) has done: 'Your current score (3.01716 MAE, lower-is-better) is far above the target (0.2126), so we should improve accuracy while preserving your exact “inspiratory-train median lookup via merge cascade + pressure-grid rounding + overwrite u_out==1” core logic. The main issue is that your binning/lookup can still miss often because bins are derived from raw `u_in` and `cum_u_in`, which vary by lung setting and breath intensity; adding one more deterministic, high-signal history feature (`cum_u_in_diff1`, i.e., per-step increment) improves conditioning without changing the approach. Concretely, I add `cum_u_in_diff1` and a small fixed binning for it, then insert one additional (still median-table) merge level that uses `(R,C,u_out,step,u_in_bin,cum_bin,cum_u_in_diff1_bin)` before falling back to your existing tables. This is a minimal, deterministic extension that should reduce inspiratory MAE and move the score closer to the target without touching model architectures/training loops (none exist here).'
- What this solution (achieved 3.01613) has done: 'Your current score (3.01613 MAE) is far above the target (0.2126), so we need a legitimate accuracy gain while keeping your exact “median lookup via merge-cascade + pressure-grid rounding + overwrite u_out==1” approach. The smallest high-impact issue is that your binning is computed on inspiratory rows only, but you currently assign bins for *all* test rows (including u_out==1), which can distort bin edges/assignments and increases merge misses/noise before you later overwrite expiratory predictions. I compute bins and do all median-table merges only on test inspiratory rows, then fill expiratory rows with the global inspiratory median (as you already do), preserving evaluation semantics and reducing miss-rate. Additionally, I make the bin assignment functions robust by initializing the output with zeros so no uninitialized values can leak in when an (R,C) combination is absent (rare but possible), which stabilizes merges and should move MAE down toward the target.'
- What this solution (achieved 3.01613) has done: 'Your current score (3.01613 MAE, lower is better) is far above the target (0.2126), so we should improve accuracy while keeping your exact “inspiratory-train median lookup via merge-cascade + pressure-grid rounding + overwrite u_out==1” core logic. The smallest high-impact fix is to stop “quantile-binning by (R,C,step)” from being distorted by mixing very different `u_out` regimes: compute bin edges only from inspiratory (`u_out==0`) rows (as you already do for `insp`), but also compute `(R,C,step)` edges using only those inspiratory rows *and* only assign bins/perform merges on test inspiratory rows (you intended this, but your edge-building currently includes `u_out` in group keys downstream and still uses `u_out` during merges, which fragments tables). I remove `u_out` from all median-table groupby keys (since for training it’s constant 0 anyway) and from all merge keys, which increases table density and reduces lookup misses without changing the approach. Finally, I keep your rule to overwrite `u_out==1` predictions with the global inspiratory median and then apply the same pressure-grid rounding/clipping for a valid submission.'
- What this solution (achieved 2.77562) has done: 'Your current score (3.01613 MAE; lower is better) is far above the target (0.2126), so we should improve accuracy while preserving your existing “inspiratory-train median lookup via merge-cascade + pressure-grid rounding + overwrite u_out==1” core logic. The most impactful minimal change is to stop discretizing `u_in`/`cum_u_in` into bins (which still causes many lookup misses) and instead use exact keys on the values that are already highly discrete in this dataset after sorting by step: `u_in` and `cum_u_in` rounded to 1 decimal, plus step and (R,C). We keep your merge-cascade structure, but add a new top-level exact-lookup table `(R,C,step,u_in_1dp,cum_u_in_1dp)->median pressure` and then fall back to your existing binned tables when exact matches are missing. This should materially reduce inspiratory MAE (fewer misses and more specific conditioning) without changing the approach or introducing any model/training loop, and it still produce valid `submission0.csv` and `submission1.csv`.'
- What this solution (achieved 4.04396) has done: 'Your current score (2.77562 MAE, lower-is-better) is still far above the target (0.2126), so we should improve the fallback predictions while keeping the same “inspiratory-only train medians via merge-cascade + pressure-grid rounding + overwrite u_out==1” core logic. The smallest high-impact fix is that the top-level “exact lookup” currently keys on `(u_in_1dp, cum_u_in_1dp)`, which is still too continuous and leads to many misses; we switch that exact lookup to use `(u_in_1dp, cum_diff1_1dp)` (where `cum_diff1` is essentially the per-step increment, close to `u_in` but more stable) to increase exact-match hit rate without introducing a new modeling approach. We keep all your existing binned-table cascade unchanged as fallbacks, and we keep the same expiratory overwrite and the same pressure-grid rounding/clipping. This should legitimately reduce inspiratory MAE and move the score closer to the target while remaining deterministic and fast.'

# 9. Code solution

## === cell 0
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

Ensembles = [
    "../input/sub-files/Submission files/sub (1).csv",
    "../input/sub-files/Submission files/sub (2).csv",
    "../input/sub-files/Submission files/sub (3).csv",
    "../input/sub-files/Submission files/sub (4).csv",
    "../input/sub-files/Submission files/sub (5).csv",
    "../input/sub-files/Submission files/sub (6).csv",
    "../input/sub-files/Submission files/sub (7).csv",
    "../input/sub-files/Submission files/sub (8).csv",
    "../input/sub-files/Submission files/sub (9).csv",
    "../input/sub-files/Submission files/sub (10).csv",
    "../input/sub-files/Submission files/sub (11).csv",
    "../input/sub-files/Submission files/sub (12).csv",
    "../input/sub-files/Submission files/sub (13).csv",
    "../input/sub-files/Submission files/sub (14).csv",
    "../input/sub-files/Submission files/sub (15).csv",
    "../input/sub-files/Submission files/sub (16).csv",
    "../input/sub-files/Submission files/sub (17).csv",
    "../input/sub-files/Submission files/sub (18).csv",
    "../input/sub-files/Submission files/sub (19).csv",
    "../input/sub-files/Submission files/sub (20).csv",
    "../input/sub-files/Submission files/sub (21).csv",
    "../input/sub-files/Submission files/sub (22).csv",
]

tf = None
keras = None



## === cell 1
train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")

train_sorted = train.sort_values(
    ["breath_id", "time_step"], kind="mergesort"
).reset_index(drop=True)
targets = train_sorted["pressure"].to_numpy().reshape(-1, 80, 1)

train_sorted["step"] = (
    train_sorted.groupby("breath_id", sort=False).cumcount().astype(np.int16)
)

test_sorted = test.sort_values(
    ["breath_id", "time_step"], kind="mergesort"
).reset_index()
test_sorted["step"] = (
    test_sorted.groupby("breath_id", sort=False).cumcount().astype(np.int16)
)
n_test = len(test)



## === cell 2
unique_pressures = np.unique(targets)
sorted_pressures = np.sort(unique_pressures)

PRESSURE_STEP = float((unique_pressures[1] - unique_pressures[0]).item())
PRESSURE_MIN = float(sorted_pressures[0].item())
PRESSURE_MAX = float(sorted_pressures[-1].item())



## === cell 3
test_preds = []
loaded_files = []

for predicted in Ensembles:
    if not os.path.exists(predicted):
        continue
    try:
        sub_df = pd.read_csv(predicted)
    except Exception:
        continue
    if "pressure" not in sub_df.columns:
        continue
    arr = sub_df["pressure"].to_numpy()
    if arr.shape[0] != n_test:
        continue
    test_preds.append(arr.reshape(-1, 1))
    loaded_files.append(predicted)



## === cell 4
if len(test_preds) == 0:
    insp = train_sorted[train_sorted["u_out"] == 0].copy()

    insp["cum_u_in"] = (
        insp.groupby("breath_id", sort=False)["u_in"].cumsum().astype(np.float32)
    )

    insp["u_in_lag1"] = (
        insp.groupby("breath_id", sort=False)["u_in"]
        .shift(1)
        .fillna(0.0)
        .astype(np.float32)
    )
    insp["u_in_diff1"] = (insp["u_in"].astype(np.float32) - insp["u_in_lag1"]).astype(
        np.float32
    )

    insp["cum_u_in_lag1"] = (
        insp.groupby("breath_id", sort=False)["cum_u_in"]
        .shift(1)
        .fillna(0.0)
        .astype(np.float32)
    )

    insp["cum_u_in_diff1"] = (
        insp["cum_u_in"].astype(np.float32) - insp["cum_u_in_lag1"].astype(np.float32)
    ).astype(np.float32)

    test_sorted_feat = test_sorted.copy()
    test_sorted_feat["cum_u_in"] = (
        test_sorted_feat.groupby("breath_id", sort=False)["u_in"]
        .cumsum()
        .astype(np.float32)
    )
    test_sorted_feat["u_in_lag1"] = (
        test_sorted_feat.groupby("breath_id", sort=False)["u_in"]
        .shift(1)
        .fillna(0.0)
        .astype(np.float32)
    )
    test_sorted_feat["u_in_diff1"] = (
        test_sorted_feat["u_in"].astype(np.float32) - test_sorted_feat["u_in_lag1"]
    ).astype(np.float32)
    test_sorted_feat["cum_u_in_lag1"] = (
        test_sorted_feat.groupby("breath_id", sort=False)["cum_u_in"]
        .shift(1)
        .fillna(0.0)
        .astype(np.float32)
    )

    test_sorted_feat["cum_u_in_diff1"] = (
        test_sorted_feat["cum_u_in"].astype(np.float32)
        - test_sorted_feat["cum_u_in_lag1"].astype(np.float32)
    ).astype(np.float32)

    q_u = np.linspace(0.0, 1.0, 17)  # 16 bins
    q_c = np.linspace(0.0, 1.0, 25)  # 24 bins

    def _make_edges(vals, q, default_edges):
        if vals.size == 0:
            return default_edges
        edges = np.quantile(vals, q).astype(np.float32)
        edges = np.maximum.accumulate(edges)
        edges = np.unique(edges)
        if edges.size < 3:
            return default_edges
        return edges

    step_u_edges_global = {}
    for s in range(80):
        vals = insp.loc[insp["step"] == s, "u_in"].to_numpy(np.float32)
        step_u_edges_global[s] = _make_edges(
            vals, q_u, np.array([0.0, 50.0, 100.0], dtype=np.float32)
        )

    step_cum_edges_global = {}
    for s in range(80):
        vals = insp.loc[insp["step"] == s, "cum_u_in"].to_numpy(np.float32)
        if vals.size == 0:
            step_cum_edges_global[s] = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        else:
            step_cum_edges_global[s] = _make_edges(
                vals,
                q_c,
                np.array(
                    [vals.min(), vals.mean(), vals.max() + 1e-3], dtype=np.float32
                ),
            )

    MIN_SAMPLES_FOR_RC_STEP_EDGES = 400  # deterministic, avoids over-fragmentation

    step_u_edges_rc = {}
    step_cum_edges_rc = {}

    for (r, c), g_rc in insp.groupby(["R", "C"], sort=False):
        for s in range(80):
            g = g_rc[g_rc["step"] == s]
            if len(g) >= MIN_SAMPLES_FOR_RC_STEP_EDGES:
                step_u_edges_rc[(r, c, s)] = _make_edges(
                    g["u_in"].to_numpy(np.float32), q_u, step_u_edges_global[s]
                )
                step_cum_edges_rc[(r, c, s)] = _make_edges(
                    g["cum_u_in"].to_numpy(np.float32), q_c, step_cum_edges_global[s]
                )

    def assign_step_quantile_bin_rc(df, col, edges_rc, edges_global):
        out = np.zeros(len(df), dtype=np.int16)
        steps = df["step"].to_numpy(np.int16)
        r_arr = df["R"].to_numpy(np.int16)
        c_arr = df["C"].to_numpy(np.int16)
        x = df[col].to_numpy(np.float32)

        for r in (5, 20, 50):
            for c in (10, 20, 50):
                mask_rc = (r_arr == r) & (c_arr == c)
                if not mask_rc.any():
                    continue
                for s in range(80):
                    mask = mask_rc & (steps == s)
                    if not mask.any():
                        continue
                    edges = edges_rc.get((r, c, s), edges_global.get(s))
                    if edges is None or edges.size < 3:
                        out[mask] = 0
                    else:
                        out[mask] = np.digitize(x[mask], edges[1:-1]).astype(np.int16)
        return out

    def assign_cum_bin_rc(df, col, edges_rc, edges_global):
        out = np.zeros(len(df), dtype=np.int16)
        steps = df["step"].to_numpy(np.int16)
        r_arr = df["R"].to_numpy(np.int16)
        c_arr = df["C"].to_numpy(np.int16)
        x = df[col].to_numpy(np.float32)

        for r in (5, 20, 50):
            for c in (10, 20, 50):
                mask_rc = (r_arr == r) & (c_arr == c)
                if not mask_rc.any():
                    continue
                for s in range(80):
                    mask = mask_rc & (steps == s)
                    if not mask.any():
                        continue
                    edges = edges_rc.get((r, c, s), edges_global.get(s))
                    if edges is None or edges.size < 3:
                        out[mask] = 0
                    else:
                        out[mask] = np.digitize(x[mask], edges[1:-1]).astype(np.int16)
        return out

    insp["u_in_bin"] = assign_step_quantile_bin_rc(
        insp, "u_in", step_u_edges_rc, step_u_edges_global
    )
    insp["u_in_lag1_bin"] = assign_step_quantile_bin_rc(
        insp, "u_in_lag1", step_u_edges_rc, step_u_edges_global
    )

    test_insp = test_sorted_feat[test_sorted_feat["u_out"] == 0].copy()

    test_insp["u_in_bin"] = assign_step_quantile_bin_rc(
        test_insp, "u_in", step_u_edges_rc, step_u_edges_global
    )
    test_insp["u_in_lag1_bin"] = assign_step_quantile_bin_rc(
        test_insp, "u_in_lag1", step_u_edges_rc, step_u_edges_global
    )

    diff_bins = np.array(
        [
            -100.0,
            -25.0,
            -12.0,
            -6.0,
            -2.0,
            -0.5,
            0.0,
            0.5,
            2.0,
            6.0,
            12.0,
            25.0,
            100.0001,
        ],
        dtype=np.float32,
    )
    insp["u_in_diff1_bin"] = np.digitize(
        np.clip(insp["u_in_diff1"].to_numpy(np.float32), -100.0, 100.0), diff_bins
    ).astype(np.int16)
    test_insp["u_in_diff1_bin"] = np.digitize(
        np.clip(test_insp["u_in_diff1"].to_numpy(np.float32), -100.0, 100.0), diff_bins
    ).astype(np.int16)

    insp["cum_bin"] = assign_cum_bin_rc(
        insp, "cum_u_in", step_cum_edges_rc, step_cum_edges_global
    )
    insp["cum_lag1_bin"] = assign_cum_bin_rc(
        insp, "cum_u_in_lag1", step_cum_edges_rc, step_cum_edges_global
    )

    test_insp["cum_bin"] = assign_cum_bin_rc(
        test_insp, "cum_u_in", step_cum_edges_rc, step_cum_edges_global
    )
    test_insp["cum_lag1_bin"] = assign_cum_bin_rc(
        test_insp, "cum_u_in_lag1", step_cum_edges_rc, step_cum_edges_global
    )

    cumdiff_bins = np.array(
        [
            -1e-3,
            0.0,
            0.5,
            1.0,
            2.0,
            4.0,
            6.0,
            8.0,
            10.0,
            15.0,
            20.0,
            30.0,
            50.0,
            100.0001,
        ],
        dtype=np.float32,
    )
    insp["cum_diff1_bin"] = np.digitize(
        np.clip(insp["cum_u_in_diff1"].to_numpy(np.float32), 0.0, 100.0), cumdiff_bins
    ).astype(np.int16)
    test_insp["cum_diff1_bin"] = np.digitize(
        np.clip(test_insp["cum_u_in_diff1"].to_numpy(np.float32), 0.0, 100.0),
        cumdiff_bins,
    ).astype(np.int16)

    insp["u_in_1dp"] = np.round(insp["u_in"].to_numpy(np.float32), 1).astype(np.float32)
    insp["cum_diff1_1dp"] = np.round(
        insp["cum_u_in_diff1"].to_numpy(np.float32), 1
    ).astype(np.float32)

    test_insp["u_in_1dp"] = np.round(test_insp["u_in"].to_numpy(np.float32), 1).astype(
        np.float32
    )
    test_insp["cum_diff1_1dp"] = np.round(
        test_insp["cum_u_in_diff1"].to_numpy(np.float32), 1
    ).astype(np.float32)

    grp_med_exact = insp.groupby(
        ["R", "C", "step", "u_in_1dp", "cum_diff1_1dp"], sort=False
    )["pressure"].median()

    grp_med_rclags = insp.groupby(
        ["R", "C", "step", "u_in_bin", "u_in_lag1_bin", "u_in_diff1_bin"], sort=False
    )["pressure"].median()

    grp_med_step_uin_cum = insp.groupby(
        ["R", "C", "step", "u_in_bin", "cum_bin", "cum_lag1_bin"], sort=False
    )["pressure"].median()

    grp_med_step_uin_cumdiff = insp.groupby(
        ["R", "C", "step", "u_in_bin", "cum_bin", "cum_diff1_bin"], sort=False
    )["pressure"].median()

    grp_med_step_uin = insp.groupby(["R", "C", "step", "u_in_bin"], sort=False)[
        "pressure"
    ].median()
    grp_med_step_cum = insp.groupby(["R", "C", "step", "cum_bin"], sort=False)[
        "pressure"
    ].median()
    grp_med_step = insp.groupby(["R", "C", "step"], sort=False)["pressure"].median()
    grp_med_rc = insp.groupby(["R", "C"], sort=False)["pressure"].median()
    global_med = float(insp["pressure"].median())

    base_df = test_insp[
        [
            "R",
            "C",
            "step",
            "u_in_1dp",
            "cum_diff1_1dp",
            "u_in_bin",
            "u_in_lag1_bin",
            "u_in_diff1_bin",
            "cum_bin",
            "cum_lag1_bin",
            "cum_diff1_bin",
        ]
    ].copy()

    base_df["pressure"] = base_df.merge(
        grp_med_exact.rename("pexact").reset_index(),
        on=["R", "C", "step", "u_in_1dp", "cum_diff1_1dp"],
        how="left",
    )["pexact"].to_numpy()

    miss = np.isnan(base_df["pressure"].to_numpy())
    if miss.any():
        tmp = (
            base_df.loc[
                miss, ["R", "C", "step", "u_in_bin", "u_in_lag1_bin", "u_in_diff1_bin"]
            ]
            .merge(
                grp_med_rclags.rename("p0").reset_index(),
                on=["R", "C", "step", "u_in_bin", "u_in_lag1_bin", "u_in_diff1_bin"],
                how="left",
            )["p0"]
            .to_numpy()
        )
        base_df.loc[miss, "pressure"] = tmp

    miss = np.isnan(base_df["pressure"].to_numpy())
    if miss.any():
        tmp = (
            base_df.loc[
                miss, ["R", "C", "step", "u_in_bin", "cum_bin", "cum_diff1_bin"]
            ]
            .merge(
                grp_med_step_uin_cumdiff.rename("p1a").reset_index(),
                on=["R", "C", "step", "u_in_bin", "cum_bin", "cum_diff1_bin"],
                how="left",
            )["p1a"]
            .to_numpy()
        )
        base_df.loc[miss, "pressure"] = tmp

    miss = np.isnan(base_df["pressure"].to_numpy())
    if miss.any():
        tmp = (
            base_df.loc[miss, ["R", "C", "step", "u_in_bin", "cum_bin", "cum_lag1_bin"]]
            .merge(
                grp_med_step_uin_cum.rename("p1b").reset_index(),
                on=["R", "C", "step", "u_in_bin", "cum_bin", "cum_lag1_bin"],
                how="left",
            )["p1b"]
            .to_numpy()
        )
        base_df.loc[miss, "pressure"] = tmp

    miss = np.isnan(base_df["pressure"].to_numpy())
    if miss.any():
        tmp = (
            base_df.loc[miss, ["R", "C", "step", "u_in_bin"]]
            .merge(
                grp_med_step_uin.rename("p1").reset_index(),
                on=["R", "C", "step", "u_in_bin"],
                how="left",
            )["p1"]
            .to_numpy()
        )
        base_df.loc[miss, "pressure"] = tmp

    miss = np.isnan(base_df["pressure"].to_numpy())
    if miss.any():
        tmp = (
            base_df.loc[miss, ["R", "C", "step", "cum_bin"]]
            .merge(
                grp_med_step_cum.rename("p2").reset_index(),
                on=["R", "C", "step", "cum_bin"],
                how="left",
            )["p2"]
            .to_numpy()
        )
        base_df.loc[miss, "pressure"] = tmp

    miss = np.isnan(base_df["pressure"].to_numpy())
    if miss.any():
        tmp = (
            base_df.loc[miss, ["R", "C", "step"]]
            .merge(
                grp_med_step.rename("p3").reset_index(),
                on=["R", "C", "step"],
                how="left",
            )["p3"]
            .to_numpy()
        )
        base_df.loc[miss, "pressure"] = tmp

    miss = np.isnan(base_df["pressure"].to_numpy())
    if miss.any():
        tmp = (
            base_df.loc[miss, ["R", "C"]]
            .merge(grp_med_rc.rename("p4").reset_index(), on=["R", "C"], how="left")[
                "p4"
            ]
            .fillna(global_med)
            .to_numpy()
        )
        base_df.loc[miss, "pressure"] = tmp

    baseline = np.empty(n_test, dtype=np.float64)

    baseline_sorted = np.full(len(test_sorted_feat), np.nan, dtype=np.float64)
    baseline_sorted[test_sorted_feat["u_out"].to_numpy() == 0] = base_df[
        "pressure"
    ].to_numpy(dtype=np.float64)

    baseline[test_sorted["index"].to_numpy()] = baseline_sorted

    baseline[test["u_out"].to_numpy() == 1] = global_med

    nan_mask = np.isnan(baseline)
    if nan_mask.any():
        baseline[nan_mask] = global_med

    predictions_median = baseline.reshape(-1, 1)
else:
    predictions_median = np.concatenate(test_preds, axis=-1)



## === cell 5
median_pre = np.median(predictions_median, axis=-1)
rounding_pre = (
    np.round((median_pre - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP + PRESSURE_MIN
)
clipped_pre = np.clip(rounding_pre, PRESSURE_MIN, PRESSURE_MAX)



## === cell 6
submission_file = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)
submission_file["pressure"] = clipped_pre.reshape(-1)
submission_file.to_csv("submission0.csv", index=False)



## === cell 7
mean_pre = np.mean(predictions_median, axis=-1)
rounding_pre = (
    np.round((mean_pre - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP + PRESSURE_MIN
)
clipped_pre = np.clip(rounding_pre, PRESSURE_MIN, PRESSURE_MAX)



## === cell 8
submission_file = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)
submission_file["pressure"] = clipped_pre.reshape(-1)
submission_file.to_csv("submission1.csv", index=False)
