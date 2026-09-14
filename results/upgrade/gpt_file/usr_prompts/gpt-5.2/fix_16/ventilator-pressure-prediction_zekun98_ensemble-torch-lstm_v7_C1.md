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

0.1494432479927215

# 6. Current score

1.64434

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.92091) has done: 'I remove the dependency on missing external Kaggle datasets (`../input/keraslstm151`, `../input/lstmfold10146`) and make the script robust to the case where no `../input/torch*` folders exist (which currently causes “No objects to concatenate”). Then I generate a valid `submission.csv` directly from the provided competition `train.csv/test.csv` using a minimal, fast, deterministic fallback model (median pressure per (R,C,time_step) with a global median backup), which preserves the evaluation semantics and produces a reasonable baseline score instead of failing. Finally, I keep the original ensembling/averaging logic if those external files ever do exist, but guard it so it never crashes and always writes a valid `.csv` with the correct columns.'
- What this solution (achieved 9.91095) has done: 'Your current fallback uses only (R, C, time_step) medians, which ignores the key control inputs and makes MAE very large; we can keep the same “groupby-median lookup + global backup” core logic but include `u_in` and `u_out` in the aggregation keys to better match the data-generating process. To keep it robust and still fast, we add a small hierarchy: first try a median lookup on (R,C,time_step,u_out,u_in_rounded), then back off to (R,C,time_step,u_out), then (R,C,time_step), then global median. This stays deterministic, avoids any new modeling/training, preserves the quantization semantics, and should move the score substantially toward the target without relying on missing external submissions. We also keep your optional ensembling logic intact and ensure final `submission.csv` is always aligned to test `id`s.'
- What this solution (achieved 3.76984) has done: 'Your current fallback is still far from the target because it doesn’t respect the competition metric (only inspiratory timesteps are scored) and because exact matching on floating `time_step` causes many missed joins, forcing frequent fallback to coarse/global medians. I keep your core “groupby-median lookup + hierarchical backoff + quantize” approach, but (1) compute medians only on inspiratory rows (`u_out==0`) and (2) join on a robust integer `time_step` index (`time_step*100` rounded) to dramatically reduce merge misses without changing modeling/training logic. I also make the `cell 4` ensembling branch safe: if external submissions exist but are misaligned in length/ids, it fall back to the already-built `submission.csv` instead of producing a wrong submission. These minimal changes should move MAE strongly downward toward your target while staying deterministic and fast.'
- What this solution (achieved 3.04213) has done: 'I keep your existing “hierarchical groupby-median lookup + quantize + safe ensembling” core logic, but make two minimal changes that directly improve MAE for this competition: (1) learn separate median mappings for inspiratory vs expiratory (`u_out==0` vs `u_out==1`) instead of training only on inspiratory and forcing expiratory to fall back, and (2) add a small sequence-aware feature (`u_in` first-difference within each breath) into the finest-grain key (with rounding) so the lookup better matches the dynamics without introducing any training loop or model change. These changes should reduce the large remaining gap from 3.77 toward your 0.149 target while staying deterministic, fast, and producing a valid `submission.csv`. I also keep your external-submission ensembling guards intact and ensure the fallback submission always aligns to test `id`s.'
- What this solution (achieved 2.67963) has done: 'Your current fallback is still far from the target because it predicts pressure without explicitly using the strongest simple physics proxy available in this dataset: cumulative inspired volume (integral of `u_in` over time). I keep the same core “hierarchical median lookup + backoff + quantize” approach, but add two minimal, deterministic sequence features computed per breath (`u_in_cum` and a short moving-average of `u_in`) and use them only in the finest-grain key, so merges remain robust and backoff still works. This should reduce MAE substantially (move toward 0.149) without introducing any training loop, new model, or external data. I also keep your external-submission ensembling guards intact and ensure we still always write a valid `submission.csv`.'
- What this solution (achieved 2.25906) has done: 'Your current fallback is a deterministic hierarchical median-lookup; the biggest remaining gap to the target is that the “finest key” is still too sparse and misses often, so it falls back to coarser medians that wash out dynamics. I keep the exact same core logic (groupby medians + backoff + quantize), but make merges denser by (1) adding a breath-relative `step` index (0–79) as an additional stable key alongside `time_idx`, and (2) slightly coarsening the most granular bins for the sequence features (`u_in_diff`, `u_in_cum`, `u_in_ma5`) to reduce lookup sparsity while still using them. This should reduce fallback frequency and move MAE downward toward your target without changing architecture/training (none exists) or adding approximations beyond key quantization you already do. I also make the external-submission ensemble branch validate length/id alignment (like cell 4) so it can’t silently misalign and hurt score.'
- What this solution (achieved 3.8629) has done: 'We keep your exact “hierarchical groupby-median lookup + backoff + quantize + safe external ensembling” core logic, but make the finest-grain lookups much denser by replacing the volatile engineered sequence bins with stable, highly-informative keys already present in the data: the per-breath step index (0–79) and the `(u_in, u_out)` controls. Concretely, we (1) make `agg1` and `agg2` use `u_in_bin` at finer resolution and drop `u_in_diff/u_in_cum/u_in_ma5` from the key to avoid sparsity/missed joins, and (2) add a new intermediate backoff that uses `(R,C,step,u_out,u_in_bin)` (no `time_idx`) which tends to match well even if `time_step` alignment is slightly off. These changes preserve the same modeling approach (median-lookup with deterministic quantization), but should reduce fallback frequency and move MAE substantially downward toward your target. All output paths and submission format remain unchanged, and the external-submission ensemble guards stay in place.'
- What this solution (achieved 5.17102) has done: 'Your current score (3.8629 MAE) is far above the target (0.1494), so we need a real accuracy jump while keeping your “deterministic hierarchical median lookup + backoff + quantize” core logic intact. The biggest issue is that the fallback currently tries to predict continuous pressure directly, but in this competition pressure takes on a fixed discrete grid; predicting the nearest known pressure level per control-state dramatically reduces MAE. I keep your same aggregation/merge/backoff structure, but change the aggregated target from median(pressure) to the most frequent discrete pressure “level” (mode) within each key, and ensure all aggregations are computed on the already-quantized pressure grid. This is a minimal change (no new model/training) and should move your score much closer to the target band.'
- What this solution (achieved 4.99587) has done: 'We need to move your MAE down toward 0.149 (lower is better) from 5.17, but we must keep your core “hierarchical grouped lookup with backoff + quantization + safe ensembling” logic intact. The main issue is that your current keys are still too sparse/misaligned: you combine both `time_idx` and `step` in some aggs (redundant/noisy) and you bin `u_in` extremely finely (0.1), causing many unseen-key fallbacks to very coarse aggregates. I keep the same structure but (1) make the finest lookups denser by using slightly coarser `u_in` bins (1.0) and (2) add a strong, stable intermediate backoff on `(R,C,step,u_out,u_in_bin)` before any time-based keys, plus (3) use inspiratory-only statistics for inspiratory (`u_out==0`) lookups while keeping an expiratory mapping for `u_out==1`. These are minimal, deterministic changes that preserve your approach but should substantially reduce fallback frequency and lower MAE toward the target.'
- What this solution (achieved 5.0802) has done: 'Your current fallback is still missing a key signal: the pressure dynamics depend strongly on the combination of `u_out` and control state at each step, and right now the most-used (top-priority) mappings ignore `u_out` entirely and only use `u_in_bin` at a single coarse resolution, which causes frequent backoff to much coarser aggregates. I keep your exact “hierarchical grouped lookup with backoff + quantization + safe external ensembling” approach, but (1) include `u_out` in the finest/intermediate inspiratory/expiratory mappings (so the correct regime is matched directly), and (2) use a small two-level `u_in` binning (0.5 then 1.0) so we get better precision where data supports it while still avoiding sparsity. I also compute the global fallback separately for `u_out==0` and `u_out==1` so expiratory rows don’t inherit an inspiratory median (and vice-versa), which reduces large systematic errors without changing the modeling paradigm. All I/O paths and your external-submission guards remain unchanged, and it still always write a valid `submission.csv`.'
- What this solution (achieved 3.98998) has done: 'Your current MAE (5.0802, lower is better) is far above the target (0.1494), so we should improve accuracy while keeping your core “hierarchical grouped lookup with backoff + quantization + safe external ensembling” logic unchanged. The biggest issue is that using `mode` for pressure within keys is unstable/sparse here (pressure is quasi-continuous and spread across many levels), so it often collapses to arbitrary single counts and hurts predictions; switching back to a robust `median` on the already-quantized pressure grid usually reduces MAE substantially without changing the overall approach. I keep the same keys, binning, merges, and backoff order, but replace the mode aggregator with a median aggregator (quantized) and keep all I/O and ensembling guards intact. This is a minimal change focused directly on lowering MAE toward your target and still produces a valid `submission.csv`.'
- What this solution (achieved 2.06453) has done: 'Your current MAE (3.98998, lower is better) is still far above the target (0.14944), so we need a meaningful accuracy gain while keeping your same “hierarchical grouped median lookup + backoff + quantization + safe ensembling” approach. The smallest high-impact fix is to add one more deterministic, physics-relevant per-breath state feature: the cumulative inspired volume proxy `u_in_cum` (integral of `u_in` over time), then use a coarsely binned version of it only in the *finest* lookup keys so joins stay dense and backoff still works. This helps distinguish different pressure states that share the same `(R,C,step,u_out,u_in_bin)` but differ by how much air has already been delivered, which is crucial for this competition. Everything else (median aggregation, backoff order, quantization, I/O paths, external-submission guards) is preserved, and the script still always writes a valid `submission.csv`.'
- What this solution (achieved 1.7375) has done: 'We need to reduce MAE (lower is better) from 2.06453 toward 0.14944, so the smallest likely high-impact improvement is to keep your exact hierarchical median-lookup/backoff approach but make the cumulative-volume proxy closer to the true “integral of flow over time” by weighting `u_in` by the per-step `dt` rather than a plain cumulative sum. This preserves your core logic (same medians, same backoff, same quantization), but gives the finest-grain keys a more physically meaningful state signal, which should reduce ambiguity and lower MAE. To keep lookups dense (avoid sparsity), we also bin this weighted cumulative feature at a modest resolution and use it only in the finest keys exactly like your current `u_in_cum_bin`. All I/O paths and your safe external-submission ensembling guards remain intact, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 1.78156) has done: 'We need to reduce MAE (lower is better) from 1.7375 toward 0.1494, so we keep your exact hierarchical median-lookup + backoff structure but make the “state” key less sparse and more physically aligned by (1) using `u_in_int_cum` (cumulative of `u_in*dt`) directly instead of a coarse `u_in_cum_bin` and (2) binning it at a slightly finer but still dense resolution. Additionally, we split the time-indexed aggregates by `step` rather than `time_idx` where possible, because `step` is perfectly aligned across breaths and avoids float-to-int time alignment misses. These are minimal, deterministic changes that preserve your core approach (no training, same median aggregation/backoff, same quantization), but should reduce fallback frequency and move the score down toward the target. All I/O paths and your safe external-submission ensembling guards remain intact, and we still always write a valid `submission.csv`.'
- What this solution (achieved 1.64434) has done: 'Your current score (1.78156 MAE, lower is better) is still far above the target (0.14944), so we should improve accuracy with the smallest possible change to your existing hierarchical “grouped median lookup + backoff + quantize” fallback. The main issue is that your finest keys are likely too sparse because `u_in_int_cum_bin` at 0.5 creates many unseen combinations, causing frequent fallback to much coarser aggregates; we make that bin coarser to increase match rate while keeping the same logic and features. Additionally, we add one intermediate backoff level that uses `u_in_int_cum_bin` without `u_in_bin/u_in_bin_f` (still deterministic) to further reduce coarse/global fallback. All I/O, external-submission guards, quantization, and submission alignment remain unchanged.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import os

paths = glob.glob("../input/torch*")




## === cell 1
class config:
    paths = {
        "train": "../input/ventilator-pressure-prediction/train.csv",
        "test": "../input/ventilator-pressure-prediction/test.csv",
        "ss": "../input/ventilator-pressure-prediction/sample_submission.csv",
    }

    model_params = {
        "is_train": True,
        "debug": False,
        "EPOCH": 300,
        "BATCH_SIZE": 1024,
        "NUM_FOLDS": 10,
    }

    post_processing = {
        "max_pressure": 64.82099173863948,
        "min_pressure": -1.8957442945646408,
        "diff_pressure": 0.07030215,
    }




## === cell 2
def _quantize_pressure(x: np.ndarray) -> np.ndarray:
    mn = config.post_processing["min_pressure"]
    mx = config.post_processing["max_pressure"]
    dp = config.post_processing["diff_pressure"]
    x = np.round((x - mn) / dp) * dp + mn
    return np.clip(x, mn, mx)


def _make_time_idx(time_step: pd.Series) -> pd.Series:
    return np.rint(time_step.to_numpy(dtype=np.float64) * 100.0).astype(np.int16)


def _group_median_pressure(
    df: pd.DataFrame, keys: list[str], out_col: str
) -> pd.DataFrame:
    med = (
        df.groupby(keys, sort=False, observed=True)["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": out_col})
    )
    return med


def make_fallback_submission(
    train_path: str, test_path: str, ss_path: str
) -> pd.DataFrame:
    train = pd.read_csv(
        train_path,
        usecols=["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
    )
    test = pd.read_csv(
        test_path, usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
    )
    sub = pd.read_csv(ss_path, usecols=["id", "pressure"])

    train["time_idx"] = _make_time_idx(train["time_step"])
    test["time_idx"] = _make_time_idx(test["time_step"])

    train = train.sort_values(["breath_id", "time_idx"], kind="mergesort")
    test = test.sort_values(["breath_id", "time_idx"], kind="mergesort")
    train["step"] = train.groupby("breath_id", sort=False).cumcount().astype(np.int16)
    test["step"] = test.groupby("breath_id", sort=False).cumcount().astype(np.int16)

    uin_bin_fine = 0.5
    uin_bin_coarse = 1.0

    train["u_in_bin_f"] = (
        np.round(train["u_in"].to_numpy(dtype=np.float64) / uin_bin_fine) * uin_bin_fine
    ).astype(np.float32)
    test["u_in_bin_f"] = (
        np.round(test["u_in"].to_numpy(dtype=np.float64) / uin_bin_fine) * uin_bin_fine
    ).astype(np.float32)

    train["u_in_bin"] = (
        np.round(train["u_in"].to_numpy(dtype=np.float64) / uin_bin_coarse)
        * uin_bin_coarse
    ).astype(np.float32)
    test["u_in_bin"] = (
        np.round(test["u_in"].to_numpy(dtype=np.float64) / uin_bin_coarse)
        * uin_bin_coarse
    ).astype(np.float32)

    train["pressure"] = _quantize_pressure(
        train["pressure"].to_numpy(dtype=np.float64)
    ).astype(np.float32)

    train["dt"] = (
        train.groupby("breath_id", sort=False)["time_step"]
        .diff()
        .fillna(train["time_step"])
        .astype(np.float32)
    )
    test["dt"] = (
        test.groupby("breath_id", sort=False)["time_step"]
        .diff()
        .fillna(test["time_step"])
        .astype(np.float32)
    )

    train["u_in_int"] = (train["u_in"].astype(np.float32) * train["dt"]).astype(
        np.float32
    )
    test["u_in_int"] = (test["u_in"].astype(np.float32) * test["dt"]).astype(np.float32)

    train["u_in_int_cum"] = (
        train.groupby("breath_id", sort=False)["u_in_int"].cumsum().astype(np.float32)
    )
    test["u_in_int_cum"] = (
        test.groupby("breath_id", sort=False)["u_in_int"].cumsum().astype(np.float32)
    )

    uin_int_cum_bin = 1.0
    train["u_in_int_cum_bin"] = (
        np.round(train["u_in_int_cum"].to_numpy(dtype=np.float64) / uin_int_cum_bin)
        * uin_int_cum_bin
    ).astype(np.float32)
    test["u_in_int_cum_bin"] = (
        np.round(test["u_in_int_cum"].to_numpy(dtype=np.float64) / uin_int_cum_bin)
        * uin_int_cum_bin
    ).astype(np.float32)

    train_in = train[train["u_out"] == 0].copy()
    train_out = train[train["u_out"] == 1].copy()

    agg_in_1f = _group_median_pressure(
        train_in,
        keys=["R", "C", "step", "u_out", "u_in_bin_f", "u_in_int_cum_bin"],
        out_col="pred_in_1f",
    )
    agg_in_1 = _group_median_pressure(
        train_in,
        keys=["R", "C", "step", "u_out", "u_in_bin", "u_in_int_cum_bin"],
        out_col="pred_in_1",
    )

    agg_in_1c = _group_median_pressure(
        train_in,
        keys=["R", "C", "step", "u_out", "u_in_int_cum_bin"],
        out_col="pred_in_1c",
    )

    agg_in_2 = _group_median_pressure(
        train_in,
        keys=["R", "C", "step", "u_out", "u_in_bin"],
        out_col="pred_in_2",
    )
    agg_in_3 = _group_median_pressure(
        train_in,
        keys=["R", "C", "step", "u_out"],
        out_col="pred_in_3",
    )
    agg_in_4 = _group_median_pressure(
        train_in,
        keys=["R", "C", "time_idx", "u_out"],
        out_col="pred_in_4",
    )

    agg_out_1f = _group_median_pressure(
        train_out,
        keys=["R", "C", "step", "u_out", "u_in_bin_f", "u_in_int_cum_bin"],
        out_col="pred_out_1f",
    )
    agg_out_1 = _group_median_pressure(
        train_out,
        keys=["R", "C", "step", "u_out", "u_in_bin", "u_in_int_cum_bin"],
        out_col="pred_out_1",
    )

    agg_out_1c = _group_median_pressure(
        train_out,
        keys=["R", "C", "step", "u_out", "u_in_int_cum_bin"],
        out_col="pred_out_1c",
    )

    agg_out_2 = _group_median_pressure(
        train_out,
        keys=["R", "C", "step", "u_out", "u_in_bin"],
        out_col="pred_out_2",
    )
    agg_out_3 = _group_median_pressure(
        train_out,
        keys=["R", "C", "step", "u_out"],
        out_col="pred_out_3",
    )
    agg_out_4 = _group_median_pressure(
        train_out,
        keys=["R", "C", "time_idx", "u_out"],
        out_col="pred_out_4",
    )

    test2 = test.copy()

    test2 = test2.merge(
        agg_in_1f,
        on=["R", "C", "step", "u_out", "u_in_bin_f", "u_in_int_cum_bin"],
        how="left",
    )
    test2 = test2.merge(
        agg_in_1,
        on=["R", "C", "step", "u_out", "u_in_bin", "u_in_int_cum_bin"],
        how="left",
    )
    test2 = test2.merge(
        agg_in_1c,
        on=["R", "C", "step", "u_out", "u_in_int_cum_bin"],
        how="left",
    )
    test2 = test2.merge(
        agg_in_2, on=["R", "C", "step", "u_out", "u_in_bin"], how="left"
    )
    test2 = test2.merge(agg_in_3, on=["R", "C", "step", "u_out"], how="left")
    test2 = test2.merge(agg_in_4, on=["R", "C", "time_idx", "u_out"], how="left")

    test2 = test2.merge(
        agg_out_1f,
        on=["R", "C", "step", "u_out", "u_in_bin_f", "u_in_int_cum_bin"],
        how="left",
    )
    test2 = test2.merge(
        agg_out_1,
        on=["R", "C", "step", "u_out", "u_in_bin", "u_in_int_cum_bin"],
        how="left",
    )
    test2 = test2.merge(
        agg_out_1c,
        on=["R", "C", "step", "u_out", "u_in_int_cum_bin"],
        how="left",
    )
    test2 = test2.merge(
        agg_out_2, on=["R", "C", "step", "u_out", "u_in_bin"], how="left"
    )
    test2 = test2.merge(agg_out_3, on=["R", "C", "step", "u_out"], how="left")
    test2 = test2.merge(agg_out_4, on=["R", "C", "time_idx", "u_out"], how="left")

    global_fallback_in = (
        float(np.median(train_in["pressure"].to_numpy(dtype=np.float64)))
        if len(train_in)
        else float(np.median(train["pressure"].to_numpy(dtype=np.float64)))
    )
    global_fallback_out = (
        float(np.median(train_out["pressure"].to_numpy(dtype=np.float64)))
        if len(train_out)
        else float(np.median(train["pressure"].to_numpy(dtype=np.float64)))
    )

    u_out = test2["u_out"].to_numpy(dtype=np.int8)

    p_in_1f = test2["pred_in_1f"].to_numpy(dtype=np.float64)
    p_in_1 = test2["pred_in_1"].to_numpy(dtype=np.float64)
    p_in_1c = test2["pred_in_1c"].to_numpy(dtype=np.float64)
    p_in_2 = test2["pred_in_2"].to_numpy(dtype=np.float64)
    p_in_3 = test2["pred_in_3"].to_numpy(dtype=np.float64)
    p_in_4 = test2["pred_in_4"].to_numpy(dtype=np.float64)

    p_out_1f = test2["pred_out_1f"].to_numpy(dtype=np.float64)
    p_out_1 = test2["pred_out_1"].to_numpy(dtype=np.float64)
    p_out_1c = test2["pred_out_1c"].to_numpy(dtype=np.float64)
    p_out_2 = test2["pred_out_2"].to_numpy(dtype=np.float64)
    p_out_3 = test2["pred_out_3"].to_numpy(dtype=np.float64)
    p_out_4 = test2["pred_out_4"].to_numpy(dtype=np.float64)

    preds_in = np.where(
        ~np.isnan(p_in_1f),
        p_in_1f,
        np.where(
            ~np.isnan(p_in_1),
            p_in_1,
            np.where(
                ~np.isnan(p_in_1c),
                p_in_1c,
                np.where(
                    ~np.isnan(p_in_2),
                    p_in_2,
                    np.where(
                        ~np.isnan(p_in_3),
                        p_in_3,
                        np.where(~np.isnan(p_in_4), p_in_4, global_fallback_in),
                    ),
                ),
            ),
        ),
    )
    preds_out = np.where(
        ~np.isnan(p_out_1f),
        p_out_1f,
        np.where(
            ~np.isnan(p_out_1),
            p_out_1,
            np.where(
                ~np.isnan(p_out_1c),
                p_out_1c,
                np.where(
                    ~np.isnan(p_out_2),
                    p_out_2,
                    np.where(
                        ~np.isnan(p_out_3),
                        p_out_3,
                        np.where(~np.isnan(p_out_4), p_out_4, global_fallback_out),
                    ),
                ),
            ),
        ),
    )

    preds = np.where(u_out == 0, preds_in, preds_out)
    preds = _quantize_pressure(preds.astype(np.float64))

    sub = sub.merge(test[["id"]], on="id", how="right")
    sub["pressure"] = preds
    sub = sub[["id", "pressure"]].sort_values("id").reset_index(drop=True)
    return sub


ext1 = "../input/keraslstm151/submission_median_round_LB153.csv"
ext2 = "../input/lstmfold10146/submission_median_round_LB153.csv"

if os.path.exists(ext1) and os.path.exists(ext2):
    df1 = pd.read_csv(ext1)
    df2 = pd.read_csv(ext2)

    if not {"id", "pressure"}.issubset(df1.columns) or not {"id", "pressure"}.issubset(
        df2.columns
    ):
        submission = make_fallback_submission(
            config.paths["train"], config.paths["test"], config.paths["ss"]
        )
    else:
        test_ids_sorted = (
            pd.read_csv(config.paths["test"], usecols=["id"])["id"]
            .sort_values()
            .to_numpy()
        )
        df1 = df1.sort_values("id").reset_index(drop=True)
        df2 = df2.sort_values("id").reset_index(drop=True)

        if (
            len(df1) != len(test_ids_sorted)
            or len(df2) != len(test_ids_sorted)
            or not np.array_equal(df1["id"].to_numpy(), test_ids_sorted)
            or not np.array_equal(df2["id"].to_numpy(), test_ids_sorted)
        ):
            submission = make_fallback_submission(
                config.paths["train"], config.paths["test"], config.paths["ss"]
            )
        else:
            df1["pressure"] = (
                df1["pressure"].astype(np.float64) * 0.15
                + df2["pressure"].astype(np.float64) * 0.85
            )
            submission = df1[["id", "pressure"]].copy()
            submission["pressure"] = _quantize_pressure(
                submission["pressure"].to_numpy()
            )
else:
    submission = make_fallback_submission(
        config.paths["train"], config.paths["test"], config.paths["ss"]
    )

submission.to_csv("submission.csv", index=False)
submission.to_csv("sub.csv", index=False)


## === cell 3
oof_files = []
for p in paths:
    f = os.path.join(p, "oof.csv")
    if os.path.exists(f):
        oof_files.append(f)

if len(oof_files) > 0:
    df = pd.concat([pd.read_csv(f) for f in oof_files], ignore_index=True)
    if "pred" in df.columns and "pressure" in df.columns:
        df = df[df["pred"] != 0]
        oof_mae = float(
            np.mean(np.abs(df["pred"].to_numpy() - df["pressure"].to_numpy()))
        )
    else:
        oof_mae = None
else:
    df = None
    oof_mae = None

oof_mae


## === cell 4
sub = pd.read_csv(config.paths["ss"])
test_ids = pd.read_csv(config.paths["test"], usecols=["id"])["id"].values

sub = sub.merge(pd.DataFrame({"id": test_ids}), on="id", how="right")

sub_paths = []
for p in paths:
    f = os.path.join(p, "submission.csv")
    if os.path.exists(f):
        sub_paths.append(f)

if len(sub_paths) > 0:
    preds_stack = []
    ok = True
    for f in sub_paths:
        tmp = (
            pd.read_csv(f, usecols=["id", "pressure"])
            .sort_values("id")
            .reset_index(drop=True)
        )
        if len(tmp) != len(test_ids) or not np.array_equal(
            tmp["id"].to_numpy(), np.sort(test_ids)
        ):
            ok = False
            break
        preds_stack.append(tmp["pressure"].to_numpy(dtype=np.float64))

    if ok and len(preds_stack) > 0:
        mean_preds = np.mean(np.vstack(preds_stack), axis=0)
        sub = sub.sort_values("id").reset_index(drop=True)
        sub["pressure"] = _quantize_pressure(mean_preds)
    else:
        sub = pd.read_csv("submission.csv")
else:
    sub = pd.read_csv("submission.csv")


## === cell 5
sub.to_csv("submission.csv", index=False)
