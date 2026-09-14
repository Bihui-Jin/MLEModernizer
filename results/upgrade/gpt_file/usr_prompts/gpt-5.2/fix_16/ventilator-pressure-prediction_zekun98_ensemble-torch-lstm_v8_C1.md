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

0.1449394944940853

# 6. Current score

4.18428

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.73281) has done: 'I remove the hard dependency on missing external Kaggle datasets (the `keraslstm151`, `lstmfold10146`, `youdontknow`, and `../input/torch*` folders) which currently crash the notebook, and replace it with a robust fallback that trains a simple per-(R,C,time_step,u_in,u_out) median lookup model directly from `train.csv`. This preserves the original “median/rounding” post-processing semantics and ensures a valid `submission.csv` is always written. I also make the OOF/submission ensembling cells conditional: if those external artifacts exist they be used; otherwise the fallback generates predictions end-to-end. Finally, I fix the bug where `df1.to_csv` was used instead of the blended/processed `submission.to_csv`, ensuring the correct file is saved.'
- What this solution (achieved 9.06816) has done: 'Your current fallback is a very weak “exact key median lookup” on continuous `u_in`/`time_step`, so it misses most rows and collapses to a global median, causing the huge MAE. I keep the same core lookup+median+rounding semantics, but change the key to a breath-sequence-friendly form by adding lag features (`u_in_lag1`, `u_out_lag1`) and cumulative volume (`u_in_cumsum`) computed per `breath_id`, then take a median over that richer key. This remains a pure train→median→merge model (no new learning loop/architecture), but drastically reduces the fallback error toward your target. I also ensure exact alignment to `sample_submission` ids and keep your post-processing unchanged.'
- What this solution (achieved 3.4003) has done: 'I keep your current “train→median lookup→merge→round/clip” core logic intact, but make the lookup key match train/test more often so the fallback doesn’t default to the global median (which is driving the huge MAE). Specifically, I (1) make the sequence-derived features consistent and more merge-friendly by using fixed quantization and stable per-breath ordering, and (2) add a tiny hierarchical backoff: try the full rich key first, then progressively drop the noisiest components (cumsum/lag) before finally using the global median. This remains the same modeling approach (median tables + merge) and same post-processing, but should substantially reduce error toward your 0.145 target without changing architecture/training loops. I also ensure the merged predictions stay aligned to `sample_submission` ids exactly.'
- What this solution (achieved 3.1561) has done: 'Your current score (3.4003 MAE; lower is better) is far from the target (0.1449), so we need a real accuracy lift while keeping your “median-lookup with rounding/clip” core approach intact. The biggest issue is the keying/backoff: using quantized `time_step/u_in` plus lag/cumsum still leaves many misses and doesn’t explicitly use the competition’s “only inspiratory phase is scored” fact, so the lookup is trying to fit both phases. I keep the same pipeline (train→groupby median tables→merge with hierarchical backoff→round/clip), but (1) train all median tables only on inspiratory rows (`u_out==0`) and (2) add a final per-(R,C,time_step_q,u_out) backoff so expiratory rows don’t collapse to a global median. This is a minimal, metric-aligned change that should reduce MAE substantially toward your target without changing the overall modeling semantics.'
- What this solution (achieved 3.1561) has done: 'I fix the pandas boolean-indexing bug inside `predict_with_backoff` that causes the “Unalignable boolean Series” crash by using numpy boolean masks (positional) instead of an index-aligned Series, which preserves your exact median-merge + hierarchical backoff logic. Then I ensure `submission` is always created even if earlier cells skip external artifacts, so cell 8 cannot see `None`. Finally, I add a small safety check to align predictions to `sample_submission` ids deterministically (no score change, just correctness) and keep your rounding/clip post-processing unchanged.'
- What this solution (achieved 3.38514) has done: 'Your current fallback is still far from the target MAE, so we need a substantial accuracy lift while preserving the same “median lookup tables + hierarchical backoff + rounding/clip” core logic. The smallest high-impact fix is to replace the overly-specific lag/cumsum keys (which still miss often) with a breath-shape key that matches train/test much more reliably: per-breath cumulative sums of `u_in` and `u_out` plus their lags, all quantized, then do the same median-table backoff. We still train tables on inspiratory rows for metric alignment, but we also use those breath-shape features in the tables/backoff so inspiratory predictions stop collapsing to generic medians. Finally, we keep your exact post-processing (round to known pressure grid and clip) and ensure strict id alignment to `sample_submission`.'
- What this solution (achieved 7.31275) has done: 'Your current MAE (3.385) is far above the target (0.145; lower is better), so we need a real accuracy lift while keeping the same “median lookup tables + hierarchical backoff + rounding/clip” approach. The biggest win with minimal semantic change is to make the lookup keys much more matchable: remove aggressive quantization (which creates mismatched bins) and instead use exact `time_step` (it’s on a fixed grid) plus a lightly-rounded `u_in`, while keeping your per-breath lag/cumsum “shape” features and the same backoff ladder. Also, train inspiratory tables only on `u_out==0` (as you already do), but make the phase backoff table *separate for insp/exp* so expiratory rows don’t borrow inspiratory medians incorrectly. Finally, keep your exact post-processing (round to pressure grid + clip) and strict `sample_submission` id alignment.'
- What this solution (achieved 2.38843) has done: 'We need to move your MAE down from 7.31275 toward 0.1449 (lower is better), so the fallback median-lookup must stop missing and collapsing to generic medians. I keep your exact “train → median tables → hierarchical backoff → round/clip” core logic, but make the keys much more matchable by (1) using the known fixed `time_step` grid as an integer step index, and (2) using `u_in`/lag/cumsum quantization that’s consistent with the 0.03s sampling (coarser than 0.1 to reduce key sparsity). I also add one minimal extra backoff table keyed by `(R,C,step,u_out,u_in)` (dropping lag/cum but keeping phase) to catch many remaining misses without changing semantics. These changes should substantially reduce lookup sparsity (and thus MAE) while preserving your approach and post-processing.'
- What this solution (achieved 2.77556) has done: 'Your current MAE (2.388) is still far above the target (0.145; lower is better), so we should reduce “misses” in the median-lookup so predictions don’t frequently fall back to coarse/global medians. Keeping the same core logic (sequence features → median tables → hierarchical backoff → round/clip), the smallest high-impact change is to stop over-rounding `u_in`/lags/cums to integers and instead quantize them a bit finer (0.5) while keeping `time_step` as an integer step index, which typically increases train/test key matches without changing the modeling approach. I also add one additional backoff table keyed by `(R,C,step,u_out,u_in_key)` (placed between the existing backoffs) to catch cases where lag/cum are noisy but `u_in` at that step is still informative. These changes should move the score materially downward toward your target while preserving evaluation semantics and still writing a valid `submission.csv`.'
- What this solution (achieved 4.57218) has done: 'Your current MAE (2.77556; lower is better) is still far above the target (0.14494), so we need a meaningful accuracy lift while preserving your median-lookup + hierarchical backoff + rounding/clip semantics. The smallest high-impact change is to make the lookup keys less sparse by (1) using the exact fixed `u_in` grid (rounded to 2 decimals, which matches the dataset’s typical precision) rather than 0.5-quantization, and (2) slightly coarsening only the *cumulative* features (which explode key cardinality) so more test rows hit the higher-quality tables before backing off. We keep training the main tables only on inspiratory rows (`u_out==0`) to match the metric, and we keep the same backoff ladder and post-processing unchanged. This should reduce “miss → coarse fallback” frequency and move MAE downward toward your target without changing the overall approach.'
- What this solution (achieved 4.52082) has done: 'Your current MAE (4.57218, lower is better) is far above the target (0.14494), so we should make a meaningful but still “same approach” improvement: reduce lookup-table sparsity so fewer rows fall through to coarse/global medians. I keep your exact pipeline (sequence features → median tables → hierarchical backoff → rounding/clip) but adjust only the keying: (1) use `R,C,u_out,step`-aware tables everywhere (including `basic`) and (2) quantize cumulative features by *step-dependent expected magnitude* (coarser later in the breath) to increase train/test key matches without changing the modeling semantics. I also add one minimal intermediate backoff table keyed by `(R,C,step,u_out,u_in_key)` (phase-aware) before dropping `u_out`, which is consistent with your existing ladder and should reduce misses. Submission writing and id alignment remain unchanged.'
- What this solution (achieved 4.71982) has done: 'Your current MAE (4.52082; lower is better) is far above the target (0.14494), so we need a real accuracy lift while keeping the exact same “median lookup tables + hierarchical backoff + rounding/clip” core approach. The biggest issue is that the current lookup keys remain too sparse (especially due to cumsums), so many rows still fall through to coarse backoffs/global medians. I make the per-breath sequence keys more matchable by switching the cumulative features to a *step-wise delta window sum* (rolling sum over recent steps) and quantizing it mildly; this keeps the same type of breath-shape information but drastically reduces key cardinality and improves train/test matches. I also insert one additional backoff level that drops the delta-sum but keeps `(R,C,step,u_out,u_in,u_in_lag1)` so we catch many misses before falling back further, without changing the overall modeling semantics.'
- What this solution (achieved 4.71982) has done: 'I keep your exact “median lookup tables + hierarchical backoff + round/clip” approach, but make one minimal, high-impact correction: for inspiratory rows (the only ones scored), we should not be training/looking up with `u_out_cum_key` because `u_out` is always 0 there, making that feature constant and adding needless sparsity that increases fall-through to coarse/global medians (hurting MAE). I drop `u_out_cum_key` from the inspiratory lookup keys/tables only (expiratory path kept as-is), which preserves your modeling semantics but increases hit-rate of the highest-quality tables. I also add a small sanity print of hit-rates per backoff level (no functional change) to confirm fewer misses. The script still run end-to-end and write `submission.csv` with the required columns.'
- What this solution (achieved 4.18428) has done: 'Your current MAE (4.71982; lower is better) is still far from the target (0.14494), so we need a meaningful accuracy lift while preserving your exact “median lookup tables + hierarchical backoff + rounding/clip” approach. The smallest high-impact change is to reduce key sparsity by removing the rolling-window sum feature (`u_in_win5_key`) from the *primary* lookup levels (it explodes cardinality and causes many misses), while keeping the rest of your sequence features and the same backoff ladder. To compensate and still use breath-shape information, we add one low-cardinality “cumulative u_in” feature quantized coarsely and use it only in a mid-level backoff table (so it helps when available without causing mass misses). All I/O paths, submission schema, and your rounding/clip post-processing remain unchanged, and the script still writes `submission.csv` end-to-end.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

paths = glob.glob("../input/torch*")
print(f"Found {len(paths)} torch* paths")


## === cell 1
src_copy = "../input/youdontknow/submission_median_round_LB153.csv"
if os.path.exists(src_copy):
    os.system(f"cp -r {src_copy} ./")
    print("Copied external submission file.")
else:
    print("External submission file not found; skipping copy.")




## === cell 2
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




## === cell 3
p1 = "../input/keraslstm151/submission_median_round_LB153.csv"
p2 = "../input/lstmfold10146/submission_median_round_LB153.csv"

submission = None

if os.path.exists(p1) and os.path.exists(p2):
    df1 = pd.read_csv(p1)
    df2 = pd.read_csv(p2)

    if not {"id", "pressure"}.issubset(df1.columns) or not {"id", "pressure"}.issubset(
        df2.columns
    ):
        raise ValueError("External submissions must contain columns: id, pressure")

    df1 = df1.sort_values("id").reset_index(drop=True)
    df2 = df2.sort_values("id").reset_index(drop=True)
    if not np.array_equal(df1["id"].values, df2["id"].values):
        df2 = df2.set_index("id").reindex(df1["id"].values).reset_index()

    df1["pressure"] = df1["pressure"] * 0.15 + df2["pressure"] * 0.85
    submission = df1[["id", "pressure"]].copy()

    submission["pressure"] = (
        np.round(
            (submission.pressure - config.post_processing["min_pressure"])
            / config.post_processing["diff_pressure"]
        )
        * config.post_processing["diff_pressure"]
        + config.post_processing["min_pressure"]
    )
    submission["pressure"] = np.clip(
        submission["pressure"],
        config.post_processing["min_pressure"],
        config.post_processing["max_pressure"],
    )

    submission.to_csv("sub.csv", index=False)
    print("Wrote blended external submission to sub.csv")
else:
    print(
        "External blend files not found; will generate predictions from train.csv/test.csv in later cells."
    )


## === cell 4
df = None
if len(paths) > 0:
    oof_files = [p + "/oof.csv" for p in paths if os.path.exists(p + "/oof.csv")]
    if len(oof_files) > 0:
        df = pd.concat([pd.read_csv(f) for f in oof_files], ignore_index=True)
        if "pred" in df.columns:
            df = df[df.pred != 0]
        print(f"Loaded OOF from {len(oof_files)} files: {df.shape}")
    else:
        print("No oof.csv found under ../input/torch* paths; skipping OOF concat.")
else:
    print("No ../input/torch* paths; skipping OOF concat.")


## === cell 5
if df is not None and {"pred", "pressure"}.issubset(df.columns):
    mae = np.mean(np.abs(df["pred"].values - df["pressure"].values))
    print("OOF MAE:", mae)
else:
    print("OOF MAE not computed (missing df or required columns).")


## === cell 6
if submission is None:
    train_path = config.paths["train"]
    test_path = config.paths["test"]
    ss_path = config.paths["ss"]

    usecols_train = [
        "id",
        "breath_id",
        "R",
        "C",
        "time_step",
        "u_in",
        "u_out",
        "pressure",
    ]
    usecols_test = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]

    train = pd.read_csv(train_path, usecols=usecols_train)
    test = pd.read_csv(test_path, usecols=usecols_test)
    ss = pd.read_csv(ss_path, usecols=["id"])

    def add_seq_features(d: pd.DataFrame) -> pd.DataFrame:
        d = d.sort_values(["breath_id", "time_step", "id"]).reset_index(drop=True)

        d["step"] = np.rint(d["time_step"].to_numpy(dtype=np.float32) / 0.03).astype(
            np.int16
        )

        uin = d["u_in"].to_numpy(dtype=np.float32)
        d["u_in_key"] = np.round(uin, 2).astype(np.float32)

        d["u_in_lag1"] = (
            d.groupby("breath_id")["u_in"].shift(1).fillna(0.0).astype(np.float32)
        )
        d["u_out_lag1"] = (
            d.groupby("breath_id")["u_out"].shift(1).fillna(0).astype(np.int8)
        )
        d["u_in_lag1_key"] = np.round(d["u_in_lag1"], 2).astype(np.float32)

        u_in = d["u_in"].astype(np.float32)
        u_out = d["u_out"].astype(np.int8)

        d["u_in_win5"] = (
            u_in.groupby(d["breath_id"])
            .rolling(window=5, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
            .astype(np.float32)
        )
        d["u_in_win5_key"] = (np.round(d["u_in_win5"] / 0.5) * 0.5).astype(np.float32)

        d["u_in_cum"] = u_in.groupby(d["breath_id"]).cumsum().astype(np.float32)
        d["u_in_cum_key"] = (np.round(d["u_in_cum"] / 5.0) * 5.0).astype(np.float32)

        d["u_out_cum"] = (
            u_out.astype(np.int16).groupby(d["breath_id"]).cumsum().astype(np.int16)
        )
        d["u_out_cum_key"] = d["u_out_cum"].astype(np.int16)
        return d

    train_f = add_seq_features(train)
    test_f = add_seq_features(test)

    train_insp = train_f[train_f["u_out"] == 0].copy()
    train_exp = train_f[train_f["u_out"] == 1].copy()

    global_median_insp = float(train_insp["pressure"].median())
    global_median_exp = (
        float(train_exp["pressure"].median())
        if len(train_exp)
        else float(train_f["pressure"].median())
    )

    key_full_insp = [
        "R",
        "C",
        "step",
        "u_in_key",
        "u_out",
        "u_in_lag1_key",
        "u_out_lag1",
    ]
    key_drop_win_insp = [
        "R",
        "C",
        "step",
        "u_in_key",
        "u_out",
        "u_in_lag1_key",
        "u_out_lag1",
    ]

    key_full_exp = [
        "R",
        "C",
        "step",
        "u_in_key",
        "u_out",
        "u_in_lag1_key",
        "u_out_lag1",
        "u_out_cum_key",
    ]
    key_drop_win_exp = [
        "R",
        "C",
        "step",
        "u_in_key",
        "u_out",
        "u_in_lag1_key",
        "u_out_lag1",
        "u_out_cum_key",
    ]

    key_drop_lags = ["R", "C", "step", "u_in_key", "u_out"]
    key_uin_lag = ["R", "C", "step", "u_out", "u_in_key", "u_in_lag1_key"]
    key_phase_uin = ["R", "C", "step", "u_out", "u_in_key"]

    key_basic_phase = ["R", "C", "step", "u_out", "u_in_key"]
    key_basic_nophase = ["R", "C", "step", "u_in_key"]
    key_phase = ["R", "C", "step", "u_out"]

    key_cum_uin = ["R", "C", "step", "u_out", "u_in_cum_key"]

    def build_tables(d: pd.DataFrame, key_full, key_drop_win):
        med_full = d.groupby(key_full, sort=False)["pressure"].median().reset_index()
        med_dw = d.groupby(key_drop_win, sort=False)["pressure"].median().reset_index()
        med_ul = d.groupby(key_uin_lag, sort=False)["pressure"].median().reset_index()
        med_dl = d.groupby(key_drop_lags, sort=False)["pressure"].median().reset_index()
        med_cum = d.groupby(key_cum_uin, sort=False)["pressure"].median().reset_index()
        med_puin = (
            d.groupby(key_phase_uin, sort=False)["pressure"].median().reset_index()
        )
        med_basic_phase = (
            d.groupby(key_basic_phase, sort=False)["pressure"].median().reset_index()
        )
        med_basic_nophase = (
            d.groupby(key_basic_nophase, sort=False)["pressure"].median().reset_index()
        )
        return (
            med_full,
            med_dw,
            med_ul,
            med_dl,
            med_cum,
            med_puin,
            med_basic_phase,
            med_basic_nophase,
        )

    insp_tables = build_tables(train_insp, key_full_insp, key_drop_win_insp)
    exp_tables = (
        build_tables(train_exp, key_full_exp, key_drop_win_exp)
        if len(train_exp)
        else insp_tables
    )

    med_phase_insp = (
        train_insp.groupby(key_phase, sort=False)["pressure"].median().reset_index()
    )
    med_phase_exp = (
        train_exp.groupby(key_phase, sort=False)["pressure"].median().reset_index()
        if len(train_exp)
        else med_phase_insp
    )

    is_exp_mask = test_f["u_out"].to_numpy(dtype=np.int8) == 1
    pred_series = pd.Series(np.nan, index=test_f.index, dtype="float64")

    def predict_with_backoff(
        out_part: pd.DataFrame,
        tables,
        phase_fill_df: pd.DataFrame,
        global_fill: float,
        key_full,
        key_drop_win,
        debug_name: str,
    ) -> pd.Series:
        (
            med_full,
            med_dw,
            med_ul,
            med_dl,
            med_cum,
            med_puin,
            med_basic_phase,
            med_basic_nophase,
        ) = tables

        n = len(out_part)
        hit = {}

        p = out_part.merge(med_full, on=key_full, how="left")["pressure"].astype(
            "float64"
        )
        hit["full"] = int(p.notna().sum())

        m = p.isna().to_numpy()
        if m.any():
            idx = np.where(m)[0]
            p2 = (
                out_part.iloc[idx][key_drop_win]
                .merge(med_dw, on=key_drop_win, how="left")["pressure"]
                .astype("float64")
                .to_numpy()
            )
            p.iloc[idx] = p2
        hit["drop_win"] = int(p.notna().sum())

        m = p.isna().to_numpy()
        if m.any():
            idx = np.where(m)[0]
            p2b = (
                out_part.iloc[idx][key_uin_lag]
                .merge(med_ul, on=key_uin_lag, how="left")["pressure"]
                .astype("float64")
                .to_numpy()
            )
            p.iloc[idx] = p2b
        hit["uin_lag"] = int(p.notna().sum())

        m = p.isna().to_numpy()
        if m.any():
            idx = np.where(m)[0]
            p3 = (
                out_part.iloc[idx][key_drop_lags]
                .merge(med_dl, on=key_drop_lags, how="left")["pressure"]
                .astype("float64")
                .to_numpy()
            )
            p.iloc[idx] = p3
        hit["drop_lags"] = int(p.notna().sum())

        m = p.isna().to_numpy()
        if m.any():
            idx = np.where(m)[0]
            p3c = (
                out_part.iloc[idx][key_cum_uin]
                .merge(med_cum, on=key_cum_uin, how="left")["pressure"]
                .astype("float64")
                .to_numpy()
            )
            p.iloc[idx] = p3c
        hit["cum_uin"] = int(p.notna().sum())

        m = p.isna().to_numpy()
        if m.any():
            idx = np.where(m)[0]
            p3b = (
                out_part.iloc[idx][key_phase_uin]
                .merge(med_puin, on=key_phase_uin, how="left")["pressure"]
                .astype("float64")
                .to_numpy()
            )
            p.iloc[idx] = p3b
        hit["phase_uin"] = int(p.notna().sum())

        m = p.isna().to_numpy()
        if m.any():
            idx = np.where(m)[0]
            p4a = (
                out_part.iloc[idx][key_basic_phase]
                .merge(med_basic_phase, on=key_basic_phase, how="left")["pressure"]
                .astype("float64")
                .to_numpy()
            )
            p.iloc[idx] = p4a
        hit["basic_phase"] = int(p.notna().sum())

        m = p.isna().to_numpy()
        if m.any():
            idx = np.where(m)[0]
            p4b = (
                out_part.iloc[idx][key_basic_nophase]
                .merge(med_basic_nophase, on=key_basic_nophase, how="left")["pressure"]
                .astype("float64")
                .to_numpy()
            )
            p.iloc[idx] = p4b
        hit["basic_nophase"] = int(p.notna().sum())

        m = p.isna().to_numpy()
        if m.any():
            idx = np.where(m)[0]
            p5 = (
                out_part.iloc[idx][key_phase]
                .merge(phase_fill_df, on=key_phase, how="left")["pressure"]
                .astype("float64")
                .to_numpy()
            )
            p.iloc[idx] = p5
        hit["phase_only"] = int(p.notna().sum())

        if p.isna().any():
            p = p.fillna(global_fill)
        hit["global_fill"] = int(p.notna().sum())

        print(
            f"[{debug_name}] n={n} hits:"
            + ", ".join([f"{k}={v}" for k, v in hit.items()])
        )

        return p.astype(np.float64)

    if (~is_exp_mask).any():
        idx_insp = np.where(~is_exp_mask)[0]
        out_insp = test_f.iloc[idx_insp][
            ["id"] + key_full_insp + ["u_in_cum_key"]
        ].copy()
        pred_series.iloc[idx_insp] = predict_with_backoff(
            out_insp,
            insp_tables,
            med_phase_insp,
            global_median_insp,
            key_full_insp,
            key_drop_win_insp,
            debug_name="insp",
        ).to_numpy()

    if is_exp_mask.any():
        idx_exp = np.where(is_exp_mask)[0]
        out_exp = test_f.iloc[idx_exp][["id"] + key_full_exp + ["u_in_cum_key"]].copy()
        pred_series.iloc[idx_exp] = predict_with_backoff(
            out_exp,
            exp_tables,
            med_phase_exp,
            global_median_exp,
            key_full_exp,
            key_drop_win_exp,
            debug_name="exp",
        ).to_numpy()

    pred = pred_series.to_numpy(dtype=np.float64)

    pred = (
        np.round(
            (pred - config.post_processing["min_pressure"])
            / config.post_processing["diff_pressure"]
        )
        * config.post_processing["diff_pressure"]
        + config.post_processing["min_pressure"]
    )
    pred = np.clip(
        pred,
        config.post_processing["min_pressure"],
        config.post_processing["max_pressure"],
    )

    ss_sorted = ss.sort_values("id").reset_index(drop=True)
    id_to_pred = pd.DataFrame({"id": test_f["id"].values, "pressure": pred}).set_index(
        "id"
    )
    ss_sorted["pressure"] = id_to_pred.reindex(ss_sorted["id"].values)[
        "pressure"
    ].to_numpy()

    submission = ss_sorted[["id", "pressure"]].copy()
    print(
        "Generated fallback submission via inspiratory-trained median lookup + hierarchical backoff + rounding/clip (reduced sparsity keys + cum-u_in backoff)."
    )


## === cell 7
if len(paths) > 0 and submission is not None:
    sub_files = [
        p + "/submission.csv" for p in paths if os.path.exists(p + "/submission.csv")
    ]
    if len(sub_files) > 0:
        base = submission[["id"]].copy().sort_values("id").reset_index(drop=True)
        preds = []
        for f in sub_files:
            tmp = pd.read_csv(f)
            tmp = tmp[["id", "pressure"]].sort_values("id").reset_index(drop=True)
            if not np.array_equal(tmp["id"].values, base["id"].values):
                tmp = tmp.set_index("id").reindex(base["id"].values).reset_index()
            preds.append(tmp["pressure"].values.astype(np.float64))
        base["pressure"] = np.mean(np.vstack(preds), axis=0)

        base["pressure"] = (
            np.round(
                (base.pressure - config.post_processing["min_pressure"])
                / config.post_processing["diff_pressure"]
            )
            * config.post_processing["diff_pressure"]
            + config.post_processing["min_pressure"]
        )
        base["pressure"] = np.clip(
            base["pressure"],
            config.post_processing["min_pressure"],
            config.post_processing["max_pressure"],
        )
        submission = base
        print(f"Ensembled {len(sub_files)} torch* submissions.")
    else:
        print(
            "No submission.csv found under ../input/torch* paths; using current submission."
        )
else:
    print("No ../input/torch* paths or no submission yet; using current submission.")


## === cell 8
if submission is None:
    raise RuntimeError("submission is None; prediction generation failed upstream.")

submission = submission[["id", "pressure"]].copy()
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv:", submission.shape)
print(submission.head())
