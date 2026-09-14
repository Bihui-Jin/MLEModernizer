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

0.1395558108671873

# 6. Current score

9.90855

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.02313) has done: 'I fix the runtime failure by removing the hard dependency on external Kaggle Dataset paths that don’t exist in your environment, while keeping the “ensemble submissions” core idea intact. The code now (a) try to load those external submissions if present, but (b) safely fall back to a simple, valid baseline prediction derived only from the provided competition data when they are missing. This ensures the notebook always runs end-to-end and always writes a correctly formatted `submission.csv`. The fallback uses a per-(R,C,time_step) median pressure lookup from the training data, which is a standard non-leaky baseline for this competition and should give a non-trivial score compared to all-zeros.'
- What this solution (achieved 4.00111) has done: 'Your current fallback baseline is too weak for this competition because it ignores the two control inputs (`u_in`, `u_out`) and the sequential nature of the breath, which likely explains the very large MAE (6.02 vs target ~0.14). To move the score much closer to the target while keeping changes minimal and not introducing a new model, I keep the same “lookup-from-train then merge onto test” core idea but enrich the lookup key to include `u_out` and a rounded `u_in` (plus `R`,`C`,`time_step`). I also add a robust two-stage backoff (exact key → drop `u_in` → global median) so every row gets a prediction without errors. This preserves the overall approach (non-leaky median mapping) while making predictions much more conditionally accurate, which should substantially reduce MAE toward your target.'
- What this solution (achieved 3.85706) has done: 'Your current fallback “median lookup” is still far from the target because it doesn’t respect the key scoring rule: only inspiratory rows (`u_out==0`) matter, and expiratory rows are ignored, so you can safely “copy through” the last inspiratory pressure into expiratory timesteps to reduce error without changing the core lookup logic. I keep your same groupby-median mapping, but (1) build predictions only for inspiratory rows and (2) post-process per breath to forward-fill predictions over expiratory rows, which is a standard, minimal, non-leaky adjustment aligned with the metric. I also add a tiny additional backoff using a coarse binned `u_in` (integer) before dropping `u_in` entirely, to reduce NaNs/mismatches while preserving the same lookup approach. These changes are small, deterministic, and should move MAE substantially down toward your target.'
- What this solution (achieved 9.91154) has done: 'Your current median-lookup + inspiratory-only + forward-fill approach is still too coarse because rounding `u_in`/`time_step` causes many key mismatches and “wrong median” collisions. To move the MAE down toward your target while keeping the same core logic (train-derived groupby median → merge onto test → backoff → per-breath forward-fill), I (1) stop rounding `time_step` (it already matches exactly at 0.03 increments) and (2) use a finer but still stable `u_in` bin (0.1) plus a coarser fallback (0.5) before falling back to integer and then dropping `u_in`. These are minimal changes that increase match rate and reduce conditional error without changing the modeling paradigm. The submission writing and alignment by `id` remain unchanged.'
- What this solution (achieved 9.90855) has done: 'Your current score is far worse than the target (lower is better), so we should improve the fallback without changing the overall “train-derived median lookup → merge onto test → backoff → breath-wise postprocess → submission.csv” logic. The biggest correctness issue is that the current code matches `u_in` and `time_step` at the same instant, but pressure depends strongly on *history within the breath*; we can keep the same lookup idea while adding a minimal, cheap history feature (`u_in_cum` within breath) and using it only as an additional key before falling back. We compute `u_in_cum` for train/test, add a first-stage median map on `(R,C,ts,u_out,u_in_01,u_in_cum_bin)` (and a coarser backoff bin), then keep your existing fallback chain and the same forward-fill over expiratory steps. This stays deterministic, avoids new models, and should materially reduce MAE toward your target while keeping runtime within limits.'
- What this solution (achieved 9.90855) has done: 'Your current score is far worse than the target (lower is better), so we should improve the same “train-derived median lookup → merge onto test → backoff → breath-wise forward-fill → submission.csv” pipeline without introducing a new model. The biggest bug hurting accuracy is that `id` is **not** globally ordered by time across the whole file, so sorting by `["breath_id","id"]` breaks the within-breath temporal order and makes the forward-fill step propagate wrong values. I change the post-process ordering to use `time_step` (true within-breath time) and keep everything else the same, plus ensure we return to the original test row order by merging back on `id`. This is a minimal, metric-aligned fix that should materially reduce MAE while preserving your core logic.'
- What this solution (achieved 9.90855) has done: 'Your current score is far worse than the target (lower is better), so we should improve accuracy while keeping the same “train-derived groupby-median lookup → merge onto test → backoff → breath-wise forward-fill → submission.csv” core logic. The main accuracy bug is that `u_in_cum` is computed in the file’s row order (which is not guaranteed to be within-breath time order), so the history feature is effectively wrong; we compute it after sorting by `breath_id,time_step` and then merge it back by row. We also make the forward-fill strictly metric-aligned by only filling expiratory rows from the last inspiratory prediction (instead of forward-filling through all-NaN breaths indiscriminately), while still ensuring every row has a value via a final global-median fill. These are minimal, deterministic fixes that should materially reduce MAE toward your target without changing the overall approach.'
- What this solution (achieved 9.90855) has done: 'Your current score is much worse than the target (lower is better), so we should make a small but high-impact correction rather than adding new modeling. The biggest correctness bug is that `test_key["id"]` is not a unique row identifier (it resets each breath), yet the code sorts and aligns predictions back to `sub` using that non-unique `id`, which scrambles predictions and explodes MAE. I keep your exact median-lookup/backoff and breath-wise forward-fill logic, but carry the true row identifier from `test.csv` (the global timestep `id` column is actually unique in the original competition; in your environment it’s not, so we must use row order instead) by adding a `row_id` and using it for stable restore-to-original-order. This is minimal, deterministic, and should dramatically reduce MAE toward your target without changing the core approach.'
- What this solution (achieved 9.90855) has done: 'Your current MAE is far above the target (lower is better), and the largest likely cause is misalignment: in this environment `test.csv`’s `id` is not globally unique, but the code still writes predictions into `sub` by row position, which can scramble pressures vs ids and explode MAE. I keep your exact median-lookup/backoff and breath-wise forward-fill logic, but rebuild the submission by merging predictions back onto `sample_submission.csv` using a stable within-file row index instead of trusting `id`. I also ensure the fallback path reads `sample_submission.csv` and `test.csv` from the same base path so their row order matches before we attach predictions. These are minimal, deterministic fixes intended to materially reduce error without changing your core approach.'
- What this solution (achieved 9.90855) has done: 'We keep your exact “train-derived median lookup → multi-stage backoff → inspiratory-only then per-breath forward-fill → submission.csv” pipeline, but fix the likely catastrophic mismatch between `sample_submission.csv` and `test.csv` row alignment in this environment where `id` is not globally unique. Concretely, we build the submission directly from `test.csv`’s own `id` column and attach predictions by a stable `row_id` (row order), instead of merging onto `sample_submission.csv` which may not correspond 1:1 by row. This is a minimal change that preserves evaluation semantics while preventing scrambled `(id, pressure)` pairs, which is the most plausible reason the MAE is ~9.9. We also add a small safety assert to ensure lengths match and we always write a valid `submission.csv`.'
- What this solution (achieved 9.90855) has done: 'Your current MAE is catastrophically high relative to the target, so the most likely issue is not the median-lookup logic itself but submission misalignment: in your environment `test.csv`’s `id` is not unique and does not match the competition’s intended global timestep id, so writing `{"id": test["id"], "pressure": preds}` scrambles predictions against the required IDs. I keep your exact lookup/backoff and per-breath forward-fill logic, but rebuild the submission using `sample_submission.csv`’s `id` order and attach predictions strictly by stable row order (a `row_id` index) so `(id,pressure)` pairs line up. I also add a couple of hard assertions to fail fast if row counts diverge, preventing silent bad submissions. This is a minimal change aimed at drastically reducing MAE toward your target without changing the modeling approach.'
- What this solution (achieved 9.90855) has done: 'The current MAE is catastrophically worse than the target, which strongly suggests a submission alignment bug rather than a “model quality” issue. In your environment, `test.csv`’s `id` clearly isn’t globally unique (it ranges 1–2000), so using `sample_submission.csv`’s `id` column (which is also 1–2000 here) can scramble mapping and yield huge error; the safest minimal fix is to build the submission from `test.csv`’s own `id` and attach predictions by stable row order. I keep your exact median-lookup + multi-stage backoff + inspiratory-only + per-breath forward-fill logic unchanged, and only change how the final `(id, pressure)` pairs are constructed (plus add strict asserts to guarantee 1:1 row alignment). This should move the score sharply down toward the target without changing the core approach.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
BASE_PATH = "../input/ventilator-pressure-prediction"
train_path = os.path.join(BASE_PATH, "train.csv")
test_path = os.path.join(BASE_PATH, "test.csv")
sample_sub_path = os.path.join(BASE_PATH, "sample_submission.csv")

sub_template = pd.read_csv(sample_sub_path)

external_paths = [
    ("sub_0", "../input/gb-vpp-pulp-fiction/median_submission.csv", 0.13),
    ("sub_1", "../input/new-ensemble-of-public-notebooks/submission.csv", 0.21),
    ("sub_2", "../input/gaps-features-tf-lstm-resnet-like-ff/sub.csv", 0.10),
    ("sub_3", "../input/vent-011-median-wins/submission.csv", 0.56),
]

loaded_subs = []
loaded_wts = []

for name, path, wt in external_paths:
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "pressure" in df.columns and len(df) == len(sub_template):
            loaded_subs.append(df["pressure"].to_numpy(dtype=np.float64))
            loaded_wts.append(wt)



## === cell 2
if len(loaded_subs) > 0:
    w = np.array(loaded_wts, dtype=np.float64)
    w = w / w.sum()
    preds = np.zeros(len(sub_template), dtype=np.float64)
    for i, arr in enumerate(loaded_subs):
        preds += w[i] * arr

    sub = pd.DataFrame(
        {"id": sub_template["id"].to_numpy(), "pressure": preds.astype(np.float64)}
    )
else:
    train = pd.read_csv(
        train_path,
        usecols=["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
    )
    test = pd.read_csv(
        test_path, usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
    )

    test_row_id = np.arange(len(test), dtype=np.int64)
    assert len(test) == len(
        sub_template
    ), "test.csv and sample_submission.csv must have identical row count for row-order alignment"

    train_ts = train["time_step"].to_numpy(dtype=np.float64)
    test_ts = test["time_step"].to_numpy(dtype=np.float64)

    train_uin_01 = np.round(train["u_in"].to_numpy(dtype=np.float64), 1)
    test_uin_01 = np.round(test["u_in"].to_numpy(dtype=np.float64), 1)

    train_uin_05 = (
        np.round(train["u_in"].to_numpy(dtype=np.float64) * 2.0) / 2.0
    ).astype(np.float64)
    test_uin_05 = (
        np.round(test["u_in"].to_numpy(dtype=np.float64) * 2.0) / 2.0
    ).astype(np.float64)

    train_uin_int = np.round(train["u_in"].to_numpy(dtype=np.float64), 0).astype(
        np.int16
    )
    test_uin_int = np.round(test["u_in"].to_numpy(dtype=np.float64), 0).astype(np.int16)

    train_idx = np.arange(len(train), dtype=np.int64)
    test_idx = np.arange(len(test), dtype=np.int64)

    train_tmp = pd.DataFrame(
        {
            "idx": train_idx,
            "breath_id": train["breath_id"].to_numpy(),
            "time_step": train_ts,
            "u_in": train["u_in"].to_numpy(),
        }
    ).sort_values(["breath_id", "time_step", "idx"], kind="mergesort")
    train_tmp["u_in_cum"] = train_tmp.groupby("breath_id", sort=False)["u_in"].cumsum()
    train_uin_cum = train_tmp.sort_values("idx", kind="mergesort")["u_in_cum"].to_numpy(
        dtype=np.float64
    )

    test_tmp = pd.DataFrame(
        {
            "idx": test_idx,
            "breath_id": test["breath_id"].to_numpy(),
            "time_step": test_ts,
            "u_in": test["u_in"].to_numpy(),
        }
    ).sort_values(["breath_id", "time_step", "idx"], kind="mergesort")
    test_tmp["u_in_cum"] = test_tmp.groupby("breath_id", sort=False)["u_in"].cumsum()
    test_uin_cum = test_tmp.sort_values("idx", kind="mergesort")["u_in_cum"].to_numpy(
        dtype=np.float64
    )

    train_uin_cum_10 = np.round(train_uin_cum / 10.0) * 10.0
    test_uin_cum_10 = np.round(test_uin_cum / 10.0) * 10.0
    train_uin_cum_25 = np.round(train_uin_cum / 25.0) * 25.0
    test_uin_cum_25 = np.round(test_uin_cum / 25.0) * 25.0

    train_key = pd.DataFrame(
        {
            "breath_id": train["breath_id"].to_numpy(np.int32),
            "R": train["R"].to_numpy(np.int16),
            "C": train["C"].to_numpy(np.int16),
            "ts": train_ts,
            "u_out": train["u_out"].to_numpy(np.int8),
            "u_in_01": train_uin_01,
            "u_in_05": train_uin_05,
            "u_in_int": train_uin_int,
            "u_in_cum_10": train_uin_cum_10.astype(np.float64),
            "u_in_cum_25": train_uin_cum_25.astype(np.float64),
            "pressure": train["pressure"].to_numpy(np.float64),
        }
    )
    test_key = pd.DataFrame(
        {
            "row_id": test_row_id,  # stable unique row identifier
            "breath_id": test["breath_id"].to_numpy(np.int32),
            "R": test["R"].to_numpy(np.int16),
            "C": test["C"].to_numpy(np.int16),
            "ts": test_ts,
            "u_out": test["u_out"].to_numpy(np.int8),
            "u_in_01": test_uin_01,
            "u_in_05": test_uin_05,
            "u_in_int": test_uin_int,
            "u_in_cum_10": test_uin_cum_10.astype(np.float64),
            "u_in_cum_25": test_uin_cum_25.astype(np.float64),
        }
    )

    insp_mask = test_key["u_out"].to_numpy(dtype=np.int8) == 0
    test_insp = test_key.loc[
        insp_mask,
        [
            "row_id",
            "breath_id",
            "R",
            "C",
            "ts",
            "u_out",
            "u_in_01",
            "u_in_05",
            "u_in_int",
            "u_in_cum_10",
            "u_in_cum_25",
        ],
    ].copy()

    med_map_hist_10 = (
        train_key.groupby(
            ["R", "C", "ts", "u_out", "u_in_01", "u_in_cum_10"], sort=False
        )["pressure"]
        .median()
        .reset_index()
    )
    pred_insp = test_insp.merge(
        med_map_hist_10,
        on=["R", "C", "ts", "u_out", "u_in_01", "u_in_cum_10"],
        how="left",
    )["pressure"]

    if pred_insp.isna().any():
        med_map_hist_25 = (
            train_key.groupby(
                ["R", "C", "ts", "u_out", "u_in_01", "u_in_cum_25"], sort=False
            )["pressure"]
            .median()
            .reset_index()
        )
        need = pred_insp.isna().to_numpy()
        tmp = test_insp.loc[
            need, ["R", "C", "ts", "u_out", "u_in_01", "u_in_cum_25"]
        ].merge(
            med_map_hist_25,
            on=["R", "C", "ts", "u_out", "u_in_01", "u_in_cum_25"],
            how="left",
        )[
            "pressure"
        ]
        pred_insp.loc[need] = tmp.to_numpy(dtype=np.float64)

    if pred_insp.isna().any():
        med_map_full = (
            train_key.groupby(["R", "C", "ts", "u_out", "u_in_01"], sort=False)[
                "pressure"
            ]
            .median()
            .reset_index()
        )
        need = pred_insp.isna().to_numpy()
        tmp = test_insp.loc[need, ["R", "C", "ts", "u_out", "u_in_01"]].merge(
            med_map_full, on=["R", "C", "ts", "u_out", "u_in_01"], how="left"
        )["pressure"]
        pred_insp.loc[need] = tmp.to_numpy(dtype=np.float64)

    if pred_insp.isna().any():
        med_map_uin_05 = (
            train_key.groupby(["R", "C", "ts", "u_out", "u_in_05"], sort=False)[
                "pressure"
            ]
            .median()
            .reset_index()
        )
        need = pred_insp.isna().to_numpy()
        tmp = test_insp.loc[need, ["R", "C", "ts", "u_out", "u_in_05"]].merge(
            med_map_uin_05, on=["R", "C", "ts", "u_out", "u_in_05"], how="left"
        )["pressure"]
        pred_insp.loc[need] = tmp.to_numpy(dtype=np.float64)

    if pred_insp.isna().any():
        med_map_uin_int = (
            train_key.groupby(["R", "C", "ts", "u_out", "u_in_int"], sort=False)[
                "pressure"
            ]
            .median()
            .reset_index()
        )
        need = pred_insp.isna().to_numpy()
        tmp = test_insp.loc[need, ["R", "C", "ts", "u_out", "u_in_int"]].merge(
            med_map_uin_int, on=["R", "C", "ts", "u_out", "u_in_int"], how="left"
        )["pressure"]
        pred_insp.loc[need] = tmp.to_numpy(dtype=np.float64)

    if pred_insp.isna().any():
        med_map_backoff = (
            train_key.groupby(["R", "C", "ts", "u_out"], sort=False)["pressure"]
            .median()
            .reset_index()
        )
        need = pred_insp.isna().to_numpy()
        tmp = test_insp.loc[need, ["R", "C", "ts", "u_out"]].merge(
            med_map_backoff, on=["R", "C", "ts", "u_out"], how="left"
        )["pressure"]
        pred_insp.loc[need] = tmp.to_numpy(dtype=np.float64)

    global_med = float(train["pressure"].median())
    pred_insp = pred_insp.fillna(global_med).to_numpy(dtype=np.float64)

    pred_full = np.full(len(test_key), np.nan, dtype=np.float64)
    pred_full[insp_mask] = pred_insp

    tmp_df = pd.DataFrame(
        {
            "row_id": test_key["row_id"].to_numpy(),
            "breath_id": test_key["breath_id"].to_numpy(),
            "time_step": test_key["ts"].to_numpy(),
            "u_out": test_key["u_out"].to_numpy(),
            "pred": pred_full,
        }
    ).sort_values(["breath_id", "time_step", "row_id"], kind="mergesort")

    tmp_df["pred_insp_only"] = tmp_df["pred"].where(
        tmp_df["u_out"].to_numpy() == 0, np.nan
    )
    tmp_df["pred_filled"] = tmp_df.groupby("breath_id", sort=False)[
        "pred_insp_only"
    ].ffill()
    tmp_df["pred_filled"] = tmp_df["pred_filled"].fillna(global_med)

    tmp_df = tmp_df.sort_values("row_id", kind="mergesort")
    preds = tmp_df["pred_filled"].to_numpy(dtype=np.float64)

    assert len(preds) == len(test), "Prediction length mismatch vs test.csv"

    sub = pd.DataFrame(
        {"id": test["id"].to_numpy(), "pressure": preds.astype(np.float64)}
    )



## === cell 3
sub = sub[["id", "pressure"]]

assert list(sub.columns) == ["id", "pressure"]
assert len(sub) > 0

sub.to_csv("submission.csv", index=False)
sub.head()
