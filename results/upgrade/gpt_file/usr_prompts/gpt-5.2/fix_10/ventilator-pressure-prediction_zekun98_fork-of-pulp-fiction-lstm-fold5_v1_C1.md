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

0.148543004919559

# 6. Current score

3.79188

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 7.54738) has done: 'Your code doesn’t yield a Kaggle score because it never writes the required `submission.csv` filename, and it also tries to read fold submissions from a path that likely doesn’t exist in your environment (so it falls back to all-zeros predictions). I make the smallest changes needed to (1) robustly load `sample_submission.csv` from the available dataset path, (2) ensure the final file is written as `submission.csv` with the correct columns, and (3) add a tiny, legitimate improvement over all-zeros by using a simple train-set median pressure for inspiratory phase (`u_out==0`) as the fallback prediction (still preserving your ensemble/median + quantization core logic). This should move the score substantially toward the target because a constant near-typical pressure is far better than predicting all zeros. All other logic (fold discovery, median ensembling, rounding to pressure grid, clipping) stays intact.'
- What this solution (achieved 6.10591) has done: 'Your current score (7.54738 MAE) is far above the target (0.1485), so we should legitimately improve predictions while keeping your ensemble/median + pressure-grid quantization logic intact. The biggest issue is that you’re not using any model signal at all (either missing fold files or falling back to a constant), so we create a minimal, fast, non-iterative per-(R,C,time_step) lookup from the training set inspiratory phase and use it to generate test predictions; this keeps the same “load predictions → median ensemble → quantize” pipeline but replaces the constant fallback with a much stronger baseline. To avoid leakage and respect the metric, we only learn from `u_out==0` rows and we use a sensible hierarchical fallback (exact key → (R,C,time_step) mean → (R,C) mean → global inspiratory median). We keep your file paths, output `submission.csv`, and keep the rounding/clipping exactly as you had it.'
- What this solution (achieved 4.08571) has done: 'Your current score (6.10591 MAE) is still far above the target (0.14854), so we should legitimately improve predictions while keeping your existing “fallback predictions → 5-fold-like stack → median ensemble → pressure-grid rounding/clipping” core logic unchanged. The main missing signal is that the fallback ignores `u_in` (the key control input), so we minimally extend your training-set lookup to use `(R, C, ts_round, u_in_round)` means for inspiratory rows, with hierarchical fallback back to your existing `(R,C,ts_round) → (R,C) → global` path. This stays non-iterative (fast), uses only train data, and respects the metric by learning only from `u_out==0` while still providing a reasonable value for `u_out==1` test rows. Output filenames/paths and the post-processing quantization remain exactly as in your pipeline.'
- What this solution (achieved 8.28043) has done: 'Your current MAE (4.08571, lower is better) is still far above the target (0.14854), so we should legitimately improve predictions while keeping your existing “train-lookup fallback → 5x stack → median ensemble → pressure-grid rounding/clipping” pipeline intact. The smallest high-impact change is to make the lookup closer to the true simulator behavior by including a simple “integrated u_in” (cumulative inhaled air proxy) feature per breath, but without changing to any ML model or training loop. We compute `area_u_in` per breath from `u_in * delta_time` on inspiratory rows, bin it, and use it as an additional key in the same hierarchical mean-lookup with safe fallbacks to your existing keys. All file paths and submission writing stay the same; runtime remains fast (pure pandas groupby/merge).'
- What this solution (achieved 3.79188) has done: 'Your run currently yields no Kaggle score because there’s no indication you successfully submitted a non-empty, correctly aligned `submission.csv` (and some of your ID ranges shown are inconsistent with the real competition files), so I first harden the data loading to always use the canonical `/kaggle/input/ventilator-pressure-prediction`-style paths and verify row counts/ID alignment before writing. Then, to legitimately improve MAE toward the target while preserving your existing “train lookup fallback → 5x stack → median ensemble → pressure-grid rounding/clipping” core logic, I add one minimal, high-signal lookup layer based on the well-known discrete pressure grid: for inspiratory rows, map `(R,C,step,u_in_bin,prev_pressure_bin)` to the most likely next pressure (mode), with safe fallback to your existing mean-based hierarchy. This keeps everything non-iterative (pure pandas), fast, and aligned with the metric (learn only from `u_out==0`). Finally, I ensure `submission.csv` is written with exactly `id,pressure` and in the same order as `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import os

CANDIDATE_BASES = [
    "../input/ventilator-pressure-prediction",
    "/kaggle/input/ventilator-pressure-prediction",
    "/kaggle/data/ventilator-pressure-prediction",
    "/kaggle/input",
    "/kaggle/data",
]


def _first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


ss_path = _first_existing(
    *[os.path.join(b, "sample_submission.csv") for b in CANDIDATE_BASES]
)
train_path = _first_existing(*[os.path.join(b, "train.csv") for b in CANDIDATE_BASES])
test_path = _first_existing(*[os.path.join(b, "test.csv") for b in CANDIDATE_BASES])

if ss_path is None or train_path is None or test_path is None:
    raise FileNotFoundError(
        "Could not locate competition files. Checked bases: "
        + ", ".join(CANDIDATE_BASES)
    )

submission = pd.read_csv(ss_path)
if "id" not in submission.columns:
    raise ValueError(
        f"sample_submission.csv must contain 'id', got {submission.columns.tolist()}"
    )



## === cell 1
tr = pd.read_csv(
    train_path,
    usecols=["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"],
)
te = pd.read_csv(
    test_path, usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]
)

te = te.merge(submission[["id"]], on="id", how="inner")

tr_insp = tr[tr["u_out"] == 0].copy()

tr_insp = tr_insp.sort_values(["breath_id", "time_step"], kind="mergesort")
te = te.sort_values(["breath_id", "time_step"], kind="mergesort")

tr_insp["step"] = tr_insp.groupby("breath_id", sort=False).cumcount().astype("int16")
te["step"] = te.groupby("breath_id", sort=False).cumcount().astype("int16")

tr_insp["ts_round"] = tr_insp["time_step"].round(2)
te["ts_round"] = te["time_step"].round(2)

tr_insp["u_in_round"] = tr_insp["u_in"].round(1)
te["u_in_round"] = te["u_in"].round(1)

tr_insp["dt"] = tr_insp.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0)
te["dt"] = te.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0)

tr_insp["area_u_in"] = (
    (tr_insp["u_in"] * tr_insp["dt"]).groupby(tr_insp["breath_id"], sort=False).cumsum()
)
te["area_u_in"] = (te["u_in"] * te["dt"]).groupby(te["breath_id"], sort=False).cumsum()

tr_insp["area_bin"] = (tr_insp["area_u_in"] / 0.5).round().astype("int32")
te["area_bin"] = (te["area_u_in"] / 0.5).round().astype("int32")

tr_insp["u_in_bin"] = (tr_insp["u_in"] / 1.0).round().astype("int16")  # ~0..100
te["u_in_bin"] = (te["u_in"] / 1.0).round().astype("int16")

insp = tr_insp["pressure"]
fallback_pressure = (
    float(insp.median()) if len(insp) else float(tr["pressure"].median())
)
global_pred = fallback_pressure

mean_by_rc_step_uin = (
    tr_insp.groupby(["R", "C", "step", "u_in_bin"], sort=False)["pressure"]
    .mean()
    .reset_index()
)
mean_by_rc_step_area = (
    tr_insp.groupby(["R", "C", "step", "area_bin"], sort=False)["pressure"]
    .mean()
    .reset_index()
)
mean_by_rc_step = (
    tr_insp.groupby(["R", "C", "step"], sort=False)["pressure"].mean().reset_index()
)

mean_by_rc_ts_uin_area = (
    tr_insp.groupby(["R", "C", "ts_round", "u_in_round", "area_bin"], sort=False)[
        "pressure"
    ]
    .mean()
    .reset_index()
)
mean_by_rc_ts_uin = (
    tr_insp.groupby(["R", "C", "ts_round", "u_in_round"], sort=False)["pressure"]
    .mean()
    .reset_index()
)
mean_by_rc_ts = (
    tr_insp.groupby(["R", "C", "ts_round"], sort=False)["pressure"].mean().reset_index()
)
mean_by_rc = tr_insp.groupby(["R", "C"], sort=False)["pressure"].mean().reset_index()

tr_exp = tr[tr["u_out"] == 1].copy()
tr_exp = tr_exp.sort_values(["breath_id", "time_step"], kind="mergesort")
tr_exp["step"] = tr_exp.groupby("breath_id", sort=False).cumcount().astype("int16")
exp_global_pred = (
    float(tr_exp["pressure"].mean()) if len(tr_exp) else float(global_pred)
)
mean_exp_by_step = tr_exp.groupby(["step"], sort=False)["pressure"].mean().reset_index()

mean_by_rc_step_uin.head(), fallback_pressure



## === cell 2
diff_pressure = 0.07030215
min_pressure = -1.8957442945646408
max_pressure = 64.82099173863948


def _to_pressure_bin(p):
    return np.rint((p - min_pressure) / diff_pressure).astype(np.int16)


def _from_pressure_bin(b):
    return b.astype(np.float64) * diff_pressure + min_pressure


tr_insp2 = tr_insp.copy()
tr_insp2["pressure_bin"] = _to_pressure_bin(tr_insp2["pressure"].values)
tr_insp2["prev_pressure_bin"] = (
    tr_insp2.groupby("breath_id", sort=False)["pressure_bin"]
    .shift(1)
    .fillna(-32768)
    .astype(np.int16)
)

mode_key_cols = ["R", "C", "step", "u_in_bin", "prev_pressure_bin"]
mode_counts = (
    tr_insp2.groupby(mode_key_cols + ["pressure_bin"], sort=False)
    .size()
    .reset_index(name="cnt")
)

mode_counts = mode_counts.sort_values(
    mode_key_cols + ["cnt"],
    ascending=[True, True, True, True, True, False],
    kind="mergesort",
)
mode_map = mode_counts.drop_duplicates(mode_key_cols, keep="first")[
    mode_key_cols + ["pressure_bin"]
]
mode_map = mode_map.rename(columns={"pressure_bin": "pred_mode_bin"})

mode_counts2 = (
    tr_insp2.groupby(["R", "C", "step", "u_in_bin", "pressure_bin"], sort=False)
    .size()
    .reset_index(name="cnt")
)
mode_counts2 = mode_counts2.sort_values(
    ["R", "C", "step", "u_in_bin", "cnt"],
    ascending=[True, True, True, True, False],
    kind="mergesort",
)
mode_map2 = mode_counts2.drop_duplicates(["R", "C", "step", "u_in_bin"], keep="first")[
    ["R", "C", "step", "u_in_bin", "pressure_bin"]
]
mode_map2 = mode_map2.rename(columns={"pressure_bin": "pred_mode2_bin"})



## === cell 3
fold_files = sorted(glob.glob("../input/pulp-fiction-fold5-fold*/submission.csv"))

if len(fold_files) == 0:
    te_pred = te[
        [
            "id",
            "breath_id",
            "R",
            "C",
            "step",
            "u_in_bin",
            "ts_round",
            "u_in_round",
            "area_bin",
            "u_out",
        ]
    ].copy()

    te_pred["pred_bin"] = np.int16(-32768)
    te_pred["prev_pred_bin"] = np.int16(-32768)

    te_pred = te_pred.sort_values(["breath_id", "step"], kind="mergesort")

    base = te_pred.merge(
        mean_by_rc_step_uin, on=["R", "C", "step", "u_in_bin"], how="left"
    ).rename(columns={"pressure": "pred_rc_step_uin"})
    base = base.merge(
        mean_by_rc_step_area, on=["R", "C", "step", "area_bin"], how="left"
    ).rename(columns={"pressure": "pred_rc_step_area"})
    base = base.merge(mean_by_rc_step, on=["R", "C", "step"], how="left").rename(
        columns={"pressure": "pred_rc_step"}
    )
    base = base.merge(
        mean_by_rc_ts_uin_area,
        on=["R", "C", "ts_round", "u_in_round", "area_bin"],
        how="left",
    ).rename(columns={"pressure": "pred_rc_ts_uin_area"})
    base = base.merge(
        mean_by_rc_ts_uin, on=["R", "C", "ts_round", "u_in_round"], how="left"
    ).rename(columns={"pressure": "pred_rc_ts_uin"})
    base = base.merge(mean_by_rc_ts, on=["R", "C", "ts_round"], how="left").rename(
        columns={"pressure": "pred_rc_ts"}
    )
    base = base.merge(mean_by_rc, on=["R", "C"], how="left").rename(
        columns={"pressure": "pred_rc"}
    )

    base = base.merge(mean_exp_by_step, on=["step"], how="left").rename(
        columns={"pressure": "pred_exp_step"}
    )

    base["pred_pressure"] = np.nan

    for bid, idx in base.groupby("breath_id", sort=False).indices.items():
        rows = idx
        prev_bin = np.int16(-32768)
        for j in rows:
            if int(base.at[j, "u_out"]) == 1:
                p = base.at[j, "pred_exp_step"]
                if pd.isna(p):
                    p = exp_global_pred
                base.at[j, "pred_pressure"] = float(p)
                continue

            key = (
                base.at[j, "R"],
                base.at[j, "C"],
                base.at[j, "step"],
                base.at[j, "u_in_bin"],
                prev_bin,
            )

    mode_dict = {
        tuple(r): int(p)
        for r, p in zip(
            mode_map[mode_key_cols].to_numpy(), mode_map["pred_mode_bin"].to_numpy()
        )
    }
    mode2_dict = {
        tuple(r): int(p)
        for r, p in zip(
            mode_map2[["R", "C", "step", "u_in_bin"]].to_numpy(),
            mode_map2["pred_mode2_bin"].to_numpy(),
        )
    }

    for bid, rows in base.groupby("breath_id", sort=False).indices.items():
        prev_bin = np.int16(-32768)
        for j in rows:
            if int(base.at[j, "u_out"]) == 1:
                p = base.at[j, "pred_exp_step"]
                if pd.isna(p):
                    p = exp_global_pred
                base.at[j, "pred_pressure"] = float(p)
                continue

            k1 = (
                int(base.at[j, "R"]),
                int(base.at[j, "C"]),
                int(base.at[j, "step"]),
                int(base.at[j, "u_in_bin"]),
                int(prev_bin),
            )
            pb = mode_dict.get(k1, None)
            if pb is None:
                k2 = (
                    int(base.at[j, "R"]),
                    int(base.at[j, "C"]),
                    int(base.at[j, "step"]),
                    int(base.at[j, "u_in_bin"]),
                )
                pb = mode2_dict.get(k2, None)

            if pb is not None:
                p = float(_from_pressure_bin(np.array([pb], dtype=np.int16))[0])
            else:
                p = base.at[j, "pred_rc_step_uin"]
                if pd.isna(p):
                    p = base.at[j, "pred_rc_step_area"]
                if pd.isna(p):
                    p = base.at[j, "pred_rc_step"]
                if pd.isna(p):
                    p = base.at[j, "pred_rc_ts_uin_area"]
                if pd.isna(p):
                    p = base.at[j, "pred_rc_ts_uin"]
                if pd.isna(p):
                    p = base.at[j, "pred_rc_ts"]
                if pd.isna(p):
                    p = base.at[j, "pred_rc"]
                if pd.isna(p):
                    p = global_pred
                p = float(p)

            base.at[j, "pred_pressure"] = p
            prev_bin = np.int16(_to_pressure_bin(np.array([p], dtype=np.float64))[0])

    base_pred = (
        base.sort_values("id", kind="mergesort")["pred_pressure"]
        .astype("float64")
        .values
    )

    test_pred = pd.DataFrame({"id": submission["id"].values})
    for i in range(5):
        test_pred[f"pressure_{i}"] = base_pred
else:
    fold_dfs = []
    for i, fp in enumerate(fold_files):
        df = pd.read_csv(fp)
        if "id" not in df.columns or "pressure" not in df.columns:
            raise ValueError(
                f"Fold submission at {fp} must have columns ['id','pressure'], got {df.columns.tolist()}"
            )
        df = df[["id", "pressure"]].copy()
        df = df.rename(columns={"pressure": f"pressure_{i}"})
        fold_dfs.append(df)

    test_pred = fold_dfs[0]
    for df in fold_dfs[1:]:
        test_pred = test_pred.merge(df, on="id", how="inner")

    test_pred = submission[["id"]].merge(test_pred, on="id", how="left")



## === cell 4
test_pred.to_csv("submission_all.csv", index=False)



## === cell 5
pressure_cols = [c for c in test_pred.columns if c.startswith("pressure_")]
if len(pressure_cols) == 0:
    test_preds = [np.zeros(len(submission), dtype=float) for _ in range(5)]
else:
    pressure_cols = sorted(pressure_cols, key=lambda x: int(x.split("_")[1]))
    cols_used = pressure_cols[:5]
    test_preds = [test_pred[c].values for c in cols_used]
    if len(test_preds) < 5:
        last = test_preds[-1]
        while len(test_preds) < 5:
            test_preds.append(last)




## === cell 6
class config:
    paths = {
        "train": train_path,
        "test": test_path,
        "ss": ss_path,
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




## === cell 7
submission = submission[["id"]].copy()

submission["pressure"] = np.median(np.vstack(test_preds), axis=0)

submission["pressure"] = (
    np.round(
        (submission["pressure"] - config.post_processing["min_pressure"])
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

if submission.shape[0] == 0:
    raise ValueError("Submission is empty.")
if submission["id"].isna().any():
    raise ValueError("Submission contains NaN ids.")
if submission["pressure"].isna().any():
    raise ValueError("Submission contains NaN pressures.")
if submission["id"].nunique() != submission.shape[0]:
    raise ValueError("Submission contains duplicate ids.")

print(submission.head())
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)

submission.to_csv("submission_base.csv", index=False)
print("Wrote submission_base.csv with shape:", submission.shape)
