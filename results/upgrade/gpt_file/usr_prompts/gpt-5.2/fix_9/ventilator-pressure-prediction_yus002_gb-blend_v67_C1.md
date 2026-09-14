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

0.1503243562841331

# 6. Current score

6.87757

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'The crash happens because `../input/gb-blending` is not present in your environment, so `glob` finds no files and the blending code ends up producing a single scalar instead of a 603600-length prediction vector. I make `g()` robust: it validate that input prediction files exist, ensure every loaded prediction has the correct length, and if none exist it fall back to creating a valid submission using the sample submission (all zeros) so the notebook always yields a `.csv`. I also fix the output filename to be a safe `.csv` name (no spaces needed, but allowed) and keep the nearest-pressure snapping logic unchanged. No model/training logic is introduced; this is purely to unblock end-to-end execution and guarantee a valid submission file.'
- What this solution (achieved 8.13673) has done: 'I fix the baseline fallback that currently drops/renames the `pressure` column due to a merge creating `pressure_x/pressure_y`, which triggers the `KeyError: 'pressure'`. I make `_baseline_submission_mean_by_rc_timestep()` build the submission by aligning on `id` and explicitly writing into the existing `pressure` column, preserving the expected submission format. I also make `g()` only consider `.csv` files and validate required columns/lengths to avoid loading non-submission artifacts. These changes are minimal, keep the blending/snapping logic intact, and ensure an end-to-end run that produces a valid `submission.csv`.'
- What this solution (achieved 7.81735) has done: 'Your current score is far worse than the target (lower-is-better), and the biggest issue is that you’re effectively producing a weak baseline because the blending directory isn’t present. I keep your blending/nearest-pressure logic intact, but change the pipeline to generate multiple diverse, fully-valid submissions from the provided train/test (simple group-mean baselines) and then feed them through your existing random-weight + median ensembling in `g()`. This is a minimal change that uses only the available data, keeps runtime under control, and should move MAE substantially toward the target without introducing any new model/training approach. The output remains a standard `submission.csv` with `id,pressure`.'
- What this solution (achieved 7.67885) has done: 'I keep your blending/median-over-random-weights and nearest-pressure snapping intact, but I make the locally-generated blend files much stronger by switching from simple group means to out-of-fold (by breath_id) target-encoding for each grouping, which reduces leakage and usually lowers MAE substantially. I also fix the specs that currently reference `u_in/u_out` without loading those columns (so those blend files were silently failing), and I ensure all generated predictions are aligned to `id` and have the exact expected length before ensembling. These are minimal, metric-aligned changes that preserve your overall approach (generate multiple submissions → blend them) while moving your 7.817 score closer to the 0.150 target. The script still run end-to-end and write `submission.csv` with `id,pressure`.'
- What this solution (achieved 7.67885) has done: 'Your current MAE (7.68, lower-is-better) is far from the target (0.15), so we need a meaningful improvement while keeping your blending + nearest-pressure snapping logic intact. The biggest leverage with minimal change is to generate stronger component predictions for your existing blender by adding a simple, metric-aligned correction: predict pressure only for inspiratory steps (u_out==0) and fill expiratory steps (u_out==1, not scored) with a safe constant (global mean snapped), which typically avoids large outlier errors. To preserve your approach, I only (1) ensure required test columns are loaded for group specs, (2) apply the u_out gating consistently in every generated submission and the baseline, and (3) keep the rest of your ensembling unchanged so it still writes a valid `submission.csv`.'
- What this solution (achieved 7.51033) has done: 'Your current MAE (7.67885, lower-is-better) is still very far from the target (0.1503), so we need a meaningful uplift while keeping your “generate a few baseline submissions → random-weight blending → nearest-pressure snapping” core logic unchanged. The biggest issue is that your group-mean components are too coarse: they ignore the strong sequential structure within a breath, so errors remain huge even after blending. With minimal disruption, I add two stronger-but-still-simple component generators: (1) a deterministic “within-breath incremental u_in integral” linear calibration per (R,C) fitted on train (captures pressure’s dependence on cumulative flow), and (2) an additional group-mean keyed on (R,C,u_in, time_step) but with time_step rounded to reduce sparsity and improve matching. These extra components are written as additional CSVs into the same blending folder and then your existing `g()` blend them as usual; submission format and nearest-pressure snapping remain identical.'
- What this solution (achieved 6.87757) has done: 'Your score is still far above the target (lower-is-better), so we need a small but meaningful uplift without changing your overall “generate a few baseline submissions → random-weight blending → nearest-pressure snapping” approach. The biggest bug hurting these baselines is that all group-mean components are trained on *all* timesteps, even though the metric ignores expiratory timesteps (u_out==1); this contaminates the mean targets and makes inspiratory predictions worse. I keep the same generators and blending, but compute all group means using only inspiratory rows (u_out==0), and I also compute the linear calibration using only inspiratory rows (still predicting for all rows, with the existing u_out gating for test). These changes are metric-aligned, minimal, and should move MAE down toward the target while preserving your pipeline and submission format.'

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
    l = []
    arrs = []
    for p in input_list:
        try:
            public_lb_score = int(p.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1
        l.append(public_lb_score)

        s = pd.read_csv(p)
        if "pressure" not in s.columns:
            raise ValueError(f"File {p} does not contain a 'pressure' column.")
        arrs.append(s["pressure"].to_numpy().ravel())

    l_sum = sum(l) if sum(l) != 0 else len(l)

    if len(arrs) == 1:
        return arrs[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        return arrs[0] * weight1 + arrs[1] * weight2


def _baseline_submission_mean_by_rc_timestep():
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    test_path = "../input/ventilator-pressure-prediction/test.csv"

    sub = pd.read_csv(sample_path, usecols=["id", "pressure"])
    test = pd.read_csv(test_path, usecols=["id", "R", "C", "time_step", "u_out"])

    train_insp = df_train.loc[
        df_train["u_out"] == 0, ["R", "C", "time_step", "pressure"]
    ]
    grp = (
        train_insp.groupby(["R", "C", "time_step"], sort=False)["pressure"]
        .mean()
        .reset_index()
    )

    test = test.merge(grp, on=["R", "C", "time_step"], how="left")

    global_mean = float(train_insp["pressure"].mean())
    pred = test.set_index("id")["pressure"].fillna(global_mean)

    sub = sub.set_index("id")
    sub["pressure"] = pred.reindex(sub.index).to_numpy(dtype=np.float64)

    uout = test.set_index("id")["u_out"].reindex(sub.index).to_numpy()
    const_exp = find_nearest(global_mean)
    sub.loc[uout == 1, "pressure"] = const_exp

    sub = sub.reset_index()
    sub["pressure"] = sub["pressure"].apply(find_nearest)

    sub.to_csv("submission.csv", index=False)
    return sub


def _make_groupmean_submission_oof(
    group_cols,
    out_path,
    n_folds=5,
    add_global_fallback=True,
):
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    test_path = "../input/ventilator-pressure-prediction/test.csv"

    sub = pd.read_csv(sample_path, usecols=["id", "pressure"])

    test_use = ["id"] + [c for c in group_cols if c != "id"]
    if "u_out" not in test_use:
        test_use.append("u_out")
    test = pd.read_csv(test_path, usecols=test_use)

    breaths = df_train["breath_id"].unique()
    rng = np.random.RandomState(2021)
    rng.shuffle(breaths)
    folds = np.array_split(breaths, n_folds)

    train_insp = df_train.loc[df_train["u_out"] == 0]
    global_mean = float(train_insp["pressure"].mean())

    key_cols = [c for c in group_cols if c != "id"]
    need_cols = ["breath_id", "id"] + key_cols + ["pressure"]
    train_keys = train_insp[need_cols].copy()

    oof_pred = np.empty(len(train_keys), dtype=np.float64)

    for f in range(n_folds):
        val_b = set(folds[f].tolist())
        is_val = train_keys["breath_id"].isin(val_b).to_numpy()

        tr = train_keys.loc[~is_val, key_cols + ["pressure"]]
        val = train_keys.loc[is_val, key_cols]

        grp = tr.groupby(key_cols, sort=False)["pressure"].mean()
        pred = val.merge(grp.rename("m").reset_index(), on=key_cols, how="left")[
            "m"
        ].to_numpy()

        if add_global_fallback:
            pred = np.where(np.isnan(pred), global_mean, pred)

        oof_pred[is_val] = pred

    full_grp = train_insp.groupby(key_cols, sort=False)["pressure"].mean().reset_index()
    merged_test = test.merge(full_grp, on=key_cols, how="left")

    if add_global_fallback:
        test_pred = merged_test.set_index("id")["pressure"].fillna(global_mean)
    else:
        test_pred = merged_test.set_index("id")["pressure"]

    sub = sub.set_index("id")
    sub["pressure"] = test_pred.reindex(sub.index).to_numpy(dtype=np.float64)

    uout = test.set_index("id")["u_out"].reindex(sub.index).to_numpy()
    const_exp = find_nearest(global_mean)
    sub.loc[uout == 1, "pressure"] = const_exp

    sub = sub.reset_index()
    sub["pressure"] = sub["pressure"].apply(find_nearest)
    sub.to_csv(out_path, index=False)
    return out_path


def _make_rc_linear_cumuin_submission(out_path):
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    test_path = "../input/ventilator-pressure-prediction/test.csv"

    sub = pd.read_csv(sample_path, usecols=["id", "pressure"])
    test = pd.read_csv(
        test_path, usecols=["id", "breath_id", "R", "C", "u_in", "u_out", "time_step"]
    )

    train_full = df_train[
        ["breath_id", "R", "C", "u_in", "u_out", "time_step", "pressure"]
    ].copy()

    train_full["dt"] = (
        train_full.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0)
    )
    test["dt"] = test.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0)

    train_full["u_in_int"] = (
        (train_full["u_in"] * train_full["dt"])
        .groupby(train_full["breath_id"], sort=False)
        .cumsum()
    )
    test["u_in_int"] = (
        (test["u_in"] * test["dt"]).groupby(test["breath_id"], sort=False).cumsum()
    )

    train = train_full.loc[train_full["u_out"] == 0].copy()
    global_mean = float(train["pressure"].mean())

    X1 = np.ones(len(train), dtype=np.float64)
    X2 = train["u_in_int"].to_numpy(np.float64)
    X3 = train["u_in"].to_numpy(np.float64)
    X4 = train["u_out"].to_numpy(np.float64)
    y = train["pressure"].to_numpy(np.float64)

    tmp = pd.DataFrame(
        {
            "R": train["R"].to_numpy(),
            "C": train["C"].to_numpy(),
            "x1": X1,
            "x2": X2,
            "x3": X3,
            "x4": X4,
            "y": y,
            "x1x1": X1 * X1,
            "x1x2": X1 * X2,
            "x1x3": X1 * X3,
            "x1x4": X1 * X4,
            "x2x2": X2 * X2,
            "x2x3": X2 * X3,
            "x2x4": X2 * X4,
            "x3x3": X3 * X3,
            "x3x4": X3 * X4,
            "x4x4": X4 * X4,
            "x1y": X1 * y,
            "x2y": X2 * y,
            "x3y": X3 * y,
            "x4y": X4 * y,
        }
    )

    agg = tmp.groupby(["R", "C"], sort=False).sum(numeric_only=True).reset_index()

    coefs = {}
    for _, r in agg.iterrows():
        A = np.array(
            [
                [r["x1x1"], r["x1x2"], r["x1x3"], r["x1x4"]],
                [r["x1x2"], r["x2x2"], r["x2x3"], r["x2x4"]],
                [r["x1x3"], r["x2x3"], r["x3x3"], r["x3x4"]],
                [r["x1x4"], r["x2x4"], r["x3x4"], r["x4x4"]],
            ],
            dtype=np.float64,
        )
        b = np.array([r["x1y"], r["x2y"], r["x3y"], r["x4y"]], dtype=np.float64)

        A = A + np.eye(4, dtype=np.float64) * 1e-6
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.array([global_mean, 0.0, 0.0, 0.0], dtype=np.float64)
        coefs[(int(r["R"]), int(r["C"]))] = w

    X1t = np.ones(len(test), dtype=np.float64)
    X2t = test["u_in_int"].to_numpy(np.float64)
    X3t = test["u_in"].to_numpy(np.float64)
    X4t = test["u_out"].to_numpy(np.float64)

    pred = np.empty(len(test), dtype=np.float64)
    rc = list(zip(test["R"].to_numpy(), test["C"].to_numpy()))
    for i, key in enumerate(rc):
        w = coefs.get((int(key[0]), int(key[1])))
        if w is None:
            pred[i] = global_mean
        else:
            pred[i] = w[0] * X1t[i] + w[1] * X2t[i] + w[2] * X3t[i] + w[3] * X4t[i]

    sub = sub.set_index("id")
    pred_ser = pd.Series(pred, index=test["id"].to_numpy()).reindex(sub.index)
    sub["pressure"] = pred_ser.to_numpy(dtype=np.float64)

    uout = test.set_index("id")["u_out"].reindex(sub.index).to_numpy()
    const_exp = find_nearest(global_mean)
    sub.loc[uout == 1, "pressure"] = const_exp

    sub = sub.reset_index()
    sub["pressure"] = sub["pressure"].apply(find_nearest)
    sub.to_csv(out_path, index=False)
    return out_path


def _make_groupmean_submission_time_rounded(group_cols, out_path, time_round=2):
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    test_path = "../input/ventilator-pressure-prediction/test.csv"

    sub = pd.read_csv(sample_path, usecols=["id", "pressure"])

    use_test = ["id", "u_out"] + [c for c in group_cols if c not in ("id", "u_out")]
    test = pd.read_csv(test_path, usecols=use_test)

    train_need = ["id", "pressure", "u_out"] + [c for c in group_cols if c != "id"]
    train = df_train[train_need].copy()
    train = train.loc[train["u_out"] == 0].copy()

    if "time_step" in group_cols:
        train["time_step"] = train["time_step"].round(time_round)
        test["time_step"] = test["time_step"].round(time_round)

    key_cols = [c for c in group_cols if c != "id"]
    grp = train.groupby(key_cols, sort=False)["pressure"].mean().reset_index()

    merged = test.merge(grp, on=key_cols, how="left")
    global_mean = float(train["pressure"].mean())
    pred = merged.set_index("id")["pressure"].fillna(global_mean)

    sub = sub.set_index("id")
    sub["pressure"] = pred.reindex(sub.index).to_numpy(dtype=np.float64)

    uout = test.set_index("id")["u_out"].reindex(sub.index).to_numpy()
    const_exp = find_nearest(global_mean)
    sub.loc[uout == 1, "pressure"] = const_exp

    sub = sub.reset_index()
    sub["pressure"] = sub["pressure"].apply(find_nearest)
    sub.to_csv(out_path, index=False)
    return out_path


def _prepare_local_blends(dp):
    os.makedirs(dp, exist_ok=True)
    created = []

    specs = [
        (["R", "C", "time_step"], "01_rc_timestep.csv"),
        (["R", "time_step"], "02_r_timestep.csv"),
        (["C", "time_step"], "03_c_timestep.csv"),
        (["R", "C", "u_out", "time_step"], "04_rc_uout_timestep.csv"),
        (["R", "C", "u_in", "time_step"], "05_rc_uin_timestep.csv"),
        (["R", "C", "u_out", "u_in", "time_step"], "06_rc_uout_uin_timestep.csv"),
        (["R", "C", "u_out"], "07_rc_uout.csv"),
        (["R", "C", "u_in"], "08_rc_uin.csv"),
    ]

    for cols, name in specs:
        out_path = os.path.join(dp, name)
        if not os.path.exists(out_path):
            try:
                _make_groupmean_submission_oof(cols, out_path, n_folds=5)
            except Exception:
                continue

        try:
            s = pd.read_csv(out_path, usecols=["id", "pressure"])
            if len(s) == 603600:
                created.append(out_path)
        except Exception:
            continue

    extra1 = os.path.join(dp, "09_rc_linear_cumuin.csv")
    if not os.path.exists(extra1):
        try:
            _make_rc_linear_cumuin_submission(extra1)
        except Exception:
            pass
    try:
        s = pd.read_csv(extra1, usecols=["id", "pressure"])
        if len(s) == 603600:
            created.append(extra1)
    except Exception:
        pass

    extra2 = os.path.join(dp, "10_rc_uin_timestep_round2.csv")
    if not os.path.exists(extra2):
        try:
            _make_groupmean_submission_time_rounded(
                ["R", "C", "u_in", "time_step"], extra2, time_round=2
            )
        except Exception:
            pass
    try:
        s = pd.read_csv(extra2, usecols=["id", "pressure"])
        if len(s) == 603600:
            created.append(extra2)
    except Exception:
        pass

    return created


def g(dp):
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    output = pd.read_csv(sample_path, usecols=["id", "pressure"])
    n_expected = len(output)

    files = sorted(
        [
            p
            for p in glob.iglob(f"{dp}/*")
            if os.path.isfile(p) and p.lower().endswith(".csv")
        ]
    )

    if len(files) == 0:
        _prepare_local_blends(dp)
        files = sorted(
            [
                p
                for p in glob.iglob(f"{dp}/*")
                if os.path.isfile(p) and p.lower().endswith(".csv")
            ]
        )

    if len(files) == 0:
        return _baseline_submission_mean_by_rc_timestep()

    flist = []
    file_count = len(files)
    splits = max(1, file_count // 2)
    for i in range(splits):
        if i == splits - 1:
            chunk = files[i * round(len(files) / splits) :]
        else:
            chunk = files[
                i * round(len(files) / splits) : (i + 1) * round(len(files) / splits)
            ]
        if len(chunk) > 0:
            flist.append(chunk)

    pred_arrays = []
    for chunk in flist:
        try:
            arr = wc(chunk)
            if getattr(arr, "shape", None) is None or arr.shape[0] != n_expected:
                continue
            pred_arrays.append(arr.astype(np.float64, copy=False))
        except Exception:
            continue

    if len(pred_arrays) == 0:
        return _baseline_submission_mean_by_rc_timestep()

    loop_time = 150
    pred_list = []
    for k in range(loop_time):
        set_seed(k)
        weight = [rd() for _ in range(len(pred_arrays))]
        weight_sum = sum(weight) if sum(weight) != 0 else 1.0
        weight = [w / weight_sum for w in weight]
        weight.sort(reverse=True)

        temp = np.zeros(n_expected, dtype=np.float64)
        for i in range(len(pred_arrays)):
            temp += pred_arrays[i] * weight[i]
        pred_list.append(temp)
        gc.collect()

    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)

    output.to_csv("submission.csv", index=False)
    return output


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.5 + b.pressure * 0.5
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
g("./gb-blending")
