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

0.1463992680053852

# 6. Current score

5.20747

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'Your code fails because it expects external Kaggle “dataset” folders (`../input/gb-rwbt-files` and `../input/gb-vpp-whoppity-dub-dub`) that are not present in this environment, leading to zero files and missing CSVs. I keep the ensemble/median “core logic” intact but make it robust: if those external files aren’t available, it fall back to producing a valid submission using the provided `sample_submission.csv` (pressure=0) so a `.csv` is always generated. I also fix the cell numbering and add safe path resolution so it uses the available `/kaggle/input/ventilator-pressure-prediction/` data. This is mainly correctness/stability; without the external prediction files the score won’t reach the target, but you get a valid submission file end-to-end.'
- What this solution (achieved 8.6492) has done: 'Your current score is far above the target (lower is better), and the main reason is that your pipeline falls back to predicting all zeros because the expected external Kaggle dataset folders aren’t present. To move the score much closer to the target without changing your ensemble/median “core logic”, I (1) make the code discover and use any available CSV submissions under the provided `/kaggle/.../ventilator-pressure-prediction/` tree (and still support your original external paths if they exist), and (2) ensure all blends align by `id` before taking medians/weighted sums to prevent silent row-order mismatches. If no external predictions exist anywhere, it still generate a valid submission (but then the score remain poor), and all I/O paths remain within the Kaggle filesystem while always writing `submission.csv`.'
- What this solution (achieved 6.10817) has done: 'Your current score (8.6492, lower-is-better) is far worse than the target (0.1464), so we need a real prediction source rather than the “all zeros / random-blend of whatever CSVs exist” fallback. Keeping your ensemble/median/blending core logic intact, I add a score-relevant fallback that uses the provided `train.csv` to build a simple, deterministic nearest-neighbor baseline: for each `(R, C)` and time step within a breath, predict the median inspiratory-phase pressure from training, and use 0 during expiratory phase (`u_out=1`) to match the metric. This does not change your existing blending behavior when valid external prediction CSVs exist; it only activates when no usable submission CSVs are found. The output is still snapped to the known discrete pressure grid via your existing `find_nearest`, and it always writes a valid `submission.csv`.'
- What this solution (achieved 4.34794) has done: 'Your current score is far worse than the target (lower is better), and the biggest remaining gap is that the fallback baseline can still be improved without changing your ensemble/blending “core logic.” I keep your discovery/blending pipeline intact, but strengthen the fallback by (1) predicting only for inspiratory rows (u_out==0) using a more informative median grouped by (R,C,step,u_in_rounded) with safe backoffs, and (2) ensuring the final submission uses `u_out` masking (0 for expiratory) and snaps predictions to the known discrete pressure grid. This should materially reduce MAE versus the current (R,C,step) median-only baseline while staying deterministic and within the same overall semantics. All paths remain the same, it still writes `submission.csv`, and if external submission CSVs are present it continue to prefer them.'
- What this solution (achieved 3.72418) has done: 'Your current score is far above the target (lower is better), and the remaining gap is mainly because the fallback baseline is still too crude. I keep your ensemble discovery/blending logic intact, but make the fallback more score-aligned by (1) learning the typical pressure trajectory per (R,C) over time_step using inspiratory rows only, and (2) adding a small deterministic correction based on binned u_in at each step, with safe backoffs. I also enforce the competition’s evaluation semantics by masking expiratory rows (`u_out==1`) to 0 and snapping predictions to the known discrete pressure grid (your existing `find_nearest`). All paths and outputs remain the same, and it still always write a valid `submission.csv`.'
- What this solution (achieved 5.20747) has done: 'Your current score (3.72418, lower-is-better) is still far above the target (0.1464), so we need a stronger fallback prediction when no external model submission CSVs are found—without changing your ensemble/blending core logic. I keep your discovery + random-weight blending + median aggregation intact, but improve only the fallback baseline to better match the metric by learning per-(R,C) “pressure response vs u_in” curves at each step/time and then predicting from the test u_in (with safe hierarchical backoffs). I also keep your correctness guards: align by `id`, mask expiratory rows (`u_out==1`) to 0, and snap to the known discrete pressure grid via `find_nearest`. This should materially reduce MAE versus the current median+small-correction fallback and move closer to the target band while staying deterministic and within constraints.'

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
CANDIDATE_BASES = [
    "../input/ventilator-pressure-prediction",
    "/kaggle/input/ventilator-pressure-prediction",
    "/kaggle/data/ventilator-pressure-prediction",
    "/kaggle/input",
    "/kaggle/data",
]


def _first_existing_file(rel_path):
    for b in CANDIDATE_BASES:
        p = os.path.join(b, rel_path) if not rel_path.startswith("/") else rel_path
        if os.path.exists(p):
            return p
    return None


train_path = _first_existing_file("train.csv")
test_path = _first_existing_file("test.csv")
sample_sub_path = _first_existing_file("sample_submission.csv")

if train_path is None or sample_sub_path is None or test_path is None:
    raise FileNotFoundError(
        f"Could not locate train.csv/test.csv/sample_submission.csv in any of: {CANDIDATE_BASES}"
    )

df_train = pd.read_csv(train_path, usecols=["pressure"])
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


def _load_pressure_by_id(csv_path, sample_ids):
    """
    Score-relevant correctness fix: ensure any blended file aligns by id (row order is not guaranteed).
    Returns pressure array aligned to sample_submission id order.
    """
    df = pd.read_csv(csv_path, usecols=["id", "pressure"])
    df = df.drop_duplicates("id", keep="last")
    df = df.set_index("id").reindex(sample_ids)
    return df["pressure"].fillna(0.0).to_numpy(dtype=float)


def wc(input_list, sample_ids):
    """
    Weighted combine for a small set of submission files.
    Original code expects filenames containing a public LB score token; keep behavior but make robust.
    Also aligns predictions by id to avoid silent misalignment hurting MAE.
    """
    l = []
    preds = []
    for p in input_list:
        fname = os.path.basename(p)
        try:
            public_lb_score = int(fname.split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1  # fallback weight if naming pattern doesn't match
        l.append(public_lb_score)
        preds.append(_load_pressure_by_id(p, sample_ids))

    l_sum = sum(l) if sum(l) != 0 else 1
    if len(preds) == 1:
        output = preds[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        output = preds[0] * weight1 + preds[1] * weight2
    return output


def _discover_csvs_in_dir(dp):
    if dp is None:
        return []
    out = []
    for i in glob.iglob(f"{dp}/**/*.csv", recursive=True):
        if os.path.isfile(i):
            try:
                head = pd.read_csv(i, nrows=5)
                if "id" in head.columns and "pressure" in head.columns:
                    out.append(i)
            except Exception:
                continue
    return sorted(out)


def _fallback_rc_step_time_uin_regression_submission(out_csv_path="rwb_fallback.csv"):
    """
    Score-relevant fallback improvement (only when no usable submission CSVs are found):
    Keep a deterministic, non-ML baseline, but make it closer to the metric by learning
    a typical inspiratory pressure response to u_in for each (R,C,step,time_bin).

    Approach (still "median-based", same overall semantics):
      - Use inspiratory-only train rows (u_out==0).
      - Bin time_step to 0.01s and u_in to 1.0 to reduce noise.
      - For each group g=(R,C,step,time_bin), estimate a robust slope:
            slope_g = median( dP / dU ) over adjacent u_in bins within g.
        and an intercept using a robust central point:
            intercept_g = median(P) - slope_g * median(u_in)
      - Predict: P_hat = intercept_g + slope_g * u_in_test
      - Backoffs:
          (R,C,step,time_bin) -> (R,C,step) -> (step) -> global median.
      - Expiratory rows (u_out==1): predict 0.0 (not scored).
      - Snap to discrete pressure grid with find_nearest (as in your code).
    """
    usecols_train = ["R", "C", "breath_id", "u_out", "u_in", "time_step", "pressure"]
    usecols_test = ["id", "R", "C", "breath_id", "u_out", "u_in", "time_step"]
    tr = pd.read_csv(train_path, usecols=usecols_train)
    te = pd.read_csv(test_path, usecols=usecols_test)

    tr["step"] = tr.groupby("breath_id").cumcount().astype(np.int16)
    te["step"] = te.groupby("breath_id").cumcount().astype(np.int16)

    tr["time_bin"] = np.round(tr["time_step"].to_numpy(dtype=np.float32), 2).astype(
        np.float32
    )
    te["time_bin"] = np.round(te["time_step"].to_numpy(dtype=np.float32), 2).astype(
        np.float32
    )

    U_BIN = 1.0
    tr["u_in_bin"] = (
        np.round(tr["u_in"].to_numpy(dtype=np.float32) / U_BIN) * U_BIN
    ).astype(np.float32)

    tr_insp = tr[tr["u_out"] == 0].copy()

    p_by_u = (
        tr_insp.groupby(["R", "C", "step", "time_bin", "u_in_bin"], sort=False)[
            "pressure"
        ]
        .median()
        .reset_index(name="p_med")
    )

    p_by_u = p_by_u.sort_values(
        ["R", "C", "step", "time_bin", "u_in_bin"], kind="mergesort"
    )
    p_by_u["du"] = p_by_u.groupby(["R", "C", "step", "time_bin"], sort=False)[
        "u_in_bin"
    ].diff()
    p_by_u["dp"] = p_by_u.groupby(["R", "C", "step", "time_bin"], sort=False)[
        "p_med"
    ].diff()
    p_by_u["slope"] = p_by_u["dp"] / p_by_u["du"]
    p_by_u.loc[~np.isfinite(p_by_u["slope"]), "slope"] = np.nan

    slope_g = (
        p_by_u.groupby(["R", "C", "step", "time_bin"], sort=False)["slope"]
        .median()
        .reset_index(name="slope_g")
    )

    center_g = (
        tr_insp.groupby(["R", "C", "step", "time_bin"], sort=False)
        .agg(p_med=("pressure", "median"), u_med=("u_in", "median"))
        .reset_index()
    )
    g = center_g.merge(slope_g, on=["R", "C", "step", "time_bin"], how="left")
    g["slope_g"] = g["slope_g"].fillna(0.0)
    g["intercept_g"] = g["p_med"] - g["slope_g"] * g["u_med"]

    p_by_u2 = (
        tr_insp.groupby(["R", "C", "step", "u_in_bin"], sort=False)["pressure"]
        .median()
        .reset_index(name="p_med")
    )
    p_by_u2 = p_by_u2.sort_values(["R", "C", "step", "u_in_bin"], kind="mergesort")
    p_by_u2["du"] = p_by_u2.groupby(["R", "C", "step"], sort=False)["u_in_bin"].diff()
    p_by_u2["dp"] = p_by_u2.groupby(["R", "C", "step"], sort=False)["p_med"].diff()
    p_by_u2["slope"] = p_by_u2["dp"] / p_by_u2["du"]
    p_by_u2.loc[~np.isfinite(p_by_u2["slope"]), "slope"] = np.nan
    slope_rcs = (
        p_by_u2.groupby(["R", "C", "step"], sort=False)["slope"]
        .median()
        .reset_index(name="slope_rcs")
    )
    center_rcs = (
        tr_insp.groupby(["R", "C", "step"], sort=False)
        .agg(p_med=("pressure", "median"), u_med=("u_in", "median"))
        .reset_index()
    )
    rcs = center_rcs.merge(slope_rcs, on=["R", "C", "step"], how="left")
    rcs["slope_rcs"] = rcs["slope_rcs"].fillna(0.0)
    rcs["intercept_rcs"] = rcs["p_med"] - rcs["slope_rcs"] * rcs["u_med"]

    med_step = (
        tr_insp.groupby(["step"], sort=False)["pressure"]
        .median()
        .reset_index(name="p_step")
    )
    global_med = float(tr_insp["pressure"].median())

    te2 = te.merge(
        g[["R", "C", "step", "time_bin", "slope_g", "intercept_g"]],
        on=["R", "C", "step", "time_bin"],
        how="left",
    )
    te2 = te2.merge(
        rcs[["R", "C", "step", "slope_rcs", "intercept_rcs"]],
        on=["R", "C", "step"],
        how="left",
    )
    te2 = te2.merge(med_step, on="step", how="left")

    u = te2["u_in"].to_numpy(dtype=float)
    pred = np.full(len(te2), np.nan, dtype=float)

    m1 = te2["intercept_g"].notna().to_numpy() & te2["slope_g"].notna().to_numpy()
    if m1.any():
        pred[m1] = (
            te2.loc[m1, "intercept_g"].to_numpy(dtype=float)
            + te2.loc[m1, "slope_g"].to_numpy(dtype=float) * u[m1]
        )

    m2 = (
        np.isnan(pred)
        & te2["intercept_rcs"].notna().to_numpy()
        & te2["slope_rcs"].notna().to_numpy()
    )
    if m2.any():
        pred[m2] = (
            te2.loc[m2, "intercept_rcs"].to_numpy(dtype=float)
            + te2.loc[m2, "slope_rcs"].to_numpy(dtype=float) * u[m2]
        )

    m3 = np.isnan(pred)
    if m3.any():
        pred[m3] = te2.loc[m3, "p_step"].to_numpy(dtype=float)

    m4 = np.isnan(pred)
    if m4.any():
        pred[m4] = global_med

    pred[te2["u_out"].to_numpy() == 1] = 0.0

    pred = np.array([find_nearest(x) for x in pred], dtype=float)

    out = pd.DataFrame({"id": te2["id"].to_numpy(dtype=np.int64), "pressure": pred})
    out.to_csv(out_csv_path, index=False)
    return out_csv_path


def g(dp):
    """
    Create a random-weighted blend over files in a directory, then median-aggregate across loops.
    If expected external dir is missing, try to discover any available submission CSVs under known
    Kaggle input/data trees; if none exist, use a deterministic RC+step+time linear-u_in baseline.
    """
    sample_df = pd.read_csv(sample_sub_path, usecols=["id", "pressure"])
    sample_ids = sample_df["id"].to_numpy()

    files = _discover_csvs_in_dir(dp)

    if len(files) == 0:
        for root in [
            "/kaggle/input/ventilator-pressure-prediction",
            "/kaggle/data/ventilator-pressure-prediction",
            "/kaggle/input",
            "/kaggle/data",
        ]:
            if os.path.exists(root):
                cand = _discover_csvs_in_dir(root)
                cand = [
                    p
                    for p in cand
                    if os.path.basename(p) != "sample_submission.csv"
                    and os.path.basename(p) != "train.csv"
                    and os.path.basename(p) != "test.csv"
                ]
                if len(cand) > 0:
                    files = cand
                    break

    file_count = len(files)

    if file_count == 0:
        out_name = _fallback_rc_step_time_uin_regression_submission(
            out_csv_path="rwb_fallback.csv"
        )
        return out_name

    loop_time = max(1, 500 // file_count)
    splits = max(1, file_count // 2)

    flist = []
    for i in range(splits):
        if i == splits - 1:
            flist.append(files[i * round(len(files) / splits) :])
        else:
            flist.append(
                files[
                    i
                    * round(len(files) / splits) : (i + 1)
                    * round(len(files) / splits)
                ]
            )

    for i in range(len(flist)):
        flist[i] = wc(flist[i], sample_ids)

    pred_list = []
    for loop_idx in range(loop_time):
        weight = []
        set_seed(loop_idx)
        for _ in range(len(flist)):
            weight.append(rd())

        weight_sum = sum(weight)
        if weight_sum == 0:
            weight_sum = 1.0
        for j in range(len(weight)):
            weight[j] /= weight_sum

        weight.sort(reverse=True)
        temp = 0.0
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()

    output = pd.read_csv(sample_sub_path)
    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)

    out_name = f"rwb {loop_time} loops.csv"
    output.to_csv(out_name, index=False)
    return out_name




## === cell 2
rwb_path = g("../input/gb-rwbt-files")
print("Generated:", rwb_path)



## === cell 3
external_sub_path = "../input/gb-vpp-whoppity-dub-dub/median_submission.csv"

sample_df = pd.read_csv(sample_sub_path, usecols=["id", "pressure"])
sample_ids = sample_df["id"].to_numpy()

df_2_pressure = _load_pressure_by_id(rwb_path, sample_ids)

if os.path.exists(external_sub_path):
    df_1_pressure = _load_pressure_by_id(external_sub_path, sample_ids)
    final_pressure = np.median(
        np.stack([df_1_pressure, df_2_pressure], axis=1),
        axis=1,
    )
else:
    final_pressure = df_2_pressure

df_final = pd.DataFrame({"id": sample_ids, "pressure": final_pressure})
df_final.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_final.shape)
print(
    "submission.csv pressure stats:",
    df_final["pressure"].min(),
    df_final["pressure"].max(),
)
