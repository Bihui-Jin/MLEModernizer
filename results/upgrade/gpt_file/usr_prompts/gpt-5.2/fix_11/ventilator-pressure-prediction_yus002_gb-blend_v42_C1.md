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

0.1535684900142998

# 6. Current score

3.39903

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.86378) has done: 'I fix the immediate runtime error by removing the dependency on non-existent external blend files and instead produce a valid submission directly from the provided competition data. To preserve the original core intent (pressure value “snapping” to known train pressure levels), I keep and reuse your `find_nearest()` logic. Since no model is present in the current script, I generate a simple deterministic baseline prediction (median train pressure) and then apply `find_nearest()` for valid pressure levels; this run end-to-end and write `submission.csv` with the required `id,pressure` columns. I also correct the cell numbering to be sequential so the notebook/script format is consistent.'
- What this solution (achieved 3.51595) has done: 'Your current score is far worse than the target (lower-is-better), so the smallest legitimate improvement is to replace the constant-median baseline with a simple physics-inspired baseline that uses the known relationship between pressure and integrated flow. I keep your existing `find_nearest()` “snap to valid pressure levels” logic (core intent preserved) but compute predictions per-breath using cumulative `u_in` and a per-(R,C) linear calibration fit on the training inspiratory phase (`u_out==0`), which directly targets the MAE metric. This avoids adding any new ML libraries, keeps runtime reasonable, and still produces a valid `submission.csv` with the required `id,pressure` columns and correct row alignment.'
- What this solution (achieved 3.35435) has done: 'Your current MAE (3.51595, lower-is-better) is far from the target, so we need a small but meaningful improvement without changing the overall “cum_u_in linear fit per (R,C) + snap to valid pressures” core logic. The largest issue is that the current linear fit ignores `time_step` and any direct `u_out` handling at prediction time, which can heavily skew pressures; we minimally extend the same linear regression to use `cum_u_in_dt = cumsum(u_in * dt)` (a better proxy for delivered volume) and we force expiratory predictions (`u_out==1`) to a stable baseline learned from training expiratory rows per (R,C). Finally, to better match the scoring (only inspiratory is scored), we fit the inspiratory regression exactly as before but on the improved integrated feature, keeping the same snapping behavior and output format.'
- What this solution (achieved 3.30375) has done: 'Your current MAE (3.35435, lower-is-better) is still far from the target, so we make the smallest change that improves correctness without changing the overall “per-(R,C) linear fit on an integrated u_in feature + expiratory baseline + snap-to-known-pressures” core logic. The main issue is that you are integrating `u_in * dt` with `dt=0` at the first step of each breath, which breaks the physical proxy early in the sequence; we instead compute `dt` from the per-breath `time_step` differences but set the first `dt` to the first `time_step` (≈0.0/0.03) and clip any negative `dt` to 0. We also add a very small, stable second feature `u_in` into the same closed-form regression (still linear regression, same training approach), which typically reduces MAE materially while preserving the same modeling intent and staying within runtime. Submission writing is kept identical (`id,pressure`) and we keep your `find_nearest()` snapping unchanged.'
- What this solution (achieved 3.25718) has done: 'Your MAE is still far above the target (lower-is-better), so we need a small but meaningful improvement without changing the overall “per-(R,C) linear fit on integrated u_in + expiratory baseline + snap-to-known-pressures” core logic. The biggest correctness issue left is that the regression is being fit to **all inspiratory rows**, but the metric only scores a subset of inspiratory timesteps; we can better match the evaluation by fitting the same regression only on the **inspiratory (u_out==0) AND early phase (time_step <= 1.0s)** where pressure dynamics are most informative for the scored region. To reduce bias at prediction time without changing the model class, we also learn a simple per-(R,C) **intercept correction** from training residuals (still linear, just an additive calibration), then apply snapping exactly as before. Submission writing is kept identical and still produces `submission.csv` with `id,pressure`.'
- What this solution (achieved 3.25718) has done: 'Your current MAE (3.25718, lower-is-better) is still far above the target (0.1536), so we should make small, legitimate changes that better match the metric without changing your overall approach (per-(R,C) linear regression on integrated u_in features + expiratory baseline + snapping to known pressure levels). The biggest issue is training on only `time_step<=1.0`, which discards most inspiratory dynamics and causes a large bias; we fit on the full inspiratory phase (`u_out==0`) while keeping the same 2-feature linear closed-form regression. To better align with the scoring (only inspiratory is scored), we keep expiratory handling but make it deterministic and stable by using the last inspiratory prediction carried forward within each breath (instead of a global median-like baseline), which typically reduces discontinuities without changing model class. Finally, we fix submission row alignment to exactly match `sample_submission` order by mapping predictions by `id` (avoids any merge/order pitfalls).'
- What this solution (achieved 3.39911) has done: 'We keep your exact modeling approach (per-(R,C) closed-form linear regression on `cum_u_in_dt` and `u_in`, expiratory handling, and snapping to known pressure levels) but fix two high-impact correctness issues that likely drive the large MAE. First, we compute `dt` robustly by setting the first step `dt` to the *median positive* within-breath timestep (instead of ≈0), which makes `cum_u_in_dt` physically meaningful from the start of each breath. Second, we replace the slow Python loop snapping with a vectorized `np.searchsorted` snapper (same semantics) to stay within the 600s budget while allowing the full pipeline to run reliably; predictions and submission format remain unchanged.'
- What this solution (achieved 3.39911) has done: 'Your MAE (3.39911, lower-is-better) is far above the target (0.1536), so we need a legitimate improvement while keeping the same core approach: per-(R,C) closed-form linear regression on `cum_u_in_dt` and `u_in`, expiratory handling, and snapping to known pressure levels. The biggest minimal gain available without changing model class is to fit and predict **only on the inspiratory phase (u_out==0)**, because the competition metric ignores expiratory timesteps; then for `u_out==1` we carry-forward the last inspiratory prediction (same as you already do) but we **not let expiratory rows influence the regression calibration** at all. Additionally, we make the `dt` computation strictly per-breath and stable by using the within-breath median positive `dt` as the first-step fallback (same intent as you have), but we also ensure we don’t leak cross-breath diffs by sorting within each breath before diff/cumsum. These are minimal correctness-aligned changes that should reduce MAE meaningfully while preserving your overall logic and submission semantics.'
- What this solution (achieved 3.39903) has done: 'Your current MAE (3.39911, lower-is-better) is still far from the target (0.1536), so we should improve the *same* per-(R,C) closed-form linear regression core by fixing a major feature-misalignment: your integrated feature `cum_u_in_dt` depends heavily on `dt`, but `dt` is being imputed with a single global median that can differ across (R,C) and across breaths. I keep the exact model form (same 2 features, same closed-form solve, same snapping) but compute a **per-(R,C) median positive dt fallback from train**, and at inference pick the correct fallback for each row via merge; this is a minimal change that makes `cum_u_in_dt` much more consistent between train/test. I also make the integration strictly per-breath with a safer first-step `dt` (fill using the per-group fallback, else global), without changing any training loop/architecture (still no ML library). Submission writing stays `id,pressure` and aligned to `sample_submission`.'
- What this solution (achieved 3.39903) has done: 'We keep your exact modeling family (per-(R,C) closed-form linear regression on `cum_u_in_dt` and `u_in`, expiratory carry-forward handling, and snapping to known pressure levels) but fix a major alignment bug: your test/train `id` values are not globally unique, so mapping predictions by `id` collapses many rows and produces largely wrong submissions (driving MAE up). The minimal, metric-aligned fix is to write predictions in the same row order as `sample_submission` by predicting in test row order and assigning directly (no `id` map). To preserve robustness, we also enforce that `test` is sorted back to its original row order after feature construction (so the direct assignment is correct) while leaving all regression logic unchanged. This should materially reduce MAE toward the target without changing your core approach.'

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


def snap_to_known_pressures(pred_array: np.ndarray) -> np.ndarray:
    x = np.asarray(pred_array, dtype=np.float64)
    idx = np.searchsorted(sorted_pressures, x, side="left")

    idx0 = np.clip(idx, 0, total_pressures_len - 1)
    idxm1 = np.clip(idx - 1, 0, total_pressures_len - 1)

    upper = sorted_pressures[idx0]
    lower = sorted_pressures[idxm1]

    choose_lower = np.abs(lower - x) < np.abs(upper - x)
    out = np.where(
        idx == 0,
        sorted_pressures[0],
        np.where(
            idx == total_pressures_len,
            sorted_pressures[-1],
            np.where(choose_lower, lower, upper),
        ),
    )
    return out.astype(np.float64, copy=False)




## === cell 2
def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list):
    l = []
    for i in range(len(input_list)):
        try:
            public_lb_score = int(
                input_list[i].split("/")[-1].split(".")[1].split(" ")[0]
            )
        except Exception:
            public_lb_score = 0
        l.append(public_lb_score)
        input_list[i] = (pd.read_csv(input_list[i]).pressure).to_numpy().ravel()

    if len(input_list) == 1:
        return input_list[0]
    else:
        weight1 = 0.6
        weight2 = 0.4
        return input_list[0] * weight1 + input_list[1] * weight2


def g(dp):
    l = [i for i in glob.iglob(f"{dp}/*")]
    file_count = len(l)
    if file_count == 0:
        raise FileNotFoundError(f"No files found to blend in directory: {dp}")

    loop_time = file_count**3
    splits = max(1, file_count // 2)
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
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output["pressure"] = 0.0

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
            output["pressure"] += flist[j] * weight[j]

    output["pressure"] /= loop_time
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb_{loop_time}_loops.csv", index=False)
    return output


def blend(a, b, out_path="blend.csv"):
    a_df = pd.read_csv(a)
    b_df = pd.read_csv(b)

    if "pressure" not in a_df.columns or "pressure" not in b_df.columns:
        raise ValueError("Both input files must contain a 'pressure' column.")
    if len(a_df) != len(b_df):
        raise ValueError("Blend inputs must have the same number of rows.")

    a_df["pressure"] = a_df["pressure"] * 0.6 + b_df["pressure"] * 0.4
    a_df["pressure"] = a_df["pressure"].apply(find_nearest)
    a_df.to_csv(out_path, index=False)
    return a_df




## === cell 3
def compute_dt_fallbacks(train_df: pd.DataFrame):
    t = train_df[["breath_id", "R", "C", "time_step"]].copy()
    t = t.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
        drop=True
    )
    t["dt_raw"] = t.groupby("breath_id", sort=False)["time_step"].diff()

    pos = t.loc[(t["dt_raw"] > 0) & np.isfinite(t["dt_raw"]), ["R", "C", "dt_raw"]]
    rc_dt = (
        pos.groupby(["R", "C"], sort=False)["dt_raw"]
        .median()
        .rename("first_dt_fallback")
        .reset_index()
    )
    global_dt = float(pos["dt_raw"].median()) if len(pos) else 0.03
    return rc_dt, global_dt


def add_dt_and_integral_with_fallbacks(
    df: pd.DataFrame, rc_dt: pd.DataFrame, global_dt: float
):
    df = df.copy()
    df = df.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
        drop=True
    )

    df = df.merge(rc_dt, on=["R", "C"], how="left")
    df["first_dt_fallback"] = (
        df["first_dt_fallback"].fillna(global_dt).astype(np.float64)
    )

    g_ts = df.groupby("breath_id", sort=False)["time_step"]
    dt = g_ts.diff()

    dt = dt.fillna(df["first_dt_fallback"])
    dt = dt.clip(lower=0.0).astype(np.float64)

    df["dt"] = dt
    df["cum_u_in_dt"] = (
        (df["u_in"].astype(np.float64) * df["dt"])
        .groupby(df["breath_id"], sort=False)
        .cumsum()
    )
    return df


def fit_group_linear_2f(train_insp: pd.DataFrame):
    grp = train_insp.groupby(["R", "C"], sort=False)

    def _fit_one(gdf):
        x1 = gdf["cum_u_in_dt"].to_numpy(np.float64)
        x2 = gdf["u_in"].to_numpy(np.float64)
        y = gdf["pressure"].to_numpy(np.float64)
        n = len(y)
        if n < 3:
            return pd.Series({"b0": np.nan, "b1": np.nan, "b2": np.nan})

        s1 = x1.sum()
        s2 = x2.sum()
        sy = y.sum()
        s11 = np.dot(x1, x1)
        s22 = np.dot(x2, x2)
        s12 = np.dot(x1, x2)
        s1y = np.dot(x1, y)
        s2y = np.dot(x2, y)

        XtX = np.array([[n, s1, s2], [s1, s11, s12], [s2, s12, s22]], dtype=np.float64)
        Xty = np.array([sy, s1y, s2y], dtype=np.float64)

        try:
            b = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            return pd.Series({"b0": np.nan, "b1": np.nan, "b2": np.nan})

        return pd.Series({"b0": b[0], "b1": b[1], "b2": b[2]})

    coef2 = grp.apply(_fit_one).reset_index()
    return coef2


rc_dt, global_dt = compute_dt_fallbacks(df_train)

train = add_dt_and_integral_with_fallbacks(df_train, rc_dt=rc_dt, global_dt=global_dt)

train_insp = train.loc[
    (train["u_out"] == 0),
    ["breath_id", "R", "C", "cum_u_in_dt", "u_in", "pressure", "time_step"],
].copy()

coef2 = fit_group_linear_2f(train_insp)

x1 = train_insp["cum_u_in_dt"].to_numpy(np.float64)
x2 = train_insp["u_in"].to_numpy(np.float64)
y = train_insp["pressure"].to_numpy(np.float64)
n = len(y)
if n >= 3:
    s1 = x1.sum()
    s2 = x2.sum()
    sy = y.sum()
    s11 = np.dot(x1, x1)
    s22 = np.dot(x2, x2)
    s12 = np.dot(x1, x2)
    s1y = np.dot(x1, y)
    s2y = np.dot(x2, y)
    XtX = np.array([[n, s1, s2], [s1, s11, s12], [s2, s12, s22]], dtype=np.float64)
    Xty = np.array([sy, s1y, s2y], dtype=np.float64)
    try:
        b = np.linalg.solve(XtX, Xty)
        global_b0, global_b1, global_b2 = float(b[0]), float(b[1]), float(b[2])
    except np.linalg.LinAlgError:
        global_b0 = float(train_insp["pressure"].mean())
        global_b1 = 0.0
        global_b2 = 0.0
else:
    global_b0 = float(train_insp["pressure"].mean())
    global_b1 = 0.0
    global_b2 = 0.0

train_exp = train.loc[train["u_out"] == 1, ["R", "C", "pressure"]].copy()
exp_base = (
    train_exp.groupby(["R", "C"])["pressure"].median().rename("exp_base").reset_index()
)
global_exp_base = (
    float(train_exp["pressure"].median())
    if len(train_exp)
    else float(df_train["pressure"].median())
)

train_insp_pred = train_insp.merge(coef2, on=["R", "C"], how="left")
train_insp_pred["b0"] = train_insp_pred["b0"].fillna(global_b0)
train_insp_pred["b1"] = train_insp_pred["b1"].fillna(global_b1)
train_insp_pred["b2"] = train_insp_pred["b2"].fillna(global_b2)
pred0 = (
    train_insp_pred["b0"].to_numpy(np.float64)
    + train_insp_pred["b1"].to_numpy(np.float64)
    * train_insp_pred["cum_u_in_dt"].to_numpy(np.float64)
    + train_insp_pred["b2"].to_numpy(np.float64)
    * train_insp_pred["u_in"].to_numpy(np.float64)
)
train_insp_pred["resid"] = train_insp_pred["pressure"].to_numpy(np.float64) - pred0
delta_b0 = (
    train_insp_pred.groupby(["R", "C"])["resid"]
    .median()
    .rename("delta_b0")
    .reset_index()
)
global_delta_b0 = (
    float(train_insp_pred["resid"].median()) if len(train_insp_pred) else 0.0
)

test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")

test["__row_id__"] = np.arange(len(test), dtype=np.int64)

test = add_dt_and_integral_with_fallbacks(test, rc_dt=rc_dt, global_dt=global_dt)

test = test.merge(coef2, on=["R", "C"], how="left")
test["b0"] = test["b0"].fillna(global_b0)
test["b1"] = test["b1"].fillna(global_b1)
test["b2"] = test["b2"].fillna(global_b2)

test = test.merge(delta_b0, on=["R", "C"], how="left")
test["delta_b0"] = test["delta_b0"].fillna(global_delta_b0)

test = test.merge(exp_base, on=["R", "C"], how="left")
test["exp_base"] = test["exp_base"].fillna(global_exp_base)

pred_insp = (
    (test["b0"].to_numpy(np.float64) + test["delta_b0"].to_numpy(np.float64))
    + test["b1"].to_numpy(np.float64) * test["cum_u_in_dt"].to_numpy(np.float64)
    + test["b2"].to_numpy(np.float64) * test["u_in"].to_numpy(np.float64)
)

pred_series = pd.Series(pred_insp, index=test.index)

last_insp_pred = (
    pred_series.where(test["u_out"].to_numpy() == 0)
    .groupby(test["breath_id"], sort=False)
    .ffill()
)
last_insp_np = last_insp_pred.to_numpy(dtype=np.float64, copy=False)
pred = np.where(
    test["u_out"].to_numpy() == 1,
    np.where(
        last_insp_np == last_insp_np,  # NaN-safe check
        last_insp_np,
        test["exp_base"].to_numpy(dtype=np.float64, copy=False),
    ),
    pred_insp,
)

pred_snapped = snap_to_known_pressures(pred)

test["pred"] = pred_snapped
test = test.sort_values("__row_id__", kind="mergesort").reset_index(drop=True)

sub = sub.copy()
sub["pressure"] = test["pred"].to_numpy(np.float64, copy=False)
sub = sub[["id", "pressure"]]
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Using global_dt (median positive dt from train):", global_dt)
print("Global fallback coefficients (b0,b1,b2):", global_b0, global_b1, global_b2)
print("Global intercept correction delta_b0:", float(global_delta_b0))
print("Global expiratory baseline:", float(global_exp_base))
