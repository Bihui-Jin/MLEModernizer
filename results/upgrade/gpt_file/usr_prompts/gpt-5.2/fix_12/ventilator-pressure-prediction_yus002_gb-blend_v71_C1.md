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

0.1575978808590274

# 6. Current score

4.01163

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.91814) has done: 'Your notebook fails because it tries to blend two external submission files that do not exist in this Kaggle environment (`../input/gb-blending/...`). To make it run end-to-end and produce a valid `.csv` submission, I remove that dependency and instead generate a baseline prediction directly from `train.csv` using a simple per-(R,C,time_step,u_in,u_out) median lookup with sensible fallbacks. This keeps the original “snap predictions to nearest known pressure value” post-processing unchanged, which is aligned with the competition’s discrete pressure levels. The result always write `submission.csv` with the required `id,pressure` columns.'
- What this solution (achieved 7.3178) has done: 'Your current score (9.91814, lower-is-better) is far worse than the target (0.1576), so we should make a small, legitimate improvement without changing the overall “lookup-from-train then snap to nearest known pressure” core logic. The biggest gain with minimal risk here is to ensure the fallback medians are computed on *inspiratory-only* rows (`u_out==0`), matching the evaluation phase and preventing expiratory pressures from biasing the lookup. We keep the exact same lookup structure and snapping, just change the data used to compute medians and global median. This should move the MAE substantially downward toward the target while keeping the pipeline simple and deterministic and still producing `submission.csv`.'
- What this solution (achieved 7.57131) has done: 'Your current score (7.3178, lower-is-better) is still far above the target (0.1576), so we should improve the lookup accuracy while keeping the same overall “median-from-train lookup + fallbacks + snap to nearest known pressure” logic. The smallest impactful change is to (1) add a stronger fallback that uses the breath context by including `time_step`-aligned cumulative volume (`u_in` integral) and (2) keep all medians computed on inspiratory-only rows to match the metric. We compute per-`(R,C,time_step,u_out,u_in_round)` medians (where `u_in` is lightly rounded to reduce sparsity), then fall back to `(R,C,time_step)` and then a new `(R,C,time_step,vol_bin)` median before the global median. This preserves the same style of deterministic table-lookup inference and the same final snapping to discrete pressure levels.'
- What this solution (achieved 7.57131) has done: 'Your current MAE (7.57, lower-is-better) is far above the target (0.1576), so we should improve the lookup accuracy while keeping the exact same “median lookup + fallbacks + snap-to-nearest-known-pressure” approach. The biggest issue is the cumulative-volume feature: it’s currently computed across *both* inspiratory and expiratory phases in test (and effectively across the whole breath in train), which misaligns the `cum_vol`/`vol_bin` meaning with the metric (only inspiratory is scored) and injects phase-dependent noise into the fallback tables. I minimally change `add_cumvol_fast` to compute inspiratory-only cumulative volume (resetting accumulation when `u_out==1`) and also compute the fallback medians for `vol_bin` using inspiratory-only `u_out==0` rows so the feature matches evaluation. Everything else (groupby medians, merge logic, fallback order, and final snapping) stays the same, and it still write a valid `submission.csv`.'
- What this solution (achieved 7.65229) has done: 'Your current score is much worse than the target (lower-is-better), so we should make a small but meaningful accuracy improvement without changing the overall “train median lookup + fallbacks + snap to nearest known pressure” approach. The biggest issue is that your lookup keys are too sparse because `time_step` is a float and must match exactly; we can keep the same logic but discretize `time_step` onto the known 80-step grid (per-breath index) to dramatically increase match rate. We add a `t_idx` (0..79) computed within each `breath_id` and use it instead of raw `time_step` in all groupby/merge keys, while keeping the same cum_vol feature, fallback order, and nearest-pressure snapping. This preserves evaluation semantics and determinism but should reduce MAE substantially toward the target.'
- What this solution (achieved 7.68663) has done: 'Your current MAE (7.65, lower-is-better) is still far from the target (0.1576), so we need a legitimate boost while keeping the same “median lookup from train + fallbacks + snap to nearest known pressure” core logic. The biggest remaining issue is that the lookup uses raw/rounded `u_in`, which is still too sparse; a minimal, metric-aligned fix is to add a breath-context fallback keyed by the *within-breath step index* and a lightly binned *cumulative inspiratory volume* (and optionally `u_in` bin) to increase match rate without changing the approach. We keep your existing primary key, keep inspiratory-only medians, keep `t_idx`, keep your cum_vol definition, and keep the same nearest-pressure snapping; we only add one stronger fallback layer and reorder fallbacks to use the more informative table first. This should move the score materially downward toward the target while staying deterministic and within Kaggle constraints.'
- What this solution (achieved 4.01248) has done: 'Your current MAE (7.68663, lower-is-better) is far above the target (0.1576), so we should improve match-rate and reduce bias while keeping the same “median lookup from train + fallbacks + snap to nearest known pressure” core logic. The main minimal fix is to compute the within-breath step index (`t_idx`) and the cumulative inspiratory volume (`cum_vol`) on the full train/test (not just inspiratory rows), then restrict to inspiratory rows only when building the median tables—this preserves your approach but prevents feature misalignment caused by missing expiratory steps in `df_train_insp`. Additionally, we ensure merges are performed once into a single test frame to avoid inconsistent row ordering and unnecessary copies, keeping prediction alignment with `id`. All post-processing (fallback order and nearest-pressure snapping) stays the same, and we still write a valid `submission.csv`.'
- What this solution (achieved 4.01169) has done: 'Your current MAE (4.01248, lower-is-better) is still far above the target (0.1576), so we should increase match-rate while keeping the same “median lookup from train + fallbacks + snap to nearest known pressure” core logic intact. The smallest high-impact change is to stop using a continuous `vol_bin` based on cumulative volume (which won’t generalize well), and instead add a strong, stable fallback keyed by the within-breath step index and discretized `u_in` (and optionally `(R,C)`), which aligns well with the simulator’s deterministic behavior. Concretely: keep the existing primary lookup `(R,C,t_idx,u_out,u_in_r)` unchanged, but replace the vol-based fallbacks with `(R,C,t_idx,u_in_r)`, then `(R,C,t_idx,u_in_bin)`, then `(t_idx,u_in_bin,u_out)`, then `(t_idx,u_out)`, then global median; snapping to nearest known pressure remains identical. This preserves architecture/training semantics (still pure table medians + fallbacks + snapping), is deterministic, and should move the score materially toward the target without risky model changes.'
- What this solution (achieved 4.01169) has done: 'Your current MAE (4.01169, lower-is-better) is still far above the target (0.1576), so we should make a small, safe change that increases lookup match-rate without changing your core “median lookup + fallbacks + snap to nearest known pressure” approach. The main issue is that your median tables are built only on inspiratory rows, but the primary key includes `u_out`; this makes the `u_out==1` test rows fall through to weak/incorrect fallbacks. We keep your same tables and fallbacks, but additionally build the same median tables on *all* phases (u_out 0/1) and only use them to fill predictions for `u_out==1` rows, while keeping inspiratory behavior unchanged. This should reduce overall MAE by improving predictions on the unscored expiratory phase without affecting the inspiratory-phase logic that drives most leaderboard performance.'
- What this solution (achieved 4.01169) has done: 'Your current MAE (4.01169; lower-is-better) is still far above the target (0.1576), so we need a small but meaningful accuracy gain while keeping the same “median lookup from train + fallbacks + snap-to-nearest-known-pressure” core logic. The biggest easy win is to reduce key sparsity by using the already-computed `cum_vol` as a *coarse bin* in an added fallback table, rather than relying only on `(t_idx, u_in)`-style fallbacks that ignore accumulated delivered volume. Concretely, we keep your primary key and existing fallbacks unchanged, and insert one additional fallback `(R,C,t_idx,vol_bin)` computed on inspiratory rows (and an all-phase version for `u_out==1`) to improve match-rate and stabilize predictions. The submission writing, column names, and nearest-pressure snapping remain identical.'
- What this solution (achieved 4.01163) has done: 'Your current score (4.01169, lower-is-better) is still far above the target (0.1576), so we should make a small, metric-aligned improvement without changing the core “median lookup + fallbacks + snap-to-nearest-known-pressure” approach. The biggest remaining weakness is that the median tables ignore the strong dependence on recent control history, so we add a minimal history feature: previous-step `u_in` (and its binned version) computed within each breath. We then insert one new fallback table keyed by `(R,C,t_idx,u_in_r,u_in_r_prev)` (and a coarser bin version) before the weaker generic fallbacks; everything else (primary key, other fallbacks, snapping, submission writing) stays the same. This improves match-rate and disambiguates similar `u_in` values that lead to different pressures depending on the immediate past, which should move MAE downward toward the target while preserving the same overall logic and runtime constraints. The script still runs end-to-end and writes a valid `submission.csv` with `id,pressure`.'

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
    for i in range(len(input_list)):
        public_lb_score = int(input_list[i].split("/")[-1].split(".")[1].split(" ")[0])
        l.append(public_lb_score)
        input_list[i] = (pd.read_csv(input_list[i]).pressure).ravel()
    output = 0
    l_sum = sum(l)
    if len(input_list) == 1:
        output = input_list[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def g(dp):
    l = []
    for i in glob.iglob(f"{dp}/*"):
        l.append(i)
    file_count = len(l)
    loop_time = 150
    splits = file_count // 2
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
    pred_list = []
    for i in range(loop_time):
        weight = []
        set_seed(i)
        for i in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        for i in range(len(weight)):
            weight[i] /= weight_sum
        weight.sort(reverse=True)
        temp = 0
        for i in range(len(flist)):
            temp += flist[i] * weight[i]
        pred_list.append(temp)
        del temp
        gc.collect()
    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    output.pressure = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.55 + b.pressure * 0.45
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 1
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")


def add_time_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["breath_id", "time_step"], kind="mergesort").copy()
    df["t_idx"] = df.groupby("breath_id", sort=False).cumcount().astype(np.int16)
    return df


def add_cumvol_fast(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["breath_id", "time_step"], kind="mergesort").copy()
    dt = df.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0).to_numpy()
    u_in = df["u_in"].to_numpy(dtype=np.float64, copy=False)
    u_out = df["u_out"].to_numpy(dtype=np.int8, copy=False)

    flow_dt = u_in * dt
    flow_dt[u_out == 1] = 0.0
    df["cum_vol"] = (
        pd.Series(flow_dt).groupby(df["breath_id"], sort=False).cumsum().to_numpy()
    )
    return df


def add_prev_uin(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["breath_id", "time_step"], kind="mergesort").copy()
    df["u_in_prev"] = (
        df.groupby("breath_id", sort=False)["u_in"]
        .shift(1)
        .fillna(0.0)
        .astype(np.float32)
    )
    return df


df_train_feat = add_time_index(df_train)
df_test_feat = add_time_index(df_test)

df_train_feat = add_cumvol_fast(df_train_feat)
df_test_feat = add_cumvol_fast(df_test_feat)

df_train_feat = add_prev_uin(df_train_feat)
df_test_feat = add_prev_uin(df_test_feat)

df_train_insp = df_train_feat[df_train_feat["u_out"] == 0].copy()

UIN_ROUND = (
    1.0  # keep same rounding strength as before (do not change core primary key)
)
df_train_insp["u_in_r"] = (df_train_insp["u_in"] / UIN_ROUND).round().astype(np.int16)
df_test_feat["u_in_r"] = (df_test_feat["u_in"] / UIN_ROUND).round().astype(np.int16)

df_train_insp["u_in_r_prev"] = (
    (df_train_insp["u_in_prev"] / UIN_ROUND).round().astype(np.int16)
)
df_test_feat["u_in_r_prev"] = (
    (df_test_feat["u_in_prev"] / UIN_ROUND).round().astype(np.int16)
)

UIN_BIN = 2.0  # fallback-only
df_train_insp["u_in_b"] = (df_train_insp["u_in"] / UIN_BIN).round().astype(np.int16)
df_test_feat["u_in_b"] = (df_test_feat["u_in"] / UIN_BIN).round().astype(np.int16)
df_train_insp["u_in_b_prev"] = (
    (df_train_insp["u_in_prev"] / UIN_BIN).round().astype(np.int16)
)
df_test_feat["u_in_b_prev"] = (
    (df_test_feat["u_in_prev"] / UIN_BIN).round().astype(np.int16)
)

df_train_all = df_train_feat.copy()
df_train_all["u_in_r"] = (df_train_all["u_in"] / UIN_ROUND).round().astype(np.int16)
df_train_all["u_in_b"] = (df_train_all["u_in"] / UIN_BIN).round().astype(np.int16)
df_train_all["u_in_r_prev"] = (
    (df_train_all["u_in_prev"] / UIN_ROUND).round().astype(np.int16)
)
df_train_all["u_in_b_prev"] = (
    (df_train_all["u_in_prev"] / UIN_BIN).round().astype(np.int16)
)

VOL_BIN = 0.2  # coarse bin to avoid over-sparsity while adding informative context
df_train_insp["vol_b"] = (df_train_insp["cum_vol"] / VOL_BIN).round().astype(np.int16)
df_test_feat["vol_b"] = (df_test_feat["cum_vol"] / VOL_BIN).round().astype(np.int16)
df_train_all["vol_b"] = (df_train_all["cum_vol"] / VOL_BIN).round().astype(np.int16)

key_cols_r = ["R", "C", "t_idx", "u_out", "u_in_r"]

train_key_median_r_insp = (
    df_train_insp.groupby(key_cols_r, observed=True)["pressure"].median().reset_index()
)

rc_time_uinr_prev_median_insp = (
    df_train_insp.groupby(["R", "C", "t_idx", "u_in_r", "u_in_r_prev"], observed=True)[
        "pressure"
    ]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pressure_rc_time_uinr_prev"})
)
rc_time_uinb_prev_median_insp = (
    df_train_insp.groupby(["R", "C", "t_idx", "u_in_b", "u_in_b_prev"], observed=True)[
        "pressure"
    ]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pressure_rc_time_uinb_prev"})
)

rc_time_uinr_median_insp = (
    df_train_insp.groupby(["R", "C", "t_idx", "u_in_r"], observed=True)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pressure_rc_time_uinr"})
)
rc_time_uinb_median_insp = (
    df_train_insp.groupby(["R", "C", "t_idx", "u_in_b"], observed=True)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pressure_rc_time_uinb"})
)
t_uinb_uout_median_insp = (
    df_train_insp.groupby(["t_idx", "u_in_b", "u_out"], observed=True)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pressure_t_uinb_uout"})
)
t_uout_median_insp = (
    df_train_insp.groupby(["t_idx", "u_out"], observed=True)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pressure_t_uout"})
)
rc_time_median_insp = (
    df_train_insp.groupby(["R", "C", "t_idx"], observed=True)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pressure_rc_time"})
)
rc_time_volb_median_insp = (
    df_train_insp.groupby(["R", "C", "t_idx", "vol_b"], observed=True)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pressure_rc_time_volb"})
)

train_key_median_r_all = (
    df_train_all.groupby(key_cols_r, observed=True)["pressure"].median().reset_index()
)

rc_time_uinr_prev_median_all = (
    df_train_all.groupby(["R", "C", "t_idx", "u_in_r", "u_in_r_prev"], observed=True)[
        "pressure"
    ]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pressure_rc_time_uinr_prev_all"})
)
rc_time_uinb_prev_median_all = (
    df_train_all.groupby(["R", "C", "t_idx", "u_in_b", "u_in_b_prev"], observed=True)[
        "pressure"
    ]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pressure_rc_time_uinb_prev_all"})
)

rc_time_uinr_median_all = (
    df_train_all.groupby(["R", "C", "t_idx", "u_in_r"], observed=True)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pressure_rc_time_uinr_all"})
)
rc_time_uinb_median_all = (
    df_train_all.groupby(["R", "C", "t_idx", "u_in_b"], observed=True)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pressure_rc_time_uinb_all"})
)
t_uinb_uout_median_all = (
    df_train_all.groupby(["t_idx", "u_in_b", "u_out"], observed=True)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pressure_t_uinb_uout_all"})
)
t_uout_median_all = (
    df_train_all.groupby(["t_idx", "u_out"], observed=True)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pressure_t_uout_all"})
)
rc_time_median_all = (
    df_train_all.groupby(["R", "C", "t_idx"], observed=True)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pressure_rc_time_all"})
)
rc_time_volb_median_all = (
    df_train_all.groupby(["R", "C", "t_idx", "vol_b"], observed=True)["pressure"]
    .median()
    .reset_index()
    .rename(columns={"pressure": "pressure_rc_time_volb_all"})
)

test_pred = df_test_feat[
    [
        "id",
        "R",
        "C",
        "t_idx",
        "u_out",
        "u_in_r",
        "u_in_b",
        "vol_b",
        "u_in_r_prev",
        "u_in_b_prev",
    ]
].copy()

test_pred = test_pred.merge(train_key_median_r_insp, on=key_cols_r, how="left")

test_pred = test_pred.merge(
    rc_time_uinr_prev_median_insp,
    on=["R", "C", "t_idx", "u_in_r", "u_in_r_prev"],
    how="left",
)
test_pred = test_pred.merge(
    rc_time_uinb_prev_median_insp,
    on=["R", "C", "t_idx", "u_in_b", "u_in_b_prev"],
    how="left",
)

test_pred = test_pred.merge(
    rc_time_uinr_median_insp, on=["R", "C", "t_idx", "u_in_r"], how="left"
)
test_pred = test_pred.merge(
    rc_time_uinb_median_insp, on=["R", "C", "t_idx", "u_in_b"], how="left"
)
test_pred = test_pred.merge(
    t_uinb_uout_median_insp, on=["t_idx", "u_in_b", "u_out"], how="left"
)
test_pred = test_pred.merge(t_uout_median_insp, on=["t_idx", "u_out"], how="left")
test_pred = test_pred.merge(rc_time_median_insp, on=["R", "C", "t_idx"], how="left")
test_pred = test_pred.merge(
    rc_time_volb_median_insp, on=["R", "C", "t_idx", "vol_b"], how="left"
)

test_pred = test_pred.merge(
    train_key_median_r_all.rename(columns={"pressure": "pressure_all"}),
    on=key_cols_r,
    how="left",
)

test_pred = test_pred.merge(
    rc_time_uinr_prev_median_all,
    on=["R", "C", "t_idx", "u_in_r", "u_in_r_prev"],
    how="left",
)
test_pred = test_pred.merge(
    rc_time_uinb_prev_median_all,
    on=["R", "C", "t_idx", "u_in_b", "u_in_b_prev"],
    how="left",
)

test_pred = test_pred.merge(
    rc_time_uinr_median_all, on=["R", "C", "t_idx", "u_in_r"], how="left"
)
test_pred = test_pred.merge(
    rc_time_uinb_median_all, on=["R", "C", "t_idx", "u_in_b"], how="left"
)
test_pred = test_pred.merge(
    t_uinb_uout_median_all, on=["t_idx", "u_in_b", "u_out"], how="left"
)
test_pred = test_pred.merge(t_uout_median_all, on=["t_idx", "u_out"], how="left")
test_pred = test_pred.merge(rc_time_median_all, on=["R", "C", "t_idx"], how="left")
test_pred = test_pred.merge(
    rc_time_volb_median_all, on=["R", "C", "t_idx", "vol_b"], how="left"
)

pred_insp = test_pred["pressure"].to_numpy(dtype=np.float64)

fallback_rc_time_uinr_prev = test_pred["pressure_rc_time_uinr_prev"].to_numpy(
    dtype=np.float64
)
fallback_rc_time_uinb_prev = test_pred["pressure_rc_time_uinb_prev"].to_numpy(
    dtype=np.float64
)

fallback_rc_time_uinr = test_pred["pressure_rc_time_uinr"].to_numpy(dtype=np.float64)
fallback_rc_time_uinb = test_pred["pressure_rc_time_uinb"].to_numpy(dtype=np.float64)
fallback_t_uinb_uout = test_pred["pressure_t_uinb_uout"].to_numpy(dtype=np.float64)
fallback_t_uout = test_pred["pressure_t_uout"].to_numpy(dtype=np.float64)
fallback_rc_time = test_pred["pressure_rc_time"].to_numpy(dtype=np.float64)
fallback_rc_time_volb = test_pred["pressure_rc_time_volb"].to_numpy(dtype=np.float64)

global_median_insp = float(df_train_insp["pressure"].median())

pred_insp = np.where(np.isnan(pred_insp), fallback_rc_time_uinr_prev, pred_insp)
pred_insp = np.where(np.isnan(pred_insp), fallback_rc_time_uinb_prev, pred_insp)
pred_insp = np.where(np.isnan(pred_insp), fallback_rc_time_uinr, pred_insp)
pred_insp = np.where(np.isnan(pred_insp), fallback_rc_time_uinb, pred_insp)
pred_insp = np.where(np.isnan(pred_insp), fallback_t_uinb_uout, pred_insp)
pred_insp = np.where(np.isnan(pred_insp), fallback_t_uout, pred_insp)
pred_insp = np.where(np.isnan(pred_insp), fallback_rc_time, pred_insp)
pred_insp = np.where(np.isnan(pred_insp), fallback_rc_time_volb, pred_insp)
pred_insp = np.where(np.isnan(pred_insp), global_median_insp, pred_insp)

pred_all = test_pred["pressure_all"].to_numpy(dtype=np.float64)

fallback_rc_time_uinr_prev_all = test_pred["pressure_rc_time_uinr_prev_all"].to_numpy(
    dtype=np.float64
)
fallback_rc_time_uinb_prev_all = test_pred["pressure_rc_time_uinb_prev_all"].to_numpy(
    dtype=np.float64
)

fallback_rc_time_uinr_all = test_pred["pressure_rc_time_uinr_all"].to_numpy(
    dtype=np.float64
)
fallback_rc_time_uinb_all = test_pred["pressure_rc_time_uinb_all"].to_numpy(
    dtype=np.float64
)
fallback_t_uinb_uout_all = test_pred["pressure_t_uinb_uout_all"].to_numpy(
    dtype=np.float64
)
fallback_t_uout_all = test_pred["pressure_t_uout_all"].to_numpy(dtype=np.float64)
fallback_rc_time_all = test_pred["pressure_rc_time_all"].to_numpy(dtype=np.float64)
fallback_rc_time_volb_all = test_pred["pressure_rc_time_volb_all"].to_numpy(
    dtype=np.float64
)

global_median_all = float(df_train_all["pressure"].median())

pred_all = np.where(np.isnan(pred_all), fallback_rc_time_uinr_prev_all, pred_all)
pred_all = np.where(np.isnan(pred_all), fallback_rc_time_uinb_prev_all, pred_all)
pred_all = np.where(np.isnan(pred_all), fallback_rc_time_uinr_all, pred_all)
pred_all = np.where(np.isnan(pred_all), fallback_rc_time_uinb_all, pred_all)
pred_all = np.where(np.isnan(pred_all), fallback_t_uinb_uout_all, pred_all)
pred_all = np.where(np.isnan(pred_all), fallback_t_uout_all, pred_all)
pred_all = np.where(np.isnan(pred_all), fallback_rc_time_all, pred_all)
pred_all = np.where(np.isnan(pred_all), fallback_rc_time_volb_all, pred_all)
pred_all = np.where(np.isnan(pred_all), global_median_all, pred_all)

u_out_arr = test_pred["u_out"].to_numpy(dtype=np.int8, copy=False)
pred = np.where(u_out_arr == 1, pred_all, pred_insp)

pred = np.vectorize(find_nearest, otypes=[np.float64])(pred)

sub = test_pred[["id"]].copy()
sub["pressure"] = pred.astype(np.float64)
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("NaN in predictions:", int(np.isnan(sub["pressure"].to_numpy()).sum()))
print("pressure min/max:", float(sub["pressure"].min()), float(sub["pressure"].max()))
