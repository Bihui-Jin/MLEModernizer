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

0.2014321665757602

# 6. Current score

8.00666

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.86378) has done: 'The notebook fails because it tries to blend two external submission files that don’t exist in your environment (`../input/gb-blending/...`), so it never produces a valid `.csv` output. I remove that dependency and instead generate a deterministic, valid baseline submission directly from the provided competition data. To keep changes minimal and preserve your existing pressure “snapping” logic, I reuse `find_nearest` and output a `submission.csv` with the required `id,pressure` columns. This run end-to-end on Kaggle and yield a non-empty submission file.'
- What this solution (achieved 9.0062) has done: 'Your current score is far from the target (lower is better), and the main issue is that the notebook outputs a constant prediction, which yields very high MAE. To move the score much closer to the target while keeping changes minimal and preserving your “snap-to-known-pressures” core logic, I replace the constant baseline with a lightweight, deterministic, per-(R,C,time_step,u_out) lookup model computed from train. This stays within your current approach (no ML libraries, no architecture/training loop changes) and still applies `find_nearest` to match valid pressure levels. I also ensure the submission `id` alignment matches `test.csv` exactly to avoid any ordering mistakes.'
- What this solution (achieved 9.26299) has done: 'Your current score (9.0062, lower-is-better) is far from the target (0.2014), so we should improve the predictive signal while keeping your “train-median lookup + fallback + snap-to-known-pressures” core logic intact. The biggest accuracy issue is that the lookup keys are too coarse (they ignore the sequential nature of breaths), so I add minimal, deterministic per-breath cumulative features (`u_in_cum`, `u_out_cum`, `dt`) and use them only as *additional* grouping keys with a safe fallback chain to your existing groups. This keeps the same modeling approach (groupby median lookups) but makes the lookup much closer to the true pressure dynamics, typically yielding a large MAE reduction on this competition. I also ensure strict `id` alignment by writing predictions in the exact test row order (no merge-induced reordering surprises) and still apply `find_nearest` snapping.'
- What this solution (achieved 8.63244) has done: 'Your current MAE (9.26299, lower-is-better) is still far from the target (0.2014), so we should improve signal while keeping your same “groupby-median lookup + fallback + snap-to-known-pressures” core logic. The most effective minimal change is to add one more deterministic per-breath dynamic key that captures sequence position (`step` within each breath), because pressure dynamics are strongly step-dependent and your current keys can still mix different phases. We use `step` only as an additional (highest-priority) lookup table with a safe fallback chain to your existing dynamic/full/coarser tables, so behavior remains stable even when the new key misses. We also keep strict `id` order by basing the submission on the original `df_test` row order exactly.'
- What this solution (achieved 8.64363) has done: 'Your current MAE (8.63, lower-is-better) is still far from the target (0.201), so we should meaningfully reduce error while keeping the same “groupby-median lookup + fallback chain + snap-to-known-pressures” core logic. The biggest accuracy miss is that the lookup ignores the strongest driver of pressure: instantaneous `u_in` (valve opening), so I add `u_in` (lightly rounded for join stability) into the highest-priority lookup keys, while keeping your existing tables as fallbacks. I also add an `inspiration`/`u_out==0`-only median variant for the top tables, because the metric only scores inspiratory rows and expiratory dynamics differ, but this still preserves your approach (just a different groupby subset). Finally, I keep strict test row order via `df_test` and still apply `find_nearest`, producing a valid `submission.csv`.'
- What this solution (achieved 7.64407) has done: 'Your current MAE (8.64363, lower-is-better) is still very far from the target (0.2014), so we should improve accuracy while keeping your same “groupby-median lookup + fallback chain + snap-to-known-pressures” approach. The main fix is to stop using floating `time_step` as a join key (it causes many missed matches due to float representation), and instead use the deterministic within-breath `step` index (0–79) as the primary time axis for all lookup tables. We keep your existing dynamic per-breath features and your fallback chain, but rebuild the coarser tables (and add a simple `R,C,step,u_out` table) so matches are much more frequent and stable. The output remains a valid `submission.csv` with `id,pressure` in the exact `test.csv` row order and still uses your `find_nearest` pressure snapping.'
- What this solution (achieved 7.64391) has done: 'Your current MAE (7.644, lower-is-better) is still far above the target (0.201), so we should improve accuracy with the smallest possible change while keeping the same “groupby median lookup + fallback chain + snap-to-known-pressures” logic. The biggest remaining miss is that pressure depends strongly on the *recent* control history, not just cumulative totals, so we add two minimal lag features (`u_in_lag1`, `u_out_lag1`) and use them only in a new highest-priority inspiratory lookup table, falling back to your existing tables when there’s no match. This preserves your approach (deterministic lookup medians) and should reduce MAE by increasing match specificity without changing any training loops/models. We also keep strict `test.csv` row order and continue snapping predictions to valid pressure levels.'
- What this solution (achieved 7.68417) has done: 'Your current score (7.64391 MAE; lower is better) is far above the target (0.2014), so we should improve predictive signal while keeping the same core “groupby-median lookup + fallback chain + snap-to-known-pressures” approach. The smallest high-impact fix is to stop using `u_out_cum` and `dt` as join keys (they create many near-unique combinations, causing sparse groups and noisy medians) and instead use more stable short-history features: `u_in` lags (1–3) and `u_in` rolling mean, only in the highest-priority inspiratory lookup table. We keep all your existing tables and fallback order, just inserting one stronger-but-still-deterministic lookup at the top and slightly simplifying the dynamic keys to increase match rate. This stays within your non-ML, deterministic lookup design and still writes a valid `submission.csv` with `id,pressure` aligned to `test.csv` row order.'
- What this solution (achieved 7.68417) has done: 'Your current MAE (7.684, lower-is-better) is far above the target, so we need a meaningful accuracy gain while preserving your existing “groupby-median lookup + fallback chain + snap-to-known-pressures” core logic. The biggest minimal fix is to stop using unstable float-like keys (`u_out_cum`, `dt_r`, and raw `u_out_lag1` as float) in your top tables, because they create sparse groups and many missed merges; instead we keep the same features but quantize them to small integer bins and use those binned versions in the lookup keys. This increases match rate and stabilizes medians without changing the approach (still deterministic lookups + fallbacks). I also make `u_out_cum` and `u_out_lag1` integer-typed consistently and add a very safe extra fallback table keyed on `(R,C,step,u_out,u_in_r)` to recover signal when history keys miss. Submission writing and `id` alignment remain identical.'
- What this solution (achieved 8.00666) has done: 'Your current MAE (7.684) is far above the target (0.201, lower-is-better), so we need a clear but still “same-logic” improvement to the deterministic groupby-lookup approach. The minimal high-impact fix here is to stop relying on sparsely-matching cumulative/dt-based keys in the *top* tables and instead add a stable, step-aligned short-history signature: a binned delta of `u_in` (first difference) plus a binned `u_in` level, which captures pressure dynamics much better while staying in the same “median lookup + fallback + snap-to-known-pressures” framework. I keep your existing feature engineering and all your existing lookup tables/fallback chain, only inserting one new (and one slightly coarser) lookup layer above the current chain to increase match rate and median quality. Submission writing remains strictly in `test.csv` row order with the required `id,pressure` columns and the same `find_nearest` snapping.'

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
    loop_time = 1131 // file_count
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




## === cell 3
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")


def add_breath_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")

    df["step"] = df.groupby("breath_id").cumcount().astype("int16")

    df["dt"] = df.groupby("breath_id")["time_step"].diff().fillna(0.0).astype("float32")
    df["u_in_cum"] = df.groupby("breath_id")["u_in"].cumsum().astype("float32")

    df["u_out_cum"] = df.groupby("breath_id")["u_out"].cumsum().astype("int16")

    df["u_in_lag1"] = (
        df.groupby("breath_id")["u_in"].shift(1).fillna(0.0).astype("float32")
    )
    df["u_out_lag1"] = (
        df.groupby("breath_id")["u_out"].shift(1).fillna(0).astype("int8")
    )

    df["u_in_lag2"] = (
        df.groupby("breath_id")["u_in"].shift(2).fillna(0.0).astype("float32")
    )
    df["u_in_lag3"] = (
        df.groupby("breath_id")["u_in"].shift(3).fillna(0.0).astype("float32")
    )
    df["u_in_roll3"] = (
        df.groupby("breath_id")["u_in"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .astype("float32")
    )
    return df


df_train_fe = add_breath_features(df_train)
df_test_fe = add_breath_features(df_test)

df_train_fe["u_in_r"] = df_train_fe["u_in"].round(1)
df_test_fe["u_in_r"] = df_test_fe["u_in"].round(1)

df_train_fe["u_in_cum_r"] = df_train_fe["u_in_cum"].round(1)
df_test_fe["u_in_cum_r"] = df_test_fe["u_in_cum"].round(1)

df_train_fe["dt_r"] = df_train_fe["dt"].round(2)
df_test_fe["dt_r"] = df_test_fe["dt"].round(2)

df_train_fe["u_in_lag1_r"] = df_train_fe["u_in_lag1"].round(1)
df_test_fe["u_in_lag1_r"] = df_test_fe["u_in_lag1"].round(1)

df_train_fe["u_in_lag2_r"] = df_train_fe["u_in_lag2"].round(1)
df_test_fe["u_in_lag2_r"] = df_test_fe["u_in_lag2"].round(1)
df_train_fe["u_in_lag3_r"] = df_train_fe["u_in_lag3"].round(1)
df_test_fe["u_in_lag3_r"] = df_test_fe["u_in_lag3"].round(1)
df_train_fe["u_in_roll3_r"] = df_train_fe["u_in_roll3"].round(1)
df_test_fe["u_in_roll3_r"] = df_test_fe["u_in_roll3"].round(1)

df_train_fe["dt_bin"] = (
    (df_train_fe["dt"] * 100).round().astype("int16")
)  # centiseconds
df_test_fe["dt_bin"] = (df_test_fe["dt"] * 100).round().astype("int16")

df_train_fe["u_in_cum_bin"] = (
    (df_train_fe["u_in_cum"] * 10).round().astype("int32")
)  # 0.1 resolution
df_test_fe["u_in_cum_bin"] = (df_test_fe["u_in_cum"] * 10).round().astype("int32")

df_train_fe["u_out_cum_bin"] = df_train_fe["u_out_cum"].astype("int16")
df_test_fe["u_out_cum_bin"] = df_test_fe["u_out_cum"].astype("int16")

df_train_fe["du_in"] = (df_train_fe["u_in"] - df_train_fe["u_in_lag1"]).astype(
    "float32"
)
df_test_fe["du_in"] = (df_test_fe["u_in"] - df_test_fe["u_in_lag1"]).astype("float32")
df_train_fe["du_in_bin"] = (
    (df_train_fe["du_in"] * 10).round().astype("int16")
)  # 0.1 resolution
df_test_fe["du_in_bin"] = (df_test_fe["du_in"] * 10).round().astype("int16")

df_train_fe["u_in_bin"] = (
    (df_train_fe["u_in"] * 10).round().astype("int16")
)  # 0.1 resolution
df_test_fe["u_in_bin"] = (df_test_fe["u_in"] * 10).round().astype("int16")

key_cols_step_full = ["R", "C", "step", "u_out"]
key_cols_rc_step = ["R", "C", "step"]
key_cols_step = ["step"]

key_cols_dyn_step = [
    "R",
    "C",
    "step",
    "u_out",
    "u_in_cum_bin",
    "u_out_cum_bin",
    "dt_bin",
]
key_cols_dyn_step_uin = [
    "R",
    "C",
    "step",
    "u_out",
    "u_in_r",
    "u_in_cum_bin",
    "u_out_cum_bin",
    "dt_bin",
]
key_cols_dyn_step_uin_lag = [
    "R",
    "C",
    "step",
    "u_out",
    "u_in_r",
    "u_in_lag1_r",
    "u_out_lag1",
    "u_in_cum_bin",
    "u_out_cum_bin",
    "dt_bin",
]

key_cols_insp_hist = [
    "R",
    "C",
    "step",
    "u_out",
    "u_in_r",
    "u_in_lag1_r",
    "u_in_lag2_r",
    "u_in_lag3_r",
    "u_in_roll3_r",
]
key_cols_rc_step_uout_uin = ["R", "C", "step", "u_out", "u_in_r"]

key_cols_insp_uin_du = ["R", "C", "step", "u_out", "u_in_bin", "du_in_bin"]
key_cols_insp_uin_only = ["R", "C", "step", "u_out", "u_in_bin"]

df_train_insp = df_train_fe[df_train_fe["u_out"] == 0]

med_insp_uin_du = (
    df_train_insp.groupby(key_cols_insp_uin_du, sort=False)["pressure"]
    .median()
    .rename("p_med_insp_uin_du")
    .reset_index()
)

med_insp_uin_only = (
    df_train_insp.groupby(key_cols_insp_uin_only, sort=False)["pressure"]
    .median()
    .rename("p_med_insp_uin_only")
    .reset_index()
)

med_insp_hist = (
    df_train_insp.groupby(key_cols_insp_hist, sort=False)["pressure"]
    .median()
    .rename("p_med_insp_hist")
    .reset_index()
)

med_dyn_step_uin_lag_insp = (
    df_train_insp.groupby(key_cols_dyn_step_uin_lag, sort=False)["pressure"]
    .median()
    .rename("p_med_dyn_step_uin_lag_insp")
    .reset_index()
)

med_dyn_step_uin_insp = (
    df_train_insp.groupby(key_cols_dyn_step_uin, sort=False)["pressure"]
    .median()
    .rename("p_med_dyn_step_uin_insp")
    .reset_index()
)

med_dyn_step_insp = (
    df_train_insp.groupby(key_cols_dyn_step, sort=False)["pressure"]
    .median()
    .rename("p_med_dyn_step_insp")
    .reset_index()
)

med_dyn_step = (
    df_train_fe.groupby(key_cols_dyn_step, sort=False)["pressure"]
    .median()
    .rename("p_med_dyn_step")
    .reset_index()
)

med_step_full = (
    df_train_fe.groupby(key_cols_step_full, sort=False)["pressure"]
    .median()
    .rename("p_med_step_full")
    .reset_index()
)

med_rc_step_uout_uin = (
    df_train_fe.groupby(key_cols_rc_step_uout_uin, sort=False)["pressure"]
    .median()
    .rename("p_med_rc_step_uout_uin")
    .reset_index()
)

med_rc_step = (
    df_train_fe.groupby(key_cols_rc_step, sort=False)["pressure"]
    .median()
    .rename("p_med_rc_step")
    .reset_index()
)

med_step = (
    df_train_fe.groupby(key_cols_step, sort=False)["pressure"]
    .median()
    .rename("p_med_step")
    .reset_index()
)

global_med = float(df_train_fe["pressure"].median())

test_pred = df_test_fe[
    [
        "id",
        "step",
        "u_in_r",
        "u_in_bin",
        "du_in_bin",
        "u_in_lag1_r",
        "u_in_lag2_r",
        "u_in_lag3_r",
        "u_in_roll3_r",
        "u_out_lag1",
        "R",
        "C",
        "u_out",
        "u_in_cum_bin",
        "u_out_cum_bin",
        "dt_bin",
    ]
].copy()

test_pred = test_pred.merge(med_insp_uin_du, on=key_cols_insp_uin_du, how="left")
test_pred = test_pred.merge(med_insp_uin_only, on=key_cols_insp_uin_only, how="left")

test_pred = test_pred.merge(med_insp_hist, on=key_cols_insp_hist, how="left")
test_pred = test_pred.merge(
    med_dyn_step_uin_lag_insp, on=key_cols_dyn_step_uin_lag, how="left"
)
test_pred = test_pred.merge(med_dyn_step_uin_insp, on=key_cols_dyn_step_uin, how="left")
test_pred = test_pred.merge(med_dyn_step_insp, on=key_cols_dyn_step, how="left")
test_pred = test_pred.merge(med_dyn_step, on=key_cols_dyn_step, how="left")
test_pred = test_pred.merge(med_step_full, on=key_cols_step_full, how="left")
test_pred = test_pred.merge(
    med_rc_step_uout_uin, on=key_cols_rc_step_uout_uin, how="left"
)
test_pred = test_pred.merge(med_rc_step, on=key_cols_rc_step, how="left")
test_pred = test_pred.merge(med_step, on=key_cols_step, how="left")

pred = test_pred["p_med_insp_uin_du"]
pred = pred.fillna(test_pred["p_med_insp_uin_only"])
pred = pred.fillna(test_pred["p_med_insp_hist"])
pred = pred.fillna(test_pred["p_med_dyn_step_uin_lag_insp"])
pred = pred.fillna(test_pred["p_med_dyn_step_uin_insp"])
pred = pred.fillna(test_pred["p_med_dyn_step_insp"])
pred = pred.fillna(test_pred["p_med_dyn_step"])
pred = pred.fillna(test_pred["p_med_step_full"])
pred = pred.fillna(test_pred["p_med_rc_step_uout_uin"])
pred = pred.fillna(test_pred["p_med_rc_step"])
pred = pred.fillna(test_pred["p_med_step"])
pred = pred.fillna(global_med)

pred = pred.astype(float).apply(find_nearest).astype(float)

sub_out = pd.DataFrame({"id": df_test["id"].values, "pressure": pred.values})
sub_out.to_csv("submission.csv", index=False)

print(sub_out.head())
print("Wrote submission.csv with shape:", sub_out.shape)
