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

0.1377192765167948

# 6. Current score

7.83475

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.13625) has done: 'Your script fails because it tries to blend two external submission files that do not exist in this Kaggle environment. I keep your core “blend + pressure snapping to nearest train pressure” logic, but make it self-contained by generating a simple model-based prediction from the provided train/test data instead of reading missing files. This produce a valid `submission.csv` with the correct columns and should achieve a reasonable MAE (and thus move toward the target) without changing the evaluation semantics. I also fix the cell numbering to be sequential and ensure all file paths point to the existing `../input/ventilator-pressure-prediction/` dataset.'
- What this solution (achieved 7.12981) has done: 'Your current score (8.13625 MAE) is far from the target (0.1377), so we need a real modeling improvement, but we can still keep your overall “train-based lookup prediction + snap to nearest train pressure + write submission.csv” core logic. The biggest issue is that your aggregation uses `u_in` and exact `time_step` matches, which are too sparse/unstable for test, leading to many fallbacks and poor MAE; we make the lookup much denser by (1) adding simple, deterministic time-series features (lagged `u_in`, cumulative sums) and (2) binning `time_step` to a fixed grid before grouping/merging. We also explicitly set `pressure=0` when `u_out==1` (expiratory phase not scored, but this improves overall robustness and usually helps the public score too). These are minimal changes that preserve your non-ML “groupby mean + fallback + snapping” approach and should move the MAE substantially toward the target band.'
- What this solution (achieved 8.19587) has done: 'Your MAE is still extremely far from the target, so we need a legitimate but minimal modeling-strength increase without changing your core “train-groupby lookup → hierarchical fallback → snap to nearest train pressure → write submission.csv” approach. The biggest win with minimal semantic change is to make the lookup keys much more “matchable” between train and test by (1) quantizing `u_in` (and its lags/cumsums) and (2) using a fixed integer time index per breath (0–79) instead of floating `time_step` rounding, which is prone to merge misses. I keep your hierarchical fallback structure but rebuild the groupby tables on these more stable keys, which should reduce NaNs and move MAE substantially toward the target. I also keep the `u_out==1 -> 0` rule and snapping.'
- What this solution (achieved 8.28162) has done: 'Your current MAE (8.19587) is still far from the target (0.1377, lower is better), so we need a clear but minimal improvement that keeps your same “train groupby lookup → hierarchical fallback → snap to nearest train pressure → write submission.csv” core logic. The biggest issue is that including `u_in_cum_q` creates extremely high-cardinality keys, causing many test rows to miss in the merge and fall back to coarse/global means; we keep the same structure but make the cumulative feature *matchable* by compressing it to a per-breath normalized cumulative percentage (and quantizing it). We also add one extra very-cheap deterministic feature (`u_in_diff_q`) to help separate dynamics without changing the modeling approach. These changes should reduce NaNs/over-fallback and move MAE substantially toward the target while preserving your evaluation semantics and submission format.'
- What this solution (achieved 8.34772) has done: 'Your score is far worse than the target (lower-is-better), and the main cause in your current lookup approach is massive merge miss/fallback due to overly high-cardinality keys (especially `u_in_cum_pct_q` and multiple lag/diff terms). I keep your exact “groupby-mean lookup → hierarchical fallback → set u_out==1 to 0 → snap to nearest train pressure → write submission.csv” core logic, but make the first lookup tables more matchable by (1) coarsening/clip-binning the cumulative-percentage feature and (2) adding a cheap intermediate fallback that uses `u_in` bins instead of the cumulative feature. This should reduce NaNs early in the cascade and move MAE substantially toward the target without changing the overall method. I also ensure merge keys use small integer dtypes consistently to avoid subtle join mismatches.'
- What this solution (achieved 8.35923) has done: 'Your current MAE is far above the target, so we need a small but meaningful fix that improves match rates in your existing “train groupby lookup → hierarchical fallback → u_out rule → snap → submission.csv” pipeline. The biggest low-risk issue is that you compute `u_in_bin` but never actually merge it into `pred_df` before using it as a key in `agg_cols2b`, which causes that whole intermediate fallback to fail or misbehave; we merge it explicitly. Next, we add one extra *coarser* intermediate fallback keyed only on `u_in_bin` (no lags/cum features) to catch many remaining NaNs without changing the core approach. These minimal changes should reduce fallback-to-global-mean and move MAE substantially closer to your target while keeping the same overall logic and evaluation semantics.'
- What this solution (achieved 8.35923) has done: 'Your current MAE (8.359) is still far above the target (0.138, lower is better), and the biggest issue is that many test rows likely miss the higher-fidelity lookup keys and fall back to coarse/global means. I keep your exact “train groupby lookup → hierarchical fallback → u_out rule → snap to nearest train pressure → submission.csv” core logic, but add two very cheap, deterministic, low-cardinality features (`u_in_mean_bin` and `u_in_last_bin`) that are stable per breath and help the lookup match better without changing the approach. Then I insert two new intermediate fallback tables using these per-breath bins (still groupby-mean lookups) before your very coarse `pred6/global_mean` steps, reducing harmful fallbacks. This should legitimately improve match rates and move MAE toward your target while staying within Kaggle constraints and preserving evaluation semantics.'
- What this solution (achieved 3.17699) has done: 'Your MAE is still far worse than the target (lower is better), and the biggest low-risk gain within your same “groupby lookup → hierarchical fallback → u_out rule → snap → submission.csv” logic is to (1) stop forcing `pressure=0` for `u_out==1` (those rows are not scored, and zeroing can hurt the public/private score if Kaggle includes them in display or if your assumption mismatches), and (2) make the last-stage fallback more accurate by using a per-(R,C,t_idx) mean (ignoring `u_out`) before falling back to the global mean. These are minimal, deterministic changes that don’t alter your overall approach but should reduce harmful fallbacks and move MAE toward the target. I also ensure `id` alignment is preserved by building submission directly from `test_f[['id']]` order and not from `sample_submission` order assumptions.'
- What this solution (achieved 8.09945) has done: 'I fix the KeyError by ensuring `pred_df` contains all merge key columns (including `t_idx`, `R`, and `C`) before any merges, and by removing duplicate/ambiguous columns that can appear after merges. I keep your exact groupby-mean lookup + hierarchical fallback + snapping logic intact, only adjusting the data assembly so merges work reliably. I also add a lightweight alignment assertion to guarantee `id` order consistency before writing `submission.csv`, preventing silent misalignment bugs. These changes are score-neutral in intent (they mainly unblock execution and preserve your existing semantics).'
- What this solution (achieved 7.83475) has done: 'Your current MAE is still far above the target (lower is better), so we should make the smallest change that improves the lookup quality without changing the overall “train groupby lookup → hierarchical fallback → snap to nearest train pressure → write submission.csv” core logic. The most direct issue is that your keys include `u_out_cum`, which is not a causal/stable driver of pressure and creates many unnecessary train/test key mismatches, forcing coarse fallbacks and inflating MAE. I keep the same cascade, but replace the top-level table to condition on `u_out` (current valve state) instead of `u_out_cum`, and I also add a cheap intermediate fallback keyed on `u_out` so inspiratory vs expiratory states don’t get mixed in the means. Everything else (feature extraction, hierarchical fillna cascade, expiratory masking/ffill behavior, snapping to nearest train pressure, and submission writing) remains intact.'

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


def find_nearest(prediction: float) -> float:
    insert_idx = np.searchsorted(sorted_pressures, prediction)
    if insert_idx == total_pressures_len:
        return float(sorted_pressures[-1])
    elif insert_idx == 0:
        return float(sorted_pressures[0])
    lower_val = sorted_pressures[insert_idx - 1]
    upper_val = sorted_pressures[insert_idx]
    return float(
        lower_val
        if abs(lower_val - prediction) < abs(upper_val - prediction)
        else upper_val
    )


def set_seed(seed: int = 2021):
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
    loop_time = 154
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
    a.pressure = a.pressure * 0.6 + b.pressure * 0.4
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.sort_values(["breath_id", "time_step"], inplace=True)
    df["t_idx"] = df.groupby("breath_id").cumcount().astype(np.int16)

    df["u_in_q"] = (df["u_in"] * 10).round().astype(np.int16)  # 0..1000
    df["u_in_bin"] = (df["u_in"] * 2).round().astype(np.int16)  # 0.5 step -> 0..200

    df["u_in_lag1_q"] = (
        df.groupby("breath_id")["u_in_q"].shift(1).fillna(0).astype(np.int16)
    )
    df["u_in_lag2_q"] = (
        df.groupby("breath_id")["u_in_q"].shift(2).fillna(0).astype(np.int16)
    )

    df["u_in_diff_q"] = (df["u_in_q"] - df["u_in_lag1_q"]).astype(np.int16)

    u_in_cum = df.groupby("breath_id")["u_in_q"].cumsum().astype(np.int32)
    u_in_sum = df.groupby("breath_id")["u_in_q"].transform("sum").astype(np.int32)
    denom = u_in_sum.to_numpy()
    denom = np.where(denom == 0, 1, denom)
    u_in_cum_pct = np.round(u_in_cum.to_numpy() * 1000.0 / denom).astype(np.int16)

    u_in_cum_pct = np.clip(u_in_cum_pct, 0, 1000)
    df["u_in_cum_pct_q"] = u_in_cum_pct
    df["u_in_cum_pct_bin"] = ((u_in_cum_pct // 20) * 20).astype(np.int16)

    df["u_out_cum"] = df.groupby("breath_id")["u_out"].cumsum().astype(np.int16)

    u_in_mean = df.groupby("breath_id")["u_in"].transform("mean").to_numpy()
    df["u_in_mean_bin"] = (np.round(u_in_mean * 2.0)).astype(np.int16)  # 0..200

    u_in_last = df.groupby("breath_id")["u_in"].transform("last").to_numpy()
    df["u_in_last_bin"] = (np.round(u_in_last * 2.0)).astype(np.int16)  # 0..200

    df["R"] = df["R"].astype(np.int16)
    df["C"] = df["C"].astype(np.int16)
    df["u_out"] = df["u_out"].astype(np.int8)

    return df


train_f = add_features(df_train)
test_f = add_features(df_test)

pred_df = test_f[
    [
        "id",
        "breath_id",
        "R",
        "C",
        "t_idx",
        "u_out",
        "u_in_q",
        "u_in_bin",
        "u_in_lag1_q",
        "u_in_lag2_q",
        "u_in_diff_q",
        "u_in_cum_pct_bin",
        "u_out_cum",
        "u_in_mean_bin",
        "u_in_last_bin",
    ]
].copy()

agg_cols = [
    "R",
    "C",
    "t_idx",
    "u_out",
    "u_in_lag1_q",
    "u_in_lag2_q",
    "u_in_diff_q",
    "u_in_cum_pct_bin",
]
train_stats = (
    train_f.groupby(agg_cols, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pred"})
)
pred_df = pred_df.merge(train_stats, on=agg_cols, how="left")

agg_cols2 = [
    "R",
    "C",
    "t_idx",
    "u_in_lag1_q",
    "u_in_lag2_q",
    "u_in_diff_q",
    "u_in_cum_pct_bin",
]
train_stats2 = (
    train_f.groupby(agg_cols2, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pred2"})
)
pred_df = pred_df.merge(train_stats2, on=agg_cols2, how="left")

agg_cols2u = ["R", "C", "t_idx", "u_out", "u_in_bin"]
train_stats2u = (
    train_f.groupby(agg_cols2u, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pred2u"})
)
pred_df = pred_df.merge(train_stats2u, on=agg_cols2u, how="left")

agg_cols2b = ["R", "C", "t_idx", "u_in_bin", "u_in_lag1_q", "u_in_diff_q"]
train_stats2b = (
    train_f.groupby(agg_cols2b, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pred2b"})
)
pred_df = pred_df.merge(train_stats2b, on=agg_cols2b, how="left")

agg_cols2c = ["R", "C", "t_idx", "u_in_bin"]
train_stats2c = (
    train_f.groupby(agg_cols2c, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pred2c"})
)
pred_df = pred_df.merge(train_stats2c, on=agg_cols2c, how="left")

agg_cols3 = ["R", "C", "t_idx", "u_in_q", "u_in_lag1_q", "u_in_diff_q"]
train_stats3 = (
    train_f.groupby(agg_cols3, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pred3"})
)
pred_df = pred_df.merge(train_stats3, on=agg_cols3, how="left")

agg_cols4 = ["R", "C", "t_idx", "u_in_q"]
train_stats4 = (
    train_f.groupby(agg_cols4, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pred4"})
)
pred_df = pred_df.merge(train_stats4, on=agg_cols4, how="left")

agg_cols4b = ["R", "C", "t_idx", "u_in_bin", "u_in_mean_bin"]
train_stats4b = (
    train_f.groupby(agg_cols4b, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pred4b"})
)
pred_df = pred_df.merge(train_stats4b, on=agg_cols4b, how="left")

agg_cols4c = ["R", "C", "t_idx", "u_in_bin", "u_in_last_bin"]
train_stats4c = (
    train_f.groupby(agg_cols4c, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pred4c"})
)
pred_df = pred_df.merge(train_stats4c, on=agg_cols4c, how="left")

agg_cols5 = ["R", "C", "t_idx"]
train_stats5 = (
    train_f.groupby(agg_cols5, sort=False)["pressure"]
    .mean()
    .reset_index()
    .rename(columns={"pressure": "pred5"})
)
pred_df = pred_df.merge(train_stats5, on=agg_cols5, how="left")

global_mean = float(df_train["pressure"].mean())

pred = pred_df["pred"]
pred = pred.fillna(pred_df["pred2"])
pred = pred.fillna(pred_df["pred2u"])  # CHANGE: new u_out-aware intermediate fallback
pred = pred.fillna(pred_df["pred2b"])
pred = pred.fillna(pred_df["pred2c"])
pred = pred.fillna(pred_df["pred3"])
pred = pred.fillna(pred_df["pred4"])
pred = pred.fillna(pred_df["pred4b"])
pred = pred.fillna(pred_df["pred4c"])
pred = pred.fillna(pred_df["pred5"])
pred = pred.fillna(global_mean)

pred_df["pred_final"] = pred.to_numpy(dtype=np.float64)
pred_df.sort_values(["breath_id", "t_idx"], inplace=True)

pred_df["pred_final"] = pred_df["pred_final"].where(
    pred_df["u_out"].to_numpy() == 0, np.nan
)
pred_df["pred_final"] = (
    pred_df.groupby("breath_id")["pred_final"].ffill().fillna(global_mean)
)

pred_df_by_id = pred_df[["id", "pred_final"]].sort_values("id").reset_index(drop=True)
test_ids_sorted = (
    pd.Series(test_f["id"].to_numpy()).sort_values().reset_index(drop=True)
)
assert (pred_df_by_id["id"].to_numpy() == test_ids_sorted.to_numpy()).all()

out = pd.DataFrame(
    {
        "id": test_f["id"].to_numpy(),
        "pressure": pred_df_by_id["pred_final"].to_numpy(dtype=np.float64),
    }
)

out["pressure"] = out["pressure"].apply(find_nearest)
out[["id", "pressure"]].to_csv("submission.csv", index=False)

print(out.head())
print("Wrote submission.csv with shape:", out.shape)
print("pressure range:", float(out["pressure"].min()), float(out["pressure"].max()))
print(
    "NaN rates pred/pred2/pred2u/pred2b/pred2c/pred3/pred4/pred4b/pred4c/pred5:",
    float(pred_df["pred"].isna().mean()),
    float(pred_df["pred2"].isna().mean()),
    float(pred_df["pred2u"].isna().mean()),
    float(pred_df["pred2b"].isna().mean()),
    float(pred_df["pred2c"].isna().mean()),
    float(pred_df["pred3"].isna().mean()),
    float(pred_df["pred4"].isna().mean()),
    float(pred_df["pred4b"].isna().mean()),
    float(pred_df["pred4c"].isna().mean()),
    float(pred_df["pred5"].isna().mean()),
)
