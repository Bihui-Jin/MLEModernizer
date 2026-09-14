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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

0.1141916451185192

# 6. Current score

1.11531

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.4947) has done: 'I fix the execution blockers by (1) restoring the missing cell number, (2) removing the dependency on the non-existent `../input/ventmodels` file, and (3) always generating a valid `submission-postprocessing.csv` from `sample_submission.csv`. To keep the core logic intact, I preserve your pressure “matching” heuristic exactly and use it to fill predictions where available. For remaining rows (no match), I fall back to a safe baseline prediction (0.0) so the notebook always completes and writes a correctly formatted submission. I also guard against empty `BIDtrain/BIDtest` concatenations so the script doesn’t crash if no matches are found.'
- What this solution (achieved 6.16068) has done: 'Your current score is very poor because most test rows remain at the baseline `pressure=0.0`, since predictions are only filled for breaths that pass the “match” filter. To move toward the target, we keep your exact matching heuristic, but add a minimal, competition-legal fallback for all remaining rows: a per-time-step median pressure learned from the training inspiratory phase (`u_out==0`) using the same `dcount` index. This preserves your core logic (the matching still overrides), but avoids leaving the vast majority of rows at zero. We also ensure we only use inspiratory data to compute the fallback and keep submission `id` alignment intact.'
- What this solution (achieved 6.03448) has done: 'Your current gap to the target is very large (lower is better), so the most impactful minimal change is to improve the fallback for rows not covered by your existing “match” heuristic. I keep your matching logic intact, but replace the crude per-`dcount` median fallback with a more informative, still-legal fallback: a grouped median by `(R, C, dcount)` computed only on inspiratory rows (`u_out==0`), with a safe backoff chain to `(dcount)` then global inspiratory median. I also ensure we never overwrite already-matched predictions and that submission `id` alignment remains correct. These changes typically reduce MAE substantially without altering your core heuristic or introducing new modeling/training.'
- What this solution (achieved 4.12443) has done: 'Your score is still far from the target (lower is better), so the smallest safe improvement is to make your fallback predictions more informative while keeping your core “match via (SP - u_in/P)” heuristic unchanged. Right now, unmatched rows use a coarse median by (R,C,dcount), which ignores the dominant driver `u_in`; we can legally and cheaply replace that fallback with a median lookup by `(R, C, dcount, binned_u_in)` computed only on inspiratory rows (`u_out==0`). We keep the exact same override priority: matched heuristic predictions always win; the new fallback is only used for previously-unfilled rows. We also fix a subtle bug where `mask_unfilled` was incorrectly treating real 0.0 predictions as missing (pressure is never 0 in train), by initializing submission pressures as NaN and only filling NaNs.'
- What this solution (achieved 4.33396) has done: 'Your current MAE (4.124) is still far above the target (0.114; lower is better), so we should improve predictions while keeping your core “match via (SP - u_in/P)” heuristic unchanged. The most minimal, high-impact change is to make the fallback closer to the metric: compute fallback medians using only inspiratory rows **and** only for timesteps `dcount>=1` (since your matching logic predicts only for `dcount>0`), which avoids mixing in the special first step that can distort medians. Next, we restrict fallback filling to inspiratory timesteps in the test (`u_out==0`) and set expiratory (`u_out==1`) to a safe constant (global inspiratory median) because expiratory isn’t scored and this avoids contaminating any joins/logic. Finally, we make the fallback binning slightly more granular (100 bins) to better condition on `u_in` without changing the approach.'
- What this solution (achieved 3.46693) has done: 'We keep your existing “match via (SP - u_in/P)” heuristic exactly as-is and only improve the fallback used for the many unmatched rows, because that’s what dominates your MAE (and you’re far from the target). The smallest high-impact improvement is to make the fallback condition not just on `(R,C,dcount,u_in_bin)` but also on the recent history of `u_in` by adding `u_in_lag1` (previous timestep within the breath), computed identically for train and test; this preserves evaluation semantics and uses only allowed features. We also safely back off through `(R,C,dcount,u_in_bin)` → `(R,C,dcount)` → `(dcount)` → global median, and keep expiratory rows filled with a constant since they are not scored. These changes keep runtime reasonable (groupby medians) and still produce the same valid submission CSV.'
- What this solution (achieved 3.5432) has done: 'Your current score (3.46693 MAE; lower is better) is still far above the target (0.11419), so we should improve predictions while keeping your core “(SP - u_in/P) match” heuristic unchanged. The biggest remaining weakness is the fallback: it uses medians, which are robust but can be biased for MAE; switching the fallback aggregator to *mean* (still computed only on inspiratory rows) typically reduces MAE for continuous targets without changing the overall approach. To keep changes minimal and stable, I only replace the fallback tables’ `.median()` with `.mean()` and keep the same backoff chain and submission alignment. Everything else (matching logic, bins, lag feature, expiratory handling, and output CSV) stays the same.'
- What this solution (achieved 3.5386) has done: 'Your current MAE (3.5432, lower is better) is far above the target (0.1142), so we should cautiously improve it while preserving your existing “(SP - u_in/P) match” heuristic unchanged. The minimal high-impact fix is to make the fallback closer to the metric by predicting **only on inspiratory timesteps** and then explicitly setting expiratory rows to any constant (not scored), which also prevents expiratory rows from influencing the fallback lookup behavior. Additionally, we can reduce bias in the fallback by replacing the discretized `u_in` bins with a **small, stable local interpolation** between the two nearest `u_in_bin` means (still the same groupby-mean core approach; just smoother post-processing). Everything else (data paths, matching search, use of `dcount>=1`, lag feature, and submission formatting) remains the same.'
- What this solution (achieved 3.81167) has done: 'Your current MAE (3.5386, lower-is-better) is still far above the target (0.1142), so we should improve the score by making the fallback predictions more informative while keeping your core (SP − u_in/P) matching heuristic unchanged. The minimal high-impact change is to add a stronger, still-cheap fallback keyed by `(R, C, dcount, u_in_bin, u_in_lag1_bin)` with **linear interpolation on both u_in and u_in_lag1** (2D smoothing), instead of only interpolating u_in. We keep the exact same backoff chain and only apply the fallback to unmatched inspiratory rows, leaving expiratory rows filled with a constant since they are not scored. This preserves your overall approach and evaluation semantics while typically reducing error for the many unmatched rows.'
- What this solution (achieved 3.8923) has done: 'Your current MAE (3.81167, lower-is-better) is still far above the target (0.11419), so we should improve predictions while keeping your core “(SP − u_in/P) match” heuristic unchanged. The biggest remaining weakness is the fallback lookup granularity and smoothing: we keep your exact groupby-mean fallback tables and bilinear interpolation structure, but make the bins finer (200 bins) so the interpolation is less lossy. To avoid hurting stability, we also (a) compute `u_in_lag1` consistently, (b) ensure bins never go out of range with the new bin count, and (c) keep expiratory handling unchanged (not scored). These are minimal changes localized to the fallback discretization/smoothing and should move the score downward toward the target.'
- What this solution (achieved 2.89522) has done: 'Your current score (3.8923, lower-is-better) is still far above the target (0.1142), so we should improve accuracy while keeping your core “(SP − u_in/P) match” heuristic unchanged. The most impactful minimal fix is to make the fallback table condition on the *time_step* (via `dcount`) and *controls* more faithfully by using a smooth, continuous fallback based on a **KNN regressor trained only on inspiratory rows**; this does not change your heuristic and only replaces the existing coarse bin-lookup fallback for the unmatched rows. To stay within your constraints and runtime, we train KNN on a compact feature set (`R,C,dcount,u_in,u_in_lag1`) with standardization and then only predict for the unmatched inspiratory test rows; expiratory rows remain filled with a constant because they are not scored. Everything else (data loading, matching search loops, and submission formatting) stays the same, and the script still writes a valid `submission-postprocessing.csv`.'
- What this solution (achieved 2.10644) has done: 'Your current MAE (2.895) is still far above the target (0.114; lower is better), so we should cautiously improve predictive accuracy while keeping your core “(SP − u_in/P) match” heuristic unchanged. The biggest lever left (without changing the approach) is the KNN fallback: we can make it more faithful to breath dynamics by (1) using more within-breath history (`u_in` lags 1–3) as additional features and (2) including `time_step` as a continuous proxy alongside `dcount`, while still training only on inspiratory rows. To avoid the KNN being dominated by repeated breaths and to keep runtime under control, we train KNN on a compact, stratified subsample of inspiratory rows across `dcount` (same semantics, just fewer duplicated points), then predict only for unmatched inspiratory test rows as before. Submission writing, id alignment, and the heuristic override order remain unchanged.'
- What this solution (achieved 1.1085) has done: 'Your current MAE (2.10644; lower is better) is still far above the target (0.11419), so we should improve accuracy while keeping your core `(SP - u_in/P)` breath-matching heuristic unchanged. The most direct, minimal lever is to make the KNN fallback more faithful to the metric by (a) predicting **only inspiratory** rows (already done), (b) adding the key physiological cumulative signal `u_in_cumsum` as an extra feature, and (c) tuning KNN neighborhood size slightly downward so it relies less on overly-smooth global averaging. To keep runtime under 600s and avoid changing the approach, we keep the same pipeline/fit/predict flow and the same stratified per-`dcount` cap, only extending the feature set and modestly adjusting `n_neighbors`. Submission writing, id alignment, and expiratory handling remain identical.'
- What this solution (achieved 1.11531) has done: 'We keep your heuristic breath-matching logic exactly the same and focus only on making the KNN fallback closer to the competition metric without changing the overall approach. The biggest issue is that KNN is trained on all inspiratory rows (including release transitions) while Kaggle scores only where `u_out==0` and (effectively) early inspiratory timesteps; we restrict the KNN training target to the same scored region (`u_out==0` and `u_out` stays 0 at the next step), which reduces label noise and should lower MAE toward the target. We also add a minimal, metric-aligned post-processing step: snap KNN predictions to the nearest valid pressure level from the training set (pressure is quantized), which typically reduces MAE with negligible logic change. Finally, we keep runtime safe by leaving your per-`dcount` cap and pipeline structure intact, and still write the same valid `submission-postprocessing.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import gc
import os
import random
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
train = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")

train["dcount"] = train.groupby("breath_id")["id"].transform("cumcount")
test["dcount"] = test.groupby("breath_id")["id"].transform("cumcount")

train["uo"] = 80 - train.groupby("breath_id")["u_out"].transform("sum")
test["uo"] = 80 - test.groupby("breath_id")["u_out"].transform("sum")

print(train.shape)
train.head()



## === cell 2
pass



## === cell 3
p_coef = [
    0.01,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
]
i_coef = [
    0.00,
    0.01,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
]
setpoints = [10, 15, 20, 25, 30, 35]



## === cell 4
unique_pressures = list(train["pressure"].round(decimals=7).unique())
len(unique_pressures), unique_pressures[:10]



## === cell 5
max_pressure = 64.82099173863328
min_pressure = -1.895744294564641
diff_pressure = 0.0703021454512



## === cell 6
BIDtrain = []
for SP in setpoints:
    for P in p_coef:
        train["u_ctrl"] = ((SP - train["u_in"] / P)).round(decimals=7)

        train["isclass"] = 0
        train.loc[train["u_ctrl"].isin(unique_pressures), "isclass"] = 1

        dt = (
            train.loc[(train.u_out == 0) & (train.dcount >= 1)]
            .groupby("breath_id")[["isclass", "uo"]]
            .agg({"isclass": "sum", "uo": "first"})
            .reset_index()
            .sort_values("isclass", ascending=False)
            .reset_index(drop=True)
        )
        dt = dt.loc[dt["isclass"] >= (dt["uo"] - 3)]
        if dt.shape[0] > 0:
            print("matches:", dt.shape[0], "P=", P, "SP=", SP)
            dt["P"] = P
            dt["SP"] = SP
            BIDtrain.append(dt)

if len(BIDtrain) > 0:
    BIDtrain = pd.concat(BIDtrain, ignore_index=True)
else:
    BIDtrain = pd.DataFrame(columns=["breath_id", "isclass", "uo", "P", "SP"])

print(BIDtrain.shape)
BIDtrain.head(10)



## === cell 7
BIDtest = []
for SP in setpoints:
    for P in p_coef:
        test["u_ctrl"] = ((SP - test["u_in"] / P)).round(decimals=7)

        test["isclass"] = 0
        test.loc[test["u_ctrl"].isin(unique_pressures), "isclass"] = 1

        dt = (
            test.loc[(test.u_out == 0) & (test.dcount >= 1)]
            .groupby("breath_id")[["isclass", "uo"]]
            .agg({"isclass": "sum", "uo": "first"})
            .reset_index()
            .sort_values("isclass", ascending=False)
            .reset_index(drop=True)
        )
        dt = dt.loc[dt["isclass"] >= (dt["uo"] - 3)]
        if dt.shape[0] > 0:
            print("matches:", dt.shape[0], "P=", P, "SP=", SP)
            dt["P"] = P
            dt["SP"] = SP
            BIDtest.append(dt)

if len(BIDtest) > 0:
    BIDtest = pd.concat(BIDtest, ignore_index=True)
else:
    BIDtest = pd.DataFrame(columns=["breath_id", "isclass", "uo", "P", "SP"])

print(BIDtest.shape)
BIDtest.head(10)



## === cell 8
if len(BIDtrain) > 0:
    n = min(10, len(BIDtrain))
    for i in range(n):
        bid = BIDtrain.iloc[i]
        P = bid.P
        SP = bid.SP
        tmp_plot = train.loc[train.breath_id == bid.breath_id].copy()
        tmp_plot["u_ctrl"] = SP - tmp_plot["u_in"] / P
        ax = tmp_plot.loc[(tmp_plot.u_out == 0) & (tmp_plot.dcount >= 0)].plot(
            x="time_step",
            y=["pressure", "u_ctrl"],
            title="P=" + str(P) + " SP:" + str(SP),
        )
        plt.close(ax.figure)



## === cell 9
if len(BIDtest) > 0:
    n = min(10, len(BIDtest))
    for i in range(n):
        bid = BIDtest.iloc[i]
        P = bid.P
        SP = bid.SP
        tmp_plot = test.loc[test.breath_id == bid.breath_id].copy()
        tmp_plot["u_ctrl"] = SP - tmp_plot["u_in"] / P
        ax = tmp_plot.loc[(tmp_plot.u_out == 0) & (tmp_plot.dcount >= 0)].plot(
            x="time_step", y=["u_ctrl"], title="P=" + str(P) + " SP:" + str(SP)
        )
        plt.close(ax.figure)



## === cell 10
test.head()



## === cell 11
sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
sub = pd.read_csv(sample_path)
sub = sub[["id", "pressure"]].copy()

sub["pressure"] = np.nan



## === cell 12
train = train.copy()
test = test.copy()

for lag in [1, 2, 3]:
    train[f"u_in_lag{lag}"] = (
        train.groupby("breath_id")["u_in"].shift(lag).fillna(0.0).astype(float)
    )
    test[f"u_in_lag{lag}"] = (
        test.groupby("breath_id")["u_in"].shift(lag).fillna(0.0).astype(float)
    )

train["u_in_cumsum"] = train.groupby("breath_id")["u_in"].cumsum().astype(float)
test["u_in_cumsum"] = test.groupby("breath_id")["u_in"].cumsum().astype(float)

train_insp_basic = train.loc[
    (train["u_out"] == 0) & (train["dcount"] >= 1), ["pressure"]
].copy()
global_insp_median = float(train_insp_basic["pressure"].median())
print("Global inspiratory median pressure:", global_insp_median)



## === cell 13
if len(BIDtest) > 0:
    test = test.merge(BIDtest[["breath_id", "P", "SP"]], on="breath_id", how="left")
    test["pred"] = (test.SP - test["u_in"] / test.P).round(decimals=7)
    test_ids = BIDtest.breath_id.values

    tmp = test.loc[
        (test.dcount > 0) & (test.breath_id.isin(test_ids)), ["id", "pred"]
    ].copy()
else:
    tmp = pd.DataFrame(
        {"id": pd.Series(dtype=np.int64), "pred": pd.Series(dtype=np.float64)}
    )

tmp.head()



## === cell 14
ventmodels_path = "../input/ventmodels"
if os.path.exists(ventmodels_path):
    print(os.listdir(ventmodels_path)[:20])
else:
    print(f"{ventmodels_path} not found; using heuristic + fallback baseline instead.")



## === cell 15
sub = sub.merge(tmp, on="id", how="left")

mask_match = sub["pred"].notna()
sub.loc[mask_match, "pressure"] = sub.loc[mask_match, "pred"].astype(float)
sub = sub.drop(columns=["pred"])

sub = sub.merge(
    test[
        [
            "id",
            "R",
            "C",
            "dcount",
            "time_step",
            "u_in",
            "u_in_lag1",
            "u_in_lag2",
            "u_in_lag3",
            "u_in_cumsum",
            "u_out",
        ]
    ],
    on="id",
    how="left",
)

mask_insp = sub["u_out"].eq(0)
mask_unfilled_insp = sub["pressure"].isna() & mask_insp

from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

train["u_out_next"] = (
    train.groupby("breath_id")["u_out"].shift(-1).fillna(1).astype(int)
)
mask_train_scored_like = (
    (train["u_out"] == 0) & (train["u_out_next"] == 0) & (train["dcount"] >= 1)
)

train_knn = train.loc[
    mask_train_scored_like,
    [
        "R",
        "C",
        "dcount",
        "time_step",
        "u_in",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_lag3",
        "u_in_cumsum",
        "pressure",
    ],
].copy()

per_dcount_cap = 15000  # unchanged: keep runtime/memory predictable
train_knn = (
    train_knn.groupby("dcount", group_keys=False)
    .apply(lambda g: g.sample(n=min(len(g), per_dcount_cap), random_state=SEED))
    .reset_index(drop=True)
)

feature_cols = [
    "R",
    "C",
    "dcount",
    "time_step",
    "u_in",
    "u_in_lag1",
    "u_in_lag2",
    "u_in_lag3",
    "u_in_cumsum",
]

X_train = train_knn[feature_cols].astype(np.float32)
y_train = train_knn["pressure"].astype(np.float32)

knn_model = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "knn",
            KNeighborsRegressor(
                n_neighbors=35, weights="distance", metric="minkowski", p=2
            ),
        ),
    ]
)

knn_model.fit(X_train, y_train)

X_test_unfilled = sub.loc[mask_unfilled_insp, feature_cols].astype(np.float32)
knn_pred = knn_model.predict(X_test_unfilled).astype(np.float32)

knn_pred = np.clip(
    knn_pred, float(train["pressure"].min()) - 1.0, float(train["pressure"].max()) + 1.0
)

pressure_grid = np.sort(train["pressure"].round(7).unique()).astype(np.float32)
idx = np.searchsorted(pressure_grid, knn_pred, side="left")
idx = np.clip(idx, 0, len(pressure_grid) - 1)
idx_left = np.clip(idx - 1, 0, len(pressure_grid) - 1)
right = pressure_grid[idx]
left = pressure_grid[idx_left]
choose_left = (knn_pred - left) <= (right - knn_pred)
snapped = np.where(choose_left, left, right).astype(np.float32)

sub.loc[mask_unfilled_insp, "pressure"] = snapped.astype(float)

mask_exp = sub["u_out"].eq(1) & sub["pressure"].isna()
sub.loc[mask_exp, "pressure"] = global_insp_median

sub["pressure"] = sub["pressure"].fillna(global_insp_median).astype(float)

sub = sub.drop(
    columns=[
        "R",
        "C",
        "dcount",
        "time_step",
        "u_in",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_lag3",
        "u_in_cumsum",
        "u_out",
    ]
)

sub["id"] = sub["id"].astype(np.int64)
sub = sub.sort_values("id").reset_index(drop=True)

sub.head()



## === cell 16
sub.to_csv("submission-postprocessing.csv", index=False)
print("Wrote submission-postprocessing.csv with shape:", sub.shape)
print(sub.describe(include="all"))
print("Submission columns:", list(sub.columns))
print("Any NaNs?", sub.isna().any().to_dict())
