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

3.5386

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

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import gc
import os
import random
import matplotlib.pyplot as plt



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
train["u_in_lag1"] = train.groupby("breath_id")["u_in"].shift(1).fillna(0.0)
test["u_in_lag1"] = test.groupby("breath_id")["u_in"].shift(1).fillna(0.0)

train_insp = train.loc[
    (train["u_out"] == 0) & (train["dcount"] >= 1),
    ["R", "C", "dcount", "u_in", "u_in_lag1", "pressure"],
].copy()

u_bins = np.linspace(0.0, 100.0, 101)  # 100 equal-width bins

train_insp["u_in_bin"] = pd.cut(
    train_insp["u_in"], bins=u_bins, include_lowest=True, labels=False
).astype("int16")
train_insp["u_in_lag1_bin"] = pd.cut(
    train_insp["u_in_lag1"], bins=u_bins, include_lowest=True, labels=False
).astype("int16")

test["u_in_bin"] = pd.cut(
    test["u_in"], bins=u_bins, include_lowest=True, labels=False
).astype("int16")
test["u_in_lag1_bin"] = pd.cut(
    test["u_in_lag1"], bins=u_bins, include_lowest=True, labels=False
).astype("int16")

fallback_by_rcdbl = (
    train_insp.groupby(["R", "C", "dcount", "u_in_bin", "u_in_lag1_bin"])["pressure"]
    .mean()
    .astype(float)
)

fallback_by_rcdb = (
    train_insp.groupby(["R", "C", "dcount", "u_in_bin"])["pressure"]
    .mean()
    .astype(float)
)
fallback_by_rcd = (
    train_insp.groupby(["R", "C", "dcount"])["pressure"].mean().astype(float)
)
fallback_by_dcount = train_insp.groupby("dcount")["pressure"].mean().astype(float)
global_insp_median = float(train_insp["pressure"].median())

print(
    "Fallback entries:",
    "(R,C,dcount,u_in_bin,u_in_lag1_bin)=",
    int(fallback_by_rcdbl.shape[0]),
    "| (R,C,dcount,u_in_bin)=",
    int(fallback_by_rcdb.shape[0]),
    "| (R,C,dcount)=",
    int(fallback_by_rcd.shape[0]),
    "| dcount range:",
    int(fallback_by_dcount.index.min()),
    "-",
    int(fallback_by_dcount.index.max()),
)
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
    test[["id", "R", "C", "dcount", "u_in", "u_in_bin", "u_in_lag1_bin", "u_out"]],
    on="id",
    how="left",
)

mask_insp = sub["u_out"].eq(0)
mask_unfilled_insp = sub["pressure"].isna() & mask_insp

bin_width = 1.0  # because 0..100 split into 100 bins
u_in_val = sub.loc[mask_unfilled_insp, "u_in"].astype(float).clip(0.0, 100.0)
u_bin = sub.loc[mask_unfilled_insp, "u_in_bin"].astype(int)
u_bin_next = (u_bin + 1).clip(0, 99)

u_left = u_bin.astype(float) * bin_width
alpha = ((u_in_val - u_left) / bin_width).clip(0.0, 1.0)

rcdbl_keys = list(
    zip(
        sub.loc[mask_unfilled_insp, "R"],
        sub.loc[mask_unfilled_insp, "C"],
        sub.loc[mask_unfilled_insp, "dcount"],
        sub.loc[mask_unfilled_insp, "u_in_bin"],
        sub.loc[mask_unfilled_insp, "u_in_lag1_bin"],
    )
)
rcdbl_pred = pd.Series(rcdbl_keys, index=sub.index[mask_unfilled_insp]).map(
    fallback_by_rcdbl
)

rcdb_keys_lo = list(
    zip(
        sub.loc[mask_unfilled_insp, "R"],
        sub.loc[mask_unfilled_insp, "C"],
        sub.loc[mask_unfilled_insp, "dcount"],
        u_bin,
    )
)
rcdb_keys_hi = list(
    zip(
        sub.loc[mask_unfilled_insp, "R"],
        sub.loc[mask_unfilled_insp, "C"],
        sub.loc[mask_unfilled_insp, "dcount"],
        u_bin_next,
    )
)

rcdb_lo = pd.Series(rcdb_keys_lo, index=sub.index[mask_unfilled_insp]).map(
    fallback_by_rcdb
)
rcdb_hi = pd.Series(rcdb_keys_hi, index=sub.index[mask_unfilled_insp]).map(
    fallback_by_rcdb
)

rcdb_hi2 = rcdb_hi.fillna(rcdb_lo)
rcdb_lo2 = rcdb_lo.fillna(rcdb_hi2)
rcdb_interp = rcdb_lo2 * (1.0 - alpha.values) + rcdb_hi2 * alpha.values

rcd_keys = list(
    zip(
        sub.loc[mask_unfilled_insp, "R"],
        sub.loc[mask_unfilled_insp, "C"],
        sub.loc[mask_unfilled_insp, "dcount"],
    )
)
rcd_pred = pd.Series(rcd_keys, index=sub.index[mask_unfilled_insp]).map(fallback_by_rcd)
dc_pred = sub.loc[mask_unfilled_insp, "dcount"].map(fallback_by_dcount)

filled_insp = (
    rcdbl_pred.fillna(rcdb_interp)
    .fillna(rcd_pred)
    .fillna(dc_pred)
    .fillna(global_insp_median)
)
sub.loc[mask_unfilled_insp, "pressure"] = filled_insp.astype(float)

mask_exp = sub["u_out"].eq(1) & sub["pressure"].isna()
sub.loc[mask_exp, "pressure"] = global_insp_median

sub = sub.drop(
    columns=["R", "C", "dcount", "u_in", "u_in_bin", "u_in_lag1_bin", "u_out"]
)

sub["pressure"] = sub["pressure"].astype(float)
sub["id"] = sub["id"].astype(np.int64)
sub = sub.sort_values("id").reset_index(drop=True)

sub.head()



## === cell 16
sub.to_csv("submission-postprocessing.csv", index=False)
print("Wrote submission-postprocessing.csv with shape:", sub.shape)
print(sub.describe(include="all"))
print("Submission columns:", list(sub.columns))
print("Any NaNs?", sub.isna().any().to_dict())
