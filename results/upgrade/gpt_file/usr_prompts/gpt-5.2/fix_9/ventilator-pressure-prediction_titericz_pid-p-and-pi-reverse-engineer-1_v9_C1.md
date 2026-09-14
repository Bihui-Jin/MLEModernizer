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

0.1141505871263613

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.49482) has done: 'Your current script already runs to completion and writes a submission, but it writes to `submission-postprocessing.csv` (not the standard `submission.csv`) and it never fills pressure for `dcount==0` even when the PID parameters are known for that breath, leaving a preventable chunk of predictions at the baseline 0. I make two minimal changes: (1) apply the same PID-based formula to all timesteps of matched breaths (including `dcount==0`) to reduce MAE, and (2) additionally snap predictions to the known discrete pressure grid from train (a legitimate post-processing aligned to the label distribution) which typically improves MAE without changing your core approach. Finally, I also write `submission.csv` alongside your existing filename to ensure Kaggle picks it up cleanly.'
- What this solution (achieved 17.65244) has done: 'Your current run doesn’t yield a Kaggle score mainly because it’s reading from `../input/...`, which won’t exist in the paths you listed (you have `/kaggle/input/...` and `/kaggle/data/...`), so the notebook likely fails before producing a submission. I make a minimal, robustness-only path resolver that tries the known locations you provided and keeps the rest of your logic identical. I also add a hard safety check that the produced submission has exactly the sample submission ids (same count and set) so you always get a valid `.csv` to upload. No changes are made to the PID matching logic, coefficient grids, or snapping; the goal is to reliably generate a submission so you can obtain a score and then tune toward the target.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import gc
import os
import random
import matplotlib.pyplot as plt
import time

random.seed(42)
np.random.seed(42)



## === cell 1
CANDIDATE_DIRS = [
    "../input/ventilator-pressure-prediction",  # original
    "/kaggle/input/ventilator-pressure-prediction",  # standard Kaggle
    "/kaggle/data/ventilator-pressure-prediction",  # provided in your tree
    "/kaggle/input",  # fallback where files may be directly placed
    "/kaggle/data",  # fallback where files may be directly placed
]


def resolve_data_dir():
    for d in CANDIDATE_DIRS:
        if os.path.isdir(d):
            if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
                os.path.join(d, "test.csv")
            ):
                return d
            nested = os.path.join(d, "ventilator-pressure-prediction")
            if (
                os.path.isdir(nested)
                and os.path.exists(os.path.join(nested, "train.csv"))
                and os.path.exists(os.path.join(nested, "test.csv"))
            ):
                return nested
    raise FileNotFoundError(
        "Could not locate train.csv/test.csv. Tried: " + ", ".join(CANDIDATE_DIRS)
    )


DATA_DIR = resolve_data_dir()
print("Using DATA_DIR:", DATA_DIR)

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)

train["dcount"] = train.groupby("breath_id")["id"].transform("cumcount")
test["dcount"] = test.groupby("breath_id")["id"].transform("cumcount")

train["uo"] = 80 - train.groupby("breath_id")["u_out"].transform("sum")
test["uo"] = 80 - test.groupby("breath_id")["u_out"].transform("sum")

train["time_delta"] = (
    train["time_step"] - train.groupby("breath_id")["time_step"].shift(1)
).fillna(0)
test["time_delta"] = (
    test["time_step"] - test.groupby("breath_id")["time_step"].shift(1)
).fillna(0)

print("train:", train.shape, "test:", test.shape, "sample_sub:", sample_sub.shape)
train.head()



## === cell 2
train["pred"] = 0.0
train[["id", "pred"]].head()



## === cell 3
test["pred"] = np.nan
test[["id", "pred"]].head()



## === cell 4
train["error"] = (train["pressure"] - train["pred"]).abs()
train.loc[train.u_out > 0, "error"] = 0
ax = train.loc[train.u_out == 0, "error"].hist(bins=20)
plt.title("Train abs error (u_out==0) vs baseline pred=0")
plt.show()



## === cell 5
maxdrift = (
    train.loc[train.u_out == 0, "error"].mean()
    + 3 * train.loc[train.u_out == 0, "error"].std()
)
maxdrift



## === cell 6
pass



## === cell 7
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



## === cell 8
unique_pressures = train["pressure"].round(decimals=7).unique()
unique_pressures = list(np.sort(unique_pressures))
len(unique_pressures), unique_pressures[:10]



## === cell 9
maxdrift / (unique_pressures[1] - unique_pressures[0])



## === cell 10
max_pressure = 64.82099173863328
min_pressure = -1.895744294564641
diff_pressure = 0.0703021454512




## === cell 11
def generate_u_in(pressure, time_step, kp, ki, kt, integral=0):
    dt = np.diff(time_step, prepend=[0])
    preds = []
    for j in range(32):
        error = kt - pressure[j]
        integral += (error - integral) * (dt[j] / (dt[j] + 0.5))
        preds.append(kp * error + ki * integral)
    return preds


pressure = train[train["breath_id"] == 1]["pressure"].values
timestep = train[train["breath_id"] == 1]["time_step"].values
u_in_hat = generate_u_in(pressure, timestep, 0.8, 8.0, 20)
noise = train[train["breath_id"] == 1]["u_in"].values[:32] - u_in_hat

plt.figure()
plt.plot(
    timestep[:32], train[train["breath_id"] == 1]["u_in"].values[:32], label="u_in"
)
plt.plot(timestep[:32], u_in_hat, label="u_in_hat")
plt.legend()
plt.show()



## === cell 12
pass



## === cell 13
pass



## === cell 14
pass



## === cell 15
pass



## === cell 16
pass



## === cell 17
pass



## === cell 18
pass



## === cell 19
pass



## === cell 20
pass



## === cell 21
pass



## === cell 22
pass



## === cell 23
MAX_PRESSURE = PRESSURE_MAX = 64.82099173863328
MIN_PRESSURE = PRESSURE_MIN = -1.895744294564641
DIFF_PRESSURE = PRESSURE_STEP = 0.0703021454512
MIN_PRESSURE2 = PRESSURE_MIN2 = MIN_PRESSURE + DIFF_PRESSURE


def match_breath(u_in, u_out, timestep, kp, ki, kt):
    dt = np.diff(timestep)
    dt2 = dt / (dt + 0.5)
    in_len = np.sum(1 - u_out)
    preds = np.zeros(len(u_in)) - 999

    match = 0
    for t in range(1, in_len):
        if preds[t - 1] != -999:
            P0 = preds[t - 1]
        else:
            P0 = np.arange(MIN_PRESSURE, MAX_PRESSURE + DIFF_PRESSURE, DIFF_PRESSURE)

        if ki == 0:
            continue

        I0 = (u_in[t - 1] - kp * (kt - P0)) / ki

        I11 = I0 + (kt - MIN_PRESSURE - I0) * dt2[t - 1]
        u_in_hat1 = kp * (kt - MIN_PRESSURE) + ki * I11

        I12 = I0 + (kt - MIN_PRESSURE2 - I0) * dt2[t - 1]
        u_in_hat2 = kp * (kt - MIN_PRESSURE2) + ki * I12

        slope = u_in_hat2 - u_in_hat1
        if np.all(slope == 0):
            continue

        x_intersect = (u_in[t] - u_in_hat2) / slope
        diff = np.abs(np.round(x_intersect) - x_intersect)

        if diff.min() < 1e-10:
            match += 1
            pos = np.argmin(diff)

            if np.all(preds[t - 1] == -999):
                preds[t - 1] = P0[pos]
                preds[t] = MIN_PRESSURE + int(x_intersect[pos] + 1) * DIFF_PRESSURE
            else:
                preds[t] = MIN_PRESSURE + (np.round(x_intersect) + 1) * DIFF_PRESSURE

    return preds, match




## === cell 24
i = 1
pressure = train[train["breath_id"] == i]["pressure"].values.copy()
timestep = train[train["breath_id"] == i]["time_step"].values.copy()
u_in = train[train["breath_id"] == i]["u_in"].values.copy()
u_out = train[train["breath_id"] == i]["u_out"].values.copy()
ypred = train[train["breath_id"] == i]["pred"].values.copy()
pressure[:5], ypred[:5]



## === cell 25
t0 = time.time()
res, match = match_breath(u_in, u_out, timestep, 1.0, 8.0, 20.0)
print("match:", match, "elapsed_sec:", time.time() - t0)



## === cell 26
pass



## === cell 27
pass



## === cell 28
pass



## === cell 29
starttime = time.time()

PIDTEST = []
count = 0
for i in test["breath_id"].unique()[:50]:
    count += 1
    ids = test[test["breath_id"] == i]["id"].values.copy()
    timestep = test[test["breath_id"] == i]["time_step"].values.copy()
    u_in = test[test["breath_id"] == i]["u_in"].values.copy()
    u_out = test[test["breath_id"] == i]["u_out"].values.copy()
    match = 0
    for P in p_coef:
        for I in i_coef:
            for SP in setpoints:
                res, match = match_breath(u_in, u_out, timestep, P, I, SP)
                if match > 24:
                    dt = pd.DataFrame(
                        {
                            "id": ids,
                            "breath_id": i,
                            "P": P,
                            "I": I,
                            "SP": SP,
                            "pressure": res,
                        }
                    )
                    PIDTEST.append(dt)
                    print(
                        i,
                        count,
                        match,
                        P,
                        I,
                        SP,
                        (time.time() - starttime),
                        (time.time() - starttime) / count,
                    )
                    break
            if match > 24:
                break
        if match > 24:
            break

if len(PIDTEST) > 0:
    PIDTEST = pd.concat(PIDTEST).reset_index(drop=True)
else:
    PIDTEST = pd.DataFrame(columns=["id", "breath_id", "P", "I", "SP", "pressure"])

PIDTEST.shape



## === cell 30
pass



## === cell 31
PIDTEST.to_csv("pid-test-1.csv", index=False)
os.path.getsize("pid-test-1.csv")



## === cell 32
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



## === cell 33
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



## === cell 34
if len(BIDtrain) > 0:
    for i in range(min(10, len(BIDtrain))):
        bid = BIDtrain.iloc[i]
        P = bid.P
        SP = bid.SP
        tmp_plot = train.loc[train.breath_id == bid.breath_id].copy()
        tmp_plot["u_ctrl"] = SP - tmp_plot["u_in"] / P
        tmp_plot.loc[(tmp_plot.u_out == 0) & (tmp_plot.dcount >= 0)].plot(
            x="time_step",
            y=["pressure", "u_ctrl"],
            title="P=" + str(P) + " SP:" + str(SP),
        )
        plt.show()



## === cell 35
if len(BIDtest) > 0:
    for i in range(min(10, len(BIDtest))):
        bid = BIDtest.iloc[i]
        P = bid.P
        SP = bid.SP
        tmp_plot = test.loc[test.breath_id == bid.breath_id].copy()
        tmp_plot["u_ctrl"] = SP - tmp_plot["u_in"] / P
        tmp_plot.loc[(tmp_plot.u_out == 0) & (tmp_plot.dcount >= 0)].plot(
            x="time_step", y=["u_ctrl"], title="P=" + str(P) + " SP:" + str(SP)
        )
        plt.show()



## === cell 36
test.head()



## === cell 37
pass



## === cell 38
test = test.merge(BIDtest[["breath_id", "P", "SP"]], on="breath_id", how="left")

I_map = {}
if len(BIDtest) > 0:
    t0 = time.time()
    uniq_bids = (
        BIDtest[["breath_id", "P", "SP"]].drop_duplicates().reset_index(drop=True)
    )
    for k in range(len(uniq_bids)):
        bid = int(uniq_bids.loc[k, "breath_id"])
        P = float(uniq_bids.loc[k, "P"])
        SP = float(uniq_bids.loc[k, "SP"])
        rows = test.loc[test["breath_id"] == bid, ["time_step", "u_in", "u_out"]]
        u_in_b = rows["u_in"].values.astype(np.float64, copy=False)
        u_out_b = rows["u_out"].values.astype(np.int64, copy=False)
        ts_b = rows["time_step"].values.astype(np.float64, copy=False)

        chosen_I = None
        for I in i_coef:
            _, m = match_breath(u_in_b, u_out_b, ts_b, P, I, SP)
            if m > 24:
                chosen_I = float(I)
                break
        if chosen_I is None:
            chosen_I = 8.0  # fallback preserves prior behavior
        I_map[bid] = chosen_I

        if (k + 1) % 200 == 0:
            print("I-map processed:", k + 1, "elapsed_sec:", time.time() - t0)

test["I"] = test["breath_id"].map(I_map).astype(np.float64)

matched_ids = (
    BIDtest["breath_id"].values if len(BIDtest) > 0 else np.array([], dtype=np.int64)
)

pid_preds = []
t0 = time.time()
for j, bid in enumerate(matched_ids):
    rows = test.loc[
        test["breath_id"] == bid, ["id", "time_step", "u_in", "u_out", "P", "I", "SP"]
    ].copy()
    if rows.shape[0] == 0:
        continue
    P = float(rows["P"].iloc[0])
    I = float(rows["I"].iloc[0]) if pd.notna(rows["I"].iloc[0]) else 8.0
    SP = float(rows["SP"].iloc[0])

    u_in_b = rows["u_in"].values.astype(np.float64, copy=False)
    u_out_b = rows["u_out"].values.astype(np.int64, copy=False)
    ts_b = rows["time_step"].values.astype(np.float64, copy=False)

    res, _ = match_breath(u_in_b, u_out_b, ts_b, P, I, SP)

    in_len = int(np.sum(1 - u_out_b))
    if in_len > 0:
        known = np.where(res[:in_len] > -999)[0]
        if known.size > 0:
            first_k = int(known[0])
            res[:first_k] = res[first_k]

    ok = res > -999
    if np.any(ok):
        pid_preds.append(
            pd.DataFrame({"id": rows.loc[ok, "id"].values, "pred_pid": res[ok]})
        )

    if (j + 1) % 500 == 0:
        print("processed matched breaths:", j + 1, "elapsed_sec:", time.time() - t0)

if len(pid_preds) > 0:
    pid_preds = pd.concat(pid_preds, ignore_index=True)
else:
    pid_preds = pd.DataFrame(columns=["id", "pred_pid"])

pid_preds.head()



## === cell 39
sub = sample_sub.copy()
sub["pressure"] = 0.0

if pid_preds.shape[0] > 0:
    pid_preds_uniq = pid_preds.dropna().drop_duplicates(subset=["id"], keep="first")
    sub = sub.merge(pid_preds_uniq, on="id", how="left")
    sub.loc[sub["pred_pid"].notna(), "pressure"] = sub.loc[
        sub["pred_pid"].notna(), "pred_pid"
    ].astype(float)
    sub = sub.drop(columns=["pred_pid"])

sub.head()



## === cell 40
pass



## === cell 41
pid_file = "pid-test-1.csv"
if os.path.exists(pid_file) and os.path.getsize(pid_file) > 0:
    tmp2 = pd.read_csv(pid_file)
    if "pressure" in tmp2.columns and "id" in tmp2.columns:
        tmp2 = tmp2.loc[tmp2.pressure > -999, ["id", "pressure"]].reset_index(drop=True)
        tmp2.columns = ["id", "pred2"]
    else:
        tmp2 = pd.DataFrame(columns=["id", "pred2"])
else:
    tmp2 = pd.DataFrame(columns=["id", "pred2"])

tmp2.head()



## === cell 42
if tmp2.shape[0] > 0:
    tmp2_uniq = tmp2.dropna().drop_duplicates(subset=["id"], keep="first")
    sub = sub.merge(tmp2_uniq, on="id", how="left")
    sub.loc[sub["pred2"].notna(), "pressure"] = sub.loc[
        sub["pred2"].notna(), "pred2"
    ].astype(float)
    sub = sub.drop(columns=["pred2"])

sub.head()



## === cell 43
pressure_grid = np.asarray(unique_pressures, dtype=np.float64)

pred = sub["pressure"].astype(np.float64).values
idx = np.searchsorted(pressure_grid, pred, side="left")
idx = np.clip(idx, 0, len(pressure_grid) - 1)

idx_left = np.clip(idx - 1, 0, len(pressure_grid) - 1)
choose_left = np.abs(pred - pressure_grid[idx_left]) <= np.abs(
    pred - pressure_grid[idx]
)
snapped = np.where(choose_left, pressure_grid[idx_left], pressure_grid[idx])

sub["pressure"] = snapped.astype(np.float64)

sub.head()



## === cell 44
sub = sub.sort_values("id").reset_index(drop=True)
sample_ids = sample_sub["id"].values

if sub.shape[0] != sample_sub.shape[0] or not np.array_equal(
    sub["id"].values, sample_ids
):
    sub = sample_sub[["id"]].merge(sub[["id", "pressure"]], on="id", how="left")
    sub["pressure"] = sub["pressure"].astype(np.float64).fillna(0.0)
    sub = sub.sort_values("id").reset_index(drop=True)

sub.to_csv("submission-postprocessing.csv", index=False)
sub.to_csv("submission.csv", index=False)

print("Wrote submission-postprocessing.csv and submission.csv with shape:", sub.shape)
print("Expected rows:", sample_sub.shape[0], "Unique ids:", sample_sub["id"].nunique())
print(sub.head())
print(sub.tail())
print("Any NaNs in pressure?:", sub["pressure"].isna().any())
print("pressure dtype:", sub["pressure"].dtype)
print(
    "Matched breaths used:",
    len(matched_ids),
    "Rows with nonzero pressure:",
    (sub["pressure"].values != 0).sum(),
)
