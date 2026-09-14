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

0.1417466187182174

# 6. Current score

1.57926

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'The crash happens because `avg()` is called on a directory that doesn’t exist in this environment, so no files are found and you divide by zero. I keep the blending logic intact but add safe handling: detect missing/empty directories, filter to valid CSVs with the required `pressure` column, and fall back to writing the provided `sample_submission.csv` (all zeros) so a valid `.csv` is always produced. I also switch file reads to use the already-available `../input/ventilator-pressure-prediction/...` paths and ensure the output filename ends with `.csv`. These fixes are score-neutral (they don’t change predictions when blend files exist) and unblock end-to-end submission creation.'
- What this solution (achieved 6.10585) has done: 'I fix the merge bug causing the missing `pressure` column by avoiding `sample_submission`’s existing `pressure` column during merge (which creates `pressure_x/pressure_y`) and explicitly constructing the final `id,pressure` output. I also add a small alignment/length check when averaging external prediction files so it can’t silently break if a CSV has the wrong row count. These are minimal, score-neutral correctness fixes except for the fallback mean-by-(R,C,time_step) path, which is already your intended logic and should yield a valid (and much better than all-zeros) submission when no blend files exist. Finally, I ensure the script always writes `submission.csv` with the correct columns and dtype.'
- What this solution (achieved 6.10593) has done: 'Your current score (6.10585, lower-is-better) is far worse than the target (0.1417), so we should make a small, legitimate improvement that better matches the metric without changing your overall “mean-by-(R,C,time_step)” fallback logic. The main issue is that the fallback averages over *all* phases, while Kaggle only scores the inspiratory phase (`u_out==0`), so we compute the mean pressure using only inspiratory rows in train and (to stay consistent) only assign those means to inspiratory rows in test. For expiratory rows (`u_out==1`), we fill with the last predicted inspiratory pressure within each breath (a simple, stable carry-forward), which avoids injecting noisy expiratory averages that don’t help the scored portion. These changes preserve your pipeline (blend-if-available, otherwise grouped mean fallback + nearest pressure snapping) while moving the MAE toward the target.'
- What this solution (achieved 6.10593) has done: 'We keep your exact blending-vs-fallback structure and the same “mean-by-(R,C,time_step)” fallback idea, but fix the main reason the fallback underperforms: it currently assigns inspiratory-derived means to *all* test rows and then forward-fills, which can leak inspiratory values into expiratory rows in a way that perturbs later alignment within a breath. Instead, we only merge/assign the grouped mean to inspiratory rows (`u_out==0`) and for expiratory rows we deterministically carry-forward the last inspiratory prediction within each breath (which is score-neutral because expiratory isn’t evaluated, but stabilizes predictions around the insp/exp boundary). We also make the `time_step` rounding consistent and robust by using an integer time-bin key (`(time_step*100).round().astype(int)`) to avoid float merge mismatches. These are minimal, metric-aligned changes that should move MAE down from ~6 toward your target without changing the overall approach.'
- What this solution (achieved 3.89746) has done: 'We keep your blending-vs-fallback structure and the same “mean-by-(R,C,time_step)” fallback, but make one metric-aligned improvement: compute the grouped mean pressure on inspiratory rows only (`u_out==0`) *and* restrict the grouping key to inspiratory-only time bins, then assign those predictions only to inspiratory test rows. To reduce the MAE further without changing the overall approach, we add one minimal extra feature to the fallback key (`u_in` discretized to a small bin) because pressure is strongly driven by `u_in` and averaging only by `(R,C,time_bin)` is too coarse (this should move the score down toward your target). We keep the expiratory handling score-neutral (carry-forward within breath) and preserve your nearest-pressure snapping. All paths and outputs remain the same and the script still always write a valid `submission.csv`.'
- What this solution (achieved 4.29178) has done: 'Your current score (3.89746, lower-is-better) is still far above the target (0.1417), so we should improve the fallback (non-blend) predictions with minimal, metric-aligned changes. The biggest remaining issue is that the fallback groups on a coarse `u_in_bin` and doesn’t use breath history, even though pressure strongly depends on cumulative inspired volume; we can add a single engineered feature `u_in_cum` (cumulative sum of `u_in` within each breath) discretized into bins and include it in the group key, keeping the same mean-by-group core logic. This preserves your overall blending-vs-fallback structure, snapping-to-nearest-pressure postprocessing, and inspiratory-only training aggregation while making the grouped mean substantially more informative. All paths stay the same and the script still always writes a valid `submission.csv`.'
- What this solution (achieved 4.29178) has done: 'Your current MAE (4.29178, lower-is-better) is still far above the target (0.1417), so we should improve the fallback (non-blend) predictions while keeping your “grouped mean + nearest-pressure snapping” core logic intact. The smallest high-impact fix is to make the engineered `u_in_cum` feature consistent with the scored phase: compute it only over inspiratory timesteps (`u_out==0`), because expiratory timesteps (not scored) currently inflate `u_in_cum` in test and break the train↔test group-key alignment. We then merge inspiratory-only group means onto inspiratory test rows as before, and keep expiratory handling as a simple within-breath carry-forward (score-neutral). This is a minimal semantic correction (not a modeling change) that should move the MAE down toward your target.'
- What this solution (achieved 2.19097) has done: 'We keep your current “blend if files exist, otherwise grouped-mean fallback + nearest-pressure snapping” core logic intact, but make the fallback grouping a bit less brittle so more test inspiratory rows actually find a matching group (reducing MAE toward the target). Specifically, we (1) bin `u_in_cum` more coarsely (your current per-1.0 bin is too sparse and causes lots of NaNs), and (2) add a safe hierarchical backoff: if the full key is missing, fall back to a slightly coarser group mean before finally using the global mean. These are minimal, metric-aligned changes that don’t alter your overall approach, keep expiratory handling the same (carry-forward), and should improve score from 4.29 downward.'
- What this solution (achieved 1.79953) has done: 'We fix the runtime crash by ensuring `id` is actually loaded into the training dataframe used inside `avg()` (it was missing, causing `sort_values(..., "id")` to KeyError). This is a minimal bug fix that preserves your existing blending-vs-fallback logic and all feature engineering/group-mean semantics. I also keep the submission writing unchanged but ensure `avg()` always returns exactly `["id","pressure"]` and writes `avg.csv` as before. No score-tuning changes are introduced beyond making the intended fallback run correctly end-to-end.'
- What this solution (achieved 1.66248) has done: 'Your current MAE (1.79953, lower-is-better) is still far above the target (0.1417), so we should make a small, metric-aligned improvement inside your existing fallback (grouped mean + nearest-pressure snapping) without changing the overall structure. The main weakness is that the fallback does not use any “previous pressure/trajectory” information; pressure is highly autocorrelated within a breath, so adding a single lag feature and using it only as a *backoff* (not replacing your current keys) tends to reduce error while preserving your grouping approach. Concretely, we compute `u_in_lag1` (previous `u_in` within breath) for train/test inspiratory rows, include it in an additional intermediate group mean, and fill NaNs using this lag-aware mean before the coarser fallbacks. This keeps blending behavior unchanged when blend files exist, keeps inspiratory-only training aggregation, and still writes a valid `submission.csv`.'
- What this solution (achieved 1.64338) has done: 'We keep your existing “blend if files exist, otherwise grouped-mean fallback + nearest-pressure snapping” logic unchanged, but make one minimal metric-aligned improvement to reduce MAE toward the target: compute and use a discretized lag of *pressure* (previous pressure within a breath) as an additional backoff group mean. This adds trajectory information that’s strongly predictive and still fits your current grouped-mean framework (no model/loop changes). We compute `pressure_lag1_bin` only on inspiratory train rows, and in test we approximate it using the already-predicted previous timestep (so it’s causal and doesn’t leak). Then we fill missing predictions using this lag-pressure backoff before your existing lag-u_in/mid/coarse/global fallbacks, which should safely improve score from ~1.66 downward while remaining stable and fast.'
- What this solution (achieved 1.57926) has done: 'Your current MAE (1.64338, lower-is-better) is still far above the target (0.1417), so we should make a small, legitimate improvement inside your existing fallback without changing the overall “grouped mean + backoff + nearest-pressure snapping” core logic. The most impactful minimal fix is to make the pressure-lag backoff actually usable: right now you compute `grp_p_lag` but don’t merge it early enough to help when the full-key mean is missing, so many rows never benefit from the lag-pressure conditioning. I (1) apply `grp_p_lag` as an earlier backoff *after* the causal test lag bin is computed, and (2) add one additional very-light fallback that drops `time_bin` for the lag-pressure key (still grouped-mean, still inspiratory-only) to improve match coverage without changing architecture/loops. All I/O paths remain the same and the script still writes a valid `submission.csv`.'

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
    a.pressure = a.pressure * 0.7 + b.pressure * 0.3
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a


def avg(dp):
    input_list = []
    if dp is not None and os.path.isdir(dp):
        for i in glob.iglob(f"{dp}/*"):
            if str(i).lower().endswith(".csv"):
                input_list.append(i)

    sample = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    n_expected = len(sample)

    preds = []
    for path in input_list:
        try:
            df = pd.read_csv(path)
            if "pressure" not in df.columns:
                continue
            arr = df["pressure"].to_numpy().ravel()
            if arr.shape[0] != n_expected:
                continue
            preds.append(arr)
        except Exception:
            continue

    if len(preds) > 0:
        out = sample[["id"]].copy()
        out["pressure"] = (np.sum(preds, axis=0) / len(preds)).astype(float)
        out["pressure"] = out["pressure"].apply(find_nearest)
        out.to_csv("avg.csv", index=False)
        return out

    df_test = pd.read_csv(
        "../input/ventilator-pressure-prediction/test.csv",
        usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"],
    )

    tr = df_train[
        ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    ].copy()

    tr = tr[tr["u_out"] == 0].copy()
    te = df_test.copy()

    tr = tr.sort_values(["breath_id", "time_step", "id"], kind="mergesort")
    te = te.sort_values(["breath_id", "time_step", "id"], kind="mergesort")

    tr["pressure_lag1"] = (
        tr.groupby("breath_id", sort=False)["pressure"]
        .shift(1)
        .fillna(tr["pressure"].median())
    )

    tr["u_in_lag1"] = tr.groupby("breath_id", sort=False)["u_in"].shift(1).fillna(0.0)

    te["u_in_insp"] = te["u_in"].where(te["u_out"] == 0, 0.0)
    te["u_in_cum"] = te.groupby("breath_id", sort=False)["u_in_insp"].cumsum()
    te["u_in_lag1"] = (
        te.groupby("breath_id", sort=False)["u_in_insp"].shift(1).fillna(0.0)
    )
    te.drop(columns=["u_in_insp"], inplace=True)

    tr["u_in_cum"] = tr.groupby("breath_id", sort=False)["u_in"].cumsum()

    tr["time_bin"] = (tr["time_step"] * 100).round().astype(np.int16)
    te["time_bin"] = (te["time_step"] * 100).round().astype(np.int16)

    tr["u_in_bin"] = np.round(tr["u_in"]).astype(np.int16)  # ~1.0 resolution
    te["u_in_bin"] = np.round(te["u_in"]).astype(np.int16)

    tr["u_in_lag1_bin"] = np.round(tr["u_in_lag1"]).astype(np.int16)
    te["u_in_lag1_bin"] = np.round(te["u_in_lag1"]).astype(np.int16)

    UIN_CUM_BIN_WIDTH = 10.0
    tr["u_in_cum_bin"] = np.round(tr["u_in_cum"] / UIN_CUM_BIN_WIDTH).astype(np.int16)
    te["u_in_cum_bin"] = np.round(te["u_in_cum"] / UIN_CUM_BIN_WIDTH).astype(np.int16)

    pressure_step = float(np.min(np.diff(sorted_pressures)))
    tr["pressure_lag1_bin"] = np.clip(
        np.round((tr["pressure_lag1"] - sorted_pressures[0]) / pressure_step),
        0,
        total_pressures_len - 1,
    ).astype(np.int16)

    key_full = ["R", "C", "time_bin", "u_in_bin", "u_in_cum_bin"]
    key_p_lag = ["R", "C", "time_bin", "u_in_bin", "pressure_lag1_bin"]
    key_p_lag_coarse = ["R", "C", "u_in_bin", "pressure_lag1_bin"]

    key_lag = ["R", "C", "time_bin", "u_in_bin", "u_in_lag1_bin"]
    key_mid = ["R", "C", "time_bin", "u_in_bin"]
    key_coarse = ["R", "C", "time_bin"]

    grp_full = tr.groupby(key_full, sort=False)["pressure"].mean().reset_index()
    grp_p_lag = (
        tr.groupby(key_p_lag, sort=False)["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "pressure_p_lag"})
    )
    grp_p_lag_coarse = (
        tr.groupby(key_p_lag_coarse, sort=False)["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "pressure_p_lag_coarse"})
    )

    grp_lag = (
        tr.groupby(key_lag, sort=False)["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "pressure_lag"})
    )
    grp_mid = (
        tr.groupby(key_mid, sort=False)["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "pressure_mid"})
    )
    grp_coarse = (
        tr.groupby(key_coarse, sort=False)["pressure"]
        .mean()
        .reset_index()
        .rename(columns={"pressure": "pressure_coarse"})
    )

    global_mean_insp = float(tr["pressure"].mean())

    te_insp = te[te["u_out"] == 0].copy()

    te_insp = te_insp.merge(grp_full, on=key_full, how="left")
    te_insp = te_insp.merge(grp_lag, on=key_lag, how="left")
    te_insp = te_insp.merge(grp_mid, on=key_mid, how="left")
    te_insp = te_insp.merge(grp_coarse, on=key_coarse, how="left")

    te_insp["pressure_base"] = te_insp["pressure"]
    te_insp["pressure_base"] = te_insp["pressure_base"].fillna(te_insp["pressure_lag"])
    te_insp["pressure_base"] = te_insp["pressure_base"].fillna(te_insp["pressure_mid"])
    te_insp["pressure_base"] = te_insp["pressure_base"].fillna(
        te_insp["pressure_coarse"]
    )
    te_insp["pressure_base"] = te_insp["pressure_base"].fillna(global_mean_insp)

    te_insp = te_insp.sort_values(["breath_id", "time_step", "id"], kind="mergesort")
    te_insp["pressure_lag1_test"] = (
        te_insp.groupby("breath_id", sort=False)["pressure_base"]
        .shift(1)
        .fillna(global_mean_insp)
    )
    te_insp["pressure_lag1_bin"] = np.clip(
        np.round((te_insp["pressure_lag1_test"] - sorted_pressures[0]) / pressure_step),
        0,
        total_pressures_len - 1,
    ).astype(np.int16)

    te_insp = te_insp.merge(grp_p_lag, on=key_p_lag, how="left")
    te_insp = te_insp.merge(grp_p_lag_coarse, on=key_p_lag_coarse, how="left")

    te_insp["pressure"] = te_insp["pressure"].fillna(te_insp["pressure_p_lag"])
    te_insp["pressure"] = te_insp["pressure"].fillna(te_insp["pressure_p_lag_coarse"])
    te_insp["pressure"] = te_insp["pressure"].fillna(te_insp["pressure_lag"])
    te_insp["pressure"] = te_insp["pressure"].fillna(te_insp["pressure_mid"])
    te_insp["pressure"] = te_insp["pressure"].fillna(te_insp["pressure_coarse"])
    te_insp["pressure"] = te_insp["pressure"].fillna(global_mean_insp)

    te = te.copy()
    te["pressure"] = np.nan
    te = te.merge(
        te_insp[["id", "pressure"]], on="id", how="left", suffixes=("", "_insp")
    )
    te.loc[te["u_out"] == 0, "pressure"] = te.loc[te["u_out"] == 0, "pressure_insp"]
    te.drop(columns=["pressure_insp"], inplace=True)

    te = te.sort_values(["breath_id", "time_step", "id"], kind="mergesort")
    te["pressure"] = (
        te.groupby("breath_id")["pressure"].ffill().fillna(global_mean_insp)
    )

    out = sample[["id"]].merge(te[["id", "pressure"]], on="id", how="left")
    out["pressure"] = out["pressure"].astype(float).apply(find_nearest)
    out.to_csv("avg.csv", index=False)
    return out




## === cell 2
sub = avg("../input/gb-data-blending-recover")

assert isinstance(sub, pd.DataFrame)
assert list(sub.columns) == ["id", "pressure"]
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
