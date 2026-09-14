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

0.1448891434702308

# 6. Current score

2.97085

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.73281) has done: 'Your notebook fails because it tries to read external blend CSVs from a Kaggle Dataset (`gb-data-blending-recover`) that is not available in this environment, so no submission is produced. I keep the existing blending/rounding logic intact, but add a safe fallback that generates a baseline prediction directly from the provided train/test files when those external files are missing. The fallback uses the same “snap predictions to nearest valid pressure” approach already in your code, and writes a valid `submission.csv` with the required `id,pressure` columns. This should run end-to-end within the time limit and yield a reasonable score (though likely not as strong as the missing blend files).'
- What this solution (achieved 2.11849) has done: 'Your current fallback is effectively a high-cardinality exact-lookup on continuous `u_in` and `time_step`, which causes almost everything to miss and collapse to the global median—hence the very poor MAE. I keep your “snap to nearest valid pressure” core logic intact, but change the fallback mapping to use only stable/discrete keys (`R`, `C`, `u_out`, and time index within breath) and add a simple per-breath cumulative `u_in` feature that is known to correlate with pressure. This is a minimal change that should dramatically improve the score toward your target without changing the overall approach (still a train-derived lookup + nearest-pressure rounding). The script still write a valid `submission.csv` end-to-end.'
- What this solution (achieved 2.83529) has done: 'I fix the `merge_asof` failure by ensuring both the left and right DataFrames are sorted exactly as pandas requires: primarily by the `on` key and secondarily by the `by` keys, within each group. This is a runtime correctness fix (no change to the core “train-derived lookup + nearest valid pressure rounding” logic), and it allow the notebook to complete and write a valid `submission.csv`. I also add a small, safe assertion to catch any future sorting regressions early, and keep all paths/outputs unchanged. No modeling or feature logic is altered beyond the ordering needed for `merge_asof` to run.'
- What this solution (achieved 2.83529) has done: 'Your fallback currently forces `pressure=0` whenever `u_out==1`, but the competition metric ignores expiratory-phase timesteps (where `u_out==1`) so that rule tends to harm MAE by injecting large errors on any inspiratory timesteps that still have `u_out==1` in test. I keep your core approach (train-derived lookup via `merge_asof` on cumulative `u_in` + snapping to nearest valid pressure) but remove that hard override and instead rely on the lookup (with the existing backoff levels + global median) for all rows. I also add one more minimal backoff level (drop `t_idx`) only for remaining NaNs to reduce misses without changing the overall method. This should move your MAE substantially down toward the 0.1449 target while staying within the same modeling semantics and still producing a valid `submission.csv`.'
- What this solution (achieved 2.83625) has done: 'Your current fallback still misses too often because `merge_asof` on continuous `u_in_cumsum` can map to a “nearest” value that’s not actually close, especially when the local spacing differs by `(R,C,u_out,t_idx)` group; this keeps MAE high. I keep the same core approach (train-derived lookup + `merge_asof` + snap-to-nearest valid pressure) but add an explicit per-group distance gate: only accept the asof match if the absolute `u_in_cumsum` difference is small, otherwise treat it as missing and let your existing backoff levels handle it. This is a minimal, metric-aligned change that should reduce wrong matches (big errors) and move the score down toward the 0.1449 target without changing the model/logic family. I also compute the gate threshold from training data per group using adjacent `u_in_cumsum` spacing so it adapts safely across groups.'
- What this solution (achieved 2.8356) has done: 'Your current fallback is still fundamentally a “train median lookup by keys + nearest u_in_cumsum + snap-to-valid-pressure”, but it’s being hurt by (1) a bug where `_dist` is always 0 (so the nearest-match quality check is effectively disabled) and (2) using a very loose, per-group spacing-based distance gate that still allows many wrong matches. I fix the distance computation and replace the gate with a tighter, empirically safer per-group tolerance derived from the within-(R,C,u_out,t_idx) u_in_cumsum distribution (IQR-based), which should reject bad matches and force your existing backoff levels to handle them. I also ensure all `merge_asof` inputs are sorted exactly as required and keep the rest of your blending/rounding/backoff logic unchanged. These are minimal changes aimed at reducing large errors and moving MAE down toward your target.'
- What this solution (achieved 2.83685) has done: 'Your current fallback is still scoring poorly because it relies on a continuous `u_in_cumsum` nearest-neighbor match, which can pick the “closest” cumulative value but still map to an incorrect pressure when the mapping is not locally consistent; the current gating helps but not enough. To move MAE substantially down toward your 0.1449 target without changing the overall lookup-based approach, I keep your `merge_asof` + backoff structure and your “snap to nearest valid pressure” step, but I add one additional, very standard ventilator feature: within-breath `u_in` lag and integrate it (`u_in_cumsum`) only after resetting at `u_out==1` (new inspiratory segment), which better matches how pressure evolves and reduces mismatches. I also add a single extra backoff that uses the same keys but matches on this reset-cumsum (still `merge_asof`, still medians), and only applies where the existing prediction is NaN—so the core logic remains intact and changes are minimal. This should improve coverage with more correct local matches and reduce large errors, moving your score meaningfully toward the target band while still producing a valid `submission.csv`.'
- What this solution (achieved 3.15753) has done: 'I fix the runtime error in `build_du_map` caused by mixing NumPy arrays and pandas Series (`to_numpy()` on an ndarray). The change is minimal and keeps your existing lookup + gated `merge_asof` + backoff + “snap to nearest valid pressure” logic intact. I also add a small guard to ensure `max_dist` is always finite and positive to prevent unexpected all-NaN predictions. The pipeline then run end-to-end and write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 3.15753) has done: 'I fix the `merge_asof` crash by enforcing the exact sorting pandas requires: globally sorted by the `on` column first, and then by the `by` keys, for both left and right frames. The current code sorts by `key_cols` then `on_col`, which can still violate the global monotonic requirement and triggers “left keys must be sorted”. I also ensure the `_asof` helper frames used for distance computation are sorted the same way, keeping your lookup + gated nearest-asof + backoff + snapping logic unchanged. This make the notebook run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 2.97085) has done: 'Your current score is far worse than the target (lower is better), so we need a real accuracy lift while keeping your same lookup + gated `merge_asof` + backoff + snap-to-valid-pressure core. The biggest issue is that the current gate is too strict/misaligned: it computes distance using a separately-built asof join and then re-attaches `max_dist` via an extra merge, which can silently misalign and reject many good matches, collapsing you into coarse backoffs/global median. I make the gate use the exact matched `u_match` coming from the same `merge_asof` result (no misalignment), and I compute `max_dist` in a more metric-aligned way based on within-group nearest-neighbor spacing (so “acceptable nearest” is neither overly tight nor overly loose). These are minimal changes that preserve your approach and should materially reduce MAE toward the target while keeping runtime reasonable and still writing a valid `submission.csv`.'

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
    a.pressure = a.pressure * 0.6 + b.pressure * 0.4
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
a = "../input/gb-data-blending-recover/0.1426 blend.csv"
b = "../input/gb-data-blending-recover/0.144 blend.csv"

if os.path.exists(a) and os.path.exists(b):
    sub = blend(a, b)
    sub[["id", "pressure"]].to_csv("submission.csv", index=False)
else:
    train = df_train.copy()
    test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")

    train["t_idx"] = train.groupby("breath_id").cumcount().astype(np.int16)
    test["t_idx"] = test.groupby("breath_id").cumcount().astype(np.int16)

    for df in (train, test):
        df["u_in_lag1"] = (
            df.groupby("breath_id")["u_in"].shift(1).fillna(0.0).astype(np.float32)
        )
        df["u_in_insp"] = (df["u_in"] * (1 - df["u_out"])).astype(np.float32)
        df["u_in_insp_lag1"] = (
            df.groupby("breath_id")["u_in_insp"].shift(1).fillna(0.0).astype(np.float32)
        )

        seg_id = df.groupby("breath_id")["u_out"].cumsum().astype(np.int16)
        df["seg_id"] = seg_id

        df["u_in_cumsum"] = (
            df.groupby(["breath_id", "seg_id"])["u_in"].cumsum().astype(np.float32)
        )
        df["u_in_insp_cumsum"] = (
            df.groupby(["breath_id", "seg_id"])["u_in_insp"].cumsum().astype(np.float32)
        )

        df["u_in_smooth"] = (
            0.5 * df["u_in"].astype(np.float32) + 0.5 * df["u_in_lag1"]
        ).astype(np.float32)
        df["u_in_cumsum_smooth"] = (
            df.groupby(["breath_id", "seg_id"])["u_in_smooth"]
            .cumsum()
            .astype(np.float32)
        )

    for col in ["R", "C", "u_out"]:
        train[col] = train[col].astype(np.int16)
        test[col] = test[col].astype(np.int16)

    def build_du_map(train_ref, key_cols, on_col):
        tr_sorted = train_ref.sort_values(
            [*key_cols, on_col], kind="mergesort"
        ).reset_index(drop=True)

        tr_sorted["_du"] = (
            tr_sorted.groupby(key_cols, observed=True)[on_col].diff().abs()
        )
        du_global = float(tr_sorted["_du"].median(skipna=True))
        if not np.isfinite(du_global) or du_global <= 0:
            du_global = 1.0

        du_med = (
            tr_sorted.groupby(key_cols, observed=True)["_du"]
            .median()
            .reset_index()
            .rename(columns={"_du": "du_med"})
        )
        du_med["du_med"] = du_med["du_med"].fillna(du_global).astype(np.float32)

        uq = (
            train_ref.groupby(key_cols, observed=True)[on_col]
            .quantile([0.25, 0.75])
            .unstack()
            .reset_index()
            .rename(columns={0.25: "q25", 0.75: "q75"})
        )
        uq["iqr"] = (uq["q75"] - uq["q25"]).astype(np.float32)

        du_map = uq.merge(du_med, on=key_cols, how="outer")
        du_map["iqr"] = du_map["iqr"].fillna(0.0).astype(np.float32)
        du_map["du_med"] = du_map["du_med"].fillna(du_global).astype(np.float32)

        iqr = du_map["iqr"].to_numpy(dtype=np.float32, copy=False)
        du_med_np = du_map["du_med"].to_numpy(dtype=np.float32, copy=False)

        md = (3.0 * du_med_np + 0.01 * iqr).astype(np.float32, copy=False)
        md = np.clip(md, 1.0 * du_med_np, 10.0 * du_med_np).astype(
            np.float32, copy=False
        )

        md = np.where(
            np.isfinite(md) & (md > 0),
            md,
            np.float32(max(3.0 * du_global, 1.0)),
        )
        du_map["max_dist"] = md.astype(np.float32)

        du_map = du_map[key_cols + ["max_dist"]].copy()
        del tr_sorted
        gc.collect()
        return du_map, du_global

    def gated_asof_predict(train_ref, test_ref, key_cols, on_col, du_map, du_global):
        sort_cols = [on_col, *key_cols]
        train_ref_s = train_ref.sort_values(sort_cols, kind="mergesort").reset_index(
            drop=True
        )
        test_ref_s = test_ref.sort_values(sort_cols, kind="mergesort").reset_index(
            drop=True
        )

        tmp = test_ref_s.merge(du_map, on=key_cols, how="left")
        tmp["max_dist"] = (
            tmp["max_dist"]
            .fillna(np.float32(max(3.0 * du_global, 1.0)))
            .astype(np.float32)
        )
        tmp = tmp.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

        right = (
            train_ref_s[[*key_cols, on_col, "pressure"]]
            .copy()
            .rename(columns={on_col: "u_match"})
        )
        right = right.sort_values(["u_match", *key_cols], kind="mergesort").reset_index(
            drop=True
        )

        pred_df = pd.merge_asof(
            tmp,
            right,
            left_on=on_col,
            right_on="u_match",
            by=key_cols,
            direction="nearest",
            allow_exact_matches=True,
        )

        dist = pred_df["u_match"].to_numpy(dtype=np.float32) - pred_df[on_col].to_numpy(
            dtype=np.float32
        )
        dist = np.abs(dist)

        pred = pred_df["pressure"].to_numpy(dtype=np.float32)
        max_dist = pred_df["max_dist"].to_numpy(dtype=np.float32)
        pred = np.where(dist <= max_dist, pred, np.nan).astype(np.float32)

        out = pred_df[["id"]].copy()
        out["pred"] = pred
        return out

    key_cols = ["R", "C", "u_out", "t_idx"]
    on_col = "u_in_insp_cumsum"  # more metric-aligned than raw u_in_cumsum

    train_ref = (
        train.groupby(key_cols + [on_col], observed=True)["pressure"]
        .median()
        .reset_index()
    )
    du_map, du_global = build_du_map(train_ref, key_cols, on_col)

    test_ref = test[["id"] + key_cols + [on_col]].copy()
    test_pred = gated_asof_predict(
        train_ref, test_ref, key_cols, on_col, du_map, du_global
    )

    if test_pred["pred"].isna().any():
        on_col_s = "u_in_cumsum_smooth"
        key_cols_s = ["R", "C", "u_out", "t_idx"]
        train_ref_s = (
            train.groupby(key_cols_s + [on_col_s], observed=True)["pressure"]
            .median()
            .reset_index()
        )
        du_map_s, du_global_s = build_du_map(train_ref_s, key_cols_s, on_col_s)
        test_ref_s = test.loc[:, ["id"] + key_cols_s + [on_col_s]].copy()
        tmp_s = gated_asof_predict(
            train_ref_s, test_ref_s, key_cols_s, on_col_s, du_map_s, du_global_s
        )
        test_pred = test_pred.merge(tmp_s, on="id", how="left", suffixes=("", "_s"))
        test_pred["pred"] = test_pred["pred"].fillna(test_pred["pred_s"])
        test_pred.drop(columns=["pred_s"], inplace=True)

    if test_pred["pred"].isna().any():
        key_cols2 = ["R", "C", "t_idx"]
        on_col2 = on_col
        train_ref2 = (
            train.groupby(key_cols2 + [on_col2], observed=True)["pressure"]
            .median()
            .reset_index()
        )
        du_map2, du_global2 = build_du_map(train_ref2, key_cols2, on_col2)
        test_ref2 = test_pred.merge(
            test[["id", "R", "C", "t_idx", on_col2]], on="id", how="left"
        )[["id", *key_cols2, on_col2]]
        tmp2 = gated_asof_predict(
            train_ref2, test_ref2, key_cols2, on_col2, du_map2, du_global2
        )
        test_pred = test_pred.merge(tmp2, on="id", how="left", suffixes=("", "_2"))
        test_pred["pred"] = test_pred["pred"].fillna(test_pred["pred_2"])
        test_pred.drop(columns=["pred_2"], inplace=True)

    if test_pred["pred"].isna().any():
        key_cols3 = ["R", "C"]
        on_col3 = on_col
        train_ref3 = (
            train.groupby(key_cols3 + [on_col3], observed=True)["pressure"]
            .median()
            .reset_index()
        )
        du_map3, du_global3 = build_du_map(train_ref3, key_cols3, on_col3)
        test_ref3 = test_pred.merge(
            test[["id", "R", "C", on_col3]], on="id", how="left"
        )[["id", *key_cols3, on_col3]]
        tmp3 = gated_asof_predict(
            train_ref3, test_ref3, key_cols3, on_col3, du_map3, du_global3
        )
        test_pred = test_pred.merge(tmp3, on="id", how="left", suffixes=("", "_3"))
        test_pred["pred"] = test_pred["pred"].fillna(test_pred["pred_3"])
        test_pred.drop(columns=["pred_3"], inplace=True)

    global_median = float(train["pressure"].median())
    test_pred["pred"] = test_pred["pred"].fillna(global_median)

    test_pred["pressure"] = test_pred["pred"].apply(find_nearest).astype(float)

    submission = test_pred[["id", "pressure"]].copy().sort_values("id")
    submission.to_csv("submission.csv", index=False)
    print(
        "Wrote submission.csv (fallback lookup with inspiratory reset-cumsum + gated nearest-asof + gated backoffs; fixed distance gating alignment; same snapping)."
    )
    print(submission.head())
    print(submission.shape)
