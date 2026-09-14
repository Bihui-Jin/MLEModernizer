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

2.83685

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
        seg_id = df.groupby("breath_id")["u_out"].cumsum().astype(np.int16)
        df["seg_id"] = seg_id
        df["u_in_cumsum"] = (
            df.groupby(["breath_id", "seg_id"])["u_in"].cumsum().astype(np.float32)
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

    key_cols = ["R", "C", "u_out", "t_idx"]
    on_col = "u_in_cumsum"

    train_ref = (
        train.groupby(key_cols + [on_col], observed=True)["pressure"]
        .median()
        .reset_index()
    )

    uq = (
        train_ref.groupby(key_cols, observed=True)[on_col]
        .quantile([0.25, 0.75])
        .unstack()
        .reset_index()
        .rename(columns={0.25: "q25", 0.75: "q75"})
    )
    uq["iqr"] = (uq["q75"] - uq["q25"]).astype(np.float32)

    train_ref_sorted = train_ref.sort_values(
        [*key_cols, on_col], kind="mergesort"
    ).reset_index(drop=True)
    train_ref_sorted["_du"] = (
        train_ref_sorted.groupby(key_cols, observed=True)[on_col].diff().abs()
    )
    du_global = float(train_ref_sorted["_du"].median(skipna=True))
    if not np.isfinite(du_global) or du_global <= 0:
        du_global = 1.0
    du_med = (
        train_ref_sorted.groupby(key_cols, observed=True)["_du"]
        .median()
        .reset_index()
        .rename(columns={"_du": "du_med"})
    )
    du_med["du_med"] = du_med["du_med"].fillna(du_global).astype(np.float32)

    du_map = uq.merge(du_med, on=key_cols, how="outer")
    du_map["iqr"] = du_map["iqr"].fillna(0.0).astype(np.float32)
    du_map["du_med"] = du_map["du_med"].fillna(du_global).astype(np.float32)

    du_map["max_dist"] = (0.08 * du_map["iqr"]).astype(np.float32)
    du_map["max_dist"] = np.maximum(
        du_map["max_dist"].to_numpy(), (2.0 * du_map["du_med"]).to_numpy()
    ).astype(np.float32)
    du_map["max_dist"] = np.minimum(
        du_map["max_dist"].to_numpy(), (6.0 * du_map["du_med"]).to_numpy()
    ).astype(np.float32)

    du_map = du_map[key_cols + ["max_dist"]].copy()
    del train_ref_sorted
    gc.collect()

    test_ref = test[key_cols + [on_col, "id"]].copy()
    test_ref = test_ref.merge(du_map, on=key_cols, how="left")
    test_ref["max_dist"] = (
        test_ref["max_dist"]
        .fillna(np.float32(max(2.0 * du_global, 1.0)))
        .astype(np.float32)
    )

    sort_cols_asof = [on_col] + key_cols
    train_ref = train_ref.sort_values(sort_cols_asof, kind="mergesort").reset_index(
        drop=True
    )
    test_ref = test_ref.sort_values(sort_cols_asof, kind="mergesort").reset_index(
        drop=True
    )

    assert train_ref[on_col].is_monotonic_increasing, "train_ref on-key not sorted"
    assert test_ref[on_col].is_monotonic_increasing, "test_ref on-key not sorted"

    left = test_ref.copy()
    right = train_ref.copy()

    back = pd.merge_asof(
        left,
        right,
        on=on_col,
        by=key_cols,
        direction="backward",
        allow_exact_matches=True,
        suffixes=("", "_r"),
    ).rename(columns={"pressure": "pred_back"})

    fwd = pd.merge_asof(
        left,
        right,
        on=on_col,
        by=key_cols,
        direction="forward",
        allow_exact_matches=True,
        suffixes=("", "_r"),
    ).rename(columns={"pressure": "pred_fwd"})

    comb = left[["id"] + key_cols + [on_col, "max_dist"]].copy()
    comb["pred_back"] = back["pred_back"].to_numpy()
    comb["pred_fwd"] = fwd["pred_fwd"].to_numpy()

    right_back = right.rename(columns={on_col: "u_match"})
    back_u = pd.merge_asof(
        left,
        right_back[[*key_cols, "u_match", "pressure"]],
        left_on=on_col,
        right_on="u_match",
        by=key_cols,
        direction="backward",
        allow_exact_matches=True,
    )
    fwd_u = pd.merge_asof(
        left,
        right_back[[*key_cols, "u_match", "pressure"]],
        left_on=on_col,
        right_on="u_match",
        by=key_cols,
        direction="forward",
        allow_exact_matches=True,
    )

    comb["u_back"] = back_u["u_match"].to_numpy()
    comb["u_fwd"] = fwd_u["u_match"].to_numpy()

    dist_back = (comb[on_col] - comb["u_back"]).abs()
    dist_fwd = (comb["u_fwd"] - comb[on_col]).abs()

    choose_back = dist_back <= dist_fwd
    pred = np.where(
        choose_back, comb["pred_back"].to_numpy(), comb["pred_fwd"].to_numpy()
    )
    dist = np.where(choose_back, dist_back.to_numpy(), dist_fwd.to_numpy())

    pred = np.where(dist <= comb["max_dist"].to_numpy(), pred, np.nan)

    test_pred = comb.drop(columns=["pred_back", "pred_fwd", "u_back", "u_fwd"]).copy()
    test_pred["pred"] = pred.astype(np.float32)

    if test_pred["pred"].isna().any():
        on_col_s = "u_in_cumsum_smooth"
        key_cols_s = ["R", "C", "u_out", "t_idx"]

        train_ref_s = (
            train.groupby(key_cols_s + [on_col_s], observed=True)["pressure"]
            .median()
            .reset_index()
        )
        sort_cols_asof_s = [on_col_s] + key_cols_s
        train_ref_s = train_ref_s.sort_values(
            sort_cols_asof_s, kind="mergesort"
        ).reset_index(drop=True)

        tmp = test.loc[:, ["id"] + key_cols_s + [on_col_s]].copy()
        tmp = tmp.sort_values(sort_cols_asof_s, kind="mergesort").reset_index(drop=True)

        tmp_s = pd.merge_asof(
            tmp,
            train_ref_s,
            on=on_col_s,
            by=key_cols_s,
            direction="nearest",
            allow_exact_matches=True,
        ).rename(columns={"pressure": "pred_s"})

        test_pred = test_pred.merge(tmp_s[["id", "pred_s"]], on="id", how="left")
        test_pred["pred"] = test_pred["pred"].fillna(test_pred["pred_s"])
        test_pred.drop(columns=["pred_s"], inplace=True)

    if test_pred["pred"].isna().any():
        train_ref2 = (
            train.groupby(["R", "C", "t_idx", on_col], observed=True)["pressure"]
            .median()
            .reset_index()
        )

        sort_cols_asof2 = [on_col, "R", "C", "t_idx"]
        train_ref2 = train_ref2.sort_values(
            sort_cols_asof2, kind="mergesort"
        ).reset_index(drop=True)

        tmp = test_pred[["id", "R", "C", "t_idx", on_col]].copy()
        tmp = tmp.sort_values(sort_cols_asof2, kind="mergesort").reset_index(drop=True)

        tmp2 = pd.merge_asof(
            tmp,
            train_ref2,
            on=on_col,
            by=["R", "C", "t_idx"],
            direction="nearest",
            allow_exact_matches=True,
        ).rename(columns={"pressure": "pred2"})

        test_pred = test_pred.merge(tmp2[["id", "pred2"]], on="id", how="left")
        test_pred["pred"] = test_pred["pred"].fillna(test_pred["pred2"])
        test_pred.drop(columns=["pred2"], inplace=True)

    if test_pred["pred"].isna().any():
        train_ref3 = (
            train.groupby(["R", "C", on_col], observed=True)["pressure"]
            .median()
            .reset_index()
        )
        sort_cols_asof3 = [on_col, "R", "C"]
        train_ref3 = train_ref3.sort_values(
            sort_cols_asof3, kind="mergesort"
        ).reset_index(drop=True)

        tmp = test_pred[["id", "R", "C", on_col]].copy()
        tmp = tmp.sort_values(sort_cols_asof3, kind="mergesort").reset_index(drop=True)

        tmp3 = pd.merge_asof(
            tmp,
            train_ref3,
            on=on_col,
            by=["R", "C"],
            direction="nearest",
            allow_exact_matches=True,
        ).rename(columns={"pressure": "pred3"})

        test_pred = test_pred.merge(tmp3[["id", "pred3"]], on="id", how="left")
        test_pred["pred"] = test_pred["pred"].fillna(test_pred["pred3"])
        test_pred.drop(columns=["pred3"], inplace=True)

    global_median = float(train["pressure"].median())
    test_pred["pred"] = test_pred["pred"].fillna(global_median)

    test_pred["pressure"] = test_pred["pred"].apply(find_nearest).astype(float)

    submission = test_pred[["id", "pressure"]].copy().sort_values("id")
    submission.to_csv("submission.csv", index=False)
    print(
        "Wrote submission.csv (fallback lookup with reset-cumsum + smoothed-cumsum backoff; same snapping; same asof/backoffs)."
    )
    print(submission.head())
    print(submission.shape)
