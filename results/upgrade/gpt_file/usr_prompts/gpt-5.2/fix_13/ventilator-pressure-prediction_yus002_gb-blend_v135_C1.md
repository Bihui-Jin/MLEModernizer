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

0.1392686173419423

# 6. Current score

5.43372

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 6.97485) has done: 'Your notebook fails because it tries to read external Kaggle Dataset files (`gb-data-blending-recover/...`) that are not present in this environment. I keep your existing blending/rounding-to-nearest-pressure logic intact, but add a safe fallback that generates a baseline prediction directly from the provided `train.csv`/`test.csv` when those files are missing. This ensures the notebook runs end-to-end, always writes a valid `submission.csv` with the correct columns, and avoids changing semantics beyond what’s necessary to produce a submission. The fallback uses a simple `u_in`→pressure median mapping (train inspiratory only) and then applies your existing `find_nearest` quantization.'
- What this solution (achieved 2.85294) has done: 'Your current score (6.97 MAE) is far worse than the target (0.139), so we need a real modeling improvement rather than file/IO fixes. I keep your core “map test rows to a pressure estimate then quantize with `find_nearest`” logic, but replace the overly-weak `u_in`-only median map with a breath-state-aware feature map that better matches the physics: use cumulative inhaled volume proxy (`cum_u_in`), time within breath, and lung attributes (`R`,`C`) while still using a fast non-ML aggregation approach. Concretely, we build a median lookup table on train inspiratory rows grouped by `(R,C,time_step,cum_u_in_bin)` and back off smoothly to coarser groupings when a key is unseen. This stays lightweight, runs within the time limit, produces a valid `submission.csv`, and should move the MAE substantially toward your target without changing your rounding/quantization semantics.'
- What this solution (achieved 1.79356) has done: 'Your current MAE (2.85294, lower-is-better) is still far from the target (0.1393), so we need a modest but real improvement while keeping your “lookup → backoff → quantize with `find_nearest`” core logic intact. The biggest weakness is that the fallback ignores important temporal dynamics (recent input changes) and inspiratory masking, so I add two lightweight, physics-aligned features: lagged `u_in`/`u_out` and a cumulative volume proxy using trapezoidal integration over time. I also make the lookup slightly more robust by using a finer `time_step` rounding and adding a single extra backoff level, while keeping the same median-aggregation semantics and your existing rounding-to-known-pressures post-process. These changes stay purely in the fallback path, run fast, and should move the score materially toward the target without changing the overall approach.'
- What this solution (achieved 1.77071) has done: 'Your current score (1.79356 MAE; lower is better) is still far above the target (0.1393), so we should improve the fallback while keeping your existing “lookup table with backoff + find_nearest quantization” core approach unchanged. The minimal win here is to (1) build the lookup using inspiratory-only rows for both training and test-like keys (use `u_out==0` for lookups), and (2) add two tiny, physics-aligned discrete features that don’t change your approach: within-breath step index and a binned delta-`u_in` (flow change). This typically reduces ambiguity for the same integrated volume/time and makes the median maps sharper. I also keep your existing backoff cascade but insert one extra intermediate backoff keyed on `(R,C,ts_r,step, int_bin)` to reduce NaNs without jumping too coarse too early, which should move MAE meaningfully toward the target without changing the model family.'
- What this solution (achieved 1.79399) has done: 'I fix the IndexError in the fallback mapper by ensuring we always index into the prediction array with positions local to the filtered `base_df`, not the global `row_id` from the full test set. This keeps your existing lookup→backoff→quantize logic identical, but makes it robust when predicting separately for inspiratory/expiratory subsets. I also keep the external-blend path unchanged, and ensure a valid `submission.csv` is always written with the required `id,pressure` columns. No modeling/feature changes are introduced—this is a correctness fix so the notebook runs end-to-end and yields a submission.'
- What this solution (achieved 1.79399) has done: 'Your current MAE (1.79399, lower-is-better) is still far above the target (0.1393), so we should improve the fallback predictor while keeping your existing “median lookup with backoff + find_nearest quantization” approach intact. The main minimal gain is to make the cumulative-volume proxy (`int_u_in`) more physically aligned by integrating *only during inspiration* (when `u_out==0`), because pressure during expiration is not scored and mixing expiratory flow into the state hurts the lookup keys. I keep your same keying/backoff structure, but add a single extra robust feature (`int_u_in_reset`) computed with inspiratory-only integration, and use it (and its bin) in the lookup keys while preserving the rest of your logic. This is a small feature correction (not a model change) that typically reduces key collisions and should move MAE meaningfully toward the target without changing your post-processing or submission semantics.'
- What this solution (achieved 1.74104) has done: 'Your fallback predictor is still far from the target MAE, so we make a small, safe improvement that keeps your core “median lookup with backoff + find_nearest quantization” approach unchanged. The biggest low-risk gain is to stop using `time_step` as a rounded float key and instead use the within-breath `step` index (and a coarse `time_step` bin only as a backoff), which reduces key-mismatch noise because `step` is perfectly aligned between train/test. We also split the mapping by `u_out_cur` as you already do, but compute lookup keys from consistent inspiratory-integrated state (`int_u_in_reset`) and slightly tighten bins to reduce collisions. These are incremental feature/key adjustments only inside the fallback path and still write a valid `submission.csv` with the required `id,pressure` columns.'
- What this solution (achieved 1.74104) has done: 'Your current MAE (1.741) is still far above the target (0.139, lower-is-better), so we need a small but meaningful accuracy improvement without changing the overall “median lookup with backoff + find_nearest quantization” approach. The minimal change I’m making is to compute the integrated input state (`int_u_in_reset`) with the correct time delta by sorting within each `breath_id` by `time_step` before doing `diff/shift/cumsum`; without this, the integration can be inconsistent and harms the lookup keys. I also strengthen the primary lookup very slightly by adding a discretized current integrated state from the full signal (`int_bin`) as an extra key (kept within the same lookup/backoff framework) to reduce collisions when `u_in` histories differ but `int_reset_bin` matches. Everything else (same training data, same median aggregation, same backoff cascade, same pressure quantization, same submission writing) remains intact.'
- What this solution (achieved 5.44151) has done: 'I fix the runtime error in the fallback path by removing the unsupported `np.unique(..., axis=0)` call on an object array inside `snap_to_candidates`, replacing it with a pandas-based grouping that preserves identical snapping semantics. This change is purely a correctness/stability fix (no model/feature changes), so it should keep score behavior essentially the same while allowing the notebook to run end-to-end. I also ensure we always write a valid `submission.csv` with exactly `id,pressure` columns. The external blending path is left untouched and still be used if those files exist.'
- What this solution (achieved 5.43372) has done: 'Your current MAE (5.4415, lower-is-better) is far above the target (0.1393), so we need a real accuracy gain but must keep your core “median lookup with backoff + candidate snapping + find_nearest quantization” logic intact. The biggest low-risk improvement is to stop relying on `int_reset_bin`/`int_bin` that can be noisy and instead key primarily on a deterministic within-breath cumulative sum of `u_in` during inspiration (`cum_u_in_insp`), which is perfectly aligned between train/test and often correlates better with pressure than time-integral under irregular dt. I keep your whole mapping/backoff structure, but swap in this more stable state variable (plus a small bin) as the main lookup driver and add one intermediate backoff keyed on `(R,C,step,cum_bin)` to reduce fall-through. This is still the same non-ML aggregation approach and should substantially reduce error toward your target while producing the same valid `submission.csv`.'

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
    a.pressure = a.pressure * 0.65 + b.pressure * 0.35
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
def make_fallback_submission(out_path="submission.csv"):
    test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")

    tr_insp = df_train[df_train["u_out"] == 0].copy()
    tr_exp = df_train[df_train["u_out"] == 1].copy()
    te = test.copy()

    for df in (tr_insp, tr_exp, te):
        df.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")

        df["step"] = df.groupby("breath_id").cumcount().astype(np.int16)

        df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1).fillna(0.0)
        df["u_out_lag1"] = (
            df.groupby("breath_id")["u_out"].shift(1).fillna(0).astype(np.int8)
        )
        df["du_in"] = (df["u_in"] - df["u_in_lag1"]).fillna(0.0)

        df["uin_bin"] = (df["u_in"] / 1.0).round().astype(np.int16)

        df["u_out_cur"] = df["u_out"].astype(np.int8)

        df["ts_bin"] = (df["time_step"] / 0.02).round().astype(np.int16)

        u_in_insp = df["u_in"].where(df["u_out"] == 0, 0.0)
        df["cum_u_in_insp"] = (
            u_in_insp.groupby(df["breath_id"]).cumsum().astype(np.float32)
        )

    def add_int_u_in(df):
        dt = df.groupby("breath_id")["time_step"].diff().fillna(0.0)
        u_prev = df.groupby("breath_id")["u_in"].shift(1).fillna(0.0)
        df["int_u_in"] = (
            ((df["u_in"] + u_prev) * 0.5 * dt).groupby(df["breath_id"]).cumsum()
        )
        return df

    def add_int_u_in_insp_only(df):
        dt = df.groupby("breath_id")["time_step"].diff().fillna(0.0)
        u_prev = df.groupby("breath_id")["u_in"].shift(1).fillna(0.0)
        area = ((df["u_in"] + u_prev) * 0.5 * dt).where(df["u_out"] == 0, 0.0)
        df["int_u_in_reset"] = area.groupby(df["breath_id"]).cumsum()
        return df

    tr_insp = add_int_u_in(tr_insp)
    tr_exp = add_int_u_in(tr_exp)
    te = add_int_u_in(te)

    tr_insp = add_int_u_in_insp_only(tr_insp)
    tr_exp = add_int_u_in_insp_only(tr_exp)
    te = add_int_u_in_insp_only(te)

    bin_scale = 0.10
    tr_insp["int_bin"] = (tr_insp["int_u_in"] / bin_scale).round().astype(np.int32)
    tr_exp["int_bin"] = (tr_exp["int_u_in"] / bin_scale).round().astype(np.int32)
    te["int_bin"] = (te["int_u_in"] / bin_scale).round().astype(np.int32)

    tr_insp["int_reset_bin"] = (
        (tr_insp["int_u_in_reset"] / bin_scale).round().astype(np.int32)
    )
    tr_exp["int_reset_bin"] = (
        (tr_exp["int_u_in_reset"] / bin_scale).round().astype(np.int32)
    )
    te["int_reset_bin"] = (te["int_u_in_reset"] / bin_scale).round().astype(np.int32)

    tr_insp["uinlag_bin"] = (tr_insp["u_in_lag1"] / 1.0).round().astype(np.int16)
    tr_exp["uinlag_bin"] = (tr_exp["u_in_lag1"] / 1.0).round().astype(np.int16)
    te["uinlag_bin"] = (te["u_in_lag1"] / 1.0).round().astype(np.int16)

    tr_insp["du_bin"] = (tr_insp["du_in"] / 1.0).round().astype(np.int16)
    tr_exp["du_bin"] = (tr_exp["du_in"] / 1.0).round().astype(np.int16)
    te["du_bin"] = (te["du_in"] / 1.0).round().astype(np.int16)

    cum_scale = 5.0
    tr_insp["cum_bin"] = (tr_insp["cum_u_in_insp"] / cum_scale).round().astype(np.int32)
    tr_exp["cum_bin"] = (tr_exp["cum_u_in_insp"] / cum_scale).round().astype(np.int32)
    te["cum_bin"] = (te["cum_u_in_insp"] / cum_scale).round().astype(np.int32)

    key_cols = [
        "R",
        "C",
        "step",
        "cum_bin",
        "uin_bin",
        "uinlag_bin",
        "du_bin",
        "u_out_lag1",
        "u_out_cur",
    ]

    def build_maps(tr):
        med_primary = tr.groupby(key_cols, sort=False)["pressure"].median()

        med_rc_step_cum = tr.groupby(["R", "C", "step", "cum_bin"], sort=False)[
            "pressure"
        ].median()

        med_rc_cum = tr.groupby(["R", "C", "cum_bin"], sort=False)["pressure"].median()
        med_step_cum = tr.groupby(["step", "cum_bin"], sort=False)["pressure"].median()
        med_rc_tsb_cum = tr.groupby(["R", "C", "ts_bin", "cum_bin"], sort=False)[
            "pressure"
        ].median()
        med_rc = tr.groupby(["R", "C"], sort=False)["pressure"].median()
        global_median = float(tr["pressure"].median())
        return (
            med_primary,
            med_rc_step_cum,
            med_rc_cum,
            med_step_cum,
            med_rc_tsb_cum,
            med_rc,
            global_median,
        )

    maps_insp = build_maps(tr_insp)
    maps_exp = build_maps(tr_exp)

    base = te[["id", "u_out", "ts_bin"] + key_cols].copy()

    def predict_with_maps(base_df, maps):
        base_df = base_df.copy()
        base_df["pos"] = np.arange(len(base_df), dtype=np.int32)

        (
            med_primary,
            med_rc_step_cum,
            med_rc_cum,
            med_step_cum,
            med_rc_tsb_cum,
            med_rc,
            global_median,
        ) = maps

        pred = (
            base_df.merge(
                med_primary.rename("p"),
                left_on=key_cols,
                right_index=True,
                how="left",
            )
            .sort_values("pos")["p"]
            .to_numpy()
        )

        if np.isnan(pred).any():
            nan_mask = np.isnan(pred)
            tmp = base_df.loc[nan_mask, ["pos", "R", "C", "step", "cum_bin"]].merge(
                med_rc_step_cum.rename("p"),
                left_on=["R", "C", "step", "cum_bin"],
                right_index=True,
                how="left",
            )
            pred[tmp["pos"].to_numpy(dtype=np.int64)] = tmp["p"].to_numpy()

        if np.isnan(pred).any():
            nan_mask = np.isnan(pred)
            tmp = base_df.loc[nan_mask, ["pos", "R", "C", "cum_bin"]].merge(
                med_rc_cum.rename("p"),
                left_on=["R", "C", "cum_bin"],
                right_index=True,
                how="left",
            )
            pred[tmp["pos"].to_numpy(dtype=np.int64)] = tmp["p"].to_numpy()

        if np.isnan(pred).any():
            nan_mask = np.isnan(pred)
            tmp = base_df.loc[nan_mask, ["pos", "step", "cum_bin"]].merge(
                med_step_cum.rename("p"),
                left_on=["step", "cum_bin"],
                right_index=True,
                how="left",
            )
            pred[tmp["pos"].to_numpy(dtype=np.int64)] = tmp["p"].to_numpy()

        if np.isnan(pred).any():
            nan_mask = np.isnan(pred)
            tmp = base_df.loc[nan_mask, ["pos", "R", "C", "ts_bin", "cum_bin"]].merge(
                med_rc_tsb_cum.rename("p"),
                left_on=["R", "C", "ts_bin", "cum_bin"],
                right_index=True,
                how="left",
            )
            pred[tmp["pos"].to_numpy(dtype=np.int64)] = tmp["p"].to_numpy()

        if np.isnan(pred).any():
            nan_mask = np.isnan(pred)
            tmp = base_df.loc[nan_mask, ["pos", "R", "C"]].merge(
                med_rc.rename("p"),
                left_on=["R", "C"],
                right_index=True,
                how="left",
            )
            pred[tmp["pos"].to_numpy(dtype=np.int64)] = tmp["p"].to_numpy()

        pred = np.where(np.isnan(pred), global_median, pred).astype(float)
        return pred

    pred = np.empty(len(base), dtype=float)
    insp_mask = base["u_out"].to_numpy() == 0
    exp_mask = ~insp_mask

    if insp_mask.any():
        pred[insp_mask] = predict_with_maps(base.loc[insp_mask], maps_insp)
    if exp_mask.any():
        pred[exp_mask] = predict_with_maps(base.loc[exp_mask], maps_exp)

    def build_context_candidates(tr, topk=7):
        ctx_cols = ["R", "C", "step", "u_out_cur"]
        vc = (
            tr.groupby(ctx_cols, sort=False)["pressure"]
            .value_counts()
            .rename("cnt")
            .reset_index()
        )
        vc.sort_values(
            ctx_cols + ["cnt"], ascending=[True, True, True, True, False], inplace=True
        )
        vc = vc.groupby(ctx_cols, sort=False).head(topk)
        cand = (
            vc.groupby(ctx_cols, sort=False)["pressure"]
            .apply(lambda s: np.sort(s.to_numpy(dtype=np.float32)))
            .to_dict()
        )
        return cand

    cand_insp = build_context_candidates(tr_insp, topk=7)
    cand_exp = build_context_candidates(tr_exp, topk=7)

    base_ctx = te[["R", "C", "step", "u_out_cur"]].copy()

    def snap_to_candidates(pred_arr, ctx_df, cand_map, fallback_sorted_pressures):
        out = pred_arr.copy()
        tmp = ctx_df.reset_index(drop=True).copy()
        tmp["pos"] = np.arange(len(tmp), dtype=np.int32)

        for (R, C, step, uoc), grp in tmp.groupby(
            ["R", "C", "step", "u_out_cur"], sort=False
        ):
            cands = cand_map.get((int(R), int(C), int(step), int(uoc)))
            if cands is None or len(cands) == 0:
                continue
            idx = grp["pos"].to_numpy(dtype=np.int64)
            v = out[idx]

            ins = np.searchsorted(cands, v)
            ins = np.clip(ins, 0, len(cands) - 1)
            upper = cands[ins]
            lower = cands[np.maximum(ins - 1, 0)]
            out[idx] = np.where(np.abs(v - lower) <= np.abs(v - upper), lower, upper)

        return out

    if insp_mask.any():
        pred[insp_mask] = snap_to_candidates(
            pred[insp_mask],
            base_ctx.loc[insp_mask],
            cand_insp,
            sorted_pressures,
        )
    if exp_mask.any():
        pred[exp_mask] = snap_to_candidates(
            pred[exp_mask],
            base_ctx.loc[exp_mask],
            cand_exp,
            sorted_pressures,
        )

    sub = pd.DataFrame({"id": te["id"].values, "pressure": pred})
    sub["pressure"] = sub["pressure"].apply(find_nearest)
    sub.to_csv(out_path, index=False)
    return sub


a = "../input/gb-data-blending-recover/0.1384 seed 3133.csv"
b = "../input/gb-data-blending-recover/0.1388 blend.csv"

if os.path.exists(a) and os.path.exists(b):
    blend(a, b)  # writes blend.csv
    pd.read_csv("blend.csv").to_csv("submission.csv", index=False)
else:
    make_fallback_submission("submission.csv")
