# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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
    """
    Blend a list of submission files into one prediction vector.

    Safe for 0/1/2/N files; weights derived from filename score when possible.
    """
    if input_list is None or len(input_list) == 0:
        return None

    scores = []
    preds = []
    for path in input_list:
        df = pd.read_csv(path)
        if "pressure" not in df.columns:
            raise ValueError(f"File {path} does not contain a 'pressure' column.")
        preds.append(df["pressure"].to_numpy().ravel())

        base = os.path.basename(path)
        score_val = 1.0
        try:
            score_str = base.split(".")[1].split(" ")[0]
            score_val = float(score_str)
        except Exception:
            score_val = 1.0
        scores.append(score_val)

    if len(preds) == 1:
        return preds[0]

    l_sum = float(np.sum(scores)) if np.sum(scores) != 0 else float(len(scores))

    if len(preds) == 2:
        weight1 = (scores[1] / l_sum) + 0.1
        weight2 = 1.0 - weight1
        return preds[0] * weight1 + preds[1] * weight2

    weights = np.array(scores, dtype=float) / l_sum
    out = np.zeros_like(preds[0], dtype=float)
    for w, p in zip(weights, preds):
        out += w * p
    return out


def _fallback_predict_from_train_lookup(
    df_train_local: pd.DataFrame, df_test_local: pd.DataFrame
) -> pd.DataFrame:
    """
    Fallback used only when no external submission files are present.

    Core logic preserved: nonparametric train->median lookup with a backoff cascade,
    then later snapping to nearest allowed pressure.

    Bug fix:
    - te does not have 'pressure_l1_r'. We must not select it from te.loc[:, cols].
      Instead we inject it into the lookup key at prediction time only.

    Robustness/speed fix (score-neutral in intent, but enables completion):
    - Use tuple key lookup on the groupby Series index instead of building a
      MultiIndex + reindex per row, which is slow and error-prone.
    - Use iloc for row extraction after sorting to ensure positional correctness.
    """
    train_cols = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    test_cols = ["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"]

    tr = df_train_local[train_cols].copy()
    te = df_test_local[test_cols].copy()

    tr.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")
    te.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")

    tr["step"] = tr.groupby("breath_id").cumcount().astype(np.int16)
    te["step"] = te.groupby("breath_id").cumcount().astype(np.int16)

    tr["u_in_r"] = tr["u_in"].round(1)
    te["u_in_r"] = te["u_in"].round(1)

    tr["dt"] = (
        tr.groupby("breath_id")["time_step"].diff().fillna(0.0).astype(np.float32)
    )
    te["dt"] = (
        te.groupby("breath_id")["time_step"].diff().fillna(0.0).astype(np.float32)
    )

    tr["u_in_cum"] = tr.groupby("breath_id")["u_in_r"].cumsum().astype(np.float32)
    te["u_in_cum"] = te.groupby("breath_id")["u_in_r"].cumsum().astype(np.float32)

    tr["u_in_area"] = (
        (tr["u_in_r"] * tr["dt"]).groupby(tr["breath_id"]).cumsum()
    ).astype(np.float32)
    te["u_in_area"] = (
        (te["u_in_r"] * te["dt"]).groupby(te["breath_id"]).cumsum()
    ).astype(np.float32)

    tr["u_in_cum_r"] = tr["u_in_cum"].round(2)
    te["u_in_cum_r"] = te["u_in_cum"].round(2)
    tr["u_in_area_r"] = tr["u_in_area"].round(4)
    te["u_in_area_r"] = te["u_in_area"].round(4)
    tr["u_in_area_r2"] = tr["u_in_area"].round(2)
    te["u_in_area_r2"] = te["u_in_area"].round(2)

    tr["u_in_diff"] = tr.groupby("breath_id")["u_in_r"].diff().fillna(0.0).round(1)
    te["u_in_diff"] = te.groupby("breath_id")["u_in_r"].diff().fillna(0.0).round(1)

    tr_reset = tr["u_out"].eq(1).groupby(tr["breath_id"], sort=False).cumsum()
    te_reset = te["u_out"].eq(1).groupby(te["breath_id"], sort=False).cumsum()
    tr["since_out1"] = (
        tr.groupby(["breath_id", tr_reset], sort=False).cumcount().astype(np.int16)
    )
    te["since_out1"] = (
        te.groupby(["breath_id", te_reset], sort=False).cumcount().astype(np.int16)
    )

    for lag in (1, 2, 3):
        tr[f"u_in_l{lag}"] = tr.groupby("breath_id")["u_in_r"].shift(lag).fillna(0.0)
        te[f"u_in_l{lag}"] = te.groupby("breath_id")["u_in_r"].shift(lag).fillna(0.0)
        tr[f"u_out_l{lag}"] = (
            tr.groupby("breath_id")["u_out"].shift(lag).fillna(0).astype(np.int8)
        )
        te[f"u_out_l{lag}"] = (
            te.groupby("breath_id")["u_out"].shift(lag).fillna(0).astype(np.int8)
        )

    tr["pressure_l1"] = (
        tr.groupby("breath_id")["pressure"].shift(1).fillna(tr["pressure"])
    )
    tr["pressure_l1"] = tr["pressure_l1"].astype(np.float32)
    tr["pressure_l1_r"] = tr["pressure_l1"].round(2)

    insp_tr = tr["u_out"].eq(0)
    global_median = (
        float(tr.loc[insp_tr, "pressure"].median())
        if insp_tr.any()
        else float(tr["pressure"].median())
    )

    pred_sorted = np.full(len(te), global_median, dtype=float)

    backoffs = [
        [
            "R",
            "C",
            "step",
            "u_out",
            "since_out1",
            "u_in_r",
            "u_in_diff",
            "u_in_l1",
            "u_in_l2",
            "u_in_l3",
            "u_out_l1",
            "u_out_l2",
            "u_out_l3",
            "u_in_cum_r",
            "u_in_area_r",
            "pressure_l1_r",
        ],
        [
            "R",
            "C",
            "step",
            "u_out",
            "u_in_r",
            "u_in_diff",
            "u_in_l1",
            "u_in_l2",
            "u_in_l3",
            "u_out_l1",
            "u_out_l2",
            "u_out_l3",
            "u_in_cum_r",
            "u_in_area_r2",
            "pressure_l1_r",
        ],
        ["R", "C", "step", "u_out", "u_in_r", "u_in_l1", "u_out_l1", "pressure_l1_r"],
        [
            "R",
            "C",
            "step",
            "u_out",
            "since_out1",
            "u_in_r",
            "u_in_diff",
            "u_in_l1",
            "u_in_l2",
            "u_in_l3",
            "u_out_l1",
            "u_out_l2",
            "u_out_l3",
            "u_in_cum_r",
            "u_in_area_r",
        ],
        [
            "R",
            "C",
            "step",
            "u_out",
            "since_out1",
            "u_in_r",
            "u_in_diff",
            "u_in_l1",
            "u_in_l2",
            "u_in_l3",
            "u_out_l1",
            "u_out_l2",
            "u_out_l3",
            "u_in_cum_r",
        ],
        [
            "R",
            "C",
            "step",
            "u_out",
            "u_in_r",
            "u_in_diff",
            "u_in_l1",
            "u_in_l2",
            "u_in_l3",
            "u_out_l1",
            "u_out_l2",
            "u_out_l3",
            "u_in_cum_r",
            "u_in_area_r",
        ],
        [
            "R",
            "C",
            "step",
            "u_out",
            "since_out1",
            "u_in_r",
            "u_in_diff",
            "u_in_l1",
            "u_in_l2",
            "u_in_l3",
            "u_out_l1",
            "u_out_l2",
            "u_out_l3",
            "u_in_cum_r",
        ],
        [
            "R",
            "C",
            "step",
            "u_out",
            "u_in_r",
            "u_in_l1",
            "u_in_l2",
            "u_out_l1",
            "u_out_l2",
            "u_in_cum_r",
            "u_in_area_r2",
        ],
        [
            "R",
            "C",
            "step",
            "u_out",
            "since_out1",
            "u_in_r",
            "u_in_l1",
            "u_in_l2",
            "u_out_l1",
            "u_out_l2",
            "u_in_area_r2",
        ],
        [
            "R",
            "C",
            "step",
            "u_out",
            "since_out1",
            "u_in_r",
            "u_in_l1",
            "u_in_l2",
            "u_out_l1",
            "u_out_l2",
            "u_in_cum_r",
        ],
        [
            "R",
            "C",
            "step",
            "u_out",
            "since_out1",
            "u_in_r",
            "u_in_l1",
            "u_in_l2",
            "u_out_l1",
            "u_out_l2",
        ],
        ["R", "C", "step", "u_out", "since_out1", "u_in_r"],
        ["R", "C", "step", "u_out"],
        ["R", "C", "step", "u_in_r"],
    ]

    lookups = []
    for cols in backoffs:
        use_insp_only = ("u_out" in cols) or (cols == ["R", "C", "step", "u_in_r"])
        tr_sub = tr.loc[insp_tr] if use_insp_only and insp_tr.any() else tr
        lookups.append(tr_sub.groupby(cols, sort=False)["pressure"].median())

    te_breaths = te["breath_id"].to_numpy()
    te_steps = te["step"].to_numpy()

    start_positions = np.flatnonzero(te_steps == 0)
    for s in start_positions:
        b = te_breaths[s]
        e = s + 1
        while e < len(te) and te_breaths[e] == b:
            e += 1

        prev_p = global_median
        for i in range(s, e):
            pressure_l1_r = float(np.round(prev_p, 2))

            filled = False
            for cols, lookup in zip(backoffs, lookups):
                if "pressure_l1_r" in cols:
                    cols_wo = [c for c in cols if c != "pressure_l1_r"]
                    row_vals = te.iloc[i][cols_wo].to_list()
                    key = tuple(row_vals + [pressure_l1_r])
                else:
                    row_vals = te.iloc[i][cols].to_list()
                    key = tuple(row_vals)

                val = lookup.get(key, np.nan)
                if not (val is np.nan or pd.isna(val)):
                    pred_sorted[i] = float(val)
                    filled = True
                    break

            if not filled:
                pred_sorted[i] = global_median

            prev_p = pred_sorted[i]

    out_df = te[["id", "breath_id", "step", "u_out"]].copy()
    out_df["pressure_pred"] = pred_sorted

    out_df["is_exp"] = out_df["u_out"].eq(1)

    exp_start = (
        out_df.loc[out_df["is_exp"], ["breath_id", "step", "pressure_pred"]]
        .sort_values(["breath_id", "step"], kind="mergesort")
        .groupby("breath_id", sort=False)
        .first()
        .rename(columns={"pressure_pred": "exp_start_pred"})
    )
    out_df = out_df.merge(exp_start[["exp_start_pred"]], on="breath_id", how="left")

    last_insp = (
        out_df.loc[~out_df["is_exp"], ["breath_id", "step", "pressure_pred"]]
        .sort_values(["breath_id", "step"], kind="mergesort")
        .groupby("breath_id", sort=False)
        .last()
        .rename(columns={"pressure_pred": "last_insp_pred"})
    )
    out_df = out_df.merge(last_insp[["last_insp_pred"]], on="breath_id", how="left")

    stable_exp = out_df["exp_start_pred"].where(
        ~out_df["exp_start_pred"].isna(), out_df["last_insp_pred"]
    )
    out_df.loc[out_df["is_exp"], "pressure_pred"] = stable_exp.loc[
        out_df["is_exp"]
    ].to_numpy()

    out_df = out_df[["id", "pressure_pred"]]
    return out_df


def g(dp):
    """
    Main blender:
    - If external submission csvs exist in dp: blend them as before.
    - Else: use train-derived median lookup fallback.
    Always writes submission.csv with id,pressure.

    Score-relevant minimal fix:
    - Ensure predictions align by `id` (merge) rather than relying on row order.
    """
    files = []
    if dp is not None and os.path.isdir(dp):
        for i in glob.iglob(f"{dp}/*"):
            if os.path.isfile(i) and i.lower().endswith(".csv"):
                files.append(i)
    files.sort()

    loop_time = 125
    splits = 2

    output = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )

    if len(files) == 0:
        df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
        pred_df = _fallback_predict_from_train_lookup(df_train, df_test)

        output = output.merge(pred_df, on="id", how="left", validate="one_to_one")
        output["pressure"] = output["pressure_pred"].astype(float)
        output.drop(columns=["pressure_pred"], inplace=True)

        output["pressure"] = output["pressure"].apply(find_nearest)
        output.to_csv("submission.csv", index=False)
        return

    flist = []
    step = round(len(files) / splits) if splits > 0 else len(files)
    for i in range(splits):
        if i == splits - 1:
            flist.append(files[i * step :])
        else:
            flist.append(files[i * step : (i + 1) * step])

    blended = []
    for part in flist:
        pred = wc(part)
        if pred is not None:
            blended.append(pred)

    if len(blended) == 0:
        df_test = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")
        pred_df = _fallback_predict_from_train_lookup(df_train, df_test)

        output = output.merge(pred_df, on="id", how="left", validate="one_to_one")
        output["pressure"] = output["pressure_pred"].astype(float)
        output.drop(columns=["pressure_pred"], inplace=True)

        output["pressure"] = output["pressure"].apply(find_nearest)
        output.to_csv("submission.csv", index=False)
        return

    pred_list = []
    for t in range(loop_time):
        set_seed(t)
        weight = [rd() for _ in range(len(blended))]
        weight_sum = sum(weight)
        if weight_sum == 0:
            weight = [1.0 / len(weight)] * len(weight)
        else:
            weight = [w / weight_sum for w in weight]

        weight.sort(reverse=True)

        temp = 0.0
        for j in range(len(blended)):
            temp += blended[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()

    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv("submission.csv", index=False)




## === cell 2
g("../input/ventilator-pressure-high-score-submissions")
print("Wrote submission.csv")
sub = pd.read_csv("submission.csv")
print(sub.head())
print(sub.shape)
print(sub.columns.tolist())
