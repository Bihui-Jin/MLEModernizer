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

0.1380924807773337

# 6. Current score

6.94124

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 9.97034) has done: 'The crash happens because `g()` is trying to stack/median arrays that end up being length 1 (because the input folder is missing/empty or contains non-submission files), so the resulting prediction vector can’t be assigned to the 603600-row sample submission. I make `g()` robust to missing/invalid blend inputs by (1) checking the directory exists and contains readable submission-like CSVs, (2) filtering to files with a `pressure` column and the expected length, and (3) providing a safe fallback model (groupwise median pressure by `(R,C,time_step,u_in,u_out)` computed from train) to always generate a valid submission. This keeps the original blending logic intact when the expected input files are present, but prevents runtime errors and ensures a `.csv` submission is written end-to-end. The fallback is legitimate and should give a non-trivial score (and at minimum yield a valid submission instead of failing).'
- What this solution (achieved 7.44526) has done: 'Your current score (9.97 MAE) is far worse than the target (0.138), so we need a real predictive lift rather than robustness tweaks. The smallest legitimate change that preserves your overall “predict pressure + snap to nearest known pressure” core logic is to replace the weak fallback (simple groupwise median by raw features) with a strong, competition-standard nearest-neighbor lookup per breath using engineered cumulative features (`u_in_cum`, `u_in_lag1`, `u_out_lag1`) grouped by `(R,C)`; this dramatically reduces MAE while keeping inference simple and deterministic. The blender path is kept intact, but if no valid blend inputs exist it now use this stronger fallback, producing a valid `submission.csv` end-to-end. This should move the score substantially toward the target band without changing model architecture/training loops (none exist here).'
- What this solution (achieved 7.97285) has done: 'Your current MAE (7.445) is far worse than the target (0.138), so we need a meaningful lift without changing your overall “lookup-based fallback + snap to nearest known pressure” core logic. The biggest issue is that the fallback is still a coarse global median-by-keys merge, which misses the dominant “same breath trajectory repeats” structure; we keep the same approach but strengthen it by adding a deterministic within-(R,C) nearest-neighbor mapping from a discretized per-timestep state (`u_in_cum`, lags, and `time_step`) to pressure. We do this with a single-pass key encoding and `groupby().median()` lookup (fast) plus a conservative backoff chain, and we keep the blender path unchanged when valid blend CSVs exist. This should substantially reduce MAE while staying within the same inference semantics (no training loop/model) and still always writing a valid `submission.csv`.'
- What this solution (achieved 6.93) has done: 'I fix the crash in `pd.merge_asof` by ensuring both the left (`te_need`) and right (`grp_cum`) frames are sorted by the full set of `by` keys plus the `on` key, which Pandas requires for `merge_asof` (your current sort can still violate this when multiple `by` groups interleave). I also add a small safety fallback: if the asof-merge still fails for any reason, the code skip that refinement step and continue with the existing backoff merges, so a valid submission is always produced. These are execution/correctness fixes only and keep your existing lookup/blend logic intact, while letting the stronger fallback run end-to-end and produce `submission.csv`. The output format and snapping-to-nearest-known-pressure behavior are preserved.'
- What this solution (achieved 6.94124) has done: 'Your current MAE (6.93, lower-is-better) is still far from the target (0.138), so we need a meaningful but minimal lift inside your existing lookup-based fallback (since blending inputs may be weak/absent). I keep your overall pipeline intact, but strengthen the fallback by adding a more discriminative first-stage lookup key that includes `(R,C,step,u_out,u_in_cum_q,u_in_q)` at finer quantization (0.01) and a slightly tighter `merge_asof` tolerance on `u_in_cum_q` within each `(R,C,step,u_out,u_in_q)` group. This preserves the same “groupby-median lookup + asof refinement + backoff merges + snap-to-nearest-known-pressure” semantics, but should reduce the number of NaNs and improve matched pressures toward the competition’s strong baseline behavior. The blender path remains unchanged when valid blend CSVs exist, and the script still always writes a valid `submission.csv` end-to-end.'

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
        try:
            public_lb_score = int(
                input_list[i].split("/")[-1].split(".")[1].split(" ")[0]
            )
        except Exception:
            public_lb_score = 1
        l.append(public_lb_score)
        input_list[i] = (pd.read_csv(input_list[i]).pressure).to_numpy().ravel()
    output = 0
    l_sum = sum(l) if sum(l) != 0 else 1
    if len(input_list) == 1:
        output = input_list[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time_step"] = df["time_step"].round(5)
    df["u_in"] = df["u_in"].round(5)

    df.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")
    df["u_in_cum"] = df.groupby("breath_id", sort=False)["u_in"].cumsum().round(5)
    df["u_in_lag1"] = (
        df.groupby("breath_id", sort=False)["u_in"].shift(1).fillna(0.0).round(5)
    )
    df["u_out_lag1"] = (
        df.groupby("breath_id", sort=False)["u_out"].shift(1).fillna(0).astype(np.int64)
    )
    df["step"] = df.groupby("breath_id", sort=False).cumcount().astype(np.int16)

    dt = df.groupby("breath_id", sort=False)["time_step"].diff().fillna(0.0)
    du = df.groupby("breath_id", sort=False)["u_in"].diff().fillna(0.0)
    df["u_in_rate"] = (
        (du / dt.replace(0.0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .round(5)
    )

    return df


def _fallback_submission(
    train_df: pd.DataFrame, test_path: str, sample_sub_path: str, out_path: str
) -> pd.DataFrame:
    test_df = pd.read_csv(
        test_path,
        usecols=["id", "breath_id", "R", "C", "time_step", "u_in", "u_out"],
    )
    sample_sub = pd.read_csv(sample_sub_path)

    tr = train_df[
        ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    ].copy()

    tr_f = _add_features(tr)
    te_f = _add_features(test_df)

    for df in (tr_f, te_f):
        df["u_in_cum_q"] = (df["u_in_cum"] * 100).round().astype(np.int32)  # 0.01
        df["u_in_q"] = (df["u_in"] * 100).round().astype(np.int32)  # 0.01
        df["time_step_q"] = (df["time_step"] * 1000).round().astype(np.int32)  # 0.001s
        df["u_in_lag1_q"] = (df["u_in_lag1"] * 100).round().astype(np.int32)  # 0.01
        df["u_in_rate_q"] = (
            (df["u_in_rate"] * 10).round().astype(np.int32)
        )  # keep coarse (stable)

    keys_hp = ["R", "C", "step", "u_out", "u_in_q", "u_in_cum_q"]

    keys_1 = [
        "R",
        "C",
        "step",
        "time_step_q",
        "u_out",
        "u_out_lag1",
        "u_in_q",
        "u_in_lag1_q",
        "u_in_rate_q",
        "u_in_cum_q",
    ]

    keys_group = [
        "R",
        "C",
        "step",
        "time_step_q",
        "u_out",
        "u_out_lag1",
        "u_in_q",
        "u_in_lag1_q",
        "u_in_rate_q",
    ]

    keys_group_cum = ["R", "C", "step", "u_out", "u_in_q"]

    keys_2 = [
        "R",
        "C",
        "step",
        "time_step_q",
        "u_out",
        "u_in_q",
        "u_in_rate_q",
        "u_in_cum_q",
    ]
    keys_3 = ["R", "C", "step", "time_step_q", "u_out", "u_in_q", "u_in_rate_q"]
    keys_4 = ["R", "C", "step", "time_step_q", "u_out"]
    keys_5 = ["R", "C", "step"]

    grp_hp = (
        tr_f.groupby(keys_hp, sort=False, observed=True)["pressure"]
        .median()
        .reset_index()
    )

    grp1 = (
        tr_f.groupby(keys_1, sort=False, observed=True)["pressure"]
        .median()
        .reset_index()
    )

    grp_cum = (
        tr_f.groupby(keys_group_cum + ["u_in_cum_q"], sort=False, observed=True)[
            "pressure"
        ]
        .median()
        .reset_index()
    )
    grp_cum.sort_values(keys_group_cum + ["u_in_cum_q"], inplace=True, kind="mergesort")

    grp2 = (
        tr_f.groupby(keys_2, sort=False, observed=True)["pressure"]
        .median()
        .reset_index()
    )
    grp3 = (
        tr_f.groupby(keys_3, sort=False, observed=True)["pressure"]
        .median()
        .reset_index()
    )
    grp4 = (
        tr_f.groupby(keys_4, sort=False, observed=True)["pressure"]
        .median()
        .reset_index()
    )
    grp5 = (
        tr_f.groupby(keys_5, sort=False, observed=True)["pressure"]
        .median()
        .reset_index()
    )

    pred = pd.Series(np.nan, index=te_f.index, dtype=np.float64)

    m_hp = te_f.merge(grp_hp, on=keys_hp, how="left")["pressure"]
    pred = pred.fillna(m_hp)

    m1 = te_f.merge(grp1, on=keys_1, how="left")["pressure"]
    pred = pred.fillna(m1)

    need = pred.isna().to_numpy()
    if need.any():
        te_need = te_f.loc[need, keys_group_cum + ["u_in_cum_q"]].copy()
        te_need.sort_values(
            keys_group_cum + ["u_in_cum_q"], inplace=True, kind="mergesort"
        )
        te_need.reset_index(drop=False, inplace=True)

        try:
            left = pd.merge_asof(
                te_need,
                grp_cum,
                on="u_in_cum_q",
                by=keys_group_cum,
                direction="backward",
                allow_exact_matches=True,
                tolerance=200,  # 200 * 0.01 = 2.0 u_in_cum units (conservative)
            )["pressure"].to_numpy()

            right = pd.merge_asof(
                te_need,
                grp_cum,
                on="u_in_cum_q",
                by=keys_group_cum,
                direction="forward",
                allow_exact_matches=True,
                tolerance=200,
            )["pressure"].to_numpy()

            u = te_need["u_in_cum_q"].to_numpy()

            l_match = pd.merge_asof(
                te_need,
                grp_cum[keys_group_cum + ["u_in_cum_q"]].rename(
                    columns={"u_in_cum_q": "match_cum"}
                ),
                left_on="u_in_cum_q",
                right_on="match_cum",
                by=keys_group_cum,
                direction="backward",
                allow_exact_matches=True,
                tolerance=200,
            )["match_cum"].to_numpy()

            r_match = pd.merge_asof(
                te_need,
                grp_cum[keys_group_cum + ["u_in_cum_q"]].rename(
                    columns={"u_in_cum_q": "match_cum"}
                ),
                left_on="u_in_cum_q",
                right_on="match_cum",
                by=keys_group_cum,
                direction="forward",
                allow_exact_matches=True,
                tolerance=200,
            )["match_cum"].to_numpy()

            dl = np.abs(u - l_match)
            dr = np.abs(u - r_match)

            choose = np.where(
                np.isnan(left) & ~np.isnan(right),
                right,
                np.where(
                    ~np.isnan(left) & np.isnan(right),
                    left,
                    np.where(dr < dl, right, left),
                ),
            )

            pred.loc[te_need["index"].to_numpy()] = choose
        except Exception:
            pass

    m2 = te_f.merge(grp2, on=keys_2, how="left")["pressure"]
    pred = pred.fillna(m2)

    m3 = te_f.merge(grp3, on=keys_3, how="left")["pressure"]
    pred = pred.fillna(m3)

    m4 = te_f.merge(grp4, on=keys_4, how="left")["pressure"]
    pred = pred.fillna(m4)

    m5 = te_f.merge(grp5, on=keys_5, how="left")["pressure"]
    pred = pred.fillna(m5)

    global_med = float(tr_f["pressure"].median())
    pred = pred.fillna(global_med).to_numpy(dtype=np.float64)

    pred = np.array([find_nearest(x) for x in pred], dtype=np.float64)

    sample_sub["pressure"] = pred
    sample_sub.to_csv(out_path, index=False)
    return sample_sub


def g(dp):
    sample_sub_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    test_path = "../input/ventilator-pressure-prediction/test.csv"
    expected_len = len(pd.read_csv(sample_sub_path))

    if (dp is None) or (not os.path.isdir(dp)):
        return _fallback_submission(
            df_train, test_path, sample_sub_path, "submission.csv"
        )

    files = sorted(list(glob.iglob(f"{dp}/*.csv")))
    valid_files = []
    for fp in files:
        try:
            tmp = pd.read_csv(fp, usecols=["pressure"])
            if len(tmp) == expected_len:
                valid_files.append(fp)
        except Exception:
            continue

    if len(valid_files) == 0:
        return _fallback_submission(
            df_train, test_path, sample_sub_path, "submission.csv"
        )

    l = valid_files
    file_count = len(l)
    loop_time = 154

    splits = max(1, file_count // 2)
    l.sort()
    flist = []
    for i in range(splits):
        start = i * round(len(l) / splits)
        end = None if i == splits - 1 else (i + 1) * round(len(l) / splits)
        chunk = l[start:end]
        if len(chunk) == 0:
            continue
        flist.append(chunk)

    for i in range(len(flist)):
        flist[i] = wc(flist[i])

    if (len(flist) == 0) or any(
        (not hasattr(x, "__len__")) or (len(x) != expected_len) for x in flist
    ):
        return _fallback_submission(
            df_train, test_path, sample_sub_path, "submission.csv"
        )

    pred_list = []
    for k in range(loop_time):
        weight = []
        set_seed(k)
        for j in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight) if sum(weight) != 0 else 1.0
        for j in range(len(weight)):
            weight[j] /= weight_sum
        weight.sort(reverse=True)
        temp = np.zeros(expected_len, dtype=np.float64)
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()

    output = pd.read_csv(sample_sub_path)
    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)

    output.to_csv("submission.csv", index=False)
    return output


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.6 + b.pressure * 0.4
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
g("../input/gb-data-blending-recover")
print("Wrote submission.csv")
print(pd.read_csv("submission.csv").head())
print(pd.read_csv("submission.csv").shape)
