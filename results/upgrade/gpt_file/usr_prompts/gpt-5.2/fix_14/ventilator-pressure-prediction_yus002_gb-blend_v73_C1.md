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

0.1660338253150522

# 6. Current score

5.03092

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 8.1335) has done: 'I remove the dependency on missing external blend files (`../input/gb-blending/...`) that currently causes the `FileNotFoundError`, and instead generate a valid submission directly from the provided competition inputs. To keep the core “blend/round-to-nearest-known-pressure” logic intact, I build a simple, deterministic per-(R,C,time_step,u_in,u_out) lookup from the training set and use it to predict the test set, then apply your existing `find_nearest` discretization. This runs end-to-end in the Kaggle environment, writes a proper `submission.csv` with `id,pressure`, and should yield a reasonable MAE without changing any model/training approach (since none exists here). I also make the path robust by trying both known dataset locations under `../input`.'
- What this solution (achieved 5.10731) has done: 'Your current score (8.1335 MAE; lower is better) is far from the target (0.166), so we need a meaningful but still “minimal-core-logic” improvement. The largest issue is that your lookup ignores the breath’s sequential dynamics and the inspiratory-only scoring; we can keep the same deterministic lookup idea but apply it *within each breath* using cumulative engineered signals (cumulative u_in and a simple lag) and then explicitly set predictions to 0 during expiratory phase (u_out==1) to match the metric. We also make the lookup more robust by quantizing float keys (time_step/u_in) and using a hierarchical fallback based on these new features, then keep your existing `find_nearest` discretization. These changes preserve the “no model training” nature and keep runtime within limits while moving MAE much closer toward the target.'
- What this solution (achieved 5.031) has done: 'I fix the runtime error in `_add_minimal_breath_features` caused by `groupby.apply` returning a multi-column object, by replacing it with a vectorized, index-aligned cumulative sum within each `breath_id`. This preserves your current “inspiratory step counter” feature idea but makes it deterministic and compatible with pandas 2.2+. I also add a safety alignment so the submission strictly follows `sample_submission.csv`’s `id` ordering (prevents accidental misalignment), while keeping your hierarchical lookup + `find_nearest` discretization unchanged. The result run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 5.031) has done: 'Your current MAE (5.031; lower is better) is still far from the target (0.166), so we need a meaningful but still “same-core-logic” correction. The biggest metric-mismatch is that the competition **does not score expiratory phase (u_out==1)**, but your lookup predicts arbitrary pressures there; setting those predictions to a constant (0, then discretized) is a minimal post-processing change that should reduce MAE substantially without changing your lookup hierarchy. Second, your submission assembly currently merges `sub` with `df_test[['id']]` (a no-op) and does not explicitly guarantee that `pred` is aligned to `sub['id']`; we instead build predictions in **test row order** and then map them onto the sample submission’s id order. These changes keep your feature engineering + hierarchical groupby-mean lookup + `find_nearest` discretization intact and should move the score much closer toward the target.'
- What this solution (achieved 5.031) has done: 'Your current MAE (5.031, lower is better) is still far above the target (0.166), so we should make a minimal but metric-aligned improvement without changing your overall “hierarchical lookup + discretize to known pressures” approach. The biggest remaining mismatch is that the metric ignores expiratory phase (u_out==1), and setting those predictions to **0** is likely far from the typical mean pressure; instead we set expiratory predictions to a stable baseline (the per-(R,C) mean from train), then apply the same `find_nearest` rounding. Additionally, we keep inspiratory predictions unchanged, and we preserve your existing feature engineering and fallback lookup hierarchy so this is a small, low-risk change. This should reduce MAE materially while keeping runtime and core logic intact and still producing a valid `submission.csv`.'
- What this solution (achieved 5.031) has done: 'Your current MAE (5.031; lower is better) is still far above the target (0.166), so we need a meaningful improvement while keeping your “hierarchical lookup + round-to-known-pressures” core intact. The main fix is to make the lookup keys match the true time-series nature: compute cumulative inspired volume only during inspiration (reset/hold during u_out=1) and use a more informative lag (previous predicted/observed u_in within inspiration) so inspiratory rows map to much closer training analogs. We also change the expiratory baseline from a coarse per-(R,C) mean to a tighter per-(R,C,time_step_q) baseline (still from your existing g4) to reduce unscored/transition spillover risk with minimal logic change. Finally, we keep submission alignment by mapping predictions by `id` exactly as you already do and still discretize with `find_nearest`.'
- What this solution (achieved 5.031) has done: 'We keep your hierarchical per-key mean-lookup + nearest-pressure discretization intact, but fix the biggest remaining metric mismatch: during expiration (`u_out==1`) the ground-truth pressure is deterministically `0`, and those rows are **not scored**, so predicting a nonzero baseline there can only hurt around the phase boundary; we set expiratory predictions to exactly `0` before discretization. Next, we add one minimal, breath-dynamics-aligned key that doesn’t change the “no model” approach: a quantized lag of cumulative inspired input (`u_in_cum_lag1_q`), and we use it only as an additional top-level lookup (fallback remains unchanged). Finally, we make the nearest-pressure rounding vectorized for speed (no semantic change), keeping output alignment by `id` exactly as you already do.'
- What this solution (achieved 5.031) has done: 'We keep your hierarchical lookup + nearest-known-pressure discretization intact, but fix the largest source of MAE: your `id` alignment is currently wrong because `df_test["id"]` is not globally unique (it resets each breath), so mapping predictions by `id` collapses many rows and produces incorrect submissions. We instead carry predictions in the exact test row order and write them directly into `sample_submission.csv` (which is already in the correct order), guaranteeing a 1:1 alignment. Additionally, because the metric ignores expiratory phase, we force `pressure=0` for `u_out==1` **after** rounding to known pressures (so it is exactly 0.0), which avoids any accidental nonzero due to discretization. These minimal fixes should move MAE substantially toward your target without changing the core prediction logic.'
- What this solution (achieved 7.7098) has done: 'The current MAE is far above the target, so we should make a metric-aligned improvement while keeping your “hierarchical lookup + nearest-known-pressure discretization” core intact. The key fix is to correct the prediction alignment: `id` is not globally unique in this dataset dump, so writing predictions in sample submission row order without explicitly ordering test rows can scramble outputs and inflate MAE. We sort `df_test/test_feat` by `row_id` (or by `breath_id`+`time_step` as a fallback) to match `sample_submission.csv` order, generate predictions in that exact order, and then write them directly. We keep your expiratory-phase post-processing (`u_out==1 -> 0.0`) and all lookup groupings unchanged.'
- What this solution (achieved 5.031) has done: 'Your MAE (7.7098; lower is better) is far above the target (0.166), and the biggest likely cause here is submission misalignment: in this dataset dump `id` is not globally unique (it resets each breath), and sorting test rows by `breath_id,time_step` can break the required row order that matches `sample_submission.csv`. I make the smallest change to guarantee 1:1 alignment by always generating predictions in the original `test.csv` row order (which matches the sample submission order) and removing any sorting-based reordering. I keep your entire hierarchical lookup + feature engineering + nearest-pressure discretization intact, and keep the expiratory-phase override (`u_out==1 -> 0.0`) exactly as you have it. This should move the score substantially toward the target without changing the core prediction logic.'
- What this solution (achieved 5.03092) has done: 'We keep your hierarchical groupby-mean lookup + nearest-known-pressure discretization unchanged, but fix a metric-critical mismatch: the MAE is computed **only on inspiratory rows (u_out==0)**, so your training aggregations should not be “pulled” by expiratory (often zero) pressures. Concretely, we build the lookup tables (g00..g5 and global_mean) using only `u_out==0` rows from train, while still producing predictions for all test rows. We also keep your existing post-processing `u_out==1 -> 0.0` (unscored) and submission row-order alignment exactly as-is. This is a minimal change that should move MAE substantially toward the target without changing the overall approach or adding any model training.'

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
def _read_comp_csv(filename: str) -> pd.DataFrame:
    candidates = [
        f"../input/ventilator-pressure-prediction/{filename}",
        f"/kaggle/input/ventilator-pressure-prediction/{filename}",
        f"../input/{filename}",
        f"/kaggle/input/{filename}",
        f"/kaggle/data/{filename}",
        f"/kaggle/data/ventilator-pressure-prediction/{filename}",
    ]
    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(f"Could not find {filename} in any of: {candidates}")


df_train = _read_comp_csv("train.csv")

unique_pressures = df_train["pressure"].unique()
sorted_pressures = np.sort(unique_pressures).astype(np.float64)
total_pressures_len = len(sorted_pressures)


def find_nearest(prediction):
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


def find_nearest_vec(pred_arr: np.ndarray) -> np.ndarray:
    pred_arr = np.asarray(pred_arr, dtype=np.float64)
    idx = np.searchsorted(sorted_pressures, pred_arr, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)

    lower_idx = np.clip(idx - 1, 0, total_pressures_len - 1)
    upper = sorted_pressures[idx]
    lower = sorted_pressures[lower_idx]

    choose_lower = (idx > 0) & (np.abs(lower - pred_arr) < np.abs(upper - pred_arr))
    out = np.where(choose_lower, lower, upper)
    return out.astype(np.float64)


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
    loop_time = 150
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
        for j in range(len(flist)):
            weight.append(rd())
        weight_sum = sum(weight)
        for j in range(len(weight)):
            weight[j] /= weight_sum
        weight.sort(reverse=True)
        temp = 0
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()
    output = _read_comp_csv("sample_submission.csv")
    output.pressure = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)


def blend(a, b, out_path="blend.csv"):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.55 + b.pressure * 0.45
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv(out_path, index=False)
    return a




## === cell 2
df_test = _read_comp_csv("test.csv")
sub = _read_comp_csv("sample_submission.csv")


def _add_minimal_breath_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["time_step_q"] = (df["time_step"] * 1000).round().astype(np.int32)  # ms
    df["u_in_q"] = (df["u_in"] * 100).round().astype(np.int32)  # 0.01

    grp = df.groupby("breath_id", sort=False)

    if "u_out" in df.columns:
        insp = (df["u_out"].to_numpy() == 0).astype(np.int16)
        insp_s = pd.Series(insp, index=df.index)
        df["u_in_insp"] = (df["u_in"] * insp_s).astype(np.float64)
        df["u_in_cum_insp"] = grp["u_in_insp"].cumsum()
        df["u_in_cum_q"] = (df["u_in_cum_insp"] * 10).round().astype(np.int32)  # 0.1

        df["step_in_insp"] = (
            insp_s.groupby(df["breath_id"], sort=False).cumsum().astype(np.int16)
        )
    else:
        df["u_in_insp"] = df["u_in"].astype(np.float64)
        df["u_in_cum_insp"] = grp["u_in_insp"].cumsum()
        df["u_in_cum_q"] = (df["u_in_cum_insp"] * 10).round().astype(np.int32)
        df["step_in_insp"] = (grp.cumcount().astype(np.int16) + 1).astype(np.int16)

    df["u_in_lag1"] = grp["u_in"].shift(1).fillna(0.0)
    df["u_in_lag1_q"] = (df["u_in_lag1"] * 100).round().astype(np.int32)  # 0.01

    df["u_in_cum_lag1"] = grp["u_in_cum_insp"].shift(1).fillna(0.0)
    df["u_in_cum_lag1_q"] = (df["u_in_cum_lag1"] * 10).round().astype(np.int32)  # 0.1

    df.drop(columns=["u_in_insp"], inplace=True, errors="ignore")
    return df


train_feat = _add_minimal_breath_features(
    df_train[["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]]
)
test_feat = _add_minimal_breath_features(
    df_test[["breath_id", "R", "C", "time_step", "u_in", "u_out"]]
)

train_insp = train_feat[train_feat["u_out"] == 0].copy()

g00 = train_insp.groupby(
    [
        "R",
        "C",
        "u_out",
        "step_in_insp",
        "u_in_q",
        "u_in_cum_q",
        "u_in_cum_lag1_q",
        "u_in_lag1_q",
    ],
    sort=False,
)["pressure"].mean()

g0 = train_insp.groupby(
    ["R", "C", "u_out", "step_in_insp", "u_in_q", "u_in_cum_q", "u_in_lag1_q"],
    sort=False,
)["pressure"].mean()

g1 = train_insp.groupby(
    ["R", "C", "time_step_q", "u_out", "u_in_q", "u_in_cum_q", "u_in_lag1_q"],
    sort=False,
)["pressure"].mean()

g2 = train_insp.groupby(
    ["R", "C", "time_step_q", "u_out", "u_in_q", "u_in_cum_q"], sort=False
)["pressure"].mean()

g3 = train_insp.groupby(["R", "C", "time_step_q", "u_out", "u_in_q"], sort=False)[
    "pressure"
].mean()

g4 = train_insp.groupby(["R", "C", "time_step_q", "u_out"], sort=False)[
    "pressure"
].mean()

g5 = train_insp.groupby(["R", "C"], sort=False)["pressure"].mean()

global_mean = float(train_insp["pressure"].mean())

key00 = list(
    zip(
        test_feat["R"],
        test_feat["C"],
        test_feat["u_out"],
        test_feat["step_in_insp"],
        test_feat["u_in_q"],
        test_feat["u_in_cum_q"],
        test_feat["u_in_cum_lag1_q"],
        test_feat["u_in_lag1_q"],
    )
)
pred = pd.Series(key00).map(g00)

if pred.isna().any():
    key0 = list(
        zip(
            test_feat["R"],
            test_feat["C"],
            test_feat["u_out"],
            test_feat["step_in_insp"],
            test_feat["u_in_q"],
            test_feat["u_in_cum_q"],
            test_feat["u_in_lag1_q"],
        )
    )
    pred = pred.fillna(pd.Series(key0).map(g0))

if pred.isna().any():
    key1 = list(
        zip(
            test_feat["R"],
            test_feat["C"],
            test_feat["time_step_q"],
            test_feat["u_out"],
            test_feat["u_in_q"],
            test_feat["u_in_cum_q"],
            test_feat["u_in_lag1_q"],
        )
    )
    pred = pred.fillna(pd.Series(key1).map(g1))

if pred.isna().any():
    key2 = list(
        zip(
            test_feat["R"],
            test_feat["C"],
            test_feat["time_step_q"],
            test_feat["u_out"],
            test_feat["u_in_q"],
            test_feat["u_in_cum_q"],
        )
    )
    pred = pred.fillna(pd.Series(key2).map(g2))

if pred.isna().any():
    key3 = list(
        zip(
            test_feat["R"],
            test_feat["C"],
            test_feat["time_step_q"],
            test_feat["u_out"],
            test_feat["u_in_q"],
        )
    )
    pred = pred.fillna(pd.Series(key3).map(g3))

if pred.isna().any():
    key4 = list(
        zip(
            test_feat["R"], test_feat["C"], test_feat["time_step_q"], test_feat["u_out"]
        )
    )
    pred = pred.fillna(pd.Series(key4).map(g4))

if pred.isna().any():
    key5 = list(zip(test_feat["R"], test_feat["C"]))
    pred = pred.fillna(pd.Series(key5).map(g5))

pred = pred.fillna(global_mean).astype(float).to_numpy()

pred = find_nearest_vec(pred)

u_out_arr = df_test["u_out"].to_numpy()
pred[u_out_arr == 1] = 0.0

out = sub[["id"]].copy()
if len(out) != len(pred):
    raise ValueError(
        f"Row count mismatch: sample_submission={len(out)} vs pred={len(pred)}"
    )
out["pressure"] = pred.astype(float)

out.to_csv("submission.csv", index=False)

print(out.head())
print("Wrote submission.csv with shape:", out.shape)
print("Pressure stats:", out["pressure"].describe())
