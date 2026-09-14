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

0.1495717798583712

# 6. Current score

3.04934

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65244) has done: 'I fix the immediate runtime failure by removing the dependency on non-existent `../input/gb-submitted/*.csv` files and instead generate a valid submission from the provided competition inputs. To keep core logic intact (this notebook is a blender/rounder), I reuse your existing “snap to nearest training pressure” post-processing and write out `submission.csv` in the required `id,pressure` format. I also make the input path robust by falling back to the available `/kaggle/input/...` location if needed, and ensure the script always produces a `.csv` even when no external blend sources are present. These changes are score-neutral relative to this script’s intent (it previously couldn’t run), and at least yield a valid submission file.'
- What this solution (achieved 5.75307) has done: 'I fix the immediate runtime error by correcting the global aggregation code: `DataFrame.agg()` with named aggregations must be applied to a grouped object (or use explicit scalar reductions), so I replace it with direct `.sum()` calls to produce the same sufficient statistics (`n, su, st, ...`). This keeps your core modeling logic identical (closed-form ridge regression per (R,C) group with a global fallback) while making it run end-to-end. I also keep the existing pressure “snapping” post-processing unchanged and ensure `submission.csv` is written with the required `id,pressure` columns. No score-tuning changes are introduced beyond making the script execute correctly.'
- What this solution (achieved 3.1845) has done: 'Your current score (5.75307, lower-is-better) is far from the target (0.1496), so we need a meaningful but still minimal change that keeps the same closed-form per-(R,C) linear regression core. The main issue is that the model ignores the time-series “within breath” dynamics and predicts each row independently, which is especially harmful in this competition. I keep the same ridge-regularized linear solve approach, but fit it at the breath level using cumulative features (e.g., cumulative u_in and cumulative u_in*time_step) computed per breath for both train and test; this preserves the linear model logic while adding the smallest time-series signal. I also ensure we only train on inspiratory rows (u_out==0) as you already do, keep the same snapping-to-nearest-pressure postprocessing, and write a valid `submission.csv`.'
- What this solution (achieved 2.95506) has done: 'To move your MAE down toward the 0.1496 target while keeping the same “closed-form ridge linear regression per (R,C)” core, I’m adding only a few extra linear features that capture the most important within-breath dynamics: lagged `u_in`, lagged `u_out`, and cumulative `u_out` (a proxy for inspiratory/expiratory history). I keep the exact same training restriction (fit only on `u_out==0`) and the same ridge-regularized normal-equation solve, just with a slightly wider linear design matrix. I also apply the same features to test, keep your “snap to nearest training pressure” post-processing unchanged, and still write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 2.80735) has done: 'Your current MAE (2.955) is still far above the target (0.1496), so we need a modest but meaningful improvement while keeping your same closed-form ridge linear regression per-(R,C) design. The biggest low-risk gain here is to add a couple of standard linear time-series features that fit your existing framework: breath-relative time deltas and a simple interaction term capturing how u_in changes over time (`u_in * time_step`) plus cumulative time. This preserves the exact training restriction (`u_out==0`), keeps the same normal-equation solve and snapping-to-nearest-pressure postprocess, and only widens the design matrix slightly. I also keep I/O paths and the submission schema unchanged so it still writes a valid `submission.csv`.'
- What this solution (achieved 2.80735) has done: 'To reduce your MAE toward the 0.1496 target while preserving the same closed-form ridge linear regression core, I make three minimal, metric-aligned changes. First, I compute features in strict time order within each breath (sort by `breath_id,time_step`) to prevent lag/diff/cumsum features from being corrupted by any accidental row-order issues. Second, I include `R` and `C` as linear features inside each (R,C)-specific model anyway (they’re constants per group so they don’t change those fits) but importantly they improve the global fallback model used when a group is missing/noisy, without changing the training approach. Third, I enforce the competition semantics by setting predictions to 0 during expiration (`u_out==1`) before snapping to the nearest training pressure, which typically yields a meaningful MAE drop because expiration isn’t scored and pressures there are otherwise hard to predict.'
- What this solution (achieved 2.80735) has done: 'Your current MAE (2.80735, lower-is-better) is still far from the target (0.1496), so we need a meaningful improvement while keeping your same closed-form ridge linear regression core. The biggest issue hurting score is that training is restricted to inspiratory rows (`u_out==0`) but test-time features and prediction are still computed across the whole breath, letting expiratory `u_in`/time history leak into inspiratory predictions via cumsums/lags; we fix this by “freezing” the cumulative/lag dynamics after the first expiratory step so inspiratory predictions only depend on inspiratory history. This preserves your exact model form (same ridge normal-equation solve, same features names, same snapping), but makes the features metric-aligned with how data behaves and typically yields a large MAE drop. We also ensure `id` alignment is preserved and still write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 2.32988) has done: 'We keep your exact closed-form ridge linear regression core, but make two small metric-aligned fixes that should reduce MAE toward the 0.1496 target. First, we avoid forcing expiratory predictions to 0; since expiration is ignored in the metric, we should simply keep any values there (or at least avoid distorting the nearest-pressure snapping step), which can otherwise introduce avoidable errors if Kaggle’s scoring mask differs from your assumption. Second, we add a minimal and standard within-breath linear signal that your framework can exploit: a couple of extra lag/diff features on `u_in` (lag2 and delta), computed in the same breath-sorted way and frozen after expiration just like your other cumulative features. These are tiny additions (just more linear columns) and typically help a lot for this competition without changing the training approach.'
- What this solution (achieved 2.72547) has done: 'Your current MAE (2.32988, lower-is-better) is still far above the target (0.14957), so the smallest “toward-target” improvement that preserves your exact closed-form ridge-per-(R,C) linear core is to align training with the competition’s scoring mask: train only on inspiratory rows where `u_out==0` **and** `pressure` is not missing (in this dataset, `pressure` exists everywhere in train, so no change) but crucially also exclude the transition noise by dropping the first few timesteps where lag/cumsum features are least informative (this is a standard minimal tweak that often reduces MAE without changing model form). Next, we keep your feature set and freezing logic, but we fix one feature bug: `time_cum` is currently a cumsum of `time_step` (wrong units); we replace it with a cumulative sum of `dt` (elapsed time), which keeps the same “time accumulation” idea while making it physically meaningful and linear-model-friendly. Finally, we keep the same snapping-to-nearest-training-pressure postprocess and still write `submission.csv` with `id,pressure`.'
- What this solution (achieved 3.04934) has done: 'We keep your exact closed-form ridge linear regression per-(R,C) core and the same feature set, but fix two score-hurting alignment issues and one regularization detail. First, your feature engineering sorts and resets the index, which breaks the required `id`-row alignment in the submission; we preserve original row order by computing features on a sorted copy and then merging back to the original indices. Second, we ensure train/test use the same “step>=2” masking behavior by zeroing lag-based features for the first two steps (instead of dropping only in train), reducing train/test mismatch without changing the model form. Third, we slightly increase ridge from `1e-6` to `1e-3` for numerical stability of the normal equations (still the same ridge solve, just less ill-conditioned), which typically improves MAE for this linear setup.'
- What this solution (achieved 3.04934) has done: 'We keep your exact “closed-form ridge linear regression per-(R,C) with global fallback + snap to nearest training pressure” core, but fix a score-hurting feature mismatch: your “freeze after expiration” logic currently freezes *everything after the first u_out==1*, which unintentionally zeroes dynamics for the (scored) inspiratory region that occurs before expiration but also corrupts later rows in train/test differently when breaths have early u_out toggles. We instead freeze cumulative/lag features only for the expiratory region itself (u_out==1), while leaving inspiratory rows (u_out==0) computed from true within-breath history, matching the metric mask more closely. We also apply the same “step>=2” handling consistently to both train and test by defining step on the same sorted order used for feature generation, preventing subtle train/test mismatch. These are minimal, metric-aligned changes that should move MAE down from ~3.05 toward the 0.15 target without changing the model form or post-processing.'
- What this solution (achieved 3.04934) has done: 'We keep your exact closed-form ridge regression per-(R,C) with global fallback and the same feature set, but fix one metric-critical mismatch: the competition only scores inspiratory timesteps, so training only on `u_out==0` while letting expiratory timesteps influence lags/cumsums (by zeroing) can distort the inspiratory dynamics. The minimal, core-logic-preserving change is to compute all lag/diff/cumsum features from the true time-ordered sequence (no expiratory zeroing), and instead apply the scoring semantics only at the end by copying inspiratory predictions into expiratory rows within each breath (expiration isn’t scored, so this avoids harming MAE through arbitrary expiratory outputs). This keeps the model form, solve method, and snapping unchanged, but aligns feature generation and post-processing with the evaluation mask and typically moves MAE down substantially from ~3.05 toward your 0.1496 target. We also keep row alignment intact by continuing to compute on a sorted copy and restoring original order.'

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
BASE1 = "../input/ventilator-pressure-prediction"
BASE2 = "/kaggle/input/ventilator-pressure-prediction"

DATA_DIR = BASE1 if os.path.exists(BASE1) else BASE2

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

df_train = pd.read_csv(train_path)

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
        weight1 = (l[1] / l_sum) + 0.15
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
    output = pd.read_csv(sample_sub_path)
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
df_test = pd.read_csv(test_path)


def add_cum_features(df):
    df0 = df.copy()
    df0["_orig_row"] = np.arange(len(df0), dtype=np.int64)

    df = df0.sort_values(["breath_id", "time_step", "_orig_row"], kind="mergesort")
    g = df.groupby("breath_id", sort=False)

    df["u_in_cum"] = g["u_in"].cumsum()
    df["u_in_t"] = df["u_in"] * df["time_step"]
    df["u_in_t_cum"] = g["u_in_t"].cumsum()

    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0)
    df["u_in_delta"] = (df["u_in"] - df["u_in_lag1"]).astype(np.float64)

    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0.0)
    df["u_out_cum"] = g["u_out"].cumsum()

    df["dt"] = g["time_step"].diff().fillna(0.0)
    df["time_cum"] = g["dt"].cumsum()

    df["u_in_x_time"] = df["u_in"] * df["time_step"]

    step = g.cumcount()
    early = step < 2
    for col in ["u_in_lag1", "u_in_lag2", "u_in_delta", "u_out_lag1", "dt"]:
        df.loc[early, col] = 0.0

    df = df.sort_values("_orig_row", kind="mergesort").drop(columns=["_orig_row"])
    df = df.reset_index(drop=True)
    return df


df_train_feat = add_cum_features(df_train)
df_test_feat = add_cum_features(df_test)

_tmp = df_train_feat.assign(_orig_row=np.arange(len(df_train_feat), dtype=np.int64))
_tmp = _tmp.sort_values(["breath_id", "time_step", "_orig_row"], kind="mergesort")
_tmp["step_in_breath"] = _tmp.groupby("breath_id", sort=False).cumcount()
step_in_breath = _tmp.sort_values("_orig_row", kind="mergesort")[
    "step_in_breath"
].to_numpy()
del _tmp

train_mask = (df_train_feat["u_out"] == 0) & (step_in_breath >= 2)

train_insp = df_train_feat.loc[
    train_mask,
    [
        "R",
        "C",
        "u_in",
        "u_out",
        "time_step",
        "u_in_cum",
        "u_in_t_cum",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_delta",
        "u_out_lag1",
        "u_out_cum",
        "dt",
        "time_cum",
        "u_in_x_time",
        "pressure",
    ],
].copy()

train_insp["one"] = 1.0

train_insp["x1"] = train_insp["u_in"]
train_insp["x2"] = train_insp["time_step"]
train_insp["x3"] = train_insp["u_in_cum"]
train_insp["x4"] = train_insp["u_in_t_cum"]
train_insp["x5"] = train_insp["u_in_lag1"]
train_insp["x6"] = train_insp["u_in_lag2"]
train_insp["x7"] = train_insp["u_in_delta"]
train_insp["x8"] = train_insp["u_out_lag1"]
train_insp["x9"] = train_insp["u_out_cum"]
train_insp["x10"] = train_insp["dt"]
train_insp["x11"] = train_insp["time_cum"]
train_insp["x12"] = train_insp["u_in_x_time"]
train_insp["x13"] = train_insp["R"].astype(np.float64)
train_insp["x14"] = train_insp["C"].astype(np.float64)

xcols = [
    "x1",
    "x2",
    "x3",
    "x4",
    "x5",
    "x6",
    "x7",
    "x8",
    "x9",
    "x10",
    "x11",
    "x12",
    "x13",
    "x14",
]
k = 1 + len(xcols)  # intercept + features

X = train_insp[["one"] + xcols].to_numpy(dtype=np.float64)  # (N, k)
y = train_insp["pressure"].to_numpy(dtype=np.float64)  # (N,)

for i in range(k):
    for j in range(i, k):
        train_insp[f"s_{i}_{j}"] = X[:, i] * X[:, j]
for i in range(k):
    train_insp[f"t_{i}"] = y * X[:, i]

agg_dict = {f"s_{i}_{j}": "sum" for i in range(k) for j in range(i, k)}
agg_dict.update({f"t_{i}": "sum" for i in range(k)})

agg = train_insp.groupby(["R", "C"], sort=False).agg(agg_dict).reset_index()

ridge = 1e-3
coef_map = {}

for row in agg.itertuples(index=False):
    XtX = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(i, k):
            v = getattr(row, f"s_{i}_{j}")
            XtX[i, j] = v
            XtX[j, i] = v
    Xty = np.array([getattr(row, f"t_{i}") for i in range(k)], dtype=np.float64)

    XtX = XtX + ridge * np.eye(k, dtype=np.float64)
    beta = np.linalg.solve(XtX, Xty)
    coef_map[(int(row.R), int(row.C))] = beta

XtX_all = np.zeros((k, k), dtype=np.float64)
for i in range(k):
    for j in range(i, k):
        v = float(train_insp[f"s_{i}_{j}"].sum())
        XtX_all[i, j] = v
        XtX_all[j, i] = v
Xty_all = np.array(
    [float(train_insp[f"t_{i}"].sum()) for i in range(k)], dtype=np.float64
)
XtX_all = XtX_all + ridge * np.eye(k, dtype=np.float64)
beta_global = np.linalg.solve(XtX_all, Xty_all)

R_arr = df_test_feat["R"].to_numpy(dtype=np.int64)
C_arr = df_test_feat["C"].to_numpy(dtype=np.int64)

X_test = np.column_stack(
    [
        np.ones(len(df_test_feat), dtype=np.float64),
        df_test_feat["u_in"].to_numpy(dtype=np.float64),
        df_test_feat["time_step"].to_numpy(dtype=np.float64),
        df_test_feat["u_in_cum"].to_numpy(dtype=np.float64),
        df_test_feat["u_in_t_cum"].to_numpy(dtype=np.float64),
        df_test_feat["u_in_lag1"].to_numpy(dtype=np.float64),
        df_test_feat["u_in_lag2"].to_numpy(dtype=np.float64),
        df_test_feat["u_in_delta"].to_numpy(dtype=np.float64),
        df_test_feat["u_out_lag1"].to_numpy(dtype=np.float64),
        df_test_feat["u_out_cum"].to_numpy(dtype=np.float64),
        df_test_feat["dt"].to_numpy(dtype=np.float64),
        df_test_feat["time_cum"].to_numpy(dtype=np.float64),
        df_test_feat["u_in_x_time"].to_numpy(dtype=np.float64),
        df_test_feat["R"].to_numpy(dtype=np.float64),
        df_test_feat["C"].to_numpy(dtype=np.float64),
    ]
)

pred = np.empty(len(df_test_feat), dtype=np.float64)
for i in range(len(df_test_feat)):
    beta = coef_map.get((int(R_arr[i]), int(C_arr[i])), beta_global)
    pred[i] = float(X_test[i] @ beta)

tmp_pred = df_test_feat[["breath_id", "u_out"]].copy()
tmp_pred["pred"] = pred
tmp_pred["pred_insp_only"] = tmp_pred["pred"].where(tmp_pred["u_out"].to_numpy() == 0)
tmp_pred["pred_ffill_insp"] = tmp_pred.groupby("breath_id", sort=False)[
    "pred_insp_only"
].ffill()
pred = tmp_pred["pred_ffill_insp"].fillna(tmp_pred["pred"]).to_numpy(dtype=np.float64)

idx = np.searchsorted(sorted_pressures, pred, side="left")
idx = np.clip(idx, 0, total_pressures_len - 1)
lower_idx = np.clip(idx - 1, 0, total_pressures_len - 1)
upper = sorted_pressures[idx]
lower = sorted_pressures[lower_idx]
choose_lower = np.abs(lower - pred) < np.abs(upper - pred)
pred_snapped = np.where(choose_lower, lower, upper).astype(np.float64)

sub = pd.DataFrame({"id": df_test_feat["id"].to_numpy(), "pressure": pred_snapped})
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv", sub.shape, sub.columns.tolist())
print(sub.head())
print("id monotonic increasing:", pd.Series(sub["id"]).is_monotonic_increasing)
