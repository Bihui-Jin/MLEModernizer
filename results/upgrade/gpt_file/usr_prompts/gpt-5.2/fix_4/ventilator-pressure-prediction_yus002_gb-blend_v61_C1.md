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

0.1510240912581121

# 6. Current score

8.53419

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65486) has done: 'The crash happens because `g('../input/gb-blending')` finds zero prediction files in that folder, so the blending pipeline builds empty/invalid arrays and ends up trying to assign a length-1 result into a 603600-row submission. I fix this by (1) making `g()` robust to missing/empty directories and only reading valid submission-like CSVs, (2) ensuring all loaded prediction vectors match the sample submission length before blending, and (3) writing a guaranteed valid fallback submission (`pressure=0`) if no external prediction files exist so you always get a `.csv` submission. These changes are score-neutral relative to the intended blending logic when files exist, and they unblock end-to-end execution in the Kaggle environment.'
- What this solution (achieved 4.38292) has done: 'Your current score is extremely far from the target (MAE 17.65 vs 0.151; lower is better), which strongly suggests the blend is falling back to `pressure=0` because no valid prediction CSVs are found in `../input/gb-blending`. The smallest change that legitimately moves the score toward the target is to keep your blending core intact, but add an in-notebook model-based fallback that generates a reasonable pressure prediction when external blend files are missing. To preserve semantics and stay minimal, the fallback uses a simple KNN regressor trained on a small, deterministic subset of the training data and predicts on test; then your existing `find_nearest` snapping is applied. This produce a valid `submission.csv` always and should dramatically reduce MAE compared to the all-zeros fallback, moving much closer toward the target band without changing your blending when files exist.'
- What this solution (achieved 8.53419) has done: 'Your current MAE (4.38) is still far above the target (0.151), and the biggest issue is that the fallback KNN is learning from per-row features that ignore the sequential “breath” dynamics and the fact that only inspiratory phase is scored. To move the score substantially toward the target while keeping your blending logic intact, I keep the same overall fallback approach (a lightweight sklearn model) but make it sequence-aware by training on per-breath cumulative features (e.g., cumulative u_in integral and lagged u_in/u_out) and predicting per timestep, which is a minimal change to the fallback path only. I also ensure we don’t train on expiratory-phase targets by filtering to u_out==0 for training, matching the evaluation semantics and improving MAE without changing your main blend behavior when external files exist. Everything still runs end-to-end and always writes a valid `submission.csv`.'

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
    """
    Original intent: weight-combine two submissions based on a number parsed from filename.
    Bugfix: be robust when filename format doesn't contain that token; fall back to equal weights.
    Also ensure returned vector is 1D float.
    """
    preds = []
    scores = []
    for path in input_list:
        try:
            public_lb_score = int(path.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1
        scores.append(public_lb_score)

        df = pd.read_csv(path)
        if "pressure" not in df.columns:
            raise ValueError(f"File {path} does not contain 'pressure' column.")
        preds.append(df["pressure"].to_numpy(dtype=float).ravel())

    if len(preds) == 1:
        return preds[0]

    l_sum = sum(scores) if sum(scores) != 0 else len(scores)
    weight1 = (scores[1] / l_sum) + 0.1
    weight1 = float(np.clip(weight1, 0.0, 1.0))
    weight2 = 1.0 - weight1
    return preds[0] * weight1 + preds[1] * weight2


def _make_seq_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Change (score-improving, still minimal and only used in fallback):
    Add simple per-breath sequential features that better capture inspiratory dynamics without
    changing any core blending logic. These are lightweight pandas groupby transforms.
    """
    df = df.copy()

    df.sort_values(["breath_id", "time_step"], inplace=True)

    g = df.groupby("breath_id", sort=False)

    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0)
    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0).astype(np.int8)

    df["du_in"] = df["u_in"] - df["u_in_lag1"]

    dt = g["time_step"].diff().fillna(0.0)
    df["dt"] = dt.astype(np.float32)
    df["u_in_dt"] = (df["u_in"] * df["dt"]).astype(np.float32)
    df["u_in_cum"] = g["u_in_dt"].cumsum().astype(np.float32)

    df["t_rel"] = df["time_step"].astype(np.float32)

    return df


def _fallback_knn_submission():
    """
    Change (score-improving, minimal): if no external blend files are available, create a legitimate
    baseline prediction instead of all-zeros.

    Improvement over previous fallback to move MAE toward target:
    - Make features sequence-aware via simple per-breath lags and cumulative u_in integral proxy.
    - Train only on inspiratory phase (u_out==0) to better match evaluation metric.
    - Keep the same model family (KNN regressor) and deterministic subset for runtime.
    """
    from sklearn.neighbors import KNeighborsRegressor

    train_path = "../input/ventilator-pressure-prediction/train.csv"
    test_path = "../input/ventilator-pressure-prediction/test.csv"
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"

    usecols_train = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    usecols_test = ["breath_id", "R", "C", "time_step", "u_in", "u_out"]

    train_full = pd.read_csv(train_path, usecols=usecols_train)
    test = pd.read_csv(test_path, usecols=usecols_test)
    sub = pd.read_csv(sample_path)

    unique_breaths = train_full["breath_id"].unique()
    rng = np.random.RandomState(2021)
    rng.shuffle(unique_breaths)
    keep_breaths = set(unique_breaths[:8000])  # ~640k rows (80 steps/breath)

    train_small = train_full[train_full["breath_id"].isin(keep_breaths)].copy()

    train_small = _make_seq_features(train_small)
    test_feat = _make_seq_features(test)

    train_small_insp = train_small[train_small["u_out"] == 0].copy()

    feat_cols = [
        "R",
        "C",
        "t_rel",
        "u_in",
        "u_out",
        "u_in_lag1",
        "u_in_lag2",
        "du_in",
        "u_out_lag1",
        "u_in_cum",
    ]

    X_train = train_small_insp[feat_cols].to_numpy(dtype=np.float32)
    y_train = train_small_insp["pressure"].to_numpy(dtype=np.float32)
    X_test = test_feat[feat_cols].to_numpy(dtype=np.float32)

    knn = KNeighborsRegressor(
        n_neighbors=80, weights="distance", metric="minkowski", p=2
    )
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test).astype(float)

    sub["pressure"] = pd.Series(pred).apply(find_nearest).to_numpy(dtype=float)
    sub.to_csv("submission.csv", index=False)


def g(dp):
    """
    Bugfixes:
    - Handle missing/empty dp directory (previously produced empty arrays -> length mismatch).
    - Only read CSV files that look like Kaggle submissions (have 'pressure' column).
    - Ensure every prediction vector matches sample_submission length before blending.
    - Always write a valid .csv submission.
    Change (score-improving): if no valid prediction files exist, use a legitimate model-based
    fallback instead of a zero submission to move MAE toward the target.
    """
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    output = pd.read_csv(sample_path)
    n = len(output)

    if not os.path.isdir(dp):
        _fallback_knn_submission()
        return

    files = sorted([p for p in glob.iglob(f"{dp}/*") if p.lower().endswith(".csv")])
    if len(files) == 0:
        _fallback_knn_submission()
        return

    valid_preds = []
    for p in files:
        try:
            dfp = pd.read_csv(p)
            if "pressure" not in dfp.columns:
                continue
            vec = dfp["pressure"].to_numpy(dtype=float).ravel()
            if len(vec) != n:
                continue
            valid_preds.append(vec)
        except Exception:
            continue

    if len(valid_preds) == 0:
        _fallback_knn_submission()
        return

    file_count = len(valid_preds)
    loop_time = 150

    splits = max(1, file_count // 2)
    flist = []
    for i in range(splits):
        start = i * round(file_count / splits)
        end = None if i == splits - 1 else (i + 1) * round(file_count / splits)
        group = valid_preds[start:end]
        if len(group) == 0:
            continue
        if len(group) == 1:
            flist.append(group[0])
        else:
            flist.append((group[0] + group[1]) / 2.0)

    if len(flist) == 0:
        _fallback_knn_submission()
        return

    pred_list = []
    for t in range(loop_time):
        set_seed(t)
        weight = [rd() for _ in range(len(flist))]
        weight_sum = sum(weight)
        if weight_sum == 0:
            weight = [1.0 / len(flist)] * len(flist)
        else:
            weight = [w / weight_sum for w in weight]
        weight.sort(reverse=True)

        temp = np.zeros(n, dtype=float)
        for i in range(len(flist)):
            temp += flist[i] * weight[i]
        pred_list.append(temp)
        del temp
        gc.collect()

    blended = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = blended
    output["pressure"] = output["pressure"].apply(find_nearest)

    output.to_csv("submission.csv", index=False)


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.6 + b.pressure * 0.4
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
g("../input/gb-blending")
print("Wrote submission.csv")
