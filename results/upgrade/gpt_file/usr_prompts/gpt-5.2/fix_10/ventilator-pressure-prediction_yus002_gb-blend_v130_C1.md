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

0.1368853205729171

# 6. Current score

8.58384

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.86378) has done: 'I fix the immediate runtime failure by removing the dependency on a missing external blending dataset and instead generate a baseline prediction directly from the provided training data. To preserve the original core idea (snapping predictions to the nearest known pressure value), the submission predict the median training pressure and then apply your existing `find_nearest` mapping. This guarantees an end-to-end run in the Kaggle environment and writes a valid `submission.csv` with the required `id,pressure` columns. The resulting score won’t hit the target, but it produce a valid submission so you can iterate from a working baseline.'
- What this solution (achieved 3.74233) has done: 'Your current score is far worse than the target (lower is better), mainly because you’re predicting a single constant pressure for every time step. To move the MAE strongly toward the target while keeping changes minimal and staying within the “snap to known pressures” core idea, I replace the constant baseline with a lightweight kNN regressor trained on the provided train.csv using only the existing tabular inputs (R, C, time_step, u_in, u_out). I also match the evaluation semantics by training only on inspiratory points (u_out==0), since expiratory points are not scored. Finally, I keep your existing `find_nearest` snapping so outputs stay on the discrete pressure grid and write a valid `submission.csv`.'
- What this solution (achieved 8.32266) has done: 'Your current score (3.74233, lower is better) is far from the target (0.1369), so we need a meaningful but still minimal change that keeps your kNN approach intact. The biggest issue is that you train only on inspiratory rows (`u_out==0`) but you predict test rows for both phases, which need different behavior; we keep kNN for inspiratory and set expiratory predictions to a stable baseline computed from expiratory training rows. To better respect breath structure without changing the model type, we add two very lightweight “lag” features (previous u_in and previous u_out within the same breath) which usually reduces MAE substantially for this competition while preserving the same core tabular kNN logic. Finally, we keep your existing snapping to the discrete pressure grid and still write a valid `submission.csv`.'
- What this solution (achieved 8.70398) has done: 'Your current MAE (8.32266; lower is better) is far from the target (0.1369), and the main issue is that the kNN is trying to regress absolute pressure from raw inputs without using the key time-series physics proxy features that make this competition workable. Keeping your exact core approach (tabular feature engineering + StandardScaler + KNeighborsRegressor + snapping to the known discrete pressure grid), I add a few minimal, competition-standard cumulative/lag features per breath (cumulative u_in area, cumulative u_out time, and delta u_in) that strongly improve kNN locality while preserving your training/prediction flow. I also ensure the feature computation is identical for train/test and keep your “train on inspiratory only + expiratory baseline” semantics unchanged. This should move the score substantially toward the target without changing model type, loss, or introducing any approximations.'
- What this solution (achieved 8.7098) has done: 'Your current MAE is much worse than the target (lower is better), so we need a small but meaningful improvement while keeping your exact pipeline (feature engineering → StandardScaler → KNN → snap to pressure grid → expiratory baseline). The most impactful minimal change for this competition is to (1) avoid “training only on u_out==0” without incorporating the mask into prediction semantics, by keeping the inspiratory KNN but also forcing *test* expiratory points to a learned expiratory baseline per (R, C) group rather than a single global constant. Additionally, we add two very lightweight, standard per-breath cumulative features (`u_in_cumsum`, `u_out_cumsum`) and one interaction (`R*C`) that improve KNN locality without changing the modeling approach. These changes keep your core logic intact and typically reduce MAE substantially compared with a single expiratory constant and fewer state-tracking features. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 8.58384) has done: 'The timeout is dominated by fitting/predicting a scikit-learn KNN regressor on millions of inspiratory rows: exact KNN scales poorly and can easily exceed 600 seconds. The fastest correctness-preserving win is to keep the same model/metric/weights but switch the neighbor search backend to a tree algorithm (KDTree/BallTree) and to feed the estimator contiguous NumPy arrays (avoids pandas overhead and extra copies). Additionally, the feature engineering currently sorts both train/test and makes several large intermediate objects; we keep the exact same features but reduce overhead by avoiding unnecessary copies, using stable in-place sorting without `ignore_index`, and using vectorized array assembly. Finally, we avoid slow per-row Python dict lookup for expiratory baselines by precomputing keys and using pandas merge-style mapping that is equivalent but faster.'

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

train_path = "../input/ventilator-pressure-prediction/train.csv"
df_train_pressure = pd.read_csv(train_path, usecols=["pressure"])
unique_pressures = df_train_pressure["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)
del df_train_pressure
gc.collect()


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


def snap_to_known_pressures(preds: np.ndarray) -> np.ndarray:
    preds = np.asarray(preds, dtype=np.float64)
    idx = np.searchsorted(sorted_pressures, preds, side="left")

    idx0 = idx == 0
    idxn = idx == total_pressures_len

    idx_clip = idx.copy()
    idx_clip[idx0] = 1  # to allow idx-1 indexing safely; will be overwritten
    idx_clip[idxn] = total_pressures_len - 1

    lower = sorted_pressures[idx_clip - 1]
    upper = sorted_pressures[idx_clip]

    choose_lower = np.abs(lower - preds) < np.abs(upper - preds)
    snapped = np.where(choose_lower, lower, upper)

    snapped[idx0] = sorted_pressures[0]
    snapped[idxn] = sorted_pressures[-1]
    return snapped


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def wc(input_list, expected_len):
    """
    Read and (optionally) weight-combine a small list of submission files.
    Original logic expects filenames like: something.<score> something.csv
    We keep that behavior but make it robust if parsing fails.
    """
    l = []
    preds = []
    for fp in input_list:
        try:
            public_lb_score = int(fp.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1
        l.append(public_lb_score)

        df = pd.read_csv(fp)
        if "pressure" not in df.columns:
            raise ValueError(f"File {fp} has no 'pressure' column.")
        arr = df["pressure"].to_numpy().ravel()
        if len(arr) != expected_len:
            raise ValueError(f"File {fp} has {len(arr)} rows, expected {expected_len}.")
        preds.append(arr)

    l_sum = sum(l) if sum(l) != 0 else 1

    if len(preds) == 1:
        return preds[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        return preds[0] * weight1 + preds[1] * weight2


def g(dp):
    """
    Blend multiple valid submission files from a directory using randomized weights and median ensembling,
    then snap predictions to the nearest known pressure value.
    """
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    output = pd.read_csv(sample_path)
    expected_len = len(output)

    all_files = sorted(glob.glob(os.path.join(dp, "*.csv")))
    valid_files = []
    for fp in all_files:
        try:
            dfi = pd.read_csv(fp, usecols=["pressure"])
            if len(dfi) == expected_len:
                valid_files.append(fp)
        except Exception:
            continue

    if len(valid_files) == 0:
        raise FileNotFoundError(
            f"No valid submission CSVs found in {dp}. "
            f"Expected .csv files with a 'pressure' column and {expected_len} rows."
        )

    file_count = len(valid_files)
    loop_time = 154

    splits = max(1, file_count // 2)
    flist = []
    for i in range(splits):
        start = i * round(file_count / splits)
        end = None if i == splits - 1 else (i + 1) * round(file_count / splits)
        chunk = valid_files[start:end]
        if len(chunk) == 0:
            continue
        flist.append(chunk)

    for i in range(len(flist)):
        flist[i] = wc(flist[i], expected_len=expected_len)

    pred_list = []
    for seed in range(loop_time):
        weight = []
        set_seed(seed)
        for _ in range(len(flist)):
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

    output["pressure"] = np.median(np.vstack(pred_list), axis=0)
    output["pressure"] = snap_to_known_pressures(output["pressure"].to_numpy())

    out_path = f"rwb {loop_time} loops.csv"
    output.to_csv(out_path, index=False)
    return out_path


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a["pressure"] = a["pressure"] * 0.6 + b["pressure"] * 0.4
    a["pressure"] = snap_to_known_pressures(a["pressure"].to_numpy())
    a.to_csv("blend.csv", index=False)
    return a




## === cell 1
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

train_path = "../input/ventilator-pressure-prediction/train.csv"
test_path = "../input/ventilator-pressure-prediction/test.csv"
sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"

train_usecols = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
test_usecols = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "id"]
dtypes_train = {
    "breath_id": np.int32,
    "R": np.int16,
    "C": np.int16,
    "time_step": np.float32,
    "u_in": np.float32,
    "u_out": np.int8,
    "pressure": np.float32,
}
dtypes_test = {
    "breath_id": np.int32,
    "R": np.int16,
    "C": np.int16,
    "time_step": np.float32,
    "u_in": np.float32,
    "u_out": np.int8,
    "id": np.int32,
}

df_train = pd.read_csv(train_path, usecols=train_usecols, dtype=dtypes_train)
df_test = pd.read_csv(test_path, usecols=test_usecols, dtype=dtypes_test)
sub = pd.read_csv(sample_path, usecols=["id"])


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.sort_values(["breath_id", "time_step"], inplace=True, kind="mergesort")

    breath = df["breath_id"].to_numpy()
    u_in = df["u_in"].to_numpy(dtype=np.float64, copy=False)
    u_out = df["u_out"].to_numpy(copy=False)
    t = df["time_step"].to_numpy(dtype=np.float64, copy=False)

    n = len(df)
    lag1_in = np.empty(n, dtype=np.float64)
    lag2_in = np.empty(n, dtype=np.float64)
    lag1_out = np.empty(n, dtype=u_out.dtype)
    lag2_out = np.empty(n, dtype=u_out.dtype)

    same_prev = np.empty(n, dtype=bool)
    same_prev[0] = False
    same_prev[1:] = breath[1:] == breath[:-1]

    same_prev2 = np.empty(n, dtype=bool)
    same_prev2[:2] = False
    same_prev2[2:] = (breath[2:] == breath[:-2]) & (breath[2:] == breath[1:-1])

    lag1_in[0] = 0.0
    lag1_in[1:] = np.where(same_prev[1:], u_in[:-1], 0.0)
    lag2_in[:2] = 0.0
    lag2_in[2:] = np.where(same_prev2[2:], u_in[:-2], 0.0)

    lag1_out[0] = 0
    lag1_out[1:] = np.where(same_prev[1:], u_out[:-1], 0).astype(
        u_out.dtype, copy=False
    )
    lag2_out[:2] = 0
    lag2_out[2:] = np.where(same_prev2[2:], u_out[:-2], 0).astype(
        u_out.dtype, copy=False
    )

    dt = np.empty(n, dtype=np.float64)
    dt[0] = 0.0
    dt[1:] = np.where(same_prev[1:], t[1:] - t[:-1], 0.0)

    start = np.empty(n, dtype=bool)
    start[0] = True
    start[1:] = ~same_prev[1:]
    start_idx = np.flatnonzero(start)

    u_in_cumsum = np.cumsum(u_in, dtype=np.float64)
    u_out_cumsum = np.cumsum(u_out.astype(np.float64, copy=False), dtype=np.float64)
    u_in_cumarea_raw = np.cumsum(u_in * dt, dtype=np.float64)
    u_out_cumtime_raw = np.cumsum(
        u_out.astype(np.float64, copy=False) * dt, dtype=np.float64
    )

    base_in = np.zeros(n, dtype=np.float64)
    base_out = np.zeros(n, dtype=np.float64)
    base_area = np.zeros(n, dtype=np.float64)
    base_outtime = np.zeros(n, dtype=np.float64)

    prev_idx = start_idx - 1
    valid = prev_idx >= 0
    si = start_idx[valid]
    pi = prev_idx[valid]
    base_in[si] = u_in_cumsum[pi]
    base_out[si] = u_out_cumsum[pi]
    base_area[si] = u_in_cumarea_raw[pi]
    base_outtime[si] = u_out_cumtime_raw[pi]

    base_in = np.maximum.accumulate(base_in)
    base_out = np.maximum.accumulate(base_out)
    base_area = np.maximum.accumulate(base_area)
    base_outtime = np.maximum.accumulate(base_outtime)

    u_in_cumsum_b = u_in_cumsum - base_in
    u_out_cumsum_b = u_out_cumsum - base_out
    u_in_cumarea = u_in_cumarea_raw - base_area
    u_out_cumtime = u_out_cumtime_raw - base_outtime

    u_in_cumarea_lag1 = np.empty(n, dtype=np.float64)
    u_in_cumarea_lag1[0] = 0.0
    u_in_cumarea_lag1[1:] = np.where(same_prev[1:], u_in_cumarea[:-1], 0.0)

    df["u_in_lag1"] = lag1_in
    df["u_in_lag2"] = lag2_in
    df["u_out_lag1"] = lag1_out
    df["u_out_lag2"] = lag2_out
    df["u_in_diff1"] = u_in - lag1_in
    df["u_in_diff2"] = lag1_in - lag2_in
    df["dt"] = dt
    df["u_in_cumarea"] = u_in_cumarea
    df["u_in_cumarea_lag1"] = u_in_cumarea_lag1
    df["u_out_cumtime"] = u_out_cumtime
    df["u_in_cumsum"] = u_in_cumsum_b
    df["u_out_cumsum"] = u_out_cumsum_b

    r = df["R"].to_numpy(dtype=np.float64, copy=False)
    c = df["C"].to_numpy(dtype=np.float64, copy=False)
    df["RC"] = r * c

    return df


df_train_fe = add_lag_features(df_train)
df_test_fe = add_lag_features(df_test)

features = [
    "R",
    "C",
    "RC",
    "time_step",
    "u_in",
    "u_out",
    "u_in_lag1",
    "u_in_lag2",
    "u_out_lag1",
    "u_out_lag2",
    "u_in_diff1",
    "u_in_diff2",
    "dt",
    "u_in_cumarea",
    "u_in_cumarea_lag1",
    "u_out_cumtime",
    "u_in_cumsum",
    "u_out_cumsum",
]

mask_insp_train = df_train_fe["u_out"].to_numpy() == 0

X_train = np.ascontiguousarray(
    df_train_fe.loc[mask_insp_train, features].to_numpy(dtype=np.float64, copy=False)
)
y_train = np.ascontiguousarray(
    df_train_fe.loc[mask_insp_train, "pressure"].to_numpy(dtype=np.float64, copy=False)
)

model = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "knn",
            KNeighborsRegressor(
                n_neighbors=35,
                weights="distance",
                metric="minkowski",
                p=2,
                algorithm="kd_tree",
                leaf_size=40,
                n_jobs=-1,
            ),
        ),
    ]
)

model.fit(X_train, y_train)

mask_exp_train = ~mask_insp_train
df_tr_exp = df_train_fe.loc[mask_exp_train, ["R", "C", "time_step", "pressure"]]

if len(df_tr_exp) > 0:
    ts_key = df_tr_exp["time_step"].round(2)
    exp_baseline_by_rct = (
        df_tr_exp.assign(time_step_key=ts_key)
        .groupby(["R", "C", "time_step_key"], sort=False)["pressure"]
        .median()
    )
    exp_baseline_by_rc = df_tr_exp.groupby(["R", "C"], sort=False)["pressure"].median()
    global_exp_baseline = float(df_tr_exp["pressure"].median())
else:
    exp_baseline_by_rct = pd.Series(dtype=np.float64)
    exp_baseline_by_rc = pd.Series(dtype=np.float64)
    global_exp_baseline = float(df_train_fe["pressure"].median())

global_exp_baseline = float(find_nearest(global_exp_baseline))

X_test_all = np.ascontiguousarray(
    df_test_fe[features].to_numpy(dtype=np.float64, copy=False)
)
pred_insp = model.predict(X_test_all).astype(np.float64, copy=False)

pred_final = pred_insp.copy()
is_exp = df_test_fe["u_out"].to_numpy() == 1

if is_exp.any():
    exp_rows = df_test_fe.loc[is_exp, ["R", "C", "time_step"]].copy()
    exp_rows["time_step_key"] = exp_rows["time_step"].round(2)

    idx_rct = pd.MultiIndex.from_frame(exp_rows[["R", "C", "time_step_key"]])
    rct_vals = exp_baseline_by_rct.reindex(idx_rct).to_numpy()

    idx_rc = pd.MultiIndex.from_frame(exp_rows[["R", "C"]])
    rc_vals = exp_baseline_by_rc.reindex(idx_rc).to_numpy()

    exp_preds = np.where(
        np.isfinite(rct_vals),
        rct_vals,
        np.where(np.isfinite(rc_vals), rc_vals, global_exp_baseline),
    ).astype(np.float64, copy=False)

    pred_final[is_exp] = exp_preds

pred_snapped = snap_to_known_pressures(pred_final)

sub["pressure"] = pred_snapped
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print(
    "Pred pressure stats:",
    float(np.min(pred_snapped)),
    float(np.max(pred_snapped)),
    float(np.mean(pred_snapped)),
)
print("Global exp baseline (snapped):", global_exp_baseline)
print("Test u_out==1 count:", int(is_exp.sum()))
