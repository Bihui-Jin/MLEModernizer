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

0.136615405700708

# 6. Current score

1.25419

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 10.86378) has done: 'The crash happens because `avg('../input/gb-data-blending-recover')` points to a dataset directory that doesn’t exist in your provided environment, so `glob` finds zero files and `np.vstack` fails. I keep your blending/rounding-to-nearest-pressure logic intact, but add a safe fallback: if no blend files are found, create a baseline submission by predicting the global median train pressure for all rows (then snapping to nearest valid pressure). I also make path resolution robust by automatically selecting an existing competition data root (`/kaggle/input/...`), so reading `train.csv`/`sample_submission.csv` works in this environment. The script always write a valid `submission.csv` with `id,pressure`.'
- What this solution (achieved 9.92097) has done: 'Your current score (10.86378 MAE) is far worse than the target (0.1366), and the main reason is that the code falls back to predicting a constant median pressure because the blend directory doesn’t exist in this environment. To move the score sharply toward the target without changing the overall “make predictions then snap to nearest valid pressure and write submission.csv” semantics, I keep your rounding-to-nearest-pressure logic but replace the fallback with a simple, fast, fully local baseline model: per-(R,C,time_step) median pressure computed from train and joined onto test. For any unmatched rows, it safely fall back to the global median, then still apply your `find_nearest` snapping. This stays within Kaggle constraints, runs quickly on CPU, and should drastically reduce MAE compared with a constant prediction.'
- What this solution (achieved 1.62713) has done: 'Your score is still far from the target because the fallback model (median by `(R,C,time_step)`) ignores the strong within-breath dynamics driven by `u_in/u_out` and accumulated flow; it effectively predicts a coarse lookup table. I keep your overall “predict → snap to nearest valid pressure → write submission.csv” semantics, but improve the fallback to a stronger (still simple/fast) per-time-step KNN regression using only local features (`R,C,time_step,u_in,u_out` plus engineered `u_in` cumulative sums within each breath). This is a minimal change contained entirely inside `avg()`’s “no blend files found” path, so blending logic remains intact. The result should move MAE sharply down toward the target band while still finishing within the time limit and producing a valid `submission.csv`.'
- What this solution (achieved 1.25419) has done: 'Your current MAE (1.627) is still far above the target (0.1366), so we should improve the fallback model (used when no blend files exist) while keeping your overall “predict → snap to nearest valid pressure → write submission.csv” semantics unchanged. The main issue is that KNN is being fit on all 5.4M rows with unscaled features, which both hurts performance and (on Kaggle) is often too slow/memory-heavy; we can instead use a compact, breath-structured representation (80 steps per breath) and standardize features before KNN to improve neighbor quality. Concretely: train KNN on per-timestep features + lag features on a sampled set of breaths (still deterministic, no approximations to training loop since there isn’t one) and then predict test in chunks, finally applying your existing nearest-pressure snapping. These are minimal, localized changes inside the fallback path only and should move MAE substantially down toward the target band while staying within time limits and producing a valid submission.'

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
def _pick_data_root():
    candidates = [
        "../input/ventilator-pressure-prediction",
        "/kaggle/input/ventilator-pressure-prediction",
        "/kaggle/data/ventilator-pressure-prediction",
        "/kaggle/input",
        "/kaggle/data",
    ]
    for c in candidates:
        if os.path.exists(c):
            if os.path.isdir(c) and os.path.exists(os.path.join(c, "train.csv")):
                return c
            if os.path.isdir(c) and os.path.exists(
                os.path.join(c, "ventilator-pressure-prediction", "train.csv")
            ):
                return os.path.join(c, "ventilator-pressure-prediction")
    return "."


DATA_ROOT = _pick_data_root()
TRAIN_PATH = os.path.join(DATA_ROOT, "train.csv")
TEST_PATH = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

df_train = pd.read_csv(TRAIN_PATH)

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
    output = pd.read_csv(SAMPLE_SUB_PATH)
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


def avg(dp, out_path="submission.csv"):
    input_list = []
    for i in glob.iglob(f"{dp}/*"):
        if os.path.isfile(i) and i.lower().endswith(".csv"):
            input_list.append(i)

    output = pd.read_csv(SAMPLE_SUB_PATH)

    def _knn_fallback():
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.preprocessing import StandardScaler

        df_test = pd.read_csv(TEST_PATH)

        def _add_features(df, is_train):
            df = df.copy()
            g = df.groupby("breath_id", sort=False)

            df["u_in_cumsum"] = g["u_in"].cumsum()
            df["u_in_cummean"] = df["u_in_cumsum"] / (g.cumcount() + 1)

            df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0)
            df["u_out_lag1"] = g["u_out"].shift(1).fillna(0.0)
            df["du_in"] = df["u_in"] - df["u_in_lag1"]

            df["u_in_x_time"] = df["u_in"] * df["time_step"]

            return df

        tr = _add_features(
            df_train[["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]],
            is_train=True,
        )
        te = _add_features(
            df_test[["breath_id", "R", "C", "time_step", "u_in", "u_out"]],
            is_train=False,
        )

        feat_cols = [
            "R",
            "C",
            "time_step",
            "u_in",
            "u_out",
            "u_in_cumsum",
            "u_in_cummean",
            "u_in_lag1",
            "u_out_lag1",
            "du_in",
            "u_in_x_time",
        ]

        set_seed(2021)
        unique_breaths = tr["breath_id"].drop_duplicates().to_numpy()
        n_breaths = min(30000, unique_breaths.shape[0])
        chosen = np.random.choice(unique_breaths, size=n_breaths, replace=False)
        tr_small = tr[tr["breath_id"].isin(chosen)]

        X_train = tr_small[feat_cols].to_numpy(dtype=np.float32, copy=False)
        y_train = tr_small["pressure"].to_numpy(dtype=np.float32, copy=False)
        X_test = te[feat_cols].to_numpy(dtype=np.float32, copy=False)

        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train_sc = scaler.fit_transform(X_train)
        knn = KNeighborsRegressor(
            n_neighbors=35,
            weights="distance",
            metric="minkowski",
            p=2,
            n_jobs=-1,
        )
        knn.fit(X_train_sc, y_train)

        pred = np.empty(X_test.shape[0], dtype=np.float32)
        bs = 200000
        for start in range(0, X_test.shape[0], bs):
            end = min(start + bs, X_test.shape[0])
            Xb = scaler.transform(X_test[start:end])
            pred[start:end] = knn.predict(Xb).astype(np.float32, copy=False)

        out = output.copy()
        out["pressure"] = pred
        out["pressure"] = out["pressure"].apply(find_nearest)
        out.to_csv(out_path, index=False)
        return out

    if len(input_list) == 0:
        return _knn_fallback()

    preds = []
    for fp in input_list:
        dfp = pd.read_csv(fp)
        if "pressure" not in dfp.columns:
            continue
        p = dfp["pressure"].to_numpy().ravel()
        if len(p) != len(output):
            continue
        preds.append(p)

    if len(preds) == 0:
        return _knn_fallback()

    output["pressure"] = np.median(np.vstack(preds), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv(out_path, index=False)
    return output




## === cell 2
avg("../input/gb-data-blending-recover", out_path="submission.csv")
sub = pd.read_csv("submission.csv")
print("Wrote submission.csv with columns:", list(sub.columns), "and rows:", len(sub))
print(sub.head())
