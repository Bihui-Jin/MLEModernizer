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

0.1471889433807456

# 6. Current score

5.17588

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 14.73489) has done: 'I fix the failure in `g()` by making it robust to missing/empty blend directories and to reading malformed or wrong-length prediction files; this is the root cause of `pred_list` ending up with arrays of the wrong shape (often length 1), which then breaks assignment to the 603600-row submission. I also ensure the code always produces a valid `submission.csv` (correct name/suffix and columns) even when no external blend files exist, by falling back to a safe baseline prediction. Finally, I keep the original blending logic intact where possible, only adding input validation, path fallbacks, and a deterministic baseline so the notebook runs end-to-end in this environment.'
- What this solution (achieved 8.23478) has done: 'Your current score (14.73489 MAE, lower-is-better) is far from the target (0.1472), and the main reason is that the code is producing a near-constant “baseline” submission whenever `../input/gb-blending` doesn’t exist, which performs extremely poorly on this competition. I keep your overall structure (read train to get allowed pressure grid, then create a submission), but replace the fallback baseline with a simple, fully in-notebook KNN regressor trained on the provided `train.csv` features and predicting `test.csv` pressures; this is a minimal modeling addition that stays within sklearn/pandas/numpy and keeps runtime under the limit. I also ensure predictions are snapped to the nearest allowed pressure (as you already do) and that row alignment matches `sample_submission.csv` by predicting in `test.csv` order and then writing `id,pressure`. The blending path remains intact: if valid blend files exist, it still blend; otherwise it use the stronger KNN fallback to move the score much closer to the target band.'
- What this solution (achieved 5.17588) has done: 'Your current MAE (8.23478, lower-is-better) is still far from the target (0.14719), and the biggest remaining issue is that the KNN fallback is learning a pointwise mapping without any per-breath temporal context, which is crucial in this competition. I keep your blend-or-fallback structure unchanged, but minimally upgrade the fallback feature set to include simple lag/cumulative features within each `breath_id` (no new model family, still KNN) so predictions move substantially toward the target. I also keep the “snap to nearest allowed pressure” post-processing intact, and ensure we predict in `test.csv` order and write a valid `submission.csv`. Finally, I reduce memory pressure by loading only needed columns and using float32 where safe (no approximation changes to the logic).'

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
from sklearn.neighbors import KNeighborsRegressor

df_train = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv", usecols=["pressure"]
)
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
    Original intent: read each submission, infer a 'score' from filename, then do a weighted combo.
    Bugfix: robustly handle filenames without the expected pattern and files that don't match expected length.
    """
    l = []
    preds = []
    for i in range(len(input_list)):
        fp = input_list[i]
        try:
            public_lb_score = int(fp.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1
        l.append(public_lb_score)

        df = pd.read_csv(fp)
        if "pressure" not in df.columns:
            raise ValueError(f"File {fp} does not contain 'pressure' column.")
        preds.append(df["pressure"].to_numpy().ravel())

    l_sum = sum(l) if sum(l) != 0 else 1

    if len(preds) == 1:
        return preds[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        return preds[0] * weight1 + preds[1] * weight2


def _add_breath_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Why change (score-moving, minimal): KNN fallback previously used only raw timestep inputs,
    missing crucial per-breath temporal context. Adding lightweight lag/cumulative features
    preserves the same KNN approach but materially improves MAE toward the target.
    """
    df = df.sort_values(["breath_id", "time_step"], kind="mergesort").copy()

    g = df.groupby("breath_id", sort=False)

    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0)
    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0.0)

    df["u_in_cumsum"] = g["u_in"].cumsum()
    df["u_in_diff1"] = (df["u_in"] - df["u_in_lag1"]).astype(np.float32)

    df["t_idx"] = g.cumcount().astype(np.float32)

    return df


def _knn_fallback_submission(
    sample_path="../input/ventilator-pressure-prediction/sample_submission.csv",
    train_path="../input/ventilator-pressure-prediction/train.csv",
    test_path="../input/ventilator-pressure-prediction/test.csv",
):
    """
    KNN fallback (same model family/approach) but with minimal temporal features per breath
    to move MAE substantially closer to the target.
    """
    sample = pd.read_csv(sample_path)

    train = pd.read_csv(
        train_path,
        usecols=["R", "C", "time_step", "u_in", "u_out", "breath_id", "pressure"],
    )
    test = pd.read_csv(
        test_path,
        usecols=["R", "C", "time_step", "u_in", "u_out", "breath_id", "id"],
    )

    train = _add_breath_features(train)
    test = _add_breath_features(test)

    feat_cols = [
        "R",
        "C",
        "time_step",
        "t_idx",
        "u_in",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_diff1",
        "u_in_cumsum",
        "u_out",
        "u_out_lag1",
        "breath_id",
    ]

    X_train = train[feat_cols].to_numpy(dtype=np.float32, copy=False)
    y_train = train["pressure"].to_numpy(dtype=np.float32, copy=False)
    X_test = test[feat_cols].to_numpy(dtype=np.float32, copy=False)

    knn = KNeighborsRegressor(
        n_neighbors=50, weights="distance", metric="minkowski", p=2, n_jobs=-1
    )
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test).astype(np.float64)

    pred = np.vectorize(find_nearest, otypes=[np.float64])(pred)

    out = pd.DataFrame({"id": test["id"].to_numpy(), "pressure": pred})
    out = out.sort_values("id").reset_index(drop=True)

    out = sample[["id"]].merge(out, on="id", how="left")
    if out["pressure"].isna().any():
        base = float(np.median(sorted_pressures))
        base = find_nearest(base)
        out["pressure"] = out["pressure"].fillna(base)

    return out


def g(dp):
    """
    Original ensemble/blend driver.
    Bugfixes kept:
      - Handle missing directory or no candidate files.
      - Ensure each prediction array has the correct length (match sample_submission rows).
    Key improvement for score:
      - If nothing valid is found, fall back to a stronger (still KNN) in-notebook model.
    """
    sample_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
    output = pd.read_csv(sample_path)
    n_rows = len(output)

    files = []
    if dp is not None and os.path.isdir(dp):
        for fp in glob.iglob(f"{dp}/*"):
            if os.path.isfile(fp) and fp.lower().endswith(".csv"):
                files.append(fp)

    files.sort()
    if len(files) == 0:
        sub = _knn_fallback_submission(sample_path=sample_path)
        sub.to_csv("submission.csv", index=False)
        return sub

    file_count = len(files)
    loop_time = 150
    splits = file_count // 2
    if splits < 1:
        splits = 1

    flist = []
    for i in range(splits):
        start = i * round(len(files) / splits)
        end = None if i == splits - 1 else (i + 1) * round(len(files) / splits)
        chunk = files[start:end]
        if len(chunk) == 0:
            continue
        try:
            blended = wc(chunk)
        except Exception:
            continue
        flist.append(blended)

    flist_valid = []
    for arr in flist:
        arr = np.asarray(arr).ravel()
        if arr.shape[0] == n_rows:
            flist_valid.append(arr)
    flist = flist_valid

    if len(flist) == 0:
        sub = _knn_fallback_submission(sample_path=sample_path)
        sub.to_csv("submission.csv", index=False)
        return sub

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

        temp = np.zeros(n_rows, dtype=np.float64)
        for j in range(len(flist)):
            temp += flist[j] * weight[j]
        pred_list.append(temp)
        del temp
        gc.collect()

    stacked = np.vstack(pred_list)  # (loop_time, n_rows)
    output["pressure"] = np.median(stacked, axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)

    output.to_csv("submission.csv", index=False)
    output.to_csv(f"rwb {loop_time} loops.csv", index=False)
    return output


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.5 + b.pressure * 0.5
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a




## === cell 2
g("../input/gb-blending")
print("Wrote submission.csv")
print(pd.read_csv("submission.csv").head())
print(pd.read_csv("submission.csv").shape)
print(pd.read_csv("submission.csv").columns.tolist())
