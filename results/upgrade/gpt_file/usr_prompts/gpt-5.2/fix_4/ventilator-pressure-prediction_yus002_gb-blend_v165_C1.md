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

_sample_sub_path = "../input/ventilator-pressure-prediction/sample_submission.csv"
_sample_sub_df = pd.read_csv(_sample_sub_path)
EXPECTED_LEN = len(_sample_sub_df)

_test_path = "../input/ventilator-pressure-prediction/test.csv"


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


def _load_pressure_vector(csv_path, expected_len=EXPECTED_LEN):
    """
    Bugfix: enforce that every prediction file contributes a 1D vector of the
    expected length. If not, skip it.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if "pressure" not in df.columns:
        return None
    arr = df["pressure"].to_numpy()
    if arr.ndim != 1 or len(arr) != expected_len:
        return None
    return arr


def _add_baseline_features(df):
    """
    Score-fix (still minimal and within core semantics): add common leakage-free
    ventilator features computed only from inputs within each breath.
    These features are standard for improving MAE without changing the overall
    "per-row regression then snap" approach.
    """
    df = df.copy()

    df["R"] = df["R"].astype("int16")
    df["C"] = df["C"].astype("int16")

    grp = df.groupby("breath_id", sort=False)
    df["u_in_lag1"] = grp["u_in"].shift(1).fillna(0.0)
    df["u_in_lag2"] = grp["u_in"].shift(2).fillna(0.0)
    df["u_out_lag1"] = grp["u_out"].shift(1).fillna(0).astype(df["u_out"].dtype)

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_in_diff2"] = df["u_in_lag1"] - df["u_in_lag2"]

    df["dt"] = grp["time_step"].diff().fillna(0.0)
    df["area"] = (df["u_in"] * df["dt"]).groupby(df["breath_id"], sort=False).cumsum()

    for c in ["u_in_lag1", "u_in_lag2", "u_in_diff1", "u_in_diff2", "dt", "area"]:
        df[c] = df[c].astype(np.float32)

    return df


def _fallback_knn_predict(
    test_path=_test_path, train_df=df_train, expected_len=EXPECTED_LEN
):
    """
    Score-fix (minimal, only used when blending files are missing/invalid):
    Strengthen the fallback model (still scikit-learn regression) by adding
    standard per-breath lag/area features and switching to a small RandomForest
    regressor which typically improves MAE substantially over raw KNN here.

    This preserves pipeline semantics:
      inputs -> regression prediction per timestep -> nearest-pressure snapping.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception:
        return None

    try:
        test_df = pd.read_csv(test_path)
    except Exception:
        return None

    train_fe = _add_baseline_features(train_df)
    test_fe = _add_baseline_features(test_df)

    train_fe = pd.get_dummies(train_fe, columns=["R", "C"], prefix=["R", "C"])
    test_fe = pd.get_dummies(test_fe, columns=["R", "C"], prefix=["R", "C"])

    target_col = "pressure"
    drop_cols = [target_col]
    feat_train = train_fe.drop(columns=drop_cols, errors="ignore")
    feat_test = test_fe.copy()

    feat_train, feat_test = feat_train.align(
        feat_test, join="left", axis=1, fill_value=0
    )

    feat_train = feat_train.select_dtypes(include=[np.number])
    feat_test = feat_test[feat_train.columns]

    X_train = feat_train.to_numpy(dtype=np.float32, copy=False)
    y_train = train_fe[target_col].to_numpy(dtype=np.float32, copy=False)
    X_test = feat_test.to_numpy(dtype=np.float32, copy=False)

    model = RandomForestRegressor(
        n_estimators=120,
        random_state=2021,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features="sqrt",
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    if pred.ndim != 1 or len(pred) != expected_len:
        return None
    return pred


def wc(input_list):
    """
    Keep original intent: make a weighted combination of (up to) two files.
    Bugfix: be robust to unexpected filenames; if parsing fails, fall back to
    equal weighting.
    """
    l = []
    vecs = []
    for i in range(len(input_list)):
        try:
            public_lb_score = int(
                input_list[i].split("/")[-1].split(".")[1].split(" ")[0]
            )
        except Exception:
            public_lb_score = 1
        v = _load_pressure_vector(input_list[i])
        if v is None:
            continue
        l.append(public_lb_score)
        vecs.append(v)

    if len(vecs) == 0:
        return None
    if len(vecs) == 1:
        return vecs[0]

    l_sum = sum(l) if sum(l) != 0 else len(l)
    weight1 = (l[1] / l_sum) + 0.1
    weight2 = 1 - weight1
    output = vecs[0] * weight1 + vecs[1] * weight2
    return output


def g(dp):
    """
    Bugfixes:
    - Filter to valid prediction CSVs and enforce correct length.
    - Avoid empty splits that lead to invalid shapes (length 1).
    - Ensure final output vector length matches sample submission.

    Score-fix:
    - If no valid external prediction CSVs are found, use a stronger baseline
      trained on train.csv instead of writing the all-zero sample submission.
    """
    files = sorted([p for p in glob.glob(os.path.join(dp, "*")) if os.path.isfile(p)])
    files = [p for p in files if p.lower().endswith(".csv")]

    loop_time = 154

    if len(files) == 0:
        fallback = _fallback_knn_predict()
        output = pd.read_csv(_sample_sub_path)
        if fallback is None:
            output.to_csv("submission.csv", index=False)
            return output
        output["pressure"] = fallback
        output["pressure"] = output["pressure"].apply(find_nearest)
        output.to_csv("submission.csv", index=False)
        return output

    splits = 2 if len(files) >= 2 else 1
    flist_groups = np.array_split(files, splits)

    flist = []
    for grp in flist_groups:
        grp = list(grp)
        if len(grp) == 0:
            continue
        combined = wc(grp)
        if combined is None:
            continue
        if len(combined) != EXPECTED_LEN:
            continue
        flist.append(combined)

    if len(flist) == 0:
        fallback = _fallback_knn_predict()
        output = pd.read_csv(_sample_sub_path)
        if fallback is None:
            output.to_csv("submission.csv", index=False)
            return output
        output["pressure"] = fallback
        output["pressure"] = output["pressure"].apply(find_nearest)
        output.to_csv("submission.csv", index=False)
        return output

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

        temp = np.zeros(EXPECTED_LEN, dtype=np.float64)
        for i in range(len(flist)):
            temp += flist[i] * weight[i]
        pred_list.append(temp)
        del temp
        if (t + 1) % 25 == 0:
            gc.collect()

    output = pd.read_csv(_sample_sub_path)
    stacked = np.vstack(pred_list)  # (loop_time, EXPECTED_LEN)
    median_pred = np.median(stacked, axis=0)
    mean_pred = stacked.mean(axis=0)
    blended = 0.8 * median_pred + 0.2 * mean_pred

    output["pressure"] = blended
    output["pressure"] = output["pressure"].apply(find_nearest)

    output.to_csv(f"rwb {loop_time} loops.csv", index=False)
    output.to_csv("submission.csv", index=False)
    return output


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    a.pressure = a.pressure * 0.7 + b.pressure * 0.3
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    a.to_csv("submission.csv", index=False)
    return a


def avg(dp):
    input_list = []
    for i in glob.iglob(f"{dp}/*"):
        if os.path.isfile(i) and i.lower().endswith(".csv"):
            input_list.append(i)

    vecs = []
    for p in input_list:
        v = _load_pressure_vector(p)
        if v is not None:
            vecs.append(v)

    output = pd.read_csv(_sample_sub_path)
    if len(vecs) == 0:
        fallback = _fallback_knn_predict()
        if fallback is None:
            output.to_csv("submission.csv", index=False)
            return output
        output["pressure"] = fallback
        output["pressure"] = output["pressure"].apply(find_nearest)
        output.to_csv("submission.csv", index=False)
        return output

    output.pressure = np.median(np.vstack(vecs), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv("avg.csv", index=False)
    output.to_csv("submission.csv", index=False)
    return output




## === cell 2
g("../input/gb-data-blending-recover")
