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
import glob
import random
from random import random as rd
import gc

from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TRAIN_PATH_CANDIDATES = [
    "../input/ventilator-pressure-prediction/train.csv",
    "/kaggle/input/ventilator-pressure-prediction/train.csv",
    "/kaggle/data/train.csv",
    "/kaggle/input/train.csv",
]
SAMPLE_SUB_PATH_CANDIDATES = [
    "../input/ventilator-pressure-prediction/sample_submission.csv",
    "/kaggle/input/ventilator-pressure-prediction/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
]
TEST_PATH_CANDIDATES = [
    "../input/ventilator-pressure-prediction/test.csv",
    "/kaggle/input/ventilator-pressure-prediction/test.csv",
    "/kaggle/data/test.csv",
    "/kaggle/input/test.csv",
]


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these paths exist: {paths}")


TRAIN_PATH = _first_existing(TRAIN_PATH_CANDIDATES)
SAMPLE_SUB_PATH = _first_existing(SAMPLE_SUB_PATH_CANDIDATES)
TEST_PATH = _first_existing(TEST_PATH_CANDIDATES)

df_train_pressure = pd.read_csv(TRAIN_PATH, usecols=["pressure"])
unique_pressures = df_train_pressure["pressure"].unique()
sorted_pressures = np.sort(unique_pressures)
total_pressures_len = len(sorted_pressures)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
EXPECTED_LEN = len(sample_sub)


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


def set_seed(seed=2021):
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


def _is_submission_like_csv(path, expected_len=EXPECTED_LEN):
    try:
        head = pd.read_csv(path, nrows=5)
    except Exception:
        return False
    cols = [c.lower() for c in head.columns]
    if set(cols) != {"id", "pressure"}:
        return False
    try:
        n = sum(1 for _ in open(path, "rb")) - 1  # header excluded
    except Exception:
        try:
            n = len(pd.read_csv(path, usecols=["id"]))
        except Exception:
            return False
    return n == expected_len


def _read_submission_pressure(path, sample_ids):
    """Read submission, align by sample_ids if needed, and return pressure vector length EXPECTED_LEN."""
    df = pd.read_csv(path)
    if "id" in df.columns and "pressure" in df.columns and len(df) == len(sample_ids):
        if not df["id"].equals(sample_ids):
            df = df.set_index("id").reindex(sample_ids).reset_index()
    return df["pressure"].to_numpy().ravel()


def wc(input_list):
    l = []
    for i in range(len(input_list)):
        fname = input_list[i].split("/")[-1]
        try:
            public_lb_score = int(fname.split(".")[1].split(" ")[0])
        except Exception:
            public_lb_score = 1
        l.append(public_lb_score)
        input_list[i] = _read_submission_pressure(input_list[i], sample_sub["id"])
    output = 0
    l_sum = sum(l) if sum(l) != 0 else 1
    if len(input_list) == 1:
        output = input_list[0]
    else:
        weight1 = (l[1] / l_sum) + 0.1
        weight2 = 1 - weight1
        output += input_list[0] * weight1 + input_list[1] * weight2
    return output


def _add_breath_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    g = df.groupby("breath_id", sort=False)

    df["u_in_cum"] = g["u_in"].cumsum().astype(np.float32)
    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0).astype(np.float32)

    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0).astype(np.int8)
    df["time_step_lag1"] = g["time_step"].shift(1).fillna(0.0).astype(np.float32)

    df["u_in_diff1"] = (df["u_in"].astype(np.float32) - df["u_in_lag1"]).astype(
        np.float32
    )
    df["u_in_cummean"] = (
        df["u_in_cum"] / (g.cumcount() + 1).astype(np.float32)
    ).astype(np.float32)
    return df


def _make_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(
        {
            "time_step": df["time_step"].astype(np.float32),
            "u_in": df["u_in"].astype(np.float32),
            "u_out": df["u_out"].astype(np.int8),
            "R": df["R"].astype(np.int16),
            "C": df["C"].astype(np.int16),
            "u_in_time": (df["u_in"] * df["time_step"]).astype(np.float32),
            "u_in_cum": df["u_in_cum"].astype(np.float32),
            "u_in_lag1": df["u_in_lag1"].astype(np.float32),
            "u_out_lag1": df["u_out_lag1"].astype(np.int8),
            "time_step_lag1": df["time_step_lag1"].astype(np.float32),
            "u_in_diff1": df["u_in_diff1"].astype(np.float32),
            "u_in_cummean": df["u_in_cummean"].astype(np.float32),
            "pressure_lag1": df["pressure_lag1"].astype(np.float32),
        }
    )
    return X


def _train_and_predict_knn(train_path: str, test_path: str) -> np.ndarray:
    usecols_train = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "pressure"]
    usecols_test = ["breath_id", "R", "C", "time_step", "u_in", "u_out", "id"]

    train = pd.read_csv(train_path, usecols=usecols_train)
    test = pd.read_csv(test_path, usecols=usecols_test)

    train = _add_breath_features(train)
    test = _add_breath_features(test)

    gtr = train.groupby("breath_id", sort=False)
    train["pressure_lag1"] = gtr["pressure"].shift(1).fillna(0.0).astype(np.float32)

    insp_mask = train["u_out"].to_numpy() == 0
    X_train = _make_features(train.loc[insp_mask])
    y_train = train.loc[insp_mask, "pressure"].astype(np.float32).to_numpy()

    model = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        KNeighborsRegressor(
            n_neighbors=50, weights="distance", metric="minkowski", p=1
        ),
    )
    model.fit(X_train, y_train)

    test = test.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
        drop=True
    )
    pred = np.zeros(len(test), dtype=np.float32)

    prev_breath = None
    prev_p = 0.0

    for i in range(len(test)):
        b = int(test.at[i, "breath_id"])
        if prev_breath is None or b != prev_breath:
            prev_breath = b
            prev_p = 0.0

        row = test.iloc[i : i + 1].copy()
        row["pressure_lag1"] = np.float32(prev_p)
        Xi = _make_features(row)
        pi = float(model.predict(Xi)[0])
        pred[i] = np.float32(pi)
        prev_p = pi

    out = test[["id"]].copy()
    out["pressure"] = pred
    out = out.set_index("id").reindex(sample_sub["id"]).reset_index()
    return out["pressure"].to_numpy(dtype=np.float32)


def g(dp):
    output = sample_sub.copy()
    pred = _train_and_predict_knn(TRAIN_PATH, TEST_PATH)
    output["pressure"] = pred
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv("submission.csv", index=False)
    return output


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    if not a["id"].equals(b["id"]):
        b = b.set_index("id").reindex(a["id"]).reset_index()
    a["pressure"] = a["pressure"] * 0.7 + b["pressure"] * 0.3
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a


def avg(dp):
    input_list = []
    if dp is not None:
        for i in glob.iglob(f"{dp}/*"):
            if i.lower().endswith(".csv") and _is_submission_like_csv(i):
                input_list.append(i)
    output = sample_sub.copy()

    if len(input_list) == 0:
        pred = _train_and_predict_knn(TRAIN_PATH, TEST_PATH)
        output["pressure"] = pred
        output["pressure"] = output["pressure"].apply(find_nearest)
        output.to_csv("submission.csv", index=False)
        return output

    for i in range(len(input_list)):
        input_list[i] = _read_submission_pressure(input_list[i], sample_sub["id"])

    output["pressure"] = np.median(np.vstack(input_list), axis=0)
    output["pressure"] = output["pressure"].apply(find_nearest)
    output.to_csv("submission.csv", index=False)
    return output




## === cell 1
candidate_dirs = [
    "../input/gb-data-blending-recover",  # original (may not exist)
    "../input/ventilator-pressure-prediction",  # exists in Kaggle dataset structure
    "/kaggle/input/ventilator-pressure-prediction",
    "/kaggle/data",
    "/kaggle/input",
]
dp = None
for d in candidate_dirs:
    if os.path.isdir(d):
        dp = d
        break

_ = g(dp)
print("Wrote submission.csv")
sub_df = pd.read_csv("submission.csv")
print(sub_df.head())
print("submission.csv shape:", sub_df.shape)
print("submission.csv columns:", list(sub_df.columns))
print(
    "pressure min/max:",
    float(sub_df["pressure"].min()),
    float(sub_df["pressure"].max()),
)
