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

0.1358740603849016

# 6. Current score

0.85843

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 17.65244) has done: 'I fix the blending code so it only uses valid *submission-like* CSVs with the same row count as `sample_submission.csv`, which removes the broadcast error caused by accidentally reading `train.csv` (5.4M rows) as if it were a submission. I also make the CSV detection stricter (must have exactly `id` and `pressure` columns) and ensure we always align predictions to the `id` order of `sample_submission.csv` when possible. If no compatible submissions are found in the scanned directories, the script fall back to a safe baseline (median pressure snapped to the nearest valid pressure) and still write `submission.csv`. Core logic (random-weight blending + snapping to known pressures) is preserved; changes are purely for correctness and stable end-to-end execution.'
- What this solution (achieved 17.65244) has done: 'Your current score is extremely far from the target (17.65 vs 0.136 MAE), which strongly suggests the submission is essentially a constant/garbage baseline due to “no compatible submissions found”, so the minimal score-improving change is to stop blending random external CSVs and instead generate predictions from the provided train/test data. I keep the “snap to nearest valid pressure” post-processing (core evaluation semantics) but replace the fallback with a lightweight, deterministic kNN regressor trained on simple per-timestep features using scikit-learn (available), which should move MAE dramatically toward the target without changing the submission format. I also ensure predictions are set to 0 for expiratory phase (`u_out==1`) since those rows are not scored and this typically improves MAE stability. The script still always write a valid `submission.csv` with the required `id,pressure` columns.'
- What this solution (achieved 1.79498) has done: 'Your current MAE (17.65) is far from the target (0.136), which indicates the pipeline is producing near-constant/incorrect pressures; the smallest legitimate improvement is to keep your existing “snap to nearest valid pressure” semantics but make the fallback predictor materially better using the available train/test. I keep your kNN approach and feature idea, but make it time-series aware by training/predicting per `breath_id` and adding a couple of classic minimal cumulative features (`u_in_cum` and `u_in_lag1`) that don’t change the overall modeling family or loop structure. I also fix a key evaluation mismatch: don’t force `u_out==1` predictions to 0 (those rows aren’t scored, but setting them to 0 can hurt because Kaggle still compares only `u_out==0`; keeping consistent predictions is safer). Finally, I ensure the kNN fallback is always used (and fast enough) by not scanning directories for random CSVs that are unrelated to this dataset.'
- What this solution (achieved 1.18403) has done: 'Your current MAE (1.79498) is still much worse than the target (0.13587), so we should improve the fallback predictor while keeping the same overall kNN + “snap to nearest valid pressure” core logic. The smallest high-impact change is to align training with the evaluation: train only on inspiratory rows (`u_out==0`), because expiratory rows are not scored and can add noise to kNN neighbors. To preserve row order and semantics, we still predict all test rows, but for expiratory test rows we fill with a simple, stable baseline derived from inspiratory training pressure (median) before snapping. Finally, we add standardization (via `StandardScaler`) in a pipeline so kNN distances aren’t dominated by high-scale features like `u_in_cum`, which typically reduces MAE materially without changing the modeling family.'
- What this solution (achieved 1.07895) has done: 'Your current MAE (1.184) is still far above the target (0.1359), so we should improve predictive accuracy while keeping the same kNN + “snap to nearest valid pressure” core logic. The minimal high-impact fix is to remove the forced baseline for `u_out==1` rows (those rows are not scored, but overriding them can distort nearest-neighbor behavior at the end of breaths) and instead let the model predict all rows consistently. Next, we add a couple of very lightweight, classic lag/cumulative features (`u_out_lag1`, `time_step_lag1`, `u_in_diff1`, `u_in_cummean`) that preserve the same feature-extraction approach but make kNN neighbors much more time-series aware. Finally, we switch kNN to Manhattan distance (`p=1`), which often reduces MAE for high-dimensional engineered features without changing the model family or training loop.'
- What this solution (achieved 0.85843) has done: 'The timeout is dominated by the per-row `for` loop that does a DataFrame slice/copy, feature construction, and a separate `model.predict()` call 603,600 times. I keep the same KNN+StandardScaler model and the same autoregressive “pressure_lag1 uses previous prediction” logic, but remove pandas overhead by precomputing all non-lag features as NumPy arrays and predicting per-breath in one batch while updating `pressure_lag1` sequentially within each breath. I also make `find_nearest` fully vectorized (same nearest-pressure snapping) to avoid `.apply` over 603,600 rows. These changes preserve core logic and semantics while cutting constant factors drastically so it can finish within 600 seconds.'

# 9. Code solution

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


def find_nearest_vec(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float32)
    idx = np.searchsorted(sorted_pressures, pred, side="left")
    idx = idx.astype(np.int64, copy=False)

    out = np.empty_like(pred, dtype=np.float32)
    left_mask = idx <= 0
    right_mask = idx >= total_pressures_len
    mid_mask = (~left_mask) & (~right_mask)

    if left_mask.any():
        out[left_mask] = np.float32(sorted_pressures[0])
    if right_mask.any():
        out[right_mask] = np.float32(sorted_pressures[-1])

    if mid_mask.any():
        upper = sorted_pressures[idx[mid_mask]]
        lower = sorted_pressures[idx[mid_mask] - 1]
        choose_lower = np.abs(lower - pred[mid_mask]) < np.abs(upper - pred[mid_mask])
        out[mid_mask] = np.where(choose_lower, lower, upper).astype(
            np.float32, copy=False
        )
    return out


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

    n = len(test)
    pred = np.zeros(n, dtype=np.float32)

    time_step = test["time_step"].to_numpy(dtype=np.float32, copy=False)
    u_in = test["u_in"].to_numpy(dtype=np.float32, copy=False)
    u_out = test["u_out"].to_numpy(dtype=np.int8, copy=False)
    R = test["R"].to_numpy(dtype=np.int16, copy=False)
    C = test["C"].to_numpy(dtype=np.int16, copy=False)
    u_in_time = (u_in * time_step).astype(np.float32, copy=False)
    u_in_cum = test["u_in_cum"].to_numpy(dtype=np.float32, copy=False)
    u_in_lag1 = test["u_in_lag1"].to_numpy(dtype=np.float32, copy=False)
    u_out_lag1 = test["u_out_lag1"].to_numpy(dtype=np.int8, copy=False)
    time_step_lag1 = test["time_step_lag1"].to_numpy(dtype=np.float32, copy=False)
    u_in_diff1 = test["u_in_diff1"].to_numpy(dtype=np.float32, copy=False)
    u_in_cummean = test["u_in_cummean"].to_numpy(dtype=np.float32, copy=False)
    breath_id = test["breath_id"].to_numpy(dtype=np.int64, copy=False)

    change = np.nonzero(breath_id[1:] != breath_id[:-1])[0] + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [n]))

    for s, e in zip(starts, ends):
        prev_p = np.float32(0.0)
        for i in range(s, e):
            Xi = np.array(
                [
                    [
                        time_step[i],
                        u_in[i],
                        np.float32(u_out[i]),
                        np.float32(R[i]),
                        np.float32(C[i]),
                        u_in_time[i],
                        u_in_cum[i],
                        u_in_lag1[i],
                        np.float32(u_out_lag1[i]),
                        time_step_lag1[i],
                        u_in_diff1[i],
                        u_in_cummean[i],
                        prev_p,
                    ]
                ],
                dtype=np.float32,
            )
            pi = float(model.predict(Xi)[0])
            pred[i] = np.float32(pi)
            prev_p = np.float32(pi)

    out = test[["id"]].copy()
    out["pressure"] = pred
    out = out.set_index("id").reindex(sample_sub["id"]).reset_index()
    return out["pressure"].to_numpy(dtype=np.float32)


def g(dp):
    output = sample_sub.copy()
    pred = _train_and_predict_knn(TRAIN_PATH, TEST_PATH)
    output["pressure"] = pred
    output["pressure"] = find_nearest_vec(
        output["pressure"].to_numpy(dtype=np.float32, copy=False)
    )
    output.to_csv("submission.csv", index=False)
    return output


def blend(a, b):
    a = pd.read_csv(a)
    b = pd.read_csv(b)
    if not a["id"].equals(b["id"]):
        b = b.set_index("id").reindex(a["id"]).reset_index()
    a["pressure"] = a["pressure"] * 0.7 + b["pressure"] * 0.3
    a["pressure"] = find_nearest_vec(
        a["pressure"].to_numpy(dtype=np.float32, copy=False)
    )
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
        output["pressure"] = find_nearest_vec(
            output["pressure"].to_numpy(dtype=np.float32, copy=False)
        )
        output.to_csv("submission.csv", index=False)
        return output

    for i in range(len(input_list)):
        input_list[i] = _read_submission_pressure(input_list[i], sample_sub["id"])

    output["pressure"] = np.median(np.vstack(input_list), axis=0)
    output["pressure"] = find_nearest_vec(
        output["pressure"].to_numpy(dtype=np.float32, copy=False)
    )
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
