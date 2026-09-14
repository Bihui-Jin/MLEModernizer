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
import numpy as np, pandas as pd, os, random, gc, glob
from random import random as rd


def set_seed(seed=2021):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return np.random.RandomState(seed)




## === cell 1
df_train = pd.read_csv(
    "../input/ventilator-pressure-prediction/train.csv",
    dtype={
        "C": "int8",
        "R": "int8",
        "breath_id": "int32",
        "id": "int16",
        "time_step": "float32",
        "u_in": "float32",
        "u_out": "int8",
        "pressure": "float32",
    },
)
unique_pressures = np.sort(df_train["pressure"].unique())
total_pressures_len = len(unique_pressures)


def find_nearest(prediction):
    """Vectorized nearest‑lookup for one value or an array of values."""
    arr = np.atleast_1d(prediction)
    idx = np.searchsorted(unique_pressures, arr, side="left")
    idx = np.clip(idx, 0, total_pressures_len - 1)

    low_idx = np.maximum(idx - 1, 0)
    high_idx = np.minimum(idx, total_pressures_len - 1)

    low = unique_pressures[low_idx]
    high = unique_pressures[high_idx]

    choose_low = np.abs(low - arr) < np.abs(high - arr)
    result = np.where(choose_low, low, high)

    if np.isscalar(prediction):
        return result.item()
    return result


def wc(input_list):
    """Weighted combine two prediction files (used in original blending script)."""
    scores = []
    for i, path in enumerate(input_list):
        try:
            score = int(path.split("/")[-1].split(".")[1].split(" ")[0])
        except Exception:
            score = 0
        scores.append(score)
        input_list[i] = pd.read_csv(path).pressure.values.ravel()
    if len(input_list) == 1:
        return input_list[0]
    w1 = (scores[1] / sum(scores)) + 0.1
    w2 = 1 - w1
    return input_list[0] * w1 + input_list[1] * w2


def g(dp):
    """Original blending routine – kept unchanged."""
    files = sorted(glob.glob(f"{dp}/*"))
    if not files:
        raise FileNotFoundError(f"No files found in {dp}")
    splits = len(files) // 2 or 1
    groups = [files[i::splits] for i in range(splits)]
    preds = [wc(group) for group in groups]
    loop_time = 154
    pred_list = []
    for it in range(loop_time):
        set_seed(it)
        weights = np.random.dirichlet(np.ones(len(preds)), size=1).ravel()
        combined = sum(p * w for p, w in zip(preds, weights))
        pred_list.append(combined)
    out = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")
    out.pressure = np.median(np.vstack(pred_list), axis=0)
    out["pressure"] = out["pressure"].apply(find_nearest)
    out.to_csv(f"rwb_{loop_time}_loops.csv", index=False)


def blend(a_path, b_path):
    a = pd.read_csv(a_path)
    b = pd.read_csv(b_path)
    a.pressure = a.pressure * 0.7 + b.pressure * 0.3
    a["pressure"] = a["pressure"].apply(find_nearest)
    a.to_csv("blend.csv", index=False)
    return a


def avg(dp):
    """Average predictions from all CSVs in *dp*."""
    file_paths = list(glob.glob(f"{dp}/*"))
    if not file_paths:
        raise FileNotFoundError(f"No prediction files in {dp}")
    preds = [pd.read_csv(p).pressure.values.ravel() for p in file_paths]
    out = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")
    out.pressure = np.median(np.vstack(preds), axis=0)
    out.to_csv("avg.csv", index=False)
    return out


def add_features(df):
    """Create simple interaction / polynomial features to enrich the linear model."""
    df = df.copy()
    df["C_R"] = df["C"] * df["R"]
    df["u_in_sq"] = df["u_in"] ** 2
    df["time_step_sq"] = df["time_step"] ** 2
    df["u_in_times_u_out"] = df["u_in"] * df["u_out"]
    df["C_u_in"] = df["C"] * df["u_in"]
    df["R_u_in"] = df["R"] * df["u_in"]
    df["time_step_u_in"] = df["time_step"] * df["u_in"]
    df["C_sq"] = df["C"] ** 2
    df["R_sq"] = df["R"] ** 2
    df["C_time_step"] = df["C"] * df["time_step"]
    df["R_time_step"] = df["R"] * df["time_step"]
    df["C_u_out"] = df["C"] * df["u_out"]
    df["R_u_out"] = df["R"] * df["u_out"]
    df["time_step_u_out"] = df["time_step"] * df["u_out"]
    return df




## === cell 2
def train_and_predict():
    from sklearn.ensemble import HistGradientBoostingRegressor

    base_feat_cols = ["C", "R", "u_in", "u_out", "time_step"]
    df_train_rounded = df_train.copy()
    df_train_rounded["u_in_r"] = df_train_rounded["u_in"].round(2)
    df_train_rounded["time_step_r"] = df_train_rounded["time_step"].round(3)

    lookup_df = (
        df_train_rounded.groupby(
            ["C", "R", "u_in_r", "u_out", "time_step_r"], as_index=False
        )["pressure"]
        .median()
        .reset_index()
        .rename(columns={"pressure": "med_pressure"})
    )

    df_test = pd.read_csv(
        "../input/ventilator-pressure-prediction/test.csv",
        dtype={
            "C": "int8",
            "R": "int8",
            "breath_id": "int32",
            "id": "int16",
            "time_step": "float32",
            "u_in": "float32",
            "u_out": "int8",
        },
    )
    test_base = df_test[base_feat_cols]
    test_feats = add_features(test_base)

    df_test_rounded = df_test.copy()
    df_test_rounded["u_in_r"] = df_test_rounded["u_in"].round(2)
    df_test_rounded["time_step_r"] = df_test_rounded["time_step"].round(3)

    key_cols = ["C", "R", "u_in_r", "u_out", "time_step_r"]
    merged = df_test_rounded.merge(lookup_df, on=key_cols, how="left")
    lookup_preds = merged["med_pressure"].values
    mask_missing = np.isnan(lookup_preds)

    if not mask_missing.any():
        final_preds = np.clip(
            lookup_preds, df_train["pressure"].min(), df_train["pressure"].max()
        )
        final_preds = find_nearest(final_preds)
        submission = pd.read_csv(
            "../input/ventilator-pressure-prediction/sample_submission.csv"
        )
        submission["pressure"] = final_preds
        submission.to_csv("submission.csv", index=False)
        print("Generated submission.csv using only the median‑lookup (no tree needed).")
        return submission

    X = add_features(df_train[base_feat_cols]).astype(np.float32)
    y = df_train["pressure"]
    model = HistGradientBoostingRegressor(
        max_depth=12,
        learning_rate=0.03,
        max_iter=3000,
        random_state=2021,
        loss="absolute_error",
        early_stopping=False,
    )
    model.fit(X, y)

    tree_preds = model.predict(test_feats.astype(np.float32))

    final_preds = np.where(mask_missing, tree_preds, lookup_preds)

    min_p, max_p = df_train["pressure"].min(), df_train["pressure"].max()
    final_preds = np.clip(final_preds, min_p, max_p)
    final_preds = find_nearest(final_preds)

    submission = pd.read_csv(
        "../input/ventilator-pressure-prediction/sample_submission.csv"
    )
    submission["pressure"] = final_preds
    submission.to_csv("submission.csv", index=False)
    print(
        "Generated submission.csv using HistGradientBoostingRegressor for missing lookup rows."
    )
    return submission




## === cell 3
try:
    avg("../input/gb-data-blending-recover")
except Exception as e:
    print(f"Avg blending failed ({e}); using tree‑based fallback.")
    train_and_predict()
