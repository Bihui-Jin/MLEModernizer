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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
import os
import gc
import numpy as np
import pandas as pd

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")



## === cell 1
DATA_DIR = "/kaggle/input/ventilator-pressure-prediction"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

dtypes_train = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
    "pressure": "float32",
}
dtypes_test = {
    "id": "int32",
    "breath_id": "int32",
    "R": "int16",
    "C": "int16",
    "time_step": "float32",
    "u_in": "float32",
    "u_out": "int8",
}

train_df = pd.read_csv(TRAIN_PATH, dtype=dtypes_train)
test_df = pd.read_csv(TEST_PATH, dtype=dtypes_test)
submission = pd.read_csv(SAMPLE_SUB_PATH, dtype={"id": "int32", "pressure": "float32"})

print(train_df.shape, test_df.shape, submission.shape)
print(train_df.columns)




## === cell 2
def add_features(df):
    df = df.copy()

    df["one"] = 1
    gb = df.groupby("breath_id", sort=False)

    df["cross"] = df["u_in"] * df["u_out"]
    df["cross2"] = df["time_step"] * df["u_out"]

    df["area"] = (
        (df["time_step"] * df["u_in"]).groupby(df["breath_id"], sort=False).cumsum()
    )

    df["time_step_cumsum"] = gb["time_step"].cumsum()
    df["u_in_cumsum"] = gb["u_in"].cumsum()
    print("Step-1...Completed")

    u_in_g = gb["u_in"]
    u_out_g = gb["u_out"]

    for k in (1, 2, 3, 4):
        df[f"u_in_lag{k}"] = u_in_g.shift(k)
        df[f"u_out_lag{k}"] = u_out_g.shift(k)
        df[f"u_in_lag_back{k}"] = u_in_g.shift(-k)
        df[f"u_out_lag_back{k}"] = u_out_g.shift(-k)

    df = df.fillna(0)
    print("Step-2...Completed")

    u_in_max = u_in_g.transform("max")
    u_in_mean = u_in_g.transform("mean")
    df["breath_id__u_in__max"] = u_in_max
    df["breath_id__u_in__mean"] = u_in_mean
    df["breath_id__u_in__diffmax"] = u_in_max - df["u_in"]
    df["breath_id__u_in__diffmean"] = u_in_mean - df["u_in"]
    print("Step-3...Completed")

    for k in (1, 2, 3, 4):
        df[f"u_in_diff{k}"] = df["u_in"] - df[f"u_in_lag{k}"]
        df[f"u_out_diff{k}"] = df["u_out"] - df[f"u_out_lag{k}"]
    print("Step-4...Completed")

    df["count"] = gb["one"].cumsum()
    df["u_in_cummean"] = df["u_in_cumsum"] / df["count"]

    breath_id = df["breath_id"].to_numpy()
    bid_lag = np.empty_like(breath_id)
    bid_lag[0] = 0
    bid_lag[1:] = breath_id[:-1]
    bid_lag2 = np.empty_like(breath_id)
    bid_lag2[:2] = 0
    bid_lag2[2:] = breath_id[:-2]

    lagsame = (bid_lag == breath_id).astype(np.int8)
    lag2same = (bid_lag2 == breath_id).astype(np.int8)

    df["breath_id_lag"] = bid_lag
    df["breath_id_lag2"] = bid_lag2
    df["breath_id_lagsame"] = lagsame
    df["breath_id_lag2same"] = lag2same

    u_in = df["u_in"].to_numpy()
    u_in_shift1 = np.empty_like(u_in)
    u_in_shift1[0] = 0.0
    u_in_shift1[1:] = u_in[:-1]
    u_in_shift2 = np.empty_like(u_in)
    u_in_shift2[:2] = 0.0
    u_in_shift2[2:] = u_in[:-2]

    df["breath_id__u_in_lag"] = u_in_shift1 * lagsame
    df["breath_id__u_in_lag2"] = u_in_shift2 * lag2same
    print("Step-5...Completed")

    df["time_step_diff"] = gb["time_step"].diff().fillna(0)

    df["ewm_u_in_mean"] = u_in_g.ewm(halflife=9).mean().reset_index(level=0, drop=True)

    roll = u_in_g.rolling(window=15, min_periods=1)
    df["15_in_sum"] = roll.sum().reset_index(level=0, drop=True)
    df["15_in_min"] = roll.min().reset_index(level=0, drop=True)
    df["15_in_max"] = roll.max().reset_index(level=0, drop=True)
    df["15_in_mean"] = roll.mean().reset_index(level=0, drop=True)
    print("Step-6...Completed")

    df["u_in_lagback_diff1"] = df["u_in"] - df["u_in_lag_back1"]
    df["u_out_lagback_diff1"] = df["u_out"] - df["u_out_lag_back1"]
    df["u_in_lagback_diff2"] = df["u_in"] - df["u_in_lag_back2"]
    df["u_out_lagback_diff2"] = df["u_out"] - df["u_out_lag_back2"]
    print("Step-7...Completed")

    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)
    df["R__C"] = df["R"].astype(str) + "__" + df["C"].astype(str)
    df = pd.get_dummies(df)
    print("Step-8...Completed")

    return df


print("Train data...\n")
train_feat = add_features(train_df)

print("\nTest data...\n")
test_feat = add_features(test_df)



## === cell 3
pressure_all = train_feat["pressure"].to_numpy().astype("float32")

P_MIN = float(np.min(pressure_all))
P_MAX = float(np.max(pressure_all))

uniq = np.unique(pressure_all)
if uniq.shape[0] > 1:
    diffs = np.diff(uniq)
    diffs = diffs[diffs > 0]
    P_STEP = float(np.min(diffs)) if diffs.size else 0.0
else:
    P_STEP = 0.0

print("Min pressure:", P_MIN)
print("Max pressure:", P_MAX)
print("Pressure step:", P_STEP)
print("Unique values:", uniq.shape[0])

del pressure_all, uniq
gc.collect()



## === cell 4
y_all = train_feat["pressure"].astype("float32").to_numpy()
groups_all = train_df["breath_id"].to_numpy()
train_insp_mask = train_df["u_out"].to_numpy() == 0

drop_cols = [
    "pressure",
    "id",
    "breath_id",
    "one",
    "count",
    "breath_id_lag",
    "breath_id_lag2",
    "breath_id_lagsame",
    "breath_id_lag2same",
]

X_train_all = train_feat.drop(columns=drop_cols, errors="ignore")
X_test = test_feat.drop(
    columns=[c for c in drop_cols if c != "pressure"], errors="ignore"
)

X_train_all, X_test = X_train_all.align(X_test, join="left", axis=1, fill_value=0)

X_train = X_train_all.loc[train_insp_mask]
y = y_all[train_insp_mask]
groups = groups_all[train_insp_mask]

insp_fill_value = float(np.median(y))

print("X_train (insp):", X_train.shape, "X_test:", X_test.shape, "y:", y.shape)
print("Expiratory fill value (median insp pressure):", insp_fill_value)

del train_feat, test_feat, X_train_all, y_all, groups_all
gc.collect()



## === cell 5
scaler = RobustScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

X_train_s = np.ascontiguousarray(X_train_s)
X_test_s = np.ascontiguousarray(X_test_s)

del X_train, X_test
gc.collect()



## === cell 6
base_params = dict(
    loss="absolute_error",
    learning_rate=0.08,
    max_depth=6,
    l2_regularization=0.0,
    random_state=42,
)

candidates = [250, 400, 600]
gkf = GroupKFold(n_splits=3)

best_iter = candidates[0]
best_mae = float("inf")

splits = list(gkf.split(X_train_s, y, groups=groups))

for it in candidates:
    fold_maes = []
    for tr_idx, va_idx in splits:
        m = HistGradientBoostingRegressor(max_iter=it, **base_params)
        m.fit(X_train_s[tr_idx], y[tr_idx])
        pred_va = m.predict(X_train_s[va_idx]).astype("float32")
        fold_maes.append(mean_absolute_error(y[va_idx], pred_va))
    cv_mae = float(np.mean(fold_maes))
    print(f"CV MAE (insp only) for max_iter={it}: {cv_mae:.6f}")
    if cv_mae < best_mae:
        best_mae = cv_mae
        best_iter = it

print("Chosen max_iter:", best_iter, "with CV MAE:", best_mae)

model = HistGradientBoostingRegressor(max_iter=best_iter, **base_params)
model.fit(X_train_s, y)

pred_test = model.predict(X_test_s).astype("float32")
print(pred_test.shape, pred_test[:5])

del X_train_s, y, groups, splits
gc.collect()



## === cell 7
submission = submission.copy()
pred_final = pred_test.copy()

test_u_out = test_df["u_out"].to_numpy()
pred_final[test_u_out == 1] = insp_fill_value

submission["pressure"] = pred_final

if P_STEP and P_STEP > 0:
    submission["pressure"] = (
        np.round((submission["pressure"] - P_MIN) / P_STEP) * P_STEP + P_MIN
    )
submission["pressure"] = np.clip(submission["pressure"], P_MIN, P_MAX)

submission = submission[["id", "pressure"]]
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Saved submission.csv with shape:", submission.shape)

del X_test_s, pred_test, pred_final
gc.collect()
