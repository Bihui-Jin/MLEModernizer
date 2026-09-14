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
import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42

BASE_PATH = "/kaggle/input/ventilator-pressure-prediction"
TRAIN_PATH = os.path.join(BASE_PATH, "train.csv")
TEST_PATH = os.path.join(BASE_PATH, "test.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sub = pd.read_csv(SAMPLE_SUB_PATH)

required_train_cols = {
    "id",
    "breath_id",
    "R",
    "C",
    "time_step",
    "u_in",
    "u_out",
    "pressure",
}
required_test_cols = {"id", "breath_id", "R", "C", "time_step", "u_in", "u_out"}
assert required_train_cols.issubset(
    train.columns
), f"train missing columns: {required_train_cols - set(train.columns)}"
assert required_test_cols.issubset(
    test.columns
), f"test missing columns: {required_test_cols - set(test.columns)}"
assert {"id", "pressure"}.issubset(
    sub.columns
), "sample_submission must contain id,pressure"

train.shape, test.shape, sub.shape



## === cell 1
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge


def add_breath_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the same core engineered-feature approach, but add a few minimal,
    non-leaky within-breath state features.
    """
    df = df.sort_values(["breath_id", "time_step"]).copy()

    df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1)
    df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2)
    df["u_out_lag1"] = df.groupby("breath_id")["u_out"].shift(1)

    df["time_step_lag1"] = df.groupby("breath_id")["time_step"].shift(1)
    df["dt"] = df["time_step"] - df["time_step_lag1"]

    dt_med = df.groupby("breath_id")["dt"].transform("median")
    df["dt"] = df["dt"].fillna(dt_med).fillna(0.0)

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_in_cumsum"] = df.groupby("breath_id")["u_in"].cumsum()

    df["u_in_dt"] = df["u_in"] * df["dt"]
    df["u_in_dt_cumsum"] = df.groupby("breath_id")["u_in_dt"].cumsum()

    eps = 1e-6
    df["u_in_over_R"] = df["u_in"] / (df["R"].astype(np.float32) + eps)
    df["u_in_times_C"] = df["u_in"] * df["C"].astype(np.float32)

    df["t_idx"] = df.groupby("breath_id").cumcount().astype(np.int16)

    df["u_in_roll3"] = (
        df.groupby("breath_id")["u_in"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["u_in_lag1_dt"] = df["u_in_lag1"] * df["dt"]

    return df


train_fe = add_breath_features(train)
test_fe = add_breath_features(test)

train_insp = train_fe.loc[train_fe["u_out"] == 0].copy()
train_insp["pressure"] = train_insp["pressure"].astype(np.float32)

feature_cols = [
    "R",
    "C",
    "time_step",
    "t_idx",
    "u_in",
    "u_out",
    "u_in_lag1",
    "u_in_lag2",
    "u_out_lag1",
    "u_in_diff1",
    "dt",
    "u_in_cumsum",
    "u_in_dt",
    "u_in_dt_cumsum",
    "u_in_over_R",
    "u_in_times_C",
    "u_in_roll3",
    "u_in_lag1_dt",
]

pressure_grid = np.sort(train["pressure"].unique()).astype(np.float32)
min_pressure = float(pressure_grid.min())

categorical_cols = ["R", "C"]
numeric_cols = [c for c in feature_cols if c not in categorical_cols]

preprocess = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_cols,
        ),
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False, with_std=True)),
                ]
            ),
            numeric_cols,
        ),
    ],
    remainder="drop",
)

base_model = Ridge(alpha=1.0, random_state=RANDOM_STATE, fit_intercept=False)

all_groups = sorted(
    set(map(tuple, train_insp[["R", "C"]].drop_duplicates().values.tolist()))
    | set(map(tuple, test_fe[["R", "C"]].drop_duplicates().values.tolist()))
)




## === cell 2
def fit_group_models_with_oof_ar(train_group: pd.DataFrame):
    """
    Returns:
      pipe_base: fitted Pipeline on all group data
      pipe_resid: fitted Pipeline on all group data for residuals
      oof_base_pred: OOF base predictions aligned to train_group index
    """
    train_group = train_group.sort_values(["breath_id", "time_step"]).copy()
    Xg = train_group[feature_cols].copy()
    yg = train_group["pressure"].values.astype(np.float32)

    pipe_base = Pipeline([("preprocess", preprocess), ("model", base_model)])
    pipe_resid = Pipeline([("preprocess", preprocess), ("model", base_model)])

    breath_ids = train_group["breath_id"].unique()
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(breath_ids)
    folds = np.array_split(breath_ids, 5)

    oof = np.zeros(len(train_group), dtype=np.float32)

    for f_bids in folds:
        is_val = train_group["breath_id"].isin(f_bids).values
        if is_val.sum() == 0 or (~is_val).sum() == 0:
            continue
        X_tr = Xg.loc[~is_val]
        y_tr = yg[~is_val]
        X_va = Xg.loc[is_val]

        pipe_f = Pipeline([("preprocess", preprocess), ("model", base_model)])
        pipe_f.fit(X_tr, y_tr)
        oof[is_val] = pipe_f.predict(X_va).astype(np.float32)

    pipe_base.fit(Xg, yg)

    train_group["base_oof"] = oof
    train_group["base_oof_lag1"] = train_group.groupby("breath_id")["base_oof"].shift(1)
    train_group["base_oof_lag2"] = train_group.groupby("breath_id")["base_oof"].shift(2)
    train_group["base_oof_lag1"] = (
        train_group["base_oof_lag1"].fillna(train_group["base_oof"]).fillna(0.0)
    )
    train_group["base_oof_lag2"] = (
        train_group["base_oof_lag2"].fillna(train_group["base_oof_lag1"]).fillna(0.0)
    )

    ar_cols = ["base_oof_lag1", "base_oof_lag2"]
    Xg2 = pd.concat(
        [Xg.reset_index(drop=True), train_group[ar_cols].reset_index(drop=True)], axis=1
    )

    resid = (yg - oof).astype(np.float32)
    pipe_resid.fit(Xg2, resid)

    return pipe_base, pipe_resid


def predict_group_iterative_ar(
    test_group: pd.DataFrame, pipe_base, pipe_resid
) -> np.ndarray:
    """
    Iterative test-time AR:
      base_pred_t = pipe_base(features_t)
      resid_pred_t = pipe_resid(features_t + [lagged base_pred/resid-adjusted pred])
    We use lagged *final* prediction to stabilize dynamics.
    """
    test_group = test_group.sort_values(["breath_id", "time_step"]).copy()
    Xg_test = test_group[feature_cols].copy().reset_index(drop=True)

    base_pred = pipe_base.predict(Xg_test).astype(np.float32)

    final_pred = np.zeros(len(test_group), dtype=np.float32)

    breath_ids = test_group["breath_id"].values
    change = np.r_[True, breath_ids[1:] != breath_ids[:-1]]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(test_group)]

    for s, e in zip(starts, ends):
        for i in range(s, e):
            lag1 = final_pred[i - 1] if i - 1 >= s else base_pred[i]
            lag2 = final_pred[i - 2] if i - 2 >= s else lag1
            X2 = pd.concat(
                [
                    Xg_test.iloc[[i]].reset_index(drop=True),
                    pd.DataFrame({"base_oof_lag1": [lag1], "base_oof_lag2": [lag2]}),
                ],
                axis=1,
            )
            resid_pred = pipe_resid.predict(X2).astype(np.float32)[0]
            final_pred[i] = base_pred[i] + resid_pred

    return final_pred


X_test_all = test_fe[feature_cols].copy()
pred = np.full(len(test_fe), min_pressure, dtype=np.float32)

for R_val, C_val in all_groups:
    train_mask = (train_insp["R"].values == R_val) & (train_insp["C"].values == C_val)
    test_mask = (test_fe["R"].values == R_val) & (test_fe["C"].values == C_val)

    if int(train_mask.sum()) == 0 or int(test_mask.sum()) == 0:
        continue

    train_grp = train_insp.loc[train_insp.index[train_mask]].copy()
    test_grp = test_fe.loc[test_fe.index[test_mask]].copy()

    pipe_base_g, pipe_resid_g = fit_group_models_with_oof_ar(train_grp)
    pred_grp = predict_group_iterative_ar(test_grp, pipe_base_g, pipe_resid_g)

    pred[np.flatnonzero(test_mask)] = pred_grp

pred[test_fe["u_out"].values == 1] = min_pressure

mask_insp_test = test_fe["u_out"].values == 0
pred_insp = pred[mask_insp_test]
idx = np.abs(pred_insp[:, None] - pressure_grid[None, :]).argmin(axis=1)
pred[mask_insp_test] = pressure_grid[idx]

pred_df = pd.DataFrame({"id": test_fe["id"].values, "pressure": pred})
sub_out = sub[["id"]].merge(pred_df, on="id", how="left")
sub_out["pressure"] = sub_out["pressure"].fillna(min_pressure).astype(np.float32)

sub_out.to_csv("submission.csv", index=False)
sub_out.head()
