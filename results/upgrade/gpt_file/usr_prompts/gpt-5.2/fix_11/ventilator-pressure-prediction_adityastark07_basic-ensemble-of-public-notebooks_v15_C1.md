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
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

DATA_DIR = "../input/ventilator-pressure-prediction"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
SAMPLE_SUB_PATH = f"{DATA_DIR}/sample_submission.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sub = pd.read_csv(SAMPLE_SUB_PATH)

assert "id" in test.columns and "id" in sub.columns
assert len(test) == len(sub)




## === cell 1
def add_breath_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["breath_id", "time_step"], kind="mergesort")

    g = df.groupby("breath_id", sort=False)

    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0)
    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0.0)

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_out_diff1"] = df["u_out"] - df["u_out_lag1"]

    df["u_in_cumsum"] = g["u_in"].cumsum()

    df["dt"] = g["time_step"].diff().fillna(0.0).astype(np.float32)
    df["t_cumsum"] = g["time_step"].cumsum().astype(np.float32)

    return df


train_fe = add_breath_features(train)
test_fe = add_breath_features(test)



## === cell 2
base_features = ["R", "C", "time_step", "u_in", "u_out"]
extra_features = [
    "u_in_lag1",
    "u_in_lag2",
    "u_out_lag1",
    "u_in_diff1",
    "u_out_diff1",
    "u_in_cumsum",
    "dt",
    "t_cumsum",
]
features = base_features + extra_features
target = "pressure"

train_fe = train_fe.sort_values(
    ["breath_id", "time_step"], kind="mergesort"
).reset_index(drop=True)
test_fe = test_fe.sort_values(["breath_id", "time_step"], kind="mergesort").reset_index(
    drop=True
)

train_fe["t_idx"] = (
    train_fe.groupby("breath_id", sort=False).cumcount().astype(np.int16)
)
test_fe["t_idx"] = test_fe.groupby("breath_id", sort=False).cumcount().astype(np.int16)

u_in_bin_width = 1.0
train_fe["u_in_bin"] = np.floor(train_fe["u_in"].to_numpy() / u_in_bin_width).astype(
    np.int16
)
test_fe["u_in_bin"] = np.floor(test_fe["u_in"].to_numpy() / u_in_bin_width).astype(
    np.int16
)

rs = 42

max_breaths = 12000
unique_breaths = train_fe["breath_id"].unique()
if len(unique_breaths) > max_breaths:
    rng = np.random.RandomState(rs)
    keep_breaths = rng.choice(unique_breaths, size=max_breaths, replace=False)
    mask_keep = train_fe["breath_id"].isin(keep_breaths).to_numpy()
    train_fit_df = train_fe.loc[mask_keep].copy()
else:
    train_fit_df = train_fe.copy()

breaths_fit = pd.unique(train_fit_df["breath_id"].to_numpy())
b_tr, b_va = train_test_split(breaths_fit, test_size=0.1, random_state=rs)

tr_mask = train_fit_df["breath_id"].isin(b_tr).to_numpy()
va_mask = train_fit_df["breath_id"].isin(b_va).to_numpy()

train_tr = train_fit_df.loc[tr_mask].copy()
train_va = train_fit_df.loc[va_mask].copy()

train_tr_fit = train_tr.loc[train_tr["u_out"].to_numpy() == 0].copy()

global_mean_pressure_tr = float(train_tr_fit[target].mean())

prior_tbl_tr = (
    train_tr_fit.groupby(["R", "C", "t_idx"], sort=False)[target]
    .mean()
    .rename("pressure_rc_tidx_mean")
    .reset_index()
)
train_tr_fit = train_tr_fit.merge(prior_tbl_tr, on=["R", "C", "t_idx"], how="left")
train_va = train_va.merge(prior_tbl_tr, on=["R", "C", "t_idx"], how="left")

prior_tbl_full = (
    train_fe.loc[train_fe["u_out"].to_numpy() == 0]
    .groupby(["R", "C", "t_idx"], sort=False)[target]
    .mean()
    .rename("pressure_rc_tidx_mean")
    .reset_index()
)
test_fe = test_fe.merge(prior_tbl_full, on=["R", "C", "t_idx"], how="left")

train_tr_fit["pressure_rc_tidx_mean"] = (
    train_tr_fit["pressure_rc_tidx_mean"]
    .fillna(global_mean_pressure_tr)
    .astype(np.float32)
)
train_va["pressure_rc_tidx_mean"] = (
    train_va["pressure_rc_tidx_mean"].fillna(global_mean_pressure_tr).astype(np.float32)
)
insp_mean_full = float(train_fe.loc[train_fe["u_out"].to_numpy() == 0, target].mean())
test_fe["pressure_rc_tidx_mean"] = (
    test_fe["pressure_rc_tidx_mean"].fillna(insp_mean_full).astype(np.float32)
)

prior_tbl2_tr = (
    train_tr_fit.groupby(["R", "C", "t_idx", "u_in_bin"], sort=False)[target]
    .mean()
    .rename("pressure_rc_tidx_uinbin_mean")
    .reset_index()
)
train_tr_fit = train_tr_fit.merge(
    prior_tbl2_tr, on=["R", "C", "t_idx", "u_in_bin"], how="left"
)
train_va = train_va.merge(prior_tbl2_tr, on=["R", "C", "t_idx", "u_in_bin"], how="left")

prior_tbl2_full = (
    train_fe.loc[train_fe["u_out"].to_numpy() == 0]
    .groupby(["R", "C", "t_idx", "u_in_bin"], sort=False)[target]
    .mean()
    .rename("pressure_rc_tidx_uinbin_mean")
    .reset_index()
)
test_fe = test_fe.merge(prior_tbl2_full, on=["R", "C", "t_idx", "u_in_bin"], how="left")

train_tr_fit["pressure_rc_tidx_uinbin_mean"] = (
    train_tr_fit["pressure_rc_tidx_uinbin_mean"]
    .fillna(train_tr_fit["pressure_rc_tidx_mean"])
    .fillna(global_mean_pressure_tr)
    .astype(np.float32)
)
train_va["pressure_rc_tidx_uinbin_mean"] = (
    train_va["pressure_rc_tidx_uinbin_mean"]
    .fillna(train_va["pressure_rc_tidx_mean"])
    .fillna(global_mean_pressure_tr)
    .astype(np.float32)
)
test_fe["pressure_rc_tidx_uinbin_mean"] = (
    test_fe["pressure_rc_tidx_uinbin_mean"]
    .fillna(test_fe["pressure_rc_tidx_mean"])
    .fillna(insp_mean_full)
    .astype(np.float32)
)

features_plus = features + [
    "t_idx",
    "pressure_rc_tidx_mean",
    "pressure_rc_tidx_uinbin_mean",
    "u_in_bin",
]

X_tr_raw = train_tr_fit[features_plus].copy()
X_va_raw = train_va[features_plus].copy()
X_test_raw = test_fe[features_plus].copy()

X_tr = pd.get_dummies(X_tr_raw, columns=["R", "C"], prefix=["R", "C"], dtype=np.int8)
X_va = pd.get_dummies(X_va_raw, columns=["R", "C"], prefix=["R", "C"], dtype=np.int8)
X_test = pd.get_dummies(
    X_test_raw, columns=["R", "C"], prefix=["R", "C"], dtype=np.int8
)

X_tr, X_va = X_tr.align(X_va, join="left", axis=1, fill_value=0)
X_tr, X_test = X_tr.align(X_test, join="left", axis=1, fill_value=0)
X_va = X_va.reindex(columns=X_tr.columns, fill_value=0)

y_tr = train_tr_fit[target].astype(np.float32)
y_va = train_va[target].astype(np.float32)



## === cell 3
model = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "knn",
            KNeighborsRegressor(n_neighbors=45, weights="distance", p=1, n_jobs=-1),
        ),
    ]
)

model.fit(X_tr, y_tr)

va_pred = model.predict(X_va).astype(np.float32)
va_u_out = train_va["u_out"].to_numpy()
insp_mask_va = va_u_out == 0

va_mae = np.mean(
    np.abs(va_pred[insp_mask_va] - y_va.to_numpy(dtype=np.float32)[insp_mask_va])
)
print(
    "Validation MAE (breath_id split; trained on u_out==0 only; evaluated on u_out==0; R/C one-hot; +priors):",
    float(va_mae),
)



## === cell 4
test_pred = model.predict(X_test).astype(np.float32)

test_order = test_fe[["id", "breath_id", "time_step", "u_out"]].copy()
test_order["pred"] = test_pred
test_order = test_order.sort_values(["breath_id", "time_step"], kind="mergesort")

test_order["last_insp_pred"] = (
    (test_order["pred"].where(test_order["u_out"].to_numpy() == 0))
    .groupby(test_order["breath_id"], sort=False)
    .ffill()
)
test_order["last_insp_pred"] = (
    test_order["last_insp_pred"].fillna(0.0).astype(np.float32)
)
test_order["pred_filled"] = np.where(
    test_order["u_out"].to_numpy() == 1,
    test_order["last_insp_pred"].to_numpy(dtype=np.float32),
    test_order["pred"].to_numpy(dtype=np.float32),
).astype(np.float32)

pressure_levels = np.sort(train["pressure"].unique().astype(np.float32))

pred_all = test_order["pred_filled"].to_numpy(dtype=np.float32)
is_insp = test_order["u_out"].to_numpy() == 0

pred_snapped_all = pred_all.copy()
pred_insp = pred_all[is_insp]

idx = np.searchsorted(pressure_levels, pred_insp, side="left")
idx = np.clip(idx, 0, len(pressure_levels) - 1)
idx_left = np.clip(idx - 1, 0, len(pressure_levels) - 1)

right = pressure_levels[idx]
left = pressure_levels[idx_left]
choose_left = np.abs(pred_insp - left) <= np.abs(pred_insp - right)
pred_snapped_insp = np.where(choose_left, left, right).astype(np.float32)

pred_snapped_all[is_insp] = pred_snapped_insp

sub = sub.sort_values("id").reset_index(drop=True)
test_ids = test[["id"]].sort_values("id").reset_index(drop=True)
assert sub["id"].to_numpy().shape == test_ids["id"].to_numpy().shape
assert np.array_equal(sub["id"].to_numpy(), test_ids["id"].to_numpy())

pred_by_id = (
    test_order.loc[:, ["id"]]
    .assign(pressure=pred_snapped_all)
    .sort_values("id")
    .reset_index(drop=True)["pressure"]
    .to_numpy(dtype=np.float32)
)

sub["pressure"] = pred_by_id
sub.to_csv("submission.csv", index=False)

sub.head()
