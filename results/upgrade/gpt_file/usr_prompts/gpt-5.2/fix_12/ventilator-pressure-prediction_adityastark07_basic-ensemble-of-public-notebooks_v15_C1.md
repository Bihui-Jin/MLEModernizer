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

train = pd.read_csv(
    TRAIN_PATH,
    dtype={
        "id": np.int32,
        "breath_id": np.int32,
        "R": np.int16,
        "C": np.int16,
        "time_step": np.float32,
        "u_in": np.float32,
        "u_out": np.int8,
        "pressure": np.float32,
    },
)
test = pd.read_csv(
    TEST_PATH,
    dtype={
        "id": np.int32,
        "breath_id": np.int32,
        "R": np.int16,
        "C": np.int16,
        "time_step": np.float32,
        "u_in": np.float32,
        "u_out": np.int8,
    },
)
sub = pd.read_csv(SAMPLE_SUB_PATH, dtype={"id": np.int32, "pressure": np.float32})

assert "id" in test.columns and "id" in sub.columns
assert len(test) == len(sub)




## === cell 1
def add_breath_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    u_in = out["u_in"].to_numpy(np.float32, copy=False)
    u_out = out["u_out"].to_numpy(np.float32, copy=False)
    t = out["time_step"].to_numpy(np.float32, copy=False)

    n = len(out)
    t_idx = (np.arange(n, dtype=np.int16) % 80).astype(np.int16, copy=False)

    u_in_lag1 = np.zeros_like(u_in, dtype=np.float32)
    u_in_lag2 = np.zeros_like(u_in, dtype=np.float32)
    u_out_lag1 = np.zeros_like(u_out, dtype=np.float32)

    mask_ge1 = t_idx >= 1
    mask_ge2 = t_idx >= 2

    u_in_lag1[mask_ge1] = u_in[np.where(mask_ge1)[0] - 1]
    u_in_lag2[mask_ge2] = u_in[np.where(mask_ge2)[0] - 2]
    u_out_lag1[mask_ge1] = u_out[np.where(mask_ge1)[0] - 1]

    out["u_in_lag1"] = u_in_lag1
    out["u_in_lag2"] = u_in_lag2
    out["u_out_lag1"] = u_out_lag1

    out["u_in_diff1"] = (u_in - u_in_lag1).astype(np.float32)
    out["u_out_diff1"] = (u_out - u_out_lag1).astype(np.float32)

    u_in_2d = u_in.reshape(-1, 80)
    out["u_in_cumsum"] = np.cumsum(u_in_2d, axis=1, dtype=np.float32).reshape(-1)

    t_2d = t.reshape(-1, 80)
    dt = np.zeros_like(t_2d, dtype=np.float32)
    dt[:, 1:] = t_2d[:, 1:] - t_2d[:, :-1]
    out["dt"] = dt.reshape(-1)
    out["t_cumsum"] = np.cumsum(t_2d, axis=1, dtype=np.float32).reshape(-1)

    return out


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

train_fe["t_idx"] = (np.arange(len(train_fe), dtype=np.int16) % 80).astype(np.int16)
test_fe["t_idx"] = (np.arange(len(test_fe), dtype=np.int16) % 80).astype(np.int16)

u_in_bin_width = 1.0
train_fe["u_in_bin"] = np.floor(
    train_fe["u_in"].to_numpy(np.float32, copy=False) / u_in_bin_width
).astype(np.int16)
test_fe["u_in_bin"] = np.floor(
    test_fe["u_in"].to_numpy(np.float32, copy=False) / u_in_bin_width
).astype(np.int16)

rs = 42

max_breaths = 12000
unique_breaths = train_fe["breath_id"].unique()
if len(unique_breaths) > max_breaths:
    rng = np.random.RandomState(rs)
    keep_breaths = rng.choice(unique_breaths, size=max_breaths, replace=False)
    keep_set = set(keep_breaths.tolist())
    mask_keep = train_fe["breath_id"].map(keep_set.__contains__).to_numpy()
    train_fit_df = train_fe.loc[mask_keep].copy()
else:
    train_fit_df = train_fe.copy()

breaths_fit = pd.unique(train_fit_df["breath_id"].to_numpy())
b_tr, b_va = train_test_split(breaths_fit, test_size=0.1, random_state=rs)

btr_set = set(b_tr.tolist())
bva_set = set(b_va.tolist())
tr_mask = train_fit_df["breath_id"].map(btr_set.__contains__).to_numpy()
va_mask = train_fit_df["breath_id"].map(bva_set.__contains__).to_numpy()

train_tr = train_fit_df.loc[tr_mask].copy()
train_va = train_fit_df.loc[va_mask].copy()

train_tr_fit = train_tr.loc[train_tr["u_out"].to_numpy() == 0].copy()

global_mean_pressure_tr = float(train_tr_fit[target].mean())

k_tr = pd.MultiIndex.from_frame(train_tr_fit[["R", "C", "t_idx"]])
prior_s_tr = train_tr_fit.groupby(["R", "C", "t_idx"], sort=False)[target].mean()
train_tr_fit["pressure_rc_tidx_mean"] = prior_s_tr.reindex(k_tr).to_numpy()

k_va = pd.MultiIndex.from_frame(train_va[["R", "C", "t_idx"]])
train_va["pressure_rc_tidx_mean"] = prior_s_tr.reindex(k_va).to_numpy()

prior_s_full = (
    train_fe.loc[train_fe["u_out"].to_numpy() == 0]
    .groupby(["R", "C", "t_idx"], sort=False)[target]
    .mean()
)
k_te = pd.MultiIndex.from_frame(test_fe[["R", "C", "t_idx"]])
test_fe["pressure_rc_tidx_mean"] = prior_s_full.reindex(k_te).to_numpy()

train_tr_fit["pressure_rc_tidx_mean"] = (
    pd.Series(train_tr_fit["pressure_rc_tidx_mean"])
    .fillna(global_mean_pressure_tr)
    .astype(np.float32)
    .to_numpy()
)
train_va["pressure_rc_tidx_mean"] = (
    pd.Series(train_va["pressure_rc_tidx_mean"])
    .fillna(global_mean_pressure_tr)
    .astype(np.float32)
    .to_numpy()
)
insp_mean_full = float(train_fe.loc[train_fe["u_out"].to_numpy() == 0, target].mean())
test_fe["pressure_rc_tidx_mean"] = (
    pd.Series(test_fe["pressure_rc_tidx_mean"])
    .fillna(insp_mean_full)
    .astype(np.float32)
    .to_numpy()
)

k2_tr = pd.MultiIndex.from_frame(train_tr_fit[["R", "C", "t_idx", "u_in_bin"]])
prior2_s_tr = train_tr_fit.groupby(["R", "C", "t_idx", "u_in_bin"], sort=False)[
    target
].mean()
train_tr_fit["pressure_rc_tidx_uinbin_mean"] = prior2_s_tr.reindex(k2_tr).to_numpy()

k2_va = pd.MultiIndex.from_frame(train_va[["R", "C", "t_idx", "u_in_bin"]])
train_va["pressure_rc_tidx_uinbin_mean"] = prior2_s_tr.reindex(k2_va).to_numpy()

prior2_s_full = (
    train_fe.loc[train_fe["u_out"].to_numpy() == 0]
    .groupby(["R", "C", "t_idx", "u_in_bin"], sort=False)[target]
    .mean()
)
k2_te = pd.MultiIndex.from_frame(test_fe[["R", "C", "t_idx", "u_in_bin"]])
test_fe["pressure_rc_tidx_uinbin_mean"] = prior2_s_full.reindex(k2_te).to_numpy()

train_tr_fit["pressure_rc_tidx_uinbin_mean"] = (
    pd.Series(train_tr_fit["pressure_rc_tidx_uinbin_mean"])
    .fillna(pd.Series(train_tr_fit["pressure_rc_tidx_mean"]))
    .fillna(global_mean_pressure_tr)
    .astype(np.float32)
    .to_numpy()
)
train_va["pressure_rc_tidx_uinbin_mean"] = (
    pd.Series(train_va["pressure_rc_tidx_uinbin_mean"])
    .fillna(pd.Series(train_va["pressure_rc_tidx_mean"]))
    .fillna(global_mean_pressure_tr)
    .astype(np.float32)
    .to_numpy()
)
test_fe["pressure_rc_tidx_uinbin_mean"] = (
    pd.Series(test_fe["pressure_rc_tidx_uinbin_mean"])
    .fillna(pd.Series(test_fe["pressure_rc_tidx_mean"]))
    .fillna(insp_mean_full)
    .astype(np.float32)
    .to_numpy()
)

features_plus = features + [
    "t_idx",
    "pressure_rc_tidx_mean",
    "pressure_rc_tidx_uinbin_mean",
    "u_in_bin",
]

R_levels = np.array([5, 20, 50], dtype=np.int16)
C_levels = np.array([10, 20, 50], dtype=np.int16)


def build_X(df: pd.DataFrame) -> np.ndarray:
    num_cols = [c for c in features_plus if c not in ("R", "C")]
    X_num = df[num_cols].to_numpy(dtype=np.float32, copy=False)

    Rv = df["R"].to_numpy(np.int16, copy=False)
    Cv = df["C"].to_numpy(np.int16, copy=False)
    R_oh = (Rv[:, None] == R_levels[None, :]).astype(np.int8, copy=False)
    C_oh = (Cv[:, None] == C_levels[None, :]).astype(np.int8, copy=False)

    return np.concatenate([X_num, R_oh, C_oh], axis=1)


X_tr = build_X(train_tr_fit)
X_va = build_X(train_va)
X_test = build_X(test_fe)

y_tr = train_tr_fit[target].to_numpy(dtype=np.float32, copy=False)
y_va = train_va[target].to_numpy(dtype=np.float32, copy=False)




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
    np.abs(va_pred[insp_mask_va] - y_va[insp_mask_va].astype(np.float32, copy=False))
)
print(
    "Validation MAE (breath_id split; trained on u_out==0 only; evaluated on u_out==0; R/C one-hot; +priors):",
    float(va_mae),
)




## === cell 4
test_pred = model.predict(X_test).astype(np.float32)

test_order = test_fe[["id", "breath_id", "u_out"]].copy()
test_order["pred"] = test_pred

pred = test_order["pred"].to_numpy(np.float32, copy=False)
u_out = test_order["u_out"].to_numpy(np.int8, copy=False)

pred2d = pred.reshape(-1, 80).copy()  # will become pred_filled per breath
uout2d = u_out.reshape(-1, 80)

last = np.zeros((pred2d.shape[0],), dtype=np.float32)
for j in range(80):
    insp = uout2d[:, j] == 0
    last = np.where(insp, pred2d[:, j], last)
    pred2d[:, j] = np.where(insp, pred2d[:, j], last)

pred_filled = pred2d.reshape(-1)

pressure_levels = np.sort(train["pressure"].unique().astype(np.float32))

is_insp = u_out == 0
pred_all = pred_filled.astype(np.float32, copy=False)

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

ids = test_order["id"].to_numpy(np.int32, copy=False)
order = np.argsort(ids, kind="mergesort")
pred_by_id = pred_snapped_all[order].astype(np.float32, copy=False)

sub["pressure"] = pred_by_id
sub.to_csv("submission.csv", index=False)

sub.head()
