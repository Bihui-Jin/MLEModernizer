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
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler



## === cell 1
np.random.seed(42)




## === cell 2
def mae(ytrue, ypred, uout=None):
    if isinstance(uout, (pd.Series, np.ndarray)):
        print(f"MAE (Inspiration Phase):")
        uout_arr = np.asarray(uout)
        ytrue_arr = np.asarray(ytrue)
        ypred_arr = np.asarray(ypred)
        return np.mean(np.abs((ytrue_arr - ypred_arr)[uout_arr == 0]))
    else:
        print("MAE (All Phases):")
        return np.mean(np.abs(np.asarray(ytrue) - np.asarray(ypred)))




## === cell 3
TRAIN_PATH = "../input/ventilator-pressure-prediction/train.csv"
TEST_PATH = "../input/ventilator-pressure-prediction/test.csv"
SAMPLE_SUB_PATH = "../input/ventilator-pressure-prediction/sample_submission.csv"



## === cell 4
data = pd.read_csv(TRAIN_PATH, usecols=["pressure", "u_out"])
ytrue = data.pressure
uout = data.u_out



## === cell 5
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)



## === cell 6
pressure_insp = train.loc[train["u_out"] == 0, "pressure"].to_numpy()
pressure_sorted = np.sort(np.unique(pressure_insp))
PRESSURE_MIN = float(pressure_sorted[0])
PRESSURE_MAX = float(pressure_sorted[-1])
PRESSURE_STEP = float(pressure_sorted[1] - pressure_sorted[0])


def post_process(pressure):
    pressure = (
        np.round((pressure - PRESSURE_MIN) / PRESSURE_STEP) * PRESSURE_STEP
        + PRESSURE_MIN
    )
    pressure = np.clip(pressure, PRESSURE_MIN, PRESSURE_MAX)
    return pressure




## === cell 7
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    g = df.groupby("breath_id", sort=False)

    df["delta_time"] = g["time_step"].diff().fillna(0.0).astype(np.float32)
    df["u_in_dt"] = (df["u_in"] * df["delta_time"]).astype(np.float32)
    df["u_in_cum"] = g["u_in_dt"].cumsum().astype(np.float32)

    df["u_in_lag1"] = g["u_in"].shift(1).fillna(0.0)
    df["u_in_lag2"] = g["u_in"].shift(2).fillna(0.0)
    df["u_in_lag3"] = g["u_in"].shift(3).fillna(0.0)
    df["u_in_lag4"] = g["u_in"].shift(4).fillna(0.0)

    df["u_in_lag5"] = g["u_in"].shift(5).fillna(0.0)
    df["u_in_lag6"] = g["u_in"].shift(6).fillna(0.0)
    df["u_in_lag7"] = g["u_in"].shift(7).fillna(0.0)
    df["u_in_lag8"] = g["u_in"].shift(8).fillna(0.0)
    df["u_in_lag9"] = g["u_in"].shift(9).fillna(0.0)
    df["u_in_lag10"] = g["u_in"].shift(10).fillna(0.0)

    df["u_in_lead1"] = g["u_in"].shift(-1).fillna(0.0)
    df["u_in_lead2"] = g["u_in"].shift(-2).fillna(0.0)

    df["u_out_lag1"] = g["u_out"].shift(1).fillna(0).astype(np.int8)
    df["u_out_lag2"] = g["u_out"].shift(2).fillna(0).astype(np.int8)
    df["u_out_lead1"] = g["u_out"].shift(-1).fillna(0).astype(np.int8)

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_in_diff2"] = df["u_in_lag1"] - df["u_in_lag2"]
    df["u_in_diff3"] = df["u_in_lag2"] - df["u_in_lag3"]

    df["time_step2"] = df["time_step"] * df["time_step"]
    df["time_step3"] = df["time_step2"] * df["time_step"]

    df["step"] = g.cumcount().astype(np.int16)
    df["step2"] = (df["step"].astype(np.float32) ** 2).astype(np.float32)

    df["time_frac"] = (df["step"].astype(np.float32) / 79.0).astype(np.float32)

    df["R_num"] = df["R"].astype(np.float32)
    df["C_num"] = df["C"].astype(np.float32)
    df["u_in_R"] = df["u_in"] * df["R_num"]
    df["u_in_C"] = df["u_in"] * df["C_num"]
    df["u_in_cum_R"] = df["u_in_cum"] * df["R_num"]
    df["u_in_cum_C"] = df["u_in_cum"] * df["C_num"]

    u_in_cum_sum = g["u_in"].cumsum().astype(np.float32)
    df["u_in_cummean"] = (u_in_cum_sum / (df["step"].astype(np.float32) + 1.0)).astype(
        np.float32
    )
    df["u_in_cummax"] = g["u_in"].cummax().astype(np.float32)

    df["u_in_rollmean_3"] = (
        g["u_in"]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .astype(np.float32)
    )
    df["u_in_rollmean_5"] = (
        g["u_in"]
        .rolling(window=5, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .astype(np.float32)
    )
    df["u_in_rollstd_5"] = (
        g["u_in"]
        .rolling(window=5, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
        .astype(np.float32)
    )

    df["u_out_cum"] = g["u_out"].cumsum().astype(np.int16)

    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)
    df = pd.get_dummies(df, columns=["R", "C"], drop_first=False)

    if "pressure" in df.columns:
        df["pressure_lag1"] = g["pressure"].shift(1).fillna(0.0).astype(np.float32)
        df["pressure_lag2"] = g["pressure"].shift(2).fillna(0.0).astype(np.float32)
    else:
        df["pressure_lag1"] = np.float32(0.0)
        df["pressure_lag2"] = np.float32(0.0)

    drop_cols = {"pressure", "id", "breath_id"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    for c in feature_cols:
        if df[c].dtype == "bool":
            df[c] = df[c].astype(np.int8)

    return df[feature_cols]




## === cell 8
train_feat = make_features(train)
test_feat = make_features(test)

missing_in_test = [c for c in train_feat.columns if c not in test_feat.columns]
missing_in_train = [c for c in test_feat.columns if c not in train_feat.columns]
for c in missing_in_test:
    test_feat[c] = 0
for c in missing_in_train:
    train_feat[c] = 0
train_feat = train_feat.sort_index(axis=1)
test_feat = test_feat.sort_index(axis=1)

X_all = train_feat.to_numpy(dtype=np.float32)
X_test = test_feat.to_numpy(dtype=np.float32)
y_all = train["pressure"].to_numpy(dtype=np.float32)

u_out_all = train["u_out"].to_numpy(dtype=np.int8)
breath_all = train["breath_id"].to_numpy()

mask_insp_all = u_out_all == 0

print(f"X_all shape: {X_all.shape}")
print(f"y_all shape: {y_all.shape}")
print(f"Inspiration rows: {int(mask_insp_all.sum())} / {len(mask_insp_all)}")



## === cell 9
alphas = np.logspace(-3, 10, 20)

unique_breaths = np.unique(breath_all)
rng = np.random.RandomState(42)
rng.shuffle(unique_breaths)

n_folds = 5
folds = np.array_split(unique_breaths, n_folds)

feature_cols = list(train_feat.columns)
pressure_lag1_idx = feature_cols.index("pressure_lag1")
pressure_lag2_idx = feature_cols.index("pressure_lag2")

best_alpha = None
best_cv_mae = np.inf

for a in alphas:
    fold_maes = []
    for k in range(n_folds):
        val_breaths = folds[k]
        is_val = np.isin(breath_all, val_breaths)
        is_tr = ~is_val

        tr_mask = is_tr & (u_out_all == 0)

        scaler_k = StandardScaler(with_mean=True, with_std=True)
        X_tr = scaler_k.fit_transform(X_all[tr_mask])
        y_tr = y_all[tr_mask]

        model = Ridge(alpha=float(a), solver="svd")
        model.fit(X_tr, y_tr)

        X_val_ar = X_all[is_val].copy()
        X_val_ar_scaled = scaler_k.transform(X_val_ar)
        pred_val = np.zeros(X_val_ar.shape[0], dtype=np.float32)

        breath_ids_val = breath_all[is_val]
        u_out_val = u_out_all[is_val]
        unique_val_breaths = pd.unique(breath_ids_val)

        mean_k = scaler_k.mean_.astype(np.float32, copy=False)
        scale_k = scaler_k.scale_.astype(np.float32, copy=False)

        for b in unique_val_breaths:
            idx = np.where(breath_ids_val == b)[0]
            for j, ridx in enumerate(idx):
                lag1 = pred_val[idx[j - 1]] if j >= 1 else 0.0
                lag2 = pred_val[idx[j - 2]] if j >= 2 else 0.0

                X_val_ar_scaled[ridx, pressure_lag1_idx] = (
                    lag1 - mean_k[pressure_lag1_idx]
                ) / scale_k[pressure_lag1_idx]
                X_val_ar_scaled[ridx, pressure_lag2_idx] = (
                    lag2 - mean_k[pressure_lag2_idx]
                ) / scale_k[pressure_lag2_idx]

                pred_val[ridx] = np.float32(
                    model.predict(X_val_ar_scaled[ridx : ridx + 1])[0]
                )

        pred_val[u_out_val == 1] = 0.0
        pred_val = post_process(pred_val)

        y_val = y_all[is_val]
        insp_mask_val = u_out_val == 0
        fold_maes.append(
            float(np.mean(np.abs(y_val[insp_mask_val] - pred_val[insp_mask_val])))
        )

    cv_mae = float(np.mean(fold_maes))
    if cv_mae < best_cv_mae:
        best_cv_mae = cv_mae
        best_alpha = float(a)

print(
    "Breath-level CV MAE (AR val inference, inspiration only, expiratory=0, post-processed):",
    best_cv_mae,
)
print("Chosen alpha:", best_alpha)



## === cell 10
oof_pred = np.zeros(X_all.shape[0], dtype=np.float32)

for k in range(n_folds):
    val_breaths = folds[k]
    is_val = np.isin(breath_all, val_breaths)
    is_tr = ~is_val

    tr_mask = is_tr & (u_out_all == 0)

    scaler_k = StandardScaler(with_mean=True, with_std=True)
    X_tr = scaler_k.fit_transform(X_all[tr_mask])
    y_tr = y_all[tr_mask]

    model = Ridge(alpha=best_alpha, solver="svd")
    model.fit(X_tr, y_tr)

    X_val_ar = X_all[is_val].copy()
    X_val_ar_scaled = scaler_k.transform(X_val_ar)
    pred_val = np.zeros(X_val_ar.shape[0], dtype=np.float32)

    breath_ids_val = breath_all[is_val]
    u_out_val = u_out_all[is_val]
    unique_val_breaths = pd.unique(breath_ids_val)

    mean_k = scaler_k.mean_.astype(np.float32, copy=False)
    scale_k = scaler_k.scale_.astype(np.float32, copy=False)

    for b in unique_val_breaths:
        idx = np.where(breath_ids_val == b)[0]
        for j, ridx in enumerate(idx):
            lag1 = pred_val[idx[j - 1]] if j >= 1 else 0.0
            lag2 = pred_val[idx[j - 2]] if j >= 2 else 0.0

            X_val_ar_scaled[ridx, pressure_lag1_idx] = (
                lag1 - mean_k[pressure_lag1_idx]
            ) / scale_k[pressure_lag1_idx]
            X_val_ar_scaled[ridx, pressure_lag2_idx] = (
                lag2 - mean_k[pressure_lag2_idx]
            ) / scale_k[pressure_lag2_idx]

            pred_val[ridx] = np.float32(
                model.predict(X_val_ar_scaled[ridx : ridx + 1])[0]
            )

    oof_pred[is_val] = pred_val

oof_pred_pp = post_process(oof_pred.copy())
oof_pred_pp[u_out_all == 1] = 0.0

gidx = pd.Series(np.arange(len(breath_all)))
tmp = pd.DataFrame({"breath_id": breath_all, "oof": oof_pred_pp})
tmp["oof_lag1"] = (
    tmp.groupby("breath_id", sort=False)["oof"].shift(1).fillna(0.0).astype(np.float32)
)
tmp["oof_lag2"] = (
    tmp.groupby("breath_id", sort=False)["oof"].shift(2).fillna(0.0).astype(np.float32)
)

X_all_ooflag = X_all.copy()
X_all_ooflag[:, pressure_lag1_idx] = tmp["oof_lag1"].to_numpy(
    dtype=np.float32, copy=False
)
X_all_ooflag[:, pressure_lag2_idx] = tmp["oof_lag2"].to_numpy(
    dtype=np.float32, copy=False
)

scaler = StandardScaler(with_mean=True, with_std=True)
X_scaled_insp = scaler.fit_transform(X_all_ooflag[mask_insp_all])

lin_reg = Ridge(alpha=best_alpha, solver="svd")
lin_reg.fit(X_scaled_insp, y_all[mask_insp_all])

X_scaled_all = scaler.transform(X_all_ooflag)
pred_train = lin_reg.predict(X_scaled_all).astype(np.float32)
pred_train[u_out_all == 1] = 0.0
pred_train_pp = post_process(pred_train)

print(mae(y_all, pred_train_pp, uout=u_out_all))
print(f"Number of features: {lin_reg.coef_.shape[0]}")



## === cell 11
feature_cols = list(train_feat.columns)
pressure_lag1_idx = feature_cols.index("pressure_lag1")
pressure_lag2_idx = feature_cols.index("pressure_lag2")

X_test_ar = X_test.copy()

X_test_scaled = scaler.transform(X_test_ar).astype(np.float32, copy=False)
mean_f = scaler.mean_.astype(np.float32, copy=False)
scale_f = scaler.scale_.astype(np.float32, copy=False)

pred_test = np.zeros(X_test_ar.shape[0], dtype=np.float32)

breath_ids_test = test["breath_id"].to_numpy()
u_out_test = test["u_out"].to_numpy(dtype=np.int8)

unique_test_breaths = pd.unique(breath_ids_test)

for b in unique_test_breaths:
    idx = np.where(breath_ids_test == b)[0]
    for j, ridx in enumerate(idx):
        lag1 = pred_test[idx[j - 1]] if j >= 1 else 0.0
        lag2 = pred_test[idx[j - 2]] if j >= 2 else 0.0

        X_test_scaled[ridx, pressure_lag1_idx] = (
            lag1 - mean_f[pressure_lag1_idx]
        ) / scale_f[pressure_lag1_idx]
        X_test_scaled[ridx, pressure_lag2_idx] = (
            lag2 - mean_f[pressure_lag2_idx]
        ) / scale_f[pressure_lag2_idx]

        pred_test[ridx] = np.float32(lin_reg.predict(X_test_scaled[ridx : ridx + 1])[0])

pred_test[u_out_test == 1] = 0.0
pred_test_pp = post_process(pred_test)



## === cell 12
submission = pd.read_csv(SAMPLE_SUB_PATH)[["id"]].copy()
pred_df = pd.DataFrame(
    {"id": test["id"].values, "pressure": pred_test_pp.astype(np.float32)}
)

submission = submission.merge(pred_df, on="id", how="left")
submission["pressure"] = submission["pressure"].fillna(0.0).astype(np.float32)



## === cell 13
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)



## === cell 14
oof1 = oof2 = oof3 = oof4 = None
sub1 = sub2 = sub3 = sub4 = None



## === cell 15
submission.to_csv("submission_pp.csv", index=False)
print("Wrote submission_pp.csv with shape:", submission.shape)



## === cell 16
p = submission["pressure"].to_numpy()
print("pressure min/max:", float(p.min()), float(p.max()))
grid_ok = np.allclose(
    ((p - PRESSURE_MIN) / PRESSURE_STEP),
    np.round((p - PRESSURE_MIN) / PRESSURE_STEP),
    atol=1e-6,
)
print("on pressure grid:", grid_ok)
