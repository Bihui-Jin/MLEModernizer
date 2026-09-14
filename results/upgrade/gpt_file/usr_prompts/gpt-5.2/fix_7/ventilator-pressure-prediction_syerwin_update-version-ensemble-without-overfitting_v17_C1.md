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

# 5. Target score

0.1388861788602605

# 6. Current score

0.76301

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.35245) has done: 'Your notebook fails because it depends on external Kaggle datasets (missing `../input/gb-vpp-pulp-fiction/...` etc.), so `pred` is never created and all later ensemble cells crash. I remove that hard dependency by training a minimal scikit-learn baseline on your already-engineered features (no new feature logic), producing test predictions aligned to `id`. I also keep your existing pressure “grid snapping” post-processing (P_MIN/P_MAX/P_STEP) so outputs match the known discrete pressure levels, which should improve MAE versus raw regression. Finally, the code always write a valid `submission.csv` (and also keep `median_submission.csv` for compatibility).'
- What this solution (achieved 1.19628) has done: 'Your current score (1.35245, lower-is-better) is far from the target (0.1389), so we need a real lift without changing your core feature logic or model family. The biggest issue is that you’re training on *all* timesteps, but Kaggle only scores the inspiratory phase (`u_out==0`), so the model is learning many “unscored” expiratory patterns that harm the scored MAE; we train only on `u_out==0` rows while still predicting all test rows. To further align with the discrete target distribution (and keep your existing “grid snapping”), we add a tiny, safe post-process: replace predictions on expiratory rows (`u_out==1`) with the previous timestep’s prediction within each breath (these rows aren’t scored but this avoids unnatural jumps leaking into nearby inspiratory predictions for some models; it should not hurt and often helps slightly). All paths/output stay the same and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.78416) has done: 'Your current MAE (1.196) is still far from the target (0.1389), so we need a meaningful lift while keeping your feature engineering and the same model family. The smallest high-impact change is to train a separate model per (R, C) lung setting (9 groups total) because pressure dynamics differ strongly by these attributes; this preserves the same HistGradientBoostingRegressor core logic and training approach, just applied per subgroup. We also keep your inspiratory-only training (`u_out==0`) and your existing discrete “grid snapping” post-processing. Finally, we ensure one-hot columns stay aligned between each group’s train/test matrices and we still write a valid `submission.csv`.'
- What this solution (achieved 0.78416) has done: 'Your current MAE (0.78416, lower-is-better) is still far from the target (0.1389), so we should improve meaningfully without changing the core model/feature logic. The biggest safe gain here is to align train/test feature processing by fitting the scaler on the full group’s training rows (both `u_out` phases) but still training the regressor only on inspiratory rows—this reduces distribution shift from scaling on a subset while keeping the same model and metric-aligned loss. Next, we ensure the group list covers all (R,C) pairs present in test (not just train), with a safe fallback to a global model when a test group is missing in train. Finally, we keep your existing post-processing (expiratory carry-forward + grid snapping) unchanged so evaluation semantics remain aligned.'
- What this solution (achieved 0.75765) has done: 'We’re still far from the target (0.784 → 0.139, lower-is-better), so the smallest high-impact improvement that preserves your feature logic and model family is to add a second-stage *residual* model per (R,C) group: train model#1 as you already do, then train model#2 to predict the remaining error and add it to the prediction. This keeps the same algorithm (HistGradientBoostingRegressor) and the same feature set, just applied twice, and often reduces MAE substantially without changing evaluation semantics. I also ensure feature alignment within each group using `reindex` (guarding against any dummy-column mismatch) while leaving all paths and your existing post-processing (expiratory carry-forward + grid snapping) unchanged. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.76301) has done: 'Your current score (0.75765, lower-is-better) is still far from the target (0.1389), so we need a real lift but with minimal, metric-aligned changes. The biggest win that preserves your core feature engineering and model family is to (1) enforce the competition’s discrete pressure levels by mapping predictions to the nearest *observed* pressure value (stronger than uniform step snapping), and (2) avoid training on noisy transition points by training only on inspiratory rows **excluding** the first timestep of each breath (still predicting for all timesteps). These changes don’t alter your model architecture/loop (still per-(R,C) two-stage HistGBR with RobustScaler) and keep the same output semantics, but typically reduce MAE substantially on this competition. The rest of the pipeline, paths, and submission writing remain unchanged.'

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingRegressor



## === cell 1
DATA_DIR = "../input/ventilator-pressure-prediction"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

print(train_df.shape, test_df.shape, sample_sub.shape)




## === cell 2
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["cross"] = df["u_in"] * df["u_out"]
    df["cross2"] = df["time_step"] * df["u_out"]
    df["area"] = df["time_step"] * df["u_in"]
    df["area"] = df.groupby("breath_id")["area"].cumsum()
    df["time_step_cumsum"] = df.groupby(["breath_id"])["time_step"].cumsum()
    df["u_in_cumsum"] = (df["u_in"]).groupby(df["breath_id"]).cumsum()
    print("Step-1...Completed")

    df["u_in_lag1"] = df.groupby("breath_id")["u_in"].shift(1)
    df["u_out_lag1"] = df.groupby("breath_id")["u_out"].shift(1)
    df["u_in_lag_back1"] = df.groupby("breath_id")["u_in"].shift(-1)
    df["u_out_lag_back1"] = df.groupby("breath_id")["u_out"].shift(-1)
    df["u_in_lag2"] = df.groupby("breath_id")["u_in"].shift(2)
    df["u_out_lag2"] = df.groupby("breath_id")["u_out"].shift(2)
    df["u_in_lag_back2"] = df.groupby("breath_id")["u_in"].shift(-2)
    df["u_out_lag_back2"] = df.groupby("breath_id")["u_out"].shift(-2)
    df["u_in_lag3"] = df.groupby("breath_id")["u_in"].shift(3)
    df["u_out_lag3"] = df.groupby("breath_id")["u_out"].shift(3)
    df["u_in_lag_back3"] = df.groupby("breath_id")["u_in"].shift(-3)
    df["u_out_lag_back3"] = df.groupby("breath_id")["u_out"].shift(-3)
    df["u_in_lag4"] = df.groupby("breath_id")["u_in"].shift(4)
    df["u_out_lag4"] = df.groupby("breath_id")["u_out"].shift(4)
    df["u_in_lag_back4"] = df.groupby("breath_id")["u_in"].shift(-4)
    df["u_out_lag_back4"] = df.groupby("breath_id")["u_out"].shift(-4)
    df = df.fillna(0)
    print("Step-2...Completed")

    df["breath_id__u_in__max"] = df.groupby(["breath_id"])["u_in"].transform("max")
    df["breath_id__u_in__mean"] = df.groupby(["breath_id"])["u_in"].transform("mean")
    df["breath_id__u_in__diffmax"] = (
        df.groupby(["breath_id"])["u_in"].transform("max") - df["u_in"]
    )
    df["breath_id__u_in__diffmean"] = (
        df.groupby(["breath_id"])["u_in"].transform("mean") - df["u_in"]
    )
    print("Step-3...Completed")

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_out_diff1"] = df["u_out"] - df["u_out_lag1"]
    df["u_in_diff2"] = df["u_in"] - df["u_in_lag2"]
    df["u_out_diff2"] = df["u_out"] - df["u_out_lag2"]
    df["u_in_diff3"] = df["u_in"] - df["u_in_lag3"]
    df["u_out_diff3"] = df["u_out"] - df["u_out_lag3"]
    df["u_in_diff4"] = df["u_in"] - df["u_in_lag4"]
    df["u_out_diff4"] = df["u_out"] - df["u_out_lag4"]
    print("Step-4...Completed")

    df["one"] = 1
    df["count"] = (df["one"]).groupby(df["breath_id"]).cumsum()
    df["u_in_cummean"] = df["u_in_cumsum"] / df["count"]

    df["breath_id_lag"] = df["breath_id"].shift(1).fillna(0)
    df["breath_id_lag2"] = df["breath_id"].shift(2).fillna(0)
    df["breath_id_lagsame"] = np.select(
        [df["breath_id_lag"] == df["breath_id"]], [1], 0
    )
    df["breath_id_lag2same"] = np.select(
        [df["breath_id_lag2"] == df["breath_id"]], [1], 0
    )
    df["breath_id__u_in_lag"] = df["u_in"].shift(1).fillna(0)
    df["breath_id__u_in_lag"] = df["breath_id__u_in_lag"] * df["breath_id_lagsame"]
    df["breath_id__u_in_lag2"] = df["u_in"].shift(2).fillna(0)
    df["breath_id__u_in_lag2"] = df["breath_id__u_in_lag2"] * df["breath_id_lag2same"]
    print("Step-5...Completed")

    df["time_step_diff"] = df.groupby("breath_id")["time_step"].diff().fillna(0)
    df["ewm_u_in_mean"] = (
        df.groupby("breath_id")["u_in"]
        .ewm(halflife=9)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df[["15_in_sum", "15_in_min", "15_in_max", "15_in_mean"]] = (
        df.groupby("breath_id")["u_in"]
        .rolling(window=15, min_periods=1)
        .agg(
            {
                "15_in_sum": "sum",
                "15_in_min": "min",
                "15_in_max": "max",
                "15_in_mean": "mean",
            }
        )
        .reset_index(level=0, drop=True)
    )
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
train_feat = add_features(train_df.copy())

print("\nTest data...\n")
test_feat = add_features(test_df.copy())

gc.collect()



## === cell 3
y_all = train_feat["pressure"].astype("float32").to_numpy()

pressure_unique = np.unique(y_all.astype("float32"))
P_MIN = float(pressure_unique.min())
P_MAX = float(pressure_unique.max())
print(f"Min pressure: {P_MIN}")
print(f"Max pressure: {P_MAX}")
print(f"Unique pressure values: {pressure_unique.shape[0]}")

test_ids = test_df["id"].to_numpy()
test_u_out = test_df["u_out"].to_numpy()
test_breath_id = test_df["breath_id"].to_numpy()

drop_cols_train = [
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
drop_cols_test = [
    "id",
    "breath_id",
    "one",
    "count",
    "breath_id_lag",
    "breath_id_lag2",
    "breath_id_lagsame",
    "breath_id_lag2same",
]

X_train_all = train_feat.drop(drop_cols_train, axis=1)
X_test_all = test_feat.drop(drop_cols_test, axis=1)
X_test_all = X_test_all.reindex(columns=X_train_all.columns, fill_value=0)

train_insp_mask = train_df["u_out"].to_numpy() == 0
train_first_step_mask = train_df.groupby("breath_id").cumcount().to_numpy() == 0
train_insp_mask = train_insp_mask & (~train_first_step_mask)

train_R = train_df["R"].to_numpy()
train_C = train_df["C"].to_numpy()
test_R = test_df["R"].to_numpy()
test_C = test_df["C"].to_numpy()

del y_all
gc.collect()



## === cell 4
gbr_params = dict(
    loss="absolute_error",  # aligns with MAE metric
    max_depth=6,
    max_iter=300,
    learning_rate=0.05,
    random_state=42,
)

residual_params = dict(
    loss="absolute_error",
    max_depth=4,  # slightly smaller capacity to avoid overshooting corrections
    max_iter=200,
    learning_rate=0.05,
    random_state=42,
)

test_pred = np.zeros(X_test_all.shape[0], dtype=np.float32)

train_groups = set(zip(train_R.tolist(), train_C.tolist()))
test_groups = set(zip(test_R.tolist(), test_C.tolist()))
all_groups = sorted(train_groups.union(test_groups))
print("Unique (R,C) groups in train:", sorted(train_groups))
print("Unique (R,C) groups in test :", sorted(test_groups))

global_train_mask = train_insp_mask
X_tr_global = X_train_all.loc[global_train_mask]
y_tr_global = train_feat.loc[global_train_mask, "pressure"].astype("float32").to_numpy()

global_scaler = RobustScaler()
X_tr_global_scaled = global_scaler.fit_transform(X_tr_global)

global_gbr1 = HistGradientBoostingRegressor(**gbr_params)
global_gbr1.fit(X_tr_global_scaled, y_tr_global)

global_pred1_tr = global_gbr1.predict(X_tr_global_scaled).astype("float32")
global_resid = (y_tr_global - global_pred1_tr).astype("float32")

global_gbr2 = HistGradientBoostingRegressor(**residual_params)
global_gbr2.fit(X_tr_global_scaled, global_resid)

del X_tr_global, y_tr_global, X_tr_global_scaled, global_pred1_tr, global_resid
gc.collect()

for r, c in all_groups:
    train_group_all_mask = (train_R == r) & (train_C == c)
    train_group_insp_mask = train_group_all_mask & train_insp_mask
    test_group_mask = (test_R == r) & (test_C == c)

    n_tr_all = int(train_group_all_mask.sum())
    n_tr_insp = int(train_group_insp_mask.sum())
    n_te = int(test_group_mask.sum())
    print(
        f"Group (R={r}, C={c}): train_all_rows={n_tr_all}, train_insp_rows={n_tr_insp}, test_rows={n_te}"
    )

    if n_te == 0:
        continue

    if n_tr_all == 0 or n_tr_insp == 0:
        X_te = X_test_all.loc[test_group_mask]
        X_te_scaled = global_scaler.transform(X_te)

        pred1 = global_gbr1.predict(X_te_scaled).astype("float32")
        pred2 = global_gbr2.predict(X_te_scaled).astype("float32")
        test_pred[test_group_mask] = (pred1 + pred2).astype("float32")

        del X_te, X_te_scaled, pred1, pred2
        gc.collect()
        continue

    X_tr_all = X_train_all.loc[train_group_all_mask].reindex(
        columns=X_train_all.columns, fill_value=0
    )
    X_tr_insp = X_train_all.loc[train_group_insp_mask].reindex(
        columns=X_train_all.columns, fill_value=0
    )
    y_tr_insp = (
        train_feat.loc[train_group_insp_mask, "pressure"].astype("float32").to_numpy()
    )
    X_te = X_test_all.loc[test_group_mask].reindex(
        columns=X_train_all.columns, fill_value=0
    )

    scaler = RobustScaler()
    scaler.fit(X_tr_all)
    X_tr_insp_scaled = scaler.transform(X_tr_insp)
    X_te_scaled = scaler.transform(X_te)

    gbr1 = HistGradientBoostingRegressor(**gbr_params)
    gbr1.fit(X_tr_insp_scaled, y_tr_insp)
    pred1_tr = gbr1.predict(X_tr_insp_scaled).astype("float32")
    resid = (y_tr_insp - pred1_tr).astype("float32")

    gbr2 = HistGradientBoostingRegressor(**residual_params)
    gbr2.fit(X_tr_insp_scaled, resid)

    pred1_te = gbr1.predict(X_te_scaled).astype("float32")
    pred2_te = gbr2.predict(X_te_scaled).astype("float32")
    test_pred[test_group_mask] = (pred1_te + pred2_te).astype("float32")

    del (
        X_tr_all,
        X_tr_insp,
        y_tr_insp,
        X_te,
        X_tr_insp_scaled,
        X_te_scaled,
        scaler,
        gbr1,
        gbr2,
        pred1_tr,
        resid,
        pred1_te,
        pred2_te,
    )
    gc.collect()

print(
    "Pred stats (raw):",
    float(test_pred.min()),
    float(test_pred.max()),
    float(test_pred.mean()),
)



## === cell 5
pred_df = (
    pd.DataFrame(
        {
            "id": test_ids,
            "breath_id": test_breath_id,
            "u_out": test_u_out,
            "pressure": test_pred,
        }
    )
    .sort_values(["breath_id", "id"])
    .reset_index(drop=True)
)

prev_pred = pred_df.groupby("breath_id")["pressure"].shift(1)
pred_df.loc[pred_df["u_out"] == 1, "pressure"] = prev_pred.loc[
    pred_df["u_out"] == 1
].fillna(pred_df.loc[pred_df["u_out"] == 1, "pressure"])

pvals = pressure_unique  # sorted unique
idx = np.searchsorted(pvals, pred_df["pressure"].to_numpy(), side="left")
idx = np.clip(idx, 0, len(pvals) - 1)
idx_left = np.clip(idx - 1, 0, len(pvals) - 1)

p_right = pvals[idx]
p_left = pvals[idx_left]
raw = pred_df["pressure"].to_numpy()
choose_left = (idx > 0) & (np.abs(raw - p_left) <= np.abs(raw - p_right))
snapped = p_right.copy()
snapped[choose_left] = p_left[choose_left]
pred_df["pressure"] = snapped.astype(np.float32)

pred_df["pressure"] = np.clip(pred_df["pressure"], P_MIN, P_MAX)

print(
    "Pred stats (post):",
    float(pred_df["pressure"].min()),
    float(pred_df["pressure"].max()),
    float(pred_df["pressure"].mean()),
)

submission = sample_sub.copy().sort_values("id").reset_index(drop=True)
pred_df_sorted_by_id = pred_df.sort_values("id").reset_index(drop=True)
submission["pressure"] = pred_df_sorted_by_id["pressure"].to_numpy()

submission.to_csv("submission.csv", index=False)
submission.to_csv("median_submission.csv", index=False)

print(submission.head())
