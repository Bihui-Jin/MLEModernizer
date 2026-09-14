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
import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



## === cell 1
sub = pd.read_csv("../input/ventilator-pressure-prediction/sample_submission.csv")
sub.head()



## === cell 2
train_df = pd.read_csv("../input/ventilator-pressure-prediction/train.csv")
test_df = pd.read_csv("../input/ventilator-pressure-prediction/test.csv")

train_u_out_raw = train_df["u_out"].to_numpy()
train_time_raw = train_df["time_step"].to_numpy()

test_u_out_raw = test_df["u_out"].to_numpy()
test_breath_id_raw = test_df["breath_id"].to_numpy()

train_rc_raw = np.c_[
    train_df["R"].to_numpy(dtype=np.int16), train_df["C"].to_numpy(dtype=np.int16)
]
test_rc_raw = np.c_[
    test_df["R"].to_numpy(dtype=np.int16), test_df["C"].to_numpy(dtype=np.int16)
]


def add_features(df):
    df = df.copy()
    g = df.groupby("breath_id", sort=False)

    df["cross"] = df["u_in"] * df["u_out"]
    df["cross2"] = df["time_step"] * df["u_out"]
    df["area"] = (
        (df["time_step"] * df["u_in"]).groupby(df["breath_id"], sort=False).cumsum()
    )
    df["time_step_cumsum"] = g["time_step"].cumsum()
    df["u_in_cumsum"] = g["u_in"].cumsum()

    for k in (1, 2, 3, 4):
        df[f"u_in_lag{k}"] = g["u_in"].shift(k)
        df[f"u_out_lag{k}"] = g["u_out"].shift(k)
        df[f"u_in_lag_back{k}"] = g["u_in"].shift(-k)
        df[f"u_out_lag_back{k}"] = g["u_out"].shift(-k)

    df = df.fillna(0)

    u_in_max = g["u_in"].transform("max")
    u_in_mean = g["u_in"].transform("mean")
    df["breath_id__u_in__max"] = u_in_max
    df["breath_id__u_in__mean"] = u_in_mean
    df["breath_id__u_in__diffmax"] = u_in_max - df["u_in"]
    df["breath_id__u_in__diffmean"] = u_in_mean - df["u_in"]

    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_out_diff1"] = df["u_out"] - df["u_out_lag1"]
    df["u_in_diff2"] = df["u_in"] - df["u_in_lag2"]
    df["u_out_diff2"] = df["u_out"] - df["u_out_lag2"]
    df["u_in_diff3"] = df["u_in"] - df["u_in_lag3"]
    df["u_out_diff3"] = df["u_out"] - df["u_out_lag3"]
    df["u_in_diff4"] = df["u_in"] - df["u_in_lag4"]
    df["u_out_diff4"] = df["u_out"] - df["u_out_lag4"]

    df["one"] = 1
    df["count"] = df["one"].groupby(df["breath_id"], sort=False).cumsum()
    df["u_in_cummean"] = df["u_in_cumsum"] / df["count"]

    df["breath_id_lag"] = df["breath_id"].shift(1).fillna(0)
    df["breath_id_lag2"] = df["breath_id"].shift(2).fillna(0)
    df["breath_id_lagsame"] = (
        df["breath_id_lag"].to_numpy() == df["breath_id"].to_numpy()
    ).astype(np.int8)
    df["breath_id_lag2same"] = (
        df["breath_id_lag2"].to_numpy() == df["breath_id"].to_numpy()
    ).astype(np.int8)

    df["breath_id__u_in_lag"] = df["u_in"].shift(1).fillna(0) * df["breath_id_lagsame"]
    df["breath_id__u_in_lag2"] = (
        df["u_in"].shift(2).fillna(0) * df["breath_id_lag2same"]
    )

    df["time_step_diff"] = g["time_step"].diff().fillna(0)

    df["ewm_u_in_mean"] = (
        g["u_in"].ewm(halflife=9).mean().reset_index(level=0, drop=True)
    )

    roll = g["u_in"].rolling(window=15, min_periods=1)
    agg = roll.agg(["sum", "min", "max", "mean"]).reset_index(level=0, drop=True)
    df["15_in_sum"] = agg["sum"].to_numpy()
    df["15_in_min"] = agg["min"].to_numpy()
    df["15_in_max"] = agg["max"].to_numpy()
    df["15_in_mean"] = agg["mean"].to_numpy()

    df["u_in_lagback_diff1"] = df["u_in"] - df["u_in_lag_back1"]
    df["u_out_lagback_diff1"] = df["u_out"] - df["u_out_lag_back1"]
    df["u_in_lagback_diff2"] = df["u_in"] - df["u_in_lag_back2"]
    df["u_out_lagback_diff2"] = df["u_out"] - df["u_out_lag_back2"]

    df["R"] = df["R"].astype(str)
    df["C"] = df["C"].astype(str)
    df["R__C"] = df["R"] + "__" + df["C"]
    df = pd.get_dummies(df)

    return df


train = add_features(train_df)
test = add_features(test_df)

del train_df, test_df
gc.collect()



## === cell 3
target_col = "pressure"
targets = train[[target_col]].to_numpy().reshape(-1, 80)

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
train.drop(drop_cols, axis=1, inplace=True)

test_drop_cols = [
    "id",
    "breath_id",
    "one",
    "count",
    "breath_id_lag",
    "breath_id_lag2",
    "breath_id_lagsame",
    "breath_id_lag2same",
]
test = test.drop(test_drop_cols, axis=1)

train, test = train.align(test, join="outer", axis=1, fill_value=0)

print(f"train: {train.shape} \ntest: {test.shape} \ntargets: {targets.shape}")



## === cell 4
scaler = RobustScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

train_scaled = train_scaled.reshape(-1, 80, train_scaled.shape[-1])
test_scaled = test_scaled.reshape(-1, 80, train_scaled.shape[-1])

print(
    f"train: {train_scaled.shape} \ntest: {test_scaled.shape} \ntargets: {targets.shape}"
)



## === cell 5
pressure = targets.squeeze().reshape(-1, 1).astype("float32")
P_MIN = float(np.min(pressure))
P_MAX = float(np.max(pressure))

uniq_p = np.unique(pressure.ravel())
uniq_p.sort()
if uniq_p.shape[0] >= 2:
    diffs = np.diff(uniq_p)
    diffs = diffs[diffs > 1e-9]
    P_STEP = float(diffs.min()) if diffs.size else 0.0
else:
    P_STEP = 0.0

print("Min pressure: {}".format(P_MIN))
print("Max pressure: {}".format(P_MAX))
print("Pressure step: {}".format(P_STEP))
print("Unique values:  {}".format(uniq_p.shape[0]))

del pressure, uniq_p, diffs
gc.collect()



## === cell 6
from sklearn.linear_model import Ridge

X_train = train_scaled.astype(np.float32)  # (n_breaths, 80, n_feat)
X_test = test_scaled.astype(np.float32)  # (n_test_breaths, 80, n_feat)
Y_train = targets.astype(np.float32)  # (n_breaths, 80)

n_breaths = X_train.shape[0]
n_test_breaths = X_test.shape[0]
n_feat = X_train.shape[-1]

X_train_breath = X_train.reshape(n_breaths, -1)  # (n_breaths, 80*n_feat)
X_test_breath = X_test.reshape(n_test_breaths, -1)  # (n_test_breaths, 80*n_feat)

u_out_mat = train_u_out_raw.reshape(-1, 80)
first_exp = np.argmax(u_out_mat == 1, axis=1).astype(np.int16)
has_exp = (u_out_mat == 1).any(axis=1)
first_exp = np.where(has_exp, first_exp, 80).astype(np.int16)

t = np.arange(80, dtype=np.int16)[None, :]
insp_scored_mat = (t < first_exp[:, None]) & (u_out_mat == 0)  # (n_breaths, 80)

sample_weight_breath = (insp_scored_mat.sum(axis=1) / 80.0).astype(
    np.float32
)  # (n_breaths,)
sample_weight_breath = np.clip(sample_weight_breath, 0.0, 1.0)

train_rc_breath = np.asarray(
    train_rc_raw.reshape(-1, 80, 2)[:, 0, :], dtype=np.int16
)  # (n_breaths, 2)
test_rc_breath = np.asarray(
    test_rc_raw.reshape(-1, 80, 2)[:, 0, :], dtype=np.int16
)  # (n_test_breaths, 2)

alphas = [0.1, 0.3, 1.0, 3.0]
subs = []

unique_train_rc = np.unique(train_rc_breath, axis=0)

for a in alphas:
    global_model = Ridge(alpha=a, random_state=RANDOM_STATE)
    global_model.fit(X_train_breath, Y_train, sample_weight=sample_weight_breath)

    pred_breath = np.full((n_test_breaths, 80), np.nan, dtype=np.float32)

    for rc in unique_train_rc:
        rc_mask_train_b = (train_rc_breath[:, 0] == rc[0]) & (
            train_rc_breath[:, 1] == rc[1]
        )
        rc_mask_test_b = (test_rc_breath[:, 0] == rc[0]) & (
            test_rc_breath[:, 1] == rc[1]
        )

        if not np.any(rc_mask_test_b):
            continue

        w_subset_b = sample_weight_breath[rc_mask_train_b]
        n_scored_proxy = float(w_subset_b.sum() * 80.0)

        y_subset = Y_train[rc_mask_train_b]
        m_subset = insp_scored_mat[rc_mask_train_b]
        if m_subset.any():
            uniq_y = np.unique(y_subset[m_subset]).size
        else:
            uniq_y = 0

        if (n_scored_proxy < 1000.0) or (uniq_y < 10):
            pred_breath[rc_mask_test_b] = global_model.predict(
                X_test_breath[rc_mask_test_b]
            ).astype(np.float32)
        else:
            model = Ridge(alpha=a, random_state=RANDOM_STATE)
            model.fit(
                X_train_breath[rc_mask_train_b],
                Y_train[rc_mask_train_b],
                sample_weight=sample_weight_breath[rc_mask_train_b],
            )
            pred_breath[rc_mask_test_b] = model.predict(
                X_test_breath[rc_mask_test_b]
            ).astype(np.float32)

    nan_mask_b = ~np.isfinite(pred_breath).all(axis=1)
    if np.any(nan_mask_b):
        pred_breath[nan_mask_b] = global_model.predict(
            X_test_breath[nan_mask_b]
        ).astype(np.float32)

    subs.append(pred_breath.reshape(-1).astype(np.float32))

sub_0 = sub.copy()
sub_1 = sub.copy()
sub_2 = sub.copy()
sub_3 = sub.copy()
sub_0["pressure"] = subs[0]
sub_1["pressure"] = subs[1]
sub_2["pressure"] = subs[2]
sub_3["pressure"] = subs[3]

del (
    global_model,
    model,
    subs,
    X_train,
    X_test,
    Y_train,
    X_train_breath,
    X_test_breath,
    u_out_mat,
    first_exp,
    has_exp,
    t,
    insp_scored_mat,
    sample_weight_breath,
    train_time_raw,
    train_rc_raw,
    test_rc_raw,
    train_rc_breath,
    test_rc_breath,
    nan_mask_b,
    unique_train_rc,
    train,
    test,
    train_scaled,
    test_scaled,
    scaler,
)
gc.collect()



## === cell 7
pred = np.array(
    [
        np.array(sub_0["pressure"].values),
        np.array(sub_1["pressure"].values),
        np.array(sub_2["pressure"].values),
        np.array(sub_3["pressure"].values),
    ]
)
pred



## === cell 8
mean = np.mean(pred, axis=0)
med = np.median(pred, axis=0)
std = np.std(pred, axis=0)



## === cell 9
clipped_pres = np.clip(np.vstack(pred), mean - std, mean + std)
clipped_mean = np.mean(clipped_pres, axis=0)



## === cell 10
sub_mean = sub.copy()
sub_mean["pressure"] = mean
sub_mean.to_csv("submission_mean.csv", index=False)
sub_mean.head(5)



## === cell 11
sub_median = sub.copy()
sub_median["pressure"] = med
sub_median.to_csv("submission_median.csv", index=False)
sub_median.head(5)



## === cell 12
sub_clip = sub.copy()
sub_clip["pressure"] = clipped_mean
sub_clip.to_csv("submission_clipped_mean.csv", index=False)
sub_clip.head(5)



## === cell 13
submission = pd.read_csv(
    "../input/ventilator-pressure-prediction/sample_submission.csv"
)

submission["pressure"] = np.median(np.vstack(pred), axis=0).astype(np.float32)

if test_u_out_raw is not None and test_breath_id_raw is not None:
    pr = submission["pressure"].to_numpy(dtype=np.float32)
    uo = test_u_out_raw
    bid = test_breath_id_raw

    last = np.nan
    last_bid = bid[0]
    for i in range(pr.shape[0]):
        if bid[i] != last_bid:
            last_bid = bid[i]
            last = np.nan
        if uo[i] == 0:
            last = pr[i]
        else:
            if np.isfinite(last):
                pr[i] = last
    submission["pressure"] = pr

if P_STEP > 0:
    submission["pressure"] = (
        np.round((submission.pressure - P_MIN) / P_STEP) * P_STEP + P_MIN
    )
submission["pressure"] = np.clip(submission.pressure, P_MIN, P_MAX)

submission = submission[["id", "pressure"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
