# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict the fare amount for a taxi ride given the pickup and dropoff locations.

## Metric
Root mean-squared error.

## Submission Format
For each `key` in the test set, you must predict a value for the `fare_amount` variable. The file should contain a header and have the following format:

```
key,fare_amount
2015-01-27 13:08:24.0000002,11.00
2015-02-27 13:08:24.0000002,12.05
2015-03-27 13:08:24.0000002,11.23
2015-04-27 13:08:24.0000002,14.17
2015-05-27 13:08:24.0000002,15.12
etc
```

## Dataset
- **train.csv** - Input features and target `fare_amount` values for the training set (about 55M rows).
- **test.csv** - Input features for the test set (about 10K rows). Your goal is to predict `fare_amount` for each row.
- **sample_submission.csv** - a sample submission file in the correct format (columns `key` and `fare_amount`). This file 'predicts' `fare_amount` to be $`11.35` for all rows, which is the mean `fare_amount` from the training set.

### Data fields
**ID**

- **key** - Unique `string` identifying each row in both the training and test sets. Comprised of **pickup_datetime** plus a unique integer, but this doesn't matter, it should just be used as a unique ID field.Required in your submission CSV. Not necessarily needed in the training set, but could be useful to simulate a 'submission file' while doing cross-validation within the training set.

**Features**

- **pickup_datetime** - `timestamp` value indicating when the taxi ride started.
- **pickup_longitude** - `float` for longitude coordinate of where the taxi ride started.
- **pickup_latitude** - `float` for latitude coordinate of where the taxi ride started.
- **dropoff_longitude** - `float` for longitude coordinate of where the taxi ride ended.
- **dropoff_latitude** - `float` for latitude coordinate of where the taxi ride ended.
- **passenger_count** - `integer` indicating the number of passengers in the taxi ride.

**Target**

- **fare_amount** - `float` dollar amount of the cost of the taxi ride. This value is only in the training set; this is what you are predicting in the test set and it is required in your submission CSV.

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            GCP-Coupons-Instructions.rtf (486 Bytes)
            description.md (100 lines)
            labels.csv (55413943 lines)
            labels.csv.zip (1.6 GB)
            sample_submission.csv (9915 lines)
            sample_submission.csv.zip (76.2 kB)
            test.csv (9915 lines)
            test.csv.zip (273.0 kB)
            train.csv (55423857 lines)
            train.csv.zip (1.6 GB)
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
        input/
            GCP-Coupons-Instructions.rtf (486 Bytes)
            description.md (100 lines)
            labels.csv (55413943 lines)
            labels.csv.zip (1.6 GB)
            sample_submission.csv (9915 lines)
            sample_submission.csv.zip (76.2 kB)
            test.csv (9915 lines)
            test.csv.zip (273.0 kB)
            train.csv (55423857 lines)
            train.csv.zip (1.6 GB)
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
        working/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
```

-> data/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/new-york-city-taxi-fare-prediction/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/train.csv has 55423856 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/train.csv has 55423856 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import math
import numpy as np
import pandas as pd

from scipy import stats as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

plt.style.use("seaborn-whitegrid")

TRAIN_PATH = "../input/new-york-city-taxi-fare-prediction/train.csv"
TEST_PATH = "../input/new-york-city-taxi-fare-prediction/test.csv"
SAMPLE_SUB_PATH = "../input/new-york-city-taxi-fare-prediction/sample_submission.csv"

NROWS = 2_000_000

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

FARE_MIN = 2.0
FARE_MAX = 100.0




## === cell 1
def load_random_sample_csv_reservoir(path, nrows_sample, seed=42, chunksize=200_000):
    rng = np.random.RandomState(seed)
    reservoir = None
    seen_total = 0  # total number of rows processed so far (global stream index)

    for chunk in pd.read_csv(path, low_memory=True, chunksize=chunksize):
        chunk = chunk.reset_index(drop=True)
        m = len(chunk)
        if m == 0:
            continue

        if reservoir is None:
            reservoir = chunk.iloc[:0].copy()

        if len(reservoir) < nrows_sample:
            take = min(nrows_sample - len(reservoir), m)
            reservoir = pd.concat(
                [reservoir, chunk.iloc[:take]], axis=0, ignore_index=True
            )

            seen_before_chunk = seen_total
            seen_total += m

            remainder = chunk.iloc[take:].reset_index(drop=True)
            m_rem = len(remainder)
            if m_rem > 0:
                k = nrows_sample
                t_indices = np.arange(
                    seen_before_chunk + take + 1,
                    seen_before_chunk + m + 1,
                    dtype=np.int64,
                )
                j = rng.randint(0, t_indices)
                mask = j < k
                if np.any(mask):
                    replace_pos = j[mask].astype(np.int64)
                    reservoir.iloc[replace_pos] = remainder.loc[mask].to_numpy()
            continue  # important: we've already handled seen_total update for this chunk

        k = nrows_sample
        t_indices = np.arange(seen_total + 1, seen_total + m + 1, dtype=np.int64)
        j = rng.randint(0, t_indices)
        mask = j < k
        if np.any(mask):
            replace_pos = j[mask].astype(np.int64)
            reservoir.iloc[replace_pos] = chunk.loc[mask].to_numpy()

        seen_total += m  # count all processed rows

    if reservoir is None or len(reservoir) == 0:
        raise RuntimeError("Failed to load any rows from training CSV.")

    reservoir = reservoir.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return reservoir


df = load_random_sample_csv_reservoir(
    TRAIN_PATH, NROWS, seed=RANDOM_STATE, chunksize=200_000
)
df["pickup_datetime"] = df["pickup_datetime"].astype(str)

print("Train loaded+sampled shape:", df.shape)
print(df.head(2))



## === cell 2
df = df[df.passenger_count.between(1, 6)]

for c in ["pickup_longitude", "dropoff_longitude"]:
    df = df[df[c].between(-180, 180)]
for c in ["pickup_latitude", "dropoff_latitude"]:
    df = df[df[c].between(-90, 90)]

df = df[
    (df.dropoff_latitude != 0)
    & (df.pickup_longitude != 0)
    & (df.pickup_latitude != 0)
    & (df.dropoff_longitude != 0)
]

df = df[df.fare_amount > FARE_MIN]
df = df[df.fare_amount < FARE_MAX]

df = df.dropna()

dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
valid_dt = dt.notna()
df = df.loc[valid_dt].copy()
dt = dt.loc[valid_dt]

df["year"] = dt.dt.year
df["hour"] = dt.dt.hour
df["month"] = dt.dt.month
df["dow"] = dt.dt.dayofweek

df = df.dropna(subset=["year", "hour", "month", "dow"])
df[["year", "hour", "month", "dow"]] = df[["year", "hour", "month", "dow"]].astype(int)

print("After basic cleaning shape:", df.shape)
print(df[["pickup_datetime", "year", "hour", "month", "dow"]].head(3))




## === cell 3
def select_within_newYork(df_in, loc):
    return (
        (df_in.pickup_longitude >= loc[0])
        & (df_in.pickup_longitude <= loc[1])
        & (df_in.pickup_latitude >= loc[2])
        & (df_in.pickup_latitude <= loc[3])
        & (df_in.dropoff_longitude >= loc[0])
        & (df_in.dropoff_longitude <= loc[1])
        & (df_in.dropoff_latitude >= loc[2])
        & (df_in.dropoff_latitude <= loc[3])
        & (df_in.dropoff_latitude >= loc[2])
        & (df_in.dropoff_latitude <= loc[3])
    )


NYC = (-74.5, -72.8, 40.5, 41.8)
df = df[select_within_newYork(df, NYC)]

print("After NYC bounding box shape:", df.shape)




## === cell 4
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    a = (
        np.sin((lat2 - lat1) / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    )
    return 6367 * 2 * np.arcsin(np.sqrt(a)) * 0.62137


df["distance"] = haversine_np(
    df.pickup_longitude, df.pickup_latitude, df.dropoff_longitude, df.dropoff_latitude
)

print(df[["distance"]].describe())



## === cell 5
print("Co-relation b/w Fare and Distance")
print(st.pearsonr(df.distance, df.fare_amount))

df = df[df.distance <= 30]

print("After distance cutoff shape:", df.shape)



## === cell 6
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
con = (
    (df.distance < 30)
    & (df.distance > 0.5)
    & (df.fare_amount > 0)
    & (df.fare_amount < 200)
)
axs[0].scatter(df.loc[con, "fare_amount"], df.loc[con, "distance"], alpha=0.3)
axs[0].set_xlabel("Fare")
axs[0].set_ylabel("Distance")
axs[0].set_title("Distance vs Fare")
plt.close(fig)



## === cell 7
df["fare-bin"] = pd.cut(df["fare_amount"], bins=list(range(0, 50, 5))).astype(str)
df.loc[df["fare-bin"] == "nan", "fare-bin"] = "[45+]"
df.loc[df["fare-bin"] == "(5, 10]", "fare-bin"] = "(05, 10]"

df["diff_long"] = (df.dropoff_longitude - df.pickup_longitude).abs()
df["diff_lat"] = (df.dropoff_latitude - df.pickup_latitude).abs()

df["abs_pickup_longitude"] = df["pickup_longitude"].abs()
df["abs_dropoff_longitude"] = df["dropoff_longitude"].abs()
df["abs_pickup_latitude"] = df["pickup_latitude"].abs()
df["abs_dropoff_latitude"] = df["dropoff_latitude"].abs()
df["manhattan_dist"] = df["diff_lat"] + df["diff_long"]

print(df[["diff_lat", "diff_long", "distance", "manhattan_dist"]].head(3))



## === cell 8
print("corelation b/w Distance and Time of Day")
print(st.pearsonr(df.distance, df.hour))

print("corelation b/w Fare and Time of Day")
print(st.pearsonr(df.fare_amount, df.hour))



## === cell 9
test = pd.read_csv(TEST_PATH, low_memory=True)
test["pickup_datetime"] = test["pickup_datetime"].astype(str)

test["diff_lat"] = (test.dropoff_latitude - test.pickup_latitude).abs()
test["diff_long"] = (test.dropoff_longitude - test.pickup_longitude).abs()
test["distance"] = haversine_np(
    test.pickup_longitude,
    test.pickup_latitude,
    test.dropoff_longitude,
    test.dropoff_latitude,
)

test["abs_pickup_longitude"] = test["pickup_longitude"].abs()
test["abs_dropoff_longitude"] = test["dropoff_longitude"].abs()
test["abs_pickup_latitude"] = test["pickup_latitude"].abs()
test["abs_dropoff_latitude"] = test["dropoff_latitude"].abs()
test["manhattan_dist"] = test["diff_lat"] + test["diff_long"]

dt_t = pd.to_datetime(test["pickup_datetime"], errors="coerce", utc=True)
test["year"] = dt_t.dt.year
test["hour"] = dt_t.dt.hour
test["month"] = dt_t.dt.month
test["dow"] = dt_t.dt.dayofweek

for c, fillv in [("year", 2015), ("hour", 0), ("month", 1), ("dow", 0)]:
    test[c] = test[c].fillna(fillv).astype(int)

test_id = list(test["key"].values)

print("Test loaded shape:", test.shape)
print(test.head(2))



## === cell 10
lr = LinearRegression()

lr_features = [
    "diff_lat",
    "diff_long",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "distance",
    "passenger_count",
    "year",
    "hour",
    "month",
    "dow",
]

X_lr_train = df[lr_features].apply(pd.to_numeric, errors="coerce")
y_lr_train = pd.to_numeric(df["fare_amount"], errors="coerce")
m = np.isfinite(X_lr_train.to_numpy()).all(axis=1) & np.isfinite(y_lr_train.to_numpy())
X_lr_train = X_lr_train.loc[m]
y_lr_train = y_lr_train.loc[m]

p = lr.fit(X_lr_train, np.log1p(y_lr_train))

print("Intercept", round(lr.intercept_, 4))
print(
    "Lat diff coef: ",
    round(lr.coef_[0], 4),
    "\tLong diff coef:",
    round(lr.coef_[1], 4),
    "\t Pikcup Latitude  coef",
    round(lr.coef_[2], 4),
    "\t Pikcup Longitude  coef",
    round(lr.coef_[3], 4),
    "\t Dropoff Latitude  coef",
    round(lr.coef_[4], 4),
    "\t Dropoff Longitude  coef",
    round(lr.coef_[5], 4),
    "\tDistance coef:",
    round(lr.coef_[6], 4),
)



## === cell 11
X_lr_test = test[lr_features].apply(pd.to_numeric, errors="coerce").fillna(0.0)

preds_lr = np.expm1(lr.predict(X_lr_test))
preds_lr = np.clip(preds_lr, FARE_MIN, FARE_MAX)

sub_lr = pd.DataFrame({"key": test_id, "fare_amount": preds_lr})
sub_lr.to_csv("output_lr.csv", index=False)

print("Wrote output_lr.csv with shape:", sub_lr.shape)
print(sub_lr.head(2))



## === cell 12
X = df.drop(
    ["key", "fare_amount", "pickup_datetime", "fare-bin"], axis=1, errors="ignore"
)
y = df["fare_amount"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

lr_eval = LinearRegression()

X_tr = X_train[lr_features].apply(pd.to_numeric, errors="coerce")
X_va = X_val[lr_features].apply(pd.to_numeric, errors="coerce")
y_tr = pd.to_numeric(y_train, errors="coerce")
y_va = pd.to_numeric(y_val, errors="coerce")

mtr = np.isfinite(X_tr.to_numpy()).all(axis=1) & np.isfinite(y_tr.to_numpy())
mva = np.isfinite(X_va.to_numpy()).all(axis=1) & np.isfinite(y_va.to_numpy())

lr_eval.fit(X_tr.loc[mtr], np.log1p(y_tr.loc[mtr]))
y_val_pred = np.expm1(lr_eval.predict(X_va.loc[mva]))
y_val_pred = np.clip(y_val_pred, FARE_MIN, FARE_MAX)

lrmse = np.sqrt(metrics.mean_squared_error(y_va.loc[mva], y_val_pred))
print("LinearRegression (log1p target) validation RMSE:", lrmse)



## === cell 13
random_forest = rf(
    n_estimators=220,
    max_depth=16,
    min_samples_leaf=2,
    max_features="sqrt",
    oob_score=True,
    bootstrap=True,
    verbose=0,
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

rf_features = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "distance",
    "diff_lat",
    "diff_long",
    "manhattan_dist",
    "abs_pickup_latitude",
    "abs_dropoff_latitude",
    "abs_pickup_longitude",
    "abs_dropoff_longitude",
    "passenger_count",
    "year",
    "hour",
    "month",
    "dow",
]

X_rf_train = df[rf_features].apply(pd.to_numeric, errors="coerce")
y_rf_train = pd.to_numeric(df["fare_amount"], errors="coerce")
mr = np.isfinite(X_rf_train.to_numpy()).all(axis=1) & np.isfinite(y_rf_train.to_numpy())
X_rf_train = X_rf_train.loc[mr].fillna(0.0)
y_rf_train = y_rf_train.loc[mr]

random_forest.fit(X_rf_train, np.log1p(y_rf_train))

print("RandomForest fitted. OOB score available:", hasattr(random_forest, "oob_score_"))
if hasattr(random_forest, "oob_score_"):
    print("RandomForest oob_score_:", random_forest.oob_score_)



## === cell 14
X_va_rf = X_val[rf_features].apply(pd.to_numeric, errors="coerce")
X_va_lr = X_val[lr_features].apply(pd.to_numeric, errors="coerce")
X_tr_lr = X_train[lr_features].apply(pd.to_numeric, errors="coerce")
y_tr2 = pd.to_numeric(y_train, errors="coerce")
y_va2 = pd.to_numeric(y_val, errors="coerce")

mva_rf = np.isfinite(X_va_rf.to_numpy()).all(axis=1) & np.isfinite(y_va2.to_numpy())
mtr_lr = np.isfinite(X_tr_lr.to_numpy()).all(axis=1) & np.isfinite(y_tr2.to_numpy())
mva_lr = np.isfinite(X_va_lr.to_numpy()).all(axis=1) & np.isfinite(y_va2.to_numpy())

rf_val_pred = np.expm1(random_forest.predict(X_va_rf.loc[mva_rf].fillna(0.0)))
rf_val_pred = np.clip(rf_val_pred, FARE_MIN, FARE_MAX)

lr_val_for_blend = LinearRegression()
lr_val_for_blend.fit(X_tr_lr.loc[mtr_lr], np.log1p(y_tr2.loc[mtr_lr]))
lr_val_pred = np.expm1(lr_val_for_blend.predict(X_va_lr.loc[mva_lr].fillna(0.0)))
lr_val_pred = np.clip(lr_val_pred, FARE_MIN, FARE_MAX)

valid_idx = X_val.index[mva_rf & mva_lr]
rf_val_pred_al = np.expm1(random_forest.predict(X_va_rf.loc[valid_idx].fillna(0.0)))
rf_val_pred_al = np.clip(rf_val_pred_al, FARE_MIN, FARE_MAX)
lr_val_pred_al = np.expm1(lr_val_for_blend.predict(X_va_lr.loc[valid_idx].fillna(0.0)))
lr_val_pred_al = np.clip(lr_val_pred_al, FARE_MIN, FARE_MAX)
y_val_al = y_va2.loc[valid_idx]

alphas = np.linspace(0.0, 1.0, 11)
best_alpha = 0.85
best_rmse = float("inf")
for a in alphas:
    blend = a * rf_val_pred_al + (1.0 - a) * lr_val_pred_al
    blend = np.clip(blend, FARE_MIN, FARE_MAX)
    rmse = np.sqrt(metrics.mean_squared_error(y_val_al, blend))
    if rmse < best_rmse:
        best_rmse = rmse
        best_alpha = float(a)

print(
    "Chosen BLEND_ALPHA from validation grid:", best_alpha, "with val RMSE:", best_rmse
)

predictedFare = np.array(preds_lr, copy=True)  # baseline for all rows

X_test_rf = test[rf_features].apply(pd.to_numeric, errors="coerce")

ok_numeric = np.isfinite(X_test_rf.to_numpy()).all(axis=1)
dist_finite = np.isfinite(pd.to_numeric(test["distance"], errors="coerce").to_numpy())
ok = ok_numeric & dist_finite

rf_rows_idx = X_test_rf.index[ok]

pred_rf_part = np.expm1(random_forest.predict(X_test_rf.loc[rf_rows_idx].fillna(0.0)))
pred_rf_part = np.clip(pred_rf_part, FARE_MIN, FARE_MAX)

BLEND_ALPHA = best_alpha  # calibrated weight on RF
pred_lr_part = preds_lr[rf_rows_idx.to_numpy()]
pred_blend_part = BLEND_ALPHA * pred_rf_part + (1.0 - BLEND_ALPHA) * pred_lr_part
pred_blend_part = np.clip(pred_blend_part, FARE_MIN, FARE_MAX)

predictedFare[rf_rows_idx.to_numpy()] = pred_blend_part
predictedFare = np.clip(predictedFare, FARE_MIN, FARE_MAX)

sub_rf = pd.DataFrame({"key": test_id, "fare_amount": predictedFare})
sub_rf.to_csv("output_rf.csv", index=False)

print("Wrote output_rf.csv with shape:", sub_rf.shape)
print("RF+blend used on rows:", int(len(rf_rows_idx)), "out of", test.shape[0])
print(sub_rf.head(2))



## === cell 15
submission = sub_rf.copy()
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)

sample = pd.read_csv(SAMPLE_SUB_PATH)
assert list(submission.columns) == list(
    sample.columns
), "Submission columns do not match sample_submission.csv"
assert (
    submission.shape[0] == sample.shape[0]
), "Submission row count does not match sample_submission.csv"
print("Submission format validated vs sample_submission.csv")
