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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sklearn-pandas==2.2.0

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
from sklearn.impute import SimpleImputer as Imputer
import numpy as np
import pandas as pd
import os

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

pd.options.mode.chained_assignment = None
np.random.seed(21)

print(os.listdir("../input"))


## === cell 1
train_path = "../input/train.csv"
usecols_train = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtypes_train = {
    "key": "string",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "float32",
}

df = pd.read_csv(train_path, nrows=10_00_000, usecols=usecols_train, dtype=dtypes_train)
df.head()


## === cell 2
m = (df["passenger_count"] > 0) & (df["fare_amount"] > 0)
df = df.loc[m]
df.head()


## === cell 3
alpha_ang = 0.506


def distance_travel(df):
    plon = df["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    plat = df["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
    dlon = df["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
    dlat = df["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)

    abs_diff_longitude = np.abs(dlon - plon) * 50.0
    abs_diff_latitude = np.abs(dlat - plat) * 69.0
    dv = np.sqrt(
        abs_diff_latitude * abs_diff_latitude + abs_diff_longitude * abs_diff_longitude
    )

    angle = np.arctan2(abs_diff_longitude, abs_diff_latitude) - alpha_ang
    actual_long = np.abs(dv * np.sin(angle))
    actual_lat = np.abs(dv * np.cos(angle))
    df["distance_travel"] = (actual_long + actual_lat).astype("float32", copy=False)


def clean_coords(df):
    plon = df["pickup_longitude"].to_numpy(copy=False)
    plat = df["pickup_latitude"].to_numpy(copy=False)
    dlon = df["dropoff_longitude"].to_numpy(copy=False)
    dlat = df["dropoff_latitude"].to_numpy(copy=False)
    pc = df["passenger_count"].to_numpy(copy=False)

    finite_mask = (
        np.isfinite(plon)
        & np.isfinite(plat)
        & np.isfinite(dlon)
        & np.isfinite(dlat)
        & np.isfinite(pc)
    )
    if not finite_mask.all():
        df = df.loc[finite_mask]

        plon = df["pickup_longitude"].to_numpy(copy=False)
        plat = df["pickup_latitude"].to_numpy(copy=False)
        dlon = df["dropoff_longitude"].to_numpy(copy=False)
        dlat = df["dropoff_latitude"].to_numpy(copy=False)

    m = (
        (plon >= -74.3)
        & (plon <= -73.7)
        & (dlon >= -74.3)
        & (dlon <= -73.7)
        & (plat >= 40.5)
        & (plat <= 41.0)
        & (dlat >= 40.5)
        & (dlat <= 41.0)
    )
    return df.loc[m]


def repair_coords_no_drop(df):
    coord_cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    need_copy = False
    for c in coord_cols:
        if not np.issubdtype(df[c].dtype, np.number):
            need_copy = True
            break
    if need_copy:
        df = df.copy()

    for c in coord_cols:
        if not np.issubdtype(df[c].dtype, np.number):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    pc = df["passenger_count"].to_numpy(copy=False)
    pc_bad = ~np.isfinite(pc) | (pc <= 0)
    if pc_bad.any():
        df.loc[pc_bad, "passenger_count"] = 1.0

    bounds = {
        "pickup_longitude": (-74.3, -73.7),
        "dropoff_longitude": (-74.3, -73.7),
        "pickup_latitude": (40.5, 41.0),
        "dropoff_latitude": (40.5, 41.0),
    }
    for c, (lo, hi) in bounds.items():
        vals = df[c].to_numpy(copy=False)
        bad = ~np.isfinite(vals)
        if bad.any():
            med = float(np.nanmedian(vals))
            if not np.isfinite(med):
                med = (lo + hi) / 2.0
            df.loc[bad, c] = med
        df[c] = df[c].clip(lo, hi)

    return df


def add_time_features(df):
    s = df["pickup_datetime"]
    dt = pd.to_datetime(s, errors="coerce", utc=True, cache=True)

    hr = dt.dt.hour.astype("float32")
    if hr.isna().any():
        med = float(np.nanmedian(hr.to_numpy()))
        if not np.isfinite(med):
            med = 12.0
        hr = hr.fillna(med)
    df["pickup_hour"] = hr
    return df


df = repair_coords_no_drop(df)
df = add_time_features(df)
df = clean_coords(df)

distance_travel(df)
dist_tr = df["distance_travel"].to_numpy(copy=False)
bad_dist_tr = ~np.isfinite(dist_tr) | (dist_tr <= 0)
if bad_dist_tr.any():
    good = np.isfinite(dist_tr) & (dist_tr > 0)
    fill_val = float(np.nanmedian(dist_tr[good]) if np.any(good) else 0.1)
    df.loc[bad_dist_tr, "distance_travel"] = fill_val

dist_tr = df["distance_travel"].to_numpy(copy=False)
m = np.isfinite(dist_tr) & (dist_tr > 0)
df = df.loc[m]
df.head()


## === cell 4
pass


## === cell 5
TRAIN_MAX_DISTANCE = 30.0
m = (df["distance_travel"] < TRAIN_MAX_DISTANCE) & (df["fare_amount"] < 100)
df = df.loc[m]
pass


## === cell 6
l = len(df)
print(l)

rng = np.random.RandomState(21)
split_mask = rng.rand(l) < 0.7  # deterministic with the fixed seed
df_train = df.loc[split_mask]
df_test = df.loc[~split_mask]


def build_X(dfx):
    n = len(dfx)
    X = np.empty((n, 4), dtype=np.float32)
    X[:, 0] = dfx["distance_travel"].to_numpy(dtype=np.float32, copy=False)
    X[:, 1] = dfx["passenger_count"].to_numpy(dtype=np.float32, copy=False)
    X[:, 2] = dfx["pickup_hour"].to_numpy(dtype=np.float32, copy=False)
    X[:, 3] = 1.0
    return X


train_X = build_X(df_train)
test_X = build_X(df_test)

train_y = df_train.fare_amount.to_numpy(dtype=np.float32, copy=False)
test_y = df_test.fare_amount.to_numpy(dtype=np.float32, copy=False)


## === cell 7
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

imp = Imputer(missing_values=np.nan, strategy="mean")
imp = imp.fit(train_X)
train_X = imp.transform(train_X)
test_X = imp.transform(test_X)

train_y_log = np.log1p(train_y)
test_y_log = np.log1p(test_y)

regr = GradientBoostingRegressor(random_state=21, n_estimators=400)
regr.fit(train_X, train_y_log)

pred_log = regr.predict(test_X)
pred = np.expm1(pred_log)
pred = np.maximum(pred, 0.0)
rmse = mean_squared_error(test_y, pred, squared=False)
print("Holdout RMSE:", rmse)


def _process_train_chunk(chunk):
    m0 = (chunk["passenger_count"] > 0) & (chunk["fare_amount"] > 0)
    chunk = chunk.loc[m0]
    if len(chunk) == 0:
        return None, None

    chunk = repair_coords_no_drop(chunk)
    chunk = add_time_features(chunk)
    chunk = clean_coords(chunk)
    if len(chunk) == 0:
        return None, None

    distance_travel(chunk)

    dist_tr = chunk["distance_travel"].to_numpy(copy=False)
    bad_dist_tr = ~np.isfinite(dist_tr) | (dist_tr <= 0)
    if bad_dist_tr.any():
        good = np.isfinite(dist_tr) & (dist_tr > 0)
        fill_val = float(np.nanmedian(dist_tr[good]) if np.any(good) else 0.1)
        chunk.loc[bad_dist_tr, "distance_travel"] = fill_val

    dist_tr = chunk["distance_travel"].to_numpy(copy=False)
    m1 = np.isfinite(dist_tr) & (dist_tr > 0)
    chunk = chunk.loc[m1]
    if len(chunk) == 0:
        return None, None

    m2 = (chunk["distance_travel"] < TRAIN_MAX_DISTANCE) & (chunk["fare_amount"] < 100)
    chunk = chunk.loc[m2]
    if len(chunk) == 0:
        return None, None

    Xc = build_X(chunk)
    yc = chunk["fare_amount"].to_numpy(dtype=np.float32, copy=False)
    return Xc, yc


X_parts = []
y_parts = []

for chunk in pd.read_csv(
    train_path,
    nrows=10_00_000,
    usecols=usecols_train,
    dtype=dtypes_train,
    chunksize=250_000,
):
    Xc, yc = _process_train_chunk(chunk)
    if Xc is None:
        continue
    X_parts.append(Xc)
    y_parts.append(yc)

X_all = np.vstack(X_parts) if len(X_parts) else np.empty((0, 4), dtype=np.float32)
y_all = np.concatenate(y_parts) if len(y_parts) else np.empty((0,), dtype=np.float32)
y_all_log = np.log1p(y_all)

imp = Imputer(missing_values=np.nan, strategy="mean").fit(X_all)
X_all = imp.transform(X_all)

regr.fit(X_all, y_all_log)


## === cell 8
test_path = "../input/test.csv"
usecols_test = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtypes_test = {
    "key": "string",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "float32",
}
tdf = pd.read_csv(test_path, nrows=10_00_000, usecols=usecols_test, dtype=dtypes_test)

tdf = repair_coords_no_drop(tdf)
tdf = add_time_features(tdf)

distance_travel(tdf)

dist = tdf["distance_travel"].to_numpy(copy=False)
bad_dist = ~np.isfinite(dist) | (dist <= 0)
if bad_dist.any():
    good = np.isfinite(dist) & (dist > 0)
    tdf.loc[bad_dist, "distance_travel"] = float(
        np.nanmedian(dist[good]) if np.any(good) else 0.1
    )

tdf["distance_travel"] = tdf["distance_travel"].clip(
    lower=0.0, upper=TRAIN_MAX_DISTANCE
)
tdf.head()


## === cell 9
ttrain_X = build_X(tdf)
ttrain_X = imp.transform(ttrain_X)

output_log = regr.predict(ttrain_X)
output = np.expm1(output_log)

output = np.where(np.isfinite(output), output, np.nan)
if np.isnan(output).any():
    output = np.nan_to_num(output, nan=float(np.nanmean(output)))

output = np.maximum(output, 0.0)

print(output)


## === cell 10
sample = pd.read_csv("../input/sample_submission.csv")

my_submission = pd.DataFrame({"key": tdf["key"].values, "fare_amount": output})
my_submission = sample[["key"]].merge(my_submission, on="key", how="left")

if my_submission["fare_amount"].isna().any():
    my_submission["fare_amount"] = my_submission["fare_amount"].fillna(
        float(np.nanmean(output))
    )

my_submission.to_csv("submission.csv", index=False)
my_submission.head()
