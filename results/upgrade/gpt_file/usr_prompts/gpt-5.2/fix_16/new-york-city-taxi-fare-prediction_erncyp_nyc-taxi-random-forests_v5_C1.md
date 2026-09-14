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
seaborn==0.12.2
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
import os
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42


def _resolve_path(*candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


TRAIN_PATH = _resolve_path(
    "/kaggle/input/train.csv",
    "/kaggle/data/train.csv",
    "../input/train.csv",
)
TEST_PATH = _resolve_path(
    "/kaggle/input/test.csv",
    "/kaggle/data/test.csv",
    "../input/test.csv",
)
SAMPLE_SUB_PATH = _resolve_path(
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "../input/sample_submission.csv",
)

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

np.random.seed(RANDOM_STATE)



## === cell 1
read_rows = 2_000_000
usecols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtypes = {
    "fare_amount": "float32",
    "pickup_datetime": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}

try:
    train_df = pd.read_csv(
        TRAIN_PATH,
        nrows=read_rows,
        usecols=usecols,
        dtype=dtypes,
        engine="pyarrow",
        memory_map=True,
    )
except Exception:
    train_df = pd.read_csv(
        TRAIN_PATH,
        nrows=read_rows,
        usecols=usecols,
        dtype=dtypes,
        engine="c",
        memory_map=True,
    )




## === cell 2
def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees)
    All args must be of equal length.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km




## === cell 3
plon = train_df["pickup_longitude"].to_numpy(copy=False)
plat = train_df["pickup_latitude"].to_numpy(copy=False)
dlon = train_df["dropoff_longitude"].to_numpy(copy=False)
dlat = train_df["dropoff_latitude"].to_numpy(copy=False)

train_df["distance"] = haversine_np(plon, plat, dlon, dlat).astype(
    np.float32, copy=False
)



## === cell 4
train_df["pickup_datetime"] = pd.to_datetime(
    train_df["pickup_datetime"], errors="coerce", cache=True
)



## === cell 5
dt = train_df["pickup_datetime"].dt
train_df["year"] = dt.year.astype("int16", copy=False)
train_df["month"] = dt.month.astype("int8", copy=False)
train_df["day"] = dt.day.astype("int8", copy=False)
train_df["hour"] = dt.hour.astype("int8", copy=False)
train_df["minute"] = dt.minute.astype("int8", copy=False)



## === cell 6
print("Old size: %d" % len(train_df))

dt_notna = train_df["pickup_datetime"].notna().to_numpy()

plon = train_df["pickup_longitude"].to_numpy(copy=False)
plat = train_df["pickup_latitude"].to_numpy(copy=False)
dlon = train_df["dropoff_longitude"].to_numpy(copy=False)
dlat = train_df["dropoff_latitude"].to_numpy(copy=False)
dist = train_df["distance"].to_numpy(copy=False)
fare = train_df["fare_amount"].to_numpy(copy=False)
pc = train_df["passenger_count"].to_numpy(copy=False)

lon_min, lon_max = -74.3, -73.7
lat_min, lat_max = 40.5, 41.0
bbox_mask = (
    (plon >= lon_min)
    & (plon <= lon_max)
    & (dlon >= lon_min)
    & (dlon <= lon_max)
    & (plat >= lat_min)
    & (plat <= lat_max)
    & (dlat >= lat_min)
    & (dlat <= lat_max)
)

same_point = (plon == dlon) & (plat == dlat)
has_zero_coord = (plon == 0.0) | (plat == 0.0) | (dlon == 0.0) | (dlat == 0.0)

fare_per_km = fare / (dist + 1e-6)

basic_mask = (
    (fare >= 2.5)
    & (fare <= 250.0)
    & (dist > 0.01)
    & (dist <= 80.0)
    & (pc >= 1)
    & (pc <= 6)
    & (~same_point)
    & (~has_zero_coord)
    & (fare_per_km >= 0.5)
    & (fare_per_km <= 50.0)
)

max_reasonable = (3.0 + 12.0 * dist + 3.0 * np.sqrt(np.maximum(dist, 0.0))).astype(
    np.float32, copy=False
)
min_reasonable = (2.5 + 0.2 * dist).astype(np.float32, copy=False)
consistency_mask = (fare >= min_reasonable) & (
    fare <= np.minimum(250.0, max_reasonable)
)

final_mask = dt_notna & bbox_mask & basic_mask & consistency_mask
train_df = train_df.loc[final_mask].reset_index(drop=True)

print("New size: %d" % len(train_df))
print("After fare-distance consistency filter size: %d" % len(train_df))



## === cell 7
new_york_lat = 40
new_york_long = -74



## === cell 8
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



## === cell 9
regr = LinearRegression()
regr_quad = LinearRegression()



## === cell 10
X = np.ascontiguousarray(train_df[["distance"]].to_numpy(copy=False), dtype=np.float32)
Y = np.ascontiguousarray(train_df["fare_amount"].to_numpy(copy=False), dtype=np.float32)



## === cell 11
regr.fit(X, Y)



## === cell 12
X2 = np.ascontiguousarray(X * X, dtype=np.float32)
regr_quad.fit(X2, Y)



## === cell 13
y_pred = regr.predict(X)
print("chi squared linear %s" % (np.sum((Y - y_pred) ** 2.0) / len(Y)) ** 0.5)

y_pred_quad = regr_quad.predict(X2)
print("chi squared quadratic %s" % (np.sum((Y - y_pred_quad) ** 2.0) / len(Y)) ** 0.5)



## === cell 14
regr_more = LinearRegression()
X = np.ascontiguousarray(
    train_df[["distance", "year", "month", "day", "hour"]].to_numpy(copy=False),
    dtype=np.float32,
)
Y = np.ascontiguousarray(train_df["fare_amount"].to_numpy(copy=False), dtype=np.float32)



## === cell 15
regr_more.fit(X, Y)
y_pred = regr_more.predict(X)
print("chi squared linear with date %s" % (np.sum((Y - y_pred) ** 2.0) / len(Y)) ** 0.5)



## === cell 16
from sklearn.ensemble import RandomForestRegressor



## === cell 17
import joblib

MODEL_CACHE_PATH = "rf_ensemble_models.joblib"
DATA_CACHE_PATH = "train_arrays_cache.joblib"

rf_kwargs = dict(
    n_estimators=500,
    n_jobs=-1,
    min_samples_leaf=3,
    min_samples_split=6,
    max_features="sqrt",
    bootstrap=True,
    max_depth=24,
)

ENSEMBLE_SEEDS = [RANDOM_STATE, RANDOM_STATE + 1, RANDOM_STATE + 2]


def _load_cached_models(path):
    if not os.path.exists(path):
        return None
    try:
        models = joblib.load(path)
        if (
            isinstance(models, list)
            and len(models) == len(ENSEMBLE_SEEDS)
            and all(hasattr(m, "predict") for m in models)
        ):
            return models
    except Exception:
        return None
    return None


def _fit_or_resume_rf(m, X, Y, target_estimators):
    cur = getattr(m, "n_estimators", None)
    if cur is None:
        cur = target_estimators
    if cur >= target_estimators:
        if cur != target_estimators:
            m.n_estimators = target_estimators
        return m
    m.warm_start = True
    m.n_estimators = target_estimators
    m.fit(X, Y)
    return m


def _load_cached_arrays(path):
    if not os.path.exists(path):
        return None
    try:
        payload = joblib.load(path)
        if not isinstance(payload, dict):
            return None
        Xc = payload.get("X", None)
        Yc = payload.get("Y", None)
        meta = payload.get("meta", None)
        if Xc is None or Yc is None or meta is None:
            return None
        if meta.get("read_rows") != read_rows:
            return None
        if meta.get("usecols") != tuple(usecols):
            return None
        if meta.get("rf_feature_cols") != ("distance", "year", "month", "day", "hour"):
            return None
        return Xc, Yc
    except Exception:
        return None


cached = _load_cached_arrays(DATA_CACHE_PATH)
if cached is None:
    X = np.ascontiguousarray(X, dtype=np.float32)
    Y = np.ascontiguousarray(Y, dtype=np.float32)
    joblib.dump(
        {
            "X": X,
            "Y": Y,
            "meta": {
                "read_rows": read_rows,
                "usecols": tuple(usecols),
                "rf_feature_cols": ("distance", "year", "month", "day", "hour"),
            },
        },
        DATA_CACHE_PATH,
        compress=3,
    )
else:
    X, Y = cached
    X = np.ascontiguousarray(X, dtype=np.float32)
    Y = np.ascontiguousarray(Y, dtype=np.float32)

rf_models = _load_cached_models(MODEL_CACHE_PATH)
if rf_models is None:
    rf_models = []
    for seed in ENSEMBLE_SEEDS:
        m = RandomForestRegressor(random_state=seed, **rf_kwargs)
        m.fit(X, Y)
        rf_models.append(m)
    joblib.dump(rf_models, MODEL_CACHE_PATH, compress=3)
else:
    updated = False
    for i, seed in enumerate(ENSEMBLE_SEEDS):
        m = rf_models[i]
        ok = True
        for k, v in rf_kwargs.items():
            if getattr(m, k, None) != v:
                ok = False
                break
        if getattr(m, "random_state", None) != seed:
            ok = False
        if not ok:
            m = RandomForestRegressor(random_state=seed, **rf_kwargs)
            m.fit(X, Y)
            rf_models[i] = m
            updated = True
            continue

        cur_estimators = getattr(m, "n_estimators", 0)
        if cur_estimators < rf_kwargs["n_estimators"]:
            rf_models[i] = _fit_or_resume_rf(m, X, Y, rf_kwargs["n_estimators"])
            updated = True
    if updated:
        joblib.dump(rf_models, MODEL_CACHE_PATH, compress=3)



## === cell 18
test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_dtypes = {
    "key": "object",
    "pickup_datetime": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}
try:
    test_df = pd.read_csv(
        TEST_PATH,
        usecols=test_usecols,
        dtype=test_dtypes,
        engine="pyarrow",
        memory_map=True,
    )
except Exception:
    test_df = pd.read_csv(
        TEST_PATH,
        usecols=test_usecols,
        dtype=test_dtypes,
        engine="c",
        memory_map=True,
    )



## === cell 19
_LON_MIN, _LON_MAX = -74.3, -73.7
_LAT_MIN, _LAT_MAX = 40.5, 41.0

test_df["pickup_longitude"] = test_df["pickup_longitude"].clip(_LON_MIN, _LON_MAX)
test_df["dropoff_longitude"] = test_df["dropoff_longitude"].clip(_LON_MIN, _LON_MAX)
test_df["pickup_latitude"] = test_df["pickup_latitude"].clip(_LAT_MIN, _LAT_MAX)
test_df["dropoff_latitude"] = test_df["dropoff_latitude"].clip(_LAT_MIN, _LAT_MAX)



## === cell 20
test_df["distance"] = haversine_np(
    test_df["pickup_longitude"].to_numpy(copy=False),
    test_df["pickup_latitude"].to_numpy(copy=False),
    test_df["dropoff_longitude"].to_numpy(copy=False),
    test_df["dropoff_latitude"].to_numpy(copy=False),
).astype(np.float32, copy=False)



## === cell 21
test_df["distance"] = test_df["distance"].clip(0.01, 80.0)



## === cell 22
test_df["pickup_datetime"] = pd.to_datetime(
    test_df["pickup_datetime"], errors="coerce", cache=True
)



## === cell 23
dt = test_df["pickup_datetime"].dt
test_df["year"] = dt.year.astype("int16", copy=False)
test_df["month"] = dt.month.astype("int8", copy=False)
test_df["day"] = dt.day.astype("int8", copy=False)
test_df["hour"] = dt.hour.astype("int8", copy=False)
test_df["minute"] = dt.minute.astype("int8", copy=False)



## === cell 24
X_to_pred = np.ascontiguousarray(
    test_df[["distance", "year", "month", "day", "hour"]].to_numpy(copy=False),
    dtype=np.float32,
)

y_sum = np.zeros(X_to_pred.shape[0], dtype=np.float64)
for m in rf_models:
    y_sum += m.predict(X_to_pred)
y_pred = (y_sum / float(len(rf_models))).astype(np.float64, copy=False)

if np.any(~np.isfinite(y_pred)):
    med = np.nanmedian(y_pred[np.isfinite(y_pred)])
    y_pred[~np.isfinite(y_pred)] = med

y_pred = np.clip(y_pred, 0.0, None)



## === cell 25
submission = pd.DataFrame(
    {"key": test_df.key, "fare_amount": y_pred}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)



## === cell 26
if False:
    X_plot = train_df[["distance"]].values.flatten()
    Y_plot = train_df["fare_amount"].values

    with sns.axes_style("white"):
        sns.jointplot(x=X_plot, y=Y_plot, kind="hex", color="k", bins="log")



## === cell 27
if False:
    X_plot = train_df[["distance"]].values.flatten()
    Y_plot = train_df["fare_amount"].values
    mask = (X_plot < 50) & (Y_plot < 100)

    with sns.axes_style("white"):
        p = sns.jointplot(
            x=X_plot[mask], y=Y_plot[mask], kind="hex", color="k", bins="log"
        )

    x = np.arange(0, 50)
    y = regr.predict(x.reshape(-1, 1))
    p.ax_joint.plot(x, y)
