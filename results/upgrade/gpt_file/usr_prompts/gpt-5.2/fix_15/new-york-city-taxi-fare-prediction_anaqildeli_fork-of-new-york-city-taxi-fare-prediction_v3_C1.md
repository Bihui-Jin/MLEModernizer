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

3.10

# 3. Installed packages

folium==0.20.0
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
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("JOBLIB_TEMP_FOLDER", "/kaggle/working")

np.random.seed(42)

try:
    from sklearnex import patch_sklearn

    patch_sklearn()  # must run before importing sklearn estimators for maximum effect
except Exception:
    pass

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



## === cell 1
CACHE_DIR = "/kaggle/working"
CACHE_PATH = os.path.join(CACHE_DIR, "preproc_cache_v1.npz")

fields = [
    "key",
    "pickup_datetime",
    "fare_amount",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

train_dtypes = {
    "fare_amount": "float64",
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int64",
}

train = None
test = None



## === cell 2
TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
NROWS = 1_000_000

try:
    train = pd.read_csv(
        TRAIN_PATH,
        nrows=NROWS,
        usecols=fields,
        parse_dates=["pickup_datetime"],
        dtype=train_dtypes,
        engine="pyarrow",
        dtype_backend="numpy_nullable",
    )
except Exception:
    train = pd.read_csv(
        TRAIN_PATH,
        nrows=NROWS,
        usecols=fields,
        parse_dates=["pickup_datetime"],
        dtype=train_dtypes,
    )

print(f"{train.shape} shape")
train.head()



## === cell 3
_ = train.shape



## === cell 4
TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

test_dtypes = {
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int64",
}

try:
    test = pd.read_csv(
        TEST_PATH,
        parse_dates=["pickup_datetime"],
        dtype=test_dtypes,
        engine="pyarrow",
        dtype_backend="numpy_nullable",
    )
except Exception:
    test = pd.read_csv(
        TEST_PATH,
        parse_dates=["pickup_datetime"],
        dtype=test_dtypes,
    )

print(f"{test.shape} shape")
test.head()



## === cell 5
_ = test.shape



## === cell 6
pass



## === cell 7
pass



## === cell 8
pass



## === cell 9
pass



## === cell 10
pc = train["passenger_count"].to_numpy(copy=False)
fa = train["fare_amount"].to_numpy(copy=False)
m = (pc != 208) & (fa >= 0)
train = train.loc[m]



## === cell 11
tr_dt = train["pickup_datetime"].dt
train["year"] = tr_dt.year.to_numpy()
train["month"] = tr_dt.month.to_numpy()
train["day"] = tr_dt.day.to_numpy()
train["hour"] = tr_dt.hour.to_numpy()
train["minute"] = tr_dt.minute.to_numpy()

te_dt = test["pickup_datetime"].dt
test["year"] = te_dt.year.to_numpy()
test["month"] = te_dt.month.to_numpy()
test["day"] = te_dt.day.to_numpy()
test["hour"] = te_dt.hour.to_numpy()
test["minute"] = te_dt.minute.to_numpy()



## === cell 12
train = train.drop(columns=["pickup_datetime"])
test = test.drop(columns=["pickup_datetime"])



## === cell 13
_ = train.shape



## === cell 14
pass



## === cell 15
train.shape



## === cell 16
pass



## === cell 17
pl = train["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
dl = train["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)
plo = train["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
dlo = train["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)

train_mask = (
    (pl > 40)
    & (pl < 45)
    & (dl > 40)
    & (dl < 45)
    & (plo < -71)
    & (plo > -79)
    & (dlo < -71)
    & (dlo > -79)
)
train = train.loc[train_mask]

tpl = test["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
tdl = test["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)
tplo = test["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
tdlo = test["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)



## === cell 18
_ = train.shape



## === cell 19
train.shape



## === cell 20
pass




## === cell 21
def haversine_np(lon1, lat1, lon2, lat2):
    lon1r = np.radians(lon1)
    lat1r = np.radians(lat1)
    lon2r = np.radians(lon2)
    lat2r = np.radians(lat2)
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


CACHE_META = {
    "cache_version": "v1",
    "nrows": int(NROWS),
    "fields": tuple(fields),
}


def _meta_matches(cached_meta_arr):
    try:
        cached_meta = cached_meta_arr.item()
        return cached_meta == CACHE_META
    except Exception:
        return False


if os.path.exists(CACHE_PATH):
    cached = np.load(CACHE_PATH, allow_pickle=True)
    if ("meta" in cached) and _meta_matches(cached["meta"]):
        X_all = cached["X_all"]
        y = cached["y"]
        X_test_all = cached["X_test_all"]
        test_finite_mask = cached["test_finite_mask"].astype(bool)
        test_keys = cached["test_keys"].astype(object)
        X_cols = cached["X_cols"].tolist()
    else:
        os.remove(CACHE_PATH)
        cached = None

if not os.path.exists(CACHE_PATH):
    tr_plon = train["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    tr_plat = train["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
    tr_dlon = train["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
    tr_dlat = train["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)

    te_plon = test["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    te_plat = test["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
    te_dlon = test["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
    te_dlat = test["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)

    train["distance_km"] = haversine_np(tr_plon, tr_plat, tr_dlon, tr_dlat)
    test["distance_km"] = haversine_np(te_plon, te_plat, te_dlon, te_dlat)

    train["abs_lon_diff"] = np.abs(tr_plon - tr_dlon)
    train["abs_lat_diff"] = np.abs(tr_plat - tr_dlat)
    test["abs_lon_diff"] = np.abs(te_plon - te_dlon)
    test["abs_lat_diff"] = np.abs(te_plat - te_dlat)

    feature_cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "distance_km",
        "abs_lon_diff",
        "abs_lat_diff",
    ]
    target_col = "fare_amount"

    train_feat = train[feature_cols].to_numpy(dtype=np.float64, copy=False)
    train_y = train[target_col].to_numpy(dtype=np.float64, copy=False)
    train_finite_mask = np.isfinite(train_feat).all(axis=1) & np.isfinite(train_y)

    test_feat = test[feature_cols].to_numpy(dtype=np.float64, copy=False)
    test_finite_mask = np.isfinite(test_feat).all(axis=1)

    train = train.loc[train_finite_mask].copy()

    y = train["fare_amount"].to_numpy(dtype=np.float64, copy=False)
    X_df = train.drop(columns=["fare_amount", "key"], errors="ignore")
    X_cols = list(X_df.columns)
    X_all = np.ascontiguousarray(X_df.to_numpy(dtype=np.float64, copy=False))

    X_test_df = test.drop(columns=["key"], errors="ignore").reindex(
        columns=X_cols, fill_value=0
    )
    X_test_all = np.ascontiguousarray(X_test_df.to_numpy(dtype=np.float64, copy=False))

    test_keys = test["key"].values

    np.savez_compressed(
        CACHE_PATH,
        meta=np.array(CACHE_META, dtype=object),
        X_all=X_all,
        y=y,
        X_test_all=X_test_all,
        test_finite_mask=test_finite_mask.astype(np.bool_),
        test_keys=test_keys.astype(object),
        X_cols=np.array(X_cols, dtype=object),
    )

train.shape, test.shape



## === cell 22
from sklearn.ensemble import GradientBoostingRegressor

X_test_np = np.ascontiguousarray(X_test_all[test_finite_mask])

X_train_np, X_valid_np, y_train, y_valid = train_test_split(
    X_all, y, test_size=0.2, random_state=42
)

gbr = GradientBoostingRegressor(
    random_state=42,
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    loss="squared_error",
)

gbr.fit(X_train_np, y_train)

valid_pred = gbr.predict(X_valid_np)
rmse = mean_squared_error(y_valid, valid_pred, squared=False)
print("Validation RMSE:", rmse)



## === cell 23
test_pred_finite = gbr.predict(X_test_np)
test_pred_finite = np.maximum(test_pred_finite, 0)

test_pred_full = np.empty(shape=(len(test_keys),), dtype=np.float64)
if test_finite_mask.all():
    test_pred_full[:] = test_pred_finite
else:
    fill_value = float(np.mean(test_pred_finite)) if len(test_pred_finite) else 0.0
    test_pred_full[:] = fill_value
    test_pred_full[test_finite_mask] = test_pred_finite

submission = pd.DataFrame({"key": test_keys, "fare_amount": test_pred_full})

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
submission.head()
