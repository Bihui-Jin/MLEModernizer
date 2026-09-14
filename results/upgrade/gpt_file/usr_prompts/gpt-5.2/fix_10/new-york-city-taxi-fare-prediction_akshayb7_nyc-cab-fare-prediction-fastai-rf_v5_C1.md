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

fastai==2.8.5
geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
import gc
import math
import random
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from IPython.display import display


def add_datepart(df, fldname, drop=True, time=False, errors="coerce"):
    """
    Adds columns relevant to a date in the column `fldname`.
    Fix: robust to timezone-aware datetimes (e.g. datetime64[ns, UTC]) which break np.issubdtype.
    """
    if fldname not in df.columns:
        raise KeyError(f"{fldname} not found in dataframe columns")

    if not pd.api.types.is_datetime64_any_dtype(df[fldname]):
        df[fldname] = pd.to_datetime(df[fldname], errors=errors, utc=True)
    if pd.api.types.is_datetime64tz_dtype(df[fldname]):
        df[fldname] = df[fldname].dt.tz_convert(None)

    fld = df[fldname]
    prefix = fldname

    attrs = [
        "Year",
        "Month",
        "Week",
        "Day",
        "Dayofweek",
        "Dayofyear",
        "Is_month_end",
        "Is_month_start",
        "Is_quarter_end",
        "Is_quarter_start",
        "Is_year_end",
        "Is_year_start",
    ]
    for attr in attrs:
        if attr == "Week":
            df[prefix + attr] = fld.dt.isocalendar().week.astype("int16")
        else:
            df[prefix + attr] = getattr(fld.dt, attr.lower())

    if time:
        df[prefix + "Hour"] = fld.dt.hour.astype("int16")
        df[prefix + "Minute"] = fld.dt.minute.astype("int16")
        df[prefix + "Second"] = fld.dt.second.astype("int16")

    if drop:
        df.drop(columns=[fldname], inplace=True)
    return df


_old_generate_sample_indices = None


def set_rf_samples(n):
    """
    Limit bootstrap sample size per tree to n (fastai v0 trick).
    """
    global _old_generate_sample_indices
    from sklearn.ensemble import _forest

    if _old_generate_sample_indices is None:
        _old_generate_sample_indices = _forest._generate_sample_indices

    def _generate_sample_indices(random_state, n_samples, n_samples_bootstrap):
        return _old_generate_sample_indices(random_state, n_samples, n)

    _forest._generate_sample_indices = _generate_sample_indices


def reset_rf_samples():
    """
    Restore sklearn's original bootstrap sampling.
    """
    global _old_generate_sample_indices
    if _old_generate_sample_indices is None:
        return
    from sklearn.ensemble import _forest

    _forest._generate_sample_indices = _old_generate_sample_indices
    _old_generate_sample_indices = None


def rf_feat_importance(m, df):
    """
    Return feature importances DataFrame (fastai v0 style).
    """
    return pd.DataFrame(
        {"cols": df.columns, "imp": m.feature_importances_}
    ).sort_values("imp", ascending=False)




## === cell 1
PATH = "/kaggle/input"  # Kaggle canonical path
train_path = f"{PATH}/train.csv"
test_path = f"{PATH}/test.csv"

USE_COLS_TRAIN = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
DTYPES_TRAIN = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}

df_raw = pd.read_csv(
    train_path,
    usecols=USE_COLS_TRAIN,
    dtype=DTYPES_TRAIN,
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
    nrows=10_000_000,
    engine="c",
)




## === cell 2
def display_all(df):
    with pd.option_context("display.max_rows", 1000), pd.option_context(
        "display.max_columns", 1000
    ):
        display(df)




## === cell 3
pass




## === cell 4
add_datepart(df_raw, "pickup_datetime", drop=True, time=True)




## === cell 5
def distance(data):
    plon = data["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    plat = data["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
    dlon = data["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
    dlat = data["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)

    lon_tr = np.abs(dlon - plon)
    lat_tr = np.abs(dlat - plat)
    data["longitutde_traversed"] = lon_tr.astype(np.float32)
    data["latitude_traversed"] = lat_tr.astype(np.float32)
    data["manhattan_distance"] = (lon_tr + lat_tr).astype(np.float32)

    lat1 = np.radians(plat)
    lon1 = np.radians(plon)
    lat2 = np.radians(dlat)
    lon2 = np.radians(dlon)

    dlat_r = lat2 - lat1
    dlon_r = lon2 - lon1
    a = np.sin(dlat_r / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon_r / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    data["haversine_km"] = (6371.0 * c).astype(np.float32)


def add_location_flags(data):
    manhattan = (-74.047285, -73.910586, 40.683928, 40.882214)
    jfk = (-73.8352, -73.7401, 40.6195, 40.6659)
    ewr = (-74.1920, -74.1460, 40.6700, 40.7080)
    lga = (-73.8895, -73.8550, 40.7660, 40.7760)

    plon = data["pickup_longitude"].to_numpy(copy=False)
    plat = data["pickup_latitude"].to_numpy(copy=False)
    dlon = data["dropoff_longitude"].to_numpy(copy=False)
    dlat = data["dropoff_latitude"].to_numpy(copy=False)

    def in_box_np(lon, lat, box):
        lon_min, lon_max, lat_min, lat_max = box
        return (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)

    pickup_manhattan = in_box_np(plon, plat, manhattan)
    dropoff_manhattan = in_box_np(dlon, dlat, manhattan)

    pickup_jfk = in_box_np(plon, plat, jfk)
    dropoff_jfk = in_box_np(dlon, dlat, jfk)

    pickup_ewr = in_box_np(plon, plat, ewr)
    dropoff_ewr = in_box_np(dlon, dlat, ewr)

    pickup_lga = in_box_np(plon, plat, lga)
    dropoff_lga = in_box_np(dlon, dlat, lga)

    data["pickup_manhattan"] = pickup_manhattan.astype("int8")
    data["dropoff_manhattan"] = dropoff_manhattan.astype("int8")
    data["pickup_jfk"] = pickup_jfk.astype("int8")
    data["dropoff_jfk"] = dropoff_jfk.astype("int8")
    data["pickup_ewr"] = pickup_ewr.astype("int8")
    data["dropoff_ewr"] = dropoff_ewr.astype("int8")
    data["pickup_lga"] = pickup_lga.astype("int8")
    data["dropoff_lga"] = dropoff_lga.astype("int8")

    is_airport_trip = (pickup_jfk | pickup_ewr | pickup_lga) ^ (
        dropoff_jfk | dropoff_ewr | dropoff_lga
    )
    data["is_airport_trip"] = is_airport_trip.astype("int8")




## === cell 6
nyc_min_long, nyc_max_long = -74.3, -73.7
nyc_min_lat, nyc_max_lat = 40.5, 41.0

for col, lo, hi in [
    ("pickup_longitude", nyc_min_long, nyc_max_long),
    ("dropoff_longitude", nyc_min_long, nyc_max_long),
    ("pickup_latitude", nyc_min_lat, nyc_max_lat),
    ("dropoff_latitude", nyc_min_lat, nyc_max_lat),
]:
    df_raw[col] = df_raw[col].clip(lo, hi)

distance(df_raw)
add_location_flags(df_raw)




## === cell 7
pass




## === cell 8
df_raw.dropna(axis=0, how="any", inplace=True)




## === cell 9
df_raw.shape




## === cell 10
key = df_raw.key
df_raw.drop("key", axis=1, inplace=True)




## === cell 11
pass




## === cell 12
df_raw = df_raw[(df_raw.passenger_count > 0) & (df_raw.passenger_count < 10)]




## === cell 13
len(df_raw)




## === cell 14
df_raw.reset_index(drop=True, inplace=True)




## === cell 15
plon = df_raw["pickup_longitude"].to_numpy(copy=False)
dlon = df_raw["dropoff_longitude"].to_numpy(copy=False)
plat = df_raw["pickup_latitude"].to_numpy(copy=False)
dlat = df_raw["dropoff_latitude"].to_numpy(copy=False)

coord_mask = (
    (plon >= nyc_min_long)
    & (plon <= nyc_max_long)
    & (dlon >= nyc_min_long)
    & (dlon <= nyc_max_long)
    & (plat >= nyc_min_lat)
    & (plat <= nyc_max_lat)
    & (dlat >= nyc_min_lat)
    & (dlat <= nyc_max_lat)
)

fare = df_raw["fare_amount"].to_numpy(copy=False)
fare_mask = (fare > 0) & (fare < 250)

lon_tr = df_raw["longitutde_traversed"].to_numpy(copy=False)
lat_tr = df_raw["latitude_traversed"].to_numpy(copy=False)
dist_mask = (lon_tr >= 0) & (lon_tr < 2.0) & (lat_tr >= 0) & (lat_tr < 2.0)

hav = df_raw["haversine_km"].to_numpy(copy=False)
hav_mask = (hav >= 0.0) & (hav <= 100.0)

df_raw = df_raw[coord_mask & fare_mask & dist_mask & hav_mask].reset_index(drop=True)

del (
    plon,
    dlon,
    plat,
    dlat,
    fare,
    lon_tr,
    lat_tr,
    hav,
    coord_mask,
    fare_mask,
    dist_mask,
    hav_mask,
)
gc.collect()




## === cell 16
arr_lon = df_raw["longitutde_traversed"].to_numpy(copy=False)
arr_lat = df_raw["latitude_traversed"].to_numpy(copy=False)

Q1_lon, Q3_lon = np.percentile(arr_lon, [25, 75])
Q1_lat, Q3_lat = np.percentile(arr_lat, [25, 75])

step_lon = 10 * (Q3_lon - Q1_lon)
step_lat = 10 * (Q3_lat - Q1_lat)

mask_lon = (arr_lon >= Q1_lon - step_lon) & (arr_lon <= Q3_lon + step_lon)
mask_lat = (arr_lat >= Q1_lat - step_lat) & (arr_lat <= Q3_lat + step_lat)

outliers_mask = ~(mask_lon & mask_lat)
outliers = np.flatnonzero(outliers_mask).tolist()

len(outliers) / len(df_raw)




## === cell 17
df = df_raw.drop(df_raw.index[outliers]).reset_index(drop=True)

del df_raw, outliers, outliers_mask, mask_lon, mask_lat, arr_lon, arr_lat
gc.collect()




## === cell 18
len(df)




## === cell 19
y = np.log1p(df.fare_amount)
df.drop("fare_amount", axis=1, inplace=True)




## === cell 20
X_train, X_valid, y_train, y_valid = train_test_split(
    df, y, test_size=0.01, random_state=42
)

del df, y
gc.collect()




## === cell 21
def rmse(x, y):
    return math.sqrt(((x - y) ** 2).mean())


def print_score(m):
    pred_tr_log = m.predict(X_train)
    pred_va_log = m.predict(X_valid)

    pred_tr = np.expm1(pred_tr_log)
    pred_va = np.expm1(pred_va_log)

    y_tr = np.expm1(y_train)
    y_va = np.expm1(y_valid)

    res = [
        rmse(pred_tr, y_tr),
        rmse(pred_va, y_va),
        m.score(X_train, y_train),
        m.score(X_valid, y_valid),
    ]
    print(res)




## === cell 22
set_rf_samples(200_000)




## === cell 23
bad_cols = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]
if len(bad_cols) > 0:
    raise TypeError(f"Non-numeric columns remain in X_train: {bad_cols}")

train_w = 1.0 + np.clip(X_train["haversine_km"].values, 0.0, 60.0) / 10.0

m = RandomForestRegressor(
    n_estimators=200,
    n_jobs=-1,
    random_state=42,
    oob_score=True,
    bootstrap=True,
)
m.fit(X_train, y_train, sample_weight=train_w)
print_score(m)




## === cell 24
fi = rf_feat_importance(m, X_train)
fi[:10]




## === cell 25
pass




## === cell 26
USE_COLS_TEST = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
DTYPES_TEST = {
    "key": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
test_set = pd.read_csv(
    test_path,
    usecols=USE_COLS_TEST,
    dtype=DTYPES_TEST,
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
    engine="c",
)




## === cell 27
test_key = test_set.key
test_set.drop("key", axis=1, inplace=True)




## === cell 28
add_datepart(test_set, "pickup_datetime", drop=True, time=True)
distance(test_set)
add_location_flags(test_set)

coord_mask_test = (
    test_set["pickup_longitude"].between(nyc_min_long, nyc_max_long)
    & test_set["dropoff_longitude"].between(nyc_min_long, nyc_max_long)
    & test_set["pickup_latitude"].between(nyc_min_lat, nyc_max_lat)
    & test_set["dropoff_latitude"].between(nyc_min_lat, nyc_max_lat)
)

for col, lo, hi in [
    ("pickup_longitude", nyc_min_long, nyc_max_long),
    ("dropoff_longitude", nyc_min_long, nyc_max_long),
    ("pickup_latitude", nyc_min_lat, nyc_max_lat),
    ("dropoff_latitude", nyc_min_lat, nyc_max_lat),
]:
    test_set[col] = test_set[col].clip(lo, hi)

for c in X_train.columns:
    if c not in test_set.columns:
        test_set[c] = 0
test_set = test_set[X_train.columns]

bad_cols_test = [
    c for c in test_set.columns if not pd.api.types.is_numeric_dtype(test_set[c])
]
if len(bad_cols_test) > 0:
    raise TypeError(f"Non-numeric columns remain in test_set: {bad_cols_test}")

test_predictions_log = m.predict(test_set)
test_predictions = np.expm1(test_predictions_log)

test_predictions = np.clip(test_predictions, 0.0, None)

submission = pd.DataFrame({"key": test_key, "fare_amount": test_predictions})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
