# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

4.79008

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 6.23058) has done: 'I fix the datetime feature extraction to be compatible with modern pandas (replacing deprecated `.dt.week`/`.dt.weekofyear` with ISO calendar week) so your preprocessing runs. Then I ensure no datetime-typed columns leak into the model matrix by coercing all feature columns to numeric, which resolves the RandomForest `DTypePromotionError`. Finally, I keep your same model and training approach but make the train/test feature alignment explicit and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 6.27152) has done: 'Your current RMSE (6.23058) is worse than the target (4.79008), so we should make a small, legitimate improvement without changing the core model/training approach. The biggest low-risk gain here is adding standard NYC Taxi data sanity filters (latitude/longitude bounds and outlier-distance removal) which reduces label noise and typically improves RMSE for this exact competition while keeping your same RandomForest and features. I also ensure we train on a numeric, NaN-free matrix consistently and clip negative predictions to 0 to avoid obviously invalid fares that can hurt RMSE. These changes are minimal, metric-aligned, and keep the same overall pipeline structure.'
- What this solution (achieved 6.26375) has done: 'You’re worse than the target (6.27152 vs 4.79008; lower is better), so we should make a small, safe improvement without changing your core approach (same RF model, same features, same training flow). The biggest issue is your current distance filter (`distance_travelled < 1.0`) is far too strict for NYC trips and discards many valid rides, hurting generalization; widening this to a more realistic bound typically reduces RMSE. I also compute a proper geodesic (haversine) distance feature alongside your existing Euclidean-in-degrees distance (keeping your original feature intact) and add a simple `abs_lon/abs_lat` sum as a tiny extra signal—these are minimal feature additions that keep the same model/training semantics but usually help this competition. Finally, I keep your existing sanity filters but slightly tighten only the most harmful outliers (e.g., absurd fares) while preserving valid longer trips.'
- What this solution (achieved 6.27378) has done: 'Your current RMSE (6.26375) is still worse than the target (4.79008), so we make a small, legitimate improvement without changing the core approach (same RandomForestRegressor, same training flow, same basic feature engineering). The biggest low-risk gain is to correct the overly-strict `distance_travelled < 0.30` filter: that filter is in “degrees” and is effectively removing many valid trips and distorting the training distribution; switching that particular bound to use your already-computed `haversine_km` is a minimal fix that typically improves RMSE. We keep your existing NYC bounding-box and fare/passenger filters, but make the distance filtering consistent and avoid double-filtering that conflicts. The submission writing stays the same and still produce a valid `submission.csv`.'
- What this solution (achieved 6.58884) has done: 'We’re currently worse than the target (6.27378 vs 4.79008; lower is better), so the smallest legitimate push toward the target is to reduce noise in the training labels and align the model to the heavy-tailed fare distribution without changing your core model/training loop. I add two standard NYC Taxi cleanup steps: remove rows where pickup/dropoff coordinates are identical (zero-distance but nonzero fare) and drop extreme outliers in `haversine_km` vs `fare_amount` using a very conservative speed-like cap. Then I train the same RandomForest on `log1p(fare_amount)` and invert with `expm1` at prediction time (same model, same features, same training flow), which typically improves RMSE for this competition by stabilizing large-fare errors. Submission writing and file name stay identical.'
- What this solution (achieved 6.57285) has done: 'Your current RMSE (6.58884; lower is better) is still far from the target (4.79008), so we should make a small, legitimate improvement without changing the core approach (same RandomForestRegressor, same feature family, same single fit/predict flow). The biggest low-risk issue is that the log-target transform is currently trained on the full (noisy) label distribution; adding a very standard, conservative “fare per km” filter (lower and upper bounds) typically removes mislabeled/erroneous trips and improves generalization for this competition. I keep all your existing filters and features, but add one additional cleanup step based on `fare_amount / haversine_km` (with safeguards for small distances) and keep submission writing identical. This should move the score downward (better) toward the target without altering the model architecture or training semantics.'
- What this solution (achieved 6.02947) has done: 'Main bottlenecks are (1) parsing 5M datetimes with multiple `.dt` accesses and `isocalendar()` overhead, (2) repeated Pandas Series operations during cleaning, and (3) fitting a large RandomForest on a large dense matrix. The refactor keeps the exact same features, filters, model, and training semantics, but speeds things up by vectorizing date feature extraction from the raw timestamp string (no `to_datetime`), computing distance features fully in NumPy without extra float64 conversions, and moving the cleanup mask computation to NumPy arrays to avoid Pandas alignment overhead. It also enables Intel-optimized scikit-learn where available and avoids unnecessary copies when building `X_train/X_test`. These changes are provably equivalent for this dataset format (fixed `YYYY-MM-DD HH:MM:SS...`), preserve determinism, and reduce total runtime substantially.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

np.random.seed(0)

INPUT_DIR_CANDIDATES = ["../input", "/kaggle/input", "/kaggle/data"]
INPUT_DIR = None
for p in INPUT_DIR_CANDIDATES:
    if os.path.isdir(p):
        INPUT_DIR = p
        break
if INPUT_DIR is None:
    INPUT_DIR = "../input"

print("Using INPUT_DIR:", INPUT_DIR)
print(os.listdir(INPUT_DIR))




## === cell 1
train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")

if not os.path.exists(train_path):
    alt = os.path.join(INPUT_DIR, "labels.csv")
    if os.path.exists(alt):
        train_path = alt

if not os.path.exists(test_path):
    nested_test = os.path.join(
        INPUT_DIR, "new-york-city-taxi-fare-prediction", "test.csv"
    )
    if os.path.exists(nested_test):
        test_path = nested_test

if not os.path.exists(train_path):
    nested_train = os.path.join(
        INPUT_DIR, "new-york-city-taxi-fare-prediction", "train.csv"
    )
    nested_labels = os.path.join(
        INPUT_DIR, "new-york-city-taxi-fare-prediction", "labels.csv"
    )
    if os.path.exists(nested_train):
        train_path = nested_train
    elif os.path.exists(nested_labels):
        train_path = nested_labels

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
usecols_test = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

dtype_common = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
dtype_train = dict(dtype_common)
dtype_train["fare_amount"] = "float32"

read_csv_train_kwargs = {}
read_csv_test_kwargs = {}
try:
    import pyarrow  # noqa: F401

    read_csv_train_kwargs["engine"] = "pyarrow"
    read_csv_test_kwargs["engine"] = "pyarrow"
except Exception:
    read_csv_train_kwargs["engine"] = "c"
    read_csv_train_kwargs["low_memory"] = False
    read_csv_test_kwargs["engine"] = "c"
    read_csv_test_kwargs["low_memory"] = False

print("Resolved train_path:", train_path)
print("Resolved test_path :", test_path)

train = pd.read_csv(
    train_path,
    nrows=5000000,
    usecols=usecols_train,
    dtype=dtype_train,
    **read_csv_train_kwargs,
)
test = pd.read_csv(
    test_path,
    usecols=usecols_test,
    dtype=dtype_common,
    **read_csv_test_kwargs,
)

print("Train shape:", train.shape, "Test shape:", test.shape)
print("Train columns:", train.columns.tolist())
print("Test columns:", test.columns.tolist())




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1059153375.py in <cell line: 0>()
     73 print("Resolved test_path :", test_path)
     74 
---> 75 train = pd.read_csv(
     76     train_path,
     77     nrows=5000000,

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1605         self._currow = 0
   1606 
-> 1607         options = self._get_options_with_defaults(engine)
   1608         options["storage_options"] = kwds.get("storage_options", None)
   1609 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _get_options_with_defaults(self, engine)
   1641                 and value != getattr(value, "value", default)
   1642             ):
-> 1643                 raise ValueError(
   1644                     f"The {repr(argname)} option is not supported with the "
   1645                     f"'pyarrow' engine"

ValueError: The 'nrows' option is not supported with the 'pyarrow' engine

## === cell 2
def handle_date_inplace(df):
    s = df["pickup_datetime"]

    if s.dtype == object:
        arr = s.to_numpy(copy=False)
        try:
            ends_utc = np.fromiter(
                (
                    (x is not None) and isinstance(x, str) and x.endswith(" UTC")
                    for x in arr
                ),
                dtype=bool,
                count=len(arr),
            )
            if ends_utc.any():
                s2 = pd.Series(arr, index=s.index)
                s2 = s2.astype("string").str.slice(0, -4)
                dt = pd.to_datetime(s2, errors="coerce", cache=True)
            else:
                dt = pd.to_datetime(s, errors="coerce", cache=True)
        except Exception:
            dt = pd.to_datetime(s, errors="coerce", cache=True)
    else:
        dt = pd.to_datetime(s, errors="coerce", cache=True)

    dti = dt.dt
    df["hour_of_day"] = dti.hour.astype("int16")
    df["month"] = dti.month.astype("int16")
    df["year"] = dti.year.astype("int16")
    df["day_of_year"] = dti.dayofyear.astype("int16")
    iso = dti.isocalendar()
    week = iso.week.astype("int16")
    df["week"] = week
    df["week_of_year"] = week
    df["weekday"] = dti.weekday.astype("int16")
    df["quarter"] = dti.quarter.astype("int16")
    df["day_of_month"] = dti.day.astype("int16")

    df.drop(columns=["pickup_datetime"], inplace=True)
    return df


train = handle_date_inplace(train)
test = handle_date_inplace(test)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1788005465.py in <cell line: 0>()
     49 
     50 
---> 51 train = handle_date_inplace(train)
     52 test = handle_date_inplace(test)
     53 

NameError: name 'train' is not defined

## === cell 3
def handle_distance_inplace(df):
    plon = df["pickup_longitude"].to_numpy(dtype=np.float32, copy=False)
    plat = df["pickup_latitude"].to_numpy(dtype=np.float32, copy=False)
    dlon = df["dropoff_longitude"].to_numpy(dtype=np.float32, copy=False)
    dlat = df["dropoff_latitude"].to_numpy(dtype=np.float32, copy=False)

    lon_dist = np.abs(plon - dlon).astype(np.float32, copy=False)
    lat_dist = np.abs(plat - dlat).astype(np.float32, copy=False)

    df["distance_travelled"] = np.sqrt(
        lon_dist * lon_dist + lat_dist * lat_dist
    ).astype(np.float32, copy=False)

    R = np.float32(6371.0)
    deg2rad = np.float32(np.pi / 180.0)
    half = np.float32(0.5)

    plat_r = plat * deg2rad
    plon_r = plon * deg2rad
    dlat_r2 = dlat * deg2rad
    dlon_r2 = dlon * deg2rad

    dlat_r = dlat_r2 - plat_r
    dlon_r = dlon_r2 - plon_r

    sin_dlat = np.sin(dlat_r * half)
    sin_dlon = np.sin(dlon_r * half)
    cos_plat = np.cos(plat_r)
    cos_dlat = np.cos(dlat_r2)

    a = sin_dlat * sin_dlat + cos_plat * cos_dlat * (sin_dlon * sin_dlon)
    df["haversine_km"] = (np.float32(2.0) * R * np.arcsin(np.sqrt(a))).astype(
        np.float32, copy=False
    )

    df["manhattan_approx"] = (lon_dist + lat_dist).astype(np.float32, copy=False)

    df["bearing"] = np.arctan2(dlon_r, dlat_r).astype(np.float32, copy=False)

    def _haversine_to_point_km_from_radians(
        lon_r, lat_r, cos_lat_r, lon0_deg, lat0_deg
    ):
        lon0_r = np.float32(lon0_deg) * deg2rad
        lat0_r = np.float32(lat0_deg) * deg2rad
        dlatp = lat_r - lat0_r
        dlonp = lon_r - lon0_r
        sin_dlatp = np.sin(dlatp * half)
        sin_dlonp = np.sin(dlonp * half)
        a0 = sin_dlatp * sin_dlatp + cos_lat_r * np.cos(lat0_r) * (
            sin_dlonp * sin_dlonp
        )
        return (np.float32(2.0) * R * np.arcsin(np.sqrt(a0))).astype(
            np.float32, copy=False
        )

    NYC_LON, NYC_LAT = -73.985428, 40.748817  # Midtown-ish (Empire State Building)
    JFK_LON, JFK_LAT = -73.7781, 40.6413
    LGA_LON, LGA_LAT = -73.8740, 40.7769
    EWR_LON, EWR_LAT = -74.1745, 40.6895

    cos_plat_r = cos_plat
    cos_dlat_r2 = cos_dlat

    df["pickup_dist_nyc_center_km"] = _haversine_to_point_km_from_radians(
        plon_r, plat_r, cos_plat_r, NYC_LON, NYC_LAT
    )
    df["dropoff_dist_nyc_center_km"] = _haversine_to_point_km_from_radians(
        dlon_r2, dlat_r2, cos_dlat_r2, NYC_LON, NYC_LAT
    )

    df["pickup_dist_jfk_km"] = _haversine_to_point_km_from_radians(
        plon_r, plat_r, cos_plat_r, JFK_LON, JFK_LAT
    )
    df["dropoff_dist_jfk_km"] = _haversine_to_point_km_from_radians(
        dlon_r2, dlat_r2, cos_dlat_r2, JFK_LON, JFK_LAT
    )

    df["pickup_dist_lga_km"] = _haversine_to_point_km_from_radians(
        plon_r, plat_r, cos_plat_r, LGA_LON, LGA_LAT
    )
    df["dropoff_dist_lga_km"] = _haversine_to_point_km_from_radians(
        dlon_r2, dlat_r2, cos_dlat_r2, LGA_LON, LGA_LAT
    )

    df["pickup_dist_ewr_km"] = _haversine_to_point_km_from_radians(
        plon_r, plat_r, cos_plat_r, EWR_LON, EWR_LAT
    )
    df["dropoff_dist_ewr_km"] = _haversine_to_point_km_from_radians(
        dlon_r2, dlat_r2, cos_dlat_r2, EWR_LON, EWR_LAT
    )

    return df


train = handle_distance_inplace(train)
test = handle_distance_inplace(test)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3405059651.py in <cell line: 0>()
     98 
     99 
--> 100 train = handle_distance_inplace(train)
    101 test = handle_distance_inplace(test)
    102 

NameError: name 'train' is not defined

## === cell 4
def clean_up_train(train):
    train = train.dropna()

    fare = train["fare_amount"].to_numpy(dtype=np.float32, copy=False)
    pc = train["passenger_count"].to_numpy(dtype=np.int16, copy=False)
    plon = train["pickup_longitude"].to_numpy(dtype=np.float32, copy=False)
    dlon = train["dropoff_longitude"].to_numpy(dtype=np.float32, copy=False)
    plat = train["pickup_latitude"].to_numpy(dtype=np.float32, copy=False)
    dlat = train["dropoff_latitude"].to_numpy(dtype=np.float32, copy=False)
    hv = train["haversine_km"].to_numpy(dtype=np.float32, copy=False)

    mask = (
        (fare > 0)
        & (pc > 0)
        & (pc < 7)
        & (plon >= -74.3)
        & (plon <= -73.7)
        & (dlon >= -74.3)
        & (dlon <= -73.7)
        & (plat >= 40.5)
        & (plat <= 41.0)
        & (dlat >= 40.5)
        & (dlat <= 41.0)
        & (hv > 0.05)
        & (hv < 80.0)
        & (fare < 200)
    )

    mask &= (
        (np.abs(plon) > 0.001)
        & (np.abs(dlon) > 0.001)
        & (np.abs(plat) > 0.001)
        & (np.abs(dlat) > 0.001)
    )

    same_loc = (plon == dlon) & (plat == dlat)
    mask &= ~same_loc

    mask &= fare <= (2.5 + 10.0 * hv)

    denom_km = np.maximum(hv, np.float32(0.2))
    fare_per_km = fare / denom_km
    mask &= (fare_per_km >= 0.5) & (fare_per_km <= 60.0)

    return train.loc[mask]


train = clean_up_train(train)
print("Cleaned train shape:", train.shape)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3802320300.py in <cell line: 0>()
     46 
     47 
---> 48 train = clean_up_train(train)
     49 print("Cleaned train shape:", train.shape)
     50 

NameError: name 'train' is not defined

## === cell 5
feature_cols = test.columns.drop("key")
feature_cols = list(feature_cols)  # stable order

X_train = train.loc[:, feature_cols].to_numpy(dtype=np.float32, copy=False)
y = train["fare_amount"].to_numpy(dtype=np.float64, copy=False)
X_test = test.loc[:, feature_cols].to_numpy(dtype=np.float32, copy=False)

np.nan_to_num(X_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
np.nan_to_num(X_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

print("X_train:", X_train.shape, "X_test:", X_test.shape, "y:", y.shape)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2496221726.py in <cell line: 0>()
----> 1 feature_cols = test.columns.drop("key")
      2 feature_cols = list(feature_cols)  # stable order
      3 
      4 X_train = train.loc[:, feature_cols].to_numpy(dtype=np.float32, copy=False)
      5 y = train["fare_amount"].to_numpy(dtype=np.float64, copy=False)

NameError: name 'test' is not defined

## === cell 6
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=120,
    max_depth=14,
    min_samples_leaf=1,
    random_state=0,
    n_jobs=-1,
)

y_train_log = np.log1p(y)
model.fit(X_train, y_train_log)

preds_log = model.predict(X_test)
preds = np.expm1(preds_log)

preds = np.clip(preds, 0.0, None)

submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": preds}, columns=["key", "fare_amount"]
)
submission.to_csv("submission.csv", index=False)

print(submission.head(20))
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1178044822.py in <cell line: 0>()
     16 )
     17 
---> 18 y_train_log = np.log1p(y)
     19 model.fit(X_train, y_train_log)
     20 

NameError: name 'y' is not defined
