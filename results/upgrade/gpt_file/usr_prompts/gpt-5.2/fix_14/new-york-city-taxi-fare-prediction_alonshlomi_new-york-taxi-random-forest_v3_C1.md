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

3.12

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

4.069922676663

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 11.46379) has done: 'Main bottlenecks are (1) CSV parsing of millions of rows with default dtypes, (2) expensive pandas-to-numpy coercions inside `haversine_distance` and datetime parsing, and (3) fitting the RandomForest twice (validation fit + full-data refit). To stay within 600s without changing the model/feature logic, the script below uses faster `read_csv` via explicit `usecols`+dtypes, performs feature engineering with pre-extracted NumPy arrays (same haversine and time features), and avoids the redundant second full refit by training once on the full dataset and computing the validation RMSE via out-of-bag predictions (same RMSE metric; no change to architecture/loss). All randomness remains controlled with `random_state=42`, and I/O paths stay identical.'
- What this solution (achieved 11.39327) has done: 'Your current RMSE (11.46) is far worse than the target (4.07), so we need a real (but still minimal) quality boost without changing the core model/features. The biggest likely issue is train/test row misalignment caused by `test.dropna()` after loading: dropping rows changes the required submission row count/order and can severely hurt the score. I keep all test rows by imputing missing test values (instead of dropping), and I apply the same coordinate/passenger cleaning logic to test (clipping to training bounds) so feature distributions match. Finally, I add a very light, metric-consistent post-process: clip predictions to the training target range to reduce extreme outliers that inflate RMSE.'
- What this solution (achieved 11.39846) has done: 'Your current RMSE is far above the target, so we need a real (but still minimal) boost in model quality without changing the core approach (RandomForest + the same basic engineered features). The largest win with minimal semantic change is to add two standard, low-risk taxi-fare features—absolute deltas in lat/lon and a simple Manhattan-distance proxy—computed from the same existing coordinates; this keeps the model family and training loop identical but gives the forest more informative splits. I also fix a subtle train/test dtype inconsistency for the time features (they were float32 in test before casting), and I apply the same datetime parsing + row drop policy for invalid datetimes in train only (test rows are preserved with imputation). Submission writing remains unchanged and produces a valid `submission.csv`.'
- What this solution (achieved 14.83155) has done: 'Your current RMSE (11.40) is much worse than the target (4.07), so we need a real-but-minimal quality lift without changing the core model family or overall training approach. The biggest likely issue is feature distribution mismatch: you heavily filter train to NYC bounds but only clip test; instead, we also clip train coordinates (rather than dropping lots of rows) to better match test while keeping the same features. Next, we add the standard NYC taxi cleanup of zero-distance (same pickup/dropoff) and fare/coordinate validity in a minimal way to reduce noise that RandomForest struggles with. Finally, we keep submission row order intact and retain your existing clipping of predictions to prevent RMSE blow-ups from outliers.'
- What this solution (achieved 19.46908) has done: 'Your current RMSE (14.83) is far worse than the target (4.07), so we should make small changes that reduce obvious noise/mismatch without changing the core approach (RandomForest + the same coordinate/time/distance features). The biggest regression in your latest version is clipping *training* coordinates instead of filtering them, which keeps many bad/out-of-distribution points but forces them into NYC bounds, creating label noise the forest can’t learn; switching back to filtering (while still clipping test) is a minimal, high-impact fix. I also add a standard cleanup for invalid passenger_count (0, negative, very large) in train before feature engineering, and I keep submission row order/count unchanged. Everything else (model, features, metric semantics, and output format) stays the same.'
- What this solution (achieved 18.78306) has done: 'Your current RMSE (19.47) is far worse than the target (4.07), so the smallest high-impact fix is to remove systematic label noise that a RandomForest can’t learn: include the well-known “outliers to remove” rule (fares tied to unrealistic long distances) while keeping the exact same model and features. I also add one minimal, standard geographic filter for NYC-ish coordinates (a tighter bounding box) to drop obvious bad GPS points rather than forcing the model to fit them. Finally, I keep your existing “clip test, filter train” policy and preserve the submission row count/order, so the CSV remains valid while quality improves toward the target.'
- What this solution (achieved 17.79288) has done: 'Your RMSE is far above the target (lower is better), so we need a small but meaningful quality lift without changing the core model family or the overall feature set. The biggest high-impact, low-risk fix here is to stop forcing test coordinates into NYC bounds via clipping (which destroys distance geometry and creates systematic feature bias); instead we leave test coordinates as-is (after imputing) and only filter bad training rows, keeping train/test feature semantics aligned. Next, we add one standard, minimal cleanup rule to training (remove absurdly long trips) that reduces label noise RandomForest struggles with, without changing the model or loss. Finally, we keep the submission row count/order intact and keep your existing prediction clipping to avoid extreme outliers inflating RMSE.'
- What this solution (achieved 6.61851) has done: 'Your current RMSE (17.79) is far worse than the target (4.07), so we should make a small, high-impact fix that reduces systematic label noise without changing your model family, training loop, or feature set. The biggest remaining issue is that we never removed “zero/near-zero distance but non-trivial fare” artifacts and other GPS glitches beyond `distance > 0`; these heavily confuse a RandomForest and can inflate RMSE. I add a minimal, standard train-only filter using the already-computed `distance` to drop implausible short trips and extreme fare-per-km outliers, while keeping test rows untouched and preserving the exact same features and submission format. Everything else (NROWS cap, engineered features, RandomForest hyperparameters, OOB validation, prediction clipping, and I/O paths) remains the same.'
- What this solution (achieved 11.66581) has done: 'Your current RMSE (6.61851) is still worse than the target (4.0699), so we should make a small, low-risk improvement that reduces label noise without changing the model family, training loop, or core feature set. The biggest remaining quality gap is usually from “non-NYC” and otherwise bad GPS rows that slip through simple bounding-box filtering; a minimal, standard fix is to additionally filter training rows by (a) plausible trip distance range and (b) a reasonable fare-per-km band, using the already-computed `distance`. This keeps test untouched (no row drops, no clipping), preserves your exact features, and typically improves RMSE by removing hard-to-learn outliers. I also keep submission format/row order identical and leave your prediction clipping in place to prevent extreme outliers inflating RMSE.'
- What this solution (achieved 9.32718) has done: 'Your current RMSE (11.66581, lower is better) is far worse than the target (4.0699), and the most likely cause is the very aggressive post-distance outlier filtering that removes a large fraction of training data and biases the model. I make the smallest quality-focused change by relaxing the two fare-per-km filters (keep the same features/model/training loop) so you retain much more valid training signal while still removing extreme GPS/label noise. I also compute `distance/abs_*` **after** datetime coercion+drop so the arrays always stay aligned (same logic, but avoids subtle misalignment risk after dropping rows). Submission writing, row order, and prediction clipping remain unchanged to guarantee a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
    SKLEARNEX_PATCHED = True
except Exception:
    SKLEARNEX_PATCHED = False

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

BASE_INPUT = "/kaggle/input"
DATA_DIR = os.path.join(BASE_INPUT, "new-york-city-taxi-fare-prediction")

print("sklearnex patched:", SKLEARNEX_PATCHED)
print("BASE_INPUT exists:", os.path.exists(BASE_INPUT))
print("Available under /kaggle/input:", os.listdir(BASE_INPUT)[:20])
print("Using DATA_DIR:", DATA_DIR)
print("DATA_DIR listing (head):", os.listdir(DATA_DIR)[:20])

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

print("train.csv exists:", os.path.exists(train_path))
print("test.csv exists:", os.path.exists(test_path))
print("sample_submission.csv exists:", os.path.exists(sample_sub_path))



## === cell 1
NROWS = 2_000_000  # keep the same pragmatic cap as the provided solution

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

dtype_train = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
    "key": "string",
    "pickup_datetime": "string",
}
dtype_test = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
    "key": "string",
    "pickup_datetime": "string",
}

read_csv_kwargs = dict()
try:
    import pyarrow  # noqa: F401

    read_csv_kwargs["engine"] = "pyarrow"
except Exception:
    pass

train = pd.read_csv(
    train_path, nrows=NROWS, usecols=usecols_train, dtype=dtype_train, **read_csv_kwargs
)
test = pd.read_csv(test_path, usecols=usecols_test, dtype=dtype_test, **read_csv_kwargs)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2875816906.py in <cell line: 0>()
     50     pass
     51 
---> 52 train = pd.read_csv(
     53     train_path, nrows=NROWS, usecols=usecols_train, dtype=dtype_train, **read_csv_kwargs
     54 )

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
train.dropna(inplace=True)

numeric_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
medians = {c: float(train[c].median()) for c in numeric_cols}
for c in numeric_cols:
    test[c] = test[c].astype("float32", copy=False)
    test[c] = test[c].fillna(medians[c])

test["passenger_count"] = test["passenger_count"].fillna(1).astype("int16", copy=False)
test["pickup_datetime"] = test["pickup_datetime"].fillna("2009-01-01 00:00:00 UTC")

mask = (train["fare_amount"] > 0) & (train["fare_amount"] < 500)
mask &= (train["passenger_count"] >= 1) & (train["passenger_count"] <= 6)

coord_bounds = {
    "pickup_longitude": (-74.5, -72.8),
    "dropoff_longitude": (-74.5, -72.8),
    "pickup_latitude": (40.5, 41.8),
    "dropoff_latitude": (40.5, 41.8),
}
for col, (lo, hi) in coord_bounds.items():
    mask &= (train[col] >= lo) & (train[col] <= hi)

train = train.loc[mask].copy()

print("Train shape after cleaning:", train.shape)
print("Test shape (kept all rows):", test.shape)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2384346383.py in <cell line: 0>()
      1 # --- Speed: avoid repeated in-place modifications that trigger pandas copies; keep identical semantics.
----> 2 train.dropna(inplace=True)
      3 
      4 numeric_cols = [
      5     "pickup_longitude",

NameError: name 'train' is not defined

## === cell 3
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two points on Earth (in km).
    Vectorized for NumPy arrays / pandas Series.
    """
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c




## === cell 4
train["pickup_datetime"] = pd.to_datetime(
    train["pickup_datetime"], errors="coerce", utc=True
)
train.dropna(subset=["pickup_datetime"], inplace=True)

p_lat = train["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
p_lon = train["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
d_lat = train["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)
d_lon = train["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)

dist_km = haversine_distance(p_lat, p_lon, d_lat, d_lon).astype(np.float32)
abs_lat = np.abs(d_lat - p_lat).astype(np.float32)
abs_lon = np.abs(d_lon - p_lon).astype(np.float32)

train["distance"] = dist_km
train["abs_lat_diff"] = abs_lat
train["abs_lon_diff"] = abs_lon
train["manhattan"] = (abs_lat + abs_lon).astype(np.float32)

NYC_LAT, NYC_LON = 40.7128, -74.0060
pickup_to_center = haversine_distance(p_lat, p_lon, NYC_LAT, NYC_LON).astype(np.float32)
dropoff_to_center = haversine_distance(d_lat, d_lon, NYC_LAT, NYC_LON).astype(
    np.float32
)

train["pickup_to_center_km"] = pickup_to_center
train["dropoff_to_center_km"] = dropoff_to_center
train["center_delta_km"] = (dropoff_to_center - pickup_to_center).astype(np.float32)

dt = train["pickup_datetime"].dt
train["day_of_week"] = dt.dayofweek.astype(np.int8)
train["month"] = dt.month.astype(np.int8)
train["hour"] = dt.hour.astype(np.int8)

fare_amount = train["fare_amount"].to_numpy(dtype=np.float64, copy=False)
dist64 = dist_km.astype(np.float64, copy=False)

mask = dist_km > 0
mask &= ~((fare_amount < 2.5) & (dist_km > 0.5))
mask &= ~((fare_amount < 3.5) & (dist_km > 1.0))
mask &= ~((fare_amount < 5.0) & (dist_km > 2.0))
mask &= dist_km <= 100
mask &= ~((dist_km < 0.05) & (fare_amount > 10.0))
mask &= (dist_km >= 0.05) & (dist_km <= 80.0)

fare_per_km = (fare_amount / np.maximum(dist64, 0.001)).astype(np.float32)
mask &= (fare_per_km >= 0.5) & (fare_per_km <= 80.0)

train = train.loc[mask].copy()

dist = train["distance"].to_numpy(dtype=np.float64, copy=False)
fare = train["fare_amount"].to_numpy(dtype=np.float64, copy=False)
A = np.vstack([dist, np.ones_like(dist)]).T
coef, _, _, _ = np.linalg.lstsq(A, fare, rcond=None)
fare_hat = A @ coef
resid = fare - fare_hat
mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
keep = np.abs(resid) <= (12.0 * mad)
train = train.loc[keep].copy()

train.head()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4274445420.py in <cell line: 0>()
      1 # --- Speed: same feature engineering, but avoid repeated .copy() filters by using a single mask.
      2 train["pickup_datetime"] = pd.to_datetime(
----> 3     train["pickup_datetime"], errors="coerce", utc=True
      4 )
      5 train.dropna(subset=["pickup_datetime"], inplace=True)

NameError: name 'train' is not defined

## === cell 5
test["passenger_count"] = test["passenger_count"].clip(1, 6).astype("int16", copy=False)

p_lat = test["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
p_lon = test["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
d_lat = test["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)
d_lon = test["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)

dist_km = haversine_distance(p_lat, p_lon, d_lat, d_lon).astype(np.float32)
abs_lat = np.abs(d_lat - p_lat).astype(np.float32)
abs_lon = np.abs(d_lon - p_lon).astype(np.float32)

test["distance"] = dist_km
test["abs_lat_diff"] = abs_lat
test["abs_lon_diff"] = abs_lon
test["manhattan"] = (abs_lat + abs_lon).astype(np.float32)

NYC_LAT, NYC_LON = 40.7128, -74.0060
pickup_to_center = haversine_distance(p_lat, p_lon, NYC_LAT, NYC_LON).astype(np.float32)
dropoff_to_center = haversine_distance(d_lat, d_lon, NYC_LAT, NYC_LON).astype(
    np.float32
)

test["pickup_to_center_km"] = pickup_to_center
test["dropoff_to_center_km"] = dropoff_to_center
test["center_delta_km"] = (dropoff_to_center - pickup_to_center).astype(np.float32)

test["pickup_datetime"] = pd.to_datetime(
    test["pickup_datetime"], errors="coerce", utc=True
)

dt_test = test["pickup_datetime"].dt
test["day_of_week"] = dt_test.dayofweek
test["month"] = dt_test.month
test["hour"] = dt_test.hour

dow_mode = int(train["day_of_week"].mode(dropna=True).iloc[0])
month_mode = int(train["month"].mode(dropna=True).iloc[0])
hour_mode = int(train["hour"].mode(dropna=True).iloc[0])

test["day_of_week"] = test["day_of_week"].fillna(dow_mode).astype(np.int8)
test["month"] = test["month"].fillna(month_mode).astype(np.int8)
test["hour"] = test["hour"].fillna(hour_mode).astype(np.int8)

test.head()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2826665967.py in <cell line: 0>()
      1 # Feature engineering for test kept identical; minor speedups by reusing arrays and minimizing pandas overhead.
----> 2 test["passenger_count"] = test["passenger_count"].clip(1, 6).astype("int16", copy=False)
      3 
      4 p_lat = test["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
      5 p_lon = test["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)

NameError: name 'test' is not defined

## === cell 6
feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "distance",
    "abs_lat_diff",
    "abs_lon_diff",
    "manhattan",
    "pickup_to_center_km",
    "dropoff_to_center_km",
    "center_delta_km",
    "day_of_week",
    "month",
    "hour",
]

X = np.ascontiguousarray(train[feature_cols].to_numpy())
y = train["fare_amount"].to_numpy()

print("X shape:", X.shape, "y shape:", y.shape)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3137759159.py in <cell line: 0>()
     18 
     19 # --- Speed: feed contiguous NumPy arrays to sklearn to reduce pandas overhead (same values/features).
---> 20 X = np.ascontiguousarray(train[feature_cols].to_numpy())
     21 y = train["fare_amount"].to_numpy()
     22 

NameError: name 'train' is not defined

## === cell 7
random_forest = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
    oob_score=True,  # enables oob_prediction_
    bootstrap=True,  # explicit to guarantee OOB availability
)

random_forest.fit(X, y)

oob = getattr(random_forest, "oob_prediction_", None)
if oob is not None:
    mask = np.isfinite(oob)
    rmse = mean_squared_error(y[mask], oob[mask], squared=False)
    print(f"OOB RMSE (proxy for validation): {rmse:.6f}")
else:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    rf_tmp = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    rf_tmp.fit(X_train, y_train)
    y_pred = rf_tmp.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"Validation RMSE: {rmse:.6f}")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/491360053.py in <cell line: 0>()
     10 )
     11 
---> 12 random_forest.fit(X, y)
     13 
     14 oob = getattr(random_forest, "oob_prediction_", None)

NameError: name 'X' is not defined

## === cell 8
X_test = np.ascontiguousarray(test[feature_cols].to_numpy())
y_pred_test = random_forest.predict(X_test)

y_min = float(y.min())
y_max = float(y.max())
y_pred_test = np.clip(y_pred_test, y_min, y_max)
y_pred_test = np.maximum(y_pred_test, 0)

submission = pd.DataFrame({"key": test["key"].astype(str), "fare_amount": y_pred_test})
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(
    "Expected test rows:",
    pd.read_csv(test_path, usecols=["key"], **read_csv_kwargs).shape[0],
)
submission.head()

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/677486104.py in <cell line: 0>()
      1 # --- Speed: contiguous NumPy for predict; preserves identical post-processing/clipping.
----> 2 X_test = np.ascontiguousarray(test[feature_cols].to_numpy())
      3 y_pred_test = random_forest.predict(X_test)
      4 
      5 y_min = float(y.min())

NameError: name 'test' is not defined
