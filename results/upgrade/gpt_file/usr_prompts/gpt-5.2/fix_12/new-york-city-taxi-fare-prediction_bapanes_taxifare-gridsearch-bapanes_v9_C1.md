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
print("hello moto")



## === cell 1
import os
import pandas as pd
import numpy as np
import math

from sklearn.neural_network import MLPRegressor

np.random.seed(42)

DATA_DIR = "/kaggle/input"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

print("Train exists:", os.path.exists(TRAIN_PATH), TRAIN_PATH)
print("Test exists:", os.path.exists(TEST_PATH), TEST_PATH)
print("Sample exists:", os.path.exists(SAMPLE_SUB_PATH), SAMPLE_SUB_PATH)




## === cell 2
def haversine_distance_miles(
    pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude
):
    """
    Vectorized haversine distance in miles.
    NaN-safe: if any coord is NaN, output will be NaN for that row (handled later).
    """
    lon1 = np.radians(pickup_longitude.astype(float))
    lat1 = np.radians(pickup_latitude.astype(float))
    lon2 = np.radians(dropoff_longitude.astype(float))
    lat2 = np.radians(dropoff_latitude.astype(float))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    r_miles = 3958.756  # Earth radius in miles
    return r_miles * c


def add_distance_feature(df):
    df["distance"] = haversine_distance_miles(
        df["pickup_longitude"].values,
        df["pickup_latitude"].values,
        df["dropoff_longitude"].values,
        df["dropoff_latitude"].values,
    )
    return df


def add_datetime_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df["pickup_hour"] = dt.dt.hour.astype("float32")
    df["pickup_dow"] = dt.dt.dayofweek.astype("float32")
    df["pickup_year"] = dt.dt.year.astype("float32")
    return df


def add_extra_geo_features(df):
    """
    Minimal feature expansion to add strong fare signal without changing model family/training:
    - coordinate deltas and manhattan distance proxy
    - airport trip indicator (JFK/LGA/EWR proximity)
    """
    plon = df["pickup_longitude"].astype(float)
    plat = df["pickup_latitude"].astype(float)
    dlon = df["dropoff_longitude"].astype(float)
    dlat = df["dropoff_latitude"].astype(float)

    df["delta_lon"] = (dlon - plon).astype("float32")
    df["delta_lat"] = (dlat - plat).astype("float32")

    lat_rad = np.radians(((plat + dlat) / 2.0).astype(float))
    miles_per_deg_lon = 69.172 * np.cos(lat_rad)
    miles_per_deg_lat = 69.0
    manhattan = (np.abs(df["delta_lon"].astype(float)) * miles_per_deg_lon) + (
        np.abs(df["delta_lat"].astype(float)) * miles_per_deg_lat
    )
    df["manhattan_dist"] = manhattan.astype("float32")

    airports = np.array(
        [
            [-73.7781, 40.6413],  # JFK
            [-73.8740, 40.7769],  # LGA
            [-74.1745, 40.6895],  # EWR
        ],
        dtype=np.float64,
    )

    def min_airport_dist(lon, lat):
        lon = lon.astype(np.float64)[:, None]
        lat = lat.astype(np.float64)[:, None]
        alon = airports[None, :, 0]
        alat = airports[None, :, 1]
        d = haversine_distance_miles(lon, lat, alon, alat)  # broadcasts
        return np.nanmin(d, axis=1)

    pickup_min = min_airport_dist(plon.values, plat.values)
    dropoff_min = min_airport_dist(dlon.values, dlat.values)
    airport_trip = ((pickup_min < 1.5) | (dropoff_min < 1.5)).astype("float32")
    df["airport_trip"] = airport_trip

    return df


def filter_train_rows(df):
    req = [
        "fare_amount",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "distance",
        "pickup_hour",
        "pickup_dow",
        "pickup_year",
        "delta_lon",
        "delta_lat",
        "manhattan_dist",
        "airport_trip",
    ]
    df = df.dropna(subset=req)

    df = df[
        (df["pickup_longitude"].between(-74.3, -73.6))
        & (df["dropoff_longitude"].between(-74.3, -73.6))
        & (df["pickup_latitude"].between(40.5, 41.0))
        & (df["dropoff_latitude"].between(40.5, 41.0))
    ]

    df = df[df["passenger_count"].between(1, 6)]
    df = df[df["fare_amount"].between(2.5, 250.0)]
    df = df[df["distance"].between(0.03, 60.0)]
    df = df[df["manhattan_dist"].between(0.03, 80.0)]

    return df




## === cell 3
def data_to_np(input_file, nrows=900000, skiprows=None):
    """
    Read Kaggle-provided CSV and compute features on the fly.

    Correctness for chunking: keep header row (row 0) and only skip data rows.
    """
    df = pd.read_csv(
        input_file,
        sep=",",
        nrows=nrows,
        skiprows=skiprows,
        header=0,
        low_memory=False,
    )

    df = add_distance_feature(df)
    df = add_datetime_features(df)
    df = add_extra_geo_features(df)

    header_names = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "distance",
        "pickup_hour",
        "pickup_dow",
        "pickup_year",
        "delta_lon",
        "delta_lat",
        "manhattan_dist",
        "airport_trip",
    ]

    if "fare_amount" in df.columns:
        df = filter_train_rows(df)
        X = df[header_names].values.astype(np.float32)
        y = df["fare_amount"].values.astype(np.float32)
        return X, y
    else:
        X = df[header_names].values.astype(np.float32)
        return X, None




## === cell 4
def global_mean_per_column_weighted(mynp_train_list):
    total_n = 0
    sum_vec = None
    for arr in mynp_train_list:
        n = arr.shape[0]
        if n == 0:
            continue
        if sum_vec is None:
            sum_vec = np.sum(arr, axis=0, dtype=np.float64)
        else:
            sum_vec += np.sum(arr, axis=0, dtype=np.float64)
        total_n += n
    if total_n == 0:
        raise ValueError("No training rows after filtering; cannot compute mean/std.")
    return (sum_vec / total_n).astype(np.float64)


def global_std_per_column_weighted(mynp_train_list, global_mean):
    total_n = 0
    sum_sq = None
    for arr in mynp_train_list:
        n = arr.shape[0]
        if n == 0:
            continue
        diff = arr.astype(np.float64) - global_mean
        if sum_sq is None:
            sum_sq = np.sum(diff * diff, axis=0, dtype=np.float64)
        else:
            sum_sq += np.sum(diff * diff, axis=0, dtype=np.float64)
        total_n += n
    if total_n == 0:
        raise ValueError("No training rows after filtering; cannot compute mean/std.")
    var = sum_sq / total_n
    return np.sqrt(var).astype(np.float64)


def norm_mynp_train(mynp_train, mean, std):
    std_safe = np.where(std == 0, 1.0, std)
    mynp_train_norm = (mynp_train - mean) / std_safe
    return mynp_train_norm




## === cell 5
NROWS_PER_CHUNK = 250000  # 3 chunks -> up to 750k raw rows (fewer after filtering)
TRAIN_TOTAL_ROWS_EST = 55423856

start_rows = [
    0,
    TRAIN_TOTAL_ROWS_EST // 2,
    max(0, TRAIN_TOTAL_ROWS_EST - NROWS_PER_CHUNK),
]
print("Chunk starts (0-based data rows):", start_rows)

header_names = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "distance",
    "pickup_hour",
    "pickup_dow",
    "pickup_year",
    "delta_lon",
    "delta_lat",
    "manhattan_dist",
    "airport_trip",
]

usecols_train = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]


def _process_train_df_to_xy(df):
    df = add_distance_feature(df)
    df = add_datetime_features(df)
    df = add_extra_geo_features(df)
    df = filter_train_rows(df)
    X = df[header_names].to_numpy(dtype=np.float32, copy=False)
    y = df["fare_amount"].to_numpy(dtype=np.float32, copy=False)
    return X, y


targets = [
    (start_rows[0], start_rows[0] + NROWS_PER_CHUNK),
    (start_rows[1], start_rows[1] + NROWS_PER_CHUNK),
    (start_rows[2], start_rows[2] + NROWS_PER_CHUNK),
]

mynp_train_0 = np.empty((0, len(header_names)), dtype=np.float32)
mynp_label_0 = np.empty((0,), dtype=np.float32)
mynp_train_1 = np.empty((0, len(header_names)), dtype=np.float32)
mynp_label_1 = np.empty((0,), dtype=np.float32)
mynp_train_2 = np.empty((0, len(header_names)), dtype=np.float32)
mynp_label_2 = np.empty((0,), dtype=np.float32)

STREAM_CHUNKSIZE = 1_000_000

ranges = [list(r) for r in targets]  # mutable [start, end]
done = [False, False, False]
raw_row_cursor = 0  # counts data rows (excluding header)

reader = pd.read_csv(
    TRAIN_PATH,
    usecols=usecols_train,
    chunksize=STREAM_CHUNKSIZE,
    low_memory=False,
)

for df_chunk in reader:
    chunk_n = len(df_chunk)
    chunk_start = raw_row_cursor
    chunk_end = raw_row_cursor + chunk_n

    for i, (w_start, w_end) in enumerate(ranges):
        if done[i]:
            continue
        inter_start = max(chunk_start, w_start)
        inter_end = min(chunk_end, w_end)
        if inter_end > inter_start:
            s0 = inter_start - chunk_start
            s1 = inter_end - chunk_start
            df_part = df_chunk.iloc[s0:s1].copy()

            Xi, yi = _process_train_df_to_xy(df_part)

            if i == 0:
                mynp_train_0, mynp_label_0 = Xi, yi
            elif i == 1:
                mynp_train_1, mynp_label_1 = Xi, yi
            else:
                mynp_train_2, mynp_label_2 = Xi, yi

            done[i] = True

    raw_row_cursor = chunk_end
    if all(done):
        break

print(mynp_train_0.shape, mynp_train_1.shape, mynp_train_2.shape)
print("Labels:", mynp_label_0.shape, mynp_label_1.shape, mynp_label_2.shape)




## === cell 6
def shuffle_in_unison(X, y, seed=42):
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(y))
    return X[order], y[order]


mynp_train_0, mynp_label_0 = shuffle_in_unison(mynp_train_0, mynp_label_0, seed=1)
mynp_train_1, mynp_label_1 = shuffle_in_unison(mynp_train_1, mynp_label_1, seed=2)
mynp_train_2, mynp_label_2 = shuffle_in_unison(mynp_train_2, mynp_label_2, seed=3)



## === cell 7
mynp_train_list = [mynp_train_0, mynp_train_1, mynp_train_2]
mynp_label_list = [mynp_label_0, mynp_label_1, mynp_label_2]

global_mean = global_mean_per_column_weighted(mynp_train_list)
global_std = global_std_per_column_weighted(mynp_train_list, global_mean)

print("global_mean:", global_mean)
print("global_std :", global_std)



## === cell 8
raw_means = global_mean.astype(np.float32)

for i in range(len(mynp_train_list)):
    X = mynp_train_list[i]
    if X.size == 0:
        continue
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X = X.copy()
        X[nan_mask] = np.broadcast_to(raw_means, X.shape)[nan_mask]
        mynp_train_list[i] = X

mynp_train_norm_0 = norm_mynp_train(mynp_train_list[0], global_mean, global_std)
mynp_train_norm_1 = norm_mynp_train(mynp_train_list[1], global_mean, global_std)
mynp_train_norm_2 = norm_mynp_train(mynp_train_list[2], global_mean, global_std)

mynp_train_concat = np.concatenate(
    (mynp_train_norm_0, mynp_train_norm_1, mynp_train_norm_2), axis=0
)
mynp_label_concat = np.concatenate((mynp_label_0, mynp_label_1, mynp_label_2), axis=0)

print("Train concat:", mynp_train_concat.shape, mynp_label_concat.shape)
print(
    "y stats: min/mean/max:",
    float(np.min(mynp_label_concat)),
    float(np.mean(mynp_label_concat)),
    float(np.max(mynp_label_concat)),
)



## === cell 9
cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "distance",
    "pickup_hour",
    "pickup_dow",
    "pickup_year",
    "delta_lon",
    "delta_lat",
    "manhattan_dist",
    "airport_trip",
]

df_test_raw = pd.read_csv(
    TEST_PATH,
    sep=",",
    header=0,
    low_memory=False,
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
)

df_test_raw = add_distance_feature(df_test_raw)
df_test_raw = add_datetime_features(df_test_raw)
df_test_raw = add_extra_geo_features(df_test_raw)

df_test_feat = df_test_raw[cols]

df_test_feat = df_test_feat.astype(np.float32)
df_test_feat = df_test_feat.fillna(pd.Series(raw_means, index=cols).astype(np.float32))

test_key_array = df_test_raw["key"].values
mynp_test = df_test_feat.to_numpy(dtype=np.float32, copy=False)
mynp_test = norm_mynp_train(mynp_test, global_mean, global_std)

print("Test:", mynp_test.shape, test_key_array.shape)
print("Any NaN in test features after fill:", bool(np.isnan(mynp_test).any()))



## === cell 10
model_after_gridSearch = MLPRegressor(
    hidden_layer_sizes=(64, 64, 64),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size=256,
    learning_rate_init=0.001,
    max_iter=30,
    random_state=42,
    verbose=False,
)

model_after_gridSearch.fit(mynp_train_concat, mynp_label_concat)



## === cell 11
test_predictions = model_after_gridSearch.predict(mynp_test).astype(np.float32)

test_predictions = np.clip(test_predictions, 0.0, 250.0)

print(test_predictions[:10])
print(
    "pred stats: min/mean/max:",
    float(np.min(test_predictions)),
    float(np.mean(test_predictions)),
    float(np.max(test_predictions)),
)



## === cell 12
df_output = pd.DataFrame({"key": test_key_array, "fare_amount": test_predictions})

sample_sub = pd.read_csv(SAMPLE_SUB_PATH, header=0, low_memory=False)
df_output = sample_sub[["key"]].merge(df_output, on="key", how="left")

df_output["fare_amount"] = df_output["fare_amount"].fillna(
    float(np.mean(test_predictions))
)

df_output.head()



## === cell 13
out_path = "submission_file.csv"
df_output.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(df_output), "cols:", list(df_output.columns))
print(df_output.head())
