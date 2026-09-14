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
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from pandas.api.types import is_datetime64_any_dtype

DATA_DIR_CANDIDATES = [
    "/kaggle/data/new-york-city-taxi-fare-prediction",
    "/kaggle/data",
    "/kaggle/input/new-york-city-taxi-fare-prediction",
    "/kaggle/input",
    "../input",
]
DATA_DIR = None
for _d in DATA_DIR_CANDIDATES:
    if os.path.exists(_d):
        if os.path.exists(os.path.join(_d, "train.csv")) and os.path.exists(
            os.path.join(_d, "test.csv")
        ):
            DATA_DIR = _d
            break
        comp_dir = os.path.join(_d, "new-york-city-taxi-fare-prediction")
        if os.path.exists(os.path.join(comp_dir, "train.csv")) and os.path.exists(
            os.path.join(comp_dir, "test.csv")
        ):
            DATA_DIR = comp_dir
            break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find train.csv/test.csv in any known DATA_DIR candidates: "
        + str(DATA_DIR_CANDIDATES)
    )

print("Using DATA_DIR:", DATA_DIR)

_READ_COLS = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
_DTYPE_TRAIN = {
    "fare_amount": "float64",
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "float64",
    "pickup_datetime": "object",
}
_DTYPE_TEST = {
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "float64",
}


def chunck_generator(filename, chunk_size=2 * 10**5):
    for chunk in pd.read_csv(
        filename,
        delimiter=",",
        iterator=True,
        chunksize=chunk_size,
        usecols=_READ_COLS,
        dtype=_DTYPE_TRAIN,
        engine="c",
        low_memory=False,
    ):
        yield chunk




## === cell 1
alpha_ang = 0.506


def _coerce_numeric_if_needed(df, cols):
    for c in cols:
        if c in df.columns:
            s = df[c]
            if not pd.api.types.is_numeric_dtype(s.dtype):
                df[c] = pd.to_numeric(s, errors="coerce")
    return df


def distance_travel_arrays(df):
    df = _coerce_numeric_if_needed(
        df,
        [
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
        ],
    )

    plon = df["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    plat = df["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
    dlon = df["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
    dlat = df["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)

    mean_lat = (plat + dlat) / 2.0
    miles_per_deg_lat = 69.0
    miles_per_deg_lon = miles_per_deg_lat * np.cos(np.deg2rad(mean_lat))

    abs_diff_longitude = np.abs(dlon - plon) * miles_per_deg_lon
    abs_diff_latitude = np.abs(dlat - plat) * miles_per_deg_lat

    displacement_vector = np.sqrt(
        abs_diff_latitude * abs_diff_latitude + abs_diff_longitude * abs_diff_longitude
    )

    denom = abs_diff_latitude.copy()
    denom[denom == 0.0] = np.nan
    ratio = abs_diff_longitude / denom
    ang = np.arctan(ratio)

    actual_long = np.abs(displacement_vector * np.sin(ang - alpha_ang))
    actual_lat = np.abs(displacement_vector * np.cos(ang - alpha_ang))
    distance = actual_long + actual_lat

    return (
        abs_diff_longitude,
        abs_diff_latitude,
        displacement_vector,
        actual_long,
        actual_lat,
        distance,
    )


def add_haversine_array(df):
    R_km = 6371.0
    lat1 = np.deg2rad(df["pickup_latitude"].to_numpy(dtype=np.float64, copy=False))
    lat2 = np.deg2rad(df["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False))
    dlat = lat2 - lat1
    lon1 = np.deg2rad(df["pickup_longitude"].to_numpy(dtype=np.float64, copy=False))
    lon2 = np.deg2rad(df["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False))
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (
        np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R_km * c


def add_time_features_arrays(df):
    if "pickup_datetime" in df.columns:
        dt = df["pickup_datetime"]
        if not is_datetime64_any_dtype(dt):
            dt = pd.to_datetime(dt, errors="coerce", utc=True, cache=True)
        try:
            if hasattr(dt.dtype, "tz") and dt.dtype.tz is not None:
                dt = dt.dt.tz_convert(None)
        except Exception:
            dt = pd.to_datetime(dt, errors="coerce", cache=True)
        pickup_hour = dt.dt.hour.to_numpy(dtype=np.float64, copy=False)
        pickup_dayofweek = dt.dt.dayofweek.to_numpy(dtype=np.float64, copy=False)
    else:
        n = len(df)
        pickup_hour = np.full(n, np.nan, dtype=np.float64)
        pickup_dayofweek = np.full(n, np.nan, dtype=np.float64)
    return pickup_hour, pickup_dayofweek




## === cell 2
def data_clean(df):
    if "passenger_count" in df.columns:
        df = df[df.passenger_count > 0]

    if "fare_amount" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["fare_amount"].dtype):
            df["fare_amount"] = pd.to_numeric(
                df["fare_amount"], errors="coerce"
            ).astype(np.float64)
        else:
            df["fare_amount"] = df["fare_amount"].astype(np.float64, copy=False)
        df = df[df.fare_amount > 0]

    if "distance_travel" in df.columns:
        df = df[df.distance_travel > 0]
        df = df.dropna(subset=["distance_travel"])

    if "haversine_km" in df.columns:
        df = df[df.haversine_km >= 0]
        df = df.dropna(subset=["haversine_km"])

    return df


def data_clean_no_filter(df):
    df = _coerce_numeric_if_needed(df, ["passenger_count"])

    df = _coerce_numeric_if_needed(
        df,
        [
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
        ],
    )

    if "distance_travel" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["distance_travel"].dtype):
            df["distance_travel"] = pd.to_numeric(
                df["distance_travel"], errors="coerce"
            )
        df.loc[~np.isfinite(df["distance_travel"].values), "distance_travel"] = np.nan

    if "haversine_km" in df.columns:
        if not pd.api.types.is_numeric_dtype(df["haversine_km"].dtype):
            df["haversine_km"] = pd.to_numeric(df["haversine_km"], errors="coerce")
        df.loc[~np.isfinite(df["haversine_km"].values), "haversine_km"] = np.nan

    for c in ["pickup_hour", "pickup_dayofweek"]:
        if c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c].dtype):
                df[c] = pd.to_numeric(df[c], errors="coerce")

    return df




## === cell 3
def remove_outliers(df):
    if "distance_travel" in df.columns:
        df = df[df.distance_travel < 30]
    if "haversine_km" in df.columns:
        df = df[df.haversine_km < 100]
    if "fare_amount" in df.columns:
        df = df[df.fare_amount < 100]

    for c in ["pickup_longitude", "dropoff_longitude"]:
        if c in df.columns:
            df = df[(df[c] >= -75) & (df[c] <= -72)]
    for c in ["pickup_latitude", "dropoff_latitude"]:
        if c in df.columns:
            df = df[(df[c] >= 40) & (df[c] <= 42)]

    return df


def remove_outliers_no_filter(df):
    if "distance_travel" in df.columns:
        df.loc[df["distance_travel"] >= 30, "distance_travel"] = np.nan
    if "haversine_km" in df.columns:
        df.loc[df["haversine_km"] >= 100, "haversine_km"] = np.nan
    if "passenger_count" in df.columns:
        df.loc[df["passenger_count"] <= 0, "passenger_count"] = np.nan

    for c in ["pickup_longitude", "dropoff_longitude"]:
        if c in df.columns:
            bad = ~df[c].between(-75, -72)
            df.loc[bad, c] = np.nan
    for c in ["pickup_latitude", "dropoff_latitude"]:
        if c in df.columns:
            bad = ~df[c].between(40, 42)
            df.loc[bad, c] = np.nan

    return df




## === cell 4
def incremental_training(train_X, train_y, regr):
    regr.fit(train_X, train_y)
    return regr


def build_features_from_arrays(
    distance_travel, haversine_km, passenger_count, pickup_hour, pickup_dayofweek
):
    n = len(passenger_count)
    X = np.empty((n, 6), dtype=np.float64)
    X[:, 0] = distance_travel
    X[:, 1] = haversine_km
    X[:, 2] = passenger_count
    X[:, 3] = pickup_hour
    X[:, 4] = pickup_dayofweek
    X[:, 5] = 1.0
    return X


def build_features(df):
    return np.column_stack(
        (
            df["distance_travel"].to_numpy(dtype=np.float64, copy=False),
            df["haversine_km"].to_numpy(dtype=np.float64, copy=False),
            df["passenger_count"].to_numpy(dtype=np.float64, copy=False),
            df["pickup_hour"].to_numpy(dtype=np.float64, copy=False),
            df["pickup_dayofweek"].to_numpy(dtype=np.float64, copy=False),
            np.ones(len(df), dtype=np.float64),
        )
    ).astype(np.float64, copy=False)


def permuted_train_test_split_indices(n, split_frac, seed):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    split = int(split_frac * n)
    return perm[:split], perm[split:]




## === cell 5
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(train_path), f"train.csv not found at {train_path}"
assert os.path.exists(test_path), f"test.csv not found at {test_path}"
assert os.path.exists(sample_path), f"sample_submission.csv not found at {sample_path}"

regr = GradientBoostingRegressor(
    n_estimators=100 * 100, warm_start=False, random_state=42
)

t = 100  # number of chunks to train on
n_features = 6
sum_ = np.zeros(n_features, dtype=np.float64)
count_ = np.zeros(n_features, dtype=np.int64)

chunk_size = 2 * 10**5


def _process_chunk_to_train_arrays(df, seed_for_split):
    (
        _abs_diff_longitude,
        _abs_diff_latitude,
        _displacement_vector,
        _actual_long,
        _actual_lat,
        dist,
    ) = distance_travel_arrays(df)
    hav = add_haversine_array(df)

    df["distance_travel"] = dist
    df["haversine_km"] = hav

    df = data_clean(df)
    df = remove_outliers(df)
    if len(df) < 10:
        return None, None

    pickup_hour, pickup_dayofweek = add_time_features_arrays(df)
    df["pickup_hour"] = pickup_hour
    df["pickup_dayofweek"] = pickup_dayofweek

    train_idx, _test_idx = permuted_train_test_split_indices(
        len(df), 0.7, seed_for_split
    )
    df_train = df.iloc[train_idx]

    train_X = build_features(df_train)
    train_y = df_train["fare_amount"].to_numpy(dtype=np.float64, copy=False)
    return train_X, train_y


gen1 = chunck_generator(filename=train_path, chunk_size=chunk_size)
total_rows = 0
effective_chunk = 0

for i in range(t):
    df = next(gen1)
    if len(df) < 10:
        continue

    train_X, train_y = _process_chunk_to_train_arrays(df, seed_for_split=42 + i)
    if train_X is None or len(train_X) < 10:
        continue

    X2000 = train_X[:2000]
    mask = np.isfinite(X2000)
    sum_ += np.where(mask, X2000, 0.0).sum(axis=0)
    count_ += mask.sum(axis=0).astype(np.int64)

    total_rows += train_X.shape[0]
    effective_chunk += 1

if effective_chunk == 0 or total_rows == 0:
    raise RuntimeError(
        "No valid chunks found for training after cleaning/outlier removal."
    )

if np.any(count_ == 0):
    stats = np.where(count_ > 0, sum_ / count_, np.nan)
else:
    stats = sum_ / count_

imp = SimpleImputer(strategy="mean")
imp.fit(np.zeros((1, n_features), dtype=np.float64))
imp.statistics_ = stats.astype(np.float64, copy=False)

mm_X_path = os.path.join("/kaggle/working", "X_train_all_mm.dat")
mm_y_path = os.path.join("/kaggle/working", "y_train_all_mm.dat")
for p in (mm_X_path, mm_y_path):
    try:
        os.remove(p)
    except OSError:
        pass

X_mm = np.memmap(mm_X_path, mode="w+", dtype=np.float64, shape=(total_rows, n_features))
y_mm = np.memmap(mm_y_path, mode="w+", dtype=np.float64, shape=(total_rows,))

gen2 = chunck_generator(filename=train_path, chunk_size=chunk_size)
pos = 0
effective_chunk_2 = 0

for i in range(t):
    df = next(gen2)
    if len(df) < 10:
        continue

    train_X, train_y = _process_chunk_to_train_arrays(df, seed_for_split=42 + i)
    if train_X is None or len(train_X) < 10:
        continue

    n = train_X.shape[0]
    X_mm[pos : pos + n, :] = imp.transform(train_X)
    y_mm[pos : pos + n] = train_y
    pos += n
    effective_chunk_2 += 1

if pos != total_rows:
    X_mm.flush()
    y_mm.flush()
    X_mm = np.memmap(mm_X_path, mode="r+", dtype=np.float64).reshape((-1, n_features))[
        :pos
    ]
    y_mm = np.memmap(mm_y_path, mode="r+", dtype=np.float64)[:pos]
    total_rows = pos

X_mm.flush()
y_mm.flush()

regr = incremental_training(np.asarray(X_mm), np.asarray(y_mm), regr)

print(
    "Training complete. Final n_estimators:",
    regr.n_estimators,
    "effective_chunks:",
    effective_chunk,
    "effective_chunks_pass2:",
    effective_chunk_2,
    "train_rows:",
    int(total_rows),
)




## === cell 6
tdf_raw = pd.read_csv(
    test_path,
    parse_dates=["pickup_datetime"],
    dtype=_DTYPE_TEST,
    infer_datetime_format=True,
    engine="c",
    low_memory=False,
)
test_keys = tdf_raw["key"].values

(
    _abs_diff_longitude,
    _abs_diff_latitude,
    _displacement_vector,
    _actual_long,
    _actual_lat,
    dist,
) = distance_travel_arrays(tdf_raw)
hav = add_haversine_array(tdf_raw)
pickup_hour, pickup_dayofweek = add_time_features_arrays(tdf_raw)

tdf_raw["distance_travel"] = dist
tdf_raw["haversine_km"] = hav
tdf_raw["pickup_hour"] = pickup_hour
tdf_raw["pickup_dayofweek"] = pickup_dayofweek

tdf = data_clean_no_filter(tdf_raw)
tdf = remove_outliers_no_filter(tdf)

ttrain_X = build_features(tdf)
ttrain_X = imp.transform(ttrain_X)
output = regr.predict(ttrain_X)

output = np.asarray(output, dtype=np.float64)
output = np.clip(output, 0, None)

print(
    "Pred stats:", float(np.min(output)), float(np.mean(output)), float(np.max(output))
)




## === cell 7
my_submission = pd.DataFrame({"key": test_keys, "fare_amount": output})
my_submission = my_submission[["key", "fare_amount"]]
my_submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", my_submission.shape)
print(my_submission.head())
