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
import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from scipy.interpolate import griddata  # noqa: F401

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

INPUT_DIR = "/kaggle/input/new-york-city-taxi-fare-prediction"
print("Input directory exists:", os.path.exists(INPUT_DIR))
print("Files:", sorted(os.listdir(INPUT_DIR))[:20])

TRAIN_USECOLS = [
    "fare_amount",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

TRAIN_DTYPES = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}

TEST_USECOLS = [
    "key",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
TEST_DTYPES = {
    "key": "string",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}


def chunck_generator(filename, chunk_size=250_000):
    for chunk in pd.read_csv(
        filename,
        delimiter=",",
        iterator=True,
        chunksize=chunk_size,
        usecols=TRAIN_USECOLS,
        dtype=TRAIN_DTYPES,
        engine="c",
    ):
        yield chunk




## === cell 1
alpha_ang = 0.506


def compute_distance_travel_arrays(df_or_arrays):
    if isinstance(df_or_arrays, dict):
        plon = df_or_arrays["pickup_longitude"]
        plat = df_or_arrays["pickup_latitude"]
        dlon = df_or_arrays["dropoff_longitude"]
        dlat = df_or_arrays["dropoff_latitude"]
        plon = np.asarray(plon)
        plat = np.asarray(plat)
        dlon = np.asarray(dlon)
        dlat = np.asarray(dlat)
    else:
        plon = df_or_arrays["pickup_longitude"].to_numpy(copy=False)
        plat = df_or_arrays["pickup_latitude"].to_numpy(copy=False)
        dlon = df_or_arrays["dropoff_longitude"].to_numpy(copy=False)
        dlat = df_or_arrays["dropoff_latitude"].to_numpy(copy=False)

    plon = plon.astype(np.float64, copy=False)
    plat = plat.astype(np.float64, copy=False)
    dlon = dlon.astype(np.float64, copy=False)
    dlat = dlat.astype(np.float64, copy=False)

    abs_diff_longitude = np.abs(dlon - plon)
    abs_diff_longitude *= 50.0
    abs_diff_latitude = np.abs(dlat - plat)
    abs_diff_latitude *= 69.0

    displacement_vector = np.hypot(abs_diff_latitude, abs_diff_longitude)

    ratio = np.divide(
        abs_diff_longitude,
        abs_diff_latitude,
        out=np.full_like(abs_diff_longitude, np.inf, dtype=np.float64),
        where=abs_diff_latitude != 0,
    )
    angle = np.arctan(ratio)
    angle -= alpha_ang

    actual_long = displacement_vector * np.sin(angle)
    np.abs(actual_long, out=actual_long)

    actual_lat = displacement_vector * np.cos(angle)
    np.abs(actual_lat, out=actual_lat)

    distance_travel = actual_long
    distance_travel += actual_lat
    return distance_travel


def distance_travel(df):
    df["distance_travel"] = compute_distance_travel_arrays(df)
    return df




## === cell 2
def data_clean(df):
    df = df[df.passenger_count > 0]
    df.fare_amount = df.fare_amount.astype(np.float64)
    df = df[df.fare_amount > 0]
    df = df[df.distance_travel > 0]
    return df




## === cell 3
def remove_outliers(df):
    df = df[df.distance_travel < 30]
    df = df[df.fare_amount < 100]
    return df




## === cell 4
def graph_presesnt(df):
    test = df[df.passenger_count == 1]
    plot = test.iloc[: len(test)].plot.scatter("distance_travel", "fare_amount")
    plot = df.iloc[:100000].plot.scatter("distance_travel", "fare_amount")
    return plot




## === cell 5
def incremental_training(train_X, train_y, regr):
    regr.fit(train_X, train_y)
    return regr




## === cell 6
filename = os.path.join(INPUT_DIR, "train.csv")

regr = GradientBoostingRegressor(n_estimators=100, warm_start=True, random_state=42)
imp = SimpleImputer(strategy="mean")

chunk_size = 500_000
t = 55
nrows = t * chunk_size

X_all = np.empty((nrows, 3), dtype=np.float64)
y_all = np.empty((nrows,), dtype=np.float64)
filled = 0


def _train_reader_arrow_numpy(path, nrows):
    import pyarrow as pa
    import pyarrow.csv as pv

    convert_opts = pv.ConvertOptions(
        include_columns=TRAIN_USECOLS,
        column_types={
            "fare_amount": pa.float32(),
            "pickup_longitude": pa.float32(),
            "pickup_latitude": pa.float32(),
            "dropoff_longitude": pa.float32(),
            "dropoff_latitude": pa.float32(),
            "passenger_count": pa.int16(),
        },
        strings_can_be_null=True,
    )
    read_opts = pv.ReadOptions(block_size=1 << 28, use_threads=True)
    parse_opts = pv.ParseOptions(delimiter=",", newlines_in_values=False)

    with pv.open_csv(
        path,
        read_options=read_opts,
        parse_options=parse_opts,
        convert_options=convert_opts,
    ) as reader:
        read_total = 0
        while read_total < nrows:
            batch = reader.read_next_batch()
            if batch is None or batch.num_rows == 0:
                break

            remaining = nrows - read_total
            if batch.num_rows > remaining:
                batch = batch.slice(0, remaining)

            cols = batch.columns
            fare = cols[0].to_numpy(zero_copy_only=False)
            plon = cols[1].to_numpy(zero_copy_only=False)
            plat = cols[2].to_numpy(zero_copy_only=False)
            dlon = cols[3].to_numpy(zero_copy_only=False)
            dlat = cols[4].to_numpy(zero_copy_only=False)
            pcount = cols[5].to_numpy(zero_copy_only=False)

            read_total += batch.num_rows
            yield fare, plon, plat, dlon, dlat, pcount


def _train_reader_pandas(path, chunk_size, nrows):
    reader = pd.read_csv(
        path,
        delimiter=",",
        iterator=True,
        chunksize=chunk_size,
        usecols=TRAIN_USECOLS,
        dtype=TRAIN_DTYPES,
        engine="c",
        nrows=nrows,
        memory_map=True,
        low_memory=False,
        na_filter=True,
        keep_default_na=True,
    )
    for df in reader:
        yield df


rows_read = 0
seen = 0

use_arrow = False
try:
    import pyarrow  # noqa: F401

    use_arrow = True
except Exception as e:
    print("pyarrow not available; using pandas. Error:", repr(e))

if not use_arrow:
    for seen, df in enumerate(
        _train_reader_pandas(filename, chunk_size=chunk_size, nrows=nrows), start=1
    ):
        n = len(df)
        rows_read += n
        if n == 0:
            break

        plon = df["pickup_longitude"].to_numpy(copy=False)
        plat = df["pickup_latitude"].to_numpy(copy=False)
        dlon = df["dropoff_longitude"].to_numpy(copy=False)
        dlat = df["dropoff_latitude"].to_numpy(copy=False)
        dist = compute_distance_travel_arrays(
            {
                "pickup_longitude": plon,
                "pickup_latitude": plat,
                "dropoff_longitude": dlon,
                "dropoff_latitude": dlat,
            }
        )
        fare = df["fare_amount"].to_numpy(copy=False).astype(np.float64, copy=False)
        pcount = df["passenger_count"].to_numpy(copy=False).astype(np.int16, copy=False)

        mask = (pcount > 0) & (fare > 0) & (dist > 0) & (dist < 30) & (fare < 100)
        l = int(mask.sum())

        if l >= 10:
            end = filled + l
            if end > X_all.shape[0]:
                l = X_all.shape[0] - filled
                if l <= 0:
                    break
                end = filled + l

            dist_kept = dist[mask]
            fare_kept = fare[mask]
            pcount_kept = pcount[mask]
            if dist_kept.shape[0] > l:
                dist_kept = dist_kept[:l]
                fare_kept = fare_kept[:l]
                pcount_kept = pcount_kept[:l]

            X_all[filled:end, 0] = dist_kept
            X_all[filled:end, 1] = pcount_kept.astype(np.float64, copy=False)
            X_all[filled:end, 2] = 1.0
            y_all[filled:end] = fare_kept
            filled = end

        if seen <= 2 or seen % 10 == 0:
            print(f"Processed chunk {seen}, kept rows: {l}, total kept: {filled}")

        if rows_read >= nrows or filled >= nrows:
            break
else:
    for seen, (fare, plon, plat, dlon, dlat, pcount) in enumerate(
        _train_reader_arrow_numpy(filename, nrows=nrows), start=1
    ):
        n = fare.shape[0]
        rows_read += n
        if n == 0:
            break

        dist = compute_distance_travel_arrays(
            {
                "pickup_longitude": plon,
                "pickup_latitude": plat,
                "dropoff_longitude": dlon,
                "dropoff_latitude": dlat,
            }
        )
        fare64 = np.asarray(fare).astype(np.float64, copy=False)
        pcount16 = np.asarray(pcount).astype(np.int16, copy=False)

        mask = (pcount16 > 0) & (fare64 > 0) & (dist > 0) & (dist < 30) & (fare64 < 100)
        l = int(mask.sum())

        if l >= 10:
            end = filled + l
            if end > X_all.shape[0]:
                l = X_all.shape[0] - filled
                if l <= 0:
                    break
                end = filled + l

            dist_kept = dist[mask]
            fare_kept = fare64[mask]
            pcount_kept = pcount16[mask]
            if dist_kept.shape[0] > l:
                dist_kept = dist_kept[:l]
                fare_kept = fare_kept[:l]
                pcount_kept = pcount_kept[:l]

            X_all[filled:end, 0] = dist_kept
            X_all[filled:end, 1] = pcount_kept.astype(np.float64, copy=False)
            X_all[filled:end, 2] = 1.0
            y_all[filled:end] = fare_kept
            filled = end

        if seen <= 2 or seen % 10 == 0:
            print(f"Processed batch {seen}, kept rows: {l}, total kept: {filled}")

        if rows_read >= nrows or filled >= nrows:
            break

train_X = X_all[:filled]
train_y = y_all[:filled]

imp = imp.fit(train_X)
train_X = imp.transform(train_X)

regr = incremental_training(train_X, train_y, regr)
print("Final training rows:", train_X.shape[0], "features:", train_X.shape[1])




## === cell 7
tdf = pd.read_csv(
    os.path.join(INPUT_DIR, "test.csv"),
    usecols=TEST_USECOLS,
    dtype=TEST_DTYPES,
    engine="c",
    memory_map=True,
    low_memory=False,
    na_filter=True,
    keep_default_na=True,
)

test_arrays = {
    "pickup_longitude": tdf["pickup_longitude"].to_numpy(copy=False),
    "pickup_latitude": tdf["pickup_latitude"].to_numpy(copy=False),
    "dropoff_longitude": tdf["dropoff_longitude"].to_numpy(copy=False),
    "dropoff_latitude": tdf["dropoff_latitude"].to_numpy(copy=False),
}
test_dist = compute_distance_travel_arrays(test_arrays)
tdf.head()




## === cell 8
n_test = len(tdf)
ttrain_X = np.empty((n_test, 3), dtype=np.float64)
ttrain_X[:, 0] = test_dist
ttrain_X[:, 1] = (
    tdf["passenger_count"].to_numpy(copy=False).astype(np.float64, copy=False)
)
ttrain_X[:, 2] = 1.0

ttrain_X = imp.transform(ttrain_X)
output = regr.predict(ttrain_X)

output = np.clip(output, 0.0, 100.0)

print(output[:10])




## === cell 9
my_submission = pd.DataFrame({"key": tdf["key"], "fare_amount": output})
my_submission.to_csv("submission.csv", index=False)
print(my_submission.head())
print("Wrote submission.csv with shape:", my_submission.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
