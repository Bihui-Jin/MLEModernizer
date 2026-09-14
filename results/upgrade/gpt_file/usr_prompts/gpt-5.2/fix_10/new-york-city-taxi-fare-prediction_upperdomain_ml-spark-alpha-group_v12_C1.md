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

INPUT_DIR = "/kaggle/input/new-york-city-taxi-fare-prediction"
print(os.listdir(INPUT_DIR))

TRAIN_USECOLS = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
TEST_USECOLS = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

TRAIN_DTYPES = {
    "fare_amount": "float64",
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int16",
}
TEST_DTYPES = {
    "key": "object",
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int16",
}


def chunck_generator(filename, chunk_size=10**6):
    for chunk in pd.read_csv(
        filename,
        delimiter=",",
        iterator=True,
        chunksize=chunk_size,
        usecols=TRAIN_USECOLS,
        dtype=TRAIN_DTYPES,
        parse_dates=["pickup_datetime"],
        engine="c",
        low_memory=False,
    ):
        yield chunk




## === cell 1
alpha_ang = 0.506


def distance_travel(df):
    plon = df["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    plat = df["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
    dlon = df["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
    dlat = df["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)

    abs_diff_longitude = np.abs(dlon - plon) * 50.0
    abs_diff_latitude = np.abs(dlat - plat) * 69.0

    displacement_vector = np.sqrt(
        abs_diff_latitude * abs_diff_latitude + abs_diff_longitude * abs_diff_longitude
    )

    eps = 1e-12
    angle = np.arctan(abs_diff_longitude / (abs_diff_latitude + eps)) - alpha_ang
    actual_long = np.abs(displacement_vector * np.sin(angle))
    actual_lat = np.abs(displacement_vector * np.cos(angle))

    df["abs_diff_longitude"] = abs_diff_longitude
    df["abs_diff_latitude"] = abs_diff_latitude
    df["displacement_vector"] = displacement_vector
    df["actual_long"] = actual_long
    df["actual_lat"] = actual_lat
    df["distance_travel"] = actual_long + actual_lat
    return df




## === cell 2
def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=False)
    df["pickup_hour"] = dt.dt.hour.astype("float64")
    df["pickup_dow"] = dt.dt.dayofweek.astype("float64")
    return df


def data_clean(df):
    df = df[df.passenger_count > 0]
    df.fare_amount = df.fare_amount.astype(np.float64)
    df = df[df.fare_amount > 0]
    df = df[df.distance_travel > 0]
    return df


def remove_outliers(df):
    df = df[df.distance_travel < 30]
    df = df[df.fare_amount < 100]
    return df


def filter_nyc_bbox(df):
    return df[
        (df["pickup_longitude"].between(-74.3, -73.6))
        & (df["dropoff_longitude"].between(-74.3, -73.6))
        & (df["pickup_latitude"].between(40.5, 41.0))
        & (df["dropoff_latitude"].between(40.5, 41.0))
    ]


def filter_zero_gps(df):
    return df[
        ~(
            (df["pickup_longitude"] == 0.0)
            & (df["pickup_latitude"] == 0.0)
            & (df["dropoff_longitude"] == 0.0)
            & (df["dropoff_latitude"] == 0.0)
        )
    ]


def filter_fare_distance_sanity(df):
    return df[~((df["distance_travel"] > 0.2) & (df["fare_amount"] < 2.5))]


def filter_high_fare_short_trip(df):
    return df[~((df["distance_travel"] < 0.05) & (df["fare_amount"] > 30.0))]




## === cell 3
def incremental_training(train_X, train_y, regr):
    regr.fit(train_X, train_y)
    return regr




## === cell 4
filename = os.path.join(INPUT_DIR, "train.csv")
gen = chunck_generator(filename=filename, chunk_size=10**6)

TARGET_ROWS = 4_000_000
parts = []
rows = 0

while rows < TARGET_ROWS:
    df = next(gen)

    df = distance_travel(df)
    df = add_time_features(df)  # keep feature building consistent between train/test
    df = data_clean(df)
    df = remove_outliers(df)
    df = filter_nyc_bbox(df)
    df = filter_zero_gps(df)  # reduce label noise from invalid GPS rows
    df = filter_fare_distance_sanity(
        df
    )  # reduce noisy low-fare labels for meaningful distances
    df = filter_high_fare_short_trip(
        df
    )  # reduce noisy high-fare labels for near-zero distances

    if len(df) == 0:
        continue

    parts.append(df)
    rows += len(df)
    print("Collected rows:", rows)

train_df = pd.concat(parts, axis=0, ignore_index=True)
print("Final train_df shape:", train_df.shape)

train_X = np.column_stack(
    (
        np.asarray(train_df.distance_travel, dtype=np.float64),
        np.asarray(train_df.passenger_count, dtype=np.float64),
        np.asarray(train_df.pickup_hour, dtype=np.float64),
        np.asarray(train_df.pickup_dow, dtype=np.float64),
        np.ones(len(train_df), dtype=np.float64),
    )
)

train_y = np.log1p(np.asarray(train_df.fare_amount, dtype=np.float64))

imp = SimpleImputer(missing_values=np.nan, strategy="mean")
train_X = imp.fit_transform(train_X)

regr = GradientBoostingRegressor(n_estimators=100, warm_start=False, random_state=42)
regr = incremental_training(train_X, train_y, regr)
print("Training done on rows:", len(train_df))



## === cell 5
tdf = pd.read_csv(
    os.path.join(INPUT_DIR, "test.csv"),
    usecols=TEST_USECOLS,
    dtype=TEST_DTYPES,
    parse_dates=["pickup_datetime"],
    engine="c",
    low_memory=False,
)

tdf_all = tdf.copy()

tdf = distance_travel(tdf)
tdf = add_time_features(tdf)  # keep feature building consistent

tdf_valid = tdf[(tdf["passenger_count"] > 0) & (tdf["distance_travel"] > 0)]
tdf_valid = filter_zero_gps(tdf_valid)

tdf_in = filter_nyc_bbox(tdf_valid)

ttrain_X_in = np.column_stack(
    (
        np.asarray(tdf_in.distance_travel, dtype=np.float64),
        np.asarray(tdf_in.passenger_count, dtype=np.float64),
        np.asarray(tdf_in.pickup_hour, dtype=np.float64),
        np.asarray(tdf_in.pickup_dow, dtype=np.float64),
        np.ones(len(tdf_in), dtype=np.float64),
    )
)
ttrain_X_in = imp.transform(ttrain_X_in)
pred_in = regr.predict(ttrain_X_in)

ttrain_X_all = np.column_stack(
    (
        np.asarray(tdf.distance_travel, dtype=np.float64),
        np.asarray(tdf.passenger_count, dtype=np.float64),
        np.asarray(tdf.pickup_hour, dtype=np.float64),
        np.asarray(tdf.pickup_dow, dtype=np.float64),
        np.ones(len(tdf), dtype=np.float64),
    )
)
ttrain_X_all = imp.transform(ttrain_X_all)
pred_all = regr.predict(ttrain_X_all)

pred_series = pd.Series(pred_all, index=tdf.index)
pred_series.loc[tdf_in.index] = pred_in

output = np.expm1(pred_series.reindex(tdf_all.index).to_numpy())
output = np.maximum(output, 0.0)

print(output[:10])



## === cell 6
my_submission = pd.DataFrame({"key": tdf_all["key"], "fare_amount": output})
my_submission.to_csv("submission.csv", index=False)
print(my_submission.head())
print("Wrote submission.csv with shape:", my_submission.shape)
