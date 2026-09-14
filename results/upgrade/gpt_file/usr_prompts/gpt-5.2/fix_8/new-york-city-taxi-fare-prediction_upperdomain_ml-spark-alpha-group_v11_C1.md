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

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer

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
    "fare_amount": "float64",
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
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
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
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


def compute_distance_travel_arrays(df):
    plon = df["pickup_longitude"].to_numpy(copy=False)
    plat = df["pickup_latitude"].to_numpy(copy=False)
    dlon = df["dropoff_longitude"].to_numpy(copy=False)
    dlat = df["dropoff_latitude"].to_numpy(copy=False)

    abs_diff_longitude = np.abs(dlon - plon) * 50.0
    abs_diff_latitude = np.abs(dlat - plat) * 69.0

    displacement_vector = np.hypot(abs_diff_latitude, abs_diff_longitude)

    ratio = np.divide(
        abs_diff_longitude,
        abs_diff_latitude,
        out=np.full_like(abs_diff_longitude, np.inf, dtype=np.float64),
        where=abs_diff_latitude != 0,
    )
    angle = np.arctan(ratio) - alpha_ang
    actual_long = np.abs(displacement_vector * np.sin(angle))
    actual_lat = np.abs(displacement_vector * np.cos(angle))

    distance_travel = actual_long + actual_lat
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
nrows = t * 250_000  # keep identical total rows read as the original script

X_all = np.empty((nrows, 3), dtype=np.float64)
y_all = np.empty((nrows,), dtype=np.float64)
filled = 0

reader = pd.read_csv(
    filename,
    delimiter=",",
    iterator=True,
    chunksize=chunk_size,
    usecols=TRAIN_USECOLS,
    engine="c",
    nrows=nrows,
    memory_map=True,
    low_memory=False,
    na_filter=True,
    keep_default_na=True,
)

seen = 0
rows_read = 0
for df in reader:
    rows_read += len(df)

    for c in [
        "fare_amount",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    dist = compute_distance_travel_arrays(df)
    fare = df["fare_amount"].to_numpy(copy=False)
    pcount = df["passenger_count"].to_numpy(copy=False)

    mask = (pcount > 0) & (fare > 0) & (dist > 0) & (dist < 30) & (fare < 100)
    l = int(mask.sum())
    if l < 10:
        if rows_read >= nrows:
            break
        continue

    dist_m = dist[mask]
    pcount_m = pcount[mask].astype(np.float64, copy=False)
    fare_m = fare[mask].astype(np.float64, copy=False)

    end = filled + l
    if end > X_all.shape[0]:
        end = X_all.shape[0]
        l = end - filled
        if l <= 0:
            break
        dist_m = dist_m[:l]
        pcount_m = pcount_m[:l]
        fare_m = fare_m[:l]

    X_all[filled:end, 0] = dist_m
    X_all[filled:end, 1] = pcount_m
    X_all[filled:end, 2] = 1.0
    y_all[filled:end] = fare_m

    filled = end
    seen += 1

    if seen <= 3 or seen % 5 == 0:
        print(f"Processed chunk {seen}, kept rows: {l}, total kept: {filled}")

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
    engine="c",
    memory_map=True,
    low_memory=False,
    na_filter=True,
    keep_default_na=True,
)

for c in [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]:
    tdf[c] = pd.to_numeric(tdf[c], errors="coerce")

test_dist = compute_distance_travel_arrays(tdf)
tdf.head()



## === cell 8
ttrain_X = np.empty((len(tdf), 3), dtype=np.float64)
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
