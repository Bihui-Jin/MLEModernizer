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
from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.impute import SimpleImputer as Imputer
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import os

print(os.listdir("../input"))

TRAIN_PATH = r"../input/train.csv"
TEST_PATH = r"../input/test.csv"

TRAIN_USECOLS = [
    "fare_amount",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
TEST_USECOLS = [
    "key",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

DTYPES_TRAIN = {
    "fare_amount": "float64",  # keep float64
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int8",
}
DTYPES_TEST = {
    "key": "object",
    "pickup_longitude": "float64",
    "pickup_latitude": "float64",
    "dropoff_longitude": "float64",
    "dropoff_latitude": "float64",
    "passenger_count": "int8",
}


def chunck_generator(filename, chunk_size=10**6, usecols=None, dtypes=None):
    for chunk in pd.read_csv(
        filename,
        delimiter=",",
        iterator=True,
        chunksize=chunk_size,
        usecols=usecols,
        dtype=dtypes,
        engine="c",
        skipinitialspace=True,
    ):
        yield chunk




## === cell 1
alpha_ang = 0.506


def distance_travel_np(plon, plat, dlon, dlat):
    abs_diff_longitude = np.abs(dlon - plon) * 50.0
    abs_diff_latitude = np.abs(dlat - plat) * 69.0

    displacement_vector = np.sqrt(
        abs_diff_latitude * abs_diff_latitude + abs_diff_longitude * abs_diff_longitude
    )

    ratio = np.empty_like(abs_diff_longitude)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(abs_diff_longitude, abs_diff_latitude, out=ratio)

    angle = np.arctan(ratio) - alpha_ang

    actual_long = np.abs(displacement_vector * np.sin(angle))
    actual_lat = np.abs(displacement_vector * np.cos(angle))
    return actual_long + actual_lat


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

    ratio = np.empty_like(abs_diff_longitude)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(abs_diff_longitude, abs_diff_latitude, out=ratio)

    angle = np.arctan(ratio) - alpha_ang

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
def data_clean(df):
    df["fare_amount"] = df["fare_amount"].astype(np.float64)
    m = (
        (df["passenger_count"] > 0)
        & (df["fare_amount"] > 0)
        & (df["distance_travel"] > 0)
    )
    return df[m]


def remove_outliers(df):
    m = (df["distance_travel"] < 30) & (df["fare_amount"] < 100)
    return df[m]




## === cell 3
def incremental_training(train_X, train_y, regr):
    regr.fit(train_X, train_y)
    return regr




## === cell 4
CHUNK_SIZE = 1_000_000
N_CHUNKS = 56  # preserved
RANDOM_STATE = 42

FINAL_N_ESTIMATORS = 20 + 20 * N_CHUNKS  # preserved from original intent

ESTIMATORS_PER_CHUNK = 20

regr = GradientBoostingRegressor(
    n_estimators=ESTIMATORS_PER_CHUNK,
    warm_start=True,
    random_state=RANDOM_STATE,
)

sum_ = np.zeros(2, dtype=np.float64)
count_ = np.zeros(2, dtype=np.int64)

gen = chunck_generator(
    filename=TRAIN_PATH,
    chunk_size=CHUNK_SIZE,
    usecols=TRAIN_USECOLS,
    dtypes=DTYPES_TRAIN,
)

t = N_CHUNKS
chunk_idx = 0
while t > 0:
    print(f"stream train, chunks left: {t}")
    try:
        df = next(gen)
    except StopIteration:
        break

    plon = df["pickup_longitude"].to_numpy(dtype=np.float64, copy=False)
    plat = df["pickup_latitude"].to_numpy(dtype=np.float64, copy=False)
    dlon = df["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False)
    dlat = df["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False)
    dist = distance_travel_np(plon, plat, dlon, dlat)

    fare = df["fare_amount"].to_numpy(dtype=np.float64, copy=False)
    pcount = df["passenger_count"].to_numpy(copy=False).astype(np.float64, copy=False)

    m = (pcount > 0) & (fare > 0) & (dist > 0) & (dist < 30) & (fare < 100)

    if not np.any(m):
        t -= 1
        chunk_idx += 1
        continue

    X0 = dist[m]
    X1 = pcount[m]
    y = fare[m]

    m0 = ~np.isnan(X0)
    if m0.any():
        sum_[0] += X0[m0].sum(dtype=np.float64)
        count_[0] += int(m0.sum())
    m1 = ~np.isnan(X1)
    if m1.any():
        sum_[1] += X1[m1].sum(dtype=np.float64)
        count_[1] += int(m1.sum())

    train_X = np.empty((y.shape[0], 3), dtype=np.float64)
    train_X[:, 0] = X0
    train_X[:, 1] = X1
    train_X[:, 2] = 1.0

    means2_running = sum_ / np.maximum(count_, 1)
    means3_running = np.array(
        [means2_running[0], means2_running[1], 1.0], dtype=np.float64
    )

    nan0 = np.isnan(train_X[:, 0])
    if nan0.any():
        train_X[nan0, 0] = means3_running[0]
    nan1 = np.isnan(train_X[:, 1])
    if nan1.any():
        train_X[nan1, 1] = means3_running[1]

    target_estimators = ESTIMATORS_PER_CHUNK * (chunk_idx + 1)
    if target_estimators > FINAL_N_ESTIMATORS:
        target_estimators = FINAL_N_ESTIMATORS
    regr.set_params(n_estimators=target_estimators)
    incremental_training(train_X, y, regr)

    t -= 1
    chunk_idx += 1
    if target_estimators >= FINAL_N_ESTIMATORS:
        pass

means2 = sum_ / np.maximum(count_, 1)
means3 = np.array([means2[0], means2[1], 1.0], dtype=np.float64)

imp = Imputer(missing_values=np.nan, strategy="mean")
imp.fit(np.zeros((1, 3), dtype=np.float64))
imp.statistics_ = means3

print(
    f"Finished streaming training with {regr.n_estimators} estimators (target {FINAL_N_ESTIMATORS})."
)



## === cell 5
tdf = pd.read_csv(
    TEST_PATH,
    usecols=TEST_USECOLS,
    dtype=DTYPES_TEST,
    engine="c",
    skipinitialspace=True,
)
distance_travel(tdf)
tdf.head()



## === cell 6
n = len(tdf)
ttrain_X = np.empty((n, 3), dtype=np.float64)
ttrain_X[:, 0] = tdf.distance_travel.to_numpy(dtype=np.float64, copy=False)
ttrain_X[:, 1] = tdf.passenger_count.to_numpy(copy=False).astype(np.float64, copy=False)
ttrain_X[:, 2] = 1.0

ttrain_X = imp.transform(ttrain_X)

output = regr.predict(ttrain_X)
print(output)



## === cell 7
my_submission = pd.DataFrame({"key": tdf.key, "fare_amount": output})
my_submission.to_csv("submission.csv", index=False)
my_submission.head()
