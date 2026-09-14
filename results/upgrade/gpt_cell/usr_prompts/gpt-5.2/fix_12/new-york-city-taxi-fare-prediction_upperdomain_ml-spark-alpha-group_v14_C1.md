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

try:
    from sklearnex import patch_sklearn

    patch_sklearn()  # Speeds up sklearn algorithms via oneDAL where available; semantics preserved.
except Exception:
    pass

from sklearn.impute import SimpleImputer as Imputer
from sklearn.ensemble import GradientBoostingRegressor

print(os.listdir("../input"))

TRAIN_COLS = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
TEST_COLS = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

DTYPES_TRAIN_FAST = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
DTYPES_TEST_FAST = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
    "key": "string",
}

np.random.seed(42)




## === cell 1
def chunck_generator(filename, header=False, chunk_size=10**6):
    usecols = [
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    for chunk in pd.read_csv(
        filename,
        delimiter=",",
        iterator=True,
        chunksize=chunk_size,
        usecols=usecols,
        dtype=DTYPES_TRAIN_FAST,
        parse_dates=["pickup_datetime"],
        engine="c",
        memory_map=True,
    ):
        yield chunk




## === cell 2
alpha_ang = 0.506


def _distance_travel_from_arrays(pu_lon, pu_lat, do_lon, do_lat):
    abs_diff_long = np.abs(do_lon - pu_lon) * 50.0
    abs_diff_lat = np.abs(do_lat - pu_lat) * 69.0
    disp = np.sqrt(abs_diff_lat * abs_diff_lat + abs_diff_long * abs_diff_long)

    denom = abs_diff_lat.copy()
    denom[denom == 0.0] = np.nan
    angle = np.arctan(abs_diff_long / denom)
    angle = np.nan_to_num(angle, nan=0.0, posinf=0.0, neginf=0.0)

    actual_long = np.abs(disp * np.sin(angle - alpha_ang))
    actual_lat = np.abs(disp * np.cos(angle - alpha_ang))
    return actual_long + actual_lat


def distance_travel(df):
    dist = _distance_travel_from_arrays(
        df["pickup_longitude"].to_numpy(dtype=np.float64, copy=False),
        df["pickup_latitude"].to_numpy(dtype=np.float64, copy=False),
        df["dropoff_longitude"].to_numpy(dtype=np.float64, copy=False),
        df["dropoff_latitude"].to_numpy(dtype=np.float64, copy=False),
    )
    df["distance_travel"] = dist
    return df




## === cell 3
def data_clean(df):
    pc = df["passenger_count"].to_numpy(copy=False)
    mask = pc > 0

    if "fare_amount" in df.columns:
        fa = df["fare_amount"].to_numpy(dtype=np.float64, copy=False)
        mask &= fa > 0.0

    if not mask.all():
        df = df.loc[mask]  # avoid .copy(); later ops assign a new column safely
    distance_travel(df)

    dist = df["distance_travel"].to_numpy(dtype=np.float64, copy=False)
    df = df.loc[dist > 0.0]
    return df




## === cell 4
def remove_outliers(df):
    dist = df["distance_travel"].to_numpy(dtype=np.float64, copy=False)
    mask = dist < 30.0
    if "fare_amount" in df.columns:
        fa = df["fare_amount"].to_numpy(dtype=np.float64, copy=False)
        mask &= fa < 60.0
    return df.loc[mask]




## === cell 5
def graph_present(df):
    test = df[df.passenger_count == 1]
    plot = test.iloc[: len(test)].plot.scatter("distance_travel", "fare_amount")
    plot = df.iloc[:100000].plot.scatter("distance_travel", "fare_amount")




## === cell 6
def data_preprocessing(df):
    df = data_clean(df)
    df = remove_outliers(df)
    return df




## === cell 7
if False:
    df = pd.read_csv("../input/train.csv", nrows=10_00_000)
    df = df = distance_travel(df)
    df = df[df.passenger_count == 1]
    df = df[df.distance_travel < 30]
    df.distance_travel.hist(bins=50, figsize=(12, 4))
    import matplotlib.pyplot as plt

    plt.xlabel("distance miles")
    plt.title("Histogram ride distances in miles")
    df.distance_travel.describe()



## === cell 8
if False:
    df = pd.read_csv("../input/train.csv", nrows=1_00_000)
    df = df = distance_travel(df)
    df = df[df.passenger_count <= 6]
    df = df[df.distance_travel < 30]
    df = df[df.fare_amount > 0]
    df = df[df.fare_amount < 60]
    df.fare_amount.hist(bins=50, figsize=(12, 4))
    import matplotlib.pyplot as plt

    plt.xlabel("fare_amount")
    df.fare_amount.describe()



## === cell 9
filename = r"../input/train.csv"
gen = chunck_generator(filename=filename)

regr = GradientBoostingRegressor(n_estimators=50, warm_start=True, random_state=42)
imp = Imputer(missing_values=np.nan, strategy="mean")

t = 56
trees_per_chunk = 5

imputer_fitted = False
min_rows_per_chunk_after_clean = 500


def _build_time_features(pickup_dt_series):
    dt = pd.to_datetime(pickup_dt_series, errors="coerce")
    hour = dt.dt.hour.astype("float64").to_numpy(copy=False)
    dow = dt.dt.dayofweek.astype("float64").to_numpy(copy=False)
    return hour, dow


def _build_X(dist, pcount, hour, dow):
    n = dist.shape[0]
    X = np.empty((n, 5), dtype=np.float64)
    X[:, 0] = dist
    X[:, 1] = pcount
    X[:, 2] = hour
    X[:, 3] = dow
    X[:, 4] = 1.0
    return X


while t > 0:
    print(t)
    df = next(gen)
    df = data_preprocessing(df)

    if len(df) < min_rows_per_chunk_after_clean:
        print(f"skip_chunk_too_small_after_clean: {len(df)}")
        continue

    dist = df["distance_travel"].to_numpy(dtype=np.float64, copy=False)
    pcount = df["passenger_count"].to_numpy(dtype=np.float64, copy=False)

    hour, dow = _build_time_features(df["pickup_datetime"])

    train_X = _build_X(dist, pcount, hour, dow)
    train_y = df["fare_amount"].to_numpy(dtype=np.float64, copy=False)

    if not imputer_fitted:
        imp = imp.fit(train_X)
        imputer_fitted = True

    train_X = imp.transform(train_X)

    regr.set_params(n_estimators=regr.n_estimators + trees_per_chunk)
    regr = regr.fit(train_X, train_y)

    try:
        print(regr.score(train_X, train_y))
    except Exception as e:
        print("score_failed:", e)

    t -= 1



## === cell 10
test_df = pd.read_csv(
    "../input/test.csv",
    nrows=10_00_000,
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    dtype=DTYPES_TEST_FAST,
    parse_dates=["pickup_datetime"],
    engine="c",
    memory_map=True,
)

distance_travel(test_df)
test_df.head()



## === cell 11
test_dist = test_df["distance_travel"].to_numpy(dtype=np.float64, copy=False)
test_pcount = test_df["passenger_count"].to_numpy(dtype=np.float64, copy=False)

test_hour, test_dow = _build_time_features(test_df["pickup_datetime"])

test_X = _build_X(test_dist, test_pcount, test_hour, test_dow)

test_X = imp.transform(test_X)
predicted_fare = regr.predict(test_X)

predicted_fare = np.clip(predicted_fare, 0.0, None)

print(predicted_fare)

my_submission = pd.DataFrame({"key": test_df.key, "fare_amount": predicted_fare})
my_submission.to_csv("submission.csv", index=False)
my_submission.head()
