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
xgboost==2.0.3

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
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os

print(os.listdir("../input"))



## === cell 1
TRAIN_PATH = "../input/train.csv"
N_SAMPLED = 1_000_000
RANDOM_STATE = 42


def read_reservoir_sample_csv(path, n_sample, seed=42, chunksize=250_000):
    rng = np.random.default_rng(seed)
    reservoir = None
    seen = 0

    for chunk in pd.read_csv(
        path, chunksize=chunksize, parse_dates=["pickup_datetime"]
    ):
        if reservoir is None:
            reservoir = chunk.iloc[:0].copy()

        for _, row in chunk.iterrows():
            seen += 1
            if len(reservoir) < n_sample:
                reservoir.loc[len(reservoir)] = row
            else:
                j = int(rng.integers(0, seen))
                if j < n_sample:
                    reservoir.iloc[j] = row

    reservoir = reservoir.reset_index(drop=True)
    return reservoir


train_df = read_reservoir_sample_csv(
    TRAIN_PATH, N_SAMPLED, seed=RANDOM_STATE, chunksize=250_000
)
train_df.dtypes



## === cell 2
print(train_df.isnull().sum())



## === cell 3
train_df = train_df.dropna(how="any", axis="rows")



## === cell 4
train_df.head()



## === cell 5
train_df.iloc[:1000].plot.scatter("pickup_longitude", "pickup_latitude")
train_df.iloc[:1000].plot.scatter("dropoff_longitude", "dropoff_latitude")

train_df.describe()




## === cell 6
def clean_df(df):
    return df[
        (df.fare_amount > 0)
        & (df.fare_amount <= 250)
        & (df.pickup_longitude > -75.5)
        & (df.pickup_longitude < -72.5)
        & (df.pickup_latitude > 40.0)
        & (df.pickup_latitude < 41.8)
        & (df.dropoff_longitude > -75.5)
        & (df.dropoff_longitude < -72.5)
        & (df.dropoff_latitude > 40.0)
        & (df.dropoff_latitude < 41.8)
        & (df.passenger_count > 0)
        & (df.passenger_count < 10)
    ]


train_df = clean_df(train_df)
print(len(train_df))




## === cell 7
def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    R_earth = 6371
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(
        np.radians, [pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]
    )
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon / 2.0) ** 2
    )

    return 2 * R_earth * np.arcsin(np.sqrt(a))


def add_datetime_info(dataset):
    dataset["pickup_datetime"] = pd.to_datetime(
        dataset["pickup_datetime"], errors="coerce"
    )

    dataset["hour"] = dataset.pickup_datetime.dt.hour
    dataset["day"] = dataset.pickup_datetime.dt.day
    dataset["month"] = dataset.pickup_datetime.dt.month
    dataset["weekday"] = dataset.pickup_datetime.dt.weekday

    return dataset


def bearing(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    lat1 = np.radians(pickup_lat)
    lat2 = np.radians(dropoff_lat)
    dlon = np.radians(dropoff_lon - pickup_lon)
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return np.arctan2(y, x)


def add_geo_features(df):
    df["distance"] = sphere_dist(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    df["abs_lon_diff"] = (df["pickup_longitude"] - df["dropoff_longitude"]).abs()
    df["abs_lat_diff"] = (df["pickup_latitude"] - df["dropoff_latitude"]).abs()
    df["distance_sq"] = df["distance"] ** 2

    df["distance_manhattan"] = df["abs_lon_diff"] + df["abs_lat_diff"]
    df["distance_manhattan_sq"] = df["distance_manhattan"] ** 2

    df["bearing"] = bearing(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )

    JFK_LAT, JFK_LON = 40.6413, -73.7781
    LGA_LAT, LGA_LON = 40.7769, -73.8740

    df["pickup_dist_jfk"] = sphere_dist(
        df["pickup_latitude"], df["pickup_longitude"], JFK_LAT, JFK_LON
    )
    df["dropoff_dist_jfk"] = sphere_dist(
        df["dropoff_latitude"], df["dropoff_longitude"], JFK_LAT, JFK_LON
    )
    df["pickup_dist_lga"] = sphere_dist(
        df["pickup_latitude"], df["pickup_longitude"], LGA_LAT, LGA_LON
    )
    df["dropoff_dist_lga"] = sphere_dist(
        df["dropoff_latitude"], df["dropoff_longitude"], LGA_LAT, LGA_LON
    )
    df["airport_min_dist"] = np.minimum.reduce(
        [
            df["pickup_dist_jfk"],
            df["dropoff_dist_jfk"],
            df["pickup_dist_lga"],
            df["dropoff_dist_lga"],
        ]
    )

    return df


train_df = add_geo_features(train_df)
train_df = add_datetime_info(train_df)

train_df = train_df[(train_df["distance"] >= 0) & (train_df["distance"] <= 100)]
train_df = train_df.dropna(subset=["pickup_datetime"])

train_df.head()



## === cell 8
train_df.drop(columns=["key"], inplace=True)
train_df.head()



## === cell 9
y = train_df["fare_amount"]
X = train_df.drop(columns=["fare_amount"])

X = X.sort_values("pickup_datetime")
y = y.loc[X.index]

split_idx = int(len(X) * 0.8)
x_train = X.iloc[:split_idx].drop(columns=["pickup_datetime"])
y_train = y.iloc[:split_idx]
x_test = X.iloc[split_idx:].drop(columns=["pickup_datetime"])
y_test = y.iloc[split_idx:]




## === cell 10
def XGBmodel(x_train, x_test, y_train, y_test):
    y_train_log = np.log1p(y_train.values)
    y_test_log = np.log1p(y_test.values)

    matrix_train = xgb.DMatrix(x_train, label=y_train_log)
    matrix_test = xgb.DMatrix(x_test, label=y_test_log)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1.0,
        "lambda": 1.0,
        "alpha": 0.0,
        "seed": RANDOM_STATE,
        "nthread": max(1, os.cpu_count() or 1),
    }

    model = xgb.train(
        params=params,
        dtrain=matrix_train,
        num_boost_round=600,
        early_stopping_rounds=30,
        evals=[(matrix_test, "valid")],
        verbose_eval=False,
    )
    return model


model = XGBmodel(x_train, x_test, y_train, y_test)



## === cell 11
test_df = pd.read_csv("../input/test.csv", parse_dates=["pickup_datetime"])

test_df = add_geo_features(test_df)
test_df = add_datetime_info(test_df)

test_key = test_df["key"]
x_pred = test_df.drop(columns=["key", "pickup_datetime"])

dtest = xgb.DMatrix(x_pred)

if hasattr(model, "best_iteration") and model.best_iteration is not None:
    pred_log = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
else:
    pred_log = model.predict(dtest)

prediction = np.expm1(pred_log)
prediction = np.clip(prediction, 0.0, None)

submission = pd.DataFrame({"key": test_key, "fare_amount": prediction})
submission.to_csv("taxi_fare_submission.csv", index=False)
submission
