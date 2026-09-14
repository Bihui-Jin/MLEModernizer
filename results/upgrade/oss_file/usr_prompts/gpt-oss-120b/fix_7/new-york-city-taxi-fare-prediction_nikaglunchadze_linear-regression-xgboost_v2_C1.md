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

3.12

# 3. Installed packages

geopandas==0.14.4
joblib==1.5.2
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
seaborn==0.12.2
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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O
import os
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]

dtype_spec = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}

ny_latitude_min, ny_latitude_max = 40.4772, 45.0153
ny_longitude_min, ny_longitude_max = -79.7624, -71.7517


def load_and_filter_train(path):
    """Load the massive CSV in chunks and apply all filters once."""
    chunks = []
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtype_spec,
        parse_dates=False,
        chunksize=1_000_000,
    ):
        chunk = chunk.dropna()
        mask = (
            (chunk["pickup_longitude"] >= ny_longitude_min)
            & (chunk["pickup_longitude"] <= ny_longitude_max)
            & (chunk["pickup_latitude"] >= ny_latitude_min)
            & (chunk["pickup_latitude"] <= ny_latitude_max)
            & (chunk["dropoff_longitude"] >= ny_longitude_min)
            & (chunk["dropoff_longitude"] <= ny_longitude_max)
            & (chunk["dropoff_latitude"] >= ny_latitude_min)
            & (chunk["dropoff_latitude"] <= ny_latitude_max)
            & (chunk["passenger_count"] >= 1)
            & (chunk["passenger_count"] <= 6)
        )
        chunks.append(chunk[mask])
    return pd.concat(chunks, ignore_index=True)


df = load_and_filter_train("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv")

test_usecols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_dtype = {k: v for k, v in dtype_spec.items() if k != "fare_amount"}

test_df = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    usecols=test_usecols,
    dtype=test_dtype,
    parse_dates=False,
)

df.dtypes




## === cell 1
df.describe()




## === cell 2
df.head()




## === cell 3
def filter_column(df, column, range_min, range_max):
    return df[(df[column] >= range_min) & (df[column] <= range_max)]


df = filter_column(df, "fare_amount", 1, 200)




## === cell 4
df[df["fare_amount"] > 200].describe()




## === cell 5
def refactor_datetime(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["year"] = df["pickup_datetime"].dt.year.astype("int16")
    df["month"] = df["pickup_datetime"].dt.month.astype("int8")
    df["day"] = df["pickup_datetime"].dt.day.astype("int8")
    df["weekday"] = df["pickup_datetime"].dt.weekday.astype("int8")
    df["hour"] = df["pickup_datetime"].dt.hour.astype("int8")
    df.drop(columns=["pickup_datetime"], inplace=True)


refactor_datetime(df)
refactor_datetime(test_df)
df.head()




## === cell 6
def haversine(p1, p2):
    """Fully vectorised haversine distance (kilometres). Accepts scalar refs."""
    lat1, lon1 = p1
    lat2, lon2 = p2

    lat1 = np.radians(lat1.astype(np.float32))
    lon1 = np.radians(lon1.astype(np.float32))

    if np.isscalar(lat2):
        lat2 = np.full_like(lat1, lat2, dtype=np.float32)
    else:
        lat2 = np.radians(lat2.astype(np.float32))

    if np.isscalar(lon2):
        lon2 = np.full_like(lon1, lon2, dtype=np.float32)
    else:
        lon2 = np.radians(lon2.astype(np.float32))

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    dist = 2 * np.arcsin(np.sqrt(a))
    km = 6367.0 * dist
    return km


ny_center = ("ny_center", (40.7128, -74.0060))
jfk_airport = ("jfk_airport", (40.6446, -73.7797))
lga_airport = ("lga_airport", (40.7733, -73.8718))
ewr_airport = ("ewr_airport", (40.6895, -74.1745))

locs = [ny_center, jfk_airport, lga_airport, ewr_airport]




## === cell 7
def insert_haversine_dists(df, locations):
    """Add distance columns for each supplied location."""
    for loc_name, (lat_ref, lon_ref) in locations:
        df[f"pickup_dist_to_{loc_name}"] = haversine(
            (df["pickup_latitude"], df["pickup_longitude"]), (lat_ref, lon_ref)
        )
        df[f"dropoff_dist_to_{loc_name}"] = haversine(
            (df["dropoff_latitude"], df["dropoff_longitude"]), (lat_ref, lon_ref)
        )
    df["ride_distance"] = haversine(
        (df["pickup_latitude"], df["pickup_longitude"]),
        (df["dropoff_latitude"], df["dropoff_longitude"]),
    )


insert_haversine_dists(df, locs)
insert_haversine_dists(test_df, locs)




## === cell 8
df = df[df["ride_distance"] > 0]
df.describe()




## === cell 9
from sklearn.model_selection import train_test_split

train_df, validation_df = train_test_split(df, test_size=0.2, random_state=SEED)




## === cell 10
features = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "year",
    "month",
    "day",
    "hour",
    "weekday",
    "ride_distance",
]
features += [f"pickup_dist_to_{x[0]}" for x in locs]
features += [f"dropoff_dist_to_{x[0]}" for x in locs]

fare_amount = "fare_amount"

train_features = train_df[features]
train_fare_amount = train_df[fare_amount]

validation_features = validation_df[features]
validation_fare_amount = validation_df[fare_amount]

train_features.info()




## === cell 11
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()




## === cell 12
def estimate_model(model, df):
    pass




## === cell 13
linear_model.fit(train_features, train_fare_amount)




## === cell 14
from sklearn.metrics import mean_squared_error

linear_predictions = linear_model.predict(validation_features)
rmse_linear = mean_squared_error(
    validation_fare_amount, linear_predictions, squared=False
)
print("Linear Validation RMSE:", rmse_linear)




## === cell 15
from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    objective="reg:squarederror",
    learning_rate=0.05,
    n_estimators=400,
    max_depth=10,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    max_bin=256,
    n_jobs=-1,
    random_state=SEED,
    tree_method="hist",
)

xgb_model.fit(
    train_features,
    train_fare_amount,
    eval_set=[(validation_features, validation_fare_amount)],
    early_stopping_rounds=50,
    verbose=False,
)

xgb_predictions_val = xgb_model.predict(validation_features)
val_rmse = mean_squared_error(
    validation_fare_amount, xgb_predictions_val, squared=False
)
print("XGB Validation RMSE (with early stopping):", val_rmse)




## === cell 16
xgb_predictions_train = xgb_model.predict(train_features)
train_rmse = mean_squared_error(train_fare_amount, xgb_predictions_train, squared=False)
print("XGB Training RMSE:", train_rmse)




## === cell 17
xgb_predictions_test = xgb_model.predict(test_df[features])
submission = pd.DataFrame({"key": test_df["key"], "fare_amount": xgb_predictions_test})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
