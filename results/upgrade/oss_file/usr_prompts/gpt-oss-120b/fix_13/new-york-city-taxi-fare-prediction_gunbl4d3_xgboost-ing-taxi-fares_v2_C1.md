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
dtypes = {
    "fare_amount": "float32",
    "pickup_datetime": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
usecols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
train_df = pd.read_csv(
    "../input/train.csv",
    nrows=6_000_000,
    dtype=dtypes,
    usecols=usecols,
)




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
        & (df.pickup_longitude > -80)
        & (df.pickup_longitude < -70)
        & (df.pickup_latitude > 35)
        & (df.pickup_latitude < 45)
        & (df.dropoff_longitude > -80)
        & (df.dropoff_longitude < -70)
        & (df.dropoff_latitude > 35)
        & (df.dropoff_latitude < 45)
        & (df.passenger_count > 0)
        & (df.passenger_count < 10)
    ]


train_df = clean_df(train_df)
print(len(train_df))




## === cell 7
def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    R_earth = 6371.0
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
    dataset["pickup_datetime"] = pd.to_datetime(dataset["pickup_datetime"])
    dataset["hour"] = dataset.pickup_datetime.dt.hour.astype(np.float32)
    dataset["day"] = dataset.pickup_datetime.dt.day.astype(np.float32)
    dataset["month"] = dataset.pickup_datetime.dt.month.astype(np.float32)
    dataset["weekday"] = dataset.pickup_datetime.dt.weekday.astype(np.float32)
    dataset["hour_sin"] = np.sin(2 * np.pi * dataset["hour"] / 24).astype(np.float32)
    dataset["hour_cos"] = np.cos(2 * np.pi * dataset["hour"] / 24).astype(np.float32)
    dataset["month_sin"] = np.sin(2 * np.pi * dataset["month"] / 12).astype(np.float32)
    dataset["month_cos"] = np.cos(2 * np.pi * dataset["month"] / 12).astype(np.float32)
    dataset["day_sin"] = np.sin(2 * np.pi * dataset["day"] / 31).astype(np.float32)
    dataset["day_cos"] = np.cos(2 * np.pi * dataset["day"] / 31).astype(np.float32)
    dataset["weekday_sin"] = np.sin(2 * np.pi * dataset["weekday"] / 7).astype(
        np.float32
    )
    dataset["weekday_cos"] = np.cos(2 * np.pi * dataset["weekday"] / 7).astype(
        np.float32
    )
    return dataset


train_df["distance"] = sphere_dist(
    train_df["pickup_latitude"],
    train_df["pickup_longitude"],
    train_df["dropoff_latitude"],
    train_df["dropoff_longitude"],
).astype(np.float32)

train_df["log_distance"] = np.log1p(train_df["distance"]).astype(np.float32)
train_df = add_datetime_info(train_df)

train_df["distance_sq"] = (train_df["distance"] ** 2).astype(np.float32)
train_df["hour_distance"] = (train_df["hour"] * train_df["distance"]).astype(np.float32)
train_df["dist_passenger"] = (
    train_df["distance"] * train_df["passenger_count"]
).astype(np.float32)

train_df.head()




## === cell 8
train_df.drop(columns=["pickup_datetime"], inplace=True)




## === cell 9
y_raw = train_df["fare_amount"]
y = np.log1p(y_raw).astype(np.float32)

train = train_df.drop(columns=["fare_amount"])

x_train, x_test, y_train, y_test = train_test_split(
    train, y, random_state=0, test_size=0.2
)




## === cell 10
def XGBmodel(x_train, x_test, y_train, y_test):
    x_train_f32 = x_train.astype(np.float32, copy=False)
    x_test_f32 = x_test.astype(np.float32, copy=False)
    y_train_f32 = y_train.astype(np.float32, copy=False)
    y_test_f32 = y_test.astype(np.float32, copy=False)

    matrix_train = xgb.DMatrix(x_train_f32, label=y_train_f32)
    matrix_test = xgb.DMatrix(x_test_f32, label=y_test_f32)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.02,
        "max_depth": 9,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1.0,
        "seed": 0,
        "tree_method": "hist",
        "max_bin": 256,
        "nthread": os.cpu_count(),
        "verbosity": 0,
    }

    model = xgb.train(
        params=params,
        dtrain=matrix_train,
        num_boost_round=2000,
        early_stopping_rounds=200,
        evals=[(matrix_test, "validation")],
        verbose_eval=False,
    )
    return model, matrix_test


model, matrix_test = XGBmodel(x_train, x_test, y_train, y_test)

val_pred_log = model.predict(matrix_test)
val_pred = np.expm1(val_pred_log)

val_rmse = np.sqrt(((np.expm1(y_test) - val_pred) ** 2).mean())
print(f"Validation RMSE (raw fare): {val_rmse:.4f}")

coeffs = np.polyfit(val_pred, np.expm1(y_test), 1)  # coeffs[0]*pred + coeffs[1]
print(f"Bias correction coefficients: slope={coeffs[0]:.5f}, intercept={coeffs[1]:.5f}")




## === cell 11
test_df = pd.read_csv("../input/test.csv", dtype=dtypes)
test_df["distance"] = sphere_dist(
    test_df["pickup_latitude"],
    test_df["pickup_longitude"],
    test_df["dropoff_latitude"],
    test_df["dropoff_longitude"],
).astype(np.float32)
test_df["log_distance"] = np.log1p(test_df["distance"]).astype(np.float32)
test_df = add_datetime_info(test_df)
test_df["distance_sq"] = (test_df["distance"] ** 2).astype(np.float32)
test_df["hour_distance"] = (test_df["hour"] * test_df["distance"]).astype(np.float32)
test_df["dist_passenger"] = (test_df["distance"] * test_df["passenger_count"]).astype(
    np.float32
)

test_key = test_df["key"]
x_pred = test_df.drop(columns=["key", "pickup_datetime"])




## === cell 12
matrix_pred = xgb.DMatrix(x_pred.astype(np.float32, copy=False))
prediction_log = model.predict(matrix_pred)
prediction = np.expm1(prediction_log)

prediction = coeffs[0] * prediction + coeffs[1]
prediction = np.clip(prediction, a_min=0, a_max=None)

submission = pd.DataFrame({"key": test_key, "fare_amount": prediction.round(2)})
submission.to_csv("taxi_fare_submission.csv", index=False)
submission
