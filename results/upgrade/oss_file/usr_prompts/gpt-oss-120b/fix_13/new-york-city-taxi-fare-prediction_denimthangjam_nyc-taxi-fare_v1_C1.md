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

3.9

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
seaborn==0.12.2
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
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor  # new model




## === cell 1
train_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
test_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

train_data = pd.read_csv(train_path, nrows=10_000_000)
test_data = pd.read_csv(test_path)




## === cell 2
train_data["Difference_longitude"] = np.abs(
    train_data["pickup_longitude"] - train_data["dropoff_longitude"]
)
train_data["Difference_latitude"] = np.abs(
    train_data["pickup_latitude"] - train_data["dropoff_latitude"]
)

test_data["Difference_longitude"] = np.abs(
    test_data["pickup_longitude"] - test_data["dropoff_longitude"]
)
test_data["Difference_latitude"] = np.abs(
    test_data["pickup_latitude"] - test_data["dropoff_latitude"]
)




## === cell 3
print(f"Rows before dropna: {len(train_data)}")
train_data.dropna(inplace=True)
print(f"Rows after dropna: {len(train_data)}")

train_data = train_data[
    (train_data["fare_amount"] > 0) & (train_data["fare_amount"] < 300)
]

train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
]




## === cell 4
train_dt = pd.to_datetime(train_data["pickup_datetime"])
train_data["pickuptime"] = train_dt.dt.hour * 100 + train_dt.dt.minute
train_data["Hour"] = train_dt.dt.hour
train_data["Weekday"] = train_dt.dt.weekday  # 0=Monday

test_dt = pd.to_datetime(test_data["pickup_datetime"])
test_data["pickuptime"] = test_dt.dt.hour * 100 + test_dt.dt.minute
test_data["Hour"] = test_dt.dt.hour
test_data["Weekday"] = test_dt.dt.weekday




## === cell 5
train_data.drop(columns=["pickup_datetime"], inplace=True)
test_data.drop(columns=["pickup_datetime"], inplace=True)




## === cell 6
train_weekday_dummies = pd.get_dummies(train_data["Weekday"], prefix="wd")
test_weekday_dummies = pd.get_dummies(test_data["Weekday"], prefix="wd")
test_weekday_dummies = test_weekday_dummies.reindex(
    columns=train_weekday_dummies.columns, fill_value=0
)

train_hour_dummies = pd.get_dummies(train_data["Hour"], prefix="hr")
test_hour_dummies = pd.get_dummies(test_data["Hour"], prefix="hr")
test_hour_dummies = test_hour_dummies.reindex(
    columns=train_hour_dummies.columns, fill_value=0
)

train_data = pd.concat([train_data, train_weekday_dummies, train_hour_dummies], axis=1)
test_data = pd.concat([test_data, test_weekday_dummies, test_hour_dummies], axis=1)

train_data.drop(columns=["Weekday", "Hour"], inplace=True)
test_data.drop(columns=["Weekday", "Hour"], inplace=True)




## === cell 7
numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ["fare_amount", "key"]]

dummy_prefixes = ("wd_", "hr_")
numeric_cols = [
    c for c in numeric_cols if not any(c.startswith(p) for p in dummy_prefixes)
]

scaler = StandardScaler()
train_data[numeric_cols] = scaler.fit_transform(train_data[numeric_cols])
test_data[numeric_cols] = scaler.transform(test_data[numeric_cols])




## === cell 8
R = 6373.0  # Earth radius in km


def haversine(df):
    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c * 0.621  # convert km to miles


train_data["Distance"] = haversine(train_data)
test_data["Distance"] = haversine(test_data)

train_data["Distance_sq"] = train_data["Distance"] ** 2
test_data["Distance_sq"] = test_data["Distance"] ** 2

airport_lat = np.radians(40.6413111)
airport_lon = np.radians(-73.7781391)


def airport_distance(df, lat_col, lon_col, out_name):
    lat = np.radians(df[lat_col])
    lon = np.radians(df[lon_col])
    dlon = airport_lon - lon
    dlat = airport_lat - lat
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat) * np.cos(airport_lat) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df[out_name] = R * c * 0.621


airport_distance(
    train_data, "pickup_latitude", "pickup_longitude", "Pickup_Distance_airport"
)
airport_distance(
    train_data, "dropoff_latitude", "dropoff_longitude", "Dropoff_Distance_airport"
)
airport_distance(
    test_data, "pickup_latitude", "pickup_longitude", "Pickup_Distance_airport"
)
airport_distance(
    test_data, "dropoff_latitude", "dropoff_longitude", "Dropoff_Distance_airport"
)




## === cell 9
dist_cols = [
    "Distance",
    "Distance_sq",
    "Pickup_Distance_airport",
    "Dropoff_Distance_airport",
]
dist_scaler = StandardScaler()
train_data[dist_cols] = dist_scaler.fit_transform(train_data[dist_cols])
test_data[dist_cols] = dist_scaler.transform(test_data[dist_cols])




## === cell 10
X = train_data.drop(columns=["key", "fare_amount"])
y = train_data["fare_amount"]

y_log = np.log1p(y)

X_train, X_val, y_train_log, y_val_log = train_test_split(
    X, y_log, test_size=0.01, random_state=80
)

model = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_depth=7,
    random_state=42,
)

model.fit(X_train, y_train_log)

val_log_pred = model.predict(X_val)
val_pred = np.expm1(val_log_pred)

rmse = mean_squared_error(np.expm1(y_val_log), val_pred, squared=False)
print(f"Validation RMSE: {rmse:.4f}")

model.fit(X, y_log)




## === cell 11
test_features = test_data.drop(columns=["key"])
test_log_pred = model.predict(test_features)
test_pred = np.expm1(test_log_pred)  # keep full precision for better RMSE

submission = pd.DataFrame({"key": test_data["key"], "fare_amount": test_pred})
submission = submission[["key", "fare_amount"]]
submission.to_csv("submission.csv", index=False)
print("submission.csv written, shape:", submission.shape)
