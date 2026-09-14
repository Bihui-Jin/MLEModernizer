# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

5.69073

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 761.76305) has done: 'Implemented missing imports, corrected data loading, added robust preprocessing (null handling, outlier filtering, haversine distance, airport distances, hour and weekday extraction, one‑hot encoding), aligned train‑test feature columns, trained a simple LinearRegression model, and generated a valid `submission.csv` with the required `key` and `fare_amount` columns.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

train_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
test_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

train_df = pd.read_csv(train_path, nrows=10_000_000)
test_df = pd.read_csv(test_path)



## === cell 1
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

train_df = train_df[train_df["fare_amount"] > 0].copy()

train_df["abs_diff_longitude"] = (
    train_df["dropoff_longitude"] - train_df["pickup_longitude"]
).abs()
train_df["abs_diff_latitude"] = (
    train_df["dropoff_latitude"] - train_df["pickup_latitude"]
).abs()
test_df["abs_diff_longitude"] = (
    test_df["dropoff_longitude"] - test_df["pickup_longitude"]
).abs()
test_df["abs_diff_latitude"] = (
    test_df["dropoff_latitude"] - test_df["pickup_latitude"]
).abs()

mask = (train_df["abs_diff_longitude"] < 5.0) & (train_df["abs_diff_latitude"] < 5.0)
train_df = train_df[mask]



## === cell 2
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
    km = R * c
    return km * 0.621371  # miles


train_df["Distance"] = haversine(train_df).round(2)
test_df["Distance"] = haversine(test_df).round(2)


def airport_distance(df):
    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])
    lat_air = np.radians(40.6413111)
    lon_air = np.radians(-73.7781391)

    dlon_p = lon_air - lon1
    dlat_p = lat_air - lat1
    a1 = (
        np.sin(dlat_p / 2) ** 2
        + np.cos(lat1) * np.cos(lat_air) * np.sin(dlon_p / 2) ** 2
    )
    c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
    dist_pickup = R * c1 * 0.621371

    dlon_d = lon_air - lon2
    dlat_d = lat_air - lat2
    a2 = (
        np.sin(dlat_d / 2) ** 2
        + np.cos(lat2) * np.cos(lat_air) * np.sin(dlon_d / 2) ** 2
    )
    c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
    dist_dropoff = R * c2 * 0.621371

    return dist_pickup.round(2), dist_dropoff.round(2)


train_df["Pickup_Distance_airport"], train_df["Dropoff_Distance_airport"] = (
    airport_distance(train_df)
)
test_df["Pickup_Distance_airport"], test_df["Dropoff_Distance_airport"] = (
    airport_distance(test_df)
)



## === cell 3
train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])

train_df["pickuptime"] = (
    train_df["pickup_datetime"].dt.hour * 100 + train_df["pickup_datetime"].dt.minute
)
test_df["pickuptime"] = (
    test_df["pickup_datetime"].dt.hour * 100 + test_df["pickup_datetime"].dt.minute
)

train_df["Weekday"] = train_df["pickup_datetime"].dt.dayofweek
test_df["Weekday"] = test_df["pickup_datetime"].dt.dayofweek

train_week_dummies = pd.get_dummies(train_df["Weekday"], prefix="wd")
test_week_dummies = pd.get_dummies(test_df["Weekday"], prefix="wd")
train_week_dummies, test_week_dummies = train_week_dummies.align(
    test_week_dummies, join="outer", axis=1, fill_value=0
)

train_df = pd.concat([train_df, train_week_dummies], axis=1)
test_df = pd.concat([test_df, test_week_dummies], axis=1)

train_df.drop(["pickup_datetime", "Weekday"], axis=1, inplace=True)
test_df.drop(["pickup_datetime", "Weekday"], axis=1, inplace=True)



## === cell 4
drop_cols = [
    "key",
    "fare_amount",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
]
X = train_df.drop(columns=drop_cols)
y = train_df["fare_amount"]

y_log = np.log1p(y)

X_train, X_val, y_train_log, y_val = train_test_split(
    X, y_log, test_size=0.01, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)



## === cell 5
model = Ridge(alpha=1.0, random_state=42)
model.fit(X_train_scaled, y_train_log)

val_log_pred = model.predict(X_val_scaled)
val_pred = np.expm1(val_log_pred)

rmse = mean_squared_error(np.expm1(y_val), val_pred, squared=False)
print(f"Validation RMSE (after log‑transform and scaling): {rmse:.4f}")



## === cell 6
test_features = test_df[X.columns]  # keep same column order
test_features_scaled = scaler.transform(test_features)

test_log_pred = model.predict(test_features_scaled)
test_pred = np.round(np.expm1(test_log_pred), 2)

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})
submission.to_csv("submission.csv", index=False)
print("submission.csv written, rows:", len(submission))
