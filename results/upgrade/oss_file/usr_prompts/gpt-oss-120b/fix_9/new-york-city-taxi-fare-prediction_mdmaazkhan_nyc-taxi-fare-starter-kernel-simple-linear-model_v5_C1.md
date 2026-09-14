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

3.10

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

5.01424

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

print("Available input files:", os.listdir("../input"))




## === cell 1
train_path = "../input/train.csv"
test_path = "../input/test.csv"

train_df = pd.read_csv(train_path, nrows=2_000_000)
test_df = pd.read_csv(test_path)




## === cell 2
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
    df["abs_diff_latitude"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()


add_travel_vector_features(train_df)
add_travel_vector_features(test_df)

train_df.dropna(inplace=True)

mask = (train_df["abs_diff_longitude"] < 5.0) & (train_df["abs_diff_latitude"] < 5.0)
train_df = train_df[mask]
test_df = test_df[
    (test_df["abs_diff_longitude"] < 5.0) & (test_df["abs_diff_latitude"] < 5.0)
]




## === cell 3
def extract_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"])
    df["pickup_time"] = dt.dt.hour * 100 + dt.dt.minute
    df["Weekday"] = dt.dt.weekday
    return df


train_df = extract_time_features(train_df)
test_df = extract_time_features(test_df)

weekday_map = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}
train_df["Weekday"].replace(weekday_map, inplace=True)
test_df["Weekday"].replace(weekday_map, inplace=True)

train_onehot = pd.get_dummies(train_df["Weekday"], prefix="weekday")
test_onehot = pd.get_dummies(test_df["Weekday"], prefix="weekday")
train_df = pd.concat([train_df, train_onehot], axis=1)
test_df = pd.concat([test_df, test_onehot], axis=1)

train_df.drop("Weekday", axis=1, inplace=True)
test_df.drop("Weekday", axis=1, inplace=True)




## === cell 4
def add_peak_flag(df):
    flags = []
    for t in df["pickup_time"]:
        if 700 < t < 1000 or 1600 < t < 2000:
            flags.append("peak")
        else:
            flags.append("not Peak")
    df["Peak_hour"] = flags
    return df


train_df = add_peak_flag(train_df)
test_df = add_peak_flag(test_df)

train_peak_onehot = pd.get_dummies(train_df["Peak_hour"], prefix="peak")
test_peak_onehot = pd.get_dummies(test_df["Peak_hour"], prefix="peak")
train_df = pd.concat([train_df, train_peak_onehot], axis=1)
test_df = pd.concat([test_df, test_peak_onehot], axis=1)

train_df.drop("Peak_hour", axis=1, inplace=True)
test_df.drop("Peak_hour", axis=1, inplace=True)




## === cell 5
R = 6373.0  # Earth radius in km


def haversine(df):
    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])
    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance_km = R * c
    return distance_km * 0.621  # miles


train_df["Distance"] = haversine(train_df).round(2)
test_df["Distance"] = haversine(test_df).round(2)




## === cell 6
JFK_LAT = np.radians(40.6413111)
JFK_LON = np.radians(-73.7781391)


def airport_distances(df):
    lat_pick = np.radians(df["pickup_latitude"])
    lon_pick = np.radians(df["pickup_longitude"])
    lat_drop = np.radians(df["dropoff_latitude"])
    lon_drop = np.radians(df["dropoff_longitude"])

    dlat_pick = JFK_LAT - lat_pick
    dlon_pick = JFK_LON - lon_pick
    a1 = (
        np.sin(dlat_pick / 2) ** 2
        + np.cos(lat_pick) * np.cos(JFK_LAT) * np.sin(dlon_pick / 2) ** 2
    )
    c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
    dist_pick = R * c1 * 0.621

    dlat_drop = JFK_LAT - lat_drop
    dlon_drop = JFK_LON - lon_drop
    a2 = (
        np.sin(dlat_drop / 2) ** 2
        + np.cos(lat_drop) * np.cos(JFK_LAT) * np.sin(dlon_drop / 2) ** 2
    )
    c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
    dist_drop = R * c2 * 0.621

    return dist_pick.round(2), dist_drop.round(2)


train_df["pickup_Distance_airport"], train_df["Dropoff_Distance_airport"] = (
    airport_distances(train_df)
)
test_df["pickup_Distance_airport"], test_df["Dropoff_Distance_airport"] = (
    airport_distances(test_df)
)




## === cell 7
train_df["passenger_count_squared"] = train_df["passenger_count"] ** 2
test_df["passenger_count_squared"] = test_df["passenger_count"] ** 2

X = train_df.drop(["key", "fare_amount", "pickup_datetime"], axis=1, errors="ignore")
y = train_df["fare_amount"]

X_test = test_df.drop(["key", "pickup_datetime"], axis=1, errors="ignore")
X_test = X_test.reindex(columns=X.columns, fill_value=0)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)




## === cell 8
lr = Ridge(alpha=1.0, random_state=42)
lr.fit(X_train_scaled, y_train)

val_pred = lr.predict(X_val_scaled)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE: {val_rmse:.4f}")




## === cell 9
test_pred = lr.predict(X_test_scaled)
test_pred = np.clip(test_pred, 0, None)

submission = pd.DataFrame({"key": test_df["key"], "fare_amount": test_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
