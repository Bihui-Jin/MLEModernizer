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

5.68059

# 6. Current score

None

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 939.03151) has done: 'I removed the deprecated `normalize=True` argument from `LinearRegression`, and ensured the test features are aligned to the training columns before prediction so the model can run without errors and produce a proper `Submission.csv` file.'
- What this solution (achieved 938.78532) has done: 'The fix adds the missing imports (`pandas` and `numpy`) at the start of the notebook, adjusts the cell numbering to start from 1, and keeps the original pipeline unchanged so the model can train and a proper Submission.csv file is written.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np

train_data = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv")
test_data = pd.read_csv("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv")



## === cell 1
train_data["Difference_longitude"] = np.abs(
    np.asarray(train_data["pickup_longitude"] - train_data["dropoff_longitude"])
)
train_data["Difference_latitude"] = np.abs(
    np.asarray(train_data["pickup_latitude"] - train_data["dropoff_latitude"])
)

test_data["Difference_longitude"] = np.abs(
    np.asarray(test_data["pickup_longitude"] - test_data["dropoff_longitude"])
)
test_data["Difference_latitude"] = np.abs(
    np.asarray(test_data["pickup_latitude"] - test_data["dropoff_latitude"])
)



## === cell 2
print(f"Before Dropping null values: {len(train_data)}")
train_data.dropna(inplace=True)
print(f"After Dropping null values: {len(train_data)}")



## === cell 3
train_data = train_data[
    (train_data["Difference_longitude"] < 5.0)
    & (train_data["Difference_latitude"] < 5.0)
]




## === cell 4
def extract_time(s):
    return s[11:-7]


train_data["pickuptime"] = train_data["pickup_datetime"].apply(extract_time)
test_data["pickuptime"] = test_data["pickup_datetime"].apply(extract_time)




## === cell 5
def weekday_from_ts(s):
    ts = pd.Timestamp(s[:-4])  # drop seconds and timezone
    return ts.weekday()


train_data["Weekday"] = train_data["pickup_datetime"].apply(weekday_from_ts)
test_data["Weekday"] = test_data["pickup_datetime"].apply(weekday_from_ts)



## === cell 6
train_data.drop("pickup_datetime", axis=1, inplace=True)
test_data.drop("pickup_datetime", axis=1, inplace=True)



## === cell 7
weekday_names = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
train_data["Weekday"] = train_data["Weekday"].replace(
    {i: name for i, name in enumerate(weekday_names)}
)
test_data["Weekday"] = test_data["Weekday"].replace(
    {i: name for i, name in enumerate(weekday_names)}
)



## === cell 8
train_one_hot = pd.get_dummies(train_data["Weekday"])
test_one_hot = pd.get_dummies(test_data["Weekday"])
train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)



## === cell 9
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)




## === cell 10
def hhmm_to_int(t):
    h, m = t.split(":")
    return int(h) * 100 + int(m)


train_data["pickuptime"] = train_data["pickuptime"].apply(hhmm_to_int)
test_data["pickuptime"] = test_data["pickuptime"].apply(hhmm_to_int)



## === cell 11
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


train_data["Distance"] = haversine(train_data).round(2)
test_data["Distance"] = haversine(test_data).round(2)



## === cell 12
airport_lat = np.radians(40.6413111)
airport_lon = np.radians(-73.7781391)


def airport_distance(df, lat_col, lon_col):
    lat = np.radians(df[lat_col])
    lon = np.radians(df[lon_col])
    dlon = airport_lon - lon
    dlat = airport_lat - lat
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat) * np.cos(airport_lat) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return (R * c * 0.621).round(2)


train_data["Pickup_Distance_airport"] = airport_distance(
    train_data, "pickup_latitude", "pickup_longitude"
)
train_data["Dropoff_Distance_airport"] = airport_distance(
    train_data, "dropoff_latitude", "dropoff_longitude"
)
test_data["Pickup_Distance_airport"] = airport_distance(
    test_data, "pickup_latitude", "pickup_longitude"
)
test_data["Dropoff_Distance_airport"] = airport_distance(
    test_data, "dropoff_latitude", "dropoff_longitude"
)



## === cell 13
train_data.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)
test_data.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)



## === cell 14
X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]
mask = y > 0
X = X[mask]
y = y[mask]

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=80)



## === cell 15
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

from sklearn.linear_model import Ridge

ridge = Ridge(alpha=0.1, random_state=80)
ridge.fit(X_train_scaled, y_train_log)

val_pred_log = ridge.predict(X_val_scaled)
val_pred = np.expm1(val_pred_log)

from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f"Validation RMSE (original scale): {rmse:.4f}")



## === cell 16
test_features = test_data.drop("key", axis=1).reindex(
    columns=X_val.columns, fill_value=0
)
test_features_scaled = scaler.transform(test_features)

test_pred_log = ridge.predict(test_features_scaled)
test_pred = np.expm1(test_pred_log)

test_pred = np.clip(test_pred, a_min=0, a_max=None)
test_pred = np.round(test_pred, 2)



## === cell 17
submission = pd.DataFrame({"key": test_data["key"], "fare_amount": test_pred})
submission.to_csv("submission.csv", index=False)
