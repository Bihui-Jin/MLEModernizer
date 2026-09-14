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

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import os  # reading the input files we have access to

print(os.listdir("../input"))




## === cell 1
dtype_map = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_datetime": "object",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}
train_data = pd.read_csv(
    "../input/train.csv",
    nrows=10_000_000,
    dtype=dtype_map,
    usecols=list(dtype_map.keys()),
)
train_data.dtypes




## === cell 2
test_data = pd.read_csv("../input/test.csv")
test_data.head()




## === cell 3
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




## === cell 4
print(train_data.isnull().sum())




## === cell 5
print("Old size: %d" % len(train_data))
train_data = train_data.dropna(how="any", axis="rows")
print("New size: %d" % len(train_data))




## === cell 6
plot = train_data[:2000].plot.scatter("Difference_longitude", "Difference_latitude")




## === cell 7
print("Old size: %d" % len(train_data))
train_data = train_data[
    (train_data.Difference_longitude < 5.0) & (train_data.Difference_latitude < 5.0)
]
print("New size: %d" % len(train_data))




## === cell 8
train_dt_full = pd.to_datetime(train_data["pickup_datetime"])
train_data["pickuptime"] = train_dt_full.dt.strftime("%H:%M:%S")
test_dt_full = pd.to_datetime(test_data["pickup_datetime"])
test_data["pickuptime"] = test_dt_full.dt.strftime("%H:%M:%S")




## === cell 9
train_data.head()




## === cell 10
train_data["Weekday"] = pd.to_datetime(train_data["pickup_datetime"]).dt.weekday
test_data["Weekday"] = pd.to_datetime(test_data["pickup_datetime"]).dt.weekday




## === cell 11
train_data.head()




## === cell 12
train_data.drop("pickup_datetime", inplace=True, axis=1)
test_data.drop("pickup_datetime", inplace=True, axis=1)




## === cell 13
weekday_map = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}
train_data["Weekday"].replace(weekday_map, inplace=True)
test_data["Weekday"].replace(weekday_map, inplace=True)




## === cell 14
train_one_hot = pd.get_dummies(train_data["Weekday"])
test_one_hot = pd.get_dummies(test_data["Weekday"])
train_data = pd.concat([train_data, train_one_hot], axis=1)
test_data = pd.concat([test_data, test_one_hot], axis=1)




## === cell 15
train_data.drop("Weekday", axis=1, inplace=True)
test_data.drop("Weekday", axis=1, inplace=True)




## === cell 16
train_time_dt = pd.to_datetime(train_data["pickuptime"], format="%H:%M:%S")
train_data["pickuptime"] = train_time_dt.dt.hour * 100 + train_time_dt.dt.minute
test_time_dt = pd.to_datetime(test_data["pickuptime"], format="%H:%M:%S")
test_data["pickuptime"] = test_time_dt.dt.hour * 100 + test_time_dt.dt.minute




## === cell 17
R = 6373.0  # radius of earth in km
lat1 = np.radians(train_data["pickup_latitude"])
lon1 = np.radians(train_data["pickup_longitude"])
lat2 = np.radians(train_data["dropoff_latitude"])
lon2 = np.radians(train_data["dropoff_longitude"])

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
train_data["Distance"] = np.round(distance * 0.621, 2)  # convert to miles

lat1 = np.radians(test_data["pickup_latitude"])
lon1 = np.radians(test_data["pickup_longitude"])
lat2 = np.radians(test_data["dropoff_latitude"])
lon2 = np.radians(test_data["dropoff_longitude"])

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_data["Distance"] = np.round(distance * 0.621, 2)




## === cell 18
R = 6373.0
lat1 = np.radians(train_data["pickup_latitude"])
lon1 = np.radians(train_data["pickup_longitude"])
lat2 = np.radians(train_data["dropoff_latitude"])
lon2 = np.radians(train_data["dropoff_longitude"])

lat3 = np.full(len(train_data), np.radians(40.6413111))  # JFK latitude
lon3 = np.full(len(train_data), np.radians(-73.7781391))  # JFK longitude

dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
dlon_dropoff = lon3 - lon2
dlat_dropoff = lat3 - lat2

a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
train_data["Pickup_Distance_airport"] = np.round(distance1 * 0.621, 2)

a2 = (
    np.sin(dlat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(dlon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
train_data["Dropoff_Distance_airport"] = np.round(distance2 * 0.621, 2)


lat1 = np.radians(test_data["pickup_latitude"])
lon1 = np.radians(test_data["pickup_longitude"])
lat2 = np.radians(test_data["dropoff_latitude"])
lon2 = np.radians(test_data["dropoff_longitude"])

lat3 = np.full(len(test_data), np.radians(40.6413111))
lon3 = np.full(len(test_data), np.radians(-73.7781391))
dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
dlon_dropoff = lon3 - lon2
dlat_dropoff = lat3 - lat2

a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
distance1 = R * c1
test_data["Pickup_Distance_airport"] = np.round(distance1 * 0.621, 2)

a2 = (
    np.sin(dlat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(dlon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
distance2 = R * c2
test_data["Dropoff_Distance_airport"] = np.round(distance2 * 0.621, 2)




## === cell 19
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




## === cell 20
mean_long = np.mean(train_data["Difference_longitude"])
var_long = np.var(train_data["Difference_longitude"])
mean_lat = np.mean(train_data["Difference_latitude"])
var_lat = np.var(train_data["Difference_latitude"])

train_data["Difference_longitude"] = (
    np.abs(train_data["Difference_longitude"] - mean_long) / var_long
)
train_data["Difference_latitude"] = (
    np.abs(train_data["Difference_latitude"] - mean_lat) / var_lat
)

mean_long_test = np.mean(test_data["Difference_longitude"])
var_long_test = np.var(test_data["Difference_longitude"])
mean_lat_test = np.mean(test_data["Difference_latitude"])
var_lat_test = np.var(test_data["Difference_latitude"])

test_data["Difference_longitude"] = (
    np.abs(test_data["Difference_longitude"] - mean_long_test) / var_long_test
)
test_data["Difference_latitude"] = (
    np.abs(test_data["Difference_latitude"] - mean_lat_test) / var_lat_test
)




## === cell 21
from sklearn.model_selection import train_test_split

X = train_data.drop(["key", "fare_amount"], axis=1)
y = train_data["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)




## === cell 22
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)
model.fit(X_train, y_train)
print("R^2 on hold‑out set:", model.score(X_test, y_test))




## === cell 23
test_features = test_data.drop("key", axis=1).reindex(
    columns=X_train.columns, fill_value=0
)
pred = np.round(np.clip(model.predict(test_features), 0, None), 2)




## === cell 24
Submission = pd.DataFrame({"key": test_data["key"], "fare_amount": pred})
Submission = Submission[["key", "fare_amount"]]




## === cell 25
Submission.set_index("key", inplace=True)
Submission.to_csv("Submission.csv")
