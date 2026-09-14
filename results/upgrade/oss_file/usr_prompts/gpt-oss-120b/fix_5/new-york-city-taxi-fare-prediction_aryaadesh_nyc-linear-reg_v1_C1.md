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
from sklearn.metrics import mean_squared_error  # needed later

from sklearnex import patch_sklearn

patch_sklearn()

print(os.listdir("../input"))




## === cell 1
train_df = pd.read_csv("../input/train.csv", nrows=10_000_000)
train_df.dtypes




## === cell 2
train_df.head()




## === cell 3
test_df = pd.read_csv("../input/test.csv")
test_df.dtypes




## === cell 4
test_df.head()




## === cell 5
def add_travel_vector_features(df):
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


add_travel_vector_features(train_df)
add_travel_vector_features(test_df)




## === cell 6
print(train_df.isnull().sum())




## === cell 7
print(test_df.isnull().sum())




## === cell 8
print("Old size: %d" % len(train_df))
train_df = train_df.dropna(how="any", axis="rows")
print("New size: %d" % len(train_df))




## === cell 9
plot = train_df.iloc[:2000].plot.scatter("abs_diff_longitude", "abs_diff_latitude")




## === cell 10
print("Old size: %d" % len(train_df))
train_df = train_df[
    (train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)
]
print("New size: %d" % len(train_df))




## === cell 11
print("Old size: %d" % len(test_df))
test_df = test_df[
    (test_df.abs_diff_longitude < 5.0) & (test_df.abs_diff_latitude < 5.0)
]
print("New size: %d" % len(test_df))




## === cell 12
train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])
test_df["pickup_datetime"] = pd.to_datetime(test_df["pickup_datetime"])

train_df["pickup_time"] = (
    train_df["pickup_datetime"].dt.hour * 100 + train_df["pickup_datetime"].dt.minute
)
test_df["pickup_time"] = (
    test_df["pickup_datetime"].dt.hour * 100 + test_df["pickup_datetime"].dt.minute
)

train_df["Weekday"] = train_df["pickup_datetime"].dt.weekday
test_df["Weekday"] = test_df["pickup_datetime"].dt.weekday




## === cell 13
train_df.head()




## === cell 14
test_df.head()




## === cell 15
train_df.head()




## === cell 16
test_df.head()




## === cell 17
train_df.drop("pickup_datetime", inplace=True, axis=1)
test_df.drop("pickup_datetime", inplace=True, axis=1)




## === cell 18
train_df.head()




## === cell 19
test_df.head()




## === cell 20
train_one_hot = pd.get_dummies(train_df["Weekday"], prefix="Weekday")
test_one_hot = pd.get_dummies(test_df["Weekday"], prefix="Weekday")
train_df = pd.concat([train_df, train_one_hot], axis=1)
test_df = pd.concat([test_df, test_one_hot], axis=1)




## === cell 21
train_df.head()




## === cell 22
test_df.head()




## === cell 23
train_df.drop("Weekday", axis=1, inplace=True)
test_df.drop("Weekday", axis=1, inplace=True)




## === cell 24
train_df.head()




## === cell 25
test_df.head()




## === cell 26
R = 6373.0
lat1 = np.radians(train_df["pickup_latitude"])
lon1 = np.radians(train_df["pickup_longitude"])
lat2 = np.radians(train_df["dropoff_latitude"])
lon2 = np.radians(train_df["dropoff_longitude"])

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
train_df["Distance"] = distance * 0.621

lat1 = np.radians(test_df["pickup_latitude"])
lon1 = np.radians(test_df["pickup_longitude"])
lat2 = np.radians(test_df["dropoff_latitude"])
lon2 = np.radians(test_df["dropoff_longitude"])

dlon = lon2 - lon1
dlat = lat2 - lat1
a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
distance = R * c
test_df["Distance"] = distance * 0.621




## === cell 27
R = 6373.0
lat3 = np.radians(40.6413)
lon3 = np.radians(-73.7781)

lat1 = np.radians(train_df["pickup_latitude"])
lon1 = np.radians(train_df["pickup_longitude"])
lat2 = np.radians(train_df["dropoff_latitude"])
lon2 = np.radians(train_df["dropoff_longitude"])

dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
train_df["Pickup_Distance_airport"] = (R * c1) * 0.621

d_lon_dropoff = lon3 - lon2
d_lat_dropoff = lat3 - lat2
a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
train_df["Dropoff_Distance_airport"] = (R * c2) * 0.621

lat1 = np.radians(test_df["pickup_latitude"])
lon1 = np.radians(test_df["pickup_longitude"])
lat2 = np.radians(test_df["dropoff_latitude"])
lon2 = np.radians(test_df["dropoff_longitude"])

dlon_pickup = lon3 - lon1
dlat_pickup = lat3 - lat1
a1 = (
    np.sin(dlat_pickup / 2) ** 2
    + np.cos(lat1) * np.cos(lat3) * np.sin(dlon_pickup / 2) ** 2
)
c1 = 2 * np.arctan2(np.sqrt(a1), np.sqrt(1 - a1))
test_df["Pickup_Distance_airport"] = (R * c1) * 0.621

d_lon_dropoff = lon3 - lon2
d_lat_dropoff = lat3 - lat2
a2 = (
    np.sin(d_lat_dropoff / 2) ** 2
    + np.cos(lat2) * np.cos(lat3) * np.sin(d_lon_dropoff / 2) ** 2
)
c2 = 2 * np.arctan2(np.sqrt(a2), np.sqrt(1 - a2))
test_df["Dropoff_Distance_airport"] = (R * c2) * 0.621




## === cell 28
train_df["Distance"] = np.round(train_df["Distance"], 2)
train_df["Pickup_Distance_airport"] = np.round(train_df["Pickup_Distance_airport"], 2)
train_df["Dropoff_Distance_airport"] = np.round(train_df["Dropoff_Distance_airport"], 2)
test_df["Distance"] = np.round(test_df["Distance"], 2)
test_df["Pickup_Distance_airport"] = np.round(test_df["Pickup_Distance_airport"], 2)
test_df["Dropoff_Distance_airport"] = np.round(test_df["Dropoff_Distance_airport"], 2)




## === cell 29
train_df.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)
test_df.drop(
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
    axis=1,
    inplace=True,
)




## === cell 30
train_df["abs_diff_longitude"] = np.abs(train_df["abs_diff_longitude"]) - np.mean(
    train_df["abs_diff_longitude"]
)
train_df["abs_diff_longitude"] = train_df["abs_diff_longitude"] / np.var(
    train_df["abs_diff_longitude"]
)




## === cell 31
train_df["abs_diff_latitude"] = np.abs(train_df["abs_diff_latitude"]) - np.mean(
    train_df["abs_diff_latitude"]
)
train_df["abs_diff_latitude"] = train_df["abs_diff_latitude"] / np.var(
    train_df["abs_diff_latitude"]
)




## === cell 32
test_df["abs_diff_longitude"] = np.abs(test_df["abs_diff_longitude"]) - np.mean(
    test_df["abs_diff_longitude"]
)
test_df["abs_diff_longitude"] = test_df["abs_diff_longitude"] / np.var(
    test_df["abs_diff_longitude"]
)




## === cell 33
test_df["abs_diff_latitude"] = np.abs(test_df["abs_diff_latitude"]) - np.mean(
    test_df["abs_diff_latitude"]
)
test_df["abs_diff_latitude"] = test_df["abs_diff_latitude"] / np.var(
    test_df["abs_diff_latitude"]
)




## === cell 34
train_df.shape




## === cell 35
test_df.shape




## === cell 36
train_df.head()




## === cell 37
from sklearn.model_selection import train_test_split

X = train_df.drop(["key", "fare_amount"], axis=1)
y = train_df["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=80
)

X_train_np = X_train.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)
y_train_np = y_train.values.astype(np.float32)
y_test_np = y_test.values.astype(np.float32)




## === cell 38
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)
gbr.fit(X_train_np, y_train_np)
lr = gbr

pred_val = lr.predict(X_test_np)
rmse = mean_squared_error(y_test_np, pred_val, squared=False)
print(f"RMSE on validation split: {rmse:.4f}")




## === cell 39
test_features = test_df[X_train.columns].values.astype(np.float32)
pred = np.clip(lr.predict(test_features), a_min=0, a_max=None)
pred = np.round(pred, 2)




## === cell 40
pd.read_csv("../input/sample_submission.csv").head()




## === cell 41
Submission = pd.DataFrame({"key": test_df["key"], "fare_amount": pred})
Submission = Submission[["key", "fare_amount"]]




## === cell 42
Submission.set_index("key", inplace=True)




## === cell 43
Submission.to_csv("Submission.csv")
