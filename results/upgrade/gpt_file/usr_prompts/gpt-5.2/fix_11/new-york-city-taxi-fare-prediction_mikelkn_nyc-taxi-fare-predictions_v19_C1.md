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
import pandas as pd  # data processing
import os

INPUT_DIR = "../input"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "/kaggle/input"

print("Listing input dir:", INPUT_DIR)
print(os.listdir(INPUT_DIR))



## === cell 1
TRAIN_PATH = f"{INPUT_DIR}/train.csv"
TEST_PATH = f"{INPUT_DIR}/test.csv"

train_full = pd.read_csv(TRAIN_PATH, parse_dates=["pickup_datetime"])
test = pd.read_csv(TEST_PATH)

train_full = train_full.dropna(subset=["pickup_datetime"]).sort_values(
    "pickup_datetime"
)
n_target = 1_000_000
if len(train_full) > n_target:
    idx = np.linspace(0, len(train_full) - 1, n_target).round().astype(np.int64)
    train = train_full.iloc[idx].copy()
else:
    train = train_full.copy()

del train_full
train.head()



## === cell 2
test.head()



## === cell 3
train.shape



## === cell 4
test.shape



## === cell 5
train.dtypes.value_counts()



## === cell 6
test.dtypes.value_counts()



## === cell 7
train.isnull().sum()



## === cell 8
train = train.dropna()
train.isnull().sum()



## === cell 9
(train == 0).astype(int).sum()



## === cell 10
(train == 0).astype(int).sum()



## === cell 11
train.shape



## === cell 12
train.describe()



## === cell 13
train.describe()



## === cell 14
train.dtypes.value_counts()



## === cell 15
object_data = train.dtypes == object
categoricals = train.columns[object_data]
categoricals



## === cell 16
train.head()



## === cell 17
import datetime as dt


def date_extraction(data):
    data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"], errors="coerce")
    data["year"] = data["pickup_datetime"].dt.year
    data["month"] = data["pickup_datetime"].dt.month
    data["weekday"] = data["pickup_datetime"].dt.weekday
    data["hour"] = data["pickup_datetime"].dt.hour
    data["is_night"] = ((data["hour"] <= 6) | (data["hour"] >= 20)).astype(np.int8)
    data.drop("pickup_datetime", axis=1, inplace=True)
    return data


train = date_extraction(train)



## === cell 18
train.head()



## === cell 19
test = date_extraction(test)
test.head()




## === cell 20
def long_lat_distance(x):
    dlon = np.radians(x["pickup_longitude"] - x["dropoff_longitude"])
    dlat = np.radians(x["pickup_latitude"] - x["dropoff_latitude"])

    x["Longitude_distance"] = dlon
    x["Latitude_distance"] = dlat

    lat_avg = np.radians((x["pickup_latitude"] + x["dropoff_latitude"]) / 2.0)
    r_km = 6371.0
    x["distance_travelled/10e3"] = r_km * np.sqrt(
        (dlon * np.cos(lat_avg)) ** 2 + dlat**2
    )
    return x




## === cell 21
train = long_lat_distance(train)
test = long_lat_distance(test)

train.head()




## === cell 22
def harvesine(x):
    r = 6371000  # meters
    theta_1 = np.radians(x["pickup_latitude"])
    theta_2 = np.radians(x["dropoff_latitude"])
    lambda_1 = np.radians(x["pickup_longitude"])
    lambda_2 = np.radians(x["dropoff_longitude"])

    theta_diff = theta_2 - theta_1
    lambda_diff = lambda_2 - lambda_1

    a = (
        np.sin(theta_diff / 2) ** 2
        + np.cos(theta_1) * np.cos(theta_2) * np.sin(lambda_diff / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    x["harvesine/km"] = (r * c) / 1000.0
    return x




## === cell 23
train = harvesine(train)
test = harvesine(test)

train.head()



## === cell 24
train["manhattan_dist"] = (
    train["pickup_longitude"] - train["dropoff_longitude"]
).abs() + (train["pickup_latitude"] - train["dropoff_latitude"]).abs()
test["manhattan_dist"] = (
    test["pickup_longitude"] - test["dropoff_longitude"]
).abs() + (test["pickup_latitude"] - test["dropoff_latitude"]).abs()

train["abs_lon_diff"] = (train["pickup_longitude"] - train["dropoff_longitude"]).abs()
train["abs_lat_diff"] = (train["pickup_latitude"] - train["dropoff_latitude"]).abs()
test["abs_lon_diff"] = (test["pickup_longitude"] - test["dropoff_longitude"]).abs()
test["abs_lat_diff"] = (test["pickup_latitude"] - test["dropoff_latitude"]).abs()

train["abs_hour_diff"] = (train["hour"] - 12).abs()
test["abs_hour_diff"] = (test["hour"] - 12).abs()

train.dtypes.value_counts()



## === cell 25
train.head()



## === cell 26
test.head()



## === cell 27
train.describe()



## === cell 28
print("Are there any nulls\nan in the train data: ")
print(train.isnull().sum())

print("\nAre there any nulls\nans in the test data: ")
print(test.isnull().sum())



## === cell 29
datetime_int_cols = ["year", "month", "weekday", "hour", "is_night"]
for c in datetime_int_cols:
    if c in train.columns:
        train[c] = train[c].astype("Int64")
    if c in test.columns:
        test[c] = test[c].astype("Int64")

for df in (train, test):
    for c in [
        "harvesine/km",
        "distance_travelled/10e3",
        "manhattan_dist",
        "abs_lon_diff",
        "abs_lat_diff",
        "abs_hour_diff",
    ]:
        if c in df.columns:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)

median_fill_cols = [
    "harvesine/km",
    "manhattan_dist",
    "abs_lon_diff",
    "abs_lat_diff",
    "distance_travelled/10e3",
    "abs_hour_diff",
    "year",
    "month",
    "weekday",
    "hour",
]
for col in median_fill_cols:
    if col in train.columns:
        med = train[col].median()
        train[col] = train[col].fillna(med)
        if col in test.columns:
            test[col] = test[col].fillna(med)

if "is_night" in train.columns:
    night_mode = (
        int(train["is_night"].mode().iloc[0]) if train["is_night"].notna().any() else 0
    )
    train["is_night"] = train["is_night"].fillna(night_mode)
    if "is_night" in test.columns:
        test["is_night"] = test["is_night"].fillna(night_mode)

for c in datetime_int_cols:
    if c in train.columns:
        train[c] = train[c].astype(np.int16)
    if c in test.columns:
        test[c] = test[c].astype(np.int16)



## === cell 30
train = train[(train["fare_amount"] > 0) & (train["fare_amount"] <= 250)].copy()
train = train[(train["passenger_count"] >= 1) & (train["passenger_count"] <= 6)].copy()

train = train[
    (train["pickup_longitude"].between(-74.5, -72.8))
    & (train["dropoff_longitude"].between(-74.5, -72.8))
    & (train["pickup_latitude"].between(40.5, 41.8))
    & (train["dropoff_latitude"].between(40.5, 41.8))
].copy()

train = train[
    (train["pickup_longitude"] != 0)
    & (train["dropoff_longitude"] != 0)
    & (train["pickup_latitude"] != 0)
    & (train["dropoff_latitude"] != 0)
].copy()

train = train[(train["harvesine/km"] >= 0) & (train["harvesine/km"] <= 200)].copy()
train = train[~((train["harvesine/km"] < 0.05) & (train["fare_amount"] > 50))].copy()
train = train[~((train["harvesine/km"] > 50) & (train["fare_amount"] < 2.5))].copy()

eps = 1e-3
fare_per_km = train["fare_amount"] / (train["harvesine/km"] + eps)
train = train[(fare_per_km <= 200) & (fare_per_km >= 0)].copy()
train = train[train["harvesine/km"] <= 80].copy()

if "distance_travelled/10e3" in train.columns:
    train["distance_travelled/10e3"] = train["distance_travelled/10e3"].clip(0, 80)
if "distance_travelled/10e3" in test.columns:
    test["distance_travelled/10e3"] = test["distance_travelled/10e3"].clip(0, 80)

train.shape



## === cell 31
valid_test = (
    (test["passenger_count"].between(1, 6))
    & (test["pickup_longitude"].between(-74.5, -72.8))
    & (test["dropoff_longitude"].between(-74.5, -72.8))
    & (test["pickup_latitude"].between(40.5, 41.8))
    & (test["dropoff_latitude"].between(40.5, 41.8))
    & (test["pickup_longitude"] != 0)
    & (test["dropoff_longitude"] != 0)
    & (test["pickup_latitude"] != 0)
    & (test["dropoff_latitude"] != 0)
    & (test["harvesine/km"].between(0, 200))
)

feature_impute_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "Longitude_distance",
    "Latitude_distance",
    "distance_travelled/10e3",
    "harvesine/km",
    "manhattan_dist",
    "abs_lon_diff",
    "abs_lat_diff",
    "abs_hour_diff",
    "year",
    "month",
    "weekday",
    "hour",
    "is_night",
]
for col in feature_impute_cols:
    if col in test.columns and col in train.columns:
        fill_val = (
            train[col].median() if col != "is_night" else int(train[col].mode().iloc[0])
        )
        test.loc[~valid_test, col] = fill_val



## === cell 32
from sklearn.ensemble import RandomForestRegressor

feature_cols = [x for x in train.columns if x not in ["fare_amount", "key"]]
X = train[feature_cols]
y = train["fare_amount"]



## === cell 33
correlations = X.corrwith(y)
correlations = abs(correlations * 100)
correlations.sort_values(ascending=False, inplace=True)

correlations



## === cell 34
ax = correlations.plot(kind="bar")
ax.set(ylim=[-1, 1], ylabel="pearson correlation")



## === cell 35
train.head()



## === cell 36
train_1 = train.drop(
    [
        "pickup_longitude",
        "dropoff_longitude",
        "pickup_latitude",
        "dropoff_latitude",
        "Longitude_distance",
        "Latitude_distance",
    ],
    axis=1,
)

train_1.head()



## === cell 37
train_1.head()



## === cell 38
train_1.describe()



## === cell 39
test_1 = test.drop(
    [
        "pickup_longitude",
        "dropoff_longitude",
        "pickup_latitude",
        "dropoff_latitude",
        "Longitude_distance",
        "Latitude_distance",
    ],
    axis=1,
)

test_1.head()



## === cell 40
from sklearn.model_selection import train_test_split

feat_cols = [x for x in train_1.columns if x not in ["fare_amount", "key"]]
X_1 = train_1[feat_cols]
y_1 = train_1["fare_amount"]

y_1_log = np.log1p(y_1)

X_1 = X_1.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X_1, y_1_log, test_size=0.25, random_state=42
)



## === cell 41
rf = RandomForestRegressor(n_estimators=100, max_features=5, random_state=42, n_jobs=-1)
rf = rf.fit(X_train, y_train)



## === cell 42
test.head()



## === cell 43
final_prediction = rf.predict(X_test)



## === cell 44
test_features = test_1.reindex(columns=feat_cols).copy()
for c in feat_cols:
    if c not in test_features.columns:
        test_features[c] = train_1[c].median()
test_features = test_features[feat_cols].astype(np.float32)

test_features.head()



## === cell 45
final_prediction = rf.predict(test_features)
final_prediction = np.expm1(final_prediction)

final_prediction = np.clip(final_prediction, 0.0, 250.0)

NYCtaxiFare_submission = pd.DataFrame(
    {"key": test["key"].values, "fare_amount": final_prediction}
)
NYCtaxiFare_submission.to_csv("NYCtaxiFare_prediction.csv", index=False)



## === cell 46
NYCtaxiFare_submission.head()
