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
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print(os.listdir("../input"))




## === cell 1
use_cols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "key",
]
dtypes = {
    "fare_amount": np.float32,
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int8,
    "key": object,
}
train = pd.read_csv("../input/train.csv", usecols=use_cols, dtype=dtypes, nrows=2000000)
test = pd.read_csv(
    "../input/test.csv",
    usecols=[c for c in use_cols if c != "fare_amount"],
    dtype={k: v for k, v in dtypes.items() if k != "fare_amount"},
)
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
train = train.loc[~(train == 0).any(axis=1)]




## === cell 11
(train == 0).astype(int).sum()




## === cell 12
train.shape




## === cell 13
train.describe()




## === cell 14
train.describe()




## === cell 15
train.dtypes.value_counts()




## === cell 16
object_data = train.dtypes == object
categoricals = train.columns[object_data]
categoricals




## === cell 17
train.drop("key", axis=1, inplace=True)
train.head()




## === cell 18
import datetime as dt


def date_extraction(data):
    data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])
    data["year"] = data["pickup_datetime"].dt.year
    data["month"] = data["pickup_datetime"].dt.month
    data["weekday"] = data["pickup_datetime"].dt.weekday  # 0 = Monday
    data["hour"] = data["pickup_datetime"].dt.hour
    data.drop("pickup_datetime", axis=1, inplace=True)
    return data


date_extraction(train)




## === cell 19
train.head()




## === cell 20
date_extraction(test)
test.head()




## === cell 21
def long_lat_distance(x):
    x["Longitude_distance"] = np.radians(x["pickup_longitude"] - x["dropoff_longitude"])
    x["Latitude_distance"] = np.radians(x["pickup_latitude"] - x["dropoff_latitude"])
    x["distance_travelled/10e3"] = (
        (x["Longitude_distance"] ** 2 + x["Latitude_distance"] ** 2) ** 0.5
    ) * 1000
    return x




## === cell 22
for x in [train, test]:
    long_lat_distance(x)
train.head()




## === cell 23
def haversine(x):
    """
    Compute Haversine distance (km) between pickup and drop‑off points.
    """
    R = 6371.0  # Earth radius in km
    lat1 = np.radians(x["pickup_latitude"])
    lon1 = np.radians(x["pickup_longitude"])
    lat2 = np.radians(x["dropoff_latitude"])
    lon2 = np.radians(x["dropoff_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    x["haversine_km"] = R * c
    return x




## === cell 24
for x in [train, test]:
    haversine(x)
train.head()




## === cell 25
train.dtypes.value_counts()




## === cell 26
train.head()




## === cell 27
test.head()




## === cell 28
train.describe()




## === cell 29
print("Are there any nulls in the train data: ")
print(train.isnull().sum())

print("\nAre there any nulls in the test data: ")
print(test.isnull().sum())




## === cell 30
from sklearnex import patch_sklearn

patch_sklearn()  # enables faster RandomForest via oneAPI optimizations

train = train[train["fare_amount"] > 0].copy()
train["fare_amount"] = np.log1p(train["fare_amount"])

from sklearn.ensemble import RandomForestRegressor

feature_cols = [x for x in train.columns if x != "fare_amount"]
X = train[feature_cols].astype(
    np.float32
)  # float32 reduces memory & speeds up computation
y = train["fare_amount"].astype(np.float32)




## === cell 31
correlations = X.corrwith(y)
correlations = abs(correlations * 100)
correlations.sort_values(ascending=False, inplace=True)

correlations




## === cell 32
ax = correlations.plot(kind="bar")
ax.set(ylim=[-1, 1], ylabel="pearson correlation")




## === cell 33
train.head()




## === cell 34
train["haversine_km"] = train["haversine_km"].round(2)
train["distance_travelled/10e3"] = train["distance_travelled/10e3"].round(2)

train.head()




## === cell 35
train.describe()




## === cell 36
test.head()




## === cell 37
from sklearn.model_selection import train_test_split

feat_cols = [x for x in train.columns if x != "fare_amount"]
X_np = train[feat_cols].astype(np.float32).values
y_np = train["fare_amount"].astype(np.float32).values

X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_np, y_np, test_size=0.25, random_state=42
)




## === cell 38
rf = RandomForestRegressor(
    n_estimators=800, max_features="sqrt", random_state=42, n_jobs=-1
)
rf = rf.fit(X_train_np, y_train_np)




## === cell 39
from sklearn.metrics import mean_squared_error

val_pred_log = rf.predict(X_test_np)
val_pred = np.expm1(val_pred_log)  # back to original fare scale
val_true = np.expm1(y_test_np)
rmse = np.sqrt(mean_squared_error(val_true, val_pred))
print(f"Validation RMSE (log‑target back‑transformed): {rmse:.5f}")




## === cell 40
test.head()




## === cell 41
test_features = test.drop("key", axis=1).astype(np.float32)




## === cell 42
final_prediction = np.expm1(rf.predict(test_features.values))

NYCtaxiFare_submission = pd.DataFrame(
    {"key": test["key"], "fare_amount": final_prediction}
)
NYCtaxiFare_submission.to_csv("NYCtaxiFare_prediction.csv", index=False)




## === cell 43
NYCtaxiFare_submission.head()
