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
lightgbm==4.6.0
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



## === cell 1
use_cols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtype_map = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
train_data = pd.read_csv(
    "../input/train.csv",
    usecols=use_cols,
    dtype=dtype_map,
    nrows=2000000,
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
)
test_data = pd.read_csv(
    "../input/test.csv",
    usecols=[c for c in use_cols if c != "fare_amount"],
    dtype={k: v for k, v in dtype_map.items() if k != "fare_amount"},
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
)
train_data.head()
test_data.head()



## === cell 2
train_data.describe()



## === cell 3
train_data.shape




## === cell 4
def changeDataType(dataset):
    dataset["passenger_count"] = dataset.passenger_count.astype("uint8")
    dataset["pickup_longitude"] = dataset.pickup_longitude.astype("float32")
    dataset["pickup_latitude"] = dataset.pickup_latitude.astype("float32")
    dataset["dropoff_longitude"] = dataset.dropoff_longitude.astype("float32")
    dataset["dropoff_latitude"] = dataset.dropoff_latitude.astype("float32")
    dataset["pickup_datetime"] = pd.to_datetime(
        arg=dataset["pickup_datetime"], format="%Y-%m-%d %H:%M:%S UTC"
    )
    dataset.info()


changeDataType(train_data)
print("--" * 40)
changeDataType(test_data)

train_data["fare_amount"] = train_data.fare_amount.astype("float32")



## === cell 5
train_data["pickup_datetime"].head()



## === cell 6
train_data.describe()



## === cell 7
train_data.isnull().sum()
train_data = train_data.dropna(axis=0)
train_data.isnull().sum()



## === cell 8
pd.set_option("float_format", "{:f}".format)
train_data.describe()



## === cell 9
pass



## === cell 10
pass



## === cell 11
p = pd.cut(train_data.fare_amount, 3)
p.value_counts()



## === cell 12
train_data = train_data[train_data.fare_amount < 400]



## === cell 13
pass



## === cell 14
pass



## === cell 15
train_data.passenger_count.describe()
train_data = train_data[train_data.passenger_count <= 6]



## === cell 16
pass



## === cell 17
train_data.describe()



## === cell 18
train_data = train_data[train_data["pickup_latitude"].between(-90, 90)]
train_data = train_data[train_data["pickup_longitude"].between(-180, 180)]
train_data = train_data[train_data["dropoff_latitude"].between(-90, 90)]
train_data = train_data[train_data["dropoff_longitude"].between(-180, 180)]



## === cell 19
train_data = train_data[
    train_data.pickup_latitude.between(
        test_data.pickup_latitude.min(), test_data.pickup_latitude.max()
    )
]
train_data = train_data[
    train_data.pickup_longitude.between(
        test_data.pickup_longitude.min(), test_data.pickup_longitude.max()
    )
]
train_data = train_data[
    train_data.dropoff_latitude.between(
        test_data.dropoff_latitude.min(), test_data.dropoff_latitude.max()
    )
]
train_data = train_data[
    train_data.dropoff_longitude.between(
        test_data.dropoff_longitude.min(), test_data.dropoff_longitude.max()
    )
]



## === cell 20
pass




## === cell 21
def degree_to_rad(degree):
    """Convert degrees to radians."""
    return degree * (np.pi / 180)


def calculate_distance(
    pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude
):
    """Haversine distance in kilometers."""
    from_lat = degree_to_rad(pickup_latitude)
    from_long = degree_to_rad(pickup_longitude)
    to_lat = degree_to_rad(dropoff_latitude)
    to_long = degree_to_rad(dropoff_longitude)

    radius = 6371.01  # Earth radius in km
    lat_diff = to_lat - from_lat
    long_diff = to_long - from_long

    a = (
        np.sin(lat_diff / 2) ** 2
        + np.cos(from_lat) * np.cos(to_lat) * np.sin(long_diff / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return radius * c




## === cell 22
train_data["distance"] = calculate_distance(
    train_data.pickup_latitude,
    train_data.pickup_longitude,
    train_data.dropoff_latitude,
    train_data.dropoff_longitude,
)



## === cell 23
train_data.describe()



## === cell 24
test_data["distance"] = calculate_distance(
    test_data.pickup_latitude,
    test_data.pickup_longitude,
    test_data.dropoff_latitude,
    test_data.dropoff_longitude,
)



## === cell 25
test_data.describe()



## === cell 26
p = pd.cut(train_data.distance, 10)
p.value_counts()



## === cell 27
train_data = train_data.loc[train_data.distance < 200]  # keep reasonable trips



## === cell 28
train_data.describe()



## === cell 29
pass



## === cell 30
train_data = train_data.drop(columns="key")



## === cell 31
train_data.describe()



## === cell 32
test_data_key = test_data["key"]
original_test_key = test_data_key.copy()  # keep a full copy for final submission
test_data = test_data.drop(columns="key")



## === cell 33
test_data.head()



## === cell 34
data = [train_data, test_data]
for i in data:
    i["Year"] = i["pickup_datetime"].dt.year
    i["Month"] = i["pickup_datetime"].dt.month
    i["Date"] = i["pickup_datetime"].dt.day
    i["Day of Week"] = i["pickup_datetime"].dt.dayofweek
    i["Hour"] = i["pickup_datetime"].dt.hour
    i["passenger_distance"] = i["passenger_count"] * i["distance"]
    i["log_distance"] = np.log1p(i["distance"])
    i["hour_sin"] = np.sin(2 * np.pi * i["Hour"] / 24)
    i["hour_cos"] = np.cos(2 * np.pi * i["Hour"] / 24)



## === cell 35
train_data.head()



## === cell 36
test_data.head()



## === cell 37
pass



## === cell 38
pass



## === cell 39
pass



## === cell 40
train_data.describe()



## === cell 41
train_data[(train_data.distance > 100) & (train_data.fare_amount < 50)]



## === cell 42
pass



## === cell 43
train_data.groupby(["Month", "Year"]).count()["fare_amount"]



## === cell 44
pass



## === cell 45
pass



## === cell 46
pass



## === cell 47
pass



## === cell 48
pass



## === cell 49
pass



## === cell 50
train_data = train_data.loc[train_data.pickup_latitude != 0]
train_data = train_data.loc[train_data.pickup_longitude != 0]
train_data = train_data.loc[train_data.dropoff_latitude != 0]
train_data = train_data.loc[train_data.dropoff_longitude != 0]



## === cell 51
test_data = test_data.loc[test_data.pickup_latitude != 0]
test_data = test_data.loc[test_data.pickup_longitude != 0]
test_data = test_data.loc[test_data.dropoff_latitude != 0]
test_data = test_data.loc[test_data.dropoff_longitude != 0]
test_data_key = test_data_key.loc[test_data.index]



## === cell 52
train_data = train_data.drop(columns="pickup_datetime", axis=1)
test_data = test_data.drop(columns="pickup_datetime", axis=1)



## === cell 53
X = train_data.loc[:, train_data.columns != "fare_amount"]
y = train_data["fare_amount"]



## === cell 54
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)



## === cell 55
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

rf = RandomForestRegressor(max_depth=200, n_estimators=50, n_jobs=-1, random_state=0)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_val)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_pred))
print("RandomForest RMSE:", rf_rmse)



## === cell 56
import lightgbm as lgb

model_lgb = lgb.LGBMRegressor(
    objective="regression",
    num_leaves=1023,  # larger leaf count for more complexity
    learning_rate=0.01,  # lower learning rate for better generalisation
    n_estimators=1000,  # more trees
    max_bin=255,
    bagging_fraction=0.8,
    bagging_freq=5,
    feature_fraction=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
model_lgb.fit(X_train, y_train)
lgb_pred = model_lgb.predict(X_val)
lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_pred))
print("LightGBM RMSE:", lgb_rmse)



## === cell 57
best_model = model_lgb if lgb_rmse < rf_rmse else rf



## === cell 58
best_model.fit(X, y)



## === cell 59
test_preds = best_model.predict(test_data)



## === cell 60
train_mean_fare = y.mean()
full_submission = pd.DataFrame({"key": original_test_key, "fare_amount": np.nan})
full_submission.loc[full_submission["key"].isin(test_data_key), "fare_amount"] = (
    test_preds
)
full_submission["fare_amount"].fillna(train_mean_fare, inplace=True)
full_submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
full_submission.head()
