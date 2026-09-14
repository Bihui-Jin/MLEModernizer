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
import matplotlib.pyplot as plt
import seaborn as sns  # for plot visualization

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt

import os

print(os.listdir("../input"))



## === cell 1
sns.set_style("darkgrid")



## === cell 2
TRAIN_NROWS = 1_000_000

test_dataset = pd.read_csv("../input/test.csv")
train_dataset = pd.read_csv("../input/train.csv", nrows=TRAIN_NROWS)



## === cell 3
train_dataset.head(5)



## === cell 4
train_dataset.tail(5)



## === cell 5
train_dataset.dtypes



## === cell 6
train_dataset.info(memory_usage="deep")



## === cell 7
for dtype in ["float", "int", "object"]:
    selected_dtype = train_dataset.select_dtypes(include=[dtype])
    mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
    mean_usage_mb = mean_usage_b / 1024**2
    print(
        "Average memory usage for {} columns: {:03.2f} MB".format(dtype, mean_usage_mb)
    )



## === cell 8
train_key = train_dataset["key"].copy()
test_key = test_dataset["key"].copy()



## === cell 9
train_dataset.info(memory_usage="deep")



## === cell 10
for ds in (train_dataset, test_dataset):
    ds["passenger_count"] = ds["passenger_count"].astype("uint8", errors="ignore")
    for col in [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]:
        ds[col] = ds[col].astype("float32", errors="ignore")
train_dataset["fare_amount"] = train_dataset["fare_amount"].astype(
    "float32", errors="ignore"
)



## === cell 11
train_dataset.info(memory_usage="deep")



## === cell 12
train_dataset.isnull().sum()



## === cell 13
len(train_dataset)



## === cell 14
print(f"Row count before drop-null operation - {train_dataset.shape[0]}")
train_dataset.dropna(inplace=True)
print(f"Row count after drop-null operation - {train_dataset.shape[0]}")



## === cell 15
train_dataset["pickup_datetime"] = pd.to_datetime(
    arg=train_dataset["pickup_datetime"], infer_datetime_format=True, errors="coerce"
)
test_dataset["pickup_datetime"] = pd.to_datetime(
    arg=test_dataset["pickup_datetime"], infer_datetime_format=True, errors="coerce"
)

before_dt = train_dataset.shape[0]
train_dataset = train_dataset[train_dataset["pickup_datetime"].notna()]
print("Dropped NaT pickup_datetime rows (train):", before_dt - train_dataset.shape[0])

test_before_dt = test_dataset.shape[0]
test_dataset = test_dataset[test_dataset["pickup_datetime"].notna()].copy()
print(
    "Dropped NaT pickup_datetime rows (test):", test_before_dt - test_dataset.shape[0]
)



## === cell 16
train_dataset.dtypes




## === cell 17
def add_new_date_time_features(dataset):
    dataset["hour"] = dataset.pickup_datetime.dt.hour
    dataset["day"] = dataset.pickup_datetime.dt.day
    dataset["month"] = dataset.pickup_datetime.dt.month
    dataset["year"] = dataset.pickup_datetime.dt.year
    dataset["day_of_week"] = dataset.pickup_datetime.dt.dayofweek
    return dataset


train_dataset = add_new_date_time_features(train_dataset)
test_dataset = add_new_date_time_features(test_dataset)



## === cell 18
train_dataset.describe()



## === cell 19
print(f"Rows before removing coordinate outliers - {train_dataset.shape[0]}")

nyc_long_min, nyc_long_max = -74.3, -72.9
nyc_lat_min, nyc_lat_max = 40.5, 41.0

train_dataset = train_dataset[
    train_dataset["pickup_longitude"].between(nyc_long_min, nyc_long_max)
    & train_dataset["dropoff_longitude"].between(nyc_long_min, nyc_long_max)
    & train_dataset["pickup_latitude"].between(nyc_lat_min, nyc_lat_max)
    & train_dataset["dropoff_latitude"].between(nyc_lat_min, nyc_lat_max)
]

print(f"Rows after removing coordinate outliers - {train_dataset.shape[0]}")



## === cell 20
train_dataset.describe()



## === cell 21
train_dataset.fare_amount[train_dataset.fare_amount <= 0].count()



## === cell 22
print(f"Row count before elimination - {train_dataset.shape[0]}")
train_dataset = train_dataset[train_dataset.fare_amount > 0]
print(f"Row count after elimination - {train_dataset.shape[0]}")



## === cell 23
print("Rows before fare upper-cap filter:", train_dataset.shape[0])
train_dataset = train_dataset[train_dataset["fare_amount"] <= 250.0]
print("Rows after fare upper-cap filter:", train_dataset.shape[0])



## === cell 24
train_dataset.passenger_count[
    (train_dataset.passenger_count < 1) | (train_dataset.passenger_count > 8)
].count()



## === cell 25
print(f"Row count before elimination - {train_dataset.shape[0]}")
train_dataset = train_dataset[train_dataset.passenger_count.between(1, 7)]
print(f"Row count after elimination - {train_dataset.shape[0]}")




## === cell 26
def degree_to_radion(degree):
    return degree * (np.pi / 180.0)


def calculate_distance(
    pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude
):
    from_lat = degree_to_radion(pickup_latitude)
    from_long = degree_to_radion(pickup_longitude)
    to_lat = degree_to_radion(dropoff_latitude)
    to_long = degree_to_radion(dropoff_longitude)

    radius = 6371.01  # km

    lat_diff = to_lat - from_lat
    long_diff = to_long - from_long

    a = (np.sin(lat_diff / 2.0) ** 2) + (
        np.cos(from_lat) * np.cos(to_lat) * (np.sin(long_diff / 2.0) ** 2)
    )
    a = np.clip(a, 0.0, 1.0)  # numerical safety
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return radius * c




## === cell 27
train_dataset["distance"] = calculate_distance(
    train_dataset.pickup_latitude,
    train_dataset.pickup_longitude,
    train_dataset.dropoff_latitude,
    train_dataset.dropoff_longitude,
)
test_dataset["distance"] = calculate_distance(
    test_dataset.pickup_latitude,
    test_dataset.pickup_longitude,
    test_dataset.dropoff_latitude,
    test_dataset.dropoff_longitude,
)



## === cell 28
train_dataset.sort_values(by="distance")



## === cell 29
train_dataset[(train_dataset.distance == 0)].count()



## === cell 30
train_dataset[
    (train_dataset.pickup_latitude != train_dataset.dropoff_latitude)
    & (train_dataset.pickup_longitude != train_dataset.dropoff_latitude)
    & (train_dataset.distance == 0)
].count()




## === cell 31
def add_distances_from_airport(dataset):
    jfk_coords = (40.639722, -73.778889)
    ewr_coords = (40.6925, -74.168611)
    lga_coords = (40.77725, -73.872611)

    dataset["pickup_jfk_distance"] = calculate_distance(
        jfk_coords[0], jfk_coords[1], dataset.pickup_latitude, dataset.pickup_longitude
    )
    dataset["dropof_jfk_distance"] = calculate_distance(
        jfk_coords[0],
        jfk_coords[1],
        dataset.dropoff_latitude,
        dataset.dropoff_longitude,
    )

    dataset["pickup_ewr_distance"] = calculate_distance(
        ewr_coords[0], ewr_coords[1], dataset.pickup_latitude, dataset.pickup_longitude
    )
    dataset["dropof_ewr_distance"] = calculate_distance(
        ewr_coords[0],
        ewr_coords[1],
        dataset.dropoff_latitude,
        dataset.dropoff_longitude,
    )

    dataset["pickup_lga_distance"] = calculate_distance(
        lga_coords[0], lga_coords[1], dataset.pickup_latitude, dataset.pickup_longitude
    )
    dataset["dropof_lga_distance"] = calculate_distance(
        lga_coords[0],
        lga_coords[1],
        dataset.dropoff_latitude,
        dataset.dropoff_longitude,
    )

    return dataset


train_dataset = add_distances_from_airport(train_dataset)
test_dataset = add_distances_from_airport(test_dataset)



## === cell 32
print("Rows before distance outlier filter:", train_dataset.shape[0])
train_dataset = train_dataset[train_dataset["distance"].between(0.0, 100.0)]
print("Rows after distance outlier filter:", train_dataset.shape[0])



## === cell 33
sns.distplot(train_dataset.fare_amount)



## === cell 34
sns.jointplot(x="distance", y="fare_amount", data=train_dataset)



## === cell 35
sns.countplot(x="day_of_week", data=train_dataset)



## === cell 36
tc = train_dataset.pivot_table(
    index="day_of_week", columns="month", values="fare_amount"
)
sns.heatmap(data=tc)



## === cell 37
selected_predictors = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "pickup_jfk_distance",
    "dropof_jfk_distance",
    "pickup_ewr_distance",
    "dropof_ewr_distance",
    "pickup_lga_distance",
    "dropof_lga_distance",
    "passenger_count",
    "hour",
    "day",
    "month",
    "year",
    "day_of_week",
    "distance",
]

X_df = train_dataset.loc[:, selected_predictors].copy()
y = train_dataset["fare_amount"].values
X_test_df = test_dataset.loc[:, selected_predictors].copy()

X_df = X_df.replace([np.inf, -np.inf], np.nan)
X_test_df = X_test_df.replace([np.inf, -np.inf], np.nan)

valid_mask = ~X_df.isna().any(axis=1) & np.isfinite(y)
X_df = X_df.loc[valid_mask]
y = y[valid_mask.values]

medians = X_df.median(numeric_only=True)
X_df = X_df.fillna(medians)
X_test_df = X_test_df.fillna(medians)

X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(medians)
X_test_df = X_test_df.replace([np.inf, -np.inf], np.nan).fillna(medians)

X = X_df.values
X_test_dataset = X_test_df.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 5, random_state=42
)
print(f"X_test size - {len(X_test)}")



## === cell 38
rand_forest_regressor = RandomForestRegressor(
    random_state=42, n_estimators=200, n_jobs=-1
)
rand_forest_regressor.fit(X_train, y_train)

y_rand_forest_predict = rand_forest_regressor.predict(X_test)
print(f"random forest size - {len(y_rand_forest_predict)}")
random_forest_model_error = sqrt(mean_squared_error(y_test, y_rand_forest_predict))
print(f" Random Forest RMSE - {random_forest_model_error}")



## === cell 39
XGB_regressor = XGBRegressor(random_state=42)
XGB_regressor.fit(X_train, y_train)

y_XGB_predict = XGB_regressor.predict(X_test)
XGB_model_error = sqrt(mean_squared_error(y_test, y_XGB_predict))
print(f" XGBoost RMSE - {XGB_model_error}")



## === cell 40
sns.barplot(
    y=list(train_dataset.loc[:, selected_predictors].columns),
    x=list(XGB_regressor.feature_importances_),
)



## === cell 41
selected_predictors = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "pickup_jfk_distance",
    "dropof_jfk_distance",
    "pickup_ewr_distance",
    "dropof_ewr_distance",
    "pickup_lga_distance",
    "dropof_lga_distance",
    "hour",
    "month",
    "year",
    "distance",
]

X_df = train_dataset.loc[:, selected_predictors].copy()
y = train_dataset["fare_amount"].values
X_test_df = test_dataset.loc[:, selected_predictors].copy()

X_df = X_df.replace([np.inf, -np.inf], np.nan)
X_test_df = X_test_df.replace([np.inf, -np.inf], np.nan)

valid_mask = ~X_df.isna().any(axis=1) & np.isfinite(y)
X_df = X_df.loc[valid_mask]
y = y[valid_mask.values]

medians = X_df.median(numeric_only=True)
X_df = X_df.fillna(medians)
X_test_df = X_test_df.fillna(medians)

X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(medians)
X_test_df = X_test_df.replace([np.inf, -np.inf], np.nan).fillna(medians)

X = X_df.values
X_test_dataset = X_test_df.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 5, random_state=42
)



## === cell 42
XGB_regressor = XGBRegressor(
    learning_rate=0.07, max_depth=6, n_estimators=300, random_state=42
)
XGB_regressor.fit(X_train, y_train)
y_XGB_predict = XGB_regressor.predict(X_test)

XGB_model_error = sqrt(mean_squared_error(y_test, y_XGB_predict))
print(f" XGBoost RMSE - {XGB_model_error}")



## === cell 43
sns.barplot(
    y=list(train_dataset.loc[:, selected_predictors].columns),
    x=list(XGB_regressor.feature_importances_),
)



## === cell 44
XGB_regressor_final = XGBRegressor(
    learning_rate=0.07, max_depth=6, n_estimators=300, random_state=42
)
XGB_regressor_final.fit(X, y)

y_XGB_predict = XGB_regressor_final.predict(X_test_dataset)

y_XGB_predict = np.where(np.isfinite(y_XGB_predict), y_XGB_predict, 0.0)
y_XGB_predict = np.maximum(y_XGB_predict, 0.0)

test_dataset["fare_amount"] = y_XGB_predict

submission = pd.DataFrame(
    {"key": test_key.loc[test_dataset.index].values, "fare_amount": y_XGB_predict}
)
submission.head(10)



## === cell 45
sns.jointplot(x="distance", y="fare_amount", data=test_dataset)



## === cell 46
test_dataset[test_dataset.distance > 100]



## === cell 47
test_dataset[test_dataset.fare_amount > 150]



## === cell 48
submission = submission[["key", "fare_amount"]].copy()
submission.to_csv("submission_new.csv", index=False)
print("Wrote submission_new.csv with shape:", submission.shape)
print("Unique keys:", submission["key"].nunique(), "Total rows:", len(submission))
