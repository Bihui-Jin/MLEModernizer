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
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearnex import patch_sklearn

patch_sklearn()  # patches sklearn in‑place, preserving API and results

print(os.listdir("../input"))



## === cell 1
usecols = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtypes = {
    "key": "object",
    "fare_amount": "float32",
    "pickup_datetime": "object",  # parsed later with pandas to_datetime
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int8",
}

train = pd.read_csv(
    "../input/train.csv",
    nrows=200_000,
    usecols=usecols,
    dtype=dtypes,
)

test_usecols = [col for col in usecols if col != "fare_amount"]
test = pd.read_csv(
    "../input/test.csv",
    usecols=test_usecols,
    dtype={k: v for k, v in dtypes.items() if k != "fare_amount"},
)

samp = pd.read_csv("../input/sample_submission.csv")



## === cell 2
train = train.dropna(how="any", axis="rows")



## === cell 3
test.shape



## === cell 4
all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(["fare_amount"], axis=1, inplace=True)

y = pd.to_numeric(train["fare_amount"], errors="coerce")
median_fare = y.median()
y = y.fillna(median_fare).values
y = np.clip(y, 0, 200)
y = np.where(np.isfinite(y), y, median_fare)
y = y.astype(np.float32)

y_log = np.log1p(y)

n_train = len(train)
n_test = len(test)
test_id = test.key




## === cell 5
def week_num(day):
    """Return the week of the month as a categorical string."""
    if day <= 7:
        return "first"
    if day <= 14:
        return "second"
    if day <= 21:
        return "third"
    if day <= 28:
        return "fourth"
    return "fifth"




## === cell 6
def add_time_features(data):
    data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])

    data["hour"] = data["pickup_datetime"].dt.hour
    data["day_of_week"] = data["pickup_datetime"].dt.day_name()
    data["day_of_month"] = data["pickup_datetime"].dt.day
    data["week_of_month"] = data["day_of_month"].map(week_num)
    data["month"] = data["pickup_datetime"].dt.month.astype(str)
    data["year"] = data["pickup_datetime"].dt.year.astype(str)

    data.drop(["day_of_month", "pickup_datetime"], axis=1, inplace=True)

    return data




## === cell 7
def add_geo_features(data):
    data["abs_diff_longitude"] = (
        data["dropoff_longitude"] - data["pickup_longitude"]
    ).abs()
    data["abs_diff_latitude"] = (
        data["dropoff_latitude"] - data["pickup_latitude"]
    ).abs()

    data["manhattan_distance"] = data["abs_diff_longitude"] + data["abs_diff_latitude"]

    data["squared_long"] = np.power(data["abs_diff_longitude"], 2)
    data["squared_lat"] = np.power(data["abs_diff_latitude"], 2)

    data["euclid_disance"] = np.sqrt(data["squared_long"] + data["squared_lat"])

    lat1 = np.radians(data["pickup_latitude"])
    lat2 = np.radians(data["dropoff_latitude"])
    lon1 = np.radians(data["pickup_longitude"])
    lon2 = np.radians(data["dropoff_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    R = 6371.0  # Earth radius in kilometers
    data["haversine_distance"] = R * c

    data["log_haversine"] = np.log1p(data["haversine_distance"])
    data["distance_per_passenger"] = (
        data["haversine_distance"] / data["passenger_count"]
    )
    data["distance_per_passenger"] = data["distance_per_passenger"].replace(
        [np.inf, -np.inf], np.nan
    )
    data["distance_per_passenger"] = data["distance_per_passenger"].fillna(0)

    data["passenger_distance"] = data["haversine_distance"] * data["passenger_count"]

    return data




## === cell 8
all_data = add_time_features(all_data)



## === cell 9
all_data = add_geo_features(all_data)



## === cell 10
features = [
    "passenger_count",
    "hour",
    "day_of_week",
    "week_of_month",
    "month",
    "year",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "euclid_disance",
    "manhattan_distance",
    "haversine_distance",
    "log_haversine",
    "distance_per_passenger",
    "passenger_distance",
]

all_data = all_data[features]

all_data = pd.get_dummies(all_data, dtype=np.float32)

all_data = all_data.fillna(0)



## === cell 11
x = all_data[:n_train].astype(np.float32)
x_test = all_data[n_train:].astype(np.float32)



## === cell 12
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=2000,  # keep original tree count
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)



## === cell 13
model.fit(x, y_log)



## === cell 14
feat_imp = pd.DataFrame(model.feature_importances_, index=x.columns, columns=["imp"])
feat_imp.sort_values("imp", ascending=False, inplace=True)



## === cell 15
pred_log = model.predict(x_test)
pred = np.expm1(pred_log)  # inverse of log1p
test_pred = np.where(pred < 0, 0, pred)  # enforce non‑negative fares



## === cell 16
sub = pd.DataFrame({"key": test_id, "fare_amount": test_pred})
sub.to_csv("submission.csv", index=False)



## === cell 17
sub.head()
