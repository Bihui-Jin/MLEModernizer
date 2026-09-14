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
import math
import os

print(os.listdir("../input")[:20])



## === cell 1
types = {
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

cols = [
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]



## === cell 2
train = pd.read_csv(
    "../input/train.csv",
    nrows=1000000,
    usecols=cols,
    dtype=types,
    parse_dates=["pickup_datetime"],
)

test_cols = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
test_types = {
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
test = pd.read_csv(
    "../input/test.csv",
    usecols=test_cols,
    dtype=test_types,
    parse_dates=["pickup_datetime"],
)

samp = pd.read_csv("../input/sample_submission.csv")



## === cell 3
train.dropna(how="any", axis="rows", inplace=True)
m = (train.fare_amount > 0) & (train["passenger_count"] <= 6)
m &= (train.pickup_latitude > -90) & (train.pickup_latitude < 90)
m &= (train.dropoff_latitude > -90) & (train.dropoff_latitude < 90)
m &= (train.pickup_longitude > -180) & (train.pickup_longitude < 180)
m &= (train.dropoff_longitude > -180) & (train.dropoff_longitude < 180)
train = train[m]



## === cell 4
test.dropna(how="any", axis="rows", inplace=True)
m = test["passenger_count"] <= 6
m &= (test.pickup_latitude > -90) & (test.pickup_latitude < 90)
m &= (test.dropoff_latitude > -90) & (test.dropoff_latitude < 90)
m &= (test.pickup_longitude > -180) & (test.pickup_longitude < 180)
m &= (test.dropoff_longitude > -180) & (test.dropoff_longitude < 180)
test = test[m]



## === cell 5
y = train.fare_amount.values
test_id = test.key




## === cell 6
def week_num(day):
    """
    given the day of the month, return the week number of the month
    """
    if day <= 7:
        return "first"
    if (day > 7) and (day <= 14):
        return "second"
    if (day > 14) and (day <= 21):
        return "third"
    if (day > 21) and (day <= 28):
        return "fourth"
    return "fifth"




## === cell 7
def add_geo_features(data):
    plon = data["pickup_longitude"].to_numpy()
    plat = data["pickup_latitude"].to_numpy()
    dlon = data["dropoff_longitude"].to_numpy()
    dlat = data["dropoff_latitude"].to_numpy()

    abs_dlong = np.abs(dlon - plon)
    abs_dlat = np.abs(dlat - plat)

    data["abs_diff_longitude"] = abs_dlong
    data["abs_diff_latitude"] = abs_dlat
    data["manhattan_distance"] = abs_dlong + abs_dlat

    data["squared_long"] = abs_dlong * abs_dlong
    data["squared_lat"] = abs_dlat * abs_dlat

    return data




## === cell 8
def add_time_features(data):
    dt = data["pickup_datetime"]

    hour = dt.dt.hour.astype("int16")
    month = dt.dt.month.astype("int16")
    year = dt.dt.year.astype("int16")

    dow = dt.dt.dayofweek.to_numpy()
    day_names = np.array(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        dtype=object,
    )
    day_of_week = day_names[dow]

    day = dt.dt.day.to_numpy(dtype=np.int16, copy=False)
    week_of_month = np.where(
        day <= 7,
        "first",
        np.where(
            day <= 14,
            "second",
            np.where(day <= 21, "third", np.where(day <= 28, "fourth", "fifth")),
        ),
    )

    data["hour"] = pd.Categorical(hour.astype(str))
    data["day_of_week"] = pd.Categorical(day_of_week)
    data["week_of_month"] = pd.Categorical(week_of_month)
    data["month"] = pd.Categorical(month.astype(str))
    data["year"] = pd.Categorical(year.astype(str))

    return data




## === cell 9
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

features = [
    "passenger_count",
    "hour",
    "day_of_week",
    "week_of_month",
    "month",
    "year",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "manhattan_distance",
]

train_feat = train.drop(columns=["fare_amount"])
test_feat = test.drop(columns=["key"])

train_feat = add_time_features(train_feat)
train_feat = add_geo_features(train_feat)
train_feat = train_feat[features]

test_feat = add_time_features(test_feat)
test_feat = add_geo_features(test_feat)
test_feat = test_feat[features]

cat_cols = ["hour", "day_of_week", "week_of_month", "month", "year"]
num_cols = [
    "passenger_count",
    "abs_diff_longitude",
    "abs_diff_latitude",
    "manhattan_distance",
]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
        ("num", "passthrough", num_cols),
    ],
    remainder="drop",
    sparse_threshold=1.0,
)

x = preprocess.fit_transform(train_feat)
x_test = preprocess.transform(test_feat)

from scipy import sparse

if sparse.issparse(x):
    x = x.tocsr()
    if x.dtype != np.float32:
        x = x.astype(np.float32)
else:
    x = np.asarray(x, dtype=np.float32, order="C")

if sparse.issparse(x_test):
    x_test = x_test.tocsr()
    if x_test.dtype != np.float32:
        x_test = x_test.astype(np.float32)
else:
    x_test = np.asarray(x_test, dtype=np.float32, order="C")



## === cell 10
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_depth=18,
    min_samples_leaf=2,
)



## === cell 11
n_fit = min(x.shape[0], len(y))
model.fit(x[:n_fit], y[:n_fit])



## === cell 12
test_pred = model.predict(x_test)

sub = pd.DataFrame()
sub["key"] = test_id.values
sub["fare_amount"] = test_pred
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with", len(sub), "rows")
