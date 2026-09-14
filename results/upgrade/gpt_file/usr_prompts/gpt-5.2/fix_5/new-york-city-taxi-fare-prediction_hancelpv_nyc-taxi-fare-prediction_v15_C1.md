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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0

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

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

print(os.listdir("../input"))

np.random.seed(42)



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
    nrows=1_000_000,
    usecols=cols,
    dtype=types,
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
    low_memory=False,
)
test = pd.read_csv(
    "../input/test.csv",
    usecols=[
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
    dtype={
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "uint8",
    },
    parse_dates=["pickup_datetime"],
    infer_datetime_format=True,
    low_memory=False,
)
samp = pd.read_csv("../input/sample_submission.csv")



## === cell 3
train.dropna(how="any", axis="rows", inplace=True)
train = train[train.fare_amount > 0]
train = train[train["passenger_count"] <= 6]



## === cell 4
latitude_mask_pickup = (train.pickup_latitude > -90) & (train.pickup_latitude < 90)
train = train[latitude_mask_pickup]

latitude_mask_dropoff = (train.dropoff_latitude > -90) & (train.dropoff_latitude < 90)
train = train[latitude_mask_dropoff]



## === cell 5
longitude_mask_pickup = (train.pickup_longitude > -180) & (train.pickup_longitude < 180)
train = train[longitude_mask_pickup]

longitude_mask_dropoff = (train.dropoff_longitude > -180) & (
    train.dropoff_longitude < 180
)
train = train[longitude_mask_dropoff]



## === cell 6
y = train.fare_amount.values
n_train = len(train)
n_test = len(test)
test_id = test.key

train_feat = train.drop(["fare_amount"], axis=1)
all_data = pd.concat(
    (train_feat, test.drop(columns=["key"])),
    axis=0,
    ignore_index=True,
    copy=False,
)




## === cell 7
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




## === cell 8
def add_time_features(data):
    dt = pd.to_datetime(data["pickup_datetime"], errors="coerce", utc=False)
    data["pickup_datetime"] = dt

    data["hour"] = dt.dt.hour.astype("Int16")
    data["day_of_week"] = dt.dt.day_name()  # already categorical-like strings

    dom = dt.dt.day
    bins = [0, 7, 14, 21, 28, 31]
    labels = ["first", "second", "third", "fourth", "fifth"]
    data["week_of_month"] = pd.cut(
        dom, bins=bins, labels=labels, include_lowest=True, right=True
    )

    data["month"] = dt.dt.month.astype("Int16")
    data["year"] = dt.dt.year.astype("Int16")

    data["hour"] = data["hour"].astype("category")
    data["month"] = data["month"].astype("category")
    data["year"] = data["year"].astype("category")
    data["day_of_week"] = data["day_of_week"].astype("category")
    data["week_of_month"] = data["week_of_month"].astype("category")

    return data




## === cell 9
def add_geo_features(data):
    plon = data["pickup_longitude"].astype(np.float32, copy=False)
    plat = data["pickup_latitude"].astype(np.float32, copy=False)
    dlon = data["dropoff_longitude"].astype(np.float32, copy=False)
    dlat = data["dropoff_latitude"].astype(np.float32, copy=False)

    abs_diff_long = (dlon - plon).abs()
    abs_diff_lat = (dlat - plat).abs()
    data["abs_diff_longitude"] = abs_diff_long
    data["abs_diff_latitude"] = abs_diff_lat

    manhattan = abs_diff_long + abs_diff_lat
    data["manhattan_distance"] = manhattan

    squared_long = np.square(abs_diff_long.to_numpy(dtype=np.float32, copy=False))
    squared_lat = np.square(abs_diff_lat.to_numpy(dtype=np.float32, copy=False))
    data["squared_long"] = squared_long
    data["squared_lat"] = squared_lat

    data["euclid_distance"] = np.sqrt(squared_long + squared_lat).astype(
        np.float32, copy=False
    )
    return data




## === cell 10
all_data = add_time_features(all_data)
all_data = add_geo_features(all_data)

if "pickup_datetime" in all_data.columns:
    all_data.drop(["pickup_datetime"], axis=1, inplace=True)



## === cell 11
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
    "euclid_distance",
]

all_data = all_data[features]

all_data = pd.get_dummies(all_data, sparse=True)
all_data = all_data.fillna(0)



## === cell 12
x = all_data[:n_train]
x_test = all_data[n_train:]



## === cell 13
from sklearn.ensemble import RandomForestRegressor

model_1 = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2,
    warm_start=True,
)



## === cell 14
for n_estimators in (100, 200, 300):
    model_1.set_params(n_estimators=n_estimators)
    model_1.fit(x, y)

model_1_pred = model_1.predict(x_test)
model_1_pred = np.maximum(model_1_pred, 0.0)

sub_1 = pd.DataFrame({"key": test_id, "fare_amount": model_1_pred})
sub_1.to_csv("submission_rf.csv", index=False)

print("Wrote submission_rf.csv with shape:", sub_1.shape)



## === cell 15
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler



## === cell 16
num_features = x.shape[1]
print("Num features:", num_features)



## === cell 17
scaler = StandardScaler(with_mean=False)
x_scaled = scaler.fit_transform(x)
x_test_scaled = scaler.transform(x_test)



## === cell 18
mlp = MLPRegressor(
    hidden_layer_sizes=(30, 15, 7, 3),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size=1024,
    learning_rate_init=0.001,
    max_iter=5,  # matches original epochs=5 intent
    shuffle=True,
    random_state=42,
    verbose=True,
)



## === cell 19
y_np = y.astype(np.float32, copy=False)
mlp.fit(x_scaled, y_np)

test_pred = mlp.predict(x_test_scaled).reshape(-1)
test_pred = np.maximum(test_pred, 0.0)

sub = pd.DataFrame({"key": test_id, "fare_amount": test_pred})
sub.to_csv("submission_nn.csv", index=False)

print("Wrote submission_nn.csv with shape:", sub.shape)
print(sub.head())
