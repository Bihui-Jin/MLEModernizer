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

3.8

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
import os
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(os.cpu_count() or 4))

np.random.seed(42)



## === cell 1
train_path = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
usecols_train = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtype_train = {
    "key": "string",
    "fare_amount": "float32",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}

df = pd.read_csv(
    train_path,
    parse_dates=["pickup_datetime"],
    date_format="mixed",
    nrows=500000,
    usecols=usecols_train,
    dtype=dtype_train,
    low_memory=False,
)
df.dropna(inplace=True)



## === cell 2
test_path = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
usecols_test = [
    "key",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
dtype_test = {
    "key": "string",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "int16",
}

test = pd.read_csv(
    test_path,
    parse_dates=["pickup_datetime"],
    date_format="mixed",
    usecols=usecols_test,
    dtype=dtype_test,
    low_memory=False,
)
test



## === cell 3
nyc_min_longitude = -74.3
nyc_max_longitude = -72
nyc_min_latitude = 40.63
nyc_max_latitude = 42



## === cell 4
plon = df["pickup_longitude"].to_numpy(copy=False)
dlon = df["dropoff_longitude"].to_numpy(copy=False)
plat = df["pickup_latitude"].to_numpy(copy=False)
dlat = df["dropoff_latitude"].to_numpy(copy=False)

mask = (
    (plon > nyc_min_longitude)
    & (plon < nyc_max_longitude)
    & (dlon > nyc_min_longitude)
    & (dlon < nyc_max_longitude)
    & (plat > nyc_min_latitude)
    & (plat < nyc_max_latitude)
    & (dlat > nyc_min_latitude)
    & (dlat < nyc_max_latitude)
)
df = df.loc[mask]



## === cell 5
pc = df["passenger_count"].to_numpy(copy=False)
pc[pc == 0] = 1
df["passenger_count"] = pc



## === cell 6
df = df[(df["fare_amount"] > 0) & (df["fare_amount"] <= 100)]




## === cell 7
def euc_distance(lat1, long1, lat2, long2):
    return ((lat1 - lat2) ** 2 + (long1 - long2) ** 2) ** 0.5


df_lat1 = df["pickup_latitude"].to_numpy(copy=False)
df_lon1 = df["pickup_longitude"].to_numpy(copy=False)
df_lat2 = df["dropoff_latitude"].to_numpy(copy=False)
df_lon2 = df["dropoff_longitude"].to_numpy(copy=False)
df["travel_distance"] = euc_distance(df_lat1, df_lon1, df_lat2, df_lon2).astype(
    np.float32
)

t_lat1 = test["pickup_latitude"].to_numpy(copy=False)
t_lon1 = test["pickup_longitude"].to_numpy(copy=False)
t_lat2 = test["dropoff_latitude"].to_numpy(copy=False)
t_lon2 = test["dropoff_longitude"].to_numpy(copy=False)
test["travel_distance"] = euc_distance(t_lat1, t_lon1, t_lat2, t_lon2).astype(
    np.float32
)



## === cell 8
df = df[(df["travel_distance"] > 0) & (df["travel_distance"] < 0.30)]



## === cell 9
dt = df["pickup_datetime"].dt
df["year"] = dt.year.astype(np.int16)
df["month"] = dt.month.astype(np.int8)
df["day"] = dt.day.astype(np.int8)
df["day_of_week"] = dt.dayofweek.astype(np.int8)
df["hour"] = dt.hour.astype(np.int8)



## === cell 10
dt_t = test["pickup_datetime"].dt
test["year"] = dt_t.year.astype(np.int16)
test["month"] = dt_t.month.astype(np.int8)
test["day"] = dt_t.day.astype(np.int8)
test["day_of_week"] = dt_t.dayofweek.astype(np.int8)
test["hour"] = dt_t.hour.astype(np.int8)



## === cell 11
pass



## === cell 12
test_key = test["key"].copy()
test.drop(["key"], axis=1, inplace=True)
df.drop(["key"], axis=1, inplace=True)



## === cell 13
print(df.isnull().sum())
print(test.isnull().sum())



## === cell 14
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

split_time = df["pickup_datetime"].quantile(0.8)
train_mask = df["pickup_datetime"] <= split_time
val_mask = ~train_mask

df = df.drop(["pickup_datetime"], axis=1)
test = test.drop(["pickup_datetime"], axis=1)

X = df.loc[:, df.columns != "fare_amount"]
y = df["fare_amount"]

X_train = X.loc[train_mask].to_numpy(dtype=np.float32, copy=False)
X_val = X.loc[val_mask].to_numpy(dtype=np.float32, copy=False)
y_train = y.loc[train_mask].to_numpy(dtype=np.float32, copy=False)
y_val = y.loc[val_mask].to_numpy(dtype=np.float32, copy=False)
X_test = test.to_numpy(dtype=np.float32, copy=False)

scaler = StandardScaler(copy=False)

X_train_scaled = np.ascontiguousarray(scaler.fit_transform(X_train), dtype=np.float32)
X_val_scaled = np.ascontiguousarray(scaler.transform(X_val), dtype=np.float32)
test_scaled = np.ascontiguousarray(scaler.transform(X_test), dtype=np.float32)



## === cell 15
model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32, 8),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    batch_size="auto",
    learning_rate="adaptive",
    learning_rate_init=0.001,
    max_iter=600,  # same total iteration budget as the previous loop
    tol=1e-4,
    n_iter_no_change=20,  # kept identical (early_stopping=False so this only affects tol-based stopping)
    shuffle=True,
    random_state=42,
    early_stopping=False,
    verbose=False,
    warm_start=False,  # not needed when training in a single fit call
)

model.fit(X_train_scaled, y_train)

train_pred = model.predict(X_train_scaled)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
val_pred = model.predict(X_val_scaled)
val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print("Train RMSE: {:0.2f}".format(train_rmse))
print("Val RMSE: {:0.2f}".format(val_rmse))
print("------------------------")



## === cell 16
pred = model.predict(test_scaled)



## === cell 17
pred = np.asarray(pred).reshape(-1).astype(float)
pred = np.clip(pred, 0.0, None)
pred.shape



## === cell 18
submission = pd.read_csv(
    "/kaggle/input/new-york-city-taxi-fare-prediction/sample_submission.csv"
)
submission["fare_amount"] = pred
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
