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
import pandas as pd  # data processing, CSV file I/O
import os
from sklearnex import patch_sklearn

patch_sklearn()

print(os.listdir("../input"))




## === cell 1
MAX_ROWS = 10_000_000  # original was 20_000_000

dtype_train = {
    "fare_amount": np.float32,
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int8,
}
usecols_train = list(dtype_train.keys())
data = pd.read_csv(
    "../input/train.csv",
    nrows=MAX_ROWS,
    dtype=dtype_train,
    usecols=usecols_train,
    low_memory=False,
)




## === cell 2
def add_diff_features(df):
    """Add cheap absolute longitude/latitude differences."""
    df["abs_diff_longitude"] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df["abs_diff_latitude"] = (df.dropoff_latitude - df.pickup_latitude).abs()


def add_distance_features(df):
    """Add haversine distance (km) – called only after filtering."""
    lon1 = np.radians(df.pickup_longitude.values.astype(np.float32))
    lat1 = np.radians(df.pickup_latitude.values.astype(np.float32))
    lon2 = np.radians(df.dropoff_longitude.values.astype(np.float32))
    lat2 = np.radians(df.dropoff_latitude.values.astype(np.float32))
    dlon = lon2 - lon1
    dlat = lat2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    df["distance"] = (6371.0 * c).astype(np.float32)  # Earth radius in km


add_diff_features(data)




## === cell 3
print(data.isnull().sum())
print("Old size: %d" % len(data))
data = data.dropna(how="any", axis="rows")
print("New size: %d" % len(data))




## === cell 4
data = data[(data["fare_amount"] > 0) & np.isfinite(data["fare_amount"])].copy()

print("After fare filtering size:", len(data))

print("Old size: %d" % len(data))
data = data[(data.abs_diff_longitude < 1.0) & (data.abs_diff_latitude < 1.0)]
print("New size: %d" % len(data))

add_distance_features(data)




## === cell 5
from sklearn.model_selection import train_test_split

y = data["fare_amount"].astype(np.float32)
X = data.drop("fare_amount", axis=1)

mask = np.isfinite(y) & (y > 0)
X = X[mask]
y = y[mask]

train_df, val_df, train_y, val_y = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_y_log = np.log1p(train_y).astype(np.float32)
val_y_log = np.log1p(val_y).astype(np.float32)




## === cell 6
def get_input_matrix(df):
    return np.column_stack(
        (
            df.abs_diff_longitude.values.astype(np.float32),
            df.abs_diff_latitude.values.astype(np.float32),
            df.distance.values.astype(np.float32),
            df.passenger_count.values.astype(np.float32),
            np.ones(len(df), dtype=np.float32),  # intercept term
        )
    )


train_X = get_input_matrix(train_df)
val_X = get_input_matrix(val_df)

print("train_X shape:", train_X.shape)
print("train_y shape:", train_y.shape)




## === cell 7
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=400,  # more trees for better fit
    max_depth=None,  # allow deeper trees
    n_jobs=-1,
    random_state=42,
)

rf.fit(train_X, train_y_log)
print("RandomForestRegressor trained on log target.")




## === cell 8
dtype_test = {
    "key": object,
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int8,
}
test_df = pd.read_csv(
    "../input/test.csv",
    dtype=dtype_test,
    low_memory=False,
)

add_diff_features(test_df)
add_distance_features(test_df)

test_X = get_input_matrix(test_df)




## === cell 9
val_y_log_pred = rf.predict(val_X)
val_y_pred = np.expm1(val_y_log_pred)

test_y_log_pred = rf.predict(test_X)
test_y_pred = np.expm1(test_y_log_pred)

from sklearn.metrics import mean_squared_error

val_rmse = np.sqrt(mean_squared_error(val_y, val_y_pred))
print("Validation RMSE:", val_rmse)

test_y_pred = np.round(test_y_pred, 2)

submission = pd.DataFrame(
    {"key": test_df["key"], "fare_amount": test_y_pred},
    columns=["key", "fare_amount"],
)
submission.to_csv("submission.csv", index=False)

print("Files in cwd:", os.listdir("."))
