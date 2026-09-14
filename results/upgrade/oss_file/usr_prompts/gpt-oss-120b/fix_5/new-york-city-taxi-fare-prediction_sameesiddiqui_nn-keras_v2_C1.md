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
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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
import os
from sklearn.ensemble import GradientBoostingRegressor

print(os.listdir("../input"))



## === cell 1
datatypes = {
    "key": "str",
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}
train_df = pd.read_csv("../input/train.csv", nrows=1_000_000, dtype=datatypes)



## === cell 2
train_df.describe()



## === cell 3
test_df = pd.read_csv("../input/test.csv", dtype=datatypes)
test_df.describe()




## === cell 4
def distance_between_points(df):
    df["diff_lat"] = abs(df["dropoff_latitude"] - df["pickup_latitude"])
    df["diff_long"] = abs(df["dropoff_longitude"] - df["pickup_longitude"])
    df["manhattan_dist"] = df["diff_lat"] + df["diff_long"]


distance_between_points(train_df)




## === cell 5
def extract_date_details(df):
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], format="%Y-%m-%d %H:%M:%S UTC"
    )
    dt = df["pickup_datetime"].dt
    df["year"] = dt.year
    df["month"] = dt.month
    df["day"] = dt.weekday
    df["hour"] = dt.hour


extract_date_details(train_df)




## === cell 6
def remove_outliers(df):
    df = df.dropna()
    mask = (
        (df["diff_lat"] < 5.0)
        & (df["diff_long"] < 5.0)
        & (df["diff_lat"] > 0.001)
        & (df["diff_long"] > 0.001)
        & (df["pickup_longitude"] < -72)
        & (df["pickup_longitude"] > -75)
        & (df["pickup_latitude"] < 42)
        & (df["pickup_latitude"] > 39)
        & (df["dropoff_longitude"] < -72)
        & (df["dropoff_longitude"] > -75)
        & (df["dropoff_latitude"] < 42)
        & (df["dropoff_latitude"] > 39)
        & (df["fare_amount"] > 2.50)
        & (df["fare_amount"] < 200)
        & (df["passenger_count"] <= 6)
        & (df["passenger_count"] > 0)
    )
    return df[mask]


train_df = remove_outliers(train_df)
print("Rows after cleaning:", len(train_df))




## === cell 7
def convert_to_one_hot(column, num_buckets, df, starting_index=0):
    df_size = df.shape[0]
    one_hots = np.zeros((df_size, num_buckets), dtype="byte")
    one_hots[np.arange(df_size), df[column].values - starting_index] = 1
    return one_hots.astype(np.float32)




## === cell 8
year_onehot = convert_to_one_hot("year", 7, train_df, 2009)
hour_onehot = convert_to_one_hot("hour", 24, train_df, 0)



## === cell 9
_quantiles_cache = {}


def _compute_quantiles(column):
    if column not in _quantiles_cache:
        _quantiles_cache[column] = np.quantile(
            train_df[column].values, [0.1 * i for i in range(1, 10)]
        )
    return _quantiles_cache[column]


def bucketize_feature(df, column):
    """Return integer bucket indices (0‑9) for the column."""
    quantiles = _compute_quantiles(column)
    binned = np.digitize(df[column].values, quantiles, right=False)
    binned = np.where((binned < 0) | (binned > 9), 9, binned).astype(np.int8)
    return binned


p_long = bucketize_feature(train_df, "pickup_longitude")
p_lat = bucketize_feature(train_df, "pickup_latitude")
d_long = bucketize_feature(train_df, "dropoff_longitude")
d_lat = bucketize_feature(train_df, "dropoff_latitude")




## === cell 10
def feature_cross(a1, a2):
    """Cross two 0‑9 integer vectors into a 100‑dim one‑hot matrix."""
    rows = a1.shape[0]
    cols = 100
    cross = np.zeros((rows, cols), dtype="byte")
    cross[np.arange(rows), (a1 * 10) + a2] = 1
    return cross.astype(np.float32)


p_lat_x_long = feature_cross(p_lat, p_long)
d_lat_x_long = feature_cross(d_lat, d_long)



## === cell 11
manhattan = (
    train_df["manhattan_dist"].values.reshape(len(train_df), 1).astype(np.float32)
)

train_X = np.concatenate(
    (p_lat_x_long, d_lat_x_long, year_onehot, hour_onehot, manhattan), axis=1
)
train_y = train_df["fare_amount"].values.astype(np.float32)

print("train_X shape:", train_X.shape)
print("train_y shape:", train_y.shape)



## === cell 12
validate_df = pd.read_csv(
    "../input/train.csv", skiprows=range(1, 1_000_001), nrows=10_000, dtype=datatypes
)




## === cell 13
def extract_features(df):
    extract_date_details(df)
    p_lo = bucketize_feature(df, "pickup_longitude")
    p_la = bucketize_feature(df, "pickup_latitude")
    d_lo = bucketize_feature(df, "dropoff_longitude")
    d_la = bucketize_feature(df, "dropoff_latitude")
    p_la_x_lo = feature_cross(p_la, p_lo)
    d_la_x_lo = feature_cross(d_la, d_lo)
    yr = convert_to_one_hot("year", 7, df, 2009)
    hr = convert_to_one_hot("hour", 24, df, 0)
    manhattan = df["manhattan_dist"].values.reshape(len(df), 1).astype(np.float32)
    X = np.concatenate((p_la_x_lo, d_la_x_lo, yr, hr, manhattan), axis=1)
    return X


distance_between_points(validate_df)
validate_df = remove_outliers(validate_df)
X_val = extract_features(validate_df)
y_val = validate_df["fare_amount"].values.astype(np.float32)

print("Validation X shape:", X_val.shape, "y shape:", y_val.shape)



## === cell 14
gbr = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42
)
gbr.fit(train_X, train_y)



## === cell 15
val_pred = gbr.predict(X_val)
diff = y_val - val_pred
rmse = np.sqrt(np.mean(diff**2))
print("Validation RMSE:", rmse)



## === cell 16
distance_between_points(test_df)
X_test = extract_features(test_df)
test_pred = gbr.predict(X_test)



## === cell 17
sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission["fare_amount"] = pd.Series(test_pred)
sample_submission.to_csv("nn_submission.csv", index=False)
print("Submission saved to nn_submission.csv")
