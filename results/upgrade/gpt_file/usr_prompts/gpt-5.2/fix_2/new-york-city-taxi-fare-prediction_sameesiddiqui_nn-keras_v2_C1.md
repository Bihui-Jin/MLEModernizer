# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

3.62636

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_DIR = "../input"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "/kaggle/input"

print("INPUT_DIR:", INPUT_DIR)
print(os.listdir(INPUT_DIR)[:20])



## === cell 1
import sys
import subprocess


def _ensure_protobuf_compat():
    try:
        import google.protobuf
        from packaging.version import parse as vparse

        cur = vparse(google.protobuf.__version__)
        if cur.major >= 5:
            print(
                "Downgrading protobuf from",
                google.protobuf.__version__,
                "to 4.25.3 for TensorFlow compatibility...",
            )
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
            )
            import importlib

            importlib.invalidate_caches()
    except Exception as e:
        print("Warning: protobuf compatibility check failed:", repr(e))


_ensure_protobuf_compat()



## === cell 2
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

train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")
sample_path = os.path.join(INPUT_DIR, "sample_submission.csv")

train_df = pd.read_csv(train_path, nrows=1_000_000, dtype=datatypes)
train_df.head()



## === cell 3
train_df.describe()



## === cell 4
test_df = pd.read_csv(
    test_path, dtype={k: v for k, v in datatypes.items() if k != "fare_amount"}
)
test_df.describe()




## === cell 5
def distance_between_points(df):
    df["diff_lat"] = (df["dropoff_latitude"] - df["pickup_latitude"]).abs()
    df["diff_long"] = (df["dropoff_longitude"] - df["pickup_longitude"]).abs()
    df["manhattan_dist"] = df["diff_lat"] + df["diff_long"]


distance_between_points(train_df)




## === cell 6
def extract_date_details(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df["pickup_datetime"] = dt
    df["year"] = dt.dt.year.astype("Int64")
    df["month"] = dt.dt.month.astype("Int64")
    df["day"] = dt.dt.weekday.astype("Int64")
    df["hour"] = dt.dt.hour.astype("Int64")


extract_date_details(train_df)
train_df.head()




## === cell 7
def remove_outliers(df):
    df = df.dropna()

    if "diff_lat" not in df.columns or "diff_long" not in df.columns:
        distance_between_points(df)

    df = df[(df["diff_lat"] < 5.0) & (df["diff_long"] < 5.0)]
    df = df[(df["diff_lat"] > 0.001) & (df["diff_long"] > 0.001)]

    df = df[(df["pickup_longitude"] < -72) & (df["pickup_longitude"] > -75)]
    df = df[(df["pickup_latitude"] < 42) & (df["pickup_latitude"] > 39)]
    df = df[(df["dropoff_longitude"] < -72) & (df["dropoff_longitude"] > -75)]
    df = df[(df["dropoff_latitude"] < 42) & (df["dropoff_latitude"] > 39)]

    if "fare_amount" in df.columns:
        df = df[
            (df["fare_amount"] > 2.50)
            & (df["fare_amount"] < 200)
            & (df["passenger_count"] <= 6)
            & (df["passenger_count"] > 0)
        ]
    else:
        df = df[(df["passenger_count"] <= 6) & (df["passenger_count"] > 0)]
    return df


train_df = remove_outliers(train_df)
len(train_df)



## === cell 8
plt.scatter(train_df[:10000]["manhattan_dist"], train_df[:10000]["fare_amount"])
plt.xlabel("manhattan distance")
plt.ylabel("fare")
plt.show()



## === cell 9
train_df.describe()




## === cell 10
def convert_to_one_hot(column, num_buckets, df, starting_index=0):
    df_size = df.shape[0]
    one_hots = np.zeros((df_size, num_buckets), dtype="byte")
    idx = df[column].astype("int32").values - starting_index
    idx = np.clip(idx, 0, num_buckets - 1)
    one_hots[np.arange(df_size), idx] = 1
    return one_hots




## === cell 11
year = convert_to_one_hot("year", 7, train_df, 2009)
hour = convert_to_one_hot("hour", 24, train_df, 0)



## === cell 12
train_df.shape




## === cell 13
def bucketize_feature(df, column):
    buckets = (
        train_df[column].quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).values
    )
    bins = np.array(df[column].values, copy=True)

    lower_bound = -100000
    for i in range(buckets.shape[0]):
        upper_bound = buckets[i]
        bins[(bins >= lower_bound) & (bins < upper_bound)] = i
        lower_bound = upper_bound
    bins[(bins < 0) | (bins > 8)] = 9
    bins = np.array(bins, dtype="byte")
    return bins


p_long = bucketize_feature(train_df, "pickup_longitude")
p_lat = bucketize_feature(train_df, "pickup_latitude")
d_long = bucketize_feature(train_df, "dropoff_longitude")
d_lat = bucketize_feature(train_df, "dropoff_latitude")



## === cell 14
print(p_long)
print(p_lat)




## === cell 15
def feature_cross(a1, a2):
    rows = a1.shape[0]
    cols = 100
    cross = np.zeros((rows, cols), dtype="byte")
    cross[np.arange(rows), (a1.astype("int32") * 10) + a2.astype("int32")] = 1
    return cross


p_lat_x_long = feature_cross(p_lat, p_long)
d_lat_x_long = feature_cross(d_lat, d_long)



## === cell 16
unique, counts = np.unique(p_long, return_counts=True)
print(np.asarray((unique, counts)).T)
unique, counts = np.unique(p_lat, return_counts=True)
print(np.asarray((unique, counts)).T)



## === cell 17
print(p_lat_x_long.shape)
print(d_lat_x_long.shape)
print(year.shape)
print(hour.shape)
print(train_df["manhattan_dist"].shape)



## === cell 18
manhattan = train_df["manhattan_dist"].values.reshape(len(train_df), 1)

train_X = np.concatenate((p_lat_x_long, d_lat_x_long, year, hour, manhattan), axis=1)
train_y = train_df["fare_amount"].values
print(train_X.shape)
print(train_y.shape)



## === cell 19
validate_df = pd.read_csv(
    train_path, skiprows=range(1, 1_000_001), nrows=10_000, dtype=datatypes
)



## === cell 20
distance_between_points(validate_df)
extract_date_details(validate_df)
validate_df = remove_outliers(validate_df)


def extract_features(df):
    if "pickup_datetime" not in df.columns or not np.issubdtype(
        df["pickup_datetime"].dtype, np.datetime64
    ):
        extract_date_details(df)
    if "manhattan_dist" not in df.columns:
        distance_between_points(df)

    p_lo = bucketize_feature(df, "pickup_longitude")
    p_la = bucketize_feature(df, "pickup_latitude")
    d_lo = bucketize_feature(df, "dropoff_longitude")
    d_la = bucketize_feature(df, "dropoff_latitude")
    p_la_x_lo = feature_cross(p_la, p_lo)
    d_la_x_lo = feature_cross(d_la, d_lo)
    yr = convert_to_one_hot("year", 7, df, 2009)
    hr = convert_to_one_hot("hour", 24, df, 0)
    manhattan_local = df["manhattan_dist"].values.reshape(len(df), 1)

    X = np.concatenate((p_la_x_lo, d_la_x_lo, yr, hr, manhattan_local), axis=1)
    return X


X = extract_features(validate_df)
true_y = validate_df["fare_amount"].values



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3692667919.py in <cell line: 0>()
     28 
     29 
---> 30 X = extract_features(validate_df)
     31 true_y = validate_df["fare_amount"].values
     32 

/tmp/ipykernel_11/3692667919.py in extract_features(df)
      7 def extract_features(df):
      8     # Ensure required engineered columns exist
----> 9     if "pickup_datetime" not in df.columns or not np.issubdtype(
     10         df["pickup_datetime"].dtype, np.datetime64
     11     ):

/usr/local/lib/python3.11/dist-packages/numpy/core/numerictypes.py in issubdtype(arg1, arg2)
    415     """
    416     if not issubclass_(arg1, generic):
--> 417         arg1 = dtype(arg1).type
    418     if not issubclass_(arg2, generic):
    419         arg2 = dtype(arg2).type

TypeError: Cannot interpret 'datetime64[ns, UTC]' as a data type

## === cell 21
import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential()
model.add(layers.Dense(128, activation="relu", input_dim=232))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(1))
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.fit(train_X, train_y, epochs=5, batch_size=256, verbose=1)



## === cell 22
result = model.predict(X, verbose=0).flatten()



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1809578996.py in <cell line: 0>()
----> 1 result = model.predict(X, verbose=0).flatten()
      2 

NameError: name 'X' is not defined

## === cell 23
mean_y = np.mean(train_df["fare_amount"].values)
result[result > 100] = mean_y
diff = true_y - result
mse = np.sum(diff**2) / len(diff)
rmse = np.sqrt(mse)
print(rmse)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1525147306.py in <cell line: 0>()
      1 mean_y = np.mean(train_df["fare_amount"].values)
----> 2 result[result > 100] = mean_y
      3 diff = true_y - result
      4 mse = np.sum(diff**2) / len(diff)
      5 rmse = np.sqrt(mse)

NameError: name 'result' is not defined

## === cell 24
distance_between_points(test_df)
extract_date_details(test_df)
X_test = extract_features(test_df)
pred_y_test = model.predict(X_test, verbose=0).flatten()



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4019011970.py in <cell line: 0>()
      2 extract_date_details(test_df)
      3 # Do not remove outliers on test (can drop rows and break submission alignment)
----> 4 X_test = extract_features(test_df)
      5 pred_y_test = model.predict(X_test, verbose=0).flatten()
      6 

/tmp/ipykernel_11/3692667919.py in extract_features(df)
      7 def extract_features(df):
      8     # Ensure required engineered columns exist
----> 9     if "pickup_datetime" not in df.columns or not np.issubdtype(
     10         df["pickup_datetime"].dtype, np.datetime64
     11     ):

/usr/local/lib/python3.11/dist-packages/numpy/core/numerictypes.py in issubdtype(arg1, arg2)
    415     """
    416     if not issubclass_(arg1, generic):
--> 417         arg1 = dtype(arg1).type
    418     if not issubclass_(arg2, generic):
    419         arg2 = dtype(arg2).type

TypeError: Cannot interpret 'datetime64[ns, UTC]' as a data type

## === cell 25
submission = pd.DataFrame(
    {"key": test_df["key"].values, "fare_amount": pred_y_test.astype(np.float32)}
)

submission.loc[submission["fare_amount"] < 0, "fare_amount"] = mean_y

submission.to_csv("nn_submission.csv", index=False)
print(submission.head())
print("Wrote nn_submission.csv with shape:", submission.shape)

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2532516926.py in <cell line: 0>()
      1 # Ensure submission uses the keys from test_df for perfect alignment.
      2 submission = pd.DataFrame(
----> 3     {"key": test_df["key"].values, "fare_amount": pred_y_test.astype(np.float32)}
      4 )
      5 

NameError: name 'pred_y_test' is not defined
