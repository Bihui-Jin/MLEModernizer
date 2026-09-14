# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

50.28306

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 47.33082) has done: 'Diagnosis: Cell 20 crashes inside TensorFlow import/initialization due to an incompatibility between TensorFlow 2.18.0 and the installed protobuf 6.33.0; TensorFlow expects the older protobuf API that provides `MessageFactory.GetPrototype`. This is a known breakage when protobuf ≥ 5/6 is used with TF versions that still rely on that API. The most localized fix is to force protobuf to use the pure-Python implementation before TensorFlow is imported, which avoids the failing C++/newer API path in many environments.  

Patch summary: In cell 20 only, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and its version) via `os.environ` immediately before importing TensorFlow, then keep the model definition/training code unchanged.  

Updated cells: Only cell 20 is changed.  

Compatibility notes for cell k+1: `model` remains a trained `tf.keras` model object exactly as before, so `model.predict(X)` in cell 21 works unchanged.  

Assumptions: The environment allows setting `os.environ` at runtime before importing TensorFlow in the cell, and the pure-Python protobuf backend is available (it is part of the protobuf package).'
- What this solution (achieved 6.33069) has done: 'The crash happens when importing TensorFlow because the installed `protobuf==6.33.0` is incompatible with TensorFlow 2.18’s generated protos in this environment, producing `MessageFactory.GetPrototype` errors. The most local, deterministic fix is to downgrade protobuf to a TF-compatible version at runtime inside the failing cell, then restart the protobuf/TensorFlow import cleanly. This keeps the model/training code unchanged while unblocking the import. The rest of the cell is left intact so `model` is created and can be used by cell 21.'
- What this solution (achieved 50.5638) has done: 'Diagnosis: The crash happens when importing/initializing TensorFlow because the environment has `protobuf==6.33.0`, but TensorFlow 2.18 expects protobuf APIs that still expose `MessageFactory.GetPrototype`; in protobuf 5/6 this API was removed/changed, causing `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. Cell 20 tries to downgrade protobuf, but it imports `google.protobuf` first and then attempts to use `import protobuf as _protobuf_pkg` (which is not a valid module), so the version check/downgrade is ineffective and TensorFlow still loads against protobuf 6. The minimal fix is to correctly detect the installed protobuf version using `google.protobuf.__version__`, and if it’s >=5, pip-install a compatible protobuf (4.25.3) before importing TensorFlow, then clear protobuf modules from `sys.modules` so the reimport uses the downgraded package.

Patch summary: In cell 20 only, replace the broken protobuf-version detection/import with a reliable check against `google.protobuf.__version__`, perform the downgrade to `protobuf==4.25.3` when needed, and purge already-imported protobuf modules before importing TensorFlow.

Updated cells: Only cell 20 is modified below.

Compatibility notes for cell k+1: The variable `model` is still created/compiled/fitted exactly as before, so `model.predict(X)` in cell 21 remains compatible and unchanged.

Assumptions: The runtime allows `pip install` during execution and has network/package access to fetch `protobuf==4.25.3` (or it is already cached).'
- What this solution (achieved 50.28306) has done: 'Your current score (50.56 RMSE) is far above the target (3.626), so we should improve accuracy with minimal, metric-aligned fixes while keeping the same model/features/training loop. The biggest issue is a train/test feature mismatch: you never call `extract_date_details(test_df)` before feature extraction, so test features miss year/hour and can lead to garbage predictions. I also fix the datetime parsing to handle both `"UTC"`-suffixed and non-suffixed formats robustly (same semantic feature extraction), and apply the same “clip extreme predictions” step to test predictions (as you already do for validation) to reduce catastrophic RMSE outliers. These are small, direct changes that preserve the core logic and should move RMSE sharply toward the target.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

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

train_df = pd.read_csv("../input/train.csv", nrows=1000000, dtype=datatypes)



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
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df["pickup_datetime"] = dt
    df["year"] = df["pickup_datetime"].dt.year.astype("Int16")
    df["month"] = df["pickup_datetime"].dt.month.astype("Int8")
    df["day"] = df["pickup_datetime"].dt.weekday.astype("Int8")
    df["hour"] = df["pickup_datetime"].dt.hour.astype("Int8")


extract_date_details(train_df)
train_df




## === cell 6
def remove_outliers(df):
    df = df.dropna()

    df = df[(df["diff_lat"] < 5.0) & (df["diff_long"] < 5.0)]
    df = df[(df["diff_lat"] > 0.001) & (df["diff_long"] > 0.001)]

    df = df[(df["pickup_longitude"] < -72) & (df["pickup_longitude"] > -75)]
    df = df[(df["pickup_latitude"] < 42) & (df["pickup_latitude"] > 39)]
    df = df[(df["dropoff_longitude"] < -72) & (df["dropoff_longitude"] > -75)]
    df = df[(df["dropoff_latitude"] < 42) & (df["dropoff_latitude"] > 39)]

    df = df[
        (df["fare_amount"] > 2.50)
        & (df["fare_amount"] < 200)
        & (df["passenger_count"] <= 6)
        & (df["passenger_count"] > 0)
    ]
    return df


train_df = remove_outliers(train_df)
len(train_df)



## === cell 7
plt.scatter(train_df[:10000]["manhattan_dist"], train_df[:10000]["fare_amount"])
plt.xlabel("manhattan distance")
plt.ylabel("fare")
plt.show()



## === cell 8
train_df.describe()




## === cell 9
def convert_to_one_hot(column, num_buckets, df, starting_index=0):
    df_size = df.shape[0]
    one_hots = np.zeros((df_size, num_buckets), dtype="byte")
    one_hots[np.arange(df_size), df[column].values - starting_index] = 1
    return one_hots




## === cell 10
year = convert_to_one_hot("year", 7, train_df, 2009)
hour = convert_to_one_hot("hour", 24, train_df, 0)



## === cell 11
train_df.shape




## === cell 12
def bucketize_feature(df, column):
    buckets = (
        train_df[column].quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).values
    )
    bins = np.array(df[column].values)

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



## === cell 13
print(p_long)
print(p_lat)




## === cell 14
def feature_cross(a1, a2):
    rows = a1.shape[0]
    cols = 100
    cross = np.zeros((rows, cols), dtype="byte")
    cross[np.arange(rows), (a1 * 10) + a2] = 1
    return cross


p_lat_x_long = feature_cross(p_lat, p_long)
d_lat_x_long = feature_cross(d_lat, d_long)



## === cell 15
unique, counts = np.unique(p_long, return_counts=True)
print(np.asarray((unique, counts)).T)
unique, counts = np.unique(p_lat, return_counts=True)
print(np.asarray((unique, counts)).T)



## === cell 16
print(p_lat_x_long.shape)
print(d_lat_x_long.shape)
print(year.shape)
print(hour.shape)
print(train_df["manhattan_dist"].shape)



## === cell 17
manhattan = train_df["manhattan_dist"].values.reshape(len(train_df), 1)

train_X = np.concatenate((p_lat_x_long, d_lat_x_long, year, hour, manhattan), axis=1)
train_y = train_df["fare_amount"].values
print(train_X.shape)
print(train_y.shape)



## === cell 18
validate_df = pd.read_csv(
    "../input/train.csv", skiprows=range(1, 1000001), nrows=10000, dtype=datatypes
)



## === cell 19
distance_between_points(validate_df)
validate_df = remove_outliers(validate_df)


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
    manhattan = df["manhattan_dist"].values.reshape(len(df), 1)

    print(p_la_x_lo.shape)
    print(d_la_x_lo.shape)
    print(yr.shape)
    print(hr.shape)
    print(manhattan.shape)

    X = np.concatenate((p_la_x_lo, d_la_x_lo, yr, hr, manhattan), axis=1)
    return X


X = extract_features(validate_df)
true_y = validate_df["fare_amount"].values



## === cell 20
import os
import sys
import subprocess

try:
    import google.protobuf as _gp  # noqa: F401
    from packaging import version as _version

    _pb_ver = _version.parse(getattr(_gp, "__version__", "0"))
    if _pb_ver.major >= 5:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        for _m in list(sys.modules.keys()):
            if _m.startswith("google.protobuf") or _m == "google.protobuf":
                sys.modules.pop(_m, None)
except Exception:
    pass

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import tensorflow as tf
from tensorflow.keras import layers

model = tf.keras.Sequential()
model.add(layers.Dense(128, activation="relu", input_dim=232))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(1))
model.compile(
    optimizer="adam", loss="mse", metrics=["mae"]  # mean squared error
)  # mean absolute error
model.fit(train_X, train_y, epochs=5, batch_size=256)



## === cell 21
result = model.predict(X).flatten()



## === cell 22
mean_y = np.mean(train_df["fare_amount"].values)
result[result > 100] = mean_y
diff = true_y - result
mse = np.sum(diff**2) / len(diff)
rmse = np.sqrt(mse)
print(rmse)



## === cell 23
distance_between_points(test_df)
extract_date_details(test_df)
X_test = extract_features(test_df)
pred_y_test = model.predict(X_test).flatten()

pred_y_test[pred_y_test > 100] = mean_y



## === cell 24
sample_submission = pd.read_csv("../input/sample_submission.csv")



## === cell 25
sample_submission["fare_amount"] = pd.Series(pred_y_test)
sample_submission.to_csv("nn_submission.csv", index=False)
