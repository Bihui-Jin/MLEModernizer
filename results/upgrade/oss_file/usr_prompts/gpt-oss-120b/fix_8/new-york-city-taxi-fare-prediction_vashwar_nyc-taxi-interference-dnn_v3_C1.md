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

No external packages required in the script and installed.

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

3.6375985209102257

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 10.02905) has done: 'I fixed the TensorFlow 1‑style code to work with TensorFlow 2, corrected the date‑part extraction (using `weekofyear` instead of the removed `week` attribute), added the missing distance feature, and built a small Keras DNN that is trained on a random subset of the training data. The script now creates the required feature columns, scales them with the provided means (`mu`) and standard deviations (`sigma`), trains the model, generates predictions for the test set, and writes a valid `submission.csv` with the correct column names.'
- What this solution (achieved 15.17834) has done: 'I safeguard the TensorFlow import, replace the broken hard‑coded scaling with statistics computed from the training data, and switch to a scikit‑learn RandomForestRegressor (which runs without TensorFlow). These fixes remove the import error, provide proper feature scaling, and give a much stronger model so the RMSE moves toward the target while keeping the overall pipeline and feature engineering unchanged.'
- What this solution (achieved 15.06927) has done: 'I increase the training sample size and ensure the RandomForest model receives the original (un‑scaled) features while keeping the scaling only for the TensorFlow path. This removes the unnecessary scaling for the tree‑based model, adds more data, and gives the forest a few more trees, which should lower the RMSE toward the target without altering the overall pipeline.'
- What this solution (achieved 6.18474) has done: 'I add the missing imports, define the required helper functions (`distance` and `add_datepart`), safely handle the optional TensorFlow import, train the model (using RandomForest when TensorFlow is unavailable), and finally generate a correct `submission.csv`. These fixes resolve the NameErrors, ensure the model is fitted, and produce a valid submission file, moving the score toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, asin

try:
    import tensorflow as tf
except Exception:
    tf = None

np.random.seed(42)
if tf is not None:
    tf.random.set_seed(42)


def distance(arr):
    """
    Vectorized Haversine distance (km) for array of shape (n,4):
    [pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude]
    Operates in float32 to avoid extra memory copies.
    """
    arr = arr.astype(np.float32, copy=False)
    lon1, lat1, lon2, lat2 = arr.T
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0 * c
    return km


def add_datepart(df, field_name, drop=False):
    """
    Expand a datetime column into many numeric parts.
    """
    fld = df[field_name]
    if not np.issubdtype(fld.dtype, np.datetime64):
        fld = pd.to_datetime(fld, errors="coerce")
    df[field_name + "Year"] = fld.dt.year
    df[field_name + "Month"] = fld.dt.month
    df[field_name + "Weekofyear"] = fld.dt.isocalendar().week.astype(int)
    df[field_name + "Day"] = fld.dt.day
    df[field_name + "Dayofweek"] = fld.dt.weekday
    df[field_name + "Dayofyear"] = fld.dt.dayofyear
    df[field_name + "Hour"] = fld.dt.hour
    df[field_name + "Elapsed"] = fld.astype("int64") // 10**9
    if drop:
        df.drop(columns=[field_name], inplace=True)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_path = "../input/new-york-city-taxi-fare-prediction/train.csv"
test_path = "../input/new-york-city-taxi-fare-prediction/test.csv"

dtype_train = {
    "key": str,
    "fare_amount": np.float32,
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int8,
}
dtype_test = {
    "key": str,
    "pickup_longitude": np.float32,
    "pickup_latitude": np.float32,
    "dropoff_longitude": np.float32,
    "dropoff_latitude": np.float32,
    "passenger_count": np.int8,
}

df_train = pd.read_csv(
    train_path,
    usecols=list(dtype_train.keys()) + ["pickup_datetime"],
    dtype=dtype_train,
    parse_dates=["pickup_datetime"],
    nrows=2_000_000,
    engine="c",
)

df_test = pd.read_csv(
    test_path,
    usecols=list(dtype_test.keys()) + ["pickup_datetime"],
    dtype=dtype_test,
    parse_dates=["pickup_datetime"],
    engine="c",
)




## === cell 2
df_train["Herv_Dist"] = distance(
    df_train[
        ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
    ].values
)
df_test["Herv_Dist"] = distance(
    df_test[
        ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
    ].values
)

add_datepart(df_train, "pickup_datetime", drop=True)
add_datepart(df_test, "pickup_datetime", drop=True)

df_train.fillna(0, inplace=True)
df_test.fillna(0, inplace=True)

y_train = df_train["fare_amount"].values.reshape(-1, 1)

feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "pickup_datetimeYear",
    "pickup_datetimeMonth",
    "pickup_datetimeWeekofyear",
    "pickup_datetimeDay",
    "pickup_datetimeDayofweek",
    "pickup_datetimeDayofyear",
    "pickup_datetimeHour",
    "pickup_datetimeElapsed",
    "Herv_Dist",
]
x_train_unscl = df_train[feature_cols].values
x_test_unscl = df_test[feature_cols].values

if tf is not None:
    mu = x_train_unscl.mean(axis=0)
    sigma = x_train_unscl.std(axis=0) + 1e-8
    x_train = (x_train_unscl - mu) / sigma
    x_test = (x_test_unscl - mu) / sigma
else:
    x_train = x_train_unscl
    x_test = x_test_unscl




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4294382924.py in <cell line: 0>()
     10 )
     11 
---> 12 add_datepart(df_train, "pickup_datetime", drop=True)
     13 add_datepart(df_test, "pickup_datetime", drop=True)
     14 

/tmp/ipykernel_11/3538114042.py in add_datepart(df, field_name, drop)
     38     """
     39     fld = df[field_name]
---> 40     if not np.issubdtype(fld.dtype, np.datetime64):
     41         fld = pd.to_datetime(fld, errors="coerce")
     42     df[field_name + "Year"] = fld.dt.year

/usr/local/lib/python3.11/dist-packages/numpy/core/numerictypes.py in issubdtype(arg1, arg2)
    415     """
    416     if not issubclass_(arg1, generic):
--> 417         arg1 = dtype(arg1).type
    418     if not issubclass_(arg2, generic):
    419         arg2 = dtype(arg2).type

TypeError: Cannot interpret 'datetime64[ns, UTC]' as a data type

## === cell 3
if tf is not None:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(x_train.shape[1],)),
            tf.keras.layers.Dense(
                2000,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            ),
            tf.keras.layers.Dense(
                1000,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            ),
            tf.keras.layers.Dense(
                500,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            ),
            tf.keras.layers.Dense(
                250,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            ),
            tf.keras.layers.Dense(
                125,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            ),
            tf.keras.layers.Dense(
                50,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            ),
            tf.keras.layers.Dense(
                25,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            ),
            tf.keras.layers.Dense(
                10,
                activation="relu",
                kernel_initializer=tf.keras.initializers.HeNormal(),
            ),
            tf.keras.layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="mse")
    model.fit(x_train, y_train, epochs=5, batch_size=1024, verbose=0)
else:
    from sklearn.ensemble import GradientBoostingRegressor

    model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=42,
    )
    model.fit(x_train, y_train.ravel())




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3584934937.py in <cell line: 0>()
      2     model = tf.keras.Sequential(
      3         [
----> 4             tf.keras.layers.Input(shape=(x_train.shape[1],)),
      5             tf.keras.layers.Dense(
      6                 2000,

NameError: name 'x_train' is not defined

## === cell 4
if tf is not None:
    y_pred_test = model.predict(x_test).reshape(-1, 1)
else:
    y_pred_test = model.predict(x_test).reshape(-1, 1)

y_pred_test = np.clip(y_pred_test, a_min=0, a_max=None)

submission = pd.DataFrame({"key": df_test["key"], "fare_amount": y_pred_test.ravel()})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"submission saved to {submission_path}, shape: {submission.shape}")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2089301110.py in <cell line: 0>()
      1 if tf is not None:
----> 2     y_pred_test = model.predict(x_test).reshape(-1, 1)
      3 else:
      4     y_pred_test = model.predict(x_test).reshape(-1, 1)
      5 

NameError: name 'model' is not defined
