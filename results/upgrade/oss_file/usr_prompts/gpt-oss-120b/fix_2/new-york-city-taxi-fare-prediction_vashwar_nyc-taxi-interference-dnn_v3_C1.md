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

10.02905

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 10.02905) has done: 'I fixed the TensorFlow 1‑style code to work with TensorFlow 2, corrected the date‑part extraction (using `weekofyear` instead of the removed `week` attribute), added the missing distance feature, and built a small Keras DNN that is trained on a random subset of the training data. The script now creates the required feature columns, scales them with the provided means (`mu`) and standard deviations (`sigma`), trains the model, generates predictions for the test set, and writes a valid `submission.csv` with the correct column names.'

# 9. Code solution

## === cell 0
import os, re, numpy as np, pandas as pd, tensorflow as tf

print("input dir contents:", os.listdir("../input"))


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_path = "../input/new-york-city-taxi-fare-prediction/train.csv"
test_path = "../input/new-york-city-taxi-fare-prediction/test.csv"

df_train = pd.read_csv(train_path, nrows=200_000)
df_test = pd.read_csv(test_path)




## === cell 2
def distance(data):
    """Haversine distance in kilometres."""
    radius = 6371.0
    lon1, lat1, lon2, lat2 = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return radius * c




## === cell 3
def add_datepart(df, fldname, drop=True):
    """Create expanded date‑time columns."""
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub("[Dd]ate$", "", fldname)
    for n in (
        "Year",
        "Month",
        "Weekofyear",
        "Day",
        "Dayofweek",
        "Dayofyear",
        "Hour",
        "Is_month_end",
        "Is_month_start",
        "Is_quarter_end",
        "Is_quarter_start",
        "Is_year_end",
        "Is_year_start",
    ):
        if n == "Weekofyear":
            df[targ_pre + n] = fld.dt.isocalendar().week.astype(int)
        else:
            df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + "Elapsed"] = fld.astype("int64") // 10**9
    if drop:
        df.drop(fldname, axis=1, inplace=True)




## === cell 4
df_train["Herv_Dist"] = distance(
    df_train[
        ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
    ].values.astype(np.float64)
)
df_test["Herv_Dist"] = distance(
    df_test[
        ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
    ].values.astype(np.float64)
)

add_datepart(df_train, "pickup_datetime", drop=True)
add_datepart(df_test, "pickup_datetime", drop=True)

y_train = df_train["fare_amount"].values.reshape(-1, 1)


## === cell 5
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


## === cell 6
mu = np.array(
    [
        -7.39752352e01,
        4.07510864e01,
        -7.39743620e01,
        4.07514412e01,
        1.69111912e00,
        2.01173779e03,
        6.26937910e00,
        2.54649417e01,
        1.57119467e01,
        3.04109087e00,
        1.75307310e02,
        1.35101716e01,
        1.33224990e09,
        3.34143936e00,
    ]
)
sigma = np.array(
    [
        4.26467712e-02,
        3.18110081e-02,
        4.13962939e-02,
        3.48417371e-02,
        1.30694141e00,
        1.86550121e00,
        3.43641982e00,
        1.49473195e01,
        8.68516050e00,
        1.94912410e00,
        1.04798866e02,
        6.51677611e00,
        5.84916113e07,
        4.08371701e00,
    ]
)

x_train = (x_train_unscl - mu) / sigma
x_test = (x_test_unscl - mu) / sigma


## === cell 7
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(
            2000, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal()
        ),
        tf.keras.layers.Dense(
            1000, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal()
        ),
        tf.keras.layers.Dense(
            500, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal()
        ),
        tf.keras.layers.Dense(
            250, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal()
        ),
        tf.keras.layers.Dense(
            125, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal()
        ),
        tf.keras.layers.Dense(
            50, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal()
        ),
        tf.keras.layers.Dense(
            25, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal()
        ),
        tf.keras.layers.Dense(
            10, activation="relu", kernel_initializer=tf.keras.initializers.HeNormal()
        ),
        tf.keras.layers.Dense(1, activation="linear"),
    ]
)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss="mse")


## === cell 8
model.fit(x_train, y_train, epochs=5, batch_size=1024, verbose=1)


## === cell 9
y_pred_test = model.predict(x_test).reshape(-1, 1)


## === cell 10
submission = pd.DataFrame({"key": df_test["key"], "fare_amount": y_pred_test.ravel()})
submission.to_csv("submission.csv", index=False)
print("submission saved to submission.csv, shape:", submission.shape)
