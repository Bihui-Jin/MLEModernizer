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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

9.42792

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

tf.get_logger().setLevel("INFO")

tf_estimator = tf.compat.v1.estimator
tf_feature_column = tf.compat.v1.feature_column
tf_metrics = tf.compat.v1.metrics  # metrics API used by Estimator is under compat.v1

np.random.seed(1)
tf.random.set_seed(1)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def add_engineered(features):
    lat1 = features["pickup_latitude"]
    lat2 = features["dropoff_latitude"]
    lon1 = features["pickup_longitude"]
    lon2 = features["dropoff_longitude"]
    latdiff = lat1 - lat2
    londiff = lon1 - lon2
    euclidean = (latdiff**2 + londiff**2) ** 0.5

    features["latdiff"] = latdiff
    features["londiff"] = londiff
    features["euclidean"] = euclidean

    return features


def build_estimator(nbuckets, hidden_units, input_columns):
    (
        plon,
        plat,
        dlon,
        dlat,
        pcount,
        year,
        month,
        day,
        hour,
        latdiff,
        londiff,
        euclidean,
    ) = input_columns

    latbuckets = np.linspace(38.0, 42.0, nbuckets).tolist()
    lonbuckets = np.linspace(-76.0, -72.0, nbuckets).tolist()
    b_plat = tf_feature_column.bucketized_column(plat, latbuckets)
    b_dlat = tf_feature_column.bucketized_column(dlat, latbuckets)
    b_plon = tf_feature_column.bucketized_column(plon, lonbuckets)
    b_dlon = tf_feature_column.bucketized_column(dlon, lonbuckets)

    ploc = tf_feature_column.crossed_column([b_plat, b_plon], nbuckets**2)
    dloc = tf_feature_column.crossed_column([b_dlat, b_dlon], nbuckets**2)
    pd_pair = tf_feature_column.crossed_column([ploc, dloc], nbuckets**4)

    wide_columns = [dloc, ploc, pd_pair, month, day, hour, year, pcount]

    deep_columns = [
        tf_feature_column.embedding_column(pd_pair, 10),
        plat,
        plon,
        dlat,
        dlon,
        latdiff,
        londiff,
        euclidean,
    ]

    estimator = tf_estimator.DNNLinearCombinedRegressor(
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
    )
    return estimator


def add_eval_metrics(labels, predictions):
    pred_values = predictions["predictions"]
    return {
        "rmse": tf_metrics.root_mean_squared_error(labels, pred_values),
        "mae": tf_metrics.mean_absolute_error(labels, pred_values),
    }


def make_input_fn(df, label=None, batch_size=128, num_epochs=1, shuffle=False):
    df = df.copy()
    if label is not None:
        y = df.pop(label).astype(np.float32)
    else:
        y = None

    def _df_to_dataset():
        x_dict = {k: df[k].values for k in df.columns}
        if y is None:
            ds = tf.data.Dataset.from_tensor_slices(x_dict)
        else:
            ds = tf.data.Dataset.from_tensor_slices((x_dict, y.values))
        if shuffle:
            ds = ds.shuffle(
                buffer_size=min(len(df), 100000), seed=1, reshuffle_each_iteration=True
            )
        ds = ds.repeat(num_epochs).batch(batch_size, drop_remainder=False)
        return ds

    return _df_to_dataset


def output_submission(df, prediction_iter, id_column, prediction_column, file_name):
    preds = []
    for p in prediction_iter:
        v = p["predictions"]
        if isinstance(v, (list, tuple, np.ndarray)):
            preds.append(float(v[0]))
        else:
            preds.append(float(v))

    out = pd.DataFrame(
        {
            id_column: df[id_column].values,
            prediction_column: np.array(preds, dtype=np.float32),
        }
    )
    out.to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows:", len(out))




## === cell 2
TRAIN_PATH = "/kaggle/input/train.csv"
TEST_PATH = "/kaggle/input/test.csv"
SUBMISSION_NAME = "submission.csv"

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
if not os.path.exists(TEST_PATH):
    TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "/kaggle/data/train.csv"
if not os.path.exists(TEST_PATH):
    TEST_PATH = "/kaggle/data/test.csv"

print("Using TRAIN_PATH:", TRAIN_PATH, "exists:", os.path.exists(TRAIN_PATH))
print("Using TEST_PATH:", TEST_PATH, "exists:", os.path.exists(TEST_PATH))



## === cell 3
CSV_COLUMNS = [
    "key",
    "fare_amount",
    "pickup_datetime",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "year",
    "month",
    "day",
    "hour",
]
LABEL_COLUMN = "fare_amount"

INPUT_COLUMNS = [
    tf_feature_column.numeric_column("pickup_longitude", dtype=tf.float32),
    tf_feature_column.numeric_column("pickup_latitude", dtype=tf.float32),
    tf_feature_column.numeric_column("dropoff_longitude", dtype=tf.float32),
    tf_feature_column.numeric_column("dropoff_latitude", dtype=tf.float32),
    tf_feature_column.numeric_column("passenger_count", dtype=tf.float32),
    tf_feature_column.numeric_column("year", dtype=tf.float32),
    tf_feature_column.categorical_column_with_identity("month", num_buckets=13),
    tf_feature_column.categorical_column_with_identity("day", num_buckets=32),
    tf_feature_column.categorical_column_with_identity("hour", num_buckets=24),
    tf_feature_column.numeric_column("latdiff", dtype=tf.float32),
    tf_feature_column.numeric_column("londiff", dtype=tf.float32),
    tf_feature_column.numeric_column("euclidean", dtype=tf.float32),
]



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4084418132.py in <cell line: 0>()
     16 
     17 INPUT_COLUMNS = [
---> 18     tf_feature_column.numeric_column("pickup_longitude", dtype=tf.float32),
     19     tf_feature_column.numeric_column("pickup_latitude", dtype=tf.float32),
     20     tf_feature_column.numeric_column("dropoff_longitude", dtype=tf.float32),

NameError: name 'tf_feature_column' is not defined

## === cell 4
train = pd.read_csv(TRAIN_PATH, nrows=4000000)
test = pd.read_csv(TEST_PATH)

print("train shape:", train.shape, "test shape:", test.shape)




## === cell 5
def clean(df):
    df = df[(-76 <= df["pickup_longitude"]) & (df["pickup_longitude"] <= -72)]
    df = df[(-76 <= df["dropoff_longitude"]) & (df["dropoff_longitude"] <= -72)]
    df = df[(38 <= df["pickup_latitude"]) & (df["pickup_latitude"] <= 42)]
    df = df[(38 <= df["dropoff_latitude"]) & (df["dropoff_latitude"] <= 42)]
    df = df[(1 <= df["passenger_count"]) & (df["passenger_count"] <= 6)]
    df = df[df["fare_amount"] > 0]
    return df


def process(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)
    df["year"] = dt.dt.year.astype("int16")
    df["month"] = dt.dt.month.astype("int8")
    df["day"] = dt.dt.day.astype("int8")
    df["hour"] = dt.dt.hour.astype("int8")
    return df




## === cell 6
train = clean(train)
train = process(train)
test = process(test)

float_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
]
for c in float_cols:
    train[c] = train[c].astype("float32")
    test[c] = test[c].astype("float32")

train["fare_amount"] = train["fare_amount"].astype("float32")

add_engineered(train)
add_engineered(test)

for c in ["latdiff", "londiff", "euclidean"]:
    train[c] = train[c].astype("float32")
    test[c] = test[c].astype("float32")

print(train.head(2))



## === cell 7
train_df, validation_df = train_test_split(train, test_size=0.2, random_state=1)

FEATURES = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "year",
    "month",
    "day",
    "hour",
    "latdiff",
    "londiff",
    "euclidean",
]

train_df = train_df[FEATURES + ["fare_amount"]].copy()
validation_df = validation_df[FEATURES + ["fare_amount"]].copy()
test_features = test[["key"] + FEATURES].copy()

print(
    "train_df:",
    train_df.shape,
    "validation_df:",
    validation_df.shape,
    "test_features:",
    test_features.shape,
)



## === cell 8
estimator = build_estimator(16, [64, 64, 64, 8], INPUT_COLUMNS)

train_spec = tf_estimator.TrainSpec(
    input_fn=make_input_fn(
        train_df, label="fare_amount", batch_size=128, num_epochs=100, shuffle=True
    ),
    max_steps=2000,
)
eval_spec = tf_estimator.EvalSpec(
    input_fn=make_input_fn(
        validation_df, label="fare_amount", batch_size=128, num_epochs=1, shuffle=False
    ),
    steps=100,
    throttle_secs=60,
)

tf_estimator.train_and_evaluate(estimator, train_spec=train_spec, eval_spec=eval_spec)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3619358205.py in <cell line: 0>()
----> 1 estimator = build_estimator(16, [64, 64, 64, 8], INPUT_COLUMNS)
      2 
      3 train_spec = tf_estimator.TrainSpec(
      4     input_fn=make_input_fn(
      5         train_df, label="fare_amount", batch_size=128, num_epochs=100, shuffle=True

NameError: name 'INPUT_COLUMNS' is not defined

## === cell 9
prediction_iter = estimator.predict(
    input_fn=make_input_fn(
        test_features[FEATURES], label=None, batch_size=128, num_epochs=1, shuffle=False
    )
)

output_submission(test_features, prediction_iter, "key", "fare_amount", SUBMISSION_NAME)
print(pd.read_csv(SUBMISSION_NAME).head())
print(
    "Wrote submission:",
    SUBMISSION_NAME,
    "size(bytes):",
    os.path.getsize(SUBMISSION_NAME),
)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3919264522.py in <cell line: 0>()
----> 1 prediction_iter = estimator.predict(
      2     input_fn=make_input_fn(
      3         test_features[FEATURES], label=None, batch_size=128, num_epochs=1, shuffle=False
      4     )
      5 )

NameError: name 'estimator' is not defined
