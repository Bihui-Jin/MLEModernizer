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
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

tf_estimator = tf.compat.v1.estimator
tf_feature_column = tf.compat.v1.feature_column
tf_metrics = tf.compat.v1.metrics




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


def pandas_train_input_fn(df, label):
    return tf.compat.v1.estimator.inputs.pandas_input_fn(
        x=df, y=label, batch_size=128, num_epochs=100, shuffle=True, queue_capacity=1000
    )


def pandas_test_input_fn(df):
    return tf.compat.v1.estimator.inputs.pandas_input_fn(
        x=df, y=None, batch_size=128, num_epochs=1, shuffle=False, queue_capacity=1000
    )


def make_feature_cols(features):
    input_columns = [tf_feature_column.numeric_column(f) for f in features]
    return input_columns


def output_submission(df, prediction_df, id_column, prediction_column, file_name):
    def _to_scalar(v):
        if isinstance(v, (list, tuple, np.ndarray)):
            return float(v[0])
        return float(v)

    df[prediction_column] = prediction_df["predictions"].apply(_to_scalar)
    df[[id_column, prediction_column]].to_csv(file_name, index=False)
    print("Output complete:", file_name)




## === cell 2
TRAIN_PATH = "/kaggle/input/train.csv"
TEST_PATH = "/kaggle/input/test.csv"
SUBMISSION_NAME = "submission.csv"

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
if not os.path.exists(TEST_PATH):
    TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"



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
DEFAULTS = [
    ["nokey"],
    [1.0],
    ["2009-06-15 17:26:21 UTC"],
    [-74.0],
    [40.0],
    [-74.0],
    [40.7],
    [1.0],
    [2009],
    [6],
    [15],
    [17],
]
INPUT_COLUMNS = [
    tf_feature_column.numeric_column("pickup_longitude"),
    tf_feature_column.numeric_column("pickup_latitude"),
    tf_feature_column.numeric_column("dropoff_longitude"),
    tf_feature_column.numeric_column("dropoff_latitude"),
    tf_feature_column.numeric_column("passenger_count"),
    tf_feature_column.numeric_column("year"),
    tf_feature_column.categorical_column_with_identity("month", num_buckets=13),
    tf_feature_column.categorical_column_with_identity("day", num_buckets=32),
    tf_feature_column.categorical_column_with_identity("hour", num_buckets=24),
    tf_feature_column.numeric_column("latdiff"),
    tf_feature_column.numeric_column("londiff"),
    tf_feature_column.numeric_column("euclidean"),
]



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/3372867016.py in <cell line: 0>()
     29 ]
     30 INPUT_COLUMNS = [
---> 31     tf_feature_column.numeric_column("pickup_longitude"),
     32     tf_feature_column.numeric_column("pickup_latitude"),
     33     tf_feature_column.numeric_column("dropoff_longitude"),

NameError: name 'tf_feature_column' is not defined

## === cell 4
train = pd.read_csv(TRAIN_PATH, nrows=4000000)
test = pd.read_csv(TEST_PATH)




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
    df["year"] = df["pickup_datetime"].apply(lambda x: int(x[:4]))
    df["month"] = df["pickup_datetime"].apply(lambda x: int(x[5:7]))
    df["day"] = df["pickup_datetime"].apply(lambda x: int(x[8:10]))
    df["hour"] = df["pickup_datetime"].apply(lambda x: int(x[11:13]))
    return df




## === cell 6
train = clean(train)



## === cell 7
train = process(train)
train[["fare_amount"]] = train[["fare_amount"]].astype("float64")
test = process(test)



## === cell 8
add_engineered(train)
add_engineered(test)
train.head(5)



## === cell 9
train_df, validation_df = train_test_split(train, test_size=0.2, random_state=1)



## === cell 10
estimator = build_estimator(16, [64, 64, 64, 8], INPUT_COLUMNS)

train_spec = tf_estimator.TrainSpec(
    input_fn=pandas_train_input_fn(train_df, train_df["fare_amount"]), max_steps=2000
)
eval_spec = tf_estimator.EvalSpec(
    input_fn=pandas_train_input_fn(validation_df, validation_df["fare_amount"]),
    steps=100,
    throttle_secs=60,
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/4174547495.py in <cell line: 0>()
----> 1 estimator = build_estimator(16, [64, 64, 64, 8], INPUT_COLUMNS)
      2 
      3 train_spec = tf_estimator.TrainSpec(
      4     input_fn=pandas_train_input_fn(train_df, train_df["fare_amount"]), max_steps=2000
      5 )

NameError: name 'INPUT_COLUMNS' is not defined

## === cell 11
tf_estimator.train_and_evaluate(estimator, train_spec=train_spec, eval_spec=eval_spec)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/7472412.py in <cell line: 0>()
----> 1 tf_estimator.train_and_evaluate(estimator, train_spec=train_spec, eval_spec=eval_spec)
      2 

NameError: name 'tf_estimator' is not defined

## === cell 12
prediction = estimator.predict(pandas_test_input_fn(test))



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/3245677553.py in <cell line: 0>()
----> 1 prediction = estimator.predict(pandas_test_input_fn(test))
      2 

NameError: name 'estimator' is not defined

## === cell 13
prediction_df = pd.DataFrame(prediction)
output_submission(test, prediction_df, "key", "fare_amount", SUBMISSION_NAME)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/1539623275.py in <cell line: 0>()
----> 1 prediction_df = pd.DataFrame(prediction)
      2 output_submission(test, prediction_df, "key", "fare_amount", SUBMISSION_NAME)

NameError: name 'prediction' is not defined
