# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.7

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver

    _pb_major = int(_pb_ver.split(".", 1)[0])
except Exception:
    _pb_major = None

if _pb_major is None or _pb_major >= 4:
    import sys
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
    )
    import importlib
    import google.protobuf  # type: ignore

    importlib.reload(google.protobuf)

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

tf.get_logger().setLevel("INFO")


## === cell 1
def add_engineered(features):
    lat1 = features['pickup_latitude']
    lat2 = features['dropoff_latitude']
    lon1 = features['pickup_longitude']
    lon2 = features['dropoff_longitude']
    latdiff = (lat1 - lat2)
    londiff = (lon1 - lon2)
    euclidean = (latdiff ** 2 + londiff ** 2) ** 0.5

    features['latdiff'] = latdiff
    features['londiff'] = londiff
    features['euclidean'] = euclidean

    return features


def build_estimator(nbuckets, hidden_units, input_columns):
    (plon, plat, dlon, dlat, pcount, year, month, day, hour, latdiff, londiff, euclidean) = input_columns

    latbuckets = np.linspace(38.0, 42.0, nbuckets).tolist()
    lonbuckets = np.linspace(-76.0, -72.0, nbuckets).tolist()
    b_plat = tf.feature_column.bucketized_column(plat, latbuckets)
    b_dlat = tf.feature_column.bucketized_column(dlat, latbuckets)
    b_plon = tf.feature_column.bucketized_column(plon, lonbuckets)
    b_dlon = tf.feature_column.bucketized_column(dlon, lonbuckets)

    ploc = tf.feature_column.crossed_column([b_plat, b_plon], nbuckets ** 2)
    dloc = tf.feature_column.crossed_column([b_dlat, b_dlon], nbuckets ** 2)
    pd_pair = tf.feature_column.crossed_column([ploc, dloc], nbuckets ** 4)

    wide_columns = [
        dloc, ploc, pd_pair,

        month, day, hour,

        year, pcount
    ]

    deep_columns = [
        tf.feature_column.embedding_column(pd_pair, 10),

        plat, plon, dlat, dlon,
        latdiff, londiff, euclidean
    ]

    estimator = tf.estimator.DNNLinearCombinedRegressor(
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units)


    return estimator


def add_eval_metrics(labels, predictions):
    pred_values = predictions['predictions']
    return {
        'rmse': tf.metrics.root_mean_squared_error(labels, pred_values),
        'mae': tf.metrics.mean_absolute_error(labels, pred_values)
    }


def pandas_train_input_fn(df, label):
    return tf.estimator.inputs.pandas_input_fn(
        x=df,
        y=label,
        batch_size=128,
        num_epochs=100,
        shuffle=True,
        queue_capacity=1000
    )


def pandas_test_input_fn(df):
    return tf.estimator.inputs.pandas_input_fn(
        x=df,
        y=None,
        batch_size=128,
        num_epochs=1,
        shuffle=True,
        queue_capacity=1000
    )


def make_feature_cols(features):
    input_columns = [tf.feature_column.numeric_column(f) for f in features]
    return input_columns


def output_submission(df, prediction_df, id_column, prediction_column, file_name):
    df[prediction_column] = prediction_df['predictions'].apply(lambda x: x[0])
    df[[id_column, prediction_column]].to_csv((file_name), index=False)
    print('Output complete')


## === cell 2
TRAIN_PATH = '../input/train.csv'
TEST_PATH = '../input/test.csv'
SUBMISSION_NAME = 'submission.csv'


## === cell 3
CSV_COLUMNS = ['key', 'fare_amount', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
               'dropoff_latitude', 'passenger_count', 'year', 'month', 'day', 'hour']
LABEL_COLUMN = 'fare_amount'
DEFAULTS = [['nokey'], [1.0], ['2009-06-15 17:26:21 UTC'], [-74.0], [40.0], [-74.0], [40.7], [1.0], [2009], [6], [15], [17]]
INPUT_COLUMNS = [
    tf.feature_column.numeric_column('pickup_longitude'),
    tf.feature_column.numeric_column('pickup_latitude'),
    tf.feature_column.numeric_column('dropoff_longitude'),
    tf.feature_column.numeric_column('dropoff_latitude'),
    tf.feature_column.numeric_column('passenger_count'),

    tf.feature_column.numeric_column('year'),
    tf.feature_column.categorical_column_with_identity('month', num_buckets=13),
    tf.feature_column.categorical_column_with_identity('day', num_buckets=32),
    tf.feature_column.categorical_column_with_identity('hour', num_buckets=24),

    tf.feature_column.numeric_column('latdiff'),
    tf.feature_column.numeric_column('londiff'),
    tf.feature_column.numeric_column('euclidean')
]


## === cell 4
train = pd.read_csv(TRAIN_PATH, nrows=4000000)
test = pd.read_csv(TEST_PATH)


## === cell 5
def clean(df):    
    df=df[(-76 <= df['pickup_longitude']) & (df['pickup_longitude'] <= -72)]
    df=df[(-76 <= df['dropoff_longitude']) & (df['dropoff_longitude'] <= -72)]
    df=df[(38 <= df['pickup_latitude']) & (df['pickup_latitude'] <= 42)]
    df=df[(38 <= df['dropoff_latitude']) & (df['dropoff_latitude'] <= 42)]
    df=df[(1 <= df['passenger_count']) & (df['passenger_count'] <= 6)]
    df=df[df['fare_amount'] > 0 ]
    
    return df

def process(df):  
    df['year'] = df['pickup_datetime'].apply(lambda x: int(x[:4]))
    df['month'] = df['pickup_datetime'].apply(lambda x: int(x[5:7]))
    df['day'] = df['pickup_datetime'].apply(lambda x: int(x[8:10]))
    df['hour'] = df['pickup_datetime'].apply(lambda x: int(x[11:13]))
    
    return df   


## === cell 6
train = clean(train)


## === cell 7
train = process(train)
train[['fare_amount']] = train[['fare_amount']].astype('float64')
test = process(test)


## === cell 8
add_engineered(train)
add_engineered(test)
train.head(5)


## === cell 9
train_df, validation_df = train_test_split(train, test_size=0.2, random_state=1)


## === cell 10
if (not hasattr(tf, "estimator")) or (tf.estimator is None):
    raise RuntimeError(
        "This runtime's TensorFlow build does not provide `tf.estimator`, but this notebook "
        "requires Estimator APIs (DNNLinearCombinedRegressor, train_and_evaluate). "
        "On TF 2.18 these APIs are typically unavailable/incompatible, and installing "
        "`tensorflow-estimator` via pip is not compatible here."
    )

if (not hasattr(tf.estimator, "DNNLinearCombinedRegressor")) or (
    not hasattr(tf.estimator, "train_and_evaluate")
):
    raise RuntimeError(
        "This runtime's `tf.estimator` is missing required APIs "
        "(`DNNLinearCombinedRegressor` and/or `train_and_evaluate`). "
        "This notebook requires TensorFlow Estimator support, which is not compatible with TF 2.18 here."
    )

estimator = build_estimator(16, [64, 64, 64, 8], INPUT_COLUMNS)

train_spec = tf.estimator.TrainSpec(
    input_fn=pandas_train_input_fn(train_df, train_df["fare_amount"]), max_steps=10000
)
eval_spec = tf.estimator.EvalSpec(
    input_fn=pandas_train_input_fn(validation_df, validation_df["fare_amount"]),
    steps=100,
    throttle_secs=60,
)


## --- ERROR in cell 10, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mRuntimeError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2969723771.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      4[0m [0;31m# Instead, perform a strict compatibility check and fail fast with a clear error message.[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;32mif[0m [0;34m([0m[0;32mnot[0m [0mhasattr[0m[0;34m([0m[0mtf[0m[0;34m,[0m [0;34m"estimator"[0m[0;34m)[0m[0;34m)[0m [0;32mor[0m [0;34m([0m[0mtf[0m[0;34m.[0m[0mestimator[0m [0;32mis[0m [0;32mNone[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 6[0;31m     raise RuntimeError(
[0m[1;32m      7[0m         [0;34m"This runtime's TensorFlow build does not provide `tf.estimator`, but this notebook "[0m[0;34m[0m[0;34m[0m[0m
[1;32m      8[0m         [0;34m"requires Estimator APIs (DNNLinearCombinedRegressor, train_and_evaluate). "[0m[0;34m[0m[0;34m[0m[0m

[0;31mRuntimeError[0m: This runtime's TensorFlow build does not provide `tf.estimator`, but this notebook requires Estimator APIs (DNNLinearCombinedRegressor, train_and_evaluate). On TF 2.18 these APIs are typically unavailable/incompatible, and installing `tensorflow-estimator` via pip is not compatible here.

## === cell 11
tf.estimator.train_and_evaluate(estimator, train_spec=train_spec, eval_spec=eval_spec)
