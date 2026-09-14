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
seaborn==0.12.2
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
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

try:
    from google.protobuf import message_factory as _message_factory

    if not hasattr(_message_factory.MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            if hasattr(self, "GetMessageClass"):
                return self.GetMessageClass(descriptor)
            if hasattr(_message_factory, "GetMessageClass"):
                return _message_factory.GetMessageClass(descriptor)
            raise AttributeError(
                "Cannot provide MessageFactory.GetPrototype: no GetMessageClass available"
            )

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import shutil

print("../", os.listdir("../"))
print("../input", os.listdir("../input"))
print("tf version: ", tf.__version__)


## === cell 1
df =  pd.read_csv('../input/train.csv', nrows= 100000, parse_dates=['pickup_datetime'])
test = pd.read_csv('../input/test.csv',parse_dates=['pickup_datetime'])


## === cell 2
df.pickup_datetime.dt.weekday


## === cell 3
from math import cos, asin, sqrt
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295     #Pi/180
    a = 0.5 - cos((lat2 - lat1) * p)/2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 12742 * asin(sqrt(a)) #2*R*asin...


## === cell 4
def add_feats(df):
    df['distance'] = pd.concat([pd.DataFrame([distance(df['pickup_latitude'][i],df['pickup_longitude'][i],df['dropoff_latitude'][i],df['dropoff_longitude'][i])], columns=['distance']) for i in range(len(df))], ignore_index=True)
    df['hour'] = df.pickup_datetime.dt.hour
    df['weekday'] = df.pickup_datetime.dt.weekday
    return df


## === cell 5
add_feats(df)


## === cell 6
dfc = df[((df.pickup_longitude >= -75.0) & (df.pickup_longitude <= -72)) 
         & ((df.pickup_latitude >= 38) & (df.pickup_latitude <= 42)) 
         & ((df.dropoff_longitude >= -75.0) & (df.dropoff_longitude <= -72)) 
         & ((df.dropoff_latitude >= 38) & (df.dropoff_latitude <= 42)) 
         & (df.fare_amount > 2.5) & (df.passenger_count > 0) & (df.passenger_count < 7) & (df.distance > 0.2)]


## === cell 7
dfc


## === cell 8
np.random.seed(seed=1) #makes result reproducible
msk = np.random.rand(len(dfc)) < 0.8
traindf = dfc[msk].drop(['key', 'pickup_datetime'], axis=1)
evaldf = dfc[~msk].drop(['key', 'pickup_datetime'], axis=1)


## === cell 9
testdf = add_feats(test)


## === cell 10
testdf = test.drop(['key', 'pickup_datetime'], axis=1)


## === cell 11
testdf


## === cell 12
def build_model_columns(nbuckets = 10):
    """Builds a set of wide and deep feature columns."""
    plon = tf.feature_column.numeric_column('pickup_longitude')
    plat = tf.feature_column.numeric_column('pickup_latitude')
    dlon = tf.feature_column.numeric_column('dropoff_longitude')
    dlat = tf.feature_column.numeric_column('dropoff_latitude')
    pcount = tf.feature_column.numeric_column('passenger_count')
    dist = tf.feature_column.numeric_column('distance') # this should be an engineered feature for the final model
    
    wday = tf.feature_column.numeric_column('weekday')
    wday_b = tf.feature_column.categorical_column_with_identity('weekday', num_buckets= 7)    
    hour = tf.feature_column.numeric_column('hour')
    
    
    latbuckets = np.linspace(38.0, 42.0, nbuckets).tolist()
    lonbuckets = np.linspace(-75.0, -72.0, nbuckets).tolist()
    b_plat = tf.feature_column.bucketized_column(plat, latbuckets)
    b_dlat = tf.feature_column.bucketized_column(dlat, latbuckets)
    b_plon = tf.feature_column.bucketized_column(plon, lonbuckets)
    b_dlon = tf.feature_column.bucketized_column(dlon, lonbuckets)
    

    ploc = tf.feature_column.crossed_column([b_plat, b_plon], nbuckets * nbuckets)
    dloc = tf.feature_column.crossed_column([b_dlat, b_dlon], nbuckets * nbuckets)
    pd_pair = tf.feature_column.crossed_column([ploc, dloc], nbuckets ** 4 )
    
    
    wide_columns = [
        dloc, ploc, pd_pair,

        wday, hour,

        pcount 
    ]
    
    deep_columns = [
        tf.feature_column.embedding_column(pd_pair, 10),

        plat, plon, dlat, dlon, dist
    ]
    return wide_columns, deep_columns


## === cell 13
def build_estimator(model_dir, nbuckets = 10):
    """Build an estimator appropriate for the given model type."""
    wide_columns, deep_columns = build_model_columns()
    hidden_units = [128, 32, 4]
    
    run_config = tf.estimator.RunConfig().replace(
    session_config=tf.ConfigProto(device_count={'GPU': 0}))
    return tf.estimator.DNNLinearCombinedRegressor(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config,    
        batch_norm=False
    )


## === cell 14
OUTDIR = './taxi_trained'


## === cell 15
BATCH_SIZE = 512


def _make_pandas_input_fn(
    x_df, y_series=None, num_epochs=1, batch_size=128, shuffle=False
):
    x_df = x_df.copy()
    x_np = {col: x_df[col].to_numpy(dtype=np.float32) for col in x_df.columns}

    if y_series is not None:
        y_np = y_series.to_numpy(dtype=np.float32)

    def _input_fn():
        if y_series is None:
            ds = tf.data.Dataset.from_tensor_slices(x_np)
        else:
            ds = tf.data.Dataset.from_tensor_slices((x_np, y_np))

        if shuffle:
            ds = ds.shuffle(
                buffer_size=min(len(x_df), 10000), seed=1, reshuffle_each_iteration=True
            )

        ds = ds.repeat(count=num_epochs)
        ds = ds.batch(batch_size, drop_remainder=False)
        return ds

    return _input_fn


train_input_fn = _make_pandas_input_fn(
    x_df=traindf[list(traindf.drop(["fare_amount"], axis=1).keys())],
    y_series=traindf["fare_amount"],
    num_epochs=None,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

eval_input_fn = _make_pandas_input_fn(
    x_df=evaldf[list(traindf.drop(["fare_amount"], axis=1).keys())],
    y_series=evaldf["fare_amount"],
    num_epochs=1,
    batch_size=len(evaldf),
    shuffle=False,
)

predict_input_fn = _make_pandas_input_fn(
    x_df=testdf[list(testdf.keys())],
    y_series=None,
    num_epochs=1,
    batch_size=len(testdf),
    shuffle=False,
)


## === cell 16
if not hasattr(tf, "estimator"):
    import tensorflow_estimator as tf_estimator  # packaged separately in TF2 environments

    tf.estimator = tf_estimator.estimator

shutil.rmtree(OUTDIR, ignore_errors=True)  # start fresh each time
num_train_steps = (100 * len(traindf)) / BATCH_SIZE
estimator = build_estimator(OUTDIR)


def rmse(labels, predictions):
    pred_values = tf.cast(predictions["predictions"], tf.float64)
    return {"rmse": tf.metrics.root_mean_squared_error(labels, pred_values)}


estimator = tf.estimator.add_metrics(estimator, rmse)

train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps)
eval_spec = tf.estimator.EvalSpec(
    input_fn=eval_input_fn,
    steps=None,
    start_delay_secs=1,  # start evaluating after N seconds
    throttle_secs=10,  # evaluate every N seconds
)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


## --- ERROR in cell 16, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/4014984679.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0;31m# TF 2.18 may not expose Estimator API as tf.estimator; bind it from tensorflow_estimator for compatibility.[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0;32mif[0m [0;32mnot[0m [0mhasattr[0m[0;34m([0m[0mtf[0m[0;34m,[0m [0;34m"estimator"[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 3[0;31m     [0;32mimport[0m [0mtensorflow_estimator[0m [0;32mas[0m [0mtf_estimator[0m  [0;31m# packaged separately in TF2 environments[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m [0;34m[0m[0m
[1;32m      5[0m     [0mtf[0m[0;34m.[0m[0mestimator[0m [0;34m=[0m [0mtf_estimator[0m[0;34m.[0m[0mestimator[0m[0;34m[0m[0;34m[0m[0m

[0;31mModuleNotFoundError[0m: No module named 'tensorflow_estimator'

## === cell 17
predictions = estimator.predict(input_fn= predict_input_fn)
