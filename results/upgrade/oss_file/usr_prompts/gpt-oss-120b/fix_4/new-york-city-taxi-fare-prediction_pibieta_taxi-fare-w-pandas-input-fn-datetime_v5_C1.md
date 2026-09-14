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

4.29156

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

print("../", os.listdir("../"))
print("../input", os.listdir("../input"))
print("tf version: ", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df = pd.read_csv("../input/train.csv", nrows=100000, parse_dates=["pickup_datetime"])
test = pd.read_csv("../input/test.csv", parse_dates=["pickup_datetime"])



## === cell 2
df.pickup_datetime.dt.weekday



## === cell 3
from math import cos, asin, sqrt


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295  # Pi/180
    a = (
        0.5
        - cos((lat2 - lat1) * p) / 2
        + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    )
    return 12742 * asin(sqrt(a))  # 2*R*asin...




## === cell 4
def add_feats(df):
    df["distance"] = pd.concat(
        [
            pd.DataFrame(
                [
                    distance(
                        df["pickup_latitude"][i],
                        df["pickup_longitude"][i],
                        df["dropoff_latitude"][i],
                        df["dropoff_longitude"][i],
                    )
                ],
                columns=["distance"],
            )
            for i in range(len(df))
        ],
        ignore_index=True,
    )
    df["hour"] = df.pickup_datetime.dt.hour
    df["weekday"] = df.pickup_datetime.dt.weekday
    return df




## === cell 5
add_feats(df)



## === cell 6
dfc = df[
    (df.pickup_longitude >= -75.0)
    & (df.pickup_longitude <= -72)
    & (df.pickup_latitude >= 38)
    & (df.pickup_latitude <= 42)
    & (df.dropoff_longitude >= -75.0)
    & (df.dropoff_longitude <= -72)
    & (df.dropoff_latitude >= 38)
    & (df.dropoff_latitude <= 42)
    & (df.fare_amount > 2.5)
    & (df.passenger_count > 0)
    & (df.passenger_count < 7)
    & (df.distance > 0.2)
]



## === cell 7
dfc



## === cell 8
np.random.seed(seed=1)  # makes result reproducible
msk = np.random.rand(len(dfc)) < 0.8
traindf = dfc[msk].drop(["key", "pickup_datetime"], axis=1)
evaldf = dfc[~msk].drop(["key", "pickup_datetime"], axis=1)



## === cell 9
testdf = add_feats(test)



## === cell 10
testdf = testdf.drop(["key", "pickup_datetime"], axis=1)



## === cell 11
testdf




## === cell 12
def build_model_columns(nbuckets=10):
    """Builds a set of wide and deep feature columns."""
    plon = tf.feature_column.numeric_column("pickup_longitude")
    plat = tf.feature_column.numeric_column("pickup_latitude")
    dlon = tf.feature_column.numeric_column("dropoff_longitude")
    dlat = tf.feature_column.numeric_column("dropoff_latitude")
    pcount = tf.feature_column.numeric_column("passenger_count")
    dist = tf.feature_column.numeric_column("distance")

    wday = tf.feature_column.numeric_column("weekday")
    wday_b = tf.feature_column.categorical_column_with_identity(
        "weekday", num_buckets=7
    )
    hour = tf.feature_column.numeric_column("hour")

    latbuckets = np.linspace(38.0, 42.0, nbuckets).tolist()
    lonbuckets = np.linspace(-75.0, -72.0, nbuckets).tolist()
    b_plat = tf.feature_column.bucketized_column(plat, latbuckets)
    b_dlat = tf.feature_column.bucketized_column(dlat, latbuckets)
    b_plon = tf.feature_column.bucketized_column(plon, lonbuckets)
    b_dlon = tf.feature_column.bucketized_column(dlon, lonbuckets)

    ploc = tf.feature_column.crossed_column([b_plat, b_plon], nbuckets * nbuckets)
    dloc = tf.feature_column.crossed_column([b_dlat, b_dlon], nbuckets * nbuckets)
    pd_pair = tf.feature_column.crossed_column([ploc, dloc], nbuckets**4)

    wide_columns = [dloc, ploc, pd_pair, wday, hour, pcount]

    deep_columns = [
        tf.feature_column.embedding_column(pd_pair, 10),
        plat,
        plon,
        dlat,
        dlon,
        dist,
    ]
    return wide_columns, deep_columns




## === cell 13
def build_estimator(model_dir, nbuckets=10):
    """Build an estimator appropriate for the given model type."""
    wide_columns, deep_columns = build_model_columns(nbuckets)
    hidden_units = [128, 32, 4]

    run_config = tf.compat.v1.estimator.RunConfig().replace(
        session_config=tf.compat.v1.ConfigProto(device_count={"GPU": 0})
    )
    return tf.compat.v1.estimator.DNNLinearCombinedRegressor(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config,
        batch_norm=False,
    )




## === cell 14
OUTDIR = "./taxi_trained"



## === cell 15
BATCH_SIZE = 512


def pandas_input_fn(
    features, labels=None, batch_size=512, num_epochs=None, shuffle=True
):
    """Convert pandas DataFrames to a tf.data.Dataset input function."""

    def input_fn():
        feature_dict = {k: np.array(v) for k, v in features.items()}
        if labels is not None:
            ds = tf.data.Dataset.from_tensor_slices((feature_dict, np.array(labels)))
        else:
            ds = tf.data.Dataset.from_tensor_slices(feature_dict)
        if shuffle:
            ds = ds.shuffle(buffer_size=len(features))
        ds = ds.batch(batch_size)
        if num_epochs is None:
            ds = ds.repeat()
        else:
            ds = ds.take(num_epochs)
        return ds

    return input_fn


train_input_fn = pandas_input_fn(
    traindf.drop(["fare_amount"], axis=1),
    traindf["fare_amount"],
    batch_size=BATCH_SIZE,
    num_epochs=None,
    shuffle=True,
)

eval_input_fn = pandas_input_fn(
    evaldf.drop(["fare_amount"], axis=1),
    evaldf["fare_amount"],
    batch_size=len(evaldf),
    num_epochs=1,
    shuffle=False,
)

predict_input_fn = pandas_input_fn(
    testdf,
    labels=None,
    batch_size=len(testdf),
    num_epochs=1,
    shuffle=False,
)



## === cell 16
shutil.rmtree(OUTDIR, ignore_errors=True)

num_train_steps = int((100 * len(traindf)) / BATCH_SIZE)

estimator = build_estimator(OUTDIR)

estimator.train(input_fn=train_input_fn, steps=num_train_steps)

_ = estimator.evaluate(input_fn=eval_input_fn)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/2135007146.py in <cell line: 0>()
      3 num_train_steps = int((100 * len(traindf)) / BATCH_SIZE)
      4 
----> 5 estimator = build_estimator(OUTDIR)
      6 
      7 estimator.train(input_fn=train_input_fn, steps=num_train_steps)

/tmp/ipykernel_55/1626825422.py in build_estimator(model_dir, nbuckets)
      4     hidden_units = [128, 32, 4]
      5 
----> 6     run_config = tf.compat.v1.estimator.RunConfig().replace(
      7         session_config=tf.compat.v1.ConfigProto(device_count={"GPU": 0})
      8     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/module_wrapper.py in _getattr(self, name)
    230     """
    231     try:
--> 232       attr = getattr(self._tfmw_wrapped_module, name)
    233     except AttributeError:
    234     # Placeholder for Google-internal contrib error

AttributeError: module 'tensorflow._api.v2.compat.v1.compat.v1' has no attribute 'estimator'

## === cell 17
predictions = estimator.predict(input_fn=predict_input_fn)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1932203927.py in <cell line: 0>()
----> 1 predictions = estimator.predict(input_fn=predict_input_fn)
      2 

NameError: name 'estimator' is not defined

## === cell 18
predlist = list(predictions)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2910921604.py in <cell line: 0>()
----> 1 predlist = list(predictions)
      2 

NameError: name 'predictions' is not defined

## === cell 19
predval = [p["predictions"] for p in predlist]



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/485118615.py in <cell line: 0>()
----> 1 predval = [p["predictions"] for p in predlist]
      2 

NameError: name 'predlist' is not defined

## === cell 20
pconc = np.array(predval).reshape(-1, 1)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4208261681.py in <cell line: 0>()
----> 1 pconc = np.array(predval).reshape(-1, 1)
      2 

NameError: name 'predval' is not defined

## === cell 21
pconc.shape



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3140495923.py in <cell line: 0>()
----> 1 pconc.shape
      2 

NameError: name 'pconc' is not defined

## === cell 22
test_keys = test["key"].values.reshape(-1, 1)



## === cell 23
output = np.hstack((test_keys, pconc))



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1989310822.py in <cell line: 0>()
----> 1 output = np.hstack((test_keys, pconc))
      2 

NameError: name 'pconc' is not defined

## === cell 24
output[:5]



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1325896683.py in <cell line: 0>()
----> 1 output[:5]
      2 

NameError: name 'output' is not defined

## === cell 25
dataset_output = pd.DataFrame({"key": output[:, 0], "fare_amount": output[:, 1]})



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/614304581.py in <cell line: 0>()
----> 1 dataset_output = pd.DataFrame({"key": output[:, 0], "fare_amount": output[:, 1]})
      2 

NameError: name 'output' is not defined

## === cell 26
dataset_output.to_csv("submission_file.csv", index=False)

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/359905751.py in <cell line: 0>()
----> 1 dataset_output.to_csv("submission_file.csv", index=False)

NameError: name 'dataset_output' is not defined
