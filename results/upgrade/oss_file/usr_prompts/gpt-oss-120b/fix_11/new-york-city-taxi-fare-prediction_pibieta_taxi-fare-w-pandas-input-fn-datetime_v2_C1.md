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

4.3003

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, shutil
import numpy as np, pandas as pd

try:
    from google.protobuf import message_factory

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):

        def _get_prototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        message_factory.MessageFactory.GetPrototype = _get_prototype
except Exception:
    pass
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

print("tf version:", tf.__version__)
print("cwd files:", os.listdir("."))
print("input files:", os.listdir("../input"))




## === cell 1
df = pd.read_csv("../input/train.csv", nrows=100000, parse_dates=["pickup_datetime"])
test = pd.read_csv("../input/test.csv", parse_dates=["pickup_datetime"])




## === cell 2
df["weekday_name"] = df["pickup_datetime"].dt.day_name()




## === cell 3
from math import sin, cos, sqrt, asin, radians


def haversine_vec(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km


def add_feats(df):
    df = df.copy()
    df["distance"] = haversine_vec(
        df["pickup_latitude"],
        df["pickup_longitude"],
        df["dropoff_latitude"],
        df["dropoff_longitude"],
    )
    df["hour"] = df["pickup_datetime"].dt.hour
    df["weekday"] = df["pickup_datetime"].dt.weekday
    return df




## === cell 4
df = add_feats(df)




## === cell 5
print(df.dtypes.head())




## === cell 6
dfc = df[
    (df["pickup_longitude"].between(-75.0, -72.0))
    & (df["pickup_latitude"].between(38.0, 42.0))
    & (df["dropoff_longitude"].between(-75.0, -72.0))
    & (df["dropoff_latitude"].between(38.0, 42.0))
    & (df["fare_amount"] > 2.5)
    & (df["passenger_count"] > 0)
    & (df["passenger_count"] < 7)
    & (df["distance"] > 0.2)
]




## === cell 7
np.random.seed(1)
msk = np.random.rand(len(dfc)) < 0.8
traindf = dfc[msk].drop(["key", "pickup_datetime"], axis=1).reset_index(drop=True)
evaldf = dfc[~msk].drop(["key", "pickup_datetime"], axis=1).reset_index(drop=True)




## === cell 8
testdf = add_feats(test)
test_features = testdf.drop(["key", "pickup_datetime"], axis=1)




## === cell 9
def build_model_columns(nbuckets=10):
    plon = tf.feature_column.numeric_column("pickup_longitude")
    plat = tf.feature_column.numeric_column("pickup_latitude")
    dlon = tf.feature_column.numeric_column("dropoff_longitude")
    dlat = tf.feature_column.numeric_column("dropoff_latitude")
    pcount = tf.feature_column.numeric_column("passenger_count")
    dist = tf.feature_column.numeric_column("distance")
    wday = tf.feature_column.numeric_column("weekday")
    hour = tf.feature_column.numeric_column("hour")
    wday_b = tf.feature_column.categorical_column_with_identity(
        "weekday", num_buckets=7
    )
    hour_b = tf.feature_column.categorical_column_with_identity("hour", num_buckets=24)

    latbuckets = np.linspace(38.0, 42.0, nbuckets).tolist()
    lonbuckets = np.linspace(-75.0, -72.0, nbuckets).tolist()
    b_plat = tf.feature_column.bucketized_column(plat, latbuckets)
    b_dlat = tf.feature_column.bucketized_column(dlat, latbuckets)
    b_plon = tf.feature_column.bucketized_column(plon, lonbuckets)
    b_dlon = tf.feature_column.bucketized_column(dlon, lonbuckets)

    ploc = tf.feature_column.crossed_column([b_plat, b_plon], nbuckets * nbuckets)
    dloc = tf.feature_column.crossed_column([b_dlat, b_dlon], nbuckets * nbuckets)
    pd_pair = tf.feature_column.crossed_column([ploc, dloc], nbuckets**4)
    day_hr = tf.feature_column.crossed_column([hour_b, wday_b], 24 * 7)

    wide_columns = [dloc, ploc, pd_pair, day_hr, wday, hour, pcount]
    deep_columns = [
        tf.feature_column.embedding_column(pd_pair, 10),
        tf.feature_column.embedding_column(day_hr, 10),
        plat,
        plon,
        dlat,
        dlon,
        dist,
    ]
    return wide_columns, deep_columns




## === cell 10
def build_estimator(model_dir, nbuckets=10):
    wide_cols, deep_cols = build_model_columns(nbuckets)
    hidden_units = [128, 32, 4]
    run_cfg = tf.estimator.RunConfig(
        session_config=tf.compat.v1.ConfigProto(device_count={"GPU": 0})
    )
    return tf.estimator.DNNLinearCombinedRegressor(
        model_dir=model_dir,
        linear_feature_columns=wide_cols,
        dnn_feature_columns=deep_cols,
        dnn_hidden_units=hidden_units,
        config=run_cfg,
    )




## === cell 11
OUTDIR = "./taxi_trained"
shutil.rmtree(OUTDIR, ignore_errors=True)

BATCH_SIZE = 512


def make_input_fn(df, y=None, batch_size=128, num_epochs=None, shuffle=False):
    """Return a tf.data.Dataset suitable for Estimator."""
    if y is not None:
        dataset = tf.data.Dataset.from_tensor_slices((dict(df), y.values))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(dict(df))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))
    dataset = dataset.batch(batch_size).repeat(num_epochs)
    return dataset


train_input_fn = lambda: make_input_fn(
    traindf.drop(["fare_amount"], axis=1),
    y=traindf["fare_amount"],
    batch_size=BATCH_SIZE,
    num_epochs=None,
    shuffle=True,
)

eval_input_fn = lambda: make_input_fn(
    evaldf.drop(["fare_amount"], axis=1),
    y=evaldf["fare_amount"],
    batch_size=len(evaldf),
    num_epochs=1,
    shuffle=False,
)

predict_input_fn = lambda: make_input_fn(
    test_features,
    y=None,
    batch_size=len(test_features),
    num_epochs=1,
    shuffle=False,
)




## === cell 12
estimator = build_estimator(OUTDIR)
steps = max(1, int((100 * len(traindf)) / BATCH_SIZE))  # ensure at least one step
tf.estimator.train_and_evaluate(
    estimator,
    train_spec=tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=steps),
    eval_spec=tf.estimator.EvalSpec(
        input_fn=eval_input_fn, steps=None, start_delay_secs=1, throttle_secs=10
    ),
)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3964533208.py in <cell line: 0>()
----> 1 estimator = build_estimator(OUTDIR)
      2 steps = max(1, int((100 * len(traindf)) / BATCH_SIZE))  # ensure at least one step
      3 tf.estimator.train_and_evaluate(
      4     estimator,
      5     train_spec=tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=steps),

/tmp/ipykernel_55/2712897770.py in build_estimator(model_dir, nbuckets)
      3     wide_cols, deep_cols = build_model_columns(nbuckets)
      4     hidden_units = [128, 32, 4]
----> 5     run_cfg = tf.estimator.RunConfig(
      6         session_config=tf.compat.v1.ConfigProto(device_count={"GPU": 0})
      7     )

AttributeError: module 'tensorflow' has no attribute 'estimator'

## === cell 13
pred_gen = estimator.predict(input_fn=predict_input_fn)
pred_vals = np.array([p["predictions"] for p in pred_gen], dtype=np.float64)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1397076602.py in <cell line: 0>()
----> 1 pred_gen = estimator.predict(input_fn=predict_input_fn)
      2 pred_vals = np.array([p["predictions"] for p in pred_gen], dtype=np.float64)
      3 
      4 

NameError: name 'estimator' is not defined

## === cell 14
submission = pd.DataFrame({"key": test["key"].values, "fare_amount": pred_vals})
submission_path = "submission_file.csv"
submission.to_csv(submission_path, index=False)
print("Submission written to", submission_path)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3771130776.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"key": test["key"].values, "fare_amount": pred_vals})
      2 submission_path = "submission_file.csv"
      3 submission.to_csv(submission_path, index=False)
      4 print("Submission written to", submission_path)

NameError: name 'pred_vals' is not defined
