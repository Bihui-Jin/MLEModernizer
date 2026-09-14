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
import os
import shutil
import numpy as np
import pandas as pd

import tensorflow as tf

print("../", os.listdir("../"))
print("../input", os.listdir("../input"))
print("tf version: ", tf.__version__)
print("tf.estimator available:", hasattr(tf, "estimator"))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df = pd.read_csv("../input/train.csv", nrows=100000, parse_dates=["pickup_datetime"])
test = pd.read_csv("../input/test.csv", parse_dates=["pickup_datetime"])



## === cell 2
_ = df.pickup_datetime.dt.day_name()



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
def add_feats(dfin: pd.DataFrame) -> pd.DataFrame:
    df = dfin.copy()

    p = 0.017453292519943295  # Pi/180
    lat1 = df["pickup_latitude"].astype("float64").to_numpy()
    lon1 = df["pickup_longitude"].astype("float64").to_numpy()
    lat2 = df["dropoff_latitude"].astype("float64").to_numpy()
    lon2 = df["dropoff_longitude"].astype("float64").to_numpy()

    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    df["distance"] = (12742 * np.arcsin(np.sqrt(a))).astype("float32")

    df["hour"] = df.pickup_datetime.dt.hour.astype("int16")
    df["weekday"] = df.pickup_datetime.dt.weekday.astype("int16")
    return df




## === cell 5
df = add_feats(df)
df.dtypes



## === cell 6
df.pickup_datetime.isnull().sum().sum()



## === cell 7
dfc = df[
    ((df.pickup_longitude >= -75.0) & (df.pickup_longitude <= -72))
    & ((df.pickup_latitude >= 38) & (df.pickup_latitude <= 42))
    & ((df.dropoff_longitude >= -75.0) & (df.dropoff_longitude <= -72))
    & ((df.dropoff_latitude >= 38) & (df.dropoff_latitude <= 42))
    & (df.fare_amount > 2.5)
    & (df.passenger_count > 0)
    & (df.passenger_count < 7)
    & (df.distance > 0.2)
].copy()



## === cell 8
np.random.seed(seed=1)  # makes result reproducible
msk = np.random.rand(len(dfc)) < 0.8
traindf = dfc[msk].drop(["key", "pickup_datetime"], axis=1).copy()
evaldf = dfc[~msk].drop(["key", "pickup_datetime"], axis=1).copy()



## === cell 9
testdf = add_feats(test)
testdf = testdf.drop(["key", "pickup_datetime"], axis=1).copy()



## === cell 10
traindf.weekday.head()




## === cell 11
def build_model_columns(nbuckets=10):
    """Builds a set of wide and deep feature columns."""
    fc = tf.feature_column

    plon = fc.numeric_column("pickup_longitude")
    plat = fc.numeric_column("pickup_latitude")
    dlon = fc.numeric_column("dropoff_longitude")
    dlat = fc.numeric_column("dropoff_latitude")
    pcount = fc.numeric_column("passenger_count")
    dist = fc.numeric_column("distance")

    wday = fc.numeric_column("weekday")
    wday_b = fc.categorical_column_with_identity("weekday", num_buckets=7)
    hour = fc.numeric_column("hour")
    hour_b = fc.categorical_column_with_identity("hour", num_buckets=24)

    latbuckets = np.linspace(38.0, 42.0, nbuckets).tolist()
    lonbuckets = np.linspace(-75.0, -72.0, nbuckets).tolist()
    b_plat = fc.bucketized_column(plat, latbuckets)
    b_dlat = fc.bucketized_column(dlat, latbuckets)
    b_plon = fc.bucketized_column(plon, lonbuckets)
    b_dlon = fc.bucketized_column(dlon, lonbuckets)

    ploc = fc.crossed_column([b_plat, b_plon], nbuckets * nbuckets)
    dloc = fc.crossed_column([b_dlat, b_dlon], nbuckets * nbuckets)
    pd_pair = fc.crossed_column([ploc, dloc], nbuckets**4)
    day_hr = fc.crossed_column([hour_b, wday_b], 24 * 7)

    wide_columns = [dloc, ploc, pd_pair, day_hr, wday, hour, pcount]

    deep_columns = [
        fc.embedding_column(pd_pair, 10),
        fc.embedding_column(day_hr, 10),
        plat,
        plon,
        dlat,
        dlon,
        dist,
    ]
    return wide_columns, deep_columns




## === cell 12
def build_estimator(model_dir, nbuckets=10):
    wide_columns, deep_columns = build_model_columns(nbuckets=nbuckets)
    hidden_units = [128, 32, 4]

    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.compat.v1.ConfigProto(device_count={"GPU": 0})
    )

    return tf.estimator.DNNLinearCombinedRegressor(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config,
    )




## === cell 13
OUTDIR = "./taxi_trained"



## === cell 14
BATCH_SIZE = 512

feature_cols = list(traindf.drop(["fare_amount"], axis=1).columns)

train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=traindf[feature_cols],
    y=traindf["fare_amount"],
    num_epochs=None,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

eval_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=evaldf[feature_cols],
    y=evaldf["fare_amount"],
    num_epochs=1,
    batch_size=len(evaldf),
    shuffle=False,
)

predict_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
    x=testdf[list(testdf.columns)],
    y=None,
    num_epochs=1,
    batch_size=len(testdf),
    shuffle=False,
)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/956355449.py in <cell line: 0>()
      4 
      5 # Fix: tensorflow_estimator is not imported; use tf.compat.v1.estimator.inputs (same API).
----> 6 train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
      7     x=traindf[feature_cols],
      8     y=traindf["fare_amount"],

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/module_wrapper.py in _getattr(self, name)
    230     """
    231     try:
--> 232       attr = getattr(self._tfmw_wrapped_module, name)
    233     except AttributeError:
    234     # Placeholder for Google-internal contrib error

AttributeError: module 'tensorflow._api.v2.compat.v1' has no attribute 'estimator'

## === cell 15
shutil.rmtree(OUTDIR, ignore_errors=True)  # start fresh each time
num_train_steps = int((100 * len(traindf)) / BATCH_SIZE)


def rmse_metric_fn(labels, predictions):
    pred_values = tf.cast(predictions["predictions"], tf.float32)
    rmse = tf.compat.v1.metrics.root_mean_squared_error(
        labels=labels, predictions=pred_values
    )
    return {"rmse": rmse}


estimator = build_estimator(OUTDIR)
estimator = tf.estimator.add_metrics(estimator, rmse_metric_fn)

train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps)
eval_spec = tf.estimator.EvalSpec(
    input_fn=eval_input_fn, steps=None, start_delay_secs=1, throttle_secs=10
)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1042734840.py in <cell line: 0>()
     11 
     12 
---> 13 estimator = build_estimator(OUTDIR)
     14 estimator = tf.estimator.add_metrics(estimator, rmse_metric_fn)
     15 

/tmp/ipykernel_11/3134559708.py in build_estimator(model_dir, nbuckets)
      4 
      5     # Fix: use tf.estimator RunConfig (built-in) and keep CPU-only to avoid GPU issues.
----> 6     run_config = tf.estimator.RunConfig().replace(
      7         session_config=tf.compat.v1.ConfigProto(device_count={"GPU": 0})
      8     )

AttributeError: module 'tensorflow' has no attribute 'estimator'

## === cell 16
predictions = estimator.predict(input_fn=predict_input_fn)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1932203927.py in <cell line: 0>()
----> 1 predictions = estimator.predict(input_fn=predict_input_fn)
      2 

NameError: name 'estimator' is not defined

## === cell 17
predlist = list(predictions)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2900556086.py in <cell line: 0>()
      1 # Fix: streaming generator -> collect safely and ensure 1D float array aligned to test rows
----> 2 predlist = list(predictions)
      3 

NameError: name 'predictions' is not defined

## === cell 18
predval = [predlist[i].get("predictions") for i in range(len(predlist))]



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3602787469.py in <cell line: 0>()
----> 1 predval = [predlist[i].get("predictions") for i in range(len(predlist))]
      2 

NameError: name 'predlist' is not defined

## === cell 19
pconc = np.concatenate(predval).reshape(-1).astype(np.float64)

assert len(pconc) == len(test), f"Pred length {len(pconc)} != test length {len(test)}"



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3833235814.py in <cell line: 0>()
----> 1 pconc = np.concatenate(predval).reshape(-1).astype(np.float64)
      2 
      3 assert len(pconc) == len(test), f"Pred length {len(pconc)} != test length {len(test)}"
      4 

NameError: name 'predval' is not defined

## === cell 20
output = pd.DataFrame({"key": test["key"].values, "fare_amount": pconc})



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1708049607.py in <cell line: 0>()
----> 1 output = pd.DataFrame({"key": test["key"].values, "fare_amount": pconc})
      2 

NameError: name 'pconc' is not defined

## === cell 21
output.to_csv("submission_file.csv", index=False)
print("Wrote submission_file.csv with shape:", output.shape)
print(output.head())

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3211318973.py in <cell line: 0>()
----> 1 output.to_csv("submission_file.csv", index=False)
      2 print("Wrote submission_file.csv with shape:", output.shape)
      3 print(output.head())

NameError: name 'output' is not defined
