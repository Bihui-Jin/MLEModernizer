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

4.29243

# 6. Current score

5.59106

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 5.59106) has done: 'I fix the TensorFlow import crash caused by an incompatibility between TensorFlow 2.18 and protobuf 6 by forcing the pure-Python protobuf implementation before importing TensorFlow. Then I update deprecated/removed pandas and TensorFlow APIs: replace `weekday_name` with `day_name()` and replace the removed `tf.estimator` / `tf.contrib` pipeline with an equivalent Keras wide-and-deep style model (linear + DNN) trained on the same engineered features. I also correct a logic bug where engineered features on `test` were accidentally discarded, and vectorize the distance feature computation so it runs fast and reliably within the time limit. Finally, I ensure a valid `submission_file.csv` is written with exactly `key,fare_amount` aligned to the test rows.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import shutil

print("../", os.listdir("../"))
print("../input", os.listdir("../input"))
print("tf version: ", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df = pd.read_csv("../input/train.csv", nrows=200000, parse_dates=["pickup_datetime"])
test = pd.read_csv("../input/test.csv", parse_dates=["pickup_datetime"])



## === cell 2
df.pickup_datetime.dt.day_name().head()



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
def add_feats(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
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
    df["distance"] = 12742.0 * np.arcsin(np.sqrt(a))

    df["hour"] = df["pickup_datetime"].dt.hour.astype("int16")
    df["weekday"] = df["pickup_datetime"].dt.weekday.astype("int8")
    return df




## === cell 5
df = add_feats(df)



## === cell 6
df.dtypes



## === cell 7
df.pickup_datetime.isnull().sum().sum()



## === cell 8
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



## === cell 9
np.random.seed(seed=1)  # makes result reproducible
msk = np.random.rand(len(dfc)) < 0.8
traindf = dfc[msk].drop(["key", "pickup_datetime"], axis=1)
evaldf = dfc[~msk].drop(["key", "pickup_datetime"], axis=1)



## === cell 10
testdf = add_feats(test)



## === cell 11
testdf = testdf.drop(["key", "pickup_datetime"], axis=1)



## === cell 12
traindf.weekday.head()




## === cell 13
def build_model_columns(nbuckets=10):
    return nbuckets




## === cell 14
def _prepare_keras_inputs(df_in: pd.DataFrame, nbuckets: int = 10):
    df = df_in.copy()

    num_cols = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "distance",
    ]
    X_num = df[num_cols].astype("float32")

    lat_edges = np.linspace(38.0, 42.0, nbuckets).astype("float32")
    lon_edges = np.linspace(-75.0, -72.0, nbuckets).astype("float32")

    def bucketize(series, edges):
        b = np.digitize(series.to_numpy(dtype="float32"), edges, right=False)
        b = np.clip(b, 0, len(edges))
        return b.astype("int32")

    b_plat = bucketize(df["pickup_latitude"], lat_edges)
    b_dlat = bucketize(df["dropoff_latitude"], lat_edges)
    b_plon = bucketize(df["pickup_longitude"], lon_edges)
    b_dlon = bucketize(df["dropoff_longitude"], lon_edges)

    weekday = df["weekday"].astype("int32").to_numpy()
    hour = df["hour"].astype("int32").to_numpy()

    nb = len(lat_edges) + 1  # number of buckets produced by digitize
    ploc = (b_plat * nb + b_plon).astype("int32")
    dloc = (b_dlat * nb + b_dlon).astype("int32")
    pd_pair = (ploc * (nb * nb) + dloc).astype("int32")

    day_hr = (hour * 7 + weekday).astype("int32")

    X_cat = {
        "pd_pair": pd_pair,
        "day_hr": day_hr,
        "weekday": weekday,
        "hour": hour,
        "passenger_count_int": df["passenger_count"].astype("int32").to_numpy(),
    }
    return X_num, X_cat




## === cell 15
def build_estimator(model_dir, nbuckets=10):
    nb = nbuckets + 1  # buckets count used for discretized lat/lon
    pd_pair_vocab = (nb * nb) * (nb * nb)  # (ploc options) * (dloc options)
    day_hr_vocab = 24 * 7

    num_in = keras.Input(shape=(6,), name="numeric")

    pd_pair_in = keras.Input(shape=(), dtype=tf.int32, name="pd_pair")
    day_hr_in = keras.Input(shape=(), dtype=tf.int32, name="day_hr")
    weekday_in = keras.Input(shape=(), dtype=tf.int32, name="weekday")
    hour_in = keras.Input(shape=(), dtype=tf.int32, name="hour")
    pcount_in = keras.Input(shape=(), dtype=tf.int32, name="passenger_count_int")

    wide_wday = keras.layers.Embedding(input_dim=7, output_dim=1, name="wide_wday")(
        weekday_in
    )
    wide_hour = keras.layers.Embedding(input_dim=24, output_dim=1, name="wide_hour")(
        hour_in
    )
    wide_pcount = keras.layers.Embedding(input_dim=8, output_dim=1, name="wide_pcount")(
        pcount_in
    )
    wide_sum = keras.layers.Add(name="wide_sum")(
        [
            keras.layers.Flatten()(wide_wday),
            keras.layers.Flatten()(wide_hour),
            keras.layers.Flatten()(wide_pcount),
        ]
    )

    emb_pd = keras.layers.Embedding(
        input_dim=pd_pair_vocab, output_dim=10, name="emb_pd_pair"
    )(pd_pair_in)
    emb_dayhr = keras.layers.Embedding(
        input_dim=day_hr_vocab, output_dim=10, name="emb_day_hr"
    )(day_hr_in)

    deep = keras.layers.Concatenate(name="deep_concat")(
        [
            num_in,
            keras.layers.Flatten()(emb_pd),
            keras.layers.Flatten()(emb_dayhr),
        ]
    )

    x = keras.layers.Dense(128, activation="relu", name="dense_128")(deep)
    x = keras.layers.Dense(32, activation="relu", name="dense_32")(x)
    x = keras.layers.Dense(4, activation="relu", name="dense_4")(x)

    deep_out = keras.layers.Dense(1, activation=None, name="deep_out")(x)
    out = keras.layers.Add(name="fare_amount")(
        [deep_out, keras.layers.Reshape((1,))(wide_sum)]
    )

    model = keras.Model(
        inputs={
            "numeric": num_in,
            "pd_pair": pd_pair_in,
            "day_hr": day_hr_in,
            "weekday": weekday_in,
            "hour": hour_in,
            "passenger_count_int": pcount_in,
        },
        outputs=out,
        name="wide_deep_keras",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="mse",
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model




## === cell 16
OUTDIR = "./taxi_trained"
shutil.rmtree(OUTDIR, ignore_errors=True)
os.makedirs(OUTDIR, exist_ok=True)

BATCH_SIZE = 512

nbuckets = 10
Xnum_train, Xcat_train = _prepare_keras_inputs(traindf, nbuckets=nbuckets)
Xnum_eval, Xcat_eval = _prepare_keras_inputs(evaldf, nbuckets=nbuckets)
Xnum_test, Xcat_test = _prepare_keras_inputs(testdf, nbuckets=nbuckets)

y_train = traindf["fare_amount"].astype("float32").to_numpy()
y_eval = evaldf["fare_amount"].astype("float32").to_numpy()

train_inputs = {"numeric": Xnum_train.to_numpy(), **Xcat_train}
eval_inputs = {"numeric": Xnum_eval.to_numpy(), **Xcat_eval}
test_inputs = {"numeric": Xnum_test.to_numpy(), **Xcat_test}

tf.random.set_seed(1)
model = build_estimator(OUTDIR, nbuckets=nbuckets)

history = model.fit(
    train_inputs,
    y_train,
    validation_data=(eval_inputs, y_eval),
    epochs=100,
    batch_size=BATCH_SIZE,
    shuffle=True,
    verbose=2,
)



## === cell 17
pred = model.predict(test_inputs, batch_size=1024, verbose=0).reshape(-1)



## === cell 18
pred = np.clip(pred, 0.0, None)



## === cell 19
dataset_output = pd.DataFrame(
    {"key": test["key"].values, "fare_amount": pred.astype("float64")}
)



## === cell 20
dataset_output.head()



## === cell 21
dataset_output.to_csv("submission_file.csv", index=False)
print("Wrote submission_file.csv with shape:", dataset_output.shape)
print(dataset_output.columns.tolist())
