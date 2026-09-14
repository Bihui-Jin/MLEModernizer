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

4.13939939564504

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf1 = tf.compat.v1

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(0)
tf1.set_random_seed(0)

print("Listing ../input:")
print(os.listdir("../input"))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR = "../input/new-york-city-taxi-fare-prediction"
print("Data dir exists:", os.path.exists(DATA_DIR))
print("Files:", [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])



## === cell 2
df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
df_test.shape, df_test.columns.tolist()




## === cell 3
def add_datepart(df, fldname, drop=True):
    """
    Bugfix: Pandas removed .dt.week; use ISO calendar week instead.
    Also ensure we create exactly the same column names expected later (incl. 'hour' lower-case).
    """
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(
            fld, infer_datetime_format=True, errors="coerce"
        )

    targ_pre = re.sub("[Dd]ate$", "", fldname)

    df[targ_pre + "Year"] = fld.dt.year
    df[targ_pre + "Month"] = fld.dt.month

    try:
        df[targ_pre + "Week"] = fld.dt.isocalendar().week.astype(np.int16)
    except Exception:
        df[targ_pre + "Week"] = fld.dt.strftime("%V").astype(np.int16)

    df[targ_pre + "Day"] = fld.dt.day
    df[targ_pre + "Dayofweek"] = fld.dt.dayofweek
    df[targ_pre + "Dayofyear"] = fld.dt.dayofyear
    df[targ_pre + "hour"] = fld.dt.hour

    df[targ_pre + "Is_month_end"] = fld.dt.is_month_end.astype(np.int8)
    df[targ_pre + "Is_month_start"] = fld.dt.is_month_start.astype(np.int8)
    df[targ_pre + "Is_quarter_end"] = fld.dt.is_quarter_end.astype(np.int8)
    df[targ_pre + "Is_quarter_start"] = fld.dt.is_quarter_start.astype(np.int8)
    df[targ_pre + "Is_year_end"] = fld.dt.is_year_end.astype(np.int8)
    df[targ_pre + "Is_year_start"] = fld.dt.is_year_start.astype(np.int8)

    df[targ_pre + "Elapsed"] = (fld.view("int64") // 10**9).astype(np.int64)

    if drop:
        df.drop(fldname, axis=1, inplace=True)




## === cell 4
he_init = tf1.keras.initializers.VarianceScaling(
    scale=2.0, mode="fan_in", distribution="truncated_normal"
)


def dnn(
    inputs,
    training,
    dropout_rate,
    n_hidden_layers=5,
    n_neurons=100,
    name=None,
    activation=tf.nn.relu,
    initializer=he_init,
):
    with tf1.variable_scope(name, "dnn"):
        for layer in range(n_hidden_layers):
            inputs = tf1.layers.dense(
                inputs,
                n_neurons,
                kernel_initializer=initializer,
                name="hidden%d" % (layer + 1),
            )
            inputs = tf.nn.relu(inputs, name="hidden%d_out" % (layer + 1))
        return inputs




## === cell 5
add_datepart(df_test, "pickup_datetime", drop=True)

feature_cols = [
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "passenger_count",
    "pickup_datetimeYear",
    "pickup_datetimeMonth",
    "pickup_datetimeWeek",
    "pickup_datetimeDay",
    "pickup_datetimeDayofweek",
    "pickup_datetimeDayofyear",
    "pickup_datetimehour",
    "pickup_datetimeElapsed",
]

missing = [c for c in feature_cols if c not in df_test.columns]
print("Missing test columns:", missing)
df_test[feature_cols].head()



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
    ]
)

x_test_unscl = df_test[feature_cols].astype(np.float32).values
x_test = (x_test_unscl - mu.astype(np.float32)) / sigma.astype(np.float32)
x_test.shape



## === cell 7
train_path = os.path.join(DATA_DIR, "train.csv")

N_TRAIN = 600000  # chosen to fit in time/memory on Kaggle CPU; no approximations beyond subsetting for feasibility.
usecols = ["fare_amount", "pickup_datetime"] + feature_cols[
    :5
]  # pickup/dropoff/passenger + datetime

df_train = pd.read_csv(train_path, usecols=usecols, nrows=N_TRAIN)
df_train = df_train.dropna()

df_train = df_train[(df_train["fare_amount"] > 0) & (df_train["fare_amount"] < 250)]

add_datepart(df_train, "pickup_datetime", drop=True)

for c in feature_cols:
    if c not in df_train.columns:
        df_train[c] = 0

X_all_unscl = df_train[feature_cols].astype(np.float32).values
y_all = df_train["fare_amount"].astype(np.float32).values.reshape(-1, 1)

X_all = (X_all_unscl - mu.astype(np.float32)) / sigma.astype(np.float32)

split = int(0.9 * len(df_train))
X_train, X_valid = X_all[:split], X_all[split:]
y_train, y_valid = y_all[:split], y_all[split:]

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape



## === cell 8
tf1.reset_default_graph()

n_inputs = 13
n_outputs = 1

X = tf1.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf1.placeholder(tf.float32, shape=(None, 1), name="y")
training = tf1.placeholder_with_default(False, shape=(), name="training")

with tf.name_scope("dnn"):
    y_pred = tf1.layers.dense(
        dnn(X, training, 0.5),
        n_outputs,
        name="Outputs",
        kernel_initializer=he_init,
        activation=tf.nn.relu,
    )

with tf.name_scope("loss"):
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="MSE")

l_rate = 0.0001
with tf.name_scope("train"):
    optimizer = tf1.train.AdamOptimizer(learning_rate=l_rate)
    t_op = optimizer.minimize(mse)

init = tf1.global_variables_initializer()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/884443819.py in <cell line: 0>()
      9 
     10 with tf.name_scope("dnn"):
---> 11     y_pred = tf1.layers.dense(
     12         dnn(X, training, 0.5),
     13         n_outputs,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/lazy_loader.py in __getattr__(self, item)
    205           "__internal__.legacy."
    206       ):
--> 207         raise AttributeError(
    208             f"`{item}` is not available with Keras 3."
    209         )

AttributeError: `dense` is not available with Keras 3.

## === cell 9
n_epochs = 3
batch_size = 2048


def batch_iter(Xd, yd, bs):
    n = Xd.shape[0]
    for i in range(0, n, bs):
        yield Xd[i : i + bs], yd[i : i + bs]


with tf1.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for Xb, yb in batch_iter(X_train, y_train, batch_size):
            sess.run(t_op, feed_dict={X: Xb, y: yb, training: True})
        val_mse = sess.run(mse, feed_dict={X: X_valid, y: y_valid, training: False})
        print(f"epoch {epoch+1}/{n_epochs} - valid_rmse={np.sqrt(val_mse):.5f}")

    y_test = sess.run(y_pred, feed_dict={X: x_test, training: False}).reshape(-1)

y_test = np.clip(y_test.astype(np.float32), 0.0, None)
y_test[:5], y_test.shape



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/982517153.py in <cell line: 0>()
     11 
     12 with tf1.Session() as sess:
---> 13     sess.run(init)
     14     for epoch in range(n_epochs):
     15         # one pass (deterministic order)

NameError: name 'init' is not defined

## === cell 10
my_submission = pd.DataFrame(
    {"key": df_test["key"].astype(str).values, "fare_amount": y_test}
)
my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)
print(my_submission.head())

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/227694934.py in <cell line: 0>()
      1 # Build and write submission with the correct format.
      2 my_submission = pd.DataFrame(
----> 3     {"key": df_test["key"].astype(str).values, "fare_amount": y_test}
      4 )
      5 my_submission.to_csv("submission.csv", index=False)

NameError: name 'y_test' is not defined
