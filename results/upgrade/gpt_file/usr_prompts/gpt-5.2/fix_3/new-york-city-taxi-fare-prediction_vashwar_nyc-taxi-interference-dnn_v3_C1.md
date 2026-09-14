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

10.02927

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 10.02927) has done: 'I fix the TensorFlow incompatibilities by switching the code to TF1-compat mode (`tf.compat.v1`) and disabling eager execution so placeholders, layers, Saver, and session work again. I also fix the datetime feature engineering bug caused by `dt.week` being removed in newer pandas by using ISO week (`dt.isocalendar().week`) while keeping the same feature names expected by `feature_cols`. Because the referenced `../input/dnn-model` checkpoint doesn’t exist in your environment, I make the script fall back safely to a simple, deterministic baseline prediction (mean fare) to still produce a valid `submission.csv` end-to-end rather than crashing. All paths and the required submission columns (`key,fare_amount`) are preserved.'
- What this solution (achieved 10.02927) has done: 'I fix the TensorFlow/Keras 3 incompatibility by replacing the removed `tf.compat.v1.layers.dense` calls with a small wrapper around `tf.keras.layers.Dense`, which preserves the same network structure and restores variables so `Saver` works again. I also fix the early crash (`MessageFactory.GetPrototype`) by avoiding importing TensorFlow at module import time and forcing the pure-Python protobuf implementation before TensorFlow is imported. Then the script attempt to restore the provided checkpoint if it exists; otherwise it fall back to the same deterministic baseline so a valid `submission.csv` is always produced. These changes should allow the intended DNN checkpoint to load and significantly improve RMSE toward your target when the checkpoint is present, while remaining score-neutral in the fallback case.'

# 9. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

INPUT_ROOT = "../input"
print("Listing ../input:")
try:
    print(os.listdir(INPUT_ROOT))
except Exception as e:
    print("Could not list ../input:", repr(e))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DNN_MODEL_DIR = os.path.join(INPUT_ROOT, "dnn-model")
if os.path.isdir(DNN_MODEL_DIR):
    print("Found dnn-model dir. Contents:", os.listdir(DNN_MODEL_DIR))
else:
    print("dnn-model dir not found at:", DNN_MODEL_DIR)



## === cell 2
df_test = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")



## === cell 3
df_test.head()




## === cell 4
def add_datepart(df, fldname, drop=True):
    """
    Adds date-related columns with the SAME NAMES as the original code expected,
    but fixes pandas 'dt.week' removal by using ISO week.
    """
    fld = df[fldname]
    if not np.issubdtype(fld.dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True, utc=False)

    targ_pre = re.sub("[Dd]ate$", "", fldname)

    df[targ_pre + "Year"] = fld.dt.year
    df[targ_pre + "Month"] = fld.dt.month

    try:
        df[targ_pre + "Week"] = fld.dt.isocalendar().week.astype(np.int64).values
    except Exception:
        df[targ_pre + "Week"] = fld.dt.weekofyear

    df[targ_pre + "Day"] = fld.dt.day
    df[targ_pre + "Dayofweek"] = fld.dt.dayofweek
    df[targ_pre + "Dayofyear"] = fld.dt.dayofyear
    df[targ_pre + "hour"] = fld.dt.hour

    df[targ_pre + "Is_month_end"] = fld.dt.is_month_end
    df[targ_pre + "Is_month_start"] = fld.dt.is_month_start
    df[targ_pre + "Is_quarter_end"] = fld.dt.is_quarter_end
    df[targ_pre + "Is_quarter_start"] = fld.dt.is_quarter_start
    df[targ_pre + "Is_year_end"] = fld.dt.is_year_end
    df[targ_pre + "Is_year_start"] = fld.dt.is_year_start

    df[targ_pre + "Elapsed"] = fld.astype("int64") // 10**9

    if drop:
        df.drop(fldname, axis=1, inplace=True)


def distance(data):
    radius = 6371  # km
    lon1 = data[:, 0]
    lat1 = data[:, 1]
    lon2 = data[:, 2]
    lat2 = data[:, 3]
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) * np.sin(dlat / 2) + np.cos(np.radians(lat1)) * np.cos(
        np.radians(lat2)
    ) * np.sin(dlon / 2) * np.sin(dlon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = radius * c
    return d




## === cell 5
he_init = tf.compat.v1.keras.initializers.VarianceScaling(
    scale=2.0, mode="fan_in", distribution="truncated_normal"
)


def dense_compat(inputs, units, name, kernel_initializer, activation=None):
    layer = tf.keras.layers.Dense(
        units=units,
        activation=activation,
        kernel_initializer=kernel_initializer,
        name=name,
    )
    return layer(inputs)


def dnn(
    inputs,
    training,
    n_hidden_layers=8,
    name=None,
    activation=tf.nn.relu,
    initializer=he_init,
):
    n_neurons = [2000, 1000, 500, 250, 125, 50, 25, 10]
    with tf.compat.v1.variable_scope(name, "dnn"):
        for layer in range(n_hidden_layers):
            inputs = dense_compat(
                inputs,
                n_neurons[layer],
                kernel_initializer=initializer,
                name="hidden%d" % (layer + 1),
                activation=None,
            )
            inputs = tf.nn.relu(inputs, name="hidden%d_out" % (layer + 1))
        return inputs




## === cell 6
df_test["Herv_Dist"] = distance(np.float64(df_test.values[:, 2:6]))



## === cell 7
add_datepart(df_test, "pickup_datetime", drop=True)



## === cell 8
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
    "Herv_Dist",
]



## === cell 9
df_test[feature_cols].head()



## === cell 10
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



## === cell 11
x_test_unscl = df_test[feature_cols].values.astype(np.float64)



## === cell 12
x_test = (x_test_unscl - mu) / sigma
x_test = x_test.astype(np.float32)



## === cell 13
tf.compat.v1.reset_default_graph()
n_inputs = 14
n_outputs = 1



## === cell 14
X = tf.compat.v1.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.compat.v1.placeholder(tf.float32, shape=(None, 1), name="y")
training = tf.compat.v1.placeholder_with_default(False, shape=(), name="training")



## === cell 15
with tf.name_scope("dnn"):
    y_pred = dense_compat(
        dnn(X, training),
        n_outputs,
        name="Outputs",
        kernel_initializer=he_init,
        activation=tf.nn.relu,
    )



## === cell 16
with tf.name_scope("loss"):
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="MSE")



## === cell 17
l_rate = 0.0001
with tf.name_scope("train"):
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=l_rate)
    t_op = optimizer.minimize(mse)



## === cell 18
init = tf.compat.v1.global_variables_initializer()

saver = tf.compat.v1.train.Saver()



## === cell 19
sess = tf.compat.v1.InteractiveSession()
sess.run(init)



## === cell 20
ckpt_prefix = os.path.join(DNN_MODEL_DIR, "model_.ckpt")
ckpt_exists = os.path.exists(ckpt_prefix) or os.path.exists(ckpt_prefix + ".index")

restored = False
if ckpt_exists:
    try:
        saver.restore(sess, ckpt_prefix)
        restored = True
        print("Restored model from:", ckpt_prefix)
    except Exception as e:
        restored = False
        print("Checkpoint restore failed; will fall back to baseline. Error:", repr(e))
else:
    print("Checkpoint not found; will fall back to baseline:", ckpt_prefix)



## === cell 21
if restored:
    y_test = sess.run(y_pred, feed_dict={X: x_test})
else:
    y_test = np.full((len(df_test), 1), 11.35, dtype=np.float32)



## === cell 22
sess.close()



## === cell 23
fare_pred = y_test.reshape(-1).astype(np.float64)
my_submission = pd.DataFrame(
    {"key": df_test["key"].astype(str).values, "fare_amount": fare_pred}
)



## === cell 24
my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)
print(my_submission.head())



## === cell 25
y_test[:5]
