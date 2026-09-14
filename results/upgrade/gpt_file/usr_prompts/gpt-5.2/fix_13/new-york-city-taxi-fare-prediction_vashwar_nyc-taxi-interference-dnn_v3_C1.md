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

None

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 10.02927) has done: 'I fix the TensorFlow incompatibilities by switching the code to TF1-compat mode (`tf.compat.v1`) and disabling eager execution so placeholders, layers, Saver, and session work again. I also fix the datetime feature engineering bug caused by `dt.week` being removed in newer pandas by using ISO week (`dt.isocalendar().week`) while keeping the same feature names expected by `feature_cols`. Because the referenced `../input/dnn-model` checkpoint doesn’t exist in your environment, I make the script fall back safely to a simple, deterministic baseline prediction (mean fare) to still produce a valid `submission.csv` end-to-end rather than crashing. All paths and the required submission columns (`key,fare_amount`) are preserved.'
- What this solution (achieved 10.02927) has done: 'I fix the TensorFlow/Keras 3 incompatibility by replacing the removed `tf.compat.v1.layers.dense` calls with a small wrapper around `tf.keras.layers.Dense`, which preserves the same network structure and restores variables so `Saver` works again. I also fix the early crash (`MessageFactory.GetPrototype`) by avoiding importing TensorFlow at module import time and forcing the pure-Python protobuf implementation before TensorFlow is imported. Then the script attempt to restore the provided checkpoint if it exists; otherwise it fall back to the same deterministic baseline so a valid `submission.csv` is always produced. These changes should allow the intended DNN checkpoint to load and significantly improve RMSE toward your target when the checkpoint is present, while remaining score-neutral in the fallback case.'
- What this solution (achieved 10.02927) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf runtime and clearing any preloaded protobuf modules before importing TensorFlow, which is the root cause of your current runtime failure. Then I keep the existing DNN graph/checkpoint-restore logic unchanged so that, if the `dnn-model` checkpoint is present, predictions come from the intended network (which should move RMSE down toward your target). If the checkpoint is absent, the script still fall back to the constant baseline and always write a valid `submission.csv` with the required columns. I also make the input path resolution robust for this environment by using `/kaggle/input` when available (without changing the dataset-relative file paths you already use).'
- What this solution (achieved 6.58114) has done: 'You’re crashing before any model logic runs because importing TensorFlow triggers a protobuf API mismatch (`MessageFactory.GetPrototype`). I fix this by (1) forcing the pure-Python protobuf runtime earlier and (2) avoiding importing TensorFlow at module import time—so the script can always run end-to-end. Since the `dnn-model` checkpoint directory is not present in your environment, the current code always falls back to a constant baseline (hence the poor RMSE); to move score toward your target without changing the DNN architecture/training semantics, I add a lightweight, deterministic fallback model (a small sklearn regression trained on a small chunk of train.csv with the exact same engineered features) and use it only when the checkpoint can’t be restored. The submission format and paths stay the same, and it still write `submission.csv`.'
- What this solution (achieved 5.19521) has done: 'Your current score (6.58 RMSE) is worse than the target (3.64), so we should legitimately reduce RMSE without changing the overall approach. The biggest win with minimal disruption is to make the fallback RandomForest behave better by training on a slightly larger (but still feasible) cleaned sample and adding two standard NYC-taxi features (manhattan distance + bearing) computed from the same columns—this keeps the same “engineer features → train simple model → predict” logic intact. I also make the fallback use a stable train/validation split to print a local RMSE sanity-check (doesn’t affect the submission) and keep the submission formatting unchanged. No changes are made to the TF checkpoint path/restore logic; this only improves the path you’re actually using (checkpoint missing).'

# 9. Code solution

## === cell 0
import os
import re
import sys
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

INPUT_ROOT = "/kaggle/input" if os.path.isdir("/kaggle/input") else "../input"
print("INPUT_ROOT:", INPUT_ROOT)
try:
    print("Listing INPUT_ROOT:", os.listdir(INPUT_ROOT)[:50])
except Exception as e:
    print("Could not list INPUT_ROOT:", repr(e))

np.random.seed(42)



## === cell 1
DNN_MODEL_DIR = os.path.join(INPUT_ROOT, "dnn-model")
if os.path.isdir(DNN_MODEL_DIR):
    print("Found dnn-model dir. Contents:", os.listdir(DNN_MODEL_DIR)[:50])
else:
    print("dnn-model dir not found at:", DNN_MODEL_DIR)



## === cell 2
test_path = os.path.join(INPUT_ROOT, "new-york-city-taxi-fare-prediction", "test.csv")
df_test = pd.read_csv(
    test_path,
    dtype={
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int16",
        "key": "object",
    },
    parse_dates=["pickup_datetime"],
)



## === cell 3
df_test.head()




## === cell 4
def add_datepart(df, fldname, drop=True):
    """
    Same feature logic as original; optimized to avoid repeated Series lookups and
    to assign using NumPy arrays where possible (less pandas overhead, same values).
    """
    fld = df[fldname]

    if not pd.api.types.is_datetime64_any_dtype(fld):
        df[fldname] = fld = pd.to_datetime(
            fld, infer_datetime_format=True, utc=False, errors="coerce"
        )

    if pd.api.types.is_datetime64tz_dtype(fld):
        df[fldname] = fld = fld.dt.tz_localize(None)

    targ_pre = re.sub("[Dd]ate$", "", fldname)
    dt = fld.dt

    df[targ_pre + "Year"] = dt.year.to_numpy()
    df[targ_pre + "Month"] = dt.month.to_numpy()

    try:
        df[targ_pre + "Week"] = dt.isocalendar().week.astype(np.int64).to_numpy()
    except Exception:
        df[targ_pre + "Week"] = dt.weekofyear.to_numpy()

    df[targ_pre + "Day"] = dt.day.to_numpy()
    df[targ_pre + "Dayofweek"] = dt.dayofweek.to_numpy()
    df[targ_pre + "Dayofyear"] = dt.dayofyear.to_numpy()
    df[targ_pre + "hour"] = dt.hour.to_numpy()

    df[targ_pre + "Is_month_end"] = dt.is_month_end
    df[targ_pre + "Is_month_start"] = dt.is_month_start
    df[targ_pre + "Is_quarter_end"] = dt.is_quarter_end
    df[targ_pre + "Is_quarter_start"] = dt.is_quarter_start
    df[targ_pre + "Is_year_end"] = dt.is_year_end
    df[targ_pre + "Is_year_start"] = dt.is_year_start

    df[targ_pre + "Elapsed"] = (fld.astype("int64") // 10**9).to_numpy()

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


def manhattan_approx_km_np(pick_lat, pick_lon, drop_lat, drop_lon):
    dlat = np.abs(drop_lat - pick_lat)
    dlon = np.abs(drop_lon - pick_lon)
    return (111.0 * dlat + 85.0 * dlon).astype(np.float64)


def bearing_deg_np(pick_lat, pick_lon, drop_lat, drop_lon):
    lon1 = np.radians(pick_lon)
    lat1 = np.radians(pick_lat)
    lon2 = np.radians(drop_lon)
    lat2 = np.radians(drop_lat)
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brng = np.degrees(np.arctan2(y, x))
    return brng.astype(np.float64)




## === cell 5
pick_lon = df_test["pickup_longitude"].to_numpy(copy=False)
pick_lat = df_test["pickup_latitude"].to_numpy(copy=False)
drop_lon = df_test["dropoff_longitude"].to_numpy(copy=False)
drop_lat = df_test["dropoff_latitude"].to_numpy(copy=False)

coords = np.stack([pick_lon, pick_lat, drop_lon, drop_lat], axis=1).astype(
    np.float64, copy=False
)
df_test["Herv_Dist"] = distance(coords)
df_test["Manhattan_Dist"] = manhattan_approx_km_np(
    pick_lat, pick_lon, drop_lat, drop_lon
)
df_test["Bearing"] = bearing_deg_np(pick_lat, pick_lon, drop_lat, drop_lon)



## === cell 6
add_datepart(df_test, "pickup_datetime", drop=True)



## === cell 7
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
    "Manhattan_Dist",
    "Bearing",
]



## === cell 8
df_test[feature_cols].head()



## === cell 9
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



## === cell 10
tf_feature_cols = [
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

x_test_unscl_tf = df_test[tf_feature_cols].to_numpy(dtype=np.float64, copy=False)
x_test = (x_test_unscl_tf - mu) / sigma
x_test = x_test.astype(np.float32, copy=False)



## === cell 11
ckpt_prefix = os.path.join(DNN_MODEL_DIR, "model_.ckpt")
ckpt_exists = os.path.exists(ckpt_prefix) or os.path.exists(ckpt_prefix + ".index")
print("Checkpoint exists:", ckpt_exists, "|", ckpt_prefix)




## === cell 12
def train_fallback_model_and_predict(df_test, feature_cols):
    """
    Original fallback preserved for completeness, but it cannot meet a hard 600s limit
    on this dataset. It is intentionally NOT used in this optimized timeout-safe run.
    """
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    train_path = os.path.join(
        INPUT_ROOT, "new-york-city-taxi-fare-prediction", "train.csv"
    )

    nrows = 3_000_000

    usecols = [
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]

    df_tr = pd.read_csv(
        train_path,
        nrows=nrows,
        usecols=usecols,
        dtype={
            "fare_amount": "float32",
            "pickup_longitude": "float32",
            "pickup_latitude": "float32",
            "dropoff_longitude": "float32",
            "dropoff_latitude": "float32",
            "passenger_count": "int16",
        },
        parse_dates=["pickup_datetime"],
    )

    df_tr = df_tr.dropna()
    df_tr = df_tr[(df_tr["fare_amount"] > 0) & (df_tr["fare_amount"] < 250)]
    df_tr = df_tr[(df_tr["passenger_count"] >= 1) & (df_tr["passenger_count"] <= 6)]

    for col in ["pickup_longitude", "dropoff_longitude"]:
        df_tr = df_tr[(df_tr[col] > -80) & (df_tr[col] < -70)]
    for col in ["pickup_latitude", "dropoff_latitude"]:
        df_tr = df_tr[(df_tr[col] > 35) & (df_tr[col] < 45)]

    tr_pick_lon = df_tr["pickup_longitude"].to_numpy(copy=False)
    tr_pick_lat = df_tr["pickup_latitude"].to_numpy(copy=False)
    tr_drop_lon = df_tr["dropoff_longitude"].to_numpy(copy=False)
    tr_drop_lat = df_tr["dropoff_latitude"].to_numpy(copy=False)

    tr_coords = np.stack(
        [tr_pick_lon, tr_pick_lat, tr_drop_lon, tr_drop_lat], axis=1
    ).astype(np.float64, copy=False)
    df_tr["Herv_Dist"] = distance(tr_coords)
    df_tr["Manhattan_Dist"] = manhattan_approx_km_np(
        tr_pick_lat, tr_pick_lon, tr_drop_lat, tr_drop_lon
    )
    df_tr["Bearing"] = bearing_deg_np(
        tr_pick_lat, tr_pick_lon, tr_drop_lat, tr_drop_lon
    )

    add_datepart(df_tr, "pickup_datetime", drop=True)

    X = df_tr[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y = df_tr["fare_amount"].to_numpy(dtype=np.float32, copy=False)

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.1, random_state=42)

    model = ExtraTreesRegressor(
        n_estimators=600,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=False,
    )
    model.fit(X_tr, y_tr)

    va_pred = model.predict(X_va).astype(np.float64)
    rmse = mean_squared_error(y_va, va_pred, squared=False)
    print("Fallback ExtraTrees local RMSE (sanity check, not Kaggle):", rmse)

    A = np.vstack([va_pred, np.ones_like(va_pred)]).T
    a, b = np.linalg.lstsq(A, y_va.astype(np.float64), rcond=None)[0]
    print("Calibration (a, b):", float(a), float(b))

    X_te = df_test[feature_cols].to_numpy(dtype=np.float32, copy=False)
    pred = model.predict(X_te).astype(np.float64)
    pred = a * pred + b

    pred = np.clip(pred, 0.0, 500.0)
    return pred




## === cell 13
fare_pred = None

ckpt_to_use = None

if not os.path.isdir(DNN_MODEL_DIR):
    raise FileNotFoundError(
        f"Expected pretrained model directory not found: {DNN_MODEL_DIR}. "
        "The fallback training path cannot complete within the 600-second timeout."
    )

import tensorflow as tf

tf.compat.v1.disable_eager_execution()
try:
    tf.compat.v1.set_random_seed(42)
except Exception:
    pass

if ckpt_exists:
    ckpt_to_use = ckpt_prefix
else:
    latest = None
    try:
        latest = tf.train.latest_checkpoint(DNN_MODEL_DIR)
    except Exception:
        latest = None
    if latest is not None and (
        os.path.exists(latest) or os.path.exists(latest + ".index")
    ):
        ckpt_to_use = latest

if ckpt_to_use is None:
    raise FileNotFoundError(
        f"No TensorFlow checkpoint found in {DNN_MODEL_DIR}. "
        "Cannot run the fast inference path required to meet the 600-second timeout."
    )

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


tf.compat.v1.reset_default_graph()
n_inputs = 14
n_outputs = 1

X_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, n_inputs), name="X")
training_ph = tf.compat.v1.placeholder_with_default(False, shape=(), name="training")

with tf.name_scope("dnn"):
    y_pred = dense_compat(
        dnn(X_ph, training_ph),
        n_outputs,
        name="Outputs",
        kernel_initializer=he_init,
        activation=tf.nn.relu,
    )

saver = tf.compat.v1.train.Saver()

config = tf.compat.v1.ConfigProto(
    allow_soft_placement=True,
    intra_op_parallelism_threads=2,
    inter_op_parallelism_threads=2,
)

with tf.compat.v1.Session(config=config) as sess:
    saver.restore(sess, ckpt_to_use)
    print("Restored model from:", ckpt_to_use)
    y_test = sess.run(y_pred, feed_dict={X_ph: x_test})
    fare_pred = y_test.reshape(-1).astype(np.float64)
    fare_pred = np.clip(fare_pred, 0.0, 500.0)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/284445519.py in <cell line: 0>()
      6 # This preserves intended core logic (restore pretrained TF checkpoint + inference).
      7 if not os.path.isdir(DNN_MODEL_DIR):
----> 8     raise FileNotFoundError(
      9         f"Expected pretrained model directory not found: {DNN_MODEL_DIR}. "
     10         "The fallback training path cannot complete within the 600-second timeout."

FileNotFoundError: Expected pretrained model directory not found: /kaggle/input/dnn-model. The fallback training path cannot complete within the 600-second timeout.

## === cell 14
my_submission = pd.DataFrame(
    {"key": df_test["key"].astype(str).values, "fare_amount": fare_pred}
)



## === cell 15
my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)
print(my_submission.head())



## === cell 16
my_submission["fare_amount"].head()
