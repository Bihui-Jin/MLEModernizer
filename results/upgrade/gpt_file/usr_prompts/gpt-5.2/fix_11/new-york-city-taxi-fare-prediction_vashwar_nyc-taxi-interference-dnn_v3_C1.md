# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import re
import sys
import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

INPUT_ROOT = "/kaggle/input" if os.path.isdir("/kaggle/input") else "../input"
print("INPUT_ROOT:", INPUT_ROOT)
try:
    print("Listing INPUT_ROOT:", os.listdir(INPUT_ROOT)[:50])
except Exception as e:
    print("Could not list INPUT_ROOT:", repr(e))



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
    Adds date-related columns with the SAME NAMES as the original code expected.
    Bugfix: handle timezone-aware datetimes (e.g., datetime64[ns, UTC]) without
    triggering np.issubdtype TypeError; keep identical feature meanings.
    """
    fld = df[fldname]

    if not pd.api.types.is_datetime64_any_dtype(fld):
        df[fldname] = fld = pd.to_datetime(
            fld, infer_datetime_format=True, utc=False, errors="coerce"
        )

    if pd.api.types.is_datetime64tz_dtype(fld):
        df[fldname] = fld = fld.dt.tz_localize(None)

    targ_pre = re.sub("[Dd]ate$", "", fldname)

    df[targ_pre + "Year"] = fld.dt.year
    df[targ_pre + "Month"] = fld.dt.month

    try:
        df[targ_pre + "Week"] = fld.dt.isocalendar().week.astype(np.int64).to_numpy()
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


def manhattan_approx_km(df):
    dlat = np.abs(df["dropoff_latitude"].to_numpy() - df["pickup_latitude"].to_numpy())
    dlon = np.abs(
        df["dropoff_longitude"].to_numpy() - df["pickup_longitude"].to_numpy()
    )
    return (111.0 * dlat + 85.0 * dlon).astype(np.float64)


def bearing_deg(df):
    lon1 = np.radians(df["pickup_longitude"].to_numpy())
    lat1 = np.radians(df["pickup_latitude"].to_numpy())
    lon2 = np.radians(df["dropoff_longitude"].to_numpy())
    lat2 = np.radians(df["dropoff_latitude"].to_numpy())
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brng = np.degrees(np.arctan2(y, x))
    return brng.astype(np.float64)




## === cell 5
coords = (
    df_test[
        ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
    ]
    .to_numpy(copy=False)
    .astype(np.float64, copy=False)
)
df_test["Herv_Dist"] = distance(coords)
df_test["Manhattan_Dist"] = manhattan_approx_km(df_test)
df_test["Bearing"] = bearing_deg(df_test)



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
    Score-improving fallback used only when the TF checkpoint is unavailable.
    Keeps the same engineered features and learns a simple regression on a chunk.
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

    tr_coords = (
        df_tr[
            [
                "pickup_longitude",
                "pickup_latitude",
                "dropoff_longitude",
                "dropoff_latitude",
            ]
        ]
        .to_numpy(copy=False)
        .astype(np.float64, copy=False)
    )
    df_tr["Herv_Dist"] = distance(tr_coords)
    df_tr["Manhattan_Dist"] = manhattan_approx_km(df_tr)
    df_tr["Bearing"] = bearing_deg(df_tr)

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

use_fallback = True
ckpt_to_use = None

if os.path.isdir(DNN_MODEL_DIR):
    if ckpt_exists:
        ckpt_to_use = ckpt_prefix
        use_fallback = False
    else:
        try:
            import tensorflow as tf  # only import if we might actually use it

            tf.compat.v1.disable_eager_execution()
            try:
                tf.compat.v1.set_random_seed(42)
            except Exception:
                pass
            latest = None
            try:
                latest = tf.train.latest_checkpoint(DNN_MODEL_DIR)
            except Exception:
                latest = None
            if latest is not None and (
                os.path.exists(latest) or os.path.exists(latest + ".index")
            ):
                ckpt_to_use = latest
                use_fallback = False
        except Exception as e:
            print("TF import/checkpoint resolution failed:", repr(e))
            use_fallback = True

if use_fallback:
    fare_pred = train_fallback_model_and_predict(df_test, feature_cols)
else:
    import tensorflow as tf

    tf.compat.v1.disable_eager_execution()
    try:
        tf.compat.v1.set_random_seed(42)
    except Exception:
        pass

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
    training_ph = tf.compat.v1.placeholder_with_default(
        False, shape=(), name="training"
    )

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
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
        allow_soft_placement=True,
    )

    with tf.compat.v1.Session(config=config) as sess:
        saver.restore(sess, ckpt_to_use)
        print("Restored model from:", ckpt_to_use)
        y_test = sess.run(y_pred, feed_dict={X_ph: x_test})
        fare_pred = y_test.reshape(-1).astype(np.float64)
        fare_pred = np.clip(fare_pred, 0.0, 500.0)



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
