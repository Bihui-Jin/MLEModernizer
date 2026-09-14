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

10.04587

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.59106) has done: 'I fix the TensorFlow import crash caused by an incompatibility between TensorFlow 2.18 and protobuf 6 by forcing the pure-Python protobuf implementation before importing TensorFlow. Then I update deprecated/removed pandas and TensorFlow APIs: replace `weekday_name` with `day_name()` and replace the removed `tf.estimator` / `tf.contrib` pipeline with an equivalent Keras wide-and-deep style model (linear + DNN) trained on the same engineered features. I also correct a logic bug where engineered features on `test` were accidentally discarded, and vectorize the distance feature computation so it runs fast and reliably within the time limit. Finally, I ensure a valid `submission_file.csv` is written with exactly `key,fare_amount` aligned to the test rows.'
- What this solution (achieved 75.71925) has done: 'I fix the TensorFlow import crash by switching to the safer protobuf compatibility setting (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` plus a `protobuf` 3.x API mode env var) and importing TensorFlow only after those env vars are set. I also make the wide-and-deep categorical vocab sizes consistent with the bucketization logic (digitize produces `nbuckets+1` levels per axis, so combined-location vocab must use `(nbuckets+2)` not `(nbuckets+1)`), which is a logic bug that can both hurt RMSE and occasionally cause out-of-range embedding indices. Finally, I keep the architecture/training loop the same but add a small learning-rate adjustment (score-improving but minimal) and ensure the submission CSV is always written with exactly `key,fare_amount` aligned to the test rows.'
- What this solution (achieved 5.6237) has done: 'I fix the TensorFlow import crash caused by the protobuf 6 / TF 2.18 incompatibility by forcing the pure-Python protobuf runtime and (critically) downgrading protobuf inside the notebook before importing TensorFlow. I also make the data path robust to your provided directory layout by reading from `../input/new-york-city-taxi-fare-prediction/` when available (falling back to `../input/`). These changes are execution-blocking bug fixes and keep the model/features/training loop identical, so score behavior should remain consistent while producing a valid `submission_file.csv`.'
- What this solution (achieved 5.25493) has done: 'Your current RMSE (5.6237) is worse than the target (4.29243), so we should improve performance with the smallest safe changes that don’t alter the model architecture or training loop structure. The biggest score drag here is that you train on only 200k rows and you *don’t apply the same geographic/validity filters to test*, causing out-of-distribution inputs at inference; we (1) increase the training sample size modestly within time/memory, and (2) apply the same coordinate/passenger constraints to test and fill invalid rows with a stable fallback prediction (training median). These changes keep the same features, model, loss, optimizer, and epoch schedule, but make train/test distributions consistent and usually reduce RMSE substantially. We also keep submission row alignment exactly matching the original test order by predicting only for valid rows and then reassembling.'
- What this solution (achieved 5.2502) has done: 'Your current score (5.25493 RMSE; lower is better) is still worse than the target (4.29243), so we should make small changes that legitimately improve generalization without altering the model architecture, loss, or training loop structure. The biggest low-risk gain here is to reduce training noise and distribution mismatch by tightening the training data cleaning to remove extreme/clearly-wrong fares (which otherwise dominate MSE) while keeping your existing geographic/passenger/distance filters intact. To avoid harming inference semantics, we keep the same test validity mask/fallback logic and only adjust the training filter plus make the fallback a bit more robust by using the cleaned-training median. These changes are minimal, fast, and typically move RMSE down toward the target.'
- What this solution (achieved 5.23136) has done: 'We need to reduce RMSE from 5.2502 toward 4.29243 (lower is better), so we should make small, low-risk changes that improve generalization without changing the model architecture or training loop. The biggest remaining mismatch is that training removes extreme/outlier coordinates but still allows unrealistic long trips and some noisy/invalid labels; adding a very light, standard NYC-taxi cleaning step (cap extreme distances and remove obviously-bad fares relative to distance) typically reduces MSE without altering features or the model. To keep inference semantics consistent, we apply the same distance cap to the test validity mask and keep the exact same fallback/reassembly logic so submission order and format remain correct. These are minimal filtering tweaks plus deterministic seeding, and they keep the same Keras wide+deep model, loss, optimizer, epochs, and feature engineering.'
- What this solution (achieved 5.27656) has done: 'Your current RMSE (5.23136, lower is better) is still worse than the target (4.29243), so we should make a small, low-risk improvement without changing the wide+deep Keras model, features, loss, optimizer, or training loop structure. The biggest remaining issue is label noise/outliers still slipping through the training set, which disproportionately hurts MSE/RMSE; we add one standard NYC-taxi cleaning rule that’s consistent with your existing distance feature: remove “implausibly cheap/expensive for the trip length” rides using a slightly tighter fare-per-km band. To keep train/test semantics consistent, we keep your test validity mask and fallback logic unchanged (only training is cleaned more), and we also use a stable split by key-hash instead of `np.random.rand` to reduce variance between runs while preserving the same 80/20 split idea. These changes are minimal and are expected to move RMSE down toward the target without altering the model itself.'
- What this solution (achieved 10.04587) has done: 'Your current RMSE (5.27656; lower is better) is still above the target (4.29243), so the smallest reliable improvement is to reduce label noise/outliers that dominate MSE without changing the model/features/training loop. I keep your wide+deep Keras model, engineered features, optimizer, and 100-epoch training exactly the same, but I (1) add two standard NYC taxi cleaning rules on the training set only: remove extreme `fare_amount` values (above a higher but safe cap) and remove rides with implausible average speeds given your already-computed distance, which typically reduces RMSE meaningfully. I also ensure the test-time validity mask uses the same “distance>0.2 and <MAX_DIST_KM” (already done) and keep your fallback/reassembly logic unchanged to preserve submission alignment. These are minimal filter tweaks intended to move the score downward toward the target band without altering the modeling semantics.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
import shutil

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _ensure_protobuf_3x():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pbver

        major = int(pbver.split(".")[0])
        if major >= 4:
            raise RuntimeError(
                f"protobuf version {pbver} is too new for stable TF import here"
            )
    except Exception:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "--no-deps",
                "protobuf==3.20.3",
            ]
        )
        import importlib
        import google.protobuf

        importlib.reload(google.protobuf)


_ensure_protobuf_3x()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

print("../", os.listdir("../"))
print("../input", os.listdir("../input"))
print("tf version: ", tf.__version__)



## === cell 1
BASE = "../input/new-york-city-taxi-fare-prediction"
if not os.path.exists(os.path.join(BASE, "train.csv")):
    BASE = "../input"

NROWS = 1_000_000

df = pd.read_csv(
    os.path.join(BASE, "train.csv"), nrows=NROWS, parse_dates=["pickup_datetime"]
)
test = pd.read_csv(os.path.join(BASE, "test.csv"), parse_dates=["pickup_datetime"])



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
MAX_DIST_KM = 50.0  # light cap; typical NYC rides are far below this

dfc = df[
    ((df.pickup_longitude >= -75.0) & (df.pickup_longitude <= -72))
    & ((df.pickup_latitude >= 38) & (df.pickup_latitude <= 42))
    & ((df.dropoff_longitude >= -75.0) & (df.dropoff_longitude <= -72))
    & ((df.dropoff_latitude >= 38) & (df.dropoff_latitude <= 42))
    & (df.fare_amount > 2.5)
    & (df.fare_amount < 250.0)
    & (df.passenger_count > 0)
    & (df.passenger_count < 7)
    & (df.distance > 0.2)
    & (df.distance < MAX_DIST_KM)
].copy()

fare_per_km = dfc["fare_amount"] / np.maximum(dfc["distance"], 0.5)

dfc = dfc[(fare_per_km > 1.5) & (fare_per_km < 30.0)].copy()

trip_hours = dfc["pickup_datetime"].astype("datetime64[ns]").diff()
short_mask = dfc["distance"] < 1.0
dfc = dfc[~(short_mask & (dfc["fare_amount"] > 50.0))].copy()
long_mask = dfc["distance"] > 20.0
dfc = dfc[~(long_mask & (dfc["fare_amount"] < 15.0))].copy()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2731619019.py in <cell line: 0>()
     27 # which are often GPS glitches and noisy labels; this improves RMSE without changing
     28 # any model/features/training loop.
---> 29 trip_hours = dfc["pickup_datetime"].astype("datetime64[ns]").diff()
     30 # Use per-row trip duration from the key timestamp isn't available; instead, use a
     31 # conservative proxy: filter only based on distance itself by bounding "speed" using

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in astype(self, dtype, copy, errors)
   6641         else:
   6642             # else, only a single dtype is given
-> 6643             new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
   6644             res = self._constructor_from_mgr(new_data, axes=new_data.axes)
   6645             return res.__finalize__(self, method="astype")

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in astype(self, dtype, copy, errors)
    428             copy = False
    429 
--> 430         return self.apply(
    431             "astype",
    432             dtype=dtype,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in apply(self, f, align_keys, **kwargs)
    361                 applied = b.apply(f, **kwargs)
    362             else:
--> 363                 applied = getattr(b, f)(**kwargs)
    364             result_blocks = extend_blocks(applied, result_blocks)
    365 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in astype(self, dtype, copy, errors, using_cow, squeeze)
    756             values = values[0, :]  # type: ignore[call-overload]
    757 
--> 758         new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
    759 
    760         new_values = maybe_coerce_values(new_values)

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array_safe(values, dtype, copy, errors)
    235 
    236     try:
--> 237         new_values = astype_array(values, dtype, copy=copy)
    238     except (ValueError, TypeError):
    239         # e.g. _astype_nansafe can fail on object-dtype of strings

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array(values, dtype, copy)
    177     if not isinstance(values, np.ndarray):
    178         # i.e. ExtensionArray
--> 179         values = values.astype(dtype, copy=copy)
    180 
    181     else:

/usr/local/lib/python3.11/dist-packages/pandas/core/arrays/datetimes.py in astype(self, dtype, copy)
    718             #  values.tz_convert("UTC").tz_localize(None), which did not match
    719             #  the Series behavior
--> 720             raise TypeError(
    721                 "Cannot use .astype to convert from timezone-aware dtype to "
    722                 "timezone-naive dtype. Use obj.tz_localize(None) or "

TypeError: Cannot use .astype to convert from timezone-aware dtype to timezone-naive dtype. Use obj.tz_localize(None) or obj.tz_convert('UTC').tz_localize(None) instead.

## === cell 9
split_hash = pd.util.hash_pandas_object(dfc["key"], index=False).astype("uint64")
msk = (split_hash % 10) < 8  # ~80% train, ~20% eval

traindf = dfc[msk].drop(["key", "pickup_datetime"], axis=1)
evaldf = dfc[~msk].drop(["key", "pickup_datetime"], axis=1)



## === cell 10
testdf = add_feats(test)

test_valid_mask = (
    ((testdf.pickup_longitude >= -75.0) & (testdf.pickup_longitude <= -72))
    & ((testdf.pickup_latitude >= 38) & (testdf.pickup_latitude <= 42))
    & ((testdf.dropoff_longitude >= -75.0) & (testdf.dropoff_longitude <= -72))
    & ((testdf.dropoff_latitude >= 38) & (testdf.dropoff_latitude <= 42))
    & (testdf.passenger_count > 0)
    & (testdf.passenger_count < 7)
    & (testdf.distance > 0.2)
    & (testdf.distance < MAX_DIST_KM)
)

testdf = testdf.drop(["key", "pickup_datetime"], axis=1)



## === cell 11
traindf.weekday.head()




## === cell 12
def build_model_columns(nbuckets=10):
    return nbuckets




## === cell 13
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

    nb = len(lat_edges) + 1  # digitize yields values in [0..len(edges)]
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




## === cell 14
def build_estimator(model_dir, nbuckets=10):
    nb_axis = nbuckets + 1  # bins per axis produced by digitize
    ploc_vocab = nb_axis * nb_axis
    pd_pair_vocab = ploc_vocab * ploc_vocab
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
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss="mse",
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model




## === cell 15
OUTDIR = "./taxi_trained"
shutil.rmtree(OUTDIR, ignore_errors=True)
os.makedirs(OUTDIR, exist_ok=True)

BATCH_SIZE = 512

nbuckets = 10
Xnum_train, Xcat_train = _prepare_keras_inputs(traindf, nbuckets=nbuckets)
Xnum_eval, Xcat_eval = _prepare_keras_inputs(evaldf, nbuckets=nbuckets)

testdf_valid = testdf.loc[test_valid_mask.values].copy()
Xnum_test_valid, Xcat_test_valid = _prepare_keras_inputs(
    testdf_valid, nbuckets=nbuckets
)

y_train = traindf["fare_amount"].astype("float32").to_numpy()
y_eval = evaldf["fare_amount"].astype("float32").to_numpy()

train_inputs = {"numeric": Xnum_train.to_numpy(), **Xcat_train}
eval_inputs = {"numeric": Xnum_eval.to_numpy(), **Xcat_eval}
test_inputs_valid = {"numeric": Xnum_test_valid.to_numpy(), **Xcat_test_valid}

np.random.seed(1)
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



## === cell 16
pred_valid = model.predict(test_inputs_valid, batch_size=1024, verbose=0).reshape(-1)



## === cell 17
fallback = float(np.median(y_train))

pred_full = np.full(shape=(len(test),), fill_value=fallback, dtype="float64")
pred_full[test_valid_mask.values] = pred_valid.astype("float64")

pred_full = np.clip(pred_full, 0.0, None)



## === cell 18
dataset_output = pd.DataFrame({"key": test["key"].values, "fare_amount": pred_full})



## === cell 19
dataset_output.head()



## === cell 20
dataset_output.to_csv("submission_file.csv", index=False)
print("Wrote submission_file.csv with shape:", dataset_output.shape)
print(dataset_output.columns.tolist())
print("Valid test rows:", int(test_valid_mask.sum()), " / ", len(test))
print("Fallback fare used for invalid rows:", fallback)
print("Training rows used after cleaning:", len(traindf), "Eval rows:", len(evaldf))
