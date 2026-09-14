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

geopandas==0.14.4
ipywidgets==8.1.5
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0
tqdm==4.67.1

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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["figure.figsize"] = 16, 9
from tqdm import tqdm

try:
    from ipywidgets import widgets  # noqa: F401
except Exception:
    widgets = None

RUN_EDA = False




## === cell 1
def read_csv_sampled(
    path,
    frac,
    random_state=1989,
    n_rows_hint=55_423_857,
    chunksize=1_000_000,
    cache_dir="/kaggle/working",
):
    """
    Deterministically read an approximate fraction of a large CSV by chunked sampling.

    - Deterministic given random_state.
    - Equivalent intent to original: trains on a random subset of original rows.
    - Much faster than skiprows(callback) which triggers a Python call per input row.
    """
    frac = float(frac)
    if not (0 < frac <= 1.0):
        raise ValueError("frac must be in (0, 1].")

    dtypes = {
        "fare_amount": "float32",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int16",
        "key": "string",
        "pickup_datetime": "string",
    }

    base = os.path.splitext(os.path.basename(path))[0]
    cache_path = os.path.join(
        cache_dir, f"{base}_sample_frac{frac:.3f}_seed{random_state}.parquet"
    )
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        df["pickup_datetime"] = pd.to_datetime(
            df["pickup_datetime"], errors="coerce", utc=False
        )
        return df

    rng = np.random.RandomState(random_state)

    n_data_rows = int(n_rows_hint) - 1
    n_keep_target = int(round(n_data_rows * frac))
    n_keep_target = max(1, min(n_keep_target, n_data_rows))

    kept = []
    kept_rows = 0

    reader = pd.read_csv(
        path,
        dtype=dtypes,
        low_memory=False,
        chunksize=chunksize,
    )

    for chunk_idx, chunk in enumerate(reader):
        chunk_rng = np.random.RandomState(random_state + chunk_idx)
        n = len(chunk)
        k = int(round(n * frac))
        if k <= 0:
            continue

        idx = chunk_rng.choice(n, size=k, replace=False)
        sampled = chunk.iloc[idx]
        kept.append(sampled)
        kept_rows += len(sampled)

        if kept_rows >= n_keep_target:
            break

    df = pd.concat(kept, ignore_index=True)
    df["pickup_datetime"] = pd.to_datetime(
        df["pickup_datetime"], errors="coerce", utc=False
    )

    try:
        df.to_parquet(cache_path, index=False)
    except Exception:
        pass

    return df




## === cell 2
TRAIN_CANDIDATES = [
    "/kaggle/input/train.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv",
    "/kaggle/data/train.csv",
    "/kaggle/data/new-york-city-taxi-fare-prediction/train.csv",
    "../input/train.csv",
]
TEST_CANDIDATES = [
    "/kaggle/input/test.csv",
    "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv",
    "/kaggle/data/test.csv",
    "/kaggle/data/new-york-city-taxi-fare-prediction/test.csv",
    "../input/test.csv",
]


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these paths exist: {paths}")


TRAIN_PATH = _first_existing(TRAIN_CANDIDATES)
TEST_PATH = _first_existing(TEST_CANDIDATES)

print("Using TRAIN_PATH:", TRAIN_PATH)
print("Using TEST_PATH :", TEST_PATH)

df_train_sample = read_csv_sampled(
    TRAIN_PATH, 0.10, random_state=1989, n_rows_hint=55_423_857
)




## === cell 3
if RUN_EDA:
    df_train_sample.info()




## === cell 4
df_test = pd.read_csv(
    TEST_PATH,
    parse_dates=["pickup_datetime"],
    dtype={
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int16",
        "key": "string",
    },
)
if RUN_EDA:
    df_test.info()




## === cell 5
df_train_sample.dropna(inplace=True)
df_train_sample = df_train_sample[df_train_sample["pickup_datetime"].notna()].copy()




## === cell 6
is_weird = df_train_sample["fare_amount"] < 0
is_weird |= ~df_train_sample["pickup_latitude"].between(40, 42)
is_weird |= ~df_train_sample["pickup_longitude"].between(-75, -72)
is_weird |= ~df_train_sample["dropoff_latitude"].between(40, 42)
is_weird |= ~df_train_sample["dropoff_longitude"].between(-75, -72)
is_weird |= df_train_sample["passenger_count"] == 0
print(is_weird.sum())

df_train_sample = df_train_sample[~is_weird].copy()




## === cell 7
if RUN_EDA:
    df_train_sample.describe()




## === cell 8
if RUN_EDA:
    df_test.describe()




## === cell 9
if RUN_EDA:
    sns.histplot(
        df_train_sample["fare_amount"].apply(np.log1p).dropna(), bins=100, kde=False
    )




## === cell 10
if RUN_EDA:
    plt.scatter(
        df_train_sample["pickup_longitude"],
        df_train_sample["pickup_latitude"],
        c=df_train_sample["fare_amount"].apply(np.log1p),
        alpha=0.7,
        s=1,
        lw=0,
    )
    plt.xlim(*np.percentile(df_train_sample["pickup_longitude"], [1, 99]))
    plt.ylim(*np.percentile(df_train_sample["pickup_latitude"], [1, 99]))
    plt.colorbar()




## === cell 11
def prep_data(df, shuffle=False):
    """
    Prepare categorical time/passenger features and scaled lat/lng degrees.

    Keep core logic: same feature set, same embedding-id approach.
    """
    dt = df["pickup_datetime"]

    iso_week = dt.dt.isocalendar().week.astype(np.int16)

    hour = dt.dt.hour.astype(np.int16)  # 0-23
    weekday = dt.dt.weekday.astype(np.int16) + 24  # 24-30
    dayofyear = dt.dt.dayofyear.astype(np.int16) + 31  # 1..366 -> 32..397
    iso_week_cat = iso_week.astype(np.int16) + 397  # 1..53 -> 398..450

    year = dt.dt.year.astype(np.int16)
    year_cat = (year - 2009 + 451).astype(np.int16)  # start after 450
    passenger = (df["passenger_count"].astype(np.int16) + 470).astype(np.int16)

    X_cat = np.vstack([hour, weekday, dayofyear, iso_week_cat, year_cat, passenger]).T

    X_deg = df[
        ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
    ].values.astype("float32")
    X_deg /= 180.0

    if "fare_amount" in df.columns:
        y = df["fare_amount"].values.astype("float32")

    if shuffle:
        rnd_ind = np.random.permutation(len(df))
        X_cat = X_cat[rnd_ind]
        X_deg = X_deg[rnd_ind]
        if "fare_amount" in df.columns:
            y = y[rnd_ind]

    if "fare_amount" in df.columns:
        return (X_cat, X_deg), y
    return (X_cat, X_deg)




## === cell 12
np.random.seed(1989)
(X_cat, X_deg), y = prep_data(df_train_sample, shuffle=True)




## === cell 13
import tensorflow as tf
from tensorflow.keras import layers as lyr
from tensorflow.keras import activations as act
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

try:
    tf.random.set_seed(1989)
except Exception:
    pass




## === cell 14
def K_haversine_bearing(x):
    R = 6371e3

    x_rad = x * np.pi
    lat1, lng1, lat2, lng2 = x_rad[:, 0], x_rad[:, 1], x_rad[:, 2], x_rad[:, 3]

    dlat = lat2 - lat1
    dlng = lng2 - lng1

    a = K.sin(dlat / 2) * K.sin(dlat / 2) + K.cos(lat1) * K.cos(lat2) * K.sin(
        dlng / 2
    ) * K.sin(dlng / 2)
    c = 2 * tf.atan2(K.sqrt(a), K.sqrt(1 - a))

    d = K.log((R * c) + 1)

    x_b = K.sin(dlng) * K.cos(lat2)
    y_b = (K.cos(lat1) * K.sin(lat2)) - (K.sin(lat1) * K.cos(lat2) * K.cos(dlng))
    b = tf.atan2(x_b, y_b) / np.pi

    return K.concatenate([K.reshape(d, (-1, 1)), K.reshape(b, (-1, 1))])




## === cell 15
(X_cat_test_tmp, X_deg_test_tmp) = prep_data(df_test)
max_cat_id = int(max(X_cat.max(), X_cat_test_tmp.max())) + 1

lyr_latlng_input = lyr.Input(X_deg.shape[1:])
lyr_haversine_bearing = lyr.Lambda(K_haversine_bearing, name="haversine")(
    lyr_latlng_input
)

lyr_cat_input = lyr.Input(X_cat.shape[1:])
lyr_cat_embeddings = lyr.Embedding(max_cat_id, 16)(lyr_cat_input)
lyr_cat_flatten = lyr.Flatten()(lyr_cat_embeddings)

lyr_concat = lyr.concatenate([lyr_latlng_input, lyr_haversine_bearing, lyr_cat_flatten])

lyr_dense1 = lyr.Dense(256, activation=act.selu)(lyr_concat)
lyr_dense2 = lyr.Dense(256, activation=act.selu)(lyr_dense1)
lyr_dense3 = lyr.Dense(256, activation=act.selu)(lyr_dense2)
lyr_out = lyr.Dense(1, activation=act.selu)(lyr_dense3)

model = Model([lyr_latlng_input, lyr_cat_input], lyr_out)
model.summary()




## === cell 16
model.compile(loss="mse", optimizer="nadam")




## === cell 17
model.fit([X_deg, X_cat], y, batch_size=512, epochs=50, validation_split=0.3)




## === cell 18
n_total = len(X_deg)
n_val = int(n_total * 0.3)
X_deg_val = X_deg[-n_val:]
X_cat_val = X_cat[-n_val:]
y_val = y[-n_val:]

y_pred = np.clip(
    model.predict([X_deg_val, X_cat_val], batch_size=1024, verbose=1)[:, 0],
    0,
    None,
)
err = pd.Series(y_val - y_pred)
if RUN_EDA:
    sns.histplot(err, bins=100, kde=False)
print(f"rmse: {(err**2).mean()**0.5:.3f}")
if RUN_EDA:
    err.describe()




## === cell 19
X_cat_test, X_deg_test = X_cat_test_tmp, X_deg_test_tmp




## === cell 20
y_test = np.clip(
    model.predict([X_deg_test, X_cat_test], batch_size=512, verbose=1), 0, None
)




## === cell 21
if RUN_EDA:
    sns.histplot(np.log1p(y), bins=100, kde=False)
    sns.histplot(np.log1p(y_test[:, 0]), bins=100, kde=False)




## === cell 22
df_sub = pd.DataFrame({"key": df_test["key"].values, "fare_amount": y_test[:, 0]})
if RUN_EDA:
    df_sub.head()




## === cell 23
df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)
print(df_sub.head())
