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

# 5. Target score

3.97307

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 8.74843) has done: 'I fix the datetime feature bug by replacing the removed `dt.weekofyear` accessor with the stable ISO calendar week (`dt.isocalendar().week`) while keeping the same categorical offset scheme. I also fix the Keras import/runtime crash by switching to `tf.keras` (compatible with the provided TensorFlow) while preserving the exact model architecture and training loop. Next, I correct a small data-prep unpacking mistake for the test set and ensure the submission file has exactly the required columns and is written as `submission.csv`. These changes are execution-stability fixes and should also improve the score versus “no submission” by enabling the model to train and predict properly.'
- What this solution (achieved 7.30552) has done: 'I fix the TensorFlow/Keras import crash caused by an incompatible protobuf version by forcing the pure-Python protobuf implementation before importing TensorFlow, which is a common Kaggle runtime workaround and should be score-neutral. I also fix a small bug in test feature unpacking (the prep function returns a tuple) so inference runs correctly end-to-end and actually writes `submission.csv`. Finally, I correct the train/validation split logic used for the quick RMSE check so it evaluates the same validation subset that `validation_split=0.3` uses (last 30% after shuffling), without changing the model or training behavior; this is score-neutral but prevents misleading diagnostics.'
- What this solution (achieved 5.63691) has done: 'I fix the TensorFlow import crash that comes from the protobuf runtime mismatch by forcing the pure‑Python protobuf implementation earlier and using a safe, consistent TensorFlow/Keras import path. I also fix the test-set unpacking bug (your `prep_data` returns a tuple of arrays) so inference uses the correct `[X_deg_test, X_cat_test]` inputs and runs end-to-end. To move RMSE toward the target without changing the model/training loop, I increase the training sample fraction modestly (still chunk-sampled) so the exact same network trains on more data, which is the smallest legitimate lever here. Finally, I keep the submission formatting strict (`key,fare_amount`) and ensure the file is written as `submission.csv`.'
- What this solution (achieved 6.32668) has done: 'I fix the TensorFlow import crash by enforcing the pure-Python protobuf implementation *before* any TensorFlow/Keras import and by also setting the `protobuf` runtime version to 3 (which avoids the `MessageFactory.GetPrototype` mismatch seen in Kaggle images). I also correct the test feature unpacking to match `prep_data`’s return structure so inference always uses `[X_deg_test, X_cat_test]` in the right order. These changes are execution-stability fixes and should be score-improving vs. the current broken run, while keeping the same model architecture, loss, and training loop. Finally, I keep the submission format strict and always write `submission.csv` with columns `key,fare_amount`.'
- What this solution (achieved 5.55942) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by ensuring the protobuf runtime is forced to the pure-Python implementation *and* by importing `google.protobuf` (and disabling C descriptors) before TensorFlow is ever imported; this is the minimal, standard Kaggle workaround and is score-neutral. I also fix the test feature unpacking bug (your `prep_data` returns `(X_cat, X_deg)`, but later code expects `X_cat_test, X_deg_test`) to ensure inference uses the correct input ordering. Finally, I update the input paths to the actual provided Kaggle paths (fallback-safe), so the script reliably reads the data and always writes a valid `submission.csv` with exactly `key,fare_amount`.'
- What this solution (achieved 5.91088) has done: 'I fix the TensorFlow import crash caused by the protobuf 6.x incompatibility by forcing the pure-Python protobuf runtime and additionally downgrading the protobuf package in-notebook to a TF-compatible 4.x version before importing TensorFlow. This is an execution-stability fix and does not change your model architecture, loss, or training loop semantics. I also fix the test feature unpacking so inference uses the correct `(X_cat, X_deg)` ordering returned by `prep_data`, which prevents silent input mixups and should improve RMSE toward your target. Finally, I keep the same submission formatting but ensure the output is always written as `submission.csv` with exactly `key,fare_amount`.'
- What this solution (achieved 5.61415) has done: 'Your current gap to the target RMSE is large (5.91088 → 3.97307, lower is better), so the smallest legitimate lever that preserves your exact model/training loop is to train on more representative data without changing architecture or loss. I (1) increase the chunk-sampling fraction modestly and make sampling deterministic across chunks, (2) fix a subtle validation bug (you currently evaluate on the last 30% of a shuffled array, which doesn’t match Keras’s `validation_split` behavior), and (3) fix the test feature unpacking so you always feed `[X_deg, X_cat]` consistently (it’s currently reversed in cell 19). These are minimal changes aimed at improving generalization and ensuring evaluation/prediction alignment, and they still write a valid `submission.csv`.'
- What this solution (achieved 5.29112) has done: 'Your current RMSE (5.61415) is worse than the target (3.97307), so we should improve generalization with the smallest changes that don’t alter the model or training loop. The biggest score drag here is a subtle but severe feature bug: `dt.dt.dayofyear + 30` overlaps numerically with `iso_week + 396`, mixing two different categorical fields inside the shared embedding space; fixing the offsets keeps the exact same feature set and embedding approach but prevents collisions. I also fix the test-set unpacking so `prep_data` outputs are assigned in the correct order (it currently swaps them), ensuring inference uses the same feature mapping as training. These two fixes are directly score-relevant while preserving architecture, loss, and training semantics, and the script still write a valid `submission.csv`.'
- What this solution (achieved 5.43464) has done: 'Your current RMSE (5.29112) is still well above the target (3.97307), so the smallest legitimate way to move toward the target without changing the model/training loop is to improve data quality and make training/test feature handling consistent. I (1) fix a remaining bug where the test-set outputs from `prep_data` are unpacked in the wrong order (this can severely hurt predictions), (2) correct the categorical offset scheme in `prep_data` to truly avoid collisions (the current comment says “no overlap” but the math still overlaps), and (3) make the embedding vocabulary size robust by using the max id from both train and test categorical features so unseen test years/passenger counts don’t cause out-of-vocabulary issues. These are minimal, score-relevant fixes that preserve architecture, loss, and training behavior, and still write a valid `submission.csv`.'
- What this solution (achieved 6.16783) has done: 'The timeout is dominated by reading and sampling 10% of a 55M-row CSV (hundreds of chunk reads + per-chunk `.sample()` + concatenation) and by an unnecessary runtime `pip install` of protobuf. To keep the same core logic and training semantics, the fastest safe fix is to read a deterministic random subset directly via `skiprows` (single-pass CSV read of only the needed rows) with explicit dtypes and fast parsing, then run the exact same cleaning, feature engineering, model, and training. We also remove the protobuf install block (it forces slow network/pip work) and rely on the already-installed environment; correctness of the ML pipeline is unchanged. Finally, we avoid computing `X_cat_test_tmp` twice by building it once and reusing it, reducing redundant preprocessing.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

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
    cache_dir="/kaggle/working",
    chunksize=2_000_000,
):
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
        "key": "object",
        "pickup_datetime": "object",
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

    read_kwargs = dict(
        dtype=dtypes,
        low_memory=False,
        chunksize=chunksize,
    )
    try:
        read_kwargs["engine"] = "pyarrow"
    except Exception:
        pass

    reader = pd.read_csv(path, **read_kwargs)

    parquet_ok = True
    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
    except Exception:
        parquet_ok = False

    if parquet_ok:
        import pyarrow as pa
        import pyarrow.parquet as pq

        writer = None
        wrote_any = False
        for chunk in reader:
            n = len(chunk)
            keep = rng.random_sample(n) < frac
            if not keep.any():
                continue
            sub = chunk.loc[keep]
            table = pa.Table.from_pandas(sub, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    cache_path, table.schema, compression="snappy", use_dictionary=True
                )
            writer.write_table(table)
            wrote_any = True
        if writer is not None:
            writer.close()

        if not wrote_any:
            df = pd.DataFrame(columns=list(dtypes.keys()))
        else:
            df = pd.read_parquet(cache_path)
        df["pickup_datetime"] = pd.to_datetime(
            df["pickup_datetime"], errors="coerce", utc=False
        )
        return df

    kept_parts = []
    for chunk in reader:
        n = len(chunk)
        keep = rng.random_sample(n) < frac
        if keep.any():
            kept_parts.append(chunk.loc[keep])

    if not kept_parts:
        df = pd.DataFrame(columns=list(dtypes.keys()))
    else:
        df = pd.concat(kept_parts, axis=0, ignore_index=True, copy=False)

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



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2194310510.py in <cell line: 0>()
     28 print("Using TEST_PATH :", TEST_PATH)
     29 
---> 30 df_train_sample = read_csv_sampled(
     31     TRAIN_PATH, 0.10, random_state=1989, n_rows_hint=55_423_857
     32 )

/tmp/ipykernel_11/4221640811.py in read_csv_sampled(path, frac, random_state, n_rows_hint, cache_dir, chunksize)
     54         pass
     55 
---> 56     reader = pd.read_csv(path, **read_kwargs)
     57 
     58     # Incremental parquet write to avoid holding all sampled chunks in RAM and avoid a huge concat cost.

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    606 
    607         if chunksize is not None:
--> 608             raise ValueError(
    609                 "The 'chunksize' option is not supported with the 'pyarrow' engine"
    610             )

ValueError: The 'chunksize' option is not supported with the 'pyarrow' engine

## === cell 3
if RUN_EDA:
    df_train_sample.info()



## === cell 4
df_test_read_kwargs = dict(
    parse_dates=["pickup_datetime"],
    dtype={
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "passenger_count": "int16",
        "key": "object",
    },
)
try:
    df_test_read_kwargs["engine"] = "pyarrow"
except Exception:
    pass

df_test = pd.read_csv(TEST_PATH, **df_test_read_kwargs)
if RUN_EDA:
    df_test.info()



## === cell 5
df_train_sample.dropna(inplace=True)
df_train_sample = df_train_sample[df_train_sample["pickup_datetime"].notna()].copy()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2483707669.py in <cell line: 0>()
----> 1 df_train_sample.dropna(inplace=True)
      2 df_train_sample = df_train_sample[df_train_sample["pickup_datetime"].notna()].copy()
      3 

NameError: name 'df_train_sample' is not defined

## === cell 6
is_weird = df_train_sample["fare_amount"] < 0
is_weird |= ~df_train_sample["pickup_latitude"].between(40, 42)
is_weird |= ~df_train_sample["pickup_longitude"].between(-75, -72)
is_weird |= ~df_train_sample["dropoff_latitude"].between(40, 42)
is_weird |= ~df_train_sample["dropoff_longitude"].between(-75, -72)
is_weird |= df_train_sample["passenger_count"] == 0
print(is_weird.sum())

df_train_sample = df_train_sample[~is_weird].copy()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1567041333.py in <cell line: 0>()
----> 1 is_weird = df_train_sample["fare_amount"] < 0
      2 is_weird |= ~df_train_sample["pickup_latitude"].between(40, 42)
      3 is_weird |= ~df_train_sample["pickup_longitude"].between(-75, -72)
      4 is_weird |= ~df_train_sample["dropoff_latitude"].between(40, 42)
      5 is_weird |= ~df_train_sample["dropoff_longitude"].between(-75, -72)

NameError: name 'df_train_sample' is not defined

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
    dt = df["pickup_datetime"]

    iso_week = dt.dt.isocalendar().week.to_numpy(dtype=np.int16, copy=False)
    hour = dt.dt.hour.to_numpy(dtype=np.int16, copy=False)  # 0-23
    weekday = (dt.dt.weekday.to_numpy(dtype=np.int16, copy=False) + 24).astype(
        np.int16, copy=False
    )  # 24-30
    dayofyear = (dt.dt.dayofyear.to_numpy(dtype=np.int16, copy=False) + 31).astype(
        np.int16, copy=False
    )  # 32..397
    iso_week_cat = (iso_week + 397).astype(np.int16, copy=False)  # 398..450

    year = dt.dt.year.to_numpy(dtype=np.int16, copy=False)
    year_cat = (year - 2009 + 451).astype(np.int16, copy=False)  # start after 450
    passenger = (
        df["passenger_count"].to_numpy(dtype=np.int16, copy=False) + 470
    ).astype(np.int16, copy=False)

    X_cat = np.stack(
        [hour, weekday, dayofyear, iso_week_cat, year_cat, passenger], axis=1
    )

    X_deg = df[
        ["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]
    ].to_numpy(dtype="float32", copy=False)
    X_deg = X_deg / 180.0

    has_y = "fare_amount" in df.columns
    if has_y:
        y = df["fare_amount"].to_numpy(dtype="float32", copy=False)

    if shuffle:
        rnd_ind = np.random.permutation(len(df))
        X_cat = X_cat[rnd_ind]
        X_deg = X_deg[rnd_ind]
        if has_y:
            y = y[rnd_ind]

    if has_y:
        return (X_cat, X_deg), y
    return (X_cat, X_deg)




## === cell 12
np.random.seed(1989)
(X_cat, X_deg), y = prep_data(df_train_sample, shuffle=True)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3033377549.py in <cell line: 0>()
      1 np.random.seed(1989)
----> 2 (X_cat, X_deg), y = prep_data(df_train_sample, shuffle=True)
      3 

NameError: name 'df_train_sample' is not defined

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

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2324020571.py in <cell line: 0>()
      1 (X_cat_test_tmp, X_deg_test_tmp) = prep_data(df_test)
----> 2 max_cat_id = int(max(X_cat.max(), X_cat_test_tmp.max())) + 1
      3 
      4 lyr_latlng_input = lyr.Input(X_deg.shape[1:])
      5 lyr_haversine_bearing = lyr.Lambda(K_haversine_bearing, name="haversine")(

NameError: name 'X_cat' is not defined

## === cell 16
model.compile(loss="mse", optimizer="nadam")



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/957350833.py in <cell line: 0>()
----> 1 model.compile(loss="mse", optimizer="nadam")
      2 

NameError: name 'model' is not defined

## === cell 17
n_total = len(X_deg)
n_val = int(n_total * 0.3)
n_train = n_total - n_val

X_deg_train, X_cat_train, y_train = X_deg[:n_train], X_cat[:n_train], y[:n_train]
X_deg_val, X_cat_val, y_val = X_deg[n_train:], X_cat[n_train:], y[n_train:]

train_ds = tf.data.Dataset.from_tensor_slices(((X_deg_train, X_cat_train), y_train))
train_ds = train_ds.batch(512, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices(((X_deg_val, X_cat_val), y_val))
val_ds = val_ds.batch(512, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

model.fit(train_ds, epochs=50, validation_data=val_ds)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/58759399.py in <cell line: 0>()
      3 # - same validation_split=0.3 by slicing (equivalent to Keras' internal split on already-shuffled data)
      4 # - same batch_size and epochs
----> 5 n_total = len(X_deg)
      6 n_val = int(n_total * 0.3)
      7 n_train = n_total - n_val

NameError: name 'X_deg' is not defined

## === cell 18
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



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2061011794.py in <cell line: 0>()
      1 y_pred = np.clip(
----> 2     model.predict([X_deg_val, X_cat_val], batch_size=1024, verbose=1)[:, 0],
      3     0,
      4     None,
      5 )

NameError: name 'model' is not defined

## === cell 19
X_cat_test, X_deg_test = X_cat_test_tmp, X_deg_test_tmp



## === cell 20
y_test = np.clip(
    model.predict([X_deg_test, X_cat_test], batch_size=512, verbose=1), 0, None
)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4098591943.py in <cell line: 0>()
      1 y_test = np.clip(
----> 2     model.predict([X_deg_test, X_cat_test], batch_size=512, verbose=1), 0, None
      3 )
      4 

NameError: name 'model' is not defined

## === cell 21
if RUN_EDA:
    sns.histplot(np.log1p(y), bins=100, kde=False)
    sns.histplot(np.log1p(y_test[:, 0]), bins=100, kde=False)



## === cell 22
df_sub = pd.DataFrame({"key": df_test["key"].values, "fare_amount": y_test[:, 0]})
if RUN_EDA:
    df_sub.head()



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2012312942.py in <cell line: 0>()
----> 1 df_sub = pd.DataFrame({"key": df_test["key"].values, "fare_amount": y_test[:, 0]})
      2 if RUN_EDA:
      3     df_sub.head()
      4 

NameError: name 'y_test' is not defined

## === cell 23
df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)
print(df_sub.head())

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3924369527.py in <cell line: 0>()
----> 1 df_sub.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", df_sub.shape)
      3 print(df_sub.head())

NameError: name 'df_sub' is not defined
