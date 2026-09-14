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
import numpy as np
import pandas as pd

import tensorflow as tf

print("/kaggle:", os.listdir("/kaggle"))
print("/kaggle/input:", os.listdir("/kaggle/input")[:10])
print("tf version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
TRAIN_PATH = "/kaggle/input/train.csv"
TEST_PATH = "/kaggle/input/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/sample_submission.csv"

df = pd.read_csv(TRAIN_PATH, nrows=100000, parse_dates=["pickup_datetime"])
test = pd.read_csv(TEST_PATH, parse_dates=["pickup_datetime"])
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

print(
    "train shape:",
    df.shape,
    "test shape:",
    test.shape,
    "sample_submission shape:",
    sample_sub.shape,
)
print("train cols:", df.columns.tolist())
print("test cols:", test.columns.tolist())



## === cell 2
_ = df.pickup_datetime.dt.day_name()
print("Datetime parsing OK. Example weekday:", df.pickup_datetime.dt.day_name().iloc[0])



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
print(df.dtypes)



## === cell 6
print("Null pickup_datetime:", int(df.pickup_datetime.isnull().sum()))



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

print("Filtered train shape:", dfc.shape)



## === cell 8
np.random.seed(seed=1)
msk = np.random.rand(len(dfc)) < 0.8
traindf = dfc[msk].drop(["key", "pickup_datetime"], axis=1).copy()
evaldf = dfc[~msk].drop(["key", "pickup_datetime"], axis=1).copy()

print("traindf:", traindf.shape, "evaldf:", evaldf.shape)



## === cell 9
testdf = add_feats(test)
testdf = testdf.drop(["key", "pickup_datetime"], axis=1).copy()
print("testdf:", testdf.shape)



## === cell 10
print("weekday head:", traindf.weekday.head().tolist())




## === cell 11
def build_wide_and_deep_model(nbuckets=10):
    lat_cuts = np.linspace(38.0, 42.0, nbuckets + 1)[1:-1].tolist()
    lon_cuts = np.linspace(-75.0, -72.0, nbuckets + 1)[1:-1].tolist()

    inputs = {
        "pickup_longitude": tf.keras.Input(
            shape=(1,), name="pickup_longitude", dtype=tf.float32
        ),
        "pickup_latitude": tf.keras.Input(
            shape=(1,), name="pickup_latitude", dtype=tf.float32
        ),
        "dropoff_longitude": tf.keras.Input(
            shape=(1,), name="dropoff_longitude", dtype=tf.float32
        ),
        "dropoff_latitude": tf.keras.Input(
            shape=(1,), name="dropoff_latitude", dtype=tf.float32
        ),
        "passenger_count": tf.keras.Input(
            shape=(1,), name="passenger_count", dtype=tf.float32
        ),
        "distance": tf.keras.Input(shape=(1,), name="distance", dtype=tf.float32),
        "weekday": tf.keras.Input(shape=(1,), name="weekday", dtype=tf.int32),
        "hour": tf.keras.Input(shape=(1,), name="hour", dtype=tf.int32),
    }

    def bucketize(x, boundaries):
        return tf.keras.layers.Discretization(bin_boundaries=boundaries)(x)

    b_plat = bucketize(inputs["pickup_latitude"], lat_cuts)
    b_plon = bucketize(inputs["pickup_longitude"], lon_cuts)
    b_dlat = bucketize(inputs["dropoff_latitude"], lat_cuts)
    b_dlon = bucketize(inputs["dropoff_longitude"], lon_cuts)

    ploc_tok = tf.strings.join(
        [tf.strings.as_string(b_plat), tf.strings.as_string(b_plon)], separator="_"
    )
    dloc_tok = tf.strings.join(
        [tf.strings.as_string(b_dlat), tf.strings.as_string(b_dlon)], separator="_"
    )
    pd_pair_tok = tf.strings.join([ploc_tok, dloc_tok], separator="__")
    day_hr_tok = tf.strings.join(
        [tf.strings.as_string(inputs["weekday"]), tf.strings.as_string(inputs["hour"])],
        separator="-",
    )

    wide_features = []

    for name, tok, bins in [
        ("ploc", ploc_tok, nbuckets * nbuckets),
        ("dloc", dloc_tok, nbuckets * nbuckets),
        ("pd_pair", pd_pair_tok, nbuckets**4),
        ("day_hr", day_hr_tok, 24 * 7),
    ]:
        hashed = tf.keras.layers.Hashing(num_bins=bins, name=f"{name}_hash")(tok)
        onehot = tf.keras.layers.CategoryEncoding(
            num_tokens=bins, output_mode="one_hot", name=f"{name}_onehot"
        )(hashed)
        wide_features.append(onehot)

    wide_features.append(tf.cast(inputs["weekday"], tf.float32))
    wide_features.append(tf.cast(inputs["hour"], tf.float32))
    wide_features.append(inputs["passenger_count"])

    wide = tf.keras.layers.Concatenate(name="wide_concat")(wide_features)

    deep_features = []

    for name, tok, bins, emb_dim in [
        ("pd_pair", pd_pair_tok, nbuckets**4, 10),
        ("day_hr", day_hr_tok, 24 * 7, 10),
    ]:
        hashed = tf.keras.layers.Hashing(num_bins=bins, name=f"{name}_hash_deep")(tok)
        emb = tf.keras.layers.Embedding(
            input_dim=bins, output_dim=emb_dim, name=f"{name}_emb"
        )(hashed)
        emb = tf.keras.layers.Reshape((emb_dim,), name=f"{name}_emb_flat")(emb)
        deep_features.append(emb)

    deep_numeric = tf.keras.layers.Concatenate(name="deep_numeric")(
        [
            inputs["pickup_latitude"],
            inputs["pickup_longitude"],
            inputs["dropoff_latitude"],
            inputs["dropoff_longitude"],
            inputs["distance"],
        ]
    )
    deep = tf.keras.layers.Concatenate(name="deep_concat")(
        deep_features + [deep_numeric]
    )

    x = tf.keras.layers.Dense(128, activation="relu")(deep)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(4, activation="relu")(x)

    all_features = tf.keras.layers.Concatenate(name="wide_deep_concat")([wide, x])
    out = tf.keras.layers.Dense(1, name="fare_amount")(all_features)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model




## === cell 12
OUTDIR = "./taxi_trained"
os.makedirs(OUTDIR, exist_ok=True)




## === cell 13
def df_to_model_inputs(dfin: pd.DataFrame):
    x = {
        "pickup_longitude": dfin["pickup_longitude"].astype("float32").to_numpy(),
        "pickup_latitude": dfin["pickup_latitude"].astype("float32").to_numpy(),
        "dropoff_longitude": dfin["dropoff_longitude"].astype("float32").to_numpy(),
        "dropoff_latitude": dfin["dropoff_latitude"].astype("float32").to_numpy(),
        "passenger_count": dfin["passenger_count"].astype("float32").to_numpy(),
        "distance": dfin["distance"].astype("float32").to_numpy(),
        "weekday": dfin["weekday"].astype("int32").to_numpy(),
        "hour": dfin["hour"].astype("int32").to_numpy(),
    }
    return x


BATCH_SIZE = 512
EPOCHS = 5

x_train = df_to_model_inputs(traindf.drop(["fare_amount"], axis=1))
y_train = traindf["fare_amount"].astype("float32").to_numpy()

x_eval = df_to_model_inputs(evaldf.drop(["fare_amount"], axis=1))
y_eval = evaldf["fare_amount"].astype("float32").to_numpy()

x_test = df_to_model_inputs(testdf)

print("Prepared model inputs. Train y mean:", float(np.mean(y_train)))



## === cell 14
tf.random.set_seed(1)
np.random.seed(1)

model = build_wide_and_deep_model(nbuckets=10)
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_eval, y_eval),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=2,
)

eval_metrics = model.evaluate(x_eval, y_eval, batch_size=4096, verbose=0)
print("Eval metrics:", dict(zip(model.metrics_names, eval_metrics)))



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3224164562.py in <cell line: 0>()
      2 np.random.seed(1)
      3 
----> 4 model = build_wide_and_deep_model(nbuckets=10)
      5 history = model.fit(
      6     x_train,

/tmp/ipykernel_11/3563710790.py in build_wide_and_deep_model(nbuckets)
     35 
     36     ploc_tok = tf.strings.join(
---> 37         [tf.strings.as_string(b_plat), tf.strings.as_string(b_plon)], separator="_"
     38     )
     39     dloc_tok = tf.strings.join(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_string_ops.py in as_string(input, precision, scientific, shortest, width, fill, name)
     84       if _result is not NotImplemented:
     85         return _result
---> 86       return as_string_eager_fallback(
     87           input, precision=precision, scientific=scientific,
     88           shortest=shortest, width=width, fill=fill, name=name, ctx=_ctx)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_string_ops.py in as_string_eager_fallback(input, precision, scientific, shortest, width, fill, name, ctx)
    165     fill = ""
    166   fill = _execute.make_str(fill, "fill")
--> 167   _attr_T, (input,) = _execute.args_to_matching_eager([input], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.int64, _dtypes.bfloat16, _dtypes.uint16, _dtypes.half, _dtypes.uint32, _dtypes.uint64, _dtypes.complex64, _dtypes.complex128, _dtypes.bool, _dtypes.variant, _dtypes.string, ])
    168   _inputs_flat = [input]
    169   _attrs = ("T", _attr_T, "precision", precision, "scientific", scientific,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in args_to_matching_eager(***failed resolving arguments***)
    249       # not list allowed dtypes, in which case we should skip this.
    250       if dtype is None and allowed_dtypes:
--> 251         tensor = tensor_conversion_registry.convert(t)
    252         # If we did not match an allowed dtype, try again with the default
    253         # dtype. This could be because we have an empty tensor and thus we

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor_conversion_registry.py in convert(value, dtype, name, as_ref, preferred_dtype, accepted_result_types)
    207   overload = getattr(value, "__tf_tensor__", None)
    208   if overload is not None:
--> 209     return overload(dtype, name)  #  pylint: disable=not-callable
    210 
    211   for base_type, conversion_func in get(type(value)):

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/keras_tensor.py in __tf_tensor__(self, dtype, name)
    136 
    137     def __tf_tensor__(self, dtype=None, name=None):
--> 138         raise ValueError(
    139             "A KerasTensor cannot be used as input to a TensorFlow function. "
    140             "A KerasTensor is a symbolic placeholder for a shape and dtype, "

ValueError: A KerasTensor cannot be used as input to a TensorFlow function. A KerasTensor is a symbolic placeholder for a shape and dtype, used when constructing Keras Functional models or Keras Functions. You can only use it as input to a Keras layer or a Keras operation (from the namespaces `keras.layers` and `keras.operations`). You are likely doing something like:

```
x = Input(...)
...
tf_fn(x)  # Invalid.
```

What you should do instead is wrap `tf_fn` in a layer:

```
class MyLayer(Layer):
    def call(self, x):
        return tf_fn(x)

x = MyLayer()(x)
```


## === cell 15
pred = model.predict(x_test, batch_size=4096, verbose=0).reshape(-1).astype(np.float64)
assert len(pred) == len(test), f"Pred length {len(pred)} != test length {len(test)}"
print("Pred stats:", float(np.min(pred)), float(np.mean(pred)), float(np.max(pred)))



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/990674835.py in <cell line: 0>()
----> 1 pred = model.predict(x_test, batch_size=4096, verbose=0).reshape(-1).astype(np.float64)
      2 assert len(pred) == len(test), f"Pred length {len(pred)} != test length {len(test)}"
      3 print("Pred stats:", float(np.min(pred)), float(np.mean(pred)), float(np.max(pred)))
      4 

NameError: name 'model' is not defined

## === cell 16
pred = np.clip(pred, 0.0, None)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2515027978.py in <cell line: 0>()
----> 1 pred = np.clip(pred, 0.0, None)
      2 

NameError: name 'pred' is not defined

## === cell 17
output = pd.DataFrame({"key": test["key"].values, "fare_amount": pred})

assert list(output.columns) == ["key", "fare_amount"]
assert output.shape[0] == sample_sub.shape[0] == test.shape[0]
print(output.head())



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2876687259.py in <cell line: 0>()
----> 1 output = pd.DataFrame({"key": test["key"].values, "fare_amount": pred})
      2 
      3 assert list(output.columns) == ["key", "fare_amount"]
      4 assert output.shape[0] == sample_sub.shape[0] == test.shape[0]
      5 print(output.head())

NameError: name 'pred' is not defined

## === cell 18
SUB_PATH = "submission_file.csv"
output.to_csv(SUB_PATH, index=False)
print(f"Wrote {SUB_PATH} with shape:", output.shape)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2771474330.py in <cell line: 0>()
      1 SUB_PATH = "submission_file.csv"
----> 2 output.to_csv(SUB_PATH, index=False)
      3 print(f"Wrote {SUB_PATH} with shape:", output.shape)
      4 

NameError: name 'output' is not defined

## === cell 19
chk = pd.read_csv(SUB_PATH, nrows=5)
print("Submission preview:")
print(chk)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4269619853.py in <cell line: 0>()
----> 1 chk = pd.read_csv(SUB_PATH, nrows=5)
      2 print("Submission preview:")
      3 print(chk)
      4 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'submission_file.csv'

## === cell 20
print("Done. Submission ready at:", os.path.abspath(SUB_PATH))
