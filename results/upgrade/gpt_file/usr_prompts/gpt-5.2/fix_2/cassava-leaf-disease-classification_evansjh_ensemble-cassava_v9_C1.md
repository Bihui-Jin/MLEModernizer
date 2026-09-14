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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8890903596252644

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
import random
import numpy as np
import pandas as pd

import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
model_path_1 = (
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/BestModel_3454_8937.h5"
)
model_path_2 = (
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/best_model_0.37458707.h5"
)
model_path_3 = (
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/googlenet_inceptionv3.h5"
)
model_path_4 = (
    "/kaggle/input/bestmodel_550_2/tensorflow2/default/1/BestModel_3577_8940.h5"
)
model_path_5 = (
    "/kaggle/input/bestmodel_8878/tensorflow2/default/1/BestModel_8878_0358.h5"
)
model_path_6 = "/kaggle/input/bestmodel_8875/tensorflow2/default/1/BestModel_8875.h5"

test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"

assert os.path.exists(sample_path), f"Missing sample submission at {sample_path}"
assert os.path.exists(test_image_dir), f"Missing test images dir at {test_image_dir}"
assert os.path.exists(train_csv_path), f"Missing train.csv at {train_csv_path}"

sample_csv = pd.read_csv(sample_path)
train_df = pd.read_csv(train_csv_path)

print("sample_submission:", sample_csv.shape, sample_csv.columns.tolist())
print("train_df:", train_df.shape, train_df.columns.tolist())
print(
    "test images:",
    len([f for f in os.listdir(test_image_dir) if f.lower().endswith(".jpg")]),
)




## === cell 2
def try_load_model(path):
    if not os.path.exists(path):
        return None
    try:
        m = tf.keras.models.load_model(path, compile=False)
        return m
    except Exception as e:
        print(f"Warning: failed to load model at {path}: {type(e).__name__}: {e}")
        return None


def infer_input_size(model):
    shp = model.inputs[0].shape  # (None, H, W, C)
    h, w = int(shp[1]), int(shp[2])
    return (h, w)


models_info = [
    (model_path_1, (550, 550)),
    (model_path_2, (512, 512)),
    (model_path_3, (448, 448)),
    (model_path_5, (512, 512)),
    (model_path_6, (512, 512)),
]

models = []
for path, declared_size in models_info:
    m = try_load_model(path)
    if m is not None:
        size = (
            infer_input_size(m)
            if (m.inputs and m.inputs[0].shape.rank == 4)
            else declared_size
        )
        models.append((m, size))
        print("Loaded:", os.path.basename(path), "input:", size)

print("Loaded model count:", len(models))




## === cell 3
def build_fallback_model(img_size=(512, 512), num_classes=5):
    import tensorflow_hub as hub

    inputs = tf.keras.Input(shape=(img_size[0], img_size[1], 3))
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)

    feature_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"
    x = hub.KerasLayer(feature_url, trainable=False)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_train_val_datasets(
    df, img_dir, img_size=(512, 512), batch_size=32, val_split=0.1
):
    df = df.copy()
    df["filepath"] = df["image_id"].apply(lambda x: os.path.join(img_dir, x))
    df = df[df["filepath"].apply(os.path.exists)].reset_index(drop=True)

    rng = np.random.RandomState(SEED)
    val_idx = []
    for label, group in df.groupby("label"):
        idx = group.index.values
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_split))
        val_idx.extend(idx[:n_val])
    val_idx = set(val_idx)

    val_df = df[df.index.isin(val_idx)].reset_index(drop=True)
    trn_df = df[~df.index.isin(val_idx)].reset_index(drop=True)

    def _load(path, label=None):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, img_size, method="bilinear")
        img = tf.cast(img, tf.float32)
        if label is None:
            return img
        return img, tf.cast(label, tf.int32)

    trn_ds = tf.data.Dataset.from_tensor_slices(
        (trn_df["filepath"].values, trn_df["label"].values)
    )
    val_ds = tf.data.Dataset.from_tensor_slices(
        (val_df["filepath"].values, val_df["label"].values)
    )

    trn_ds = (
        trn_ds.shuffle(min(8192, len(trn_df)), seed=SEED, reshuffle_each_iteration=True)
        .map(_load, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        val_ds.map(_load, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return trn_ds, val_ds, trn_df, val_df


fallback_model = None
fallback_img_size = (512, 512)

if len(models) == 0:
    batch_size = 32
    epochs = 3  # keep runtime within 600s; avoids early stopping / sampling changes.
    fallback_img_size = (512, 512)

    trn_ds, val_ds, trn_df, val_df = make_train_val_datasets(
        train_df,
        train_image_dir,
        img_size=fallback_img_size,
        batch_size=batch_size,
        val_split=0.1,
    )
    print("Training fallback model. Train/Val:", len(trn_df), len(val_df))

    fallback_model = build_fallback_model(img_size=fallback_img_size, num_classes=5)
    history = fallback_model.fit(
        trn_ds, validation_data=val_ds, epochs=epochs, verbose=1
    )

    models = [(fallback_model, fallback_img_size)]
    print("Fallback model trained and added for inference.")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4163208569.py in <cell line: 0>()
     91     print("Training fallback model. Train/Val:", len(trn_df), len(val_df))
     92 
---> 93     fallback_model = build_fallback_model(img_size=fallback_img_size, num_classes=5)
     94     history = fallback_model.fit(
     95         trn_ds, validation_data=val_ds, epochs=epochs, verbose=1

/tmp/ipykernel_11/4163208569.py in build_fallback_model(img_size, num_classes)
     11     # Using 224/256/300/512 variants depends on availability; 512 generally works with B0 feature-vector.
     12     feature_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"
---> 13     x = hub.KerasLayer(feature_url, trainable=False)(x)
     14     x = tf.keras.layers.Dropout(0.2)(x)
     15     outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow_hub/keras_layer.py in call(self, inputs, training)
    248         # Behave like BatchNormalization. (Dropout is different, b/181839368.)
    249         training = False
--> 250       result = smart_cond.smart_cond(training,
    251                                      lambda: f(training=True),
    252                                      lambda: f(training=False))

/usr/local/lib/python3.11/dist-packages/tensorflow_hub/keras_layer.py in <lambda>()
    250       result = smart_cond.smart_cond(training,
    251                                      lambda: f(training=True),
--> 252                                      lambda: f(training=False))
    253 
    254     # Unwrap dicts returned by signatures.

/usr/local/lib/python3.11/dist-packages/tensorflow/core/function/polymorphism/function_type.py in canonicalize_to_monomorphic(args, kwargs, default_values, capture_types, polymorphic_type)
    581     else:
    582       parameters.append(
--> 583           _make_validated_mono_param(name, arg, poly_parameter.kind,
    584                                      type_context,
    585                                      poly_parameter.type_constraint))

/usr/local/lib/python3.11/dist-packages/tensorflow/core/function/polymorphism/function_type.py in _make_validated_mono_param(name, value, kind, type_context, poly_type)
    520 ) -> Parameter:
    521   """Generates and validates a parameter for Monomorphic FunctionType."""
--> 522   mono_type = trace_type.from_value(value, type_context)
    523 
    524   if poly_type and not mono_type.is_subtype_of(poly_type):

/usr/local/lib/python3.11/dist-packages/tensorflow/core/function/trace_type/trace_type_builder.py in from_value(value, context)
    183 
    184   if util.is_np_ndarray(value):
--> 185     ndarray = value.__array__()
    186     return default_types.TENSOR(ndarray.shape, ndarray.dtype)
    187 

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/keras_tensor.py in __array__(self)
    106 
    107     def __array__(self):
--> 108         raise ValueError(
    109             "A KerasTensor is symbolic: it's a placeholder for a shape "
    110             "an a dtype. It doesn't have any actual numerical value. "

ValueError: Exception encountered when calling layer 'keras_layer' (type KerasLayer).

A KerasTensor is symbolic: it's a placeholder for a shape an a dtype. It doesn't have any actual numerical value. You cannot convert it to a NumPy array.

Call arguments received by layer 'keras_layer' (type KerasLayer):
  • inputs=<KerasTensor shape=(None, 512, 512, 3), dtype=float32, sparse=False, name=keras_tensor_1>
  • training=None

## === cell 4
from collections import Counter

test_ids = sample_csv["image_id"].tolist()


def predict_one_model(model, input_size, image_paths, batch_size=32):
    def _load(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, input_size, method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0
        return img

    ds = (
        tf.data.Dataset.from_tensor_slices(image_paths)
        .map(_load, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    preds = model.predict(ds, verbose=0)
    return preds


test_paths = [os.path.join(test_image_dir, image_id) for image_id in test_ids]
missing = [p for p in test_paths if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(
        f"Some test images listed in sample_submission are missing, e.g. {missing[:3]}"
    )

all_model_probs = []
for model, input_size in models:
    probs = predict_one_model(model, input_size, test_paths, batch_size=32)
    if probs.ndim != 2 or probs.shape[1] != 5:
        raise ValueError(
            f"Model output shape unexpected: {probs.shape}, expected (N,5)"
        )
    all_model_probs.append(probs)

avg_probs = np.mean(all_model_probs, axis=0)
pred_labels = np.argmax(avg_probs, axis=1).astype(int)

submission_df = pd.DataFrame({"image_id": test_ids, "label": pred_labels})
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Wrote:", submission_path, "shape:", submission_df.shape)
submission_df.head()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AxisError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1755502896.py in <cell line: 0>()
     43 
     44 avg_probs = np.mean(all_model_probs, axis=0)
---> 45 pred_labels = np.argmax(avg_probs, axis=1).astype(int)
     46 
     47 submission_df = pd.DataFrame({"image_id": test_ids, "label": pred_labels})

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in argmax(a, axis, out, keepdims)
   1227     """
   1228     kwds = {'keepdims': keepdims} if keepdims is not np._NoValue else {}
-> 1229     return _wrapfunc(a, 'argmax', axis=axis, out=out, **kwds)
   1230 
   1231 

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in _wrapfunc(obj, method, *args, **kwds)
     57 
     58     try:
---> 59         return bound(*args, **kwds)
     60     except TypeError:
     61         # A TypeError occurs if the object does have such a method in its

AxisError: axis 1 is out of bounds for array of dimension 1

## === cell 5
sub = pd.read_csv("/kaggle/working/submission.csv")
assert sub.shape[0] == sample_csv.shape[0], "Row count mismatch vs sample_submission"
assert sub.columns.tolist() == ["image_id", "label"], "Column mismatch"
assert (
    sub["image_id"].tolist() == sample_csv["image_id"].tolist()
), "image_id ordering mismatch"
assert sub["label"].between(0, 4).all(), "Labels out of range [0,4]"
print("Submission OK.")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3040537306.py in <cell line: 0>()
      1 # Final sanity checks for submission validity
----> 2 sub = pd.read_csv("/kaggle/working/submission.csv")
      3 assert sub.shape[0] == sample_csv.shape[0], "Row count mismatch vs sample_submission"
      4 assert sub.columns.tolist() == ["image_id", "label"], "Column mismatch"
      5 assert (

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/submission.csv'
