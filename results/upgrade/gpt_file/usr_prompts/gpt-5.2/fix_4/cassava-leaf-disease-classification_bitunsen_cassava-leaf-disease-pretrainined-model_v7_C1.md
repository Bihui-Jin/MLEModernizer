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

3.9

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

0.8724690238742823

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import json
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "") == "python":
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")




## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

train_csv_path = os.path.join(BASE_DIR, "train.csv")
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")

assert os.path.exists(train_csv_path), f"Missing: {train_csv_path}"
assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"
assert os.path.isdir(TRAIN_DIR), f"Missing dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing dir: {TEST_DIR}"




## === cell 2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

print("TF version:", tf.__version__)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())

print(json.dumps(map_classes, indent=2))




## === cell 4
label_list = [int(key) for key in map_classes.keys()]
print("Labels:", label_list)




## === cell 5
train_df_tmp = pd.read_csv(train_csv_path, usecols=["image_id", "label"])
print(f"Number of train images: {train_df_tmp.shape[0]}")
del train_df_tmp




## === cell 6
IMG_HEIGHT = 400
IMG_WIDTH = 400
batch_size = 32

PRE_TRAINED_MODEL = "../input/xceptionv6/Cassava_Best_Xception_Model_V05.hdf5"
print(
    "Pretrained model exists?",
    os.path.exists(PRE_TRAINED_MODEL),
    "| Path:",
    PRE_TRAINED_MODEL,
)




## === cell 7
AUTOTUNE = tf.data.AUTOTUNE


def _read_decode_resize(image_path):
    img_bytes = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)  # uint8 [H,W,3]
    img = tf.image.resize(
        img, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.LANCZOS3
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _aug_hflip(img, p=0.5):
    do = tf.random.uniform(()) < p
    return tf.cond(do, lambda: tf.image.flip_left_right(img), lambda: img)


def _aug_rand_brightness(img, limit=0.2, p=0.5):
    do = tf.random.uniform(()) < p

    def _apply():
        delta = tf.random.uniform((), -limit, limit, dtype=tf.float32)
        out = img + delta
        return tf.clip_by_value(out, 0.0, 1.0)

    return tf.cond(do, _apply, lambda: img)


def _aug_rand_contrast(img, limit=0.2, p=0.5):
    do = tf.random.uniform(()) < p

    def _apply():
        factor = 1.0 + tf.random.uniform((), -limit, limit, dtype=tf.float32)
        mean = tf.reduce_mean(img, axis=[0, 1], keepdims=True)
        out = (img - mean) * factor + mean
        return tf.clip_by_value(out, 0.0, 1.0)

    return tf.cond(do, _apply, lambda: img)


def _augment_train_like_original(img):
    do_full = tf.random.uniform(()) > 0.5

    def _full():
        x = img
        x = _aug_hflip(x, p=0.5)
        x = _aug_rand_contrast(x, limit=0.2, p=0.5)
        x = _aug_rand_brightness(x, limit=0.2, p=0.5)
        return x

    return tf.cond(do_full, _full, lambda: img)


def _make_dataset(image_ids, labels, data_dir, mode):
    image_paths = tf.strings.join([data_dir, image_ids])
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(image_paths)
    else:
        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if mode == "TRAIN":
        ds = ds.shuffle(
            buffer_size=tf.minimum(tf.shape(image_ids)[0], 8192),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    def _map_train(path, y):
        img = _read_decode_resize(path)
        img = _augment_train_like_original(img)
        return img, y

    def _map_val(path, y):
        img = _read_decode_resize(path)
        return img, y

    def _map_test(path):
        img = _read_decode_resize(path)
        return img

    if mode == "TEST":
        ds = ds.map(_map_test, num_parallel_calls=AUTOTUNE, deterministic=True)
        ds = ds.cache()
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    if mode == "TRAIN":
        ds = ds.map(_map_train, num_parallel_calls=AUTOTUNE, deterministic=False)
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    ds = ds.map(_map_val, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 8
train_df = pd.read_csv(train_csv_path)
print(train_df.head())
print("Train shape:", train_df.shape)
num_classes = train_df["label"].nunique()
print("Num classes:", num_classes)

sample_sub = pd.read_csv(sample_sub_path)
test_df = sample_sub[["image_id"]].copy()
test_samples = test_df.shape[0]
print("Test samples:", test_samples)




## === cell 9
from sklearn.model_selection import train_test_split

train_ids, val_ids, train_y, val_y = train_test_split(
    train_df["image_id"].values,
    train_df["label"].values,
    test_size=0.1,
    random_state=SEED,
    stratify=train_df["label"].values,
)

train_ds = _make_dataset(
    image_ids=tf.constant(train_ids),
    labels=tf.constant(train_y, dtype=tf.int32),
    data_dir=tf.constant(TRAIN_DIR),
    mode="TRAIN",
)
val_ds = _make_dataset(
    image_ids=tf.constant(val_ids),
    labels=tf.constant(val_y, dtype=tf.int32),
    data_dir=tf.constant(TRAIN_DIR),
    mode="VALIDATE",
)
test_ds = _make_dataset(
    image_ids=tf.constant(test_df["image_id"].values),
    labels=None,
    data_dir=tf.constant(TEST_DIR),
    mode="TEST",
)

train_steps = int(np.ceil(len(train_ids) / batch_size))
val_steps = int(np.ceil(len(val_ids) / batch_size))
test_steps = int(np.ceil(len(test_df) / batch_size))
print("Batches - train/val/test:", train_steps, val_steps, test_steps)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1982779059.py in <cell line: 0>()
     13 # - caches validation/test (identical semantics, avoids repeated disk I/O)
     14 # - prefetches to overlap model compute with input pipeline
---> 15 train_ds = _make_dataset(
     16     image_ids=tf.constant(train_ids),
     17     labels=tf.constant(train_y, dtype=tf.int32),

/tmp/ipykernel_11/2321686963.py in _make_dataset(image_ids, labels, data_dir, mode)
     67     # NOTE: Deterministic ordering for validation/test; for train we still shuffle deterministically by seed.
     68     if mode == "TRAIN":
---> 69         ds = ds.shuffle(
     70             buffer_size=tf.minimum(tf.shape(image_ids)[0], 8192),
     71             seed=SEED,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in shuffle(self, buffer_size, seed, reshuffle_each_iteration, name)
   1508       A new `Dataset` with the transformation applied as described above.
   1509     """
-> 1510     return shuffle_op._shuffle(  # pylint: disable=protected-access
   1511         self, buffer_size, seed, reshuffle_each_iteration, name=name)
   1512 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/shuffle_op.py in _shuffle(input_dataset, buffer_size, seed, reshuffle_each_iteration, name)
     30     name=None,
     31 ):
---> 32   return _ShuffleDataset(
     33       input_dataset, buffer_size, seed, reshuffle_each_iteration, name=name)
     34 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/shuffle_op.py in __init__(self, input_dataset, buffer_size, seed, reshuffle_each_iteration, name)
     47     """See `Dataset.shuffle()` for details."""
     48     self._input_dataset = input_dataset
---> 49     self._buffer_size = ops.convert_to_tensor(
     50         buffer_size, dtype=dtypes.int64, name="buffer_size")
     51     self._seed, self._seed2 = random_seed.get_seed(seed)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/profiler/trace.py in wrapped(*args, **kwargs)
    181         with Trace(trace_name, **trace_kwargs):
    182           return func(*args, **kwargs)
--> 183       return func(*args, **kwargs)
    184 
    185     return wrapped

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in convert_to_tensor(value, dtype, name, as_ref, preferred_dtype, dtype_hint, ctx, accepted_result_types)
    730   # TODO(b/142518781): Fix all call-sites and remove redundant arg
    731   preferred_dtype = preferred_dtype or dtype_hint
--> 732   return tensor_conversion_registry.convert(
    733       value, dtype, name, as_ref, preferred_dtype, accepted_result_types
    734   )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor_conversion_registry.py in convert(value, dtype, name, as_ref, preferred_dtype, accepted_result_types)
    207   overload = getattr(value, "__tf_tensor__", None)
    208   if overload is not None:
--> 209     return overload(dtype, name)  #  pylint: disable=not-callable
    210 
    211   for base_type, conversion_func in get(type(value)):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in __tf_tensor__(self, dtype, name)
    625                 name=name))
    626       return graph.capture(self, name=name)
--> 627     return super().__tf_tensor__(dtype, name)
    628 
    629   def _capture_as_const(self, name) -> Optional[tensor_lib.Tensor]:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor.py in __tf_tensor__(self, dtype, name)
    759       ) -> "Tensor":
    760     if dtype is not None and not dtype.is_compatible_with(self.dtype):
--> 761       raise ValueError(
    762           _add_error_prefix(
    763               f"Tensor conversion requested dtype {dtype.name} "

ValueError: buffer_size: Tensor conversion requested dtype int64 for Tensor with dtype int32: <tf.Tensor: shape=(), dtype=int32, numpy=8192>

## === cell 10
def build_model(img_height=IMG_HEIGHT, img_width=IMG_WIDTH, n_classes=5):
    inputs = keras.Input(shape=(img_height, img_width, 3))
    x = keras.applications.xception.preprocess_input(inputs * 255.0)
    base = keras.applications.Xception(
        include_top=False, weights="imagenet", input_tensor=x
    )
    base.trainable = False  # keep fast and stable

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model


model = build_model(n_classes=5)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()




## === cell 11
EPOCHS = 2

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/550030011.py in <cell line: 0>()
      2 
      3 # NOTE: Training loop is unchanged (same epochs/optimizer/loss); only the input pipeline is faster.
----> 4 history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
      5 
      6 

NameError: name 'train_ds' is not defined

## === cell 12
pred_probs = model.predict(test_ds, verbose=1)
pred_labels = np.argmax(pred_probs, axis=1).astype(int)

print("Pred shape:", pred_probs.shape, "Labels shape:", pred_labels.shape)
assert len(pred_labels) == len(test_df), "Prediction length mismatch with test_df"
assert np.all(
    (pred_labels >= 0) & (pred_labels <= 4)
), "Predicted labels out of range 0..4"




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1175171880.py in <cell line: 0>()
      1 # NOTE: Prediction semantics unchanged; faster input pipeline (cached + prefetched).
----> 2 pred_probs = model.predict(test_ds, verbose=1)
      3 pred_labels = np.argmax(pred_probs, axis=1).astype(int)
      4 
      5 print("Pred shape:", pred_probs.shape, "Labels shape:", pred_labels.shape)

NameError: name 'test_ds' is not defined

## === cell 13
submission = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": pred_labels}
)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission.head(10))

check = pd.read_csv(submission_path)
print(check.shape)
print(check.head(3))
print("Columns:", list(check.columns))
assert list(check.columns) == ["image_id", "label"]
assert check["image_id"].nunique() == len(check), "Duplicate image_id in submission"
assert check["label"].between(0, 4).all(), "Labels out of expected range 0..4"
assert os.path.exists(submission_path) and submission_path.endswith(".csv")

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1049009317.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"image_id": test_df["image_id"].values, "label": pred_labels}
      3 )
      4 
      5 submission_path = "submission.csv"

NameError: name 'pred_labels' is not defined
