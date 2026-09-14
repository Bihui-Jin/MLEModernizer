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

0.8245693563009973

# 6. Current score

0.18348

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.18348) has done: 'The main bottleneck is that you build `val_ds` by re-reading and preprocessing the *entire* training TFRecord set (and even caching it), rather than only the validation fraction; that roughly doubles input pipeline work and can blow memory/time. The fix is to construct train/val datasets from the TFRecords using a deterministic per-example split (hash of `image_name`) so each epoch reads each example only once, while preserving the same model/training loop and semantics. I also remove the unused CSV-based dataset code and switch test inference to a single `model.predict` call (same computation, less Python overhead). Finally, I add `ignore_order` and tune `interleave`/`prefetch` options in a deterministic-safe way to reduce input overhead without changing results.'

# 9. Code solution

## === cell 0
TRAIN_IMG_LOC = "../input/cassava-leaf-disease-classification/train_images"
TEST_IMG_LOC = "../input/cassava-leaf-disease-classification/test_images"
TRAIN_CSV = "../input/cassava-leaf-disease-classification/train.csv"
SAMPLE_CSV = "../input/cassava-leaf-disease-classification/sample_submission.csv"

import os
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint

print("TensorFlow:", tf.__version__)
print("ALL required modules loaded successfully")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

NUM_CLASSES = 5
IMG_SIZE = 300
BATCH_SIZE = 32

AUTOTUNE = tf.data.AUTOTUNE

TRAIN_TFREC_GLOB = (
    "../input/cassava-leaf-disease-classification/train_tfrecords/*.tfrec"
)
TEST_TFREC_GLOB = "../input/cassava-leaf-disease-classification/test_tfrecords/*.tfrec"




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
train_df["filepath"] = TRAIN_IMG_LOC + "/" + train_df["image_id"].astype(str)

train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

val_frac = 0.1
val_size = int(len(train_df) * val_frac)
val_df = train_df.iloc[:val_size].copy()
trn_df = train_df.iloc[val_size:].copy()

print("Train size:", len(trn_df), "Val size:", len(val_df))
print(trn_df.head())




## === cell 2
from tensorflow.keras.applications.efficientnet import preprocess_input

_FEATURE_DESC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


def decode_and_preprocess(
    img_bytes_or_path, label=None, training=False, from_bytes=False
):
    if from_bytes:
        img_bytes = img_bytes_or_path
    else:
        img_bytes = tf.io.read_file(img_bytes_or_path)

    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)  # EfficientNet expects this preprocessing

    if training:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.1)

    if label is None:
        return img
    label = tf.cast(label, tf.int32)
    return img, label


def _parse_tfrecord(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESC)
    return ex["image"], ex["image_name"], tf.cast(ex["target"], tf.int32)


def _parse_test_tfrecord(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESC)
    return ex["image"], ex["image_name"]


def _is_val_example(image_name, val_mod=10, val_threshold=1):
    bucket = tf.strings.to_hash_bucket_fast(image_name, val_mod)
    return bucket < val_threshold


def make_ds_from_tfrecords_split(
    tfrecord_files, training, cache=False, val_mod=10, val_threshold=1
):
    options = tf.data.Options()
    options.experimental_deterministic = True

    files = tf.constant(list(tfrecord_files))
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    if training:
        files_ds = files_ds.shuffle(
            tf.shape(files)[0], seed=SEED, reshuffle_each_iteration=True
        )

    ds = files_ds.interleave(
        lambda fn: tf.data.TFRecordDataset(fn, num_parallel_reads=AUTOTUNE),
        cycle_length=min(8, int(len(tfrecord_files))),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    ).with_options(options)

    ds = ds.map(_parse_tfrecord, num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.filter(
            lambda img_bytes, name, y: tf.logical_not(
                _is_val_example(name, val_mod, val_threshold)
            )
        )
    else:
        ds = ds.filter(
            lambda img_bytes, name, y: _is_val_example(name, val_mod, val_threshold)
        )

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda img_bytes, name, y: decode_and_preprocess(
            img_bytes, y, training=training, from_bytes=True
        ),
        num_parallel_calls=AUTOTUNE,
    )

    if cache:
        ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


train_tfrecord_files = tf.io.gfile.glob(TRAIN_TFREC_GLOB)
if not train_tfrecord_files:
    raise FileNotFoundError(f"No TFRecords found at glob: {TRAIN_TFREC_GLOB}")

train_ds = make_ds_from_tfrecords_split(
    train_tfrecord_files, training=True, cache=False, val_mod=10, val_threshold=1
)
val_ds = make_ds_from_tfrecords_split(
    train_tfrecord_files, training=False, cache=True, val_mod=10, val_threshold=1
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2776882551.py in <cell line: 0>()
    111 # --- Speed fix: only read/process each example once per epoch by using a split, not "full train for train + full train for val".
    112 # Cache only the (small) validation subset.
--> 113 train_ds = make_ds_from_tfrecords_split(
    114     train_tfrecord_files, training=True, cache=False, val_mod=10, val_threshold=1
    115 )

/tmp/ipykernel_11/2776882551.py in make_ds_from_tfrecords_split(tfrecord_files, training, cache, val_mod, val_threshold)
     60     files_ds = tf.data.Dataset.from_tensor_slices(files)
     61     if training:
---> 62         files_ds = files_ds.shuffle(
     63             tf.shape(files)[0], seed=SEED, reshuffle_each_iteration=True
     64         )

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

ValueError: buffer_size: Tensor conversion requested dtype int64 for Tensor with dtype int32: <tf.Tensor: shape=(), dtype=int32, numpy=14>

## === cell 3
base = tf.keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.trainable = False  # quick and stable; avoids long training

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## === cell 4
ckpt_path = "best_model.keras"
callbacks = [
    ModelCheckpoint(
        ckpt_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
]

EPOCHS = 3
history = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2
)

model = tf.keras.models.load_model(ckpt_path)
print("Model training complete and best checkpoint loaded.")




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3653338213.py in <cell line: 0>()
     13 EPOCHS = 3
     14 history = model.fit(
---> 15     train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=2
     16 )
     17 

NameError: name 'train_ds' is not defined

## === cell 5
ss = pd.read_csv(SAMPLE_CSV)

test_tfrecord_files = tf.io.gfile.glob(TEST_TFREC_GLOB)
if not test_tfrecord_files:
    raise FileNotFoundError(f"No TFRecords found at glob: {TEST_TFREC_GLOB}")

options = tf.data.Options()
options.experimental_deterministic = True

files_ds = tf.data.Dataset.from_tensor_slices(sorted(test_tfrecord_files))
test_ds = files_ds.interleave(
    lambda fn: tf.data.TFRecordDataset(fn, num_parallel_reads=AUTOTUNE),
    cycle_length=min(8, len(test_tfrecord_files)),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
).with_options(options)

test_ds = test_ds.map(_parse_test_tfrecord, num_parallel_calls=AUTOTUNE)

test_ds = test_ds.map(
    lambda img_bytes, name: (
        decode_and_preprocess(img_bytes, label=None, training=False, from_bytes=True),
        name,
    ),
    num_parallel_calls=AUTOTUNE,
)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

probs = model.predict(test_ds.map(lambda x, name: x), verbose=0)
names = np.concatenate([n.numpy() for _, n in test_ds], axis=0).astype("U")

preds = np.argmax(probs, axis=1).astype(int)
pred_map = dict(zip(names, preds))

ordered_preds = ss["image_id"].map(pred_map).values.astype(int)

submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": ordered_preds})
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
