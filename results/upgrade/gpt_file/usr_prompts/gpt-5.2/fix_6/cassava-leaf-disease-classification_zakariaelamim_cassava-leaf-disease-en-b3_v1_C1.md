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

0.8643094590510728

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

ROOT_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(ROOT_DIR, "train.csv")
SAMPLE_SUB = os.path.join(ROOT_DIR, "sample_submission.csv")
TRAIN_DIR = os.path.join(ROOT_DIR, "train_images")
TEST_DIR = os.path.join(ROOT_DIR, "test_images")
TRAIN_TFRECORDS_DIR = os.path.join(ROOT_DIR, "train_tfrecords")
TEST_TFRECORDS_DIR = os.path.join(ROOT_DIR, "test_tfrecords")

print("ROOT_DIR exists:", os.path.exists(ROOT_DIR))
print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Train images dir exists:", os.path.exists(TRAIN_DIR))
print("Test images dir exists:", os.path.exists(TEST_DIR))
print("Train tfrecords dir exists:", os.path.exists(TRAIN_TFRECORDS_DIR))
print("Test tfrecords dir exists:", os.path.exists(TEST_TFRECORDS_DIR))
print("Num test images (dir listing):", len(os.listdir(TEST_DIR)))

train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

print(train_df.head())
print(sample_sub.head())



## === cell 1
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__)

try:
    tf.random.set_seed(SEED)
except Exception as e:
    print("tf.random.set_seed failed:", repr(e))

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception as e:
    print("enable_op_determinism not available:", repr(e))

try:
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception as e:
    print("threading config failed:", repr(e))



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
IMG_SIZE = 300
BATCH_SIZE = 16
NUM_CLASSES = 5
EPOCHS = 5  # keep modest to fit 600s budget

train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
val_frac = 0.1
val_size = int(len(train_df) * val_frac)
val_df = train_df.iloc[:val_size].copy()
trn_df = train_df.iloc[val_size:].copy()


def build_paths_and_labels(df):
    paths = (TRAIN_DIR + "/" + df["image_id"].astype(str)).to_numpy()
    labels = df["label"].astype("int32").to_numpy()
    return paths, labels


trn_paths, trn_labels = build_paths_and_labels(trn_df)
val_paths, val_labels = build_paths_and_labels(val_df)


def _list_tfrec_files(folder, prefix):
    if not os.path.exists(folder):
        return []
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.startswith(prefix) and f.endswith(".tfrec")
    ]
    return sorted(files)


train_tfrec_files = _list_tfrec_files(TRAIN_TFRECORDS_DIR, "ld_train")
test_tfrec_files = _list_tfrec_files(TEST_TFRECORDS_DIR, "ld_test")

print("Train TFRecord shards:", len(train_tfrec_files))
print("Test TFRecord shards:", len(test_tfrec_files))

TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
    "image_id": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def decode_and_resize_single(path, label):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear", antialias=False)
    img = tf.cast(img, tf.float32) / 255.0
    return img, tf.one_hot(label, NUM_CLASSES)


@tf.function
def decode_and_resize_test_single(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear", antialias=False)
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def _parse_tfrec_train(example_proto):
    ex = tf.io.parse_single_example(example_proto, TFREC_FEATURES)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear", antialias=False)
    img = tf.cast(img, tf.float32) / 255.0
    label = tf.cast(ex["label"], tf.int32)
    return img, tf.one_hot(label, NUM_CLASSES)


@tf.function
def _parse_tfrec_test(example_proto):
    ex = tf.io.parse_single_example(example_proto, TFREC_FEATURES)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear", antialias=False)
    img = tf.cast(img, tf.float32) / 255.0
    return img


AUTOTUNE = tf.data.AUTOTUNE

options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True

if len(train_tfrec_files) > 0:
    tfrec_files = np.array(train_tfrec_files)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(tfrec_files))
    tfrec_files = tfrec_files[perm]
    val_n_files = max(1, int(round(len(tfrec_files) * val_frac)))
    val_tfrec_files = tfrec_files[:val_n_files].tolist()
    trn_tfrec_files = tfrec_files[val_n_files:].tolist()

    raw_train = tf.data.TFRecordDataset(trn_tfrec_files, num_parallel_reads=AUTOTUNE)
    raw_val = tf.data.TFRecordDataset(val_tfrec_files, num_parallel_reads=AUTOTUNE)

    train_ds = raw_train.with_options(options)
    train_ds = train_ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    train_ds = train_ds.map(_parse_tfrec_train, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.cache()  # in-memory cache of preprocessed tensors
    train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
    train_ds = train_ds.prefetch(AUTOTUNE)

    val_ds = raw_val.with_options(options)
    val_ds = val_ds.map(_parse_tfrec_train, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.cache()
    val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False)
    val_ds = val_ds.prefetch(AUTOTUNE)
else:
    train_ds = tf.data.Dataset.from_tensor_slices((trn_paths, trn_labels))
    train_ds = train_ds.with_options(options)
    train_ds = train_ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    train_ds = train_ds.map(decode_and_resize_single, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.cache()
    train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
    train_ds = train_ds.prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.with_options(options)
    val_ds = val_ds.map(decode_and_resize_single, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.cache()
    val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False)
    val_ds = val_ds.prefetch(AUTOTUNE)

print("Train batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("Val batches:", tf.data.experimental.cardinality(val_ds).numpy())



## === cell 3
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=32,
)
model.summary()



## === cell 4
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/183724393.py in <cell line: 0>()
----> 1 history = model.fit(
      2     train_ds,
      3     validation_data=val_ds,
      4     epochs=EPOCHS,
      5     verbose=1,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

InvalidArgumentError: Graph execution error:

Detected at node ParseSingleExample/ParseExample/ParseExampleV2 defined at (most recent call last):
<stack traces unavailable>
Error in user-defined function passed to ParallelMapDatasetV2:6 transformation with iterator: Iterator::Root::Prefetch::BatchV2::MemoryCacheImpl::ParallelMapV2: Feature: image_id (data type: string) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNextAsOptional]] [Op:__inference_multi_step_on_iterator_2131]

## === cell 5
test_image_ids = sample_sub["image_id"].to_numpy()
test_paths = (TEST_DIR + "/" + sample_sub["image_id"].astype(str)).to_numpy()

if len(test_tfrec_files) > 0:
    TFREC_TEST_FEATURES = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_id": tf.io.FixedLenFeature([], tf.string),
    }

    @tf.function
    def _parse_tfrec_test_with_id(example_proto):
        ex = tf.io.parse_single_example(example_proto, TFREC_TEST_FEATURES)
        img = tf.image.decode_jpeg(ex["image"], channels=3)
        img = tf.image.resize(
            img, [IMG_SIZE, IMG_SIZE], method="bilinear", antialias=False
        )
        img = tf.cast(img, tf.float32) / 255.0
        return ex["image_id"], img

    raw_test = tf.data.TFRecordDataset(
        test_tfrec_files, num_parallel_reads=AUTOTUNE
    ).with_options(options)
    test_ds = raw_test.map(_parse_tfrec_test_with_id, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    all_ids = []
    all_probs = []
    for batch_ids, batch_imgs in test_ds:
        batch_probs = model(batch_imgs, training=False).numpy()
        all_probs.append(batch_probs)
        all_ids.append(batch_ids.numpy())

    probs = np.concatenate(all_probs, axis=0)
    tfrec_ids = np.concatenate(all_ids, axis=0).astype("U")

    preds = probs.argmax(axis=1).astype(int)

    id_to_pred = dict(zip(tfrec_ids.tolist(), preds.tolist()))
    preds = np.array([id_to_pred[i] for i in test_image_ids], dtype=int)
else:
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = test_ds.with_options(options)
    test_ds = test_ds.map(decode_and_resize_test_single, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False)
    test_ds = test_ds.prefetch(AUTOTUNE)

    probs = model.predict(test_ds, verbose=1)
    preds = probs.argmax(axis=1).astype(int)

print("Preds length:", len(preds), "Expected:", len(test_image_ids))



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/3634270858.py in <cell line: 0>()
     31     all_ids = []
     32     all_probs = []
---> 33     for batch_ids, batch_imgs in test_ds:
     34         batch_probs = model(batch_imgs, training=False).numpy()
     35         all_probs.append(batch_probs)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in __next__(self)
    824   def __next__(self):
    825     try:
--> 826       return self._next_internal()
    827     except errors.OutOfRangeError:
    828       raise StopIteration

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in _next_internal(self)
    774     # to communicate that there is no more data to iterate over.
    775     with context.execution_mode(context.SYNC):
--> 776       ret = gen_dataset_ops.iterator_get_next(
    777           self._iterator_resource,
    778           output_types=self._flat_output_types,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_dataset_ops.py in iterator_get_next(iterator, output_types, output_shapes, name)
   3084       return _result
   3085     except _core._NotOkStatusException as e:
-> 3086       _ops.raise_from_not_ok_status(e, name)
   3087     except _core._FallbackException:
   3088       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Feature: image_id (data type: string) is required but could not be found.
	 [[{{function_node __inference__parse_tfrec_test_with_id_2172}}{{node ParseSingleExample/ParseExample/ParseExampleV2}}]] [Op:IteratorGetNext] name: 

## === cell 6
sub = pd.DataFrame({"image_id": test_image_ids, "label": preds})
print(sub.head())

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(sub))
print("Submission columns:", list(sub.columns))

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/120832581.py in <cell line: 0>()
----> 1 sub = pd.DataFrame({"image_id": test_image_ids, "label": preds})
      2 print(sub.head())
      3 
      4 out_path = "submission.csv"
      5 sub.to_csv(out_path, index=False)

NameError: name 'preds' is not defined
