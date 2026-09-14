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

0.7733454215775158

# 6. Current score

0.12967

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.12967) has done: 'The timeout is almost certainly dominated by JPEG decoding + resizing the full training set for every epoch (even with `.cache()` this spills to disk and still costs a lot) and by recomputing the `shuffle(buffer_size=len(train))` each epoch. I keep the exact same model, loss, optimizer, epochs, and dataset semantics, but switch training/validation input to the provided TFRecords (same images/labels) to avoid per-file JPEG overhead and leverage sequential reads. I also make caching explicitly file-backed (deterministic and stable) and keep deterministic options, while preserving the stratified split and label mapping exactly. Prediction remains on JPEGs (test TFRecords don’t include filenames), but input pipeline is tuned with parallelism and prefetch for speed.'
- What this solution (achieved 0.12967) has done: 'I first fix the TensorFlow import crash in cell 1 by setting the pure-Python protobuf implementation before importing TensorFlow (this resolves the `MessageFactory.GetPrototype` issue seen in some Kaggle images with newer Python/protobuf). Next, I fix the TFRecord parsing bug by using the correct feature key for the filename (the Cassava TFRecords store `image_name`, not `image_id`), and I still map labels via your existing CSV-driven `id_to_label` to preserve the original label encoding semantics. Finally, I ensure the TFRecord `image_name` matches your CSV `image_id` format by appending “.jpg” when needed, so the train/valid filtering and label lookup work correctly and training can run end-to-end to produce `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

print("TF version:", tf.__version__)

SEED = 42
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)
train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

train_csv["disease"] = train_csv["label"].map(label_to_disease)
train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"]
)

train_csv["disease"] = train_csv["disease"].astype(str)
train_csv["label"] = train_csv["label"].astype(str)

disease_categories = np.sort(train_csv["disease"].unique())
disease_to_int = {d: i for i, d in enumerate(disease_categories)}
train_csv["label_encoded"] = train_csv["disease"].map(disease_to_int).astype(np.int64)

rng = np.random.default_rng(SEED)
valid_frac = 0.2
valid_indices = []
for lbl, idx in train_csv.groupby("label", sort=True).indices.items():
    idx = np.asarray(idx, dtype=np.int64)
    rng.shuffle(idx)
    n_valid = int(np.round(len(idx) * valid_frac))
    valid_indices.append(idx[:n_valid])
valid_indices = np.concatenate(valid_indices)
valid_mask = np.zeros(len(train_csv), dtype=bool)
valid_mask[valid_indices] = True

valid = train_csv.loc[valid_mask].reset_index(drop=True)
train = train_csv.loc[~valid_mask].reset_index(drop=True)

id_to_label = dict(
    zip(
        train_csv["image_id"].values.tolist(),
        train_csv["label_encoded"].values.tolist(),
    )
)
train_ids = set(train["image_id"].values.tolist())
valid_ids = set(valid["image_id"].values.tolist())



## === cell 2
AUTOTUNE = tf.data.AUTOTUNE

TRAIN_TFREC_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords"
train_tfrecords = tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec"))
train_tfrecords = sorted(train_tfrecords)

options = tf.data.Options()
options.experimental_deterministic = True
try:
    options.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass

train_ids_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(list(train_ids), dtype=tf.string),
        values=tf.ones([len(train_ids)], dtype=tf.int32),
    ),
    default_value=tf.constant(0, dtype=tf.int32),
)
valid_ids_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(list(valid_ids), dtype=tf.string),
        values=tf.ones([len(valid_ids)], dtype=tf.int32),
    ),
    default_value=tf.constant(0, dtype=tf.int32),
)
labels_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(list(id_to_label.keys()), dtype=tf.string),
        values=tf.constant(list(id_to_label.values()), dtype=tf.int64),
    ),
    default_value=tf.constant(-1, dtype=tf.int64),
)

_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
}


def _ensure_jpg_suffix(name):
    name = tf.strings.strip(name)
    has_jpg = tf.strings.regex_full_match(tf.strings.lower(name), r".*\.jpg$")
    return tf.cond(has_jpg, lambda: name, lambda: tf.strings.join([name, ".jpg"]))


def _parse_tfrec(serialized):
    ex = tf.io.parse_single_example(serialized, _FEATURES)
    image = tf.image.decode_jpeg(ex["image"], channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)

    image_id = _ensure_jpg_suffix(ex["image_name"])

    label = labels_table.lookup(image_id)
    return image_id, image, label


def _is_in_train(image_id, image, label):
    return tf.equal(train_ids_table.lookup(image_id), 1)


def _is_in_valid(image_id, image, label):
    return tf.equal(valid_ids_table.lookup(image_id), 1)


def _drop_id(image_id, image, label):
    return image, label


files_ds = tf.data.Dataset.from_tensor_slices(train_tfrecords).with_options(options)
raw_ds = files_ds.interleave(
    lambda fn: tf.data.TFRecordDataset(fn, num_parallel_reads=AUTOTUNE),
    cycle_length=min(len(train_tfrecords), 8),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)

parsed = raw_ds.map(_parse_tfrec, num_parallel_calls=AUTOTUNE)


def _label_is_valid(image_id, image, label):
    return tf.greater_equal(label, 0)


parsed = parsed.filter(_label_is_valid)

train_ds = (
    parsed.filter(_is_in_train)
    .map(_drop_id, num_parallel_calls=AUTOTUNE)
    .cache("/kaggle/working/train_cache.tfdata")
    .shuffle(buffer_size=len(train), seed=SEED, reshuffle_each_iteration=True)
    .batch(32, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

valid_ds = (
    parsed.filter(_is_in_valid)
    .map(_drop_id, num_parallel_calls=AUTOTUNE)
    .cache("/kaggle/working/valid_cache.tfdata")
    .batch(32, drop_remainder=False)
    .prefetch(AUTOTUNE)
)



## === cell 3
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback


class EarlyStoppingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if self.model.stop_training:
            print(f"Early stopping triggered at epoch {epoch + 1}.")


early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1
)



## === cell 4
NUM_CLASSES = 5

base_model = EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
x = GlobalAveragePooling2D()(base_model.output)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=5,
    callbacks=[early_stopping, learning_rate_reduction],
    verbose=1,
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/160922946.py in <cell line: 0>()
     14 )
     15 
---> 16 history = model.fit(
     17     train_ds,
     18     validation_data=valid_ds,

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
Error in user-defined function passed to ParallelMapDatasetV2:3 transformation with iterator: Iterator::Root::Prefetch::BatchV2::Shuffle::FileCacheImpl::Map::Filter::Filter::Map: Feature: label (data type: int64) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_49622]

## === cell 5
import pandas as pd
import numpy as np
import os

sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"

image_ids = sample_sub["image_id"].tolist()
test_paths = [os.path.join(test_dir, image_id) for image_id in image_ids]


def load_test_only(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image


AUTOTUNE = tf.data.AUTOTUNE
options = tf.data.Options()
options.experimental_deterministic = True
try:
    options.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .with_options(options)
    .map(load_test_only, num_parallel_calls=AUTOTUNE)
    .batch(64, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

probs = model.predict(test_ds, verbose=0)
pred_labels = np.argmax(probs, axis=1).astype(np.int64)

submission_df = pd.DataFrame({"image_id": image_ids, "label": pred_labels})
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Submission file created:", submission_path)
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.shape[1])
print("Label value counts:\n", submission_df["label"].value_counts().sort_index())
