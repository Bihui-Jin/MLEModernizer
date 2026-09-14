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

0.7265034753702024

# 6. Current score

0.12631

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.12631) has done: 'The timeout is dominated by slow Python-based image augmentation/loading (Keras `ImageDataGenerator` with JPEG decode in Python) and a per-image Python loop for test inference. I keep the same EfficientNetB0 model, losses, epochs, and fine-tuning logic, but switch the data pipeline to `tf.data` reading the provided TFRecords with identical preprocessing and equivalent augmentations executed inside TensorFlow for much higher throughput. I also vectorize test-time inference by building a batched `tf.data` dataset instead of `load_img` in a Python loop. Finally, I enable deterministic execution and add `prefetch`/parallelism knobs that are correctness-preserving and significantly reduce input bottlenecks.'
- What this solution (achieved 0.12631) has done: 'We fix the TensorFlow import crash in cell 1 by removing the protobuf implementation environment tweak that’s incompatible with the Kaggle TF/protobuf build, while keeping determinism settings intact. Then we fix the dataset augmentation error by providing correct 2-int seeds to `stateless_random_flip_*` (the current code builds a 3-element seed tuple, which TensorFlow rejects). With those two fixes, `train_ds/valid_ds` build successfully so training and fine-tuning run, and the existing TFRecord-based batched test inference produce a valid `submission.csv`. These changes are execution-blocking bug fixes and should also raise accuracy substantially from the current 0.12631 by enabling the intended training pipeline to actually run.'
- What this solution (achieved 0.12631) has done: 'I fix the TensorFlow import crash by setting the protobuf implementation to `python` before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` incompatibility and lets the pipeline start. Then I fix the dataset mapping error by ensuring `_augment` preserves the third element (`image_id`) so the later `map(lambda img,y,image_id: ...)` receives the expected 3-tuple; this unblocks `train_ds/valid_ds` creation and therefore training. Finally, I keep the model/training logic unchanged, but add a small safety fallback in test prediction ordering to guarantee submission rows align with `sample_submission.csv` even if TFRecord iteration order differs, producing a valid `submission.csv`.'
- What this solution (achieved 0.12631) has done: 'The timeout is dominated by expensive per-example `ds.filter()` with a `StaticHashTable`, plus repeated Python-side materialization of all test IDs and an extra pass over the test dataset before prediction. I keep the exact model/training logic intact, but replace the TFRecord “read everything then filter” approach with index-based shard + in-shard record selection so we only read the needed examples for train/valid, preserving the exact split semantics. I also make test prediction single-pass (collect IDs and predict in the same loop) to remove an entire dataset traversal and large intermediate concatenations. Finally, I add `cache()` for validation and tune `tf.data` options to reduce pipeline overhead while keeping determinism.'

# 9. Code solution

## === cell 0
import os
import random
from datetime import datetime

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import numpy as np
import pandas as pd
import tensorflow as tf

print("TF version:", tf.__version__)
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.config.run_functions_eagerly(False)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input

label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)

train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")
train_csv["label"] = train_csv["label"].astype(int)

train, valid = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=SEED
)

NUM_CLASSES = int(train_csv["label"].nunique())
print("Detected classes:", NUM_CLASSES)

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
AUTO = tf.data.AUTOTUNE

TRAIN_TFREC_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords"
TEST_TFREC_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords"

train_tfrec_files = sorted(tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
test_tfrec_files = sorted(tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))

_SHARD_SIZE = 1338

all_image_ids = train_csv["image_id"].astype(str).to_numpy()
id_to_global_idx = {iid: i for i, iid in enumerate(all_image_ids)}

train_global_idx = np.fromiter(
    (id_to_global_idx[iid] for iid in train["image_id"].astype(str).tolist()),
    dtype=np.int64,
    count=len(train),
)
valid_global_idx = np.fromiter(
    (id_to_global_idx[iid] for iid in valid["image_id"].astype(str).tolist()),
    dtype=np.int64,
    count=len(valid),
)


def _select_tfrec_files_for_global_indices(
    global_indices, tfrec_files, shard_size=_SHARD_SIZE
):
    if len(global_indices) == 0:
        return tfrec_files
    shards = {int(i) // shard_size for i in global_indices}
    return [tfrec_files[s] for s in sorted(shards) if 0 <= s < len(tfrec_files)]


train_tfrec_files_sel = _select_tfrec_files_for_global_indices(
    train_global_idx, train_tfrec_files
)
valid_tfrec_files_sel = _select_tfrec_files_for_global_indices(
    valid_global_idx, train_tfrec_files
)


def _build_shard_to_local_indices(global_indices, shard_size=_SHARD_SIZE):
    shard_to_locals = {}
    for g in np.asarray(global_indices, dtype=np.int64):
        s = int(g) // shard_size
        l = int(g) - s * shard_size
        shard_to_locals.setdefault(s, []).append(l)
    for s in shard_to_locals:
        shard_to_locals[s].sort()
    return shard_to_locals


train_shard_to_locals = _build_shard_to_local_indices(train_global_idx)
valid_shard_to_locals = _build_shard_to_local_indices(valid_global_idx)

_FEATURES_FULL_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
}


def _basename_image_id(image_name):
    parts = tf.strings.split(image_name, sep="/")
    return parts[-1]


def _decode_full_train(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES_FULL_TRAIN)
    label = tf.cast(ex["target"], tf.int32)

    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)  # keep identical preprocessing intent

    y = tf.one_hot(label, depth=NUM_CLASSES, dtype=tf.float32)
    return img, y, _basename_image_id(ex["image_name"])


def _augment(img, y, image_id):
    h = tf.strings.to_hash_bucket_fast(image_id, 2**31 - 1)
    h32 = tf.cast(h, tf.int32)

    seed_lr = tf.stack([tf.cast(SEED, tf.int32), h32 + 1], axis=0)
    seed_ud = tf.stack([tf.cast(SEED, tf.int32), h32 + 2], axis=0)

    img = tf.image.stateless_random_flip_left_right(img, seed=seed_lr)
    img = tf.image.stateless_random_flip_up_down(img, seed=seed_ud)

    seed = tf.stack([tf.cast(SEED, tf.int32), h32], axis=0)

    rot = tf.random.stateless_uniform([], seed=seed, minval=-45.0, maxval=45.0) * (
        np.pi / 180.0
    )
    seed2 = seed + tf.constant([0, 13], tf.int32)
    shx = tf.random.stateless_uniform([], seed=seed2, minval=-0.2, maxval=0.2)
    seed3 = seed + tf.constant([0, 29], tf.int32)
    shy = tf.random.stateless_uniform([], seed=seed3, minval=-0.2, maxval=0.2)
    seed4 = seed + tf.constant([0, 47], tf.int32)
    zoom = tf.random.stateless_uniform([], seed=seed4, minval=0.8, maxval=1.2)
    seed5 = seed + tf.constant([0, 61], tf.int32)
    dx = tf.random.stateless_uniform([], seed=seed5, minval=-0.2, maxval=0.2) * tf.cast(
        IMG_SIZE[1], tf.float32
    )
    seed6 = seed + tf.constant([0, 79], tf.int32)
    dy = tf.random.stateless_uniform([], seed=seed6, minval=-0.2, maxval=0.2) * tf.cast(
        IMG_SIZE[0], tf.float32
    )

    c = tf.math.cos(rot)
    s = tf.math.sin(rot)

    a0 = (c / zoom) + shx
    a1 = -s / zoom
    b0 = s / zoom
    b1 = (c / zoom) + shy

    cx = (IMG_SIZE[1] - 1) / 2.0
    cy = (IMG_SIZE[0] - 1) / 2.0

    a2 = cx - a0 * cx - a1 * cy + dx
    b2 = cy - b0 * cx - b1 * cy + dy

    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0], axis=0)
    transform = tf.reshape(transform, [1, 8])

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(img, 0),
        transforms=transform,
        output_shape=tf.constant([IMG_SIZE[0], IMG_SIZE[1]], tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    img = tf.squeeze(img, 0)

    return img, y, image_id


def _make_dataset_from_tfrecs_selected(
    tfrec_files, training: bool, shard_to_locals, shard_size=_SHARD_SIZE
):
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.threading.private_threadpool_size = 16
    opts.threading.max_intra_op_parallelism = 1
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True

    shard_ids = [i for i in range(len(tfrec_files)) if i in shard_to_locals]
    if len(shard_ids) == 0:
        empty = tf.data.Dataset.from_tensors(tf.constant("", tf.string)).take(0)
        ds = empty.map(
            lambda _: (
                tf.zeros([*IMG_SIZE, 3], tf.float32),
                tf.zeros([NUM_CLASSES], tf.float32),
            )
        )
        return ds.with_options(opts)

    shard_files = [tfrec_files[i] for i in shard_ids]
    locals_lists = [np.asarray(shard_to_locals[i], dtype=np.int64) for i in shard_ids]
    locals_ragged = tf.ragged.constant(locals_lists, dtype=tf.int64)

    def _dataset_for_one_shard(file_path, local_idx_vec):
        local_idx_vec = tf.cast(local_idx_vec, tf.int64)
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=local_idx_vec,
                values=tf.ones([tf.shape(local_idx_vec)[0]], tf.int32),
            ),
            default_value=0,
        )
        ds0 = tf.data.TFRecordDataset(file_path)
        ds0 = ds0.enumerate()
        ds0 = ds0.filter(lambda i, _: table.lookup(tf.cast(i, tf.int64)) > 0)
        ds0 = ds0.map(lambda _, ex: ex, num_parallel_calls=AUTO)
        return ds0

    file_ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant(shard_files), locals_ragged)
    )
    ds = file_ds.interleave(
        lambda fp, locs: _dataset_for_one_shard(fp, locs),
        cycle_length=min(8, len(shard_files)),
        num_parallel_calls=AUTO,
        deterministic=True,
        block_length=1,
    ).with_options(opts)

    ds = ds.map(_decode_full_train, num_parallel_calls=AUTO)

    if training:
        ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(_augment, num_parallel_calls=AUTO)

    ds = ds.map(lambda img, y, image_id: (img, y), num_parallel_calls=AUTO)

    if not training:
        ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=bool(training))
    ds = ds.prefetch(AUTO)
    return ds


train_ds = _make_dataset_from_tfrecs_selected(
    train_tfrec_files, training=True, shard_to_locals=train_shard_to_locals
)
valid_ds = _make_dataset_from_tfrecs_selected(
    train_tfrec_files, training=False, shard_to_locals=valid_shard_to_locals
)

train_steps = int(np.ceil(len(train) / BATCH_SIZE))
valid_steps = int(np.ceil(len(valid) / BATCH_SIZE))
print("Train/Valid sizes:", len(train), len(valid), "steps:", train_steps, valid_steps)
print(
    "Train shards selected:", len(train_tfrec_files_sel), "of", len(train_tfrec_files)
)
print(
    "Valid shards selected:", len(valid_tfrec_files_sel), "of", len(train_tfrec_files)
)



## === cell 2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback


class EarlyStoppingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if self.model.stop_training:
            print(f"Early stopping triggered at epoch {epoch + 1}.")


early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1
)



## === cell 3
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)
base_model.trainable = False  # stable + fast within time limit

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2, seed=SEED)(x)
preds = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=preds)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()



## === cell 4
EPOCHS = 8

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=valid_steps,
    callbacks=[early_stopping, learning_rate_reduction, EarlyStoppingCallback()],
    verbose=1,
)

base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history_ft = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=4,
    steps_per_epoch=train_steps,
    validation_steps=valid_steps,
    callbacks=[early_stopping, learning_rate_reduction, EarlyStoppingCallback()],
    verbose=1,
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FailedPreconditionError                   Traceback (most recent call last)
/tmp/ipykernel_11/3517077506.py in <cell line: 0>()
      1 EPOCHS = 8
      2 
----> 3 history = model.fit(
      4     train_ds,
      5     validation_data=valid_ds,

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

FailedPreconditionError: Graph execution error:

Detected at node key_value_init/LookupTableImportV2 defined at (most recent call last):
<stack traces unavailable>
Error in user-defined function passed to ParallelInterleaveDatasetV4:1 transformation with iterator: Iterator::Root::Prefetch::BatchV2::Map::Map::Shuffle::Map::Interleave: Table was already initialized with different data.
	 [[{{node key_value_init/LookupTableImportV2}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_16023]

## === cell 5
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
sub = pd.read_csv(sample_sub_path)

test_ids = sub["image_id"].astype(str).tolist()

_FEATURES_FULL_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _decode_full_test(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES_FULL_TEST)
    image_id = _basename_image_id(ex["image_name"])
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img, image_id


test_opts = tf.data.Options()
test_opts.experimental_deterministic = True
test_opts.threading.private_threadpool_size = 16
test_opts.threading.max_intra_op_parallelism = 1
test_opts.experimental_optimization.map_parallelization = True
test_opts.experimental_optimization.parallel_batch = True

num_reads_test = min(8, max(1, len(test_tfrec_files)))
test_ds = tf.data.TFRecordDataset(
    test_tfrec_files, num_parallel_reads=num_reads_test
).with_options(test_opts)
test_ds = test_ds.map(_decode_full_test, num_parallel_calls=AUTO)
test_ds = test_ds.batch(64, drop_remainder=False).prefetch(AUTO)

all_ids_list = []
preds_list = []
for batch_imgs, batch_ids in test_ds:
    batch_probs = model(batch_imgs, training=False).numpy()
    preds_list.append(np.argmax(batch_probs, axis=1).astype(np.int64))
    all_ids_list.append(batch_ids.numpy())

all_ids = np.concatenate(all_ids_list, axis=0).astype("U")
pred_idx = np.concatenate(preds_list, axis=0).astype(np.int64)

id_to_pred = dict(zip(all_ids.tolist(), pred_idx.tolist()))
pred_labels = [int(id_to_pred.get(iid, 0)) for iid in test_ids]

submission_df = pd.DataFrame({"image_id": sub["image_id"], "label": pred_labels})
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Submission file created:", submission_path)
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.columns.tolist())
assert submission_df.shape[0] == sub.shape[0]
assert submission_df.columns.tolist() == ["image_id", "label"]
assert submission_path.endswith(".csv")
