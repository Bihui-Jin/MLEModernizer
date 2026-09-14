# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" in os.environ:
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import glob
import math
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.io import read_file
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from sklearn.model_selection import StratifiedKFold

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
tf.get_logger().setLevel("ERROR")

BATCH_SIZE = 64
ROW = 224
COL = 224

train_csv_loc = "../input/cassava-leaf-disease-classification/train.csv"
train_location = "../input/cassava-leaf-disease-classification/train_images/"
test_location = "../input/cassava-leaf-disease-classification/test_images/"

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

os.makedirs("tf_cache", exist_ok=True)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass




## === cell 1
@tf.function
def _tf_augment(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.10)
    img = tf.image.random_contrast(img, lower=0.90, upper=1.10)
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    img = tf.image.rot90(img, k=k)
    return img




## === cell 2
@tf.function
def fetch_image_without_aug(filename, label):
    img_bytes = read_file(filename)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [ROW, COL])
    img = preprocess_input(img)
    return img, label




## === cell 3
@tf.function
def fetch_image_with_aug(filename, label):
    img_bytes = read_file(filename)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [ROW, COL])
    img = _tf_augment(img)
    img = preprocess_input(img)
    return img, label




## === cell 4
@tf.function
def _decode_resize_for_cache(filename, label):
    img_bytes = read_file(filename)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [ROW, COL])
    return img, label


@tf.function
def _augment_and_preprocess(img, label):
    img = _tf_augment(img)
    img = preprocess_input(img)
    return img, label




## === cell 5
def _make_data_options(deterministic=True):
    opts = tf.data.Options()
    opts.deterministic = deterministic

    try:
        opts.experimental_slack = True
    except Exception:
        pass

    try:
        opts.threading.private_threadpool_size = max(4, (os.cpu_count() or 8) - 1)
    except Exception:
        pass

    try:
        eo = opts.experimental_optimization
        if hasattr(eo, "map_and_batch_fusion"):
            eo.map_and_batch_fusion = True
        if hasattr(eo, "parallel_batch"):
            eo.parallel_batch = True
        if hasattr(eo, "autotune_buffers"):
            eo.autotune_buffers = True
    except Exception:
        pass

    return opts


def build_global_train_cache(all_files, all_labels):
    ds = Dataset.from_tensor_slices((all_files, all_labels)).with_options(
        _make_data_options(deterministic=True)
    )
    ds = ds.map(_decode_resize_for_cache, num_parallel_calls=AUTOTUNE)
    ds = ds.cache(os.path.join("tf_cache", "train_all_decode_resize"))
    ds = ds.prefetch(AUTOTUNE)
    return ds


def materialize_global_cache_to_numpy(global_cached_ds, n_total):
    imgs = np.empty((n_total, ROW, COL, 3), dtype=np.float32)
    labels = np.empty((n_total,), dtype=np.int64)

    i = 0
    for batch_imgs, batch_labels in global_cached_ds.batch(256).prefetch(AUTOTUNE):
        bi = int(batch_imgs.shape[0])
        imgs[i : i + bi] = batch_imgs.numpy()
        labels[i : i + bi] = batch_labels.numpy().astype(np.int64, copy=False)
        i += bi
    if i != n_total:
        raise RuntimeError(f"Materialization mismatch: got {i}, expected {n_total}")
    return imgs, labels


def getDatasetFromMaterializedCache(
    cached_imgs_np, cached_labels_np, train_index, val_index, n_train, n_val
):
    train_opts = _make_data_options(deterministic=True)
    val_opts = _make_data_options(deterministic=True)

    x_train = cached_imgs_np[np.asarray(train_index, dtype=np.int64)]
    y_train = cached_labels_np[np.asarray(train_index, dtype=np.int64)]
    x_val = cached_imgs_np[np.asarray(val_index, dtype=np.int64)]
    y_val = cached_labels_np[np.asarray(val_index, dtype=np.int64)]

    train_ds = Dataset.from_tensor_slices((x_train, y_train)).with_options(train_opts)
    train_ds = train_ds.shuffle(n_train, seed=SEED, reshuffle_each_iteration=True)
    train_ds = train_ds.map(_augment_and_preprocess, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
    train_ds = train_ds.prefetch(AUTOTUNE)

    val_ds = Dataset.from_tensor_slices((x_val, y_val)).with_options(val_opts)
    val_ds = val_ds.map(
        lambda img, label: (preprocess_input(img), label), num_parallel_calls=AUTOTUNE
    )
    val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds




## === cell 6
def scheduler(epoch, lr):
    if epoch < 8:
        return lr
    else:
        return lr * np.exp(-0.05)




## === cell 7
def create_callbacks(folder_name):
    os.makedirs(os.path.join("Weights", folder_name), exist_ok=True)

    lr_scheduler = LearningRateScheduler(scheduler)

    weight_save_only = ModelCheckpoint(
        filepath=os.path.join("Weights", f"{folder_name}.weights.h5"),
        monitor="val_accuracy",
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
    )

    callbacks = [lr_scheduler, weight_save_only]
    return callbacks




## === cell 8
def create_model(training=True, weights="imagenet"):
    base_model = EfficientNetB0(
        weights=weights, include_top=False, input_shape=(ROW, COL, 3)
    )

    x = Input(shape=(ROW, COL, 3))
    out_1 = base_model(x, training=training)
    out_1 = GlobalAveragePooling2D(name="encoding")(out_1)
    out_1 = Dropout(0.5)(out_1)
    output = Dense(5, activation="softmax")(out_1)

    final_model = Model(inputs=x, outputs=output)
    return final_model




## === cell 9
train_csv = pd.read_csv(train_csv_loc)
train_csv["image_id"] = train_csv["image_id"].map(
    lambda x: os.path.join(train_location, x)
)

all_files = train_csv["image_id"].values
all_labels = train_csv["label"].values.astype(np.int64)
n_total = len(all_files)

global_cached_ds = build_global_train_cache(all_files, all_labels)

cached_imgs_np, cached_labels_np = materialize_global_cache_to_numpy(
    global_cached_ds, n_total
)

create_k_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

base_folder_name = "Exp_"
fold_number = 1

for train_index, test_index in create_k_folds.split(all_files, all_labels):
    print(f"Fold Number {fold_number} is starting it's training")

    folder_name = base_folder_name + str(fold_number)
    print("Creating Callbacks...")
    callbacks = create_callbacks(folder_name)

    n_train = int(len(train_index))
    n_val = int(len(test_index))

    print("Importing Datasets...")
    train_ds, val_ds = getDatasetFromMaterializedCache(
        cached_imgs_np, cached_labels_np, train_index, test_index, n_train, n_val
    )

    final_model = create_model(training=True, weights="imagenet")
    final_model.compile(
        optimizer=RMSprop(learning_rate=1e-4),
        loss=SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
        jit_compile=True,
    )

    steps_per_epoch = int(math.ceil(n_train / float(BATCH_SIZE)))
    validation_steps = int(math.ceil(n_val / float(BATCH_SIZE)))

    print("Starting training...")
    final_model.fit(
        train_ds,
        epochs=15,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=2,
    )

    fold_number += 1



## === cell 10
checkpoints = sorted(glob.glob(os.path.join("Weights", "Exp_*.weights.h5")))
if len(checkpoints) == 0:
    checkpoints = sorted(glob.glob(os.path.join("Weights", "*.weights.h5")))

if len(checkpoints) == 0:
    raise FileNotFoundError(
        "No fold checkpoints found under Weights/. Training likely failed before saving."
    )

Models = []
for checkpoint in checkpoints:
    m = create_model(training=False, weights=None)
    m.load_weights(checkpoint)
    Models.append(m)

print(f"Models loaded: {len(Models)}")
print("Example checkpoint:", checkpoints[0])



## === cell 11
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
sample_sub = pd.read_csv(sample_sub_path)

test_files = sample_sub["image_id"].map(lambda x: os.path.join(test_location, x)).values


@tf.function
def _fetch_test_image(filename):
    img_bytes = read_file(filename)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [ROW, COL])
    img = preprocess_input(img)
    return img


test_ds = Dataset.from_tensor_slices(test_files).with_options(
    _make_data_options(deterministic=True)
)
test_ds = test_ds.map(_fetch_test_image, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.cache(os.path.join("tf_cache", "test"))
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

preds_sum = None
for model in Models:
    p = model.predict(test_ds, verbose=0)
    if preds_sum is None:
        preds_sum = p
    else:
        preds_sum += p
preds_mean = preds_sum / float(len(Models))

results = np.argmax(preds_mean, axis=1).astype(int).tolist()

submission = pd.DataFrame({"image_id": sample_sub["image_id"], "label": results})
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
