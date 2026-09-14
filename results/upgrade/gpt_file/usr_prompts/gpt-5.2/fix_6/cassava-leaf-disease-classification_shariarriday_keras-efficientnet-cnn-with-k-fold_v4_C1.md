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

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import glob
import math
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from sklearn.model_selection import StratifiedKFold

BATCH_SIZE = 64
ROW = 300
COL = 300

train_csv_loc = "../input/cassava-leaf-disease-classification/train.csv"
train_location = "../input/cassava-leaf-disease-classification/train_images/"
test_location = "../input/cassava-leaf-disease-classification/test_images/"
sample_sub_loc = "../input/cassava-leaf-disease-classification/sample_submission.csv"

train_tfrecords_glob = (
    "../input/cassava-leaf-disease-classification/train_tfrecords/*.tfrec"
)
test_tfrecords_glob = (
    "../input/cassava-leaf-disease-classification/test_tfrecords/*.tfrec"
)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF version:", tf.__version__)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

DATASET_OPTIONS = tf.data.Options()
DATASET_OPTIONS.experimental_deterministic = True
DATASET_OPTIONS.threading.private_threadpool_size = 32
DATASET_OPTIONS.threading.max_intra_op_parallelism = 0




## === cell 1
def augmentations_tf(file):
    file = tf.io.read_file(file)
    file = tf.image.decode_jpeg(file, channels=3)
    file = tf.image.resize(file, [ROW, COL])

    file = tf.image.random_flip_left_right(file)
    file = tf.image.random_flip_up_down(file)
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    file = tf.image.rot90(file, k)

    file = tf.image.random_brightness(file, max_delta=0.15)
    file = tf.image.random_contrast(file, lower=0.85, upper=1.15)
    file = tf.image.random_saturation(file, lower=0.85, upper=1.15)
    file = tf.image.random_hue(file, max_delta=0.05)

    file = preprocess_input(file)
    return file




## === cell 2
def fetch_image_without_aug(filename, label):
    image_file = tf.io.read_file(filename)
    image_file = tf.image.decode_jpeg(image_file, channels=3)
    image_file = tf.image.resize(image_file, [ROW, COL])
    image_file = preprocess_input(image_file)
    return image_file, label




## === cell 3
def fetch_image_with_aug(filename, label):
    aug_img = augmentations_tf(filename)
    aug_img.set_shape((ROW, COL, 3))
    return aug_img, label




## === cell 4
_FEATURE_DESC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


def _decode_from_tfrecord(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESC)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, [ROW, COL])
    img = preprocess_input(img)
    label = tf.cast(ex["target"], tf.int32)
    return img, label


def _decode_from_tfrecord_with_aug(example_proto):
    img, label = _decode_from_tfrecord(example_proto)

    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    img = tf.image.rot90(img, k)

    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    img = tf.image.random_saturation(img, lower=0.85, upper=1.15)
    img = tf.image.random_hue(img, max_delta=0.05)

    img.set_shape((ROW, COL, 3))
    return img, label


def _make_tfrecord_dataset(tfrecord_files, training):
    ds = tf.data.TFRecordDataset(tfrecord_files, num_parallel_reads=AUTOTUNE)
    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(_decode_from_tfrecord_with_aug, num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(_decode_from_tfrecord, num_parallel_calls=AUTOTUNE)
        ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.with_options(DATASET_OPTIONS)
    return ds




## === cell 5
def getDatasetFromDataframe(train_files, train_labels, val_files, val_labels):
    train_ds = Dataset.from_tensor_slices((train_files.values, train_labels.values))
    train_ds = train_ds.shuffle(
        len(train_files), seed=SEED, reshuffle_each_iteration=True
    )
    train_ds = train_ds.map(fetch_image_with_aug, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
    train_ds = train_ds.prefetch(AUTOTUNE)

    val_ds = Dataset.from_tensor_slices((val_files.values, val_labels.values))
    val_ds = val_ds.map(fetch_image_without_aug, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.cache()
    val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False)
    val_ds = val_ds.prefetch(AUTOTUNE)

    train_ds = train_ds.with_options(DATASET_OPTIONS)
    val_ds = val_ds.with_options(DATASET_OPTIONS)

    return train_ds, val_ds




## === cell 6
def scheduler(epoch, lr):
    if epoch < 8:
        return lr
    else:
        return lr * np.exp(-0.05)




## === cell 7
def create_callbacks(Folder_name):
    os.makedirs("Weights", exist_ok=True)
    os.makedirs(os.path.join("Weights", "logs"), exist_ok=True)
    os.makedirs(os.path.join("Weights", Folder_name), exist_ok=True)
    os.makedirs(os.path.join("Weights", "logs", Folder_name), exist_ok=True)

    lr_scheduler = LearningRateScheduler(scheduler)

    weight_save = ModelCheckpoint(
        os.path.join("Weights", Folder_name, "best_model.keras"),
        monitor="val_accuracy",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
    )

    tensorboard = TensorBoard(
        os.path.join("Weights", "logs", Folder_name),
        histogram_freq=0,
        write_graph=False,
        update_freq="epoch",
    )

    callbacks = [lr_scheduler, weight_save, tensorboard]
    histories = []
    return callbacks, histories




## === cell 8
def create_model(training=True, weights="imagenet"):
    base_model = EfficientNetB3(
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
train_csv["image_id"] = train_csv["image_id"].map(lambda x: train_location + x)

Folder_name_base = "Exp_"
create_k_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

train_tfrec_files = sorted(glob.glob(train_tfrecords_glob))
if len(train_tfrec_files) == 0:
    raise FileNotFoundError(f"No train TFRecords found at {train_tfrecords_glob}")

fold_number = 1
all_histories = []

for train_index, test_index in create_k_folds.split(
    train_csv["image_id"], train_csv["label"]
):
    print(f"Fold Number {fold_number} is starting it's training")
    print("Creating Callbacks...")
    Folder_name = Folder_name_base + str(fold_number)
    callbacks, histories = create_callbacks(Folder_name)

    print("Importing Datasets...")
    X_train, X_test = (
        train_csv["image_id"].iloc[train_index],
        train_csv["image_id"].iloc[test_index],
    )
    y_train, y_test = (
        train_csv["label"].iloc[train_index],
        train_csv["label"].iloc[test_index],
    )

    train_ds, val_ds = getDatasetFromDataframe(X_train, y_train, X_test, y_test)

    final_model = create_model()

    final_model.compile(
        optimizer=RMSprop(learning_rate=1e-4),
        loss=SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    print("Starting training...")
    hist = final_model.fit(
        train_ds, epochs=15, validation_data=val_ds, callbacks=callbacks, verbose=2
    )
    histories.append(hist.history)
    all_histories.append(hist.history)

    fold_number += 1

print("Training complete. Folds trained:", fold_number - 1)




## === cell 10
def load_trained_models():
    Models = []

    local_model_ckpts = sorted(
        glob.glob(os.path.join("Weights", "Exp_*", "best_model.keras"))
    )

    for model_path in local_model_ckpts:
        m = tf.keras.models.load_model(model_path, compile=False)
        Models.append(m)

    return Models


Models = load_trained_models()
if len(Models) == 0:
    raise FileNotFoundError(
        "No model checkpoints found. Expected Weights/Exp_*/best_model.keras (from training)."
    )

print(f"Models loaded: {len(Models)}")

sub_df = pd.read_csv(sample_sub_loc)

test_tfrec_files = sorted(glob.glob(test_tfrecords_glob))
if len(test_tfrec_files) == 0:
    test_files = (
        sub_df["image_id"].apply(lambda x: os.path.join(test_location, x)).values
    )

    def fetch_test_image(filename):
        img = tf.io.read_file(filename)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [ROW, COL])
        img = preprocess_input(img)
        return img

    test_ds = Dataset.from_tensor_slices(test_files)
    test_ds = test_ds.map(fetch_test_image, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False)
    test_ds = test_ds.prefetch(AUTOTUNE).with_options(DATASET_OPTIONS)
else:
    _TEST_FEATURE_DESC = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }

    def _decode_test(example_proto):
        ex = tf.io.parse_single_example(example_proto, _TEST_FEATURE_DESC)
        img = tf.image.decode_jpeg(ex["image"], channels=3)
        img = tf.image.resize(img, [ROW, COL])
        img = preprocess_input(img)
        return img, ex["image_name"]

    test_ds = tf.data.TFRecordDataset(test_tfrec_files, num_parallel_reads=AUTOTUNE)
    test_ds = test_ds.map(_decode_test, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False)
    test_ds = test_ds.prefetch(AUTOTUNE).with_options(DATASET_OPTIONS)

num_models = float(len(Models))


@tf.function(reduce_retracing=True)
def ensemble_predict_batch(batch):
    s = 0.0
    for m in Models:
        s += m(batch, training=False)
    return s / num_models


if len(test_tfrec_files) == 0:
    results = []
    for batch in test_ds:
        probs = ensemble_predict_batch(batch)
        preds = tf.argmax(probs, axis=1, output_type=tf.int32)
        results.extend(preds.numpy().tolist())
else:
    name_to_pred = {}
    for batch_imgs, batch_names in test_ds:
        probs = ensemble_predict_batch(batch_imgs)
        preds = tf.argmax(probs, axis=1, output_type=tf.int32).numpy()
        names = batch_names.numpy()
        for n, p in zip(names, preds):
            name_to_pred[n.decode("utf-8")] = int(p)

    results = [name_to_pred[img_id] for img_id in sub_df["image_id"].tolist()]

print("Inference complete. Predictions:", len(results))



## === cell 11
submission = pd.DataFrame(
    {"image_id": pd.read_csv(sample_sub_loc)["image_id"], "label": results}
)
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Saved to:", os.path.abspath("submission.csv"))
