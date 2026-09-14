# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.8916591115140526

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'We fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation before importing TensorFlow (this removes the `MessageFactory.GetPrototype` error and lets the notebook run end-to-end). Then we fix the invalid submission by ensuring `image_id` values exactly match the competition’s expected test order: we join predictions onto `sample_submission.csv` by `image_id` and write labels in that exact row order (this removes any mismatch caused by TFRecord reading order, missing/blank ids, or sorting). We also add a small safety check to drop any empty `image_id` records coming from TFRecords (score-neutral, correctness-only). No model/training logic is changed.'
- What this solution (achieved 0.05531) has done: 'We fix the immediate runtime crash by pinning protobuf to the pure-Python backend *before any TensorFlow import*, and also setting the protobuf implementation version to avoid TF/protobuf API mismatches that cause `MessageFactory.GetPrototype` failures. Then we correct a major label-mapping logic bug: the model already outputs the correct numeric class indices (0–4) from TFRecords, but the current code remaps those indices through disease names and a `LabelEncoder`, which can scramble labels and destroy accuracy. Finally, we keep the rest of the pipeline identical (same TFRecord reading, same EfficientNetB0 head, same training loop) and still align predictions to `sample_submission.csv` order to guarantee a valid submission file.'
- What this solution (achieved 0.05531) has done: 'You’re crashing before training due to a TensorFlow/protobuf incompatibility that isn’t fully solved just by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION`; we force the pure-Python protobuf earlier and monkey-patch the missing `MessageFactory.GetPrototype` method (aliasing it to `GetMessageClass`) before importing TensorFlow. Then, to fix the extremely low accuracy (0.055) without changing the model/training approach, we remove the incorrect label remapping via disease names/LabelEncoder and instead train on the original numeric labels (0–4) that the TFRecords already contain. Finally, we keep the same TFRecord pipeline and EfficientNetB0 head, and still write `submission.csv` by merging predictions onto `sample_submission.csv` to guarantee correct row order and a valid file.'

# 9. Code solution

## === cell 0
import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):
        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception as e:
    print("protobuf patch skipped:", repr(e))

import tensorflow as tf

print("TF version:", tf.__version__)

try:
    pass
except Exception as e:
    print("Thread config skipped:", repr(e))




## === cell 1
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)
train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

train_csv["label"] = train_csv["label"].astype(int)
train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"]
)

train, valid = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=42
)

TFREC_TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords"
TFREC_TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords"

train_tfrec_files = sorted(
    [
        os.path.join(TFREC_TRAIN_DIR, f)
        for f in os.listdir(TFREC_TRAIN_DIR)
        if f.endswith(".tfrec") or f.endswith(".tfrecord") or f.endswith(".tfrecords")
    ]
)
test_tfrec_files = sorted(
    [
        os.path.join(TFREC_TEST_DIR, f)
        for f in os.listdir(TFREC_TEST_DIR)
        if f.endswith(".tfrec") or f.endswith(".tfrecord") or f.endswith(".tfrecords")
    ]
)

num_classes = int(train_csv["label"].nunique())
assert num_classes == 5, f"Unexpected num_classes={num_classes}; expected 5"
print("num_classes:", num_classes)

GLOBAL_SEED = 42


def _parse_example(example_proto, labeled=True):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    }
    ex = tf.io.parse_single_example(example_proto, feature_description)
    img = tf.io.decode_jpeg(ex["image"], channels=3)
    return img, ex


IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def _preprocess(img):
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img


def _stateless_augment(img, image_id):
    h = tf.strings.to_hash_bucket_fast(image_id, 2**31 - 1)
    seed = tf.stack([tf.constant(GLOBAL_SEED, tf.int32), tf.cast(h, tf.int32)], axis=0)

    img = tf.image.stateless_random_flip_left_right(img, seed=seed)
    img = tf.image.stateless_random_flip_up_down(
        img, seed=seed + tf.constant([1, 1], tf.int32)
    )

    angle = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([2, 2], tf.int32), minval=-45.0, maxval=45.0
    ) * (np.pi / 180.0)

    dx = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([3, 3], tf.int32), minval=-0.2, maxval=0.2
    ) * tf.cast(IMG_SIZE[1], tf.float32)
    dy = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([4, 4], tf.int32), minval=-0.2, maxval=0.2
    ) * tf.cast(IMG_SIZE[0], tf.float32)

    zoom = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([5, 5], tf.int32), minval=0.8, maxval=1.2
    )

    shear = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([6, 6], tf.int32), minval=-0.2, maxval=0.2
    )

    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)

    a0 = zoom * cos_a
    a1 = zoom * (-sin_a + shear)
    b0 = zoom * sin_a
    b1 = zoom * cos_a

    cx = tf.cast(IMG_SIZE[1], tf.float32) / 2.0
    cy = tf.cast(IMG_SIZE[0], tf.float32) / 2.0

    tx = cx - (a0 * cx + a1 * cy) + dx
    ty = cy - (b0 * cx + b1 * cy) + dy

    transform = tf.stack([a0, a1, tx, b0, b1, ty, 0.0, 0.0])[None, :]
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=transform,
        output_shape=tf.constant([IMG_SIZE[0], IMG_SIZE[1]], tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]
    return img


train_ids = set(train["image_id"].tolist())
valid_ids = set(valid["image_id"].tolist())

train_ids_tf = tf.constant(sorted(list(train_ids)))
valid_ids_tf = tf.constant(sorted(list(valid_ids)))
train_ids_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        train_ids_tf, tf.ones_like(train_ids_tf, dtype=tf.int32)
    ),
    default_value=0,
)
valid_ids_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        valid_ids_tf, tf.ones_like(valid_ids_tf, dtype=tf.int32)
    ),
    default_value=0,
)


def _is_in_train(ex):
    return tf.equal(train_ids_table.lookup(ex["image_id"]), 1)


def _is_in_valid(ex):
    return tf.equal(valid_ids_table.lookup(ex["image_id"]), 1)


def _label_from_ex(ex):
    lbl = tf.where(ex["label"] >= 0, ex["label"], ex["target"])
    return tf.cast(lbl, tf.int32)


def make_ds(files, subset):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(lambda x: _parse_example(x), num_parallel_calls=tf.data.AUTOTUNE)

    if subset == "train":
        ds = ds.filter(lambda img, ex: _is_in_train(ex))
    elif subset == "valid":
        ds = ds.filter(lambda img, ex: _is_in_valid(ex))

    def _map_labeled(img, ex, training):
        image_id = ex["image_id"]
        img = _preprocess(img)
        if training:
            img = _stateless_augment(img, image_id)
        y = tf.one_hot(_label_from_ex(ex), depth=num_classes, dtype=tf.float32)
        return img, y

    ds = ds.map(
        lambda img, ex: _map_labeled(img, ex, subset == "train"),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if subset == "train":
        ds = ds.shuffle(8192, seed=GLOBAL_SEED, reshuffle_each_iteration=True)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.cache()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_generator = make_ds(train_tfrec_files, "train")
valid_generator = make_ds(train_tfrec_files, "valid")




## === cell 2
tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception as e:
    print("Determinism setting skipped:", repr(e))

base_model = EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
base_model.trainable = False  # stable + faster; avoids changing approach drastically

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.2)(x)
outputs = Dense(num_classes, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## === cell 3
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




## === cell 4
EPOCHS = 8  # conservative to fit runtime; early stopping may stop earlier

steps_per_epoch = int(np.ceil(len(train) / BATCH_SIZE))
validation_steps = int(np.ceil(len(valid) / BATCH_SIZE))

history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[early_stopping, learning_rate_reduction, EarlyStoppingCallback()],
    verbose=1,
)




## === cell 5
import numpy as np
import pandas as pd

img_size = (224, 224)
BATCH_SIZE = 32


def make_test_ds(files):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(
        lambda x: _parse_example(x, labeled=False), num_parallel_calls=tf.data.AUTOTUNE
    )

    def _map(img, ex):
        img = _preprocess(img)
        return img, ex["image_id"]

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


test_ds = make_test_ds(test_tfrec_files)

all_preds = []
all_ids = []
for batch_imgs, batch_ids in test_ds:
    batch_pred = model(batch_imgs, training=False).numpy()
    all_preds.append(batch_pred)
    all_ids.append(batch_ids.numpy())

preds = np.concatenate(all_preds, axis=0)
image_ids = np.concatenate(all_ids, axis=0)

image_ids = [x.decode("utf-8") for x in image_ids.astype("S").tolist()]
pred_idx = np.argmax(preds, axis=1).astype(int)

pred_df = pd.DataFrame({"image_id": image_ids, "label": pred_idx})
pred_df = pred_df[pred_df["image_id"].astype(str).str.len() > 0].copy()

sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

merged = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

missing_pred = merged["label"].isna().sum()
if missing_pred:
    print("WARNING: missing predictions for", int(missing_pred), "rows; filling with 0")
    merged["label"] = merged["label"].fillna(0).astype(int)
else:
    merged["label"] = merged["label"].astype(int)

submission_df = merged[["image_id", "label"]].copy()
assert submission_df.shape[0] == sample_sub.shape[0], (
    submission_df.shape,
    sample_sub.shape,
)
assert (
    submission_df["image_id"].tolist() == sample_sub["image_id"].tolist()
), "image_id order mismatch"
assert submission_df.columns.tolist() == ["image_id", "label"]

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Submission file created:", submission_path)
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.shape[1])
print("Label value counts:\n", submission_df["label"].value_counts().sort_index())
