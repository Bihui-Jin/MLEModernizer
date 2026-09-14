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

# 5. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

try:
    keras.utils.set_random_seed(SEED)
except Exception:
    tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = f"{BASE_PATH}/train.csv"
SAMPLE_SUB = f"{BASE_PATH}/sample_submission.csv"
TRAIN_IMG_DIR = f"{BASE_PATH}/train_images"
TEST_IMG_DIR = f"{BASE_PATH}/test_images"

print("TensorFlow:", tf.__version__)
print(
    "Train exists:",
    os.path.exists(TRAIN_CSV),
    "Test images exists:",
    os.path.exists(TEST_IMG_DIR),
)




## === cell 1
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5

train_df = pd.read_csv(TRAIN_CSV)
train_df["label"] = train_df["label"].astype(int)
train_df["image_id"] = train_df["image_id"].astype(str)
train_df["path"] = TRAIN_IMG_DIR + "/" + train_df["image_id"]

train_split, valid_split = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df["label"],
    random_state=SEED,
)


@tf.function
def _decode_resize_jpeg(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)  # fixed JPEG decoder, faster
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)
    return img


@tf.function
def _apply_keras_preprocess(img):
    return preprocess_input(img)


_rot_layer = keras.layers.RandomRotation(
    factor=45.0 / 180.0, fill_mode="nearest", seed=SEED
)
_trans_layer = keras.layers.RandomTranslation(
    height_factor=0.2, width_factor=0.2, fill_mode="nearest", seed=SEED
)
_zoom_layer = keras.layers.RandomZoom(
    height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), fill_mode="nearest", seed=SEED
)


@tf.function
def _augment_image(img, seed_pair):
    seed_pair = tf.cast(seed_pair, tf.int32)
    img = tf.image.stateless_random_flip_left_right(img, seed_pair)

    seed2 = tf.bitwise.bitwise_xor(seed_pair, tf.constant([1, 0], dtype=tf.int32))
    img = tf.image.stateless_random_flip_up_down(img, seed2)

    img = _rot_layer(img, training=True)
    img = _trans_layer(img, training=True)
    img = _zoom_layer(img, training=True)

    seed7 = tf.bitwise.bitwise_xor(seed_pair, tf.constant([6, 0], dtype=tf.int32))
    shear = tf.random.stateless_uniform(
        [],
        seed7,
        minval=-0.2,
        maxval=0.2,
        dtype=tf.float32,
    )

    cx = (IMG_SIZE[1] - 1) / 2.0
    cy = (IMG_SIZE[0] - 1) / 2.0

    z = 1.0
    sh = -shear  # inverse approx for output->input
    a0 = z
    a1 = sh
    b0 = 0.0
    b1 = z

    a2 = cx - a0 * cx - a1 * cy
    b2 = cy - b0 * cx - b1 * cy

    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[None, :]
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=transform,
        output_shape=tf.constant([IMG_SIZE[0], IMG_SIZE[1]], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0,
    )[0]
    return img


def make_dataset(df, training: bool):
    paths = df["path"].to_numpy(dtype=str)
    labels = df["label"].to_numpy(dtype=np.int32)
    sample_ids = np.arange(len(df), dtype=np.int64)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels, sample_ids))

    if training:
        shuffle_buf = int(min(len(df), 4096))
        ds = ds.shuffle(
            buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
        )

    @tf.function
    def _pipeline(path, label, sid):
        img = _decode_resize_jpeg(path)
        if training:
            base_seed = tf.constant([SEED, SEED], dtype=tf.int32)
            seed_pair = tf.random.experimental.stateless_fold_in(
                base_seed, tf.cast(sid, tf.int64)
            )
            seed_pair = tf.cast(seed_pair, tf.int32)
            img = _augment_image(img, seed_pair)
        img = _apply_keras_preprocess(img)
        y = tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)
        return img, y

    ds = ds.map(_pipeline, num_parallel_calls=AUTOTUNE, deterministic=True)

    if not training:
        ds = ds.cache()

    opts = tf.data.Options()
    opts.deterministic = True
    opts.experimental_optimization.map_and_batch_fusion = True
    opts.experimental_optimization.parallel_batch = True
    ds = ds.with_options(opts)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(train_split, training=True)
valid_ds = make_dataset(valid_split, training=False)

print("Prepared tf.data datasets:")
print("Train batches:", int(np.ceil(len(train_split) / BATCH_SIZE)))
print("Valid batches:", int(np.ceil(len(valid_split) / BATCH_SIZE)))




## === cell 2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
)

learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    patience=2,
    factor=0.5,
    min_lr=1e-6,
    verbose=1,
)

USE_EXTERNAL_ENSEMBLE = False
external_model_paths = {
    "cropnet": "/kaggle/input/cropnet_model/tensorflow2/default/1/kaggle/working/model_feature_extraction_tf",
    "densenet": "/kaggle/input/densenet_model/tensorflow2/default/1/kaggle/working/kaggle/working/densenet_model_tf",
    "efficientnetb4": "/kaggle/input/efficientnetb4_model/tensorflow2/default/1/kaggle/working/kaggle/working/efficientnet_model_tf",
}
if all(os.path.exists(p) for p in external_model_paths.values()):
    USE_EXTERNAL_ENSEMBLE = True

print("USE_EXTERNAL_ENSEMBLE =", USE_EXTERNAL_ENSEMBLE)

inp = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inp)
x = GlobalAveragePooling2D()(base.output)
out = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=inp, outputs=out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## === cell 3
EPOCHS = 8  # keep identical

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, learning_rate_reduction],
    verbose=1,
)




## === cell 4
sample_sub = pd.read_csv(SAMPLE_SUB)
sample_sub["image_id"] = sample_sub["image_id"].astype(str)
sample_sub["path"] = TEST_IMG_DIR + "/" + sample_sub["image_id"]
test_paths = sample_sub["path"].to_numpy(dtype=str)

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)


@tf.function
def _test_pipeline(path):
    img = _decode_resize_jpeg(path)
    img = _apply_keras_preprocess(img)
    return img


test_ds = test_ds.map(
    _test_pipeline, num_parallel_calls=AUTOTUNE, deterministic=True
).cache()

opts = tf.data.Options()
opts.deterministic = True
opts.experimental_optimization.map_and_batch_fusion = True
opts.experimental_optimization.parallel_batch = True
test_ds = test_ds.with_options(opts)

test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

probs = model.predict(test_ds, verbose=1)
preds = np.argmax(probs, axis=1).astype(int)

submission_df = pd.DataFrame(
    {
        "image_id": sample_sub["image_id"].values,
        "label": preds,
    }
)

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission_df.head())
print("Submission shape:", submission_df.shape)
assert submission_df.shape[0] == sample_sub.shape[0]
assert list(submission_df.columns) == ["image_id", "label"]
assert out_path.endswith(".csv")
