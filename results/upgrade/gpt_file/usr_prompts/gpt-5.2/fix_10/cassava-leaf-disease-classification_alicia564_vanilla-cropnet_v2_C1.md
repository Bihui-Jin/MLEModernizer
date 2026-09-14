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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Could not enable full TF determinism in this environment:", repr(e))

print("TF version:", tf.__version__)



## === cell 1
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input

label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)

train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

train_csv["disease"] = train_csv["label"].map(label_to_disease).astype(str)
train_csv["label"] = train_csv["label"].astype(str)
train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"]
)

train, valid = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=SEED
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

class_names = sorted(train["disease"].unique().tolist())
num_classes = len(class_names)
class_to_index = {name: i for i, name in enumerate(class_names)}
print("Num classes:", num_classes)
print("Class indices (disease->index):", class_to_index)

train_paths = train["path"].to_numpy()
train_labels = train["disease"].map(class_to_index).to_numpy(dtype=np.int32)
valid_paths = valid["path"].to_numpy()
valid_labels = valid["disease"].map(class_to_index).to_numpy(dtype=np.int32)



@tf.function
def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img


@tf.function
def _stateless_seeds(seed, n):
    idx = tf.range(1, n + 1, dtype=tf.int32)  # [n]
    s0 = seed[0] ^ (idx * tf.constant(0x9E3779B9, tf.int32))
    s1 = seed[1] ^ (idx * tf.constant(0x85EBCA6B, tf.int32))
    return tf.stack([s0, s1], axis=1)  # [n, 2]


@tf.function
def _random_transform(img, seed):
    seed = tf.convert_to_tensor(seed, dtype=tf.int32)  # shape [2]
    seeds = _stateless_seeds(seed, 7)  # [7, 2]
    s1, s2, s3, s4, s5, s6, s7 = tf.unstack(seeds, axis=0)

    img = tf.image.stateless_random_flip_left_right(img, seed=s1)
    img = tf.image.stateless_random_flip_up_down(img, seed=s2)

    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)

    angle = tf.random.stateless_uniform((), s3, minval=-45.0, maxval=45.0) * (
        np.pi / 180.0
    )

    tx = tf.random.stateless_uniform((), s4, minval=-0.2, maxval=0.2) * h
    ty = tf.random.stateless_uniform((), s5, minval=-0.2, maxval=0.2) * w

    zoom = tf.random.stateless_uniform((), s6, minval=0.8, maxval=1.2)

    shear = tf.random.stateless_uniform((), s7, minval=-0.2, maxval=0.2)

    cx = (w - 1.0) / 2.0
    cy = (h - 1.0) / 2.0

    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)

    sh = tf.tan(shear)
    a00 = (cos_a + sin_a * sh) / zoom
    a01 = (-sin_a + cos_a * sh) / zoom
    a10 = (sin_a) / zoom
    a11 = (cos_a) / zoom

    a02 = cx - a00 * cx - a01 * cy + ty
    a12 = cy - a10 * cx - a11 * cy + tx

    transform = tf.stack([a00, a01, a02, a10, a11, a12, 0.0, 0.0])[tf.newaxis, :]

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[tf.newaxis, ...],
        transforms=transform,
        output_shape=tf.shape(img)[:2],
        fill_mode="NEAREST",
        interpolation="BILINEAR",
        fill_value=0.0,
    )[0]
    return img


@tf.function
def _load_train(idx, path, label):
    img = _decode_resize(path)
    seed = tf.stack([tf.cast(SEED, tf.int32), tf.cast(idx, tf.int32)])
    img = _random_transform(img, seed)
    y = tf.one_hot(label, depth=num_classes, dtype=tf.float32)
    return img, y


@tf.function
def _load_eval(path, label):
    img = _decode_resize(path)
    y = tf.one_hot(label, depth=num_classes, dtype=tf.float32)
    return img, y


def _make_ds(paths, labels, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(
            buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True
        )
        ds = ds.enumerate()
        ds = ds.map(
            lambda idx, pl: _load_train(idx, pl[0], pl[1]),
            num_parallel_calls=AUTOTUNE,
            deterministic=False,
        )
    else:
        ds = ds.map(_load_eval, num_parallel_calls=AUTOTUNE, deterministic=True)
        ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False, deterministic=True)
    ds = ds.prefetch(AUTOTUNE)

    opts = tf.data.Options()
    if training:
        opts.experimental_deterministic = False
    else:
        opts.experimental_deterministic = True
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.map_and_batch_fusion = True
    opts.experimental_optimization.parallel_batch = True
    ds = ds.with_options(opts)
    return ds


train_generator = _make_ds(train_paths, train_labels, training=True)
valid_generator = _make_ds(valid_paths, valid_labels, training=False)



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
    monitor="val_loss",
    patience=2,
    factor=0.5,
    min_lr=1e-7,
    verbose=1,
)



## === cell 3
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)
base_model.trainable = False  # keep minimal/fast and stable

inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
outputs = Dense(num_classes, activation="softmax")(x)
model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()



## === cell 4
EPOCHS = 5

history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    callbacks=[early_stopping, learning_rate_reduction, EarlyStoppingCallback()],
    verbose=1,
)



## === cell 5
val_loss, val_acc = model.evaluate(valid_generator, verbose=0)
print(f"Validation accuracy (generator): {val_acc:.4f}")



## === cell 6
from tensorflow.keras.layers import TFSMLayer

cropnet_model = None
cropnet_model_path = "/kaggle/input/vanilla_kaggle_cropnet/tensorflow2/default/1/kaggle/working/cropnet_model_tf"

if os.path.isdir(cropnet_model_path) and (
    os.path.exists(os.path.join(cropnet_model_path, "saved_model.pb"))
    or os.path.exists(os.path.join(cropnet_model_path, "saved_model.pbtxt"))
):
    try:
        layer = TFSMLayer(cropnet_model_path, call_endpoint="serving_default")
        input_layer = Input(shape=(224, 224, 3))
        output_layer = layer(input_layer)
        cropnet_model = Model(inputs=input_layer, outputs=output_layer)
        print("Loaded CropNet SavedModel successfully.")
    except Exception as e:
        print(
            "Failed to load CropNet model; falling back to EfficientNet. Error:",
            repr(e),
        )
        cropnet_model = None
else:
    print("CropNet SavedModel not found at path; falling back to EfficientNet.")
    cropnet_model = None



## === cell 7
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
sample_sub = pd.read_csv(sample_sub_path)

image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
img_size = (224, 224)



@tf.function
def _load_test_image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, img_size, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img


test_paths = tf.constant(
    [os.path.join(image_dir, f) for f in sample_sub["image_id"].tolist()],
    dtype=tf.string,
)

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(_load_test_image, num_parallel_calls=AUTOTUNE, deterministic=True)
test_ds = test_ds.cache()
test_ds = test_ds.batch(64, drop_remainder=False, deterministic=True).prefetch(AUTOTUNE)

opts = tf.data.Options()
opts.experimental_deterministic = True
opts.experimental_optimization.map_parallelization = True
opts.experimental_optimization.map_and_batch_fusion = True
opts.experimental_optimization.parallel_batch = True
test_ds = test_ds.with_options(opts)

if cropnet_model is not None:
    preds = cropnet_model.predict(test_ds, verbose=0)
    if isinstance(preds, dict):
        preds = next(iter(preds.values()))
else:
    preds = model.predict(test_ds, verbose=0)

predictions = np.argmax(np.asarray(preds), axis=1).astype(int).tolist()

submission_df = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": predictions}
)
submission_df["label"] = submission_df["label"].astype(int)

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)

print("Submission file created:", out_path)
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.shape[1])
assert out_path.endswith(".csv")
assert list(submission_df.columns) == ["image_id", "label"]
assert len(submission_df) == len(sample_sub)
