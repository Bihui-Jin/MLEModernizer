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
import random
import numpy as np
import pandas as pd



## === cell 1
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import tensorflow as tf
from math import ceil
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint



## === cell 2
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

path = "/kaggle/input/cassava-leaf-disease-classification/"
train_csv_path = os.path.join(path, "train.csv")
sample_sub_path = os.path.join(path, "sample_submission.csv")
train_img_dir = os.path.join(path, "train_images") + "/"
test_img_dir = os.path.join(path, "test_images") + "/"

train_df = pd.read_csv(train_csv_path)
sample_submission = pd.read_csv(sample_sub_path)

print("train_df:", train_df.shape, "sample_submission:", sample_submission.shape)
print(
    "train images dir exists:",
    os.path.isdir(train_img_dir),
    "test images dir exists:",
    os.path.isdir(test_img_dir),
)



## === cell 3
gm_exp = tf.Variable(3.0, dtype=tf.float32, trainable=True, name="gm_exp")


def generalized_mean_pool_2d(X):
    pool = (
        tf.reduce_mean(tf.abs(X ** (gm_exp)), axis=[1, 2], keepdims=False) + 1.0e-7
    ) ** (1.0 / gm_exp)
    return pool


def create_model(input_shape):
    inp = Input(shape=input_shape)

    x_model = ResNet50(
        weights=None,  # preserve original (no ImageNet weights)
        include_top=False,
        input_tensor=inp,
        pooling=None,
        classes=None,
    )
    for layer in x_model.layers:
        layer.trainable = True

    lambda_layer = Lambda(generalized_mean_pool_2d, name="gem_pool")
    lambda_layer.trainable_weights.extend([gm_exp])
    x = lambda_layer(x_model.output)

    out = Dense(5, activation="softmax", name="plan_diseases")(x)
    model = Model(inputs=x_model.input, outputs=out)
    return model




## === cell 4
AUTOTUNE = tf.data.AUTOTUNE

_DATASET_OPTIONS = tf.data.Options()
_DATASET_OPTIONS.experimental_deterministic = True
try:
    _DATASET_OPTIONS.experimental_optimization.map_parallelization = True
    _DATASET_OPTIONS.experimental_optimization.parallel_batch = True
except Exception:
    pass


def _build_paths_and_labels(df, img_dir, indices=None):
    if indices is None:
        ids = df["image_id"].to_numpy(dtype=str, copy=False)
        labels = None
    else:
        ids = df.iloc[indices]["image_id"].to_numpy(dtype=str, copy=False)
        labels = df.iloc[indices]["label"].to_numpy(dtype=np.int32, copy=False)

    paths = np.char.add(img_dir, ids).astype(str, copy=False)
    return paths, labels


@tf.function
def _decode_resize_scale(path, img_h, img_w):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, (img_h, img_w), method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    img.set_shape([None, None, 3])
    return img


def make_train_dataset(
    df, img_dir, indices, batch_size, img_size=(224, 224, 3), shuffle=True, seed=SEED
):
    paths, labels = _build_paths_and_labels(df, img_dir, indices)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        buffer_size = min(len(paths), 4096)
        ds = ds.shuffle(
            buffer_size=buffer_size, seed=seed, reshuffle_each_iteration=True
        )

    img_h, img_w = int(img_size[0]), int(img_size[1])

    ds = ds.map(
        lambda p, y: (_decode_resize_scale(p, img_h, img_w), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.with_options(_DATASET_OPTIONS)
    return ds


def make_test_dataset(df, img_dir, batch_size, img_size=(224, 224, 3)):
    paths, _ = _build_paths_and_labels(df, img_dir, indices=None)
    ds = tf.data.Dataset.from_tensor_slices(paths)
    img_h, img_w = int(img_size[0]), int(img_size[1])

    ds = ds.map(
        lambda p: _decode_resize_scale(p, img_h, img_w),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.with_options(_DATASET_OPTIONS)
    return ds




## === cell 5
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
train_idx, val_idx = next(skf.split(train_df["image_id"], train_df["label"]))

print("Train size:", len(train_idx), "Val size:", len(val_idx))



## === cell 6
input_shape = (224, 224, 3)
model = create_model(input_shape)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

batch_size = 16

train_ds = make_train_dataset(
    train_df,
    train_img_dir,
    train_idx,
    batch_size=batch_size,
    img_size=input_shape,
    shuffle=True,
    seed=SEED,
)
val_ds = make_train_dataset(
    train_df,
    train_img_dir,
    val_idx,
    batch_size=batch_size,
    img_size=input_shape,
    shuffle=False,
    seed=SEED,
)

ckpt_path = "/kaggle/working/best.weights.h5"
ckpt = ModelCheckpoint(
    ckpt_path,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    callbacks=[ckpt],
    verbose=1,
)

if os.path.exists(ckpt_path):
    model.load_weights(ckpt_path)



## === cell 7
test_df = sample_submission[["image_id"]].copy()
test_ds = make_test_dataset(
    test_df, test_img_dir, batch_size=batch_size, img_size=input_shape
)

preds = model.predict(test_ds, verbose=1)
labels = np.argmax(preds, axis=1).astype(int)

submission = sample_submission.copy()
submission["label"] = labels
submission = submission[["image_id", "label"]]
submission["label"] = submission["label"].astype(int)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", submission.shape)
print(submission.head())
