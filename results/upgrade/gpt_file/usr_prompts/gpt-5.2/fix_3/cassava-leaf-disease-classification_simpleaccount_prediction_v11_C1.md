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

0.5480507706255666

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



## === cell 1
import cv2
import tensorflow as tf
from math import ceil
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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


def _build_paths_and_labels(df, img_dir, indices=None):
    if indices is None:
        ids = df["image_id"].to_numpy()
        labels = None
    else:
        ids = df.loc[indices, "image_id"].to_numpy()
        labels = df.loc[indices, "label"].to_numpy(dtype=np.int64)
    paths = np.char.add(img_dir, ids).astype(str)
    return paths, labels


def _decode_resize_scale(path, img_size_hw):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, img_size_hw, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


def make_train_dataset(
    df, img_dir, indices, batch_size, img_size=(224, 224, 3), shuffle=True, seed=SEED
):
    paths, labels = _build_paths_and_labels(df, img_dir, indices)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True
        )
    img_size_hw = (img_size[0], img_size[1])
    ds = ds.map(
        lambda p, y: (_decode_resize_scale(p, img_size_hw), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_dataset(df, img_dir, batch_size, img_size=(224, 224, 3)):
    paths, _ = _build_paths_and_labels(df, img_dir, indices=None)
    ds = tf.data.Dataset.from_tensor_slices(paths)
    img_size_hw = (img_size[0], img_size[1])
    ds = ds.map(
        lambda p: _decode_resize_scale(p, img_size_hw),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
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



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2706749208.py in <cell line: 0>()
     11 
     12 # Use tf.data datasets (faster) with identical preprocessing semantics.
---> 13 train_ds = make_train_dataset(
     14     train_df,
     15     train_img_dir,

/tmp/ipykernel_11/4257517079.py in make_train_dataset(df, img_dir, indices, batch_size, img_size, shuffle, seed)
     30     df, img_dir, indices, batch_size, img_size=(224, 224, 3), shuffle=True, seed=SEED
     31 ):
---> 32     paths, labels = _build_paths_and_labels(df, img_dir, indices)
     33     ds = tf.data.Dataset.from_tensor_slices((paths, labels))
     34     if shuffle:

/tmp/ipykernel_11/4257517079.py in _build_paths_and_labels(df, img_dir, indices)
     12         ids = df.loc[indices, "image_id"].to_numpy()
     13         labels = df.loc[indices, "label"].to_numpy(dtype=np.int64)
---> 14     paths = np.char.add(img_dir, ids).astype(str)
     15     return paths, labels
     16 

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in add(x1, x2)
    330         # object dtype itemsize as num chars (worked on short strings).
    331         # bytes + void worked but promoting void->bytes is dubious also.
--> 332         raise TypeError(
    333             "np.char.add() requires both arrays of the same dtype kind, but "
    334             f"got dtypes: '{arr1.dtype}' and '{arr2.dtype}' (the few cases "

TypeError: np.char.add() requires both arrays of the same dtype kind, but got dtypes: '<U63' and 'object' (the few cases where this used to work often lead to incorrect results).

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

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4252662269.py in <cell line: 0>()
      1 test_df = sample_submission[["image_id"]].copy()
----> 2 test_ds = make_test_dataset(
      3     test_df, test_img_dir, batch_size=batch_size, img_size=input_shape
      4 )
      5 

/tmp/ipykernel_11/4257517079.py in make_test_dataset(df, img_dir, batch_size, img_size)
     49 
     50 def make_test_dataset(df, img_dir, batch_size, img_size=(224, 224, 3)):
---> 51     paths, _ = _build_paths_and_labels(df, img_dir, indices=None)
     52     ds = tf.data.Dataset.from_tensor_slices(paths)
     53     img_size_hw = (img_size[0], img_size[1])

/tmp/ipykernel_11/4257517079.py in _build_paths_and_labels(df, img_dir, indices)
     12         ids = df.loc[indices, "image_id"].to_numpy()
     13         labels = df.loc[indices, "label"].to_numpy(dtype=np.int64)
---> 14     paths = np.char.add(img_dir, ids).astype(str)
     15     return paths, labels
     16 

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in add(x1, x2)
    330         # object dtype itemsize as num chars (worked on short strings).
    331         # bytes + void worked but promoting void->bytes is dubious also.
--> 332         raise TypeError(
    333             "np.char.add() requires both arrays of the same dtype kind, but "
    334             f"got dtypes: '{arr1.dtype}' and '{arr2.dtype}' (the few cases "

TypeError: np.char.add() requires both arrays of the same dtype kind, but got dtypes: '<U62' and 'object' (the few cases where this used to work often lead to incorrect results).
