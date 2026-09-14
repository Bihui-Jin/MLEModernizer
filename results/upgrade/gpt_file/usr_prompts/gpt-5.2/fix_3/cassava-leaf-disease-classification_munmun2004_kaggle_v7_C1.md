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
import json
import random
import warnings
import math
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB3

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

print("TensorFlow:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("Could not enable XLA JIT:", repr(e))
try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Could not enable op determinism:", repr(e))




## === cell 1
work_dir = "../input/cassava-leaf-disease-classification/"
train_path = os.path.join(work_dir, "train_images")
test_path = os.path.join(work_dir, "test_images")

print("work_dir exists:", os.path.exists(work_dir))
print("train_path exists:", os.path.exists(train_path))
print("test_path exists:", os.path.exists(test_path))




## === cell 2
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


seed = 21
seed_everything(seed)




## === cell 3
data = pd.read_csv(os.path.join(work_dir, "train.csv"))
print("train.csv shape:", data.shape)
print(data["label"].value_counts())




## === cell 4
with open(os.path.join(work_dir, "label_num_to_disease_map.json"), "r") as f:
    real_labels = json.load(f)
real_labels = {int(k): v for k, v in real_labels.items()}

data["class_name"] = data["label"].map(real_labels)
print(data[["image_id", "label", "class_name"]].head())




## === cell 5
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    data,
    test_size=0.05,
    random_state=42,
    stratify=data["label"],
)

print("train/val sizes:", train_df.shape, val_df.shape)




## === cell 6
IMG_SIZE = 456
size = (IMG_SIZE, IMG_SIZE)
n_CLASS = 5
BATCH_SIZE = 15
EPOCHS = 20

AUTOTUNE = tf.data.AUTOTUNE

class_names_sorted = sorted(data["class_name"].unique().tolist())
class_name_to_idx = {name: i for i, name in enumerate(class_names_sorted)}
idx_to_class_name = {i: name for name, i in class_name_to_idx.items()}
print("Class mapping (class_name -> index):", class_name_to_idx)




## === cell 7


def _decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )  # interpolation="nearest"
    img = tf.cast(img, tf.float32)
    return img


def _effnet_preprocess(img):
    return tf.keras.applications.efficientnet.preprocess_input(img)


def _augment(img, seed_pair):
    s0 = seed_pair
    s1 = seed_pair + tf.constant([1, 0], dtype=tf.int32)
    s2 = seed_pair + tf.constant([2, 0], dtype=tf.int32)
    s3 = seed_pair + tf.constant([3, 0], dtype=tf.int32)
    s4 = seed_pair + tf.constant([4, 0], dtype=tf.int32)
    s5 = seed_pair + tf.constant([5, 0], dtype=tf.int32)
    s6 = seed_pair + tf.constant([6, 0], dtype=tf.int32)

    img = tf.image.stateless_random_flip_left_right(img, seed=s0)
    img = tf.image.stateless_random_flip_up_down(img, seed=s1)

    angle = tf.random.stateless_uniform([], seed=s2, minval=-40.0, maxval=40.0) * (
        math.pi / 180.0
    )

    scale = tf.random.stateless_uniform([], seed=s3, minval=0.8, maxval=1.2)

    tx = tf.random.stateless_uniform([], seed=s4, minval=-0.2, maxval=0.2) * tf.cast(
        IMG_SIZE, tf.float32
    )
    ty = tf.random.stateless_uniform([], seed=s5, minval=-0.2, maxval=0.2) * tf.cast(
        IMG_SIZE, tf.float32
    )

    shear = tf.random.stateless_uniform([], seed=s6, minval=-0.2, maxval=0.2)

    cx = (tf.cast(IMG_SIZE, tf.float32) - 1.0) / 2.0
    cy = (tf.cast(IMG_SIZE, tf.float32) - 1.0) / 2.0

    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)

    a0 = scale * (cos_a + shear * sin_a)
    a1 = scale * (-sin_a + shear * cos_a)
    a2 = tf.zeros([], tf.float32)

    b0 = scale * (sin_a)
    b1 = scale * (cos_a)
    b2 = tf.zeros([], tf.float32)

    det = a0 * b1 - a1 * b0
    inv_a0 = b1 / det
    inv_a1 = -a1 / det
    inv_b0 = -b0 / det
    inv_b1 = a0 / det

    t_x = cx + tx - (a0 * cx + a1 * cy)
    t_y = cy + ty - (b0 * cx + b1 * cy)

    c = -(inv_a0 * t_x + inv_a1 * t_y)
    f = -(inv_b0 * t_x + inv_b1 * t_y)

    transform = tf.stack([inv_a0, inv_a1, c, inv_b0, inv_b1, f, 0.0, 0.0])[None, :]

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=transform,
        output_shape=tf.constant([IMG_SIZE, IMG_SIZE], dtype=tf.int32),
        interpolation="NEAREST",
        fill_mode="NEAREST",  # fill_mode="nearest"
        fill_value=0.0,
    )[0]
    return img


def make_dataset(df, directory, training):
    paths = (directory.rstrip("/") + "/" + df["image_id"].astype(str)).values
    path_ds = tf.data.Dataset.from_tensor_slices(paths)

    if training:
        labels = df["class_name"].map(class_name_to_idx).astype(np.int32).values
        y_ds = tf.data.Dataset.from_tensor_slices(labels)
        ds = tf.data.Dataset.zip((path_ds, y_ds))
    else:
        ds = path_ds

    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)

    if training:
        ds = ds.shuffle(
            buffer_size=min(len(df), 4096), seed=42, reshuffle_each_iteration=True
        )

        seed_ds = tf.data.Dataset.range(len(df)).map(
            lambda i: tf.stack([tf.cast(seed, tf.int32), tf.cast(i, tf.int32)]),
            num_parallel_calls=AUTOTUNE,
        )
        ds = tf.data.Dataset.zip((ds, seed_ds))

        def _load_aug(pair, s):
            path, y = pair
            img = _decode_and_resize(path)
            img = _augment(img, s)
            img = _effnet_preprocess(img)
            y = tf.one_hot(y, depth=n_CLASS, dtype=tf.float32)
            return img, y

        ds = ds.map(_load_aug, num_parallel_calls=AUTOTUNE)
    else:

        def _load(path):
            img = _decode_and_resize(path)
            img = _effnet_preprocess(img)
            return img

        ds = ds.map(_load, num_parallel_calls=AUTOTUNE)

    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(train_df, train_path, training=True)

val_paths = (train_path.rstrip("/") + "/" + val_df["image_id"].astype(str)).values
val_labels = val_df["class_name"].map(class_name_to_idx).astype(np.int32).values
val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
options = tf.data.Options()
options.experimental_deterministic = True
val_ds = val_ds.with_options(options)


def _val_load(path, y):
    img = _decode_and_resize(path)
    img = _effnet_preprocess(img)
    y = tf.one_hot(y, depth=n_CLASS, dtype=tf.float32)
    return img, y


val_ds = (
    val_ds.map(_val_load, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

STEP_SIZE_TRAIN = int(math.ceil(len(train_df) / BATCH_SIZE))
STEP_SIZE_VAL = int(math.ceil(len(val_df) / BATCH_SIZE))
print("STEP_SIZE_TRAIN:", STEP_SIZE_TRAIN, "STEP_SIZE_VAL:", STEP_SIZE_VAL)




## === cell 8
def create_model():
    model = Sequential()
    model.add(
        EfficientNetB3(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights="imagenet",
        )
    )
    model.add(GlobalAveragePooling2D())
    model.add(
        Dense(
            256,
            activation="relu",
            bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001),
        )
    )
    model.add(Dropout(0.5))
    model.add(Dense(n_CLASS, activation="softmax"))
    return model


leaf_model = create_model()
leaf_model.summary()




## === cell 9
loss = tf.keras.losses.CategoricalCrossentropy(
    from_logits=False,
    label_smoothing=0.0001,
    name="categorical_crossentropy",
)

leaf_model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=loss,
    metrics=["categorical_accuracy"],
)

model_save = ModelCheckpoint(
    filepath="Cassava_best.weights.h5",
    save_best_only=True,
    save_weights_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1,
)

early_stop = EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=5,
    mode="min",
    verbose=1,
    restore_best_weights=True,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_delta=0.001,
    mode="min",
    verbose=1,
)




## === cell 10
history = leaf_model.fit(
    train_ds,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=val_ds,
    validation_steps=STEP_SIZE_VAL,
    epochs=EPOCHS,
    callbacks=[model_save, early_stop, reduce_lr],
    verbose=1,
)

if os.path.exists("Cassava_best.weights.h5"):
    leaf_model.load_weights("Cassava_best.weights.h5")

loaded_model = leaf_model




## === cell 11
plt.figure(figsize=(8, 4))
plt.plot(history.epoch, history.history.get("loss", []), "-o", label="training_loss")
plt.plot(
    history.epoch, history.history.get("val_loss", []), "-o", label="validation_loss"
)
plt.legend()
plt.xlim(left=0)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(
    history.epoch,
    history.history.get("categorical_accuracy", []),
    "-o",
    label="train_accuracy",
)
plt.plot(
    history.epoch,
    history.history.get("val_categorical_accuracy", []),
    "-o",
    label="validation_accuracy",
)
plt.legend()
plt.xlim(left=0)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.show()




## === cell 12
from sklearn.metrics import confusion_matrix, classification_report

Y_pred = loaded_model.predict(val_ds, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)

y_true = val_df["class_name"].map(class_name_to_idx).astype(int).values
print("Confusion Matrix")
print(confusion_matrix(y_true, y_pred))

target_names = [idx_to_class_name[i] for i in range(n_CLASS)]
print(classification_report(y_true, y_pred, target_names=target_names))




## === cell 13
ss = pd.read_csv(os.path.join(work_dir, "sample_submission.csv"))

test_paths = (test_path.rstrip("/") + "/" + ss["image_id"].astype(str)).values
test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
options = tf.data.Options()
options.experimental_deterministic = True
test_ds = test_ds.with_options(options)


def _test_load(path):
    img = _decode_and_resize(path)
    img = _effnet_preprocess(img)
    return img


test_ds = (
    test_ds.map(_test_load, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

probs = loaded_model.predict(test_ds, verbose=1)
preds = np.argmax(probs, axis=1).astype(int)

submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": preds})
submission.to_csv("submission.csv", index=False)

print("Saved submission.csv with shape:", submission.shape)
print(submission.head())
