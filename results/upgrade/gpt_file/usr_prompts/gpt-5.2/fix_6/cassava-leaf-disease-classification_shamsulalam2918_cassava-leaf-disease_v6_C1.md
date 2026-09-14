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
import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    Callback,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

plt.rcParams["figure.figsize"] = (17, 6)

SEED = 42
tf.keras.utils.set_random_seed(SEED)

try:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## === cell 1
TRAINING_DIR = "../input/cassava-leaf-disease-classification/train_images"
TRAINING_CSV = "../input/cassava-leaf-disease-classification/train.csv"
JSON_LABELS = (
    "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
)
TEST_DIR = "../input/cassava-leaf-disease-classification/test_images"
SAMPLE_SUB_PATH = "../input/cassava-leaf-disease-classification/sample_submission.csv"



## === cell 2
train_df = pd.read_csv(TRAINING_CSV)
train_df["label"] = train_df["label"].astype(
    "string"
)  # required by flow_from_dataframe for categorical class_mode
train_df.head()



## === cell 3
total_images_count = len(train_df.index)
total_train_img_count = int(total_images_count * 0.8)
total_val_img_count = total_images_count - total_train_img_count
print("Expected images counts:")
print(f"\nTotal Images from original directory: {total_images_count}")
print(f"Training Images: {total_train_img_count}")
print(f"Validation Images: {total_val_img_count}")



## === cell 4
label_df = pd.read_json(JSON_LABELS, orient="index")
label_df = label_df.values.flatten().tolist()
label_df



## === cell 5
train_label_0 = train_df[train_df["label"] == "0"]
train_label_1 = train_df[train_df["label"] == "1"]
train_label_2 = train_df[train_df["label"] == "2"]
train_label_3 = train_df[train_df["label"] == "3"]
train_label_4 = train_df[train_df["label"] == "4"]
print(
    "Per-class counts:",
    [
        len(train_label_0),
        len(train_label_1),
        len(train_label_2),
        len(train_label_3),
        len(train_label_4),
    ],
)



## === cell 6
if False:
    training_images_dir = os.path.join(TRAINING_DIR, "*.jpg")
    print(training_images_dir)
    training_images = glob.glob(training_images_dir)

    if len(training_images) > 0:
        plt.figure(figsize=(12, 12))
        for i in range(1, 10):
            training_image = np.random.choice(training_images)
            training_image_RGB = cv2.imread(training_image)[..., ::-1]
            plt.subplot(3, 3, i)
            plt.imshow(training_image_RGB)
            plt.axis("off")
        plt.show()



## === cell 7
training_datagen = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=100,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.7, 1.4],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
    validation_split=0.2,
)

validation_datagen = ImageDataGenerator(
    rescale=1 / 255.0,
    validation_split=0.2,
)



## === cell 8
BATCH_SIZE = 24
IMG_WIDTH = 300
IMG_HEIGHT = 300
CHANNEL = 3

AUTOTUNE = tf.data.AUTOTUNE

train_df_sorted = train_df.reset_index(drop=True)
n_total = len(train_df_sorted)
n_val = int(round(n_total * 0.2))
val_df = train_df_sorted.iloc[:n_val].copy()
trn_df = train_df_sorted.iloc[n_val:].copy()

print("\nTraining Dataset")
print(f"Found {len(trn_df)} training images belonging to 5 classes.")
print("\nValidation Dataset")
print(f"Found {len(val_df)} validation images belonging to 5 classes.")

class_indices = {str(i): i for i in range(5)}
print("\nClass Indices:")
print(class_indices)

trn_paths = (TRAINING_DIR + "/" + trn_df["image_id"].astype(str)).to_numpy()
val_paths = (TRAINING_DIR + "/" + val_df["image_id"].astype(str)).to_numpy()
trn_labels = trn_df["label"].astype(int).to_numpy()
val_labels = val_df["label"].astype(int).to_numpy()


@tf.function(jit_compile=False)
def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, [IMG_WIDTH, IMG_HEIGHT], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function(jit_compile=False)
def _augment_stateless(img, seed):
    rot = tf.random.stateless_uniform([], seed=seed, minval=-100.0, maxval=100.0) * (
        np.pi / 180.0
    )
    tx = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([1, 0], tf.int32), minval=-0.2, maxval=0.2
    ) * tf.cast(IMG_HEIGHT, tf.float32)
    ty = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([0, 1], tf.int32), minval=-0.2, maxval=0.2
    ) * tf.cast(IMG_WIDTH, tf.float32)
    shx = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([2, 0], tf.int32), minval=-0.2, maxval=0.2
    )
    shy = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([0, 2], tf.int32), minval=-0.2, maxval=0.2
    )
    zx = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([3, 0], tf.int32), minval=0.7, maxval=1.3
    )
    zy = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([0, 3], tf.int32), minval=0.7, maxval=1.3
    )

    cx = (IMG_WIDTH - 1) / 2.0
    cy = (IMG_HEIGHT - 1) / 2.0

    cos_r = tf.cos(rot)
    sin_r = tf.sin(rot)

    a00 = cos_r * zx + (-sin_r) * shy * zx
    a01 = cos_r * shx * zy + (-sin_r) * zy
    a10 = sin_r * zx + (cos_r) * shy * zx
    a11 = sin_r * shx * zy + (cos_r) * zy

    t0 = (cx - a00 * cx - a01 * cy) + ty
    t1 = (cy - a10 * cx - a11 * cy) + tx

    transform = tf.stack([a00, a01, t0, a10, a11, t1, 0.0, 0.0])

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(img, 0),
        transforms=tf.expand_dims(transform, 0),
        output_shape=tf.constant([IMG_HEIGHT, IMG_WIDTH], tf.int32),
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0,
    )
    img = tf.squeeze(img, 0)

    do_h = (
        tf.random.stateless_uniform(
            [], seed=seed + tf.constant([4, 0], tf.int32), minval=0.0, maxval=1.0
        )
        < 0.5
    )
    do_v = (
        tf.random.stateless_uniform(
            [], seed=seed + tf.constant([0, 4], tf.int32), minval=0.0, maxval=1.0
        )
        < 0.5
    )
    img = tf.cond(do_h, lambda: tf.image.flip_left_right(img), lambda: img)
    img = tf.cond(do_v, lambda: tf.image.flip_up_down(img), lambda: img)

    br = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([5, 0], tf.int32), minval=0.7, maxval=1.4
    )
    img = tf.clip_by_value(img * br, 0.0, 1.0)
    return img


def _with_perf_options(ds):
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_slack = (
        True  # allows input pipeline to prefetch more effectively
    )
    options.threading.private_threadpool_size = 0  # let TF pick
    return ds.with_options(options)


def _make_train_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)

    @tf.function(jit_compile=False)
    def _map_fn(path, label):
        img = _decode_resize(path)
        h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
        seed = tf.stack([tf.cast(SEED, tf.int32), tf.cast(h, tf.int32)])
        img = _augment_stateless(img, seed)
        y = tf.one_hot(tf.cast(label, tf.int32), depth=5)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = _with_perf_options(ds)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_val_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    @tf.function(jit_compile=False)
    def _map_fn(path, label):
        img = _decode_resize(path)
        y = tf.one_hot(tf.cast(label, tf.int32), depth=5)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = _with_perf_options(ds)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = _make_train_ds(trn_paths, trn_labels)
validation_ds = _make_val_ds(val_paths, val_labels)




## === cell 9
class theCallBacks(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if (logs.get("val_accuracy", 0) > 0.92) and (logs.get("accuracy", 0) > 0.92):
            print(
                "\nTraining Accuracy> 0.92 & Validation Accuracy> 0.92\nCancelling training!"
            )
            self.model.stop_training = True


callback_on_metrics = theCallBacks()

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    verbose=1,
    cooldown=1,
    min_lr=0.0001,
)



## === cell 10
loss_func = CategoricalCrossentropy()



## === cell 11
import datetime


class LearningRateLogger(Callback):
    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if "learning_rate" not in logs:
            lr = getattr(self.model.optimizer, "learning_rate", None)
            try:
                logs["learning_rate"] = float(tf.keras.backend.get_value(lr))
            except Exception:
                pass


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0)



## === cell 12
new_input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, CHANNEL))



## === cell 13
DROPOUT_RATE = 0.5

base_model = InceptionResNetV2(
    include_top=False,
    weights="imagenet",
    input_tensor=new_input,
    pooling="avg",
)

for layer in base_model.layers:
    layer.trainable = False

x = Flatten(name="flatten")(base_model.output)
x = Dropout(DROPOUT_RATE)(x)
x = Dense(4096, activation="relu", name="fc6")(x)
x = Dropout(DROPOUT_RATE)(x)
x = Dense(1024, activation="relu", name="fc7")(x)
x = Dropout(DROPOUT_RATE)(x)

out = Dense(5, activation="softmax", name="classifier")(x)
model = Model(inputs=base_model.input, outputs=out)

SGD_LEARNING_RATE = 0.01
SGD_DECAY = 0.0001
sgd = tf.keras.optimizers.SGD(
    learning_rate=SGD_LEARNING_RATE, momentum=0.9, decay=SGD_DECAY, nesterov=True
)

model.compile(optimizer=sgd, loss=loss_func, metrics=["accuracy"])
model.summary()



## === cell 14
num_epochs = 10

steps_per_epoch = int(np.ceil(len(trn_paths) / BATCH_SIZE))
validation_steps = int(np.ceil(len(val_paths) / BATCH_SIZE))
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)



## === cell 15
model_checkpoint_path = "./best_model.keras"
checkpoint = ModelCheckpoint(
    filepath=model_checkpoint_path,
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    mode="max",
)

history = model.fit(
    train_ds,
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_ds,
    validation_steps=validation_steps,
    callbacks=[
        checkpoint,
        reduce_lr,
        callback_on_metrics,
        LearningRateLogger(),
        tensorboard_callback,
    ],
    verbose=1,
)

model = tf.keras.models.load_model(model_checkpoint_path)



## === cell 16
sample_submission = pd.read_csv(SAMPLE_SUB_PATH)

test_paths = (TEST_DIR + "/" + sample_submission["image_id"].astype(str)).to_numpy()


def _make_test_ds(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)

    @tf.function(jit_compile=False)
    def _map_fn(path):
        img = _decode_resize(path)
        return img

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = _with_perf_options(ds)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = _make_test_ds(test_paths)
test_steps = int(np.ceil(len(test_paths) / BATCH_SIZE))

probs = model.predict(
    test_ds,
    steps=test_steps,
    verbose=1,
)

predicted = np.argmax(probs, axis=1)[: len(sample_submission)].astype(int).tolist()

submission = pd.DataFrame({"image_id": sample_submission.image_id, "label": predicted})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with", len(submission), "rows")
