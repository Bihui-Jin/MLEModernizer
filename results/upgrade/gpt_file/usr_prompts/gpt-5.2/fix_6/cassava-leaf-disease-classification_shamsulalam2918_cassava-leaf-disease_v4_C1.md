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

0.1000302206104563

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

INPUT_DIR = "/kaggle/input/cassava-leaf-disease-classification"
WORKING_DIR = "/kaggle/working"
os.makedirs(WORKING_DIR, exist_ok=True)

print("INPUT_DIR exists:", os.path.exists(INPUT_DIR))
print("Listing input root:")
print(os.listdir("/kaggle/input")[:10])




## === cell 1
import numpy as np
import pandas as pd

print(
    "Skipping os.walk directory scan for runtime (no effect on model training or predictions)."
)




## === cell 2
import glob
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.callbacks import (
    Callback,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Input, Model
from tensorflow.keras.applications import InceptionResNetV2

plt.rcParams["figure.figsize"] = (17, 6)

print("TensorFlow:", tf.__version__)

SEED = 42
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
    print("Enabled TF op determinism.")
except Exception as e:
    print("Could not enable TF determinism:", e)

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("Could not enable XLA JIT:", e)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception as e:
    print("Could not set TF threading:", e)




## === cell 3
TRAINING_DIR = os.path.join(INPUT_DIR, "train_images")
TRAINING_CSV = os.path.join(INPUT_DIR, "train.csv")
JSON_LABELS = os.path.join(INPUT_DIR, "label_num_to_disease_map.json")
TEST_DIR = os.path.join(INPUT_DIR, "test_images")
SAMPLE_SUB = os.path.join(INPUT_DIR, "sample_submission.csv")

assert os.path.exists(TRAINING_DIR), TRAINING_DIR
assert os.path.exists(TRAINING_CSV), TRAINING_CSV
assert os.path.exists(TEST_DIR), TEST_DIR
assert os.path.exists(SAMPLE_SUB), SAMPLE_SUB




## === cell 4
train_df = pd.read_csv(TRAINING_CSV)
train_df["label"] = train_df["label"].astype(
    "string"
)  # keep label dtype consistent with original semantics
train_df.head()




## === cell 5
total_images_count = len(train_df.index)
total_train_img_count = int(total_images_count * 0.8)
total_val_img_count = total_images_count - total_train_img_count

print("Expected images counts:")
print(f"Total Images from original directory: {total_images_count}")
print(f"Training Images: {total_train_img_count}")
print(f"Validation Images: {total_val_img_count}")




## === cell 6
label_df = pd.read_json(JSON_LABELS, orient="index")
label_names = label_df.values.flatten().tolist()
print("Label names:", label_names)




## === cell 7
print(
    "Skipping training image glob/plot for runtime (no effect on model training or predictions)."
)




## === cell 8
BATCH_SIZE = 24
IMG_WIDTH = 300
IMG_HEIGHT = 300
CHANNEL = 3

print("Building tf.data datasets (same 80/20 split, same augmentations)...")

train_df_ordered = train_df.reset_index(drop=True)
train_df_train = train_df_ordered.iloc[:total_train_img_count].copy()
train_df_val = train_df_ordered.iloc[total_train_img_count:].copy()

classes = sorted(train_df_ordered["label"].unique().tolist(), key=lambda x: str(x))
class_to_index = {c: i for i, c in enumerate(classes)}
print("Class Indices:")
print(class_to_index)

train_labels_int = train_df_train["label"].map(class_to_index).astype(np.int32).values
val_labels_int = train_df_val["label"].map(class_to_index).astype(np.int32).values

train_image_ids = train_df_train["image_id"].astype(str).values
val_image_ids = train_df_val["image_id"].astype(str).values

AUTOTUNE = tf.data.AUTOTUNE


def _read_decode_resize(image_id):
    img_path = tf.strings.join([TRAINING_DIR, os.sep, image_id])
    img_bytes = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_WIDTH, IMG_HEIGHT], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.ensure_shape(img, (IMG_HEIGHT, IMG_WIDTH, CHANNEL))
    return img


def _augment(img):
    b = tf.random.uniform([], minval=0.7, maxval=1.4, dtype=tf.float32)
    img = tf.clip_by_value(img * b, 0.0, 1.0)

    img = tf.cond(
        tf.random.uniform([]) < 0.5, lambda: tf.image.flip_left_right(img), lambda: img
    )
    img = tf.cond(
        tf.random.uniform([]) < 0.5, lambda: tf.image.flip_up_down(img), lambda: img
    )

    angle = tf.random.uniform([], minval=-100.0, maxval=100.0, dtype=tf.float32) * (
        np.pi / 180.0
    )

    tx = tf.random.uniform([], minval=-0.2, maxval=0.2, dtype=tf.float32) * tf.cast(
        IMG_HEIGHT, tf.float32
    )
    ty = tf.random.uniform([], minval=-0.2, maxval=0.2, dtype=tf.float32) * tf.cast(
        IMG_WIDTH, tf.float32
    )

    zoom = tf.random.uniform([], minval=0.7, maxval=1.3, dtype=tf.float32)
    shear = tf.random.uniform([], minval=-0.2, maxval=0.2, dtype=tf.float32)

    cx = (IMG_WIDTH - 1) / 2.0
    cy = (IMG_HEIGHT - 1) / 2.0

    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)

    a0 = cos_a / zoom
    a1 = (-sin_a + shear) / zoom
    b0 = sin_a / zoom
    b1 = (cos_a + shear) / zoom

    a2 = cx + ty - a0 * cx - a1 * cy
    b2 = cy + tx - b0 * cx - b1 * cy

    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[None, :]

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=transform,
        output_shape=tf.constant([IMG_HEIGHT, IMG_WIDTH], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0,
    )[0]
    img = tf.ensure_shape(img, (IMG_HEIGHT, IMG_WIDTH, CHANNEL))
    return img


def _to_onehot(label_int):
    return tf.one_hot(label_int, depth=5, dtype=tf.float32)


_DATASET_OPTIONS = tf.data.Options()
_DATASET_OPTIONS.experimental_deterministic = True
try:
    _DATASET_OPTIONS.autotune.enabled = True
except Exception:
    pass


def _make_ds(image_ids, label_ints, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((image_ids, label_ints))
    ds = ds.with_options(_DATASET_OPTIONS)
    if training:
        ds = ds.shuffle(
            buffer_size=len(image_ids), seed=SEED, reshuffle_each_iteration=True
        )

    ds = ds.map(
        lambda img_id, y: (_read_decode_resize(img_id), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    if training:
        ds = (
            ds.cache()
        )  # caches decoded/resized tensors (not augmented), avoids repeated JPEG decode each epoch
        ds = ds.map(
            lambda x, y: (_augment(x), _to_onehot(y)),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )
    else:
        ds = ds.map(
            lambda x, y: (x, _to_onehot(y)),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        ).cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_tfds = _make_ds(train_image_ids, train_labels_int, training=True)
val_tfds = _make_ds(val_image_ids, val_labels_int, training=False)

print("\nTraining Dataset:", train_tfds)
print("Validation Dataset:", val_tfds)




## === cell 9
class theCallBacks(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if (logs.get("val_accuracy", 0) > 0.92) and (logs.get("accuracy", 0) > 0.92):
            print(
                "\nTraining Accuracy > 0.92 & Validation Accuracy > 0.92\nCancelling training!"
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
import datetime


class LearningRateLogger(Callback):
    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if "learning_rate" in logs:
            return
        lr = getattr(self.model.optimizer, "learning_rate", None)
        try:
            logs["learning_rate"] = float(tf.keras.backend.get_value(lr))
        except Exception:
            pass


log_dir = os.path.join(
    WORKING_DIR, "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0)




## === cell 11
new_input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, CHANNEL))

base_model = InceptionResNetV2(
    include_top=False,
    weights="imagenet",
    input_tensor=new_input,
    pooling="avg",
)

for layer in base_model.layers:
    layer.trainable = False

x = Dropout(0.5)(base_model.output)
x = Dense(4096, activation="relu", name="fc6")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu", name="fc7")(x)
x = Dropout(0.5)(x)
out = Dense(5, activation="softmax", name="classifier")(x)

model = Model(inputs=base_model.input, outputs=out)

sgd = tf.keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.9, decay=0.0001, nesterov=True
)
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

try:
    model.steps_per_execution = 16
except Exception as e:
    print("Could not set steps_per_execution:", e)

model.summary()




## === cell 12
num_epochs = 10
steps_per_epoch = max(1, total_train_img_count // BATCH_SIZE)
validation_steps = max(1, total_val_img_count // BATCH_SIZE)

model_checkpoint_path = os.path.join(WORKING_DIR, "cassava_best.keras")
checkpoint = ModelCheckpoint(
    filepath=model_checkpoint_path,
    monitor="val_accuracy",
    verbose=1,
    save_best_only=True,
    mode="max",
)

history = model.fit(
    train_tfds,
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_tfds,
    validation_steps=validation_steps,
    callbacks=[
        checkpoint,
        reduce_lr,
        callback_on_metrics,
        tensorboard_callback,
        LearningRateLogger(),
    ],
    verbose=1,
)

print("Best model saved to:", model_checkpoint_path)
print("Checkpoint exists:", os.path.exists(model_checkpoint_path))




## === cell 13
if os.path.exists(model_checkpoint_path):
    best_model = tf.keras.models.load_model(model_checkpoint_path, compile=False)
    print("Loaded best_model from checkpoint.")
else:
    best_model = model
    print("Checkpoint not found; using in-memory trained model for prediction.")

sample_submission = pd.read_csv(SAMPLE_SUB)
assert {"image_id", "label"}.issubset(sample_submission.columns)

test_image_ids = sample_submission["image_id"].values.astype(str)


def _load_and_preprocess(image_id):
    img_path = tf.strings.join([TEST_DIR, os.sep, image_id])
    img_bytes = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize_with_crop_or_pad(
        img,
        tf.minimum(tf.shape(img)[0], tf.shape(img)[1]),
        tf.minimum(tf.shape(img)[0], tf.shape(img)[1]),
    )
    img = tf.image.resize(
        img, [IMG_WIDTH, IMG_HEIGHT], method=tf.image.ResizeMethod.BILINEAR
    )
    img = img / 255.0
    img = tf.ensure_shape(img, (IMG_HEIGHT, IMG_WIDTH, CHANNEL))
    return img


test_ds = tf.data.Dataset.from_tensor_slices(
    tf.convert_to_tensor(test_image_ids, dtype=tf.string)
).with_options(_DATASET_OPTIONS)
test_ds = test_ds.map(
    _load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
)
test_ds = test_ds.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

probs = best_model.predict(test_ds, verbose=0)
predicted = np.argmax(probs, axis=1).astype(int).tolist()

submission = pd.DataFrame(
    {"image_id": sample_submission["image_id"], "label": predicted}
)
submission_path = os.path.join(WORKING_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.head())
print("Rows:", len(submission))
assert submission_path.endswith(".csv") and os.path.exists(submission_path)
assert len(submission) == len(sample_submission)
assert list(submission.columns) == ["image_id", "label"]
