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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

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
import tensorflow as tf

from tensorflow.keras.callbacks import (
    Callback,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard,
)
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Input, Model
from tensorflow.keras.applications import InceptionResNetV2

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

TRAIN_TFRECORDS_DIR = os.path.join(INPUT_DIR, "train_tfrecords")
TEST_TFRECORDS_DIR = os.path.join(INPUT_DIR, "test_tfrecords")

assert os.path.exists(TRAINING_DIR), TRAINING_DIR
assert os.path.exists(TRAINING_CSV), TRAINING_CSV
assert os.path.exists(TEST_DIR), TEST_DIR
assert os.path.exists(SAMPLE_SUB), SAMPLE_SUB
assert os.path.exists(TRAIN_TFRECORDS_DIR), TRAIN_TFRECORDS_DIR
assert os.path.exists(TEST_TFRECORDS_DIR), TEST_TFRECORDS_DIR




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

train_df_ordered = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
train_df_train = train_df_ordered.iloc[:total_train_img_count].copy()
train_df_val = train_df_ordered.iloc[total_train_img_count:].copy()

classes = sorted(train_df_ordered["label"].unique().tolist(), key=lambda x: str(x))
class_to_index = {c: i for i, c in enumerate(classes)}
print("Class Indices:")
print(class_to_index)

_img_ids_all = train_df_ordered["image_id"].astype(str).to_numpy()
_lbl_int_all = train_df_ordered["label"].map(class_to_index).astype(np.int32).to_numpy()

AUTOTUNE = tf.data.AUTOTUNE

TRAIN_TFRECORD_FILES = sorted(glob.glob(os.path.join(TRAIN_TFRECORDS_DIR, "*.tfrec")))
TEST_TFRECORD_FILES = sorted(glob.glob(os.path.join(TEST_TFRECORDS_DIR, "*.tfrec")))
assert len(TRAIN_TFRECORD_FILES) > 0, "No train tfrecords found"
assert len(TEST_TFRECORD_FILES) > 0, "No test tfrecords found"

_FEATURE_DESCRIPTION_IMAGEONLY = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}
_FEATURE_DESCRIPTION_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}

_DATASET_OPTIONS = tf.data.Options()
_DATASET_OPTIONS.experimental_deterministic = True
try:
    _DATASET_OPTIONS.autotune.enabled = True
except Exception:
    pass
try:
    _DATASET_OPTIONS.experimental_optimization.map_and_batch_fusion = True
    _DATASET_OPTIONS.experimental_optimization.parallel_batch = True
    _DATASET_OPTIONS.experimental_optimization.map_parallelization = True
except Exception:
    pass

train_keys = tf.constant(_img_ids_all, dtype=tf.string)
train_vals = tf.constant(_lbl_int_all, dtype=tf.int32)
_label_lookup = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(train_keys, train_vals),
    default_value=tf.constant(-1, tf.int32),
)


@tf.function
def _decode_resize_from_bytes(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_WIDTH, IMG_HEIGHT], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.ensure_shape(img, (IMG_HEIGHT, IMG_WIDTH, CHANNEL))
    return img


@tf.function
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
        tf.constant(np.pi / 180.0, tf.float32)
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


@tf.function
def _to_onehot(label_int):
    return tf.one_hot(label_int, depth=5, dtype=tf.float32)


@tf.function
def _parse_to_name_img_label(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESCRIPTION_IMAGEONLY)
    name = ex["image_name"]
    img = _decode_resize_from_bytes(ex["image"])
    label_int = _label_lookup.lookup(name)
    return name, img, _to_onehot(label_int)


def _make_train_val_from_tfrecords(all_train_files):
    base = tf.data.TFRecordDataset(
        all_train_files, num_parallel_reads=AUTOTUNE
    ).with_options(_DATASET_OPTIONS)

    base = base.map(
        _parse_to_name_img_label, num_parallel_calls=AUTOTUNE, deterministic=True
    )

    base_enum = base.enumerate()

    train_enum = base_enum.take(total_train_img_count)
    val_enum = base_enum.skip(total_train_img_count)

    train_ds = train_enum.map(
        lambda i, x: x, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    val_ds = val_enum.map(
        lambda i, x: x, num_parallel_calls=AUTOTUNE, deterministic=True
    )

    train_ds = train_ds.map(
        lambda name, img, y: (_augment(img), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    ).repeat()

    val_ds = val_ds.map(
        lambda name, img, y: (img, y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    shuffle_buf = min(total_train_img_count, 4096)
    train_ds = train_ds.shuffle(
        buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
    )

    val_ds = val_ds.cache()

    train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return train_ds, val_ds


train_tfds, val_tfds = _make_train_val_from_tfrecords(TRAIN_TFRECORD_FILES)

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
    model.steps_per_execution = 32
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


@tf.function
def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESCRIPTION_TEST)
    img = _decode_resize_from_bytes(ex["image"])
    image_name = ex["image_name"]
    return image_name, img


test_ds = tf.data.TFRecordDataset(
    TEST_TFRECORD_FILES, num_parallel_reads=tf.data.AUTOTUNE
).with_options(_DATASET_OPTIONS)

test_ds = test_ds.map(
    _parse_test_example, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
).cache()

expected_names = sample_submission["image_id"].astype(str).tolist()
test_names = []
for bn, _ in test_ds.batch(BATCH_SIZE):
    test_names.extend([s.decode("utf-8") for s in bn.numpy().tolist()])

if test_names != expected_names:
    raise RuntimeError(
        "Test TFRecord order does not match sample_submission.csv order; "
        "cannot skip reordering without changing semantics."
    )

test_imgs_ds = (
    test_ds.map(lambda n, x: x, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

probs = best_model.predict(test_imgs_ds, verbose=0)
preds_out = np.asarray(np.argmax(probs, axis=1), dtype=np.int64)

submission = pd.DataFrame(
    {"image_id": sample_submission["image_id"], "label": preds_out}
)
submission_path = os.path.join(WORKING_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.head())
print("Rows:", len(submission))
assert submission_path.endswith(".csv") and os.path.exists(submission_path)
assert len(submission) == len(sample_submission)
assert list(submission.columns) == ["image_id", "label"]
