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

0.9073738289513448

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'The main timeout drivers are (1) parsing the full training TFRecords and then filtering with hash-table lookups (which still decodes/resizes *every* example twice for train/valid), (2) expensive per-image augmentation (notably the custom shear) running in the input pipeline, and (3) extra passes over the test dataset to separately collect `image_id`s in Python. To preserve core logic and accuracy, the changes below keep the same model/optimizer/loss/epochs and the same augmentation operations, but make the pipeline asymptotically cheaper by reading only the needed examples via TFRecord sharding + `take/skip` (so we don’t decode images that be filtered out), fuse prediction + id collection into a single forward pass, and remove redundant options/copies that add overhead. Determinism and seeds are preserved; no sampling, reduced precision, or early stopping behavior changes are introduced.'
- What this solution (achieved 0.05531) has done: 'The crash in the first cell comes from an upstream protobuf/TensorFlow incompatibility that triggers during `import tensorflow as tf`, so I add a safe environment workaround to force the pure-Python protobuf implementation before importing TensorFlow. I also remove/disable EarlyStopping (it changes training semantics and can severely hurt score here) while keeping the same model, optimizer, loss, augmentations, and epoch count. Finally, I make the TFRecord `image_id` decoding robust (strip any path prefix) to ensure predictions align with `sample_submission.csv`, which fixes the “fillna(0)” mass-mapping that can destroy accuracy. These changes are minimal, unblock execution, and should substantially increase accuracy toward your target by ensuring the model actually trains for 10 epochs and the submission mapping is correct.'
- What this solution (achieved 0.05531) has done: 'You’re failing before any training because `import tensorflow as tf` crashes with a protobuf API mismatch (`MessageFactory.GetPrototype`), so the pipeline never reaches model fitting and your submission ends up effectively random/mostly zeros. I add a small, safe “import guard” that forces a compatible protobuf runtime (and, if needed, automatically falls back to the TFRecord-free image-folder pipeline so the notebook can still run end-to-end). I also fix the submission key mismatch by converting TFRecord `image_id` bytes to proper strings and stripping any path prefix consistently, eliminating the massive `.fillna(0)` mapping that destroys accuracy. These are execution- and alignment-fixes (not architectural changes) and should move accuracy dramatically toward your target.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

os.environ["PYTHONHASHSEED"] = "42"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from google.protobuf import message_factory as _message_factory

    if (
        hasattr(_message_factory, "MessageFactory")
        and not hasattr(_message_factory.MessageFactory, "GetPrototype")
        and hasattr(_message_factory.MessageFactory, "GetMessageClass")
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF version:", tf.__version__)
AUTOTUNE = tf.data.AUTOTUNE



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.preprocessing import LabelEncoder

label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)
train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

train_csv["disease"] = train_csv["label"].map(label_to_disease)
train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"]
)

train_csv["label_encoded"] = LabelEncoder().fit_transform(train_csv["disease"])
train_csv["label"] = train_csv["label"].astype(str)
train_csv["disease"] = train_csv["disease"].astype(str)

train, valid = train_test_split(
    train_csv,
    test_size=0.2,
    stratify=train_csv["label"],
    random_state=42,
)

_unique_labels_sorted = sorted(train_csv["label"].unique().tolist())
class_indices = {lab: i for i, lab in enumerate(_unique_labels_sorted)}
NUM_CLASSES = len(class_indices)
print("Detected classes:", NUM_CLASSES)
print("Class indices:", class_indices)

idx_to_label = {v: int(k) for k, v in class_indices.items()}
print("idx_to_label:", idx_to_label)

_ds_options = tf.data.Options()
_ds_options.experimental_deterministic = True
_ds_options.autotune.enabled = True

TRAIN_TFREC_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords"
TEST_TFREC_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords"

train_tfrecs = sorted(
    [
        os.path.join(TRAIN_TFREC_DIR, f)
        for f in os.listdir(TRAIN_TFREC_DIR)
        if f.endswith(".tfrec")
    ]
)
test_tfrecs = sorted(
    [
        os.path.join(TEST_TFREC_DIR, f)
        for f in os.listdir(TEST_TFREC_DIR)
        if f.endswith(".tfrec")
    ]
)

print("Train tfrecords:", len(train_tfrecs), "Test tfrecords:", len(test_tfrecs))

_FEATURES_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    "id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
}
_FEATURES_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    "id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
}

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def _decode_and_resize(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)
    return img


def _preprocess(img):
    return preprocess_input(img)


def _normalize_image_id(image_id):
    image_id = tf.strings.regex_replace(image_id, rb"^.*\/", b"")
    return image_id


def _get_image_id_from_parsed(ex):
    image_id = ex.get("image_id", tf.constant(b"", dtype=tf.string))
    fallback_id = ex.get("id", tf.constant(b"", dtype=tf.string))
    image_id = tf.where(tf.equal(image_id, b""), fallback_id, image_id)
    image_id = _normalize_image_id(image_id)
    return image_id


def _parse_train_with_id(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES_TRAIN)
    img = _decode_and_resize(ex["image"])
    img = _preprocess(img)
    label = tf.cast(ex["target"], tf.int32)
    label_oh = tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)
    image_id = _get_image_id_from_parsed(ex)
    return img, label_oh, image_id


def _parse_train(example_proto):
    img, label_oh, _ = _parse_train_with_id(example_proto)
    return img, label_oh


def _parse_test(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES_TEST)
    img = _decode_and_resize(ex["image"])
    img = _preprocess(img)
    image_id = _get_image_id_from_parsed(ex)
    return img, image_id


import math


@tf.function
def _random_shear(img, shear_range=0.2):
    shear_deg = tf.random.uniform([], -shear_range, shear_range, dtype=tf.float32)
    shear = shear_deg * (math.pi / 180.0)
    sinv = tf.math.sin(shear)
    transform = tf.stack([1.0, -sinv, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])[tf.newaxis, :]
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[tf.newaxis, ...],
        transforms=transform,
        output_shape=tf.constant([IMG_SIZE[0], IMG_SIZE[1]], dtype=tf.int32),
        fill_value=0.0,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
    )[0]
    return out


augment_layers = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=42),
        tf.keras.layers.RandomRotation(
            factor=45.0 / 360.0, fill_mode="nearest", seed=42
        ),
        tf.keras.layers.RandomTranslation(0.2, 0.2, fill_mode="nearest", seed=42),
        tf.keras.layers.RandomZoom(0.2, 0.2, fill_mode="nearest", seed=42),
    ],
    name="augment",
)


def _augment(img, label):
    img = augment_layers(img, training=True)
    img = _random_shear(img, shear_range=0.2)
    return img, label


n_train = len(train)
n_valid = len(valid)
print("CSV split sizes:", n_train, n_valid)

USE_TFRECORDS = True

train_steps = (n_train + BATCH_SIZE - 1) // BATCH_SIZE
valid_steps = (n_valid + BATCH_SIZE - 1) // BATCH_SIZE

try:
    train_raw_all = tf.data.TFRecordDataset(
        train_tfrecs, num_parallel_reads=AUTOTUNE
    ).with_options(_ds_options)

    train_ids = tf.constant(train["image_id"].values.astype("S"), dtype=tf.string)
    valid_ids = tf.constant(valid["image_id"].values.astype("S"), dtype=tf.string)

    train_id_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            train_ids, tf.ones_like(train_ids, dtype=tf.int64)
        ),
        default_value=0,
    )
    valid_id_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            valid_ids, tf.ones_like(valid_ids, dtype=tf.int64)
        ),
        default_value=0,
    )

    def _parse_id_only_train(example_proto):
        ex = tf.io.parse_single_example(
            example_proto,
            {
                "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
                "id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
            },
        )
        image_id = _get_image_id_from_parsed(ex)
        return example_proto, image_id

    def _is_in_train_serialized(example_proto, image_id):
        return tf.equal(train_id_table.lookup(image_id), 1)

    def _is_in_valid_serialized(example_proto, image_id):
        return tf.equal(valid_id_table.lookup(image_id), 1)

    train_serialized_with_id = train_raw_all.map(
        _parse_id_only_train, num_parallel_calls=AUTOTUNE
    ).with_options(_ds_options)

    train_raw = train_serialized_with_id.filter(_is_in_train_serialized).map(
        lambda ex, _id: ex, num_parallel_calls=AUTOTUNE
    )
    valid_raw = train_serialized_with_id.filter(_is_in_valid_serialized).map(
        lambda ex, _id: ex, num_parallel_calls=AUTOTUNE
    )

    SHUFFLE_BUF = 8192
    train_ds = (
        train_raw.map(_parse_train, num_parallel_calls=AUTOTUNE)
        .shuffle(SHUFFLE_BUF, seed=42, reshuffle_each_iteration=False)
        .map(_augment, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
        .with_options(_ds_options)
    )
    valid_ds = (
        valid_raw.map(_parse_train, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .cache()
        .prefetch(AUTOTUNE)
        .with_options(_ds_options)
    )

except Exception as e:
    print(
        "TFRecord pipeline failed; falling back to image-folder pipeline. Error:",
        repr(e),
    )
    USE_TFRECORDS = False

if not USE_TFRECORDS:
    def _load_from_path(path, label_int):
        img_bytes = tf.io.read_file(path)
        img = _decode_and_resize(img_bytes)
        img = _preprocess(img)
        label_oh = tf.one_hot(
            tf.cast(label_int, tf.int32), NUM_CLASSES, dtype=tf.float32
        )
        return img, label_oh

    train_paths = train["path"].values.astype(str)
    valid_paths = valid["path"].values.astype(str)
    train_labels = train["label"].astype(int).values
    valid_labels = valid["label"].astype(int).values

    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        .shuffle(8192, seed=42, reshuffle_each_iteration=False)
        .map(lambda p, y: _load_from_path(p, y), num_parallel_calls=AUTOTUNE)
        .map(_augment, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
        .with_options(_ds_options)
    )
    valid_ds = (
        tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
        .map(lambda p, y: _load_from_path(p, y), num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .cache()
        .prefetch(AUTOTUNE)
        .with_options(_ds_options)
    )



## === cell 2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.layers import Input


class EarlyStoppingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if self.model.stop_training:
            print(f"Early stopping triggered at epoch {epoch + 1}.")


early_stopping = None

learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1
)



## === cell 3
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)
x = base_model.output
x = GlobalAveragePooling2D()(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()



## === cell 4
callbacks = [learning_rate_reduction]
if early_stopping is not None:
    callbacks.append(early_stopping)
    callbacks.append(EarlyStoppingCallback())

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    steps_per_epoch=train_steps,
    validation_steps=valid_steps,
    epochs=10,
    callbacks=callbacks,
    verbose=1,
)



## === cell 5
import pandas as pd
import numpy as np
import os

sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)


def _bytes_to_str_no_prefix(arr_bytes):
    out = []
    for b in arr_bytes.tolist():
        if isinstance(b, (bytes, bytearray)):
            s = b.decode("utf-8", errors="ignore")
        else:
            s = str(b)
        s = re.sub(r"^.*\/", "", s)
        out.append(s)
    return np.array(out, dtype=object)


pred_map = {}

if USE_TFRECORDS:
    test_raw = tf.data.TFRecordDataset(
        test_tfrecs, num_parallel_reads=AUTOTUNE
    ).with_options(_ds_options)
    test_parsed = test_raw.map(_parse_test, num_parallel_calls=AUTOTUNE).with_options(
        _ds_options
    )
    test_ds = (
        test_parsed.batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
        .with_options(_ds_options)
    )

    test_image_ids = []
    pred_labels_list = []

    for batch_imgs, batch_ids in test_ds:
        probs = model(batch_imgs, training=False).numpy()
        pred_idx = np.argmax(probs, axis=1).astype(int)
        pred_labels = np.vectorize(idx_to_label.get)(pred_idx).astype(int)
        pred_labels_list.append(pred_labels)

        test_image_ids.append(_bytes_to_str_no_prefix(batch_ids.numpy()))

    test_image_ids = np.concatenate(test_image_ids, axis=0)
    pred_labels = np.concatenate(pred_labels_list, axis=0)

    pred_map = dict(zip(test_image_ids.tolist(), pred_labels.tolist()))
else:
    test_img_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
    test_ids = sample_sub["image_id"].astype(str).tolist()
    test_paths = [os.path.join(test_img_dir, x) for x in test_ids]

    def _load_test(path):
        img_bytes = tf.io.read_file(path)
        img = _decode_and_resize(img_bytes)
        img = _preprocess(img)
        return img

    test_ds = (
        tf.data.Dataset.from_tensor_slices(test_paths)
        .map(_load_test, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
        .with_options(_ds_options)
    )

    pred_labels_list = []
    for batch_imgs in test_ds:
        probs = model(batch_imgs, training=False).numpy()
        pred_idx = np.argmax(probs, axis=1).astype(int)
        pred_labels = np.vectorize(idx_to_label.get)(pred_idx).astype(int)
        pred_labels_list.append(pred_labels)
    pred_labels = np.concatenate(pred_labels_list, axis=0)
    pred_map = dict(zip(test_ids, pred_labels.tolist()))

submission_df = sample_sub.copy()
submission_df["image_id"] = (
    submission_df["image_id"].astype(str).str.replace(r"^.*\/", "", regex=True)
)

submission_df["label"] = submission_df["image_id"].map(pred_map)
missing = submission_df["label"].isna().sum()
if missing:
    print(
        "WARNING: missing mapped ids:",
        int(missing),
        "-> filling with 0 to keep valid submission",
    )
    submission_df["label"] = submission_df["label"].fillna(0)

submission_df["label"] = submission_df["label"].astype(int)

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Submission file created:", submission_path)
print(submission_df.head())
print("Submission shape:", submission_df.shape)
print("Missing mapped ids:", int(missing))
print("Label distribution:\n", submission_df["label"].value_counts().sort_index())
