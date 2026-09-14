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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import random
import numpy as np
import tensorflow as tf

SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    pass
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass
try:
    tf.config.optimizer.set_experimental_options({"layout_optimizer": True})
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## === cell 1
DATA_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_IMG_DIR = os.path.join(DATA_PATH, "train_images")
TEST_IMG_DIR = os.path.join(DATA_PATH, "test_images")
TRAIN_TFREC_DIR = os.path.join(DATA_PATH, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_PATH, "test_tfrecords")

assert os.path.exists(DATA_PATH), f"Missing DATA_PATH: {DATA_PATH}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing TRAIN_IMG_DIR: {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing TEST_IMG_DIR: {TEST_IMG_DIR}"
assert os.path.exists(TRAIN_TFREC_DIR), f"Missing TRAIN_TFREC_DIR: {TRAIN_TFREC_DIR}"
assert os.path.exists(TEST_TFREC_DIR), f"Missing TEST_TFREC_DIR: {TEST_TFREC_DIR}"




## === cell 2
import pandas as pd
import json

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau




## === cell 3
def first_existing_path(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None




## === cell 4
path = DATA_PATH

train_df = pd.read_csv(os.path.join(path, "train.csv"))
print("Total images for Train (from train.csv): ", len(train_df))

with open(os.path.join(path, "label_num_to_disease_map.json")) as file:
    classes = json.loads(file.read())

print(json.dumps(classes, indent=4))

train_df["class"] = train_df["label"].map({int(i): c for i, c in classes.items()})
train_df.head()




## === cell 5
DO_PLOTS = False

if DO_PLOTS:
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.subplots(figsize=(12, 8))
    ax = sns.countplot(x="class", data=train_df)

    for a in ax.patches:
        ax.annotate("{:1}".format(a.get_height()), (a.get_x() + 0.3, a.get_height()))
    plt.xticks(rotation=90)
    ax.set_title("classes", fontdict={"fontsize": 15})
    plt.show()




## === cell 6
def plot_images(class_id, label, images_number, verbose=0):
    import cv2  # local import to avoid hard dependency when plotting is disabled
    import matplotlib.pyplot as plt

    plot_list = (
        train_df[train_df["label"] == class_id]
        .sample(images_number, random_state=SEED)["image_id"]
        .tolist()
    )

    if verbose:
        print(plot_list)

    labels = [label for _ in range(len(plot_list))]
    size = np.sqrt(images_number)
    if int(size) * int(size) < images_number:
        size = int(size) + 1

    plt.figure(figsize=(20, 20))

    for ind, (image_id, label) in enumerate(zip(plot_list, labels)):
        plt.subplot(size, size, ind + 1)
        image = cv2.imread(os.path.join(path, "train_images", image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image)
        plt.title(label, fontsize=12)
        plt.axis("off")

    plt.show()


if DO_PLOTS:
    plot_images(class_id=4, label="Healthy", images_number=6, verbose=1)
    plot_images(
        class_id=3, label="Cassava Mosaic Disease (CMD)", images_number=6, verbose=1
    )
    plot_images(
        class_id=2, label="Cassava Green Mottle (CGM)", images_number=6, verbose=1
    )
    plot_images(
        class_id=1,
        label="Cassava Brown Streak Disease (CBSD)",
        images_number=6,
        verbose=1,
    )
    plot_images(
        class_id=0, label="Cassava Bacterial Blight (CBB)", images_number=6, verbose=1
    )




## === cell 7
train_df["label"] = train_df["label"].astype(np.int64)




## === cell 8
TARGET_SIZE = (380, 380)
BATCH_SIZE = 16
EPOCHS = 5




## === cell 9
AUTO = tf.data.AUTOTUNE

_FEATURES_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
}
_FEATURES_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _decode_and_resize(image_bytes):
    img = tf.image.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(img, TARGET_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img


def _augment_equivalent(img, seed_pair):
    s0, s1 = seed_pair[0], seed_pair[1]

    img = tf.image.stateless_random_flip_left_right(img, seed=[s0, s1])
    img = tf.image.stateless_random_flip_up_down(img, seed=[s0, s1 + 1])

    angle = tf.random.stateless_uniform(
        [], seed=[s0, s1 + 2], minval=-45.0, maxval=45.0
    ) * (np.pi / 180.0)
    c = tf.math.cos(angle)
    s = tf.math.sin(angle)

    h = tf.cast(TARGET_SIZE[0], tf.float32)
    w = tf.cast(TARGET_SIZE[1], tf.float32)
    cx = (w - 1.0) / 2.0
    cy = (h - 1.0) / 2.0

    scale = tf.random.stateless_uniform([], seed=[s0, s1 + 3], minval=0.8, maxval=1.2)
    inv_scale = tf.math.reciprocal(scale)

    max_dy = tf.cast(tf.round(0.2 * h), tf.int32)
    max_dx = tf.cast(tf.round(0.2 * w), tf.int32)
    dy = tf.cast(
        tf.random.stateless_uniform(
            [], seed=[s0, s1 + 4], minval=-max_dy, maxval=max_dy + 1, dtype=tf.int32
        ),
        tf.float32,
    )
    dx = tf.cast(
        tf.random.stateless_uniform(
            [], seed=[s0, s1 + 5], minval=-max_dx, maxval=max_dx + 1, dtype=tf.int32
        ),
        tf.float32,
    )

    a00 = c * inv_scale
    a01 = -s * inv_scale
    a10 = s * inv_scale
    a11 = c * inv_scale

    a02 = cx - a00 * cx - a01 * cy - dx
    a12 = cy - a10 * cx - a11 * cy - dy

    transform = tf.stack([a00, a01, a02, a10, a11, a12, 0.0, 0.0])

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(img, 0),
        transforms=tf.expand_dims(transform, 0),
        output_shape=tf.constant(TARGET_SIZE, dtype=tf.int32),
        interpolation="NEAREST",
        fill_mode="NEAREST",
        fill_value=0.0,
    )
    img = tf.squeeze(img, 0)
    return img


def _preprocess(img):
    return preprocess_input(img)


def _parse_train(ex):
    ex = tf.io.parse_single_example(ex, _FEATURES_TRAIN)
    img = _decode_and_resize(ex["image"])
    label = tf.where(ex["label"] >= 0, ex["label"], ex["target"])
    label = tf.cast(label, tf.float32)  # keep identical dtype/signature
    return img, label


def _parse_test(ex):
    ex = tf.io.parse_single_example(ex, _FEATURES_TEST)
    img = _decode_and_resize(ex["image"])
    return img, ex["image_name"]


def _with_stateless_seed(idx):
    idx = tf.cast(idx, tf.int32)
    return tf.stack([tf.cast(SEED, tf.int32), idx], axis=0)


train_tfrecs = tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec"))
test_tfrecs = tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec"))
assert len(train_tfrecs) > 0, "No train tfrecords found"
assert len(test_tfrecs) > 0, "No test tfrecords found"
train_tfrecs = sorted(train_tfrecs)
test_tfrecs = sorted(test_tfrecs)

n_train_files = int(len(train_tfrecs) * 0.8)
train_files = train_tfrecs[:n_train_files]
val_files = (
    train_tfrecs[n_train_files:]
    if n_train_files < len(train_tfrecs)
    else train_tfrecs[-1:]
)


@tf.function
def _train_map_fused(i, ex):
    img, label = _parse_train(ex)
    img = _augment_equivalent(img, _with_stateless_seed(i))
    img = _preprocess(img)
    return img, label


@tf.function
def _val_map_fused(ex):
    img, label = _parse_train(ex)
    img = _preprocess(img)
    return img, label


@tf.function
def _test_map_fused(ex):
    img, name = _parse_test(ex)
    img = _preprocess(img)
    return img, name


_ds_opts = tf.data.Options()
_ds_opts.experimental_deterministic = True
_ds_opts.experimental_optimization.apply_default_optimizations = True
_ds_opts.experimental_optimization.map_fusion = True
_ds_opts.experimental_optimization.map_parallelization = True

_TFREC_READ_BUFFER = 8 * 1024 * 1024  # 8MB


def _make_train_ds(files):
    ds = tf.data.TFRecordDataset(
        files, num_parallel_reads=AUTO, buffer_size=_TFREC_READ_BUFFER
    )
    ds = ds.with_options(_ds_opts)
    ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.enumerate()
    ds = ds.map(_train_map_fused, num_parallel_calls=AUTO)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


def _make_val_ds(files):
    ds = tf.data.TFRecordDataset(
        files, num_parallel_reads=AUTO, buffer_size=_TFREC_READ_BUFFER
    )
    ds = ds.with_options(_ds_opts)
    ds = ds.map(_val_map_fused, num_parallel_calls=AUTO)
    ds = (
        ds.cache()
    )  # validation has no augmentation; safe and avoids re-decoding every epoch
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


train_ds = _make_train_ds(train_files)
val_ds = _make_val_ds(val_files)

STEPS_PER_EPOCH = None
VALIDATION_STEPS = None

print("Train TFRecords:", len(train_files), "Val TFRecords:", len(val_files))
print(
    "STEPS_PER_EPOCH:",
    STEPS_PER_EPOCH,
    "VALIDATION_STEPS:",
    VALIDATION_STEPS,
    "EPOCHS:",
    EPOCHS,
)




## === cell 10
candidate_notop_weights = [
    "/kaggle/input/efficientnetb0/efficientnetb0_notop.h5",
    "/kaggle/input/efficientnetb0-notop/efficientnetb0_notop.h5",
]
weights_path = first_existing_path(candidate_notop_weights)

if weights_path is not None:
    eff_weights = weights_path
    print("Using EfficientNetB0 notop weights from:", eff_weights)
else:
    eff_weights = "imagenet"
    print(
        "EfficientNetB0 notop weights file not found; falling back to weights='imagenet'."
    )

basemodel = EfficientNetB0(
    weights=eff_weights,
    include_top=False,
    input_shape=TARGET_SIZE + (3,),
)

headmodel = layers.GlobalAveragePooling2D()(basemodel.output)
headmodel = layers.Dense(5, activation="softmax")(headmodel)
model = keras.Model(inputs=basemodel.input, outputs=headmodel)




## === cell 11
candidate_finetuned = [
    "../input/best-weights-efficient/best.h5",
    "/kaggle/input/best-weights-efficient/best.h5",
    "/kaggle/input/best-weights-efficientnet/best.h5",
]
finetuned_path = first_existing_path(candidate_finetuned)

if finetuned_path is not None:
    try:
        model.load_weights(finetuned_path)
        print("Loaded fine-tuned weights from:", finetuned_path)
    except Exception as e:
        print(
            "Warning: found fine-tuned weights but failed to load; continuing without them."
        )
        print("Load error:", repr(e))
else:
    print("Fine-tuned weights not found; continuing with base weights.")

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)




## === cell 12
model_save = ModelCheckpoint(
    "./best_weights.h5",
    save_best_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1,
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-6,
    mode="min",
    verbose=1,
)
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    mode="min",
    verbose=1,
    restore_best_weights=True,
)

history = model.fit(
    train_ds,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=val_ds,
    validation_steps=VALIDATION_STEPS,
    callbacks=[model_save, early_stop, reduce_lr],
)

model.save("model.h5")

if os.path.exists("./best_weights.h5"):
    try:
        model.load_weights("./best_weights.h5")
        print("Loaded best checkpoint from ./best_weights.h5 for inference.")
    except Exception as e:
        print("Warning: failed to load ./best_weights.h5; using current model weights.")
        print("Load error:", repr(e))




## === cell 13
ss = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))

test_ds = tf.data.TFRecordDataset(
    test_tfrecs, num_parallel_reads=AUTO, buffer_size=_TFREC_READ_BUFFER
)
test_ds = test_ds.with_options(_ds_opts)
test_ds = test_ds.map(_test_map_fused, num_parallel_calls=AUTO)
test_ds = test_ds.batch(64, drop_remainder=False)
test_ds = test_ds.prefetch(AUTO)


@tf.function
def _drop_name(x, name):
    return x


proba = model.predict(test_ds.map(_drop_name, num_parallel_calls=AUTO), verbose=0)
test_preds = np.argmax(proba, axis=1).astype(int)

ss["label"] = test_preds[: len(ss)]
ss.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", ss.shape)
print(ss.head())
assert os.path.exists("submission.csv") and os.path.getsize("submission.csv") > 0
assert list(ss.columns) == ["image_id", "label"]
assert len(ss) == len(pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv")))
