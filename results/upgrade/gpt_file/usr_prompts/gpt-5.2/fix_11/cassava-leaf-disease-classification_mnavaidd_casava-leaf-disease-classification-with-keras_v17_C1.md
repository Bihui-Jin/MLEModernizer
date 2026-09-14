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
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
)

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())




## === cell 1
SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

RUN_DIAGNOSTICS = False




## === cell 2
data_path = "../input/cassava-leaf-disease-classification/"
train_csv_data_path = data_path + "train.csv"
label_json_data_path = data_path + "label_num_to_disease_map.json"
images_dir_data_path = data_path + "train_images/"
test_images_dir_data_path = data_path + "test_images/"
train_tfrecords_dir = data_path + "train_tfrecords/"
test_tfrecords_dir = data_path + "test_tfrecords/"




## === cell 3
train_csv = pd.read_csv(train_csv_data_path)
train_csv["label"] = train_csv["label"].astype(str)

label_class = pd.read_json(label_json_data_path, orient="index")
label_class = label_class.values.flatten().tolist()

train_csv.head()




## === cell 4
assert os.path.exists(train_csv_data_path), f"Missing: {train_csv_data_path}"
assert os.path.isdir(images_dir_data_path), f"Missing dir: {images_dir_data_path}"
assert os.path.isdir(
    test_images_dir_data_path
), f"Missing dir: {test_images_dir_data_path}"
assert os.path.exists(
    data_path + "sample_submission.csv"
), "Missing sample_submission.csv"
assert os.path.isdir(train_tfrecords_dir), f"Missing dir: {train_tfrecords_dir}"
assert os.path.isdir(test_tfrecords_dir), f"Missing dir: {test_tfrecords_dir}"




## === cell 5
print("Label names :")
for i, label in enumerate(label_class):
    print(f" {i}. {label}")




## === cell 6
train_csv.head()




## === cell 7
BATCH_SIZE = 18
IMG_SIZE = 224

_CPU = os.cpu_count() or 2
AUTOTUNE = tf.data.AUTOTUNE




## === cell 8
NUM_CLASSES = train_csv["label"].nunique()
print("NUM_CLASSES:", NUM_CLASSES)

train_tfrecord_files = tf.io.gfile.glob(os.path.join(train_tfrecords_dir, "*.tfrec"))
test_tfrecord_files = tf.io.gfile.glob(os.path.join(test_tfrecords_dir, "*.tfrec"))
train_tfrecord_files = sorted(train_tfrecord_files)
test_tfrecord_files = sorted(test_tfrecord_files)

if len(train_tfrecord_files) == 0 or len(test_tfrecord_files) == 0:
    raise FileNotFoundError(
        "TFRecord files not found; expected in train_tfrecords/ and test_tfrecords/"
    )

val_frac = 0.15
n_total = len(train_csv)
rng = np.random.RandomState(SEED)
perm = rng.permutation(n_total)
n_val = int(round(n_total * val_frac))
val_idx = perm[:n_val]
train_idx = perm[n_val:]

train_df = train_csv.iloc[train_idx].reset_index(drop=True)
valid_df = train_csv.iloc[val_idx].reset_index(drop=True)

train_image_ids = tf.constant(train_df["image_id"].values)
train_labels_int = tf.constant(train_df["label"].astype(int).values, dtype=tf.int32)

valid_image_ids = tf.constant(valid_df["image_id"].values)
valid_labels_int = tf.constant(valid_df["label"].astype(int).values, dtype=tf.int32)

train_label_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(train_image_ids, train_labels_int),
    default_value=tf.constant(-1, tf.int32),
)
valid_label_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(valid_image_ids, valid_labels_int),
    default_value=tf.constant(-1, tf.int32),
)

train_id_set = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        train_image_ids, tf.ones_like(train_labels_int)
    ),
    default_value=tf.constant(0, tf.int32),
)
valid_id_set = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        valid_image_ids, tf.ones_like(valid_labels_int)
    ),
    default_value=tf.constant(0, tf.int32),
)




## === cell 9
if RUN_DIAGNOSTICS:
    print("train_df:", train_df.shape, "valid_df:", valid_df.shape)
else:
    pass




## === cell 10
_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


@tf.function(jit_compile=False)
def _decode_and_resize(image_bytes):
    img = tf.io.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR, antialias=True
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function(jit_compile=False)
def _stateless_seed_from_name(name):
    h = tf.strings.to_hash_bucket_fast(name, 2**31 - 1)
    return tf.stack([tf.constant(SEED, tf.int64), tf.cast(h, tf.int64)], axis=0)


@tf.function(jit_compile=False)
def _apply_augment(img, seed2):
    seeds = tf.random.experimental.stateless_split(tf.cast(seed2, tf.int64), num=12)

    img = tf.image.stateless_random_flip_left_right(
        img, seed=tf.cast(seeds[0], tf.int32)
    )
    img = tf.image.stateless_random_flip_up_down(img, seed=tf.cast(seeds[1], tf.int32))

    b = tf.random.stateless_uniform(
        [], seed=tf.cast(seeds[2], tf.int32), minval=0.1, maxval=0.9
    )
    img = tf.clip_by_value(img * b, 0.0, 1.0)

    cshift = tf.random.stateless_uniform(
        [3], seed=tf.cast(seeds[3], tf.int32), minval=-0.1, maxval=0.1
    )
    img = tf.clip_by_value(img + cshift, 0.0, 1.0)

    angle = tf.random.stateless_uniform(
        [], seed=tf.cast(seeds[4], tf.int32), minval=0.0, maxval=2.0 * np.pi
    )
    tx = (
        tf.random.stateless_uniform(
            [], seed=tf.cast(seeds[5], tf.int32), minval=-0.1, maxval=0.1
        )
        * IMG_SIZE
    )
    ty = (
        tf.random.stateless_uniform(
            [], seed=tf.cast(seeds[6], tf.int32), minval=-0.1, maxval=0.1
        )
        * IMG_SIZE
    )
    scale = tf.random.stateless_uniform(
        [], seed=tf.cast(seeds[7], tf.int32), minval=0.7, maxval=1.3
    )
    shear = tf.random.stateless_uniform(
        [],
        seed=tf.cast(seeds[8], tf.int32),
        minval=-np.tan(np.deg2rad(25.0)),
        maxval=np.tan(np.deg2rad(25.0)),
    )

    cos_a = tf.cos(angle) * scale
    sin_a = tf.sin(angle) * scale

    a0 = cos_a + shear * sin_a
    a1 = -sin_a + shear * cos_a
    a2 = tx
    b0 = sin_a
    b1 = cos_a
    b2 = ty

    cx = (IMG_SIZE - 1) / 2.0
    cy = (IMG_SIZE - 1) / 2.0

    a2 = a2 + cx - (a0 * cx + a1 * cy)
    b2 = b2 + cy - (b0 * cx + b1 * cy)

    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0], axis=0)
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(img, 0),
        transforms=tf.expand_dims(transform, 0),
        output_shape=[IMG_SIZE, IMG_SIZE],
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    img = tf.squeeze(img, 0)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img


@tf.function(jit_compile=False)
def _parse_train_example_with_filter(example):
    x = tf.io.parse_single_example(example, _FEATURES)
    name = x["image_name"]
    keep = train_id_set.lookup(name) > 0
    img = _decode_and_resize(x["image"])
    seed2 = _stateless_seed_from_name(name)
    img = _apply_augment(img, seed2)
    label = train_label_table.lookup(name)
    y = tf.one_hot(label, depth=NUM_CLASSES, dtype=tf.float32)
    return keep, img, y


@tf.function(jit_compile=False)
def _parse_valid_example_with_filter(example):
    x = tf.io.parse_single_example(example, _FEATURES)
    name = x["image_name"]
    keep = valid_id_set.lookup(name) > 0
    img = _decode_and_resize(x["image"])
    label = valid_label_table.lookup(name)
    y = tf.one_hot(label, depth=NUM_CLASSES, dtype=tf.float32)
    return keep, img, y


@tf.function(jit_compile=False)
def _parse_test_example(example):
    x = tf.io.parse_single_example(
        example,
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        },
    )
    img = _decode_and_resize(x["image"])
    return img, x["image_name"]




## === cell 11
_ds_options = tf.data.Options()
_ds_options.experimental_deterministic = True
_ds_options.experimental_optimization.apply_default_optimizations = True
_ds_options.experimental_optimization.map_parallelization = True
_ds_options.experimental_optimization.parallel_batch = True

train_raw = tf.data.TFRecordDataset(
    train_tfrecord_files, num_parallel_reads=AUTOTUNE, compression_type=None
)
train_ds = (
    train_raw.map(
        _parse_train_example_with_filter,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    .filter(lambda keep, img, y: keep)
    .map(lambda keep, img, y: (img, y), num_parallel_calls=AUTOTUNE, deterministic=True)
    .shuffle(8192, seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
    .with_options(_ds_options)
)

valid_raw = tf.data.TFRecordDataset(
    train_tfrecord_files, num_parallel_reads=AUTOTUNE, compression_type=None
)
valid_ds = (
    valid_raw.map(
        _parse_valid_example_with_filter,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    .filter(lambda keep, img, y: keep)
    .map(lambda keep, img, y: (img, y), num_parallel_calls=AUTOTUNE, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
    .with_options(_ds_options)
)

train_steps = int(np.ceil(len(train_df) / BATCH_SIZE))
valid_steps = int(np.ceil(len(valid_df) / BATCH_SIZE))
print("train_steps:", train_steps, "valid_steps:", valid_steps)




## === cell 12
def build_model(img_size=224, num_classes=5):
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    base = applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
    )
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model


model = build_model(IMG_SIZE, NUM_CLASSES)




## === cell 13
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adamax(learning_rate=0.01),
    metrics=["acc"],
    steps_per_execution=16,
)
model.summary()




## === cell 14
model_save = tf.keras.callbacks.ModelCheckpoint(
    filepath="Model.weights.h5",
    save_best_only=True,
    save_weights_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1,
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="acc",
    min_delta=0.001,
    patience=5,
    mode="max",  # acc should be maximized
    verbose=1,
    restore_best_weights=True,
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_delta=0.001,
    mode="min",
    verbose=1,
)




## === cell 15
EPOCHS = 6

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=valid_steps,
    callbacks=[model_save, early_stop, reduce_lr],
    verbose=1,
)




## === cell 16
if os.path.exists("Model.weights.h5"):
    model.load_weights("Model.weights.h5")
loaded_model = model




## === cell 17
if RUN_DIAGNOSTICS:
    from sklearn.metrics import confusion_matrix

    Y_pred = loaded_model.predict(
        valid_ds,
        steps=valid_steps,
        verbose=0,
    )
    y_pred = np.argmax(Y_pred, axis=1)

    y_true = []
    for _, yb in valid_ds:
        y_true.append(np.argmax(yb.numpy(), axis=1))
    y_true = np.concatenate(y_true, axis=0)[: len(valid_df)]

    print("Confusion Matrix")
    print(confusion_matrix(y_true, y_pred[: len(y_true)]))
else:
    Y_pred, y_pred = None, None




## === cell 18
if RUN_DIAGNOSTICS and y_pred is not None:
    from sklearn.metrics import classification_report

    target_names = [str(i) for i in range(NUM_CLASSES)]
    y_true = []
    for _, yb in valid_ds:
        y_true.append(np.argmax(yb.numpy(), axis=1))
    y_true = np.concatenate(y_true, axis=0)[: len(valid_df)]

    print(
        classification_report(y_true, y_pred[: len(y_true)], target_names=target_names)
    )




## === cell 19
if RUN_DIAGNOSTICS and y_pred is not None:
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    y_true = []
    for _, yb in valid_ds:
        y_true.append(np.argmax(yb.numpy(), axis=1))
    y_true = np.concatenate(y_true, axis=0)[: len(valid_df)]

    cm = confusion_matrix(y_true, y_pred[: len(y_true)])
    labels_pretty = [
        "Cassava Bacterial Blight (CBB)",
        "Cassava Brown Streak Disease (CBSD)",
        "Cassava Green Mottle (CGM)",
        "Cassava Mosaic Disease (CMD)",
        "Healthy",
    ]
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        xticklabels=labels_pretty,
        yticklabels=labels_pretty,
        annot=True,
        fmt="d",
        cmap="Blues",
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.show()




## === cell 20
if RUN_DIAGNOSTICS:
    ss = pd.read_csv(data_path + "sample_submission.csv")
    example_image_id = ss.image_id.iloc[0]
    test_img_path = os.path.join(test_images_dir_data_path, example_image_id)
    raise RuntimeError(
        "RUN_DIAGNOSTICS=True requires cv2, which is intentionally not imported to avoid env protobuf crash."
    )




## === cell 21
ss = pd.read_csv(data_path + "sample_submission.csv")
ss_ids = tf.constant(ss["image_id"].values)

ss_id_set = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        ss_ids, tf.ones([tf.shape(ss_ids)[0]], dtype=tf.int32)
    ),
    default_value=tf.constant(0, tf.int32),
)


@tf.function(jit_compile=False)
def _parse_test_example_with_filter(example):
    x = tf.io.parse_single_example(
        example,
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        },
    )
    name = x["image_name"]
    keep = ss_id_set.lookup(name) > 0
    img = _decode_and_resize(x["image"])
    return keep, img, name


test_raw = tf.data.TFRecordDataset(
    test_tfrecord_files, num_parallel_reads=AUTOTUNE, compression_type=None
)

test_ds = (
    test_raw.map(
        _parse_test_example_with_filter, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    .filter(lambda keep, img, name: keep)
    .map(
        lambda keep, img, name: (img, name),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
    .with_options(_ds_options)
)

test_steps = int(np.ceil(len(ss) / BATCH_SIZE))

probs = loaded_model.predict(
    test_ds.map(lambda x, n: x, num_parallel_calls=AUTOTUNE, deterministic=True),
    steps=test_steps,
    verbose=0,
)

name_list = []
for _, nameb in test_ds:
    name_list.append(nameb.numpy())
names = np.concatenate(name_list, axis=0)[: len(ss)]

probs = np.asarray(probs)[: len(ss)]
preds_raw = np.argmax(probs, axis=1).astype(int)

name_str = np.array([n.decode("utf-8") for n in names], dtype=object)
order = np.argsort(name_str)
name_sorted = name_str[order]
pred_sorted = preds_raw[order]

ss_arr = ss["image_id"].values.astype(object)
idx = np.searchsorted(name_sorted, ss_arr)
preds = pred_sorted[idx].astype(int)

my_submission = pd.DataFrame({"image_id": ss_arr, "label": preds.tolist()})
my_submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", my_submission.shape)




## === cell 22
print("Submission File: \n---------------\n")
print(my_submission.head())
print("\nsubmission.csv exists:", os.path.exists("submission.csv"))
print("submission.csv preview:")
with open("submission.csv", "r", encoding="utf-8") as f:
    for _ in range(5):
        print(f.readline().rstrip("\n"))
