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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
    Input,
)
from tensorflow.keras.models import Model

print("TF version:", tf.__version__)




## === cell 1
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

options = tf.data.Options()
options.experimental_deterministic = True
try:
    opt = options.experimental_optimization
    for k, v in {
        "apply_default_optimizations": True,
        "map_and_batch_fusion": True,
        "map_parallelization": True,
        "parallel_batch": True,
    }.items():
        if hasattr(opt, k):
            setattr(opt, k, v)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE




## === cell 2
data_path = "../input/cassava-leaf-disease-classification/"
train_csv_data_path = data_path + "train.csv"
label_json_data_path = data_path + "label_num_to_disease_map.json"
images_dir_data_path = data_path + "train_images/"

test_images_dir_data_path = data_path + "test_images/"
sample_sub_path = data_path + "sample_submission.csv"

train_tfrecords_dir = data_path + "train_tfrecords/"
test_tfrecords_dir = data_path + "test_tfrecords/"




## === cell 3
train_csv = pd.read_csv(train_csv_data_path)
train_csv["label"] = train_csv["label"].astype("string")

label_class = pd.read_json(label_json_data_path, orient="index")
label_class = label_class.values.flatten().tolist()




## === cell 4
train_csv = shuffle(train_csv, random_state=SEED).reset_index(drop=True)




## === cell 5
print("Label names :")
for i, label in enumerate(label_class):
    print(f" {i}. {label}")




## === cell 6
train_csv.head()




## === cell 7
print("Train size:", len(train_csv))
print(train_csv["label"].value_counts())




## === cell 8
BATCH_SIZE = 18
IMG_SIZE = 224
N_CLASSES = 5
VAL_SPLIT = 0.15




## === cell 9
n_total = len(train_csv)
n_valid = int(round(n_total * VAL_SPLIT))
valid_df = train_csv.iloc[:n_valid].copy()
train_df = train_csv.iloc[n_valid:].copy()

print("Split sizes | train:", len(train_df), "valid:", len(valid_df))

N_WORKERS = min(8, max(1, (os.cpu_count() or 2) - 1))
print("CPU count:", os.cpu_count(), "| N_WORKERS:", N_WORKERS)




## === cell 10
def _tfrecord_files(pattern_dir):
    pattern = os.path.join(pattern_dir, "*.tfrec")
    files = tf.io.gfile.glob(pattern)
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No TFRecords found at: {pattern}")
    return files


train_tfrec_files = _tfrecord_files(train_tfrecords_dir)
test_tfrec_files = _tfrecord_files(test_tfrecords_dir)

print("Train TFRecords:", len(train_tfrec_files))
print("Test TFRecords:", len(test_tfrec_files))




## === cell 11
_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}

_decode_sig = [tf.TensorSpec([], tf.string)]
_img_lbl_name_sig = [
    tf.TensorSpec([IMG_SIZE, IMG_SIZE, 3], tf.float32),
    tf.TensorSpec([], tf.int32),
    tf.TensorSpec([], tf.string),
]


@tf.function(input_signature=_decode_sig)
def _decode_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    label = tf.cast(ex["target"], tf.int32)
    return img, label, ex["image_name"]


@tf.function
def _one_hot(img, label):
    return img, tf.one_hot(label, N_CLASSES, dtype=tf.float32)


@tf.function
def _random_channel_shift_stateless(img, seed2, max_shift=0.1):
    shift = tf.random.stateless_uniform(
        [3], seed=seed2, minval=-max_shift, maxval=max_shift
    )
    img = img + shift
    return tf.clip_by_value(img, 0.0, 1.0)


@tf.function
def _random_brightness_scale_stateless(img, seed2, lo=0.1, hi=0.9):
    scale = tf.random.stateless_uniform([], seed=seed2, minval=lo, maxval=hi)
    img = img * scale
    return tf.clip_by_value(img, 0.0, 1.0)


@tf.function
def _projective_transform(img, transform):
    transform = tf.cast(transform, tf.float32)
    img4 = tf.expand_dims(img, 0)
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img4,
        transforms=tf.expand_dims(transform, 0),
        output_shape=[IMG_SIZE, IMG_SIZE],
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    return tf.squeeze(out, 0)


@tf.function
def _compose_affine(theta, shear, zoom, tx, ty):
    c = (IMG_SIZE - 1) / 2.0
    cos_t = tf.math.cos(theta)
    sin_t = tf.math.sin(theta)

    R = tf.stack([tf.stack([cos_t, -sin_t]), tf.stack([sin_t, cos_t])])
    Sh = tf.stack([tf.stack([1.0, tf.math.tan(shear)]), tf.stack([0.0, 1.0])])
    Z = tf.stack([tf.stack([zoom, 0.0]), tf.stack([0.0, zoom])])

    A = tf.linalg.matmul(R, tf.linalg.matmul(Sh, Z))  # 2x2

    center = tf.constant([c, c], tf.float32)
    shift = tf.stack([tx, ty])
    b = center + shift - tf.linalg.matvec(A, center)

    a0, a1 = A[0, 0], A[0, 1]
    a3, a4 = A[1, 0], A[1, 1]
    a2, a5 = b[0], b[1]
    return tf.stack([a0, a1, a2, a3, a4, a5, 0.0, 0.0])


@tf.function(input_signature=_img_lbl_name_sig)
def _augment_with_name_seed(img, label, image_name):
    h = tf.strings.to_hash_bucket_fast(image_name, 2**31 - 1)
    base_seed = tf.stack([tf.cast(SEED, tf.int32), tf.cast(h, tf.int32)])

    img = tf.image.stateless_random_flip_left_right(img, seed=base_seed)
    img = tf.image.stateless_random_flip_up_down(
        img, seed=base_seed + tf.constant([1, 0], tf.int32)
    )

    img = _random_channel_shift_stateless(
        img, base_seed + tf.constant([2, 0], tf.int32), max_shift=0.1
    )
    img = _random_brightness_scale_stateless(
        img, base_seed + tf.constant([3, 0], tf.int32), lo=0.1, hi=0.9
    )

    theta = tf.random.stateless_uniform(
        [],
        seed=base_seed + tf.constant([4, 0], tf.int32),
        minval=0.0,
        maxval=2.0 * tf.constant(np.pi, tf.float32),
    )
    shear = tf.random.stateless_uniform(
        [], seed=base_seed + tf.constant([5, 0], tf.int32), minval=-25.0, maxval=25.0
    ) * (tf.constant(np.pi, tf.float32) / 180.0)
    zoom = tf.random.stateless_uniform(
        [],
        seed=base_seed + tf.constant([6, 0], tf.int32),
        minval=1.0 - 0.3,
        maxval=1.0 + 0.3,
    )
    tx = (
        tf.random.stateless_uniform(
            [], seed=base_seed + tf.constant([7, 0], tf.int32), minval=-0.1, maxval=0.1
        )
        * IMG_SIZE
    )
    ty = (
        tf.random.stateless_uniform(
            [], seed=base_seed + tf.constant([8, 0], tf.int32), minval=-0.1, maxval=0.1
        )
        * IMG_SIZE
    )

    transform = _compose_affine(theta, shear, zoom, tx, ty)
    img = _projective_transform(img, transform)
    return img, label


def _split_tfrec_files(files, n_train_steps, n_valid_steps, batch_size):
    shard_examples = 1338
    need_train = int(n_train_steps * batch_size)
    need_valid = int(n_valid_steps * batch_size)

    n_train_shards = int(np.ceil(need_train / shard_examples))
    n_valid_shards = int(np.ceil(need_valid / shard_examples))

    train_files = files[:n_train_shards]
    valid_files = files[n_train_shards : n_train_shards + n_valid_shards]
    if not valid_files:
        valid_files = files[-1:]
    return train_files, valid_files


def _build_dataset_from_tfrec(files, training: bool, cache: bool = False):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE).with_options(
        options
    )

    if training:
        ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)

        def _train_map(example_proto):
            img, label, name = _decode_example(example_proto)
            img, label = _augment_with_name_seed(img, label, name)
            return _one_hot(img, label)

        ds = ds.map(_train_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    else:

        def _valid_decode_map(example_proto):
            img, label, _ = _decode_example(example_proto)
            return img, label

        ds = ds.map(_valid_decode_map, num_parallel_calls=AUTOTUNE, deterministic=True)
        if cache:
            ds = ds.cache()

        def _valid_onehot_map(img, label):
            return _one_hot(img, label)

        ds = ds.map(_valid_onehot_map, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 12
train_steps = int(np.ceil(len(train_df) / BATCH_SIZE))
valid_steps = int(np.ceil(len(valid_df) / BATCH_SIZE))

train_files_split, valid_files_split = _split_tfrec_files(
    train_tfrec_files, train_steps, valid_steps, BATCH_SIZE
)

train_ds = _build_dataset_from_tfrec(train_files_split, training=True, cache=False)
valid_ds = _build_dataset_from_tfrec(valid_files_split, training=False, cache=True)





## === cell 13
images, labels = next(iter(train_ds))
print("Batch images:", images.shape, "Batch labels:", labels.shape)




## === cell 14
print("Class indices:", {str(i): i for i in range(N_CLASSES)})




## === cell 15
SHOW_PLOTS = False

if SHOW_PLOTS:
    plt.figure(figsize=(12, 9))
    for i in range(min(6, images.shape[0])):
        plt.subplot(2, 3, i + 1)
        plt.axis("off")
        plt.imshow(images[i].numpy())
        plt.title(label_class[int(np.argmax(labels[i].numpy()))])
    plt.show()




## === cell 16
base = applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
)
base.trainable = False  # start with frozen backbone (stable and fast)

x = base.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
outputs = Dense(N_CLASSES, activation="softmax")(x)

model = Model(inputs=base.input, outputs=outputs)




## === cell 17
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adamax(learning_rate=0.01),
    metrics=["acc"],
)

model.summary()




## === cell 18
model_save = tf.keras.callbacks.ModelCheckpoint(
    filepath="best_model.weights.h5",
    save_best_only=True,
    save_weights_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1,
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=5,
    mode="min",
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




## === cell 19
EPOCHS = 8

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    steps_per_epoch=train_steps,
    validation_steps=valid_steps,
    epochs=EPOCHS,
    callbacks=[model_save, early_stop, reduce_lr],
    verbose=1,
)




## === cell 20
if os.path.exists("best_model.weights.h5"):
    model.load_weights("best_model.weights.h5")




## === cell 21
RUN_VALIDATION_REPORT = False

if RUN_VALIDATION_REPORT:
    from sklearn.metrics import classification_report, confusion_matrix

    Y_pred = model.predict(
        valid_ds,
        steps=valid_steps,
        verbose=1,
    )
    y_pred = np.argmax(Y_pred, axis=1)

    y_true = []
    for _, yb in valid_ds.unbatch().take(len(valid_df)):
        y_true.append(int(tf.argmax(yb).numpy()))
    y_true = np.array(y_true, dtype=np.int32)

    print("Confusion Matrix")
    print(confusion_matrix(y_true, y_pred[: len(y_true)]))

    target_names = [str(i) for i in range(N_CLASSES)]
    print(
        classification_report(y_true, y_pred[: len(y_true)], target_names=target_names)
    )




## === cell 22
ss = pd.read_csv(sample_sub_path)

if SHOW_PLOTS:
    example_image = ss.image_id.iloc[0]
    test_img_path = os.path.join(test_images_dir_data_path, example_image)

    img = cv2.imread(test_img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read test image at: {test_img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = (
        cv2.resize(img, (IMG_SIZE, IMG_SIZE)).reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
    )

    plt.figure(figsize=(8, 4))
    plt.title(f"TEST IMAGE: {example_image}")
    plt.axis("off")
    plt.imshow(resized_img[0])
    plt.show()




## === cell 23
@tf.function(input_signature=[tf.TensorSpec([], tf.string)])
def _decode_test_example(example_proto):
    ex = tf.io.parse_single_example(
        example_proto,
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        },
    )
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    return img, ex["image_name"]


test_ds = tf.data.TFRecordDataset(
    test_tfrec_files, num_parallel_reads=AUTOTUNE
).with_options(options)
test_ds = test_ds.map(
    _decode_test_example, num_parallel_calls=AUTOTUNE, deterministic=True
)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False)
test_ds = test_ds.prefetch(AUTOTUNE)

test_images_ds = test_ds.map(
    lambda x, n: x, num_parallel_calls=AUTOTUNE, deterministic=True
)
test_names_ds = test_ds.map(
    lambda x, n: n, num_parallel_calls=AUTOTUNE, deterministic=True
)

pred_probs = model.predict(test_images_ds, verbose=1)
preds = np.argmax(pred_probs, axis=1).astype(int)

names = np.concatenate([nb.numpy().astype("U") for nb in test_names_ds], axis=0)
pred_map = dict(zip(names.tolist(), preds.tolist()))

ordered_preds = ss["image_id"].map(pred_map).astype(int).values

my_submission = pd.DataFrame(
    {"image_id": ss["image_id"].values, "label": ordered_preds.tolist()}
)
my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)




## === cell 24
print("Submission File:\n---------------\n")
print(my_submission.head())
print("\nSubmission label distribution:")
print(my_submission["label"].value_counts().sort_index())
