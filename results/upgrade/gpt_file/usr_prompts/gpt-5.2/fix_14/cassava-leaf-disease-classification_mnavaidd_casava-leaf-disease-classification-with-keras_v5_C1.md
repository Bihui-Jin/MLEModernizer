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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.0018

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.09753) has done: 'The timeout is dominated by training a very large ResNet152V2 at 320×320 for 20 epochs, plus slower Python-side image augmentation/loading from `ImageDataGenerator`. To keep core logic identical while cutting wall time, the key changes are: (1) freeze the ImageNet base during the existing training loop (same architecture/loss/optimizer/epochs, but far fewer trainable ops), and (2) switch the training/validation input pipeline to a `tf.data` pipeline that performs the same preprocessing and augmentations using TensorFlow ops with parallelism/prefetch (removes Python generator overhead). I also remove plotting/`model.save()`+reload from the critical path (they don’t affect predictions) and ensure deterministic settings remain in place. Prediction already uses `tf.data`; I keep it but add `.cache()` for the test decode/resize to avoid any repeated work.'
- What this solution (achieved 0.11584) has done: 'I fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation early, which avoids the `MessageFactory.GetPrototype` incompatibility in this environment. Then I fix the `tf.image.rotate` error by switching to `tensorflow_addons.image.rotate` (available on Kaggle for this competition) so your existing augmentation logic still works. Finally, I ensure the dataset build cell completes so `val_ds` is defined (the earlier map error prevented it), and keep the rest of the model/training/prediction logic unchanged so the score behavior stays consistent while producing a valid `submission.csv`.'
- What this solution (achieved 0.10912) has done: 'I fix the TensorFlow/protobuf crash by setting the additional environment flag that forces the pure-Python protobuf implementation consistently in this Kaggle runtime. Then I fix the `tfa` NameError by making the `tensorflow_addons` import reliable and re-importing it right before the augmentation pipeline is defined, so the `tf.data` map function can resolve it during tracing. These changes unblock dataset creation so `val_ds` is defined and training proceeds unchanged (same model, loss, optimizer, epochs, and augmentation logic), producing a valid `submission.csv`. I not make any score-tuning changes since the primary issue is runtime failure and your current score already comes from a working variant.'
- What this solution (achieved 0.09753) has done: 'The timeout is dominated by an extremely expensive input pipeline: decoding + resizing 320×320 images and then running heavy, per-sample projective transforms (rotate + shear/zoom) for every epoch, which becomes the bottleneck even though the backbone is frozen. To preserve core logic and accuracy, the main speedups are: (1) replace the slow custom rotation/shear projective code with the graph-fused `tf.image.*` equivalents (same semantics: rotate, translate, shear, zoom, flips, brightness/contrast-like scaling), (2) restructure the dataset so we don’t keep re-creating path datasets and we apply `.with_options()` once at the end, and (3) remove the “warm-up” full materialization loops that force cache population up-front. These changes are provably equivalent at the algorithm level (same augmentations and training loop), but reduce Python overhead and expensive raw ops, improving throughput enough to fit in 600 seconds.'
- What this solution (achieved 0.16629) has done: 'I fix the two root runtime blockers so the notebook runs end-to-end: the TensorFlow/protobuf crash (by forcing the Python protobuf implementation *before* any TF import, and doing it in a way that works reliably in this Kaggle runtime), and the missing `tf.image.rotate` API in TF 2.18 (by using `tensorflow_addons.image.rotate`, which preserves the intended augmentation behavior). Once the `train_ds/val_ds` build succeeds, the downstream `NameError`s in training/prediction disappear automatically. I keep the model, training loop, epochs, optimizer, and data split unchanged to avoid unnecessary score changes (your score is already far above the target band). The script still write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.10127) has done: 'I fix the early TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation in a way that works reliably in this Kaggle runtime (must be set before importing `google.protobuf`/`tensorflow`). Then I fix the `tfa` scoping/tracing `NameError` by importing `tensorflow_addons` inside the augmentation function (so the traced graph can always resolve it), while keeping the same augmentation behavior. These changes unblock `train_ds/val_ds` creation so training and inference run end-to-end unchanged, and the script writes a valid `submission.csv` in `/kaggle/working`. No score-tuning changes are introduced since your current score is already far above the target band and the goal is stability/correctness.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("PYTHONHASHSEED", "42")

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2

import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, GlobalAveragePooling2D

random.seed(42)
np.random.seed(42)
tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    for _gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_path = "../input/cassava-leaf-disease-classification/"
train_csv_data_path = data_path + "train.csv"
label_json_data_path = data_path + "label_num_to_disease_map.json"
images_dir_data_path = data_path + "train_images"

train_tfrecords_dir = data_path + "train_tfrecords/"
test_tfrecords_dir = data_path + "test_tfrecords/"



## === cell 2
train_csv = pd.read_csv(train_csv_data_path)
train_csv["label"] = train_csv["label"].astype("string")

label_class = pd.read_json(label_json_data_path, orient="index")
label_class = label_class.values.flatten().tolist()



## === cell 3
train_data_label_3 = train_csv[train_csv["label"] == "3"]
train_data_label_3 = shuffle(train_data_label_3, random_state=42)
train_data_label_3 = train_data_label_3[:3000]

train_data_label_not_3 = train_csv[train_csv["label"] != "3"]

train_csv = pd.concat([train_data_label_3, train_data_label_not_3], ignore_index=True)



## === cell 4
print("Label names :")
for i, label in enumerate(label_class):
    print(f" {i}. {label}")



## === cell 5
train_csv.head()



## === cell 6
BATCH_SIZE = 18
IMG_SIZE = 320
VAL_SPLIT = 0.15

train_csv_sorted = train_csv.sort_values("image_id").reset_index(drop=True)
n_total = len(train_csv_sorted)
n_val = int(np.floor(n_total * VAL_SPLIT))

val_df = train_csv_sorted.iloc[:n_val].copy()
train_df = train_csv_sorted.iloc[n_val:].copy()

train_paths = (images_dir_data_path + "/" + train_df["image_id"]).to_numpy()
val_paths = (images_dir_data_path + "/" + val_df["image_id"]).to_numpy()

train_labels_int = train_df["label"].astype(int).to_numpy()
val_labels_int = val_df["label"].astype(int).to_numpy()

NUM_CLASSES = 5



## === cell 7
AUTOTUNE = tf.data.AUTOTUNE


def _seed_from_path(path):
    h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
    return tf.stack([tf.cast(h, tf.int32), tf.constant(42, tf.int32)], axis=0)


def _decode_resize_rescale_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img,
        [IMG_SIZE, IMG_SIZE],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _decode_resize_rescale(path):
    img_bytes = tf.io.read_file(path)
    return _decode_resize_rescale_from_bytes(img_bytes)


def _apply_projective(img, transform):
    transforms = tf.reshape(transform, [1, 8])
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(img, 0),
        transforms=transforms,
        output_shape=tf.constant([IMG_SIZE, IMG_SIZE], tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    return tf.squeeze(out, axis=0)


def _augment(img, seed):
    angle = tf.random.stateless_uniform(
        [], seed=seed, minval=-np.pi, maxval=np.pi, dtype=tf.float32
    )
    ca = tf.cos(angle)
    sa = tf.sin(angle)

    cx = (IMG_SIZE - 1) / 2.0
    cy = (IMG_SIZE - 1) / 2.0

    a0 = ca
    a1 = -sa
    b0 = sa
    b1 = ca
    a2 = cx - a0 * cx - a1 * cy
    b2 = cy - b0 * cx - b1 * cy
    rot_transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])
    img = _apply_projective(img, rot_transform)

    s2 = seed + tf.constant([1, 0], tf.int32)
    tx = (
        tf.random.stateless_uniform(
            [], seed=s2, minval=-0.1, maxval=0.1, dtype=tf.float32
        )
        * IMG_SIZE
    )
    s3 = seed + tf.constant([2, 0], tf.int32)
    ty = (
        tf.random.stateless_uniform(
            [], seed=s3, minval=-0.1, maxval=0.1, dtype=tf.float32
        )
        * IMG_SIZE
    )
    img = tf.roll(img, shift=tf.cast(tf.stack([ty, tx]), tf.int32), axis=[0, 1])

    s4 = seed + tf.constant([3, 0], tf.int32)
    b = tf.random.stateless_uniform(
        [], seed=s4, minval=0.1, maxval=0.9, dtype=tf.float32
    )
    img = tf.clip_by_value(img * b, 0.0, 1.0)

    s5 = seed + tf.constant([4, 0], tf.int32)
    shear = tf.random.stateless_uniform(
        [], seed=s5, minval=-25.0, maxval=25.0, dtype=tf.float32
    ) * (np.pi / 180.0)
    s6 = seed + tf.constant([5, 0], tf.int32)
    zoom = tf.random.stateless_uniform(
        [], seed=s6, minval=1.0 - 0.3, maxval=1.0 + 0.3, dtype=tf.float32
    )

    sh = tf.tan(shear)

    fa0 = zoom
    fa1 = zoom * sh
    fb0 = 0.0
    fb1 = zoom
    fa2 = cx - fa0 * cx - fa1 * cy
    fb2 = cy - fb0 * cx - fb1 * cy

    det = fa0 * fb1 - fa1 * fb0
    ia0 = fb1 / det
    ia1 = -fa1 / det
    ib0 = -fb0 / det
    ib1 = fa0 / det
    ia2 = -(ia0 * fa2 + ia1 * fb2)
    ib2 = -(ib0 * fa2 + ib1 * fb2)

    aff_transform = tf.stack([ia0, ia1, ia2, ib0, ib1, ib2, 0.0, 0.0])
    img = _apply_projective(img, aff_transform)

    s7 = seed + tf.constant([6, 0], tf.int32)
    cshift = tf.random.stateless_uniform(
        [1, 1, 3], seed=s7, minval=-0.1, maxval=0.1, dtype=tf.float32
    )
    img = tf.clip_by_value(img + cshift, 0.0, 1.0)

    s8 = seed + tf.constant([7, 0], tf.int32)
    do_h = tf.random.stateless_uniform([], seed=s8, minval=0.0, maxval=1.0) < 0.5
    img = tf.cond(do_h, lambda: tf.image.flip_left_right(img), lambda: img)

    s9 = seed + tf.constant([8, 0], tf.int32)
    do_v = tf.random.stateless_uniform([], seed=s9, minval=0.0, maxval=1.0) < 0.5
    img = tf.cond(do_v, lambda: tf.image.flip_up_down(img), lambda: img)

    return img


def _train_from_cached(img, path, y):
    seed = _seed_from_path(path)
    img = _augment(img, seed)
    y = tf.one_hot(tf.cast(y, tf.int32), NUM_CLASSES)
    return img, y


def _val_from_cached(img, y):
    y = tf.one_hot(tf.cast(y, tf.int32), NUM_CLASSES)
    return img, y


options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.parallel_batch = True
options.experimental_optimization.map_parallelization = True

train_id_to_label = dict(zip(train_df["image_id"].tolist(), train_labels_int.tolist()))
val_id_to_label = dict(zip(val_df["image_id"].tolist(), val_labels_int.tolist()))


def _make_label_table(dct):
    keys = tf.constant(list(dct.keys()), dtype=tf.string)
    vals = tf.constant(list(dct.values()), dtype=tf.int64)
    init = tf.lookup.KeyValueTensorInitializer(keys, vals)
    return tf.lookup.StaticHashTable(init, default_value=tf.constant(-1, tf.int64))


train_label_table = _make_label_table(train_id_to_label)
val_label_table = _make_label_table(val_id_to_label)


def _parse_train_tfrec(example_proto):
    feats = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_id": tf.io.FixedLenFeature([], tf.string),
    }
    ex = tf.io.parse_single_example(example_proto, feats)
    img = _decode_resize_rescale_from_bytes(ex["image"])
    image_id = ex["image_id"]
    y = train_label_table.lookup(image_id)
    path = tf.strings.join([images_dir_data_path, "/", image_id])
    return img, path, y


def _parse_val_tfrec(example_proto):
    feats = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_id": tf.io.FixedLenFeature([], tf.string),
    }
    ex = tf.io.parse_single_example(example_proto, feats)
    img = _decode_resize_rescale_from_bytes(ex["image"])
    y = val_label_table.lookup(ex["image_id"])
    return img, y


train_cache_path = "/kaggle/working/train_decoded_cache"
val_cache_path = "/kaggle/working/val_decoded_cache"

train_tfrec_files = tf.io.gfile.glob(train_tfrecords_dir + "*.tfrec")
train_tfrec_files = sorted(train_tfrec_files)

val_tfrec_files = train_tfrec_files  # same source; labels filtered via table

train_raw = tf.data.TFRecordDataset(train_tfrec_files, num_parallel_reads=AUTOTUNE)
val_raw = tf.data.TFRecordDataset(val_tfrec_files, num_parallel_reads=AUTOTUNE)

shuffle_buf = int(min(len(train_paths), 4096))

train_ds = (
    train_raw.map(_parse_train_tfrec, num_parallel_calls=AUTOTUNE)
    .filter(lambda img, path, y: y >= 0)
    .shuffle(buffer_size=shuffle_buf, seed=42, reshuffle_each_iteration=True)
    .cache(train_cache_path)
    .map(
        lambda img, path, y: _train_from_cached(img, path, y),
        num_parallel_calls=AUTOTUNE,
    )
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
    .with_options(options)
)

val_ds = (
    val_raw.map(_parse_val_tfrec, num_parallel_calls=AUTOTUNE)
    .filter(lambda img, y: y >= 0)
    .cache(val_cache_path)
    .map(_val_from_cached, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
    .with_options(options)
)



## === cell 8
for _x, _y in train_ds.take(1):
    images, labels = _x, _y
    break
print("Sanity check batch shapes:", images.shape, labels.shape)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/828420230.py in <cell line: 0>()
----> 1 for _x, _y in train_ds.take(1):
      2     images, labels = _x, _y
      3     break
      4 print("Sanity check batch shapes:", images.shape, labels.shape)
      5 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in __next__(self)
    824   def __next__(self):
    825     try:
--> 826       return self._next_internal()
    827     except errors.OutOfRangeError:
    828       raise StopIteration

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in _next_internal(self)
    774     # to communicate that there is no more data to iterate over.
    775     with context.execution_mode(context.SYNC):
--> 776       ret = gen_dataset_ops.iterator_get_next(
    777           self._iterator_resource,
    778           output_types=self._flat_output_types,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_dataset_ops.py in iterator_get_next(iterator, output_types, output_shapes, name)
   3084       return _result
   3085     except _core._NotOkStatusException as e:
-> 3086       _ops.raise_from_not_ok_status(e, name)
   3087     except _core._FallbackException:
   3088       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to ParallelMapDatasetV2:4 transformation with iterator: Iterator::Root::Prefetch::FiniteTake::Prefetch::MapAndBatch::FileCacheImpl::Shuffle::Filter::ParallelMapV2: Feature: image_id (data type: string) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]] [Op:IteratorGetNext] name: 

## === cell 9
pass



## === cell 10
base = tf.keras.applications.ResNet152V2(
    include_top=False, weights="imagenet", input_shape=[IMG_SIZE, IMG_SIZE, 3]
)

base.trainable = False



## === cell 11
base.summary()



## === cell 12
model = tf.keras.Sequential()
model.add(base)
model.add(BatchNormalization(axis=-1))
model.add(GlobalAveragePooling2D())
model.add(Dense(5, activation="softmax"))



## === cell 13
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adamax(learning_rate=0.01),
    metrics=["acc"],
)



## === cell 14
model.summary()



## === cell 15
history = model.fit(
    train_ds,
    epochs=20,
    validation_data=val_ds,
)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
AlreadyExistsError                        Traceback (most recent call last)
/tmp/ipykernel_11/3844465766.py in <cell line: 0>()
----> 1 history = model.fit(
      2     train_ds,
      3     epochs=20,
      4     validation_data=val_ds,
      5 )

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

AlreadyExistsError: Graph execution error:

Detected at node IteratorGetNext defined at (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main

  File "<frozen runpy>", line 88, in _run_code

  File "/usr/local/lib/python3.11/dist-packages/colab_kernel_launcher.py", line 37, in <module>

  File "/usr/local/lib/python3.11/dist-packages/traitlets/config/application.py", line 992, in launch_instance

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelapp.py", line 712, in start

  File "/usr/local/lib/python3.11/dist-packages/tornado/platform/asyncio.py", line 211, in start

  File "/usr/lib/python3.11/asyncio/base_events.py", line 608, in run_forever

  File "/usr/lib/python3.11/asyncio/base_events.py", line 1936, in _run_once

  File "/usr/lib/python3.11/asyncio/events.py", line 84, in _run

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 510, in dispatch_queue

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 499, in process_one

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 406, in dispatch_shell

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 730, in execute_request

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/ipkernel.py", line 383, in do_execute

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/zmqshell.py", line 528, in run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 2975, in run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3030, in _run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/async_helpers.py", line 78, in _pseudo_sync_runner

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3257, in run_cell_async

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3473, in run_ast_nodes

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3553, in run_code

  File "/tmp/ipykernel_11/3844465766.py", line 1, in <cell line: 0>

  File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 371, in fit

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 219, in function

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 132, in multi_step_on_iterator

Detected at node IteratorGetNext defined at (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main

  File "<frozen runpy>", line 88, in _run_code

  File "/usr/local/lib/python3.11/dist-packages/colab_kernel_launcher.py", line 37, in <module>

  File "/usr/local/lib/python3.11/dist-packages/traitlets/config/application.py", line 992, in launch_instance

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelapp.py", line 712, in start

  File "/usr/local/lib/python3.11/dist-packages/tornado/platform/asyncio.py", line 211, in start

  File "/usr/lib/python3.11/asyncio/base_events.py", line 608, in run_forever

  File "/usr/lib/python3.11/asyncio/base_events.py", line 1936, in _run_once

  File "/usr/lib/python3.11/asyncio/events.py", line 84, in _run

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 510, in dispatch_queue

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 499, in process_one

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 406, in dispatch_shell

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 730, in execute_request

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/ipkernel.py", line 383, in do_execute

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/zmqshell.py", line 528, in run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 2975, in run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3030, in _run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/async_helpers.py", line 78, in _pseudo_sync_runner

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3257, in run_cell_async

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3473, in run_ast_nodes

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3553, in run_code

  File "/tmp/ipykernel_11/3844465766.py", line 1, in <cell line: 0>

  File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 371, in fit

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 219, in function

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 132, in multi_step_on_iterator

2 root error(s) found.
  (0) ALREADY_EXISTS:  There appears to be a concurrent caching iterator running - cache lockfile already exists ('/kaggle/working/train_decoded_cache_0.lockfile'). If you are sure no other running TF computations are using this cache prefix, delete the lockfile and re-initialize the iterator. Lockfile contents: Created at: 1768374129
	 [[{{node IteratorGetNext}}]]
	 [[IteratorGetNext/_4]]
  (1) ALREADY_EXISTS:  There appears to be a concurrent caching iterator running - cache lockfile already exists ('/kaggle/working/train_decoded_cache_0.lockfile'). If you are sure no other running TF computations are using this cache prefix, delete the lockfile and re-initialize the iterator. Lockfile contents: Created at: 1768374129
	 [[{{node IteratorGetNext}}]]
0 successful operations.
0 derived errors ignored. [Op:__inference_multi_step_on_iterator_29943]

## === cell 16
pass



## === cell 17
pass



## === cell 18
pass



## === cell 19
ss = pd.read_csv(data_path + "sample_submission.csv")

test_tfrec_files = tf.io.gfile.glob(test_tfrecords_dir + "*.tfrec")
test_tfrec_files = sorted(test_tfrec_files)


def _parse_test_tfrec(example_proto):
    feats = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_id": tf.io.FixedLenFeature([], tf.string),
    }
    ex = tf.io.parse_single_example(example_proto, feats)
    img = _decode_resize_rescale_from_bytes(ex["image"])
    return img, ex["image_id"]


test_raw = tf.data.TFRecordDataset(test_tfrec_files, num_parallel_reads=AUTOTUNE)
test_cache_path = "/kaggle/working/test_decoded_cache"
test_ds = (
    test_raw.map(_parse_test_tfrec, num_parallel_calls=AUTOTUNE)
    .cache(test_cache_path)
    .batch(64, drop_remainder=False)
    .prefetch(AUTOTUNE)
    .with_options(options)
)

probs = model.predict(test_ds.map(lambda img, image_id: img), verbose=0)
preds = probs.argmax(axis=1).astype(np.int64)

test_ids = []
for _, batch_ids in test_ds.map(lambda img, image_id: image_id).unbatch().batch(1024):
    test_ids.append(batch_ids.numpy())
test_ids = np.concatenate(test_ids).astype("U")

pred_df = pd.DataFrame({"image_id": test_ids, "label": preds})
my_submission = ss[["image_id"]].merge(pred_df, on="image_id", how="left")
my_submission["label"] = my_submission["label"].astype(np.int64)

my_submission.to_csv("submission.csv", index=False)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/2141695956.py in <cell line: 0>()
     27 )
     28 
---> 29 probs = model.predict(test_ds.map(lambda img, image_id: img), verbose=0)
     30 preds = probs.argmax(axis=1).astype(np.int64)
     31 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to ParallelMapDatasetV2:30 transformation with iterator: Iterator::Root::ParallelMapV2::Prefetch::BatchV2::FileCacheImpl::ParallelMapV2: Feature: image_id (data type: string) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]] [Op:IteratorGetNext] name: 

## === cell 20
print("Submission File: \n---------------\n")
print(my_submission.head())  # Predicted Output
print("\nSaved to: /kaggle/working/submission.csv")

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3181959020.py in <cell line: 0>()
      1 print("Submission File: \n---------------\n")
----> 2 print(my_submission.head())  # Predicted Output
      3 print("\nSaved to: /kaggle/working/submission.csv")

NameError: name 'my_submission' is not defined
