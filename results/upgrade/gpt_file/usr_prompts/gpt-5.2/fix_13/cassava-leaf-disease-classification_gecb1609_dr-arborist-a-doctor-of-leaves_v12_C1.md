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
import warnings
from pathlib import Path

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adam

warnings.simplefilter("ignore")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(False)  # keep semantics stable; avoid XLA numerical diffs

AUTOTUNE = tf.data.AUTOTUNE

print("TensorFlow version:", tf.__version__)

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)



## === cell 1
WORK_DIR = "../input/cassava-leaf-disease-classification"
base_path = Path(WORK_DIR)

print("WORK_DIR exists:", os.path.exists(WORK_DIR))

with open(os.path.join(WORK_DIR, "label_num_to_disease_map.json")) as file:
    _label_map = json.loads(file.read())
print("Loaded label map keys:", sorted(list(_label_map.keys())))

train_labels = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
train_labels.head()



## === cell 2
BATCH_SIZE = 8
EPOCHS = 20
TARGET_SIZE = 350

STEPS_PER_EPOCH = None
VALIDATION_STEPS = None



## === cell 3
train_img_dir = base_path / "train_images"
test_img_dir = base_path / "test_images"



## === cell 4
train_df = train_labels.copy()
diseaseMapping = pd.read_json(base_path / "label_num_to_disease_map.json", typ="series")



## === cell 5
diseaseMapping



## === cell 6
mappingDict = diseaseMapping.to_dict()



## === cell 7
train_df.head()



## === cell 8
train_df = train_df.replace(mappingDict)



## === cell 9
labelCounts = train_df["label"].value_counts().reset_index()
labelCounts.columns = ["Label", "Number of Observations"]
labelCounts.head()



## === cell 10
uniqueIds = train_df["image_id"].nunique()
if uniqueIds == len(train_df):
    print("There are no repeating Image IDs in the dataset")
else:
    print(f"There are {len(train_df) - uniqueIds} repeating Image IDs")



## === cell 11
pass



## === cell 12
healthyImages = train_df[train_df["label"] == "Healthy"]["image_id"].to_list()
cbbImages = train_df[train_df["label"] == "Cassava Bacterial Blight (CBB)"][
    "image_id"
].to_list()
cbsdImages = train_df[train_df["label"] == "Cassava Brown Streak Disease (CBSD)"][
    "image_id"
].to_list()
cgmImages = train_df[train_df["label"] == "Cassava Green Mottle (CGM)"][
    "image_id"
].to_list()
cmdImages = train_df[train_df["label"] == "Cassava Mosaic Disease (CMD)"][
    "image_id"
].to_list()




## === cell 13
def showImages(images):
    return




## === cell 14
def showHistogram(sample_img, title):
    return




## === cell 15
def load_image_pil_rgb(image_path, target_size):
    raise NotImplementedError("Not used; tf.data pipeline handles image loading.")




## === cell 16
pass



## === cell 17
pass



## === cell 18
pass



## === cell 19
pass



## === cell 20
pass



## === cell 21
pass



## === cell 22
pass



## === cell 23
pass



## === cell 24
pass



## === cell 25
pass



## === cell 26
pass



## === cell 27
pass



## === cell 28
pass



## === cell 29
pass



## === cell 30
pass



## === cell 31
pass



## === cell 32
pass



## === cell 33
pass



## === cell 34
channelIntensityDf = pd.DataFrame(
    {
        "Leaf Type": ["Healthy", "CBB", "CBSD", "CGM", "CMD"],
        "Red Channel Mean": [108, 102, 106, 113, 110],
        "Green Channel Mean": [126, 117, 123, 128, 128],
        "Blue Channel Mean": [80, 66, 72, 85, 80],
    }
)
channelIntensityDf



## === cell 35
train_labels.label = train_labels.label.astype("str")



## === cell 36
from tensorflow.keras.layers import (
    RandomRotation,
    RandomZoom,
    RandomFlip,
    RandomTranslation,
)


def _apply_shear_x(image, shear):
    shape = tf.shape(image)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)
    cx = w / 2.0
    cy = h / 2.0

    M = tf.stack([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]])
    S = tf.stack([[1.0, shear, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    B = tf.stack([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])

    A = tf.linalg.matmul(tf.linalg.matmul(M, S), B)  # 3x3

    a0, a1, a2 = A[0, 0], A[0, 1], A[0, 2]
    b0, b1, b2 = A[1, 0], A[1, 1], A[1, 2]
    c0, c1 = A[2, 0], A[2, 1]
    transform = tf.stack([a0, a1, a2, b0, b1, b2, c0, c1])[tf.newaxis, :]

    image4 = image[tf.newaxis, ...]
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=image4,
        transforms=transform,
        output_shape=tf.cast(tf.stack([shape[0], shape[1]]), tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    return out[0]


_aug_rotation = RandomRotation(factor=0.25, fill_mode="reflect", seed=SEED)
_aug_zoom = RandomZoom(
    height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), fill_mode="reflect", seed=SEED
)
_aug_flip = RandomFlip(mode="horizontal_and_vertical", seed=SEED)
_aug_translate = RandomTranslation(
    height_factor=0.1, width_factor=0.1, fill_mode="reflect", seed=SEED
)


@tf.function(reduce_retracing=True)
def _decode_resize_rescale_from_bytes(jpeg_bytes):
    img = tf.image.decode_jpeg(jpeg_bytes, channels=3)
    img = tf.image.resize(
        img, [TARGET_SIZE, TARGET_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0  # rescale=1/255
    return img


@tf.function(reduce_retracing=True)
def _decode_resize_rescale(path):
    img = tf.io.read_file(path)
    return _decode_resize_rescale_from_bytes(img)


@tf.function(reduce_retracing=True)
def _augment(image, label, seed0, seed1):
    seed = tf.stack([seed0, seed1])
    image = _aug_rotation(image, training=True)
    image = _aug_zoom(image, training=True)
    image = _aug_flip(image, training=True)
    image = _aug_translate(image, training=True)

    shear = tf.random.stateless_uniform(
        [], seed=seed, minval=-0.1, maxval=0.1, dtype=tf.float32
    )
    image = _apply_shear_x(image, shear)
    return image, label


train_labels_shuf = train_labels.sample(frac=1.0, random_state=SEED).reset_index(
    drop=True
)

n_total = len(train_labels_shuf)
n_val = int(np.floor(0.2 * n_total))
n_train = n_total - n_val

train_df_split = train_labels_shuf.iloc[:n_train].reset_index(drop=True)
val_df_split = train_labels_shuf.iloc[n_train:].reset_index(drop=True)

classes = np.sort(train_labels["label"].unique())
class_to_index = {c: i for i, c in enumerate(classes)}
num_classes = len(classes)
assert num_classes == 5, f"Expected 5 classes, got {num_classes}"

train_paths = (
    train_img_dir.as_posix() + "/" + train_df_split["image_id"].astype(str)
).to_numpy()
train_y = train_df_split["label"].map(class_to_index).to_numpy(np.int64)

val_paths = (
    train_img_dir.as_posix() + "/" + val_df_split["image_id"].astype(str)
).to_numpy()
val_y = val_df_split["label"].map(class_to_index).to_numpy(np.int64)



## === cell 37
options = tf.data.Options()
options.experimental_deterministic = True
try:
    options.experimental_threading.private_threadpool_size = max(
        8, (os.cpu_count() or 8)
    )
    options.experimental_threading.max_intra_op_parallelism = 1
except Exception:
    pass

SHUFFLE_BUFFER = min(n_train, 2048)

train_tfrec_paths = tf.io.gfile.glob(str(base_path / "train_tfrecords" / "*.tfrec"))
test_tfrec_paths = tf.io.gfile.glob(str(base_path / "test_tfrecords" / "*.tfrec"))
train_tfrec_paths = sorted(train_tfrec_paths)
test_tfrec_paths = sorted(test_tfrec_paths)

assert len(train_tfrec_paths) > 0, "No train TFRecords found"
assert len(test_tfrec_paths) > 0, "No test TFRecords found"

train_id_to_label = dict(zip(train_df_split["image_id"].astype(str).values, train_y))
val_id_to_label = dict(zip(val_df_split["image_id"].astype(str).values, val_y))

train_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(list(train_id_to_label.keys()), dtype=tf.string),
        values=tf.constant(list(train_id_to_label.values()), dtype=tf.int64),
    ),
    default_value=tf.constant(-1, dtype=tf.int64),
)
val_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(list(val_id_to_label.keys()), dtype=tf.string),
        values=tf.constant(list(val_id_to_label.values()), dtype=tf.int64),
    ),
    default_value=tf.constant(-1, dtype=tf.int64),
)

feature_description_train = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}

feature_description_test = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
}


@tf.function(reduce_retracing=True)
def _parse_train_example(serialized):
    ex = tf.io.parse_single_example(serialized, feature_description_train)
    image_bytes = ex["image"]
    image_name = ex["image_name"]  # e.g. b"123.jpg"
    img = _decode_resize_rescale_from_bytes(image_bytes)
    return img, image_name


@tf.function(reduce_retracing=True)
def _parse_test_example(serialized):
    ex = tf.io.parse_single_example(serialized, feature_description_test)
    image_bytes = ex["image"]
    image_name = tf.where(
        tf.size(ex["image_name"]) > 0, ex["image_name"], ex["image_id"]
    )
    img = _decode_resize_rescale_from_bytes(image_bytes)
    return img, image_name


@tf.function(reduce_retracing=True)
def _train_map_and_augment(img, image_name):
    h = tf.strings.to_hash_bucket_fast(image_name, 2**31 - 1)
    seed0 = tf.cast(h % (2**31 - 1), tf.int32)
    seed1 = tf.cast((h // 7 + SEED) % (2**31 - 1), tf.int32)
    label = tf.cast(train_table.lookup(image_name), tf.int32)
    img, label = _augment(img, label, seed0, seed1)
    return img, label


@tf.function(reduce_retracing=True)
def _val_map(img, image_name):
    label = tf.cast(val_table.lookup(image_name), tf.int32)
    return img, label


def _is_in_train(img, image_name):
    return train_table.lookup(image_name) >= 0


def _is_in_val(img, image_name):
    return val_table.lookup(image_name) >= 0


raw_train = tf.data.TFRecordDataset(
    train_tfrec_paths, num_parallel_reads=AUTOTUNE
).with_options(options)
raw_train = raw_train.map(
    _parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=True
)
raw_train = raw_train.filter(_is_in_train)
raw_train = raw_train.shuffle(
    buffer_size=SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True
)
raw_train = raw_train.map(
    _train_map_and_augment, num_parallel_calls=AUTOTUNE, deterministic=True
)
train_ds = raw_train.cache()
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

raw_val = tf.data.TFRecordDataset(
    train_tfrec_paths, num_parallel_reads=AUTOTUNE
).with_options(options)
raw_val = raw_val.map(
    _parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=True
)
raw_val = raw_val.filter(_is_in_val)
raw_val = raw_val.map(_val_map, num_parallel_calls=AUTOTUNE, deterministic=True)
val_ds = raw_val.cache()
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

STEPS_PER_EPOCH = int(np.ceil(n_train / BATCH_SIZE))
VALIDATION_STEPS = int(np.ceil(n_val / BATCH_SIZE))

print("n_train:", n_train, "n_val:", n_val)
print("STEPS_PER_EPOCH:", STEPS_PER_EPOCH, "VALIDATION_STEPS:", VALIDATION_STEPS)



## === cell 38
train_generator = train_ds
validation_generator = val_ds




## === cell 39
def create_model():
    conv_base = EfficientNetB3(
        include_top=False, weights=None, input_shape=(TARGET_SIZE, TARGET_SIZE, 3)
    )
    x = conv_base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(5, activation="softmax")(x)
    model = models.Model(conv_base.input, x)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["acc"],
        run_eagerly=False,
        steps_per_execution=16,
    )
    return model




## === cell 40
model = create_model()
model.summary()

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS,
    verbose=2,
)



## === cell 41
model.save("./EffNetB0_512_8.h5")



## === cell 42
ss = pd.read_csv(os.path.join(WORK_DIR, "sample_submission.csv"))
ss.head()



## === cell 43
preds = []



## === cell 44
name_to_index = {n: i for i, n in enumerate(ss["image_id"].astype(str).values)}
num_test = len(ss)
probs_out = np.zeros((num_test, 5), dtype=np.float32)

test_raw = tf.data.TFRecordDataset(
    test_tfrec_paths, num_parallel_reads=AUTOTUNE
).with_options(options)
test_raw = test_raw.map(
    _parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=True
)
test_raw = test_raw.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

all_names = []
all_probs = []

for batch_imgs, batch_names in test_raw:
    batch_prob = model(batch_imgs, training=False).numpy()
    all_probs.append(batch_prob)
    all_names.append(batch_names.numpy())

all_probs = np.concatenate(all_probs, axis=0)
all_names = np.concatenate(all_names, axis=0).astype("U")

missing = 0
for i in range(len(all_names)):
    n = all_names[i]
    if n in name_to_index:
        idx = name_to_index[n]
        probs_out[idx] = all_probs[i]
    else:
        missing += 1

if missing:
    print(
        "Warning: test TFRecords contained",
        missing,
        "names not found in sample_submission.csv",
    )

probs = probs_out
preds = probs.argmax(axis=1).astype(np.int64).tolist()



## === cell 45
y_predict = probs
y_predict



## === cell 46
print("Number of predictions:", len(preds))
print("First 10 predictions:", preds[:10])



## === cell 47
assert len(preds) == len(ss), f"Pred length {len(preds)} != submission length {len(ss)}"



## === cell 48
ss["label"] = pd.Series(preds, dtype="int64")
ss.head()



## === cell 49
ss.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", ss.shape)

assert os.path.exists("submission.csv")
check = pd.read_csv("submission.csv")
assert list(check.columns) == ["image_id", "label"]
assert len(check) == 2676
print(check.head())
