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

0.7586884255061952

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.57661) has done: 'The main timeout culprit is stage-2 dataset creation from TFRecords: `enumerate().filter(reduce_any(equal(i, idx_keep)))` forces a very expensive per-example membership test and effectively scans all records multiple times. I keep the same stage-1/stage-2 models, losses, epochs, and overall training/prediction semantics, but rebuild the stage-2 input pipeline to select the exact training/validation examples by joining on `image_id` inside TFRecords (a provably equivalent split to your CSV-based split) rather than by global index filtering. I also remove an environment setting that forces the slow pure-Python protobuf implementation, and make caching behavior robust (cache only the decoded/resized images) to avoid repeated JPEG decode/resize overhead. These changes preserve core logic and accuracy while cutting input-pipeline cost enough to fit under 600 seconds.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

from pathlib import Path

print("TensorFlow:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE = Path("/kaggle/input/cassava-leaf-disease-classification")
train_csv_path = str(BASE / "train.csv")
label_json_path = str(BASE / "label_num_to_disease_map.json")
train_images_dir = BASE / "train_images"
test_images_dir = BASE / "test_images"
sample_sub_path = str(BASE / "sample_submission.csv")

assert Path(train_csv_path).exists(), f"Missing: {train_csv_path}"
assert Path(sample_sub_path).exists(), f"Missing: {sample_sub_path}"
assert train_images_dir.exists(), f"Missing: {train_images_dir}"
assert test_images_dir.exists(), f"Missing: {test_images_dir}"



## === cell 2
train_csv = pd.read_csv(train_csv_path)
train_csv["label"] = train_csv["label"].astype("string")
train_csv["label_int"] = train_csv["label"].astype(int)

label_class_j = pd.read_json(label_json_path, orient="index")
label_class = label_class_j.values.flatten().tolist()

print(train_csv.head())
print("Num classes:", len(label_class))



## === cell 3
SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)



## === cell 4
img_ids = train_csv.image_id.values[:8]
img_lbls = train_csv.label.values[:8]



## === cell 5
RUN_VIS = False

images_collection = []
if RUN_VIS:
    for img_id in img_ids:
        path = str(train_images_dir / str(img_id))
        img_arr = cv2.imread(path)
        if img_arr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        img_arr = cv2.resize(img_arr, (150, 150))
        images_collection.append(img_arr)



## === cell 6
if RUN_VIS:
    plt.figure(figsize=(18, 10))
    for i in range(8):
        plt.subplot(2, 4, i % 8 + 1)
        plt.imshow(images_collection[i])
        plt.title(label_class[int(img_lbls[i])])
        plt.axis("off")
    plt.show()



## === cell 7
from sklearn.model_selection import train_test_split

df_train, df_val = train_test_split(
    train_csv,
    test_size=0.15,
    random_state=SEED,
    stratify=train_csv["label_int"],
)

print("Train size:", len(df_train), "Val size:", len(df_val))
print(
    "Train label distribution:\n",
    df_train["label_int"].value_counts(normalize=True).sort_index(),
)



## === cell 8
NUM_CLASSES = 5
IMG_SIZE = 320
model_1_img_size = 32
model_2_img_size = 320

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE_1 = 128
BATCH_SIZE_2 = 16

_DS_OPTIONS = tf.data.Options()
try:
    _DS_OPTIONS.deterministic = False
except Exception:
    pass
try:
    _DS_OPTIONS.experimental_deterministic = False
except Exception:
    pass
try:
    _DS_OPTIONS.experimental_slack = True
except Exception:
    pass

try:
    _DS_OPTIONS.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass
try:
    _DS_OPTIONS.experimental_optimization.map_parallelization = True
except Exception:
    pass
try:
    _DS_OPTIONS.experimental_optimization.map_and_batch_fusion = True
except Exception:
    pass
try:
    _DS_OPTIONS.experimental_optimization.parallel_batch = True
except Exception:
    pass


@tf.function
def _read_image(path, size):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [size, size], method="bilinear")
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


@tf.function
def _augment(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img


def make_ds(
    df, images_root, size, batch_size, training, label_mode, cache=False, cache_name=""
):
    image_ids = df["image_id"].values
    paths = np.char.add(str(images_root) + "/", image_ids.astype(str))

    if label_mode == "binary_healthy":
        y = (df["label_int"].values == 4).astype(np.int32)
    elif label_mode == "multiclass":
        y = df["label_int"].values.astype(np.int32)
    else:
        raise ValueError("Unknown label_mode")

    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    ds = ds.with_options(_DS_OPTIONS)

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, lbl: (_read_image(p, tf.constant(size, tf.int32)), lbl),
        num_parallel_calls=AUTOTUNE,
        deterministic=False,
    )

    if training:
        ds = ds.map(
            lambda img, lbl: (_augment(img), lbl),
            num_parallel_calls=AUTOTUNE,
            deterministic=False,
        )

    if cache:
        ds = ds.cache(cache_name if cache_name else None)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds_1 = make_ds(
    df_train,
    train_images_dir,
    model_1_img_size,
    BATCH_SIZE_1,
    True,
    "binary_healthy",
    cache=False,
    cache_name="stage1_train",
)
val_ds_1 = make_ds(
    df_val,
    train_images_dir,
    model_1_img_size,
    BATCH_SIZE_1,
    False,
    "binary_healthy",
    cache=True,
    cache_name="stage1_val.cache",
)


def build_stage1(input_size):
    inp = tf.keras.Input(shape=(input_size, input_size, 3))
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


model_1 = build_stage1(model_1_img_size)
model_1.summary()



## === cell 9
EPOCHS_1 = 3

steps_per_epoch_1 = int(np.ceil(len(df_train) / BATCH_SIZE_1))
val_steps_1 = int(np.ceil(len(df_val) / BATCH_SIZE_1))

history_1 = model_1.fit(
    train_ds_1,
    validation_data=val_ds_1,
    epochs=EPOCHS_1,
    steps_per_epoch=steps_per_epoch_1,
    validation_steps=val_steps_1,
    verbose=2,
)



## === cell 10
TFREC_TRAIN_DIR = BASE / "train_tfrecords"
TFREC_TRAIN_FILES = sorted([str(p) for p in TFREC_TRAIN_DIR.glob("*.tfrec")])
assert len(TFREC_TRAIN_FILES) > 0, f"No TFRecords found in: {TFREC_TRAIN_DIR}"


def _tfrec_format():
    return {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
        "image_name": tf.io.FixedLenFeature([], tf.string),  # used for filtering by id
    }


@tf.function
def _decode_tfrec(example_proto, size):
    ex = tf.io.parse_single_example(example_proto, _tfrec_format())
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, [size, size], method="bilinear")
    img = tf.image.convert_image_dtype(img, tf.float32)
    lbl = tf.cast(ex["target"], tf.int32)
    name = ex["image_name"]
    return img, lbl, name


def _build_idx_to_file(tfrec_files):
    idx_to_file = {}
    for f in tfrec_files:
        name = Path(f).name
        stem = name.split(".")[0]
        part = stem.split("-")[0]
        shard_str = "".join([c for c in part if c.isdigit()])
        if shard_str:
            idx_to_file[int(shard_str)] = f
    return idx_to_file


def _tfrec_files_for_split(
    df_split, train_csv_full, tfrec_files, shard_size=1338, idx_to_file=None
):
    if idx_to_file is None:
        idx_to_file = _build_idx_to_file(tfrec_files)

    id_to_pos = dict(
        zip(train_csv_full["image_id"].astype(str).values, range(len(train_csv_full)))
    )
    split_ids = df_split["image_id"].astype(str).values
    split_pos = np.fromiter(
        (id_to_pos[i] for i in split_ids), dtype=np.int64, count=len(split_ids)
    )

    shard_idxs = np.unique((split_pos // shard_size).astype(np.int64))
    files = [idx_to_file[i] for i in shard_idxs if i in idx_to_file]
    if not files:
        return tfrec_files
    return sorted(files)


def _build_split_shard_offsets(df_split, train_csv_full, shard_size=1338):
    id_to_pos = dict(
        zip(train_csv_full["image_id"].astype(str).values, range(len(train_csv_full)))
    )
    split_ids = df_split["image_id"].astype(str).values
    split_pos = np.fromiter(
        (id_to_pos[i] for i in split_ids), dtype=np.int64, count=len(split_ids)
    )
    shard_idx = (split_pos // shard_size).astype(np.int64)
    offset = (split_pos % shard_size).astype(np.int64)
    return shard_idx, offset


def make_ds_stage2_from_tfrecords(
    df, tfrec_files, size, batch_size, training, cache=False, cache_name=""
):
    idx_to_file = _build_idx_to_file(tfrec_files)
    shard_idx, offset = _build_split_shard_offsets(df, train_csv, shard_size=1338)

    shard_to_offsets = {}
    for s, o in zip(shard_idx.tolist(), offset.tolist()):
        if s in idx_to_file:
            shard_to_offsets.setdefault(s, []).append(o)

    shard_keys = sorted(shard_to_offsets.keys())
    if not shard_keys:
        shard_keys = sorted(idx_to_file.keys())

    def _dataset_for_shard(shard_int):
        shard_int = int(shard_int)
        f = idx_to_file[shard_int]
        ds_file = tf.data.TFRecordDataset(f, num_parallel_reads=1)
        if shard_int in shard_to_offsets:
            offs = shard_to_offsets[shard_int]
            offs_ds = tf.data.Dataset.from_tensor_slices(
                np.asarray(offs, dtype=np.int64)
            )

            def _read_at(o):
                o = tf.cast(o, tf.int64)
                return ds_file.skip(o).take(1)

            return offs_ds.interleave(
                _read_at,
                cycle_length=AUTOTUNE,
                num_parallel_calls=AUTOTUNE,
                deterministic=False,
                block_length=1,
            )
        else:
            return ds_file

    files_ds = tf.data.Dataset.from_tensor_slices(
        np.asarray(shard_keys, dtype=np.int64)
    )
    if training:
        files_ds = files_ds.shuffle(
            len(shard_keys), seed=SEED, reshuffle_each_iteration=True
        )

    ds = files_ds.interleave(
        lambda s: _dataset_for_shard(s),
        cycle_length=AUTOTUNE,
        num_parallel_calls=AUTOTUNE,
        deterministic=False,
        block_length=16,
    )
    ds = ds.with_options(_DS_OPTIONS)

    if training:
        ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda x: _decode_tfrec(x, tf.constant(size, tf.int32)),
        num_parallel_calls=AUTOTUNE,
        deterministic=False,
    )

    ds = ds.map(
        lambda img, lbl, name: (img, lbl),
        num_parallel_calls=AUTOTUNE,
        deterministic=False,
    )

    if training:
        ds = ds.map(
            lambda img, lbl: (_augment(img), lbl),
            num_parallel_calls=AUTOTUNE,
            deterministic=False,
        )

    if cache:
        ds = ds.cache(cache_name if cache_name else None)

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


_idx_to_file = _build_idx_to_file(TFREC_TRAIN_FILES)

TFREC_TRAIN_FILES_TRAIN = _tfrec_files_for_split(
    df_train, train_csv, TFREC_TRAIN_FILES, shard_size=1338, idx_to_file=_idx_to_file
)
TFREC_TRAIN_FILES_VAL = _tfrec_files_for_split(
    df_val, train_csv, TFREC_TRAIN_FILES, shard_size=1338, idx_to_file=_idx_to_file
)

train_ds_2 = make_ds_stage2_from_tfrecords(
    df_train,
    TFREC_TRAIN_FILES_TRAIN,
    model_2_img_size,
    BATCH_SIZE_2,
    True,
    cache=False,
    cache_name="stage2_train",
)
val_ds_2 = make_ds_stage2_from_tfrecords(
    df_val,
    TFREC_TRAIN_FILES_VAL,
    model_2_img_size,
    BATCH_SIZE_2,
    False,
    cache=True,
    cache_name="stage2_val.cache",
)


def build_stage2(input_size, num_classes):
    inp = tf.keras.Input(shape=(input_size, input_size, 3))
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


model_2 = build_stage2(model_2_img_size, NUM_CLASSES)
model_2.summary()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1558661964.py in <cell line: 0>()
    179 )
    180 
--> 181 train_ds_2 = make_ds_stage2_from_tfrecords(
    182     df_train,
    183     TFREC_TRAIN_FILES_TRAIN,

/tmp/ipykernel_11/1558661964.py in make_ds_stage2_from_tfrecords(df, tfrec_files, size, batch_size, training, cache, cache_name)
    131         )
    132 
--> 133     ds = files_ds.interleave(
    134         lambda s: _dataset_for_shard(s),
    135         cycle_length=AUTOTUNE,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in interleave(self, map_func, cycle_length, block_length, num_parallel_calls, deterministic, name)
   2532     # pylint: disable=g-import-not-at-top,protected-access
   2533     from tensorflow.python.data.ops import interleave_op
-> 2534     return interleave_op._interleave(self, map_func, cycle_length, block_length,
   2535                                      num_parallel_calls, deterministic, name)
   2536     # pylint: enable=g-import-not-at-top,protected-access

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/interleave_op.py in _interleave(input_dataset, map_func, cycle_length, block_length, num_parallel_calls, deterministic, name)
     47         input_dataset, map_func, cycle_length, block_length, name=name)
     48   else:
---> 49     return _ParallelInterleaveDataset(
     50         input_dataset,
     51         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/interleave_op.py in __init__(self, input_dataset, map_func, cycle_length, block_length, num_parallel_calls, buffer_output_elements, prefetch_input_elements, deterministic, name)
    117     """See `Dataset.interleave()` for details."""
    118     self._input_dataset = input_dataset
--> 119     self._map_func = structured_function.StructuredFunctionWrapper(
    120         map_func, self._transformation_name(), dataset=input_dataset)
    121     if not isinstance(self._map_func.output_structure, dataset_ops.DatasetSpec):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in __init__(self, func, transformation_name, dataset, input_classes, input_shapes, input_types, input_structure, add_to_graph, use_legacy_function, defun_kwargs)
    263         fn_factory = trace_tf_function(defun_kwargs)
    264 
--> 265     self._function = fn_factory()
    266     # There is no graph to add in eager mode.
    267     add_to_graph &= not context.executing_eagerly()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in get_concrete_function(self, *args, **kwargs)
   1249   def get_concrete_function(self, *args, **kwargs):
   1250     # Implements PolymorphicFunction.get_concrete_function.
-> 1251     concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
   1252     concrete._garbage_collector.release()  # pylint: disable=protected-access
   1253     return concrete

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _get_concrete_function_garbage_collected(self, *args, **kwargs)
   1219       if self._variable_creation_config is None:
   1220         initializers = []
-> 1221         self._initialize(args, kwargs, add_initializers_to=initializers)
   1222         self._initialize_uninitialized_variables(initializers)
   1223 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _initialize(self, args, kwds, add_initializers_to)
    694     )
    695     # Force the definition of the function for these arguments
--> 696     self._concrete_variable_creation_fn = tracing_compilation.trace_function(
    697         args, kwds, self._variable_creation_config
    698     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in trace_function(args, kwargs, tracing_options)
    176       kwargs = {}
    177 
--> 178     concrete_function = _maybe_define_function(
    179         args, kwargs, tracing_options
    180     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _maybe_define_function(args, kwargs, tracing_options)
    281         else:
    282           target_func_type = lookup_func_type
--> 283         concrete_function = _create_concrete_function(
    284             target_func_type, lookup_func_context, func_graph, tracing_options
    285         )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _create_concrete_function(function_type, type_context, func_graph, tracing_options)
    308       attributes_lib.DISABLE_ACD, False
    309   )
--> 310   traced_func_graph = func_graph_module.func_graph_from_py_func(
    311       tracing_options.name,
    312       tracing_options.python_function,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, create_placeholders)
   1057 
   1058     _, original_func = tf_decorator.unwrap(python_func)
-> 1059     func_outputs = python_func(*func_args, **func_kwargs)
   1060 
   1061     # invariant: `func_outputs` contains only Tensors, CompositeTensors,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in wrapped_fn(*args, **kwds)
    597         # the function a weak reference to itself to avoid a reference cycle.
    598         with OptionalXlaContext(compile_with_xla):
--> 599           out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    600         return out
    601 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapped_fn(*args)
    229       # Note: wrapper_helper will apply autograph based on context.
    230       def wrapped_fn(*args):  # pylint: disable=missing-docstring
--> 231         ret = wrapper_helper(*args)
    232         ret = structure.to_tensor_list(self._output_structure, ret)
    233         return [ops.convert_to_tensor(t) for t in ret]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapper_helper(*args)
    159       if not _should_unpack(nested_args):
    160         nested_args = (nested_args,)
--> 161       ret = autograph.tf_convert(self._func, ag_ctx)(*nested_args)
    162       ret = variable_utils.convert_variables_to_tensors(ret)
    163       if _should_pack(ret):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):
--> 693           raise e.ag_error_metadata.to_exception(e)
    694         else:
    695           raise

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    688       try:
    689         with conversion_ctx:
--> 690           return converted_call(f, args, kwargs, options=options)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    437     try:
    438       if kwargs is not None:
--> 439         result = converted_f(*effective_args, **kwargs)
    440       else:
    441         result = converted_f(*effective_args)

/tmp/__autograph_generated_fileamu4mug5.py in <lambda>(s)
      4 
      5     def inner_factory(ag__):
----> 6         tf__lam = lambda s: ag__.with_function_scope(lambda lscope: ag__.converted_call(_dataset_for_shard, (s,), None, lscope), 'lscope', ag__.STD)
      7         return tf__lam
      8     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_fileamu4mug5.py in <lambda>(lscope)
      4 
      5     def inner_factory(ag__):
----> 6         tf__lam = lambda s: ag__.with_function_scope(lambda lscope: ag__.converted_call(_dataset_for_shard, (s,), None, lscope), 'lscope', ag__.STD)
      7         return tf__lam
      8     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_file90c393zq.py in tf___dataset_for_shard(shard_int)
     11                 retval_ = ag__.UndefinedReturnValue()
     12                 shard_int = ag__.converted_call(ag__.ld(int), (ag__.ld(shard_int),), None, fscope)
---> 13                 f = ag__.ld(idx_to_file)[ag__.ld(shard_int)]
     14                 ds_file = ag__.converted_call(ag__.ld(tf).data.TFRecordDataset, (ag__.ld(f),), dict(num_parallel_reads=1), fscope)
     15 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor.py in __hash__(self)
    609     g = getattr(self, "graph", None)
    610     if (Tensor._USE_EQUALITY and (g is None or g.building_function)):
--> 611       raise TypeError("Tensor is unhashable. "
    612                       "Instead, use tensor.ref() as the key.")
    613     else:

TypeError: in user code:

    File "/tmp/ipykernel_11/1558661964.py", line 134, in None  *
        lambda s: _dataset_for_shard(s)
    File "/tmp/ipykernel_11/1558661964.py", line 100, in _dataset_for_shard  *
        f = idx_to_file[shard_int]

    TypeError: Tensor is unhashable. Instead, use tensor.ref() as the key.


## === cell 11
EPOCHS_2 = 3

steps_per_epoch_2 = int(np.ceil(len(df_train) / BATCH_SIZE_2))
val_steps_2 = int(np.ceil(len(df_val) / BATCH_SIZE_2))

history_2 = model_2.fit(
    train_ds_2,
    validation_data=val_ds_2,
    epochs=EPOCHS_2,
    steps_per_epoch=steps_per_epoch_2,
    validation_steps=val_steps_2,
    verbose=2,
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4228864223.py in <cell line: 0>()
      4 val_steps_2 = int(np.ceil(len(df_val) / BATCH_SIZE_2))
      5 
----> 6 history_2 = model_2.fit(
      7     train_ds_2,
      8     validation_data=val_ds_2,

NameError: name 'model_2' is not defined

## === cell 12
print("IMG_SIZE:", IMG_SIZE)
print("model_1_img_size:", model_1_img_size)
print("model_2_img_size:", model_2_img_size)



## === cell 13
if RUN_VIS:
    ss_tmp = pd.read_csv(sample_sub_path)
    test_img_path = str(test_images_dir / ss_tmp["image_id"].iloc[0])

    img = cv2.imread(test_img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read test image: {test_img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_img = (
        cv2.resize(img, (IMG_SIZE, IMG_SIZE)).reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
    )

    plt.figure(figsize=(8, 4))
    plt.title(f"TEST IMAGE: {Path(test_img_path).name}")
    plt.imshow(resized_img[0])
    plt.axis("off")
    plt.show()



## === cell 14
HEALTHY_THRESHOLD = 0.5


def make_test_ds(image_paths, size, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(image_paths.astype(str))
    ds = ds.with_options(_DS_OPTIONS)
    ds = ds.map(
        lambda p: _read_image(p, tf.constant(size, tf.int32)),
        num_parallel_calls=AUTOTUNE,
        deterministic=False,
    )
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds




## === cell 15
ss = pd.read_csv(sample_sub_path)

test_paths = np.char.add(str(test_images_dir) + "/", ss["image_id"].astype(str).values)

test_ds_1 = make_test_ds(test_paths, model_1_img_size, batch_size=512)
p1 = model_1.predict(test_ds_1, verbose=0).reshape(-1)  # probability healthy

healthy_mask = p1 >= HEALTHY_THRESHOLD
preds = np.empty(len(ss), dtype=np.int64)
preds[healthy_mask] = 4

idx_nonhealthy = np.where(~healthy_mask)[0]
if idx_nonhealthy.size > 0:
    nonhealthy_paths = test_paths[idx_nonhealthy]
    test_ds_2 = make_test_ds(
        nonhealthy_paths, model_2_img_size, batch_size=BATCH_SIZE_2
    )
    p2 = model_2.predict(test_ds_2, verbose=0)
    preds[idx_nonhealthy] = np.argmax(p2, axis=1).astype(np.int64)

my_submission = pd.DataFrame(
    {"image_id": ss["image_id"].values, "label": preds.astype(int)}
)
my_submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", my_submission.shape)
print(my_submission.head())



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2225786871.py in <cell line: 0>()
      1 ss = pd.read_csv(sample_sub_path)
      2 
----> 3 test_paths = np.char.add(str(test_images_dir) + "/", ss["image_id"].astype(str).values)
      4 
      5 test_ds_1 = make_test_ds(test_paths, model_1_img_size, batch_size=512)

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in add(x1, x2)
    330         # object dtype itemsize as num chars (worked on short strings).
    331         # bytes + void worked but promoting void->bytes is dubious also.
--> 332         raise TypeError(
    333             "np.char.add() requires both arrays of the same dtype kind, but "
    334             f"got dtypes: '{arr1.dtype}' and '{arr2.dtype}' (the few cases "

TypeError: np.char.add() requires both arrays of the same dtype kind, but got dtypes: '<U62' and 'object' (the few cases where this used to work often lead to incorrect results).

## === cell 16
print("Submission File: \n---------------\n")
print(my_submission.head())
print("\nSaved to: /kaggle/working/submission.csv")

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3311604638.py in <cell line: 0>()
      1 print("Submission File: \n---------------\n")
----> 2 print(my_submission.head())
      3 print("\nSaved to: /kaggle/working/submission.csv")

NameError: name 'my_submission' is not defined
