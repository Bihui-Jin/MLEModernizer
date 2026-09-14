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

0.57661

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.57661) has done: 'The main timeout culprit is stage-2 dataset creation from TFRecords: `enumerate().filter(reduce_any(equal(i, idx_keep)))` forces a very expensive per-example membership test and effectively scans all records multiple times. I keep the same stage-1/stage-2 models, losses, epochs, and overall training/prediction semantics, but rebuild the stage-2 input pipeline to select the exact training/validation examples by joining on `image_id` inside TFRecords (a provably equivalent split to your CSV-based split) rather than by global index filtering. I also remove an environment setting that forces the slow pure-Python protobuf implementation, and make caching behavior robust (cache only the decoded/resized images) to avoid repeated JPEG decode/resize overhead. These changes preserve core logic and accuracy while cutting input-pipeline cost enough to fit under 600 seconds.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
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


@tf.function
def _read_image(path, size):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [size, size], method="bilinear")
    img = tf.image.convert_image_dtype(img, tf.float32)  # == tf.cast(...)/255.0
    return img


@tf.function
def _augment(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img


_CACHE_DIR = Path("/kaggle/working/tf_cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_DS_OPTIONS = tf.data.Options()
_DS_OPTIONS.deterministic = True  # keep global determinism expectation
_DS_OPTIONS.experimental_slack = (
    True  # allow input/compute overlap without changing results
)


def make_ds(
    df, images_root, size, batch_size, training, label_mode, cache=False, cache_name=""
):
    paths = (images_root / df["image_id"].values).astype(str)
    if label_mode == "binary_healthy":
        y = (df["label_int"].values == 4).astype(np.int32)
    elif label_mode == "multiclass":
        y = df["label_int"].values.astype(np.int32)
    else:
        raise ValueError("Unknown label_mode")

    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    ds = ds.with_options(_DS_OPTIONS)

    def _map_read(p, lbl):
        img = _read_image(p, size)
        return img, lbl

    ds = ds.map(_map_read, num_parallel_calls=AUTOTUNE, deterministic=False)

    if cache:
        cache_path = str(
            _CACHE_DIR
            / f"cache_{cache_name}_{'train' if training else 'val'}_{size}.tf-data"
        )
        ds = ds.cache(cache_path)

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

        def _map_aug(img, lbl):
            return _augment(img), lbl

        ds = ds.map(_map_aug, num_parallel_calls=AUTOTUNE, deterministic=False)

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
    cache=True,
    cache_name="stage1",
)
val_ds_1 = make_ds(
    df_val,
    train_images_dir,
    model_1_img_size,
    BATCH_SIZE_1,
    False,
    "binary_healthy",
    cache=True,
    cache_name="stage1",
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
history_1 = model_1.fit(
    train_ds_1,
    validation_data=val_ds_1,
    epochs=EPOCHS_1,
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
        "image_id": tf.io.FixedLenFeature([], tf.string),
    }


@tf.function
def _decode_tfrec(example_proto, size):
    ex = tf.io.parse_single_example(example_proto, _tfrec_format())
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, [size, size], method="bilinear")
    img = tf.image.convert_image_dtype(img, tf.float32)
    lbl = tf.cast(ex["target"], tf.int32)
    img_id = ex["image_id"]
    return img, lbl, img_id


def make_ds_from_tfrecords_by_image_id(
    tfrec_files, image_id_keep, size, batch_size, training, cache=False, cache_name=""
):
    keys = tf.constant(np.asarray(image_id_keep, dtype=np.bytes_))
    vals = tf.ones([tf.shape(keys)[0]], dtype=tf.int32)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, vals), default_value=0
    )

    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(_DS_OPTIONS)

    ds = ds.map(
        lambda x: _decode_tfrec(x, size),
        num_parallel_calls=AUTOTUNE,
        deterministic=False,
    )

    ds = ds.filter(lambda img, lbl, img_id: table.lookup(img_id) > 0)

    ds = ds.map(lambda img, lbl, img_id: (img, lbl), num_parallel_calls=AUTOTUNE)

    if cache:
        cache_path = str(
            _CACHE_DIR
            / f"cache_{cache_name}_{'train' if training else 'val'}_{size}.tf-data"
        )
        ds = ds.cache(cache_path)

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(
            lambda img, lbl: (_augment(img), lbl),
            num_parallel_calls=AUTOTUNE,
            deterministic=False,
        )

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


train_ids = df_train["image_id"].values
val_ids = df_val["image_id"].values

train_ds_2 = make_ds_from_tfrecords_by_image_id(
    TFREC_TRAIN_FILES,
    train_ids,
    model_2_img_size,
    BATCH_SIZE_2,
    True,
    cache=True,
    cache_name="stage2_tfrec",
)
val_ds_2 = make_ds_from_tfrecords_by_image_id(
    TFREC_TRAIN_FILES,
    val_ids,
    model_2_img_size,
    BATCH_SIZE_2,
    False,
    cache=True,
    cache_name="stage2_tfrec",
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



## === cell 11
EPOCHS_2 = 3
history_2 = model_2.fit(
    train_ds_2,
    validation_data=val_ds_2,
    epochs=EPOCHS_2,
    verbose=2,
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/2118399446.py in <cell line: 0>()
      1 EPOCHS_2 = 3
----> 2 history_2 = model_2.fit(
      3     train_ds_2,
      4     validation_data=val_ds_2,
      5     epochs=EPOCHS_2,

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

InvalidArgumentError: Graph execution error:

Detected at node ParseSingleExample/ParseExample/ParseExampleV2 defined at (most recent call last):
<stack traces unavailable>
Error in user-defined function passed to ParallelMapDatasetV2:19 transformation with iterator: Iterator::Root::Prefetch::MapAndBatch::Shuffle::FileCacheImpl::ParallelMapV2::Filter::ParallelMapV2: Feature: image_id (data type: string) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_6317]

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
        lambda p: _read_image(p, size), num_parallel_calls=AUTOTUNE, deterministic=False
    )
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds




## === cell 15
ss = pd.read_csv(sample_sub_path)
test_paths = (test_images_dir / ss["image_id"].values).astype(str)

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

my_submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": preds})
my_submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", my_submission.shape)
print(my_submission.head())



## === cell 16
print("Submission File: \n---------------\n")
print(my_submission.head())
print("\nSaved to: /kaggle/working/submission.csv")
