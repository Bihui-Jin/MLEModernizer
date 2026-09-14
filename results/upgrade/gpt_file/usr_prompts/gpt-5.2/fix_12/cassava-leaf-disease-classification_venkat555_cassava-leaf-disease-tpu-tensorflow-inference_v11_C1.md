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

0.8792686612269568

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix two hard runtime issues that currently prevent any submission from being created: (1) the TensorFlow import crash caused by a protobuf incompatibility by pinning the pure-Python protobuf implementation before importing TensorFlow, and (2) the TTA mapping error where `data_augment` is applied to already-batched tensors (rank-4), causing `random_crop` shape mismatch. I adjust the TTA stage to apply augmentation per-image via `tf.map_fn` while keeping the same augmentation logic and preserving deterministic dataset ordering so `image_id` aligns with predictions. Finally, I ensure `submission.csv` is always written with the required columns and row count matched to `sample_submission.csv`.'
- What this solution (achieved 0.61099) has done: 'The timeout is dominated by (1) Python-protobuf TFRecord parsing forced by environment variables, (2) expensive per-image TTA implemented via `tf.map_fn` inside the input pipeline, and (3) `model.predict_on_batch` being invoked from Python for every batch (and for every model). I switch protobuf back to the default (C++ fast path), move augmentation to a fully vectorized, batch-level TensorFlow implementation (no `map_fn`), and wrap per-model forward passes in `tf.function` to reduce Python overhead while preserving identical architecture, weights, dataset order, and TTA semantics. I also keep deterministic ordering for ID alignment and avoid any changes to image size, batch size, model, or the number of TTA steps.'
- What this solution (achieved 0.61099) has done: 'I fix the TensorFlow import crash by switching protobuf to the safe pure-Python implementation before importing TensorFlow (the current `cpp` setting is what triggers the `_message` ImportError in this environment). Then I make the TFRecord parsing + dataset pipeline consistent for test inference: ensure deterministic ordering, correct resize, and apply TTA only on `(images, ids)` batches (not on mislabeled `(image,label)` pairs). Finally, I ensure `submission.csv` is always written with exactly the `image_id,label` columns aligned to `sample_submission.csv`, and add a small safety check that the number of produced predictions matches the sample submission rows.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import math, re, warnings, random, glob
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential

warnings.filterwarnings("ignore")

tf.config.optimizer.set_jit(False)

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print(f"Running on TPU {tpu.master()}")
except Exception:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f"REPLICAS: {REPLICAS}")



## === cell 2
BATCH_SIZE = 16 * REPLICAS
HEIGHT = 512
WIDTH = 512
CHANNELS = 3
N_CLASSES = 5
TTA_STEPS = 3  # Do TTA if > 0
IMAGE_SIZE = [512, 512]
SEED = 555
AUG_BATCH = BATCH_SIZE

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)




## === cell 3
def data_augment(image, label):
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    IMG_SIZE = IMAGE_SIZE[0]
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
    image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])
    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.random_saturation(image, 0, 2)
    image = tf.image.adjust_saturation(image, 3)
    return image, label




## === cell 4
def get_name(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    name = parts[-1]
    return name


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.ensure_shape(image, [None, None, 3])
    return image


def resize_image(image, label_or_id):
    image = tf.image.resize(image, [HEIGHT, WIDTH])
    image = tf.reshape(image, [HEIGHT, WIDTH, CHANNELS])
    return image, label_or_id


def process_path(file_path):
    name = get_name(file_path)
    img = tf.io.read_file(file_path)
    img = decode_image(img)
    return img, name


def get_dataset(files_path, shuffled=False, tta=False, extension="jpg"):
    dataset = tf.data.Dataset.list_files(f"{files_path}*{extension}", shuffle=shuffled)
    dataset = dataset.map(process_path, num_parallel_calls=AUTO)
    if tta:
        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.map(resize_image, num_parallel_calls=AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset


def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    image_name = example["image_name"]
    return image, image_name


def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    label = tf.cast(example["target"], tf.int32)
    return image, label


def load_dataset(filenames, labeled=True, ordered=False):
    opts = tf.data.Options()
    opts.experimental_deterministic = bool(ordered)

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(opts)
    dataset = dataset.map(
        read_labeled_tfrecord if labeled else read_unlabeled_tfrecord,
        num_parallel_calls=AUTO,
        deterministic=ordered,
    )
    return dataset


def count_data_items(filenames):
    n = [
        int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
        for filename in filenames
    ]
    return np.sum(n)




## === cell 5
database_base_path = "/kaggle/input/cassava-leaf-disease-classification/"
submission = pd.read_csv(f"{database_base_path}sample_submission.csv")
print(submission.head())

TEST_FILENAMES = tf.io.gfile.glob(f"{database_base_path}test_tfrecords/*.tfrec")
TEST_FILENAMES = sorted(TEST_FILENAMES)
NUM_TEST_IMAGES = int(count_data_items(TEST_FILENAMES))
print(f"Test images (from TFRecords): {NUM_TEST_IMAGES}")
print(f"Sample submission rows: {len(submission)}")

if NUM_TEST_IMAGES != len(submission):
    print(
        f"Warning: TFRecord count ({NUM_TEST_IMAGES}) != sample_submission rows ({len(submission)})."
    )



## === cell 6
model_path_list = glob.glob(
    "/kaggle/input/casavaleafclassificationdensenet201e25f3/*.h5"
)
model_path_list.sort()

print("Models to predict:")
if len(model_path_list) == 0:
    print("(none found at /kaggle/input/casavaleafclassificationdensenet201e25f3/*.h5)")
else:
    print(*model_path_list, sep="\n")



## === cell 7
from tensorflow import keras

models = []

with strategy.scope():
    if len(model_path_list) > 0:
        for model_path in model_path_list:
            print(f"Loading model: {model_path}")
            models.append(keras.models.load_model(model_path))
    else:
        print(
            "Falling back to EfficientNetB0(ImageNet) + Dense(5) head (no external .h5 found)."
        )
        backbone = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(HEIGHT, WIDTH, CHANNELS),
            pooling="avg",
        )
        x = backbone.output
        out = L.Dense(N_CLASSES, activation="softmax")(x)
        fallback_model = tf.keras.Model(backbone.input, out)
        models.append(fallback_model)

print(f"Loaded {len(models)} model(s).")



## === cell 8
models[0].summary()




## === cell 9
@tf.function
def batch_data_augment(images, ids):
    ks = tf.random.uniform([tf.shape(images)[0]], minval=0, maxval=4, dtype=tf.int32)
    r0 = images
    r1 = tf.image.rot90(images, k=1)
    r2 = tf.image.rot90(images, k=2)
    r3 = tf.image.rot90(images, k=3)
    stacked = tf.stack([r0, r1, r2, r3], axis=1)  # [B,4,H,W,C]
    idx_b = tf.range(tf.shape(images)[0], dtype=tf.int32)
    idx = tf.stack([idx_b, ks], axis=1)
    images = tf.gather_nd(stacked, idx)

    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_flip_up_down(images)

    IMG_SIZE = IMAGE_SIZE[0]
    images = tf.image.resize_with_crop_or_pad(images, IMG_SIZE + 6, IMG_SIZE + 6)
    images = tf.image.random_crop(
        images, size=[tf.shape(images)[0], IMG_SIZE, IMG_SIZE, 3]
    )

    images = tf.image.random_brightness(images, max_delta=0.5)
    images = tf.image.random_saturation(images, 0, 2)
    images = tf.image.adjust_saturation(images, 3)
    return images, ids


print(" TTA_STEPS = {} ".format(TTA_STEPS))

tta_steps = max(int(TTA_STEPS), 0)
if tta_steps == 0:
    tta_steps = 1
    use_tta = False
else:
    use_tta = True

base_options = tf.data.Options()
base_options.experimental_deterministic = True
try:
    base_options.experimental_optimization.map_parallelization = True
    base_options.experimental_optimization.parallel_batch = True
except Exception:
    pass

base_test_ds = load_dataset(TEST_FILENAMES, labeled=False, ordered=True).with_options(
    base_options
)
base_test_ds = base_test_ds.map(
    resize_image, num_parallel_calls=AUTO, deterministic=True
)
base_test_ds = base_test_ds.batch(BATCH_SIZE, drop_remainder=False)
base_test_ds = base_test_ds.cache()
base_test_ds = base_test_ds.prefetch(AUTO)

predict_fns = [tf.function(lambda x, m=m: m(x, training=False)) for m in models]

probs_sum = np.zeros((NUM_TEST_IMAGES, N_CLASSES), dtype=np.float32)
all_test_ids = np.empty((NUM_TEST_IMAGES,), dtype=object)

for step in range(tta_steps):
    print(f"TTA step {step+1}/{tta_steps}")

    ds = base_test_ds
    if use_tta:
        ds = ds.map(
            batch_data_augment,
            num_parallel_calls=AUTO,
            deterministic=True,
        )

    write_pos = 0
    for images, ids in ds:
        bs = int(images.shape[0])

        if step == 0:
            all_test_ids[write_pos : write_pos + bs] = ids.numpy().astype("U")

        batch_p = None
        for fn in predict_fns:
            p = fn(images)
            p = tf.convert_to_tensor(p)
            batch_p = p if batch_p is None else (batch_p + p)
        batch_p = batch_p / float(len(models))

        probs_sum[write_pos : write_pos + bs] += batch_p.numpy().astype(
            np.float32, copy=False
        )
        write_pos += bs

    if write_pos != NUM_TEST_IMAGES:
        raise RuntimeError(
            f"Dataset yielded {write_pos} items, expected {NUM_TEST_IMAGES}"
        )

probabilities = probs_sum / float(tta_steps)
predictions = np.argmax(probabilities, axis=-1).astype(np.int64)

print("Predictions shape:", predictions.shape)
print("test_ids shape:", all_test_ids.shape)



## === cell 10
print("Generating submission.csv file...")

print("NUM_TEST_IMAGES:", NUM_TEST_IMAGES)
print("len(sample_submission):", len(submission))

pred_df = pd.DataFrame({"image_id": all_test_ids, "label": predictions})
pred_df["image_id"] = pred_df["image_id"].astype(str)
pred_df["label"] = pred_df["label"].astype(int)

sub_df = submission[["image_id"]].merge(pred_df, on="image_id", how="left")

missing = int(sub_df["label"].isna().sum())
if missing:
    print(f"Warning: {missing} image_id(s) missing predictions; filling with 0.")
    sub_df["label"] = sub_df["label"].fillna(0).astype(int)

sub_df = sub_df[["image_id", "label"]]
assert len(sub_df) == len(submission), "Submission row count mismatch."

sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head())

with open("submission.csv", "r", encoding="utf-8") as f:
    for _ in range(5):
        print(f.readline().rstrip("\n"))
