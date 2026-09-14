# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict whether a lesion is malignant (0 denotes **benign**, and 1 indicates **malignant**).

## Metric
Area under the ROC curve.

## Submission Format
For each `image_name` in the test set, you must predict the probability (`target`) that the sample is **malignant**. The file should contain a header and have the following format:

```
image_name,target
ISIC_0052060,0.7
ISIC_0052349,0.9
ISIC_0058510,0.8
ISIC_0073313,0.5
ISIC_0073502,0.5
etc.
```

## Dataset 
The images are provided in DICOM format.

Images are also provided in JPEG and TFRecord format (in the `jpeg` and `tfrecords` directories, respectively). Images in TFRecord format have been resized to a uniform 1024x1024.

Metadata is also provided outside of the DICOM format, in CSV files. See the `Columns` section for a description.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `image_name` - unique identifier, points to filename of related DICOM image
- `patient_id` - unique patient identifier
- `sex` - the sex of the patient (when unknown, will be blank)
- `age_approx` - approximate patient age at time of imaging
- `anatom_site_general_challenge` - location of imaged site
- `diagnosis` - detailed diagnosis information (train only)
- `benign_malignant` - indicator of malignancy of imaged lesion
- `target` - binarized version of the target variable

# 2. Python version

3.8

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
        input/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
        working/
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
```

-> data/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/siim-isic-melanoma-classification/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> data/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> input/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

import re
import math
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF pick
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TensorFlow:", tf.__version__)



## === cell 1
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import (
    preprocess_input as effnet_preprocess,
)



## === cell 2
DATASET_DIR = "/kaggle/input/siim-isic-melanoma-classification"

if not os.path.exists(DATASET_DIR):
    alt = "/kaggle/data/siim-isic-melanoma-classification"
    if os.path.exists(alt):
        DATASET_DIR = alt

print("Using DATASET_DIR:", DATASET_DIR)



## === cell 3
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.master())
except Exception:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)



## === cell 4
AUTO = tf.data.experimental.AUTOTUNE

DEBUG = False
N_FOLD = 4
EPOCHS = 1 if DEBUG else 7
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
IMAGE_SIZE = [1024, 1024]

TFRECORD_DIR = os.path.join(DATASET_DIR, "tfrecords")
print("TFRECORD_DIR:", TFRECORD_DIR)



## === cell 5
test_files = tf.io.gfile.glob(os.path.join(TFRECORD_DIR, "test*.tfrec"))
if len(test_files) == 0:
    raise FileNotFoundError(f"No TFRecord files found under: {TFRECORD_DIR}")

test_files = sorted(test_files)
print("Num test tfrecords:", len(test_files))
print("Example tfrecord:", test_files[0])

train_files = tf.io.gfile.glob(os.path.join(TFRECORD_DIR, "train*.tfrec"))
if len(train_files) == 0:
    raise FileNotFoundError(f"No TFRecord files found under: {TFRECORD_DIR}")

train_files = sorted(train_files)
print("Num train tfrecords:", len(train_files))
print("Example tfrecord:", train_files[0])



## === cell 6
sub_path = os.path.join(DATASET_DIR, "sample_submission.csv")
sub = pd.read_csv(sub_path)
print("Sample submission shape:", sub.shape)
sub.head()




## === cell 7
@tf.function
def decode_image(image_data):
    image = tf.io.decode_jpeg(image_data, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear", antialias=False)
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
    image = image * 255.0  # preprocess_input expects [0,255]
    image = effnet_preprocess(image)
    return image


@tf.function
def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    target = tf.cast(example["target"], tf.float32)
    return image, target


@tf.function
def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    idnum = example["image_name"]
    return image, idnum


def load_dataset(filenames, labeled=True, ordered=False):
    options = tf.data.Options()
    options.experimental_deterministic = bool(ordered)

    try:
        options.experimental_optimization.apply_default_optimizations = True
    except Exception:
        pass
    try:
        options.experimental_optimization.map_parallelization = True
    except Exception:
        pass
    try:
        options.experimental_optimization.parallel_batch = True
    except Exception:
        pass

    ds_files = tf.data.Dataset.from_tensor_slices(filenames)
    if not ordered:
        ds_files = ds_files.shuffle(
            len(filenames), seed=SEED, reshuffle_each_iteration=True
        )

    cycle_len = min(len(filenames), 16) if not ordered else min(len(filenames), 8)
    dataset = ds_files.interleave(
        lambda fn: tf.data.TFRecordDataset(fn, num_parallel_reads=AUTO),
        cycle_length=cycle_len,
        num_parallel_calls=AUTO,
        deterministic=bool(ordered),
    )
    dataset = dataset.with_options(options)
    dataset = dataset.map(
        read_labeled_tfrecord if labeled else read_unlabeled_tfrecord,
        num_parallel_calls=AUTO,
        deterministic=bool(ordered),
    )
    return dataset


def get_test_dataset(test_files, ordered=False, cache=False):
    dataset = load_dataset(test_files, labeled=False, ordered=ordered)
    if cache:
        dataset = dataset.cache()
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)
    dataset = dataset.prefetch(AUTO)
    return dataset


def get_train_dataset(
    train_files,
    ordered=False,
    shuffle_buffer=2048,
    repeat=True,
    cache=False,
    shuffle=True,
    drop_remainder=True,
):
    dataset = load_dataset(train_files, labeled=True, ordered=ordered)
    if cache:
        dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(
            shuffle_buffer, seed=SEED, reshuffle_each_iteration=True
        )
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(AUTO)
    return dataset


_count_re = re.compile(r"-([0-9]*)\.")


def count_data_items(filenames):
    return int(sum(int(_count_re.search(fn).group(1)) for fn in filenames))




## === cell 8
num_test = count_data_items(test_files)
num_train = count_data_items(train_files)
print("Counted test items from filenames:", num_test)
print("Counted train items from filenames:", num_train)
print("Sample submission rows:", len(sub))




## === cell 9
def get_model():
    with strategy.scope():
        model = tf.keras.Sequential(
            [
                EfficientNetB3(
                    input_shape=(*IMAGE_SIZE, 3), weights="imagenet", include_top=False
                ),
                L.GlobalAveragePooling2D(),
                L.Dense(1, activation="sigmoid"),
            ]
        )
    return model




## === cell 10
model = get_model()
with strategy.scope():
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(name="auc")],
        jit_compile=True,
    )

rng = np.random.RandomState(SEED)
idx = np.arange(len(train_files))
rng.shuffle(idx)
split = max(1, int(0.85 * len(train_files)))
train_f = [train_files[i] for i in idx[:split]]
val_f = [train_files[i] for i in idx[split:]]

train_items = count_data_items(train_f)
val_items = count_data_items(val_f)

steps_per_epoch = max(1, (train_items // BATCH_SIZE))
val_steps = max(1, int(math.ceil(val_items / BATCH_SIZE)))

print("Train tfrecs:", len(train_f), "Val tfrecs:", len(val_f))
print("steps_per_epoch:", steps_per_epoch, "val_steps:", val_steps)

train_ds = get_train_dataset(
    train_f, ordered=False, repeat=True, cache=False, shuffle=True, drop_remainder=True
)

val_ds = get_train_dataset(
    val_f,
    ordered=True,
    repeat=False,
    cache=False,
    shuffle=False,
    drop_remainder=False,
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)



## === cell 11
test_ds = get_test_dataset(test_files, ordered=True, cache=False)

test_ids_list = []
for _, id_batch in test_ds:
    test_ids_list.append(id_batch.numpy())
test_ids = np.concatenate(test_ids_list, axis=0).astype("U")

probabilities = model.predict(test_ds, verbose=1).reshape(-1)

n = min(len(test_ids), len(probabilities))
pred_df = pd.DataFrame({"image_name": test_ids[:n], "target": probabilities[:n]})

print("pred_df shape:", pred_df.shape)
pred_df.head()



## === cell 12
if "image_name" not in pred_df.columns or "target" not in pred_df.columns:
    raise KeyError(
        f"pred_df missing required columns. Columns: {pred_df.columns.tolist()}"
    )

if pred_df["image_name"].is_unique:
    pred_map = pd.Series(pred_df["target"].values, index=pred_df["image_name"].values)
    sub_out = sub[["image_name"]].copy()
    sub_out["target"] = sub_out["image_name"].map(pred_map)
    if sub_out["target"].isna().any():
        sub_out["target"] = sub_out["target"].fillna(float(pred_df["target"].mean()))
else:
    mean_pred_df = pred_df.groupby("image_name", as_index=False)["target"].mean()
    sub_out = sub[["image_name"]].merge(mean_pred_df, on="image_name", how="left")
    if sub_out["target"].isna().any():
        sub_out["target"] = sub_out["target"].fillna(
            float(mean_pred_df["target"].mean())
        )

sub_out["target"] = pd.to_numeric(sub_out["target"], errors="coerce").fillna(
    float(sub_out["target"].mean())
)
sub_out["target"] = sub_out["target"].clip(0.0, 1.0)

sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
print(sub_out.head())
print(
    "submission.csv exists:",
    os.path.exists("submission.csv"),
    "size:",
    os.path.getsize("submission.csv"),
)
