# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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
tf_keras==2.18.0
tqdm==4.67.1

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

# 5. Target score

0.8237

# 6. Current score

0.68896

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.35669) has done: 'Diagnosis: Cell 6 crashes because `KaggleDatasets().get_gcs_path(...)` calls a Kaggle backend service that requires authentication and fails in this environment with `BackendError: Unauthenticated`. This prevents construction of `GCS_DS_PATH`, which is required by cell 7 to build TFRecord file globs. The dataset is available locally under the provided `/kaggle/data/...` paths, so we can safely fall back to that local directory when the GCS lookup fails.

Patch summary: Wrap the `get_gcs_path()` call in a `try/except` and, on failure, set `GCS_DS_PATH` to the existing local dataset directory (`/kaggle/data/siim-isic-melanoma-classification` or `/kaggle/input/siim-isic-melanoma-classification`). Keep the variable name and semantics (a base directory containing `tfrecords/`) unchanged so downstream code continues to work.

Updated cells: Only cell 6 is modified.

Compatibility notes for cell k+1: `GCS_DS_PATH` remains a string path prefix; cell 7’s `tf.io.gfile.glob(GCS_DS_PATH + "/tfrecords/train*")` and `.../test*` work against the local filesystem via `tf.io.gfile`.

Assumptions: The dataset directory exists locally at one of the two standard Kaggle locations and contains a `tfrecords/` subdirectory (as shown in the provided file tree).'
- What this solution (achieved 0.61285) has done: 'Your score is far below the target, so the smallest safe way to move AUC up is to fix a likely label/TFRecord mismatch rather than changing the model or training loop. In this competition’s TFRecords the label key is typically `label` (not `target`), and reading the wrong key can silently destroy training signal and yield near-random predictions like your 0.356 AUC. I modify only the TFRecord parsing to automatically use `label` when present (and fall back to `target` for safety), keeping all architecture/training logic identical. This should materially increase AUC toward your 0.8237 target without changing your approach.'
- What this solution (achieved 0.58907) has done: 'We keep your pipeline and EfficientNetB0 training exactly as-is, and focus on one likely reason your AUC is stuck around ~0.61: the train/validation TFRecord split is currently done by taking the first 10% of TFRecord shards as validation, which can be distribution-shifted (often sorted/grouped) and hurts generalization. I change only the split logic to a deterministic shuffle of TFRecord filenames (fixed seed) before splitting, which typically raises AUC without changing model/training semantics. I also ensure the label tensor is shaped consistently as float32 for `binary_crossentropy` to avoid any subtle casting/broadcasting issues (doesn’t change the approach, just makes it numerically consistent). Everything else (data decoding, augmentation, model, epochs, submission writing) remains unchanged.'
- What this solution (achieved 0.68896) has done: 'I keep your TFRecord pipeline and EfficientNetB0 training loop intact, but adjust three small things that commonly hold AUC down around ~0.60 in this competition: (1) ensure filenames (train/val/test) are deterministically sorted so the ordered test pipeline aligns perfectly with `image_name`, (2) compute `steps_per_epoch`/`validation_steps` using `ceil` instead of floor so each epoch/validation covers all images rather than systematically dropping the tail (which can undertrain and bias AUC), and (3) switch the head loss to `BinaryCrossentropy(label_smoothing=0.05)` which is a minimal, semantics-preserving regularization that often improves ranking/AUC without changing architecture or training procedure. These are minimal changes aimed at increasing your score toward 0.8237 without introducing early stopping, approximations, or changing the model. The script still run end-to-end and write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os



## === cell 1
from datetime import datetime

start_time = datetime.now()



## === cell 2
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
)

import gc
import json
import math
import re
import cv2
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import tensorflow as tf
from tqdm import tqdm

from tensorflow.keras import layers

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass



## === cell 3
from tensorflow.keras.applications import EfficientNetB0



## === cell 4
from kaggle_datasets import KaggleDatasets



## === cell 5
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)



## === cell 6
try:
    GCS_DS_PATH = KaggleDatasets().get_gcs_path("siim-isic-melanoma-classification")
except Exception:
    local_candidates = [
        "/kaggle/data/siim-isic-melanoma-classification",
        "/kaggle/input/siim-isic-melanoma-classification",
    ]
    for p in local_candidates:
        if tf.io.gfile.exists(p):
            GCS_DS_PATH = p
            break
    else:
        raise



## === cell 7
TRAINING_FILENAMES = sorted(tf.io.gfile.glob(GCS_DS_PATH + "/tfrecords/train*"))
TEST_FILENAMES = sorted(tf.io.gfile.glob(GCS_DS_PATH + "/tfrecords/test*"))

BATCH_SIZE = 8 * strategy.num_replicas_in_sync
IMAGE_SIZE = [1024, 1024]
AUTO = tf.data.experimental.AUTOTUNE
imSize = 224
EPOCHS = 2




## === cell 8
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    image = tf.image.resize(image, [imSize, imSize])
    return image


def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example["image"])

    label = tf.where(example["label"] >= 0, example["label"], example["target"])
    label = tf.cast(label, tf.float32)
    label = tf.reshape(label, [1])
    return image, label


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
    if not ordered:
        options.experimental_deterministic = False
    else:
        options.experimental_deterministic = True

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(options)
    dataset = dataset.map(
        read_labeled_tfrecord if labeled else read_unlabeled_tfrecord,
        num_parallel_calls=AUTO,
    )
    return dataset


def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label


def count_data_items(filenames):
    n = [
        int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
        for filename in filenames
    ]
    return np.sum(n)


NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH = int(math.ceil(NUM_TRAINING_IMAGES / BATCH_SIZE))

print(
    "Dataset: {} training images, {} unlabeled test images".format(
        NUM_TRAINING_IMAGES, NUM_TEST_IMAGES
    )
)

rng = np.random.RandomState(2020)
shuffled = TRAINING_FILENAMES.copy()
rng.shuffle(shuffled)

n_files = len(shuffled)
n_val = max(1, int(0.10 * n_files))
VAL_FILENAMES = shuffled[:n_val]
TRAIN_FILENAMES = shuffled[n_val:]

NUM_VAL_IMAGES = count_data_items(VAL_FILENAMES)
NUM_TRAIN_IMAGES = count_data_items(TRAIN_FILENAMES)

TRAIN_STEPS_PER_EPOCH = int(math.ceil(NUM_TRAIN_IMAGES / BATCH_SIZE))
VAL_STEPS = max(1, int(math.ceil(NUM_VAL_IMAGES / BATCH_SIZE)))

print(
    f"Split TFRecords -> train_files={len(TRAIN_FILENAMES)}, val_files={len(VAL_FILENAMES)}"
)
print(
    f"Split counts    -> train_images={NUM_TRAIN_IMAGES}, val_images={NUM_VAL_IMAGES}"
)


def get_training_dataset():
    dataset = load_dataset(TRAIN_FILENAMES, labeled=True, ordered=False)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset


def get_validation_dataset():
    dataset = load_dataset(VAL_FILENAMES, labeled=True, ordered=True)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset


def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset




## === cell 9
print("Training data shapes:")
for image, label in get_training_dataset().take(1):
    print(image.numpy().shape, label.numpy().shape)
print("Training data label examples:", label.numpy()[:20].reshape(-1))

print("Validation data shapes:")
for image, label in get_validation_dataset().take(1):
    print(image.numpy().shape, label.numpy().shape)

print("Test data shapes:")
for image, idnum in get_test_dataset().take(1):
    print(image.numpy().shape, idnum.numpy().shape)
print("Test data IDs:", idnum.numpy().astype("U")[:5])



## === cell 10
with strategy.scope():
    model = tf.keras.Sequential(
        [
            EfficientNetB0(
                input_shape=(imSize, imSize, 3), weights="imagenet", include_top=False
            ),
            layers.GlobalAveragePooling2D(),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    model.summary()



## === cell 11
history = model.fit(
    get_training_dataset(),
    steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=get_validation_dataset(),
    validation_steps=VAL_STEPS,
)



## === cell 12
test_ds = get_test_dataset(ordered=True)

print("Computing predictions...")
test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities = model.predict(test_images_ds, verbose=1).reshape(-1)

print("Generating submission.csv file...")
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype("U")

assert len(test_ids) == len(
    probabilities
), f"ids({len(test_ids)}) != probs({len(probabilities)})"

sub = pd.DataFrame({"image_name": test_ids, "target": probabilities})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)



## === cell 13
sub_check = pd.read_csv("submission.csv")
sub_check.head()



## === cell 14
print("Notebook Runtime: ", datetime.now() - start_time)
