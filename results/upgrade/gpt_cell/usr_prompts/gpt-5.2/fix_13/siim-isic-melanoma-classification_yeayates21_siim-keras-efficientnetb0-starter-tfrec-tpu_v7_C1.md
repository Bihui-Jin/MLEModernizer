# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.8

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

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
        "/kaggle/data",
        "/kaggle/input",
    ]
    for p in local_candidates:
        if tf.io.gfile.exists(os.path.join(p, "tfrecords")):
            GCS_DS_PATH = p
            break
        if tf.io.gfile.exists(
            os.path.join(p, "siim-isic-melanoma-classification", "tfrecords")
        ):
            GCS_DS_PATH = os.path.join(p, "siim-isic-melanoma-classification")
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
    image = tf.io.decode_raw(image_data, tf.uint8)
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    image = tf.cast(image, tf.float32) / 255.0
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

train_csv_path = None
for p in [
    "/kaggle/data/train.csv",
    "/kaggle/input/train.csv",
    os.path.join(GCS_DS_PATH, "train.csv"),
]:
    if tf.io.gfile.exists(p):
        train_csv_path = p
        break
if train_csv_path is None:
    raise FileNotFoundError("Could not locate train.csv for patient_id split.")

train_df = pd.read_csv(train_csv_path, usecols=["image_name", "patient_id"])
patient_ids = train_df["patient_id"].astype(str).values
unique_p = pd.unique(patient_ids)

rng = np.random.RandomState(2020)
rng.shuffle(unique_p)
n_val_p = max(1, int(0.10 * len(unique_p)))
val_patients = set(unique_p[:n_val_p])

val_image_names = set(
    train_df.loc[
        train_df["patient_id"].astype(str).isin(val_patients), "image_name"
    ].values
)


def _name_in_val_from_id(idbytes):
    s = tf.strings.decode_utf8(idbytes)
    return tf.constant(s.numpy().decode("utf-8") in val_image_names)


def _filter_to_val(img, lbl):
    return True



n_shards = len(TRAINING_FILENAMES)
if n_shards > 0:
    shard_idx = (
        pd.util.hash_pandas_object(train_df["image_name"], index=False).astype(np.int64)
        % n_shards
    ).values
    train_df = train_df.assign(_shard=shard_idx)
    shard_patient_frac = (
        train_df.groupby("_shard")["patient_id"]
        .apply(lambda x: x.astype(str).isin(val_patients).mean())
        .to_dict()
    )
    shard_order = sorted(
        range(n_shards), key=lambda i: shard_patient_frac.get(i, 0.0), reverse=True
    )
    n_val = max(1, int(0.10 * n_shards))
    val_shards = set(shard_order[:n_val])

    VAL_FILENAMES = [TRAINING_FILENAMES[i] for i in range(n_shards) if i in val_shards]
    TRAIN_FILENAMES = [
        TRAINING_FILENAMES[i] for i in range(n_shards) if i not in val_shards
    ]
else:
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



## --- ERROR in cell 9, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mInvalidArgumentError[0m                      Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1678995288.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0mprint[0m[0;34m([0m[0;34m"Training data shapes:"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0;32mfor[0m [0mimage[0m[0;34m,[0m [0mlabel[0m [0;32min[0m [0mget_training_dataset[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mtake[0m[0;34m([0m[0;36m1[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m     [0mprint[0m[0;34m([0m[0mimage[0m[0;34m.[0m[0mnumpy[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mshape[0m[0;34m,[0m [0mlabel[0m[0;34m.[0m[0mnumpy[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mshape[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0mprint[0m[0;34m([0m[0;34m"Training data label examples:"[0m[0;34m,[0m [0mlabel[0m[0;34m.[0m[0mnumpy[0m[0;34m([0m[0;34m)[0m[0;34m[[0m[0;34m:[0m[0;36m20[0m[0;34m][0m[0;34m.[0m[0mreshape[0m[0;34m([0m[0;34m-[0m[0;36m1[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py[0m in [0;36m__next__[0;34m(self)[0m
[1;32m    824[0m   [0;32mdef[0m [0m__next__[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    825[0m     [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 826[0;31m       [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_next_internal[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    827[0m     [0;32mexcept[0m [0merrors[0m[0;34m.[0m[0mOutOfRangeError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    828[0m       [0;32mraise[0m [0mStopIteration[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py[0m in [0;36m_next_internal[0;34m(self)[0m
[1;32m    774[0m     [0;31m# to communicate that there is no more data to iterate over.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    775[0m     [0;32mwith[0m [0mcontext[0m[0;34m.[0m[0mexecution_mode[0m[0;34m([0m[0mcontext[0m[0;34m.[0m[0mSYNC[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 776[0;31m       ret = gen_dataset_ops.iterator_get_next(
[0m[1;32m    777[0m           [0mself[0m[0;34m.[0m[0m_iterator_resource[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    778[0m           [0moutput_types[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0m_flat_output_types[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_dataset_ops.py[0m in [0;36miterator_get_next[0;34m(iterator, output_types, output_shapes, name)[0m
[1;32m   3084[0m       [0;32mreturn[0m [0m_result[0m[0;34m[0m[0;34m[0m[0m
[1;32m   3085[0m     [0;32mexcept[0m [0m_core[0m[0;34m.[0m[0m_NotOkStatusException[0m [0;32mas[0m [0me[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 3086[0;31m       [0m_ops[0m[0;34m.[0m[0mraise_from_not_ok_status[0m[0;34m([0m[0me[0m[0;34m,[0m [0mname[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   3087[0m     [0;32mexcept[0m [0m_core[0m[0;34m.[0m[0m_FallbackException[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   3088[0m       [0;32mpass[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py[0m in [0;36mraise_from_not_ok_status[0;34m(e, name)[0m
[1;32m   6000[0m [0;32mdef[0m [0mraise_from_not_ok_status[0m[0;34m([0m[0me[0m[0;34m,[0m [0mname[0m[0;34m)[0m [0;34m->[0m [0mNoReturn[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   6001[0m   [0me[0m[0;34m.[0m[0mmessage[0m [0;34m+=[0m [0;34m([0m[0;34m" name: "[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mname[0m [0;32mif[0m [0mname[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m [0;32melse[0m [0;34m""[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 6002[0;31m   [0;32mraise[0m [0mcore[0m[0;34m.[0m[0m_status_to_exception[0m[0;34m([0m[0me[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m  [0;31m# pylint: disable=protected-access[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   6003[0m [0;34m[0m[0m
[1;32m   6004[0m [0;34m[0m[0m

[0;31mInvalidArgumentError[0m: {{function_node __wrapped__IteratorGetNext_output_types_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to ParallelMapDatasetV2:3 transformation with iterator: Iterator::Root::Prefetch::FiniteTake::Prefetch::BatchV2::Shuffle::ForeverRepeat[0]::ParallelMapV2::ParallelMapV2: Input to reshape is a tensor with 181693 values, but the requested shape has 3145728
	 [[{{node Reshape}}]] [Op:IteratorGetNext] name: 

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
