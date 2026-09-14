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
Detect the presence and position of catheters and lines on chest x-rays.

## Metric
Area under the ROC curve for each label, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each ID in the test set, you must predict a probability for all target variables. The file should contain a header and have the following format:
```
StudyInstanceUID,ETT - Abnormal,ETT - Borderline,ETT - Normal,NGT - Abnormal,NGT - Borderline,NGT - Incompletely Imaged,NGT - Normal,CVC - Abnormal,CVC - Borderline,CVC - Normal,Swan Ganz Catheter Present
1.2.826.0.1.3680043.8.498.62451881164053375557257228990443168843,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.83721761279899623084220697845011427274,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.12732270010839808189235995393981377825,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.11769539755086084996287023095028033598,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.87838627504097587943394933987052577153,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.53211840524738036417560823327351887819,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.93555795394184819372299157360228027866,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.52241894131170494723503100795076463919,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.36500167484503936720548852591033878284,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.86199852603457900780565655267977637728,0,0,0,0,0,0,0,0,0,0,0
```

## Dataset
`train.csv` contains image IDs, binary labels, and patient IDs.

TFRecords are available for both train and test.

We've also included `train_annotations.csv`. These are segmentation annotations for training samples that have them. They are included solely as additional information for competitors.

- train.csv - contains image IDs, binary labels, and patient IDs.
- sample_submission.csv - a sample submission file in the correct format
- test - test images
- train - training images

### Columns
- `StudyInstanceUID` - unique ID for each image
- `ETT - Abnormal` - endotracheal tube placement abnormal
- `ETT - Borderline` - endotracheal tube placement borderline abnormal
- `ETT - Normal` - endotracheal tube placement normal
- `NGT - Abnormal` - nasogastric tube placement abnormal
- `NGT - Borderline` - nasogastric tube placement borderline abnormal
- `NGT - Incompletely Imaged` - nasogastric tube placement inconclusive due to imaging
- `NGT - Normal` - nasogastric tube placement borderline normal
- `CVC - Abnormal` - central venous catheter placement abnormal
- `CVC - Borderline` - central venous catheter placement borderline abnormal
- `CVC - Normal` - central venous catheter placement normal
- `Swan Ganz Catheter Present`
- `PatientID` - unique ID for each patient in the dataset

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
        input/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
        working/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
```

-> data/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/ranzcr-clip-catheter-line-classification/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/ranzcr-clip-catheter-line-classification/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> data/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> (stopped after 10 files for performance)

# 5. Target score

0.5084721608910747

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

import math, re, warnings, random, glob
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
SEED = 555
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
warnings.filterwarnings("ignore")

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass



## === cell 2
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



## === cell 3
database_base_path = "/kaggle/input/ranzcr-clip-catheter-line-classification/"



## === cell 4
BATCH_SIZE = 16 * REPLICAS
HEIGHT = 512
WIDTH = 512
CHANNELS = 3
N_CLASSES = 11
TTA_STEPS = 0  # keep deterministic/stable and within runtime
IMAGE_SIZE = [512, 512]
AUG_BATCH = BATCH_SIZE




## === cell 5
def count_tfrecord_items(filenames):
    if not filenames:
        return 0
    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    return int(ds.reduce(tf.constant(0, dtype=tf.int64), lambda x, _: x + 1).numpy())




## === cell 6
def data_augment(image, label):
    image = tf.image.rot90(
        image, k=tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32, seed=SEED)
    )
    image = tf.image.random_flip_left_right(image, seed=SEED)
    image = tf.image.random_flip_up_down(image, seed=SEED)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_saturation(image, 0.8, 1.2, seed=SEED)
    return image, label




## === cell 7
train_csv_path = os.path.join(database_base_path, "train.csv")
sample_sub_path = os.path.join(database_base_path, "sample_submission.csv")
train_img_dir = os.path.join(database_base_path, "train")
test_img_dir = os.path.join(database_base_path, "test")

train_df = pd.read_csv(train_csv_path)
submission = pd.read_csv(sample_sub_path)

print("train_df:", train_df.shape)
print("sample_submission:", submission.shape)

target_cols_full = [
    "ETT - Abnormal",
    "ETT - Borderline",
    "ETT - Normal",
    "NGT - Abnormal",
    "NGT - Borderline",
    "NGT - Incompletely Imaged",
    "NGT - Normal",
    "CVC - Abnormal",
    "CVC - Borderline",
    "CVC - Normal",
    "Swan Ganz Catheter Present",
]
missing = [c for c in target_cols_full if c not in train_df.columns]
if missing:
    raise ValueError(f"train.csv missing expected target columns: {missing}")




## === cell 8
def to_float32_2(image, label):
    max_val = tf.reduce_max(label, axis=-1, keepdims=True)
    cond = tf.equal(label, max_val)
    label = tf.where(cond, tf.ones_like(label), tf.zeros_like(label))
    return tf.cast(image, tf.float32), tf.cast(label, tf.int32)


def to_float32(image, label):
    return tf.cast(image, tf.float32), label


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    return image


TRAIN_FEATURE_DESCRIPTION = {
    "CVC - Abnormal": tf.io.FixedLenFeature([], tf.int64),
    "CVC - Borderline": tf.io.FixedLenFeature([], tf.int64),
    "CVC - Normal": tf.io.FixedLenFeature([], tf.int64),
    "ETT - Abnormal": tf.io.FixedLenFeature([], tf.int64),
    "ETT - Borderline": tf.io.FixedLenFeature([], tf.int64),
    "ETT - Normal": tf.io.FixedLenFeature([], tf.int64),
    "NGT - Abnormal": tf.io.FixedLenFeature([], tf.int64),
    "NGT - Borderline": tf.io.FixedLenFeature([], tf.int64),
    "NGT - Incompletely Imaged": tf.io.FixedLenFeature([], tf.int64),
    "NGT - Normal": tf.io.FixedLenFeature([], tf.int64),
    "StudyInstanceUID": tf.io.FixedLenFeature([], tf.string),
    "Swan Ganz Catheter Present": tf.io.FixedLenFeature([], tf.int64),
    "image": tf.io.FixedLenFeature([], tf.string),
}
UNLABELED_TFREC_FORMAT = {
    "StudyInstanceUID": tf.io.FixedLenFeature([], tf.string),
    "image": tf.io.FixedLenFeature([], tf.string),
}


def read_labeled_tfrecord(example):
    example = tf.io.parse_single_example(example, TRAIN_FEATURE_DESCRIPTION)
    image = decode_image(example["image"])
    image = tf.image.resize(image, [IMAGE_SIZE[0], IMAGE_SIZE[1]])

    values = tf.stack(
        [
            tf.cast(example["ETT - Abnormal"], tf.int32),
            tf.cast(example["ETT - Borderline"], tf.int32),
            tf.cast(example["ETT - Normal"], tf.int32),
            tf.cast(example["NGT - Abnormal"], tf.int32),
            tf.cast(example["NGT - Borderline"], tf.int32),
            tf.cast(example["NGT - Incompletely Imaged"], tf.int32),
            tf.cast(example["NGT - Normal"], tf.int32),
            tf.cast(example["CVC - Abnormal"], tf.int32),
            tf.cast(example["CVC - Borderline"], tf.int32),
            tf.cast(example["CVC - Normal"], tf.int32),
            tf.cast(example["Swan Ganz Catheter Present"], tf.int32),
        ],
        axis=0,
    )
    label = tf.argmax(values, axis=0, output_type=tf.int32)
    return image, label


def read_unlabeled_tfrecord(example):
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    image = tf.image.resize(image, [IMAGE_SIZE[0], IMAGE_SIZE[1]])
    image_name = example["StudyInstanceUID"]
    return image, image_name


def load_dataset(filenames, labeled=True, ordered=False):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = bool(ordered)

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(
        read_labeled_tfrecord if labeled else read_unlabeled_tfrecord,
        num_parallel_calls=AUTO,
    )
    return dataset




## === cell 9
print(submission.head())

TRAIN_FILENAMES = sorted(
    tf.io.gfile.glob(f"{database_base_path}train_tfrecords/*.tfrec")
)
TEST_FILENAMES = sorted(tf.io.gfile.glob(f"{database_base_path}test_tfrecords/*.tfrec"))
print(f"Found TFRecords: train={len(TRAIN_FILENAMES)} test={len(TEST_FILENAMES)}")

if not TRAIN_FILENAMES:
    raise FileNotFoundError(
        "No train TFRecords found at /kaggle/input/.../train_tfrecords/*.tfrec"
    )
if not TEST_FILENAMES:
    raise FileNotFoundError(
        "No test TFRecords found at /kaggle/input/.../test_tfrecords/*.tfrec"
    )

n_total = int(len(train_df))
val_size = int(0.1 * n_total)
train_size = n_total - val_size
print("Split sizes:", train_size, val_size)

full_train_ds = load_dataset(TRAIN_FILENAMES, labeled=True, ordered=True)

shuffle_buf = int(min(n_total, 4096))
full_train_ds = full_train_ds.shuffle(
    shuffle_buf, seed=SEED, reshuffle_each_iteration=False
)

val_ds = full_train_ds.take(val_size)
train_ds = full_train_ds.skip(val_size)

train_ds = train_ds.map(data_augment, num_parallel_calls=AUTO)

val_ds = val_ds.cache()

train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)

test_ds = (
    load_dataset(TEST_FILENAMES, labeled=False, ordered=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

NUM_TEST_IMAGES = int(len(submission))
print(f"NUM_TEST_IMAGES (from sample_submission): {NUM_TEST_IMAGES}")

print("Prepared datasets from TFRecords:")
print(" train:", train_size, "val:", val_size, "test:", NUM_TEST_IMAGES)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3174164427.py in <cell line: 0>()
     11 
     12 if not TRAIN_FILENAMES:
---> 13     raise FileNotFoundError(
     14         "No train TFRecords found at /kaggle/input/.../train_tfrecords/*.tfrec"
     15     )

FileNotFoundError: No train TFRecords found at /kaggle/input/.../train_tfrecords/*.tfrec

## === cell 10
with strategy.scope():
    K.clear_session()
    model = Sequential(
        [
            L.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
            L.Conv2D(16, 3, padding="same", activation="relu"),
            L.MaxPool2D(),
            L.Conv2D(32, 3, padding="same", activation="relu"),
            L.MaxPool2D(),
            L.Conv2D(64, 3, padding="same", activation="relu"),
            L.GlobalAveragePooling2D(),
            L.Dense(64, activation="relu"),
            L.Dense(N_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=[],
    )

EPOCHS = 1
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1563982723.py in <cell line: 0>()
     21 
     22 EPOCHS = 1
---> 23 history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
     24 

NameError: name 'train_ds' is not defined

## === cell 11
models = [model]
print("Using in-notebook trained model (no external .h5 found).")




## === cell 12
def predict_proba(models, test_ds, tta_steps=0):
    if tta_steps and tta_steps > 0:
        probs_steps = []
        for step in range(tta_steps):
            print(f"TTA step {step+1}/{tta_steps}")
            aug_ds = test_ds.map(lambda img, uid: (img, uid), num_parallel_calls=AUTO)
            aug_ds = aug_ds.map(
                lambda img, uid: (data_augment(img, tf.constant(0))[0], uid),
                num_parallel_calls=AUTO,
            )
            imgs_only = aug_ds.map(lambda img, uid: img, num_parallel_calls=AUTO)
            probs = np.average(
                [m.predict(imgs_only, verbose=0) for m in models], axis=0
            )
            probs_steps.append(probs)
        probabilities = np.mean(probs_steps, axis=0)
    else:
        imgs_only = test_ds.map(lambda img, uid: img, num_parallel_calls=AUTO)
        probabilities = np.average(
            [m.predict(imgs_only, verbose=0) for m in models], axis=0
        )

    probabilities = np.asarray(probabilities)
    probabilities = np.clip(probabilities, 0.0, 1.0)
    return probabilities




## === cell 13
print(" TTA_STEPS = {} ".format(TTA_STEPS))
probabilities = predict_proba(models, test_ds, tta_steps=TTA_STEPS)
print("Predictions shape:", probabilities.shape)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2302460907.py in <cell line: 0>()
      1 print(" TTA_STEPS = {} ".format(TTA_STEPS))
----> 2 probabilities = predict_proba(models, test_ds, tta_steps=TTA_STEPS)
      3 print("Predictions shape:", probabilities.shape)
      4 

NameError: name 'test_ds' is not defined

## === cell 14
if probabilities.ndim != 2 or probabilities.shape[1] != N_CLASSES:
    raise ValueError(
        f"Unexpected probabilities shape {probabilities.shape}; expected (*, {N_CLASSES})"
    )

probs_full = probabilities  # shape [N_test, 11] aligned with target_cols_full



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3147057112.py in <cell line: 0>()
----> 1 if probabilities.ndim != 2 or probabilities.shape[1] != N_CLASSES:
      2     raise ValueError(
      3         f"Unexpected probabilities shape {probabilities.shape}; expected (*, {N_CLASSES})"
      4     )
      5 

NameError: name 'probabilities' is not defined

## === cell 15
expected_cols = list(submission.columns)
id_col = expected_cols[0]
target_cols = expected_cols[1:]

print(f"Sample submission targets: {len(target_cols)} columns")
print("Targets:", target_cols)

test_uids = submission[id_col].values.astype(str)

sub_df = pd.DataFrame({id_col: test_uids})

for c in target_cols:
    if c in target_cols_full:
        j = target_cols_full.index(c)
        sub_df[c] = probs_full[:, j].astype(np.float32)
    else:
        sub_df[c] = 0.0

sub_df = sub_df[expected_cols]



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/802797435.py in <cell line: 0>()
     14     if c in target_cols_full:
     15         j = target_cols_full.index(c)
---> 16         sub_df[c] = probs_full[:, j].astype(np.float32)
     17     else:
     18         sub_df[c] = 0.0

NameError: name 'probs_full' is not defined

## === cell 16
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print(f"Wrote {sub_path} with shape {sub_df.shape} and columns: {list(sub_df.columns)}")

with open(sub_path, "r") as f:
    for _ in range(5):
        print(f.readline().rstrip("\n"))

## --- ERROR in outputing the csv:
Invalid submission: Class ETT - Abnormal is not in the submission.
