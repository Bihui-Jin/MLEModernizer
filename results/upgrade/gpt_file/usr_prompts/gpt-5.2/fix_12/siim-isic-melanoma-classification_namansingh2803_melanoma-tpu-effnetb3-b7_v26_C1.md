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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import re
import math
import random

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L

from sklearn.model_selection import train_test_split  # (kept; harmless even if unused)

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TensorFlow:", tf.__version__)



## === cell 1
print("Listing /kaggle/input exists:", os.path.exists("/kaggle/input"))
print(
    "Listing competition dataset folder exists:",
    os.path.exists("/kaggle/input/siim-isic-melanoma-classification"),
)



## === cell 2
BASE_PATH_CANDIDATES = [
    "/kaggle/input/siim-isic-melanoma-classification",
    "/kaggle/data/siim-isic-melanoma-classification",
    "/kaggle/input",
    "/kaggle/data",
]
BASE_PATH = None
for p in BASE_PATH_CANDIDATES:
    if os.path.exists(p) and os.path.exists(os.path.join(p, "train.csv")):
        BASE_PATH = p
        break
if BASE_PATH is None:
    BASE_PATH = "/kaggle/input/siim-isic-melanoma-classification"

print("Resolved BASE_PATH:", BASE_PATH)



## === cell 3
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.master())
except Exception:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)



## === cell 4
AUTO = tf.data.AUTOTUNE

EPOCHS = 10
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
IMAGE_SIZE = [1024, 1024]

print("BASE_PATH:", BASE_PATH)
print("BATCH_SIZE:", BATCH_SIZE, "IMAGE_SIZE:", IMAGE_SIZE, "EPOCHS:", EPOCHS)



## === cell 5
HEIGHT = IMAGE_SIZE[0]
WIDTH = IMAGE_SIZE[1]
CHANNELS = 3




## === cell 6
def append_path(pre):
    return np.vectorize(lambda file: os.path.join(BASE_PATH, pre, file))




## === cell 7
sub = pd.read_csv(f"{BASE_PATH}/sample_submission.csv")
print("sample_submission:", sub.shape, sub.columns.tolist())
sub.head()



## === cell 8
pass



## === cell 9
pass



## === cell 10
TRAINING_FILENAMES_ALL = sorted(tf.io.gfile.glob(f"{BASE_PATH}/tfrecords/train*.tfrec"))
TEST_FILENAMES = sorted(tf.io.gfile.glob(f"{BASE_PATH}/tfrecords/test*.tfrec"))

if len(TRAINING_FILENAMES_ALL) == 0 or len(TEST_FILENAMES) == 0:
    raise FileNotFoundError(
        "TFRecord files not found. Expected under "
        f"{BASE_PATH}/tfrecords. Found train={len(TRAINING_FILENAMES_ALL)}, test={len(TEST_FILENAMES)}"
    )

VALIDATION_FRACTION = 0.1
n_val = max(1, int(len(TRAINING_FILENAMES_ALL) * VALIDATION_FRACTION))
VALIDATION_FILENAMES = TRAINING_FILENAMES_ALL[:n_val]
TRAINING_FILENAMES = TRAINING_FILENAMES_ALL[n_val:]

print(
    "TFRecords:",
    len(TRAINING_FILENAMES_ALL),
    "train files,",
    len(TRAINING_FILENAMES),
    "used for training,",
    len(VALIDATION_FILENAMES),
    "used for validation",
)
print("Example train tfrec:", TRAINING_FILENAMES[0])
print("Example test tfrec:", TEST_FILENAMES[0])

CLASSES = [0, 1]



## === cell 11
pass



## === cell 12
LABELED_TFREC_FORMAT = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
UNLABELED_TFREC_FORMAT = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function(reduce_retracing=True)
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    return image


@tf.function(reduce_retracing=True)
def read_labeled_tfrecord(example):
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    label = tf.cast(example["target"], tf.int32)
    return image, label


@tf.function(reduce_retracing=True)
def read_unlabeled_tfrecord(example):
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    idnum = example["image_name"]
    return image, idnum


def load_dataset(filenames, labeled=True, ordered=False):
    opts = tf.data.Options()
    try:
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_optimization.autotune_buffers = True
    except Exception:
        pass
    try:
        opts.experimental_deterministic = bool(ordered)
    except Exception:
        pass

    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    ds = ds.with_options(opts)
    ds = ds.map(
        read_labeled_tfrecord if labeled else read_unlabeled_tfrecord,
        num_parallel_calls=AUTO,
        deterministic=ordered,
    )
    return ds


@tf.function(reduce_retracing=True)
def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label


def get_training_dataset():
    ds = load_dataset(TRAINING_FILENAMES, labeled=True, ordered=False)
    ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.repeat()
    ds = ds.map(data_augment, num_parallel_calls=AUTO)
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    ds = ds.prefetch(AUTO)
    return ds


def get_validation_dataset(ordered=False):
    ds = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTO)
    return ds


def get_test_dataset(ordered=False):
    ds = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTO)
    return ds


def count_data_items(filenames):
    n = [
        int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
        for filename in filenames
    ]
    return np.sum(n)


NUM_TRAINING_IMAGES = int(count_data_items(TRAINING_FILENAMES))
NUM_VALIDATION_IMAGES = int(count_data_items(VALIDATION_FILENAMES))
NUM_TEST_IMAGES = int(count_data_items(TEST_FILENAMES))

STEPS_PER_EPOCH = max(1, NUM_TRAINING_IMAGES // BATCH_SIZE)
VALIDATION_STEPS = max(1, math.ceil(NUM_VALIDATION_IMAGES / BATCH_SIZE))
TEST_STEPS = max(1, math.ceil(NUM_TEST_IMAGES / BATCH_SIZE))

print(
    f"Dataset: {NUM_TRAINING_IMAGES} train, {NUM_VALIDATION_IMAGES} val, {NUM_TEST_IMAGES} test"
)
print(
    "STEPS_PER_EPOCH:",
    STEPS_PER_EPOCH,
    "VALIDATION_STEPS:",
    VALIDATION_STEPS,
    "TEST_STEPS (unused for id alignment):",
    TEST_STEPS,
)




## === cell 13
def build_lrfn(
    lr_start=0.00001,
    lr_max=0.0001,
    lr_min=0.000001,
    lr_rampup_epochs=20,
    lr_sustain_epochs=0,
    lr_exp_decay=0.8,
):
    lr_max = lr_max * strategy.num_replicas_in_sync

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_exp_decay ** (
                epoch - lr_rampup_epochs - lr_sustain_epochs
            ) + lr_min
        return lr

    return lrfn




## === cell 14
with strategy.scope():
    model = tf.keras.Sequential(
        [
            tf.keras.applications.EfficientNetB6(
                input_shape=(*IMAGE_SIZE, 3), weights="imagenet", include_top=False
            ),
            L.GlobalAveragePooling2D(),
            L.Dense(512, activation="relu"),
            L.Dense(128, activation="relu"),
            L.Dense(1, activation="sigmoid"),
        ]
    )

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    jit_compile=True,
)
model.summary()



## === cell 15
with strategy.scope():
    model2 = tf.keras.Sequential(
        [
            tf.keras.applications.EfficientNetB3(
                input_shape=(*IMAGE_SIZE, 3), weights="imagenet", include_top=False
            ),
            L.GlobalAveragePooling2D(),
            L.Dense(512, activation="relu"),
            L.Dropout(0.3),
            L.Dense(256, activation="relu"),
            L.Dropout(0.25),
            L.Dense(128, activation="relu"),
            L.Dropout(0.2),
            L.Dense(1, activation="sigmoid"),
        ]
    )

model2.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    jit_compile=True,
)
model2.summary()



## === cell 16
lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=0)

STEPS_PER_EPOCH = max(1, NUM_TRAINING_IMAGES // BATCH_SIZE)

train_ds = get_training_dataset()
val_ds = get_validation_dataset(ordered=True)



## === cell 17
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=[lr_schedule],
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=val_ds,
    validation_steps=VALIDATION_STEPS,
    verbose=1,
)



## === cell 18
history2 = model2.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=[lr_schedule],
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=val_ds,
    validation_steps=VALIDATION_STEPS,
    verbose=1,
)



## === cell 19
pass



## === cell 20
test_ds = get_test_dataset(ordered=True)
print("Computing predictions...")

all_ids = []
all_p1 = []
all_p2 = []

for batch_images, batch_ids in test_ds:
    p1 = model(batch_images, training=False)
    p2 = model2(batch_images, training=False)
    all_p1.append(p1.numpy())
    all_p2.append(p2.numpy())
    all_ids.append(batch_ids.numpy())

test_ids = np.concatenate(all_ids, axis=0).astype("U")
probabilities1 = np.concatenate(all_p1, axis=0)
probabilities2 = np.concatenate(all_p2, axis=0)

probabilities = 0.5 * probabilities1 + 0.5 * probabilities2
probabilities = np.clip(probabilities.reshape(-1), 0.0, 1.0)

print("Collected test ids:", len(test_ids), "preds:", len(probabilities))



## === cell 21
sub1 = sub.copy()
sub2 = sub.copy()



## === cell 22
print("Generating submission.csv file...")
print("Test IDs:", len(test_ids), "Preds:", probabilities.shape)



## === cell 23
pred_df = pd.DataFrame({"image_name": test_ids, "target": probabilities})
pred_df.head()



## === cell 24
pred_df = pred_df.drop_duplicates(subset=["image_name"], keep="first")
pred_map = dict(zip(pred_df["image_name"].values, pred_df["target"].values))

sub_out = sub.copy()
sub_out["target"] = sub_out["image_name"].map(pred_map)

if sub_out["target"].isna().any():
    sub_out["target"] = sub_out["target"].fillna(
        float(np.nanmean(pred_df["target"].values))
    )

sub_out["target"] = np.clip(sub_out["target"].astype(np.float32), 0.0, 1.0)

sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
sub_out.head()



## === cell 25
sub1.head()



## === cell 26
sub2.head()



## === cell 27
pass



## === cell 28
pass



## === cell 29
pass



## === cell 30
sub_out.to_csv("submission-sample.csv", index=False)
sub_out.head()



## === cell 31
pass



## === cell 32
pass



## === cell 33
pass



## === cell 34
print("Final submission file ready at: submission.csv")



## === cell 35
model.save("EffNetB6-Melanoma.h5")
model2.save("EffNetB3-Melanoma.h5")
print("Saved model files.")
