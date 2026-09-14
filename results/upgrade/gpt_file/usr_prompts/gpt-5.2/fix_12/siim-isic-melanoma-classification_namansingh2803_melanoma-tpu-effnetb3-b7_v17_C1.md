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
import re
import math
import numpy as np
import pandas as pd

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
os.environ["TF_ENABLE_XLA"] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_experimental_options(
        {
            "layout_optimizer": True,
            "constant_folding": True,
            "shape_optimization": True,
            "remapping": True,
            "arithmetic_optimization": True,
            "dependency_optimization": True,
            "loop_optimization": True,
            "function_optimization": True,
        }
    )
except Exception:
    pass

print("TF:", tf.__version__)



## === cell 1
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



## === cell 2
AUTO = tf.data.experimental.AUTOTUNE

BASE_PATH = "/kaggle/input/siim-isic-melanoma-classification"

EPOCHS = 10
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
IMAGE_SIZE = [1024, 1024]

HEIGHT = IMAGE_SIZE[0]
WIDTH = IMAGE_SIZE[1]
CHANNELS = 3

print("BASE_PATH:", BASE_PATH)
print("BATCH_SIZE:", BATCH_SIZE, "IMAGE_SIZE:", IMAGE_SIZE)



## === cell 3
sub = pd.read_csv(f"{BASE_PATH}/sample_submission.csv")
train = pd.read_csv(f"{BASE_PATH}/train.csv")
test = pd.read_csv(f"{BASE_PATH}/test.csv")

print("train:", train.shape, "test:", test.shape, "sub:", sub.shape)
print(sub.head())



## === cell 4
TRAINING_FILENAMES = tf.io.gfile.glob(
    os.path.join(BASE_PATH, "tfrecords", "train*.tfrec")
)
TEST_FILENAMES = tf.io.gfile.glob(os.path.join(BASE_PATH, "tfrecords", "test*.tfrec"))

CLASSES = [0, 1]

print(
    "TFRecords:",
    len(TRAINING_FILENAMES),
    "train files,",
    len(TEST_FILENAMES),
    "test files",
)
assert len(TRAINING_FILENAMES) > 0, "No training TFRecords found."
assert len(TEST_FILENAMES) > 0, "No test TFRecords found."



## === cell 5
AUTO = tf.data.experimental.AUTOTUNE

LABELED_TFREC_FORMAT = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
UNLABELED_TFREC_FORMAT = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def read_labeled_tfrecord(example):
    ex = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = tf.image.decode_jpeg(ex["image"], channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.ensure_shape(image, [IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    label = tf.cast(ex["target"], tf.int32)
    return image, label


@tf.function
def read_unlabeled_tfrecord(example):
    ex = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = tf.image.decode_jpeg(ex["image"], channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.ensure_shape(image, [IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    image_name = ex["image_name"]
    return image, image_name


@tf.function
def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label


def load_dataset(filenames, labeled=True, ordered=False):
    options = tf.data.Options()
    options.experimental_deterministic = bool(ordered)
    try:
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        options.experimental_optimization.map_and_batch_fusion = True
        options.experimental_optimization.autotune_buffers = True
    except Exception:
        pass

    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    ds = ds.with_options(options)

    if labeled:
        ds = ds.map(
            read_labeled_tfrecord, num_parallel_calls=AUTO, deterministic=ordered
        )
    else:
        ds = ds.map(
            read_unlabeled_tfrecord, num_parallel_calls=AUTO, deterministic=ordered
        )
    return ds


def _maybe_prefetch_to_device(ds):
    try:
        gpus = tf.config.list_logical_devices("GPU")
        if gpus:
            return ds.apply(
                tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=1)
            )
    except Exception:
        pass
    return ds


def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES, labeled=True, ordered=False)

    shuffle_buf = min(8192, max(2048, BATCH_SIZE * 128))
    dataset = dataset.shuffle(shuffle_buf, seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.repeat()

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO, deterministic=False)

    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)
    dataset = _maybe_prefetch_to_device(dataset)
    return dataset


def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)
    dataset = dataset.prefetch(AUTO)
    dataset = _maybe_prefetch_to_device(dataset)
    return dataset


def count_data_items(filenames):
    n = [
        int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
        for filename in filenames
    ]
    return np.sum(n)


NUM_TRAINING_IMAGES = int(count_data_items(TRAINING_FILENAMES))
NUM_TEST_IMAGES = int(count_data_items(TEST_FILENAMES))
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
TEST_STEPS = int(math.ceil(NUM_TEST_IMAGES / BATCH_SIZE))

print(
    f"Dataset: {NUM_TRAINING_IMAGES} training images, {NUM_TEST_IMAGES} unlabeled test images"
)
print("STEPS_PER_EPOCH:", STEPS_PER_EPOCH)
print("TEST_STEPS:", TEST_STEPS)




## === cell 6
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


lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)



## === cell 7
with strategy.scope():
    model2 = tf.keras.Sequential(
        [
            tf.keras.applications.EfficientNetB0(
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
    metrics=["accuracy"],
    jit_compile=True,
    steps_per_execution=32,
)
model2.summary()



## === cell 8
CKPT_PATH = "/kaggle/working/model2.weights.h5"

train_ds = get_training_dataset()

if tf.io.gfile.exists(CKPT_PATH):
    print("Found cached weights, loading:", CKPT_PATH)
    model2.load_weights(CKPT_PATH)
    history2 = None
else:
    history2 = model2.fit(
        train_ds,
        epochs=EPOCHS,
        callbacks=[lr_schedule],
        steps_per_epoch=STEPS_PER_EPOCH,
    )
    model2.save_weights(CKPT_PATH)
    print("Saved weights to:", CKPT_PATH)



## === cell 9
test_ds = get_test_dataset(ordered=True)

print("Computing predictions...")
test_ids = test["image_name"].to_numpy(dtype=str)

test_images_ds = test_ds.map(
    lambda image, image_name: image, num_parallel_calls=AUTO, deterministic=True
)

probabilities2 = model2.predict(test_images_ds, steps=TEST_STEPS, verbose=1)
probabilities2 = np.asarray(probabilities2).reshape(-1)

min_len = min(len(test_ids), len(probabilities2))
test_ids = test_ids[:min_len]
probabilities2 = probabilities2[:min_len]

print(
    "Preds:",
    probabilities2.shape,
    "min/max:",
    float(probabilities2.min()),
    float(probabilities2.max()),
)

print("Generating submission.csv file...")
pred_df2 = pd.DataFrame({"image_name": test_ids, "target": probabilities2})

submission = sub[["image_name"]].merge(pred_df2, on="image_name", how="left")

if submission["target"].isna().any():
    submission["target"] = submission["target"].fillna(float(np.mean(probabilities2)))

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
assert submission.shape[0] == sub.shape[0]
assert list(submission.columns) == ["image_name", "target"]
assert submission["target"].notna().all()
