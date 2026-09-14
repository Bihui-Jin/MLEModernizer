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

# 5. Target score

0.8956746504152059

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import re
import math
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K

from sklearn.model_selection import train_test_split

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

tf.config.optimizer.set_jit(True)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception:
    pass

print("TF version:", tf.__version__)
print("Listing input root (first 20):", os.listdir("/kaggle/input")[:20])



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR = "/kaggle/input/siim-isic-melanoma-classification"
assert os.path.exists(DATA_DIR), f"Expected dataset at {DATA_DIR}"
print("Dataset dir OK:", DATA_DIR)
print("Has tfrecords:", os.path.exists(os.path.join(DATA_DIR, "tfrecords")))



## === cell 2
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



## === cell 3
AUTO = tf.data.experimental.AUTOTUNE

GCS_PATH = DATA_DIR  # keep variable name to preserve core logic downstream

EPOCHS = 10
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
IMAGE_SIZE = [1024, 1024]

print("BATCH_SIZE:", BATCH_SIZE, "IMAGE_SIZE:", IMAGE_SIZE)



## === cell 4
HEIGHT = IMAGE_SIZE[0]
WIDTH = IMAGE_SIZE[1]
CHANNELS = 3




## === cell 5
def append_path(pre):
    base = os.path.join(GCS_PATH, "jpeg", pre)

    def _fn(files):
        files = np.asarray(files, dtype=str)
        return np.char.add(np.char.add(base + os.sep, files), "")

    return _fn




## === cell 6
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
print(sub.head())
print("Sample submission rows:", len(sub))



## === cell 7
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
print(train[["image_name", "target"]].head())
print("Train rows:", len(train), "pos rate:", train["target"].mean())



## === cell 8
pass



## === cell 9
TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + "/tfrecords/train*.tfrec")
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + "/tfrecords/test*.tfrec")

TRAINING_FILENAMES = sorted(TRAINING_FILENAMES)
TEST_FILENAMES = sorted(TEST_FILENAMES)
trn_files, val_files = train_test_split(
    TRAINING_FILENAMES, test_size=0.1, random_state=SEED
)
TRAINING_FILENAMES = sorted(trn_files)
VALIDATION_FILENAMES = sorted(val_files)

CLASSES = [0, 1]

print(
    "TFRecord shards:",
    len(TRAINING_FILENAMES),
    "train,",
    len(VALIDATION_FILENAMES),
    "val,",
    len(TEST_FILENAMES),
    "test",
)



## === cell 10
_DIM = int(IMAGE_SIZE[0])
_XDIM = _DIM % 2

_x = tf.repeat(tf.range(_DIM // 2, -_DIM // 2, -1), _DIM)
_y = tf.tile(tf.range(-_DIM // 2, _DIM // 2), [_DIM])
_z = tf.ones([_DIM * _DIM], dtype="int32")
_IDX = tf.stack([_x, _y, _z])  # shape (3, DIM*DIM)


@tf.function
def transform_rotation(image):
    rotation = 15.0 * tf.random.normal([1], dtype=tf.float32)
    rotation = tf.constant(math.pi, tf.float32) * rotation / 180.0

    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1], dtype=tf.float32)
    zero = tf.constant([0], dtype=tf.float32)
    rotation_matrix = tf.reshape(
        tf.concat([c1, s1, zero, -s1, c1, zero, zero, zero, one], axis=0), [3, 3]
    )

    idx2 = K.dot(rotation_matrix, tf.cast(_IDX, dtype=tf.float32))
    idx2 = K.cast(idx2, dtype="int32")
    idx2 = K.clip(idx2, -_DIM // 2 + _XDIM + 1, _DIM // 2)

    idx3 = tf.stack([_DIM // 2 - idx2[0, :], _DIM // 2 - 1 + idx2[1, :]])
    d = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d, [_DIM, _DIM, 3])


@tf.function
def transform_shear(image):
    shear = 5.0 * tf.random.normal([1], dtype=tf.float32)
    shear = tf.constant(math.pi, tf.float32) * shear / 180.0

    one = tf.constant([1], dtype=tf.float32)
    zero = tf.constant([0], dtype=tf.float32)
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_matrix = tf.reshape(
        tf.concat([one, s2, zero, zero, c2, zero, zero, zero, one], axis=0), [3, 3]
    )

    idx2 = K.dot(shear_matrix, tf.cast(_IDX, dtype=tf.float32))
    idx2 = K.cast(idx2, dtype="int32")
    idx2 = K.clip(idx2, -_DIM // 2 + _XDIM + 1, _DIM // 2)

    idx3 = tf.stack([_DIM // 2 - idx2[0, :], _DIM // 2 - 1 + idx2[1, :]])
    d = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d, [_DIM, _DIM, 3])


@tf.function
def transform_shift(image):
    height_shift = 16.0 * tf.random.normal([1], dtype=tf.float32)
    width_shift = 16.0 * tf.random.normal([1], dtype=tf.float32)
    one = tf.constant([1], dtype=tf.float32)
    zero = tf.constant([0], dtype=tf.float32)

    shift_matrix = tf.reshape(
        tf.concat(
            [one, zero, height_shift, zero, one, width_shift, zero, zero, one], axis=0
        ),
        [3, 3],
    )

    idx2 = K.dot(shift_matrix, tf.cast(_IDX, dtype=tf.float32))
    idx2 = K.cast(idx2, dtype="int32")
    idx2 = K.clip(idx2, -_DIM // 2 + _XDIM + 1, _DIM // 2)

    idx3 = tf.stack([_DIM // 2 - idx2[0, :], _DIM // 2 - 1 + idx2[1, :]])
    d = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d, [_DIM, _DIM, 3])


@tf.function
def transform_zoom(image):
    height_zoom = 1.0 + tf.random.normal([1], dtype=tf.float32) / 10.0
    width_zoom = 1.0 + tf.random.normal([1], dtype=tf.float32) / 10.0
    one = tf.constant([1], dtype=tf.float32)
    zero = tf.constant([0], dtype=tf.float32)

    zoom_matrix = tf.reshape(
        tf.concat(
            [
                one / height_zoom,
                zero,
                zero,
                zero,
                one / width_zoom,
                zero,
                zero,
                zero,
                one,
            ],
            axis=0,
        ),
        [3, 3],
    )

    idx2 = K.dot(zoom_matrix, tf.cast(_IDX, dtype=tf.float32))
    idx2 = K.cast(idx2, dtype="int32")
    idx2 = K.clip(idx2, -_DIM // 2 + _XDIM + 1, _DIM // 2)

    idx3 = tf.stack([_DIM // 2 - idx2[0, :], _DIM // 2 - 1 + idx2[1, :]])
    d = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d, [_DIM, _DIM, 3])




## === cell 11
def data_augment_chaotic(image, label):
    p_spatial = tf.random.uniform([1], 0, 1, dtype="float32")
    p_spatial2 = tf.random.uniform([1], 0, 1, dtype="float32")
    p_pixel = tf.random.uniform([1], 0, 1, dtype="float32")
    p_crop = tf.random.uniform([1], 0, 1, dtype="float32")

    if p_spatial >= 0.2:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

    if p_crop >= 0.7:
        if p_crop >= 0.95:
            image = tf.image.random_crop(
                image, size=[int(HEIGHT * 0.6), int(WIDTH * 0.6), CHANNELS]
            )
        elif p_crop >= 0.85:
            image = tf.image.random_crop(
                image, size=[int(HEIGHT * 0.7), int(WIDTH * 0.7), CHANNELS]
            )
        elif p_crop >= 0.8:
            image = tf.image.random_crop(
                image, size=[int(HEIGHT * 0.8), int(WIDTH * 0.8), CHANNELS]
            )
        else:
            image = tf.image.random_crop(
                image, size=[int(HEIGHT * 0.9), int(WIDTH * 0.9), CHANNELS]
            )
        image = tf.image.resize(image, size=[HEIGHT, WIDTH])

    if p_spatial2 >= 0.6:
        if p_spatial2 >= 0.9:
            image = transform_rotation(image)
        elif p_spatial2 >= 0.8:
            image = transform_zoom(image)
        elif p_spatial2 >= 0.7:
            image = transform_shift(image)
        else:
            image = transform_shear(image)

    if p_pixel >= 0.4:
        if p_pixel >= 0.85:
            image = tf.image.random_saturation(image, lower=0, upper=2)
        elif p_pixel >= 0.65:
            image = tf.image.random_contrast(image, lower=0.8, upper=2)
        elif p_pixel >= 0.5:
            image = tf.image.random_brightness(image, max_delta=0.2)
        else:
            image = tf.image.adjust_gamma(image, gamma=0.6)

    return image, label




## === cell 12
@tf.function
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image


@tf.function
def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    label = tf.cast(example["target"], tf.int32)
    return image, label


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
    opts = tf.data.Options()
    opts.experimental_deterministic = bool(ordered)
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True
    opts.experimental_optimization.autotune_buffers = True
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(opts)
    dataset = dataset.map(
        read_labeled_tfrecord if labeled else read_unlabeled_tfrecord,
        num_parallel_calls=AUTO,
        deterministic=bool(ordered),
    )
    return dataset


@tf.function
def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label


_TRAIN_DS = None
_VAL_DS_ORDERED = None
_TEST_DS_ORDERED = None
_TEST_IMAGES_DS_ORDERED = None
_TEST_IDS_DS_ORDERED = None


def get_training_dataset():
    global _TRAIN_DS
    if _TRAIN_DS is None:
        dataset = load_dataset(TRAINING_FILENAMES, labeled=True, ordered=False)

        dataset = dataset.cache()

        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
        dataset = dataset.prefetch(AUTO)
        _TRAIN_DS = dataset
    return _TRAIN_DS


def get_validation_dataset(ordered=False):
    global _VAL_DS_ORDERED
    if ordered:
        if _VAL_DS_ORDERED is None:
            dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=True)
            dataset = dataset.batch(BATCH_SIZE)
            dataset = dataset.prefetch(AUTO)
            _VAL_DS_ORDERED = dataset
        return _VAL_DS_ORDERED

    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=False)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset


def get_test_dataset(ordered=False):
    global _TEST_DS_ORDERED
    if ordered:
        if _TEST_DS_ORDERED is None:
            dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=True)
            dataset = dataset.batch(BATCH_SIZE)
            dataset = dataset.prefetch(AUTO)
            _TEST_DS_ORDERED = dataset
        return _TEST_DS_ORDERED

    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=False)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset


_COUNT_RE = re.compile(r"-([0-9]*)\.")


def count_data_items(filenames):
    n = [int(_COUNT_RE.search(filename).group(1)) for filename in filenames]
    return int(np.sum(n))


NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
VALIDATION_STEPS = max(1, NUM_VALIDATION_IMAGES // BATCH_SIZE)

print(
    f"Dataset: {NUM_TRAINING_IMAGES} train, {NUM_VALIDATION_IMAGES} val, {NUM_TEST_IMAGES} test"
)
print("STEPS_PER_EPOCH:", STEPS_PER_EPOCH, "VALIDATION_STEPS:", VALIDATION_STEPS)




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
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"], jit_compile=True
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
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"], jit_compile=True
)

model2.summary()



## === cell 16
lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

STEPS_PER_EPOCH = max(1, NUM_TRAINING_IMAGES // BATCH_SIZE)
VALIDATION_STEPS = max(1, NUM_VALIDATION_IMAGES // BATCH_SIZE)



## === cell 17
history = model.fit(
    get_training_dataset(),
    epochs=EPOCHS,
    callbacks=[lr_schedule],
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=get_validation_dataset(ordered=True),
    validation_steps=VALIDATION_STEPS,
    verbose=2,
)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2537213669.py in <cell line: 0>()
      1 history = model.fit(
----> 2     get_training_dataset(),
      3     epochs=EPOCHS,
      4     callbacks=[lr_schedule],
      5     steps_per_epoch=STEPS_PER_EPOCH,

/tmp/ipykernel_11/3827258525.py in get_training_dataset()
     67     global _TRAIN_DS
     68     if _TRAIN_DS is None:
---> 69         dataset = load_dataset(TRAINING_FILENAMES, labeled=True, ordered=False)
     70 
     71         # Speed: cache decoded+parsed examples before repeat/shuffle to avoid re-decoding every epoch.

/tmp/ipykernel_11/3827258525.py in load_dataset(filenames, labeled, ordered)
     40     opts.experimental_optimization.map_parallelization = True
     41     opts.experimental_optimization.parallel_batch = True
---> 42     opts.experimental_optimization.autotune_buffers = True
     43     dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
     44     dataset = dataset.with_options(opts)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 18
history2 = model2.fit(
    get_training_dataset(),
    epochs=EPOCHS,
    callbacks=[lr_schedule],
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=get_validation_dataset(ordered=True),
    validation_steps=VALIDATION_STEPS,
    verbose=2,
)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2773752949.py in <cell line: 0>()
      1 history2 = model2.fit(
----> 2     get_training_dataset(),
      3     epochs=EPOCHS,
      4     callbacks=[lr_schedule],
      5     steps_per_epoch=STEPS_PER_EPOCH,

/tmp/ipykernel_11/3827258525.py in get_training_dataset()
     67     global _TRAIN_DS
     68     if _TRAIN_DS is None:
---> 69         dataset = load_dataset(TRAINING_FILENAMES, labeled=True, ordered=False)
     70 
     71         # Speed: cache decoded+parsed examples before repeat/shuffle to avoid re-decoding every epoch.

/tmp/ipykernel_11/3827258525.py in load_dataset(filenames, labeled, ordered)
     40     opts.experimental_optimization.map_parallelization = True
     41     opts.experimental_optimization.parallel_batch = True
---> 42     opts.experimental_optimization.autotune_buffers = True
     43     dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
     44     dataset = dataset.with_options(opts)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 19
test_ds = get_test_dataset(ordered=True)
print("Computing predictions...")

if _TEST_IMAGES_DS_ORDERED is None:
    _TEST_IMAGES_DS_ORDERED = test_ds.map(
        lambda image, idnum: image, num_parallel_calls=AUTO
    ).prefetch(AUTO)
if _TEST_IDS_DS_ORDERED is None:
    _TEST_IDS_DS_ORDERED = test_ds.map(
        lambda image, idnum: idnum, num_parallel_calls=AUTO
    ).unbatch()

prob1 = model.predict(_TEST_IMAGES_DS_ORDERED, verbose=1)
prob2 = model2.predict(_TEST_IMAGES_DS_ORDERED, verbose=1)
probabilities = 0.5 * prob1 + 0.5 * prob2



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3888676841.py in <cell line: 0>()
      1 # Speed: build test image/id datasets exactly once and reuse for both predict calls (no duplicate mapping work).
----> 2 test_ds = get_test_dataset(ordered=True)
      3 print("Computing predictions...")
      4 
      5 if _TEST_IMAGES_DS_ORDERED is None:

/tmp/ipykernel_11/3827258525.py in get_test_dataset(ordered)
    103     if ordered:
    104         if _TEST_DS_ORDERED is None:
--> 105             dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=True)
    106             # Speed: avoid caching full test set in RAM; batch+prefetch is enough and preserves order.
    107             dataset = dataset.batch(BATCH_SIZE)

/tmp/ipykernel_11/3827258525.py in load_dataset(filenames, labeled, ordered)
     40     opts.experimental_optimization.map_parallelization = True
     41     opts.experimental_optimization.parallel_batch = True
---> 42     opts.experimental_optimization.autotune_buffers = True
     43     dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
     44     dataset = dataset.with_options(opts)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 20
print("Generating submission.csv file...")
test_ids = next(iter(_TEST_IDS_DS_ORDERED.batch(NUM_TEST_IMAGES))).numpy().astype("U")

pred_df = pd.DataFrame({"image_name": test_ids, "target": probabilities.reshape(-1)})
pred_df.head()



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1405470583.py in <cell line: 0>()
      1 print("Generating submission.csv file...")
----> 2 test_ids = next(iter(_TEST_IDS_DS_ORDERED.batch(NUM_TEST_IMAGES))).numpy().astype("U")
      3 
      4 pred_df = pd.DataFrame({"image_name": test_ids, "target": probabilities.reshape(-1)})
      5 pred_df.head()

AttributeError: 'NoneType' object has no attribute 'batch'

## === cell 21
sub_out = sub[["image_name"]].merge(pred_df, on="image_name", how="left")

if sub_out["target"].isna().any():
    sub_out["target"] = sub_out["target"].fillna(
        float(np.nanmean(sub_out["target"].values))
    )

sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
print(sub_out.head())



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2712052483.py in <cell line: 0>()
----> 1 sub_out = sub[["image_name"]].merge(pred_df, on="image_name", how="left")
      2 
      3 if sub_out["target"].isna().any():
      4     sub_out["target"] = sub_out["target"].fillna(
      5         float(np.nanmean(sub_out["target"].values))

NameError: name 'pred_df' is not defined

## === cell 22
assert os.path.exists("submission.csv")
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["image_name", "target"]
assert len(chk) == len(sub)
assert chk["target"].between(0, 1).all()
print("submission.csv validated.")



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/4007288417.py in <cell line: 0>()
----> 1 assert os.path.exists("submission.csv")
      2 chk = pd.read_csv("submission.csv")
      3 assert list(chk.columns) == ["image_name", "target"]
      4 assert len(chk) == len(sub)
      5 assert chk["target"].between(0, 1).all()

AssertionError: 

## === cell 23
model.save("EffNetB6-Melanoma.h5")
model2.save("EffNetB3-Melanoma.h5")
print("Saved models.")
