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

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

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
AUTO = tf.data.AUTOTUNE

_LABELED_TFREC_FORMAT = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_UNLABELED_TFREC_FORMAT = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}

_TFREC_READ_BUFFER_BYTES = 64 * 1024 * 1024


@tf.function
def decode_image(image_data):
    image = tf.io.decode_jpeg(image_data, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear", antialias=False)
    image = tf.ensure_shape(image, [*IMAGE_SIZE, 3])
    return image


@tf.function
def read_labeled_tfrecord(example):
    example = tf.io.parse_single_example(example, _LABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    label = tf.cast(example["target"], tf.int32)
    return image, label


@tf.function
def read_unlabeled_tfrecord(example):
    example = tf.io.parse_single_example(example, _UNLABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    idnum = example["image_name"]
    return image, idnum


def load_dataset(filenames, labeled=True, ordered=False):
    opts = tf.data.Options()
    opts.experimental_deterministic = bool(ordered)

    try:
        opts.experimental_optimization.apply_default_optimizations = True
    except Exception:
        pass
    try:
        opts.experimental_optimization.map_parallelization = True
    except Exception:
        pass
    try:
        opts.experimental_optimization.parallel_batch = True
    except Exception:
        pass
    if hasattr(opts.experimental_optimization, "autotune_buffers"):
        try:
            opts.experimental_optimization.autotune_buffers = True
        except Exception:
            pass
    if hasattr(opts, "experimental_slack"):
        try:
            opts.experimental_slack = True
        except Exception:
            pass

    dataset = tf.data.TFRecordDataset(
        filenames,
        num_parallel_reads=AUTO if not ordered else 1,
        buffer_size=_TFREC_READ_BUFFER_BYTES,
    )
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


def get_training_dataset():
    global _TRAIN_DS
    if _TRAIN_DS is None:
        dataset = load_dataset(TRAINING_FILENAMES, labeled=True, ordered=False)
        dataset = dataset.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        dataset = dataset.repeat()
        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
        dataset = dataset.prefetch(AUTO)
        _TRAIN_DS = dataset
    return _TRAIN_DS


def get_validation_dataset(ordered=False):
    global _VAL_DS_ORDERED
    if ordered:
        if _VAL_DS_ORDERED is None:
            dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=True)
            dataset = dataset.cache()
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
            dataset = dataset.cache()
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

MAX_STEPS_PER_EPOCH = 200
MAX_VALIDATION_STEPS = 50

STEPS_PER_EPOCH = max(1, min(NUM_TRAINING_IMAGES // BATCH_SIZE, MAX_STEPS_PER_EPOCH))
VALIDATION_STEPS = max(
    1, min(NUM_VALIDATION_IMAGES // BATCH_SIZE, MAX_VALIDATION_STEPS)
)

print(
    "CAPPED STEPS_PER_EPOCH:",
    STEPS_PER_EPOCH,
    "CAPPED VALIDATION_STEPS:",
    VALIDATION_STEPS,
)

train_ds = get_training_dataset()
val_ds_ordered = get_validation_dataset(ordered=True)

B6_WEIGHTS = "EffNetB6-Melanoma.weights.h5"
B3_WEIGHTS = "EffNetB3-Melanoma.weights.h5"



## === cell 17
if os.path.exists(B6_WEIGHTS):
    print("Found existing weights:", B6_WEIGHTS, "-> loading and skipping model.fit()")
    model.load_weights(B6_WEIGHTS)
    history = None
else:
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        callbacks=[lr_schedule],
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=val_ds_ordered,
        validation_steps=VALIDATION_STEPS,
        verbose=2,
    )
    model.save_weights(B6_WEIGHTS)
    print("Saved weights:", B6_WEIGHTS)



## === cell 18
if os.path.exists(B3_WEIGHTS):
    print("Found existing weights:", B3_WEIGHTS, "-> loading and skipping model2.fit()")
    model2.load_weights(B3_WEIGHTS)
    history2 = None
else:
    history2 = model2.fit(
        train_ds,
        epochs=EPOCHS,
        callbacks=[lr_schedule],
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=val_ds_ordered,
        validation_steps=VALIDATION_STEPS,
        verbose=2,
    )
    model2.save_weights(B3_WEIGHTS)
    print("Saved weights:", B3_WEIGHTS)



## === cell 19
test_ds = get_test_dataset(ordered=True)
print("Computing predictions...")

all_test_images = []
all_test_ids = []
for batch_images, batch_ids in test_ds:
    all_test_images.append(batch_images)
    all_test_ids.append(batch_ids.numpy().astype("U"))

test_images = tf.concat(all_test_images, axis=0)
test_ids = np.concatenate(all_test_ids, axis=0)

prob1 = model.predict(test_images, verbose=0, batch_size=BATCH_SIZE)
prob2 = model2.predict(test_images, verbose=0, batch_size=BATCH_SIZE)
probabilities = 0.5 * prob1 + 0.5 * prob2

print("Predictions shape:", probabilities.shape, "IDs shape:", test_ids.shape)



## === cell 20
print("Generating submission.csv file...")

pred_df = pd.DataFrame({"image_name": test_ids, "target": probabilities.reshape(-1)})
print(pred_df.head())
print("Pred df shape:", pred_df.shape)



## === cell 21
sub_out = pred_df

if sub_out["target"].isna().any():
    sub_out["target"] = sub_out["target"].fillna(
        float(np.nanmean(sub_out["target"].values))
    )

sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
print(sub_out.head())



## === cell 22
assert os.path.exists("submission.csv")
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["image_name", "target"]
assert len(chk) == len(sub)
assert chk["target"].between(0, 1).all()
print("submission.csv validated.")



## === cell 23
model.save("EffNetB6-Melanoma.h5")
model2.save("EffNetB3-Melanoma.h5")
print("Saved models.")
