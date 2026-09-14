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

0.8887837980508441

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

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

try:
    tf.config.optimizer.set_jit(False)
except Exception as e:
    print("Warning: could not set jit flag:", repr(e))

print("TF version:", tf.__version__)
print("Listing /kaggle/input:", os.listdir("/kaggle/input")[:10])




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR = "/kaggle/input/siim-isic-melanoma-classification"
TFREC_DIR = os.path.join(DATA_DIR, "tfrecords")

print("TFRecord dir exists:", os.path.isdir(TFREC_DIR))
print("TFRecord files sample:", sorted(os.listdir(TFREC_DIR))[:5])




## === cell 2
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU", tpu.master())
except Exception:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS:", strategy.num_replicas_in_sync)




## === cell 3
AUTO = tf.data.experimental.AUTOTUNE

EPOCHS = 10
BATCH_SIZE = 8 * strategy.num_replicas_in_sync

IMAGE_SIZE = [256, 256]

HEIGHT = IMAGE_SIZE[0]
WIDTH = IMAGE_SIZE[1]
CHANNELS = 3

TRAINING_FILENAMES = tf.io.gfile.glob(os.path.join(TFREC_DIR, "train*.tfrec"))
TEST_FILENAMES = tf.io.gfile.glob(os.path.join(TFREC_DIR, "test*.tfrec"))

print("Num training tfrecs:", len(TRAINING_FILENAMES))
print("Num test tfrecs:", len(TEST_FILENAMES))




## === cell 4
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

print("train shape:", train.shape, "test shape:", test.shape, "sub shape:", sub.shape)
print("sub columns:", sub.columns.tolist())




## === cell 5
def transform_rotation(image):
    DIM = IMAGE_SIZE[0]
    XDIM = DIM % 2

    rotation = 15.0 * tf.random.normal([1], dtype="float32")
    rotation = math.pi * rotation / 180.0

    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1], dtype="float32")
    zero = tf.constant([0], dtype="float32")
    rotation_matrix = tf.reshape(
        tf.concat([c1, s1, zero, -s1, c1, zero, zero, zero, one], axis=0), [3, 3]
    )

    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
    z = tf.ones([DIM * DIM], dtype="int32")
    idx = tf.stack([x, y, z])

    idx2 = K.dot(rotation_matrix, tf.cast(idx, dtype="float32"))
    idx2 = K.cast(idx2, dtype="int32")
    idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)

    idx3 = tf.stack([DIM // 2 - idx2[0,], DIM // 2 - 1 + idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d, [DIM, DIM, 3])


def transform_shear(image):
    DIM = IMAGE_SIZE[0]
    XDIM = DIM % 2

    shear = 5.0 * tf.random.normal([1], dtype="float32")
    shear = math.pi * shear / 180.0

    one = tf.constant([1], dtype="float32")
    zero = tf.constant([0], dtype="float32")
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_matrix = tf.reshape(
        tf.concat([one, s2, zero, zero, c2, zero, zero, zero, one], axis=0), [3, 3]
    )

    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
    z = tf.ones([DIM * DIM], dtype="int32")
    idx = tf.stack([x, y, z])

    idx2 = K.dot(shear_matrix, tf.cast(idx, dtype="float32"))
    idx2 = K.cast(idx2, dtype="int32")
    idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)

    idx3 = tf.stack([DIM // 2 - idx2[0,], DIM // 2 - 1 + idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d, [DIM, DIM, 3])


def transform_shift(image):
    DIM = IMAGE_SIZE[0]
    XDIM = DIM % 2

    height_shift = 16.0 * tf.random.normal([1], dtype="float32")
    width_shift = 16.0 * tf.random.normal([1], dtype="float32")
    one = tf.constant([1], dtype="float32")
    zero = tf.constant([0], dtype="float32")

    shift_matrix = tf.reshape(
        tf.concat(
            [one, zero, height_shift, zero, one, width_shift, zero, zero, one], axis=0
        ),
        [3, 3],
    )

    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
    z = tf.ones([DIM * DIM], dtype="int32")
    idx = tf.stack([x, y, z])

    idx2 = K.dot(shift_matrix, tf.cast(idx, dtype="float32"))
    idx2 = K.cast(idx2, dtype="int32")
    idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)

    idx3 = tf.stack([DIM // 2 - idx2[0,], DIM // 2 - 1 + idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d, [DIM, DIM, 3])


def transform_zoom(image):
    DIM = IMAGE_SIZE[0]
    XDIM = DIM % 2

    height_zoom = 1.0 + tf.random.normal([1], dtype="float32") / 10.0
    width_zoom = 1.0 + tf.random.normal([1], dtype="float32") / 10.0
    one = tf.constant([1], dtype="float32")
    zero = tf.constant([0], dtype="float32")

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

    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
    z = tf.ones([DIM * DIM], dtype="int32")
    idx = tf.stack([x, y, z])

    idx2 = K.dot(zoom_matrix, tf.cast(idx, dtype="float32"))
    idx2 = K.cast(idx2, dtype="int32")
    idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)

    idx3 = tf.stack([DIM // 2 - idx2[0,], DIM // 2 - 1 + idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d, [DIM, DIM, 3])




## === cell 6
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




## === cell 7
@tf.function
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(
        image, IMAGE_SIZE, method="bilinear"
    )  # faster than antialias=True
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


_BASE_DATA_OPTS = tf.data.Options()
_BASE_DATA_OPTS.experimental_deterministic = False
_BASE_DATA_OPTS.experimental_optimization.map_parallelization = True
_BASE_DATA_OPTS.experimental_optimization.parallel_batch = True
_BASE_DATA_OPTS.experimental_optimization.map_and_batch_fusion = True
_BASE_DATA_OPTS.experimental_optimization.autotune_buffers = True
_BASE_DATA_OPTS.threading.private_threadpool_size = 0  # let TF decide
_BASE_DATA_OPTS.threading.max_intra_op_parallelism = 0


def load_dataset(filenames, labeled=True, ordered=False):
    opts = tf.data.Options()
    opts.experimental_optimization = _BASE_DATA_OPTS.experimental_optimization
    opts.threading = _BASE_DATA_OPTS.threading
    opts.experimental_deterministic = ordered  # deterministic only when ordered=True

    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    ds = ds.with_options(opts)
    ds = ds.map(
        read_labeled_tfrecord if labeled else read_unlabeled_tfrecord,
        num_parallel_calls=AUTO,
        deterministic=ordered,
    )
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds


@tf.function
def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image, label


def get_training_dataset():
    ds = load_dataset(TRAINING_FILENAMES, labeled=True, ordered=False)

    ds = ds.cache()

    ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.repeat()

    ds = ds.map(data_augment, num_parallel_calls=AUTO, deterministic=False)
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    ds = ds.prefetch(AUTO)
    return ds


def get_test_dataset(ordered=False):
    ds = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(fn).group(1)) for fn in filenames]
    return int(np.sum(n))


NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print(f"Dataset: {NUM_TRAINING_IMAGES} training images, {NUM_TEST_IMAGES} test images")
print("BATCH_SIZE:", BATCH_SIZE, "STEPS_PER_EPOCH:", STEPS_PER_EPOCH)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1563596185.py in <cell line: 0>()
     42 _BASE_DATA_OPTS.experimental_optimization.parallel_batch = True
     43 _BASE_DATA_OPTS.experimental_optimization.map_and_batch_fusion = True
---> 44 _BASE_DATA_OPTS.experimental_optimization.autotune_buffers = True
     45 _BASE_DATA_OPTS.threading.private_threadpool_size = 0  # let TF decide
     46 _BASE_DATA_OPTS.threading.max_intra_op_parallelism = 0

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 8
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
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=0)




## === cell 9
EfficientNetB7 = tf.keras.applications.EfficientNetB7
EfficientNetB0 = tf.keras.applications.EfficientNetB0

with strategy.scope():
    model = tf.keras.Sequential(
        [
            EfficientNetB7(
                input_shape=(*IMAGE_SIZE, 3), weights="imagenet", include_top=False
            ),
            L.GlobalAveragePooling2D(),
            L.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

with strategy.scope():
    model2 = tf.keras.Sequential(
        [
            EfficientNetB0(
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
    model2.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

print("Models built.")




## === cell 10
train_ds = get_training_dataset()

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=[lr_schedule],
    steps_per_epoch=STEPS_PER_EPOCH,
    verbose=2,
)

history2 = model2.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=[lr_schedule],
    steps_per_epoch=STEPS_PER_EPOCH,
    verbose=2,
)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/178627481.py in <cell line: 0>()
----> 1 train_ds = get_training_dataset()
      2 
      3 history = model.fit(
      4     train_ds,
      5     epochs=EPOCHS,

NameError: name 'get_training_dataset' is not defined

## === cell 11
test_ds = get_test_dataset(ordered=True)


@tf.function
def _get_images_and_ids(image, image_id):
    return image, image_id


@tf.function
def _drop_ids(image, image_id):
    return image


print("Computing predictions and collecting ids...")

test_ids = np.concatenate([ids.numpy().astype("U") for _, ids in test_ds], axis=0)

test_images_ds = test_ds.map(
    _drop_ids, num_parallel_calls=AUTO, deterministic=True
).prefetch(AUTO)

probabilities_b7 = model.predict(test_images_ds, verbose=1).reshape(-1)
probabilities_b0 = model2.predict(test_images_ds, verbose=1).reshape(-1)

print("Pred shapes:", probabilities_b7.shape, probabilities_b0.shape)
print("IDs collected:", test_ids.shape)

n = min(len(test_ids), len(probabilities_b7), len(probabilities_b0))
test_ids = test_ids[:n]
probabilities_b7 = probabilities_b7[:n]
probabilities_b0 = probabilities_b0[:n]

pred = 0.5 * probabilities_b7 + 0.5 * probabilities_b0

pred_df = pd.DataFrame({"image_name": test_ids, "target": pred.astype(np.float32)})

submission = sub[["image_name"]].merge(pred_df, on="image_name", how="left")
if submission["target"].isna().any():
    submission["target"] = submission["target"].fillna(float(np.mean(pred)))

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv:", submission.shape)
print(submission.head())

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2090668444.py in <cell line: 0>()
      2 # We keep the same evaluation semantics by building one dataset that yields (image, id),
      3 # then deriving an images-only view for model.predict without re-decoding.
----> 4 test_ds = get_test_dataset(ordered=True)
      5 
      6 

NameError: name 'get_test_dataset' is not defined
