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

0.8774900521692278

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
import random
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K

try:
    import seaborn as sns
    from matplotlib import pyplot as plt
except Exception:
    sns = None
    plt = None

print("TensorFlow:", tf.__version__)
print("Eager:", tf.executing_eagerly())

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
    print("Determinism: enabled")
except Exception as e:
    print("Determinism: could not enable:", repr(e))

try:
    tf.config.optimizer.set_jit(True)
    print("XLA JIT: enabled")
except Exception as e:
    print("XLA JIT: could not enable:", repr(e))

try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    if gpus:
        print("GPU memory growth: enabled")
except Exception as e:
    print("GPU memory growth: could not enable:", repr(e))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
missing_path = "/kaggle/input/subeffnetb0/submissionB0.csv"
print("Checking for optional external submission file:", missing_path)
print("Exists:", os.path.exists(missing_path))




## === cell 2
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




## === cell 3
AUTO = tf.data.experimental.AUTOTUNE

INPUT_DIR = "/kaggle/input/siim-isic-melanoma-classification"
TFRECORD_DIR = os.path.join(INPUT_DIR, "tfrecords")

EPOCHS = 10
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
IMAGE_SIZE = [1024, 1024]

print("INPUT_DIR:", INPUT_DIR)
print("TFRECORD_DIR exists:", os.path.isdir(TFRECORD_DIR))
print("BATCH_SIZE:", BATCH_SIZE, "IMAGE_SIZE:", IMAGE_SIZE)




## === cell 4
HEIGHT = IMAGE_SIZE[0]
WIDTH = IMAGE_SIZE[1]
CHANNELS = 3




## === cell 5
def append_path(pre, base_dir):
    def _join(arr):
        arr = np.asarray(arr, dtype=str)
        return np.char.add(
            np.char.add(np.char.add(base_dir, os.sep), pre + os.sep), arr
        )

    return _join




## === cell 6
sub = pd.read_csv(
    "/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv"
)
print(sub.shape)
sub.head()




## === cell 7
train = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/train.csv")
print(train.shape)
train.head()




## === cell 8
if sns is not None:
    _ = sns.countplot(x=train["target"])
    if plt is not None:
        plt.show()
else:
    print("seaborn not available; skipping plot.")




## === cell 9
TRAINING_FILENAMES = tf.io.gfile.glob(os.path.join(TFRECORD_DIR, "train*.tfrec"))
TEST_FILENAMES = tf.io.gfile.glob(os.path.join(TFRECORD_DIR, "test*.tfrec"))

CLASSES = [0, 1]

print("Training TFRecords:", len(TRAINING_FILENAMES))
print("Test TFRecords:", len(TEST_FILENAMES))
assert len(TRAINING_FILENAMES) > 0, "No training TFRecords found."
assert len(TEST_FILENAMES) > 0, "No test TFRecords found."




## === cell 10
def _to_projective_transform(matrix_3x3):
    m = tf.reshape(matrix_3x3, [3, 3])
    return tf.stack(
        [m[0, 0], m[0, 1], m[0, 2], m[1, 0], m[1, 1], m[1, 2], m[2, 0], m[2, 1]], axis=0
    )


@tf.function
def transform_rotation(image):
    rotation = 15.0 * tf.random.normal([1], dtype=tf.float32)
    rotation = math.pi * rotation / 180.0
    c = tf.math.cos(rotation)[0]
    s = tf.math.sin(rotation)[0]

    cx = (tf.cast(WIDTH, tf.float32) - 1.0) / 2.0
    cy = (tf.cast(HEIGHT, tf.float32) - 1.0) / 2.0

    t1 = tf.stack([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], axis=0)
    r = tf.stack([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], axis=0)
    t2 = tf.stack([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]], axis=0)
    m = tf.linalg.matmul(t2, tf.linalg.matmul(r, t1))
    tr = _to_projective_transform(m)[tf.newaxis, :]
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=image[tf.newaxis, ...],
        transforms=tr,
        output_shape=[HEIGHT, WIDTH],
        interpolation="BILINEAR",
        fill_value=0.0,
    )[0]


@tf.function
def transform_shear(image):
    shear = 5.0 * tf.random.normal([1], dtype=tf.float32)
    shear = math.pi * shear / 180.0
    sh = tf.math.tan(shear)[0]

    cx = (tf.cast(WIDTH, tf.float32) - 1.0) / 2.0
    cy = (tf.cast(HEIGHT, tf.float32) - 1.0) / 2.0

    t1 = tf.stack([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], axis=0)
    s = tf.stack([[1.0, sh, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], axis=0)
    t2 = tf.stack([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]], axis=0)
    m = tf.linalg.matmul(t2, tf.linalg.matmul(s, t1))
    tr = _to_projective_transform(m)[tf.newaxis, :]
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=image[tf.newaxis, ...],
        transforms=tr,
        output_shape=[HEIGHT, WIDTH],
        interpolation="BILINEAR",
        fill_value=0.0,
    )[0]


@tf.function
def transform_shift(image):
    height_shift = 16.0 * tf.random.normal([1], dtype=tf.float32)[0]
    width_shift = 16.0 * tf.random.normal([1], dtype=tf.float32)[0]
    m = tf.stack(
        [[1.0, 0.0, height_shift], [0.0, 1.0, width_shift], [0.0, 0.0, 1.0]], axis=0
    )
    tr = _to_projective_transform(m)[tf.newaxis, :]
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=image[tf.newaxis, ...],
        transforms=tr,
        output_shape=[HEIGHT, WIDTH],
        interpolation="BILINEAR",
        fill_value=0.0,
    )[0]


@tf.function
def transform_zoom(image):
    height_zoom = 1.0 + tf.random.normal([1], dtype=tf.float32)[0] / 10.0
    width_zoom = 1.0 + tf.random.normal([1], dtype=tf.float32)[0] / 10.0

    cx = (tf.cast(WIDTH, tf.float32) - 1.0) / 2.0
    cy = (tf.cast(HEIGHT, tf.float32) - 1.0) / 2.0

    t1 = tf.stack([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], axis=0)
    z = tf.stack(
        [[1.0 / height_zoom, 0.0, 0.0], [0.0, 1.0 / width_zoom, 0.0], [0.0, 0.0, 1.0]],
        axis=0,
    )
    t2 = tf.stack([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]], axis=0)
    m = tf.linalg.matmul(t2, tf.linalg.matmul(z, t1))
    tr = _to_projective_transform(m)[tf.newaxis, :]
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=image[tf.newaxis, ...],
        transforms=tr,
        output_shape=[HEIGHT, WIDTH],
        interpolation="BILINEAR",
        fill_value=0.0,
    )[0]




## === cell 11
@tf.function
def data_augment_chaotic(image, label):
    p_spatial = tf.random.uniform([1], minval=0, maxval=1, dtype=tf.float32)[0]
    p_spatial2 = tf.random.uniform([1], minval=0, maxval=1, dtype=tf.float32)[0]
    p_pixel = tf.random.uniform([1], minval=0, maxval=1, dtype=tf.float32)[0]
    p_crop = tf.random.uniform([1], minval=0, maxval=1, dtype=tf.float32)[0]

    def _flip(im):
        im = tf.image.random_flip_left_right(im)
        im = tf.image.random_flip_up_down(im)
        return im

    image = tf.cond(p_spatial >= 0.2, lambda: _flip(image), lambda: image)

    def _crop_resize(im):
        def crop(sz):
            return tf.image.resize(
                tf.image.random_crop(im, size=sz), size=[HEIGHT, WIDTH]
            )

        im2 = tf.cond(
            p_crop >= 0.95,
            lambda: crop([int(HEIGHT * 0.6), int(WIDTH * 0.6), CHANNELS]),
            lambda: tf.cond(
                p_crop >= 0.85,
                lambda: crop([int(HEIGHT * 0.7), int(WIDTH * 0.7), CHANNELS]),
                lambda: tf.cond(
                    p_crop >= 0.8,
                    lambda: crop([int(HEIGHT * 0.8), int(WIDTH * 0.8), CHANNELS]),
                    lambda: crop([int(HEIGHT * 0.9), int(WIDTH * 0.9), CHANNELS]),
                ),
            ),
        )
        return im2

    image = tf.cond(p_crop >= 0.7, lambda: _crop_resize(image), lambda: image)

    def _spatial(im):
        return tf.cond(
            p_spatial2 >= 0.9,
            lambda: transform_rotation(im),
            lambda: tf.cond(
                p_spatial2 >= 0.8,
                lambda: transform_zoom(im),
                lambda: tf.cond(
                    p_spatial2 >= 0.7,
                    lambda: transform_shift(im),
                    lambda: transform_shear(im),
                ),
            ),
        )

    image = tf.cond(p_spatial2 >= 0.6, lambda: _spatial(image), lambda: image)

    def _pixel(im):
        return tf.cond(
            p_pixel >= 0.85,
            lambda: tf.image.random_saturation(im, lower=0.0, upper=2.0),
            lambda: tf.cond(
                p_pixel >= 0.65,
                lambda: tf.image.random_contrast(im, lower=0.8, upper=2.0),
                lambda: tf.cond(
                    p_pixel >= 0.5,
                    lambda: tf.image.random_brightness(im, max_delta=0.2),
                    lambda: tf.image.adjust_gamma(im, gamma=0.6),
                ),
            ),
        )

    image = tf.cond(p_pixel >= 0.4, lambda: _pixel(image), lambda: image)

    return image, label




## === cell 12
@tf.function
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear")
    image = tf.ensure_shape(image, [*IMAGE_SIZE, 3])
    return image


@tf.function
def decode_image_file(file, label=None):
    bits = tf.io.read_file(file)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear")
    image = tf.ensure_shape(image, [*IMAGE_SIZE, 3])
    if label is None:
        return image
    return image, label


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
    options = tf.data.Options()
    options.experimental_deterministic = bool(ordered)
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.autotune_buffers = True
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(options)
    dataset = dataset.map(
        read_labeled_tfrecord if labeled else read_unlabeled_tfrecord,
        num_parallel_calls=AUTO,
        deterministic=ordered,
    )
    return dataset


@tf.function
def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image, label


def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES, labeled=True, ordered=False)
    dataset = dataset.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO, deterministic=False)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)
    return dataset


def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO)
    return dataset


_COUNT_RE = re.compile(r"-([0-9]*)\.")


def count_data_items(filenames):
    n = [int(_COUNT_RE.search(filename).group(1)) for filename in filenames]
    return np.sum(n)


NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
print(
    f"Dataset: {NUM_TRAINING_IMAGES} training images, {NUM_TEST_IMAGES} unlabeled test images"
)
print("STEPS_PER_EPOCH:", STEPS_PER_EPOCH)




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
pass




## === cell 15
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

model2.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model2.summary()




## === cell 16
lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE




## === cell 17
history = model2.fit(
    get_training_dataset(),
    epochs=EPOCHS,
    callbacks=[lr_schedule],
    steps_per_epoch=STEPS_PER_EPOCH,
)




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1354321538.py in <cell line: 0>()
      1 history = model2.fit(
----> 2     get_training_dataset(),
      3     epochs=EPOCHS,
      4     callbacks=[lr_schedule],
      5     steps_per_epoch=STEPS_PER_EPOCH,

/tmp/ipykernel_11/2448648346.py in get_training_dataset()
     71 
     72 def get_training_dataset():
---> 73     dataset = load_dataset(TRAINING_FILENAMES, labeled=True, ordered=False)
     74     dataset = dataset.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
     75     dataset = dataset.map(data_augment, num_parallel_calls=AUTO, deterministic=False)

/tmp/ipykernel_11/2448648346.py in load_dataset(filenames, labeled, ordered)
     52     options.experimental_optimization.parallel_batch = True
     53     options.experimental_optimization.map_and_batch_fusion = True
---> 54     options.experimental_optimization.autotune_buffers = True
     55     dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
     56     dataset = dataset.with_options(options)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 18
pass




## === cell 19
pass




## === cell 20
test_ds = get_test_dataset(ordered=True)

print("Computing predictions + collecting ids in one pass...")
all_ids = []
all_preds = []

for batch_images, batch_ids in test_ds:
    batch_probs = model2(batch_images, training=False)
    all_preds.append(batch_probs.numpy())
    all_ids.append(batch_ids.numpy())

probabilities2 = np.concatenate(all_preds, axis=0)
test_ids = np.concatenate(all_ids, axis=0).astype("U")

print("test_ids:", test_ids.shape)
print("probabilities2:", probabilities2.shape)




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3502731704.py in <cell line: 0>()
      1 # Explanation: avoid two full passes over the test dataset (one for predict, one to collect ids).
      2 # We iterate once, collecting ids and predictions batch-by-batch; identical semantics.
----> 3 test_ds = get_test_dataset(ordered=True)
      4 
      5 print("Computing predictions + collecting ids in one pass...")

/tmp/ipykernel_11/2448648346.py in get_test_dataset(ordered)
     81 
     82 def get_test_dataset(ordered=False):
---> 83     dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
     84     dataset = dataset.batch(BATCH_SIZE)
     85     # Safe: test set is iterated; caching avoids repeated decode/resize if reused.

/tmp/ipykernel_11/2448648346.py in load_dataset(filenames, labeled, ordered)
     52     options.experimental_optimization.parallel_batch = True
     53     options.experimental_optimization.map_and_batch_fusion = True
---> 54     options.experimental_optimization.autotune_buffers = True
     55     dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
     56     dataset = dataset.with_options(options)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 21
sub2 = sub.copy()




## === cell 22
print("Generating submission.csv file...")




## === cell 23
preds = probabilities2.reshape(-1)
assert len(preds) == len(test_ids), "Prediction count does not match test id count."

pred_df2 = pd.DataFrame({"image_name": test_ids, "target": preds})
pred_df2.head()




## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/783070655.py in <cell line: 0>()
----> 1 preds = probabilities2.reshape(-1)
      2 assert len(preds) == len(test_ids), "Prediction count does not match test id count."
      3 
      4 pred_df2 = pd.DataFrame({"image_name": test_ids, "target": preds})
      5 pred_df2.head()

NameError: name 'probabilities2' is not defined

## === cell 24
sub_out = sub2[["image_name"]].merge(pred_df2, on="image_name", how="left")
assert sub_out["target"].isna().sum() == 0, "Some test images are missing predictions."

sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
sub_out.head()




## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4095567253.py in <cell line: 0>()
----> 1 sub_out = sub2[["image_name"]].merge(pred_df2, on="image_name", how="left")
      2 assert sub_out["target"].isna().sum() == 0, "Some test images are missing predictions."
      3 
      4 sub_out.to_csv("submission.csv", index=False)
      5 print("Wrote submission.csv with shape:", sub_out.shape)

NameError: name 'pred_df2' is not defined

## === cell 25
print("Done. submission.csv is ready in the current working directory.")




## === cell 26
sub_out.head()




## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2488442099.py in <cell line: 0>()
----> 1 sub_out.head()
      2 
      3 

NameError: name 'sub_out' is not defined

## === cell 27
missing_path_b7 = "/kaggle/input/submissionb7/submissionB7.csv"
print("Checking for optional external submission file:", missing_path_b7)
print("Exists:", os.path.exists(missing_path_b7))




## === cell 28
pass




## === cell 29
sub_out.to_csv("submission_EffnetB0.csv", index=False)
print("Also wrote submission_EffnetB0.csv")




## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2983827293.py in <cell line: 0>()
----> 1 sub_out.to_csv("submission_EffnetB0.csv", index=False)
      2 print("Also wrote submission_EffnetB0.csv")
      3 
      4 

NameError: name 'sub_out' is not defined

## === cell 30
pass




## === cell 31
pass




## === cell 32
print("Final submission file:", os.path.abspath("submission.csv"))
print(pd.read_csv("submission.csv").head())

## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2848877140.py in <cell line: 0>()
      1 print("Final submission file:", os.path.abspath("submission.csv"))
----> 2 print(pd.read_csv("submission.csv").head())

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'submission.csv'
