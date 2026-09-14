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

0.738010361034356

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("PYTHONHASHSEED", "0")

import re
import math
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K

np.random.seed(0)
tf.random.set_seed(0)

print("TF version:", tf.__version__)
print("Working dir:", os.getcwd())
print("Listing /kaggle/input:", os.listdir("/kaggle/input")[:10])



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path_to_check = "/kaggle/input/subeffnetb0/submissionB0.csv"
print("Exists:", os.path.exists(path_to_check), "|", path_to_check)



## === cell 2
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
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Enabled GPU memory growth for", len(gpus), "GPU(s)")
    except Exception as e:
        print("Could not set GPU memory growth:", repr(e))

    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)



## === cell 3
AUTO = tf.data.experimental.AUTOTUNE

BASE_PATH = "/kaggle/input/siim-isic-melanoma-classification"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_CSV = os.path.join(BASE_PATH, "test.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

TRAIN_JPEG_DIR = os.path.join(BASE_PATH, "jpeg", "train")
TEST_JPEG_DIR = os.path.join(BASE_PATH, "jpeg", "test")

EPOCHS = 10
BATCH_SIZE = 8 * strategy.num_replicas_in_sync

IMAGE_SIZE = [1024, 1024]

print("BASE_PATH:", BASE_PATH)
print("Train JPEG dir exists:", os.path.isdir(TRAIN_JPEG_DIR))
print("Test  JPEG dir exists:", os.path.isdir(TEST_JPEG_DIR))
print("BATCH_SIZE:", BATCH_SIZE, "EPOCHS:", EPOCHS, "IMAGE_SIZE:", IMAGE_SIZE)



## === cell 4
HEIGHT = IMAGE_SIZE[0]
WIDTH = IMAGE_SIZE[1]
CHANNELS = 3




## === cell 5
def append_path(pre):
    base = os.path.join(BASE_PATH, pre)

    def _fn(files):
        files = np.asarray(files, dtype=str)
        return np.char.add(np.char.add(base + os.sep, files), "")

    return _fn




## === cell 6
sub = pd.read_csv(SAMPLE_SUB)
sub.head()



## === cell 7
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)

print(
    "train shape:",
    train.shape,
    "| test shape:",
    test.shape,
    "| sample_submission shape:",
    sub.shape,
)
print("train columns:", list(train.columns))
print("test columns:", list(test.columns))



## === cell 8
print("Target value counts:\n", train["target"].value_counts(dropna=False))



## === cell 9
train_paths = (TRAIN_JPEG_DIR + "/" + train["image_name"].astype(str) + ".jpg").values
train_labels = train["target"].astype(np.int32).values

test_paths = (TEST_JPEG_DIR + "/" + test["image_name"].astype(str) + ".jpg").values
test_ids = test["image_name"].values.astype(str)

missing_train = int(
    np.sum(~np.fromiter((os.path.exists(p) for p in train_paths[:200]), dtype=bool))
)
missing_test = int(
    np.sum(~np.fromiter((os.path.exists(p) for p in test_paths[:200]), dtype=bool))
)
print("Missing in first 200 train jpg:", missing_train)
print("Missing in first 200 test jpg :", missing_test)

CLASSES = [0, 1]




## === cell 10
@tf.function
def transform_rotation(image):
    DIM = IMAGE_SIZE[0]
    XDIM = DIM % 2  # fix for size 331 (kept from original)

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


@tf.function
def transform_shear(image):
    DIM = IMAGE_SIZE[0]
    XDIM = DIM % 2  # fix for size 331

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


@tf.function
def transform_shift(image):
    DIM = IMAGE_SIZE[0]
    XDIM = DIM % 2  # fix for size 331

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


@tf.function
def transform_zoom(image):
    DIM = IMAGE_SIZE[0]
    XDIM = DIM % 2  # fix for size 331

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




## === cell 11
@tf.function
def data_augment_chaotic(image, label):
    p_spatial = tf.random.uniform([1], minval=0, maxval=1, dtype="float32")
    p_spatial2 = tf.random.uniform([1], minval=0, maxval=1, dtype="float32")
    p_pixel = tf.random.uniform([1], minval=0, maxval=1, dtype="float32")
    p_crop = tf.random.uniform([1], minval=0, maxval=1, dtype="float32")

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
def decode_image_file(file_path, label=None):
    bits = tf.io.read_file(file_path)

    img_shape = tf.io.extract_jpeg_shape(bits)
    h = img_shape[0]
    w = img_shape[1]
    crop_window = tf.stack([0, 0, h, w])

    dct_scale = tf.constant([1, 2], dtype=tf.int32)

    image = tf.image.decode_and_crop_jpeg(
        bits,
        crop_window=crop_window,
        channels=3,
        dct_method="INTEGER_FAST",
        dct_scale=dct_scale,
    )
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    if label is None:
        return image
    return image, label


@tf.function
def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image, label


def get_training_dataset():
    ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    ds = ds.with_options(options)
    ds = ds.shuffle(2048, seed=0, reshuffle_each_iteration=True)
    ds = ds.map(decode_image_file, num_parallel_calls=AUTO)
    ds = ds.map(data_augment, num_parallel_calls=AUTO)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    ds = ds.prefetch(AUTO)
    return ds


def get_test_dataset(ordered=True):
    ds = tf.data.Dataset.from_tensor_slices(test_paths)

    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_optimization.apply_default_optimizations = True
    try:
        options.experimental_optimization.map_and_batch_fusion = True
        options.experimental_optimization.parallel_batch = True
    except Exception:
        pass
    ds = ds.with_options(options)

    ds = ds.map(decode_image_file, num_parallel_calls=AUTO)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)

    if tf.config.list_logical_devices("GPU"):
        try:
            ds = ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
        except Exception:
            ds = ds.prefetch(AUTO)
    else:
        ds = ds.prefetch(AUTO)

    return ds


NUM_TRAINING_IMAGES = len(train_paths)
NUM_TEST_IMAGES = len(test_paths)
STEPS_PER_EPOCH = max(1, NUM_TRAINING_IMAGES // BATCH_SIZE)

print(f"Dataset: {NUM_TRAINING_IMAGES} training images, {NUM_TEST_IMAGES} test images")
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
path_sub_b7 = "/kaggle/input/submissionb7/submissionB7.csv"
path_sub_b0 = "/kaggle/input/subeffnetb0/submissionB0.csv"

B7_WEIGHTS_PATH = "effnetb7_trained.weights.h5"
B0_WEIGHTS_PATH = "effnetb0_trained.weights.h5"

have_local_b7 = os.path.exists(B7_WEIGHTS_PATH)
have_local_b0 = os.path.exists(B0_WEIGHTS_PATH)
have_external_b7 = os.path.exists(path_sub_b7)
have_external_b0 = os.path.exists(path_sub_b0)

print("Local weights available: B7:", have_local_b7, "| B0:", have_local_b0)
print(
    "External submissions available: B7:", have_external_b7, "| B0:", have_external_b0
)



## === cell 15
try:
    tf.config.optimizer.set_jit(True)
    print("XLA JIT: enabled")
except Exception as e:
    print("XLA JIT: could not enable:", repr(e))

model = None
model2 = None
b7_status = "not_built"
b0_status = "not_built"

if have_local_b7:
    with strategy.scope():
        model = tf.keras.Sequential(
            [
                tf.keras.applications.EfficientNetB7(
                    input_shape=(*IMAGE_SIZE, 3), weights="imagenet", include_top=False
                ),
                L.GlobalAveragePooling2D(),
                L.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
    model.load_weights(B7_WEIGHTS_PATH)
    b7_status = "loaded"
else:
    b7_status = "skipped"

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

if have_local_b0:
    model2.load_weights(B0_WEIGHTS_PATH)
    b0_status = "loaded"
else:
    b0_status = "imagenet_backbone_untrained_head"

print("B7 status:", b7_status, "| B0 status:", b0_status)



## === cell 16
lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=0)
STEPS_PER_EPOCH = max(1, NUM_TRAINING_IMAGES // BATCH_SIZE)



## === cell 17
pass



## === cell 18
print("Computing predictions...")

INFER_BATCH_SIZE = BATCH_SIZE
if tpu:
    INFER_BATCH_SIZE = max(BATCH_SIZE, 64)
elif tf.config.list_logical_devices("GPU"):
    INFER_BATCH_SIZE = max(BATCH_SIZE, 64)
else:
    INFER_BATCH_SIZE = max(BATCH_SIZE, 16)

_orig_batch = BATCH_SIZE
BATCH_SIZE = INFER_BATCH_SIZE
print("Inference BATCH_SIZE:", BATCH_SIZE, "(train batch was:", _orig_batch, ")")

test_ds = get_test_dataset(ordered=True)

num_test_steps = int(math.ceil(NUM_TEST_IMAGES / BATCH_SIZE))
print("Predict steps:", num_test_steps, "| NUM_TEST_IMAGES:", NUM_TEST_IMAGES)

probabilities2 = model2.predict(test_ds, steps=num_test_steps, verbose=1)

if model is not None and b7_status == "loaded":
    probabilities1 = model.predict(test_ds, steps=num_test_steps, verbose=1)
    probabilities = 0.5 * probabilities1 + 0.5 * probabilities2
else:
    probabilities1 = probabilities2
    probabilities = probabilities2

probabilities = np.asarray(probabilities).reshape(-1, 1).astype(np.float32)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1589463859.py in <cell line: 0>()
     13 print("Inference BATCH_SIZE:", BATCH_SIZE, "(train batch was:", _orig_batch, ")")
     14 
---> 15 test_ds = get_test_dataset(ordered=True)
     16 
     17 num_test_steps = int(math.ceil(NUM_TEST_IMAGES / BATCH_SIZE))

/tmp/ipykernel_11/1808543266.py in get_test_dataset(ordered)
     68     ds = ds.with_options(options)
     69 
---> 70     ds = ds.map(decode_image_file, num_parallel_calls=AUTO)
     71 
     72     # CHANGE (timeout fix): do not cache huge decoded float32 tensors; test set is only iterated once.

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in map(self, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
   2339     from tensorflow.python.data.ops import map_op
   2340 
-> 2341     return map_op._map_v2(
   2342         self,
   2343         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in _map_v2(input_dataset, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
     55           num_parallel_calls,
     56       )
---> 57     return _ParallelMapDataset(
     58         input_dataset,
     59         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in __init__(self, input_dataset, map_func, num_parallel_calls, deterministic, use_inter_op_parallelism, preserve_cardinality, use_legacy_function, use_unbounded_threadpool, name)
    200     self._input_dataset = input_dataset
    201     self._use_inter_op_parallelism = use_inter_op_parallelism
--> 202     self._map_func = structured_function.StructuredFunctionWrapper(
    203         map_func,
    204         self._transformation_name(),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in __init__(self, func, transformation_name, dataset, input_classes, input_shapes, input_types, input_structure, add_to_graph, use_legacy_function, defun_kwargs)
    263         fn_factory = trace_tf_function(defun_kwargs)
    264 
--> 265     self._function = fn_factory()
    266     # There is no graph to add in eager mode.
    267     add_to_graph &= not context.executing_eagerly()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in get_concrete_function(self, *args, **kwargs)
   1249   def get_concrete_function(self, *args, **kwargs):
   1250     # Implements PolymorphicFunction.get_concrete_function.
-> 1251     concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
   1252     concrete._garbage_collector.release()  # pylint: disable=protected-access
   1253     return concrete

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _get_concrete_function_garbage_collected(self, *args, **kwargs)
   1219       if self._variable_creation_config is None:
   1220         initializers = []
-> 1221         self._initialize(args, kwargs, add_initializers_to=initializers)
   1222         self._initialize_uninitialized_variables(initializers)
   1223 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _initialize(self, args, kwds, add_initializers_to)
    694     )
    695     # Force the definition of the function for these arguments
--> 696     self._concrete_variable_creation_fn = tracing_compilation.trace_function(
    697         args, kwds, self._variable_creation_config
    698     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in trace_function(args, kwargs, tracing_options)
    176       kwargs = {}
    177 
--> 178     concrete_function = _maybe_define_function(
    179         args, kwargs, tracing_options
    180     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _maybe_define_function(args, kwargs, tracing_options)
    281         else:
    282           target_func_type = lookup_func_type
--> 283         concrete_function = _create_concrete_function(
    284             target_func_type, lookup_func_context, func_graph, tracing_options
    285         )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _create_concrete_function(function_type, type_context, func_graph, tracing_options)
    308       attributes_lib.DISABLE_ACD, False
    309   )
--> 310   traced_func_graph = func_graph_module.func_graph_from_py_func(
    311       tracing_options.name,
    312       tracing_options.python_function,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, create_placeholders)
   1057 
   1058     _, original_func = tf_decorator.unwrap(python_func)
-> 1059     func_outputs = python_func(*func_args, **func_kwargs)
   1060 
   1061     # invariant: `func_outputs` contains only Tensors, CompositeTensors,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in wrapped_fn(*args, **kwds)
    597         # the function a weak reference to itself to avoid a reference cycle.
    598         with OptionalXlaContext(compile_with_xla):
--> 599           out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    600         return out
    601 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapped_fn(*args)
    229       # Note: wrapper_helper will apply autograph based on context.
    230       def wrapped_fn(*args):  # pylint: disable=missing-docstring
--> 231         ret = wrapper_helper(*args)
    232         ret = structure.to_tensor_list(self._output_structure, ret)
    233         return [ops.convert_to_tensor(t) for t in ret]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapper_helper(*args)
    159       if not _should_unpack(nested_args):
    160         nested_args = (nested_args,)
--> 161       ret = autograph.tf_convert(self._func, ag_ctx)(*nested_args)
    162       ret = variable_utils.convert_variables_to_tensors(ret)
    163       if _should_pack(ret):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/tmp/__autograph_generated_filec2gtj4uj.py in tf__decode_image_file(file_path, label)
     14                 crop_window = ag__.converted_call(ag__.ld(tf).stack, ([0, 0, ag__.ld(h), ag__.ld(w)],), None, fscope)
     15                 dct_scale = ag__.converted_call(ag__.ld(tf).constant, ([1, 2],), dict(dtype=ag__.ld(tf).int32), fscope)
---> 16                 image = ag__.converted_call(ag__.ld(tf).image.decode_and_crop_jpeg, (ag__.ld(bits),), dict(crop_window=ag__.ld(crop_window), channels=3, dct_method='INTEGER_FAST', dct_scale=ag__.ld(dct_scale)), fscope)
     17                 image = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(image), ag__.ld(IMAGE_SIZE)), dict(method='bilinear'), fscope)
     18                 image = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(image), ag__.ld(tf).float32), None, fscope) / 255.0

TypeError: in user code:

    File "/tmp/ipykernel_11/1808543266.py", line 21, in decode_image_file  *
        image = tf.image.decode_and_crop_jpeg(

    TypeError: Got an unexpected keyword argument 'dct_scale'


## === cell 19
sub2 = sub.copy()



## === cell 20
print("Generating submission.csv file...")
test_ids_from_ds = test_ids.astype("U")

print("Pred shape:", probabilities.shape, "IDs shape:", test_ids_from_ds.shape)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2607101679.py in <cell line: 0>()
      2 test_ids_from_ds = test_ids.astype("U")
      3 
----> 4 print("Pred shape:", probabilities.shape, "IDs shape:", test_ids_from_ds.shape)
      5 

NameError: name 'probabilities' is not defined

## === cell 21
pred_df = pd.DataFrame(
    {"image_name": test_ids_from_ds, "target": probabilities.reshape(-1)}
)
pred_df.head()



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2145006658.py in <cell line: 0>()
      1 pred_df = pd.DataFrame(
----> 2     {"image_name": test_ids_from_ds, "target": probabilities.reshape(-1)}
      3 )
      4 pred_df.head()
      5 

NameError: name 'probabilities' is not defined

## === cell 22
pred_df2 = pred_df.copy()
pred_df2.head()



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2465780928.py in <cell line: 0>()
----> 1 pred_df2 = pred_df.copy()
      2 pred_df2.head()
      3 

NameError: name 'pred_df' is not defined

## === cell 23
missing_in_pred = set(sub2["image_name"]) - set(pred_df2["image_name"])
extra_in_pred = set(pred_df2["image_name"]) - set(sub2["image_name"])
print("Missing in predictions:", len(missing_in_pred))
print("Extra in predictions:", len(extra_in_pred))



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1664028517.py in <cell line: 0>()
----> 1 missing_in_pred = set(sub2["image_name"]) - set(pred_df2["image_name"])
      2 extra_in_pred = set(pred_df2["image_name"]) - set(sub2["image_name"])
      3 print("Missing in predictions:", len(missing_in_pred))
      4 print("Extra in predictions:", len(extra_in_pred))
      5 

NameError: name 'pred_df2' is not defined

## === cell 24
sub2.head()



## === cell 25
path_sub_b7 = "/kaggle/input/submissionb7/submissionB7.csv"
print("External submission exists:", os.path.exists(path_sub_b7), "|", path_sub_b7)



## === cell 26
pass



## === cell 27
sub2 = sub2.drop(columns=["target"], errors="ignore")
sub2 = sub2.merge(pred_df2, on="image_name", how="left")

if sub2["target"].isna().any():
    sub2["target"] = sub2["target"].fillna(float(np.nanmean(sub2["target"].values)))

sub2["target"] = sub2["target"].astype(np.float32).clip(0.0, 1.0)

sub2.to_csv("submission_EffnetB0.csv", index=False)
sub2.head()



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3099601859.py in <cell line: 0>()
      1 sub2 = sub2.drop(columns=["target"], errors="ignore")
----> 2 sub2 = sub2.merge(pred_df2, on="image_name", how="left")
      3 
      4 if sub2["target"].isna().any():
      5     sub2["target"] = sub2["target"].fillna(float(np.nanmean(sub2["target"].values)))

NameError: name 'pred_df2' is not defined

## === cell 28
path_sub_b0 = "/kaggle/input/subeffnetb0/submissionB0.csv"
print("External submission exists:", os.path.exists(path_sub_b0), "|", path_sub_b0)



## === cell 29
sub3 = None
print("sub3 is not available in this environment; proceeding without it.")



## === cell 30
sub_es = sub2[["image_name", "target"]].copy()
sub_es.to_csv("submission.csv", index=False)

print("Wrote:", os.path.abspath("submission.csv"))
print(sub_es.head())
print("Submission shape:", sub_es.shape)
print("Columns:", list(sub_es.columns))

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1237547433.py in <cell line: 0>()
----> 1 sub_es = sub2[["image_name", "target"]].copy()
      2 sub_es.to_csv("submission.csv", index=False)
      3 
      4 print("Wrote:", os.path.abspath("submission.csv"))
      5 print(sub_es.head())

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['target'] not in index"
