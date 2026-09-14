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
    image = tf.image.decode_jpeg(bits, channels=3)
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



## === cell 19
sub2 = sub.copy()



## === cell 20
print("Generating submission.csv file...")
test_ids_from_ds = test_ids.astype("U")

print("Pred shape:", probabilities.shape, "IDs shape:", test_ids_from_ds.shape)



## === cell 21
pred_df = pd.DataFrame(
    {"image_name": test_ids_from_ds, "target": probabilities.reshape(-1)}
)
pred_df.head()



## === cell 22
pred_df2 = pred_df.copy()
pred_df2.head()



## === cell 23
missing_in_pred = set(sub2["image_name"]) - set(pred_df2["image_name"])
extra_in_pred = set(pred_df2["image_name"]) - set(sub2["image_name"])
print("Missing in predictions:", len(missing_in_pred))
print("Extra in predictions:", len(extra_in_pred))



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
