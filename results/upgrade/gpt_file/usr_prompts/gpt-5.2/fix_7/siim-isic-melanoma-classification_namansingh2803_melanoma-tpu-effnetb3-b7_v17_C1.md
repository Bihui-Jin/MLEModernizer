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
_DIM = IMAGE_SIZE[0]
_XDIM = _DIM % 2
_x_grid = tf.repeat(tf.range(_DIM // 2, -_DIM // 2, -1), _DIM)
_y_grid = tf.tile(tf.range(-_DIM // 2, _DIM // 2), [_DIM])
_z_grid = tf.ones([_DIM * _DIM], dtype="int32")
_idx_grid = tf.stack([_x_grid, _y_grid, _z_grid])  # [3, DIM*DIM]

_one_f32 = tf.constant([1.0], dtype="float32")
_zero_f32 = tf.constant([0.0], dtype="float32")


def transform_rotation(image):
    DIM = IMAGE_SIZE[0]
    XDIM = DIM % 2

    rotation = 15.0 * tf.random.normal([1], dtype="float32")
    rotation = math.pi * rotation / 180.0

    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    rotation_matrix = tf.reshape(
        tf.concat(
            [c1, s1, _zero_f32, -s1, c1, _zero_f32, _zero_f32, _zero_f32, _one_f32],
            axis=0,
        ),
        [3, 3],
    )

    idx2 = K.dot(rotation_matrix, tf.cast(_idx_grid, dtype="float32"))
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

    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_matrix = tf.reshape(
        tf.concat(
            [
                _one_f32,
                s2,
                _zero_f32,
                _zero_f32,
                c2,
                _zero_f32,
                _zero_f32,
                _zero_f32,
                _one_f32,
            ],
            axis=0,
        ),
        [3, 3],
    )

    idx2 = K.dot(shear_matrix, tf.cast(_idx_grid, dtype="float32"))
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

    shift_matrix = tf.reshape(
        tf.concat(
            [
                _one_f32,
                _zero_f32,
                height_shift,
                _zero_f32,
                _one_f32,
                width_shift,
                _zero_f32,
                _zero_f32,
                _one_f32,
            ],
            axis=0,
        ),
        [3, 3],
    )

    idx2 = K.dot(shift_matrix, tf.cast(_idx_grid, dtype="float32"))
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

    zoom_matrix = tf.reshape(
        tf.concat(
            [
                _one_f32 / height_zoom,
                _zero_f32,
                _zero_f32,
                _zero_f32,
                _one_f32 / width_zoom,
                _zero_f32,
                _zero_f32,
                _zero_f32,
                _one_f32,
            ],
            axis=0,
        ),
        [3, 3],
    )

    idx2 = K.dot(zoom_matrix, tf.cast(_idx_grid, dtype="float32"))
    idx2 = K.cast(idx2, dtype="int32")
    idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)

    idx3 = tf.stack([DIM // 2 - idx2[0,], DIM // 2 - 1 + idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))
    return tf.reshape(d, [DIM, DIM, 3])




## === cell 6
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




## === cell 7
@tf.function
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image


LABELED_TFREC_FORMAT = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
UNLABELED_TFREC_FORMAT = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def read_labeled_tfrecord_batch(examples):
    ex = tf.io.parse_example(examples, LABELED_TFREC_FORMAT)
    images = tf.map_fn(
        decode_image,
        ex["image"],
        fn_output_signature=tf.float32,
        parallel_iterations=16,
    )
    labels = tf.cast(ex["target"], tf.int32)
    return images, labels


@tf.function
def read_unlabeled_tfrecord_batch(examples):
    ex = tf.io.parse_example(examples, UNLABELED_TFREC_FORMAT)
    images = tf.map_fn(
        decode_image,
        ex["image"],
        fn_output_signature=tf.float32,
        parallel_iterations=16,
    )
    ids = ex["image_name"]
    return images, ids


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
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        ds = ds.map(
            read_labeled_tfrecord_batch, num_parallel_calls=AUTO, deterministic=ordered
        )
    else:
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        ds = ds.map(
            read_unlabeled_tfrecord_batch,
            num_parallel_calls=AUTO,
            deterministic=ordered,
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
    dataset = dataset.map(
        lambda x, y: (
            tf.map_fn(
                lambda im: data_augment(im, 0)[0], x, fn_output_signature=tf.float32
            ),
            y,
        ),
        num_parallel_calls=AUTO,
        deterministic=False,
    )
    dataset = dataset.unbatch()
    dataset = dataset.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)
    dataset = _maybe_prefetch_to_device(dataset)
    return dataset


def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
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
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)



## === cell 9
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
)
model2.summary()



## === cell 10
train_ds = get_training_dataset()

history2 = model2.fit(
    train_ds,
    epochs=EPOCHS,
    callbacks=[lr_schedule],
    steps_per_epoch=STEPS_PER_EPOCH,
)



## === cell 11
test_ds = get_test_dataset(ordered=True)

print("Computing predictions...")
test_ids = test["image_name"].to_numpy(dtype=str)

test_images_ds = test_ds.map(lambda image, idnum: image, num_parallel_calls=AUTO)
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
