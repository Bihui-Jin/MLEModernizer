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

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K

from sklearn.model_selection import train_test_split

print("TF version:", tf.__version__)
print("Num GPUs available:", len(tf.config.list_physical_devices("GPU")))

tf.random.set_seed(42)
np.random.seed(42)

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("Could not enable XLA JIT:", e)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass




## === cell 1
BASE_PATH = "/kaggle/input/siim-isic-melanoma-classification"
print("Exists:", os.path.exists(BASE_PATH))
print("Top-level files:", [p for p in os.listdir(BASE_PATH) if p.endswith(".csv")][:10])
print("TFRecords dir exists:", os.path.exists(os.path.join(BASE_PATH, "tfrecords")))
print(
    "JPEG train dir exists:", os.path.exists(os.path.join(BASE_PATH, "jpeg", "train"))
)
print("JPEG test dir exists:", os.path.exists(os.path.join(BASE_PATH, "jpeg", "test")))




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

GCS_PATH = BASE_PATH  # local path

EPOCHS = 10
BATCH_SIZE = 8 * strategy.num_replicas_in_sync

IMAGE_SIZE = [384, 384]

print("Using path:", GCS_PATH)
print("BATCH_SIZE:", BATCH_SIZE, "EPOCHS:", EPOCHS, "IMAGE_SIZE:", IMAGE_SIZE)




## === cell 4
HEIGHT = IMAGE_SIZE[0]
WIDTH = IMAGE_SIZE[1]
CHANNELS = 3




## === cell 5
def append_path(pre, base=BASE_PATH):
    def _join(files):
        files = np.asarray(files, dtype=str)
        prefix = os.path.join(base, pre) + os.sep
        return np.char.add(prefix, files)

    return _join




## === cell 6
sub = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))
sub.head()




## === cell 7
train = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
train.head()




## === cell 8
print("Train rows:", len(train))
print(train["target"].value_counts(dropna=False))




## === cell 9
TRAIN_JPEG_DIR = os.path.join(BASE_PATH, "jpeg", "train")
TEST_JPEG_DIR = os.path.join(BASE_PATH, "jpeg", "test")

train_paths = (
    TRAIN_JPEG_DIR + os.sep + train["image_name"].astype(str) + ".jpg"
).values
train_labels = train["target"].astype(np.int32).values

test_df = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))
test_paths = (
    TEST_JPEG_DIR + os.sep + test_df["image_name"].astype(str) + ".jpg"
).values
test_ids = test_df["image_name"].astype(str).values

NUM_TRAINING_IMAGES = len(train_paths)
NUM_TEST_IMAGES = len(test_paths)

STEPS_PER_EPOCH = max(1, NUM_TRAINING_IMAGES // BATCH_SIZE)

print(f"Dataset: {NUM_TRAINING_IMAGES} training images, {NUM_TEST_IMAGES} test images")
print("STEPS_PER_EPOCH:", STEPS_PER_EPOCH)

CLASSES = [0, 1]




## === cell 10
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




## === cell 11
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
def decode_jpeg_resize(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    img = tf.ensure_shape(img, [IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    return img


@tf.function
def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label


DATASET_OPTS = tf.data.Options()
DATASET_OPTS.experimental_deterministic = True
try:
    DATASET_OPTS.autotune.enabled = True
except Exception:
    pass


def get_training_dataset():
    ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    ds = ds.with_options(DATASET_OPTS)
    ds = ds.shuffle(2048, reshuffle_each_iteration=True)
    ds = ds.repeat()
    ds = ds.map(lambda p, y: (decode_jpeg_resize(p), y), num_parallel_calls=AUTO)
    ds = ds.map(data_augment, num_parallel_calls=AUTO)
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    ds = ds.prefetch(AUTO)
    return ds


def get_test_dataset(ordered=True):
    ds = tf.data.Dataset.from_tensor_slices((test_paths, test_ids))
    ds = ds.with_options(DATASET_OPTS)
    ds = ds.map(lambda p, iid: (decode_jpeg_resize(p), iid), num_parallel_calls=AUTO)
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTO)
    return ds




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
    backbone = tf.keras.applications.EfficientNetB6(
        input_shape=(*IMAGE_SIZE, 3), weights="imagenet", include_top=False
    )
    model = tf.keras.Sequential(
        [
            backbone,
            L.GlobalAveragePooling2D(),
            L.Dense(512, activation="relu"),
            L.Dense(128, activation="relu"),
            L.Dense(1, activation="sigmoid"),
        ]
    )

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=16,
)
model.summary()




## === cell 15
pass




## === cell 16
lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
STEPS_PER_EPOCH = max(1, NUM_TRAINING_IMAGES // BATCH_SIZE)




## === cell 17
history = model.fit(
    get_training_dataset(),
    epochs=EPOCHS,
    callbacks=[lr_schedule],
    steps_per_epoch=STEPS_PER_EPOCH,
)




## === cell 18
test_ds = get_test_dataset(ordered=True)
print("Computing predictions...")

probabilities_list = []
ids_list = []
for batch_images, batch_ids in test_ds:
    probabilities_list.append(model(batch_images, training=False))
    ids_list.append(batch_ids)

probabilities = tf.concat(probabilities_list, axis=0).numpy().reshape(-1)
test_ids_from_ds = tf.concat(ids_list, axis=0).numpy().astype("U")

print("Pred shape:", probabilities.shape)
print("IDs shape:", test_ids_from_ds.shape)




## === cell 19
print("Generating submission.csv file...")
pred_df = pd.DataFrame({"image_name": test_ids_from_ds, "target": probabilities})
pred_df.head()




## === cell 20
sub_out = sub[["image_name"]].merge(pred_df, on="image_name", how="left")

if sub_out["target"].isna().any():
    sub_out["target"] = sub_out["target"].fillna(
        float(np.nanmean(sub_out["target"].values))
    )

sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
print(sub_out.head())

model.save("EffNetB6-Melanoma.h5")
print("Saved model to EffNetB6-Melanoma.h5")
