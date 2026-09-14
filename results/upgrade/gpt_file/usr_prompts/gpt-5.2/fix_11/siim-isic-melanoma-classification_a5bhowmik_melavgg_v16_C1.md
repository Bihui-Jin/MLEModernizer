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
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import re
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

tf.config.optimizer.set_jit(False)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TF version:", tf.__version__)


## === cell 1
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    Dropout,
    LeakyReLU,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.vgg19 import VGG19

try:
    from kaggle_datasets import KaggleDatasets
except Exception:
    KaggleDatasets = None


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
input_path = "/kaggle/input/siim-isic-melanoma-classification"
output_path = "/kaggle/working/"

train_data = pd.read_csv(
    os.path.join(input_path, "train.csv"), usecols=["image_name", "target"]
)
test_data = pd.read_csv(os.path.join(input_path, "test.csv"))

NUM_TEST_IMAGES = test_data.shape[0]
print("Train rows:", train_data.shape[0], "Test rows:", test_data.shape[0])




## === cell 4
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(
            alpha * K.pow(1.0 - pt_1, gamma) * K.log(pt_1 + K.epsilon())
        ) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1.0 - pt_0 + K.epsilon()))

    return focal_loss_fixed




## === cell 5
with strategy.scope():
    base = VGG19(input_shape=(256, 256, 3), weights="imagenet", include_top=False)
    base.trainable = True
    model_copy = tf.keras.Sequential(
        [
            base,
            Dense(4096, name="fc1"),
            LeakyReLU(alpha=0.2),
            Dropout(0.2),
            Dense(4096, name="fc2"),
            LeakyReLU(alpha=0.2),
            Dropout(0.2),
            Dense(1000, name="fc3"),
            LeakyReLU(alpha=0.2),
            GlobalAveragePooling2D(),
            Dropout(0.2),
            Dense(1, activation="sigmoid", trainable=True),
        ]
    )

    adadelta = tf.keras.optimizers.Adadelta(learning_rate=0.001)
    model_copy.compile(
        optimizer=adadelta,
        loss="binary_crossentropy",
        metrics=["AUC", "Recall", "Precision"],
    )

model_copy.summary()


## === cell 6
LR_START = 0.00001
LR_MAX = 0.00005 * strategy.num_replicas_in_sync
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 5
LR_SUSTAIN_EPOCHS = 0
LR_EXP_DECAY = 0.8


def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (
            epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
        ) + LR_MIN
    return lr


lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)


## === cell 7
IMAGE_SIZE = [256, 256]




## === cell 8
@tf.function
def decode_image(image_data):
    image = tf.io.decode_jpeg(image_data, channels=3, dct_method="INTEGER_FAST")
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.ensure_shape(image, [1024, 1024, 3])
    image = tf.image.resize(image, [256, 256])
    return image


@tf.function
def decode_image_generated(image_data):
    image = tf.io.decode_jpeg(image_data, channels=3, dct_method="INTEGER_FAST")
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, [256, 256])
    image = tf.ensure_shape(image, [256, 256, 3])
    return image


@tf.function
def decode_image_test(image_data):
    image = tf.io.decode_jpeg(image_data, channels=3, dct_method="INTEGER_FAST")
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.ensure_shape(image, [1024, 1024, 3])
    image = tf.image.resize(image, [256, 256])
    return image




## === cell 9
@tf.function
def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image, label




## === cell 10
AUTO = tf.data.experimental.AUTOTUNE


def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example["image"])
    label = tf.cast(example["target"], tf.int32)
    return image, label


def read_labeled_generated_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image_generated(example["image"])
    label = tf.cast(example["target"], tf.int32)
    return image, label


def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image_test(example["image"])
    idnum = example["image_name"]
    return image, idnum


def load_dataset(filenames, labeled=True, ordered=False, generated=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False
    else:
        ignore_order.experimental_deterministic = True

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(
        (
            read_labeled_generated_tfrecord
            if (labeled and generated)
            else read_labeled_tfrecord if labeled else read_unlabeled_tfrecord
        ),
        num_parallel_calls=AUTO,
    )
    return dataset


JPEG_TRAIN_DIR = os.path.join(input_path, "jpeg", "train")
JPEG_TEST_DIR = os.path.join(input_path, "jpeg", "test")


def _jpeg_path_from_name(name, train=True):
    base = JPEG_TRAIN_DIR if train else JPEG_TEST_DIR
    return tf.strings.join([base, "/", name, ".jpg"])


@tf.function
def read_labeled_jpeg(name, label):
    path = _jpeg_path_from_name(name, train=True)
    image_bytes = tf.io.read_file(path)
    image = decode_image_generated(image_bytes)
    return image, tf.cast(label, tf.int32)


@tf.function
def read_unlabeled_jpeg(name):
    path = _jpeg_path_from_name(name, train=False)
    image_bytes = tf.io.read_file(path)
    image = decode_image_generated(image_bytes)
    return image, name


def load_jpeg_train_dataset_from_df(df, ordered=False):
    opts = tf.data.Options()
    opts.experimental_deterministic = bool(ordered)

    names = tf.convert_to_tensor(df["image_name"].values, dtype=tf.string)
    labels = tf.convert_to_tensor(df["target"].values, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((names, labels))
    ds = ds.with_options(opts)
    ds = ds.map(read_labeled_jpeg, num_parallel_calls=AUTO)
    return ds


def load_jpeg_test_dataset_from_df(df, ordered=True):
    opts = tf.data.Options()
    opts.experimental_deterministic = bool(ordered)

    names = tf.convert_to_tensor(df["image_name"].values, dtype=tf.string)
    ds = tf.data.Dataset.from_tensor_slices(names)
    ds = ds.with_options(opts)
    ds = ds.map(read_unlabeled_jpeg, num_parallel_calls=AUTO)
    return ds




## === cell 11
TRAINING_FILENAMES = tf.io.gfile.glob(
    os.path.join(input_path, "tfrecords", "train*.tfrec")
)
TEST_FILENAMES = tf.io.gfile.glob(os.path.join(input_path, "tfrecords", "test*.tfrec"))

GENERATED_BASE = "/kaggle/input/generated"
Generated = (
    tf.io.gfile.glob(os.path.join(GENERATED_BASE, "generated*.tfrec"))
    if tf.io.gfile.exists(GENERATED_BASE)
    else []
)

print("Found train tfrecords:", len(TRAINING_FILENAMES))
print("Found test tfrecords:", len(TEST_FILENAMES))
print("Found generated tfrecords:", len(Generated))

print("JPEG train dir exists:", tf.io.gfile.exists(JPEG_TRAIN_DIR))
print("JPEG test dir exists:", tf.io.gfile.exists(JPEG_TEST_DIR))


## === cell 12
BATCH_SIZE = 16
EPOCHS = 1


def get_test_dataset(ordered=False):
    if tf.io.gfile.exists(JPEG_TEST_DIR):
        dataset = load_jpeg_test_dataset_from_df(test_data, ordered=True)
    else:
        dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)
    dataset = dataset.prefetch(AUTO)
    return dataset




## === cell 13
def count_data_items(filenames):
    n = [
        int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
        for filename in filenames
    ]
    return int(np.sum(n))


dataset1_cnt = count_data_items(TRAINING_FILENAMES)
print("Training items (base):", dataset1_cnt)

if len(Generated) > 0:
    dataset2_cnt = count_data_items(Generated)
    print("Training items (generated):", dataset2_cnt)
else:
    dataset2_cnt = 0

train_size_1 = int(0.8 * dataset1_cnt) + 1

if tf.io.gfile.exists(JPEG_TRAIN_DIR):
    train_df_shuf = train_data.sample(frac=1.0, random_state=SEED).reset_index(
        drop=True
    )
    train_df_1 = train_df_shuf.iloc[:train_size_1]
    valid_df_1 = train_df_shuf.iloc[train_size_1:]

    train_dataset_1 = load_jpeg_train_dataset_from_df(train_df_1, ordered=False).map(
        data_augment, num_parallel_calls=AUTO
    )
    valid_dataset_1 = load_jpeg_train_dataset_from_df(valid_df_1, ordered=True)
else:
    _base = load_dataset(TRAINING_FILENAMES, labeled=True).shuffle(
        2048, seed=SEED, reshuffle_each_iteration=False
    )
    train_dataset_1 = _base.take(train_size_1).map(
        data_augment, num_parallel_calls=AUTO
    )
    valid_dataset_1 = _base.skip(train_size_1)

if dataset2_cnt > 0:
    train_size_2 = int(0.8 * dataset2_cnt) + 1

    _gen = load_dataset(Generated, labeled=True, generated=True).shuffle(
        2048, seed=SEED + 1, reshuffle_each_iteration=False
    )

    train_dataset_2 = _gen.take(train_size_2).map(data_augment, num_parallel_calls=AUTO)
    valid_dataset_2 = _gen.skip(train_size_2)

    train_dataset = train_dataset_1.concatenate(train_dataset_2)
    valid_dataset = valid_dataset_1.concatenate(valid_dataset_2)
else:
    train_dataset = train_dataset_1
    valid_dataset = valid_dataset_1
    train_size_2 = 0  # ensure defined

options = tf.data.Options()
options.experimental_deterministic = True
try:
    options.experimental_optimization.map_parallelization = True
except Exception:
    pass

train_dataset = train_dataset.with_options(options)
valid_dataset = valid_dataset.with_options(options)

train_dataset = train_dataset.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTO)
valid_dataset = valid_dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)

TRAIN_ITEMS = train_size_1 + (train_size_2 if dataset2_cnt > 0 else 0)
VALID_ITEMS = (dataset1_cnt - train_size_1) + (
    (dataset2_cnt - train_size_2) if dataset2_cnt > 0 else 0
)
STEPS_PER_EPOCH = max(1, TRAIN_ITEMS // BATCH_SIZE)  # matches drop_remainder=True

VALIDATION_STEPS = max(1, min(200, int(np.ceil(VALID_ITEMS / BATCH_SIZE))))

print("Train items used:", TRAIN_ITEMS, "Valid items used:", VALID_ITEMS)
print("Steps/epoch:", STEPS_PER_EPOCH, "Val steps (capped):", VALIDATION_STEPS)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.1, patience=4, verbose=1, min_lr=1e-8, mode="auto"
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", min_delta=0, verbose=1, patience=5, mode="auto"
)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(output_path, "model.h5"),
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode="auto",
)


## === cell 14
history = model_copy.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=valid_dataset,
    validation_steps=VALIDATION_STEPS,
    callbacks=[reduce_lr, early_stopping, checkpoint],
)


## === cell 15
test_ds = get_test_dataset(ordered=True)
print("Computing predictions...")

all_ids = []
all_probs = []

for batch_imgs, batch_ids in test_ds:
    probs = model_copy(batch_imgs, training=False)
    probs = tf.reshape(probs, [-1])
    all_probs.append(probs)
    all_ids.append(batch_ids)

test_ids = tf.concat(all_ids, axis=0).numpy().astype("U")
pred_probs = tf.concat(all_probs, axis=0).numpy().astype(np.float32)

pred_df = pd.DataFrame({"image_name": test_ids, "target": pred_probs})
sub = test_data[["image_name"]].merge(pred_df, on="image_name", how="left")

if sub["target"].isna().any():
    fill_val = float(np.nanmean(pred_probs)) if np.isfinite(pred_probs).any() else 0.5
    sub["target"] = sub["target"].fillna(fill_val)

sub_path = os.path.join(output_path, "submission.csv")
sub.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "Rows:", len(sub), "NaNs:", int(sub["target"].isna().sum()))
print(sub.head())


## === cell 16
try:
    lr_value = float(tf.keras.backend.get_value(model_copy.optimizer.learning_rate))
    print("Current learning rate:", lr_value)
except Exception as e:
    print("Could not read learning rate:", repr(e))
