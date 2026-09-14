# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect the presence and position of catheters and lines on chest x-rays.

## Metric
Area under the ROC curve for each label, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each ID in the test set, you must predict a probability for all target variables. The file should contain a header and have the following format:
```
StudyInstanceUID,ETT - Abnormal,ETT - Borderline,ETT - Normal,NGT - Abnormal,NGT - Borderline,NGT - Incompletely Imaged,NGT - Normal,CVC - Abnormal,CVC - Borderline,CVC - Normal,Swan Ganz Catheter Present
1.2.826.0.1.3680043.8.498.62451881164053375557257228990443168843,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.83721761279899623084220697845011427274,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.12732270010839808189235995393981377825,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.11769539755086084996287023095028033598,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.87838627504097587943394933987052577153,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.53211840524738036417560823327351887819,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.93555795394184819372299157360228027866,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.52241894131170494723503100795076463919,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.36500167484503936720548852591033878284,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.86199852603457900780565655267977637728,0,0,0,0,0,0,0,0,0,0,0
```

## Dataset
`train.csv` contains image IDs, binary labels, and patient IDs.

TFRecords are available for both train and test.

We've also included `train_annotations.csv`. These are segmentation annotations for training samples that have them. They are included solely as additional information for competitors.

- train.csv - contains image IDs, binary labels, and patient IDs.
- sample_submission.csv - a sample submission file in the correct format
- test - test images
- train - training images

### Columns
- `StudyInstanceUID` - unique ID for each image
- `ETT - Abnormal` - endotracheal tube placement abnormal
- `ETT - Borderline` - endotracheal tube placement borderline abnormal
- `ETT - Normal` - endotracheal tube placement normal
- `NGT - Abnormal` - nasogastric tube placement abnormal
- `NGT - Borderline` - nasogastric tube placement borderline abnormal
- `NGT - Incompletely Imaged` - nasogastric tube placement inconclusive due to imaging
- `NGT - Normal` - nasogastric tube placement borderline normal
- `CVC - Abnormal` - central venous catheter placement abnormal
- `CVC - Borderline` - central venous catheter placement borderline abnormal
- `CVC - Normal` - central venous catheter placement normal
- `Swan Ganz Catheter Present`
- `PatientID` - unique ID for each patient in the dataset

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
        input/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
        working/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
```

-> data/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/ranzcr-clip-catheter-line-classification/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/ranzcr-clip-catheter-line-classification/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> data/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import math, warnings, random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras import Sequential

warnings.filterwarnings("ignore")




## === cell 1
SEED = 555
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF version:", tf.__version__)




## === cell 2
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print(f"Running on TPU {tpu.master()}")
except Exception:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f"REPLICAS: {REPLICAS}")




## === cell 3
database_base_path = "/kaggle/input/ranzcr-clip-catheter-line-classification/"
TRAIN_IMG_DIR = os.path.join(database_base_path, "train")
TEST_IMG_DIR = os.path.join(database_base_path, "test")

train_csv_path = os.path.join(database_base_path, "train.csv")
sample_sub_path = os.path.join(database_base_path, "sample_submission.csv")

assert tf.io.gfile.exists(train_csv_path), f"Missing: {train_csv_path}"
assert tf.io.gfile.exists(sample_sub_path), f"Missing: {sample_sub_path}"
assert tf.io.gfile.exists(TRAIN_IMG_DIR), f"Missing: {TRAIN_IMG_DIR}"
assert tf.io.gfile.exists(TEST_IMG_DIR), f"Missing: {TEST_IMG_DIR}"

train_df = pd.read_csv(train_csv_path)
submission = pd.read_csv(sample_sub_path)

print("train_df:", train_df.shape)
print("submission:", submission.shape)
print("submission columns:", list(submission.columns))




## === cell 4
BATCH_SIZE = 16 * REPLICAS
HEIGHT = 512
WIDTH = 512
CHANNELS = 3

N_CLASSES = 5

TTA_STEPS = 3  # Do TTA if > 0
IMAGE_SIZE = [512, 512]

AUG_BATCH = BATCH_SIZE




## === cell 5
label_cols_all = [
    "ETT - Abnormal",
    "ETT - Borderline",
    "ETT - Normal",
    "NGT - Abnormal",
    "NGT - Borderline",
    "NGT - Incompletely Imaged",
    "NGT - Normal",
    "CVC - Abnormal",
    "CVC - Borderline",
    "CVC - Normal",
    "Swan Ganz Catheter Present",
]
for c in label_cols_all:
    if c not in train_df.columns:
        raise ValueError(f"Expected label column missing in train.csv: {c}")

pos_counts = train_df[label_cols_all].sum().sort_values(ascending=False)
selected_label_cols = list(pos_counts.head(N_CLASSES).index)
print("Selected 5 label columns for single-class target:", selected_label_cols)
print(pos_counts.head(11))




## === cell 6
def data_augment(image, label):
    image = tf.image.random_flip_left_right(image, seed=SEED)
    image = tf.image.random_flip_up_down(image, seed=SEED)
    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.random_saturation(image, 0, 2, seed=SEED)
    image = tf.image.adjust_saturation(image, 3)
    return image, label




## === cell 7
def decode_jpg(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMAGE_SIZE, method=tf.image.ResizeMethod.AREA)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def make_single_class_label(row_values):
    row_values = tf.cast(row_values, tf.int32)
    maxv = tf.reduce_max(row_values)
    return tf.cond(
        maxv > 0,
        lambda: tf.argmax(row_values, output_type=tf.int32),
        lambda: tf.constant(0, tf.int32),
    )


def onehot(image, label):
    return image, tf.one_hot(label, N_CLASSES)




## === cell 8
_DATA_OPTS = tf.data.Options()
_DATA_OPTS.deterministic = True  # preserve reproducibility

try:
    _DATA_OPTS.experimental_optimization.autotune = True
except Exception:
    pass
try:
    _DATA_OPTS.experimental_optimization.map_parallelization = True
except Exception:
    pass
try:
    _DATA_OPTS.experimental_optimization.map_and_batch_fusion = True
except Exception:
    pass

try:
    _DATA_OPTS.threading.private_threadpool_size = max(8, os.cpu_count() or 8)
except Exception:
    pass

TRAIN_IMG_DIR_T = tf.constant(TRAIN_IMG_DIR)
TEST_IMG_DIR_T = tf.constant(TEST_IMG_DIR)


@tf.function
def _augment_batch_images(images):
    images = tf.image.random_flip_left_right(images, seed=SEED)
    images = tf.image.random_flip_up_down(images, seed=SEED)
    images = tf.image.random_brightness(images, max_delta=0.5)
    images = tf.image.random_saturation(images, 0, 2, seed=SEED)
    images = tf.image.adjust_saturation(images, 3)
    return images


def make_train_dataset(df, training=True):
    uids = df["StudyInstanceUID"].astype(str).values
    y_mat = df[selected_label_cols].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((uids, y_mat))
    ds = ds.with_options(_DATA_OPTS)

    def _load(uid, yrow):
        path = tf.strings.join([TRAIN_IMG_DIR_T, "/", uid, ".jpg"])
        image = decode_jpg(path)
        label = make_single_class_label(yrow)
        return image, label

    ds = ds.map(_load, num_parallel_calls=AUTO, deterministic=True)

    if training:
        ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)

    if training:
        ds = ds.map(
            lambda x, y: (_augment_batch_images(x), y),
            num_parallel_calls=AUTO,
            deterministic=True,
        )

    ds = ds.prefetch(AUTO)
    return ds


def make_valid_dataset(df):
    return make_train_dataset(df, training=False)


def make_test_images_dataset(uids, cache_in_memory=True):
    uids = np.asarray([str(u) for u in uids])
    ds = tf.data.Dataset.from_tensor_slices(uids)
    ds = ds.with_options(_DATA_OPTS)

    def _load(uid):
        path = tf.strings.join([TEST_IMG_DIR_T, "/", uid, ".jpg"])
        image = decode_jpg(path)
        return image

    ds = ds.map(_load, num_parallel_calls=AUTO, deterministic=True)
    if cache_in_memory:
        ds = ds.cache()  # in-memory cache of decoded images; avoids 3x disk I/O for TTA
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds




## === cell 9
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
tr_idx, va_idx = next(gss.split(train_df, groups=train_df["PatientID"]))
tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
va_df = train_df.iloc[va_idx].reset_index(drop=True)

print("Train/Valid sizes:", tr_df.shape, va_df.shape)

train_ds = make_train_dataset(tr_df, training=True)
valid_ds = make_valid_dataset(va_df)

steps_per_epoch = max(1, len(tr_df) // BATCH_SIZE)
valid_steps = max(1, math.ceil(len(va_df) / BATCH_SIZE))
print("steps_per_epoch:", steps_per_epoch, "valid_steps:", valid_steps)




## === cell 10
with strategy.scope():
    model = Sequential(
        [
            L.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
            L.Conv2D(16, 3, padding="same", activation="relu"),
            L.MaxPool2D(),
            L.Conv2D(32, 3, padding="same", activation="relu"),
            L.MaxPool2D(),
            L.Conv2D(64, 3, padding="same", activation="relu"),
            L.GlobalAveragePooling2D(),
            L.Dense(64, activation="relu"),
            L.Dense(N_CLASSES, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )

model.summary()




## === cell 11
EPOCHS = 2
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_ds,
    validation_steps=valid_steps,
    verbose=2,
)

models = [model]




## === cell 12
test_uids = submission["StudyInstanceUID"].astype(str).tolist()

test_images_ds_base = make_test_images_dataset(test_uids, cache_in_memory=True)

print("Num test ids:", len(test_uids))




## === cell 13
print(" TTA_STEPS = {} ".format(TTA_STEPS))


@tf.function
def _augment_images_only(batch_images):
    return _augment_batch_images(batch_images)


if TTA_STEPS > 0:
    probs_tta = []
    for _ in range(TTA_STEPS):
        test_images_aug_ds = test_images_ds_base.map(
            _augment_images_only, num_parallel_calls=AUTO, deterministic=True
        )
        probs = np.average(
            [m.predict(test_images_aug_ds, verbose=0) for m in models], axis=0
        )
        probs_tta.append(probs)

    probabilities = np.mean(np.stack(probs_tta, axis=0), axis=0)
else:
    probabilities = np.average(
        [models[0].predict(test_images_ds_base, verbose=0)], axis=0
    )

print("probabilities shape:", probabilities.shape)




## === cell 14
print("Generating submission.csv file...")

test_ids = np.array(test_uids, dtype="U")

if len(test_ids) == 0 or probabilities.shape[0] == 0:
    raise RuntimeError(
        "No test ids or probabilities produced; cannot write submission."
    )

n = min(len(test_ids), probabilities.shape[0], len(submission))
test_ids = test_ids[:n]
probabilities = probabilities[:n]

submission_cols = list(submission.columns)
needed_target_cols = [c for c in label_cols_all if c != "PatientID"]
for c in needed_target_cols:
    if c not in submission_cols:
        submission_cols.append(c)

sub_df = pd.DataFrame(0.0, index=np.arange(n), columns=submission_cols)
sub_df["StudyInstanceUID"] = test_ids

for i, col in enumerate(selected_label_cols):
    if col in sub_df.columns and i < probabilities.shape[1]:
        sub_df[col] = probabilities[:, i].astype(np.float32)

sub_df = sub_df[submission_cols]
sub_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub_df.shape)
print("Columns:", list(sub_df.columns))
print(sub_df.head())




## === cell 15
try:
    with open("submission.csv", "r") as f:
        for _ in range(5):
            print(f.readline().rstrip("\n"))
except Exception as e:
    print("Could not preview submission.csv:", e)
