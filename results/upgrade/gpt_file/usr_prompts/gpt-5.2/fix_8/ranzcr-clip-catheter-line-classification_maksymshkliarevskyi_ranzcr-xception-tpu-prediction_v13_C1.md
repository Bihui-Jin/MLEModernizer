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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import warnings

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam

tf.get_logger().setLevel("ERROR")

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())


## === cell 1
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    ncpu = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(max(1, ncpu // 2))
    tf.config.threading.set_inter_op_parallelism_threads(max(1, ncpu // 2))
except Exception:
    pass


## === cell 2
WORK_DIR = "../input/ranzcr-clip-catheter-line-classification"
if not os.path.exists(WORK_DIR):
    WORK_DIR = "/kaggle/input/ranzcr-clip-catheter-line-classification"
print("WORK_DIR:", WORK_DIR)
print("Exists:", os.path.exists(WORK_DIR))


## === cell 3
train_dir = os.path.join(WORK_DIR, "train")
test_dir = os.path.join(WORK_DIR, "test")
print("Train dir exists:", os.path.exists(train_dir))
print("Test dir exists:", os.path.exists(test_dir))


## === cell 4
train = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))

TARGET_COLS = [
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
missing = [c for c in TARGET_COLS if c not in train.columns]
if missing:
    raise ValueError(f"Train is missing expected target columns: {missing}")

ss_in = pd.read_csv(os.path.join(WORK_DIR, "sample_submission.csv"))
if "StudyInstanceUID" not in ss_in.columns:
    raise ValueError("sample_submission.csv must contain StudyInstanceUID")

train_images = (
    WORK_DIR + "/train/" + train["StudyInstanceUID"].astype(str) + ".jpg"
).values
test_images = (
    WORK_DIR + "/test/" + ss_in["StudyInstanceUID"].astype(str) + ".jpg"
).values

labels = train[TARGET_COLS].values.astype(np.float32)

train_annot_path = os.path.join(WORK_DIR, "train_annotations.csv")
train_annot = (
    pd.read_csv(train_annot_path) if os.path.exists(train_annot_path) else None
)

print("Targets:\n", "*" * 20, "\n", np.array(TARGET_COLS))
print("*" * 50)
print(train.head(2))
print("Sample submission columns:", list(ss_in.columns))


## === cell 5
BATCH_SIZE = 8
EPOCHS = 30
TARGET_SIZE = 750




## === cell 6
def build_decoder(with_labels=True, target_size=(TARGET_SIZE, TARGET_SIZE), ext="jpg"):
    def decode(path):
        file_bytes = tf.io.read_file(path)
        if ext == "png":
            img = tf.image.decode_png(file_bytes, channels=3)
        elif ext in ["jpg", "jpeg"]:
            img = tf.image.decode_jpeg(file_bytes, channels=3)
        else:
            raise ValueError("Image extension not supported")

        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, target_size)
        return img

    def decode_with_labels(path, label):
        return decode(path), label

    return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True):
    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)

        saturation = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        if saturation >= 0.5:
            img = tf.image.random_saturation(img, lower=0.9, upper=1.1)

        contrast = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        if contrast >= 0.5:
            img = tf.image.random_contrast(img, lower=0.9, upper=1.1)

        brightness = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        if brightness >= 0.5:
            img = tf.image.random_brightness(img, max_delta=0.1)

        return img

    def augment_with_labels(img, label):
        return augment(img), label

    return augment_with_labels if with_labels else augment


def build_dataset(
    paths,
    labels=None,
    bsize=32,
    cache=True,
    decode_fn=None,
    augment_fn=None,
    augment=True,
    repeat=True,
    shuffle=1024,
    cache_dir="",
):
    options = tf.data.Options()
    options.deterministic = True
    try:
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_parallelization = True
    except Exception:
        pass

    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)

    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)

    AUTO = tf.data.AUTOTUNE
    slices = paths if labels is None else (paths, labels)

    dset = tf.data.Dataset.from_tensor_slices(slices)
    dset = dset.with_options(options)

    dset = dset.map(decode_fn, num_parallel_calls=AUTO, deterministic=True)

    if cache:
        if cache_dir:
            os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
            dset = dset.cache(cache_dir)
        else:
            dset = dset.cache()

    if augment:
        dset = dset.map(augment_fn, num_parallel_calls=AUTO, deterministic=True)

    if repeat:
        dset = dset.repeat()

    if shuffle:
        dset = dset.shuffle(shuffle, seed=SEED, reshuffle_each_iteration=True)

    dset = dset.batch(bsize, drop_remainder=False)
    dset = dset.prefetch(AUTO)
    return dset




## === cell 7
patients = train["PatientID"].values
unique_patients = np.unique(patients)

rng = np.random.RandomState(SEED)
perm = rng.permutation(unique_patients)
val_size = int(round(0.2 * len(unique_patients)))
val_p = perm[:val_size]
train_p = perm[val_size:]

is_val = np.isin(patients, val_p)
tr_paths = train_images[~is_val]
va_paths = train_images[is_val]
tr_labels = labels[~is_val]
va_labels = labels[is_val]

print("Train samples:", len(tr_paths), "Valid samples:", len(va_paths))

STEPS_PER_EPOCH = int(np.ceil(len(tr_paths) / BATCH_SIZE))
VALIDATION_STEPS = int(np.ceil(len(va_paths) / BATCH_SIZE))
TEST_STEPS = int(np.ceil(len(test_images) / BATCH_SIZE))

CACHE_BASE = "/kaggle/working/tf_cache_ranzcr"
train_cache = os.path.join(CACHE_BASE, f"train_decoded_{TARGET_SIZE}")
valid_cache = os.path.join(CACHE_BASE, f"valid_{TARGET_SIZE}")
test_cache = os.path.join(CACHE_BASE, f"test_{TARGET_SIZE}")

decoded_train_ds = build_dataset(
    tr_paths,
    tr_labels,
    bsize=BATCH_SIZE,
    repeat=True,
    shuffle=2048,
    augment=False,  # augmentation applied after cache below
    cache=True,
    cache_dir=train_cache,
)

AUTO = tf.data.AUTOTUNE
decoded_train_ds = decoded_train_ds.map(
    build_augmenter(with_labels=True),
    num_parallel_calls=AUTO,
    deterministic=True,
).prefetch(AUTO)
train_ds = decoded_train_ds

valid_ds = build_dataset(
    va_paths,
    va_labels,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=True,
    cache_dir=valid_cache,
)
test_ds = build_dataset(
    test_images,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=True,
    cache_dir=test_cache,
)




## === cell 8
def build_model(input_shape=(TARGET_SIZE, TARGET_SIZE, 3), n_classes=len(TARGET_COLS)):
    base = Xception(include_top=False, weights="imagenet", input_shape=input_shape)

    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(n_classes, activation="sigmoid")(x)
    model = models.Model(inputs=base.input, outputs=out)
    return model


model = build_model()
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    jit_compile=True,
)
model.summary()


## === cell 9
ckpt_path = "best_model.keras"
callbacks = [
    ModelCheckpoint(
        ckpt_path, monitor="val_loss", save_best_only=True, mode="min", verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
    ),
]

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    callbacks=callbacks,
    verbose=1,
)


## === cell 10
if os.path.exists(ckpt_path):
    model = tf.keras.models.load_model(ckpt_path)
else:
    print("WARNING: checkpoint not found; using last-epoch model weights.")


## === cell 11
pred = model.predict(test_ds, verbose=1, steps=TEST_STEPS)
pred = np.clip(pred, 0.0, 1.0)
print("Pred shape:", pred.shape)


## === cell 12
submission = pd.DataFrame(
    {"StudyInstanceUID": ss_in["StudyInstanceUID"].astype(str).values}
)

if pred.shape[1] != len(TARGET_COLS):
    raise ValueError(
        f"Model output shape {pred.shape} does not match n_targets={len(TARGET_COLS)}"
    )

for j, col in enumerate(TARGET_COLS):
    submission[col] = pred[:, j]

desired_cols = ["StudyInstanceUID"] + TARGET_COLS
submission = submission[desired_cols]

assert submission.shape[0] == len(ss_in), "Submission row count mismatch"
assert list(submission.columns) == desired_cols, "Submission columns mismatch"

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())


## === cell 13
chk = pd.read_csv("submission.csv")
print(chk.shape)
print(chk.columns.tolist())
print(chk.head())
