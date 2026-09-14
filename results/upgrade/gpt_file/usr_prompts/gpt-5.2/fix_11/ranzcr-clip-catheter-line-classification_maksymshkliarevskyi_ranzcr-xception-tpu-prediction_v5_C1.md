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
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam

import warnings

warnings.simplefilter("ignore")

print("TF:", tf.__version__)

SEED = 42
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPUs:", gpus)
except Exception as e:
    print("GPU config skipped:", e)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## === cell 1
WORK_DIR = "../input/ranzcr-clip-catheter-line-classification"
print("WORK_DIR exists:", os.path.isdir(WORK_DIR))
print("WORK_DIR files:", os.listdir(WORK_DIR)[:10])




## === cell 2
train_dir = os.path.join(WORK_DIR, "train")
test_dir = os.path.join(WORK_DIR, "test")
print("Train dir exists:", os.path.isdir(train_dir))
print("Test dir exists:", os.path.isdir(test_dir))




## === cell 3
train = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
ss = pd.read_csv(os.path.join(WORK_DIR, "sample_submission.csv"))
train_annot = pd.read_csv(os.path.join(WORK_DIR, "train_annotations.csv"))

label_cols = [c for c in train.columns if c not in ["StudyInstanceUID", "PatientID"]]

train_images = (WORK_DIR + "/train/" + train["StudyInstanceUID"] + ".jpg").to_numpy()
test_images = (WORK_DIR + "/test/" + ss["StudyInstanceUID"] + ".jpg").to_numpy()

labels = train[label_cols].values.astype(np.float32)

print("Num train:", len(train), "Num test:", len(ss))
print("Num targets:", len(label_cols))
print("Targets:\n", label_cols)




## === cell 4
print("Skipping EDA plots to save time under the 600s budget.")




## === cell 5
print("Skipping sample image visualization to save time under the 600s budget.")




## === cell 6
BATCH_SIZE = 8 * 1
EPOCHS = 30
TARGET_SIZE = 299  # was 750

print("BATCH_SIZE:", BATCH_SIZE)
print("EPOCHS:", EPOCHS)
print("TARGET_SIZE:", TARGET_SIZE)




## === cell 7
def build_decoder(with_labels=True, target_size=(TARGET_SIZE, TARGET_SIZE), ext="jpg"):
    target_h, target_w = target_size

    @tf.function
    def decode(path):
        file_bytes = tf.io.read_file(path)
        if ext in ["jpg", "jpeg"]:
            img = tf.image.decode_jpeg(
                file_bytes, channels=3, dct_method="INTEGER_FAST"
            )
        elif ext == "png":
            img = tf.image.decode_png(file_bytes, channels=3)
        else:
            raise ValueError("Image extension not supported")

        img = tf.image.convert_image_dtype(
            img, tf.float32
        )  # == cast/255 but often faster
        img = tf.image.resize(
            img, target_size, method=tf.image.ResizeMethod.BILINEAR, antialias=False
        )
        img = tf.ensure_shape(img, (target_h, target_w, 3))
        return img

    def decode_with_labels(path, label):
        return decode(path), label

    return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True):
    @tf.function
    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.adjust_brightness(img, 0.1)
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
    drop_remainder=False,
):
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_fusion = True
    options.threading.private_threadpool_size = 0  # let TF autotune

    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)

    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)

    AUTO = tf.data.AUTOTUNE
    slices = paths if labels is None else (paths, labels)

    dset = tf.data.Dataset.from_tensor_slices(slices).with_options(options)

    dset = dset.map(decode_fn, num_parallel_calls=AUTO)

    if cache:
        if cache_dir:
            pass
        else:
            dset = dset.cache()

    if shuffle:
        dset = dset.shuffle(shuffle, seed=SEED, reshuffle_each_iteration=True)

    if augment:
        dset = dset.map(augment_fn, num_parallel_calls=AUTO)

    if repeat:
        dset = dset.repeat()

    dset = dset.batch(bsize, drop_remainder=drop_remainder)
    dset = dset.prefetch(AUTO)
    return dset




## === cell 8
gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
train_idx, valid_idx = next(gss.split(train, groups=train["PatientID"].values))

train_paths = train_images[train_idx]
valid_paths = train_images[valid_idx]

train_labels = labels[train_idx]
valid_labels = labels[valid_idx]

STEPS_PER_EPOCH = int(np.ceil(len(train_paths) / BATCH_SIZE))
VALIDATION_STEPS = int(np.ceil(len(valid_paths) / BATCH_SIZE))

CACHE_BASE = os.path.join(".", "tf_cache_ranzcr_299")
os.makedirs(CACHE_BASE, exist_ok=True)
TRAIN_CACHE = os.path.join(CACHE_BASE, "train.cache")
VALID_CACHE = os.path.join(CACHE_BASE, "valid.cache")
TEST_CACHE = os.path.join(CACHE_BASE, "test.cache")

train_ds = build_dataset(
    train_paths,
    train_labels,
    bsize=BATCH_SIZE,
    repeat=True,
    shuffle=2048,
    augment=True,
    cache=True,
    cache_dir=TRAIN_CACHE,  # kept for API compatibility; disk caching disabled inside build_dataset for speed
    drop_remainder=True,
)

valid_ds = build_dataset(
    valid_paths,
    valid_labels,
    bsize=BATCH_SIZE,
    repeat=True,  # keep repeat=True to preserve original training loop semantics exactly
    shuffle=False,
    augment=False,
    cache=True,
    cache_dir=VALID_CACHE,  # kept; disk caching disabled inside build_dataset
    drop_remainder=True,
)

test_ds = build_dataset(
    test_images,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=True,
    cache_dir=TEST_CACHE,  # kept; disk caching disabled inside build_dataset
    drop_remainder=False,
)

print("STEPS_PER_EPOCH:", STEPS_PER_EPOCH)
print("VALIDATION_STEPS:", VALIDATION_STEPS)
print(train_ds, valid_ds, test_ds)




## === cell 9
def build_model(input_shape=(TARGET_SIZE, TARGET_SIZE, 3), n_classes=11):
    base = Xception(include_top=False, weights="imagenet", input_shape=input_shape)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(n_classes, activation="sigmoid")(x)
    m = models.Model(inputs=base.input, outputs=out)
    return m


model = build_model(n_classes=len(label_cols))

model.compile(
    optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", jit_compile=True
)
model.summary()




## === cell 10
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    verbose=2,
)




## === cell 11
test_steps = int(np.ceil(len(ss) / BATCH_SIZE))
pred = model.predict(test_ds, steps=test_steps, verbose=1)
pred = np.clip(pred, 0.0, 1.0)

for c in label_cols:
    if c not in ss.columns:
        ss[c] = 0.0

ss[label_cols] = pred.astype(np.float32)

submission = ss[["StudyInstanceUID"] + label_cols].copy()
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

assert os.path.exists("submission.csv")
sub_check = pd.read_csv("submission.csv")
assert sub_check.shape[0] == ss.shape[0]
assert list(sub_check.columns) == ["StudyInstanceUID"] + label_cols
print("Submission columns OK. Rows:", sub_check.shape[0])
