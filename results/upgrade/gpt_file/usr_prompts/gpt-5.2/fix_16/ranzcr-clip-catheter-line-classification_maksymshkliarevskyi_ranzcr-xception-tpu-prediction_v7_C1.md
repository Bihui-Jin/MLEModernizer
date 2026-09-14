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
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", "42")

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam

import warnings

warnings.simplefilter("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TF version:", tf.__version__)
print("GPUs:", gpus)



## === cell 1
WORK_DIR = "../input/ranzcr-clip-catheter-line-classification"
if not os.path.exists(WORK_DIR):
    WORK_DIR = "/kaggle/input/ranzcr-clip-catheter-line-classification"
print("WORK_DIR:", WORK_DIR)
print("Has train.csv:", os.path.exists(os.path.join(WORK_DIR, "train.csv")))
print(
    "Has sample_submission.csv:",
    os.path.exists(os.path.join(WORK_DIR, "sample_submission.csv")),
)



## === cell 2
pass



## === cell 3
pass



## === cell 4
train = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
ss = pd.read_csv(os.path.join(WORK_DIR, "sample_submission.csv"))
train_annot = pd.read_csv(os.path.join(WORK_DIR, "train_annotations.csv"))

ALL_LABEL_COLS = [
    c for c in train.columns if c not in ["StudyInstanceUID", "PatientID"]
]

for c in ALL_LABEL_COLS:
    if c not in ss.columns:
        ss[c] = 0.0
ss = ss[["StudyInstanceUID"] + ALL_LABEL_COLS]

train_images = (
    WORK_DIR + "/train/" + train["StudyInstanceUID"].astype(str) + ".jpg"
).values
test_images = (WORK_DIR + "/test/" + ss["StudyInstanceUID"].astype(str) + ".jpg").values

labels = train[ALL_LABEL_COLS].values.astype(np.float32)

print("Labels:\n", "*" * 20, "\n", np.array(ALL_LABEL_COLS))
print("*" * 50)
print("train shape:", train.shape, "ss shape:", ss.shape)
train.head()



## === cell 5
DO_PLOTS = False
pass



## === cell 6
pass



## === cell 7
BATCH_SIZE = 16
EPOCHS = (
    2  # keep runtime safe for Kaggle execution; core training loop remains the same
)
TARGET_SIZE = 750

STEPS_PER_EPOCH = None
VALIDATION_STEPS = None




## === cell 8
def build_decoder(target_size=(TARGET_SIZE, TARGET_SIZE)):
    target_h, target_w = int(target_size[0]), int(target_size[1])

    @tf.function(reduce_retracing=True)
    def decode(path):
        file_bytes = tf.io.read_file(path)
        img = tf.io.decode_jpeg(file_bytes, channels=3, dct_method="INTEGER_FAST")
        img = tf.image.convert_image_dtype(img, tf.float32)  # identical scale
        img = tf.image.resize(img, (target_h, target_w), antialias=False)
        img.set_shape([target_h, target_w, 3])
        return img

    return decode


def build_augmenter():
    @tf.function(reduce_retracing=True)
    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.adjust_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        img = tf.image.random_saturation(img, 0.9, 1.1)
        return img

    return augment


def build_dataset(
    paths,
    labels=None,
    bsize=32,
    decode_fn=None,
    augment_fn=None,
    augment=True,
    repeat=True,
    shuffle=1024,
    drop_remainder=False,
    cache=False,
    cache_path=None,
):
    AUTO = tf.data.AUTOTUNE
    if decode_fn is None:
        decode_fn = build_decoder()
    if augment_fn is None:
        augment_fn = build_augmenter()

    options = tf.data.Options()
    try:
        options.deterministic = False
    except Exception:
        pass
    try:
        options.experimental_deterministic = False
    except Exception:
        pass

    def _apply_cache(ds):
        if not cache:
            return ds
        if cache_path is not None:
            return ds.cache(cache_path)
        return ds.cache()

    if labels is None:
        dset = tf.data.Dataset.from_tensor_slices(paths)
        dset = dset.map(decode_fn, num_parallel_calls=AUTO, deterministic=False)
        dset = _apply_cache(dset)
        if shuffle:
            dset = dset.shuffle(shuffle, seed=SEED, reshuffle_each_iteration=True)
        if repeat:
            dset = dset.repeat()
        dset = dset.batch(bsize, drop_remainder=drop_remainder)
        dset = dset.prefetch(AUTO)
        dset = dset.with_options(options)
        return dset

    dset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dset = dset.shuffle(shuffle, seed=SEED, reshuffle_each_iteration=True)

    if augment:

        def _map_img_lbl(path, label):
            img = decode_fn(path)
            img = augment_fn(img)
            return img, label

    else:

        def _map_img_lbl(path, label):
            img = decode_fn(path)
            return img, label

    dset = dset.map(_map_img_lbl, num_parallel_calls=AUTO, deterministic=False)
    dset = _apply_cache(dset)
    if repeat:
        dset = dset.repeat()
    dset = dset.batch(bsize, drop_remainder=drop_remainder)
    dset = dset.prefetch(AUTO)
    dset = dset.with_options(options)
    return dset




## === cell 9
patients = train["PatientID"].values
unique_patients = np.unique(patients)

train_p, val_p = train_test_split(
    unique_patients, test_size=0.2, random_state=SEED, shuffle=True
)
train_mask = np.isin(patients, train_p)
val_mask = np.isin(patients, val_p)

train_paths = train_images[train_mask]
val_paths = train_images[val_mask]
train_labels = labels[train_mask]
val_labels = labels[val_mask]

STEPS_PER_EPOCH = int(np.ceil(len(train_paths) / BATCH_SIZE))
VALIDATION_STEPS = int(np.ceil(len(val_paths) / BATCH_SIZE))
TEST_STEPS = int(np.ceil(len(test_images) / BATCH_SIZE))

print("Train/Val sizes:", len(train_paths), len(val_paths))
print("Steps:", STEPS_PER_EPOCH, VALIDATION_STEPS, "Test steps:", TEST_STEPS)

_shared_decode = build_decoder()
_shared_aug = build_augmenter()

train_ds = build_dataset(
    train_paths,
    train_labels,
    bsize=BATCH_SIZE,
    decode_fn=_shared_decode,
    augment_fn=_shared_aug,
    repeat=True,
    shuffle=2048,
    augment=True,
    drop_remainder=False,
    cache=False,  # must not cache augmented pipeline (would change randomness/semantics)
)

val_ds = build_dataset(
    val_paths,
    val_labels,
    bsize=BATCH_SIZE,
    decode_fn=_shared_decode,
    augment_fn=_shared_aug,
    repeat=False,
    shuffle=False,
    augment=False,
    drop_remainder=False,
    cache=False,
    cache_path=None,
)

test_ds = build_dataset(
    test_images,
    labels=None,
    bsize=BATCH_SIZE,
    decode_fn=_shared_decode,
    augment_fn=_shared_aug,
    repeat=False,
    shuffle=False,  # do not shuffle test: avoids extra work and preserves ordering for submission
    augment=False,
    drop_remainder=False,
    cache=False,
    cache_path=None,
)




## === cell 10
def build_model(
    input_shape=(TARGET_SIZE, TARGET_SIZE, 3), n_classes=len(ALL_LABEL_COLS)
):
    base = Xception(include_top=False, weights="imagenet", input_shape=input_shape)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(n_classes, activation="sigmoid")(x)
    model = models.Model(inputs=base.input, outputs=out)
    return model


model = build_model()

try:
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(
                multi_label=True, num_labels=len(ALL_LABEL_COLS), name="auc"
            )
        ],
        steps_per_execution=16,
    )
except TypeError:
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(
                multi_label=True, num_labels=len(ALL_LABEL_COLS), name="auc"
            )
        ],
    )

model.summary()



## === cell 11
ckpt_path = "best_model.keras"
callbacks = [
    ModelCheckpoint(
        ckpt_path, monitor="val_auc", mode="max", save_best_only=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_auc", mode="max", factor=0.5, patience=1, verbose=1, min_lr=1e-6
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    callbacks=callbacks,
    verbose=1,
)

if os.path.exists(ckpt_path):
    model = tf.keras.models.load_model(ckpt_path)




## === cell 12
def activation_layer_vis(img, activation_layer=0, layers_n=10):
    return None


def all_activations_vis(img, layers_n=10):
    return None




## === cell 13
one_img_ds = None



## === cell 14
print("Visualization dataset ready:", one_img_ds)



## === cell 15
pred = model.predict(test_ds, steps=TEST_STEPS, verbose=1)
pred = np.clip(pred, 0.0, 1.0)

ss.loc[:, ALL_LABEL_COLS] = pred.astype(np.float32)
ss.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", ss.shape)
print(ss.head())



## === cell 16
assert ss.columns.tolist() == ["StudyInstanceUID"] + ALL_LABEL_COLS
assert not ss[ALL_LABEL_COLS].isna().any().any()
print("Submission format OK. File exists:", os.path.exists("submission.csv"))
