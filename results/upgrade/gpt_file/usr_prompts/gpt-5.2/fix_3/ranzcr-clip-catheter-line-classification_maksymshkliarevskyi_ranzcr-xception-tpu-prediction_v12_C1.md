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
import warnings

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam

tf.get_logger().setLevel("ERROR")
print("TensorFlow:", tf.__version__)



## === cell 1
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## === cell 2
WORK_DIR = "../input/ranzcr-clip-catheter-line-classification"
print("WORK_DIR exists:", os.path.exists(WORK_DIR))
print("WORK_DIR sample:", os.listdir(WORK_DIR)[:10])



## === cell 3
print("Train images:", len(os.listdir(os.path.join(WORK_DIR, "train"))))
print("Test images:", len(os.listdir(os.path.join(WORK_DIR, "test"))))



## === cell 4
train = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
ss = pd.read_csv(os.path.join(WORK_DIR, "sample_submission.csv"))

label_cols = [c for c in train.columns if c not in ["StudyInstanceUID", "PatientID"]]
print("Number of label columns:", len(label_cols))
print("Labels:\n", label_cols)

train_images = (WORK_DIR + "/train/" + train["StudyInstanceUID"] + ".jpg").astype(str)
test_images = (WORK_DIR + "/test/" + ss["StudyInstanceUID"] + ".jpg").astype(str)

labels = train[label_cols].values.astype(np.float32)

train_annot = pd.read_csv(os.path.join(WORK_DIR, "train_annotations.csv"))
train.head()



## === cell 5
print("Skipping label count plots for runtime.")



## === cell 6
BATCH_SIZE = 8
EPOCHS = 3  # unchanged
TARGET_SIZE = 750

print(
    "Configured BATCH_SIZE:", BATCH_SIZE, "EPOCHS:", EPOCHS, "TARGET_SIZE:", TARGET_SIZE
)




## === cell 7
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
        img = tf.image.resize(
            img, target_size, method=tf.image.ResizeMethod.BILINEAR, antialias=False
        )
        img.set_shape([target_size[0], target_size[1], 3])
        return img

    def decode_with_labels(path, label):
        return decode(path), label

    return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True):
    def augment(img):
        s = tf.constant([SEED, 0], dtype=tf.int32)
        r0 = tf.random.stateless_uniform(
            [], seed=s, minval=0, maxval=1.0, dtype=tf.float32
        )
        img = tf.cond(r0 > 0.5, lambda: tf.image.flip_left_right(img), lambda: img)

        r1 = tf.random.stateless_uniform(
            [],
            seed=s + tf.constant([0, 1], tf.int32),
            minval=0,
            maxval=1.0,
            dtype=tf.float32,
        )
        img = tf.cond(r1 > 0.5, lambda: tf.image.flip_up_down(img), lambda: img)

        rotate = tf.random.stateless_uniform(
            [],
            seed=s + tf.constant([0, 2], tf.int32),
            minval=0,
            maxval=1.0,
            dtype=tf.float32,
        )
        img = tf.cond(
            rotate > 0.75,
            lambda: tf.image.rot90(img, k=3),
            lambda: tf.cond(
                rotate > 0.5,
                lambda: tf.image.rot90(img, k=2),
                lambda: tf.cond(
                    rotate > 0.25, lambda: tf.image.rot90(img, k=1), lambda: img
                ),
            ),
        )

        saturation = tf.random.stateless_uniform(
            [],
            seed=s + tf.constant([0, 3], tf.int32),
            minval=0,
            maxval=1.0,
            dtype=tf.float32,
        )
        img = tf.cond(
            saturation >= 0.6,
            lambda: tf.image.stateless_random_saturation(
                img, lower=0.85, upper=1.15, seed=s + tf.constant([0, 33], tf.int32)
            ),
            lambda: img,
        )

        contrast = tf.random.stateless_uniform(
            [],
            seed=s + tf.constant([0, 4], tf.int32),
            minval=0,
            maxval=1.0,
            dtype=tf.float32,
        )
        img = tf.cond(
            contrast >= 0.6,
            lambda: tf.image.stateless_random_contrast(
                img, lower=0.85, upper=1.15, seed=s + tf.constant([0, 44], tf.int32)
            ),
            lambda: img,
        )

        brightness = tf.random.stateless_uniform(
            [],
            seed=s + tf.constant([0, 5], tf.int32),
            minval=0,
            maxval=1.0,
            dtype=tf.float32,
        )
        img = tf.cond(
            brightness >= 0.4,
            lambda: tf.image.stateless_random_brightness(
                img, max_delta=0.1, seed=s + tf.constant([0, 55], tf.int32)
            ),
            lambda: img,
        )
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
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)

    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)

    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)

    AUTO = tf.data.AUTOTUNE
    slices = paths if labels is None else (paths, labels)

    opts = tf.data.Options()
    opts.experimental_deterministic = True

    dset = tf.data.Dataset.from_tensor_slices(slices).with_options(opts)
    dset = dset.map(decode_fn, num_parallel_calls=AUTO)

    dset = dset.cache(cache_dir) if cache else dset

    dset = dset.map(augment_fn, num_parallel_calls=AUTO) if augment else dset

    if shuffle:
        dset = dset.shuffle(shuffle, seed=SEED, reshuffle_each_iteration=True)
    dset = dset.repeat() if repeat else dset

    dset = dset.batch(bsize, drop_remainder=False).prefetch(AUTO)
    return dset




## === cell 8
x_train, x_valid, y_train, y_valid = train_test_split(
    train_images.values, labels, test_size=0.15, random_state=SEED, shuffle=True
)

STEPS_PER_EPOCH = int(np.ceil(len(x_train) / BATCH_SIZE))
VALIDATION_STEPS = int(np.ceil(len(x_valid) / BATCH_SIZE))
print("Train size:", len(x_train), "Valid size:", len(x_valid))
print("STEPS_PER_EPOCH:", STEPS_PER_EPOCH, "VALIDATION_STEPS:", VALIDATION_STEPS)

CACHE_BASE = os.path.join("/kaggle/working", "tf_cache_ranzcr")
train_cache = os.path.join(CACHE_BASE, "train.cache")
valid_cache = os.path.join(CACHE_BASE, "valid.cache")
test_cache = os.path.join(CACHE_BASE, "test.cache")

train_ds = build_dataset(
    x_train,
    y_train,
    bsize=BATCH_SIZE,
    repeat=True,
    shuffle=2048,
    augment=True,
    cache=True,
    cache_dir=train_cache,
)
valid_ds = build_dataset(
    x_valid,
    y_valid,
    bsize=BATCH_SIZE,
    repeat=True,
    shuffle=False,
    augment=False,
    cache=True,
    cache_dir=valid_cache,
)

test_ds = build_dataset(
    test_images.values,
    labels=None,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=True,
    cache_dir=test_cache,
)

print(train_ds, valid_ds, test_ds)



## === cell 9
inputs = layers.Input(shape=(TARGET_SIZE, TARGET_SIZE, 3))
base = Xception(
    include_top=False, weights="imagenet", input_tensor=inputs, pooling="avg"
)
x = base.output
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(label_cols), activation="sigmoid")(x)
model = models.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy")

model.summary()



## === cell 10
ckpt_path = "xception_ranzcr.weights.h5"
callbacks = [
    ModelCheckpoint(
        ckpt_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6, verbose=1
    ),
]

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    callbacks=callbacks,
    verbose=2,
)

if os.path.exists(ckpt_path):
    model.load_weights(ckpt_path)



## === cell 11
pred = model.predict(test_ds, verbose=1)
print("Pred shape:", pred.shape)



## === cell 12
sub = ss.copy()

for c in label_cols:
    if c not in sub.columns:
        sub[c] = 0.0

sub = sub[["StudyInstanceUID"] + label_cols]

sub.loc[:, label_cols] = pred.astype(np.float32)

sub[label_cols] = sub[label_cols].clip(0.0, 1.0)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
sub.head()



## === cell 13
assert os.path.exists("submission.csv")
check = pd.read_csv("submission.csv")
assert check.shape[0] == ss.shape[0]
assert check.columns[0] == "StudyInstanceUID"
for c in label_cols:
    assert c in check.columns
print("Submission looks valid.")
