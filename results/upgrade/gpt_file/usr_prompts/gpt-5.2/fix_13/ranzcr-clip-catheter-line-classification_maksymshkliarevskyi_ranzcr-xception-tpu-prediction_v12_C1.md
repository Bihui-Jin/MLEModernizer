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
    _cpu = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(_cpu)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass



## === cell 2
WORK_DIR = "../input/ranzcr-clip-catheter-line-classification"
print("WORK_DIR exists:", os.path.exists(WORK_DIR))
print("WORK_DIR sample:", os.listdir(WORK_DIR)[:10])



## === cell 3
train_dir = os.path.join(WORK_DIR, "train")
test_dir = os.path.join(WORK_DIR, "test")
print("Train dir exists:", os.path.exists(train_dir))
print("Test dir exists:", os.path.exists(test_dir))



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

TARGET_SIZE = 299

print(
    "Configured BATCH_SIZE:", BATCH_SIZE, "EPOCHS:", EPOCHS, "TARGET_SIZE:", TARGET_SIZE
)



## === cell 7
from tensorflow.keras.applications.xception import (
    preprocess_input as xception_preprocess,
)


def build_decoder(with_labels=True, target_size=(TARGET_SIZE, TARGET_SIZE), ext="jpg"):
    @tf.function(reduce_retracing=True)
    def _decode(path):
        file_bytes = tf.io.read_file(path)
        if ext == "png":
            img = tf.image.decode_png(file_bytes, channels=3)
        elif ext in ["jpg", "jpeg"]:
            img = tf.image.decode_jpeg(file_bytes, channels=3)
        else:
            raise ValueError("Image extension not supported")

        img = tf.cast(img, tf.float32)
        img = tf.image.resize(
            img, target_size, method=tf.image.ResizeMethod.BILINEAR, antialias=False
        )
        img.set_shape([target_size[0], target_size[1], 3])
        img = xception_preprocess(img)
        return img

    @tf.function(reduce_retracing=True)
    def decode_with_labels(path, label):
        return _decode(path), label

    return decode_with_labels if with_labels else _decode


def build_augmenter(with_labels=True):
    @tf.function(reduce_retracing=True)
    def _seed_from_path(path):
        h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
        return tf.stack([tf.cast(SEED, tf.int32), tf.cast(h, tf.int32)], axis=0)

    @tf.function(reduce_retracing=True)
    def _augment_from_path_and_img(path, img):
        s = _seed_from_path(path)

        img2 = tf.image.stateless_random_flip_left_right(
            img, seed=s + tf.constant([0, 10], tf.int32)
        )
        img2 = tf.image.stateless_random_flip_up_down(
            img2, seed=s + tf.constant([0, 11], tf.int32)
        )

        r = tf.random.stateless_uniform(
            [],
            seed=s + tf.constant([0, 12], tf.int32),
            minval=0,
            maxval=4,
            dtype=tf.int32,
        )
        img2 = tf.image.rot90(img2, k=r)

        u0 = tf.random.stateless_uniform([], seed=s + tf.constant([0, 13], tf.int32))
        img2 = tf.cond(
            u0 >= 0.6,
            lambda: tf.image.stateless_random_saturation(
                img2, lower=0.85, upper=1.15, seed=s + tf.constant([0, 33], tf.int32)
            ),
            lambda: img2,
        )

        u1 = tf.random.stateless_uniform([], seed=s + tf.constant([0, 14], tf.int32))
        img2 = tf.cond(
            u1 >= 0.6,
            lambda: tf.image.stateless_random_contrast(
                img2, lower=0.85, upper=1.15, seed=s + tf.constant([0, 44], tf.int32)
            ),
            lambda: img2,
        )

        u2 = tf.random.stateless_uniform([], seed=s + tf.constant([0, 15], tf.int32))
        img2 = tf.cond(
            u2 >= 0.4,
            lambda: tf.image.stateless_random_brightness(
                img2, max_delta=0.1, seed=s + tf.constant([0, 55], tf.int32)
            ),
            lambda: img2,
        )
        return img2

    @tf.function(reduce_retracing=True)
    def augment_with_labels(path, img, label):
        return _augment_from_path_and_img(path, img), label

    @tf.function(reduce_retracing=True)
    def augment_no_labels(path, img):
        return _augment_from_path_and_img(path, img)

    return augment_with_labels if with_labels else augment_no_labels


def build_dataset(
    paths,
    labels=None,
    bsize=32,
    decode_fn=None,
    augment_fn=None,
    augment=True,
    repeat=True,
    shuffle=1024,
    cache=False,
    cache_path=None,
):
    AUTO = tf.data.AUTOTUNE
    with_labels = labels is not None

    if decode_fn is None:
        decode_fn = build_decoder(with_labels)

    if augment_fn is None:
        augment_fn = build_augmenter(with_labels)

    opts = tf.data.Options()
    opts.experimental_deterministic = False
    try:
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_optimization.map_and_batch_fusion = True
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_optimization.autotune_buffers = True
    except Exception:
        pass

    if with_labels:
        dset = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(opts)
        if shuffle:
            dset = dset.shuffle(shuffle, seed=SEED, reshuffle_each_iteration=True)
        if repeat:
            dset = dset.repeat()

        if augment:

            @tf.function(reduce_retracing=True)
            def _decode_and_augment(p, y):
                img = decode_fn(p, y)[0]
                img, y2 = augment_fn(p, img, y)
                return img, y2

            dset = dset.map(
                _decode_and_augment, num_parallel_calls=AUTO, deterministic=False
            )
        else:
            dset = dset.map(decode_fn, num_parallel_calls=AUTO, deterministic=False)
            if cache:
                dset = dset.cache(cache_path) if cache_path else dset.cache()
    else:
        dset = tf.data.Dataset.from_tensor_slices(paths).with_options(opts)
        if shuffle:
            dset = dset.shuffle(shuffle, seed=SEED, reshuffle_each_iteration=True)
        if repeat:
            dset = dset.repeat()

        if augment:

            @tf.function(reduce_retracing=True)
            def _decode_and_augment_no_label(p):
                img = decode_fn(p)
                img = augment_fn(p, img)
                return img

            dset = dset.map(
                _decode_and_augment_no_label,
                num_parallel_calls=AUTO,
                deterministic=False,
            )
        else:
            dset = dset.map(decode_fn, num_parallel_calls=AUTO, deterministic=False)
            if cache:
                dset = dset.cache(cache_path) if cache_path else dset.cache()

    dset = dset.apply(tf.data.experimental.ignore_errors())
    dset = dset.batch(bsize, drop_remainder=False)
    dset = dset.prefetch(AUTO)
    return dset




## === cell 8
x_train, x_valid, y_train, y_valid = train_test_split(
    train_images.values, labels, test_size=0.15, random_state=SEED, shuffle=True
)

STEPS_PER_EPOCH = int(np.ceil(len(x_train) / BATCH_SIZE))
VALIDATION_STEPS = int(np.ceil(len(x_valid) / BATCH_SIZE))
print("Train size:", len(x_train), "Valid size:", len(x_valid))
print("STEPS_PER_EPOCH:", STEPS_PER_EPOCH, "VALIDATION_STEPS:", VALIDATION_STEPS)

train_ds = build_dataset(
    x_train,
    y_train,
    bsize=BATCH_SIZE,
    repeat=True,
    shuffle=2048,
    augment=True,
    cache=False,
)

valid_cache_path = os.path.join("/kaggle/working", "valid_cache.tfdata")
valid_ds = build_dataset(
    x_valid,
    y_valid,
    bsize=BATCH_SIZE,
    repeat=True,
    shuffle=False,
    augment=False,
    cache=True,
    cache_path=valid_cache_path,
)

test_cache_path = os.path.join("/kaggle/working", "test_cache.tfdata")
test_ds = build_dataset(
    test_images.values,
    labels=None,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=True,
    cache_path=test_cache_path,
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

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    jit_compile=True,
    steps_per_execution=32,
)

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
