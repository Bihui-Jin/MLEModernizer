# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Classify plant seedlings into their respective species.

## Metric
Micro-averaged F1-score.

## Submission Format
For each `file` in the test set, you must predict a probability for the `species` variable. The file should contain a header and have the following format:

```
file,species
0021e90e4.png,Maize
003d61042.png,Sugar beet
007b3da8b.png,Common wheat
etc.
```

## Dataset
The list of species is as follows:

```
Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet
```

- **train.csv** - the training set, with plant species organized by folder
- **test.csv** - the test set, you need to predict the species of each image
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        input/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        working/
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
```

-> data/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> data/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> working/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

# 5. Target score

0.94962

# 6. Current score

0.06757

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.06757) has done: 'The timeout is dominated by the training input pipeline: it decodes/resizes each image, then applies `ImageDataGenerator` augmentation via `tf.numpy_function` and a Python loop per image, which is extremely slow. I keep the same model, loss, optimizer, callbacks, and epoch/step semantics, but replace the slow Python/Numpy augmentation path with an equivalent pure-TensorFlow augmentation pipeline that matches the same augmentation parameters and still uses `preprocess_input`. I also avoid the redundant `flow_from_directory()` generators for train/val (they were only used to list filepaths/classes) by building the file list once with `tf.keras.utils.image_dataset_from_directory` while preserving the same class order, split, batch sizing, and drop-remainder behavior. Finally, I add `cache()` after decode/resize (pre-augmentation) so expensive disk decode happens once per epoch without changing training semantics (augmentation still changes per epoch).'
- What this solution (achieved 0.06757) has done: 'The timeout is dominated by the `tf.data` augmentation pipeline: it uses multiple `tf.map_fn` passes plus per-image stateless random ops and dynamic cropping, which is much slower than the model itself. I keep the exact same model, loss, optimizer, scheduler, early-stopping, and overall training loop, but rewrite the augmentation to be fully vectorized (no `tf.map_fn`) while preserving the same semantics (flip, brightness scale, zoom via crop+resize, translation, rotation) and determinism. I also remove redundant dataset enumeration work by attaching indices before batching in a cheaper way, and ensure caching/prefetching stay optimal. These changes cut input pipeline overhead dramatically without changing what the model sees (beyond negligible float-level differences), bringing runtime under 600s.'
- What this solution (achieved 0.06757) has done: 'I fix the TensorFlow import crash by pinning a protobuf version that is compatible with TF 2.18 in this Kaggle image, then re-import TensorFlow cleanly. Next I fix the `Dataset.map(_pack)` signature mismatch by unpacking `(idxs, (img, y))` as two positional arguments, which allow `train_ds/val_ds` to be created and training to run. Finally, I keep your model/training logic unchanged and ensure the test dataset loads from the correct directory structure (the competition test folder contains images directly), so predictions align with `sample_submission.csv` and a valid `submission.csv` is written.'
- What this solution (achieved 0.06757) has done: 'I fix the training dataset construction crash by correcting how stateless random flip seeds are provided: `tf.image.stateless_random_flip_*` expects a single seed vector of shape `[2]`, not a per-example `[batch, 2]` seed matrix. To keep your augmentation logic and determinism intact while avoiding Python loops, I replace only the flip ops with equivalent vectorized `tf.where` gating using per-example stateless uniforms, leaving the rest of the augmentation pipeline (brightness/zoom/shift/rotation and `preprocess_input`) unchanged. This allow `train_ds` to be created, training to run, and the script to write a valid `submission.csv`. No model/training hyperparameters or core architecture are changed.'
- What this solution (achieved 0.06757) has done: 'I fix the dataset pipeline crash by removing the dtype-based `tf.cond` router (which tries to create a Tensor from a `tf.DType`) and instead make the augmentation/preprocess functions accept float32, matching what `image_dataset_from_directory` actually yields in TF 2.18. I update `_augment_batch` and `_preprocess_only` input signatures accordingly so both train and test `.map()` calls bind correctly, which unblocks training and inference. These changes preserve your model, optimizer, loss, callbacks, and epoch/step semantics; they only correct dtype handling so the same augmentations and `preprocess_input` run without errors. Finally, I ensure a valid `/kaggle/working/submission.csv` is written using the sample submission’s file order.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
import numpy as np
import pandas as pd

print("Input root exists:", os.path.exists("/kaggle/input"))
print(
    "Train dir exists:",
    os.path.exists("/kaggle/input/plant-seedlings-classification/train"),
)
print(
    "Test dir exists:",
    os.path.exists("/kaggle/input/plant-seedlings-classification/test"),
)



## === cell 1
TRAIN_ROOT = "/kaggle/input/plant-seedlings-classification/train"
print("Train root:", TRAIN_ROOT)
print("Skipping train.csv generation (unused by training/prediction).")



## === cell 2
classes = sorted(
    [d for d in os.listdir(TRAIN_ROOT) if os.path.isdir(os.path.join(TRAIN_ROOT, d))]
)
print(f"Number of classes: {len(classes)}")
print("Classes:", classes)



## === cell 3
print("Skipping plots for runtime.")



## === cell 4
print("Skipping sample image visualization for runtime.")



## === cell 5
print("Skipping image-size histogram for runtime.")



## === cell 6
try:
    import google.protobuf  # noqa: F401
    import protobuf  # type: ignore # noqa: F401
except Exception:
    pass


def _ensure_protobuf_compatible():
    try:
        import google.protobuf as gp

        ver = getattr(gp, "__version__", "")
        print("Detected protobuf version:", ver)
        major = int(ver.split(".")[0]) if ver else 0
        if major >= 5:
            print("Downgrading protobuf to 4.25.3 for TF compatibility...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
            )
            import importlib

            importlib.invalidate_caches()
            for m in list(sys.modules.keys()):
                if m.startswith("google.protobuf") or m == "protobuf":
                    sys.modules.pop(m, None)
    except Exception as e:
        print("[WARN] Could not validate/downgrade protobuf:", repr(e))


_ensure_protobuf_compatible()

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    LearningRateScheduler,
    EarlyStopping,
    ModelCheckpoint,
)
from math import exp

print("TensorFlow:", tf.__version__)
print("Using tf.keras:", tf.keras.__name__)



## === cell 7
file = "/kaggle/working/pretrained_ResNet50V2_vFINAL"
batch_size = 32
seed = 42
val_split = 0.2
image_size = (256, 256)

PROYECT_FOLDER_TRAIN = "/kaggle/input/plant-seedlings-classification/train/"

train_list_ds = tf.keras.utils.image_dataset_from_directory(
    PROYECT_FOLDER_TRAIN,
    labels="inferred",
    label_mode="int",
    class_names=classes,  # keep identical class->index mapping
    color_mode="rgb",
    batch_size=None,
    image_size=image_size,
    shuffle=True,
    seed=seed,
    validation_split=val_split,
    subset="training",
    interpolation="bilinear",
)

val_list_ds = tf.keras.utils.image_dataset_from_directory(
    PROYECT_FOLDER_TRAIN,
    labels="inferred",
    label_mode="int",
    class_names=classes,
    color_mode="rgb",
    batch_size=None,
    image_size=image_size,
    shuffle=True,
    seed=seed,
    validation_split=val_split,
    subset="validation",
    interpolation="bilinear",
)

num_classes = len(classes)
print("Class indices:", {c: i for i, c in enumerate(classes)})
print("num_classes:", num_classes)

train_count = int(tf.data.experimental.cardinality(train_list_ds).numpy())
val_count = int(tf.data.experimental.cardinality(val_list_ds).numpy())

train_steps = train_count // batch_size
val_steps = val_count // batch_size
print("train_count:", train_count, "val_count:", val_count)
print("train_steps:", train_steps, "val_steps:", val_steps)


def _to_onehot(y_int):
    return tf.one_hot(tf.cast(y_int, tf.int32), depth=num_classes, dtype=tf.float32)


_rot_layer = layers.RandomRotation(
    factor=30.0 / 180.0,  # ±30 degrees
    fill_mode="reflect",
    interpolation="bilinear",
    seed=seed,
)


@tf.function(reduce_retracing=True)
def _stateless_uniform_per_example(ex_seeds, minval, maxval, dtype=tf.float32):
    return tf.random.stateless_uniform(
        shape=(tf.shape(ex_seeds)[0],),
        seed=tf.cast(ex_seeds, tf.int64),
        minval=tf.cast(minval, dtype),
        maxval=tf.cast(maxval, dtype),
        dtype=dtype,
    )


@tf.function(reduce_retracing=True)
def _stateless_uniform_int_per_example(ex_seeds, minval, maxval):
    return tf.random.stateless_uniform(
        shape=(tf.shape(ex_seeds)[0],),
        seed=tf.cast(ex_seeds, tf.int64),
        minval=tf.cast(minval, tf.int32),
        maxval=tf.cast(maxval, tf.int32),
        dtype=tf.int32,
    )


_augment_batch = tf.function(
    lambda imgs_f32, y_ints, idxs: (
        (
            lambda imgs_uint8_or_float, y_ints, idxs: (
                (
                    lambda imgs_uint8_or_float, y_ints, idxs: (
                        (
                            lambda imgs, y, ex_seeds: (
                                (
                                    lambda x: (
                                        (
                                            lambda p_lr: (
                                                (
                                                    lambda x: (
                                                        (
                                                            lambda p_ud: (
                                                                (
                                                                    lambda x: (
                                                                        (
                                                                            lambda br: (
                                                                                (
                                                                                    lambda x: (
                                                                                        (
                                                                                            lambda h, w, zoom: (
                                                                                                (
                                                                                                    lambda crop_h, crop_w: (
                                                                                                        (
                                                                                                            lambda max_off_y, max_off_x: (
                                                                                                                (
                                                                                                                    lambda off_y, off_x: (
                                                                                                                        (
                                                                                                                            lambda y1, x1, y2, x2: (
                                                                                                                                (
                                                                                                                                    lambda boxes, box_indices: (
                                                                                                                                        (
                                                                                                                                            lambda x: (
                                                                                                                                                (
                                                                                                                                                    lambda max_dx, max_dy: (
                                                                                                                                                        (
                                                                                                                                                            lambda dx, dy: (
                                                                                                                                                                (
                                                                                                                                                                    lambda pad_x, pad_y: (
                                                                                                                                                                        (
                                                                                                                                                                            lambda pad_x_max, pad_y_max: (
                                                                                                                                                                                (
                                                                                                                                                                                    lambda x_pad: (
                                                                                                                                                                                        (
                                                                                                                                                                                            lambda off_y2, off_x2, Hp, Wp: (
                                                                                                                                                                                                (
                                                                                                                                                                                                    lambda y1p, x1p, y2p, x2p: (
                                                                                                                                                                                                        (
                                                                                                                                                                                                            lambda boxes2: (
                                                                                                                                                                                                                (
                                                                                                                                                                                                                    lambda x: (
                                                                                                                                                                                                                        (
                                                                                                                                                                                                                            lambda x: (
                                                                                                                                                                                                                                (
                                                                                                                                                                                                                                    lambda x: (
                                                                                                                                                                                                                                        preprocess_input(
                                                                                                                                                                                                                                            x
                                                                                                                                                                                                                                        ),
                                                                                                                                                                                                                                        y,
                                                                                                                                                                                                                                    )
                                                                                                                                                                                                                                )(
                                                                                                                                                                                                                                    tf.clip_by_value(
                                                                                                                                                                                                                                        x,
                                                                                                                                                                                                                                        0.0,
                                                                                                                                                                                                                                        1.0,
                                                                                                                                                                                                                                    )
                                                                                                                                                                                                                                    * 255.0
                                                                                                                                                                                                                                )
                                                                                                                                                                                                                            )
                                                                                                                                                                                                                        )(
                                                                                                                                                                                                                            _rot_layer(
                                                                                                                                                                                                                                x,
                                                                                                                                                                                                                                training=True,
                                                                                                                                                                                                                            )
                                                                                                                                                                                                                        )
                                                                                                                                                                                                                    )
                                                                                                                                                                                                                )(
                                                                                                                                                                                                                    tf.image.crop_and_resize(
                                                                                                                                                                                                                        x_pad,
                                                                                                                                                                                                                        boxes2,
                                                                                                                                                                                                                        box_indices,
                                                                                                                                                                                                                        crop_size=image_size,
                                                                                                                                                                                                                        method="bilinear",
                                                                                                                                                                                                                        extrapolation_value=0.0,
                                                                                                                                                                                                                    )
                                                                                                                                                                                                                )
                                                                                                                                                                                                            )
                                                                                                                                                                                                        )(
                                                                                                                                                                                                            tf.stack(
                                                                                                                                                                                                                [
                                                                                                                                                                                                                    y1p,
                                                                                                                                                                                                                    x1p,
                                                                                                                                                                                                                    y2p,
                                                                                                                                                                                                                    x2p,
                                                                                                                                                                                                                ],
                                                                                                                                                                                                                axis=1,
                                                                                                                                                                                                            )
                                                                                                                                                                                                        )
                                                                                                                                                                                                    )
                                                                                                                                                                                                )(
                                                                                                                                                                                                    tf.cast(
                                                                                                                                                                                                        off_y2,
                                                                                                                                                                                                        tf.float32,
                                                                                                                                                                                                    )
                                                                                                                                                                                                    / tf.cast(
                                                                                                                                                                                                        Hp,
                                                                                                                                                                                                        tf.float32,
                                                                                                                                                                                                    ),
                                                                                                                                                                                                    tf.cast(
                                                                                                                                                                                                        off_x2,
                                                                                                                                                                                                        tf.float32,
                                                                                                                                                                                                    )
                                                                                                                                                                                                    / tf.cast(
                                                                                                                                                                                                        Wp,
                                                                                                                                                                                                        tf.float32,
                                                                                                                                                                                                    ),
                                                                                                                                                                                                    tf.cast(
                                                                                                                                                                                                        off_y2
                                                                                                                                                                                                        + image_size[
                                                                                                                                                                                                            0
                                                                                                                                                                                                        ],
                                                                                                                                                                                                        tf.float32,
                                                                                                                                                                                                    )
                                                                                                                                                                                                    / tf.cast(
                                                                                                                                                                                                        Hp,
                                                                                                                                                                                                        tf.float32,
                                                                                                                                                                                                    ),
                                                                                                                                                                                                    tf.cast(
                                                                                                                                                                                                        off_x2
                                                                                                                                                                                                        + image_size[
                                                                                                                                                                                                            1
                                                                                                                                                                                                        ],
                                                                                                                                                                                                        tf.float32,
                                                                                                                                                                                                    )
                                                                                                                                                                                                    / tf.cast(
                                                                                                                                                                                                        Wp,
                                                                                                                                                                                                        tf.float32,
                                                                                                                                                                                                    ),
                                                                                                                                                                                                )
                                                                                                                                                                                            )
                                                                                                                                                                                        )(
                                                                                                                                                                                            pad_y_max
                                                                                                                                                                                            - dy,
                                                                                                                                                                                            pad_x_max
                                                                                                                                                                                            - dx,
                                                                                                                                                                                            tf.shape(
                                                                                                                                                                                                x_pad
                                                                                                                                                                                            )[
                                                                                                                                                                                                1
                                                                                                                                                                                            ],
                                                                                                                                                                                            tf.shape(
                                                                                                                                                                                                x_pad
                                                                                                                                                                                            )[
                                                                                                                                                                                                2
                                                                                                                                                                                            ],
                                                                                                                                                                                        )
                                                                                                                                                                                    )
                                                                                                                                                                                )(
                                                                                                                                                                                    tf.pad(
                                                                                                                                                                                        x,
                                                                                                                                                                                        [
                                                                                                                                                                                            [
                                                                                                                                                                                                0,
                                                                                                                                                                                                0,
                                                                                                                                                                                            ],
                                                                                                                                                                                            [
                                                                                                                                                                                                pad_y_max,
                                                                                                                                                                                                pad_y_max,
                                                                                                                                                                                            ],
                                                                                                                                                                                            [
                                                                                                                                                                                                pad_x_max,
                                                                                                                                                                                                pad_x_max,
                                                                                                                                                                                            ],
                                                                                                                                                                                            [
                                                                                                                                                                                                0,
                                                                                                                                                                                                0,
                                                                                                                                                                                            ],
                                                                                                                                                                                        ],
                                                                                                                                                                                        mode="REFLECT",
                                                                                                                                                                                    )
                                                                                                                                                                                )
                                                                                                                                                                            )
                                                                                                                                                                        )(
                                                                                                                                                                            tf.reduce_max(
                                                                                                                                                                                pad_x
                                                                                                                                                                            ),
                                                                                                                                                                            tf.reduce_max(
                                                                                                                                                                                pad_y
                                                                                                                                                                            ),
                                                                                                                                                                        )
                                                                                                                                                                    )
                                                                                                                                                                )(
                                                                                                                                                                    tf.abs(
                                                                                                                                                                        dx
                                                                                                                                                                    ),
                                                                                                                                                                    tf.abs(
                                                                                                                                                                        dy
                                                                                                                                                                    ),
                                                                                                                                                                )
                                                                                                                                                            )
                                                                                                                                                        )(
                                                                                                                                                            _stateless_uniform_int_per_example(
                                                                                                                                                                ex_seeds
                                                                                                                                                                + tf.constant(
                                                                                                                                                                    [
                                                                                                                                                                        0,
                                                                                                                                                                        14,
                                                                                                                                                                    ],
                                                                                                                                                                    tf.int64,
                                                                                                                                                                ),
                                                                                                                                                                -tf.cast(
                                                                                                                                                                    tf.round(
                                                                                                                                                                        0.2
                                                                                                                                                                        * tf.cast(
                                                                                                                                                                            image_size[
                                                                                                                                                                                1
                                                                                                                                                                            ],
                                                                                                                                                                            tf.float32,
                                                                                                                                                                        )
                                                                                                                                                                    ),
                                                                                                                                                                    tf.int32,
                                                                                                                                                                ),
                                                                                                                                                                tf.cast(
                                                                                                                                                                    tf.round(
                                                                                                                                                                        0.2
                                                                                                                                                                        * tf.cast(
                                                                                                                                                                            image_size[
                                                                                                                                                                                1
                                                                                                                                                                            ],
                                                                                                                                                                            tf.float32,
                                                                                                                                                                        )
                                                                                                                                                                    ),
                                                                                                                                                                    tf.int32,
                                                                                                                                                                )
                                                                                                                                                                + 1,
                                                                                                                                                            ),
                                                                                                                                                            _stateless_uniform_int_per_example(
                                                                                                                                                                ex_seeds
                                                                                                                                                                + tf.constant(
                                                                                                                                                                    [
                                                                                                                                                                        0,
                                                                                                                                                                        15,
                                                                                                                                                                    ],
                                                                                                                                                                    tf.int64,
                                                                                                                                                                ),
                                                                                                                                                                -tf.cast(
                                                                                                                                                                    tf.round(
                                                                                                                                                                        0.2
                                                                                                                                                                        * tf.cast(
                                                                                                                                                                            image_size[
                                                                                                                                                                                0
                                                                                                                                                                            ],
                                                                                                                                                                            tf.float32,
                                                                                                                                                                        )
                                                                                                                                                                    ),
                                                                                                                                                                    tf.int32,
                                                                                                                                                                ),
                                                                                                                                                                tf.cast(
                                                                                                                                                                    tf.round(
                                                                                                                                                                        0.2
                                                                                                                                                                        * tf.cast(
                                                                                                                                                                            image_size[
                                                                                                                                                                                0
                                                                                                                                                                            ],
                                                                                                                                                                            tf.float32,
                                                                                                                                                                        )
                                                                                                                                                                    ),
                                                                                                                                                                    tf.int32,
                                                                                                                                                                )
                                                                                                                                                                + 1,
                                                                                                                                                            ),
                                                                                                                                                        )
                                                                                                                                                    )
                                                                                                                                                )(
                                                                                                                                                    tf.cast(
                                                                                                                                                        tf.round(
                                                                                                                                                            0.2
                                                                                                                                                            * tf.cast(
                                                                                                                                                                image_size[
                                                                                                                                                                    1
                                                                                                                                                                ],
                                                                                                                                                                tf.float32,
                                                                                                                                                            )
                                                                                                                                                        ),
                                                                                                                                                        tf.int32,
                                                                                                                                                    ),
                                                                                                                                                    tf.cast(
                                                                                                                                                        tf.round(
                                                                                                                                                            0.2
                                                                                                                                                            * tf.cast(
                                                                                                                                                                image_size[
                                                                                                                                                                    0
                                                                                                                                                                ],
                                                                                                                                                                tf.float32,
                                                                                                                                                            )
                                                                                                                                                        ),
                                                                                                                                                        tf.int32,
                                                                                                                                                    ),
                                                                                                                                                )
                                                                                                                                            )
                                                                                                                                        )(
                                                                                                                                            tf.image.crop_and_resize(
                                                                                                                                                x,
                                                                                                                                                boxes,
                                                                                                                                                tf.range(
                                                                                                                                                    tf.shape(
                                                                                                                                                        x
                                                                                                                                                    )[
                                                                                                                                                        0
                                                                                                                                                    ],
                                                                                                                                                    dtype=tf.int32,
                                                                                                                                                ),
                                                                                                                                                crop_size=image_size,
                                                                                                                                                method="bilinear",
                                                                                                                                                extrapolation_value=0.0,
                                                                                                                                            )
                                                                                                                                        )
                                                                                                                                    )
                                                                                                                                )(
                                                                                                                                    boxes,
                                                                                                                                    tf.range(
                                                                                                                                        tf.shape(
                                                                                                                                            x
                                                                                                                                        )[
                                                                                                                                            0
                                                                                                                                        ],
                                                                                                                                        dtype=tf.int32,
                                                                                                                                    ),
                                                                                                                                )
                                                                                                                            )
                                                                                                                        )(
                                                                                                                            tf.cast(
                                                                                                                                off_y,
                                                                                                                                tf.float32,
                                                                                                                            )
                                                                                                                            / tf.cast(
                                                                                                                                h,
                                                                                                                                tf.float32,
                                                                                                                            ),
                                                                                                                            tf.cast(
                                                                                                                                off_x,
                                                                                                                                tf.float32,
                                                                                                                            )
                                                                                                                            / tf.cast(
                                                                                                                                w,
                                                                                                                                tf.float32,
                                                                                                                            ),
                                                                                                                            tf.cast(
                                                                                                                                off_y
                                                                                                                                + crop_h,
                                                                                                                                tf.float32,
                                                                                                                            )
                                                                                                                            / tf.cast(
                                                                                                                                h,
                                                                                                                                tf.float32,
                                                                                                                            ),
                                                                                                                            tf.cast(
                                                                                                                                off_x
                                                                                                                                + crop_w,
                                                                                                                                tf.float32,
                                                                                                                            )
                                                                                                                            / tf.cast(
                                                                                                                                w,
                                                                                                                                tf.float32,
                                                                                                                            ),
                                                                                                                        )
                                                                                                                    )
                                                                                                                )(
                                                                                                                    tf.minimum(
                                                                                                                        _stateless_uniform_int_per_example(
                                                                                                                            ex_seeds
                                                                                                                            + tf.constant(
                                                                                                                                [
                                                                                                                                    0,
                                                                                                                                    4,
                                                                                                                                ],
                                                                                                                                tf.int64,
                                                                                                                            ),
                                                                                                                            0,
                                                                                                                            tf.reduce_max(
                                                                                                                                tf.maximum(
                                                                                                                                    h
                                                                                                                                    - crop_h,
                                                                                                                                    0,
                                                                                                                                )
                                                                                                                            )
                                                                                                                            + 1,
                                                                                                                        ),
                                                                                                                        tf.maximum(
                                                                                                                            h
                                                                                                                            - crop_h,
                                                                                                                            0,
                                                                                                                        ),
                                                                                                                    ),
                                                                                                                    tf.minimum(
                                                                                                                        _stateless_uniform_int_per_example(
                                                                                                                            ex_seeds
                                                                                                                            + tf.constant(
                                                                                                                                [
                                                                                                                                    0,
                                                                                                                                    5,
                                                                                                                                ],
                                                                                                                                tf.int64,
                                                                                                                            ),
                                                                                                                            0,
                                                                                                                            tf.reduce_max(
                                                                                                                                tf.maximum(
                                                                                                                                    w
                                                                                                                                    - crop_w,
                                                                                                                                    0,
                                                                                                                                )
                                                                                                                            )
                                                                                                                            + 1,
                                                                                                                        ),
                                                                                                                        tf.maximum(
                                                                                                                            w
                                                                                                                            - crop_w,
                                                                                                                            0,
                                                                                                                        ),
                                                                                                                    ),
                                                                                                                )
                                                                                                            )
                                                                                                        )(
                                                                                                            h
                                                                                                            - crop_h,
                                                                                                            w
                                                                                                            - crop_w,
                                                                                                        )
                                                                                                    )
                                                                                                )(
                                                                                                    tf.clip_by_value(
                                                                                                        tf.cast(
                                                                                                            tf.cast(
                                                                                                                h,
                                                                                                                tf.float32,
                                                                                                            )
                                                                                                            / zoom,
                                                                                                            tf.int32,
                                                                                                        ),
                                                                                                        1,
                                                                                                        h,
                                                                                                    ),
                                                                                                    tf.clip_by_value(
                                                                                                        tf.cast(
                                                                                                            tf.cast(
                                                                                                                w,
                                                                                                                tf.float32,
                                                                                                            )
                                                                                                            / zoom,
                                                                                                            tf.int32,
                                                                                                        ),
                                                                                                        1,
                                                                                                        w,
                                                                                                    ),
                                                                                                )
                                                                                            )
                                                                                        )(
                                                                                            tf.shape(
                                                                                                x
                                                                                            )[
                                                                                                1
                                                                                            ],
                                                                                            tf.shape(
                                                                                                x
                                                                                            )[
                                                                                                2
                                                                                            ],
                                                                                            _stateless_uniform_per_example(
                                                                                                ex_seeds
                                                                                                + tf.constant(
                                                                                                    [
                                                                                                        0,
                                                                                                        13,
                                                                                                    ],
                                                                                                    tf.int64,
                                                                                                ),
                                                                                                0.8,
                                                                                                1.2,
                                                                                                dtype=tf.float32,
                                                                                            ),
                                                                                        )
                                                                                    )
                                                                                )(
                                                                                    tf.clip_by_value(
                                                                                        x
                                                                                        * br[
                                                                                            :,
                                                                                            None,
                                                                                            None,
                                                                                            None,
                                                                                        ],
                                                                                        0.0,
                                                                                        1.0,
                                                                                    )
                                                                                )
                                                                            )
                                                                        )(
                                                                            _stateless_uniform_per_example(
                                                                                ex_seeds
                                                                                + tf.constant(
                                                                                    [
                                                                                        0,
                                                                                        12,
                                                                                    ],
                                                                                    tf.int64,
                                                                                ),
                                                                                0.7,
                                                                                1.3,
                                                                                dtype=tf.float32,
                                                                            )
                                                                        )
                                                                    )
                                                                )(
                                                                    tf.where(
                                                                        p_ud[
                                                                            :,
                                                                            None,
                                                                            None,
                                                                            None,
                                                                        ]
                                                                        < 0.5,
                                                                        tf.image.flip_up_down(
                                                                            x
                                                                        ),
                                                                        x,
                                                                    )
                                                                )
                                                            )
                                                        )(
                                                            _stateless_uniform_per_example(
                                                                ex_seeds
                                                                + tf.constant(
                                                                    [0, 11], tf.int64
                                                                ),
                                                                0.0,
                                                                1.0,
                                                            )
                                                        )
                                                    )
                                                )(
                                                    tf.where(
                                                        p_lr[:, None, None, None] < 0.5,
                                                        tf.image.flip_left_right(x),
                                                        x,
                                                    )
                                                )
                                            )
                                        )(
                                            _stateless_uniform_per_example(
                                                ex_seeds
                                                + tf.constant([0, 10], tf.int64),
                                                0.0,
                                                1.0,
                                            )
                                        )
                                    )
                                )(tf.clip_by_value(imgs / 255.0, 0.0, 1.0))
                            )
                        )(
                            tf.cast(imgs_uint8_or_float, tf.float32),
                            _to_onehot(y_ints),
                            tf.stack(
                                [
                                    tf.fill(tf.shape(idxs), tf.cast(seed, tf.int64)),
                                    tf.cast(idxs, tf.int64),
                                ],
                                axis=1,
                            ),
                        )
                    )
                )(imgs_uint8_or_float, y_ints, idxs)
            )
        )(imgs_f32, y_ints, idxs)
    ),
    input_signature=[
        tf.TensorSpec(shape=(None, image_size[0], image_size[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
        tf.TensorSpec(shape=(None,), dtype=tf.int64),
    ],
    reduce_retracing=True,
)


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, image_size[0], image_size[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    ],
    reduce_retracing=True,
)
def _preprocess_val_batch(imgs, y_ints):
    y = _to_onehot(y_ints)
    x = preprocess_input(tf.cast(imgs, tf.float32))
    return x, y


def _make_train_ds(ds_unbatched):
    ds = ds_unbatched.cache()  # caches deterministic decode/resize output
    ds = ds.enumerate()  # (idx, (img, y))
    ds = ds.batch(batch_size, drop_remainder=True)

    def _pack(idxs, xy):
        imgs, y_ints = xy
        return imgs, y_ints, tf.cast(idxs, tf.int64)

    ds = ds.map(_pack, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

    ds = ds.map(_augment_batch, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

    try:
        gpus = tf.config.list_logical_devices("GPU")
        if gpus:
            ds = ds.apply(tf.data.experimental.copy_to_device("/GPU:0"))
            ds = ds.prefetch(tf.data.AUTOTUNE)
        else:
            ds = ds.prefetch(tf.data.AUTOTUNE)
    except Exception:
        ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def _make_val_ds(ds_unbatched):
    ds = ds_unbatched.cache()
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(
        _preprocess_val_batch, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    )

    try:
        gpus = tf.config.list_logical_devices("GPU")
        if gpus:
            ds = ds.apply(tf.data.experimental.copy_to_device("/GPU:0"))
            ds = ds.prefetch(tf.data.AUTOTUNE)
        else:
            ds = ds.prefetch(tf.data.AUTOTUNE)
    except Exception:
        ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


train_ds = _make_train_ds(train_list_ds)
val_ds = _make_val_ds(val_list_ds)

print("Train/Val datasets ready.")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4040939207.py in <cell line: 0>()
    733 
    734 
--> 735 train_ds = _make_train_ds(train_list_ds)
    736 val_ds = _make_val_ds(val_list_ds)
    737 

/tmp/ipykernel_11/4040939207.py in _make_train_ds(ds_unbatched)
    698 
    699     # Bug fix: always use float32 augmentation; avoids dtype-router crash.
--> 700     ds = ds.map(_augment_batch, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    701 
    702     try:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in map(self, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
   2339     from tensorflow.python.data.ops import map_op
   2340 
-> 2341     return map_op._map_v2(
   2342         self,
   2343         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in _map_v2(input_dataset, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
     55           num_parallel_calls,
     56       )
---> 57     return _ParallelMapDataset(
     58         input_dataset,
     59         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in __init__(self, input_dataset, map_func, num_parallel_calls, deterministic, use_inter_op_parallelism, preserve_cardinality, use_legacy_function, use_unbounded_threadpool, name)
    200     self._input_dataset = input_dataset
    201     self._use_inter_op_parallelism = use_inter_op_parallelism
--> 202     self._map_func = structured_function.StructuredFunctionWrapper(
    203         map_func,
    204         self._transformation_name(),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in __init__(self, func, transformation_name, dataset, input_classes, input_shapes, input_types, input_structure, add_to_graph, use_legacy_function, defun_kwargs)
    263         fn_factory = trace_tf_function(defun_kwargs)
    264 
--> 265     self._function = fn_factory()
    266     # There is no graph to add in eager mode.
    267     add_to_graph &= not context.executing_eagerly()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in get_concrete_function(self, *args, **kwargs)
   1249   def get_concrete_function(self, *args, **kwargs):
   1250     # Implements PolymorphicFunction.get_concrete_function.
-> 1251     concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
   1252     concrete._garbage_collector.release()  # pylint: disable=protected-access
   1253     return concrete

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _get_concrete_function_garbage_collected(self, *args, **kwargs)
   1219       if self._variable_creation_config is None:
   1220         initializers = []
-> 1221         self._initialize(args, kwargs, add_initializers_to=initializers)
   1222         self._initialize_uninitialized_variables(initializers)
   1223 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _initialize(self, args, kwds, add_initializers_to)
    694     )
    695     # Force the definition of the function for these arguments
--> 696     self._concrete_variable_creation_fn = tracing_compilation.trace_function(
    697         args, kwds, self._variable_creation_config
    698     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in trace_function(args, kwargs, tracing_options)
    176       kwargs = {}
    177 
--> 178     concrete_function = _maybe_define_function(
    179         args, kwargs, tracing_options
    180     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _maybe_define_function(args, kwargs, tracing_options)
    281         else:
    282           target_func_type = lookup_func_type
--> 283         concrete_function = _create_concrete_function(
    284             target_func_type, lookup_func_context, func_graph, tracing_options
    285         )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _create_concrete_function(function_type, type_context, func_graph, tracing_options)
    308       attributes_lib.DISABLE_ACD, False
    309   )
--> 310   traced_func_graph = func_graph_module.func_graph_from_py_func(
    311       tracing_options.name,
    312       tracing_options.python_function,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, create_placeholders)
   1057 
   1058     _, original_func = tf_decorator.unwrap(python_func)
-> 1059     func_outputs = python_func(*func_args, **func_kwargs)
   1060 
   1061     # invariant: `func_outputs` contains only Tensors, CompositeTensors,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in wrapped_fn(*args, **kwds)
    597         # the function a weak reference to itself to avoid a reference cycle.
    598         with OptionalXlaContext(compile_with_xla):
--> 599           out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    600         return out
    601 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapped_fn(*args)
    229       # Note: wrapper_helper will apply autograph based on context.
    230       def wrapped_fn(*args):  # pylint: disable=missing-docstring
--> 231         ret = wrapper_helper(*args)
    232         ret = structure.to_tensor_list(self._output_structure, ret)
    233         return [ops.convert_to_tensor(t) for t in ret]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapper_helper(*args)
    159       if not _should_unpack(nested_args):
    160         nested_args = (nested_args,)
--> 161       ret = autograph.tf_convert(self._func, ag_ctx)(*nested_args)
    162       ret = variable_utils.convert_variables_to_tensors(ret)
    163       if _should_pack(ret):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/tmp/__autograph_generated_fileroe7juud.py in <lambda>(imgs_f32, y_ints, idxs)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda imgs_f32, y_ints, idxs: ag__.with_function_scope(lambda lscope: ag__.converted_call(ag__.autograph_artifact(lambda imgs_uint8_or_float, y_ints, idxs: ag__.converted_call(ag__.autograph_artifact(lambda imgs_uint8_or_float, y_ints, idxs: ag__.converted_call(ag__.autograph_artifact(lambda imgs, y, ex_seeds: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda p_lr: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda p_ud: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda br: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda h, w, zoom: ag__.converted_call(ag__.autograph_artifact(lambda crop_h, crop_w: ag__.converted_call(ag__.autograph_artifact(lambda max_off_y, max_off_x: ag__.converted_call(ag__.autograph_artifact(lambda off_y, off_x: ag__.converted_call(ag__.autograph_artifact(lambda y1, x1, y2, x2: ag__.converted_call(ag__.autograph_artifact(lambda boxes, box_indices: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda max_dx, max_dy: ag__.converted_call(ag__.autograph_artifact(lambda dx, dy: ag__.converted_call(ag__.autograph_artifact(lambda pad_x, pad_y: ag__.converted_call(ag__.autograph_artifact(lambda pad_x_max, pad_y_max: ag__.converted_call(ag__.autograph_artifact(lambda x_pad: ag__.converted_call(ag__.autograph_artifact(lambda off_y2, off_x2, Hp, Wp: ag__.converted_call(ag__.autograph_artifact(lambda y1p, x1p, y2p, x2p: ag__.converted_call(ag__.autograph_artifact(lambda boxes2: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda x: (ag__.converted_call(preprocess_input, (x,), None, lscope), y)), (ag__.converted_call(tf.clip_by_value, (x, 0.0, 1.0), None, lscope) * 255.0,), None, lscope)), (ag__.converted_call(_rot_layer, (x,), dict(training=True), lscope),), None, lscope)), (ag__.converted_call(tf.image.crop_and_resize, (x_pad, boxes2, box_indices), dict(crop_size=image_size, method='bilinear', extrapolation_value=0.0), lscope),), None, lscope)), (ag__.converted_call(tf.stack, ([y1p, x1p, y2p, x2p],), dict(axis=1), lscope),), None, lscope)), (ag__.converted_call(tf.cast, (off_y2, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (Hp, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_x2, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (Wp, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_y2 + image_size[0], tf.float32), None, lscope) / ag__.converted_call(tf.cast, (Hp, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_x2 + image_size[1], tf.float32), None, lscope) / ag__.converted_call(tf.cast, (Wp, tf.float32), None, lscope)), None, lscope)), (pad_y_max - dy, pad_x_max - dx, ag__.converted_call(tf.shape, (x_pad,), None, lscope)[1], ag__.converted_call(tf.shape, (x_pad,), None, lscope)[2]), None, lscope)), (ag__.converted_call(tf.pad, (x, [[0, 0], [pad_y_max, pad_y_max], [pad_x_max, pad_x_max], [0, 0]]), dict(mode='REFLECT'), lscope),), None, lscope)), (ag__.converted_call(tf.reduce_max, (pad_x,), None, lscope), ag__.converted_call(tf.reduce_max, (pad_y,), None, lscope)), None, lscope)), (ag__.converted_call(tf.abs, (dx,), None, lscope), ag__.converted_call(tf.abs, (dy,), None, lscope)), None, lscope)), (ag__.converted_call(_stateless_uniform_int_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 14], tf.int64), None, lscope), -ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[1], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope), ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[1], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope) + 1), None, lscope), ag__.converted_call(_stateless_uniform_int_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 15], tf.int64), None, lscope), -ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[0], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope), ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[0], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope) + 1), None, lscope)), None, lscope)), (ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[1], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope), ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[0], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope)), None, lscope)), (ag__.converted_call(tf.image.crop_and_resize, (x, boxes, ag__.converted_call(tf.range, (ag__.converted_call(tf.shape, (x,), None, lscope)[0],), dict(dtype=tf.int32), lscope)), dict(crop_size=image_size, method='bilinear', extrapolation_value=0.0), lscope),), None, lscope)), (boxes, ag__.converted_call(tf.range, (ag__.converted_call(tf.shape, (x,), None, lscope)[0],), dict(dtype=tf.int32), lscope)), None, lscope)), (ag__.converted_call(tf.cast, (off_y, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (h, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_x, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (w, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_y + crop_h, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (h, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_x + crop_w, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (w, tf.float32), None, lscope)), None, lscope)), (ag__.converted_call(tf.minimum, (ag__.converted_call(_stateless_uniform_int_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 4], tf.int64), None, lscope), 0, ag__.converted_call(tf.reduce_max, (ag__.converted_call(tf.maximum, (h - crop_h, 0), None, lscope),), None, lscope) + 1), None, lscope), ag__.converted_call(tf.maximum, (h - crop_h, 0), None, lscope)), None, lscope), ag__.converted_call(tf.minimum, (ag__.converted_call(_stateless_uniform_int_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 5], tf.int64), None, lscope), 0, ag__.converted_call(tf.reduce_max, (ag__.converted_call(tf.maximum, (w - crop_w, 0), None, lscope),), None, lscope) + 1), None, lscope), ag__.converted_call(tf.maximum, (w - crop_w, 0), None, lscope)), None, lscope)), None, lscope)), (h - crop_h, w - crop_w), None, lscope)), (ag__.converted_call(tf.clip_by_value, (ag__.converted_call(tf.cast, (ag__.converted_call(tf.cast, (h, tf.float32), None, lscope) / zoom, tf.int32), None, lscope), 1, h), None, lscope), ag__.converted_call(tf.clip_by_value, (ag__.converted_call(tf.cast, (ag__.converted_call(tf.cast, (w, tf.float32), None, lscope) / zoom, tf.int32), None, lscope), 1, w), None, lscope)), None, lscope)), (ag__.converted_call(tf.shape, (x,), None, lscope)[1], ag__.converted_call(tf.shape, (x,), None, lscope)[2], ag__.converted_call(_stateless_uniform_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 13], tf.int64), None, lscope), 0.8, 1.2), dict(dtype=tf.float32), lscope)), None, lscope)), (ag__.converted_call(tf.clip_by_value, (x * br[:, None, None, None], 0.0, 1.0), None, lscope),), None, lscope)), (ag__.converted_call(_stateless_uniform_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 12], tf.int64), None, lscope), 0.7, 1.3), dict(dtype=tf.float32), lscope),), None, lscope)), (ag__.converted_call(tf.where, (p_ud[:, None, None, None] < 0.5, ag__.converted_call(tf.image.flip_up_down, (x,), None, lscope), x), None, lscope),), None, lscope)), (ag__.converted_call(_stateless_uniform_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 11], tf.int64), None, lscope), 0.0, 1.0), None, lscope),), None, lscope)), (ag__.converted_call(tf.where, (p_lr[:, None, None, None] < 0.5, ag__.converted_call(tf.image.flip_left_right, (x,), None, lscope), x), None, lscope),), None, lscope)), (ag__.converted_call(_stateless_uniform_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 10], tf.int64), None, lscope), 0.0, 1.0), None, lscope),), None, lscope)), (ag__.converted_call(tf.clip_by_value, (imgs / 255.0, 0.0, 1.0), None, lscope),), None, lscope)), (ag__.converted_call(tf.cast, (imgs_uint8_or_float, tf.float32), None, lscope), ag__.converted_call(_to_onehot, (y_ints,), None, lscope), ag__.converted_call(tf.stack, ([ag__.converted_call(tf.fill, (ag__.converted_call(tf.shape, (idxs,), None, lscope), ag__.converted_call(tf.cast, (seed, tf.int64), None, lscope)), None, lscope), ag__.converted_call(tf.cast, (idxs, tf.int64), None, lscope)],), dict(axis=1), lscope)), None, lscope)), (imgs_uint8_or_float, y_ints, idxs), None, lscope)), (imgs_f32, y_ints, idxs), None, lscope), 'lscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True))
      6         return tf__lam
      7     return inner_factory

/tmp/__autograph_generated_fileroe7juud.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda imgs_f32, y_ints, idxs: ag__.with_function_scope(lambda lscope: ag__.converted_call(ag__.autograph_artifact(lambda imgs_uint8_or_float, y_ints, idxs: ag__.converted_call(ag__.autograph_artifact(lambda imgs_uint8_or_float, y_ints, idxs: ag__.converted_call(ag__.autograph_artifact(lambda imgs, y, ex_seeds: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda p_lr: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda p_ud: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda br: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda h, w, zoom: ag__.converted_call(ag__.autograph_artifact(lambda crop_h, crop_w: ag__.converted_call(ag__.autograph_artifact(lambda max_off_y, max_off_x: ag__.converted_call(ag__.autograph_artifact(lambda off_y, off_x: ag__.converted_call(ag__.autograph_artifact(lambda y1, x1, y2, x2: ag__.converted_call(ag__.autograph_artifact(lambda boxes, box_indices: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda max_dx, max_dy: ag__.converted_call(ag__.autograph_artifact(lambda dx, dy: ag__.converted_call(ag__.autograph_artifact(lambda pad_x, pad_y: ag__.converted_call(ag__.autograph_artifact(lambda pad_x_max, pad_y_max: ag__.converted_call(ag__.autograph_artifact(lambda x_pad: ag__.converted_call(ag__.autograph_artifact(lambda off_y2, off_x2, Hp, Wp: ag__.converted_call(ag__.autograph_artifact(lambda y1p, x1p, y2p, x2p: ag__.converted_call(ag__.autograph_artifact(lambda boxes2: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda x: (ag__.converted_call(preprocess_input, (x,), None, lscope), y)), (ag__.converted_call(tf.clip_by_value, (x, 0.0, 1.0), None, lscope) * 255.0,), None, lscope)), (ag__.converted_call(_rot_layer, (x,), dict(training=True), lscope),), None, lscope)), (ag__.converted_call(tf.image.crop_and_resize, (x_pad, boxes2, box_indices), dict(crop_size=image_size, method='bilinear', extrapolation_value=0.0), lscope),), None, lscope)), (ag__.converted_call(tf.stack, ([y1p, x1p, y2p, x2p],), dict(axis=1), lscope),), None, lscope)), (ag__.converted_call(tf.cast, (off_y2, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (Hp, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_x2, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (Wp, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_y2 + image_size[0], tf.float32), None, lscope) / ag__.converted_call(tf.cast, (Hp, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_x2 + image_size[1], tf.float32), None, lscope) / ag__.converted_call(tf.cast, (Wp, tf.float32), None, lscope)), None, lscope)), (pad_y_max - dy, pad_x_max - dx, ag__.converted_call(tf.shape, (x_pad,), None, lscope)[1], ag__.converted_call(tf.shape, (x_pad,), None, lscope)[2]), None, lscope)), (ag__.converted_call(tf.pad, (x, [[0, 0], [pad_y_max, pad_y_max], [pad_x_max, pad_x_max], [0, 0]]), dict(mode='REFLECT'), lscope),), None, lscope)), (ag__.converted_call(tf.reduce_max, (pad_x,), None, lscope), ag__.converted_call(tf.reduce_max, (pad_y,), None, lscope)), None, lscope)), (ag__.converted_call(tf.abs, (dx,), None, lscope), ag__.converted_call(tf.abs, (dy,), None, lscope)), None, lscope)), (ag__.converted_call(_stateless_uniform_int_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 14], tf.int64), None, lscope), -ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[1], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope), ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[1], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope) + 1), None, lscope), ag__.converted_call(_stateless_uniform_int_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 15], tf.int64), None, lscope), -ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[0], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope), ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[0], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope) + 1), None, lscope)), None, lscope)), (ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[1], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope), ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[0], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope)), None, lscope)), (ag__.converted_call(tf.image.crop_and_resize, (x, boxes, ag__.converted_call(tf.range, (ag__.converted_call(tf.shape, (x,), None, lscope)[0],), dict(dtype=tf.int32), lscope)), dict(crop_size=image_size, method='bilinear', extrapolation_value=0.0), lscope),), None, lscope)), (boxes, ag__.converted_call(tf.range, (ag__.converted_call(tf.shape, (x,), None, lscope)[0],), dict(dtype=tf.int32), lscope)), None, lscope)), (ag__.converted_call(tf.cast, (off_y, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (h, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_x, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (w, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_y + crop_h, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (h, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_x + crop_w, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (w, tf.float32), None, lscope)), None, lscope)), (ag__.converted_call(tf.minimum, (ag__.converted_call(_stateless_uniform_int_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 4], tf.int64), None, lscope), 0, ag__.converted_call(tf.reduce_max, (ag__.converted_call(tf.maximum, (h - crop_h, 0), None, lscope),), None, lscope) + 1), None, lscope), ag__.converted_call(tf.maximum, (h - crop_h, 0), None, lscope)), None, lscope), ag__.converted_call(tf.minimum, (ag__.converted_call(_stateless_uniform_int_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 5], tf.int64), None, lscope), 0, ag__.converted_call(tf.reduce_max, (ag__.converted_call(tf.maximum, (w - crop_w, 0), None, lscope),), None, lscope) + 1), None, lscope), ag__.converted_call(tf.maximum, (w - crop_w, 0), None, lscope)), None, lscope)), None, lscope)), (h - crop_h, w - crop_w), None, lscope)), (ag__.converted_call(tf.clip_by_value, (ag__.converted_call(tf.cast, (ag__.converted_call(tf.cast, (h, tf.float32), None, lscope) / zoom, tf.int32), None, lscope), 1, h), None, lscope), ag__.converted_call(tf.clip_by_value, (ag__.converted_call(tf.cast, (ag__.converted_call(tf.cast, (w, tf.float32), None, lscope) / zoom, tf.int32), None, lscope), 1, w), None, lscope)), None, lscope)), (ag__.converted_call(tf.shape, (x,), None, lscope)[1], ag__.converted_call(tf.shape, (x,), None, lscope)[2], ag__.converted_call(_stateless_uniform_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 13], tf.int64), None, lscope), 0.8, 1.2), dict(dtype=tf.float32), lscope)), None, lscope)), (ag__.converted_call(tf.clip_by_value, (x * br[:, None, None, None], 0.0, 1.0), None, lscope),), None, lscope)), (ag__.converted_call(_stateless_uniform_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 12], tf.int64), None, lscope), 0.7, 1.3), dict(dtype=tf.float32), lscope),), None, lscope)), (ag__.converted_call(tf.where, (p_ud[:, None, None, None] < 0.5, ag__.converted_call(tf.image.flip_up_down, (x,), None, lscope), x), None, lscope),), None, lscope)), (ag__.converted_call(_stateless_uniform_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 11], tf.int64), None, lscope), 0.0, 1.0), None, lscope),), None, lscope)), (ag__.converted_call(tf.where, (p_lr[:, None, None, None] < 0.5, ag__.converted_call(tf.image.flip_left_right, (x,), None, lscope), x), None, lscope),), None, lscope)), (ag__.converted_call(_stateless_uniform_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 10], tf.int64), None, lscope), 0.0, 1.0), None, lscope),), None, lscope)), (ag__.converted_call(tf.clip_by_value, (imgs / 255.0, 0.0, 1.0), None, lscope),), None, lscope)), (ag__.converted_call(tf.cast, (imgs_uint8_or_float, tf.float32), None, lscope), ag__.converted_call(_to_onehot, (y_ints,), None, lscope), ag__.converted_call(tf.stack, ([ag__.converted_call(tf.fill, (ag__.converted_call(tf.shape, (idxs,), None, lscope), ag__.converted_call(tf.cast, (seed, tf.int64), None, lscope)), None, lscope), ag__.converted_call(tf.cast, (idxs, tf.int64), None, lscope)],), dict(axis=1), lscope)), None, lscope)), (imgs_uint8_or_float, y_ints, idxs), None, lscope)), (imgs_f32, y_ints, idxs), None, lscope), 'lscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True))
      6         return tf__lam
      7     return inner_factory

/tmp/__autograph_generated_fileroe7juud.py in <lambda>(imgs_uint8_or_float, y_ints, idxs)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda imgs_f32, y_ints, idxs: ag__.with_function_scope(lambda lscope: ag__.converted_call(ag__.autograph_artifact(lambda imgs_uint8_or_float, y_ints, idxs: ag__.converted_call(ag__.autograph_artifact(lambda imgs_uint8_or_float, y_ints, idxs: ag__.converted_call(ag__.autograph_artifact(lambda imgs, y, ex_seeds: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda p_lr: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda p_ud: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda br: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda h, w, zoom: ag__.converted_call(ag__.autograph_artifact(lambda crop_h, crop_w: ag__.converted_call(ag__.autograph_artifact(lambda max_off_y, max_off_x: ag__.converted_call(ag__.autograph_artifact(lambda off_y, off_x: ag__.converted_call(ag__.autograph_artifact(lambda y1, x1, y2, x2: ag__.converted_call(ag__.autograph_artifact(lambda boxes, box_indices: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda max_dx, max_dy: ag__.converted_call(ag__.autograph_artifact(lambda dx, dy: ag__.converted_call(ag__.autograph_artifact(lambda pad_x, pad_y: ag__.converted_call(ag__.autograph_artifact(lambda pad_x_max, pad_y_max: ag__.converted_call(ag__.autograph_artifact(lambda x_pad: ag__.converted_call(ag__.autograph_artifact(lambda off_y2, off_x2, Hp, Wp: ag__.converted_call(ag__.autograph_artifact(lambda y1p, x1p, y2p, x2p: ag__.converted_call(ag__.autograph_artifact(lambda boxes2: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda x: (ag__.converted_call(preprocess_input, (x,), None, lscope), y)), (ag__.converted_call(tf.clip_by_value, (x, 0.0, 1.0), None, lscope) * 255.0,), None, lscope)), (ag__.converted_call(_rot_layer, (x,), dict(training=True), lscope),), None, lscope)), (ag__.converted_call(tf.image.crop_and_resize, (x_pad, boxes2, box_indices), dict(crop_size=image_size, method='bilinear', extrapolation_value=0.0), lscope),), None, lscope)), (ag__.converted_call(tf.stack, ([y1p, x1p, y2p, x2p],), dict(axis=1), lscope),), None, lscope)), (ag__.converted_call(tf.cast, (off_y2, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (Hp, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_x2, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (Wp, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_y2 + image_size[0], tf.float32), None, lscope) / ag__.converted_call(tf.cast, (Hp, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_x2 + image_size[1], tf.float32), None, lscope) / ag__.converted_call(tf.cast, (Wp, tf.float32), None, lscope)), None, lscope)), (pad_y_max - dy, pad_x_max - dx, ag__.converted_call(tf.shape, (x_pad,), None, lscope)[1], ag__.converted_call(tf.shape, (x_pad,), None, lscope)[2]), None, lscope)), (ag__.converted_call(tf.pad, (x, [[0, 0], [pad_y_max, pad_y_max], [pad_x_max, pad_x_max], [0, 0]]), dict(mode='REFLECT'), lscope),), None, lscope)), (ag__.converted_call(tf.reduce_max, (pad_x,), None, lscope), ag__.converted_call(tf.reduce_max, (pad_y,), None, lscope)), None, lscope)), (ag__.converted_call(tf.abs, (dx,), None, lscope), ag__.converted_call(tf.abs, (dy,), None, lscope)), None, lscope)), (ag__.converted_call(_stateless_uniform_int_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 14], tf.int64), None, lscope), -ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[1], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope), ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[1], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope) + 1), None, lscope), ag__.converted_call(_stateless_uniform_int_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 15], tf.int64), None, lscope), -ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[0], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope), ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[0], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope) + 1), None, lscope)), None, lscope)), (ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[1], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope), ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[0], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope)), None, lscope)), (ag__.converted_call(tf.image.crop_and_resize, (x, boxes, ag__.converted_call(tf.range, (ag__.converted_call(tf.shape, (x,), None, lscope)[0],), dict(dtype=tf.int32), lscope)), dict(crop_size=image_size, method='bilinear', extrapolation_value=0.0), lscope),), None, lscope)), (boxes, ag__.converted_call(tf.range, (ag__.converted_call(tf.shape, (x,), None, lscope)[0],), dict(dtype=tf.int32), lscope)), None, lscope)), (ag__.converted_call(tf.cast, (off_y, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (h, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_x, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (w, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_y + crop_h, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (h, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_x + crop_w, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (w, tf.float32), None, lscope)), None, lscope)), (ag__.converted_call(tf.minimum, (ag__.converted_call(_stateless_uniform_int_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 4], tf.int64), None, lscope), 0, ag__.converted_call(tf.reduce_max, (ag__.converted_call(tf.maximum, (h - crop_h, 0), None, lscope),), None, lscope) + 1), None, lscope), ag__.converted_call(tf.maximum, (h - crop_h, 0), None, lscope)), None, lscope), ag__.converted_call(tf.minimum, (ag__.converted_call(_stateless_uniform_int_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 5], tf.int64), None, lscope), 0, ag__.converted_call(tf.reduce_max, (ag__.converted_call(tf.maximum, (w - crop_w, 0), None, lscope),), None, lscope) + 1), None, lscope), ag__.converted_call(tf.maximum, (w - crop_w, 0), None, lscope)), None, lscope)), None, lscope)), (h - crop_h, w - crop_w), None, lscope)), (ag__.converted_call(tf.clip_by_value, (ag__.converted_call(tf.cast, (ag__.converted_call(tf.cast, (h, tf.float32), None, lscope) / zoom, tf.int32), None, lscope), 1, h), None, lscope), ag__.converted_call(tf.clip_by_value, (ag__.converted_call(tf.cast, (ag__.converted_call(tf.cast, (w, tf.float32), None, lscope) / zoom, tf.int32), None, lscope), 1, w), None, lscope)), None, lscope)), (ag__.converted_call(tf.shape, (x,), None, lscope)[1], ag__.converted_call(tf.shape, (x,), None, lscope)[2], ag__.converted_call(_stateless_uniform_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 13], tf.int64), None, lscope), 0.8, 1.2), dict(dtype=tf.float32), lscope)), None, lscope)), (ag__.converted_call(tf.clip_by_value, (x * br[:, None, None, None], 0.0, 1.0), None, lscope),), None, lscope)), (ag__.converted_call(_stateless_uniform_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 12], tf.int64), None, lscope), 0.7, 1.3), dict(dtype=tf.float32), lscope),), None, lscope)), (ag__.converted_call(tf.where, (p_ud[:, None, None, None] < 0.5, ag__.converted_call(tf.image.flip_up_down, (x,), None, lscope), x), None, lscope),), None, lscope)), (ag__.converted_call(_stateless_uniform_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 11], tf.int64), None, lscope), 0.0, 1.0), None, lscope),), None, lscope)), (ag__.converted_call(tf.where, (p_lr[:, None, None, None] < 0.5, ag__.converted_call(tf.image.flip_left_right, (x,), None, lscope), x), None, lscope),), None, lscope)), (ag__.converted_call(_stateless_uniform_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 10], tf.int64), None, lscope), 0.0, 1.0), None, lscope),), None, lscope)), (ag__.converted_call(tf.clip_by_value, (imgs / 255.0, 0.0, 1.0), None, lscope),), None, lscope)), (ag__.converted_call(tf.cast, (imgs_uint8_or_float, tf.float32), None, lscope), ag__.converted_call(_to_onehot, (y_ints,), None, lscope), ag__.converted_call(tf.stack, ([ag__.converted_call(tf.fill, (ag__.converted_call(tf.shape, (idxs,), None, lscope), ag__.converted_call(tf.cast, (seed, tf.int64), None, lscope)), None, lscope), ag__.converted_call(tf.cast, (idxs, tf.int64), None, lscope)],), dict(axis=1), lscope)), None, lscope)), (imgs_uint8_or_float, y_ints, idxs), None, lscope)), (imgs_f32, y_ints, idxs), None, lscope), 'lscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True))
      6         return tf__lam
      7     return inner_factory

/tmp/__autograph_generated_fileroe7juud.py in <lambda>(imgs_uint8_or_float, y_ints, idxs)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda imgs_f32, y_ints, idxs: ag__.with_function_scope(lambda lscope: ag__.converted_call(ag__.autograph_artifact(lambda imgs_uint8_or_float, y_ints, idxs: ag__.converted_call(ag__.autograph_artifact(lambda imgs_uint8_or_float, y_ints, idxs: ag__.converted_call(ag__.autograph_artifact(lambda imgs, y, ex_seeds: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda p_lr: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda p_ud: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda br: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda h, w, zoom: ag__.converted_call(ag__.autograph_artifact(lambda crop_h, crop_w: ag__.converted_call(ag__.autograph_artifact(lambda max_off_y, max_off_x: ag__.converted_call(ag__.autograph_artifact(lambda off_y, off_x: ag__.converted_call(ag__.autograph_artifact(lambda y1, x1, y2, x2: ag__.converted_call(ag__.autograph_artifact(lambda boxes, box_indices: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda max_dx, max_dy: ag__.converted_call(ag__.autograph_artifact(lambda dx, dy: ag__.converted_call(ag__.autograph_artifact(lambda pad_x, pad_y: ag__.converted_call(ag__.autograph_artifact(lambda pad_x_max, pad_y_max: ag__.converted_call(ag__.autograph_artifact(lambda x_pad: ag__.converted_call(ag__.autograph_artifact(lambda off_y2, off_x2, Hp, Wp: ag__.converted_call(ag__.autograph_artifact(lambda y1p, x1p, y2p, x2p: ag__.converted_call(ag__.autograph_artifact(lambda boxes2: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda x: (ag__.converted_call(preprocess_input, (x,), None, lscope), y)), (ag__.converted_call(tf.clip_by_value, (x, 0.0, 1.0), None, lscope) * 255.0,), None, lscope)), (ag__.converted_call(_rot_layer, (x,), dict(training=True), lscope),), None, lscope)), (ag__.converted_call(tf.image.crop_and_resize, (x_pad, boxes2, box_indices), dict(crop_size=image_size, method='bilinear', extrapolation_value=0.0), lscope),), None, lscope)), (ag__.converted_call(tf.stack, ([y1p, x1p, y2p, x2p],), dict(axis=1), lscope),), None, lscope)), (ag__.converted_call(tf.cast, (off_y2, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (Hp, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_x2, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (Wp, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_y2 + image_size[0], tf.float32), None, lscope) / ag__.converted_call(tf.cast, (Hp, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_x2 + image_size[1], tf.float32), None, lscope) / ag__.converted_call(tf.cast, (Wp, tf.float32), None, lscope)), None, lscope)), (pad_y_max - dy, pad_x_max - dx, ag__.converted_call(tf.shape, (x_pad,), None, lscope)[1], ag__.converted_call(tf.shape, (x_pad,), None, lscope)[2]), None, lscope)), (ag__.converted_call(tf.pad, (x, [[0, 0], [pad_y_max, pad_y_max], [pad_x_max, pad_x_max], [0, 0]]), dict(mode='REFLECT'), lscope),), None, lscope)), (ag__.converted_call(tf.reduce_max, (pad_x,), None, lscope), ag__.converted_call(tf.reduce_max, (pad_y,), None, lscope)), None, lscope)), (ag__.converted_call(tf.abs, (dx,), None, lscope), ag__.converted_call(tf.abs, (dy,), None, lscope)), None, lscope)), (ag__.converted_call(_stateless_uniform_int_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 14], tf.int64), None, lscope), -ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[1], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope), ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[1], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope) + 1), None, lscope), ag__.converted_call(_stateless_uniform_int_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 15], tf.int64), None, lscope), -ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[0], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope), ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[0], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope) + 1), None, lscope)), None, lscope)), (ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[1], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope), ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[0], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope)), None, lscope)), (ag__.converted_call(tf.image.crop_and_resize, (x, boxes, ag__.converted_call(tf.range, (ag__.converted_call(tf.shape, (x,), None, lscope)[0],), dict(dtype=tf.int32), lscope)), dict(crop_size=image_size, method='bilinear', extrapolation_value=0.0), lscope),), None, lscope)), (boxes, ag__.converted_call(tf.range, (ag__.converted_call(tf.shape, (x,), None, lscope)[0],), dict(dtype=tf.int32), lscope)), None, lscope)), (ag__.converted_call(tf.cast, (off_y, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (h, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_x, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (w, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_y + crop_h, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (h, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_x + crop_w, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (w, tf.float32), None, lscope)), None, lscope)), (ag__.converted_call(tf.minimum, (ag__.converted_call(_stateless_uniform_int_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 4], tf.int64), None, lscope), 0, ag__.converted_call(tf.reduce_max, (ag__.converted_call(tf.maximum, (h - crop_h, 0), None, lscope),), None, lscope) + 1), None, lscope), ag__.converted_call(tf.maximum, (h - crop_h, 0), None, lscope)), None, lscope), ag__.converted_call(tf.minimum, (ag__.converted_call(_stateless_uniform_int_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 5], tf.int64), None, lscope), 0, ag__.converted_call(tf.reduce_max, (ag__.converted_call(tf.maximum, (w - crop_w, 0), None, lscope),), None, lscope) + 1), None, lscope), ag__.converted_call(tf.maximum, (w - crop_w, 0), None, lscope)), None, lscope)), None, lscope)), (h - crop_h, w - crop_w), None, lscope)), (ag__.converted_call(tf.clip_by_value, (ag__.converted_call(tf.cast, (ag__.converted_call(tf.cast, (h, tf.float32), None, lscope) / zoom, tf.int32), None, lscope), 1, h), None, lscope), ag__.converted_call(tf.clip_by_value, (ag__.converted_call(tf.cast, (ag__.converted_call(tf.cast, (w, tf.float32), None, lscope) / zoom, tf.int32), None, lscope), 1, w), None, lscope)), None, lscope)), (ag__.converted_call(tf.shape, (x,), None, lscope)[1], ag__.converted_call(tf.shape, (x,), None, lscope)[2], ag__.converted_call(_stateless_uniform_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 13], tf.int64), None, lscope), 0.8, 1.2), dict(dtype=tf.float32), lscope)), None, lscope)), (ag__.converted_call(tf.clip_by_value, (x * br[:, None, None, None], 0.0, 1.0), None, lscope),), None, lscope)), (ag__.converted_call(_stateless_uniform_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 12], tf.int64), None, lscope), 0.7, 1.3), dict(dtype=tf.float32), lscope),), None, lscope)), (ag__.converted_call(tf.where, (p_ud[:, None, None, None] < 0.5, ag__.converted_call(tf.image.flip_up_down, (x,), None, lscope), x), None, lscope),), None, lscope)), (ag__.converted_call(_stateless_uniform_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 11], tf.int64), None, lscope), 0.0, 1.0), None, lscope),), None, lscope)), (ag__.converted_call(tf.where, (p_lr[:, None, None, None] < 0.5, ag__.converted_call(tf.image.flip_left_right, (x,), None, lscope), x), None, lscope),), None, lscope)), (ag__.converted_call(_stateless_uniform_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 10], tf.int64), None, lscope), 0.0, 1.0), None, lscope),), None, lscope)), (ag__.converted_call(tf.clip_by_value, (imgs / 255.0, 0.0, 1.0), None, lscope),), None, lscope)), (ag__.converted_call(tf.cast, (imgs_uint8_or_float, tf.float32), None, lscope), ag__.converted_call(_to_onehot, (y_ints,), None, lscope), ag__.converted_call(tf.stack, ([ag__.converted_call(tf.fill, (ag__.converted_call(tf.shape, (idxs,), None, lscope), ag__.converted_call(tf.cast, (seed, tf.int64), None, lscope)), None, lscope), ag__.converted_call(tf.cast, (idxs, tf.int64), None, lscope)],), dict(axis=1), lscope)), None, lscope)), (imgs_uint8_or_float, y_ints, idxs), None, lscope)), (imgs_f32, y_ints, idxs), None, lscope), 'lscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True))
      6         return tf__lam
      7     return inner_factory

/tmp/__autograph_generated_fileroe7juud.py in <lambda>(imgs, y, ex_seeds)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda imgs_f32, y_ints, idxs: ag__.with_function_scope(lambda lscope: ag__.converted_call(ag__.autograph_artifact(lambda imgs_uint8_or_float, y_ints, idxs: ag__.converted_call(ag__.autograph_artifact(lambda imgs_uint8_or_float, y_ints, idxs: ag__.converted_call(ag__.autograph_artifact(lambda imgs, y, ex_seeds: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda p_lr: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda p_ud: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda br: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda h, w, zoom: ag__.converted_call(ag__.autograph_artifact(lambda crop_h, crop_w: ag__.converted_call(ag__.autograph_artifact(lambda max_off_y, max_off_x: ag__.converted_call(ag__.autograph_artifact(lambda off_y, off_x: ag__.converted_call(ag__.autograph_artifact(lambda y1, x1, y2, x2: ag__.converted_call(ag__.autograph_artifact(lambda boxes, box_indices: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda max_dx, max_dy: ag__.converted_call(ag__.autograph_artifact(lambda dx, dy: ag__.converted_call(ag__.autograph_artifact(lambda pad_x, pad_y: ag__.converted_call(ag__.autograph_artifact(lambda pad_x_max, pad_y_max: ag__.converted_call(ag__.autograph_artifact(lambda x_pad: ag__.converted_call(ag__.autograph_artifact(lambda off_y2, off_x2, Hp, Wp: ag__.converted_call(ag__.autograph_artifact(lambda y1p, x1p, y2p, x2p: ag__.converted_call(ag__.autograph_artifact(lambda boxes2: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda x: (ag__.converted_call(preprocess_input, (x,), None, lscope), y)), (ag__.converted_call(tf.clip_by_value, (x, 0.0, 1.0), None, lscope) * 255.0,), None, lscope)), (ag__.converted_call(_rot_layer, (x,), dict(training=True), lscope),), None, lscope)), (ag__.converted_call(tf.image.crop_and_resize, (x_pad, boxes2, box_indices), dict(crop_size=image_size, method='bilinear', extrapolation_value=0.0), lscope),), None, lscope)), (ag__.converted_call(tf.stack, ([y1p, x1p, y2p, x2p],), dict(axis=1), lscope),), None, lscope)), (ag__.converted_call(tf.cast, (off_y2, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (Hp, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_x2, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (Wp, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_y2 + image_size[0], tf.float32), None, lscope) / ag__.converted_call(tf.cast, (Hp, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_x2 + image_size[1], tf.float32), None, lscope) / ag__.converted_call(tf.cast, (Wp, tf.float32), None, lscope)), None, lscope)), (pad_y_max - dy, pad_x_max - dx, ag__.converted_call(tf.shape, (x_pad,), None, lscope)[1], ag__.converted_call(tf.shape, (x_pad,), None, lscope)[2]), None, lscope)), (ag__.converted_call(tf.pad, (x, [[0, 0], [pad_y_max, pad_y_max], [pad_x_max, pad_x_max], [0, 0]]), dict(mode='REFLECT'), lscope),), None, lscope)), (ag__.converted_call(tf.reduce_max, (pad_x,), None, lscope), ag__.converted_call(tf.reduce_max, (pad_y,), None, lscope)), None, lscope)), (ag__.converted_call(tf.abs, (dx,), None, lscope), ag__.converted_call(tf.abs, (dy,), None, lscope)), None, lscope)), (ag__.converted_call(_stateless_uniform_int_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 14], tf.int64), None, lscope), -ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[1], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope), ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[1], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope) + 1), None, lscope), ag__.converted_call(_stateless_uniform_int_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 15], tf.int64), None, lscope), -ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[0], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope), ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[0], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope) + 1), None, lscope)), None, lscope)), (ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[1], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope), ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[0], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope)), None, lscope)), (ag__.converted_call(tf.image.crop_and_resize, (x, boxes, ag__.converted_call(tf.range, (ag__.converted_call(tf.shape, (x,), None, lscope)[0],), dict(dtype=tf.int32), lscope)), dict(crop_size=image_size, method='bilinear', extrapolation_value=0.0), lscope),), None, lscope)), (boxes, ag__.converted_call(tf.range, (ag__.converted_call(tf.shape, (x,), None, lscope)[0],), dict(dtype=tf.int32), lscope)), None, lscope)), (ag__.converted_call(tf.cast, (off_y, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (h, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_x, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (w, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_y + crop_h, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (h, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_x + crop_w, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (w, tf.float32), None, lscope)), None, lscope)), (ag__.converted_call(tf.minimum, (ag__.converted_call(_stateless_uniform_int_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 4], tf.int64), None, lscope), 0, ag__.converted_call(tf.reduce_max, (ag__.converted_call(tf.maximum, (h - crop_h, 0), None, lscope),), None, lscope) + 1), None, lscope), ag__.converted_call(tf.maximum, (h - crop_h, 0), None, lscope)), None, lscope), ag__.converted_call(tf.minimum, (ag__.converted_call(_stateless_uniform_int_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 5], tf.int64), None, lscope), 0, ag__.converted_call(tf.reduce_max, (ag__.converted_call(tf.maximum, (w - crop_w, 0), None, lscope),), None, lscope) + 1), None, lscope), ag__.converted_call(tf.maximum, (w - crop_w, 0), None, lscope)), None, lscope)), None, lscope)), (h - crop_h, w - crop_w), None, lscope)), (ag__.converted_call(tf.clip_by_value, (ag__.converted_call(tf.cast, (ag__.converted_call(tf.cast, (h, tf.float32), None, lscope) / zoom, tf.int32), None, lscope), 1, h), None, lscope), ag__.converted_call(tf.clip_by_value, (ag__.converted_call(tf.cast, (ag__.converted_call(tf.cast, (w, tf.float32), None, lscope) / zoom, tf.int32), None, lscope), 1, w), None, lscope)), None, lscope)), (ag__.converted_call(tf.shape, (x,), None, lscope)[1], ag__.converted_call(tf.shape, (x,), None, lscope)[2], ag__.converted_call(_stateless_uniform_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 13], tf.int64), None, lscope), 0.8, 1.2), dict(dtype=tf.float32), lscope)), None, lscope)), (ag__.converted_call(tf.clip_by_value, (x * br[:, None, None, None], 0.0, 1.0), None, lscope),), None, lscope)), (ag__.converted_call(_stateless_uniform_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 12], tf.int64), None, lscope), 0.7, 1.3), dict(dtype=tf.float32), lscope),), None, lscope)), (ag__.converted_call(tf.where, (p_ud[:, None, None, None] < 0.5, ag__.converted_call(tf.image.flip_up_down, (x,), None, lscope), x), None, lscope),), None, lscope)), (ag__.converted_call(_stateless_uniform_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 11], tf.int64), None, lscope), 0.0, 1.0), None, lscope),), None, lscope)), (ag__.converted_call(tf.where, (p_lr[:, None, None, None] < 0.5, ag__.converted_call(tf.image.flip_left_right, (x,), None, lscope), x), None, lscope),), None, lscope)), (ag__.converted_call(_stateless_uniform_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 10], tf.int64), None, lscope), 0.0, 1.0), None, lscope),), None, lscope)), (ag__.converted_call(tf.clip_by_value, (imgs / 255.0, 0.0, 1.0), None, lscope),), None, lscope)), (ag__.converted_call(tf.cast, (imgs_uint8_or_float, tf.float32), None, lscope), ag__.converted_call(_to_onehot, (y_ints,), None, lscope), ag__.converted_call(tf.stack, ([ag__.converted_call(tf.fill, (ag__.converted_call(tf.shape, (idxs,), None, lscope), ag__.converted_call(tf.cast, (seed, tf.int64), None, lscope)), None, lscope), ag__.converted_call(tf.cast, (idxs, tf.int64), None, lscope)],), dict(axis=1), lscope)), None, lscope)), (imgs_uint8_or_float, y_ints, idxs), None, lscope)), (imgs_f32, y_ints, idxs), None, lscope), 'lscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True))
      6         return tf__lam
      7     return inner_factory

/tmp/__autograph_generated_fileroe7juud.py in <lambda>(x)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda imgs_f32, y_ints, idxs: ag__.with_function_scope(lambda lscope: ag__.converted_call(ag__.autograph_artifact(lambda imgs_uint8_or_float, y_ints, idxs: ag__.converted_call(ag__.autograph_artifact(lambda imgs_uint8_or_float, y_ints, idxs: ag__.converted_call(ag__.autograph_artifact(lambda imgs, y, ex_seeds: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda p_lr: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda p_ud: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda br: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda h, w, zoom: ag__.converted_call(ag__.autograph_artifact(lambda crop_h, crop_w: ag__.converted_call(ag__.autograph_artifact(lambda max_off_y, max_off_x: ag__.converted_call(ag__.autograph_artifact(lambda off_y, off_x: ag__.converted_call(ag__.autograph_artifact(lambda y1, x1, y2, x2: ag__.converted_call(ag__.autograph_artifact(lambda boxes, box_indices: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda max_dx, max_dy: ag__.converted_call(ag__.autograph_artifact(lambda dx, dy: ag__.converted_call(ag__.autograph_artifact(lambda pad_x, pad_y: ag__.converted_call(ag__.autograph_artifact(lambda pad_x_max, pad_y_max: ag__.converted_call(ag__.autograph_artifact(lambda x_pad: ag__.converted_call(ag__.autograph_artifact(lambda off_y2, off_x2, Hp, Wp: ag__.converted_call(ag__.autograph_artifact(lambda y1p, x1p, y2p, x2p: ag__.converted_call(ag__.autograph_artifact(lambda boxes2: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda x: ag__.converted_call(ag__.autograph_artifact(lambda x: (ag__.converted_call(preprocess_input, (x,), None, lscope), y)), (ag__.converted_call(tf.clip_by_value, (x, 0.0, 1.0), None, lscope) * 255.0,), None, lscope)), (ag__.converted_call(_rot_layer, (x,), dict(training=True), lscope),), None, lscope)), (ag__.converted_call(tf.image.crop_and_resize, (x_pad, boxes2, box_indices), dict(crop_size=image_size, method='bilinear', extrapolation_value=0.0), lscope),), None, lscope)), (ag__.converted_call(tf.stack, ([y1p, x1p, y2p, x2p],), dict(axis=1), lscope),), None, lscope)), (ag__.converted_call(tf.cast, (off_y2, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (Hp, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_x2, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (Wp, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_y2 + image_size[0], tf.float32), None, lscope) / ag__.converted_call(tf.cast, (Hp, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_x2 + image_size[1], tf.float32), None, lscope) / ag__.converted_call(tf.cast, (Wp, tf.float32), None, lscope)), None, lscope)), (pad_y_max - dy, pad_x_max - dx, ag__.converted_call(tf.shape, (x_pad,), None, lscope)[1], ag__.converted_call(tf.shape, (x_pad,), None, lscope)[2]), None, lscope)), (ag__.converted_call(tf.pad, (x, [[0, 0], [pad_y_max, pad_y_max], [pad_x_max, pad_x_max], [0, 0]]), dict(mode='REFLECT'), lscope),), None, lscope)), (ag__.converted_call(tf.reduce_max, (pad_x,), None, lscope), ag__.converted_call(tf.reduce_max, (pad_y,), None, lscope)), None, lscope)), (ag__.converted_call(tf.abs, (dx,), None, lscope), ag__.converted_call(tf.abs, (dy,), None, lscope)), None, lscope)), (ag__.converted_call(_stateless_uniform_int_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 14], tf.int64), None, lscope), -ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[1], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope), ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[1], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope) + 1), None, lscope), ag__.converted_call(_stateless_uniform_int_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 15], tf.int64), None, lscope), -ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[0], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope), ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[0], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope) + 1), None, lscope)), None, lscope)), (ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[1], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope), ag__.converted_call(tf.cast, (ag__.converted_call(tf.round, (0.2 * ag__.converted_call(tf.cast, (image_size[0], tf.float32), None, lscope),), None, lscope), tf.int32), None, lscope)), None, lscope)), (ag__.converted_call(tf.image.crop_and_resize, (x, boxes, ag__.converted_call(tf.range, (ag__.converted_call(tf.shape, (x,), None, lscope)[0],), dict(dtype=tf.int32), lscope)), dict(crop_size=image_size, method='bilinear', extrapolation_value=0.0), lscope),), None, lscope)), (boxes, ag__.converted_call(tf.range, (ag__.converted_call(tf.shape, (x,), None, lscope)[0],), dict(dtype=tf.int32), lscope)), None, lscope)), (ag__.converted_call(tf.cast, (off_y, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (h, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_x, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (w, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_y + crop_h, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (h, tf.float32), None, lscope), ag__.converted_call(tf.cast, (off_x + crop_w, tf.float32), None, lscope) / ag__.converted_call(tf.cast, (w, tf.float32), None, lscope)), None, lscope)), (ag__.converted_call(tf.minimum, (ag__.converted_call(_stateless_uniform_int_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 4], tf.int64), None, lscope), 0, ag__.converted_call(tf.reduce_max, (ag__.converted_call(tf.maximum, (h - crop_h, 0), None, lscope),), None, lscope) + 1), None, lscope), ag__.converted_call(tf.maximum, (h - crop_h, 0), None, lscope)), None, lscope), ag__.converted_call(tf.minimum, (ag__.converted_call(_stateless_uniform_int_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 5], tf.int64), None, lscope), 0, ag__.converted_call(tf.reduce_max, (ag__.converted_call(tf.maximum, (w - crop_w, 0), None, lscope),), None, lscope) + 1), None, lscope), ag__.converted_call(tf.maximum, (w - crop_w, 0), None, lscope)), None, lscope)), None, lscope)), (h - crop_h, w - crop_w), None, lscope)), (ag__.converted_call(tf.clip_by_value, (ag__.converted_call(tf.cast, (ag__.converted_call(tf.cast, (h, tf.float32), None, lscope) / zoom, tf.int32), None, lscope), 1, h), None, lscope), ag__.converted_call(tf.clip_by_value, (ag__.converted_call(tf.cast, (ag__.converted_call(tf.cast, (w, tf.float32), None, lscope) / zoom, tf.int32), None, lscope), 1, w), None, lscope)), None, lscope)), (ag__.converted_call(tf.shape, (x,), None, lscope)[1], ag__.converted_call(tf.shape, (x,), None, lscope)[2], ag__.converted_call(_stateless_uniform_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 13], tf.int64), None, lscope), 0.8, 1.2), dict(dtype=tf.float32), lscope)), None, lscope)), (ag__.converted_call(tf.clip_by_value, (x * br[:, None, None, None], 0.0, 1.0), None, lscope),), None, lscope)), (ag__.converted_call(_stateless_uniform_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 12], tf.int64), None, lscope), 0.7, 1.3), dict(dtype=tf.float32), lscope),), None, lscope)), (ag__.converted_call(tf.where, (p_ud[:, None, None, None] < 0.5, ag__.converted_call(tf.image.flip_up_down, (x,), None, lscope), x), None, lscope),), None, lscope)), (ag__.converted_call(_stateless_uniform_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 11], tf.int64), None, lscope), 0.0, 1.0), None, lscope),), None, lscope)), (ag__.converted_call(tf.where, (p_lr[:, None, None, None] < 0.5, ag__.converted_call(tf.image.flip_left_right, (x,), None, lscope), x), None, lscope),), None, lscope)), (ag__.converted_call(_stateless_uniform_per_example, (ex_seeds + ag__.converted_call(tf.constant, ([0, 10], tf.int64), None, lscope), 0.0, 1.0), None, lscope),), None, lscope)), (ag__.converted_call(tf.clip_by_value, (imgs / 255.0, 0.0, 1.0), None, lscope),), None, lscope)), (ag__.converted_call(tf.cast, (imgs_uint8_or_float, tf.float32), None, lscope), ag__.converted_call(_to_onehot, (y_ints,), None, lscope), ag__.converted_call(tf.stack, ([ag__.converted_call(tf.fill, (ag__.converted_call(tf.shape, (idxs,), None, lscope), ag__.converted_call(tf.cast, (seed, tf.int64), None, lscope)), None, lscope), ag__.converted_call(tf.cast, (idxs, tf.int64), None, lscope)],), dict(axis=1), lscope)), None, lscope)), (imgs_uint8_or_float, y_ints, idxs), None, lscope)), (imgs_f32, y_ints, idxs), None, lscope), 'lscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True))
      6         return tf__lam
      7     return inner_factory

/tmp/__autograph_generated_fileo3ea7b18.py in tf___stateless_uniform_per_example(ex_seeds, minval, maxval, dtype)
     13                 except:
     14                     do_return = False
---> 15                     raise
     16                 return fscope.ret(retval_, do_return)
     17         return tf___stateless_uniform_per_example

ValueError: in user code:

    File "/tmp/ipykernel_11/4040939207.py", line 90, in None  *
        )
    File "/tmp/ipykernel_11/4040939207.py", line 71, in _stateless_uniform_per_example  *
        dtype=dtype,

    ValueError: Shape must be rank 1 but is rank 2 for '{{node stateless_random_uniform/StatelessRandomGetKeyCounter}} = StatelessRandomGetKeyCounter[Tseed=DT_INT64](ex_seeds)' with input shapes: [?,2].


## === cell 8
input_shape_c = (image_size[0], image_size[1], 3)
print(input_shape_c)

base_model = ResNet50V2(
    weights="imagenet", include_top=False, input_shape=input_shape_c
)



## === cell 9
for layer in base_model.layers:
    if layer.name == "conv5_block1_1_conv":
        break
    layer.trainable = False

pre_trained_model = Sequential()
pre_trained_model.add(base_model)
pre_trained_model.add(layers.Flatten())
pre_trained_model.add(layers.Dense(512, activation="relu"))
pre_trained_model.add(Dropout(0.5))
pre_trained_model.add(BatchNormalization())
pre_trained_model.add(layers.Dense(num_classes, activation="softmax"))
pre_trained_model.summary()



## === cell 10
epochs = 200

print("[INFO]: Compiling the model...")
pre_trained_model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=1e-3),
    metrics=["accuracy"],
)


def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * exp(-0.1)


annealer = LearningRateScheduler(scheduler)

earlystop = EarlyStopping(
    patience=5,
    monitor="val_loss",
)

best_path = file + ".weights.h5"
modelsave = ModelCheckpoint(
    filepath=best_path, save_best_only=True, save_weights_only=True, verbose=1
)

print("[INFO]: Training the network...")
H_pre = pre_trained_model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    epochs=epochs,
    callbacks=[annealer, earlystop, modelsave],
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3429255679.py in <cell line: 0>()
     30 print("[INFO]: Training the network...")
     31 H_pre = pre_trained_model.fit(
---> 32     train_ds,
     33     validation_data=val_ds,
     34     steps_per_epoch=train_steps,

NameError: name 'train_ds' is not defined

## === cell 11
print("[INFO]: Training finished. Skipping curve plots for runtime.")
print("Epochs run:", len(H_pre.history.get("loss", [])))



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1968910585.py in <cell line: 0>()
      1 print("[INFO]: Training finished. Skipping curve plots for runtime.")
----> 2 print("Epochs run:", len(H_pre.history.get("loss", [])))
      3 

NameError: name 'H_pre' is not defined

## === cell 12
csv_testfile = "/kaggle/working/test.csv"
print("Skipping test.csv generation for runtime. (Not needed.)")



## === cell 13
print("Skipping loading test.csv for runtime.")



## === cell 14
if os.path.exists(best_path):
    print("[INFO]: Loading best checkpoint weights for inference:", best_path)
    pre_trained_model.load_weights(best_path)
else:
    print("[WARN]: Best checkpoint weights not found; using in-memory model weights.")

test_batch_size = 32
PROYECT_FOLDER_TEST = "/kaggle/input/plant-seedlings-classification/test"

test_files_ds = tf.keras.utils.image_dataset_from_directory(
    PROYECT_FOLDER_TEST,
    labels=None,
    label_mode=None,
    class_names=None,
    color_mode="rgb",
    batch_size=test_batch_size,
    image_size=image_size,
    shuffle=False,
    interpolation="bilinear",
)


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=(None, image_size[0], image_size[1], 3), dtype=tf.float32)
    ],
    reduce_retracing=True,
)
def _preprocess_only(img):
    return preprocess_input(tf.cast(img, tf.float32))


test_ds = test_files_ds.map(
    _preprocess_only, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
)

try:
    gpus = tf.config.list_logical_devices("GPU")
    if gpus:
        test_ds = test_ds.apply(tf.data.experimental.copy_to_device("/GPU:0"))
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    else:
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
except Exception:
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

list_of_files = [os.path.basename(p) for p in test_files_ds.file_paths]
print("Example test filename from dataset:", list_of_files[0])
print("Num test files:", len(list_of_files))



## === cell 15
predicted_class = pre_trained_model.predict(
    test_ds,
    verbose=1,
)



## === cell 16
predicted_class_number = np.argmax(predicted_class, axis=1)

idx_to_class = {i: c for i, c in enumerate(classes)}
classes_ordered = [idx_to_class[i] for i in range(len(idx_to_class))]

print("Ordered classes:", classes_ordered[:5], "...", len(classes_ordered))
print("Predicted class indices sample:", predicted_class_number[:10])



## === cell 17
sample_path = "/kaggle/input/plant-seedlings-classification/sample_submission.csv"
if not os.path.exists(sample_path):
    sample_path = "/kaggle/input/sample_submission.csv"

sample = pd.read_csv(sample_path)
if "file" not in sample.columns or "species" not in sample.columns:
    raise ValueError(f"Unexpected sample submission columns: {sample.columns.tolist()}")

pred_files = list_of_files
pred_species = [classes_ordered[i] for i in predicted_class_number]
pred_df = pd.DataFrame({"file": pred_files, "species": pred_species})

submission = sample[["file"]].merge(pred_df, on="file", how="left")
if submission["species"].isna().any():
    missing = submission.loc[submission["species"].isna(), "file"].head(10).tolist()
    raise ValueError(f"Missing predictions for some files, e.g.: {missing}")

submission = submission[["file", "species"]]

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.shape)
print(submission.head())
print("Submission columns:", submission.columns.tolist())
print("Done. Submission is ready at /kaggle/working/submission.csv")
