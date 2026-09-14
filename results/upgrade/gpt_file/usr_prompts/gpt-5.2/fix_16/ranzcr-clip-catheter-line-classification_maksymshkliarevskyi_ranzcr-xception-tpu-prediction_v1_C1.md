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
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

print("TensorFlow:", tf.__version__)

tf.keras.backend.set_image_data_format("channels_last")

try:
    tf.config.experimental.enable_op_determinism()
    print("Determinism: enabled")
except Exception as _:
    print("Determinism: not available in this TF build")

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF choose
    tf.config.threading.set_inter_op_parallelism_threads(0)  # let TF choose
except Exception as _:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception as _:
    pass

try:
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision policy:", mixed_precision.global_policy())
except Exception as e:
    print("Mixed precision skipped:", e)



## === cell 1
try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPUs:", gpus)
except Exception as e:
    print("GPU config skipped:", e)




## === cell 2
def build_decoder(with_labels=True, target_size=(512, 512), ext="jpg"):
    def decode(path):
        file_bytes = tf.io.read_file(path)
        if ext == "png":
            img = tf.image.decode_png(file_bytes, channels=3)
        elif ext in ["jpg", "jpeg"]:
            img = tf.image.decode_jpeg(
                file_bytes, channels=3, dct_method="INTEGER_FAST"
            )
        else:
            raise ValueError("Image extension not supported")

        img.set_shape([None, None, 3])
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, target_size, antialias=False)
        img.set_shape([target_size[0], target_size[1], 3])
        return img

    def decode_with_labels(path, label):
        return decode(path), label

    return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True):
    def augment(img, seed):
        img = tf.image.stateless_random_flip_left_right(img, seed=seed)
        seed2 = tf.random.experimental.stateless_split(seed, num=2)[1]
        img = tf.image.stateless_random_flip_up_down(img, seed=seed2)
        return img

    def augment_with_labels(img, label, seed):
        return augment(img, seed), label

    def augment_no_labels(img, seed):
        return augment(img, seed)

    return augment_with_labels if with_labels else augment_no_labels


def build_dataset(
    paths,
    labels=None,
    bsize=32,
    cache=False,
    decode_fn=None,
    augment_fn=None,
    augment=True,
    repeat=True,
    shuffle=1024,
    cache_dir="",
    deterministic=True,
    drop_remainder=False,
):
    """
    Timeout fix, correctness preserved:
    - Add aggressive tf.data options (parallel map/batch/prefetch) to improve throughput.
    - Keep augmentation stateless/deterministic per-element via enumerate() seeds (same augment semantics).
    - Allow non-deterministic scheduling for training pipeline (does not change data/labels, only execution order),
      improving performance under tf.data parallelism.
    - Support disk cache for decoded+resized data (when enabled by caller) to avoid repeated JPEG decode each epoch.
    """
    if decode_fn is None:
        decode_fn = build_decoder(
            labels is not None, target_size=(TARGET_SIZE, TARGET_SIZE)
        )

    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)

    AUTO = tf.data.AUTOTUNE
    slices = paths if labels is None else (paths, labels)
    dset = tf.data.Dataset.from_tensor_slices(slices)

    options = tf.data.Options()
    options.experimental_deterministic = bool(deterministic)
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    try:
        options.experimental_optimization.autotune_buffers = True
    except Exception:
        pass
    options.threading.private_threadpool_size = 0
    options.threading.max_intra_op_parallelism = 0
    dset = dset.with_options(options)

    if shuffle:
        dset = dset.shuffle(shuffle, seed=SEED, reshuffle_each_iteration=True)

    dset = dset.map(decode_fn, num_parallel_calls=AUTO)
    dset = dset.apply(tf.data.experimental.ignore_errors())

    if cache:
        if cache_dir:
            os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
            dset = dset.cache(cache_dir)
        else:
            dset = dset.cache()

    if augment:

        def _aug_enum(i, xy):
            seed = tf.stack(
                [
                    tf.cast(i, tf.int32) + tf.constant(SEED * 1000003, tf.int32),
                    tf.cast(i, tf.int32) + tf.constant(SEED * 9176 + 17, tf.int32),
                ],
                axis=0,
            )
            if labels is not None:
                img, lab = xy
                return augment_fn(img, lab, seed)
            else:
                return augment_fn(xy, seed)

        dset = dset.enumerate().map(_aug_enum, num_parallel_calls=AUTO)

    if repeat:
        dset = dset.repeat()

    dset = dset.batch(bsize, drop_remainder=bool(drop_remainder))
    dset = dset.prefetch(AUTO)
    return dset




## === cell 3
WORK_DIR = "../input/ranzcr-clip-catheter-line-classification"
print("WORK_DIR exists:", os.path.exists(WORK_DIR))



## === cell 4
train = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))

label_cols = [c for c in train.columns if c not in ["StudyInstanceUID", "PatientID"]]
assert len(label_cols) == 11, f"Expected 11 labels, got {len(label_cols)}: {label_cols}"

train_images = (
    WORK_DIR + "/train/" + train["StudyInstanceUID"].astype(str) + ".jpg"
).to_numpy()
labels = train[label_cols].to_numpy(dtype=np.float32)

ss_raw = pd.read_csv(os.path.join(WORK_DIR, "sample_submission.csv"))
test_images = (
    WORK_DIR + "/test/" + ss_raw["StudyInstanceUID"].astype(str) + ".jpg"
).to_numpy()

print("Num train:", len(train), "Num test:", len(ss_raw))
print("Labels:", label_cols)



## === cell 5
pass



## === cell 6
pass



## === cell 7
BATCH_SIZE = 16
EPOCHS = 15

TARGET_SIZE = 256

STEPS_PER_EPOCH = None
VALIDATION_STEPS = None

print(
    "Configured BATCH_SIZE:", BATCH_SIZE, "EPOCHS:", EPOCHS, "TARGET_SIZE:", TARGET_SIZE
)



## === cell 8
patients = train["PatientID"].values
unique_patients = np.unique(patients)

pt_train, pt_val = train_test_split(
    unique_patients, test_size=0.2, random_state=SEED, shuffle=True
)
is_val = train["PatientID"].isin(pt_val).values

train_paths = train_images[~is_val]
val_paths = train_images[is_val]

train_labels = labels[~is_val]
val_labels = labels[is_val]

print("Train samples:", len(train_paths), "Val samples:", len(val_paths))

STEPS_PER_EPOCH = int(len(train_paths) // BATCH_SIZE)
VALIDATION_STEPS = int(len(val_paths) // BATCH_SIZE)
print("STEPS_PER_EPOCH:", STEPS_PER_EPOCH, "VALIDATION_STEPS:", VALIDATION_STEPS)

train_cache_path = os.path.join("/kaggle/working", f"train_cache_{TARGET_SIZE}.tfdata")
train_df = build_dataset(
    train_paths,
    train_labels,
    bsize=BATCH_SIZE,
    repeat=True,
    shuffle=2048,
    augment=True,
    cache=True,
    cache_dir=train_cache_path,
    deterministic=False,  # allow faster parallel scheduling; augmentation remains stateless/deterministic per element
    drop_remainder=True,  # fixed shapes reduce XLA retracing/overhead
)

valid_cache_path = os.path.join("/kaggle/working", f"valid_cache_{TARGET_SIZE}.tfdata")
valid_df = build_dataset(
    val_paths,
    val_labels,
    bsize=BATCH_SIZE,
    repeat=True,
    shuffle=False,
    augment=False,
    cache=True,
    cache_dir=valid_cache_path,
    deterministic=True,
    drop_remainder=True,
)

test_df = build_dataset(
    test_images,
    labels=None,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=False,
    cache_dir="",
    deterministic=True,
    drop_remainder=False,
)

print("Datasets built.")




## === cell 9
def build_model(input_shape=(TARGET_SIZE, TARGET_SIZE, 3), n_classes=11):
    base = Xception(include_top=False, weights="imagenet", input_shape=input_shape)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(n_classes, activation="sigmoid", dtype="float32")(x)
    model = models.Model(inputs=base.input, outputs=out)
    return model


model = build_model()

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.AUC(multi_label=True, num_labels=len(label_cols), name="auc")
    ],
    jit_compile=True,
)
model.summary()



## === cell 10
ckpt_path = "xception_best.keras"
callbacks = [
    ModelCheckpoint(
        ckpt_path,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
    ),
    ReduceLROnPlateau(
        monitor="val_auc", mode="max", factor=0.5, patience=2, verbose=1, min_lr=1e-6
    ),
]

history = model.fit(
    train_df,
    validation_data=valid_df,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    callbacks=callbacks,
    verbose=1,
)

model = tf.keras.models.load_model(ckpt_path)



## === cell 11
print("Our Xception CNN has %d layers" % len(model.layers))



## === cell 12
pass



## === cell 13
pass



## === cell 14
test_steps = int(np.ceil(len(test_images) / BATCH_SIZE))

preds = model.predict(test_df, steps=test_steps, verbose=1)
preds = np.clip(preds, 0.0, 1.0)



## === cell 15
submission = pd.DataFrame({"StudyInstanceUID": ss_raw["StudyInstanceUID"].values})
for j, col in enumerate(label_cols):
    submission[col] = preds[:, j].astype(np.float32)

expected_cols = ["StudyInstanceUID"] + label_cols
assert list(submission.columns) == expected_cols, "Submission columns mismatch"
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
