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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

try:
    import google.protobuf  # noqa: F401
    from importlib.metadata import version as _pkg_version

    pb_ver = _pkg_version("protobuf")
    if int(pb_ver.split(".")[0]) >= 6:
        os.system("python -m pip install -q --no-deps 'protobuf<6'")
except Exception:
    pass

import random
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TF:", tf.__version__)
print("Eager:", tf.executing_eagerly())




## === cell 1
DATA_ROOT = "/kaggle/input/ranzcr-clip-catheter-line-classification"
if not os.path.exists(DATA_ROOT):
    DATA_ROOT = "/kaggle/data/ranzcr-clip-catheter-line-classification"

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_DIR = os.path.join(DATA_ROOT, "test")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

train_df = pd.read_csv(TRAIN_CSV)

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

missing = [c for c in TARGET_COLS if c not in train_df.columns]
if missing:
    raise ValueError(f"Missing target columns in train.csv: {missing}")

if os.path.exists(SAMPLE_SUB):
    sample_df = pd.read_csv(SAMPLE_SUB)
else:
    sample_df = pd.read_csv("/kaggle/data/sample_submission.csv")

if "StudyInstanceUID" not in sample_df.columns:
    raise ValueError("sample_submission.csv missing StudyInstanceUID column")

test_uids = sample_df["StudyInstanceUID"].astype(str).tolist()

print("Train rows:", len(train_df), "Test images:", len(test_uids))
print("Sample submission columns:", list(sample_df.columns))




## === cell 2
IMG_SIZE = 260
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

CACHE_DIR = "/kaggle/working/tf_cache_ranzcr"
os.makedirs(CACHE_DIR, exist_ok=True)


def _decode_resize_normalize(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize_with_pad(
        img, IMG_SIZE, IMG_SIZE, method="bilinear", antialias=False
    )
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img


def _seed_from_path(path):
    h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
    return tf.stack([tf.cast(SEED, tf.int32), tf.cast(h, tf.int32)], axis=0)


def _augment_stateless(img, seed2):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed2)
    img = tf.image.stateless_random_brightness(img, max_delta=0.08, seed=seed2 + 1)
    img = tf.image.stateless_random_contrast(img, lower=0.9, upper=1.1, seed=seed2 + 2)
    return img


def _build_paths(base_dir, uids):
    u = np.asarray(uids, dtype=str)
    p = np.char.add(np.char.add(np.char.add(base_dir, os.sep), u), ".jpg")
    return p.astype(str)


def _base_ds_options():
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True
    return opts


def make_train_ds(df, training=True, cache_name="train_cache"):
    uids = df["StudyInstanceUID"].values.astype(str)
    labels = df[TARGET_COLS].values.astype("float32")
    paths = _build_paths(TRAIN_DIR, uids)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(min(len(df), 8192), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.with_options(_base_ds_options())

    def _load(path, y):
        img_bytes = tf.io.read_file(path)
        img = _decode_resize_normalize(img_bytes)
        if training:
            seed2 = _seed_from_path(path)
            img = _augment_stateless(img, seed2)
        return img, y

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE, drop_remainder=training)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_ds(uids):
    paths = _build_paths(TEST_DIR, uids)
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.with_options(_base_ds_options())

    def _load(path):
        img_bytes = tf.io.read_file(path)
        img = _decode_resize_normalize(img_bytes)
        return img

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds




## === cell 3
patients = train_df["PatientID"].astype(str).values
unique_patients = np.unique(patients)
rng = np.random.RandomState(SEED)
rng.shuffle(unique_patients)

val_frac = 0.1
n_val = int(len(unique_patients) * val_frac)
val_patients = set(unique_patients[:n_val])

is_val = train_df["PatientID"].astype(str).isin(val_patients)
tr_df = train_df.loc[~is_val].reset_index(drop=True)
va_df = train_df.loc[is_val].reset_index(drop=True)

print("Train split:", tr_df.shape, "Val split:", va_df.shape)

train_ds = make_train_ds(tr_df, training=True, cache_name="train")
val_ds = make_train_ds(va_df, training=False, cache_name="val")




## === cell 4
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(len(TARGET_COLS), activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)

auc_metric = tf.keras.metrics.AUC(
    multi_label=True, num_labels=len(TARGET_COLS), name="auc"
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[auc_metric],
    jit_compile=True,
    steps_per_execution=32,
)
model.summary()




## === cell 5
EPOCHS = 6
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)




## === cell 6
test_ds = make_test_ds(test_uids)

pred = model.predict(test_ds, verbose=1)
pred = np.clip(pred, 0.0, 1.0)

sub = pd.DataFrame({"StudyInstanceUID": test_uids})
for i, c in enumerate(TARGET_COLS):
    sub[c] = pred[:, i]

sub = sub[["StudyInstanceUID"] + TARGET_COLS]
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv:", sub.shape)
print(sub.head())
print("Submission columns:", list(sub.columns))
