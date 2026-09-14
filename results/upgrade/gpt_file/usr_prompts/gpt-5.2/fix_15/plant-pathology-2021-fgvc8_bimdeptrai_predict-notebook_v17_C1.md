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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.7808442158026662

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import random
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.preprocessing import MultiLabelBinarizer

SEED = 42
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



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print(train.shape, submissions.shape)
display(train.head())



## === cell 2
h_target = 384
w_target = 384
batch_size = 32

AUTOTUNE = tf.data.AUTOTUNE
SHUFFLE_BUFFER = min(len(train), 4096)

USE_RAM_CACHE = False
USE_DISK_CACHE = True
CACHE_DIR = "/kaggle/working/tfdata_cache"
os.makedirs(CACHE_DIR, exist_ok=True)



## === cell 3
label_split = train["labels"].str.split()

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_split)
classes = list(mlb.classes_)

labels_df = pd.DataFrame(y, columns=classes)
display(labels_df.head())
print("Num classes:", len(classes))
print("Classes:", classes)




## === cell 4
def _maybe_set(obj, name, value):
    try:
        setattr(obj, name, value)
        return True
    except Exception:
        return False


def _ds_options(deterministic: bool):
    options = tf.data.Options()
    try:
        options.experimental_deterministic = deterministic
    except Exception:
        pass

    opt = options.experimental_optimization
    _maybe_set(opt, "apply_default_optimizations", True)
    _maybe_set(opt, "map_parallelization", True)
    _maybe_set(opt, "parallel_batch", True)
    _maybe_set(opt, "autotune_buffers", True)
    _maybe_set(opt, "filter_fusion", True)
    _maybe_set(opt, "map_and_batch_fusion", True)
    _maybe_set(opt, "inject_prefetch", True)
    _maybe_set(opt, "parallelize_interleave", True)
    return options


def _image_ds_from_dir(directory, batch_size, shuffle, seed):
    return keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        image_size=(h_target, w_target),
        interpolation="bilinear",
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )


def make_train_ds(df, y_arr, batch_size=32, shuffle=True, cache_path=None):
    filenames = df["image"].to_numpy(dtype=str, copy=False)

    if shuffle:
        rng = np.random.RandomState(SEED)
        perm = rng.permutation(len(filenames))
        filenames = filenames[perm]
        y_arr = y_arr[perm]

    img_ds = _image_ds_from_dir(
        TRAIN_IMG_DIR, batch_size=batch_size, shuffle=False, seed=SEED
    )
    paths = tf.strings.join([TRAIN_IMG_DIR, os.sep, tf.constant(filenames)])
    labels = tf.constant(y_arr.astype(np.float32, copy=False))

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(_ds_options(deterministic=False if shuffle else True))

    if shuffle:
        ds = ds.shuffle(
            buffer_size=SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True
        )

    @tf.function
    def _decode_resize_norm(path, lab):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
        img = tf.image.resize(
            img, [h_target, w_target], method="bilinear", antialias=False
        )
        img = tf.cast(img, tf.float32) * (1.0 / 255.0)
        img.set_shape([h_target, w_target, 3])
        return img, lab

    ds = ds.map(_decode_resize_norm, num_parallel_calls=AUTOTUNE)

    if USE_RAM_CACHE:
        ds = ds.cache()
    elif cache_path is not None and USE_DISK_CACHE:
        ds = ds.cache(cache_path)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_ds(df, batch_size=32, cache_path=None):
    images = df["image"].to_numpy(dtype=str, copy=False)
    paths = tf.strings.join([TEST_IMG_DIR, os.sep, tf.constant(images)])

    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.with_options(_ds_options(deterministic=True))

    @tf.function
    def _decode_resize_norm(path):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
        img = tf.image.resize(
            img, [h_target, w_target], method="bilinear", antialias=False
        )
        img = tf.cast(img, tf.float32) * (1.0 / 255.0)
        img.set_shape([h_target, w_target, 3])
        return img

    ds = ds.map(_decode_resize_norm, num_parallel_calls=AUTOTUNE)

    if USE_RAM_CACHE:
        ds = ds.cache()
    elif cache_path is not None and USE_DISK_CACHE:
        ds = ds.cache(cache_path)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


idx = np.arange(len(train))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
val_size = int(0.1 * len(train))
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

train_df = train.iloc[tr_idx].reset_index(drop=True)
val_df = train.iloc[val_idx].reset_index(drop=True)
y_train = y[tr_idx]
y_val = y[val_idx]

train_cache = os.path.join(
    CACHE_DIR, f"train_{h_target}x{w_target}_bs{batch_size}.cache"
)
val_cache = os.path.join(CACHE_DIR, f"val_{h_target}x{w_target}_bs{batch_size}.cache")
test_cache = os.path.join(CACHE_DIR, f"test_{h_target}x{w_target}_bs{batch_size}.cache")

assert os.path.isdir(TRAIN_IMG_DIR), f"Missing TRAIN_IMG_DIR: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing TEST_IMG_DIR: {TEST_IMG_DIR}"

train_ds = make_train_ds(
    train_df, y_train, batch_size=batch_size, shuffle=True, cache_path=train_cache
)
val_ds = make_train_ds(
    val_df, y_val, batch_size=batch_size, shuffle=False, cache_path=val_cache
)
test_ds = make_test_ds(submissions, batch_size=batch_size, cache_path=test_cache)


train_batches = (len(train_df) + batch_size - 1) // batch_size
val_batches = (len(val_df) + batch_size - 1) // batch_size
print("Train batches:", train_batches)
print("Val batches:", val_batches)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1984220123.py in <cell line: 0>()
    154 assert os.path.isdir(TEST_IMG_DIR), f"Missing TEST_IMG_DIR: {TEST_IMG_DIR}"
    155 
--> 156 train_ds = make_train_ds(
    157     train_df, y_train, batch_size=batch_size, shuffle=True, cache_path=train_cache
    158 )

/tmp/ipykernel_11/1984220123.py in make_train_ds(df, y_arr, batch_size, shuffle, cache_path)
     60         y_arr = y_arr[perm]
     61 
---> 62     img_ds = _image_ds_from_dir(
     63         TRAIN_IMG_DIR, batch_size=batch_size, shuffle=False, seed=SEED
     64     )

/tmp/ipykernel_11/1984220123.py in _image_ds_from_dir(directory, batch_size, shuffle, seed)
     35 def _image_ds_from_dir(directory, batch_size, shuffle, seed):
     36     # returns (images, inferred_label) where inferred_label is ignored later
---> 37     return keras.utils.image_dataset_from_directory(
     38         directory,
     39         labels="inferred",

/usr/local/lib/python3.11/dist-packages/keras/src/utils/image_dataset_utils.py in image_dataset_from_directory(directory, labels, label_mode, class_names, color_mode, batch_size, image_size, shuffle, seed, validation_split, subset, interpolation, follow_links, crop_to_aspect_ratio, pad_to_aspect_ratio, data_format, verbose)
    327         )
    328         if not image_paths:
--> 329             raise ValueError(
    330                 f"No images found in directory {directory}. "
    331                 f"Allowed formats: {ALLOWLIST_FORMATS}"

ValueError: No images found in directory ../input/plant-pathology-2021-fgvc8/train_images. Allowed formats: ('.bmp', '.gif', '.jpeg', '.jpg', '.png')

## === cell 5
base = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(h_target, w_target, 3),
    pooling="avg",
)
base.trainable = False  # keep training stable and within time

inp = keras.Input(shape=(h_target, w_target, 3))
x = inp
x = tf.keras.applications.resnet50.preprocess_input(x * 255.0)
x = base(x, training=False)
x = keras.layers.Dropout(0.2)(x)
out = keras.layers.Dense(len(classes), activation="sigmoid")(x)

model = keras.Model(inp, out)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="binary_crossentropy"
)

model.summary()



## === cell 6
EPOCHS = 3
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3781331062.py in <cell line: 0>()
      1 EPOCHS = 3
----> 2 history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
      3 

NameError: name 'train_ds' is not defined

## === cell 7
preds = model.predict(test_ds, verbose=1)
print("Preds shape:", preds.shape)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1486111183.py in <cell line: 0>()
----> 1 preds = model.predict(test_ds, verbose=1)
      2 print("Preds shape:", preds.shape)
      3 

NameError: name 'test_ds' is not defined

## === cell 8
thresh = 0.1

preds_np = np.asarray(preds)
n, c = preds_np.shape
cls_arr = np.asarray(classes, dtype=object)

top_idx = preds_np.argmax(axis=1)
top_labels = cls_arr[top_idx]

mask = preds_np >= thresh
has_any = mask.any(axis=1)
if not np.all(has_any):
    mask = mask.copy()
    mask[~has_any, :] = False
    mask[~has_any, top_idx[~has_any]] = True

healthy_i = classes.index("healthy") if "healthy" in classes else None
if healthy_i is not None:
    top_is_healthy = top_idx == healthy_i
else:
    top_is_healthy = np.zeros(n, dtype=bool)

if healthy_i is None:
    labels_per_row = [" ".join(map(str, cls_arr[row_mask])) for row_mask in mask]
else:
    chosen_has_healthy = mask[:, healthy_i]
    only_top = top_labels.astype(str)
    join_all = [" ".join(map(str, cls_arr[row_mask])) for row_mask in mask]
    labels_per_row = np.where(
        top_is_healthy,
        "healthy",
        np.where(chosen_has_healthy, only_top, np.asarray(join_all, dtype=object)),
    ).tolist()

submissions = submissions.copy()
submissions["labels"] = labels_per_row

submissions = submissions[["image", "labels"]]
submissions.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submissions.shape)
display(submissions.head())



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4080423129.py in <cell line: 0>()
      1 thresh = 0.1
      2 
----> 3 preds_np = np.asarray(preds)
      4 n, c = preds_np.shape
      5 cls_arr = np.asarray(classes, dtype=object)

NameError: name 'preds' is not defined

## === cell 9
assert os.path.exists("submission.csv")
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["image", "labels"]
assert len(chk) == len(submissions)
print(chk["labels"].head(10).tolist())
print("submission.csv OK")

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/1122907816.py in <cell line: 0>()
----> 1 assert os.path.exists("submission.csv")
      2 chk = pd.read_csv("submission.csv")
      3 assert list(chk.columns) == ["image", "labels"]
      4 assert len(chk) == len(submissions)
      5 print(chk["labels"].head(10).tolist())

AssertionError:
