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

0.7879593721144982

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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

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

AUTOTUNE = tf.data.AUTOTUNE

try:
    options = tf.data.Options()
    options.deterministic = True
except Exception:
    options = None



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
train.head()



## === cell 2
label_split = train["labels"].str.split()
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_split)
classes = list(mlb.classes_)
labels = pd.DataFrame(y, columns=classes)
labels.head()



## === cell 3
h_target = 256
w_target = 256
batch_size = 32

idx = np.arange(len(train))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
val_frac = 0.1
val_size = int(len(idx) * val_frac)
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

train_df = train.iloc[tr_idx].reset_index(drop=True)
val_df = train.iloc[val_idx].reset_index(drop=True)

train_df[classes] = labels.loc[tr_idx, classes].to_numpy()
val_df[classes] = labels.loc[val_idx, classes].to_numpy()

train_df.head()




## === cell 4
def _read_decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img,
        (h_target, w_target),
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _augment(img, seed2):
    angle = tf.random.stateless_uniform([], seed=seed2, minval=-15.0, maxval=15.0) * (
        np.pi / 180.0
    )
    c = tf.math.cos(angle)
    s = tf.math.sin(angle)
    cx = (w_target - 1) / 2.0
    cy = (h_target - 1) / 2.0
    a0 = c
    a1 = -s
    a2 = cx - c * cx + s * cy
    b0 = s
    b1 = c
    b2 = cy - s * cx - c * cy
    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[None, :]
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=transform,
        output_shape=tf.constant([h_target, w_target], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]

    max_dx = tf.cast(tf.round(0.05 * tf.cast(w_target, tf.float32)), tf.int32)
    max_dy = tf.cast(tf.round(0.05 * tf.cast(h_target, tf.float32)), tf.int32)
    dx = tf.random.stateless_uniform(
        [],
        seed=seed2 + tf.constant([1, 0], tf.int32),
        minval=-max_dx,
        maxval=max_dx + 1,
        dtype=tf.int32,
    )
    dy = tf.random.stateless_uniform(
        [],
        seed=seed2 + tf.constant([0, 1], tf.int32),
        minval=-max_dy,
        maxval=max_dy + 1,
        dtype=tf.int32,
    )
    img = tf.roll(img, shift=[dy, dx], axis=[0, 1])

    scale = tf.random.stateless_uniform(
        [], seed=seed2 + tf.constant([2, 2], tf.int32), minval=0.9, maxval=1.1
    )
    new_h = tf.cast(tf.round(scale * tf.cast(h_target, tf.float32)), tf.int32)
    new_w = tf.cast(tf.round(scale * tf.cast(w_target, tf.float32)), tf.int32)
    img2 = tf.image.resize(img, (new_h, new_w), method="bilinear", antialias=False)
    img2 = tf.image.resize_with_crop_or_pad(img2, h_target, w_target)

    do_flip = tf.random.stateless_uniform(
        [], seed=seed2 + tf.constant([3, 3], tf.int32), minval=0.0, maxval=1.0
    )
    img2 = tf.cond(do_flip < 0.5, lambda: img2, lambda: tf.image.flip_left_right(img2))
    return img2


def _maybe_cache(ds, cache_name):
    try:
        return ds.cache()
    except Exception:
        cache_path = os.path.join("/kaggle/working", cache_name)
        return ds.cache(cache_path)


def make_ds(df, img_dir, training, cache_name):
    img_dir2 = img_dir.rstrip("/") + "/"
    img_names = df["image"].astype(str).to_numpy()
    paths_np = np.char.add(img_dir2, img_names).astype("U")

    ys_np = df[classes].to_numpy(dtype=np.float32)

    paths = tf.convert_to_tensor(paths_np, dtype=tf.string)
    ys = tf.convert_to_tensor(ys_np, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, ys))
    if options is not None:
        ds = ds.with_options(options)

    @tf.function
    def _decode_only(path, y):
        return _read_decode_resize(path), y

    ds = ds.map(_decode_only, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = _maybe_cache(ds, cache_name)

    if training:
        shuffle_buf = min(len(df), 4096)
        ds = ds.shuffle(
            buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
        )
        ds = ds.enumerate()

        @tf.function
        def _aug_map(i, xy):
            img, y = xy
            seed2 = tf.stack([tf.cast(SEED, tf.int32), tf.cast(i, tf.int32)])
            img = _augment(img, seed2)
            return img, y

        ds = ds.map(_aug_map, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


train_gen = make_ds(
    train_df, TRAIN_IMG_DIR, training=True, cache_name="cache_train_decode_resize"
)
val_gen = make_ds(
    val_df, TRAIN_IMG_DIR, training=False, cache_name="cache_val_decode_resize"
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1378145400.py in <cell line: 0>()
    119 
    120 
--> 121 train_gen = make_ds(
    122     train_df, TRAIN_IMG_DIR, training=True, cache_name="cache_train_decode_resize"
    123 )

/tmp/ipykernel_11/1378145400.py in make_ds(df, img_dir, training, cache_name)
     81     img_dir2 = img_dir.rstrip("/") + "/"
     82     img_names = df["image"].astype(str).to_numpy()
---> 83     paths_np = np.char.add(img_dir2, img_names).astype("U")
     84 
     85     ys_np = df[classes].to_numpy(dtype=np.float32)

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in add(x1, x2)
    330         # object dtype itemsize as num chars (worked on short strings).
    331         # bytes + void worked but promoting void->bytes is dubious also.
--> 332         raise TypeError(
    333             "np.char.add() requires both arrays of the same dtype kind, but "
    334             f"got dtypes: '{arr1.dtype}' and '{arr2.dtype}' (the few cases "

TypeError: np.char.add() requires both arrays of the same dtype kind, but got dtypes: '<U49' and 'object' (the few cases where this used to work often lead to incorrect results).

## === cell 5
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(h_target, w_target, 3),
)
base.trainable = False  # keep training stable and fast under 600s

inp = keras.Input(shape=(h_target, w_target, 3))
x = base(inp, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
out = keras.layers.Dense(len(classes), activation="sigmoid")(x)
model = keras.Model(inp, out)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()



## === cell 6
steps_per_epoch = int(np.ceil(len(train_df) / batch_size))
val_steps = int(np.ceil(len(val_df) / batch_size))

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=3,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/301554110.py in <cell line: 0>()
      3 
      4 history = model.fit(
----> 5     train_gen,
      6     validation_data=val_gen,
      7     epochs=3,

NameError: name 'train_gen' is not defined

## === cell 7
submissions = pd.read_csv(SAMPLE_SUB)
submissions.head()



## === cell 8
test_dir2 = TEST_IMG_DIR.rstrip("/") + "/"
test_img_names = submissions["image"].astype(str).to_numpy()
test_paths_np = np.char.add(test_dir2, test_img_names).astype("U")
test_paths = tf.convert_to_tensor(test_paths_np, dtype=tf.string)

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
if options is not None:
    test_ds = test_ds.with_options(options)


@tf.function
def _read_only(p):
    return _read_decode_resize(p)


test_ds = test_ds.map(_read_only, num_parallel_calls=AUTOTUNE, deterministic=True)
test_ds = _maybe_cache(test_ds, "cache_test_decode_resize")
test_ds = test_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

preds = model.predict(test_ds, verbose=1)
preds.shape



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3799328746.py in <cell line: 0>()
      2 test_dir2 = TEST_IMG_DIR.rstrip("/") + "/"
      3 test_img_names = submissions["image"].astype(str).to_numpy()
----> 4 test_paths_np = np.char.add(test_dir2, test_img_names).astype("U")
      5 test_paths = tf.convert_to_tensor(test_paths_np, dtype=tf.string)
      6 

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in add(x1, x2)
    330         # object dtype itemsize as num chars (worked on short strings).
    331         # bytes + void worked but promoting void->bytes is dubious also.
--> 332         raise TypeError(
    333             "np.char.add() requires both arrays of the same dtype kind, but "
    334             f"got dtypes: '{arr1.dtype}' and '{arr2.dtype}' (the few cases "

TypeError: np.char.add() requires both arrays of the same dtype kind, but got dtypes: '<U48' and 'object' (the few cases where this used to work often lead to incorrect results).

## === cell 9
thresh = {
    "complex": 0.25,
    "frog_eye_leaf_spot": 0.7,
    "healthy": 0.25,
    "powdery_mildew": 0.25,
    "rust": 0.5,
    "scab": 0.5,
}

thr_vec = np.array([thresh[c] for c in classes], dtype=np.float32)



## === cell 10
pred_bin = preds >= thr_vec[None, :]

healthy_idx = classes.index("healthy") if "healthy" in classes else None

row_max_idx = np.argmax(preds, axis=1)
if healthy_idx is not None:
    is_healthy_max = row_max_idx == healthy_idx
else:
    is_healthy_max = np.zeros((preds.shape[0],), dtype=bool)

chosen_mask = pred_bin.copy()

empty = chosen_mask.sum(axis=1) == 0
if healthy_idx is not None:
    includes_healthy = chosen_mask[:, healthy_idx]
else:
    includes_healthy = np.zeros((preds.shape[0],), dtype=bool)

need_fallback = empty | includes_healthy
row_max = preds.max(axis=1, keepdims=True)
fallback_mask = preds >= row_max  # includes ties

final_mask = np.where(need_fallback[:, None], fallback_mask, chosen_mask)

classes_arr = np.asarray(classes, dtype=object)
pred_labels = [""] * preds.shape[0]
for i in range(preds.shape[0]):
    if is_healthy_max[i]:
        pred_labels[i] = "healthy"
        continue
    labs = classes_arr[final_mask[i]]
    lab_str = " ".join(labs.tolist()).strip()
    pred_labels[i] = lab_str if lab_str != "" else "healthy"

submissions = submissions.copy()
submissions["labels"] = pred_labels
submissions[["image", "labels"]].to_csv("submission.csv", index=False)

submissions.head()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3348025794.py in <cell line: 0>()
----> 1 pred_bin = preds >= thr_vec[None, :]
      2 
      3 healthy_idx = classes.index("healthy") if "healthy" in classes else None
      4 
      5 row_max_idx = np.argmax(preds, axis=1)

NameError: name 'preds' is not defined

## === cell 11
assert os.path.exists("submission.csv")
check = pd.read_csv("submission.csv")
print(check.shape)
print(check.columns.tolist())
print(check.head())
print("Unique label strings (sample):", check["labels"].head(10).tolist())

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/3046407804.py in <cell line: 0>()
----> 1 assert os.path.exists("submission.csv")
      2 check = pd.read_csv("submission.csv")
      3 print(check.shape)
      4 print(check.columns.tolist())
      5 print(check.head())

AssertionError:
