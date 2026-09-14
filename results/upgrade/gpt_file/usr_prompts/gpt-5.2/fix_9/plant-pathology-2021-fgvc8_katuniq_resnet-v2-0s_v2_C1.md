# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd


import tensorflow as tf
import tensorflow.keras as keras

from sklearn.preprocessing import MultiLabelBinarizer

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
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception:
    pass

print("TF:", tf.__version__)
print("Num GPUs:", len(tf.config.list_physical_devices("GPU")))



## === cell 1
train = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
train.head()



## === cell 2
label_split = train.labels.apply(lambda x: x.split())
trans_label = MultiLabelBinarizer().fit(label_split)

class_names = list(trans_label.classes_)

labels = pd.DataFrame(trans_label.transform(label_split), columns=class_names)
labels.head()



## === cell 3
for label in labels.columns:
    print(label, labels[label].value_counts(normalize=True))



## === cell 4
submissions = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")
submissions.head()



## === cell 5
h_target = 224
w_target = 224
batch_size = 16

train_img_dir = (
    "../input/plant-pathology-2021-fgvc8/train_images"
    if os.path.exists("../input/plant-pathology-2021-fgvc8/train_images")
    else "../input/train_images"
)

test_img_dir = (
    "../input/plant-pathology-2021-fgvc8/test_images"
    if os.path.exists("../input/plant-pathology-2021-fgvc8/test_images")
    else "../input/test_images"
)

print("train_img_dir:", train_img_dir)
print("test_img_dir :", test_img_dir)



## === cell 6
train_df = train.copy()
for c in class_names:
    train_df[c] = labels[c].values

idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
val_size = int(0.1 * len(train_df))
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

trn_df = train_df.iloc[trn_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

AUTOTUNE = tf.data.AUTOTUNE
SHUFFLE_BUFFER = min(len(trn_df), 2048)


@tf.function
def _decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [h_target, w_target], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([h_target, w_target, 3])
    return img


@tf.function
def _decode_augment(path, seed2):
    img_bytes = tf.io.read_file(path)

    z = tf.random.stateless_uniform(
        [], seed=seed2 + tf.constant([2, 0], tf.int32), minval=0.9, maxval=1.1
    )

    shape = tf.io.extract_jpeg_shape(img_bytes)
    ih = tf.cast(shape[0], tf.float32)
    iw = tf.cast(shape[1], tf.float32)

    crop_h = tf.cast(tf.round(ih / z), tf.int32)
    crop_w = tf.cast(tf.round(iw / z), tf.int32)
    crop_h = tf.clip_by_value(crop_h, 1, tf.cast(shape[0], tf.int32))
    crop_w = tf.clip_by_value(crop_w, 1, tf.cast(shape[1], tf.int32))

    max_off_y = tf.cast(shape[0], tf.int32) - crop_h
    max_off_x = tf.cast(shape[1], tf.int32) - crop_w

    off_y = tf.cond(
        max_off_y > 0,
        lambda: tf.random.stateless_uniform(
            [],
            seed=seed2 + tf.constant([6, 0], tf.int32),
            minval=0,
            maxval=max_off_y + 1,
            dtype=tf.int32,
        ),
        lambda: tf.constant(0, dtype=tf.int32),
    )
    off_x = tf.cond(
        max_off_x > 0,
        lambda: tf.random.stateless_uniform(
            [],
            seed=seed2 + tf.constant([7, 0], tf.int32),
            minval=0,
            maxval=max_off_x + 1,
            dtype=tf.int32,
        ),
        lambda: tf.constant(0, dtype=tf.int32),
    )

    crop_window = tf.stack([off_y, off_x, crop_h, crop_w])
    img = tf.io.decode_and_crop_jpeg(img_bytes, crop_window, channels=3)

    img = tf.image.resize(
        img, [h_target, w_target], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0

    img = tf.image.stateless_random_flip_left_right(img, seed=seed2)

    max_dx = int(round(0.05 * w_target))
    max_dy = int(round(0.05 * h_target))
    dx = tf.random.stateless_uniform(
        [],
        seed=seed2 + tf.constant([3, 0], tf.int32),
        minval=-max_dx,
        maxval=max_dx + 1,
        dtype=tf.int32,
    )
    dy = tf.random.stateless_uniform(
        [],
        seed=seed2 + tf.constant([4, 0], tf.int32),
        minval=-max_dy,
        maxval=max_dy + 1,
        dtype=tf.int32,
    )
    img = tf.roll(img, shift=[dy, dx], axis=[0, 1])
    img.set_shape([h_target, w_target, 3])
    return img


def _dataset_options():
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    try:
        options.experimental_slack = True
    except Exception:
        pass
    return options


def _make_paths(img_dir, image_series):
    return (img_dir.rstrip("/") + "/" + image_series.astype(str)).to_numpy(dtype=str)


def make_train_ds(df):
    paths_np = _make_paths(train_img_dir, df["image"])
    paths = tf.convert_to_tensor(paths_np, dtype=tf.string)
    y = tf.convert_to_tensor(
        df[class_names].to_numpy(dtype=np.float32), dtype=tf.float32
    )

    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    ds = ds.shuffle(
        buffer_size=SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True
    )
    ds = ds.enumerate()
    ds = ds.with_options(_dataset_options())

    def _map_fn(i, data):
        p, label = data
        i = tf.cast(i, tf.int32)
        seed2 = tf.stack([i, tf.constant(SEED, tf.int32)], axis=0)
        img = _decode_augment(p, seed2)
        return img, label

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds(df):
    paths_np = _make_paths(train_img_dir, df["image"])
    paths = tf.convert_to_tensor(paths_np, dtype=tf.string)
    y = tf.convert_to_tensor(
        df[class_names].to_numpy(dtype=np.float32), dtype=tf.float32
    )

    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    ds = ds.with_options(_dataset_options())

    def _map_fn(p, label):
        img = _decode_resize(p)
        return img, label

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_ds(df):
    paths_np = _make_paths(test_img_dir, df["image"])
    paths = tf.convert_to_tensor(paths_np, dtype=tf.string)

    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.with_options(_dataset_options())

    def _map_fn(p):
        return _decode_resize(p)

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_generator = make_train_ds(trn_df)
val_generator = make_val_ds(val_df)
test_generator = make_test_ds(submissions)



## === cell 7
base = tf.keras.applications.ResNet50V2(
    include_top=False,
    weights="imagenet",
    input_shape=(h_target, w_target, 3),
    pooling="avg",
)
base.trainable = False  # keep core architecture; train only the head for speed

x = tf.keras.layers.Dropout(0.2)(base.output)
out = tf.keras.layers.Dense(len(class_names), activation="sigmoid", name="pred")(x)
model = tf.keras.Model(inputs=base.input, outputs=out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=3,
    verbose=1,
)

preds = model.predict(test_generator, verbose=1)
print(preds[:2])



## === cell 8
thresh = {
    "complex": 0.23,
    "frog_eye_leaf_spot": 0.23,
    "healthy": 0.23,
    "powdery_mildew": 0.23,
    "rust": 0.23,
    "scab": 0.23,
}

missing = set(class_names) - set(thresh.keys())
extra = set(thresh.keys()) - set(class_names)
if missing or extra:
    raise ValueError(
        f"Class mismatch. missing_in_thresh={missing}, extra_in_thresh={extra}"
    )



## === cell 9
preds_np = np.asarray(preds)
thr_vec = np.array([thresh[c] for c in class_names], dtype=preds_np.dtype)

sel = preds_np > thr_vec[None, :]  # (N, C) boolean
argmax_idx = preds_np.argmax(axis=1)

label_arr = np.array(class_names, dtype=object)

healthy_idx = (
    int(np.where(label_arr == "healthy")[0][0]) if "healthy" in class_names else -1
)

sel_count = sel.sum(axis=1)
healthy_selected = (
    sel[:, healthy_idx] if healthy_idx >= 0 else np.zeros(sel.shape[0], dtype=bool)
)

use_argmax = (sel_count == 0) | (healthy_selected & (sel_count > 1))

rows = np.empty(sel.shape[0], dtype=object)
rows[use_argmax] = label_arr[argmax_idx[use_argmax]]

need_join_idx = np.where(~use_argmax)[0]
for i in need_join_idx:
    parts = label_arr[sel[i]]
    rows[i] = " ".join(parts.tolist()).strip()

submissions["labels"] = rows
submissions = submissions[["image", "labels"]]
submissions.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submissions.shape)
print(submissions.head())



## === cell 10
submissions.head()
