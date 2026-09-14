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
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ["TF_DETERMINISTIC_OPS"] = "1"
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass
try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TF version:", tf.__version__)



## === cell 1
DATA_DIR = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print(train.shape, submissions.shape)
train.head()



## === cell 2
label_split = train["labels"].str.split()
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_split)
labels = pd.DataFrame(y, columns=mlb.classes_)

print("Classes:", list(mlb.classes_))
labels.head()



## === cell 3
for label in labels.columns:
    vc = labels[label].value_counts(normalize=True)
    print(label, vc.to_dict())



## === cell 4
h_target = 256
w_target = 256
batch_size = 32



## === cell 5
train_df = train.copy()
train_df["filepath"] = TRAIN_IMG_DIR + "/" + train_df["image"].astype(str)

for c in mlb.classes_:
    train_df[c] = labels[c].values

idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
val_size = int(0.1 * len(train_df))
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

trn_df = train_df.iloc[trn_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

print("Train/Val:", trn_df.shape, val_df.shape)



## === cell 6
AUTOTUNE = tf.data.AUTOTUNE
target_cols = list(mlb.classes_)
num_classes = len(target_cols)

_RESCALE = tf.constant(1.0 / 255.0, dtype=tf.float32)


@tf.function
def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # dataset is jpg
    img = tf.image.resize(
        img,
        [h_target, w_target],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32) * _RESCALE
    return img


@tf.function
def _augment(img):
    angle = tf.random.stateless_uniform(
        [],
        seed=[SEED, tf.random.uniform([], maxval=2**31 - 1, dtype=tf.int32)],
        minval=-20.0,
        maxval=20.0,
        dtype=tf.float32,
    ) * (np.pi / 180.0)
    try:
        img = tf.image.rotate(img, angle, fill_mode="nearest")
    except Exception:
        pass

    max_dx = tf.cast(tf.round(0.10 * tf.cast(w_target, tf.float32)), tf.int32)
    max_dy = tf.cast(tf.round(0.10 * tf.cast(h_target, tf.float32)), tf.int32)
    dx = tf.random.stateless_uniform(
        [],
        seed=[SEED + 1, tf.random.uniform([], maxval=2**31 - 1, dtype=tf.int32)],
        minval=-max_dx,
        maxval=max_dx + 1,
        dtype=tf.int32,
    )
    dy = tf.random.stateless_uniform(
        [],
        seed=[SEED + 2, tf.random.uniform([], maxval=2**31 - 1, dtype=tf.int32)],
        minval=-max_dy,
        maxval=max_dy + 1,
        dtype=tf.int32,
    )
    try:
        img = tf.raw_ops.ImageProjectiveTransformV3(
            images=img[None, ...],
            transforms=tf.cast(
                [
                    [
                        1.0,
                        0.0,
                        -tf.cast(dx, tf.float32),
                        0.0,
                        1.0,
                        -tf.cast(dy, tf.float32),
                        0.0,
                        0.0,
                    ]
                ],
                tf.float32,
            ),
            output_shape=[h_target, w_target],
            fill_mode="NEAREST",
            interpolation="BILINEAR",
            fill_value=0.0,
        )[0]
    except Exception:
        pass

    zoom = tf.random.stateless_uniform(
        [],
        seed=[SEED + 3, tf.random.uniform([], maxval=2**31 - 1, dtype=tf.int32)],
        minval=0.90,
        maxval=1.10,
        dtype=tf.float32,
    )
    new_h = tf.cast(tf.round(tf.cast(h_target, tf.float32) / zoom), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(w_target, tf.float32) / zoom), tf.int32)
    new_h = tf.clip_by_value(new_h, 1, h_target)
    new_w = tf.clip_by_value(new_w, 1, w_target)
    if_zoom_in = zoom > 1.0

    def _zoom_in():
        cropped = tf.image.random_crop(img, size=[new_h, new_w, 3], seed=SEED)
        return tf.image.resize(
            cropped,
            [h_target, w_target],
            method=tf.image.ResizeMethod.BILINEAR,
            antialias=False,
        )

    def _zoom_out():
        padded = tf.image.resize_with_pad(
            img, new_h, new_w, method=tf.image.ResizeMethod.BILINEAR, antialias=False
        )
        return tf.image.resize(
            padded,
            [h_target, w_target],
            method=tf.image.ResizeMethod.BILINEAR,
            antialias=False,
        )

    img = tf.cond(if_zoom_in, _zoom_in, _zoom_out)

    img = tf.image.stateless_random_flip_left_right(
        img, seed=[SEED + 4, tf.random.uniform([], maxval=2**31 - 1, dtype=tf.int32)]
    )
    return img


def _with_fast_deterministic_options(ds):
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True
    opts.experimental_optimization.map_fusion = True
    try:
        opts.threading.private_threadpool_size = max(4, (os.cpu_count() or 4))
    except Exception:
        pass
    return ds.with_options(opts)


def make_train_ds(df):
    paths = df["filepath"].values.astype(str)
    y_arr = df[target_cols].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, y_arr))

    ds = ds.map(
        lambda p, y: (_decode_resize(p), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds = ds.cache()

    ds = ds.shuffle(
        buffer_size=min(len(df), 8192), seed=SEED, reshuffle_each_iteration=True
    )
    ds = ds.map(
        lambda img, y: (_augment(img), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    ds = _with_fast_deterministic_options(ds)
    return ds


def make_val_ds(df):
    paths = df["filepath"].values.astype(str)
    y_arr = df[target_cols].values.astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths, y_arr))
    ds = ds.map(
        lambda p, y: (_decode_resize(p), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.cache().batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    ds = _with_fast_deterministic_options(ds)
    return ds


def make_test_ds(df):
    paths = TEST_IMG_DIR + "/" + df["image"].values.astype(str)
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(
        lambda p: _decode_resize(p), num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds = ds.cache().batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    ds = _with_fast_deterministic_options(ds)
    return ds


train_ds = make_train_ds(trn_df)
val_ds = make_val_ds(val_df)
test_ds = make_test_ds(submissions)



## === cell 7
base = tf.keras.applications.DenseNet201(
    include_top=False,
    weights="imagenet",
    input_shape=(h_target, w_target, 3),
    pooling="avg",
)
x = base.output
x = tf.keras.layers.Dropout(0.25)(x)
out = tf.keras.layers.Dense(len(target_cols), activation="sigmoid")(x)
model = tf.keras.Model(inputs=base.input, outputs=out)

base.trainable = False
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(multi_label=True, name="auc")],
)

model.summary()



## === cell 8
steps_per_epoch = int(np.ceil(len(trn_df) / batch_size))
val_steps = int(np.ceil(len(val_df) / batch_size))

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)

base.trainable = True
for layer in base.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(multi_label=True, name="auc")],
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)



## === cell 9
preds = model.predict(test_ds, verbose=1)
print("preds shape:", preds.shape)
print(preds[:2])



## === cell 10
thresh = {
    "complex": 0.25,
    "frog_eye_leaf_spot": 0.25,
    "healthy": 0.25,
    "powdery_mildew": 0.25,
    "rust": 0.25,
    "scab": 0.25,
}

label_order = target_cols
assert set(thresh.keys()) == set(label_order), "Threshold keys must match class names."



## === cell 11
sub_df = submissions.copy()
preds_np = np.asarray(preds, dtype=np.float32)
label_order = list(label_order)
lab2i = {lab: i for i, lab in enumerate(label_order)}

healthy_i = lab2i["healthy"]
row_max_i = preds_np.argmax(axis=1)
is_healthy_max = row_max_i == healthy_i

thr_vec = np.array([thresh[lab] for lab in label_order], dtype=np.float32)
above = preds_np > thr_vec[None, :]

out_labels = np.empty(len(sub_df), dtype=object)
out_labels[is_healthy_max] = "healthy"

nonhealthy_idx = np.where(~is_healthy_max)[0]
for i in nonhealthy_idx:
    mask = above[i]
    if not mask.any():
        out_labels[i] = label_order[int(row_max_i[i])]
        continue
    idxs = np.flatnonzero(mask)
    comb = [label_order[j] for j in idxs]
    if "healthy" in comb and len(comb) > 1:
        comb = [l for l in comb if l != "healthy"]
    out_labels[i] = " ".join(comb)

sub_df["labels"] = out_labels.tolist()
sub_df[["image", "labels"]].to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with", len(sub_df), "rows")



## === cell 12
assert os.path.exists("submission.csv"), "submission.csv was not created"
check = pd.read_csv("submission.csv")
assert list(check.columns) == ["image", "labels"]
assert len(check) == len(submissions)
check.head()
