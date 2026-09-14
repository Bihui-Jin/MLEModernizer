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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

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
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

print("TF:", tf.__version__)




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
h_target = 384
w_target = 384
batch_size = 32




## === cell 3
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(label_split)
class_names = list(mlb.classes_)

labels_df = pd.DataFrame(Y, columns=class_names)
print("Num classes:", len(class_names))
print("Classes:", class_names)

class_to_idx = {c: i for i, c in enumerate(class_names)}
healthy_idx = class_to_idx.get("healthy", None)
print("healthy_idx:", healthy_idx)




## === cell 4
train_with_y = train.copy()
for j, c in enumerate(class_names):
    train_with_y[c] = Y[:, j].astype(np.float32)

idx = np.arange(len(train_with_y))
rng = np.random.default_rng(SEED)
rng.shuffle(idx)
val_frac = 0.1
n_val = int(len(train_with_y) * val_frac)
val_idx = idx[:n_val]
trn_idx = idx[n_val:]

train_trn = train_with_y.iloc[trn_idx].reset_index(drop=True)
train_val = train_with_y.iloc[val_idx].reset_index(drop=True)

print("Train/Val:", train_trn.shape, train_val.shape)




## === cell 5
AUTOTUNE = tf.data.AUTOTUNE

train_paths = tf.constant(
    [os.path.join(TRAIN_IMG_DIR, f) for f in train_trn["image"].values.tolist()]
)
val_paths = tf.constant(
    [os.path.join(TRAIN_IMG_DIR, f) for f in train_val["image"].values.tolist()]
)
test_paths = tf.constant(
    [os.path.join(TEST_IMG_DIR, f) for f in submissions["image"].values.tolist()]
)

y_trn = train_trn[class_names].values.astype(np.float32)
y_val = train_val[class_names].values.astype(np.float32)

steps_per_epoch = int(np.ceil(len(train_trn) / batch_size))
val_steps = int(np.ceil(len(train_val) / batch_size))
test_steps = int(np.ceil(len(submissions) / batch_size))
print(
    "steps_per_epoch:",
    steps_per_epoch,
    "val_steps:",
    val_steps,
    "test_steps:",
    test_steps,
)


@tf.function
def _decode_resize_from_path(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, (h_target, w_target), method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _augment(img, idx):
    idx = tf.cast(idx, tf.int32)
    s0 = tf.stack([tf.cast(SEED, tf.int32), idx])

    img = tf.image.stateless_random_flip_left_right(img, seed=s0)

    angle = tf.random.stateless_uniform([], seed=s0 + 1, minval=-15.0, maxval=15.0) * (
        np.pi / 180.0
    )

    c2 = tf.cast(tf.cos(-angle), tf.float32)
    s2 = tf.cast(tf.sin(-angle), tf.float32)
    cx = tf.cast(w_target, tf.float32) / 2.0
    cy = tf.cast(h_target, tf.float32) / 2.0
    a0 = c2
    a1 = -s2
    a2 = cx - c2 * cx + s2 * cy
    b0 = s2
    b1 = c2
    b2 = cy - s2 * cx - c2 * cy
    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[None, :]
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=transform,
        output_shape=[h_target, w_target],
        interpolation="NEAREST",
        fill_mode="NEAREST",
        fill_value=0.0,
    )[0]

    max_dx = tf.cast(tf.round(0.05 * tf.cast(w_target, tf.float32)), tf.int32)
    max_dy = tf.cast(tf.round(0.05 * tf.cast(h_target, tf.float32)), tf.int32)
    dx = tf.random.stateless_uniform(
        [], seed=s0 + 2, minval=-max_dx, maxval=max_dx + 1, dtype=tf.int32
    )
    dy = tf.random.stateless_uniform(
        [], seed=s0 + 3, minval=-max_dy, maxval=max_dy + 1, dtype=tf.int32
    )

    transform = tf.stack(
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
    )[None, :]
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=transform,
        output_shape=[h_target, w_target],
        interpolation="NEAREST",
        fill_mode="NEAREST",
        fill_value=0.0,
    )[0]

    zoom = tf.random.stateless_uniform([], seed=s0 + 4, minval=0.9, maxval=1.1)
    new_h = tf.cast(tf.round(zoom * tf.cast(h_target, tf.float32)), tf.int32)
    new_w = tf.cast(tf.round(zoom * tf.cast(w_target, tf.float32)), tf.int32)
    img_zoom = tf.image.resize(
        img, (new_h, new_w), method=tf.image.ResizeMethod.BILINEAR
    )
    img_zoom = tf.image.resize_with_crop_or_pad(img_zoom, h_target, w_target)
    return img_zoom


def _with_options(ds):
    opts = tf.data.Options()
    opts.experimental_deterministic = True

    try:
        opts.threading.private_threadpool_size = 0  # let TF manage
        opts.threading.max_intra_op_parallelism = 0
    except Exception:
        pass

    try:
        opts.experimental_optimization.map_parallelization = True
    except Exception:
        pass
    try:
        opts.experimental_optimization.parallel_batch = True
    except Exception:
        pass
    if hasattr(opts.experimental_optimization, "autotune_buffers"):
        try:
            opts.experimental_optimization.autotune_buffers = True
        except Exception:
            pass
    if hasattr(opts.experimental_optimization, "map_fusion"):
        try:
            opts.experimental_optimization.map_fusion = True
        except Exception:
            pass
    if hasattr(opts.experimental_optimization, "apply_default_optimizations"):
        try:
            opts.experimental_optimization.apply_default_optimizations = True
        except Exception:
            pass
    try:
        opts.experimental_slack = True
    except Exception:
        pass

    return ds.with_options(opts)


_SHUFFLE_BUFFER = min(len(train_trn), 4096)


def _make_train_ds(paths, y):
    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    ds = ds.shuffle(
        buffer_size=_SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True
    )

    def _decode_pair(path, label):
        return _decode_resize_from_path(path), label

    ds = ds.map(_decode_pair, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)

    @tf.function
    def _augment_batch(imgs, labels):
        b = tf.shape(imgs)[0]
        idxs = tf.range(b, dtype=tf.int32)
        imgs_aug = tf.map_fn(
            lambda t: _augment(t[0], t[1]),
            (imgs, idxs),
            fn_output_signature=tf.float32,
            parallel_iterations=16,
        )
        return imgs_aug, labels

    ds = ds.map(_augment_batch, num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(AUTOTUNE)
    return _with_options(ds)


def _make_eval_ds(paths, y=None):
    paths_ds = tf.data.Dataset.from_tensor_slices(paths)
    img_ds = paths_ds.map(_decode_resize_from_path, num_parallel_calls=AUTOTUNE)

    if y is None:
        ds = img_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
        return _with_options(ds)

    lbls = tf.data.Dataset.from_tensor_slices(y)
    ds = (
        tf.data.Dataset.zip((img_ds, lbls))
        .batch(batch_size, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )
    return _with_options(ds)


train_ds = _make_train_ds(train_paths, y_trn)
val_ds = _make_eval_ds(val_paths, y_val)
test_ds = _make_eval_ds(test_paths, None)

print("Datasets built:", train_ds, val_ds, test_ds)




## === cell 6
inputs = keras.Input(shape=(h_target, w_target, 3))
base = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_tensor=inputs,
    pooling="avg",
)
x = base.output
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(len(class_names), activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

base.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()




## === cell 7
history1 = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=val_steps,
    epochs=2,
    verbose=1,
)

base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
)

history2 = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=val_steps,
    epochs=1,
    verbose=1,
)




## === cell 8
preds = model.predict(test_ds, steps=test_steps, verbose=1)
preds = preds[: len(submissions)]
print("preds shape:", preds.shape)
print(preds[:2])




## === cell 9
thresh = 0.2

p = preds
n, k = p.shape
chosen_mask = p >= thresh

argmax_idx = np.argmax(p, axis=1)
none_chosen = ~chosen_mask.any(axis=1)
chosen_mask[none_chosen, :] = False
chosen_mask[none_chosen, argmax_idx[none_chosen]] = True

if healthy_idx is not None:
    healthy_and_others = chosen_mask[:, healthy_idx] & (chosen_mask.sum(axis=1) > 1)
    chosen_mask[healthy_and_others, healthy_idx] = False

    if k > 1:
        p_others_max = np.max(np.delete(p, healthy_idx, axis=1), axis=1)
    else:
        p_others_max = np.zeros(n, dtype=p.dtype)
    force_healthy = (argmax_idx == healthy_idx) & (p_others_max < thresh)
    chosen_mask[force_healthy, :] = False
    chosen_mask[force_healthy, healthy_idx] = True

cn = np.asarray(class_names, dtype=object)
out_labels = [""] * n
rows = np.flatnonzero(chosen_mask.any(axis=1))
for i in rows:
    out_labels[i] = " ".join(cn[np.flatnonzero(chosen_mask[i])].tolist())

submissions_out = submissions.copy()
submissions_out["labels"] = out_labels
submissions_out = submissions_out[["image", "labels"]]
submissions_out.to_csv("submission.csv", index=False)

print(submissions_out.head())
print("Wrote submission.csv with shape:", submissions_out.shape)
