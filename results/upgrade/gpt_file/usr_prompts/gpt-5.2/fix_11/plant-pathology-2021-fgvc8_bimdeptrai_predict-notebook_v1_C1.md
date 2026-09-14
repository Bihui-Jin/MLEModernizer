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

0.1578947368421052

# 6. Current score

0.71756

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.71756) has done: 'Main bottlenecks are input pipeline cost (JPEG decode/resize + expensive custom projective rotation) and per-epoch dataset re-creation/shuffle overhead; the model itself is lightweight. I keep the exact same model, loss, epochs, and augmentation semantics, but make the tf.data pipeline faster by (1) caching decoded+resized images (pre-augmentation) to avoid repeating JPEG decode/resize every epoch, (2) using `shuffle(..., reshuffle_each_iteration=False)` with a deterministic per-example seed that depends on `(epoch, index)` so augmentation still changes each epoch while keeping determinism, and (3) removing Python-level overhead in label splitting and submission label formatting via vectorized/compiled operations. These changes are equivalent in outputs up to negligible float diffs (same ops, same augmentation distributions), but cut redundant CPU work so it finishes within the 600s limit.'
- What this solution (achieved 0.71756) has done: 'The main timeout driver is the CPU-heavy training pipeline: per-step projective rotation/resize zoom inside `tf.data` plus deterministic execution forces slow single-threaded-ish kernels and high Python/graph overhead. To keep the exact same model and training semantics, the optimizations focus on (1) making the dataset pipeline fully graph-compiled and cheaper per element (cache decoded images in-memory instead of on-disk, fuse/enforce static shapes, and precompute constants), (2) reducing input-pipeline contention by setting `drop_remainder` consistently and avoiding repeated shape inference, and (3) ensuring we don’t do any redundant work at predict time. None of these change the architecture, loss, number of steps, augmentations used, or thresholds; they only remove overhead while preserving deterministic behavior.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF choose
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TF:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_PATH = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

train.head(), submissions.head(), train.shape, submissions.shape




## === cell 2
label_split = train["labels"].str.split(" ")
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_split)
classes = mlb.classes_.tolist()

labels_df = pd.DataFrame(y, columns=classes)
labels_df.head(), len(classes), classes




## === cell 3
train_df = train.copy()
for c in classes:
    train_df[c] = labels_df[c].values

idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)

val_frac = 0.1
val_size = int(len(idx) * val_frac)
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

tr_df.shape, va_df.shape




## === cell 4
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

AUTOTUNE = tf.data.AUTOTUNE

IMG_H = tf.constant(IMG_SIZE[0], tf.int32)
IMG_W = tf.constant(IMG_SIZE[1], tf.int32)
OUT_SHAPE = tf.constant([IMG_SIZE[0], IMG_SIZE[1]], dtype=tf.int32)
PI = tf.constant(np.pi, tf.float32)


def _build_paths_and_labels(df, img_dir, with_labels: bool):
    img_dir = str(img_dir).rstrip("/")
    paths = np.array(
        [f"{img_dir}/{img}" for img in df["image"].astype(str).values], dtype=str
    )
    if with_labels:
        labels = df[classes].values.astype(np.float32, copy=False)
        return paths, labels
    return paths


tr_paths, tr_y = _build_paths_and_labels(tr_df, TRAIN_IMG_DIR, with_labels=True)
va_paths, va_y = _build_paths_and_labels(va_df, TRAIN_IMG_DIR, with_labels=True)
te_paths = _build_paths_and_labels(submissions, TEST_IMG_DIR, with_labels=False)

tr_paths_tf = tf.constant(tr_paths)
va_paths_tf = tf.constant(va_paths)
te_paths_tf = tf.constant(te_paths)

tr_y_tf = tf.constant(tr_y)
va_y_tf = tf.constant(va_y)


@tf.function(reduce_retracing=True)
def _decode_resize_rescale(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    img = tf.ensure_shape(img, (IMG_SIZE[0], IMG_SIZE[1], 3))
    return img


@tf.function(reduce_retracing=True)
def _stateless_uniform(shape, seed):
    return tf.random.stateless_uniform(
        shape=shape, seed=seed, minval=0.0, maxval=1.0, dtype=tf.float32
    )


@tf.function(reduce_retracing=True)
def _rotate_projective(img, angle_rad):
    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)
    cx = (w - 1.0) / 2.0
    cy = (h - 1.0) / 2.0

    cos_a = tf.math.cos(angle_rad)
    sin_a = tf.math.sin(angle_rad)

    a0 = cos_a
    a1 = sin_a
    a2 = (1.0 - cos_a) * cx - sin_a * cy
    b0 = -sin_a
    b1 = cos_a
    b2 = sin_a * cx + (1.0 - cos_a) * cy

    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[None, :]

    img4 = img[None, ...]
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img4,
        transforms=transform,
        output_shape=OUT_SHAPE,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    out0 = out[0]
    out0 = tf.ensure_shape(out0, (IMG_SIZE[0], IMG_SIZE[1], 3))
    return out0


@tf.function(reduce_retracing=True)
def _augment(img, seed2):
    r = _stateless_uniform([], seed2)
    img = tf.cond(r < 0.5, lambda: tf.image.flip_left_right(img), lambda: img)

    angle = (
        _stateless_uniform([], seed2 + tf.constant([11, 17], tf.int32)) * 2.0 - 1.0
    ) * (15.0 * PI / 180.0)
    img = _rotate_projective(img, angle)

    z = (_stateless_uniform([], seed2 + tf.constant([23, 5], tf.int32)) * 0.2) + 0.9
    new_h = tf.cast(tf.round(z * tf.cast(IMG_H, tf.float32)), tf.int32)
    new_w = tf.cast(tf.round(z * tf.cast(IMG_W, tf.float32)), tf.int32)
    img2 = tf.image.resize(
        img, [new_h, new_w], method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img2 = tf.image.resize_with_crop_or_pad(img2, IMG_SIZE[0], IMG_SIZE[1])
    img2 = tf.ensure_shape(img2, (IMG_SIZE[0], IMG_SIZE[1], 3))
    return img2


def _make_train_ds_cached(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    ds = ds.map(
        lambda p, y: (_decode_resize_rescale(p), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    ).cache()

    ds = ds.shuffle(
        buffer_size=len(tr_df), seed=SEED, reshuffle_each_iteration=False
    ).repeat()

    @tf.function(reduce_retracing=True)
    def _map_fn(i, xy, epoch):
        img, y = xy
        seed2 = tf.stack(
            [tf.cast(SEED, tf.int32) + tf.cast(epoch, tf.int32), tf.cast(i, tf.int32)]
        )
        img = _augment(img, seed2)
        return img, y

    ds = ds.enumerate()

    return ds, _map_fn


def _make_eval_ds(paths, labels=None, drop_remainder=True):
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(
            _decode_resize_rescale, num_parallel_calls=AUTOTUNE, deterministic=True
        )
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(
        lambda p, y: (_decode_resize_rescale(p), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=drop_remainder)
    ds = ds.prefetch(AUTOTUNE)
    return ds


epoch_var = tf.Variable(0, dtype=tf.int32, trainable=False)
train_base_ds, _train_map_fn = _make_train_ds_cached(tr_paths_tf, tr_y_tf)

train_ds = (
    train_base_ds.map(
        lambda i, xy: _train_map_fn(i, xy, epoch_var),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTOTUNE)
)

val_ds = _make_eval_ds(va_paths_tf, va_y_tf, drop_remainder=True)
test_ds = _make_eval_ds(te_paths_tf, labels=None)




## === cell 5
base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, weights="imagenet"
)
base.trainable = False  # keep fast and stable

inp = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(inp, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
out = tf.keras.layers.Dense(len(classes), activation="sigmoid")(x)
model = tf.keras.Model(inp, out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()




## === cell 6
EPOCHS = 3

steps_per_epoch = len(tr_df) // BATCH_SIZE
validation_steps = len(va_df) // BATCH_SIZE


class _EpochVarCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        epoch_var.assign(epoch)


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
    callbacks=[_EpochVarCallback()],
)




## === cell 7
test_steps = int(np.ceil(len(submissions) / BATCH_SIZE))

preds = model.predict(
    test_ds,
    steps=test_steps,
    verbose=1,
)
preds = preds[: len(submissions)]  # safety
preds.shape, preds[:2]




## === cell 8
thresh = 0.5

mask = preds >= thresh
top_idx = np.argmax(preds, axis=1).astype(np.int32)

class_arr = np.asarray(classes, dtype=object)

pred_labels = []
for i in range(mask.shape[0]):
    cols = np.flatnonzero(mask[i])
    if cols.size:
        pred_labels.append(" ".join(class_arr[cols].tolist()))
    else:
        pred_labels.append(class_arr[int(top_idx[i])])

submissions_out = submissions.copy()
submissions_out["labels"] = pred_labels

submissions_out.to_csv("submission.csv", index=False)
submissions_out.head(), submissions_out.shape




## === cell 9
assert os.path.exists("submission.csv")
assert list(submissions_out.columns) == ["image", "labels"]
assert len(submissions_out) == len(submissions)
print("Wrote submission.csv with", len(submissions_out), "rows")
print(submissions_out.sample(5, random_state=SEED))
