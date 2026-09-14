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

tf.config.experimental.enable_op_determinism()

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
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer().fit(label_split)
labels = pd.DataFrame(mlb.transform(label_split), columns=mlb.classes_)
labels.head()



## === cell 3
for label in labels.columns:
    vc = labels[label].value_counts(normalize=True)
    print(label, dict(vc))



## === cell 4
h_target = 256
w_target = 256
batch_size = 32

train_df = train.copy()
for c in labels.columns:
    train_df[c] = labels[c].values

idx = np.arange(len(train_df))
rng = np.random.default_rng(SEED)
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

df_tr = train_df.iloc[tr_idx].reset_index(drop=True)
df_va = train_df.iloc[va_idx].reset_index(drop=True)

print("Train/Val:", df_tr.shape, df_va.shape)



## === cell 5
y_cols = list(labels.columns)

AUTOTUNE = tf.data.AUTOTUNE
SHUFFLE_BUFFER = min(len(df_tr), 2048)

CACHE_DIR = "./tfdata_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

TRAIN_DECODE_CACHE = os.path.join(CACHE_DIR, "train_decode.cache")


def _remove_cache_prefix(prefix_path: str):
    d = os.path.dirname(prefix_path) or "."
    base = os.path.basename(prefix_path)
    if not os.path.isdir(d):
        return
    for fn in os.listdir(d):
        if fn == base or fn.startswith(base + "."):
            fp = os.path.join(d, fn)
            try:
                os.remove(fp)
            except FileNotFoundError:
                pass


_remove_cache_prefix(TRAIN_DECODE_CACHE)
_remove_cache_prefix(os.path.join(CACHE_DIR, "val.cache"))
_remove_cache_prefix(os.path.join(CACHE_DIR, "test.cache"))


def _build_paths_and_labels_np(df, img_dir, y_cols=None):
    paths = (img_dir + "/" + df["image"].astype(str)).to_numpy()
    if y_cols is None:
        y = None
    else:
        y = df[y_cols].to_numpy(dtype=np.float32, copy=False)
    return paths, y


tr_paths_np, tr_y_np = _build_paths_and_labels_np(df_tr, TRAIN_IMG_DIR, y_cols=y_cols)
va_paths_np, va_y_np = _build_paths_and_labels_np(df_va, TRAIN_IMG_DIR, y_cols=y_cols)
te_paths_np, _ = _build_paths_and_labels_np(submissions, TEST_IMG_DIR, y_cols=None)


@tf.function(reduce_retracing=True)
def _decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize_with_pad(
        img, h_target, w_target, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    img.set_shape([h_target, w_target, 3])
    return img


@tf.function(reduce_retracing=True)
def _augment(img, seed):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed)

    angle = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([1, 0], tf.int32), minval=-10.0, maxval=10.0
    ) * (np.pi / 180.0)
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(img, 0),
        transforms=tf.expand_dims(
            tf.stack(
                [
                    tf.cos(angle),
                    -tf.sin(angle),
                    0.0,
                    tf.sin(angle),
                    tf.cos(angle),
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            0,
        ),
        output_shape=tf.constant([h_target, w_target], dtype=tf.int32),
        fill_mode="REFLECT",
        interpolation="BILINEAR",
        fill_value=0.0,
    )
    img = tf.squeeze(img, 0)

    tx = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([2, 0], tf.int32), minval=-0.05, maxval=0.05
    ) * tf.cast(w_target, tf.float32)
    ty = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([3, 0], tf.int32), minval=-0.05, maxval=0.05
    ) * tf.cast(h_target, tf.float32)

    z = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([4, 0], tf.int32), minval=0.9, maxval=1.1
    )

    cx = (w_target - 1) / 2.0
    cy = (h_target - 1) / 2.0

    a0 = 1.0 / z
    a4 = 1.0 / z
    a2 = cx - cx / z - tx
    a5 = cy - cy / z - ty

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(img, 0),
        transforms=tf.expand_dims(tf.stack([a0, 0.0, a2, 0.0, a4, a5, 0.0, 0.0]), 0),
        output_shape=tf.constant([h_target, w_target], dtype=tf.int32),
        fill_mode="REFLECT",
        interpolation="BILINEAR",
        fill_value=0.0,
    )
    img = tf.squeeze(img, 0)
    img.set_shape([h_target, w_target, 3])
    return img


@tf.function(reduce_retracing=True)
def _seed_from_index(i):
    i = tf.cast(i, tf.int32)
    return tf.stack([tf.cast(SEED, tf.int32), i])


def _dataset_options(deterministic=False):
    opts = tf.data.Options()
    opts.deterministic = deterministic
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.map_and_batch_fusion = True
    opts.experimental_optimization.parallel_batch = True
    opts.threading.private_threadpool_size = 0
    return opts


def _maybe_prefetch_to_device(ds):
    try:
        gpus = tf.config.list_logical_devices("GPU")
    except Exception:
        gpus = []
    if gpus:
        ds = ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
    return ds


def make_train_ds(paths_np, y_np):
    base_ds = tf.data.Dataset.from_tensor_slices((paths_np, y_np))
    base_ds = base_ds.with_options(_dataset_options(deterministic=True))

    @tf.function(reduce_retracing=True)
    def _decode_map(path, label):
        return _decode_and_resize(path), label

    base_ds = base_ds.map(_decode_map, num_parallel_calls=AUTOTUNE)
    base_ds = base_ds.cache(TRAIN_DECODE_CACHE)

    base_ds = base_ds.shuffle(
        buffer_size=SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True
    )
    base_ds = base_ds.with_options(_dataset_options(deterministic=False))

    base_ds = base_ds.enumerate()

    @tf.function(reduce_retracing=True)
    def _aug_map(i, img_label):
        img, label = img_label
        seed = _seed_from_index(i)
        img = _augment(img, seed)
        return img, label

    base_ds = base_ds.map(_aug_map, num_parallel_calls=AUTOTUNE)
    base_ds = base_ds.batch(batch_size, drop_remainder=True)
    base_ds = _maybe_prefetch_to_device(base_ds)
    base_ds = base_ds.prefetch(AUTOTUNE)
    return base_ds


def make_eval_ds(paths_np, y_np, with_labels, cache_in_memory=False, cache_name=None):
    if with_labels:
        ds = tf.data.Dataset.from_tensor_slices((paths_np, y_np))

        @tf.function(reduce_retracing=True)
        def _map_fn(path, label):
            img = _decode_and_resize(path)
            return img, label

    else:
        ds = tf.data.Dataset.from_tensor_slices(paths_np)

        @tf.function(reduce_retracing=True)
        def _map_fn(path):
            img = _decode_and_resize(path)
            return img

    ds = ds.with_options(_dataset_options(deterministic=True))
    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)

    if cache_in_memory:
        ds = ds.cache()
    elif cache_name is not None:
        ds = ds.cache(os.path.join(CACHE_DIR, cache_name))

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = _maybe_prefetch_to_device(ds)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(tr_paths_np, tr_y_np)
val_ds = make_eval_ds(
    va_paths_np,
    va_y_np,
    with_labels=True,
    cache_in_memory=True,  # was disk cache; RAM cache is faster and equivalent
    cache_name=None,
)
test_ds = make_eval_ds(
    te_paths_np,
    None,
    with_labels=False,
    cache_in_memory=True,  # was disk cache; RAM cache is faster and equivalent
    cache_name=None,
)



## === cell 6
base = tf.keras.applications.ResNet50V2(
    include_top=False,
    weights="imagenet",
    input_shape=(h_target, w_target, 3),
    pooling="avg",
)
base.trainable = False  # keep training light and stable

inp = keras.Input(shape=(h_target, w_target, 3))
x = inp
x = tf.keras.layers.Lambda(lambda t: t * 2.0 - 1.0)(x)
x = base(x, training=False)
x = keras.layers.Dropout(0.2)(x)
out = keras.layers.Dense(len(y_cols), activation="sigmoid")(x)
model = keras.Model(inp, out)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()



## === cell 7
steps_per_epoch = len(df_tr) // batch_size
val_steps = (len(df_va) + batch_size - 1) // batch_size

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)

base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)



## === cell 8
preds = model.predict(test_ds, verbose=1)
print("preds shape:", preds.shape)
print("classes:", y_cols)



## === cell 9
thresh = {
    "complex": 0.25,
    "frog_eye_leaf_spot": 0.25,
    "healthy": 0.25,
    "powdery_mildew": 0.25,
    "rust": 0.25,
    "scab": 0.25,
}

class_to_idx = {c: i for i, c in enumerate(y_cols)}
th_labs = list(thresh.keys())
th_vals = np.array([thresh[k] for k in th_labs], dtype=np.float32)
th_indices = np.array([class_to_idx[k] for k in th_labs], dtype=np.int32)
healthy_idx = class_to_idx["healthy"]

argmax_idx = np.argmax(preds, axis=1).astype(np.int32)

p_sel = preds[:, th_indices] > th_vals[None, :]
has_label = p_sel.any(axis=1)

pred_labels = np.empty(len(submissions), dtype=object)

is_healthy_am = argmax_idx == healthy_idx
pred_labels[is_healthy_am] = "healthy"

not_healthy = ~is_healthy_am
needs_thresh = not_healthy & has_label
fallback = not_healthy & (~has_label)

pred_labels[fallback] = np.array(y_cols, dtype=object)[argmax_idx[fallback]]

needs_idx = np.flatnonzero(needs_thresh)
y_cols_arr = np.array(y_cols, dtype=object)
for i in needs_idx:
    idxs = np.flatnonzero(p_sel[i])
    label_comb = [th_labs[k] for k in idxs]
    final = " ".join(label_comb)
    if ("healthy" in label_comb) or (final == ""):
        final = y_cols_arr[argmax_idx[i]]
    pred_labels[i] = final

submissions["labels"] = pred_labels.tolist()
submissions.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submissions.shape)



## === cell 10
submissions.head()
