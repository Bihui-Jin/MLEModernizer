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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

import numpy as np
import pandas as pd
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TF version:", tf.__version__)




## === cell 1
def auto_select_accelerator():
    """
    Reference:
        * https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu
        * https://www.kaggle.com/xhlulu/ranzcr-efficientnet-tpu-training
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except Exception:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy




## === cell 2
CANDIDATE_ROOTS = [
    "/kaggle/input/plant-pathology-2021-fgvc8/",
    "/kaggle/input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8/",
]
load_dir = None
for p in CANDIDATE_ROOTS:
    if os.path.exists(os.path.join(p, "train.csv")) and os.path.isdir(
        os.path.join(p, "test_images")
    ):
        load_dir = p
        break
if load_dir is None:
    raise FileNotFoundError(
        "Could not find train.csv under expected Kaggle input paths."
    )

IMSIZES = (224, 240, 260, 300, 380, 456, 528, 600)
im_size = IMSIZES[7]

df = pd.read_csv(os.path.join(load_dir, "train.csv"))

strategy = auto_select_accelerator()

BATCH_SIZE = 32

test_dir = os.path.join(load_dir, "test_images")
if not os.path.isdir(test_dir):
    raise FileNotFoundError(f"Missing test_images directory at: {test_dir}")

test_df = pd.DataFrame({"image": sorted(os.listdir(test_dir))})

n_labels = 5

print("Dataset root:", load_dir)
print("Train rows:", len(df), "Test rows:", len(test_df))




## === cell 3
AUTOTUNE = tf.data.AUTOTUNE

_tfdata_opts = tf.data.Options()
_tfdata_opts.experimental_deterministic = False
try:
    _tfdata_opts.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass
_tfdata_opts.experimental_optimization.map_parallelization = True
_tfdata_opts.experimental_optimization.parallel_batch = True
_tfdata_opts.experimental_optimization.map_and_batch_fusion = True
try:
    _tfdata_opts.experimental_slack = True
except Exception:
    pass

test_paths = tf.constant([os.path.join(test_dir, f) for f in test_df["image"].values])


@tf.function
def _read_bytes(path):
    return tf.io.read_file(path)


@tf.function
def _decode_and_resize_from_bytes(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)  # exact decode; no ratio downscale
    img = tf.image.resize(
        img, [im_size, im_size], method=tf.image.ResizeMethod.BILINEAR, antialias=True
    )
    img = tf.cast(img, tf.float32)
    return img


cache_path_bytes = os.path.join("/kaggle/working", f"pp2021_test_jpegbytes.cache")

bytes_ds = tf.data.Dataset.from_tensor_slices(test_paths)
bytes_ds = bytes_ds.map(_read_bytes, num_parallel_calls=AUTOTUNE)
bytes_ds = bytes_ds.cache(cache_path_bytes)
bytes_ds = bytes_ds.with_options(_tfdata_opts)

base_ds = bytes_ds.map(_decode_and_resize_from_bytes, num_parallel_calls=AUTOTUNE)
base_ds = base_ds.with_options(_tfdata_opts)

_batched_base = base_ds.batch(BATCH_SIZE, drop_remainder=False).with_options(
    _tfdata_opts
)


@tf.function(jit_compile=True)
def _augment_and_preprocess_batch(imgs, seed0, seed1):
    seeds = tf.stack([tf.cast(seed0, tf.int32), tf.cast(seed1, tf.int32)], axis=0)

    imgs = tf.image.stateless_random_flip_left_right(imgs, seed=seeds)
    imgs = tf.image.stateless_random_flip_up_down(
        imgs, seed=seeds + tf.constant([1, 7], tf.int32)
    )
    imgs = tf.image.stateless_random_brightness(
        imgs, max_delta=0.2, seed=seeds + tf.constant([3, 11], tf.int32)
    )

    imgs = tf.clip_by_value(imgs, 0.0, 255.0)
    imgs = tf.keras.applications.efficientnet.preprocess_input(imgs)
    return imgs


def make_test_ds(tta_idx=0):
    tta_idx = int(tta_idx)
    seed0 = tf.constant(42, tf.int32)
    seed1 = tf.constant(1000 + tta_idx, tf.int32)
    ds = _batched_base.map(
        lambda x: _augment_and_preprocess_batch(x, seed0, seed1),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.with_options(_tfdata_opts)
    return ds


test_steps = int(np.ceil(len(test_df) / BATCH_SIZE))
print("Test steps:", test_steps)
print("Bytes cache path:", cache_path_bytes)




## === cell 4
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

WEIGHTS_PATH = "/kaggle/input/effnetb7-2/besteffb7_2.h5"

with strategy.scope():
    base_weights = None
    if not os.path.exists(WEIGHTS_PATH):
        base_weights = "imagenet"

    base = tf.keras.applications.EfficientNetB7(
        weights=base_weights,
        include_top=False,
        input_shape=(im_size, im_size, 3),
    )

    model = Sequential(
        [
            base,
            GlobalAveragePooling2D(),
            Dense(n_labels, activation="sigmoid"),
        ]
    )

if os.path.exists(WEIGHTS_PATH):
    model.load_weights(WEIGHTS_PATH)
    print("Loaded weights:", WEIGHTS_PATH)
else:
    print("WARNING: Weight file not found:", WEIGHTS_PATH)
    print("Using EfficientNetB7 imagenet weights as fallback.")

try:
    model.compile(run_eagerly=False)
except Exception:
    pass




## === cell 5
TTA = 3


def _maybe_stage_to_device(ds):
    try:
        gpus = tf.config.list_logical_devices("GPU")
        if gpus:
            dev = gpus[0].name
            ds = ds.apply(tf.data.experimental.copy_to_device(dev))
            ds = ds.prefetch(tf.data.AUTOTUNE)
            return ds
    except Exception:
        pass
    return ds


pred_sum = None
for t in range(TTA):
    ds = _maybe_stage_to_device(make_test_ds(tta_idx=t))
    p = model.predict(
        ds,
        steps=test_steps,
        verbose=1,
    )
    if pred_sum is None:
        pred_sum = p
    else:
        pred_sum += p

pred = pred_sum / float(TTA)
print("Pred shape:", pred.shape)




## === cell 6
name = {
    0: "complex",
    1: "scab",
    2: "frog_eye_leaf_spot",
    3: "rust",
    4: "powdery_mildew",
}
healthy_label = "healthy"

threshold = {
    0: 0.25,
    1: 0.35,
    2: 0.7,
    3: 0.8,
    4: 0.8,
}

thr = np.array([threshold[i] for i in range(n_labels)], dtype=pred.dtype)
mask = pred > thr  # (N, n_labels)

label_names = np.array([name[i] for i in range(n_labels)], dtype=object)

pred_string = []
for row in mask:
    if row.any():
        pred_string.append(" ".join(label_names[row].tolist()))
    else:
        pred_string.append(healthy_label)

sub = pd.DataFrame({"image": test_df["image"].values, "labels": pred_string})
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)

assert list(sub.columns) == ["image", "labels"]
assert sub["image"].isna().sum() == 0
assert sub["labels"].isna().sum() == 0
assert (sub["labels"].str.len() > 0).all()
print(sub["labels"].value_counts().head(10))
