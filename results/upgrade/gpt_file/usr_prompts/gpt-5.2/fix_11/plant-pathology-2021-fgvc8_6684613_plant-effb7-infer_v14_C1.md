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

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TF version:", tf.__version__)

AUTOTUNE = tf.data.AUTOTUNE




## === cell 1
def auto_select_accelerator():
    """
    TPU/GPU/CPU strategy selector.
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except Exception:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy




## === cell 2
load_dir = "/kaggle/input/plant-pathology-2021-fgvc8/"
train_csv_path = os.path.join(load_dir, "train.csv")
test_dir = os.path.join(load_dir, "test_images")
sample_sub_path = os.path.join(load_dir, "sample_submission.csv")

IMSIZES = (224, 240, 260, 300, 380, 456, 528, 600)
im_size = IMSIZES[7]  # keep original choice (600)

strategy = auto_select_accelerator()
BATCH_SIZE = 32

df = pd.read_csv(train_csv_path)

sample_df = pd.read_csv(sample_sub_path)[["image"]].copy()

if os.path.isdir(test_dir):
    test_images_set = set(
        f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if (
        sample_df["image"].isin(list(test_images_set)[:3]).any()
        or len(test_images_set) >= 3
    ):
        test_df = sample_df
    else:
        test_images = sorted(list(test_images_set))
        test_df = pd.DataFrame({"image": test_images})
else:
    test_df = sample_df

n_labels = 5

print("Train rows:", len(df), "Test rows:", len(test_df))
print("Example test images:", test_df["image"].head().tolist())
assert len(test_df) > 0




## === cell 3
def _build_test_dataset_cached(test_df, test_dir, im_size, batch_size):
    paths = tf.constant([os.path.join(test_dir, f) for f in test_df["image"].tolist()])

    def _read_decode_resize_to_float(path):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
        img = tf.image.resize(
            img,
            [im_size, im_size],
            method=tf.image.ResizeMethod.BILINEAR,
            antialias=False,
        )
        return tf.cast(img, tf.float32)

    ds = tf.data.Dataset.from_tensor_slices(paths)

    options = tf.data.Options()
    options.experimental_deterministic = True
    try:
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.autotune_buffers = True
    except Exception:
        pass
    ds = ds.with_options(options)

    ds = ds.map(_read_decode_resize_to_float, num_parallel_calls=AUTOTUNE)

    cache_path = os.path.join(
        "/kaggle/working",
        f"pp2021_test_cache_{os.path.basename(os.path.normpath(test_dir))}_{im_size}.tfdata",
    )
    ds = ds.cache(cache_path)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = _build_test_dataset_cached(test_df, test_dir, im_size, BATCH_SIZE)




## === cell 4
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

try:
    from tensorflow.keras import mixed_precision

    _has_accel = bool(tf.config.list_physical_devices("GPU")) or (
        len(tf.config.list_logical_devices("TPU")) > 0
    )
    if _has_accel:
        mixed_precision.set_global_policy("mixed_float16")
        print("Enabled mixed precision policy:", mixed_precision.global_policy())
    else:
        print("Mixed precision not enabled (no GPU/TPU detected).")
except Exception as e:
    print("Mixed precision not enabled:", repr(e))

with strategy.scope():
    base = tf.keras.applications.EfficientNetB7(
        weights=None,  # keep as original
        include_top=False,
        input_shape=(im_size, im_size, 3),
    )

    model = Sequential(
        [
            base,
            GlobalAveragePooling2D(),
            Dense(n_labels, activation="sigmoid", dtype="float32"),
        ]
    )


def _find_weights_file():
    candidates = [
        "/kaggle/input/effnetb7-2/besteffb7_2.h5",
        "/kaggle/input/effnetb7-2/best_effb7_2.h5",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    target_names = {"besteffb7_2.h5"}
    for root, _, files in os.walk("/kaggle/input"):
        for fn in files:
            if fn in target_names:
                return os.path.join(root, fn)
    return None


weights_path = _find_weights_file()
if weights_path is None:
    raise FileNotFoundError(
        "Required weights file 'besteffb7_2.h5' not found under /kaggle/input. "
        "Upload/attach the dataset containing this weights file (e.g., 'effnetb7-2') "
        "so the notebook produces a meaningful submission score."
    )

model.load_weights(weights_path)
print("Loaded weights:", weights_path)




## === cell 5
@tf.function
def _augment_and_preprocess(batch_images):
    x = batch_images
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.random_brightness(x, max_delta=0.2)
    x = tf.clip_by_value(x, 0.0, 255.0)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    return x


@tf.function
def _predict_tta_step_vectorized(batch_images, tta: tf.Tensor):
    tta = tf.cast(tta, tf.int32)
    b = tf.shape(batch_images)[0]

    x = tf.tile(batch_images, [tta, 1, 1, 1])
    x = _augment_and_preprocess(x)
    y = model(x, training=False)  # (tta*b, n_labels), float32 due to final Dense dtype
    y = tf.cast(y, tf.float32)

    y = tf.reshape(y, [tta, b, n_labels])
    return tf.reduce_mean(y, axis=0)


@tf.function
def _tta_predict_tf(ds, tta: tf.Tensor):
    preds_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    i = tf.constant(0, tf.int32)
    for batch in ds:
        preds_ta = preds_ta.write(i, _predict_tta_step_vectorized(batch, tta))
        i += 1
    return preds_ta.concat()


def _tta_predict(test_ds, tta):
    tta_t = tf.constant(int(tta), dtype=tf.int32)
    return _tta_predict_tf(test_ds, tta_t).numpy()


TTA = 3
pred = _tta_predict(test_ds, TTA)

print("Pred shape:", pred.shape)
assert pred.shape[0] == len(test_df) and pred.shape[1] == n_labels




## === cell 6
name = {
    0: "complex",
    1: "scab",
    2: "frog_eye_leaf_spot",
    3: "rust",
    4: "powdery_mildew",
}
healthy_label = "healthy"

threshold = {0: 0.25, 1: 0.35, 2: 0.8, 3: 0.8, 4: 0.8}

thr = np.array([threshold[i] for i in range(n_labels)], dtype=np.float32)
mask = pred > thr[None, :]

class_names = np.array([name[i] for i in range(n_labels)], dtype=object)
pred_string = []
for row in mask:
    labels = class_names[row].tolist()
    pred_string.append(" ".join(labels) if labels else healthy_label)

test_df["labels"] = pred_string

sub_path = "submission.csv"
test_df[["image", "labels"]].to_csv(sub_path, index=False)
print("Wrote", sub_path)
test_df.head()




## === cell 7
assert os.path.exists("submission.csv")
sub = pd.read_csv("submission.csv")
assert list(sub.columns) == ["image", "labels"]
assert len(sub) == len(test_df)
print(sub.head())
print("Unique labels examples:", sub["labels"].head(20).tolist())
