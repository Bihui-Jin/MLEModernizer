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
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import pandas as pd
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
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
    Tries TPU first; falls back to default strategy on CPU/GPU.
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
IMSIZES = (224, 240, 260, 300, 380, 456, 528, 600)
im_size = IMSIZES[7]

load_dir = "/kaggle/input/plant-pathology-2021-fgvc8/"
train_csv_path = os.path.join(load_dir, "train.csv")
df = pd.read_csv(train_csv_path)

strategy = auto_select_accelerator()


def _pick_batch_size(strategy, base=32):
    r = int(getattr(strategy, "num_replicas_in_sync", 1) or 1)
    if "TPU" in strategy.__class__.__name__.upper():
        return max(base, 128 * r)
    if r > 1:
        return max(base, 64 * r)
    return max(base, 64)


BATCH_SIZE = _pick_batch_size(strategy, base=32)
print("Inference BATCH_SIZE:", BATCH_SIZE)

test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
test_df = pd.DataFrame({"image": sorted(os.listdir(test_dir))})

n_labels = 5

print("Train rows:", len(df), "Test rows:", len(test_df))



## === cell 3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

with strategy.scope():
    base = tf.keras.applications.EfficientNetB7(
        weights=None,
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

weights_path = "/kaggle/input/effnetb7-2/besteffb7_2.h5"
if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print(f"Loaded weights: {weights_path}")
else:
    print(
        f"WARNING: weights not found at {weights_path}. "
        "Proceeding with randomly initialized model (submission will be valid but low-scoring)."
    )



## === cell 4
AUTOTUNE = tf.data.AUTOTUNE


def _build_test_dataset_tta_batched(image_names, directory, batch_size):
    directory = directory.rstrip("/")
    paths = tf.strings.join(
        [tf.constant(directory + "/", dtype=tf.string), image_names]
    )

    def _load(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(
            img, (im_size, im_size), method=tf.image.ResizeMethod.BILINEAR
        )
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img

    def _tta_stack(img):
        return tf.stack(
            [img, tf.image.flip_left_right(img), tf.image.flip_up_down(img)], axis=0
        )

    ds = tf.data.Dataset.from_tensor_slices(paths)

    options = tf.data.Options()
    options.experimental_deterministic = True
    try:
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        options.experimental_optimization.autotune_buffers = True
        options.threading.private_threadpool_size = 32
        options.threading.max_intra_op_parallelism = 0
    except Exception:
        pass
    ds = ds.with_options(options)

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.map(_tta_stack, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


@tf.function(reduce_retracing=True)
def _infer_tta_batch(x_b3hwc):
    b = tf.shape(x_b3hwc)[0]
    x = tf.reshape(x_b3hwc, (b * 3, im_size, im_size, 3))
    y = model(x, training=False)  # [B*3, n_labels]
    y = tf.reshape(y, (b, 3, n_labels))
    return tf.reduce_mean(y, axis=1)  # [B, n_labels]


def predict_with_tta_tfdata_batched(test_ds, n_test, verbose=1):
    if verbose:
        print("TTA inference (batched-fused): starting...")
    out = np.empty((n_test, n_labels), dtype=np.float32)
    idx = 0
    for batch in test_ds:
        y = _infer_tta_batch(batch)
        y_np = y.numpy()
        bs = y_np.shape[0]
        out[idx : idx + bs] = y_np
        idx += bs
    if idx != n_test:
        out = out[:n_test]
    if verbose:
        print(f"TTA inference: done. Processed {out.shape[0]} images")
    return out


test_image_names_tf = tf.constant(test_df["image"].values, dtype=tf.string)
test_ds = _build_test_dataset_tta_batched(test_image_names_tf, test_dir, BATCH_SIZE)

pred = predict_with_tta_tfdata_batched(test_ds, n_test=int(len(test_df)), verbose=1)
print("Prediction shape:", pred.shape)



## === cell 5
name = {
    0: "complex",
    1: "scab",
    2: "frog_eye_leaf_spot",
    3: "rust",
    4: "powdery_mildew",
    5: "healthy",
}

threshold = {i: 0.25 for i in range(n_labels)}

pred_bin = (
    pred > np.array([threshold[i] for i in range(n_labels)], dtype=pred.dtype)[None, :]
)
label_names = np.array([name[i] for i in range(n_labels)], dtype=object)

parts = label_names[None, :] * pred_bin
parts = np.where(pred_bin, parts, "")
pred_string = np.array([" ".join(row[row != ""]) for row in parts], dtype=object)
pred_string[pred_string == ""] = name[5]

test_df["labels"] = pred_string.tolist()

sub_path = "submission.csv"
test_df[["image", "labels"]].to_csv(sub_path, index=False)
print("Wrote:", sub_path)
test_df.head()



## === cell 6
assert os.path.exists("submission.csv")
sub = pd.read_csv("submission.csv")
assert list(sub.columns) == ["image", "labels"]
assert len(sub) == len(test_df)
print(sub.head())
