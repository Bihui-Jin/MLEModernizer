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
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf

print("TensorFlow:", tf.__version__)

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
    print("XLA JIT: enabled")
except Exception as e:
    print("XLA JIT: not enabled:", type(e).__name__)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## === cell 1
def auto_select_accelerator():
    """
    TPU if available, else default strategy (GPU/CPU).
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except Exception as e:
        strategy = tf.distribute.get_strategy()
        print("TPU not found, using default strategy. Reason:", type(e).__name__)
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy




## === cell 2
IMSIZES = (224, 240, 260, 300, 380, 456, 528, 600)
im_size = IMSIZES[7]  # keep original choice (600)

load_dir = "/kaggle/input/plant-pathology-2021-fgvc8/"
train_csv_path = os.path.join(load_dir, "train.csv")
df = pd.read_csv(train_csv_path)

df["labels"] = df["labels"].astype(str)
all_tokens = sorted(
    {tok for s in df["labels"].values for tok in s.split() if tok.strip()}
)
class_name = all_tokens
n_labels = len(class_name)

print("Classes:", class_name)
print("n_labels:", n_labels)



## === cell 3
strategy = auto_select_accelerator()

DEFAULT_BATCH_SIZE = 64
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", DEFAULT_BATCH_SIZE))

test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
test_images = sorted(os.listdir(test_dir))
test_df = pd.DataFrame({"image": test_images})
print("Test images:", len(test_df))
print("BATCH_SIZE:", BATCH_SIZE)



## === cell 4
AUTOTUNE = tf.data.AUTOTUNE


@tf.function
def _read_decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # uint8
    img = tf.image.resize(
        img, (im_size, im_size), method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    return img


@tf.function
def _random_zoom(img, seed_pair):
    s0, s1 = seed_pair[0], seed_pair[1]
    z = tf.random.stateless_uniform(
        [], seed=[s0, s1], minval=0.9, maxval=1.1, dtype=tf.float32
    )
    new_size = tf.cast(tf.round(z * tf.cast(im_size, tf.float32)), tf.int32)
    new_size = tf.maximum(new_size, 1)
    img2 = tf.image.resize(
        img, (new_size, new_size), method=tf.image.ResizeMethod.BILINEAR
    )
    img2 = tf.image.resize_with_crop_or_pad(img2, im_size, im_size)
    return img2


@tf.function
def _augment(img, tta_idx, sample_idx):
    base_seed = tf.stack(
        [tf.cast(SEED + 1000 * tta_idx, tf.int32), tf.cast(sample_idx, tf.int32)],
        axis=0,
    )
    rh = (
        tf.random.stateless_uniform([], seed=base_seed + tf.constant([1, 0], tf.int32))
        < 0.5
    )
    rv = (
        tf.random.stateless_uniform([], seed=base_seed + tf.constant([2, 0], tf.int32))
        < 0.5
    )
    img = tf.cond(rh, lambda: tf.image.flip_left_right(img), lambda: img)
    img = tf.cond(rv, lambda: tf.image.flip_up_down(img), lambda: img)
    img = _random_zoom(img, base_seed + tf.constant([3, 0], tf.int32))
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


def make_base_test_dataset(file_names):
    paths = tf.constant([os.path.join(test_dir, fn) for fn in file_names])
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.enumerate()  # (idx, path)

    def _map(idx, path):
        img = _read_decode_resize(path)
        return idx, img

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_dataset_from_base(base_ds, tta_idx):
    def _map(idx, img):
        img = _augment(img, tf.cast(tta_idx, tf.int32), tf.cast(idx, tf.int32))
        return img

    ds = base_ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 5
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense

with strategy.scope():
    base = tf.keras.applications.EfficientNetB7(
        weights=None,
        include_top=False,
        input_shape=(im_size, im_size, 3),
    )

    model = Sequential(
        [
            base,
            GlobalMaxPooling2D(),
            Dense(n_labels, activation="softmax"),
        ]
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=4e-4),
        metrics=["accuracy"],
    )

model.summary()



## === cell 6
weights_path = "/kaggle/input/model222/bestmodel_tpu_aug.h5"
if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print("Loaded weights:", weights_path)
else:
    print("WARNING: Weights file not found:", weights_path)
    print(
        "Proceeding with randomly initialized weights (submission will be low-scoring but valid)."
    )



## === cell 7
TTA = 6
N = len(test_df)
steps = math.ceil(N / BATCH_SIZE)

base_ds = make_base_test_dataset(test_df["image"].values.tolist())

pred_sum = np.zeros((N, n_labels), dtype=np.float32)

for tta_idx in range(TTA):
    ds = make_test_dataset_from_base(base_ds, tta_idx)
    p = model.predict(ds, steps=steps, verbose=1)
    pred_sum += p[:N].astype(np.float32, copy=False)

pred = pred_sum / float(TTA)

top1 = np.argmax(pred, axis=1)
test_df["labels"] = [class_name[i] for i in top1]

sub_path = "submission.csv"
test_df[["image", "labels"]].to_csv(sub_path, index=False)
print("Wrote", sub_path)
test_df.head()



## === cell 8
pred
