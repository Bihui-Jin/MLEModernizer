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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0, efficientnet

try:
    import tensorflow_addons as tfa
except Exception:  # broadened to catch non‑ImportError failures
    tfa = None

tf.config.optimizer.set_jit(True)  # XLA JIT compiler
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

np.random.seed(42)
tf.random.set_seed(42)


def get_strategy():
    """
    Returns a TensorFlow distribution strategy.
    Uses default strategy (CPU/GPU) to avoid TPU/protobuf issues.
    """
    strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy




## === cell 1
BASE_PATH = "/kaggle/input/plant-pathology-2021-fgvc8/"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images/")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images/")

df = pd.read_csv(TRAIN_CSV)

label_names = {
    0: "complex",
    1: "scab",
    2: "frog_eye_leaf_spot",
    3: "rust",
    4: "powdery_mildew",
    6: "healthy",  # not part of the sigmoid output vector
}
n_labels = 5  # number of disease classes (exclude healthy)

_label_to_idx = {name: idx for idx, name in label_names.items() if idx < n_labels}


def labels_to_vector(label_str):
    """Convert space‑delimited label string to a binary vector of length n_labels."""
    vec = np.zeros(n_labels, dtype=np.float32)
    for lab in label_str.split():
        idx = _label_to_idx.get(lab)
        if idx is not None:
            vec[idx] = 1.0
    return vec


df["targets"] = df["labels"].apply(labels_to_vector)

train_df, val_df = train_test_split(
    df, test_size=0.1, random_state=42, stratify=df["labels"]
)




## === cell 2
IMG_SIZE = 224
BATCH_SIZE = 256  # larger batch reduces steps per epoch
AUTOTUNE = tf.data.AUTOTUNE


def make_dataset(dataframe, img_dir, shuffle=True):
    """Build a tf.data.Dataset yielding (image, label_vector)."""
    img_paths = dataframe["image"].apply(lambda x: os.path.join(img_dir, x)).values
    labels = np.stack(dataframe["targets"].values)

    ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))

    def _load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = efficientnet.preprocess_input(img)
        return img, label

    ds = ds.map(_load_image, num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=4096, reshuffle_each_iteration=True)

    ds = ds.batch(BATCH_SIZE)

    opts = tf.data.Options()
    opts.experimental_deterministic = False
    ds = ds.with_options(opts)

    ds = ds.prefetch(AUTOTUNE)
    return ds


train_dataset = make_dataset(train_df, TRAIN_IMG_DIR, shuffle=True)
val_dataset = make_dataset(val_df, TRAIN_IMG_DIR, shuffle=False)




## === cell 3
strategy = get_strategy()
with strategy.scope():
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg",
    )
    base_model.trainable = False

    model = models.Sequential(
        [base_model, layers.Dense(n_labels, activation="sigmoid")]
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )
model.summary()




## === cell 4
EPOCHS_HEAD = 4
EPOCHS_FINE = 2

model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS_HEAD, verbose=2)

base_model.trainable = True
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["binary_accuracy"],
)
model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS_FINE, verbose=2)




## === cell 5
from sklearn.metrics import f1_score

val_preds = model.predict(val_dataset, verbose=0)  # (num_val, n_labels)
val_true = np.stack(val_df["targets"].values)  # (num_val, n_labels)

thresholds_opt = {}
candidate_thr = np.arange(0.1, 0.91, 0.05)

for i in range(n_labels):
    best_thr = 0.5
    best_f1 = 0.0
    for thr in candidate_thr:
        pred_bin = (val_preds[:, i] > thr).astype(int)
        f1 = f1_score(val_true[:, i], pred_bin, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    thresholds_opt[i] = best_thr

print("Optimized thresholds:", thresholds_opt)




## === cell 6
test_files = sorted(
    [
        f
        for f in os.listdir(TEST_IMG_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
)
test_df = pd.DataFrame({"image": test_files})


def make_test_dataset(img_dir):
    img_paths = [os.path.join(img_dir, f) for f in test_files]
    ds = tf.data.Dataset.from_tensor_slices(img_paths)

    def _load_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = efficientnet.preprocess_input(img)
        return img

    ds = ds.map(_load_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)

    opts = tf.data.Options()
    opts.experimental_deterministic = False
    ds = ds.with_options(opts)

    ds = ds.prefetch(AUTOTUNE)
    return ds


test_dataset = make_test_dataset(TEST_IMG_DIR)




## === cell 7
preds = model.predict(test_dataset, verbose=0)  # shape (num_test, n_labels)




## === cell 8
pred_strings = []
for probs in preds:
    labs = []
    for i in range(n_labels):
        if probs[i] > thresholds_opt[i]:
            labs.append(label_names[i])
    if not labs:
        labs.append(label_names[6])  # healthy fallback
    pred_strings.append(" ".join(labs))

test_df["labels"] = pred_strings
submission_path = "submission.csv"
test_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print(test_df.head())
