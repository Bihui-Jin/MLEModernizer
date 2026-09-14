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

import re, random, math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

cpu_count = os.cpu_count() or 1
tf.config.threading.set_inter_op_parallelism_threads(cpu_count)
tf.config.threading.set_intra_op_parallelism_threads(cpu_count)

print("TF:", tf.__version__)



## === cell 1
BASE_INPUT = "/kaggle/input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
TEST_IMG_DIR = os.path.join(BASE_INPUT, "test_images")

CLASS_NAMES = [
    "scab",
    "frog_eye_leaf_spot",
    "complex",
    "rust",
    "powdery_mildew",
    "healthy",
]
NUM_CLASSES = len(CLASS_NAMES)

LABEL2IDX = {lbl: i for i, lbl in enumerate(CLASS_NAMES)}




## === cell 2
def decode_image(path, label=None, img_size=(224, 224)):
    """Read an image file, decode, resize and optionally return label."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0
    if label is None:
        return img
    else:
        return img, label


def labels_to_multihot(label_str):
    """Convert space‑separated label string to a multi‑hot vector using a dict lookup."""
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    for lbl in label_str.split():
        idx = LABEL2IDX.get(lbl)
        if idx is not None:
            vec[idx] = 1.0
    return vec




## === cell 3
train_df = pd.read_csv(TRAIN_CSV)

train_img_paths = [
    os.path.join(BASE_INPUT, "train_images", img_name)
    for img_name in train_df["image"].values
]

train_labels = np.stack(train_df["labels"].apply(labels_to_multihot).values)



## === cell 4
BATCH_SIZE = 128
AUTOTUNE = tf.data.AUTOTUNE

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_img_paths, train_labels))
    .shuffle(buffer_size=1024, seed=42)
    .map(
        lambda p, l: decode_image(p, l, img_size=(224, 224)),
        num_parallel_calls=AUTOTUNE,
        deterministic=False,
    )
    .cache("train_cache.tfdata")
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)



## === cell 5
inputs = layers.Input(shape=(224, 224, 3))
x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation="relu")(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(NUM_CLASSES, activation="sigmoid")(x)

model = models.Model(inputs, outputs)
model.compile(optimizer=optimizers.Adam(1e-3), loss="binary_crossentropy")



## === cell 6
model.fit(train_ds, epochs=10, verbose=1)



## === cell 7
test_files = sorted(
    [
        f
        for f in os.listdir(TEST_IMG_DIR)
        if re.search(r"\.(jpg|jpeg|png)$", f, re.IGNORECASE)
    ]
)
TEST_PATHS = [os.path.join(TEST_IMG_DIR, f) for f in test_files]

test_ds = (
    tf.data.Dataset.from_tensor_slices(TEST_PATHS)
    .map(
        lambda p: decode_image(p, img_size=(224, 224)),
        num_parallel_calls=AUTOTUNE,
        deterministic=False,
    )
    .cache("test_cache.tfdata")
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)



## === cell 8
probs = model.predict(test_ds, verbose=0)  # shape: (num_test, NUM_CLASSES)



## === cell 9
threshold = 0.5  # more realistic threshold for multi‑label prediction
pred_strings = []
for prob_vec in probs:
    labels = [CLASS_NAMES[i] for i, p in enumerate(prob_vec) if p > threshold]
    if not labels:  # fallback to 'healthy' if nothing passes
        labels = ["healthy"]
    pred_strings.append(" ".join(labels))



## === cell 10
submission = pd.DataFrame({"image": test_files, "labels": pred_strings})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
display(submission.head())
