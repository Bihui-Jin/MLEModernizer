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

import re
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.applications import ResNet50

tf.config.optimizer.set_jit(True)

if tf.config.list_physical_devices("GPU"):
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    from tensorflow.keras.mixed_precision import experimental as mixed_precision

    mixed_precision.set_policy("mixed_float16")

tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)

print(tf.__version__)
print(tf.keras.__version__)




## === cell 1
def decode_image(filename, label=None, image_size=(224, 224)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    else:
        return image, label




## === cell 2
FEATURE_BATCH_SIZE = 1024  # larger batch reduces per‑batch overhead
TRAIN_BATCH_SIZE = 64  # larger training batch for faster epochs
AUTO = tf.data.experimental.AUTOTUNE



## === cell 3
test_src = os.path.abspath("../input/plant-pathology-2021-fgvc8/test_images")
IMAGE_PATHS = [
    os.path.join(test_src, f)
    for f in os.listdir(test_src)
    if re.search(r"([a-zA-Z0-9\s_\\.\-\(\):])+(\.jpg|\.jpeg|\.png)$", f, re.IGNORECASE)
]

print(f"Found {len(IMAGE_PATHS)} test images.")

options = tf.data.Options()
options.experimental_deterministic = False

test_image_dataset = (
    tf.data.Dataset.from_tensor_slices(IMAGE_PATHS)
    .with_options(options)
    .map(lambda f: decode_image(f), num_parallel_calls=AUTO, deterministic=False)
    .batch(FEATURE_BATCH_SIZE)
    .prefetch(AUTO)
)



## === cell 4
base = ResNet50(weights="imagenet", include_top=False, pooling="avg")
base.trainable = False
print("ResNet50 base model built successfully.")



## === cell 5
train_csv_path = os.path.abspath("../input/plant-pathology-2021-fgvc8/train.csv")
train_df = pd.read_csv(train_csv_path)

train_img_dir = os.path.abspath("../input/plant-pathology-2021-fgvc8/train_images")

label_to_idx = {
    "scab": 0,
    "frog_eye_leaf_spot": 1,
    "complex": 2,
    "rust": 3,
    "powdery_mildew": 4,
}


def labels_to_vector(label_str):
    vec = np.zeros(5, dtype=np.float32)
    if isinstance(label_str, str):
        for lab in label_str.split():
            if lab in label_to_idx:
                vec[label_to_idx[lab]] = 1.0
    return vec


train_paths = [os.path.join(train_img_dir, fname) for fname in train_df["image"].values]
train_labels = np.stack(train_df["labels"].apply(labels_to_vector).values)

train_image_dataset = (
    tf.data.Dataset.from_tensor_slices(train_paths)
    .map(lambda f: decode_image(f), num_parallel_calls=AUTO, deterministic=False)
    .batch(FEATURE_BATCH_SIZE)
    .prefetch(AUTO)
)

print("Extracting ResNet features for training set...")
train_features = base.predict(train_image_dataset, verbose=0)
print(f"Training features shape: {train_features.shape}")

inputs = Input(shape=train_features.shape[1])
outputs = Dense(5, activation="sigmoid")(inputs)
model = Model(inputs=inputs, outputs=outputs)
print("Top model (dense head) built successfully.")

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
)

train_feat_dataset = (
    tf.data.Dataset.from_tensor_slices((train_features, train_labels))
    .shuffle(1000)
    .batch(TRAIN_BATCH_SIZE)
    .prefetch(AUTO)
)

model.fit(train_feat_dataset, epochs=3, verbose=1)



## === cell 6
print("Extracting ResNet features for test set...")
test_features = base.predict(test_image_dataset, verbose=0)
print(f"Test features shape: {test_features.shape}")

probs = model.predict(test_features, batch_size=FEATURE_BATCH_SIZE, verbose=0)
print(f"Predictions shape: {probs.shape}")



## === cell 7
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    6: "healthy",
}
threshold = {i: 0.35 for i in range(5)}
threshold2 = {i: 0.25 for i in range(5)}

pred_string = []
for line in probs:
    s = ""
    count = 0
    for i in range(5):
        if line[i] > threshold[i]:
            s += name[i] + " "
    for i in range(5):
        if line[i] > threshold2[i]:
            count += 1
    if count >= 2:
        notComplex = True
        for i in range(5):
            if line[i] > threshold[i] and name[i] == "complex":
                notComplex = False
                break
        if notComplex:
            s += "complex" + " "
    if s.strip() == "":
        s = name[6]
    pred_string.append(s.strip())



## === cell 8
image_names = [os.path.basename(p) for p in IMAGE_PATHS]
df = pd.DataFrame({"image": image_names, "labels": pred_string})
assert len(df) == len(image_names) == len(pred_string)

submission_path = "submission.csv"
df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
display(df.head())
