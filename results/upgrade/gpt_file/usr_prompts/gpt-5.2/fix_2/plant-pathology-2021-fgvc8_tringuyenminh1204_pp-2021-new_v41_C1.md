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
import os, re, math, random, pathlib
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K

print("TF:", tf.__version__)
print("Keras:", tf.keras.__version__ if hasattr(tf.keras, "__version__") else "bundled")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



## === cell 1
import pathlib




## === cell 2
def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    else:
        return image, label




## === cell 3
BATCH_SIZE = 32



## === cell 4
source = "../input/plant-pathology-2021-fgvc8/test_images"

IMAGE_FILES = sorted(
    [f for f in os.listdir(source) if re.search(r"(?i)\.(jpg|jpeg|png)$", f)]
)
IMAGE_PATHS = [os.path.join(source, f) for f in IMAGE_FILES]

print("Num test images found:", len(IMAGE_PATHS))
print("First 3:", IMAGE_FILES[:3])



## === cell 5
IMAGE_PATHS[:5]



## === cell 7
AUTO = tf.data.experimental.AUTOTUNE



## === cell 8
test_dataset = (
    tf.data.Dataset.from_tensor_slices(IMAGE_PATHS)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)



## === cell 9
import tensorflow as tf
from tensorflow import keras




## === cell 10
class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = K.shape(inputs)
        noise_shape = [
            symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)
        ]
        return tuple(noise_shape)




## === cell 11
def find_model_file():
    candidates = [
        "../input/3smresnet50/3SMResNet50.h5",
        "/kaggle/input/3smresnet50/3SMResNet50.h5",
        "../input/3smresnet50/3SMResNet50.hdf5",
        "/kaggle/input/3smresnet50/3SMResNet50.hdf5",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    roots = ["../input", "/kaggle/input"]
    for root in roots:
        if os.path.exists(root):
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    if (
                        fn.lower() in {"3smresnet50.h5", "3smresnet50.hdf5"}
                        or fn.lower() == "3smresnet50.h5"
                    ):
                        return os.path.join(dirpath, fn)
                    if (
                        fn.lower() == "3smresnet50.h5"
                        or fn.lower() == "3smresnet50.hdf5"
                    ):
                        return os.path.join(dirpath, fn)
                    if fn.lower() == "3smresnet50.h5":
                        return os.path.join(dirpath, fn)
                    if fn.lower() == "3smresnet50.hdf5":
                        return os.path.join(dirpath, fn)
                if "3SMResNet50.h5" in filenames:
                    return os.path.join(dirpath, "3SMResNet50.h5")
                if "3SMResNet50.hdf5" in filenames:
                    return os.path.join(dirpath, "3SMResNet50.hdf5")
    return None


model_path = find_model_file()
model = None

if model_path is not None:
    print("Loading model from:", model_path)
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={"FixedDropout": FixedDropout},
    )
else:
    print("Pretrained model file not found. Training fallback model...")

    TRAIN_CSV = "../input/plant-pathology-2021-fgvc8/train.csv"
    TRAIN_IMG_DIR = "../input/plant-pathology-2021-fgvc8/train_images"

    train_df = pd.read_csv(TRAIN_CSV)
    train_df["filepath"] = train_df["image"].apply(
        lambda x: os.path.join(TRAIN_IMG_DIR, x)
    )

    classes = ["scab", "frog_eye_leaf_spot", "complex", "rust", "powdery_mildew"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    def encode_labels(label_str):
        y = np.zeros(len(classes), dtype=np.float32)
        for lab in str(label_str).split():
            if lab in class_to_idx:
                y[class_to_idx[lab]] = 1.0
        return y

    y = np.stack([encode_labels(s) for s in train_df["labels"].values], axis=0)

    idx = np.arange(len(train_df))
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    split = int(0.9 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]

    tr_files = train_df.loc[tr_idx, "filepath"].values
    va_files = train_df.loc[va_idx, "filepath"].values
    y_tr = y[tr_idx]
    y_va = y[va_idx]

    IMAGE_SIZE = (512, 512)

    def decode_with_label(filename, label):
        return decode_image(filename, label, image_size=IMAGE_SIZE)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((tr_files, y_tr))
        .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        .map(decode_with_label, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((va_files, y_va))
        .map(decode_with_label, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
        pooling="avg",
    )
    x = tf.keras.layers.Dropout(0.2)(base.output)
    out = tf.keras.layers.Dense(len(classes), activation="sigmoid")(x)
    model = tf.keras.Model(inputs=base.input, outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
    )

    model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=1)



## === cell 12
probs = model.predict(test_dataset, verbose=1)
temp_probs = probs

print("Pred shape:", temp_probs.shape)



## === cell 13
temp_probs[:2]



## === cell 14
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    6: "healthy",
}

threshold = {0: 0.35, 1: 0.35, 2: 0.35, 3: 0.35, 4: 0.35}

pred_string = []
for line in temp_probs:
    labels = []
    count = 0
    for i in range(5):
        if line[i] > threshold[i]:
            labels.append(name[i])
            count += 1

    if count >= 2:
        if "complex" not in labels:
            labels.append("complex")

    if len(labels) == 0:
        labels = [name[6]]

    pred_string.append(" ".join(labels))



## === cell 15
pred_string[:10], len(pred_string)



## === cell 16
df = pd.DataFrame({"image": IMAGE_FILES, "labels": pred_string})
df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
