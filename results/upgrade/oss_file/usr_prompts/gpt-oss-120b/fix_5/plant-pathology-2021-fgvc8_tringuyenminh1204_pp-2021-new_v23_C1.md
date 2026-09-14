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

0.2320406278855045

# 6. Current score

0.30532

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.28659) has done: 'The changes mainly reduce the heavy image‑processing load: the image size is lowered from 512×512 to MobileNetV2’s native 224×224, cutting the amount of data each step must handle, and the `.cache()` call that tried to keep all decoded images in RAM is removed to avoid excessive memory use and slowdowns. These adjustments keep the exact model architecture, training loop, and prediction logic unchanged, preserving result accuracy while fitting comfortably inside the 600‑second limit.'
- What this solution (achieved 0.30532) has done: 'The fix adds a protobuf compatibility setting before importing TensorFlow to stop the import error, and slightly raises the prediction thresholds so the F1‑score moves closer to the target value while keeping the original model and workflow unchanged. The rest of the pipeline is left intact, and the script still writes a correctly‑formatted `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = (
    "python"  # fix TensorFlow protobuf issue
)
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, backend as K

print(tf.__version__)
print(tf.keras.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_PATH = "../input/plant-pathology-2021-fgvc8/"
train = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
test = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))



## === cell 2
label_lists = train["labels"].str.split(" ")
unique_labels = sorted({lbl for sublist in label_lists for lbl in sublist})
label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
num_classes = len(unique_labels)


def multilabel_to_onehot(labels):
    vec = np.zeros(num_classes, dtype=np.int32)
    for l in labels:
        vec[label_to_idx[l]] = 1
    return vec


onehot = np.stack(label_lists.apply(multilabel_to_onehot).values)
onehot_df = pd.DataFrame(onehot, columns=unique_labels)
new_train = pd.concat([train[["image"]], onehot_df], axis=1)



## === cell 3
train_paths = []
for root, _, files in os.walk(os.path.join(BASE_PATH, "train_images")):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            train_paths.append(os.path.join(root, f))

test_paths = []
for root, _, files in os.walk(os.path.join(BASE_PATH, "test_images")):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            test_paths.append(os.path.join(root, f))



## === cell 4
IMG_SIZE = (224, 224)  # changed from (512, 512) for faster processing


def decode_image(path, label=None):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    if label is None:
        return img
    return img, label




## === cell 5
BATCH_SIZE = 128
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_labels = new_train[unique_labels].values.astype(np.float32)
train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.map(lambda p, l: (decode_image(p), l), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.shuffle(2048).batch(BATCH_SIZE).prefetch(AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = (
    test_ds.map(decode_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)



## === cell 6
base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
)
base.trainable = False  # freeze pretrained layers

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(num_classes, activation="sigmoid")(x)
model = models.Model(inputs, outputs)

model.compile(optimizer=optimizers.Adam(), loss="binary_crossentropy")



## === cell 7
model.fit(train_ds, epochs=1, verbose=1)



## === cell 8
probs = model.predict(test_ds, verbose=0)

orig_name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    6: "healthy",
}

threshold = {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2}

pred_strings = []
for line in probs:
    s = ""
    cnt = 0
    for i in range(5):
        idx = label_to_idx.get(orig_name[i])
        if idx is not None and line[idx] > threshold[i]:
            cnt += 1
    if cnt >= 3:
        max_val = -1
        key = None
        for i in range(5):
            idx = label_to_idx.get(orig_name[i])
            if idx is not None:
                val = line[idx]
                if val > max_val:
                    max_val = val
                    key = i
        if orig_name[key] != "complex":
            s += orig_name[key] + " "
        s += "complex "
    else:
        for i in range(5):
            idx = label_to_idx.get(orig_name[i])
            if idx is not None and line[idx] > threshold[i]:
                s += orig_name[i] + " "
    if s == "":
        s = orig_name[6]  # healthy
    pred_strings.append(s.strip())

test["labels"] = pred_strings



## === cell 9
submission_path = "submission.csv"
test.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print(test.head())
