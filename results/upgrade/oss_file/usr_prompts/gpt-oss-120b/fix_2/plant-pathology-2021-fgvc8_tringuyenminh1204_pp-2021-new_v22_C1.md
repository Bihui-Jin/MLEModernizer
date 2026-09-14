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
import os, numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras import layers, models, optimizers, applications

print(tf.__version__)



## === cell 1
BASE = "../input/plant-pathology-2021-fgvc8/"
train_df = pd.read_csv(os.path.join(BASE, "train.csv"))
test_df = pd.read_csv(os.path.join(BASE, "sample_submission.csv"))

train_img_dir = os.path.join(BASE, "train_images")
test_img_dir = os.path.join(BASE, "test_images")

train_paths = [
    os.path.join(root, f) for root, _, files in os.walk(train_img_dir) for f in files
]
test_paths = [
    os.path.join(root, f) for root, _, files in os.walk(test_img_dir) for f in files
]

test_path_dict = {os.path.basename(p): p for p in test_paths}
test_paths_ordered = [test_path_dict[fn] for fn in test_df["image"]]



## === cell 2
DISEASES = [
    "scab",
    "frog_eye_leaf_spot",
    "complex",
    "rust",
    "powdery_mildew",
    "healthy",
]
num_classes = len(DISEASES)


def multi_hot(label_str):
    arr = np.zeros(num_classes, dtype=np.float32)
    for token in label_str.split():
        if token in DISEASES:
            arr[DISEASES.index(token)] = 1.0
    return arr


train_labels = np.stack(train_df["labels"].apply(multi_hot).values)



## === cell 3
IMG_SIZE = (224, 224)
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64


def decode_image(path, label=None):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    if label is None:
        return img
    else:
        return img, label


train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .map(decode_image, num_parallel_calls=AUTOTUNE)
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths_ordered)
    .map(lambda p: decode_image(p), num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)



## === cell 4
base_model = applications.ResNet50(
    weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,)
)
base_model.trainable = False  # freeze ImageNet weights

inputs = layers.Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(num_classes, activation="sigmoid")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=optimizers.Adam(),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(name="auc")],
)



## === cell 5
model.fit(train_dataset, epochs=3, verbose=2)



## === cell 6
probs = model.predict(test_dataset)  # shape (num_test, num_classes)



## === cell 7
threshold = {0: 0.2, 1: 0.3, 2: 0.15, 3: 0.3, 4: 0.35}
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    6: "healthy",
}


def predict_string(prob_row):
    cnt = sum(prob_row[i] > threshold[i] for i in range(5))
    if cnt >= 3:
        key = np.argmax(prob_row[:5])
        s = ""
        if name[key] != "complex":
            s += name[key] + " "
        s += "complex "
    else:
        s = " ".join(name[i] for i in range(5) if prob_row[i] > threshold[i])
    if not s.strip():
        s = name[6]  # healthy
    return s.strip()


pred_strings = [predict_string(row) for row in probs]



## === cell 8
submission = pd.DataFrame({"image": test_df["image"], "labels": pred_strings})
submission.to_csv("submission.csv", index=False)
print("submission.csv written, preview:")
print(submission.head())
