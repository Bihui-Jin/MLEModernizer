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
import os, random, math, re
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model

print("TF:", tf.__version__)
print("Keras:", tf.keras.__version__)


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



## === cell 1
path = "../input/plant-pathology-2021-fgvc8/"
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(
    path + "sample_submission.csv"
)  # contains the correct test image order
sub = pd.read_csv(path + "sample_submission.csv")

train.head(), test.head(), train.shape, test.shape



## === cell 2
AUTO = tf.data.experimental.AUTOTUNE



## === cell 3
from matplotlib import pyplot as plt

img = plt.imread(path + "train_images/800113bb65efe69e.jpg")
print(img.shape)
plt.imshow(img)
plt.axis("off")



## === cell 4
import pathlib



## === cell 5
train_paths = []
for root, dirnames, files in os.walk(path + "train_images"):
    for file in files:
        train_paths.append(os.path.join(root, file))

test_paths = []
for root, dirnames, files in os.walk(path + "test_images"):
    for file in files:
        test_paths.append(os.path.join(root, file))

len(train_paths), len(test_paths)



## === cell 6
kind = np.unique(train["labels"])
kind[:10], len(kind)



## === cell 7
all_labels = [
    "scab",
    "frog_eye_leaf_spot",
    "rust",
    "complex",
    "powdery_mildew",
    "healthy",
]
label2idx = {l: i for i, l in enumerate(all_labels)}


def multilabel_to_vec(s: str):
    vec = np.zeros(len(all_labels), dtype=np.float32)
    for lab in str(s).split():
        if lab in label2idx:
            vec[label2idx[lab]] = 1.0
    return vec


y = np.vstack(train["labels"].apply(multilabel_to_vec).values)

new_train = pd.concat([train[["image"]], pd.DataFrame(y, columns=all_labels)], axis=1)
new_train.head()



## === cell 8
new_train.describe()




## === cell 9
def decode_image(filename, label=None, image_size=(224, 224)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
    image = tf.image.resize(image, image_size, method="bilinear")
    if label is None:
        return image
    return image, label




## === cell 10
test_paths = [
    os.path.join(path, "test_images", img_id) for img_id in test["image"].values
]
test_paths[:3], len(test_paths)



## === cell 11
BATCH_SIZE = 32



## === cell 12
test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(
        lambda x: decode_image(x, label=None, image_size=(224, 224)),
        num_parallel_calls=AUTO,
    )
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)



## === cell 13
from tensorflow import keras



## === cell 14
base = tf.keras.applications.ResNet50(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling="avg"
)
inputs = keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.resnet50.preprocess_input(inputs * 255.0)
x = base(x, training=False)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(all_labels), activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss="binary_crossentropy"
)

model.summary()



## === cell 15
from sklearn.model_selection import train_test_split

train_paths_ordered = [
    os.path.join(path, "train_images", img_id) for img_id in train["image"].values
]
X_train, X_val, y_train, y_val = train_test_split(
    train_paths_ordered, y, test_size=0.1, random_state=SEED, stratify=None
)


def make_ds(paths_list, labels_array=None, training=False):
    if labels_array is None:
        ds = tf.data.Dataset.from_tensor_slices(paths_list)
        ds = ds.map(
            lambda p: decode_image(p, label=None, image_size=(224, 224)),
            num_parallel_calls=AUTO,
        )
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths_list, labels_array))
        ds = ds.map(
            lambda p, lab: decode_image(p, label=lab, image_size=(224, 224)),
            num_parallel_calls=AUTO,
        )
    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTO)
    return ds


train_ds = make_ds(X_train, y_train, training=True)
val_ds = make_ds(X_val, y_val, training=False)



## === cell 16
EPOCHS = 2
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)



## === cell 17
probs = model.predict(test_dataset, verbose=1)
probs.shape



## === cell 18
temp_probs = probs



## === cell 19
idx2name = {i: n for i, n in enumerate(all_labels)}

threshold = {
    label2idx["scab"]: 0.05,
    label2idx["frog_eye_leaf_spot"]: 0.05,
    label2idx["rust"]: 0.05,
    label2idx["complex"]: 0.05,
    label2idx["powdery_mildew"]: 0.05,
}
healthy_idx = label2idx["healthy"]

pred_string = []
for line in temp_probs:
    s = ""
    count = 0

    nonhealthy_indices = [
        label2idx["scab"],
        label2idx["frog_eye_leaf_spot"],
        label2idx["complex"],
        label2idx["rust"],
        label2idx["powdery_mildew"],
    ]
    for i in nonhealthy_indices:
        if line[i] > threshold[i]:
            s += idx2name[i] + " "
            count += 1

    if count >= 2:
        if not (line[label2idx["complex"]] > threshold[label2idx["complex"]]):
            s += "complex" + " "

    if s == "":
        s = idx2name[healthy_idx]

    pred_string.append(s.strip())

test["labels"] = pred_string
test.to_csv("submission.csv", index=False)
test.head()



## === cell 20
assert os.path.exists("submission.csv")
sub_check = pd.read_csv("submission.csv")
print(sub_check.shape)
print(sub_check.columns.tolist())
print(sub_check.head())
print("Unique label strings (sample):", sub_check["labels"].head(10).tolist())
