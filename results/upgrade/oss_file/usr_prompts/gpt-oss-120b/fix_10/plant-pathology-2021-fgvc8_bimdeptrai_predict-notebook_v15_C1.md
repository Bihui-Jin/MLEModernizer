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

0.7964542936288113

# 6. Current score

0.29129

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.29129) has done: 'I remove the problematic protobuf setting and the unavailable TensorFlow Addons usage, simplify the augmentation to only flips, add a separate image‑loading function for the test set that points to the correct directory, and ensure the validation dataset is created correctly. These fixes eliminate the runtime errors and allow the pipeline to generate a valid `submission.csv` file.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split


tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

if tf.config.list_physical_devices("GPU"):
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
submissions = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")




## === cell 2
IMG_H, IMG_W = 384, 384
BATCH_SIZE = 128
EPOCHS = 2
THRESH = 0.2
TRAIN_IMG_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_IMG_DIR = "../input/plant-pathology-2021-fgvc8/test_images"
AUTOTUNE = tf.data.AUTOTUNE




## === cell 3
label_split = train_df["labels"].apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
binary_labels = mlb.fit_transform(label_split)
label_names = mlb.classes_




## === cell 4
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.1,
    random_state=42,
    stratify=binary_labels.max(axis=1),
)
train_df_split = train_df.iloc[train_idx].reset_index(drop=True)
val_df_split = train_df.iloc[val_idx].reset_index(drop=True)
train_labels = binary_labels[train_idx]
val_labels = binary_labels[val_idx]




## === cell 5
def _load_and_preprocess(path):
    img = tf.io.read_file(tf.strings.join([TRAIN_IMG_DIR, "/", path]))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_H, IMG_W])
    img = img / 255.0
    return img


def _load_and_preprocess_test(path):
    img = tf.io.read_file(tf.strings.join([TEST_IMG_DIR, "/", path]))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_H, IMG_W])
    img = img / 255.0
    return img


def _augment(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    return img


train_paths = train_df_split["image"].values
train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.map(
    lambda p, l: (_load_and_preprocess(p), tf.cast(l, tf.float32)),
    num_parallel_calls=AUTOTUNE,
)
train_ds = train_ds.map(
    lambda img, lbl: (_augment(img), lbl),
    num_parallel_calls=AUTOTUNE,
)
train_ds = train_ds.shuffle(buffer=len(train_paths), seed=42)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_paths = val_df_split["image"].values
val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(
    lambda p, l: (_load_and_preprocess(p), tf.cast(l, tf.float32)),
    num_parallel_calls=AUTOTUNE,
)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3762480557.py in <cell line: 0>()
     32     num_parallel_calls=AUTOTUNE,
     33 )
---> 34 train_ds = train_ds.shuffle(buffer=len(train_paths), seed=42)
     35 train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
     36 

TypeError: DatasetV2.shuffle() got an unexpected keyword argument 'buffer'

## === cell 6
base_model = keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_H, IMG_W, 3),
    pooling="avg",
)
x = base_model.output
output = keras.layers.Dense(len(label_names), activation="sigmoid")(x)
model = keras.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
)




## === cell 7
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/334432989.py in <cell line: 0>()
      1 model.fit(
      2     train_ds,
----> 3     validation_data=val_ds,
      4     epochs=EPOCHS,
      5     verbose=1,

NameError: name 'val_ds' is not defined

## === cell 8
test_paths = submissions["image"].values
test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(
    lambda p: _load_and_preprocess_test(p),
    num_parallel_calls=AUTOTUNE,
)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)




## === cell 9
preds = model.predict(
    test_ds,
    verbose=1,
)




## === cell 10
pred_labels = preds >= THRESH

for i, img_name in enumerate(submissions["image"]):
    selected = np.where(pred_labels[i])[0]
    if len(selected) == 0:
        selected = [np.argmax(preds[i])]
    label_str = " ".join([label_names[idx] for idx in selected])
    if "healthy" in label_str and len(selected) > 1:
        label_str = " ".join([l for l in label_str.split() if l != "healthy"])
        if label_str == "":
            label_str = "healthy"
    submissions.at[i, "labels"] = label_str

submissions.to_csv("submission.csv", index=False)




## === cell 11
print("Submission saved. First rows:")
print(submissions.head())
