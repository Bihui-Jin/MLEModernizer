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

0.8258541089566027

# 6. Current score

0.29888

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.29888) has done: 'I removed the unused `KaggleDatasets` import that caused the protobuf error, fixed the image‑folder paths, and replaced the missing custom model file with a lightweight pretrained MobileNetV2 that is fine‑tuned on the provided training data. The new pipeline builds a proper multi‑label dataset, trains for a few epochs, predicts on the test images, thresholds the probabilities, and writes a correctly‑sized `submission.csv` with the required “image,labels” format. This resolves all runtime errors and produces a valid submission file while keeping the original modelling approach unchanged.'

# 9. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image as keras_image
from pathlib import Path

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_INPUT = Path("../input/plant-pathology-2021-fgvc8")
TRAIN_CSV = BASE_INPUT / "train.csv"
TEST_IMAGES_DIR = BASE_INPUT / "test_images"
TRAIN_IMAGES_DIR = BASE_INPUT / "train_images"




## === cell 2
def decode_image(filename, label=None, image_size=(224, 224)):
    bits = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(bits, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.resize(img, image_size)
    if label is None:
        return img
    return img, label




## === cell 3
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE
THRESHOLD = 0.5



## === cell 4
train_df = pd.read_csv(TRAIN_CSV)

class_names = [
    "scab",
    "frog_eye_leaf_spot",
    "complex",
    "rust",
    "powdery_mildew",
    "healthy",
]
num_classes = len(class_names)
class_to_idx = {c: i for i, c in enumerate(class_names)}


def labels_to_onehot(label_str):
    vec = np.zeros(num_classes, dtype=np.float32)
    for lab in label_str.split():
        if lab in class_to_idx:
            vec[class_to_idx[lab]] = 1.0
    return vec


train_image_paths = [str(TRAIN_IMAGES_DIR / fname) for fname in train_df["image"]]
train_labels = np.stack(train_df["labels"].apply(labels_to_onehot).values)



## === cell 5
train_ds = (
    tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
    .shuffle(buffer=len(train_image_paths), reshuffle_each_iteration=True)
    .map(lambda x, y: decode_image(x, y), num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3191960749.py in <cell line: 0>()
      2 train_ds = (
      3     tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
----> 4     .shuffle(buffer=len(train_image_paths), reshuffle_each_iteration=True)
      5     .map(lambda x, y: decode_image(x, y), num_parallel_calls=AUTOTUNE)
      6     .batch(BATCH_SIZE)

TypeError: DatasetV2.shuffle() got an unexpected keyword argument 'buffer'

## === cell 6
val_size = int(0.1 * len(train_image_paths))
val_ds = train_ds.take(val_size // BATCH_SIZE)
train_ds = train_ds.skip(val_size // BATCH_SIZE)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4253189364.py in <cell line: 0>()
      1 # Simple validation split (10%)
      2 val_size = int(0.1 * len(train_image_paths))
----> 3 val_ds = train_ds.take(val_size // BATCH_SIZE)
      4 train_ds = train_ds.skip(val_size // BATCH_SIZE)
      5 

NameError: name 'train_ds' is not defined

## === cell 7
base_model = MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
base_model.trainable = False  # freeze backbone for quick fine‑tuning

inputs = layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="sigmoid")(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.BinaryAccuracy()],
)

model.summary()



## === cell 8
EPOCHS = 3
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1087989373.py in <cell line: 0>()
      1 # Train for a few epochs (lightweight fine‑tuning)
      2 EPOCHS = 3
----> 3 model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)
      4 

NameError: name 'train_ds' is not defined

## === cell 9
test_filenames = sorted(
    [
        f
        for f in os.listdir(TEST_IMAGES_DIR)
        if re.search(r"([a-zA-Z0-9\s_\\.\-\(\):])+(\.jpg|\.jpeg|\.png)$", f, re.I)
    ]
)
test_paths = [str(TEST_IMAGES_DIR / f) for f in test_filenames]

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(lambda x: decode_image(x), num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

probs = model.predict(test_ds, verbose=0)



## === cell 10
pred_strings = []
for prob_vec in probs:
    labels = [class_names[i] for i, p in enumerate(prob_vec) if p > THRESHOLD]
    if not labels:
        labels = ["healthy"]
    pred_strings.append(" ".join(labels))

submission = pd.DataFrame({"image": test_filenames, "labels": pred_strings})

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
display(submission.head())
