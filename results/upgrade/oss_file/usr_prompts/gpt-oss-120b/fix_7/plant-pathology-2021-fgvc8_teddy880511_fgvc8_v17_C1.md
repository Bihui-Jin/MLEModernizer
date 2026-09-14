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

0.3635457063711896

# 6. Current score

0.21672

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.21672) has done: 'We replace the eager NumPy image loading with an efficient tf.data pipeline that reads, decodes, resizes, and normalizes images on‑the‑fly in parallel, so we never materialize the full training or test arrays in memory. The same file lists and label encoding are kept, and the ResNet‑50 model is trained on the dataset object, preserving the exact architecture, loss, optimizer, epochs, and batch size. Prediction also uses the pipeline, guaranteeing the original order of test files for the submission.'

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd, cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder

BASE_PATH = "/kaggle/input/plant-pathology-2021-fgvc8"

train_img_path = os.path.join(BASE_PATH, "train_images")
train_csv_path = os.path.join(BASE_PATH, "train.csv")

train_files = sorted(
    [
        f
        for f in os.listdir(train_img_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
)

train_filepaths = [os.path.join(train_img_path, f) for f in train_files]


def _load_and_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [160, 240])  # height, width
    img = tf.cast(img, tf.float32) / 255.0
    return img




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
y_df = pd.read_csv(train_csv_path)
le_sc = LabelEncoder()
y_encoded = le_sc.fit_transform(y_df["labels"])
num_classes = len(le_sc.classes_)
y_train = to_categorical(y_encoded, num_classes)

batch_size = 32
train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_filepaths, y_train))
    .map(lambda p, l: (_load_and_preprocess(p), l), num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(buffer=len(train_filepaths))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1176353828.py in <cell line: 0>()
     10     tf.data.Dataset.from_tensor_slices((train_filepaths, y_train))
     11     .map(lambda p, l: (_load_and_preprocess(p), l), num_parallel_calls=tf.data.AUTOTUNE)
---> 12     .shuffle(buffer=len(train_filepaths))
     13     .batch(batch_size)
     14     .prefetch(tf.data.AUTOTUNE)

TypeError: DatasetV2.shuffle() got an unexpected keyword argument 'buffer'

## === cell 2
model = ResNet50(
    include_top=True,
    weights=None,
    input_shape=(160, 240, 3),
    classes=num_classes,
)

model.compile(optimizer=SGD(), loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_dataset, epochs=2, verbose=2)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1149039590.py in <cell line: 0>()
      8 
      9 model.compile(optimizer=SGD(), loss="categorical_crossentropy", metrics=["accuracy"])
---> 10 model.fit(train_dataset, epochs=2, verbose=2)
     11 

NameError: name 'train_dataset' is not defined

## === cell 3
test_img_path = os.path.join(BASE_PATH, "test_images")
test_files = sorted(
    [
        f
        for f in os.listdir(test_img_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
)

test_filepaths = [os.path.join(test_img_path, f) for f in test_files]

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_filepaths)
    .map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

pred_probs = model.predict(test_dataset, verbose=0)
pred_labels = np.argmax(pred_probs, axis=1)
pred_class_names = le_sc.inverse_transform(pred_labels)

submission = pd.DataFrame({"image": test_files, "labels": pred_class_names})
submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")
