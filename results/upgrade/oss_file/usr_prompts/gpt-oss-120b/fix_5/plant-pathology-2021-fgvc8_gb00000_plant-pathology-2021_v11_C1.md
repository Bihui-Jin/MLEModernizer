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
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import math

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))




## === cell 1
BASE_PATH = "/kaggle/input/plant-pathology-2021-fgvc8"
train_dir = os.path.join(BASE_PATH, "train_images")
test_dir = os.path.join(BASE_PATH, "test_images")
train_csv_path = os.path.join(BASE_PATH, "train.csv")
sample_submission_path = os.path.join(BASE_PATH, "sample_submission.csv")




## === cell 2
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(
    sample_submission_path
)  # contains correct ordering of test images
test_ids = test_df["image"].tolist()




## === cell 3
train_df["label_list"] = train_df["labels"].apply(lambda x: x.split(" "))




## === cell 4
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(train_df["label_list"])
class_names = mlb.classes_.tolist()
num_classes = len(class_names)




## === cell 5
class ImageSequence(Sequence):
    def __init__(self, df, directory, y=None, batch_size=32, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.directory = directory
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return math.ceil(len(self.df) / self.batch_size)

    def __getitem__(self, idx):
        batch_idxs = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_idxs]

        batch_images = []
        for fname in batch_df["image"]:
            img_path = os.path.join(self.directory, fname)
            img = load_img(img_path, target_size=(224, 336))
            arr = img_to_array(img)
            arr = preprocess_input(arr)
            batch_images.append(arr)
        X = np.stack(batch_images, axis=0)

        if self.y is not None:
            Y = self.y[batch_idxs]
            return X, Y
        else:
            return X

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)




## === cell 6
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.1, random_state=42, stratify=y
)

train_seq = ImageSequence(
    df=train_df.iloc[train_idx],
    directory=train_dir,
    y=y[train_idx],
    batch_size=32,
    shuffle=True,
)

val_seq = ImageSequence(
    df=train_df.iloc[val_idx],
    directory=train_dir,
    y=y[val_idx],
    batch_size=32,
    shuffle=False,
)

test_seq = ImageSequence(
    df=test_df,
    directory=test_dir,
    y=None,
    batch_size=32,
    shuffle=False,
)




## === cell 7
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(224, 336, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dense(num_classes, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])




## === cell 8
model.fit(
    train_seq,
    validation_data=val_seq,
    epochs=10,
    verbose=2,
)




## === cell 9
test_preds = model.predict(test_seq, verbose=0)




## === cell 10
threshold = 0.5
pred_labels = []
for probs in test_preds:
    indices = np.where(probs >= threshold)[0]
    if len(indices) == 0:
        indices = [np.argmax(probs)]
    labels = [class_names[i] for i in indices]
    pred_labels.append(" ".join(labels))




## === cell 11
submission = pd.DataFrame({"image": test_ids, "labels": pred_labels})
submission.to_csv("submission.csv", index=False)
