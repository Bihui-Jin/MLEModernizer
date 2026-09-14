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

3.10

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
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import class_weight
from PIL import Image
from joblib import Parallel, delayed  # parallel image loading

np.random.seed(42)



## === cell 1
data_path = "/kaggle/input/plant-pathology-2021-fgvc8"
train_csv_path = os.path.join(data_path, "train.csv")
train_images_dir = os.path.join(data_path, "train_images")
test_images_dir = os.path.join(data_path, "test_images")
sample_submission_path = os.path.join(data_path, "sample_submission.csv")

train_df = pd.read_csv(train_csv_path)
submission = pd.read_csv(sample_submission_path)

train_df["primary_label"] = train_df["labels"].apply(lambda x: x.split(" ")[0])
class_names = sorted(train_df["primary_label"].unique())
num_classes = len(class_names)



## === cell 2
train_split_df, val_split_df = train_test_split(
    train_df,
    test_size=0.2,
    stratify=train_df["primary_label"],
    random_state=42,
)

label_to_index = {name: i for i, name in enumerate(class_names)}

IMG_SIZE = (64, 64)  # modest size to keep memory reasonable


def load_image(path):
    """Read image, resize, convert to float16 array in [0,1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float16) / 255.0
    return arr.reshape(-1)  # flatten


def prepare_dataset(df, images_dir):
    paths = df["image"].apply(lambda x: os.path.join(images_dir, x)).tolist()
    X_list = Parallel(n_jobs=-1, backend="threading")(
        delayed(load_image)(p) for p in paths
    )
    X = np.stack(X_list)
    y = df["primary_label"].map(label_to_index).values
    return X, y


print("Loading training images...")
X_train, y_train = prepare_dataset(train_split_df, train_images_dir)
print("Loading validation images...")
X_val, y_val = prepare_dataset(val_split_df, train_images_dir)

weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=y_train,
)
class_weights_dict = {i: w for i, w in enumerate(weights)}

model = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=1000,
    class_weight=class_weights_dict,
    n_jobs=-1,
)
model.fit(X_train, y_train)

val_acc = model.score(X_val, y_val)
print(f"Validation accuracy: {val_acc:.4f}")



## === cell 3
print("Loading test images...")
test_paths = (
    submission["image"].apply(lambda x: os.path.join(test_images_dir, x)).tolist()
)
X_test = np.stack(
    Parallel(n_jobs=-1, backend="threading")(delayed(load_image)(p) for p in test_paths)
)

pred_indices = model.predict(X_test)
pred_labels = [class_names[idx] for idx in pred_indices]

submission["labels"] = pred_labels



## === cell 4
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")
