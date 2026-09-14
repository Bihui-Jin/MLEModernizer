# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.7200738688827324

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from concurrent.futures import ThreadPoolExecutor




## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
train_csv_path = "../input/plant-pathology-2021-fgvc8/train.csv"
train_images_dir = "../input/plant-pathology-2021-fgvc8/train_images/"

image_size = (
    300,
    300,
)  # not used for feature extraction but kept for possible resizing

train_df = pd.read_csv(train_csv_path)
train_df["label_list"] = train_df["labels"].apply(lambda x: x.split(" "))

mlb = MultiLabelBinarizer()
mlb.fit(train_df["label_list"])
label_names = mlb.classes_.tolist()


def extract_mean_rgb(image_path):
    """Return mean R, G, B values as a 3‑element numpy array."""
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(image_size)
        arr = np.array(img).astype(np.float32) / 255.0
        return arr.mean(axis=(0, 1))  # mean over height and width
    except Exception:
        return np.zeros(3, dtype=np.float32)


max_samples = 8000
if len(train_df) > max_samples:
    train_subset = train_df.sample(n=max_samples, random_state=42).reset_index(
        drop=True
    )
else:
    train_subset = train_df

train_image_paths = [
    os.path.join(train_images_dir, img_name) for img_name in train_subset["image"]
]
with ThreadPoolExecutor(max_workers=8) as executor:
    train_features = list(executor.map(extract_mean_rgb, train_image_paths))
X_train = np.vstack(train_features)

y_train = mlb.transform(train_subset["label_list"])

base_clf = LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="lbfgs",
    max_iter=200,
    n_jobs=-1,
    class_weight="balanced",
)
clf = OneVsRestClassifier(base_clf, n_jobs=-1)
clf.fit(X_train, y_train)

threshold_candidates = [0.3, 0.4, 0.5, 0.6, 0.7]
best_thresh = 0.5
best_f1 = 0.0

train_probs = clf.predict_proba(X_train)  # shape: (n_samples, n_labels)
for t in threshold_candidates:
    preds_binary = (train_probs >= t).astype(int)
    f1 = f1_score(y_train, preds_binary, average="macro")
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t





## === cell 2
test_image_names = sorted(os.listdir(test_dir))

test_image_paths = [os.path.join(test_dir, name) for name in test_image_names]
with ThreadPoolExecutor(max_workers=8) as executor:
    test_features = list(executor.map(extract_mean_rgb, test_image_paths))

predictions = []
for img_name, feats in zip(test_image_names, test_features):
    feats = feats.reshape(1, -1)
    probs = clf.predict_proba(feats)[0]  # shape: (n_labels,)

    selected_idx = np.where(probs > best_thresh)[0]
    if selected_idx.size == 0:
        most_common_label = label_names[np.argmax(y_train.sum(axis=0))]
        selected_labels = [most_common_label]
    else:
        selected_labels = [label_names[i] for i in selected_idx]

    predictions.append([img_name, " ".join(selected_labels).strip()])

submission_df = pd.DataFrame(predictions, columns=["image", "labels"])
submission_path = os.path.join(output_dir, "submission.csv")
submission_df.to_csv(submission_path, index=False)

print(f"Submission file written to: {submission_path}")
