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
import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import concurrent.futures

cv2.setNumThreads(0)




## === cell 1
train = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
test_submissions = pd.read_csv(
    "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
)




## === cell 2
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer().fit(label_split)
y = mlb.transform(label_split)
class_list = mlb.classes_.tolist()
num_classes = len(class_list)




## === cell 3
h_target, w_target = 64, 64  # smaller size for fast feature extraction
train_img_dir = "../input/plant-pathology-2021-fgvc8/train_images"




## === cell 4
def load_image_vector_uint8(img_path):
    img = cv2.imread(img_path)  # uint8 BGR
    if img is None:
        img = np.zeros((h_target, w_target, 3), dtype=np.uint8)
    img = cv2.resize(img, (w_target, h_target), interpolation=cv2.INTER_AREA)
    return img.ravel()  # still uint8


train_image_paths = [
    os.path.join(train_img_dir, img_name) for img_name in train["image"]
]
num_train = len(train_image_paths)
X_train_uint8 = np.empty((num_train, h_target * w_target * 3), dtype=np.uint8)

max_workers = max(1, os.cpu_count())
with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    for i, vec in enumerate(
        tqdm(
            executor.map(load_image_vector_uint8, train_image_paths, chunksize=2000),
            total=num_train,
            desc="Loading train images",
        )
    ):
        X_train_uint8[i] = vec

X_train = X_train_uint8.astype(np.float32) / 255.0

train_filenames = train["image"].tolist()




## === cell 5
base_clf = LogisticRegression(max_iter=100, solver="saga", n_jobs=-1, random_state=42)
clf = OneVsRestClassifier(base_clf, n_jobs=-1)
clf.fit(X_train, y)




## === cell 6
test_img_dir = "../input/plant-pathology-2021-fgvc8/test_images"
test_image_paths = [
    os.path.join(test_img_dir, img_name) for img_name in test_submissions["image"]
]

num_test = len(test_image_paths)
X_test_uint8 = np.empty((num_test, h_target * w_target * 3), dtype=np.uint8)

max_workers = max(1, os.cpu_count())
with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    for i, vec in enumerate(
        tqdm(
            executor.map(load_image_vector_uint8, test_image_paths, chunksize=2000),
            total=num_test,
            desc="Loading test images",
        )
    ):
        X_test_uint8[i] = vec

X_test = X_test_uint8.astype(np.float32) / 255.0




## === cell 7
preds = clf.predict_proba(X_test)  # shape (num_samples, num_classes)




## === cell 8
thresh = {
    "complex": 0.25,
    "frog_eye_leaf_spot": 0.25,
    "healthy": 0.25,
    "powdery_mildew": 0.25,
    "rust": 0.25,
    "scab": 0.25,
}
for cls in class_list:
    thresh.setdefault(cls, 0.25)

label_outputs = []
healthy_idx = class_list.index("healthy") if "healthy" in class_list else None

for i in range(len(test_submissions)):
    prob = preds[i]
    if healthy_idx is not None and prob[healthy_idx] >= 0.5:
        label_outputs.append("healthy")
    else:
        chosen = [
            label for j, label in enumerate(class_list) if prob[j] > thresh[label]
        ]
        if not chosen:
            chosen = [class_list[np.argmax(prob)]]
        label_outputs.append(" ".join(chosen))

test_submissions["labels"] = label_outputs




## === cell 9
test_submissions.to_csv("submission.csv", index=False)




## === cell 10
test_submissions.head()
