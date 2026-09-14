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
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ThreadPoolExecutor

load_dir = "/kaggle/input/plant-pathology-2021-fgvc8/"
train_path = os.path.join(load_dir, "train.csv")
train_img_dir = os.path.join(load_dir, "train_images")
test_img_dir = os.path.join(load_dir, "test_images")

train_df = pd.read_csv(train_path)
train_df["primary_label"] = train_df.labels.apply(lambda x: str(x).split()[0])
le = LabelEncoder()
y = le.fit_transform(train_df["primary_label"])


def rich_image_features(image_path):
    """
    Returns a 33‑dim float32 vector:
    [mean_R, mean_G, mean_B,
     std_R, std_G, std_B,
     8‑bin histogram for R (normalized),
     8‑bin histogram for G (normalized),
     8‑bin histogram for B (normalized),
     w_norm, h_norm, aspect_ratio]
    Width/height are divided by 1024; histograms are frequency‑scaled.
    On error returns a vector of 0.5.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            arr = np.array(img)  # H x W x 3, uint8
            mean = arr.mean(axis=(0, 1)).astype(np.float32)  # (3,)
            std = arr.std(axis=(0, 1)).astype(np.float32)  # (3,)
            hist_features = []
            for c in range(3):
                hist, _ = np.histogram(arr[:, :, c], bins=8, range=(0, 256))
                hist = hist.astype(np.float32) / hist.sum()  # normalize to sum‑1
                hist_features.extend(hist)
            w, h = img.size
            w_norm = w / 1024.0
            h_norm = h / 1024.0
            aspect = w / float(h) if h != 0 else 0.0
            feats = np.concatenate([mean, std, hist_features, [w_norm, h_norm, aspect]])
            return feats.astype(np.float32)
    except Exception:
        return np.full(33, 0.5, dtype=np.float32)


train_image_paths = [os.path.join(train_img_dir, fname) for fname in train_df["image"]]

max_workers = min(16, os.cpu_count() or 1)
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    train_features = list(
        executor.map(rich_image_features, train_image_paths, chunksize=20)
    )

X_train = np.stack(train_features)

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    n_jobs=5,
    random_state=42,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
)
model.fit(X_train, y)



## === cell 1
test_images = sorted(
    [
        f
        for f in os.listdir(test_img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
)
test_df = pd.DataFrame({"image": test_images})

test_image_paths = [os.path.join(test_img_dir, fname) for fname in test_df["image"]]

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    test_features = list(
        executor.map(rich_image_features, test_image_paths, chunksize=20)
    )

X_test = np.stack(test_features)

test_pred_idx = model.predict(X_test)
test_pred_labels = le.inverse_transform(test_pred_idx)

test_df["labels"] = test_pred_labels



## === cell 2
submission_path = "submission.csv"
test_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(test_df.head())
