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

# 5. Code solution

## === cell 0
import os
import pandas as pd
from collections import Counter
from PIL import Image, ImageStat
import numpy as np
import concurrent.futures

from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score




## === cell 1
BASE_INPUT = "/kaggle/input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
TEST_DIR = os.path.join(BASE_INPUT, "test_images")

train_df = pd.read_csv(TRAIN_CSV)

all_labels = train_df["labels"].str.split().explode()
most_common_label = Counter(all_labels).most_common(1)[0][0]
non_healthy_labels = [lbl for lbl in all_labels if lbl != "healthy"]
most_common_nonhealthy = (
    Counter(non_healthy_labels).most_common(1)[0][0]
    if non_healthy_labels
    else most_common_label
)
print(
    f"Baseline fallback: 'healthy' when green is high, otherwise '{most_common_nonhealthy}'"
)


def mean_std_rgb_with_green_ratio(path):
    """
    Feature vector (16‑dim):
    mean R,G,B (3) + std R,G,B (3) +
    green/total, red/total, blue/total (3) +
    total_mean (1) + std/mean ratios for R,G,B (3) +
    mean H,S,V (3)
    """
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            stat = ImageStat.Stat(img)
            mean = stat.mean  # list of 3 floats
            std = stat.stddev  # list of 3 floats

            total_mean = sum(mean) + 1e-6
            green_ratio = mean[1] / total_mean
            red_ratio = mean[0] / total_mean
            blue_ratio = mean[2] / total_mean
            red_std_ratio = std[0] / total_mean
            green_std_ratio = std[1] / total_mean
            blue_std_ratio = std[2] / total_mean

            hsv_img = img.convert("HSV")
            hsv_stat = ImageStat.Stat(hsv_img)
            hsv_mean = hsv_stat.mean  # list of 3 floats (H, S, V)

            return np.array(
                mean
                + std
                + [
                    green_ratio,
                    red_ratio,
                    blue_ratio,
                    total_mean,
                    red_std_ratio,
                    green_std_ratio,
                    blue_std_ratio,
                ]
                + hsv_mean,
                dtype=np.float32,
            )
    except Exception as e:
        print(f"Warning: could not read {path}: {e}")
        return np.zeros(16, dtype=np.float32)


train_images = train_df["image"].tolist()
train_paths = [os.path.join(BASE_INPUT, "train_images", name) for name in train_images]

with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    train_features = list(executor.map(mean_std_rgb_with_green_ratio, train_paths))

X_train_raw = np.vstack(train_features)  # (n_samples, 16)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train_raw)

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(train_df["labels"].str.split())

X_tr, X_val, y_tr, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

clf = OneVsRestClassifier(
    LogisticRegression(
        solver="liblinear",
        max_iter=1000,  # increased iterations for better convergence
        random_state=42,
        class_weight="balanced",
    )
)
clf.fit(X_tr, y_tr)

probs_val = clf.predict_proba(X_val)  # (n_val, n_classes)

candidate_thresholds = np.arange(0.05, 0.91, 0.01)  # expanded range up to 0.90
best_thr = 0.35
best_f1 = 0.0
for thr in candidate_thresholds:
    y_pred_bin = (probs_val >= thr).astype(int)
    f1 = f1_score(y_val, y_pred_bin, average="macro")
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

THRESHOLD = best_thr
print(
    f"Selected probability threshold: {THRESHOLD:.2f} (validation macro F1={best_f1:.4f})"
)

clf_full = OneVsRestClassifier(
    LogisticRegression(
        solver="liblinear",
        max_iter=1000,  # same increase for the final model
        random_state=42,
        class_weight="balanced",
    )
)
clf_full.fit(X_scaled, y)
clf = clf_full  # replace the model used for final predictions


def predict_labels(paths, filenames):
    """Predict space‑separated label strings for given image paths."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        feats_raw = list(executor.map(mean_std_rgb_with_green_ratio, paths))
    X_test_raw = np.vstack(feats_raw)

    X_test = scaler.transform(X_test_raw)

    probs = clf.predict_proba(X_test)

    preds = []
    for i, (prob_vec, fname) in enumerate(zip(probs, filenames)):
        label_idxs = np.where(prob_vec >= THRESHOLD)[0]
        if len(label_idxs) == 0:
            green_ratio = X_test_raw[i][6]  # green/total ratio at index 6
            pred = "healthy" if green_ratio > 0.5 else most_common_nonhealthy
            preds.append(pred)
        else:
            pred_labels = mlb.classes_[label_idxs]
            preds.append(" ".join(pred_labels))
    return preds




## === cell 2
if __name__ == "__main__":
    test_images = sorted(
        [
            f
            for f in os.listdir(TEST_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    test_paths = [os.path.join(TEST_DIR, name) for name in test_images]

    predicted_labels = predict_labels(test_paths, test_images)

    submission_df = pd.DataFrame({"image": test_images, "labels": predicted_labels})
    output_path = os.path.join(".", "submission.csv")
    submission_df.to_csv(output_path, index=False)
    print(f"Submission written to {output_path}")
