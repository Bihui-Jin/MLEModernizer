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
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import concurrent.futures  # parallel feature extraction
import multiprocessing

output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
train_img_dir = "../input/plant-pathology-2021-fgvc8/train_images/"

data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")

train_oh, val_oh = train_test_split(
    one_hot,
    test_size=0.2,
    random_state=42,
)

class_priors = one_hot.mean().values.astype(np.float32)  # (num_classes,)
dataset_labels = one_hot.columns.tolist()
num_classes = len(dataset_labels)

most_common_label = one_hot.sum().idxmax()
if most_common_label is None:
    most_common_label = "healthy"


def eval_f1(preds: np.ndarray) -> float:
    """Sample‑averaged F1 for a binary prediction matrix."""
    return f1_score(val_oh.values, preds, average="samples")


sorted_class_idxs = np.argsort(-class_priors)  # descending order
max_k = min(5, num_classes)

best_k = 1
best_f1_topk = 0.0
for k in range(1, max_k + 1):
    preds = np.zeros((val_oh.shape[0], num_classes), dtype=int)
    preds[:, sorted_class_idxs[:k]] = 1
    f1 = eval_f1(preds)
    if f1 > best_f1_topk:
        best_f1_topk = f1
        best_k = k

thresholds = np.linspace(0.0, class_priors.max(), 101)
best_thr = 0.0
best_f1_thr = 0.0
for thr in thresholds:
    mask = (class_priors >= thr).astype(int)
    preds = np.tile(mask, (val_oh.shape[0], 1))
    f1 = eval_f1(preds)
    if f1 > best_f1_thr:
        best_f1_thr = f1
        best_thr = thr

best_f1_greedy = 0.0
selected_idxs = []
preds_greedy = np.zeros((val_oh.shape[0], num_classes), dtype=int)
current_f1 = eval_f1(preds_greedy)

for idx in sorted_class_idxs:
    temp_preds = preds_greedy.copy()
    temp_preds[:, idx] = 1
    f1 = eval_f1(temp_preds)
    if f1 > current_f1 + 1e-6:
        preds_greedy = temp_preds
        current_f1 = f1
        selected_idxs.append(idx)

best_f1_greedy = current_f1


def extract_features(image_path: str) -> np.ndarray:
    """Return concatenated mean and std RGB values (scaled 0‑1) for an image."""
    with Image.open(image_path).convert("RGB") as img:
        arr = np.array(img, dtype=np.float32) / 255.0  # shape (H, W, 3)
    mean = arr.mean(axis=(0, 1))
    std = arr.std(axis=(0, 1))
    return np.concatenate([mean, std])  # shape (6,)


def compute_features(paths):
    """Extract features for a list of image paths using a thread pool.

    Pre‑allocates the output array to avoid Python‑level list overhead.
    """
    n = len(paths)
    features = np.empty((n, 6), dtype=np.float32)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=multiprocessing.cpu_count()
    ) as executor:
        for i, feat in enumerate(executor.map(extract_features, paths, chunksize=20)):
            features[i] = feat
    return features


train_imgs = data_set.loc[train_oh.index, "image"].tolist()
val_imgs = data_set.loc[val_oh.index, "image"].tolist()

train_X = compute_features([os.path.join(train_img_dir, n) for n in train_imgs])
val_X = compute_features([os.path.join(train_img_dir, n) for n in val_imgs])

clf = OneVsRestClassifier(
    LogisticRegression(
        max_iter=500,
        solver="lbfgs",
        n_jobs=-1,
        class_weight="balanced",
    )
)
clf.fit(train_X, train_oh.values)

val_probs = clf.predict_proba(val_X)  # (n_val, n_classes)

best_thr_model = 0.0
best_f1_model = 0.0
thr_grid = np.linspace(0.0, 1.0, 101)
for thr in thr_grid:
    preds = (val_probs >= thr).astype(int)
    f1 = eval_f1(preds)
    if f1 > best_f1_model:
        best_f1_model = f1
        best_thr_model = thr

if best_f1_model > max(best_f1_topk, best_f1_thr, best_f1_greedy):
    chosen_method = "prob_threshold"
    chosen_value = best_thr_model
    print(
        f"Chosen probability threshold {chosen_value:.4f} (validation F1≈{best_f1_model:.4f})"
    )
elif best_f1_thr > best_f1_topk and best_f1_thr > best_f1_greedy:
    chosen_method = "threshold"
    chosen_value = best_thr
    print(
        f"Chosen global threshold {chosen_value:.4f} (validation F1≈{best_f1_thr:.4f})"
    )
elif best_f1_topk > best_f1_greedy:
    chosen_method = "topk"
    chosen_value = best_k
    print(f"Chosen top‑k {chosen_value} (validation F1≈{best_f1_topk:.4f})")
else:
    chosen_method = "greedy"
    chosen_value = selected_idxs
    print(
        f"Chosen greedy set of {len(chosen_value)} classes (validation F1≈{best_f1_greedy:.4f})"
    )




## === cell 1
if __name__ == "__main__":

    test_images = sorted(
        [
            f
            for f in os.listdir(test_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )

    rows = []

    if chosen_method == "prob_threshold":
        test_X = compute_features(
            [os.path.join(test_dir, name) for name in test_images]
        )
        test_probs = clf.predict_proba(test_X)  # (n_test, n_classes)

    for idx, name in enumerate(test_images):
        if chosen_method == "topk":
            top_k_labels = [dataset_labels[i] for i in sorted_class_idxs[:chosen_value]]
            labels_str = " ".join(top_k_labels) if top_k_labels else most_common_label

        elif chosen_method == "threshold":
            mask = class_priors >= chosen_value
            thresh_labels = [lbl for lbl, keep in zip(dataset_labels, mask) if keep]
            labels_str = " ".join(thresh_labels) if thresh_labels else most_common_label

        elif chosen_method == "greedy":
            greedy_labels = [dataset_labels[i] for i in chosen_value]
            labels_str = " ".join(greedy_labels) if greedy_labels else most_common_label

        else:  # prob_threshold
            mask = test_probs[idx] >= chosen_value
            prob_labels = [lbl for lbl, keep in zip(dataset_labels, mask) if keep]
            labels_str = " ".join(prob_labels) if prob_labels else most_common_label

        rows.append([name, labels_str])

    submission_df = pd.DataFrame(rows, columns=["image", "labels"])
    submission_path = os.path.join(output_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")
