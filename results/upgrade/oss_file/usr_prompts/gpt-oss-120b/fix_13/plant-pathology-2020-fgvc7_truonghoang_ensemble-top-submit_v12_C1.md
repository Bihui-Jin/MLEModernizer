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
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from PIL import Image
from concurrent.futures import ThreadPoolExecutor  # threads are fine for I/O‑bound work
from functools import partial

pixel_size = 16
pixel_len = 3 * pixel_size * pixel_size


def _process_one(image_id, base_dir, size):
    """
    Returns (means, stds, flat_pixels) as NumPy arrays.
    """
    filename = image_id if str(image_id).lower().endswith(".jpg") else f"{image_id}.jpg"
    img_path = os.path.join(base_dir, filename)
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            arr = np.array(img, dtype=np.float32)  # (H, W, 3)
            means = arr.mean(axis=(0, 1))
            stds = arr.std(axis=(0, 1))

            resized_img = img.resize((size, size), Image.BILINEAR)
            resized_arr = np.array(resized_img, dtype=np.float32) / 255.0
            flat_pixels = resized_arr.reshape(-1)  # (pixel_len,)
    except Exception:
        means = np.zeros(3, dtype=np.float32)
        stds = np.zeros(3, dtype=np.float32)
        flat_pixels = np.zeros(pixel_len, dtype=np.float32)
    return means, stds, flat_pixels


def _extract_features(image_ids, base_dir, size):
    """
    Parallel extraction of means, stds, and flattened pixels for a list of image IDs.
    Returns three NumPy arrays: means (n,3), stds (n,3), pixels (n,pixel_len)

    Optimisation: pre‑allocate output arrays and fill them in‑place to avoid
    building intermediate Python lists and repeated np.vstack calls, which
    significantly reduces memory copies and runtime.
    """
    n = len(image_ids)
    means = np.empty((n, 3), dtype=np.float32)
    stds = np.empty((n, 3), dtype=np.float32)
    pixels = np.empty((n, pixel_len), dtype=np.float32)

    max_workers = min(32, (os.cpu_count() or 1) * 2)
    func = partial(_process_one, base_dir=base_dir, size=size)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, (m, s, p) in enumerate(executor.map(func, image_ids)):
            means[idx] = m
            stds[idx] = s
            pixels[idx] = p

    return means, stds, pixels




## === cell 1
train_path = "../input/plant-pathology-2020-fgvc7/train.csv"
test_path = "../input/plant-pathology-2020-fgvc7/test.csv"
sample_path = "../input/plant-pathology-2020-fgvc7/sample_submission.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

for df in (train_df, test_df):
    df["id_num"] = df["image_id"].str.extract(r"(\d+)")[0].astype(int)
    df["id_num_sq"] = df["id_num"] ** 2

images_dir = os.path.join(os.path.dirname(train_path), "images")
pixel_cols = [f"pix_{i}" for i in range(pixel_len)]

all_ids = pd.concat([train_df["image_id"], test_df["image_id"]], ignore_index=True)
means_all, stds_all, pixels_all = _extract_features(
    all_ids.tolist(), base_dir=images_dir, size=pixel_size
)

n_train = len(train_df)
train_means, test_means = means_all[:n_train], means_all[n_train:]
train_stds, test_stds = stds_all[:n_train], stds_all[n_train:]
train_pixels, test_pixels = pixels_all[:n_train], pixels_all[n_train:]

for i, ch in enumerate(["r", "g", "b"]):
    train_df[f"img_mean_{ch}"] = train_means[:, i]
    test_df[f"img_mean_{ch}"] = test_means[:, i]
    train_df[f"img_std_{ch}"] = train_stds[:, i]
    test_df[f"img_std_{ch}"] = test_stds[:, i]

train_df[pixel_cols] = train_pixels
test_df[pixel_cols] = test_pixels

feature_cols = (
    ["id_num", "id_num_sq"]
    + ["img_mean_r", "img_mean_g", "img_mean_b", "img_std_r", "img_std_g", "img_std_b"]
    + pixel_cols
)

X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values

models = {}
for col in target_cols:
    y = train_df[col].values
    gbc = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )
    gbc.fit(X_train, y)
    models[col] = gbc




## === cell 2
preds = {}
for col in target_cols:
    preds[col] = models[col].predict_proba(X_test)[:, 1]




## === cell 3
sub = pd.read_csv(sample_path)
for col in target_cols:
    sub[col] = preds[col]
sub = sub[["image_id"] + target_cols]
sub.to_csv("submission.csv", index=False)
