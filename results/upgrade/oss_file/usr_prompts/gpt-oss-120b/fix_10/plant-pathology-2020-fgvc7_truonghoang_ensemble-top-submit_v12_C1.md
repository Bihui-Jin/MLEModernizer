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
from concurrent.futures import ThreadPoolExecutor
from functools import partial  # added for partial function handling




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


def image_features(image_id, base_dir, size=16):
    """
    Returns a dict with:
        - mean/std for each RGB channel (computed on the original image)
        - flattened pixel values for each channel from a single resized RGB image
    """
    filename = image_id if str(image_id).lower().endswith(".jpg") else f"{image_id}.jpg"
    img_path = os.path.join(base_dir, filename)
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            arr = np.array(img)  # (H, W, 3)
            means = arr.mean(axis=(0, 1)).astype(float)  # R, G, B
            stds = arr.std(axis=(0, 1)).astype(float)  # R, G, B

            resized_img = img.resize((size, size), Image.BILINEAR)
            resized_arr = np.array(resized_img).astype(np.float32) / 255.0
            flat_pixels = resized_arr.reshape(-1).tolist()  # 3*size*size

            return {
                "mean_r": float(means[0]),
                "mean_g": float(means[1]),
                "mean_b": float(means[2]),
                "std_r": float(stds[0]),
                "std_g": float(stds[1]),
                "std_b": float(stds[2]),
                "pixels": flat_pixels,
            }
    except Exception:
        return {
            "mean_r": 0.0,
            "mean_g": 0.0,
            "mean_b": 0.0,
            "std_r": 0.0,
            "std_g": 0.0,
            "std_b": 0.0,
            "pixels": [0.0] * (3 * size * size),
        }


images_dir = os.path.join(os.path.dirname(train_path), "images")
pixel_size = 16
pixel_len = 3 * pixel_size * pixel_size
pixel_cols = [f"pix_{i}" for i in range(pixel_len)]


def _extract_feat_list(image_ids):
    """
    Extract features for a list/Series of image ids in parallel,
    preserving original order.
    """
    max_workers = min(8, os.cpu_count() or 1)
    func = partial(image_features, base_dir=images_dir, size=pixel_size)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, image_ids))


all_ids = pd.concat([train_df["image_id"], test_df["image_id"]], ignore_index=True)
all_feats = _extract_feat_list(all_ids)

train_feats = all_feats[: len(train_df)]
test_feats = all_feats[len(train_df) :]

for df, feats in ((train_df, train_feats), (test_df, test_feats)):
    df["img_mean_r"] = [f["mean_r"] for f in feats]
    df["img_mean_g"] = [f["mean_g"] for f in feats]
    df["img_mean_b"] = [f["mean_b"] for f in feats]
    df["img_std_r"] = [f["std_r"] for f in feats]
    df["img_std_g"] = [f["std_g"] for f in feats]
    df["img_std_b"] = [f["std_b"] for f in feats]

    pix_array = np.array(
        [f["pixels"] for f in feats], dtype=np.float32
    )  # (n_samples, pixel_len)
    for i, col in enumerate(pixel_cols):
        df[col] = pix_array[:, i]

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
