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




## === cell 1
train_path = "../input/plant-pathology-2020-fgvc7/train.csv"
test_path = "../input/plant-pathology-2020-fgvc7/test.csv"
sample_path = "../input/plant-pathology-2020-fgvc7/sample_submission.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]


def id_to_num(image_id):
    nums = re.findall(r"\d+", str(image_id))
    return int(nums[0]) if nums else 0


for df in (train_df, test_df):
    df["id_num"] = df["image_id"].apply(id_to_num)
    df["id_num_sq"] = df["id_num"] ** 2


def image_features(image_id, base_dir, size=16):
    """
    Returns a dict with:
        - mean/std for each RGB channel
        - flattened pixel values for each channel (size*size)
    """
    filename = image_id if str(image_id).lower().endswith(".jpg") else f"{image_id}.jpg"
    img_path = os.path.join(base_dir, filename)
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            arr = np.array(img)  # shape (H, W, 3)
            means = arr.mean(axis=(0, 1)).astype(float)  # R, G, B
            stds = arr.std(axis=(0, 1)).astype(float)  # R, G, B
            resized = np.stack(
                [Image.fromarray(arr[:, :, c]).resize((size, size)) for c in range(3)],
                axis=2,
            )
            pixels = resized.astype(np.float32) / 255.0
            flat_pixels = pixels.reshape(-1).tolist()  # 3 * size * size values
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

for df in (train_df, test_df):
    means_r, means_g, means_b = [], [], []
    stds_r, stds_g, stds_b = [], [], []
    pix_vals = []
    for img_id in df["image_id"]:
        feats = image_features(img_id, images_dir, size=pixel_size)
        means_r.append(feats["mean_r"])
        means_g.append(feats["mean_g"])
        means_b.append(feats["mean_b"])
        stds_r.append(feats["std_r"])
        stds_g.append(feats["std_g"])
        stds_b.append(feats["std_b"])
        pix_vals.append(feats["pixels"])
    df["img_mean_r"] = means_r
    df["img_mean_g"] = means_g
    df["img_mean_b"] = means_b
    df["img_std_r"] = stds_r
    df["img_std_g"] = stds_g
    df["img_std_b"] = stds_b
    pix_array = np.array(pix_vals)  # (n_samples, pixel_len)
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
        n_estimators=500,  # slightly more trees for richer features
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
