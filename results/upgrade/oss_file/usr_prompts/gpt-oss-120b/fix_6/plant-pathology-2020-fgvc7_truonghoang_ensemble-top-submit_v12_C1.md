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

# 5. Target score

0.9697842817939653

# 6. Current score

0.59569

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I replace the failing ensemble code with a simple baseline that computes the average label frequencies from the training set and fills the submission with those values. This removes the invalid file reads, eliminates the undefined `dsub` variable, and guarantees a correctly‑formatted `submission.csv` is written.'
- What this solution (achieved 0.54806) has done: 'I replace the constant‑mean baseline with a tiny numeric‑ID model: each image_id’s numeric part is used as a single feature and a separate LogisticRegression is fitted for every target column. This introduces variation in the predictions (instead of a flat 0.5 score) and should raise the ROC‑AUC toward the target while keeping the overall pipeline simple and unchanged otherwise. The script still reads the same CSVs, builds a `submission.csv` with the required columns, and avoids any heavy libraries or major architectural changes.'
- What this solution (achieved 0.54806) has done: 'I add two simple numeric features derived from the image id (the raw id and its square) and train each LogisticRegression model with a higher iteration limit, a larger regularisation strength, and class‑weight balancing. These low‑cost feature extensions and regularisation tweaks keep the original pipeline intact while giving the model more expressive power, which should raise the ROC‑AUC toward the target without large architectural changes.'
- What this solution (achieved 0.52556) has done: 'I add lightweight image‑based features (average pixel intensity and its standard deviation) to the existing numeric ID features, and switch the per‑column classifier from LogisticRegression to a GradientBoostingClassifier, which can capture non‑linear relationships while keeping the same overall pipeline. These changes remain minimal, preserve the original workflow, and are expected to raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.59569) has done: 'I enrich the image‑based features by adding a small down‑scaled grayscale pixel vector (8×8 = 64 values) for each picture and keep the existing numeric ID features. The GradientBoosting model use a slightly larger forest (400 trees, depth 4) to exploit the extra visual information. These minimal, targeted changes keep the original pipeline intact while providing far more discriminative data, which should raise the ROC‑AUC toward the target score.'

# 9. Code solution

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


def image_features(image_id, base_dir, size=8):
    """
    Returns:
        mean_gray, std_gray, flat_pixels (list of length size*size)
    """
    filename = image_id if str(image_id).lower().endswith(".jpg") else f"{image_id}.jpg"
    img_path = os.path.join(base_dir, filename)
    try:
        with Image.open(img_path) as img:
            gray = img.convert("L")
            arr = np.array(gray)
            mean = float(arr.mean())
            std = float(arr.std())
            resized = gray.resize((size, size))
            pixels = np.array(resized).astype(np.float32).flatten() / 255.0
            return mean, std, pixels.tolist()
    except Exception:
        return 0.0, 0.0, [0.0] * (size * size)


images_dir = os.path.join(os.path.dirname(train_path), "images")

pixel_size = 8
pixel_len = pixel_size * pixel_size
pixel_cols = [f"pix_{i}" for i in range(pixel_len)]

for df in (train_df, test_df):
    means, stds, pix_vals = [], [], []
    for img_id in df["image_id"]:
        m, s, pix = image_features(img_id, images_dir, size=pixel_size)
        means.append(m)
        stds.append(s)
        pix_vals.append(pix)
    df["img_mean"] = means
    df["img_std"] = stds
    pix_array = np.array(pix_vals)  # shape (n_samples, pixel_len)
    for i, col in enumerate(pixel_cols):
        df[col] = pix_array[:, i]

feature_cols = ["id_num", "id_num_sq", "img_mean", "img_std"] + pixel_cols
X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values

models = {}
for col in target_cols:
    y = train_df[col].values
    gbc = GradientBoostingClassifier(
        n_estimators=400,  # increased capacity
        learning_rate=0.05,
        max_depth=4,  # slightly deeper trees
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
