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
import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from PIL import Image  # added for simple image statistics

train_path = "../input/plant-pathology-2020-fgvc7/train.csv"
train_df = pd.read_csv(train_path)

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]


def id_to_num(image_id):
    digits = "".join(filter(str.isdigit, str(image_id)))
    return int(digits) if digits else 0


def digit_sum(n):
    return sum(int(d) for d in str(n))


def count_digits(s):
    return sum(c.isdigit() for c in str(s))


def first_digit(s):
    for c in str(s):
        if c.isdigit():
            return int(c)
    return 0


def last_digit(s):
    for c in reversed(str(s)):
        if c.isdigit():
            return int(c)
    return 0


def ascii_sum(s):
    return sum(ord(c) for c in str(s))


def extract_image_stats(image_id, base_dir):
    """
    Load an image and return simple RGB channel statistics:
    mean, std, min, max for each channel.
    If the image cannot be read, return zeros.
    """
    filename = f"{image_id}.jpg"
    path = os.path.join(base_dir, filename)
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0
            mean_rgb = arr.mean(axis=(0, 1))  # (3,)
            std_rgb = arr.std(axis=(0, 1))  # (3,)
            min_rgb = arr.min(axis=(0, 1))  # (3,)
            max_rgb = arr.max(axis=(0, 1))  # (3,)
            return list(mean_rgb) + list(std_rgb) + list(min_rgb) + list(max_rgb)
    except Exception:
        return [0.0] * 12


train_df["id_num"] = train_df["image_id"].apply(id_to_num)
train_df["id_mod_10"] = train_df["id_num"] % 10
train_df["id_mod_100"] = train_df["id_num"] % 100
train_df["id_len"] = train_df["image_id"].astype(str).apply(len)

train_df["id_mod_5"] = train_df["id_num"] % 5
train_df["id_mod_20"] = train_df["id_num"] % 20
train_df["id_digit_sum"] = train_df["id_num"].apply(digit_sum)
train_df["id_digit_cnt"] = train_df["image_id"].apply(count_digits)
train_df["id_first_digit"] = train_df["image_id"].apply(first_digit)
train_df["id_last_digit"] = train_df["image_id"].apply(last_digit)
train_df["id_ascii_sum"] = train_df["image_id"].apply(ascii_sum)
train_df["id_mod_2"] = train_df["id_num"] % 2
train_df["id_mod_3"] = train_df["id_num"] % 3

image_dir = "../input/plant-pathology-2020-fgvc7/images"
img_stats = train_df["image_id"].apply(lambda x: extract_image_stats(x, image_dir))
img_stats = np.vstack(img_stats.values)
(
    train_df["img_mean_r"],
    train_df["img_mean_g"],
    train_df["img_mean_b"],
    train_df["img_std_r"],
    train_df["img_std_g"],
    train_df["img_std_b"],
    train_df["img_min_r"],
    train_df["img_min_g"],
    train_df["img_min_b"],
    train_df["img_max_r"],
    train_df["img_max_g"],
    train_df["img_max_b"],
) = img_stats.T

feature_cols = [
    "id_num",
    "id_mod_10",
    "id_mod_100",
    "id_len",
    "id_mod_5",
    "id_mod_20",
    "id_digit_sum",
    "id_digit_cnt",
    "id_first_digit",
    "id_last_digit",
    "id_ascii_sum",
    "id_mod_2",
    "id_mod_3",
    "img_mean_r",
    "img_mean_g",
    "img_mean_b",
    "img_std_r",
    "img_std_g",
    "img_std_b",
    "img_min_r",
    "img_min_g",
    "img_min_b",
    "img_max_r",
    "img_max_g",
    "img_max_b",
]

models = {}
X_train = train_df[feature_cols].values

for col in target_cols:
    y = train_df[col].values
    rf = RandomForestClassifier(
        n_estimators=1200,  # increased capacity for better learning
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y)
    models[col] = rf

class_means = train_df[target_cols].mean()
print("Class mean probabilities:", class_means.to_dict())



## === cell 1
test_path = "../input/plant-pathology-2020-fgvc7/test.csv"
test_df = pd.read_csv(test_path)

submission = pd.DataFrame()
submission["image_id"] = test_df["image_id"]

test_df["id_num"] = test_df["image_id"].apply(id_to_num)
test_df["id_mod_10"] = test_df["id_num"] % 10
test_df["id_mod_100"] = test_df["id_num"] % 100
test_df["id_len"] = test_df["image_id"].astype(str).apply(len)

test_df["id_mod_5"] = test_df["id_num"] % 5
test_df["id_mod_20"] = test_df["id_num"] % 20
test_df["id_digit_sum"] = test_df["id_num"].apply(digit_sum)
test_df["id_digit_cnt"] = test_df["image_id"].apply(count_digits)
test_df["id_first_digit"] = test_df["image_id"].apply(first_digit)
test_df["id_last_digit"] = test_df["image_id"].apply(last_digit)
test_df["id_ascii_sum"] = test_df["image_id"].apply(ascii_sum)
test_df["id_mod_2"] = test_df["id_num"] % 2
test_df["id_mod_3"] = test_df["id_num"] % 3

img_stats_test = test_df["image_id"].apply(lambda x: extract_image_stats(x, image_dir))
img_stats_test = np.vstack(img_stats_test.values)
(
    test_df["img_mean_r"],
    test_df["img_mean_g"],
    test_df["img_mean_b"],
    test_df["img_std_r"],
    test_df["img_std_g"],
    test_df["img_std_b"],
    test_df["img_min_r"],
    test_df["img_min_g"],
    test_df["img_min_b"],
    test_df["img_max_r"],
    test_df["img_max_g"],
    test_df["img_max_b"],
) = img_stats_test.T

X_test = test_df[feature_cols].values

for col in target_cols:
    prob = models[col].predict_proba(X_test)[:, 1]  # probability of class=1
    prob = np.clip(prob, 0.0, 1.0)
    submission[col] = prob

print("Submission shape:", submission.shape)



## === cell 2
submission.to_csv("submission.csv", index=False)
print("Saved submission to submission.csv")
