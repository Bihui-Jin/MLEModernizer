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

0.97113

# 6. Current score

0.69045

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I replace the missing external ensemble CSV loading with a simple baseline that uses the average label frequencies from the provided training data. This removes the FileNotFoundError and the undefined *dsub* variable, ensures the script writes a correctly‑named `submission.csv` with the required columns, and keeps the core logic minimal and stable.'
- What this solution (achieved 0.54806) has done: 'I replace the constant‑mean baseline with a tiny numeric‑feature logistic‑regression model (using the numeric part of each image_id) that is trained on the provided labels. This modest model can capture simple patterns in the data and is expected to raise the ROC‑AUC from the 0.5 baseline toward the target while keeping the overall pipeline unchanged and still writing a correct `submission.csv`. If scikit‑learn is unavailable the code falls back to the original mean baseline.'
- What this solution (achieved 0.54938) has done: 'I added lightweight image‑based features (mean RGB values) and a few simple numeric descriptors derived from the image_id, and switched to a GradientBoostingClassifier which works better on this type of tabular data than a plain LogisticRegression. The code still trains one binary model per target column and falls back to the original mean baseline if Pillow isn’t available or an image can’t be read, keeping the original pipeline intact while giving a higher‑quality prediction that should move the ROC‑AUC closer to the target score.'
- What this solution (achieved 0.57319) has done: 'I enhance the feature set by adding a simple image‑metadata feature (file size) and include a more powerful tree model (HistGradientBoostingClassifier) that falls back to the original GradientBoosting if unavailable. These changes keep the overall pipeline the same while providing richer information to the learner, which should raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.69045) has done: 'I add richer image‑based features (per‑channel standard deviation and a compact color histogram) and slightly strengthen the HistGradientBoosting model (more iterations, lower learning rate). These extra features give the tree model more discriminative information while preserving the overall pipeline and per‑target binary training, which should raise the ROC‑AUC toward the target score.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np




## === cell 1
def locate(rel_path):
    candidates = [
        f"../input/plant-pathology-2020-fgvc7/{rel_path}",
        f"../input/plant-pathology/{rel_path}",
        f"../input/{rel_path}",
        f"./{rel_path}",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Unable to find {rel_path}")


print("Input folder contents (first 10):", os.listdir(locate(""))[:10])




## === cell 2
train_path = locate("train.csv")
train_df = pd.read_csv(train_path)

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]




## === cell 3
test_path = locate("test.csv")
test_df = pd.read_csv(test_path)




## === cell 4
def id_to_num(img_id):
    import re

    nums = re.findall(r"\d+", str(img_id))
    return int(nums[0]) if nums else 0


def id_features(img_id):
    """Return simple numeric features derived from the image_id."""
    s = str(img_id)
    num = id_to_num(s)
    length = len(s)
    digit_sum = sum(int(d) for d in s if d.isdigit())
    return [num, length, digit_sum]


def image_rgb_means(img_path):
    """Read an image with Pillow and return mean R,G,B values."""
    try:
        from PIL import Image

        img = Image.open(img_path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.mean(axis=(0, 1)).tolist()
    except Exception:
        return [np.nan, np.nan, np.nan]


def image_rgb_stds(img_path):
    """Return per‑channel standard deviation."""
    try:
        from PIL import Image

        img = Image.open(img_path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.std(axis=(0, 1)).tolist()
    except Exception:
        return [np.nan, np.nan, np.nan]


def image_histogram_features(img_path, bins=8):
    """
    Compute a compact color histogram (binned per channel) and return as a flat list.
    Result length = bins * 3.
    """
    try:
        from PIL import Image

        img = Image.open(img_path).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)  # 0‑255
        hist = []
        for ch in range(3):
            channel = arr[:, :, ch]
            h, _ = np.histogram(channel, bins=bins, range=(0, 256), density=True)
            hist.extend(h.tolist())
        return hist
    except Exception:
        return [np.nan] * (bins * 3)


def image_meta_features(img_path):
    """Return lightweight metadata features for an image file."""
    try:
        size_kb = os.path.getsize(img_path) / 1024.0
        return [size_kb]
    except Exception:
        return [np.nan]


def build_features(df):
    """
    Build a numeric feature matrix for a dataframe containing an `image_id` column.
    Features:
        - ID‑based numeric descriptors (3)
        - Mean RGB channels (3)
        - Std RGB channels (3)
        - Color histogram (8 bins × 3 channels = 24)
        - File size in kilobytes (1)
    """
    feats = []
    img_dir = locate("images")
    for img_id in df["image_id"]:
        f = id_features(img_id)
        img_path = os.path.join(img_dir, f"{img_id}.jpg")
        f.extend(image_rgb_means(img_path))
        f.extend(image_rgb_stds(img_path))
        f.extend(image_histogram_features(img_path, bins=8))
        f.extend(image_meta_features(img_path))
        feats.append(f)
    return np.array(feats, dtype=np.float32)


X_train = build_features(train_df)
X_test = build_features(test_df)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

try:
    from sklearn.ensemble import HistGradientBoostingClassifier

    ModelClass = HistGradientBoostingClassifier
    model_kwargs = dict(max_iter=500, learning_rate=0.05, random_state=42)
except Exception:
    from sklearn.ensemble import GradientBoostingClassifier

    ModelClass = GradientBoostingClassifier
    model_kwargs = dict(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        random_state=42,
    )

models = {}
for col in target_cols:
    y = train_df[col].values
    clf = ModelClass(**model_kwargs)
    clf.fit(X_train, y)
    models[col] = clf

preds = {col: models[col].predict_proba(X_test)[:, 1] for col in target_cols}
submission = pd.DataFrame(
    {
        "image_id": test_df["image_id"],
        "healthy": preds["healthy"],
        "multiple_diseases": preds["multiple_diseases"],
        "rust": preds["rust"],
        "scab": preds["scab"],
    }
)

print("Submission preview:")
print(submission.head())




## === cell 5
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, rows: {len(submission)}")
