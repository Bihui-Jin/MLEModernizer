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
import numpy as np
import pandas as pd
import os



## === cell 1
DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "/kaggle/data/plant-pathology-2020-fgvc7"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

target_cols = [c for c in sample_sub.columns if c != "image_id"]
assert "image_id" in sample_sub.columns, "sample_submission must contain image_id"
assert set(target_cols).issubset(
    set(train_df.columns)
), "Train is missing one or more target columns"

train_df.head(), test_df.head(), sample_sub.head()



## === cell 2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from PIL import Image

images_dir = os.path.join(DATA_DIR, "images")
if not os.path.isdir(images_dir):
    images_dir = os.path.join(DATA_DIR, "plant-pathology-2020-fgvc7", "images")


def _image_path(image_id: str) -> str:
    return os.path.join(images_dir, f"{image_id}.jpg")


def extract_features(image_id: str) -> np.ndarray:
    """
    Deterministic, lightweight features (keeps same overall approach: image -> vector -> LR):
      Base (8):
        - per-channel mean (3)
        - per-channel std (3)
        - brightness mean (1)
        - normalized green-red ratio (1)
      Added (24):
        - coarse per-channel histograms (8 bins * 3 channels), normalized
    Total = 32 features.
    """
    p = _image_path(image_id)
    with Image.open(p) as img:
        img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0  # [H,W,3] in [0,1]

    pix = arr.reshape(-1, 3)

    mean_rgb = pix.mean(axis=0)
    std_rgb = pix.std(axis=0)

    brightness = float(mean_rgb.mean())
    gr_ratio = float((mean_rgb[1] - mean_rgb[0]) / (mean_rgb[1] + mean_rgb[0] + 1e-6))

    bins = np.linspace(0.0, 1.0, 9, dtype=np.float32)  # 8 bins
    hists = []
    for ch in range(3):
        h, _ = np.histogram(pix[:, ch], bins=bins)
        h = h.astype(np.float32)
        h /= h.sum() + 1e-6  # normalize to remove dependence on image size
        hists.append(h)
    hist_feat = np.concatenate(hists, axis=0).astype(np.float32)  # 24 dims

    feats = np.concatenate(
        [
            mean_rgb.astype(np.float32),
            std_rgb.astype(np.float32),
            np.array([brightness, gr_ratio], dtype=np.float32),
            hist_feat,
        ]
    )
    return feats.astype(np.float32)


X_train = np.vstack([extract_features(i) for i in train_df["image_id"].values])
X_test = np.vstack([extract_features(i) for i in test_df["image_id"].values])

preds = np.zeros((len(test_df), len(target_cols)), dtype=np.float64)

for j, c in enumerate(target_cols):
    y = train_df[c].values.astype(int)

    if y.min() == y.max():
        preds[:, j] = float(y.mean())
        continue

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=800,
                    C=3.0,
                    class_weight="balanced",
                    random_state=0,
                ),
            ),
        ]
    )
    clf.fit(X_train, y)
    preds[:, j] = clf.predict_proba(X_test)[:, 1]

sub = sample_sub.copy()
sub = sub.merge(
    test_df[["image_id"]], on="image_id", how="right", validate="one_to_one"
)
for j, c in enumerate(target_cols):
    sub[c] = preds[:, j].astype(float)

sub = sub[["image_id"] + target_cols]
for c in target_cols:
    sub[c] = sub[c].clip(0.0, 1.0)

sub.to_csv("submission.csv", index=False)
sub.head()



## === cell 3
print("Wrote submission.csv")
print("Shape:", sub.shape)
print("Columns:", list(sub.columns))
print(sub.describe(include="all"))
