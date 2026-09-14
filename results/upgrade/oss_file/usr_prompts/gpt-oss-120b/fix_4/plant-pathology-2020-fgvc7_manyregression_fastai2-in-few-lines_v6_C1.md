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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.91678

# 6. Current score

0.67222

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.59303) has done: 'I replace the failing fastai‐based pipeline with a lightweight sklearn model that extracts simple RGB histogram features from each image, trains a One‑Vs‑Rest logistic regression, and writes the predicted probabilities to a correctly formatted `submission.csv`. This fixes all import and name errors, ensures a valid CSV output, and provides a reasonable baseline that should move the ROC‑AUC score toward the target.'
- What this solution (achieved 0.65582) has done: 'I enhance the feature extraction to use per‑channel color histograms plus simple statistics, which give richer information than raw pixel values, and replace the linear logistic regression with a GradientBoosting classifier (still wrapped in OneVsRest) that can capture non‑linear patterns. These changes keep the overall pipeline structure while providing a stronger model, which should raise the validation ROC‑AUC toward the target score.'
- What this solution (achieved 0.67222) has done: 'I increase the visual detail captured by the features (larger resize and more histogram bins) and give the GradientBoosting model more capacity (more trees and deeper depth). These modest adjustments keep the same pipeline while providing richer inputs and a stronger learner, which should raise the validation ROC‑AUC toward the target score.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score




## === cell 1
path = Path("/kaggle/input/plant-pathology-2020-fgvc7")




## === cell 2
train_df = pd.read_csv(path / "train.csv")
LABEL_COLS = ["healthy", "multiple_diseases", "rust", "scab"]
assert all(col in train_df.columns for col in LABEL_COLS)




## === cell 3
def extract_features(img_path, size=(128, 128), bins=64):
    """
    Resize image, compute per‑channel histogram (bins) and basic stats.
    Returns a 1‑D array: [hist_R, hist_G, hist_B, mean_R, std_R, mean_G, std_G, mean_B, std_B].
    """
    img = Image.open(img_path).convert("RGB")
    img = img.resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # shape (H, W, 3)

    hist_r, _ = np.histogram(arr[..., 0], bins=bins, range=(0, 1), density=True)
    hist_g, _ = np.histogram(arr[..., 1], bins=bins, range=(0, 1), density=True)
    hist_b, _ = np.histogram(arr[..., 2], bins=bins, range=(0, 1), density=True)

    mean_r = arr[..., 0].mean()
    std_r = arr[..., 0].std()
    mean_g = arr[..., 1].mean()
    std_g = arr[..., 1].std()
    mean_b = arr[..., 2].mean()
    std_b = arr[..., 2].std()

    features = np.concatenate(
        [hist_r, hist_g, hist_b, [mean_r, std_r, mean_g, std_g, mean_b, std_b]]
    )
    return features




## === cell 4
train_image_paths = [
    path / "images" / f"{img_id}.jpg" for img_id in train_df["image_id"]
]
X = np.stack([extract_features(p) for p in train_image_paths])
y = train_df[LABEL_COLS].values




## === cell 5
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1)
)

base_clf = GradientBoostingClassifier(
    random_state=42, n_estimators=300, learning_rate=0.05, max_depth=5
)
clf = OneVsRestClassifier(base_clf)
clf.fit(X_train, y_train)

val_preds = clf.predict_proba(X_val)
val_auc = np.mean([roc_auc_score(y_val[:, i], val_preds[:, i]) for i in range(4)])
print(f"Validation mean ROC‑AUC: {val_auc:.5f}")




## === cell 6
test_df = pd.read_csv(path / "test.csv")




## === cell 7
test_image_paths = [path / "images" / f"{img_id}.jpg" for img_id in test_df["image_id"]]
X_test = np.stack([extract_features(p) for p in test_image_paths])




## === cell 8
test_preds = clf.predict_proba(X_test)  # shape (n_test, 4)




## === cell 9
submission = pd.read_csv(path / "sample_submission.csv")
submission = submission.set_index("image_id").loc[test_df["image_id"]].reset_index()
submission[LABEL_COLS] = test_preds
submission.head()




## === cell 10
submission.to_csv("submission.csv", index=False, float_format="%.8f")
print("Submission saved to submission.csv")
