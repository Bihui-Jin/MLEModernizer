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
scipy==1.15.3
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

0.9669491813515626

# 6. Current score

0.58761

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The fix replaces the missing external submission files with a simple, deterministic baseline: it loads the official training data, computes the average prevalence for each disease label, and fills every test sample with these mean probabilities. This removes the FileNotFound and NameError issues, ensures a valid `submission.csv` with the correct columns, and keeps the core logic minimal while still providing a reasonable prediction for the competition.'
- What this solution (achieved 0.58761) has done: 'I replace the constant‑mean baseline with a very light image‑based model: each image is resized to 32 × 32 px, flattened, and used as input to four independent LogisticRegression classifiers (one per disease label). This adds predictive signal from the actual pictures, which should raise the ROC‑AUC from the current ~0.5 toward the target while keeping the pipeline simple and fast. The rest of the script (paths, CSV handling, and final submission write‑out) remains unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from PIL import Image



## === cell 1
base_input = Path("/kaggle/input/plant-pathology-2020-fgvc7")
train_path = base_input / "train.csv"
test_path = base_input / "test.csv"
sample_sub_path = base_input / "sample_submission.csv"

if not train_path.exists():
    train_path = Path("train.csv")
if not test_path.exists():
    test_path = Path("test.csv")
if not sample_sub_path.exists():
    sample_sub_path = Path("sample_submission.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)



## === cell 2
label_cols = ["healthy", "multiple_diseases", "rust", "scab"]
images_dir = base_input / "images"


def load_image(img_id: str, size: tuple = (32, 32)) -> np.ndarray:
    """Load an image, resize to `size`, normalize to [0,1] and flatten."""
    img_path = images_dir / f"{img_id}.jpg"
    if not img_path.exists():
        return np.zeros(size[0] * size[1] * 3, dtype=np.float32)
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(size, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.flatten()
    except Exception:
        return np.zeros(size[0] * size[1] * 3, dtype=np.float32)


X_train = np.stack([load_image(img_id) for img_id in train_df["image_id"]])
y_train = train_df[label_cols].values.astype(np.float32)

models = {}
for idx, col in enumerate(label_cols):
    lr = LogisticRegression(max_iter=200, n_jobs=5, solver="lbfgs")
    lr.fit(X_train, y_train[:, idx])
    models[col] = lr



## === cell 3
X_test = np.stack([load_image(img_id) for img_id in test_df["image_id"]])

submission = sample_sub.copy()
for col in label_cols:
    probs = models[col].predict_proba(X_test)[:, 1]
    submission[col] = probs

submission = submission.set_index("image_id").loc[test_df["image_id"]].reset_index()



## === cell 4
output_path = Path("submission.csv")
submission.to_csv(output_path, index=False)
