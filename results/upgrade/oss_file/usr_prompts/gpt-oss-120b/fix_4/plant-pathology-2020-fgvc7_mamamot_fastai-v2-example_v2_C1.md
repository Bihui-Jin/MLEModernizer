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
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                sample_submission.csv (184 lines)
                ... and 2 other files
                images/
                    Train_744.jpg (218.7 kB)
                    Train_541.jpg (194.0 kB)
                    ... and 1819 other files
        input/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                sample_submission.csv (184 lines)
                ... and 2 other files
                images/
                    Train_744.jpg (218.7 kB)
                    Train_541.jpg (194.0 kB)
                    ... and 1819 other files
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                sample_submission.csv (184 lines)
                ... and 2 other files
                images/
                    Train_744.jpg (218.7 kB)
                    Train_541.jpg (194.0 kB)
                    ... and 1819 other files
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
Here is some information about the columns:
healthy (float64) has 1 unique values: [0.25]
image_id (object) has 183 unique values. Some example values: ['Test_0', 'Test_115', 'Test_117', 'Test_118']
multiple_diseases (float64) has 1 unique values: [0.25]
rust (float64) has 1 unique values: [0.25]
scab (float64) has 1 unique values: [0.25]

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
Here is some information about the columns:
image_id (object) has 183 unique values. Some example values: ['Test_0', 'Test_115', 'Test_117', 'Test_118']

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
Here is some information about the columns:
healthy (int64) has 2 unique values: [0, 1]
image_id (object) has 1638 unique values. Some example values: ['Train_0', 'Train_1088', 'Train_1098', 'Train_1097']
multiple_diseases (int64) has 2 unique values: [0, 1]
rust (int64) has 2 unique values: [1, 0]
scab (int64) has 2 unique values: [0, 1]

-> input/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
Here is some information about the columns:
healthy (float64) has 1 unique values: [0.25]
image_id (object) has 183 unique values. Some example values: ['Test_0', 'Test_115', 'Test_117', 'Test_118']
multiple_diseases (float64) has 1 unique values: [0.25]
rust (float64) has 1 unique values: [0.25]
scab (float64) has 1 unique values: [0.25]

-> input/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
Here is some information about the columns:
image_id (object) has 183 unique values. Some example values: ['Test_0', 'Test_115', 'Test_117', 'Test_118']

-> input/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
Here is some information about the columns:
healthy (int64) has 2 unique values: [0, 1]
image_id (object) has 1638 unique values. Some example values: ['Train_0', 'Train_1088', 'Train_1098', 'Train_1097']
multiple_diseases (int64) has 2 unique values: [0, 1]
rust (int64) has 2 unique values: [1, 0]
scab (int64) has 2 unique values: [0, 1]

-> working/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
Here is some information about the columns:
healthy (float64) has 1 unique values: [0.25]
image_id (object) has 183 unique values. Some example values: ['Test_0', 'Test_115', 'Test_117', 'Test_118']
multiple_diseases (float64) has 1 unique values: [0.25]
rust (float64) has 1 unique values: [0.25]
scab (float64) has 1 unique values: [0.25]

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from concurrent.futures import ThreadPoolExecutor



## === cell 1
possible_roots = [
    Path("data/plant-pathology-2020-fgvc7"),
    Path("/kaggle/input/plant-pathology-2020-fgvc7"),
    Path("/kaggle/input/plant-pathology-2020-fgvc7/data"),
    Path("input/plant-pathology-2020-fgvc7"),
    Path("working/plant-pathology-2020-fgvc7"),
]
data_path = None
for p in possible_roots:
    if (p / "train.csv").exists():
        data_path = p
        break
if data_path is None:
    raise FileNotFoundError(
        "Could not locate train.csv in any expected data directories."
    )

train_path = data_path / "train.csv"
test_path = data_path / "test.csv"
images_dir = data_path / "images"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

label_cols = ["healthy", "multiple_diseases", "rust", "scab"]




## === cell 2
def load_images(df, img_dir, size=(64, 64)):
    """
    Reads images listed in df['image_id'] using a thread pool for I/O parallelism.
    Returns a NumPy array of shape (n_samples, size[0]*size[1]*3).
    Images are resized to `size` and converted to RGB.
    """

    def _process(img_id):
        img_path = img_dir / f"{img_id}.jpg"
        if not img_path.is_file():
            raise FileNotFoundError(f"Image file not found: {img_path}")
        with Image.open(img_path) as im:
            im = im.convert("RGB").resize(size)
            return np.asarray(im, dtype=np.float32).flatten()

    with ThreadPoolExecutor() as executor:
        img_arrays = list(executor.map(_process, df["image_id"]))
    return np.stack(img_arrays) / 255.0  # normalise to [0,1]




## === cell 3
X_full = load_images(train_df, images_dir, size=(64, 64))
y_full = train_df[label_cols].values.astype(np.int64)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)



## === cell 4
models = {}
for i, col in enumerate(label_cols):
    lr = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="saga",
        max_iter=1000,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    lr.fit(X_tr, y_tr[:, i])
    models[col] = lr



## === cell 5
val_scores = []
for i, col in enumerate(label_cols):
    prob = models[col].predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val[:, i], prob)
    val_scores.append(auc)
mean_auc = np.mean(val_scores)
print(f"Validation mean ROC‑AUC: {mean_auc:.5f}")



## === cell 6
for i, col in enumerate(label_cols):
    models[col].fit(X_full, y_full[:, i])



## === cell 7
X_test = load_images(test_df, images_dir, size=(64, 64))
preds = {}
for col in label_cols:
    preds[col] = models[col].predict_proba(X_test)[:, 1]



## === cell 8
submission = pd.DataFrame()
submission["image_id"] = test_df["image_id"]
for col in label_cols:
    submission[col] = preds[col]

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")
