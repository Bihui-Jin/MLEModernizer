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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier  # retained for reference
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from PIL import Image
from joblib import Parallel, delayed

os.environ["OMP_NUM_THREADS"] = "1"




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

label_cols = ["healthy", "multiple_diseases", "rust", "scab"]
images_dir = base_input / "images"


def load_image(img_id: str, size: tuple = (128, 128)) -> np.ndarray:
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


X_train_raw = np.stack([load_image(img_id) for img_id in train_df["image_id"]])
X_test_raw = np.stack([load_image(img_id) for img_id in test_df["image_id"]])
y_train = train_df[label_cols].values.astype(np.float32)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

test_pred = {col: np.zeros(len(test_df), dtype=np.float32) for col in label_cols}
val_scores = {}

kf = KFold(n_splits=5, shuffle=True, random_state=42)


def train_one_label(col_idx, col_name):
    """Train MLP for a single label using 5‑fold CV and return predictions."""
    fold_preds = np.zeros(len(test_df), dtype=np.float32)
    oof_preds = np.zeros(len(train_df), dtype=np.float32)

    for train_idx, val_idx in kf.split(X_train):
        mlp = MLPClassifier(
            hidden_layer_sizes=(1024, 512),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=10,
            verbose=False,
        )
        mlp.fit(X_train[train_idx], y_train[train_idx, col_idx])
        oof_preds[val_idx] = mlp.predict_proba(X_train[val_idx])[:, 1]
        fold_preds += mlp.predict_proba(X_test)[:, 1]

    test_pred_col = fold_preds / kf.n_splits
    try:
        val_auc = roc_auc_score(y_train[:, col_idx], oof_preds)
    except ValueError:
        val_auc = np.nan
    return col_name, test_pred_col, val_auc


results = Parallel(n_jobs=os.cpu_count(), backend="threading")(
    delayed(train_one_label)(i, col) for i, col in enumerate(label_cols)
)

for col_name, pred_arr, auc in results:
    test_pred[col_name] = pred_arr
    val_scores[col_name] = auc




## === cell 2
submission = sample_sub.copy()
for col in label_cols:
    submission[col] = test_pred[col]

submission = submission.set_index("image_id").loc[test_df["image_id"]].reset_index()




## === cell 3
output_path = Path("submission.csv")
submission.to_csv(output_path, index=False)
