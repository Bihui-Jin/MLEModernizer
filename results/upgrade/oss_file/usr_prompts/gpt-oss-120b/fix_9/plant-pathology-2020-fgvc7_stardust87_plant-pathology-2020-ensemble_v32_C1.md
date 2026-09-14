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
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline



## === cell 1
BASE_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "images")




## === cell 2
def load_image_array(image_id, size=(224, 224)):
    """
    Load an image, resize to `size`, and return a feature vector.
    The vector consists of the flattened RGB pixels together with
    per‑channel mean and standard deviation (6 extra features).
    Returned as float32 to speed up downstream training.
    """
    filename = image_id if image_id.lower().endswith(".jpg") else f"{image_id}.jpg"
    path = os.path.join(IMAGES_DIR, filename)
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize(size)
        arr = np.asarray(img, dtype=np.float32)  # keep as float32
        flat = arr.flatten()
        means = arr.mean(axis=(0, 1))
        stds = arr.std(axis=(0, 1))
        return np.concatenate([flat, means, stds]).astype(np.float32)




## === cell 3
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print("Loading training images...")
X_train_list = [load_image_array(img_id) for img_id in train_df["image_id"]]
X_train = np.stack(X_train_list).astype(np.float32)
y_train = train_df[["healthy", "multiple_diseases", "rust", "scab"]].values.astype(
    np.int8
)

print("Loading test images...")
X_test_list = [load_image_array(img_id) for img_id in test_df["image_id"]]
X_test = np.stack(X_test_list).astype(np.float32)

rf = RandomForestClassifier(
    n_estimators=1500,
    n_jobs=5,
    class_weight="balanced",
    random_state=42,
)
rf_model = MultiOutputClassifier(rf, n_jobs=5)

print("Training RandomForest...")
rf_model.fit(X_train, y_train)

logit_pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=5,
        solver="lbfgs",
        C=5.0,
    ),
)
logit_model = MultiOutputClassifier(logit_pipe, n_jobs=5)

print("Training Logistic Regression...")
logit_model.fit(X_train, y_train)

et = ExtraTreesClassifier(
    n_estimators=1500,
    n_jobs=5,
    class_weight="balanced",
    random_state=42,
)
et_model = MultiOutputClassifier(et, n_jobs=5)

print("Training ExtraTrees...")
et_model.fit(X_train, y_train)

pca_logit_pipe = make_pipeline(
    StandardScaler(),
    PCA(n_components=500, random_state=42),
    LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=5,
        solver="lbfgs",
        C=5.0,
    ),
)
pca_logit_model = MultiOutputClassifier(pca_logit_pipe, n_jobs=5)

print("Training PCA‑Logistic Regression...")
pca_logit_model.fit(X_train, y_train)

rf_probas = []
logit_probas = []
et_probas = []
pca_logit_probas = []

for i in range(y_train.shape[1]):
    rf_probas.append(rf_model.estimators_[i].predict_proba(X_test)[:, 1])
    logit_probas.append(logit_model.estimators_[i].predict_proba(X_test)[:, 1])
    et_probas.append(et_model.estimators_[i].predict_proba(X_test)[:, 1])
    pca_logit_probas.append(pca_logit_model.estimators_[i].predict_proba(X_test)[:, 1])

rf_probas = np.column_stack(rf_probas)
logit_probas = np.column_stack(logit_probas)
et_probas = np.column_stack(et_probas)
pca_logit_probas = np.column_stack(pca_logit_probas)

submission_avg = (rf_probas + logit_probas + et_probas + pca_logit_probas) / 4.0




## === cell 4
def make_submission_file(preds, test_dataframe, output_path="submission.csv"):
    """
    Write predictions to a CSV with the required format:
    image_id,healthy,multiple_diseases,rust,scab
    """
    sub_df = test_dataframe.copy()
    sub_df[["healthy", "multiple_diseases", "rust", "scab"]] = preds
    sub_df.to_csv(output_path, index=False)
    print(f"Submission written to {output_path}")


make_submission_file(submission_avg, test_df)
