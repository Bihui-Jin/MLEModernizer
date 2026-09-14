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

0.9646650712510072

# 6. Current score

0.61324

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.61324) has done: 'The script now safely loads the available train and test data, builds a simple image‑based model (flattened 64×64 RGB pixels with a RandomForest wrapped in a MultiOutputClassifier), predicts probabilities for each disease, and writes a correctly‑formatted `submission.csv`. It also retains the original submission‑listing logic but falls back to the model when no external submissions are found, fixing the IndexError and ensuring a valid CSV is produced. This minimal change fixes the runtime error and provides a reasonable baseline that moves the score toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier



## === cell 1
BASE_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "images")



## === cell 2
SUBMISSIONS_PATH = "/kaggle/input/submissions/"
submissions_all = []
for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
    for filename in filenames:
        submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found submissions:", submissions_all)




## === cell 3
def load_image_array(image_id, size=(64, 64)):
    """
    Load an image, resize to `size`, and return a flattened RGB array.
    image_id may already contain the .jpg extension; if not, it is added.
    """
    filename = image_id if image_id.lower().endswith(".jpg") else f"{image_id}.jpg"
    path = os.path.join(IMAGES_DIR, filename)
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize(size)
        return np.asarray(img).flatten()




## === cell 4
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print("Loading training images...")
X_train = np.stack(train_df["image_id"].apply(load_image_array).values)
y_train = train_df[["healthy", "multiple_diseases", "rust", "scab"]].values

print("Loading test images...")
X_test = np.stack(test_df["image_id"].apply(load_image_array).values)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=5,
    min_samples_split=2,
    min_samples_leaf=1,
)
model = MultiOutputClassifier(rf, n_jobs=5)
print("Training model...")
model.fit(X_train, y_train)

print("Predicting on test set...")
proba_list = []
for i, cls in enumerate(["healthy", "multiple_diseases", "rust", "scab"]):
    proba = model.estimators_[i].predict_proba(X_test)[:, 1]
    proba_list.append(proba)
submission_avg = np.column_stack(proba_list)




## === cell 5
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
