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

0.9625763959938703

# 6. Current score

0.67194

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I correct the path used to locate submission CSV files, add a safety check for an empty list, and adjust the ensemble function to handle the case where only one submission is available. This ensures the script creates a valid `submission.csv` without raising an IndexError, allowing you to generate a proper submission file.'
- What this solution (achieved 0.58929) has done: 'I fixed the data‑loading paths by checking the typical Kaggle locations (./data, ./input, /kaggle/input) and picking the one that exists. This lets the script read the CSV files and images regardless of the current working directory. I also reordered the cells slightly to keep the logic clear and ensured the final submission CSV is written with the correct columns.'
- What this solution (achieved 0.66652) has done: 'I enrich the image‐level features by adding per‑channel standard deviations and the original image dimensions (width & height) to the existing mean RGB values, and I increase the forest size to give the model more capacity. These additions keep the same overall pipeline (RandomForest + MultiOutput) while providing more discriminative information, which should raise the ROC‑AUC toward the target score. The script is otherwise unchanged and still writes a correctly formatted `submission.csv`.'
- What this solution (achieved 0.67194) has done: 'I added two simple yet potentially informative features (aspect ratio and pixel count) to the image‑level feature vector, and I increased the forest size to give the model more capacity while switching to `balanced_subsample` weighting to handle class imbalance per tree. These changes keep the exact same pipeline (RandomForest + MultiOutput) but give the model richer data and a slightly stronger ensemble, which should raise the ROC‑AUC toward the target score.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier




## === cell 1
def extract_features(image_ids, folder_path):
    """Return an (n_samples, 10) array with mean RGB, std RGB,
    width, height, aspect ratio and pixel count for each image."""
    features = []
    for img_id in image_ids:
        img_path = os.path.join(folder_path, f"{img_id}.jpg")
        if not os.path.isfile(img_path):
            features.append([0.0] * 10)
            continue
        img = Image.open(img_path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0  # normalise to [0,1]
        mean_rgb = arr.mean(axis=(0, 1))
        std_rgb = arr.std(axis=(0, 1))
        width, height = img.size
        aspect_ratio = width / height if height != 0 else 0.0
        pixel_count = width * height
        feat = np.concatenate(
            [mean_rgb, std_rgb, [width, height, aspect_ratio, pixel_count]]
        )
        features.append(feat.tolist())
    return np.stack(features)




## === cell 2
_possible_base_dirs = [
    os.path.join("data", "plant-pathology-2020-fgvc7"),
    os.path.join("input", "plant-pathology-2020-fgvc7"),
    os.path.join("/kaggle", "input", "plant-pathology-2020-fgvc7"),
]

BASE_DIR = None
for p in _possible_base_dirs:
    if os.path.isdir(p):
        BASE_DIR = p
        break

if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not locate the dataset directory. Checked paths: "
        + ", ".join(_possible_base_dirs)
    )

train_path = os.path.join(BASE_DIR, "train.csv")
test_path = os.path.join(BASE_DIR, "test.csv")
images_folder = os.path.join(BASE_DIR, "images")




## === cell 3
train_df = pd.read_csv(train_path)
label_cols = ["healthy", "multiple_diseases", "rust", "scab"]
y_train = train_df[label_cols].values
train_ids = train_df["image_id"].astype(str).values

X_train = extract_features(train_ids, images_folder)

rf = RandomForestClassifier(
    n_estimators=2000,  # more trees for stronger model
    random_state=42,
    n_jobs=-1,
    max_depth=None,
    min_samples_split=2,
    class_weight="balanced_subsample",  # per‑tree balanced weighting
)
model = MultiOutputClassifier(rf)
model.fit(X_train, y_train)




## === cell 4
test_df = pd.read_csv(test_path)
test_ids = test_df["image_id"].astype(str).values
X_test = extract_features(test_ids, images_folder)

probs = []
for estimator in model.estimators_:
    prob_pos = estimator.predict_proba(X_test)[:, 1]  # probability of class 1
    probs.append(prob_pos)

test_preds = np.stack(probs, axis=1)  # shape (n_samples, n_targets)




## === cell 5
submission = test_df.copy()
submission[label_cols] = test_preds

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"{submission_path} written successfully with shape: {submission.shape}")
