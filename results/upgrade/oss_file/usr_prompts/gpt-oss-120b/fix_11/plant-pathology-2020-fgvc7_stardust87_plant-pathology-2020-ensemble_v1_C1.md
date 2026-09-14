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
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import concurrent.futures


def get_data_dir():
    possible_dirs = [
        "./data/plant-pathology-2020-fgvc7",
        "./input/plant-pathology-2020-fgvc7",
        "/kaggle/input/plant-pathology-2020-fgvc7",
        "./data",
        "./input",
    ]
    for d in possible_dirs:
        if os.path.isdir(d):
            return d
    raise FileNotFoundError("Data directory not found. Check paths.")


data_dir = get_data_dir()
train_path = os.path.join(data_dir, "train.csv")
test_path = os.path.join(data_dir, "test.csv")
images_folder = os.path.join(data_dir, "images")




## === cell 1
def _process_image(args):
    """Helper for parallel execution: compute feature vector for a single image."""
    img_id, folder_path = args
    img_path = os.path.join(folder_path, f"{img_id}.jpg")
    if not os.path.isfile(img_path):
        return np.zeros(40, dtype=np.float32)

    img = Image.open(img_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # normalise to [0,1]

    mean_rgb = arr.mean(axis=(0, 1))
    std_rgb = arr.std(axis=(0, 1))

    eps = 1e-6
    skew_rgb = ((arr - mean_rgb) ** 3).mean(axis=(0, 1)) / (std_rgb**3 + eps)

    width, height = img.size
    aspect_ratio = width / height if height != 0 else 0.0
    pixel_count = width * height
    log_pixel_count = np.log1p(pixel_count)

    gray_arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    mean_gray = gray_arr.mean()
    std_gray = gray_arr.std()

    hist_features = []
    for c in range(3):  # R, G, B
        hist, _ = np.histogram(
            arr[..., c].ravel(), bins=8, range=(0.0, 1.0), density=True
        )
        hist_features.extend(hist.tolist())

    feat = np.concatenate(
        [
            mean_rgb,
            std_rgb,
            skew_rgb,
            [width, height, aspect_ratio, pixel_count],
            [mean_gray, std_gray, log_pixel_count],
            hist_features,
        ]
    )
    return feat.astype(np.float32)


def extract_features(image_ids, folder_path):
    """Return an (n_samples, 40) array with basic statistics, RGB histograms,
    and per‑channel skewness. This version runs in parallel across CPU cores."""
    args_iter = ((img_id, folder_path) for img_id in image_ids)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(_process_image, args_iter, chunksize=32))
    return np.stack(results)




## === cell 2
train_df = pd.read_csv(train_path)
label_cols = ["healthy", "multiple_diseases", "rust", "scab"]
y_train = train_df[label_cols].values
train_ids = train_df["image_id"].astype(str).values

X_train = extract_features(train_ids, images_folder)

rf = RandomForestClassifier(
    n_estimators=3000,
    random_state=42,
    n_jobs=-1,
    max_depth=None,
    min_samples_split=2,
    class_weight="balanced_subsample",
)
model = MultiOutputClassifier(rf)
model.fit(X_train, y_train)



## === cell 3
test_df = pd.read_csv(test_path)
test_ids = test_df["image_id"].astype(str).values

X_test = extract_features(test_ids, images_folder)

probs = []
for estimator in model.estimators_:
    prob_pos = estimator.predict_proba(X_test)[:, 1]  # probability of class 1
    probs.append(prob_pos)

test_preds = np.stack(probs, axis=1)  # shape (n_samples, n_targets)

submission = test_df.copy()
submission[label_cols] = test_preds

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"{submission_path} written successfully with shape: {submission.shape}")
