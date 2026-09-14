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

0.9699448210609528

# 6. Current score

0.66469

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The fix updates the input directory to the correct Kaggle path (`/kaggle/input/plant-pathology-2020-fgvc7`), ensures the training and sample‑submission files are loaded, computes the class‑wise mean probabilities, fills the submission template with those means, and writes a valid `submission.csv`. Cells are renumbered starting from 1 as required.'
- What this solution (achieved 0.47271) has done: 'I replace the constant‑mean baseline with a very simple nearest‑neighbor lookup: for each test image I extract the numeric part of its `image_id`, find the training image whose numeric ID is closest, and copy that training row’s four disease probabilities as the prediction. This uses only pandas and basic Python, keeps the overall workflow unchanged, and should raise the ROC‑AUC from the current ~0.5 toward the target without altering any core modeling logic. The script is renumbered starting at cell 1 and now writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.59387) has done: 'I replace the simple nearest‑neighbor lookup with a lightweight image‑based model: for each image I compute the mean RGB colour, train a separate logistic‑regression classifier for each disease label on these three features, and use the predicted probabilities for the submission. This keeps the pipeline simple, adds a genuine predictive signal, and is expected to raise the ROC‑AUC toward the target while still writing a valid `submission.csv`.'
- What this solution (achieved 0.64001) has done: 'I added richer image features (mean + standard deviation of each RGB channel) and used a balanced logistic‑regression (class_weight='balanced') so the model gets a stronger signal and mitigates label imbalance. These minimal, targeted changes keep the original pipeline (per‑label LR on colour features) while improving predictive power, moving the ROC‑AUC closer to the target.'
- What this solution (achieved 0.67496) has done: 'The code now parallelizes image feature extraction using a thread pool, which dramatically reduces the I/O‑bound processing time for the 1.8k images while keeping the exact same 12‑dim feature vector and overall workflow unchanged. The rest of the pipeline (scaling, model training, prediction, and submission creation) remains identical, ensuring result accuracy is preserved.'
- What this solution (achieved 0.66469) has done: 'I keep the overall pipeline unchanged but enrich the image representation: the feature extractor now also compute mean and std for the LAB colour space, giving a more discriminative 18‑dim vector. I also loosen regularisation by setting C=5 in the LogisticRegression (still balanced) and raise max_iter to 1000 for stable convergence. These focused tweaks add useful signal without altering the core per‑label LR workflow, so the ROC‑AUC should move noticeably closer to the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor

BASE_INPUT = "/kaggle/input/plant-pathology-2020-fgvc7"
TRAIN_PATH = os.path.join(BASE_INPUT, "train.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")
TEST_PATH = os.path.join(BASE_INPUT, "test.csv")
IMAGE_DIR = os.path.join(BASE_INPUT, "images")
OUTPUT_PATH = "submission.csv"

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]




## === cell 1
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)


def extract_features(image_path):
    """Return concatenated mean and std of RGB, HSV, and LAB channels (18‑dim vector)."""
    try:
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            rgb_arr = np.asarray(rgb, dtype=np.float32) / 255.0
            mean_rgb = rgb_arr.mean(axis=(0, 1))
            std_rgb = rgb_arr.std(axis=(0, 1))

            hsv = img.convert("HSV")
            hsv_arr = np.asarray(hsv, dtype=np.float32) / 255.0
            mean_hsv = hsv_arr.mean(axis=(0, 1))
            std_hsv = hsv_arr.std(axis=(0, 1))

            lab = img.convert("LAB")
            lab_arr = np.asarray(lab, dtype=np.float32) / 255.0
            mean_lab = lab_arr.mean(axis=(0, 1))
            std_lab = lab_arr.std(axis=(0, 1))

            return np.concatenate(
                [mean_rgb, std_rgb, mean_hsv, std_hsv, mean_lab, std_lab]
            )
    except Exception:
        return np.zeros(18, dtype=np.float32)




## === cell 2
def compute_features(ids):
    paths = [os.path.join(IMAGE_DIR, f"{img_id}.jpg") for img_id in ids]
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        features = list(executor.map(extract_features, paths))
    return np.vstack(features)


train_X = compute_features(train_df["image_id"])
test_X = compute_features(test_df["image_id"])




## === cell 3
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_X)
test_X_scaled = scaler.transform(test_X)

models = {}
for col in target_cols:
    lr = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        C=5.0,  # weaker regularisation for richer features
    )
    lr.fit(train_X_scaled, train_df[col].values)
    models[col] = lr

preds = np.column_stack(
    [models[col].predict_proba(test_X_scaled)[:, 1] for col in target_cols]
)




## === cell 4
submission_df = sample_sub.copy()
submission_df = submission_df.iloc[: len(test_df)].reset_index(drop=True)
submission_df["image_id"] = test_df["image_id"]
for idx, col in enumerate(target_cols):
    submission_df[col] = preds[:, idx]

submission_df.to_csv(OUTPUT_PATH, index=False)
print(f"Submission written to {OUTPUT_PATH}")
