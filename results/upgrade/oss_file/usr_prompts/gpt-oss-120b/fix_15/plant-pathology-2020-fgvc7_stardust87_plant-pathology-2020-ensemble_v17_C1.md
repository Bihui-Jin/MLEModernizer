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

0.9694949945692696

# 6. Current score

0.67002

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The update searches for the required CSV files wherever they reside in the repository (e.g., `./data`, `./input`, `./kaggle/data`, etc.), loads the training data to compute per‑label mean probabilities, and builds a submission using those means for every test image. This fixes the `FileNotFoundError`s and ensures a valid `submission.csv` is written while preserving the original baseline logic.'
- What this solution (achieved 0.65741) has done: 'I fixed the image‑loading logic so it correctly finds files whether the CSV ids contain the “.jpg” suffix or not, added a safe fallback that uses per‑label mean probabilities when no images are available, and guarded all later steps against missing‑image situations. This ensures the notebook runs end‑to‑end, creates a valid `submission.csv`, and keeps the original modelling approach when possible, nudging the score toward the target.'
- What this solution (achieved 0.62679) has done: 'I increase the image resolution used for feature extraction (64 × 64 instead of 32 × 32) and give the RandomForest more trees (500) to capture richer patterns. Then I blend the model’s probabilities with the simple per‑label mean baseline (85 % model + 15 % baseline), which usually nudges the ROC‑AUC upward without altering the overall pipeline.'
- What this solution (achieved 0.6297) has done: 'I increase the RandomForest size (to 800 trees) so the model can capture more patterns, and raise the blending weight of the model’s predictions from 0.85 to 0.95 (leaving only 5 % of the simple mean baseline). These modest changes keep the original pipeline intact while likely improving the ROC‑AUC toward the target score.'
- What this solution (achieved 0.63283) has done: 'I slightly boost the image resolution to 96 × 96 pixels, append simple colour statistics (means and standard deviations for each channel) to the flattened pixel vector, and rely more on the model by increasing the blending weight from 0.95 to 0.99. These minimal, non‑structural changes give the RandomForest richer features while keeping the original pipeline intact, and should nudely raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.68372) has done: 'I add modest regularisation and balance handling to the RandomForest (class_weight='balanced' and a few more trees) and lower the model‑only blending weight from 0.99 to 0.85 so the stable per‑label mean baseline contributes more. These tiny tweaks keep the original pipeline intact while encouraging a higher AUC, moving the score closer to the target.'
- What this solution (achieved 0.68372) has done: 'I increase the contribution of the RandomForest predictions by raising the `model_weight` from 0.85 to 0.95 (reducing the baseline’s influence). This small adjustment leans more on the learned model, which should improve the ROC‑AUC and move the score closer to the target while preserving the existing pipeline.'
- What this solution (achieved 0.67274) has done: 'Implemented a few targeted tweaks to push the ROC‑AUC closer to the target while preserving the original pipeline:

* Fixed the type hint for `get_image_path` to be compatible with Python 3.8.  
* Increased image resolution to 128 × 128 for richer visual features.  
* Boosted the RandomForest to 2500 trees for stronger learners.  
* Raised the model‑only blending weight to 0.98 (baseline 2 %).  
* Cleanly handle missing test images: after blending, rows corresponding to missing images are replaced with the per‑label mean baseline instead of predictions based on zero‑features.

These minimal, safe adjustments keep the core logic intact and are expected to improve the AUC toward the target score.'
- What this solution (achieved 0.67635) has done: 'I add a few cheap, model‑agnostic features (per‑channel max and min values) to the image vector and soften the aggressive 0.98 model blending to a 0.5 / 0.5 mix with the global‑mean baseline. These tweaks keep the original RandomForest pipeline intact while giving the classifier richer information and allowing the strong baseline to contribute more, which should move the ROC‑AUC noticeably closer to the target.'
- What this solution (achieved 0.66829) has done: 'I reduce the image resolution to 64 × 64 to give the RandomForest a more compact, less noisy feature set and increase the model’s contribution in the blend to 85 % (baseline 15 %). This keeps the overall pipeline unchanged while giving the model stronger influence, which should raise the ROC‑AUC toward the target.'
- What this solution (achieved 0.67002) has done: 'The changes parallelize image loading/pre‑processing with a thread pool to cut I/O and CPU overhead, pre‑allocate feature handling via lists, and reduce the RandomForest size from 3000 to 1000 trees – a still‑strong model that keeps the same algorithmic structure while fitting within the 600 s limit. The core logic, feature composition, model type, and prediction blending remain unchanged.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from typing import Optional
from concurrent.futures import ThreadPoolExecutor


def locate_file(filename: str) -> str:
    """
    Search common dataset directories for a given filename.
    Returns the first matching path as a string.
    Raises FileNotFoundError if the file cannot be found.
    """
    possible_roots = [
        Path("./data"),
        Path("./input"),
        Path("./kaggle/data"),
        Path("./kaggle/input"),
        Path("./working"),
        Path("."),
    ]
    for root in possible_roots:
        candidate = root / filename
        if candidate.is_file():
            return str(candidate)
    for path in Path(".").rglob(filename):
        if path.is_file():
            return str(path)
    raise FileNotFoundError(f"Could not locate {filename} in any expected directory.")


def locate_image_dir() -> Path:
    """
    Find the directory that contains the training images by locating any jpg file
    and returning its parent folder.
    """
    for path in Path(".").rglob("Train_*.jpg"):
        if path.is_file():
            return path.parent
    raise FileNotFoundError("Could not locate training image directory.")


def get_image_path(image_dir: Path, img_id: str) -> Optional[Path]:
    """
    Return a Path to the image file for a given id.
    Handles cases where img_id already has an extension or not.
    """
    candidate = image_dir / img_id
    if candidate.is_file():
        return candidate
    candidate_jpg = image_dir / f"{img_id}.jpg"
    if candidate_jpg.is_file():
        return candidate_jpg
    return None


def load_and_preprocess(img_path: Path, size: int = 128) -> np.ndarray:
    """
    Load an image, resize to (size, size), convert to RGB,
    flatten the pixel values, and append simple colour statistics.
    Extra stats (per‑channel max/min/median) are added to give the model a bit
    more signal without changing the overall modelling approach.
    Returns a 1‑D float array.
    """
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = img.resize((size, size))
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (size, size, 3)

    flat = arr.flatten()

    channel_means = arr.mean(axis=(0, 1))  # (3,)
    channel_stds = arr.std(axis=(0, 1))  # (3,)
    channel_max = arr.max(axis=(0, 1))  # (3,)
    channel_min = arr.min(axis=(0, 1))  # (3,)
    channel_median = np.median(arr, axis=(0, 1))  # (3,)

    features = np.concatenate(
        [flat, channel_means, channel_stds, channel_max, channel_min, channel_median]
    )
    return features.astype(np.float32)




## === cell 1
TRAIN_CSV = locate_file("train.csv")
TEST_CSV = locate_file("test.csv")
SAMPLE_SUBMISSION_CSV = locate_file("sample_submission.csv")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
X_train_ids = train_df["image_id"].astype(str).tolist()
y_all = train_df[target_cols].values.astype(np.float32)

image_dir = locate_image_dir()

IMAGE_SIZE = 128  # increased resolution for richer features


def _process_train(img_id: str):
    img_path = get_image_path(image_dir, img_id)
    if img_path is not None:
        return load_and_preprocess(img_path, size=IMAGE_SIZE), img_id
    else:
        return None, img_id


train_features = []
valid_ids = []
with ThreadPoolExecutor() as executor:
    for feat, img_id in executor.map(_process_train, X_train_ids):
        if feat is not None:
            train_features.append(feat)
            valid_ids.append(img_id)

if train_features:
    X_train = np.stack(train_features, axis=0)
    y_train = (
        train_df.set_index("image_id")
        .loc[valid_ids, target_cols]
        .values.astype(np.float32)
    )
    have_images = True
else:
    X_train = None
    y_train = None
    have_images = False




## === cell 2
if have_images:
    rf = RandomForestClassifier(
        n_estimators=1000,  # reduced from 3000 to meet runtime limits
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=1,
        class_weight="balanced",
    )
    model = MultiOutputClassifier(rf)
    model.fit(X_train, y_train)
else:
    model = None




## === cell 3
test_ids = test_df["image_id"].astype(str).tolist()
test_features = [None] * len(test_ids)
missing_test_ids = []
missing_idx = []


def _process_test(idx_img):
    idx, img_id = idx_img
    img_path = get_image_path(image_dir, img_id)
    if img_path is not None:
        return load_and_preprocess(img_path, size=IMAGE_SIZE), idx, False
    else:
        return None, idx, True


with ThreadPoolExecutor() as executor:
    for feat, idx, is_missing in executor.map(_process_test, enumerate(test_ids)):
        if not is_missing:
            test_features[idx] = feat
        else:
            missing_idx.append(idx)
            missing_test_ids.append(test_ids[idx])
            if have_images:
                test_features[idx] = np.zeros_like(train_features[0])
            else:
                test_features[idx] = np.zeros(1, dtype=np.float32)

if have_images:
    X_test = np.stack(test_features, axis=0)
    proba_list = model.predict_proba(X_test)
    model_probas = np.column_stack([p[:, 1] for p in proba_list])
else:
    model_probas = np.empty((0, len(target_cols)), dtype=np.float32)

label_means = train_df[target_cols].mean().values.reshape(1, -1)  # (1,4)

if model_probas.shape[0] == 0:
    test_probas = np.tile(label_means, (len(test_ids), 1))
else:
    model_weight = 0.95  # give more influence to the trained model
    baseline_weight = 1.0 - model_weight
    test_probas = model_weight * model_probas + baseline_weight * label_means

if missing_idx:
    test_probas[missing_idx] = label_means

submission_df = pd.DataFrame(test_probas, columns=target_cols)
submission_df.insert(0, "image_id", test_ids)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission file '{submission_path}' created with shape: {submission_df.shape}")
print(submission_df.head())
