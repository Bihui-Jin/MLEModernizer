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

3.9

# 3. Installed packages

No external packages required in the script and installed.

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

0.9039601561465894

# 6. Current score

0.6356

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The fixes remove the problematic TensorFlow import, replace the missing model loading with a simple baseline that predicts the average label frequencies from the training set, and ensure the script correctly reads the test file, creates the required columns, and writes a valid `submission.csv` with the proper format.'
- What this solution (achieved 0.47809) has done: 'The script failed because it looked for the CSV files in a fixed relative directory that does not exist in the current environment. I added a lightweight path‑search that scans common Kaggle input locations (and the current folder) to locate *train.csv* and *test.csv*. This makes the code robust to the actual data location while keeping the original grouping‑based prediction logic unchanged, ensuring a valid `submission.csv` is written.'
- What this solution (achieved 0.5382) has done: 'I keep the overall pipeline but improve the prediction logic: instead of using only the modulo‑10 group, I group images by the integer part of their id divided by 10 (e.g., 370 → 37) which provides finer, more meaningful clusters. I then blend each group’s mean label frequencies with the global means (60 % group, 40 % overall) to give a better calibrated probability while preserving the original simple‑average approach. This modest change is expected to raise the ROC‑AUC toward the target without altering the core workflow.'
- What this solution (achieved 0.6452) has done: 'I replace the simplistic group‑mean blending with a lightweight image‑based model that extracts mean RGB values from each leaf picture and trains a separate logistic‑regression classifier for every disease label. This adds informative visual features while keeping the pipeline simple and fast, which should raise the ROC‑AUC toward the target. I also add a small helper to locate the image folder relative to the CSV files and ensure the submission CSV is written with the correct columns.'
- What this solution (achieved 0.64503) has done: 'I enhance the lightweight image‑based model by adding a simple numeric identifier feature (derived from the image file name) and increase the logistic‑regression regularisation parameter C to allow a slightly more flexible decision boundary. Both the RGB mean/std features and the new id feature are standard‑scaled together before training. These modest changes keep the overall pipeline unchanged while providing extra signal that should raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.66529) has done: 'I add a few inexpensive visual features (HSV channel statistics) to the existing RGB mean/std, and give the logistic model a slightly higher regularisation‑strength (C) so it can exploit the extra signal. These changes keep the same overall pipeline (image loading, scaling, OneVsRest logistic regression) while providing more informative inputs, which should raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.6651) has done: 'I add a lightweight image‑size feature (width / height) to the existing RGB + HSV statistics and increase the logistic regression regularisation strength (C) slightly. These two small changes give the model a bit more discriminative power without altering the overall pipeline, which should raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.67041) has done: 'I slightly adjust the logistic‑regression hyper‑parameters (reduce C, raise max_iter) to improve generalisation and blend its predictions with a simple global‑frequency baseline, which adds a small amount of label‑wise prior information without altering the overall pipeline. These minimal changes keep the original feature extraction and model structure while aiming to raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.66937) has done: 'I slightly strengthen the logistic‑regression model (increase C to 2.0 and max_iter to 1000) so it can capture more signal from the visual features, and I give the model a larger influence in the final blend (raise the blending coefficient from 0.85 to 0.90). These minimal tweaks keep the same feature extraction and overall pipeline while expectedly raising the ROC‑AUC toward the target score.'
- What this solution (achieved 0.63695) has done: 'I enhance the feature set by adding a normalized 16‑bin grayscale histogram to each image’s statistics and raise the logistic‑regression regularisation strength (C) to allow the model to capture more signal. I also slightly increase the blending weight toward the learned model (0.95 vs 0.90). These minimal changes keep the overall pipeline intact while providing extra visual information expected to raise the ROC‑AUC toward the target.'
- What this solution (achieved 0.6356) has done: 'I raise the model’s capacity and give its predictions a larger influence in the final blend. Specifically, the logistic‑regression classifier use a higher C (10.0) and the ‘lbfgs’ solver with more iterations, which often yields better calibrated probabilities for multi‑label problems. Then the blended output rely 99 % on the learned model and only 1 % on the global‑frequency baseline, moving the ROC‑AUC closer to the target while keeping the overall pipeline unchanged.'

# 9. Code solution

## === cell 0
import os
import re
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler


def _find_csv(filename):
    """
    Search for `filename` in typical Kaggle data folders and the current directory.
    Returns the first matching path as a string, or raises FileNotFoundError.
    """
    search_dirs = [
        Path("./data/plant-pathology-2020-fgvc7"),
        Path("./kaggle/input/plant-pathology-2020-fgvc7"),
        Path("./input/plant-pathology-2020-fgvc7"),
        Path("."),
    ]
    for d in search_dirs:
        candidate = d / filename
        if candidate.is_file():
            return str(candidate)
    for p in Path(".").rglob(filename):
        if p.is_file():
            return str(p)
    raise FileNotFoundError(f"Unable to locate {filename} in expected locations.")


def _find_images_dir():
    """
    Locate the directory that contains the image files (Train_*.jpg, Test_*.jpg).
    It is assumed to be a sibling 'images' folder next to the CSV files.
    """
    csv_path = Path(_find_csv("train.csv"))
    possible = [
        csv_path.parent / "images",
        csv_path.parent.parent / "images",
        Path("./data/plant-pathology-2020-fgvc7/images"),
        Path("./kaggle/input/plant-pathology-2020-fgvc7/images"),
        Path("./input/plant-pathology-2020-fgvc7/images"),
    ]
    for p in possible:
        if p.is_dir():
            return str(p)
    for p in Path(".").rglob("Train_*.jpg"):
        return str(p.parent)
    raise FileNotFoundError("Unable to locate images directory.")


def _numeric_id(img_id):
    """
    Extract the first integer appearing in the image_id string.
    If none found, return 0.
    """
    m = re.search(r"\d+", str(img_id))
    return int(m.group()) if m else 0


def _extract_features(image_path):
    """
    Load an image and compute lightweight visual statistics:
      - RGB channel mean & std (3 + 3)
      - HSV channel mean & std (3 + 3)
      - Aspect ratio (width / height) (1)
      - Normalized 16‑bin grayscale histogram (16)
    Returns a 30‑dimensional feature vector.
    """
    with Image.open(image_path) as img:
        orig_w, orig_h = img.size
        img = img.convert("RGB")
        img = img.resize((48, 48))

        arr_rgb = np.array(img) / 255.0  # (48,48,3)
        rgb_means = arr_rgb.mean(axis=(0, 1))
        rgb_stds = arr_rgb.std(axis=(0, 1))

        arr_hsv = np.array(img.convert("HSV")) / 255.0
        hsv_means = arr_hsv.mean(axis=(0, 1))
        hsv_stds = arr_hsv.std(axis=(0, 1))

        gray = img.convert("L")
        gray_arr = np.array(gray).ravel()
        hist, _ = np.histogram(gray_arr, bins=16, range=(0, 256), density=True)

        aspect_ratio = np.array([orig_w / orig_h if orig_h != 0 else 0.0])

    return np.concatenate(
        [rgb_means, rgb_stds, hsv_means, hsv_stds, aspect_ratio, hist]
    )  # (30,)


def compute_image_based_predictions():
    train_path = _find_csv("train.csv")
    test_path = _find_csv("test.csv")
    images_dir = _find_images_dir()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    label_cols = ["healthy", "multiple_diseases", "rust", "scab"]

    train_feats = []
    train_id_nums = []
    for img_id in train_df["image_id"]:
        img_file = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.isfile(img_file):
            candidates = list(Path(images_dir).rglob(f"{img_id}.*"))
            img_file = str(candidates[0]) if candidates else None
        if img_file and os.path.isfile(img_file):
            feats = _extract_features(img_file)
        else:
            feats = np.zeros(30)
        train_feats.append(feats)
        train_id_nums.append(_numeric_id(img_id))

    X_color_train = np.vstack(train_feats)
    X_id_train = np.asarray(train_id_nums).reshape(-1, 1)
    X_train_raw = np.hstack([X_color_train, X_id_train])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)

    y_train = train_df[label_cols].values

    base_clf = LogisticRegression(
        solver="lbfgs",
        class_weight="balanced",
        max_iter=5000,
        C=10.0,
        random_state=42,
        multi_class="ovr",
    )
    clf = OneVsRestClassifier(base_clf)
    clf.fit(X_train, y_train)

    test_feats = []
    test_id_nums = []
    for img_id in test_df["image_id"]:
        img_file = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.isfile(img_file):
            candidates = list(Path(images_dir).rglob(f"{img_id}.*"))
            img_file = str(candidates[0]) if candidates else None
        if img_file and os.path.isfile(img_file):
            feats = _extract_features(img_file)
        else:
            feats = np.zeros(30)
        test_feats.append(feats)
        test_id_nums.append(_numeric_id(img_id))

    X_color_test = np.vstack(test_feats)
    X_id_test = np.asarray(test_id_nums).reshape(-1, 1)
    X_test_raw = np.hstack([X_color_test, X_id_test])
    X_test = scaler.transform(X_test_raw)

    probs = clf.predict_proba(X_test)  # (n_test, n_labels)

    global_means = train_df[label_cols].mean().values
    baseline = np.tile(global_means, (probs.shape[0], 1))

    blended = 0.99 * probs + 0.01 * baseline

    pred_df = pd.DataFrame(blended, columns=label_cols)
    pred_df.insert(0, "image_id", test_df["image_id"])
    return pred_df




## === cell 1
def write_submission(df, output_path="./submission.csv"):
    """
    Ensure the submission follows the exact column order required by Kaggle.
    """
    cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    df = df[cols]
    df.to_csv(output_path, index=False)
    print(f"Submission written to {output_path}")




## === cell 2
if __name__ == "__main__":
    submission_df = compute_image_based_predictions()
    print("First few predictions:")
    print(submission_df.head())
    write_submission(submission_df)
