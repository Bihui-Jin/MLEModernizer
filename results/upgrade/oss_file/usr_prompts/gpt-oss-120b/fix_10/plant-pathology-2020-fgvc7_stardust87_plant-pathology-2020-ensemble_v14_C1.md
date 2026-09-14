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

0.9700224130896464

# 6. Current score

0.53869

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The fix filters the listed CSVs to keep only real submission files (containing “submission” in the name), avoids trying to ensemble files that lack the target columns, and uses the proper sample submission as the template when writing the averaged predictions. If there are not enough valid submissions, it safely falls back to the baseline mean‑label approach, guaranteeing a `submission.csv` is produced without errors. This resolves the KeyError and ensures a correct submission file is written.'
- What this solution (achieved 0.56939) has done: 'I replace the weak baseline that only uses column means with a simple image‑based model. The script load each training image, resize it to 64 × 64, flatten the pixels and fit a scikit‑learn LogisticRegression (wrapped in MultiOutputClassifier) on the four disease labels. After training, the model predicts probabilities for every test image and writes them to `submission.csv` using the provided sample‑submission template. If any step fails the code falls back to the original mean‑label baseline, guaranteeing a valid file while moving the score much closer to the target.'
- What this solution (achieved 0.57624) has done: 'The update speeds up image loading by pre‑building a case‑insensitive filename lookup (avoiding repeated directory scans) and parallelizing the resize/flatten step with a multiprocessing pool, while keeping the exact resize size, feature ordering, and downstream logic unchanged. All other parts of the pipeline remain identical, preserving model training, prediction, and submission creation.'
- What this solution (achieved 0.60522) has done: 'I add feature scaling and dimensionality reduction (StandardScaler + PCA) before the logistic regression, which usually improves linear models on high‑dimensional image vectors and is a minimal change that keeps the overall pipeline unchanged. I also import the needed classes. This adjustment should raise the ROC‑AUC toward the target while still writing a valid submission.csv.'
- What this solution (achieved 0.5856) has done: 'I replace the linear logistic model with a RandomForest classifier (which can capture non‑linear patterns in the pixel data) and increase the retained PCA components to preserve more visual information. Both changes are small, keep the same image‑loading, scaling, and PCA pipeline, and are expected to raise the ROC‑AUC substantially toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.58734) has done: 'The changes increase model capacity and preserve the existing workflow: the PCA dimensionality is raised to keep more visual information, and the RandomForest is made richer with more trees and a `sqrt` feature‑sampling strategy, which typically improves generalisation for high‑dimensional data. These adjustments are minimal, keep the same data‑loading and prediction pipeline, and are expected to raise the ROC‑AUC toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.53869) has done: 'We modestly boost the model’s predictive power without changing the overall pipeline: increase the PCA dimensionality to keep more image information, raise the RandomForest tree count, and add a simple LogisticRegression model whose probability predictions are averaged with the RandomForest’s. These small upgrades are expected to raise the ROC‑AUC toward the target while keeping the original workflow intact and still producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import multiprocessing as mp

DATA_ROOT = "/kaggle/input/plant-pathology-2020-fgvc7"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUBMISSION = os.path.join(DATA_ROOT, "sample_submission.csv")
IMAGES_DIR = os.path.join(DATA_ROOT, "images")


def _load_single(args):
    """Helper for multiprocessing: load, resize, normalize and flatten one image."""
    img_id, img_path, size = args
    try:
        img = Image.open(img_path).convert("RGB").resize(size)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return img_id, arr.ravel()
    except Exception:
        return None  # signal failure


def load_image_vectors(df, image_dir, size=(128, 128), n_jobs=5):
    """
    Load images, resize to 128×128, and return a stacked NumPy array together with
    the list of image_ids that were successfully read.
    Optimisations:
    - Build a case‑insensitive lookup dict once instead of scanning the directory
      for every image.
    - Parallelise the per‑image processing with a multiprocessing Pool.
    The returned order matches the order of `df["image_id"]` for the images that
    exist, preserving deterministic behaviour.
    """
    file_lookup = {}
    for fname in os.listdir(image_dir):
        name, ext = os.path.splitext(fname)
        if ext.lower() in {".jpg", ".jpeg", ".png"}:
            file_lookup[name.lower()] = os.path.join(image_dir, fname)

    tasks = []
    for img_id in df["image_id"]:
        key = str(img_id).lower()
        img_path = file_lookup.get(key)
        if img_path is None:
            continue
        tasks.append((img_id, img_path, size))

    with mp.Pool(processes=n_jobs) as pool:
        results = pool.map(_load_single, tasks)

    vectors = []
    ids_kept = []
    for res in results:
        if res is not None:
            img_id, vec = res
            ids_kept.append(img_id)
            vectors.append(vec)

    if not vectors:
        raise RuntimeError("No images could be loaded.")
    return np.stack(vectors), ids_kept




## === cell 1
def train_and_predict():
    train_df = pd.read_csv(TRAIN_CSV)
    label_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    X_train, kept_ids = load_image_vectors(train_df, IMAGES_DIR)
    y_train = train_df.loc[train_df["image_id"].isin(kept_ids), label_cols].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    n_components = min(800, X_train_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)

    base_rf = RandomForestClassifier(
        n_estimators=800,  # more trees for better stability
        max_depth=None,
        max_features="sqrt",
        n_jobs=5,
        class_weight="balanced",
        random_state=42,
        bootstrap=True,
    )
    rf_clf = MultiOutputClassifier(base_rf, n_jobs=5)
    rf_clf.fit(X_train_pca, y_train)

    base_lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=5,
        solver="lbfgs",
        multi_class="ovr",
    )
    lr_clf = MultiOutputClassifier(base_lr, n_jobs=5)
    lr_clf.fit(X_train_pca, y_train)

    test_df = pd.read_csv(TEST_CSV)
    X_test, test_ids = load_image_vectors(test_df, IMAGES_DIR)

    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)

    rf_probas = rf_clf.predict_proba(X_test_pca)  # list per label
    lr_probas = lr_clf.predict_proba(X_test_pca)

    avg_proba = np.column_stack(
        [(rf[:, 1] + lr[:, 1]) / 2.0 for rf, lr in zip(rf_probas, lr_probas)]
    )

    template = pd.read_csv(SAMPLE_SUBMISSION)
    submission = pd.DataFrame(
        {
            "image_id": test_df["image_id"],
            "healthy": avg_proba[:, 0],
            "multiple_diseases": avg_proba[:, 1],
            "rust": avg_proba[:, 2],
            "scab": avg_proba[:, 3],
        }
    )
    submission = submission[template.columns]
    submission_path = "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Model‑based submission written to {submission_path}")

    return True




## === cell 2
fallback = True
try:
    import glob

    SUBMISSIONS_PATH = os.path.join(DATA_ROOT, "submissions")
    if not os.path.isdir(SUBMISSIONS_PATH):
        SUBMISSIONS_PATH = DATA_ROOT
    submission_files = [
        p
        for p in glob.glob(os.path.join(SUBMISSIONS_PATH, "**/*.csv"), recursive=True)
        if "submission" in os.path.basename(p).lower()
    ]
    if len(submission_files) >= 2:
        dfs = [
            pd.read_csv(p)[["healthy", "multiple_diseases", "rust", "scab"]].astype(
                float
            )
            for p in submission_files[:2]
        ]
        avg = (dfs[0] + dfs[1]) / 2.0
        tmpl = pd.read_csv(SAMPLE_SUBMISSION)
        tmpl[["healthy", "multiple_diseases", "rust", "scab"]] = avg.values
        tmpl.to_csv("submission.csv", index=False)
        print("Ensembled existing submissions.")
        fallback = False
except Exception as e:
    print(f"Ensembling step failed ({e}), will use model fallback.")
    fallback = True

if fallback:
    try:
        success = train_and_predict()
        if not success:
            raise RuntimeError("Model prediction failed.")
    except Exception as e:
        print(f"Model fallback failed ({e}), using mean‑label baseline.")
        train_df = pd.read_csv(TRAIN_CSV)
        baseline_probs = (
            train_df[["healthy", "multiple_diseases", "rust", "scab"]].mean().values
        )
        test_df = pd.read_csv(TEST_CSV)
        baseline_matrix = pd.DataFrame(
            [baseline_probs] * len(test_df),
            columns=["healthy", "multiple_diseases", "rust", "scab"],
        )
        submission_df = pd.concat([test_df, baseline_matrix], axis=1)
        submission_df.to_csv("submission.csv", index=False)
        print("Baseline submission.csv written successfully.")
