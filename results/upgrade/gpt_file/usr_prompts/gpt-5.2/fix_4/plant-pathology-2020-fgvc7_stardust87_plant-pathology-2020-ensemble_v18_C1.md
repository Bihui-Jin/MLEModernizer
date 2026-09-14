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

0.9694727806644228

# 6. Current score

0.5434

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The crash happens because `/kaggle/input/submissions/` doesn’t exist in this environment, so `submissions_all` is empty and indexing `[0,1,2]` fails. To make the notebook run end-to-end and always produce a valid `submission.csv`, I keep your ensemble logic but add a safe fallback: if no external submission files are found, create a baseline submission from `sample_submission.csv` with valid probabilities. I also make `ensemble()` validate indices/weights and ensure outputs are aligned to the test `image_id` order and clipped to `[0,1]`, which is score-neutral but prevents invalid submissions.'
- What this solution (achieved 0.47155) has done: 'Your current 0.5 score comes from outputting a constant 0.25 for all classes when no external submissions are found, which is essentially non-informative for ROC AUC. To move the score toward the 0.969 target with minimal changes and without changing your overall “build a submission from existing CSVs” core logic, I add a tiny fallback that trains a simple multi-label classifier on the provided `train.csv` using only `image_id` text features (character n-grams) and predicts probabilities for `test.csv`. This stays within your existing pipeline structure (producing `submission_avg` then writing `submission.csv`) and should be a meaningful improvement over 0.25 constants while remaining lightweight and deterministic. I also keep your ensemble path unchanged when `/kaggle/input/submissions/` is present.'
- What this solution (achieved 0.5434) has done: 'Your current fallback model uses only `image_id` text, which is almost unrelated to leaf health, so the AUC stays near random (~0.47–0.5). To move the score toward the 0.969 target while preserving the overall “fallback model when no external submissions” logic, I keep the same lightweight sklearn approach but switch the fallback features to the actual image pixels using a simple HOG + LogisticRegression OneVsRest pipeline. I also make the data path resolution robust to both `/kaggle/input/...` and your provided `/kaggle/data/...` layout so the code reliably finds images and CSVs. This should materially increase predictive signal (and thus ROC AUC) without changing the ensemble path or submission-writing semantics.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/"

_CANDIDATE_DATA_DIRS = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]


def _resolve_first_existing_file(rel_name: str) -> str:
    for d in _CANDIDATE_DATA_DIRS:
        p = os.path.join(d, rel_name)
        if os.path.isfile(p):
            return p
    return os.path.join(_CANDIDATE_DATA_DIRS[0], rel_name)


def _resolve_images_dir() -> str:
    for d in _CANDIDATE_DATA_DIRS:
        p = os.path.join(d, "images")
        if os.path.isdir(p):
            return p
    for d in _CANDIDATE_DATA_DIRS:
        p = os.path.join(d, "plant-pathology-2020-fgvc7", "images")
        if os.path.isdir(p):
            return p
    return os.path.join(_CANDIDATE_DATA_DIRS[0], "images")


DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
SAMPLE_SUB_PATH = _resolve_first_existing_file("sample_submission.csv")
TEST_CSV_PATH = _resolve_first_existing_file("test.csv")
TRAIN_CSV_PATH = _resolve_first_existing_file("train.csv")
IMAGES_DIR = _resolve_images_dir()

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

print("Resolved paths:")
print("  SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)
print("  TEST_CSV_PATH:", TEST_CSV_PATH)
print("  TRAIN_CSV_PATH:", TRAIN_CSV_PATH)
print("  IMAGES_DIR:", IMAGES_DIR)



## === cell 2
submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print(submissions_all)




## === cell 3
def ensemble(submissions_all, sub_idx, weights=None):
    """
    Weighted average of prediction columns across multiple submission files.

    Bugfixes / robustness:
    - Validate indices against available files.
    - Validate weights length and normalize weights to sum to 1.
    - Ensure prediction columns exist and are aligned to test image_id order.
    """
    if weights is None:
        weights = [1.0] * len(sub_idx)

    if len(sub_idx) == 0:
        raise ValueError("sub_idx is empty; nothing to ensemble.")

    if len(weights) != len(sub_idx):
        raise ValueError(
            f"weights length ({len(weights)}) must match sub_idx length ({len(sub_idx)})."
        )

    if len(submissions_all) == 0:
        raise FileNotFoundError(
            f"No submission files found under {SUBMISSIONS_PATH}. "
            "Provide CSVs there or rely on the fallback baseline submission."
        )

    valid = []
    valid_w = []
    for i, idx in enumerate(sub_idx):
        if not isinstance(idx, int):
            raise TypeError(
                f"Submission index must be int, got {type(idx)} at position {i}."
            )
        if 0 <= idx < len(submissions_all):
            valid.append(idx)
            valid_w.append(float(weights[i]))
        else:
            print(
                f"Warning: sub_idx {idx} is out of range (0..{len(submissions_all)-1}); skipping."
            )

    if len(valid) == 0:
        raise IndexError(
            "All provided sub_idx are out of range for the discovered submissions_all list."
        )

    wsum = sum(valid_w)
    if wsum == 0:
        raise ValueError("Sum of weights is 0; cannot normalize.")
    valid_w = [w / wsum for w in valid_w]

    test_ids = pd.read_csv(TEST_CSV_PATH)["image_id"].values

    submission_with_weight = []
    for i, idx in enumerate(valid):
        path = submissions_all[idx]
        print(f"I'm taking submission {path} with weight {valid_w[i]:.6f}")
        df = pd.read_csv(path)

        missing = [c for c in (["image_id"] + TARGET_COLS) if c not in df.columns]
        if missing:
            raise ValueError(f"Submission file {path} missing columns: {missing}")

        df = df[["image_id"] + TARGET_COLS].copy()
        df = df.set_index("image_id").reindex(test_ids)

        if df.isna().any().any():
            na_cols = df.columns[df.isna().any()].tolist()
            raise ValueError(
                f"Submission file {path} cannot be aligned to test.csv image_id order; NaNs in columns {na_cols}."
            )

        arr = df[TARGET_COLS].to_numpy(dtype=float)
        submission_with_weight.append(arr * valid_w[i])

    submission_avg = np.sum(submission_with_weight, axis=0)
    submission_avg = np.clip(submission_avg, 0.0, 1.0)
    return submission_avg




## === cell 4
def fallback_image_hog_model_predict():
    """
    Score-improving fallback (when no external submissions exist):
    Use simple image features (HOG) + OneVsRest LogisticRegression.

    This preserves the lightweight sklearn training approach but uses real image signal,
    which is directly relevant for ROC AUC in this competition and should move score
    much closer to the 0.969 target than image_id text.
    """
    from skimage.io import imread
    from skimage.color import rgb2gray
    from skimage.transform import resize
    from skimage.feature import hog
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    train_df = pd.read_csv(TRAIN_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)

    train_df = train_df.sort_values("image_id").reset_index(drop=True)
    test_df = test_df.sort_values("image_id").reset_index(drop=True)

    y_train = train_df[TARGET_COLS].astype(int).values

    out_size = (128, 128)

    def _img_path(image_id: str) -> str:
        return os.path.join(IMAGES_DIR, f"{image_id}.jpg")

    def _featurize(image_ids):
        feats = []
        for iid in image_ids:
            p = _img_path(iid)
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Missing image file: {p}")
            img = imread(p)
            if img.ndim == 3:
                img = rgb2gray(img)
            img = resize(
                img, out_size, anti_aliasing=True, preserve_range=False
            ).astype(np.float32)

            f = hog(
                img,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
                feature_vector=True,
            )
            feats.append(f.astype(np.float32))
        return np.vstack(feats)

    X_train = _featurize(train_df["image_id"].astype(str).values)
    X_test = _featurize(test_df["image_id"].astype(str).values)

    base = LogisticRegression(
        solver="liblinear",
        max_iter=500,
        C=2.0,
        random_state=0,
    )
    clf = OneVsRestClassifier(base)
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)
    proba = np.clip(proba, 0.0, 1.0).astype(float)

    test_ids_original = pd.read_csv(TEST_CSV_PATH)["image_id"].astype(str).values
    proba_df = pd.DataFrame(proba, columns=TARGET_COLS)
    proba_df["image_id"] = test_df["image_id"].astype(str).values
    proba_df = proba_df.set_index("image_id").reindex(test_ids_original)
    if proba_df.isna().any().any():
        raise ValueError(
            "Internal alignment error: NaNs after reindex to test.csv order."
        )
    return proba_df[TARGET_COLS].to_numpy(dtype=float)




## === cell 5
def make_submission_file(submission_avg, submissions_all=None):
    """
    Create submission.csv in the required format.
    If submissions_all is provided and non-empty, uses its first file as a template.
    Otherwise uses the official sample_submission.csv template.
    """
    if submissions_all is not None and len(submissions_all) > 0:
        template_path = submissions_all[0]
    else:
        template_path = SAMPLE_SUB_PATH

    submission_df = pd.read_csv(template_path)

    test_df = pd.read_csv(TEST_CSV_PATH)
    if "image_id" not in submission_df.columns:
        raise ValueError(f"Template {template_path} missing image_id column.")

    submission_df = submission_df.drop(
        columns=[c for c in submission_df.columns if c != "image_id"], errors="ignore"
    )
    submission_df = submission_df.merge(
        test_df[["image_id"]], on="image_id", how="right", sort=False
    )

    for c in TARGET_COLS:
        submission_df[c] = 0.0

    submission_df[TARGET_COLS] = submission_avg
    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission_df.shape)
    print(submission_df.head())




## === cell 6
if len(submissions_all) >= 3:
    submission_avg = ensemble(submissions_all, [0, 1, 2], [0.2, 0.7, 0.1])
    make_submission_file(submission_avg, submissions_all)
elif len(submissions_all) > 0:
    idxs = list(range(len(submissions_all)))
    wts = [1.0] * len(idxs)
    submission_avg = ensemble(submissions_all, idxs, wts)
    make_submission_file(submission_avg, submissions_all)
else:
    submission_avg = fallback_image_hog_model_predict()
    make_submission_file(submission_avg, submissions_all=None)
