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



## === cell 1
BASE_PATH_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]
BASE_PATH = None
for p in BASE_PATH_CANDIDATES:
    if os.path.exists(os.path.join(p, "sample_submission.csv")) and os.path.exists(
        os.path.join(p, "test.csv")
    ):
        BASE_PATH = p
        break

if BASE_PATH is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv and test.csv in expected Kaggle paths."
    )

SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")
TEST_CSV_PATH = os.path.join(BASE_PATH, "test.csv")
TRAIN_CSV_PATH = os.path.join(BASE_PATH, "train.csv")

IMAGES_DIR_CANDIDATES = [
    os.path.join(BASE_PATH, "images"),
    os.path.join(BASE_PATH, "plant-pathology-2020-fgvc7", "images"),
]
IMAGES_DIR = None
for d in IMAGES_DIR_CANDIDATES:
    if os.path.isdir(d):
        IMAGES_DIR = d
        break
if IMAGES_DIR is None:
    raise FileNotFoundError(
        "Could not locate images directory under expected Kaggle paths."
    )

print("Using BASE_PATH:", BASE_PATH)
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)
print("TEST_CSV_PATH:", TEST_CSV_PATH)
print("TRAIN_CSV_PATH:", TRAIN_CSV_PATH)
print("IMAGES_DIR:", IMAGES_DIR)



## === cell 2
SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"

submissions_all = []
if os.path.exists(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found submission files:", submissions_all)




## === cell 3
def _make_fallback_submissions(
    sample_sub_path: str, test_csv_path: str, train_csv_path: str, images_dir: str
) -> list:
    """
    Change (score-improving, minimal semantic impact):
    - Keep the same core pipeline: create two valid submission CSVs then ensemble them.
    - Strengthen the same downsampled RGB feature baseline by using a linear SVM
      (often better margins than multinomial LR on raw pixels) and then calibrate
      to probabilities via CalibratedClassifierCV so we still output proper per-class
      probabilities required by the competition metric.
    - Standardize features (important for SVM) while keeping the rest identical.
    """
    from PIL import Image
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    sample = pd.read_csv(sample_sub_path)
    test = pd.read_csv(test_csv_path)
    train = pd.read_csv(train_csv_path)

    required_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    missing = [c for c in required_cols if c not in sample.columns]
    if missing:
        raise ValueError(f"sample_submission.csv missing columns: {missing}")

    target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    for c in target_cols:
        if c not in train.columns:
            raise ValueError(f"train.csv missing required target column: {c}")

    test_ids = test["image_id"].tolist()
    sample = sample.set_index("image_id").reindex(test_ids).reset_index()
    if sample["image_id"].isna().any():
        raise ValueError(
            "sample_submission is missing some image_id entries present in test.csv."
        )

    y = train[target_cols].to_numpy(dtype=np.int64).argmax(axis=1)
    train_ids = train["image_id"].tolist()

    def featurize(image_ids, size=48):
        X = np.zeros((len(image_ids), size * size * 3), dtype=np.float32)
        missing_count = 0
        for i, img_id in enumerate(image_ids):
            img_path = os.path.join(images_dir, f"{img_id}.jpg")
            if not os.path.exists(img_path):
                missing_count += 1
                continue
            with Image.open(img_path) as im:
                im = im.convert("RGB").resize((size, size), resample=Image.BILINEAR)
                arr = np.asarray(im, dtype=np.float32) / 255.0
                X[i] = arr.reshape(-1)
        if missing_count:
            print(
                f"Warning: {missing_count} images were missing under {images_dir}. They were featurized as zeros."
            )
        return X

    X_train = featurize(train_ids, size=48)
    X_test = featurize(test_ids, size=48)

    base_svm1 = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("svm", LinearSVC(C=2.0, class_weight=None, random_state=0, max_iter=5000)),
        ]
    )
    base_svm2 = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("svm", LinearSVC(C=0.7, class_weight=None, random_state=1, max_iter=5000)),
        ]
    )

    clf1 = CalibratedClassifierCV(base_svm1, method="sigmoid", cv=3)
    clf2 = CalibratedClassifierCV(base_svm2, method="sigmoid", cv=3)

    clf1.fit(X_train, y)
    clf2.fit(X_train, y)

    P1 = clf1.predict_proba(X_test).astype(np.float64)
    P2 = clf2.predict_proba(X_test).astype(np.float64)

    def align_probs(calibrated_model, P):
        classes = list(calibrated_model.classes_)
        if classes != [0, 1, 2, 3]:
            P_aligned = np.zeros((P.shape[0], 4), dtype=np.float64)
            for j, cls in enumerate(classes):
                P_aligned[:, int(cls)] = P[:, j]
            row_sum = P_aligned.sum(axis=1, keepdims=True)
            P_aligned = np.where(row_sum > 0, P_aligned / row_sum, 0.25)
            return P_aligned
        return P

    P1 = align_probs(clf1, P1)
    P2 = align_probs(clf2, P2)

    P1 = np.clip(P1, 1e-6, 1 - 1e-6)
    P2 = np.clip(P2, 1e-6, 1 - 1e-6)

    sub1 = sample.copy()
    sub1.loc[:, target_cols] = P1
    sub2 = sample.copy()
    sub2.loc[:, target_cols] = P2

    path1 = "fallback_submission_1.csv"
    path2 = "fallback_submission_2.csv"
    sub1.to_csv(path1, index=False)
    sub2.to_csv(path2, index=False)
    return [os.path.abspath(path1), os.path.abspath(path2)]


if len(submissions_all) == 0:
    submissions_all = _make_fallback_submissions(
        SAMPLE_SUB_PATH, TEST_CSV_PATH, TRAIN_CSV_PATH, IMAGES_DIR
    )
    print("Created fallback submissions:", submissions_all)




## === cell 4
def ensemble(submissions_all, sub_idx, weights=[]):
    if len(sub_idx) == 0:
        raise ValueError("sub_idx must contain at least one index.")
    if len(weights) == 0:
        weights = [1.0 / len(sub_idx)] * len(sub_idx)
    if len(weights) != len(sub_idx):
        raise ValueError(
            f"weights length ({len(weights)}) must match sub_idx length ({len(sub_idx)})."
        )

    submission_with_weight = []
    base_len = None

    for i in range(len(sub_idx)):
        idx = sub_idx[i]
        if idx < 0 or idx >= len(submissions_all):
            raise IndexError(
                f"sub_idx[{i}]={idx} out of range for submissions_all (len={len(submissions_all)})."
            )

        print(f"I'm taking submission {submissions_all[idx]} with weight {weights[i]}")
        submission = pd.read_csv(submissions_all[idx])

        cols = ["healthy", "multiple_diseases", "rust", "scab"]
        missing_cols = [c for c in cols if c not in submission.columns]
        if missing_cols:
            raise ValueError(
                f"Submission {submissions_all[idx]} missing columns: {missing_cols}"
            )

        arr = submission.loc[:, cols].to_numpy(dtype=np.float64)

        if base_len is None:
            base_len = arr.shape[0]
        elif arr.shape[0] != base_len:
            raise ValueError(
                f"Row count mismatch across submissions: expected {base_len}, got {arr.shape[0]}"
            )

        submission_with_weight.append(arr * float(weights[i]))

    submission_avg = np.sum(submission_with_weight, axis=0)
    return submission_avg




## === cell 5
def make_submission_file(submission_avg, submissions_all):
    submission_df = pd.read_csv(SAMPLE_SUB_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)

    test_ids = test_df["image_id"].tolist()
    submission_df = submission_df.set_index("image_id").reindex(test_ids).reset_index()
    if submission_df["image_id"].isna().any():
        raise ValueError(
            "sample_submission is missing some image_id entries present in test.csv."
        )

    cols = ["healthy", "multiple_diseases", "rust", "scab"]
    if submission_avg.shape != (len(submission_df), len(cols)):
        raise ValueError(
            f"submission_avg has shape {submission_avg.shape}, expected {(len(submission_df), len(cols))}"
        )

    submission_df.loc[:, cols] = submission_avg
    submission_df[cols] = submission_df[cols].clip(0.0, 1.0)

    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission_df.shape)
    print("Columns:", list(submission_df.columns))
    print(submission_df.head())




## === cell 6
submission_avg = ensemble(submissions_all, [0, 1], [0.24, 0.76])
make_submission_file(submission_avg, submissions_all)
