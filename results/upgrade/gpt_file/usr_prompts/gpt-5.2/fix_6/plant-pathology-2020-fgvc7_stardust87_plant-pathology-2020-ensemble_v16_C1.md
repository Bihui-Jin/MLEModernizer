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

BASE_PATH = "/kaggle/input/plant-pathology-2020-fgvc7"
ALT_BASE_PATH = "/kaggle/input"

SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_PATH = os.path.join(BASE_PATH, "train.csv")
TEST_PATH = os.path.join(BASE_PATH, "test.csv")

IMAGES_DIR_CANDIDATES = [
    os.path.join(BASE_PATH, "images"),
    os.path.join(ALT_BASE_PATH, "plant-pathology-2020-fgvc7", "images"),
    os.path.join(ALT_BASE_PATH, "images"),
]
IMAGES_DIR = None
for p in IMAGES_DIR_CANDIDATES:
    if os.path.isdir(p):
        IMAGES_DIR = p
        break

SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"
SEARCH_ROOTS = [SUBMISSIONS_PATH, BASE_PATH, ALT_BASE_PATH]

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

print("IMAGES_DIR:", IMAGES_DIR)
print("SAMPLE_SUB_PATH exists:", os.path.exists(SAMPLE_SUB_PATH))
print("TRAIN_PATH exists:", os.path.exists(TRAIN_PATH))
print("TEST_PATH exists:", os.path.exists(TEST_PATH))




## === cell 1
def _looks_like_submission_csv(path: str) -> bool:
    try:
        head = pd.read_csv(path, nrows=5)
    except Exception:
        return False
    cols = set(head.columns)
    if "image_id" not in cols:
        return False
    if not any(c in cols for c in TARGET_COLS):
        return False
    base = os.path.basename(path).lower()
    if base in ("train.csv", "test.csv", "sample_submission.csv"):
        return False
    return True


submissions_all = []
seen = set()

for root in SEARCH_ROOTS:
    if root and os.path.exists(root):
        for dirname, _, filenames in os.walk(root):
            for filename in filenames:
                if not filename.lower().endswith(".csv"):
                    continue
                path = os.path.join(dirname, filename)
                if path in seen:
                    continue
                seen.add(path)
                if _looks_like_submission_csv(path):
                    submissions_all.append(path)

submissions_all.sort()
print("Found submission-like CSV files:", submissions_all)




## === cell 2
def ensemble(
    submissions_all, sub_idx, weights=None, sample_submission_path=SAMPLE_SUB_PATH
):
    if weights is None:
        weights = [1.0] * len(sub_idx)
    if len(sub_idx) != len(weights):
        raise ValueError(
            f"sub_idx (len={len(sub_idx)}) and weights (len={len(weights)}) must match."
        )

    sample = pd.read_csv(sample_submission_path)
    if "image_id" not in sample.columns:
        raise KeyError("sample_submission must contain image_id")

    submission_with_weight = []
    for i in range(len(sub_idx)):
        idx = sub_idx[i]
        if idx < 0 or idx >= len(submissions_all):
            raise IndexError(
                f"Requested submission index {idx}, but only {len(submissions_all)} files were found."
            )
        path = submissions_all[idx]
        print(f"I'm taking submission {path} with weight {weights[i]}")
        submission = pd.read_csv(path)

        missing = [
            c for c in (["image_id"] + TARGET_COLS) if c not in submission.columns
        ]
        if missing:
            raise KeyError(f"Submission {path} is missing columns: {missing}")

        submission["image_id"] = submission["image_id"].astype(str)
        sub_aligned = sample[["image_id"]].merge(
            submission[["image_id"] + TARGET_COLS],
            on="image_id",
            how="left",
            sort=False,
        )

        for c in TARGET_COLS:
            col = pd.to_numeric(sub_aligned[c], errors="coerce").astype("float64")
            if col.isna().any():
                fillv = float(col.mean()) if col.notna().any() else 0.25
                sub_aligned[c] = col.fillna(fillv)

        preds = sub_aligned[TARGET_COLS].astype("float64").values
        submission_with_weight.append(preds * float(weights[i]))

    submission_avg = sum(submission_with_weight) / float(sum(weights))
    return submission_avg




## === cell 3
def make_submission_file(submission_avg, sample_submission_path):
    submission_df = pd.read_csv(sample_submission_path)

    for c in TARGET_COLS:
        if c not in submission_df.columns:
            raise KeyError(f"sample_submission is missing expected column: {c}")

    submission_df.loc[:, TARGET_COLS] = submission_avg
    submission_df.loc[:, TARGET_COLS] = submission_df.loc[:, TARGET_COLS].clip(0.0, 1.0)

    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission_df.shape)
    print(submission_df.head())




## === cell 4
def _read_image_basic_stats(path: str):
    """
    Returns a compact deterministic feature vector from an image.

    Change (score): extend the same global-stat feature idea with a few extra
    low-cost stats that capture spotting/contrast shifts (min/max, grayscale mean/std),
    which should improve separability and move AUC upward toward the target.
    """
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.float32) / 255.0  # H,W,3
            flat = arr.reshape(-1, 3)

            mean_rgb = flat.mean(axis=0)  # (3,)
            std_rgb = flat.std(axis=0)  # (3,)
            min_rgb = flat.min(axis=0)  # (3,)
            max_rgb = flat.max(axis=0)  # (3,)

            r, g, b = mean_rgb[0], mean_rgb[1], mean_rgb[2]
            eps = 1e-6
            r_ratio = r / (g + b + eps)
            g_ratio = g / (r + b + eps)
            b_ratio = b / (r + g + eps)

            gray = 0.2989 * flat[:, 0] + 0.5870 * flat[:, 1] + 0.1140 * flat[:, 2]
            gray_mean = float(gray.mean())
            gray_std = float(gray.std())

            feats = np.array(
                [
                    mean_rgb[0],
                    mean_rgb[1],
                    mean_rgb[2],
                    std_rgb[0],
                    std_rgb[1],
                    std_rgb[2],
                    min_rgb[0],
                    min_rgb[1],
                    min_rgb[2],
                    max_rgb[0],
                    max_rgb[1],
                    max_rgb[2],
                    r_ratio,
                    g_ratio,
                    b_ratio,
                    gray_mean,
                    gray_std,
                ],
                dtype=np.float32,
            )
            return feats
    except Exception:
        return None


def build_features(image_ids, images_dir):
    n_feats = 17  # must match _read_image_basic_stats output
    X = np.zeros((len(image_ids), n_feats), dtype=np.float32)
    missing = 0
    for i, img_id in enumerate(image_ids):
        fname = str(img_id)
        if not fname.lower().endswith(".jpg"):
            fname = fname + ".jpg"
        path = os.path.join(images_dir, fname)
        feat = _read_image_basic_stats(path)
        if feat is None:
            missing += 1
            X[i] = np.array(
                [
                    0.5,
                    0.5,
                    0.5,
                    0.1,
                    0.1,
                    0.1,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.5,
                    0.1,
                ],
                dtype=np.float32,
            )
        else:
            X[i] = feat.astype(np.float32)
    return X, missing


def fallback_train_image_model_and_predict(
    train_path, test_path, sample_submission_path, images_dir
):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_df = pd.read_csv(sample_submission_path)

    test_ids = sample_df["image_id"].astype(str).tolist()
    train_ids = train_df["image_id"].astype(str).tolist()

    X_train, miss_tr = build_features(train_ids, images_dir)
    X_test, miss_te = build_features(test_ids, images_dir)

    y_train = train_df[TARGET_COLS].astype(int).values

    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    base_lr = LogisticRegression(
        solver="lbfgs",
        max_iter=800,
        C=2.0,
        class_weight=None,
        n_jobs=None,
        random_state=0,
    )
    clf = MultiOutputClassifier(
        Pipeline([("scaler", StandardScaler()), ("lr", base_lr)])
    )
    clf.fit(X_train, y_train)

    probs = np.zeros((len(test_ids), len(TARGET_COLS)), dtype=np.float64)
    for j, est in enumerate(clf.estimators_):
        p = est.predict_proba(X_test)
        if p.shape[1] == 2:
            probs[:, j] = p[:, 1]
        else:
            prev = float(train_df[TARGET_COLS[j]].mean())
            probs[:, j] = prev

    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)

    print(
        f"Fallback image-model: missing train images={miss_tr}, missing test images={miss_te}"
    )
    return probs




## === cell 5
if len(submissions_all) >= 2:
    submission_avg = ensemble(
        submissions_all, [0, 1], [0.23, 0.77], sample_submission_path=SAMPLE_SUB_PATH
    )
    make_submission_file(submission_avg, SAMPLE_SUB_PATH)
elif len(submissions_all) == 1:
    submission_avg = ensemble(
        submissions_all, [0], [1.0], sample_submission_path=SAMPLE_SUB_PATH
    )
    make_submission_file(submission_avg, SAMPLE_SUB_PATH)
else:
    if (
        os.path.exists(TRAIN_PATH)
        and os.path.exists(TEST_PATH)
        and IMAGES_DIR is not None
        and os.path.isdir(IMAGES_DIR)
    ):
        submission_avg = fallback_train_image_model_and_predict(
            TRAIN_PATH, TEST_PATH, SAMPLE_SUB_PATH, IMAGES_DIR
        )
        make_submission_file(submission_avg, SAMPLE_SUB_PATH)
    else:
        sample_df = pd.read_csv(SAMPLE_SUB_PATH)
        if os.path.exists(TRAIN_PATH):
            train_df = pd.read_csv(TRAIN_PATH)
            priors = train_df[TARGET_COLS].mean().astype("float64")
            priors = priors.clip(1e-6, 1.0 - 1e-6)
            for c in TARGET_COLS:
                sample_df.loc[:, c] = float(priors[c])
            print(
                "No external submissions and image-model unavailable; wrote prior-based submission using train label prevalence:",
                priors.to_dict(),
            )
        else:
            for c in TARGET_COLS:
                sample_df.loc[:, c] = 0.25
            print(
                "No external submissions found and train.csv missing; wrote uniform baseline (0.25) submission."
            )

        sample_df.to_csv("submission.csv", index=False)
        print("Wrote submission.csv with shape:", sample_df.shape)
        print(sample_df.head())
