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

0.9700013841179632

# 6. Current score

0.51699

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The crash happens because your notebook expects an external folder of prior submission files (`/kaggle/input/submissions/submissions/`) that doesn’t exist in this environment, so `submissions_all` is empty and indexing `[0,1]` fails. I keep your ensembling core logic, but add a safe fallback that generates two simple baseline “submissions” from the provided `sample_submission.csv` so the pipeline always runs end-to-end. I also add small validations (weights length, index bounds, row alignment) and ensure the output `submission.csv` has the required columns and correct row order for `test.csv`. This is primarily a correctness fix (your current score is “Not yielded”); the baseline is only to produce a valid submission file.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score comes from producing near-constant class probabilities (fallback submissions), which yields weak ROC AUC. To move toward the 0.97 target with minimal logic change, I keep your “ensemble CSVs” approach but generate the fallback submissions from a simple, legitimate signal: the class priors from `train.csv`, which is a standard baseline and usually scores substantially better than uniform guesses. I also ensure column order and row alignment strictly follow `test.csv` and keep your weighting/ensembling intact (still ensembling two CSVs). This should improve score while preserving your pipeline semantics and producing a valid `submission.csv`.'
- What this solution (achieved 0.45767) has done: 'Your current 0.5 score is consistent with producing almost-constant probabilities (class priors and lightly smoothed priors), which yields weak mean ROC AUC. To move toward the 0.97 target while keeping your “ensemble CSVs” core logic intact, I keep the ensembling pipeline but make the fallback submissions use a legitimate per-image signal: simple image brightness statistics from the provided JPGs via PIL (no model/architecture changes, still just producing CSVs to ensemble). This create two complementary, non-constant prediction files (different calibrations) and then ensemble them with your existing weights, which should substantially improve ROC AUC versus constants. I also add strict alignment to `test.csv` order inside the fallback creation so the generated CSVs can’t silently misorder rows.'
- What this solution (achieved 0.56502) has done: 'Your current score is far below the 0.97 target, and the main limiter is that the fallback “submissions” are based on crude brightness/contrast heuristics that don’t capture disease cues. With minimal core-logic change (still: generate two CSVs → ensemble with your existing weights), I make the fallback predictions come from a lightweight, legitimate image feature pipeline using downsampled RGB pixels + a multinomial logistic regression trained on `train.csv`. This keeps the same ensembling semantics but injects a much stronger per-image signal while staying well within the 600s budget. I also add strict file existence checks and ensure both train/test image ordering matches `test.csv`, so the submission rows align correctly.'
- What this solution (achieved 0.52933) has done: 'The timeout is dominated by the fallback path: per-image PIL loading/resizing in Python loops (done twice) plus two `CalibratedClassifierCV(cv=3)` fits, which each refit the full pipeline multiple times. To keep the exact same modeling logic (same features, same models, same calibration), the key speedups are (1) cache the computed RGB features to disk so repeated runs don’t redo image decoding, (2) make featurization faster with a single reusable worker and vectorized memory writes, (3) avoid repeated CSV reads and repeated index alignment work, and (4) reduce Python overhead in ensembling by streaming the weighted sum instead of building a list. These changes are provably equivalent in outputs (up to negligible floating-point differences) and only remove redundant work; the training/calibration procedure remains unchanged.'
- What this solution (achieved 0.51699) has done: 'The timeout is dominated by training eight calibrated SVMs (2 pipelines × 4 targets) with `CalibratedClassifierCV(cv=3)`, which refits the base estimator multiple times per target and is expensive at 48×48×3=6912 features. To preserve identical semantics, we keep the same features, SVMs, and sigmoid calibration, but enable safe parallelism inside `CalibratedClassifierCV` via `n_jobs=-1` so the CV fits run concurrently. We also cut avoidable overhead by (1) caching features with memory mapping to reduce RAM pressure and load time, (2) using `float32` consistently for `X_*` and avoiding repeated Python work in `featurize`, and (3) reading only needed columns when ensembling and building the final submission. These changes are performance-only and keep the model family, calibration method, and outputs equivalent up to negligible floating-point differences.'

# 9. Code solution

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
    Core logic unchanged:
    - Same downsampled RGB feature extraction
    - Same model family (LinearSVC + CalibratedClassifierCV(sigmoid, cv=3))
    - Same 2-model ensemble and 4 independent one-vs-rest targets

    Performance fixes (correctness-preserving):
    - Enable parallel CV fitting inside CalibratedClassifierCV via n_jobs=-1.
      This preserves identical calibration semantics while reducing wall time.
    - Use np.load(..., mmap_mode="r") for cached feature arrays to reduce load time/RAM.
    - Avoid repeated Python attribute lookups and keep float32 features; no change to values.
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

    train_ids = train["image_id"].tolist()
    Y = train[target_cols].to_numpy(dtype=np.int64, copy=True)

    def _cache_paths(size: int):
        tag = f"pp2020_rgb{size}_v1"
        xtr = os.path.abspath(f"{tag}_X_train.npy")
        xte = os.path.abspath(f"{tag}_X_test.npy")
        return xtr, xte

    def featurize(image_ids, size=48):
        X = np.zeros((len(image_ids), size * size * 3), dtype=np.float32)
        missing_count = 0

        join = os.path.join
        exists = os.path.exists
        img_open = Image.open
        bilinear = Image.BILINEAR
        asarray = np.asarray
        inv255 = np.float32(1.0 / 255.0)

        for i, img_id in enumerate(image_ids):
            img_path = join(images_dir, f"{img_id}.jpg")
            if not exists(img_path):
                missing_count += 1
                continue
            with img_open(img_path) as im:
                im = im.convert("RGB").resize((size, size), resample=bilinear)
                arr = asarray(im, dtype=np.float32)
            X[i, :] = arr.reshape(-1) * inv255

        if missing_count:
            print(
                f"Warning: {missing_count} images were missing under {images_dir}. They were featurized as zeros."
            )
        return X

    cache_train, cache_test = _cache_paths(size=48)
    if os.path.exists(cache_train) and os.path.exists(cache_test):
        X_train = np.load(cache_train, mmap_mode="r")
        X_test = np.load(cache_test, mmap_mode="r")
    else:
        X_train = featurize(train_ids, size=48)
        X_test = featurize(test_ids, size=48)
        np.save(cache_train, X_train)
        np.save(cache_test, X_test)

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

    P1 = np.zeros((X_test.shape[0], 4), dtype=np.float64)
    P2 = np.zeros((X_test.shape[0], 4), dtype=np.float64)

    for j, col in enumerate(target_cols):
        yj = Y[:, j]

        if yj.min() == yj.max():
            prior = float(yj.mean())
            P1[:, j] = prior
            P2[:, j] = prior
            continue

        clf1 = CalibratedClassifierCV(base_svm1, method="sigmoid", cv=3, n_jobs=-1)
        clf2 = CalibratedClassifierCV(base_svm2, method="sigmoid", cv=3, n_jobs=-1)

        clf1.fit(X_train, yj)
        clf2.fit(X_train, yj)

        proba1 = clf1.predict_proba(X_test).astype(np.float64, copy=False)
        proba2 = clf2.predict_proba(X_test).astype(np.float64, copy=False)

        cls1 = list(clf1.classes_)
        cls2 = list(clf2.classes_)

        if 1 in cls1:
            P1[:, j] = proba1[:, cls1.index(1)]
        else:
            P1[:, j] = float(yj.mean())

        if 1 in cls2:
            P2[:, j] = proba2[:, cls2.index(1)]
        else:
            P2[:, j] = float(yj.mean())

    np.clip(P1, 1e-6, 1 - 1e-6, out=P1)
    np.clip(P2, 1e-6, 1 - 1e-6, out=P2)

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

    cols = ["healthy", "multiple_diseases", "rust", "scab"]
    usecols = ["image_id"] + cols
    submission_sum = None
    base_len = None

    for i in range(len(sub_idx)):
        idx = sub_idx[i]
        if idx < 0 or idx >= len(submissions_all):
            raise IndexError(
                f"sub_idx[{i}]={idx} out of range for submissions_all (len={len(submissions_all)})."
            )

        w = float(weights[i])
        print(f"I'm taking submission {submissions_all[idx]} with weight {w}")
        submission = pd.read_csv(submissions_all[idx], usecols=usecols)

        missing_cols = [c for c in cols if c not in submission.columns]
        if missing_cols:
            raise ValueError(
                f"Submission {submissions_all[idx]} missing columns: {missing_cols}"
            )

        arr = submission.loc[:, cols].to_numpy(dtype=np.float64, copy=False)

        if base_len is None:
            base_len = arr.shape[0]
            submission_sum = np.zeros_like(arr, dtype=np.float64)
        elif arr.shape[0] != base_len:
            raise ValueError(
                f"Row count mismatch across submissions: expected {base_len}, got {arr.shape[0]}"
            )

        submission_sum += arr * w

    return submission_sum




## === cell 5
def make_submission_file(submission_avg, submissions_all):
    submission_df = pd.read_csv(
        SAMPLE_SUB_PATH,
        usecols=["image_id", "healthy", "multiple_diseases", "rust", "scab"],
    )
    test_df = pd.read_csv(TEST_CSV_PATH, usecols=["image_id"])

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


submission_avg = ensemble(submissions_all, [0, 1], [0.24, 0.76])
make_submission_file(submission_avg, submissions_all)
