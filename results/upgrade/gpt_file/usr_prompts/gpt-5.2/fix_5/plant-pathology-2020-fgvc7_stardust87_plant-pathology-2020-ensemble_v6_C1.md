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

0.9628698141306518

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The crash happens because the notebook expects an external `/kaggle/input/submissions/...` dataset that isn’t present, so `submissions_all` is empty and indexing `[0,1]` fails. I keep your ensembling logic intact, but add a safe fallback: if no external submissions are found, we generate a valid baseline submission from `sample_submission.csv` (constant class priors from `train.csv`), which runs end-to-end and yields a non-trivial ROC-AUC score. I also make the ensemble function validate indices/weights and normalize by total weight (so it behaves like a true weighted average and avoids scale bugs). The output be written as `submission.csv` with the exact required columns.'
- What this solution (achieved 0.55075) has done: 'Your current 0.5 score is consistent with a constant-prior fallback (or uniform 0.25), so to move toward the 0.9629 target we need a real model-based prediction while keeping the “core logic” (submission creation + optional ensembling) intact. I add a minimal, fast classical image pipeline (HOG features + LogisticRegression One-vs-Rest) trained on the provided train images, and use its probabilities for the submission when no external submissions are present. This keeps evaluation semantics (proper per-class probabilities), preserves your ensemble path unchanged, and should substantially improve AUC compared to constant priors without introducing heavy DL dependencies. I also make the fallback selection smarter: prefer the trained model if it can run; otherwise keep your prior-based fallback.'
- What this solution (achieved 0.5) has done: 'The timeout is dominated by image feature extraction (Python loop over ~1800 images with `imread` + `resize` + 3×`hog`) plus an inefficient `np.vstack([x.reshape(1,-1) for x in X])` pattern that adds extra Python overhead and copies. I keep the exact same HOG+LogReg core logic, but make featurization faster by (1) preallocating the feature matrix once, (2) using `skimage.transform.resize(..., output_shape=..., channel_axis=-1)` to resize all 3 channels at once, (3) using faster dtype handling without changing values, and (4) parallelizing the per-image featurization with joblib threads (safe here because most time is in C/NumPy code and it preserves determinism). I also speed up CSV reads in the ensemble path by loading only required columns and accumulating in a single array to avoid storing all weighted submissions. These changes are equivalent in semantics and avoid approximations, but cut constant factors enough to fit the 600s budget.'

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"

DATA_ROOT = "/kaggle/input/plant-pathology-2020-fgvc7"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")

IMAGES_DIR = os.path.join(DATA_ROOT, "images")

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]



## === cell 2
submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found external submissions:", submissions_all)




## === cell 3
def ensemble(submissions_all, sub_idx, weights=[]):
    """
    Weighted average ensemble of existing submission files.
    Bugfix: validate indices/weights and normalize by total weight so values remain probabilities.
    """
    if len(submissions_all) == 0:
        raise ValueError("No submissions found to ensemble.")
    if len(sub_idx) == 0:
        raise ValueError("sub_idx is empty.")
    if len(weights) != len(sub_idx):
        raise ValueError(
            f"weights length ({len(weights)}) must match sub_idx length ({len(sub_idx)})."
        )
    for j in sub_idx:
        if j < 0 or j >= len(submissions_all):
            raise IndexError(
                f"sub_idx contains {j}, but submissions_all has length {len(submissions_all)}."
            )
    total_w = float(sum(weights))
    if total_w <= 0:
        raise ValueError("Sum of weights must be > 0.")

    submission_sum = None
    for i in range(len(sub_idx)):
        path = submissions_all[sub_idx[i]]
        w = float(weights[i])
        print(f"I'm taking submission {path} with weight {w}")
        arr = pd.read_csv(path, usecols=TARGET_COLS)[TARGET_COLS].to_numpy(
            dtype=float, copy=False
        )
        if submission_sum is None:
            submission_sum = arr * w
        else:
            submission_sum += arr * w

    submission_avg = submission_sum / total_w
    return submission_avg




## === cell 4
def make_submission_file_from_template(
    submission_avg, template_path, out_path="submission.csv"
):
    """
    Write submission using a provided template (sample_submission or an existing submission file).
    """
    submission_df = pd.read_csv(template_path)
    if "image_id" not in submission_df.columns:
        raise ValueError("Template must include image_id.")
    for c in TARGET_COLS:
        if c not in submission_df.columns:
            raise ValueError(f"Template missing required column: {c}")

    submission_df.loc[:, TARGET_COLS] = submission_avg
    submission_df.to_csv(out_path, index=False)
    print(
        f"Wrote {out_path} with shape {submission_df.shape} and columns {list(submission_df.columns)}"
    )




## === cell 5
def make_fallback_submission(out_path="submission.csv"):
    """
    Fallback (score-improving vs all-0.25): use class priors from train.csv as constant predictions.
    This preserves evaluation semantics (valid probabilities) and ensures end-to-end execution
    when no external submission files are available.
    """
    train = pd.read_csv(TRAIN_CSV, usecols=TARGET_COLS)
    sample = pd.read_csv(SAMPLE_SUB_CSV)

    priors = train[TARGET_COLS].mean().values.astype(float)

    sub = sample[["image_id"] + TARGET_COLS].copy()
    sub.loc[:, TARGET_COLS] = priors  # broadcasts row-wise
    sub.to_csv(out_path, index=False)
    print("No external submissions found. Created fallback prior-based submission.")
    print("Priors used:", dict(zip(TARGET_COLS, priors)))
    print(f"Wrote {out_path} with shape {sub.shape}")




## === cell 6
def make_model_submission(out_path="submission.csv"):
    import numpy as np

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    from skimage.io import imread
    from skimage.transform import resize
    from skimage.feature import hog

    try:
        from joblib import Parallel, delayed, cpu_count

        _HAS_JOBLIB = True
    except Exception:
        _HAS_JOBLIB = False

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    sample = pd.read_csv(SAMPLE_SUB_CSV)

    test_ids = test_df["image_id"].tolist()

    img_size = (160, 160)
    hog_params = dict(
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )

    def _img_path(image_id):
        return os.path.join(IMAGES_DIR, f"{image_id}.jpg")

    def _zero_feats():
        dummy = np.zeros((img_size[0], img_size[1]), dtype=np.float32)
        l = hog(dummy, **hog_params).shape[0]
        return np.zeros(l * 3, dtype=np.float32)

    zero_vec = _zero_feats()
    feat_dim = int(zero_vec.shape[0])

    def _featurize_one(image_id):
        path = _img_path(image_id)
        if not os.path.exists(path):
            return zero_vec.copy(), True

        img = imread(path)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        img = resize(
            img,
            output_shape=(img_size[0], img_size[1], 3),
            anti_aliasing=True,
            preserve_range=False,
            channel_axis=-1,
        ).astype(np.float32, copy=False)

        f0 = hog(img[:, :, 0], **hog_params).astype(np.float32, copy=False)
        f1 = hog(img[:, :, 1], **hog_params).astype(np.float32, copy=False)
        f2 = hog(img[:, :, 2], **hog_params).astype(np.float32, copy=False)
        feats = np.concatenate((f0, f1, f2), axis=0).astype(np.float32, copy=False)
        return feats, False

    def _featurize_ids(image_ids):
        n = len(image_ids)
        X = np.empty((n, feat_dim), dtype=np.float32)
        missing = 0

        if _HAS_JOBLIB and n > 16:
            n_jobs = min(8, cpu_count() or 1)
            results = Parallel(n_jobs=n_jobs, prefer="threads", batch_size=16)(
                delayed(_featurize_one)(iid) for iid in image_ids
            )
            for i, (feats, is_missing) in enumerate(results):
                X[i, :] = feats
                missing += int(is_missing)
        else:
            for i, image_id in enumerate(image_ids):
                feats, is_missing = _featurize_one(image_id)
                X[i, :] = feats
                missing += int(is_missing)

        if missing:
            print(f"Warning: {missing} images missing; used zero features for them.")
        return X

    X_train = _featurize_ids(train_df["image_id"].tolist())
    y_train = train_df[TARGET_COLS].values.astype(int, copy=False)
    X_test = _featurize_ids(test_ids)

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "ovr",
                OneVsRestClassifier(
                    LogisticRegression(
                        solver="lbfgs",
                        C=4.0,
                        max_iter=1000,
                        random_state=42,
                    )
                ),
            ),
        ]
    )

    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test).astype(float, copy=False)

    proba = np.clip(proba, 1e-6, 1 - 1e-6)

    sub = pd.DataFrame({"image_id": test_ids})
    sub[TARGET_COLS] = proba
    sub = sub[["image_id"] + TARGET_COLS]
    sub.to_csv(out_path, index=False)
    print(f"Created model-based submission at {out_path} with shape {sub.shape}")




## === cell 7
if len(submissions_all) >= 2:
    submission_avg = ensemble(submissions_all, [0, 1], [0.8, 0.2])
    make_submission_file_from_template(
        submission_avg, submissions_all[0], out_path="submission.csv"
    )
elif len(submissions_all) == 1:
    single = pd.read_csv(submissions_all[0], usecols=(["image_id"] + TARGET_COLS))
    missing = [c for c in (["image_id"] + TARGET_COLS) if c not in single.columns]
    if missing:
        raise ValueError(f"Single submission missing columns: {missing}")
    single[["image_id"] + TARGET_COLS].to_csv("submission.csv", index=False)
    print("Only one external submission found; copied it to submission.csv")
else:
    try:
        make_model_submission(out_path="submission.csv")
    except Exception as e:
        print("Model-based submission failed; falling back to prior-based submission.")
        print("Error was:", repr(e))
        make_fallback_submission(out_path="submission.csv")
