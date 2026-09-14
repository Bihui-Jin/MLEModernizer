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

0.9629959556175588

# 6. Current score

0.62276

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The error happens because the code assumes an external `/kaggle/input/submissions/submissions/` dataset exists, but it doesn’t in your environment, so `submissions_all` is empty and indexing fails. To make the notebook run end-to-end and still preserve the “ensemble submissions” core logic, I add a safe fallback: if no external submissions are found, create two simple baseline submissions (using the sample_submission format) and then ensemble them with your same weighting code. I also harden `ensemble()` with basic input validation (matching lengths, index bounds) and make sure the produced `submission.csv` matches the required columns and row order from `test.csv`. This yield a valid `.csv` submission file without changing the fundamental approach (weighted averaging of submission files).'
- What this solution (achieved 0.47187) has done: 'Your current 0.5 score is caused by using fallback “flat” probabilities (uniform/prior) instead of any image-based signal; to move toward the 0.963 target while preserving your ensemble-CSV core logic, I keep the ensembling code but replace the fallback generation with a simple, legitimate model trained from `train.csv` using only installed packages (scikit-learn via `sklearn-pandas` dependency). Specifically, when no external submissions are found, we train a one-vs-rest Logistic Regression on `image_id` character n-grams to produce non-constant, per-class probabilities for the test set (and also keep a uniform fallback as the second submission), then ensemble them with your same weighted averaging. This is minimal (only changes the fallback) and should substantially increase AUC from 0.5 toward the target without changing the submission schema/paths.'
- What this solution (achieved 0.65426) has done: 'Your score is low because the fallback model uses only `image_id` text features, which has little relationship to the image labels; to move toward the 0.963 target without changing your ensemble-based core logic, I keep the CSV-ensemble approach but make the fallback “signal” come from the actual images. Specifically, when no external submissions are present, I train a lightweight image classifier using scikit-learn on simple, fast color histogram features extracted from the provided JPGs, then ensemble that submission with the existing uniform fallback using your same weighted averaging. I also harden alignment so every fallback submission is ordered exactly like `test.csv` before ensembling, preventing silent row mismatches that can destroy AUC. These are minimal, metric-relevant changes that should materially improve ROC AUC while keeping runtime under the 600s limit.'
- What this solution (achieved 0.63671) has done: 'To move your score up toward the 0.963 target without changing the overall “ensemble CSVs” core logic, I strengthen only the fallback image-based submission: keep the same simple color-histogram features and LogisticRegression+OneVsRest approach, but add a small, deterministic augmentation (original + horizontal flip) and average probabilities, which typically improves AUC noticeably for leaf images. I also add class-balancing in LogisticRegression to reduce bias toward frequent classes, which often helps mean column-wise ROC AUC with minimal risk. Finally, I slightly increase the ensemble weight on the improved image-model fallback (still ensembling with the uniform fallback) to better leverage the stronger signal while preserving your weighted-averaging logic and output format.'
- What this solution (achieved 0.62276) has done: 'Your current gap to the 0.963 target is large, and the biggest issue is that the image fallback is too weak (simple RGB histograms) and the ensemble is dominated by it anyway. To improve ROC AUC while preserving your exact “fallback submissions + weighted CSV ensembling” core logic, I only strengthen the image-feature extractor (still fast, still classic ML) by adding edge/texture information via HOG features computed from resized grayscale images. I keep the same OneVsRest + LogisticRegression training approach and the same deterministic TTA (horizontal flip averaged at test time). Finally, I remove the tiny uniform-mix degradation and ensemble two image-based submissions (original+flip and original-only) to gain a small robustness bump without changing the overall semantics (still ensembling CSVs).'

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"

DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TEST_CSV_PATH = os.path.join(DATA_DIR, "test.csv")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

print("SUBMISSIONS_PATH exists:", os.path.exists(SUBMISSIONS_PATH))
print("SAMPLE_SUB_PATH exists:", os.path.exists(SAMPLE_SUB_PATH))
print("TEST_CSV_PATH exists:", os.path.exists(TEST_CSV_PATH))
print("TRAIN_CSV_PATH exists:", os.path.exists(TRAIN_CSV_PATH))
print("IMAGES_DIR exists:", os.path.exists(IMAGES_DIR))



## === cell 2
submissions_all = []
if os.path.exists(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))

submissions_all = submissions_all[::-1]
print("Found submission files:", len(submissions_all))
print(submissions_all[:10])




## === cell 3
def _make_fallback_submissions(
    sample_sub_path: str,
    train_csv_path: str,
    test_csv_path: str,
    images_dir: str,
    out_dir: str = ".",
):
    """
    Change is directly for score: upgrade the image fallback features from pure color histograms
    to (color hist + simple moments + HOG texture/edge features), which is strongly correlated
    with leaf disease patterns and typically improves mean ROC AUC.
    Core logic preserved: still trains OneVsRest(LogisticRegression) on fixed features and
    creates CSV fallback submissions, then ensembles them downstream.
    """
    req_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]

    sample = pd.read_csv(sample_sub_path)
    missing = [c for c in req_cols if c not in sample.columns]
    if missing:
        raise ValueError(f"sample_submission is missing columns: {missing}")

    test_df = pd.read_csv(test_csv_path)
    train_df = pd.read_csv(train_csv_path)

    test_ids = test_df["image_id"].astype(str).tolist()

    sub_uniform = pd.DataFrame({"image_id": test_ids})
    for c in req_cols[1:]:
        sub_uniform[c] = 0.25
    path_uniform = os.path.join(out_dir, "fallback_uniform.csv")
    sub_uniform.to_csv(path_uniform, index=False)

    from PIL import Image
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    try:
        from skimage.feature import hog
    except Exception as e:
        raise ImportError(
            "scikit-image is required for HOG features but is not available in this environment."
        ) from e

    def _img_path_from_id(image_id: str) -> str:
        if image_id.lower().endswith(".jpg"):
            return os.path.join(images_dir, image_id)
        return os.path.join(images_dir, f"{image_id}.jpg")

    def _extract_features_from_pil(
        im,
        rgb_size=(128, 128),
        gray_size=(128, 128),
        bins=16,
        hog_ppc=(16, 16),
        hog_cpb=(2, 2),
    ) -> np.ndarray:
        im_rgb = im.convert("RGB").resize(rgb_size)
        arr = np.asarray(im_rgb, dtype=np.uint8)

        feats = []
        for ch in range(3):
            h, _ = np.histogram(arr[:, :, ch], bins=bins, range=(0, 256))
            h = h.astype(np.float32)
            h /= h.sum() + 1e-8
            feats.append(h)

        flat = arr.reshape(-1, 3).astype(np.float32) / 255.0
        mean = flat.mean(axis=0).astype(np.float32)
        std = flat.std(axis=0).astype(np.float32)
        feats.append(mean)
        feats.append(std)

        im_g = im.convert("L").resize(gray_size)
        g = np.asarray(im_g, dtype=np.float32) / 255.0

        hog_feat = hog(
            g,
            orientations=9,
            pixels_per_cell=hog_ppc,
            cells_per_block=hog_cpb,
            block_norm="L2-Hys",
            feature_vector=True,
        ).astype(np.float32)

        feats.append(hog_feat)
        return np.concatenate(feats, axis=0)

    def _extract_features(image_id: str, flip: bool = False) -> np.ndarray:
        p = _img_path_from_id(image_id)
        try:
            im = Image.open(p)
            if flip:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            return _extract_features_from_pil(im)
        except Exception:
            return None

    _zero_feat = None
    for _id in train_df["image_id"].astype(str).tolist()[:5] + test_ids[:5]:
        tmp = _extract_features(_id, flip=False)
        if tmp is not None:
            _zero_feat = np.zeros_like(tmp, dtype=np.float32)
            break
    if _zero_feat is None:
        raise RuntimeError("Could not infer feature vector size from available images.")

    def _safe_extract(image_id: str, flip: bool = False) -> np.ndarray:
        v = _extract_features(image_id, flip=flip)
        return v if v is not None else _zero_feat

    train_ids = train_df["image_id"].astype(str).tolist()
    y_train = train_df[req_cols[1:]].astype(int).values

    X_train = np.vstack([_safe_extract(i, flip=False) for i in train_ids])

    X_test = np.vstack([_safe_extract(i, flip=False) for i in test_ids])
    X_test_flip = np.vstack([_safe_extract(i, flip=True) for i in test_ids])

    base = LogisticRegression(
        solver="lbfgs",
        C=3.0,
        max_iter=600,
        n_jobs=1,
        random_state=42,
        class_weight="balanced",
    )
    clf = OneVsRestClassifier(
        Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("lr", base),
            ]
        )
    )
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)
    proba_flip = clf.predict_proba(X_test_flip)
    proba_tta = 0.5 * (proba + proba_flip)

    sub_tta = pd.DataFrame({"image_id": test_ids})
    for j, c in enumerate(req_cols[1:]):
        sub_tta[c] = proba_tta[:, j]
    path_tta = os.path.join(out_dir, "fallback_img_hog_lr_tta.csv")
    sub_tta.to_csv(path_tta, index=False)

    sub_orig = pd.DataFrame({"image_id": test_ids})
    for j, c in enumerate(req_cols[1:]):
        sub_orig[c] = proba[:, j]
    path_orig = os.path.join(out_dir, "fallback_img_hog_lr_orig.csv")
    sub_orig.to_csv(path_orig, index=False)

    return [path_tta, path_orig, path_uniform]


if len(submissions_all) == 0:
    print(
        "No external submissions found; creating fallback submissions (HOG+color LR, with/without TTA + uniform)."
    )
    submissions_all = _make_fallback_submissions(
        SAMPLE_SUB_PATH, TRAIN_CSV_PATH, TEST_CSV_PATH, IMAGES_DIR, out_dir="."
    )
    print("Fallback submissions:", submissions_all)




## === cell 4
def ensemble(submissions_all, sub_idx, weights=[]):
    if len(submissions_all) == 0:
        raise ValueError("submissions_all is empty; cannot ensemble.")
    if len(sub_idx) == 0:
        raise ValueError("sub_idx is empty; nothing to ensemble.")
    if len(weights) == 0:
        weights = [1.0] * len(sub_idx)
    if len(weights) != len(sub_idx):
        raise ValueError(
            f"weights length ({len(weights)}) must match sub_idx length ({len(sub_idx)})"
        )

    test_df = pd.read_csv(TEST_CSV_PATH)
    cols = ["healthy", "multiple_diseases", "rust", "scab"]

    submission_with_weight = []
    for i in range(len(sub_idx)):
        idx = sub_idx[i]
        if not (0 <= idx < len(submissions_all)):
            raise IndexError(
                f"sub_idx[{i}]={idx} out of range for submissions_all of length {len(submissions_all)}"
            )

        path = submissions_all[idx]
        w = float(weights[i])
        print(f"I'm taking submission {path} with weight {w}")

        submission = pd.read_csv(path)
        missing = [c for c in (["image_id"] + cols) if c not in submission.columns]
        if missing:
            raise ValueError(f"Submission file {path} missing columns: {missing}")

        submission["image_id"] = submission["image_id"].astype(str)
        test_df["image_id"] = test_df["image_id"].astype(str)

        submission = test_df[["image_id"]].merge(submission, on="image_id", how="left")
        if submission[cols].isna().any().any():
            submission[cols] = submission[cols].fillna(0.25)

        mat = submission.loc[:, cols].values
        submission_with_weight.append(mat * w)

    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 5
def make_submission_file(submission_avg, submissions_all):
    test_df = pd.read_csv(TEST_CSV_PATH)
    cols = ["healthy", "multiple_diseases", "rust", "scab"]

    submission_df = pd.read_csv(SAMPLE_SUB_PATH)

    submission_df["image_id"] = submission_df["image_id"].astype(str)
    test_df["image_id"] = test_df["image_id"].astype(str)

    submission_df = submission_df.merge(
        test_df[["image_id"]], on="image_id", how="right"
    )
    if submission_df.shape[0] != test_df.shape[0]:
        raise ValueError(
            "Submission row count does not match test row count after alignment."
        )

    if (
        submission_avg.shape[0] != submission_df.shape[0]
        or submission_avg.shape[1] != 4
    ):
        raise ValueError(
            f"submission_avg has shape {submission_avg.shape}, expected ({submission_df.shape[0]}, 4)"
        )

    submission_df[cols] = submission_avg
    submission_df[cols] = submission_df[cols].clip(0.0, 1.0)

    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission_df.shape)
    print(submission_df.head())




## === cell 6
submission_avg = ensemble(submissions_all, [0, 1], [0.8, 0.2])
make_submission_file(submission_avg, submissions_all)
