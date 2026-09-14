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

0.53037

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The crash happens because the notebook expects an external `/kaggle/input/submissions/...` dataset that isn’t present, so `submissions_all` is empty and indexing `[0,1]` fails. I keep your ensembling logic intact, but add a safe fallback: if no external submissions are found, we generate a valid baseline submission from `sample_submission.csv` (constant class priors from `train.csv`), which runs end-to-end and yields a non-trivial ROC-AUC score. I also make the ensemble function validate indices/weights and normalize by total weight (so it behaves like a true weighted average and avoids scale bugs). The output be written as `submission.csv` with the exact required columns.'
- What this solution (achieved 0.55075) has done: 'Your current 0.5 score is consistent with a constant-prior fallback (or uniform 0.25), so to move toward the 0.9629 target we need a real model-based prediction while keeping the “core logic” (submission creation + optional ensembling) intact. I add a minimal, fast classical image pipeline (HOG features + LogisticRegression One-vs-Rest) trained on the provided train images, and use its probabilities for the submission when no external submissions are present. This keeps evaluation semantics (proper per-class probabilities), preserves your ensemble path unchanged, and should substantially improve AUC compared to constant priors without introducing heavy DL dependencies. I also make the fallback selection smarter: prefer the trained model if it can run; otherwise keep your prior-based fallback.'
- What this solution (achieved 0.5) has done: 'The timeout is dominated by image feature extraction (Python loop over ~1800 images with `imread` + `resize` + 3×`hog`) plus an inefficient `np.vstack([x.reshape(1,-1) for x in X])` pattern that adds extra Python overhead and copies. I keep the exact same HOG+LogReg core logic, but make featurization faster by (1) preallocating the feature matrix once, (2) using `skimage.transform.resize(..., output_shape=..., channel_axis=-1)` to resize all 3 channels at once, (3) using faster dtype handling without changing values, and (4) parallelizing the per-image featurization with joblib threads (safe here because most time is in C/NumPy code and it preserves determinism). I also speed up CSV reads in the ensemble path by loading only required columns and accumulating in a single array to avoid storing all weighted submissions. These changes are equivalent in semantics and avoid approximations, but cut constant factors enough to fit the 600s budget.'
- What this solution (achieved 0.5) has done: 'Your 0.5 score suggests the pipeline is falling back to weak/near-constant predictions, so we improve the model-based fallback while keeping the same “HOG features + OneVsRest LogisticRegression” core approach intact. The biggest safe win is to use the full HOG descriptor on the (resized) RGB image directly (not per-channel and concatenated), which is both faster and typically more predictive; we keep the same classifier family, training loop, and probability submission semantics. We also ensure the output probabilities are aligned to `test.csv` ordering by merging with `sample_submission.csv` (so even if file order changes, submission rows match). These are minimal, directly score-relevant changes that should move you closer to the 0.9629 target without changing the overall solution structure.'
- What this solution (achieved 0.51614) has done: 'Your current 0.5 suggests the model fallback is failing and you’re ending up with weak/constant predictions, so the smallest score-moving fix is to make the model path robust and actually run end-to-end. I keep your HOG + OneVsRest LogisticRegression approach and submission/ensemble logic intact, but (1) correct the dataset paths to match your provided filesystem, (2) avoid `skimage`/`joblib` dependency failures by using only installed packages (PIL for image loading, sklearn only), and (3) ensure the submission columns/order exactly match `sample_submission.csv` and test ordering. The fallback-to-priors behavior remains as-is if images can’t be loaded, but now the model path should execute reliably and move the ROC-AUC score toward your target.'
- What this solution (achieved 0.53037) has done: 'I keep your exact overall flow (external submission ensemble if present, otherwise model fallback, otherwise priors fallback) but make the model fallback substantially more predictive with minimal changes: enrich the feature vector by concatenating color statistics to your existing HOG descriptor and add mild class-balancing in the same LogisticRegression to reduce bias on rare labels. I also ensure the data root resolves robustly to the provided filesystem so the model path actually trains on real images instead of silently hitting missing files (which effectively produces near-constant predictions and the ~0.51 score). These changes preserve the same model family (HOG-like features + OneVsRest LogisticRegression), same training approach, and same submission semantics, but should move ROC-AUC meaningfully toward your 0.9629 target. The script still write `submission.csv` with the required columns and test ordering.'

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"

DATA_ROOT_CANDIDATES = [
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/data",
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/input",
]
DATA_ROOT = None
for cand in DATA_ROOT_CANDIDATES:
    if os.path.exists(os.path.join(cand, "train.csv")) and os.path.exists(
        os.path.join(cand, "test.csv")
    ):
        DATA_ROOT = cand
        break
if DATA_ROOT is None:
    DATA_ROOT = "/kaggle/data/plant-pathology-2020-fgvc7"

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")

IMAGES_DIR = os.path.join(DATA_ROOT, "images")

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

print("DATA_ROOT:", DATA_ROOT)
print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
print("TEST_CSV exists:", os.path.exists(TEST_CSV))
print("SAMPLE_SUB_CSV exists:", os.path.exists(SAMPLE_SUB_CSV))
print("IMAGES_DIR exists:", os.path.isdir(IMAGES_DIR))



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
    Fallback: use class priors from train.csv as constant predictions.
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
    from PIL import Image

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    sample = pd.read_csv(SAMPLE_SUB_CSV)

    test_ids = test_df["image_id"].tolist()

    img_size = (160, 160)  # (W,H) for PIL resize

    def _hog_gray(gray, orientations=9, pixels_per_cell=16):
        gray = gray.astype(np.float32, copy=False)
        gx = np.zeros_like(gray, dtype=np.float32)
        gy = np.zeros_like(gray, dtype=np.float32)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]

        mag = np.sqrt(gx * gx + gy * gy)
        ang = (np.arctan2(gy, gx) * (180.0 / np.pi)) % 180.0  # [0,180)

        h, w = gray.shape
        cell = int(pixels_per_cell)
        n_cells_y = h // cell
        n_cells_x = w // cell
        if n_cells_y <= 0 or n_cells_x <= 0:
            return np.zeros((orientations,), dtype=np.float32)

        mag = mag[: n_cells_y * cell, : n_cells_x * cell]
        ang = ang[: n_cells_y * cell, : n_cells_x * cell]

        bin_width = 180.0 / orientations
        bins = np.floor(ang / bin_width).astype(np.int32)
        bins = np.clip(bins, 0, orientations - 1)

        feats = np.zeros((n_cells_y, n_cells_x, orientations), dtype=np.float32)
        for cy in range(n_cells_y):
            y0 = cy * cell
            y1 = y0 + cell
            for cx in range(n_cells_x):
                x0 = cx * cell
                x1 = x0 + cell
                b = bins[y0:y1, x0:x1].ravel()
                m = mag[y0:y1, x0:x1].ravel()
                hist = np.bincount(b, weights=m, minlength=orientations).astype(
                    np.float32, copy=False
                )
                feats[cy, cx, :] = hist

        vec = feats.reshape(-1)
        norm = np.sqrt((vec * vec).sum()) + 1e-6
        vec = vec / norm
        return vec

    def _img_path(image_id):
        return os.path.join(IMAGES_DIR, f"{image_id}.jpg")

    dummy = np.zeros((img_size[1], img_size[0]), dtype=np.float32)
    hog_dim = int(_hog_gray(dummy).shape[0])
    color_dim = 6  # mean RGB (3) + std RGB (3)
    feat_dim = hog_dim + color_dim

    def _featurize_one(image_id):
        path = _img_path(image_id)
        if not os.path.exists(path):
            return np.zeros((feat_dim,), dtype=np.float32), True

        img = Image.open(path).convert("RGB").resize(img_size, resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # H,W,3 in [0,1]

        mean_rgb = arr.reshape(-1, 3).mean(axis=0).astype(np.float32, copy=False)
        std_rgb = arr.reshape(-1, 3).std(axis=0).astype(np.float32, copy=False)

        gray = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
        hog = _hog_gray(gray, orientations=9, pixels_per_cell=16).astype(
            np.float32, copy=False
        )

        feats = np.concatenate([hog, mean_rgb, std_rgb]).astype(np.float32, copy=False)
        return feats, False

    def _featurize_ids(image_ids):
        n = len(image_ids)
        X = np.empty((n, feat_dim), dtype=np.float32)
        missing = 0
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
                        class_weight="balanced",
                    )
                ),
            ),
        ]
    )

    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test).astype(float, copy=False)
    proba = np.clip(proba, 1e-6, 1 - 1e-6)

    pred_df = pd.DataFrame({"image_id": test_ids})
    pred_df[TARGET_COLS] = proba

    sub = sample[["image_id"]].merge(pred_df, on="image_id", how="left")
    priors = train_df[TARGET_COLS].mean().values.astype(float)
    for c in TARGET_COLS:
        if sub[c].isna().any():
            sub[c] = sub[c].fillna(float(priors[TARGET_COLS.index(c)]))
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
