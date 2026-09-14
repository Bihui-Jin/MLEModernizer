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

0.69241

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The crash happens because `/kaggle/input/submissions/` doesn’t exist in this environment, so `submissions_all` is empty and indexing `[0,1,2]` fails. I keep your ensembling logic intact, but add a safe fallback: if no external submissions are found, create a valid baseline submission using `sample_submission.csv` (uniform probabilities), ensuring a `submission.csv` is always produced. I also make the path resolution robust by checking the provided competition dataset locations first, without changing any modeling/training (none exists in this script). This unblock end-to-end execution and yield a valid Kaggle submission file.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score comes from writing a uniform-probability submission (0.25 for each class) because no external CSVs are found to ensemble. To move the score toward the 0.9695 target without changing the overall approach (still “no training, just generate a submission”), I replace the uniform fallback with a minimal, legitimate heuristic: a class-prior baseline computed from `train.csv` label frequencies (and lightly smoothed), applied to every test row. This keeps the same pipeline structure (read CSVs → create constant predictions → write `submission.csv`) but yields a meaningfully better AUC than random guessing, while remaining stable and fast.'
- What this solution (achieved 0.5) has done: 'Your script currently can’t report a Kaggle score, so the first priority is to guarantee it always produces a valid `submission.csv` with correct column names and test-row alignment. To move the expected ROC AUC upward toward your 0.9695 target without changing the “no training, just generate a submission” core approach, I replace the arbitrary `image_id`-number-based variation with a safer constant baseline using smoothed class priors from `train.csv` (AUC-optimal behavior for a no-signal model) and keep the existing ensemble path unchanged when external submissions exist. I also add a strict column-name normalization step so any found external submissions using `combinations`/other aliases are mapped to `multiple_diseases`, preventing silent column mismatch failures. These are minimal, stability-focused changes that should improve over the current heuristic without introducing new modeling/training logic.'
- What this solution (achieved 0.46103) has done: 'Your current 0.5 score comes from predicting (effectively) constant probabilities for every test image, which gives near-random ranking and thus low ROC AUC. To move toward the 0.9695 target without changing the “no training / no image features” core approach, I keep the same pipeline but (1) fix a label-name mismatch (`multiple_diseases` vs `combinations`) that can silently break ensembling, and (2) replace the constant-prior fallback with an ultra-light, deterministic per-image heuristic derived only from the `image_id` string (no training loop, no model) so predictions vary across rows and can rank examples better than constant scores. I also keep strict column ordering, clipping, and test alignment so the submission is always valid. These minimal changes should improve AUC substantially versus constant priors while remaining fast and stable.'
- What this solution (achieved 0.68597) has done: 'Your current score (0.461) is far below the target (0.9695), and the main reason is that the fallback submission has essentially no real signal (it only uses `image_id` arithmetic), which won’t rank true labels well for ROC AUC. To move the score upward with minimal disruption, I keep your overall “no training loop” pipeline but replace the `image_id`-based heuristic with a tiny, legitimate model: scikit-learn logistic regression trained on simple, fast image statistics (color means/stds) extracted from the provided JPEGs. This keeps the same submission-writing and column semantics, uses only local competition files, and should substantially improve AUC while staying within time limits by downsampling images and using a small feature set. Ensembling logic remains intact: if external submissions exist, it still ensembles; otherwise it trains this lightweight image-stat model and predicts test probabilities.'
- What this solution (achieved 0.68775) has done: 'Your current score (0.68597) is far below the 0.9695 target, so we should improve ranking signal with minimal, safe changes while keeping the same “logistic regression on simple image stats” core. I keep the exact feature set and per-class LogisticRegression training loop, but fix two issues that typically hurt ROC AUC here: (1) add stratified CV out-of-fold predictions (for calibration/regularization) and average fold test predictions (still same model class, no new architecture), and (2) use class-balanced weights to better learn rare classes like `multiple_diseases`. I also make image loading faster/more consistent by using a slightly larger resize (still cheap) and add a deterministic seed; submission schema/paths stay identical and it still write `submission.csv`.'
- What this solution (achieved 0.68775) has done: 'Your current score (0.68775) is far below the target (0.96949), so we should improve ranking signal while keeping your core approach intact (simple image-stat features + per-class LogisticRegression with CV). The smallest reliable gain here is to (1) fix the image directory selection so we always use the correct `images/` folder that matches the CSVs (and don’t accidentally pick a wrong root), and (2) use CV out-of-fold (OOF) probabilities to calibrate each class via a tiny 1D logistic calibration (still LogisticRegression, no new model family), then apply that calibration to the averaged test predictions. These changes keep the same feature extraction, same per-class training loop, same metric semantics, but typically improve ROC AUC by reducing miscalibration and fold-to-fold bias. Submission format, column order, clipping, and file name remain unchanged (`submission.csv`).'
- What this solution (achieved 0.67631) has done: 'Your current gap to the target is large (0.68775 vs 0.96949, higher-is-better), so we need a real AUC lift while keeping the same core approach (simple image-stat features + per-class LogisticRegression + CV + 1D logistic calibration). The smallest high-impact change here is to use a slightly richer but still “image statistics” feature set (adds inexpensive global quantiles + excess green/ExG and a couple ratios) and to make the CV splitting more stable by avoiding folds that can become single-class for rare labels (adaptive number of splits). I’m also fixing a common hidden AUC killer: incorrect image path resolution when multiple dataset roots exist, by preferring the directory that actually contains the train/test jpgs referenced by the CSVs. Everything else (model family, per-class loop, calibration, submission schema, file name) stays the same.'
- What this solution (achieved 0.67792) has done: 'Your current score (0.67631) is far below the 0.9695 target, so we should increase AUC by improving how your existing LogisticRegression-on-image-stats pipeline generalizes, without changing the overall approach. The biggest low-risk gain is to keep the same feature family but add a few extremely cheap texture/shape statistics (edge magnitude + simple local-contrast) that help distinguish scab/rust patterns beyond color. I also switch the per-fold scaling to a proper `StandardScaler` fit on each fold (instead of global scaling that leaks validation distribution slightly), which typically improves CV-trained test predictions and thus leaderboard AUC while preserving the same CV training loop and model family. Everything else (per-class CV, 1D logistic calibration, blending with priors, submission schema/paths) stays the same.'
- What this solution (achieved 0.67951) has done: 'Your current score (0.67792) is far below the target (0.96949), so we should improve ranking signal while keeping your exact pipeline (global image-stat features → per-class LogisticRegression with StratifiedKFold → 1D logistic calibrator → blend with priors). The smallest, high-impact change is to use stronger yet still “simple statistics” features that capture lesion texture: add lightweight FFT-band energy and a small set of multi-scale gradient stats (computed on a tiny grayscale resize) while preserving your existing feature vector and training loop semantics. I also set `n_jobs=1` explicitly for determinism and slightly increase `max_iter` only where convergence can affect probability quality, without changing the model family or introducing any new training strategy. Everything else (paths, CV approach, calibration, submission schema/filename) stays the same.'
- What this solution (achieved 0.67951) has done: 'Your current score (0.67951) is far below the target (0.96949), so we need a real AUC lift while preserving your exact pipeline (simple image-stat features → per-class LogisticRegression with StratifiedKFold → 1D LR calibration → blend with priors). The highest-impact minimal fix is to correct a feature bug: you return NaNs when an image can’t be read using a hardcoded feature length (36) that doesn’t match the actual feature vector length (37), which can corrupt imputation and degrade learning. I also make image loading deterministic and more robust by using `ImageFile.LOAD_TRUNCATED_IMAGES=True` and ensuring we always compute the correct feature length constant from one place. These changes don’t alter the model family, CV loop, calibration, or semantics, but should improve stability and the learned ranking signal, nudging ROC AUC upward toward your target.'
- What this solution (achieved 0.69404) has done: 'Your current score (0.67951) is far below the 0.96949 target (higher-is-better), so we should aim for a clear AUC lift while keeping the same pipeline (simple image-stat features → per-class LogisticRegression with StratifiedKFold → 1D LR calibration → blend with priors). The smallest high-impact fix is to correct a silent class-name mismatch in your description vs actual columns (`multiple_diseases`), ensure the feature vector length is *derived from the actual feature list* (so it can’t drift), and add a tiny, safe augmentation inside the *same feature family* by including the per-channel grayscale correlation terms (still global stats, no new model/training approach). These changes keep your model family, CV loop, calibration, and submission semantics identical, but should improve stability and add a bit more discriminative signal, nudging ROC AUC upward toward the target. The submission writing, column order, clipping, and `submission.csv` output remain unchanged.'
- What this solution (achieved 0.69241) has done: 'Your current score (0.694) is far below the target (0.9695), so we should improve ROC AUC by strengthening generalization while keeping the exact same core pipeline: fixed image-stat features → per-class LogisticRegression with StratifiedKFold → 1D logistic calibration → blend with priors. The smallest high-impact fix is to remove a subtle distribution shift: you currently compute train features at size (160,160) and then downsample to ~32x32 with steps that aren’t consistent; we make the multi-scale branch explicitly resize to a fixed small size (32x32) so FFT/gradient features are comparable across images. We also add very light, deterministic regularization tuning *within the same model family* by setting `C` based on fold size (still LogisticRegression, same loop), and we ensure the HSV fallback matches the real conversion range to avoid inconsistent `v` values. These changes keep your approach intact, but typically increase stability and ranking quality, nudging AUC upward toward your target, while still finishing fast and writing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/"

DATA_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]


def find_first_existing(relpath: str, roots):
    for r in roots:
        p = os.path.join(r, relpath)
        if os.path.exists(p):
            return p
    return None


SAMPLE_SUB_PATH = find_first_existing("sample_submission.csv", DATA_CANDIDATES)
TEST_CSV_PATH = find_first_existing("test.csv", DATA_CANDIDATES)
TRAIN_CSV_PATH = find_first_existing("train.csv", DATA_CANDIDATES)


def _preferred_images_dir(train_csv_path, test_csv_path, candidates):
    def _check_dir(img_dir, ref_ids):
        if img_dir is None or (not os.path.isdir(img_dir)):
            return False
        for rid in ref_ids[:10]:
            if os.path.exists(os.path.join(img_dir, f"{rid}.jpg")):
                return True
        return False

    ref_ids = []
    for csv_path in [train_csv_path, test_csv_path]:
        if csv_path is None:
            continue
        try:
            df = pd.read_csv(csv_path)
            if "image_id" in df.columns:
                ref_ids = df["image_id"].astype(str).tolist()
                break
        except Exception:
            pass

    for csv_path in [train_csv_path, test_csv_path]:
        if csv_path is None:
            continue
        parent = os.path.dirname(csv_path)
        img_dir = os.path.join(parent, "images")
        if _check_dir(img_dir, ref_ids):
            return img_dir

    for r in candidates:
        p = os.path.join(r, "images")
        if _check_dir(p, ref_ids):
            return p

    for csv_path in [train_csv_path, test_csv_path]:
        if csv_path is None:
            continue
        parent = os.path.dirname(csv_path)
        img_dir = os.path.join(parent, "images")
        if os.path.isdir(img_dir):
            return img_dir
    for r in candidates:
        p = os.path.join(r, "images")
        if os.path.isdir(p):
            return p
    return None


IMAGES_DIR = _preferred_images_dir(TRAIN_CSV_PATH, TEST_CSV_PATH, DATA_CANDIDATES)

if SAMPLE_SUB_PATH is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in expected Kaggle input/data paths."
    )

print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)
print("TEST_CSV_PATH:", TEST_CSV_PATH)
print("TRAIN_CSV_PATH:", TRAIN_CSV_PATH)
print("IMAGES_DIR:", IMAGES_DIR)



## === cell 2
submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found submissions:", submissions_all)



## === cell 3
TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]


def _load_test_index():
    if TEST_CSV_PATH is None:
        base = pd.read_csv(SAMPLE_SUB_PATH)
        return base["image_id"].astype(str)
    test_df = pd.read_csv(TEST_CSV_PATH)
    return test_df["image_id"].astype(str)


def _normalize_submission_columns(sub: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "combinations": "multiple_diseases",
        "multiple disease": "multiple_diseases",
        "multiple_disease": "multiple_diseases",
        "multiple diseases": "multiple_diseases",
        "multiple_diseases": "multiple_diseases",
    }
    return sub.rename(columns=rename_map)


def ensemble(submissions_all, sub_idx, weights=[]):
    test_ids = _load_test_index()
    submission_with_weight = []
    for i in range(len(sub_idx)):
        path = submissions_all[sub_idx[i]]
        print(f"I'm taking submission {path} with weight {weights[i]}")
        sub = pd.read_csv(path)
        sub = _normalize_submission_columns(sub)

        if "image_id" in sub.columns:
            sub["image_id"] = sub["image_id"].astype(str)
            sub = sub.set_index("image_id")
            sub = sub.reindex(test_ids)
            if sub[TARGET_COLS].isna().any().any():
                raise ValueError(
                    f"Submission {path} is missing some test image_ids after reindexing."
                )
            arr = sub[TARGET_COLS].values
        else:
            if any(c not in sub.columns for c in TARGET_COLS):
                raise ValueError(
                    f"Submission {path} missing required columns {TARGET_COLS}."
                )
            arr = sub.loc[:, TARGET_COLS].values

        submission_with_weight.append(arr * weights[i])

    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 4
def make_submission_file(submission_avg, submissions_all):
    test_ids = _load_test_index()

    submission_df = pd.DataFrame({"image_id": test_ids})
    for j, c in enumerate(TARGET_COLS):
        submission_df[c] = submission_avg[:, j].astype(float)

    for c in TARGET_COLS:
        submission_df[c] = submission_df[c].clip(1e-6, 1 - 1e-6)

    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission_df.shape)
    print(submission_df.head())




## === cell 5
def _compute_smoothed_priors_from_train():
    if TRAIN_CSV_PATH is None:
        priors = pd.Series({c: 0.25 for c in TARGET_COLS}, dtype=float)
        return priors

    train_df = pd.read_csv(TRAIN_CSV_PATH)

    missing = [c for c in TARGET_COLS if c not in train_df.columns]
    if missing:
        priors = pd.Series({c: 0.25 for c in TARGET_COLS}, dtype=float)
        return priors

    y = train_df[TARGET_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    alpha = 1.0
    priors = (y.sum(axis=0) + alpha) / (len(y) + 2.0 * alpha)
    priors = priors.clip(1e-6, 1 - 1e-6)
    return priors


def _safe_read_image(path):
    try:
        from PIL import Image, ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = True
    except Exception:
        return None
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return img
    except Exception:
        return None


def _feature_length():
    dummy = np.zeros((10, 10, 3), dtype=np.float32)
    arr = dummy
    rgb_mean = arr.mean(axis=(0, 1))
    rgb_std = arr.std(axis=(0, 1))
    r_mean, g_mean, b_mean = rgb_mean.tolist()
    eps = 1e-6
    exg = 2.0 * g_mean - r_mean - b_mean
    ngrdi = (g_mean - r_mean) / (g_mean + r_mean + eps)
    gb_ratio = g_mean / (b_mean + eps)
    rg_ratio = r_mean / (g_mean + eps)
    gray = (
        0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
    ).astype(np.float32)
    q10, q50, q90 = np.quantile(gray, [0.10, 0.50, 0.90]).astype(np.float32).tolist()
    rq10, rq90 = np.quantile(arr[:, :, 0], [0.10, 0.90]).astype(np.float32).tolist()
    gq10, gq90 = np.quantile(arr[:, :, 1], [0.10, 0.90]).astype(np.float32).tolist()
    bq10, bq90 = np.quantile(arr[:, :, 2], [0.10, 0.90]).astype(np.float32).tolist()
    h, s, v = (0.0, 0.0, float(max(r_mean, g_mean, b_mean)))
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    gx = np.pad(gx, ((0, 0), (0, 1)), mode="edge")
    gy = np.pad(gy, ((0, 1), (0, 0)), mode="edge")
    gmag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    gmag_mean = float(gmag.mean())
    gmag_std = float(gmag.std())
    gmag_q90 = float(np.quantile(gmag, 0.90))
    mad = float(np.mean(np.abs(gray - np.median(gray))).astype(np.float32))
    chroma = arr.std(axis=2).astype(np.float32)
    chroma_mean = float(chroma.mean())
    chroma_std = float(chroma.std())
    g32 = gray
    gx2 = np.diff(g32, axis=1)
    gy2 = np.diff(g32, axis=0)
    gx2 = np.pad(gx2, ((0, 0), (0, 1)), mode="edge")
    gy2 = np.pad(gy2, ((0, 1), (0, 0)), mode="edge")
    gmag2 = np.sqrt(gx2 * gx2 + gy2 * gy2).astype(np.float32)
    gmag2_mean = float(gmag2.mean())
    gmag2_std = float(gmag2.std())
    fft_b1, fft_b2, fft_b3 = (0.0, 0.0, 0.0)
    r = arr[:, :, 0].reshape(-1)
    g = arr[:, :, 1].reshape(-1)
    b = arr[:, :, 2].reshape(-1)
    corr_rg = float(np.corrcoef(r, g)[0, 1]) if r.size > 1 else 0.0
    corr_rb = float(np.corrcoef(r, b)[0, 1]) if r.size > 1 else 0.0
    corr_gb = float(np.corrcoef(g, b)[0, 1]) if r.size > 1 else 0.0
    feats = np.array(
        [
            rgb_mean[0],
            rgb_mean[1],
            rgb_mean[2],
            rgb_std[0],
            rgb_std[1],
            rgb_std[2],
            h,
            s,
            v,
            exg,
            ngrdi,
            gb_ratio,
            rg_ratio,
            q10,
            q50,
            q90,
            rq10,
            rq90,
            gq10,
            gq90,
            bq10,
            bq90,
            float(gray.mean()),
            float(gray.std()),
            gmag_mean,
            gmag_std,
            gmag_q90,
            mad,
            chroma_mean,
            chroma_std,
            gmag2_mean,
            gmag2_std,
            fft_b1,
            fft_b2,
            fft_b3,
            float(np.quantile(gmag2, 0.90)) if gmag2.size else 0.0,
            corr_rg,
            corr_rb,
            corr_gb,
        ],
        dtype=np.float32,
    )
    return int(feats.shape[0])


N_FEATURES = _feature_length()
print("N_FEATURES:", N_FEATURES)


def _extract_image_features(
    image_id: str, images_dir: str, size=(160, 160)
) -> np.ndarray:
    """
    Same overall 'simple image-statistics' feature extraction.
    """
    img_path = os.path.join(images_dir, f"{image_id}.jpg")
    img = _safe_read_image(img_path)
    if img is None:
        return np.full(N_FEATURES, np.nan, dtype=np.float32)

    try:
        img = img.resize(size)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    except Exception:
        return np.full(N_FEATURES, np.nan, dtype=np.float32)

    rgb_mean = arr.mean(axis=(0, 1))
    rgb_std = arr.std(axis=(0, 1))

    r_mean, g_mean, b_mean = rgb_mean.tolist()

    eps = 1e-6
    exg = 2.0 * g_mean - r_mean - b_mean
    ngrdi = (g_mean - r_mean) / (g_mean + r_mean + eps)
    gb_ratio = g_mean / (b_mean + eps)
    rg_ratio = r_mean / (g_mean + eps)

    gray = (
        0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
    ).astype(np.float32)
    q10, q50, q90 = np.quantile(gray, [0.10, 0.50, 0.90]).astype(np.float32).tolist()

    rq10, rq90 = np.quantile(arr[:, :, 0], [0.10, 0.90]).astype(np.float32).tolist()
    gq10, gq90 = np.quantile(arr[:, :, 1], [0.10, 0.90]).astype(np.float32).tolist()
    bq10, bq90 = np.quantile(arr[:, :, 2], [0.10, 0.90]).astype(np.float32).tolist()

    try:
        import colorsys

        h, s, v = colorsys.rgb_to_hsv(float(r_mean), float(g_mean), float(b_mean))
    except Exception:
        h, s, v = (0.0, 0.0, float(max(r_mean, g_mean, b_mean)))

    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    gx = np.pad(gx, ((0, 0), (0, 1)), mode="edge")
    gy = np.pad(gy, ((0, 1), (0, 0)), mode="edge")
    gmag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    gmag_mean = float(gmag.mean())
    gmag_std = float(gmag.std())
    gmag_q90 = float(np.quantile(gmag, 0.90))

    mad = float(np.mean(np.abs(gray - np.median(gray))).astype(np.float32))

    chroma = arr.std(axis=2).astype(np.float32)
    chroma_mean = float(chroma.mean())
    chroma_std = float(chroma.std())

    try:
        img_small = img.resize((32, 32))
        arr_small = np.asarray(img_small, dtype=np.float32) / 255.0
        g32 = (
            0.2989 * arr_small[:, :, 0]
            + 0.5870 * arr_small[:, :, 1]
            + 0.1140 * arr_small[:, :, 2]
        ).astype(np.float32)
    except Exception:
        g32 = gray[::5, ::5].astype(np.float32)
        if g32.shape[0] < 16 or g32.shape[1] < 16:
            g32 = gray[::4, ::4].astype(np.float32)

    gx2 = np.diff(g32, axis=1)
    gy2 = np.diff(g32, axis=0)
    gx2 = np.pad(gx2, ((0, 0), (0, 1)), mode="edge")
    gy2 = np.pad(gy2, ((0, 1), (0, 0)), mode="edge")
    gmag2 = np.sqrt(gx2 * gx2 + gy2 * gy2).astype(np.float32)
    gmag2_mean = float(gmag2.mean())
    gmag2_std = float(gmag2.std())

    try:
        F = np.fft.rfft2(g32.astype(np.float32))
        P = (F.real * F.real + F.imag * F.imag).astype(np.float32)

        h2, w2 = g32.shape
        fy = np.fft.fftfreq(h2).reshape(-1, 1).astype(np.float32)
        fx = np.fft.rfftfreq(w2).reshape(1, -1).astype(np.float32)
        fr = np.sqrt(fx * fx + fy * fy)

        b1 = (
            float(P[(fr > 0.00) & (fr <= 0.10)].mean())
            if np.any((fr > 0.00) & (fr <= 0.10))
            else 0.0
        )
        b2 = (
            float(P[(fr > 0.10) & (fr <= 0.25)].mean())
            if np.any((fr > 0.10) & (fr <= 0.25))
            else 0.0
        )
        b3 = (
            float(P[(fr > 0.25) & (fr <= 0.50)].mean())
            if np.any((fr > 0.25) & (fr <= 0.50))
            else 0.0
        )
        fft_b1 = float(np.log1p(b1))
        fft_b2 = float(np.log1p(b2))
        fft_b3 = float(np.log1p(b3))
    except Exception:
        fft_b1, fft_b2, fft_b3 = (0.0, 0.0, 0.0)

    r = arr[:, :, 0].reshape(-1)
    g = arr[:, :, 1].reshape(-1)
    b = arr[:, :, 2].reshape(-1)

    def _safe_corr(x, y):
        if x.size < 2:
            return 0.0
        xc = x - x.mean()
        yc = y - y.mean()
        denom = float(np.sqrt((xc * xc).mean() * (yc * yc).mean()) + 1e-12)
        return float(((xc * yc).mean()) / denom)

    corr_rg = _safe_corr(r, g)
    corr_rb = _safe_corr(r, b)
    corr_gb = _safe_corr(g, b)

    feats = np.array(
        [
            rgb_mean[0],
            rgb_mean[1],
            rgb_mean[2],
            rgb_std[0],
            rgb_std[1],
            rgb_std[2],
            h,
            s,
            v,
            exg,
            ngrdi,
            gb_ratio,
            rg_ratio,
            q10,
            q50,
            q90,
            rq10,
            rq90,
            gq10,
            gq90,
            bq10,
            bq90,
            float(gray.mean()),
            float(gray.std()),
            gmag_mean,
            gmag_std,
            gmag_q90,
            mad,
            chroma_mean,
            chroma_std,
            gmag2_mean,
            gmag2_std,
            fft_b1,
            fft_b2,
            fft_b3,
            float(np.quantile(gmag2, 0.90)) if gmag2.size else 0.0,
            corr_rg,
            corr_rb,
            corr_gb,
        ],
        dtype=np.float32,
    )
    return feats


def _fit_1d_calibrator(oof_pred, y_true, seed=0):
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return None
    eps = 1e-6
    oof_pred = np.clip(oof_pred.astype(np.float64), eps, 1 - eps)
    logit = np.log(oof_pred / (1 - oof_pred)).reshape(-1, 1)
    y_true = y_true.astype(int)

    if len(np.unique(y_true)) < 2:
        return None

    cal = LogisticRegression(
        solver="lbfgs", max_iter=800, C=1.0, random_state=seed, n_jobs=1
    )
    cal.fit(logit, y_true)
    return cal


def _apply_1d_calibrator(cal, test_pred):
    if cal is None:
        return test_pred
    eps = 1e-6
    test_pred = np.clip(test_pred.astype(np.float64), eps, 1 - eps)
    logit = np.log(test_pred / (1 - test_pred)).reshape(-1, 1)
    return cal.predict_proba(logit)[:, 1]


def train_and_predict_image_stats_model(
    train_csv_path: str, test_ids: pd.Series, images_dir: str
):
    priors = _compute_smoothed_priors_from_train()
    if (
        (train_csv_path is None)
        or (images_dir is None)
        or (not os.path.isdir(images_dir))
    ):
        print("Images/train not available; falling back to constant priors.")
        return np.tile(priors.values.reshape(1, -1), (len(test_ids), 1))

    train_df = pd.read_csv(train_csv_path)
    train_df["image_id"] = train_df["image_id"].astype(str)

    seed = 0

    X_train = np.vstack(
        [_extract_image_features(i, images_dir) for i in train_df["image_id"].values]
    )
    X_test = np.vstack(
        [_extract_image_features(i, images_dir) for i in test_ids.values]
    )

    col_medians = np.nanmedian(X_train, axis=0)
    inds_tr = np.where(np.isnan(X_train))
    if len(inds_tr[0]) > 0:
        X_train[inds_tr] = np.take(col_medians, inds_tr[1])
    inds_te = np.where(np.isnan(X_test))
    if len(inds_te[0]) > 0:
        X_test[inds_te] = np.take(col_medians, inds_te[1])

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        print("sklearn not available; falling back to priors. Error:", repr(e))
        return np.tile(priors.values.reshape(1, -1), (len(test_ids), 1))

    preds = np.zeros((len(test_ids), len(TARGET_COLS)), dtype=np.float64)

    for j, c in enumerate(TARGET_COLS):
        y = pd.to_numeric(train_df[c], errors="coerce").fillna(0.0).values.astype(int)

        pos = int(y.sum())
        neg = int(len(y) - pos)
        max_splits = min(5, pos, neg)
        n_splits = max(2, max_splits)

        if len(np.unique(y)) < 2 or max_splits < 2:
            preds[:, j] = float(priors[c])
            continue

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        test_pred_accum = np.zeros(len(test_ids), dtype=np.float64)
        oof_pred = np.zeros(len(train_df), dtype=np.float64)

        base_C = 1.0
        if len(train_df) < 1200:
            base_C = 1.2
        elif len(train_df) > 1600:
            base_C = 0.9

        for tr_idx, va_idx in skf.split(X_train, y):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X_train[tr_idx])
            Xva = scaler.transform(X_train[va_idx])
            Xte = scaler.transform(X_test)

            clf = LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                C=base_C,
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            )
            clf.fit(Xtr, y[tr_idx])

            oof_pred[va_idx] = clf.predict_proba(Xva)[:, 1]
            test_pred_accum += clf.predict_proba(Xte)[:, 1] / n_splits

        cal = _fit_1d_calibrator(oof_pred, y, seed=seed)
        test_pred_cal = _apply_1d_calibrator(cal, test_pred_accum)

        preds[:, j] = 0.90 * test_pred_cal + 0.10 * float(priors[c])

    preds = np.clip(preds, 1e-6, 1 - 1e-6)
    return preds


if len(submissions_all) >= 3:
    submission_avg = ensemble(submissions_all, [0, 1, 2], [0.18, 0.72, 0.1])
else:
    print(
        "Not enough submissions found to ensemble (need >=3). "
        "Training a lightweight image-stat logistic baseline to improve ROC AUC."
    )
    test_ids = _load_test_index()
    submission_avg = train_and_predict_image_stats_model(
        TRAIN_CSV_PATH, test_ids, IMAGES_DIR
    )

make_submission_file(submission_avg, submissions_all)
