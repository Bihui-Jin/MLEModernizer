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

0.9678361818267092

# 6. Current score

0.69587

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The crash happens because `/kaggle/input/submissions/` doesn’t exist here, so `submissions_all` is empty and indexing `[0,2,4]` fails. I keep the ensemble logic intact, but add a safe fallback: if no external submissions are found, generate a valid baseline submission by using the class priors (mean label frequencies) from `train.csv` for every test image. This guarantees an end-to-end run and creates a correctly formatted `submission.csv` with the required columns. If the submissions folder exists and contains enough files, it behave exactly like your original ensembling code.'
- What this solution (achieved 0.64082) has done: 'Your current 0.5 score is consistent with the fallback “class prior” submission (constant probabilities), so to move toward the 0.9678 target we should use the actual images to produce non-constant predictions while keeping changes minimal. I keep your existing ensemble logic intact when external submissions exist, but replace only the fallback branch with a lightweight, deterministic image-based model using sklearn’s LogisticRegression on simple color features (per-channel mean/std), which fits fast and typically scores far above 0.5 on this dataset. I also ensure the submission rows align exactly to `test.csv` order and the columns match `sample_submission.csv`. This preserves your evaluation semantics (probabilities per class) and produces a valid `submission.csv` end-to-end.'
- What this solution (achieved 0.53619) has done: 'Your current fallback model is very weak for this image task (simple RGB mean/std), so the minimal way to move toward the 0.9678 target is to keep the same sklearn LogisticRegression per-class approach but enrich the features while staying lightweight and deterministic. I upgrade the feature extractor to include (1) downsampled grayscale pixels and (2) simple color statistics, then standardize features with `StandardScaler`, which is important for stable LR probabilities. I also add `class_weight="balanced"` to reduce bias from class imbalance (especially for `multiple_diseases`) without changing the modeling family. The external-submission ensemble branch remains unchanged, and the script still writes a valid `submission.csv` aligned to `test.csv` with the required columns.'
- What this solution (achieved 0.60549) has done: 'Your current score (0.536) is far below the target (0.968), so we should improve the fallback (image-based) branch while keeping the same per-class LogisticRegression training loop and submission semantics. The smallest meaningful lift is to extract richer but still lightweight features: add HSV statistics plus a compact HOG descriptor on a small grayscale image, while keeping your existing grayscale downsample pixels and RGB stats. We keep StandardScaler + LogisticRegression (same model family), and only tune regularization slightly (higher C) to better fit these stronger features without changing the approach. The external-submission ensembling path remains unchanged, and the script still writes a correctly formatted `submission.csv` aligned to `test.csv`.'
- What this solution (achieved 0.60549) has done: 'Your current gap to the target is large (0.6055 vs 0.9678), so we need a meaningful lift while preserving your core approach (per-class LogisticRegression on handcrafted image features). The minimal high-impact fix is to correct the HOG feature length calculation (it’s currently wrong, causing silent feature misalignment/zero-padding that harms learning) by deriving the length from an actual HOG call, then using that exact length for the feature matrix. I also make the feature extraction robust to any unexpected HOG length changes (by trimming/padding deterministically) without changing the model family, training loop, or submission semantics. Everything else (paths, ensemble branch, scaler + LR, columns/order, submission writing) stays the same.'
- What this solution (achieved 0.65096) has done: 'Your current gap to the target is large (0.6055 vs 0.9678, higher-is-better), so the fallback image-based branch needs a meaningful lift while preserving your existing approach (handcrafted features + per-class LogisticRegression). The smallest high-impact change is to keep the same model/training loop but replace the most lossy part of the features (raw downsampled pixels) with a slightly richer, still-lightweight representation (downsampled grayscale + HOG, plus simple RGB/HSV stats) and add a second HOG scale; this tends to improve separability without changing semantics. I also fix a subtle but important issue: fitting one global scaler across all features can be dominated by the high-dimensional pixel block—so I keep a single StandardScaler (same idea) but ensure feature blocks are numerically comparable by using float32 throughout and consistent extraction, which improves LR conditioning. The external-submission ensemble path remains unchanged, and the script still writes a valid `submission.csv` aligned to `test.csv` with the required columns.'
- What this solution (achieved 0.66623) has done: 'Your current score (0.65096) is far below the target (0.96784), so we should improve the fallback image-based branch while keeping the same core approach (handcrafted features → StandardScaler → per-class LogisticRegression). The most minimal, high-impact change is to align the HOG feature extraction with standard practice for this dataset by using `transform_sqrt=True` (contrast normalization) and a slightly finer gradient binning (`orientations=12`), without changing the model family or training loop. I also keep the feature dimensionality deterministic by computing HOG lengths with the exact same parameters used at extraction time, avoiding subtle train/test mismatch. The external-submission ensemble path and the submission formatting/alignment remain unchanged.'
- What this solution (achieved 0.67043) has done: 'Your current score (0.666) is far below the target (0.9678), so we should improve the fallback image-based branch while keeping the same core approach: handcrafted features → StandardScaler → per-class LogisticRegression → probabilities. The smallest high-impact change is to add a compact “color layout” feature (coarse spatial RGB means) that captures lesion localization patterns the current global stats + HOG can miss, without changing the model family or training loop. I also switch `multi_class` to the modern default (`"auto"`) and slightly increase `max_iter` to ensure convergence with the expanded feature vector (no change in semantics). The external-submission ensemble path and submission formatting/order remain unchanged.'
- What this solution (achieved 0.67043) has done: 'Your current score (0.67043) is far below the target (0.9678), so we should improve the fallback image-based branch while keeping the exact same core approach (handcrafted features → StandardScaler → per-class LogisticRegression → probabilities). The smallest high-impact fix is to correct a label/column mismatch: your code predicts `multiple_diseases`, but your earlier description and many Plant Pathology solutions use `combinations`; if the dataset variant actually contains `combinations`, training on the wrong/missing column (or silently misaligned columns) can severely cap ROC AUC. I also add a minimal safeguard to automatically detect whether the second class column is `multiple_diseases` or `combinations` and use that consistently for training and submission, without changing modeling logic. Finally, I enforce exact row alignment to `test.csv` order (by merging the template with `test_df`) so predictions can’t get mis-ordered, which can otherwise destroy AUC.'
- What this solution (achieved 0.67719) has done: 'Your current score (0.670) is far below the 0.9678 target, so we should improve the fallback image-model while keeping the exact same core pipeline (handcrafted features → StandardScaler → per-class LogisticRegression → probabilities). The most minimal high-impact change is to fix feature/label inconsistencies: use the same class column names end-to-end (including the `md_col` choice) and ensure the submission columns exactly match the sample submission for this dataset variant. Then, without changing the model family or training loop, we add only a tiny amount of extra spatial information (coarse HSV layout alongside existing RGB layout) to help LR separate localized lesions. Finally, we keep row alignment strictly to `test.csv` order to avoid any silent AUC destruction from mis-ordering.'
- What this solution (achieved 0.68217) has done: 'To move your score upward toward the 0.9678 target while preserving your exact core pipeline (handcrafted features → StandardScaler → per-class LogisticRegression), I’m making two minimal, high-impact fixes in the fallback image-model branch. First, I ensure the feature extraction is robust to occasional missing/corrupt image files by substituting a deterministic blank image instead of crashing or producing misaligned rows (misalignment can severely hurt AUC). Second, I add a very small amount of shape/structure information by including a compact “edge density” statistic per coarse grid cell (computed from simple gradient magnitude on grayscale), which complements HOG without changing the modeling family or training loop. Everything else (external submission ensemble path, class column auto-detection, column names/order, and writing `submission.csv`) remains the same.'
- What this solution (achieved 0.68638) has done: 'Your current score is far below the target (0.682 vs 0.968, higher-is-better), so we should strengthen the fallback image-based model while keeping the same core pipeline (handcrafted features → StandardScaler → per-class LogisticRegression → probabilities). The most minimal high-impact change is to improve feature conditioning and add one lightweight texture cue that LogisticRegression can exploit: per-channel histogram features (coarse bins) plus a simple Laplacian-of-gray “spot/lesion” statistic, without changing the model family or training loop. I also switch the LR `class_weight` from `"balanced"` to `None` (still the same LR) because ROC-AUC often improves when probabilities are better calibrated rather than reweighted, especially for rare classes. Everything else (paths, ensemble branch, column auto-detection, strict test order alignment, and writing `submission.csv`) remains unchanged.'
- What this solution (achieved 0.69154) has done: 'Your score (0.686) is far below the target (0.968), so we should improve the fallback image-based model while keeping your exact pipeline (handcrafted features → StandardScaler → per-class LogisticRegression → probabilities) unchanged in spirit. The most minimal high-impact improvement is to add a tiny set of vegetation-focused color indices (ExG/ExR/NDI-like) and their coarse spatial layout, which often separates “healthy” vs “diseased” leaves better than raw RGB/HSV stats alone. I also add a very small “green mask ratio” feature (how much of the image looks like leaf vs background) to stabilize predictions without changing the model family or training loop. The external-submission ensemble path remains untouched, and the script still writes a correctly formatted `submission.csv` aligned to `test.csv`.'
- What this solution (achieved 0.69587) has done: 'Your current score (0.6915) is far below the target (0.9678), so we should improve the fallback image-based branch while keeping your exact pipeline (handcrafted features → StandardScaler → per-class LogisticRegression → probabilities). The most minimal high-impact gain here is to make the handcrafted features more invariant to lighting/background: (1) add per-channel histogram features in HSV (in addition to RGB hist) and (2) add a simple “excess green” (ExG) threshold mask ratio that better captures how much of the image is leaf vs background (a stronger version of your current green ratio). These additions don’t change the model family, training loop, or submission semantics; they just add a small, fast-to-compute feature block that LogisticRegression can exploit for separability. Everything else (external submission ensemble path, column auto-detection, strict test order alignment, and writing `submission.csv`) remains unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/"

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]


def _find_file(filename: str):
    for root in DATA_ROOT_CANDIDATES:
        path = os.path.join(root, filename)
        if os.path.exists(path):
            return path
        nested = os.path.join(root, "plant-pathology-2020-fgvc7", filename)
        if os.path.exists(nested):
            return nested
    return None


def _find_dir(dirname: str):
    for root in DATA_ROOT_CANDIDATES:
        path = os.path.join(root, dirname)
        if os.path.isdir(path):
            return path
        nested = os.path.join(root, "plant-pathology-2020-fgvc7", dirname)
        if os.path.isdir(nested):
            return nested
    return None


TRAIN_CSV = _find_file("train.csv")
TEST_CSV = _find_file("test.csv")
SAMPLE_SUB_CSV = _find_file("sample_submission.csv")
IMAGES_DIR = _find_dir("images")

print("Resolved paths:")
print("TRAIN_CSV:", TRAIN_CSV)
print("TEST_CSV:", TEST_CSV)
print("SAMPLE_SUB_CSV:", SAMPLE_SUB_CSV)
print("IMAGES_DIR:", IMAGES_DIR)



## === cell 2
submissions_all = []
if os.path.exists(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found submission files:", submissions_all)




## === cell 3
def ensemble(submissions_all, sub_idx, weights=[]):
    submission_with_weight = []
    for i in range(len(sub_idx)):
        print(
            f"I'm taking submission {submissions_all[sub_idx[i]]} with weight {weights[i]}"
        )
        submission = pd.read_csv(submissions_all[sub_idx[i]])
        submission = submission.loc[
            :, ["healthy", "multiple_diseases", "rust", "scab"]
        ].values
        submission_with_weight.append(submission * weights[i])
    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 4
def make_submission_file(submission_avg, template_csv_path, test_csv_path=None):
    submission_df = pd.read_csv(template_csv_path)

    if test_csv_path is not None and os.path.exists(test_csv_path):
        test_df_local = pd.read_csv(test_csv_path)
        submission_df = submission_df.merge(
            test_df_local[["image_id"]], on="image_id", how="right", sort=False
        )

    required = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    missing = [c for c in required if c not in submission_df.columns]
    if missing:
        raise ValueError(
            f"Template submission is missing required columns {missing}. "
            f"Found columns: {list(submission_df.columns)}"
        )

    submission_df = submission_df.loc[:, required]
    submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = submission_avg
    submission_df.to_csv("submission.csv", index=False)
    print("Wrote: submission.csv")
    print(submission_df.head())




## === cell 5
if (
    TRAIN_CSV is None
    or TEST_CSV is None
    or SAMPLE_SUB_CSV is None
    or IMAGES_DIR is None
):
    raise FileNotFoundError(
        "Could not locate train.csv/test.csv/sample_submission.csv/images/ in expected Kaggle paths."
    )

train_df_probe = pd.read_csv(TRAIN_CSV, nrows=5)
sample_sub_probe = pd.read_csv(SAMPLE_SUB_CSV, nrows=5)

if (
    "multiple_diseases" in train_df_probe.columns
    and "multiple_diseases" in sample_sub_probe.columns
):
    md_col = "multiple_diseases"
elif (
    "combinations" in train_df_probe.columns
    and "combinations" in sample_sub_probe.columns
):
    md_col = "combinations"
else:
    md_col = (
        "multiple_diseases"
        if "multiple_diseases" in train_df_probe.columns
        else "combinations"
    )

target_cols_train = ["healthy", md_col, "rust", "scab"]
print("Using second class column as:", md_col)



## === cell 6
if len(submissions_all) >= 5:
    submission_avg = ensemble(submissions_all, [0, 2, 4], [0.15, 0.8, 0.05])
    make_submission_file(submission_avg, submissions_all[0], test_csv_path=TEST_CSV)
else:
    import numpy as np
    from PIL import Image, ImageFile
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from skimage.feature import hog
    from skimage.color import rgb2gray, rgb2hsv
    from skimage.filters import laplace

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

    if "image_id" not in sample_sub.columns:
        raise ValueError("sample_submission.csv must contain image_id column.")

    if (
        "multiple_diseases" not in sample_sub.columns
        and "combinations" in sample_sub.columns
    ):
        sample_sub = sample_sub.rename(columns={"combinations": "multiple_diseases"})

    sample_sub = sample_sub.merge(
        test_df[["image_id"]], on="image_id", how="right", sort=False
    )

    required_out_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    for c in required_out_cols:
        if c not in sample_sub.columns:
            sample_sub[c] = 0.0
    sample_sub = sample_sub[required_out_cols]

    def _img_path(image_id: str) -> str:
        return os.path.join(IMAGES_DIR, f"{image_id}.jpg")

    HOG_KW = dict(
        orientations=12,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )

    def _compute_hog_len(size_hog=(64, 64)):
        dummy = np.zeros((size_hog[0], size_hog[1]), dtype=np.float32)
        h = hog(dummy, **HOG_KW)
        return int(h.shape[0])

    def _coarse_layout(arr: np.ndarray, grid=(4, 4)) -> np.ndarray:
        H, W, C = arr.shape
        gh, gw = grid
        ys = np.linspace(0, H, gh + 1, dtype=int)
        xs = np.linspace(0, W, gw + 1, dtype=int)
        out = np.zeros((gh * gw * C,), dtype=np.float32)
        k = 0
        for yi in range(gh):
            y0, y1 = ys[yi], ys[yi + 1]
            for xi in range(gw):
                x0, x1 = xs[xi], xs[xi + 1]
                patch = arr[y0:y1, x0:x1, :]
                if patch.size == 0:
                    out[k : k + C] = 0.0
                else:
                    out[k : k + C] = patch.mean(axis=(0, 1))
                k += C
        return out

    def _edge_layout(gray01: np.ndarray, grid=(4, 4)) -> np.ndarray:
        g = gray01.astype(np.float32)
        gy, gx = np.gradient(g)
        mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
        H, W = mag.shape
        gh, gw = grid
        ys = np.linspace(0, H, gh + 1, dtype=int)
        xs = np.linspace(0, W, gw + 1, dtype=int)
        out = np.zeros((gh * gw,), dtype=np.float32)
        k = 0
        for yi in range(gh):
            y0, y1 = ys[yi], ys[yi + 1]
            for xi in range(gw):
                x0, x1 = xs[xi], xs[xi + 1]
                patch = mag[y0:y1, x0:x1]
                out[k] = float(patch.mean()) if patch.size else 0.0
                k += 1
        return out

    def _color_hist(arr01: np.ndarray, bins: int = 16) -> np.ndarray:
        eps = 1e-12
        feats = []
        for ch in range(3):
            h, _ = np.histogram(
                arr01[:, :, ch], bins=bins, range=(0.0, 1.0), density=False
            )
            h = h.astype(np.float32)
            h = h / (h.sum() + eps)
            feats.append(h)
        return np.concatenate(feats, axis=0).astype(np.float32)

    def _laplace_stats(gray01: np.ndarray) -> np.ndarray:
        l = laplace(gray01.astype(np.float32)).astype(np.float32)
        a = np.abs(l)
        return np.array([a.mean(), a.std(), np.quantile(a, 0.9)], dtype=np.float32)

    def _safe_open_rgb(path: str, fallback_size=(96, 96)) -> Image.Image:
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            return Image.fromarray(
                np.zeros((fallback_size[1], fallback_size[0], 3), dtype=np.uint8),
                mode="RGB",
            )

    def _veg_indices(arr01: np.ndarray) -> np.ndarray:
        r = arr01[:, :, 0].astype(np.float32)
        g = arr01[:, :, 1].astype(np.float32)
        b = arr01[:, :, 2].astype(np.float32)
        eps = 1e-6
        exg = (2.0 * g - r - b).astype(np.float32)
        exr = (1.4 * r - g).astype(np.float32)
        ndi = ((g - r) / (g + r + eps)).astype(np.float32)
        return np.stack([exg, exr, ndi], axis=-1).astype(np.float32)

    def _exg_ratio(arr01: np.ndarray) -> np.ndarray:
        veg = _veg_indices(arr01)
        exg = veg[:, :, 0]
        mask = exg > 0.05
        return np.array([mask.mean(dtype=np.float64)], dtype=np.float32)

    def _green_ratio(arr01: np.ndarray) -> np.ndarray:
        r = arr01[:, :, 0].astype(np.float32)
        g = arr01[:, :, 1].astype(np.float32)
        b = arr01[:, :, 2].astype(np.float32)
        mask = (g > r) & (g > b) & (g > 0.2)
        return np.array([mask.mean(dtype=np.float64)], dtype=np.float32)

    def _extract_features(
        image_ids,
        size_color=(96, 96),
        size_gray=(40, 40),
        size_hog1=(64, 64),
        size_hog2=(96, 96),
        layout_grid=(4, 4),
        hist_bins=16,
    ):
        hog1_len = _compute_hog_len(size_hog=size_hog1)
        hog2_len = _compute_hog_len(size_hog=size_hog2)

        layout_rgb_len = int(layout_grid[0] * layout_grid[1] * 3)
        layout_hsv_len = int(layout_grid[0] * layout_grid[1] * 3)
        edge_len = int(layout_grid[0] * layout_grid[1])
        hist_len = int(3 * hist_bins)
        lap_len = 3

        veg_len = 3  # global means of ExG/ExR/NDI
        veg_layout_len = int(layout_grid[0] * layout_grid[1] * 3)

        green_ratio_len = 1
        exg_ratio_len = 1

        hist_hsv_len = int(3 * hist_bins)

        feat_dim = (
            12
            + (size_gray[0] * size_gray[1])
            + layout_rgb_len
            + layout_hsv_len
            + edge_len
            + hist_len
            + hist_hsv_len
            + lap_len
            + veg_len
            + veg_layout_len
            + green_ratio_len
            + exg_ratio_len
            + hog1_len
            + hog2_len
        )
        X = np.zeros((len(image_ids), feat_dim), dtype=np.float32)

        for i, iid in enumerate(image_ids):
            p = _img_path(iid)
            img_rgb = _safe_open_rgb(p, fallback_size=size_color)

            img_c = img_rgb.resize(size_color)
            arr = np.asarray(img_c, dtype=np.float32) / 255.0
            mu = arr.mean(axis=(0, 1))
            sd = arr.std(axis=(0, 1))
            X[i, 0:3] = mu
            X[i, 3:6] = sd

            hsv = rgb2hsv(arr)
            mu_hsv = hsv.mean(axis=(0, 1))
            sd_hsv = hsv.std(axis=(0, 1))
            X[i, 6:9] = mu_hsv
            X[i, 9:12] = sd_hsv

            img_g_small = img_rgb.convert("L").resize(size_gray)
            g_small_2d = np.asarray(img_g_small, dtype=np.float32) / 255.0
            g_small = g_small_2d.reshape(-1)
            start = 12
            end = start + g_small.shape[0]
            X[i, start:end] = g_small

            pos = end
            layout_rgb = _coarse_layout(arr, grid=layout_grid)
            X[i, pos : pos + layout_rgb_len] = layout_rgb
            pos += layout_rgb_len

            layout_hsv = _coarse_layout(hsv.astype(np.float32), grid=layout_grid)
            X[i, pos : pos + layout_hsv_len] = layout_hsv
            pos += layout_hsv_len

            edge = _edge_layout(g_small_2d, grid=layout_grid)
            X[i, pos : pos + edge_len] = edge
            pos += edge_len

            chist_rgb = _color_hist(arr, bins=hist_bins)
            X[i, pos : pos + hist_len] = chist_rgb
            pos += hist_len

            chist_hsv = _color_hist(hsv.astype(np.float32), bins=hist_bins)
            X[i, pos : pos + hist_hsv_len] = chist_hsv
            pos += hist_hsv_len

            lst = _laplace_stats(g_small_2d)
            X[i, pos : pos + lap_len] = lst
            pos += lap_len

            veg = _veg_indices(arr)
            X[i, pos : pos + veg_len] = veg.mean(axis=(0, 1))
            pos += veg_len

            veg_layout = _coarse_layout(veg, grid=layout_grid)
            X[i, pos : pos + veg_layout_len] = veg_layout
            pos += veg_layout_len

            gr = _green_ratio(arr)
            X[i, pos : pos + green_ratio_len] = gr
            pos += green_ratio_len

            exgr = _exg_ratio(arr)
            X[i, pos : pos + exg_ratio_len] = exgr
            pos += exg_ratio_len

            img_hog1 = img_rgb.resize(size_hog1)
            arr_hog1 = np.asarray(img_hog1, dtype=np.float32) / 255.0
            gray1 = rgb2gray(arr_hog1).astype(np.float32)
            h1 = hog(gray1, **HOG_KW).astype(np.float32)
            if h1.shape[0] >= hog1_len:
                X[i, pos : pos + hog1_len] = h1[:hog1_len]
            else:
                X[i, pos : pos + h1.shape[0]] = h1
            pos += hog1_len

            img_hog2 = img_rgb.resize(size_hog2)
            arr_hog2 = np.asarray(img_hog2, dtype=np.float32) / 255.0
            gray2 = rgb2gray(arr_hog2).astype(np.float32)
            h2 = hog(gray2, **HOG_KW).astype(np.float32)
            if h2.shape[0] >= hog2_len:
                X[i, pos : pos + hog2_len] = h2[:hog2_len]
            else:
                X[i, pos : pos + h2.shape[0]] = h2

        return X

    X_train = _extract_features(train_df["image_id"].tolist())
    X_test = _extract_features(test_df["image_id"].tolist())

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    target_cols_out = ["healthy", "multiple_diseases", "rust", "scab"]
    preds = np.zeros((len(test_df), len(target_cols_out)), dtype=np.float64)

    for out_j, out_col in enumerate(target_cols_out):
        train_col = md_col if out_col == "multiple_diseases" else out_col
        if train_col not in train_df.columns:
            preds[:, out_j] = 0.0
            continue

        y = train_df[train_col].astype(int).values
        if len(np.unique(y)) < 2:
            preds[:, out_j] = float(y.mean())
            continue

        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=2500,
            C=5.0,
            class_weight=None,
            random_state=0,
            multi_class="auto",
        )
        clf.fit(X_train_s, y)
        preds[:, out_j] = clf.predict_proba(X_test_s)[:, 1]

    preds = np.clip(preds, 0.0, 1.0)

    make_submission_file(preds, SAMPLE_SUB_CSV, test_csv_path=TEST_CSV)
