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

0.9625763959938703

# 6. Current score

0.63006

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the shape-mismatch bug by ensuring every candidate submission is aligned to the competition test set (`test.csv`) by `image_id`, and I ignore any CSVs that look like train-sized predictions. I also normalize/renormalize ensemble weights so the averaged probabilities stay on a consistent scale (score-neutral but safer). Finally, I make the fallback produce a valid uniform-probability submission with the correct rows/columns and write `submission.csv` end-to-end.'
- What this solution (achieved 0.66866) has done: 'Your current 0.5 score strongly suggests the fallback uniform 0.25 submission is being used (or the discovered CSVs are not real model predictions), so the smallest safe way to move toward the 0.9626 target is to generate a real image-based model prediction inside this notebook rather than trying to ensemble unknown external CSVs. I keep your existing submission alignment/formatting logic, but add a minimal scikit-learn baseline that reads images, extracts simple color/texture features, trains a one-vs-rest logistic regression for the 4 labels, and predicts probabilities for the test set. This preserves evaluation semantics (probabilities per class) and should materially improve ROC AUC versus uniform guessing while staying within the installed-package constraints. If image reading fails for any reason, the code still fall back to your original CSV-discovery ensemble and then finally to uniform 0.25 so it always produces a valid `submission.csv`.'
- What this solution (achieved 0.58405) has done: 'Your current baseline uses only 10 global color statistics, which is usually too weak for this task and explains the big gap to the 0.9626 target. I keep the same scikit-learn OneVsRest logistic-regression pipeline, but strengthen the existing feature extractor in a minimal way by adding simple downsampled “thumbnail” pixel features (still just deterministic numeric features, no new model/training loop). I also add a tiny, deterministic image resize inside `_read_image_rgb` to make feature extraction consistent and faster, and increase `max_iter` slightly to ensure convergence rather than changing the learning approach. This should materially increase ROC AUC while preserving your overall structure and still writing a valid `submission.csv` end-to-end.'
- What this solution (achieved 0.58302) has done: 'Your current pipeline is already producing a valid submission, but the score gap to 0.9626 is large, so the most reliable minimal improvement is to strengthen the *same* scikit-learn baseline without changing its core approach (still deterministic hand-crafted features + OneVsRest logistic regression). I keep the exact training loop/model family, but (1) add a couple of very cheap texture/shape statistics (edge energy + Laplacian variance + center-vs-border color difference) that are highly relevant for leaf lesion patterns, and (2) use a slightly higher-resolution thumbnail (still small) to carry more spatial information. These changes typically improve ROC AUC materially versus global color stats while staying within your constraints and runtime. Everything else (alignment, clipping, fallback behavior, submission writing) stays the same.'
- What this solution (achieved 0.57807) has done: 'Your current score (0.583) is far below the target (0.9626), so we should improve the same sklearn baseline without changing the modeling family/training semantics. The biggest issue is that OneVsRest logistic regression treats each label independently and can’t naturally represent the “exactly one class” structure here (healthy/multiple_diseases/rust/scab), which usually hurts ROC AUC badly. A minimal, high-impact fix is to keep the same feature extraction and (still) logistic regression, but switch to a single multinomial LogisticRegression and output its class probabilities mapped back to the 4 submission columns. Everything else (image reading, features, submission alignment/format, clipping, fallback behavior, and writing `submission.csv`) stays the same.'
- What this solution (achieved 0.58868) has done: 'Your current gap to the target is large (0.578 → 0.9626), so we should improve the same sklearn-based image-feature + (multinomial) logistic regression baseline without changing its overall approach. The weakest link is the hand-crafted feature extractor: it lacks robust texture information and is sensitive to lighting; adding a few cheap, deterministic texture/color-invariant features (HSV stats + simple gradient orientation histogram) typically boosts ROC AUC materially while keeping the same model family and training semantics. I also add class-balancing in LogisticRegression (still the same algorithm/solver) to reduce the impact of label imbalance, and keep the exact same submission alignment/writing behavior. Everything still runs end-to-end and always writes a valid `submission.csv`.'
- What this solution (achieved 0.62499) has done: 'Your current score (0.58868) is far below the target (0.9626), so we should increase performance with the smallest possible change that preserves your core approach (hand-crafted features + multinomial LogisticRegression). The most likely bottleneck is that the model is trained on raw imbalanced classes without any per-image normalization, so lighting/background variations dominate; we add a simple per-image standardization of the thumbnail pixels (keeping the same features, just normalized) and add a tiny amount of label smoothing at the probability level to improve ROC AUC calibration without changing the model family. We also use stratified cross-validated out-of-fold probability calibration (still LogisticRegression, same solver) by averaging 2 folds’ test probabilities to reduce variance, which usually boosts AUC materially while staying within sklearn and within time. Submission alignment/formatting and fallbacks remain unchanged, and it still always write a valid `submission.csv`.'
- What this solution (achieved 0.58633) has done: 'Your current score (0.62499) is far below the target (0.9626), so we should improve the same core approach (hand-crafted image features + multinomial LogisticRegression) with the smallest high-impact changes. The biggest easy gain without changing the model family is to extract more informative features by using a slightly larger thumbnail (more spatial detail) and adding a simple “green-mask” feature block to focus on leaf tissue (reduces background/lighting noise). I also increase the cross-validation averaging from 2 to 3 folds to reduce variance (same training semantics, just a steadier estimate) while staying within runtime. Everything else (paths, alignment, clipping, submission writing, and fallbacks) stays the same and still always writes a valid `submission.csv`.'
- What this solution (achieved 0.64333) has done: 'Your current score is far below the target, so we should improve the same sklearn multinomial LogisticRegression pipeline by feeding it slightly more informative (but still cheap/deterministic) image features. I keep your training approach, CV averaging, model family/solver, and submission alignment unchanged, but (1) increase the working resize from 128→192 to retain more lesion detail, (2) add a compact HSV thumbnail block alongside the existing RGB thumbnail, and (3) add a simple per-channel histogram feature (captures color distribution differences between diseases). These are minimal feature-extraction changes that typically lift ROC AUC substantially without changing evaluation semantics, and the script still always write a valid `submission.csv`.'
- What this solution (achieved 0.64175) has done: 'Your current score (0.64333) is far below the target (0.96258), so we should improve performance with the smallest change that keeps your existing sklearn multinomial LogisticRegression + feature-extraction approach intact. The highest-leverage minimal fix is to make the classifier less underfit by slightly increasing model capacity (we tune `C` upward and remove `class_weight="balanced"` which often hurts AUC in this single-label setting), while keeping the same CV averaging/training semantics. We also add one cheap, disease-relevant feature block (a small 2D “lesion map” via coarse grayscale grid stats) without changing the overall pipeline style. Everything else (paths, alignment, clipping, label-smoothing, and always writing a valid `submission.csv`) remains unchanged.'
- What this solution (achieved 0.63006) has done: 'Your current score (0.64175) is far below the target (0.96258), so we should make the smallest, safest changes that increase predictive signal without changing the overall approach (hand-crafted deterministic image features + multinomial LogisticRegression + 3-fold averaging). The main low-risk upgrade is to focus the thumbnail features on leaf tissue by masking out background before computing thumbnail RGB/HSV, which usually improves AUC because disease patterns live on the leaf, not the background. To avoid changing your training semantics, we keep the same model, folds, and post-processing, and only (1) add a simple green-based leaf mask and (2) add a few very cheap “masked thumbnail” features plus masked color statistics. Everything else (paths, alignment, clipping, label smoothing, and always writing `submission.csv`) remains unchanged.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"

COMP_DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
SAMPLE_SUB_PATH = os.path.join(COMP_DATA_DIR, "sample_submission.csv")
TEST_CSV_PATH = os.path.join(COMP_DATA_DIR, "test.csv")
TRAIN_CSV_PATH = os.path.join(COMP_DATA_DIR, "train.csv")

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

test_df = pd.read_csv(TEST_CSV_PATH)
test_ids = test_df["image_id"].astype(str)
n_test = len(test_df)

sample_df = pd.read_csv(SAMPLE_SUB_PATH)
assert (
    list(sample_df.columns) == ["image_id"] + TARGET_COLS
), "Unexpected sample_submission format"
assert (
    len(sample_df) == n_test
), "sample_submission and test.csv row counts differ unexpectedly"

print("n_test:", n_test)
print("sample_submission columns:", list(sample_df.columns))



## === cell 2
submissions_all = []


def _is_candidate_submission_csv(fp: str) -> bool:
    try:
        head = pd.read_csv(fp, nrows=5)
    except Exception:
        return False
    if "image_id" not in head.columns:
        return False
    if not all(c in head.columns for c in TARGET_COLS):
        return False
    return True


if os.path.exists(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                fp = os.path.join(dirname, filename)
                if _is_candidate_submission_csv(fp):
                    submissions_all.append(fp)

if len(submissions_all) == 0 and os.path.exists("/kaggle/input"):
    for dirname, _, filenames in os.walk("/kaggle/input"):
        for filename in filenames:
            if not filename.lower().endswith(".csv"):
                continue
            fp = os.path.join(dirname, filename)
            if _is_candidate_submission_csv(fp):
                submissions_all.append(fp)

print(f"Found {len(submissions_all)} candidate submission CSV(s).")
for p in submissions_all[:20]:
    print(p)




## === cell 3
def _load_and_align_submission(fp: str, test_ids: pd.Series) -> pd.DataFrame:
    """
    Load a submission-like CSV and align it to test_ids by image_id.
    Returns a DataFrame aligned to test_ids with columns TARGET_COLS.
    Raises ValueError if it cannot be aligned safely.
    """
    df = pd.read_csv(fp)
    missing = [c for c in ["image_id"] + TARGET_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"File {fp} missing columns: {missing}")

    df = df.loc[:, ["image_id"] + TARGET_COLS].copy()
    df["image_id"] = df["image_id"].astype(str)
    df = df.drop_duplicates(subset="image_id", keep="first")

    aligned = df.set_index("image_id").reindex(test_ids.values)

    n_missing_rows = int(aligned[TARGET_COLS].isna().any(axis=1).sum())
    if n_missing_rows > 0:
        raise ValueError(
            f"File {fp} cannot be aligned to test set (missing {n_missing_rows}/{len(test_ids)} image_id rows)."
        )

    for c in TARGET_COLS:
        aligned[c] = pd.to_numeric(aligned[c], errors="coerce")
    if aligned[TARGET_COLS].isna().any().any():
        raise ValueError(f"File {fp} has non-numeric predictions after alignment.")

    aligned[TARGET_COLS] = aligned[TARGET_COLS].clip(0.0, 1.0)
    return aligned[TARGET_COLS]


def ensemble(submissions_all, sub_idx, weights):
    if len(submissions_all) == 0:
        raise ValueError("No submission files available to ensemble.")
    if len(sub_idx) == 0:
        raise ValueError("sub_idx is empty.")
    if len(sub_idx) != len(weights):
        raise ValueError("sub_idx and weights must have the same length.")
    if any(i < 0 or i >= len(submissions_all) for i in sub_idx):
        raise IndexError(
            f"sub_idx contains out-of-range indices for submissions_all (len={len(submissions_all)})."
        )

    wsum = float(sum(weights))
    if wsum == 0.0:
        raise ValueError("Sum of weights must be non-zero.")
    weights = [float(w) / wsum for w in weights]

    submission_with_weight = None
    used_files = []
    for i, w in zip(sub_idx, weights):
        fp = submissions_all[i]
        preds = _load_and_align_submission(fp, test_ids).values  # shape (n_test, 4)
        if submission_with_weight is None:
            submission_with_weight = preds * w
        else:
            submission_with_weight += preds * w
        used_files.append(fp)

    print("Ensembled files:")
    for u in used_files:
        print(" -", u)

    return submission_with_weight




## === cell 4
def make_submission_file(submission_avg):
    submission_df = pd.read_csv(SAMPLE_SUB_PATH)

    if submission_avg is not None:
        if submission_avg.shape != (len(submission_df), len(TARGET_COLS)):
            raise ValueError(
                f"submission_avg has shape {submission_avg.shape}, expected {(len(submission_df), len(TARGET_COLS))}."
            )
        submission_df.loc[:, TARGET_COLS] = submission_avg

    submission_df["image_id"] = submission_df["image_id"].astype(str)
    for c in TARGET_COLS:
        submission_df[c] = (
            pd.to_numeric(submission_df[c], errors="coerce").fillna(0.25).clip(0.0, 1.0)
        )

    submission_df.to_csv("submission.csv", index=False)
    return submission_df




## === cell 5
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def _find_images_dir(comp_dir: str) -> str:
    cand = os.path.join(comp_dir, "images")
    if os.path.isdir(cand):
        return cand
    for p in [
        "/kaggle/input/plant-pathology-2020-fgvc7/images",
        "/kaggle/data/plant-pathology-2020-fgvc7/images",
        "/kaggle/input/images",
        "/kaggle/data/images",
    ]:
        if os.path.isdir(p):
            return p
    return cand


IMAGES_DIR = _find_images_dir(COMP_DATA_DIR)
print("IMAGES_DIR:", IMAGES_DIR, "exists:", os.path.isdir(IMAGES_DIR))


def _read_image_rgb(path: str, resize_to=(192, 192)):
    try:
        from PIL import Image

        img = Image.open(path).convert("RGB")
        if resize_to is not None:
            img = img.resize(resize_to, resample=Image.BILINEAR)
        return np.asarray(img)
    except Exception:
        try:
            import matplotlib.image as mpimg

            img = mpimg.imread(path)
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            if img.shape[-1] == 4:
                img = img[..., :3]
            if img.dtype != np.uint8:
                img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            return img
        except Exception as e:
            raise RuntimeError(f"Could not read image {path}: {e}")


def _rgb_to_hsv_np(x):  # x in [0,1], shape (H,W,3)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    cmax = np.max(x, axis=-1)
    cmin = np.min(x, axis=-1)
    delta = cmax - cmin

    h = np.zeros_like(cmax, dtype=np.float32)
    mask = delta > 1e-8
    rc = (((g - b) / (delta + 1e-12)) % 6.0).astype(np.float32)
    gc = (((b - r) / (delta + 1e-12)) + 2.0).astype(np.float32)
    bc = (((r - g) / (delta + 1e-12)) + 4.0).astype(np.float32)

    rmask = mask & (cmax == r)
    gmask = mask & (cmax == g)
    bmask = mask & (cmax == b)
    h[rmask] = rc[rmask]
    h[gmask] = gc[gmask]
    h[bmask] = bc[bmask]
    h = (h / 6.0).astype(np.float32)  # [0,1)

    s = np.zeros_like(cmax, dtype=np.float32)
    nonzero = cmax > 1e-8
    s[nonzero] = (delta[nonzero] / (cmax[nonzero] + 1e-12)).astype(np.float32)

    v = cmax.astype(np.float32)
    return h, s, v


def _image_features(rgb: np.ndarray, thumb_size=32, hist_bins=8) -> np.ndarray:
    x = rgb.astype(np.float32) / 255.0

    means = x.reshape(-1, 3).mean(axis=0)
    stds = x.reshape(-1, 3).std(axis=0)

    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    gr = (g - r).reshape(-1)
    br = (b - r).reshape(-1)

    y = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)

    dx = y[:, 1:] - y[:, :-1]
    dy = y[1:, :] - y[:-1, :]
    edge_energy = float(np.mean(np.abs(dx)) + np.mean(np.abs(dy)))

    lap = -4.0 * y[1:-1, 1:-1] + y[1:-1, :-2] + y[1:-1, 2:] + y[:-2, 1:-1] + y[2:, 1:-1]
    lap_var = float(np.var(lap)) if lap.size > 0 else 0.0

    h0, w0 = y.shape
    ch0, ch1 = int(0.25 * h0), int(0.75 * h0)
    cw0, cw1 = int(0.25 * w0), int(0.75 * w0)
    center = x[ch0:ch1, cw0:cw1, :].reshape(-1, 3)
    border_mask = np.ones((h0, w0), dtype=bool)
    border_mask[ch0:ch1, cw0:cw1] = False
    border = x[border_mask].reshape(-1, 3)
    if center.size == 0 or border.size == 0:
        center_border = np.zeros(3, dtype=np.float32)
    else:
        center_border = (center.mean(axis=0) - border.mean(axis=0)).astype(np.float32)

    hh, ss, vv = _rgb_to_hsv_np(x)
    hsv_stats = np.array(
        [
            float(hh.mean()),
            float(hh.std()),
            float(ss.mean()),
            float(ss.std()),
            float(vv.mean()),
            float(vv.std()),
        ],
        dtype=np.float32,
    )

    gx = np.zeros_like(y, dtype=np.float32)
    gy = np.zeros_like(y, dtype=np.float32)
    gx[:, 1:-1] = (y[:, 2:] - y[:, :-2]) * 0.5
    gy[1:-1, :] = (y[2:, :] - y[:-2, :]) * 0.5
    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    ang = np.arctan2(gy, gx).astype(np.float32)  # [-pi, pi]
    ang01 = (ang + np.pi) / (2.0 * np.pi)  # [0,1]
    bins = 8
    idx = np.minimum((ang01 * bins).astype(np.int32), bins - 1)
    hist = np.zeros((bins,), dtype=np.float32)
    for bi in range(bins):
        m = mag[idx == bi]
        hist[bi] = float(m.mean()) if m.size else 0.0

    green_mask = (g > r) & (g > b) & (ss > 0.2) & (vv > 0.15)
    gm = green_mask.reshape(-1)
    frac_green = float(gm.mean())
    if gm.any():
        y_g = y.reshape(-1)[gm]
        s_g = ss.reshape(-1)[gm]
        v_g = vv.reshape(-1)[gm]
        green_stats = np.array(
            [
                frac_green,
                float(y_g.mean()),
                float(y_g.std()),
                float(s_g.mean()),
                float(v_g.mean()),
            ],
            dtype=np.float32,
        )
        x_flat = x.reshape(-1, 3)[gm]
        masked_means = x_flat.mean(axis=0).astype(np.float32)
        masked_stds = x_flat.std(axis=0).astype(np.float32)
        masked_hsv = np.stack([hh, ss, vv], axis=-1).reshape(-1, 3)[gm]
        masked_hsv_stats = np.array(
            [
                float(masked_hsv[:, 0].mean()),
                float(masked_hsv[:, 0].std()),
                float(masked_hsv[:, 1].mean()),
                float(masked_hsv[:, 1].std()),
                float(masked_hsv[:, 2].mean()),
                float(masked_hsv[:, 2].std()),
            ],
            dtype=np.float32,
        )
    else:
        green_stats = np.array([frac_green, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        masked_means = np.zeros(3, dtype=np.float32)
        masked_stds = np.zeros(3, dtype=np.float32)
        masked_hsv_stats = np.zeros(6, dtype=np.float32)

    hR, _ = np.histogram(r.reshape(-1), bins=hist_bins, range=(0.0, 1.0), density=True)
    hG, _ = np.histogram(g.reshape(-1), bins=hist_bins, range=(0.0, 1.0), density=True)
    hB, _ = np.histogram(b.reshape(-1), bins=hist_bins, range=(0.0, 1.0), density=True)
    hS, _ = np.histogram(ss.reshape(-1), bins=hist_bins, range=(0.0, 1.0), density=True)
    hV, _ = np.histogram(vv.reshape(-1), bins=hist_bins, range=(0.0, 1.0), density=True)
    color_hists = np.concatenate([hR, hG, hB, hS, hV]).astype(np.float32)

    grid = 12
    hh2, ww2 = y.shape
    gh = max(1, hh2 // grid)
    gw = max(1, ww2 // grid)
    y_crop = y[: gh * grid, : gw * grid]
    y_cells = y_crop.reshape(grid, gh, grid, gw).mean(axis=(1, 3))  # (grid, grid)
    y_cells_flat = y_cells.reshape(-1).astype(np.float32)
    y_cells_mu = float(y_cells_flat.mean())
    y_cells_sd = float(y_cells_flat.std())
    y_cells_flat = (y_cells_flat - y_cells_mu) / (y_cells_sd + 1e-6)

    base = np.concatenate(
        [
            means,
            stds,
            [gr.mean(), gr.std(), br.mean(), br.std()],
            [edge_energy, lap_var],
            center_border,
            hsv_stats,
            hist,
            green_stats,
            masked_means,
            masked_stds,
            masked_hsv_stats,
            color_hists,
            y_cells_flat,
        ]
    ).astype(np.float32)

    h2, w2, _ = x.shape
    th = min(thumb_size, h2)
    tw = min(thumb_size, w2)
    ch = (h2 // th) * th
    cw = (w2 // tw) * tw

    yy = x[:ch, :cw, :]
    yy = yy.reshape(th, ch // th, tw, cw // tw, 3).mean(axis=(1, 3))  # (th, tw, 3)
    thumb_rgb = yy.reshape(-1).astype(np.float32)
    tmu = float(thumb_rgb.mean())
    tsd = float(thumb_rgb.std())
    thumb_rgb = (thumb_rgb - tmu) / (tsd + 1e-6)

    hsv_img = np.stack([hh, ss, vv], axis=-1).astype(np.float32)
    yyh = hsv_img[:ch, :cw, :]
    yyh = yyh.reshape(th, ch // th, tw, cw // tw, 3).mean(axis=(1, 3))
    thumb_hsv = yyh.reshape(-1).astype(np.float32)
    hmu = float(thumb_hsv.mean())
    hsd = float(thumb_hsv.std())
    thumb_hsv = (thumb_hsv - hmu) / (hsd + 1e-6)

    gm2 = green_mask[:ch, :cw].astype(np.float32)
    gm2 = gm2.reshape(th, ch // th, tw, cw // tw).mean(axis=(1, 3))  # (th, tw) in [0,1]
    gm2 = (gm2 > 0.25).astype(np.float32)[..., None]  # hard-ish mask per cell
    yy_masked = yy * gm2
    thumb_rgb_masked = yy_masked.reshape(-1).astype(np.float32)
    if float(gm2.mean()) > 1e-6:
        mtmu = float(thumb_rgb_masked.mean())
        mtsd = float(thumb_rgb_masked.std())
        thumb_rgb_masked = (thumb_rgb_masked - mtmu) / (mtsd + 1e-6)
    else:
        thumb_rgb_masked = thumb_rgb.copy()

    yyh_masked = yyh * gm2
    thumb_hsv_masked = yyh_masked.reshape(-1).astype(np.float32)
    if float(gm2.mean()) > 1e-6:
        mhmu = float(thumb_hsv_masked.mean())
        mhsd = float(thumb_hsv_masked.std())
        thumb_hsv_masked = (thumb_hsv_masked - mhmu) / (mhsd + 1e-6)
    else:
        thumb_hsv_masked = thumb_hsv.copy()

    feats = np.concatenate(
        [base, thumb_rgb, thumb_hsv, thumb_rgb_masked, thumb_hsv_masked]
    ).astype(np.float32)
    return feats


def _build_features(image_ids: pd.Series, images_dir: str, thumb_size=32) -> np.ndarray:
    image_ids_arr = image_ids.astype(str).values
    if len(image_ids_arr) == 0:
        return np.zeros((0, 10), dtype=np.float32)

    def _resolve_fp(img_id: str) -> str:
        fp = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(fp):
            g = glob.glob(os.path.join(images_dir, img_id + ".*"))
            if len(g) > 0:
                fp = g[0]
        return fp

    fp0 = _resolve_fp(image_ids_arr[0])
    rgb0 = _read_image_rgb(fp0, resize_to=(192, 192))
    f0 = _image_features(rgb0, thumb_size=thumb_size)
    d = int(f0.shape[0])

    feats = np.zeros((len(image_ids_arr), d), dtype=np.float32)
    feats[0] = f0
    for i in range(1, len(image_ids_arr)):
        fp = _resolve_fp(image_ids_arr[i])
        rgb = _read_image_rgb(fp, resize_to=(192, 192))
        feats[i] = _image_features(rgb, thumb_size=thumb_size)
    return feats


def train_and_predict_baseline(
    train_csv_path: str, test_ids: pd.Series, images_dir: str
) -> np.ndarray:
    train_df = pd.read_csv(train_csv_path)
    train_df["image_id"] = train_df["image_id"].astype(str)

    X_train = _build_features(train_df["image_id"], images_dir, thumb_size=32)
    y_train_onehot = train_df[TARGET_COLS].astype(int).values
    y_class = np.argmax(y_train_onehot, axis=1).astype(int)

    X_test = _build_features(test_ids, images_dir, thumb_size=32)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    proba_test_accum = np.zeros((len(test_ids), len(TARGET_COLS)), dtype=np.float32)
    n_folds = 0
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_class), start=1):
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=3000,
                        C=6.0,
                        multi_class="multinomial",
                        class_weight=None,
                    ),
                ),
            ]
        )
        clf.fit(X_train[tr_idx], y_class[tr_idx])

        proba = clf.predict_proba(X_test)  # (n_test, 4) with classes_ order
        proba_full = np.zeros((len(test_ids), len(TARGET_COLS)), dtype=np.float32)
        for k, cls in enumerate(clf.named_steps["lr"].classes_):
            proba_full[:, int(cls)] = proba[:, k].astype(np.float32)

        proba_test_accum += proba_full
        n_folds += 1
        print(f"Fold {fold}: done")

    proba_full = proba_test_accum / float(max(n_folds, 1))

    eps = 1e-3
    proba_full = (1.0 - 4.0 * eps) * proba_full + eps
    proba_full = np.clip(proba_full, 0.0, 1.0)
    return proba_full




## === cell 6
submission_avg = None

try:
    if os.path.exists(TRAIN_CSV_PATH) and os.path.isdir(IMAGES_DIR):
        submission_avg = train_and_predict_baseline(
            TRAIN_CSV_PATH, test_ids, IMAGES_DIR
        )
        print("Baseline image model predictions generated:", submission_avg.shape)
except Exception as e:
    print("Baseline image model failed, falling back. Reason:", repr(e))
    submission_avg = None

if submission_avg is None and len(submissions_all) >= 2:
    try:
        submission_avg = ensemble(submissions_all, [0, 1], [0.4, 0.6])
    except Exception as e:
        print("Ensembling first two candidates failed, falling back. Reason:", repr(e))
        submission_avg = None

if submission_avg is None and len(submissions_all) >= 1:
    for idx in range(len(submissions_all)):
        try:
            submission_avg = ensemble(submissions_all, [idx], [1.0])
            break
        except Exception as e:
            print(
                f"Single-file candidate at index {idx} failed alignment, skipping. Reason:",
                repr(e),
            )
            submission_avg = None

if submission_avg is None:
    submission_avg = pd.DataFrame(0.25, index=range(n_test), columns=TARGET_COLS).values

sub_df = make_submission_file(submission_avg)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print("Columns:", list(sub_df.columns))
print("submission.csv exists:", os.path.exists("submission.csv"))
