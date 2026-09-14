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
numpy==1.26.4
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

0.9697366773573868

# 6. Current score

0.58966

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.49496) has done: 'I fix the runtime error by removing the hardcoded `../input/plantpathology/*.csv` dependencies (those files don’t exist in your environment) and instead generate predictions directly from the provided train/test images. To keep changes minimal while ensuring a solid score, I use a simple, deterministic scikit-learn pipeline: load images, extract small grayscale features, train a multi-output logistic regression, and predict probabilities for each label. I also ensure the submission columns and `image_id` ordering exactly match `sample_submission.csv`, and write `submission.csv` with the required suffix.'
- What this solution (achieved 0.53083) has done: 'Your current score is far below the target, so we should improve feature quality and model capacity while keeping the same core approach (deterministic image features → scikit-learn multi-output logistic regression → probability submission). The biggest issue is that 64×64 grayscale flattening discards most discriminative color/texture cues, so we minimally extend the feature extractor to include (1) color information, (2) simple edge/gradient cues, and (3) a few global statistics; this remains lightweight and deterministic. We also apply class-balanced training per label (via `class_weight="balanced"` inside each logistic regression) to handle label imbalance, which typically improves mean ROC AUC without changing evaluation semantics. Submission alignment still strictly follow `sample_submission.csv` and `test.csv` ordering and write `submission.csv`.'
- What this solution (achieved 0.56359) has done: 'Your current score (0.53083) is far below the target (0.9697), so we should improve discriminative power while keeping the same “handcrafted image features → standardize → multi-output logistic regression” core pipeline. The minimal high-impact change is to make the features more informative without changing the model family: add compact color histograms + simple color indices (helps separate rust/scab patterns) while keeping your existing downsampled RGB/gray/edge features. I also switch the logistic regression solver to a multinomial-capable one with the same linear model (still logistic regression) to fit probabilities more stably on high-dimensional standardized features. Submission writing/alignment stays identical and still produces `submission.csv`.'
- What this solution (achieved 0.56532) has done: 'Your score is far below the target, so we should increase performance while keeping the same core pipeline (handcrafted deterministic features → StandardScaler → MultiOutputClassifier(LogisticRegression) → probability CSV). The smallest high-impact change is to fix a feature scaling issue: mixing huge flattened pixel vectors with small histogram/stat vectors causes the StandardScaler and LR to underuse the informative small blocks; we rebalance by (a) switching to `StandardScaler(with_mean=False)` (sparse-safe behavior that avoids mean-centering huge dense vectors) and (b) applying lightweight, deterministic block-wise weights inside `extract_features` so histograms/stats/indices aren’t drowned out. We also set `n_jobs=-1` inside `MultiOutputClassifier` to speed training without changing semantics, staying well within constraints. Submission writing and column/order alignment remain identical and still produce `submission.csv`.'
- What this solution (achieved 0.56766) has done: 'Your current score (0.56532) is far below the target (0.9697), so we need a real lift while keeping the same core pipeline (handcrafted deterministic features → scaling → multi-output logistic regression). The smallest high-impact change without changing the model family is to add a compact, rotation/scale-robust texture descriptor (uniform LBP histogram) plus a small HSV histogram, which helps separate rust/scab/healthy patterns much better than raw downsampled pixels alone. To avoid distorting the training approach, we keep LogisticRegression+MultiOutputClassifier intact and only lightly tune regularization/iteration to better fit the expanded feature space. Submission writing/order/columns remain identical and still produce a valid `submission.csv`.'
- What this solution (achieved 0.57391) has done: 'Your current score (0.56766) is far below the target (0.9697), so we should increase performance with the smallest changes that keep the same core pipeline (handcrafted deterministic features → scaling → multi-output logistic regression → probability submission). The biggest likely issue is that the current feature vector is dominated by raw pixel blocks, while the more informative histogram/texture blocks may still be under-leveraged; we lightly rebalance block weights and add a tiny amount of multi-scale texture (LBP at a second radius) without changing the model family or training approach. We also switch the scaler to `StandardScaler()` (mean-centering) for dense features, which typically improves linear model conditioning and AUC for these handcrafted vectors. Submission format, column order, and image_id alignment remain exactly as required and still write `submission.csv`.'
- What this solution (achieved 0.59027) has done: 'Your score is far below the target, so we should improve discriminative power while keeping the same “handcrafted deterministic features → StandardScaler → MultiOutput LogisticRegression probabilities” core. The largest likely gap is that the current linear model is being asked to separate classes with mostly low-level pixel/hist cues; a minimal, compatible boost is to add compact, rotation-tolerant local gradient-structure via a small HOG-like orientation histogram (no new dependencies) and to compute edges from a lightly blurred grayscale to reduce noise. These additions keep the same training loop/model family, only extend the feature vector, and typically improve ROC AUC for texture-driven classes like rust/scab. Submission writing/ordering remains identical and still writes a valid `submission.csv`.'
- What this solution (achieved 0.58722) has done: 'Your current score (0.59027) is far below the target (0.9697), so we should improve performance while keeping the same core pipeline: deterministic handcrafted features → StandardScaler → MultiOutput LogisticRegression → probability submission. The minimal high-impact bug-like issue here is a feature inconsistency: LBP is computed from the unblurred gray while HOG/edges use blurred gray, and your HOG currently discards sign (0..π), which loses discriminative structure; we make these gradient/texture features consistent and slightly richer without changing the model family or training loop. Concretely, we (1) compute LBP on the same blurred grayscale used for HOG/edges, (2) switch HOG orientation to 0..2π with 18 bins (still compact) while keeping the same per-cell histogram logic, and (3) add very lightweight L2-Hys style clipping/renorm per cell histogram (still deterministic) to reduce sensitivity to illumination and help AUC. Submission writing/order/columns remain identical and still write `submission.csv`.'
- What this solution (achieved 0.58895) has done: 'Your current score is far below the target, so we should improve discriminative power while keeping the same pipeline (handcrafted deterministic features → StandardScaler → MultiOutput LogisticRegression). The smallest high-impact change here is to add a compact “green vs red” opponent-color histogram (on (G−R)/(G+R)) which is strongly correlated with rust/scab coloration but still fits your existing feature-extraction approach and linear model. To keep semantics identical and stable, we only extend the feature vector and keep the same model, training loop, and submission alignment. This should move ROC AUC upward without introducing new dependencies or changing I/O paths.'
- What this solution (achieved 0.58966) has done: 'Your current score (0.58895) is far below the target (0.9697), so we should improve discrimination while keeping the exact same overall pipeline (deterministic handcrafted features → StandardScaler → MultiOutput LogisticRegression → clipped probabilities → submission.csv). The smallest high-impact change is to add compact, rotation-tolerant shape information via a simple region-proportion feature computed from the edge map at a few thresholds (captures how “spotty” vs “smooth” a leaf is), which is often very predictive for scab/rust. To avoid changing training semantics, we keep the same model/solver/loss and only extend the feature vector; we also standardize the edge magnitude before thresholding to make these proportions stable across illumination. Submission formatting and `image_id` alignment remain identical.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
BASE_PATH = "/kaggle/data/plant-pathology-2020-fgvc7"
if not os.path.exists(BASE_PATH):
    BASE_PATH = "/kaggle/input/plant-pathology-2020-fgvc7"

TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_CSV = os.path.join(BASE_PATH, "test.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_PATH, "sample_submission.csv")
IMAGES_DIR = os.path.join(BASE_PATH, "images")

print("BASE_PATH:", BASE_PATH)
print("Exists TRAIN_CSV:", os.path.exists(TRAIN_CSV))
print("Exists TEST_CSV:", os.path.exists(TEST_CSV))
print("Exists SAMPLE_SUB_CSV:", os.path.exists(SAMPLE_SUB_CSV))
print("Exists IMAGES_DIR:", os.path.exists(IMAGES_DIR))



## === cell 2
from PIL import Image

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

target_cols = [c for c in sample_sub.columns if c != "image_id"]
assert target_cols == [
    "healthy",
    "multiple_diseases",
    "rust",
    "scab",
], f"Unexpected target columns: {target_cols}"


def _resolve_image_path(image_id: str) -> str:
    return os.path.join(IMAGES_DIR, f"{image_id}.jpg")


def _hist_channel(ch_0_1: np.ndarray, bins: int = 16) -> np.ndarray:
    h, _ = np.histogram(ch_0_1, bins=bins, range=(0.0, 1.0), density=True)
    return h.astype(np.float32)


def _hist_range(x: np.ndarray, bins: int, vmin: float, vmax: float) -> np.ndarray:
    h, _ = np.histogram(x, bins=bins, range=(vmin, vmax), density=True)
    return h.astype(np.float32)


def _lbp_uniform_riu2(gray_0_1: np.ndarray, radius: int = 1) -> np.ndarray:
    """
    Uniform LBP (P=8, R=radius), rotation-invariant (riu2): bins 0..8 for uniform patterns,
    bin 9 for non-uniform. Produces a 10-bin histogram (density).
    """
    g = (gray_0_1 * 255.0).astype(np.float32)
    H, W = g.shape
    r = int(radius)
    if H < 2 * r + 1 or W < 2 * r + 1:
        out = np.zeros(10, dtype=np.float32)
        out[-1] = 1.0
        return out

    c = g[r : H - r, r : W - r]
    n0 = g[: H - 2 * r, : W - 2 * r]
    n1 = g[: H - 2 * r, r : W - r]
    n2 = g[: H - 2 * r, 2 * r :]
    n3 = g[r : H - r, 2 * r :]
    n4 = g[2 * r :, 2 * r :]
    n5 = g[2 * r :, r : W - r]
    n6 = g[2 * r :, : W - 2 * r]
    n7 = g[r : H - r, : W - 2 * r]

    bits = [
        (n0 >= c).astype(np.uint8),
        (n1 >= c).astype(np.uint8),
        (n2 >= c).astype(np.uint8),
        (n3 >= c).astype(np.uint8),
        (n4 >= c).astype(np.uint8),
        (n5 >= c).astype(np.uint8),
        (n6 >= c).astype(np.uint8),
        (n7 >= c).astype(np.uint8),
    ]
    ones = np.zeros_like(bits[0], dtype=np.uint8)
    for b in bits:
        ones += b

    transitions = np.zeros_like(bits[0], dtype=np.uint8)
    for i in range(8):
        transitions += (bits[i] ^ bits[(i + 1) % 8]).astype(np.uint8)

    code = np.where(transitions <= 2, ones, 9).astype(np.uint8)
    hist = np.bincount(code.ravel(), minlength=10).astype(np.float32)
    hist /= hist.sum() + 1e-12
    return hist


def _gaussian_blur_3x3(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    p = np.pad(g, ((1, 1), (1, 1)), mode="reflect")
    out = (
        1.0 * p[:-2, :-2]
        + 2.0 * p[:-2, 1:-1]
        + 1.0 * p[:-2, 2:]
        + 2.0 * p[1:-1, :-2]
        + 4.0 * p[1:-1, 1:-1]
        + 2.0 * p[1:-1, 2:]
        + 1.0 * p[2:, :-2]
        + 2.0 * p[2:, 1:-1]
        + 1.0 * p[2:, 2:]
    ) / 16.0
    return out.astype(np.float32)


def _hog_orientation_histograms(
    gray_0_1: np.ndarray,
    cell_size: int = 12,
    bins: int = 18,
    clip: float = 0.2,
) -> np.ndarray:
    """
    Signed-orientation HOG (0..2π) with mild per-cell L2-Hys clipping+renorm.
    Deterministic, compact, and keeps the same handcrafted-feature approach.
    """
    g = gray_0_1.astype(np.float32)
    gx = np.zeros_like(g, dtype=np.float32)
    gy = np.zeros_like(g, dtype=np.float32)
    gx[:, 1:] = g[:, 1:] - g[:, :-1]
    gy[1:, :] = g[1:, :] - g[:-1, :]

    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    ang = np.arctan2(gy, gx).astype(np.float32)  # [-pi, pi]
    ang = np.where(ang < 0, ang + 2.0 * np.pi, ang)  # [0, 2pi)
    bin_f = (ang / (2.0 * np.pi)) * bins
    b0 = np.floor(bin_f).astype(np.int32)
    b0 = np.clip(b0, 0, bins - 1)

    H, W = g.shape
    ncy = H // cell_size
    ncx = W // cell_size
    if ncy == 0 or ncx == 0:
        return np.zeros(bins, dtype=np.float32)

    feats = []
    for cy in range(ncy):
        y0 = cy * cell_size
        y1 = y0 + cell_size
        for cx in range(ncx):
            x0 = cx * cell_size
            x1 = x0 + cell_size
            b = b0[y0:y1, x0:x1].ravel()
            m = mag[y0:y1, x0:x1].ravel()
            h = np.bincount(b, weights=m, minlength=bins).astype(np.float32)

            h /= np.sqrt((h * h).sum()) + 1e-12
            if clip is not None and clip > 0:
                h = np.minimum(h, np.float32(clip))
                h /= np.sqrt((h * h).sum()) + 1e-12

            feats.append(h)
    return np.concatenate(feats, axis=0).astype(np.float32)


def extract_features(
    image_ids,
    size=(96, 96),
    hist_bins=16,
    hsv_bins=12,
    opp_bins=24,
    w_rgb=0.22,
    w_gray=0.22,
    w_edge=0.28,
    w_stats=3.0,
    w_hist=4.0,
    w_idx=3.5,
    w_hsv=3.0,
    w_lbp_r1=5.0,
    w_lbp_r2=4.0,
    w_hog=4.5,
    w_opp=4.0,
    edge_prop_thresholds=(0.8, 1.2, 1.6, 2.0),
    w_edge_props=6.0,
):
    """
    Handcrafted deterministic features:
      - Downsampled RGB + gray + edges
      - Global stats
      - RGB/gray histograms + HSV histograms
      - Color indices (ExG/ExR stats)
      - LBP (r=1,2) on blurred gray
      - Signed HOG on blurred gray
      - Opponent-color histogram of (G-R)/(G+R)
      - NEW: edge proportion features at multiple standardized thresholds
    Core pipeline remains: handcrafted features -> scaling -> linear multi-output LR.
    """
    n = len(image_ids)
    H, W = size

    d_rgb = 3 * H * W
    d_gray = H * W
    d_edge = H * W
    d_stats = 12
    d_hist = (3 + 1) * hist_bins
    d_idx_stats = 8
    d_hsv = 3 * hsv_bins
    d_lbp = 10

    cell_size = 12
    hog_bins = 18
    hog_cells_y = H // cell_size
    hog_cells_x = W // cell_size
    d_hog = hog_cells_y * hog_cells_x * hog_bins

    d_opp = opp_bins

    d_edge_props = len(edge_prop_thresholds)

    d = (
        d_rgb
        + d_gray
        + d_edge
        + d_stats
        + d_hist
        + d_idx_stats
        + d_hsv
        + d_lbp
        + d_lbp
        + d_hog
        + d_opp
        + d_edge_props
    )
    X = np.empty((n, d), dtype=np.float32)

    for i, img_id in enumerate(image_ids):
        p = _resolve_image_path(img_id)
        with Image.open(p) as im:
            im_rgb = im.convert("RGB").resize((W, H), Image.BILINEAR)
            arr_rgb = np.asarray(im_rgb, dtype=np.float32) / 255.0

        rgb_flat = arr_rgb.reshape(-1) * np.float32(w_rgb)

        gray = (
            0.299 * arr_rgb[..., 0] + 0.587 * arr_rgb[..., 1] + 0.114 * arr_rgb[..., 2]
        ).astype(np.float32)
        gray_flat = gray.reshape(-1) * np.float32(w_gray)

        gray_blur = _gaussian_blur_3x3(gray)

        gx = np.zeros_like(gray_blur, dtype=np.float32)
        gy = np.zeros_like(gray_blur, dtype=np.float32)
        gx[:, 1:] = gray_blur[:, 1:] - gray_blur[:, :-1]
        gy[1:, :] = gray_blur[1:, :] - gray_blur[:-1, :]
        edge = np.sqrt(gx * gx + gy * gy).astype(np.float32)
        edge_flat = edge.reshape(-1) * np.float32(w_edge)

        e_mean = float(edge.mean())
        e_std = float(edge.std()) + 1e-6
        edge_z = (edge - e_mean) / e_std
        edge_props = np.asarray(
            [float((edge_z > t).mean()) for t in edge_prop_thresholds],
            dtype=np.float32,
        ) * np.float32(w_edge_props)

        stats = []
        for c in range(3):
            ch = arr_rgb[..., c]
            stats.extend(
                [float(ch.mean()), float(ch.std()), float(ch.min()), float(ch.max())]
            )
        stats = np.asarray(stats, dtype=np.float32) * np.float32(w_stats)

        h_r = _hist_channel(arr_rgb[..., 0], bins=hist_bins)
        h_g = _hist_channel(arr_rgb[..., 1], bins=hist_bins)
        h_b = _hist_channel(arr_rgb[..., 2], bins=hist_bins)
        h_gray = _hist_channel(gray, bins=hist_bins)
        hist = np.concatenate([h_r, h_g, h_b, h_gray], axis=0) * np.float32(w_hist)

        R, G, B = arr_rgb[..., 0], arr_rgb[..., 1], arr_rgb[..., 2]
        exg = (2.0 * G - R - B).astype(np.float32)
        exr = (1.4 * R - G).astype(np.float32)
        idx_stats = np.asarray(
            [
                exg.mean(),
                exg.std(),
                exg.min(),
                exg.max(),
                exr.mean(),
                exr.std(),
                exr.min(),
                exr.max(),
            ],
            dtype=np.float32,
        )
        idx_stats = idx_stats * np.float32(w_idx)

        denom = (G + R + 1e-6).astype(np.float32)
        opp = ((G - R) / denom).astype(np.float32)  # roughly in [-1, 1]
        opp_hist = _hist_range(opp, bins=opp_bins, vmin=-1.0, vmax=1.0) * np.float32(
            w_opp
        )

        with Image.open(p) as im:
            im_hsv = im.convert("HSV").resize((W, H), Image.BILINEAR)
            arr_hsv = np.asarray(im_hsv, dtype=np.float32)
            Hc = (arr_hsv[..., 0] / 255.0).astype(np.float32)
            Sc = (arr_hsv[..., 1] / 255.0).astype(np.float32)
            Vc = (arr_hsv[..., 2] / 255.0).astype(np.float32)

        h_H = _hist_channel(Hc, bins=hsv_bins)
        h_S = _hist_channel(Sc, bins=hsv_bins)
        h_V = _hist_channel(Vc, bins=hsv_bins)
        hsv_hist = np.concatenate([h_H, h_S, h_V], axis=0).astype(
            np.float32
        ) * np.float32(w_hsv)

        lbp_r1 = _lbp_uniform_riu2(gray_blur, radius=1).astype(np.float32) * np.float32(
            w_lbp_r1
        )
        lbp_r2 = _lbp_uniform_riu2(gray_blur, radius=2).astype(np.float32) * np.float32(
            w_lbp_r2
        )

        hog = _hog_orientation_histograms(
            gray_blur, cell_size=cell_size, bins=hog_bins, clip=0.2
        ) * np.float32(w_hog)

        X[i] = np.concatenate(
            [
                rgb_flat,
                gray_flat,
                edge_flat,
                stats,
                hist,
                idx_stats,
                hsv_hist,
                lbp_r1,
                lbp_r2,
                hog,
                opp_hist,
                edge_props,
            ],
            axis=0,
        )

    return X


X_train = extract_features(train_df["image_id"].values)
y_train = train_df[target_cols].values.astype(int)
X_test = extract_features(test_df["image_id"].values)

print("X_train:", X_train.shape, "y_train:", y_train.shape, "X_test:", X_test.shape)



## === cell 3
base_clf = LogisticRegression(
    solver="lbfgs",
    C=3.0,
    max_iter=8000,
    random_state=42,
    class_weight="balanced",
)

model = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", MultiOutputClassifier(base_clf, n_jobs=-1)),
    ]
)

model.fit(X_train, y_train)

probas = model.predict_proba(X_test)
pred = np.column_stack([p[:, 1] for p in probas]).astype(np.float32)

eps = 1e-6
pred = np.clip(pred, eps, 1.0 - eps)

print("pred:", pred.shape, "min/max:", float(pred.min()), float(pred.max()))



## === cell 4
sub = test_df[["image_id"]].merge(sample_sub, on="image_id", how="left")
sub[target_cols] = pred
sub = sub[["image_id"] + target_cols]

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print(sub.head())
