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

0.9675421146508574

# 6. Current score

0.82226

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the pipeline by removing the dependency on missing external Kaggle dataset submission files (the `../input/plantpathology/*.csv` ensemble), which is what currently causes the `FileNotFoundError` and prevents `dsub` from being defined. To keep the solution valid end-to-end in this environment, I instead generate a safe baseline submission directly from `sample_submission.csv`, filling each class with a constant prior computed from the training label means (score-neutral vs. uniform and typically slightly better calibrated). I also make the input paths robust to both provided folder layouts (`/kaggle/input/...` and `/kaggle/data/...`). Finally, the script always write `submission.csv` with the exact required columns.'
- What this solution (achieved 0.65733) has done: 'Your current 0.5 score comes from predicting constants, which yields ~0.5 ROC AUC because the ranking contains no information. With the very limited installed packages (no deep learning / CV libs), the smallest legitimate improvement is to extract simple image-based features (RGB channel statistics) using only the Python standard library (`PIL`) and then train a lightweight multi-output classifier (one-vs-rest logistic regression) from scikit-learn to produce non-constant probabilities. This preserves the “simple baseline” spirit while creating meaningful per-image ranking signals that should move the score substantially upward toward your 0.9675 target without changing the evaluation semantics. The script still writes a valid `submission.csv` with the exact required columns and ordering.'
- What this solution (achieved 0.64308) has done: 'You’re far below the target (0.657 vs 0.9675), so we should improve ranking signal while keeping the same “simple image stats + OneVsRest LogisticRegression” core logic. The biggest low-risk gain is to expand the feature vector beyond global RGB mean/std to include coarse spatial information (same stats computed on a 2×2 grid) plus simple color ratios; this still uses PIL+NumPy only and keeps training identical. I also add stable, deterministic image preprocessing and class-balancing in LogisticRegression (same model family/solver) to reduce bias from class imbalance and typically improve ROC-AUC without changing evaluation semantics. The pipeline still run end-to-end and write a valid `submission.csv` with the required columns and ordering.'
- What this solution (achieved 0.70427) has done: 'Your current gap to the target is large (0.643 → 0.9675), so we should improve ranking signal while preserving the same “simple RGB stats → scaled features → OneVsRest LogisticRegression” core pipeline. The smallest high-impact change is to make the handcrafted features more disease-sensitive by adding lightweight texture and color descriptors (HSV stats, edge magnitude stats, and green-red/yellow indices) computed from the same resized image array, without changing the model family or training semantics. I also set a fixed `random_state` for determinism and use a slightly stronger regularization setting (`C`) that often improves ROC-AUC for this kind of small-feature linear model, while keeping the same solver and overall approach. The script still run end-to-end, keep paths unchanged/robust, and always write a valid `submission.csv` with the required columns and order.'
- What this solution (achieved 0.78282) has done: 'Your current solution is far below the target, so we should increase ROC-AUC by strengthening the per-image ranking signal while keeping the same core pipeline (handcrafted image stats → StandardScaler → OneVsRest LogisticRegression). The smallest high-impact change without altering the model family/training approach is to add a little more spatial granularity (a 4×4 grid in addition to the existing 2×2) and a few rotation-invariant texture summaries via simple FFT magnitude statistics computed from the resized grayscale image. These features remain deterministic, lightweight, and compatible with your current sklearn setup, and they typically improve separability for leaf disease patterns (spots/patches). The rest (paths, targets, classifier, and submission writing) is kept identical so it runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.79018) has done: 'We keep your exact pipeline (handcrafted image features → StandardScaler → OneVsRest LogisticRegression) and only make small, score-directed adjustments that usually improve mean ROC-AUC ranking without changing the modeling approach. The main change is to add a few lightweight, disease-relevant texture descriptors (LBP-like local binary pattern histogram on grayscale + per-channel low-frequency DCT energy) computed from the same resized image; these add complementary information to your existing color/FFT/gradient features. We also set `n_jobs=-1` on the OneVsRest wrapper to ensure it finishes comfortably under the time limit, and we keep deterministic behavior. Submission writing/format stays identical.'
- What this solution (achieved 0.77354) has done: 'You’re well below the target (0.79018 vs 0.96754), so we should improve ranking signal while keeping the exact same core pipeline (handcrafted image features → StandardScaler → OneVsRest LogisticRegression). The smallest high-impact change is to add a few more deterministic, cheap, disease-relevant texture/color features without changing the model family: multi-scale LBP histograms (P=8 and P=16) and a simple per-channel “brownness / redness” index that helps rust vs scab separation. I also keep the same training semantics but make the logistic regression slightly less regularized (C) to better fit the richer feature set, which commonly improves ROC-AUC for linear models on small datasets. Submission writing/format and paths remain identical, and the script still runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.79648) has done: 'We keep your exact pipeline (handcrafted deterministic image features → StandardScaler → OneVsRest LogisticRegression) and only strengthen the feature set in a minimal, low-risk way to push ROC-AUC upward toward the 0.9675 target. The main change is to add a couple of very cheap but informative texture summaries (multi-scale gradient magnitude stats and per-quadrant edge stats) plus a compact color histogram in HSV space, which helps ranking without changing model family or training semantics. We also add a tiny amount of probability calibration via a strictly monotonic transform (none; we not change semantics), so predictions remain probabilities from the same model—only inputs improve. Submission writing/format and paths remain identical, and the script still produces `submission.csv` end-to-end.'
- What this solution (achieved 0.82892) has done: 'Your current gap to the target is large (0.79648 → 0.96754), so we should cautiously increase ROC-AUC by strengthening the *ranking signal* while keeping your exact pipeline (handcrafted deterministic features → StandardScaler → OneVsRest LogisticRegression). The most impactful minimal change here is to add a few more cheap, disease-relevant texture/color descriptors that still fit the same “single-pass feature extraction” pattern: per-channel gradient/edge stats (not just grayscale), simple Gabor-like orientation energy via rotated gradients, and a compact “green mask” lesion/spot contrast summary. I also set `n_jobs=-1` on the underlying LogisticRegression (still lbfgs) to ensure training finishes comfortably, and keep everything deterministic and submission-format identical. No changes to data splits (none), loss, model family, or training loop semantics—only feature enrichment.'
- What this solution (achieved 0.82956) has done: 'You’re still well below the target (0.82892 vs 0.96754), so the safest way to move ROC-AUC upward without changing your model family or training semantics is to strengthen the existing handcrafted feature extractor and slightly tune regularization within the same LogisticRegression+OVR pipeline. I add a couple of cheap, deterministic shape/spot descriptors (radial power spectrum summary and center-vs-border contrast) that tend to help leaf-lesion separability, while keeping all existing features intact. I also make a small, conservative adjustment to LogisticRegression’s `C` (same solver/loop) to better fit the now-richer feature set, and keep everything deterministic and submission formatting identical. The script still run end-to-end under the same paths and write a valid `submission.csv` with the exact required columns.'
- What this solution (achieved 0.82146) has done: 'You’re still below the target (0.82956 vs 0.96754), so we should cautiously increase mean ROC-AUC by improving probability ranking while keeping your exact pipeline (handcrafted deterministic features → StandardScaler → OneVsRest LogisticRegression). The smallest high-impact change that preserves core logic is to train the same model via out-of-fold stacking: use StratifiedKFold on the single-label argmax (derived from the one-hot targets) to generate OOF predicted probabilities, then fit a simple per-class LogisticRegression “meta” calibrator on those OOF probabilities and apply it to test probabilities. This doesn’t change architecture/loss family (still logistic regression), doesn’t add early stopping/sampling, and usually boosts ROC-AUC by correcting systematic per-class biases without needing deep CV libraries. I also make feature extraction faster and deterministic by caching extracted features per image id (no semantic change, just avoids repeated PIL work and keeps runtime under the limit). The script still runs end-to-end and writes a valid `submission.csv` with the required columns and ordering.'
- What this solution (achieved 0.81855) has done: 'You’re still well below the target (0.82146 vs 0.96754), so we should improve mean ROC-AUC by increasing the per-class ranking signal without changing your core pipeline (handcrafted deterministic features → StandardScaler → OneVsRest LogisticRegression → OOF meta-logistic). The smallest likely gain is to fix a mismatch: you train the base OVR model with `class_weight="balanced"` (good for AUC), but you also train the meta calibrators with `class_weight="balanced"` which can distort probability ordering; we remove balancing in the meta layer and add a tiny L2 regularization adjustment to stabilize it. We also add a very small, deterministic monotonic post-processing for each class (rank-based normalization) applied equally to all test predictions; this preserves ordering (so it should not hurt AUC) and can slightly improve column-wise AUC by better spreading tied/clustered probabilities. All paths, feature extraction, model family, CV loop, and submission writing remain intact and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.82226) has done: 'You’re still well below the target (0.81855 vs 0.96754), so we should increase mean ROC-AUC by improving the ranking signal while keeping your exact pipeline (handcrafted deterministic features → StandardScaler → OneVsRest LogisticRegression → OOF meta-logistic → per-column rank normalization). The smallest high-impact, low-risk change is to add simple test-time augmentation (TTA) by extracting the *same* features on a few deterministic flips/rotations and averaging the resulting probabilities; this doesn’t change the model family, loss, or training loop, but typically improves generalization for leaf images. To keep changes minimal, training remains identical; only test feature extraction is augmented and averaged before prediction. Submission writing/format, paths, and target columns remain unchanged and it still produces a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
CANDIDATE_BASES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]


def resolve_competition_dir():
    for b in CANDIDATE_BASES:
        if os.path.isdir(b):
            if os.path.isfile(os.path.join(b, "train.csv")):
                return b
            sub = os.path.join(b, "plant-pathology-2020-fgvc7")
            if os.path.isfile(os.path.join(sub, "train.csv")):
                return sub
    raise FileNotFoundError(
        "Could not locate competition directory containing train.csv"
    )


COMP_DIR = resolve_competition_dir()
COMP_DIR



## === cell 2
train_path = os.path.join(COMP_DIR, "train.csv")
test_path = os.path.join(COMP_DIR, "test.csv")
sample_path = os.path.join(COMP_DIR, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sub = pd.read_csv(sample_path)

train.shape, test.shape, sub.shape, train.columns.tolist()



## === cell 3
target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

missing_in_train = [c for c in target_cols if c not in train.columns]
missing_in_sub = [c for c in target_cols if c not in sub.columns]
if missing_in_train:
    raise ValueError(f"Missing target columns in train.csv: {missing_in_train}")
if missing_in_sub:
    raise ValueError(
        f"Missing target columns in sample_submission.csv: {missing_in_sub}"
    )
if "image_id" not in sub.columns:
    raise ValueError("sample_submission.csv must contain image_id")

IMG_DIR_CANDIDATES = [
    os.path.join(COMP_DIR, "images"),
    os.path.join(os.path.dirname(COMP_DIR), "images"),
]
IMG_DIR = None
for p in IMG_DIR_CANDIDATES:
    if os.path.isdir(p):
        IMG_DIR = p
        break
if IMG_DIR is None:
    raise FileNotFoundError(
        "Could not locate images/ directory next to competition CSVs."
    )

from PIL import Image, ImageOps  # noqa: E402


def img_path_from_id(image_id: str) -> str:
    return os.path.join(IMG_DIR, f"{image_id}.jpg")


def _safe_stats(arr: np.ndarray) -> np.ndarray:
    means = arr.mean(axis=(0, 1))
    stds = arr.std(axis=(0, 1))
    return np.concatenate([means, stds], axis=0)


def _rgb_to_hsv_np(rgb01: np.ndarray) -> np.ndarray:
    r = rgb01[:, :, 0]
    g = rgb01[:, :, 1]
    b = rgb01[:, :, 2]

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    delta = maxc - minc

    s = np.where(maxc > 0, delta / (maxc + 1e-8), 0.0)

    h = np.zeros_like(maxc, dtype=np.float32)
    mask = delta > 1e-8

    rc = (maxc - r) / (delta + 1e-8)
    gc = (maxc - g) / (delta + 1e-8)
    bc = (maxc - b) / (delta + 1e-8)

    r_is_max = (r == maxc) & mask
    g_is_max = (g == maxc) & mask
    b_is_max = (b == maxc) & mask

    h[r_is_max] = (bc - gc)[r_is_max]
    h[g_is_max] = (2.0 + rc - bc)[g_is_max]
    h[b_is_max] = (4.0 + gc - rc)[b_is_max]
    h = (h / 6.0) % 1.0

    hsv = np.stack([h, s, v], axis=2).astype(np.float32)
    return hsv


def _grid_stats(arr: np.ndarray, grid: int) -> np.ndarray:
    H, W, C = arr.shape
    feats = []
    for i in range(grid):
        for j in range(grid):
            y0 = (i * H) // grid
            y1 = ((i + 1) * H) // grid
            x0 = (j * W) // grid
            x1 = ((j + 1) * W) // grid
            patch = arr[y0:y1, x0:x1, :]
            feats.append(_safe_stats(patch))
    return np.concatenate(feats, axis=0)


def _fft_texture_features(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    g = g - float(g.mean())
    F = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(F).astype(np.float32)

    H, W = mag.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float32)

    rmax = float(rr.max()) + 1e-8
    r = rr / rmax

    bins = [(0.0, 0.10), (0.10, 0.25), (0.25, 0.45), (0.45, 0.70), (0.70, 1.01)]
    feats = []
    for lo, hi in bins:
        m = (r >= lo) & (r < hi)
        vals = mag[m]
        if vals.size == 0:
            feats.extend([0.0, 0.0, 0.0])
        else:
            feats.extend(
                [float(vals.mean()), float(vals.std()), float(np.quantile(vals, 0.90))]
            )
    return np.array(feats, dtype=np.float32)


def _lbp_hist_riu2(gray01: np.ndarray, P: int = 8) -> np.ndarray:
    g = gray01.astype(np.float32)
    center = g[1:-1, 1:-1]

    if P != 8:
        offsets = [
            (-2, 0),
            (-2, 1),
            (-2, 2),
            (-1, 2),
            (0, 2),
            (1, 2),
            (2, 2),
            (2, 1),
            (2, 0),
            (2, -1),
            (2, -2),
            (1, -2),
            (0, -2),
            (-1, -2),
            (-2, -2),
            (-2, -1),
        ]
        center = g[2:-2, 2:-2]
        neighbors = [
            g[2 + dy : g.shape[0] - 2 + dy, 2 + dx : g.shape[1] - 2 + dx]
            for dy, dx in offsets
        ]
    else:
        neighbors = [
            g[0:-2, 0:-2],
            g[0:-2, 1:-1],
            g[0:-2, 2:],
            g[1:-1, 2:],
            g[2:, 2:],
            g[2:, 1:-1],
            g[2:, 0:-2],
            g[1:-1, 0:-2],
        ]

    bits = [(n >= center).astype(np.uint8) for n in neighbors]

    b0 = bits[0]
    transitions = np.zeros_like(center, dtype=np.uint8)
    prev = b0
    for k in range(1, P):
        transitions += bits[k] ^ prev
        prev = bits[k]
    transitions += b0 ^ prev

    ones = np.zeros_like(center, dtype=np.uint8)
    for b in bits:
        ones += b
    mapped = np.where(transitions <= 2, ones, P + 1).astype(np.int32)

    nbins = P + 2
    hist = np.bincount(mapped.ravel(), minlength=nbins).astype(np.float32)
    hist /= hist.sum() + 1e-8
    return hist


def _dct2_energy_features(gray01: np.ndarray, block: int = 8) -> np.ndarray:
    x = gray01.astype(np.float32)

    def dct_1d(a: np.ndarray, axis: int) -> np.ndarray:
        a = np.swapaxes(a, axis, -1)
        n = a.shape[-1]
        v = np.concatenate([a, a[..., ::-1]], axis=-1)
        V = np.fft.rfft(v, axis=-1)
        k = np.arange(n, dtype=np.float32)
        W = np.exp(-1j * np.pi * k / (2.0 * n))
        out = (V[..., :n] * W).real.astype(np.float32)
        out = np.swapaxes(out, axis, -1)
        return out

    d = dct_1d(dct_1d(x, axis=0), axis=1)

    H, W = d.shape
    b = min(block, H, W)
    lf = d[:b, :b]
    yy, xx = np.ogrid[:b, :b]
    rr = np.sqrt(yy**2 + xx**2)
    rmax = float(rr.max()) + 1e-8
    r = rr / rmax
    bins = [(0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.01)]
    feats = []
    lf2 = (lf * lf).astype(np.float32)
    for lo, hi in bins:
        m = (r >= lo) & (r < hi)
        vals = lf2[m]
        feats.append(float(vals.mean()) if vals.size else 0.0)
        feats.append(float(np.quantile(vals, 0.90)) if vals.size else 0.0)
    return np.array(feats, dtype=np.float32)


def _hsv_hist(
    hsv: np.ndarray, h_bins: int = 12, s_bins: int = 4, v_bins: int = 4
) -> np.ndarray:
    h = hsv[:, :, 0].ravel()
    s = hsv[:, :, 1].ravel()
    v = hsv[:, :, 2].ravel()
    hb = np.clip((h * h_bins).astype(np.int32), 0, h_bins - 1)
    sb = np.clip((s * s_bins).astype(np.int32), 0, s_bins - 1)
    vb = np.clip((v * v_bins).astype(np.int32), 0, v_bins - 1)
    idx = (hb * (s_bins * v_bins) + sb * v_bins + vb).astype(np.int32)
    hist = np.bincount(idx, minlength=h_bins * s_bins * v_bins).astype(np.float32)
    hist /= hist.sum() + 1e-8
    return hist


def _multi_scale_grad_features(gray: np.ndarray) -> np.ndarray:
    feats = []
    g = gray.astype(np.float32)
    for step in (1, 2, 4):
        gs = g[::step, ::step]
        gx = np.diff(gs, axis=1)
        gy = np.diff(gs, axis=0)
        gx = np.pad(gx, ((0, 0), (0, 1)), mode="edge")
        gy = np.pad(gy, ((0, 1), (0, 0)), mode="edge")
        grad = np.sqrt(gx * gx + gy * gy).astype(np.float32)
        feats.extend(
            [float(grad.mean()), float(grad.std()), float(np.quantile(grad, 0.90))]
        )
    return np.array(feats, dtype=np.float32)


def _quad_edge_stats(gray: np.ndarray) -> np.ndarray:
    H, W = gray.shape
    h2, w2 = H // 2, W // 2
    quads = [gray[:h2, :w2], gray[:h2, w2:], gray[h2:, :w2], gray[h2:, w2:]]
    out = []
    for q in quads:
        gx = np.diff(q, axis=1)
        gy = np.diff(q, axis=0)
        gx = np.pad(gx, ((0, 0), (0, 1)), mode="edge")
        gy = np.pad(gy, ((0, 1), (0, 0)), mode="edge")
        grad = np.sqrt(gx * gx + gy * gy).astype(np.float32)
        out.extend([float(grad.mean()), float(np.quantile(grad, 0.90))])
    return np.array(out, dtype=np.float32)


def _channel_grad_stats(arr: np.ndarray) -> np.ndarray:
    feats = []
    for c in range(3):
        ch = arr[:, :, c].astype(np.float32)
        gx = np.diff(ch, axis=1)
        gy = np.diff(ch, axis=0)
        gx = np.pad(gx, ((0, 0), (0, 1)), mode="edge")
        gy = np.pad(gy, ((0, 1), (0, 0)), mode="edge")
        grad = np.sqrt(gx * gx + gy * gy).astype(np.float32)
        feats.extend(
            [float(grad.mean()), float(grad.std()), float(np.quantile(grad, 0.90))]
        )
    return np.array(feats, dtype=np.float32)


def _oriented_grad_energy(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    gx = np.diff(g, axis=1)
    gy = np.diff(g, axis=0)
    gx = np.pad(gx, ((0, 0), (0, 1)), mode="edge")
    gy = np.pad(gy, ((0, 1), (0, 0)), mode="edge")

    e0 = gx * gx
    e90 = gy * gy
    e45 = gx + gy
    e135 = gx - gy
    e45 = e45 * e45
    e135 = e135 * e135

    out = []
    for e in (e0, e45, e90, e135):
        out.extend([float(e.mean()), float(np.quantile(e, 0.90))])
    return np.array(out, dtype=np.float32)


def _green_mask_contrast(arr: np.ndarray) -> np.ndarray:
    eps = 1e-6
    r = arr[:, :, 0].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    b = arr[:, :, 2].astype(np.float32)

    greenish = (g > r) & (g > b)
    frac = float(greenish.mean())

    gray = (r + g + b) / 3.0
    if greenish.any() and (~greenish).any():
        mu_in = float(gray[greenish].mean())
        mu_out = float(gray[~greenish].mean())
        std_in = float(gray[greenish].std())
        std_out = float(gray[~greenish].std())
    else:
        mu_in = float(gray.mean())
        mu_out = float(gray.mean())
        std_in = float(gray.std())
        std_out = float(gray.std())

    return np.array([frac, mu_in - mu_out, std_in - std_out], dtype=np.float32)


def _radial_power_spectrum_features(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    g = g - float(g.mean())
    F = np.fft.fftshift(np.fft.fft2(g))
    power = (F.real * F.real + F.imag * F.imag).astype(np.float32)

    H, W = power.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float32)
    rmax = float(rr.max()) + 1e-8
    r = rr / rmax

    bins = [
        (0.0, 0.06),
        (0.06, 0.14),
        (0.14, 0.28),
        (0.28, 0.50),
        (0.50, 0.80),
        (0.80, 1.01),
    ]
    ring_means = []
    for lo, hi in bins:
        m = (r >= lo) & (r < hi)
        vals = power[m]
        ring_means.append(float(vals.mean()) if vals.size else 0.0)
    ring_means = np.array(ring_means, dtype=np.float32)

    tot = float(ring_means.sum()) + 1e-8
    frac = ring_means / tot

    x = np.arange(len(ring_means), dtype=np.float32)
    y = np.log(ring_means + 1e-6)
    x0 = x - x.mean()
    slope = float((x0 * (y - y.mean())).sum() / (x0 * x0).sum() + 1e-8)

    return np.concatenate([frac, np.array([slope], dtype=np.float32)], axis=0)


def _center_border_contrast(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    H, W = g.shape
    y0, y1 = int(0.25 * H), int(0.75 * H)
    x0, x1 = int(0.25 * W), int(0.75 * W)
    center = g[y0:y1, x0:x1]
    mask = np.ones((H, W), dtype=bool)
    mask[y0:y1, x0:x1] = False
    border = g[mask]

    c_mu, c_sd = float(center.mean()), float(center.std())
    b_mu, b_sd = float(border.mean()), float(border.std())
    return np.array([c_mu - b_mu, c_sd - b_sd], dtype=np.float32)


_FEATURE_CACHE = {}


def _apply_aug(arr01: np.ndarray, aug: str) -> np.ndarray:
    if aug == "orig":
        return arr01
    if aug == "hflip":
        return arr01[:, ::-1, :]
    if aug == "vflip":
        return arr01[::-1, :, :]
    if aug == "rot90":
        return np.rot90(arr01, k=1, axes=(0, 1)).copy()
    raise ValueError(f"Unknown augmentation: {aug}")


def extract_basic_rgb_stats(image_id: str, aug: str = "orig") -> np.ndarray:
    key = (image_id, aug)
    if key in _FEATURE_CACHE:
        return _FEATURE_CACHE[key]

    fp = img_path_from_id(image_id)
    with Image.open(fp) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im = im.resize((128, 128), resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0

    arr = _apply_aug(arr, aug)

    global_stats = _safe_stats(arr)

    gray = arr.mean(axis=2).astype(np.float32)
    gmean = float(gray.mean())
    gstd = float(gray.std())

    H, W, _ = arr.shape
    h2, w2 = H // 2, W // 2
    quads = [
        arr[:h2, :w2, :],
        arr[:h2, w2:, :],
        arr[h2:, :w2, :],
        arr[h2:, w2:, :],
    ]
    quad_stats = np.concatenate([_safe_stats(q) for q in quads], axis=0)

    grid4_stats = _grid_stats(arr, grid=4)

    eps = 1e-6
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    r_m = float(r.mean())
    g_m = float(g.mean())
    b_m = float(b.mean())
    rg = r_m / (g_m + eps)
    rb = r_m / (b_m + eps)
    gb = g_m / (b_m + eps)
    s_rgb = (r_m + g_m + b_m) / 3.0

    hsv = _rgb_to_hsv_np(arr)
    hsv_stats = _safe_stats(hsv)

    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    gx = np.pad(gx, ((0, 0), (0, 1)), mode="edge")
    gy = np.pad(gy, ((0, 1), (0, 0)), mode="edge")
    grad = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    grad_mean = float(grad.mean())
    grad_std = float(grad.std())
    grad_p90 = float(np.quantile(grad, 0.90))

    ngrdi = float(((g - r) / (g + r + eps)).mean())
    excess_green = float((2.0 * g - r - b).mean())
    yellow_index = float(((r + g) / (2.0 * b + eps)).mean())

    redness = float((r - g).mean())
    brownness = float(((r + g) * 0.5 - b).mean())
    dark_fraction = float((gray < 0.30).mean())
    sat_mean = float(hsv[:, :, 1].mean())
    val_mean = float(hsv[:, :, 2].mean())

    fft_feats = _fft_texture_features(gray)

    lbp_hist_p8 = _lbp_hist_riu2(gray, P=8)
    lbp_hist_p16 = _lbp_hist_riu2(gray, P=16)

    dct_energy = _dct2_energy_features(gray, block=8)

    hsv_hist = _hsv_hist(hsv, h_bins=12, s_bins=4, v_bins=4)
    ms_grad = _multi_scale_grad_features(gray)
    quad_edges = _quad_edge_stats(gray)

    ch_grad = _channel_grad_stats(arr)
    orient_e = _oriented_grad_energy(gray)
    green_contrast = _green_mask_contrast(arr)

    radial_ps = _radial_power_spectrum_features(gray)
    cb_contrast = _center_border_contrast(gray)

    feats = np.concatenate(
        [
            global_stats,
            quad_stats,
            grid4_stats,
            np.array([gmean, gstd, rg, rb, gb, s_rgb], dtype=np.float32),
            hsv_stats,
            np.array(
                [grad_mean, grad_std, grad_p90, ngrdi, excess_green, yellow_index],
                dtype=np.float32,
            ),
            np.array(
                [redness, brownness, dark_fraction, sat_mean, val_mean],
                dtype=np.float32,
            ),
            fft_feats,
            lbp_hist_p8,
            lbp_hist_p16,
            dct_energy,
            hsv_hist,
            ms_grad,
            quad_edges,
            ch_grad,
            orient_e,
            green_contrast,
            radial_ps,
            cb_contrast,
        ],
        axis=0,
    ).astype(np.float32)

    _FEATURE_CACHE[key] = feats
    return feats




## === cell 4
train_ids = train["image_id"].astype(str).tolist()
test_ids = test["image_id"].astype(str).tolist()

X_train = np.vstack([extract_basic_rgb_stats(i, aug="orig") for i in train_ids])
y_train = train[target_cols].astype(int).values

from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.multiclass import OneVsRestClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

base_lr = LogisticRegression(
    max_iter=800,
    solver="lbfgs",
    class_weight="balanced",
    C=4.5,
    random_state=42,
    n_jobs=-1,
)

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("ovr", OneVsRestClassifier(base_lr, n_jobs=-1)),
    ]
)

y_single = y_train.argmax(axis=1).astype(int)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

TTA_AUGS = ("orig", "hflip", "vflip", "rot90")
X_test_tta = {
    aug: np.vstack([extract_basic_rgb_stats(i, aug=aug) for i in test_ids])
    for aug in TTA_AUGS
}

oof_proba = np.zeros((X_train.shape[0], len(target_cols)), dtype=np.float32)
test_proba_folds_tta = {
    aug: np.zeros((X_test_tta[aug].shape[0], len(target_cols)), dtype=np.float32)
    for aug in TTA_AUGS
}

for tr_idx, va_idx in skf.split(X_train, y_single):
    clf_fold = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ovr", OneVsRestClassifier(base_lr, n_jobs=-1)),
        ]
    )
    clf_fold.fit(X_train[tr_idx], y_train[tr_idx])

    oof_proba[va_idx] = clf_fold.predict_proba(X_train[va_idx]).astype(np.float32)

    for aug in TTA_AUGS:
        test_proba_folds_tta[aug] += (
            clf_fold.predict_proba(X_test_tta[aug]).astype(np.float32)
            / skf.get_n_splits()
        )

test_proba_folds = np.zeros((len(test_ids), len(target_cols)), dtype=np.float32)
for aug in TTA_AUGS:
    test_proba_folds += test_proba_folds_tta[aug] / float(len(TTA_AUGS))

meta_proba = np.zeros_like(test_proba_folds, dtype=np.float32)
for j in range(len(target_cols)):
    meta = LogisticRegression(
        max_iter=800,
        solver="lbfgs",
        class_weight=None,
        C=0.5,
        random_state=42,
        n_jobs=-1,
    )
    meta.fit(oof_proba[:, [j]], y_train[:, j])
    meta_proba[:, j] = meta.predict_proba(test_proba_folds[:, [j]])[:, 1].astype(
        np.float32
    )

proba = meta_proba


def _rank_normalize_cols(p: np.ndarray) -> np.ndarray:
    out = np.empty_like(p, dtype=np.float32)
    n = p.shape[0]
    denom = max(n - 1, 1)
    for j in range(p.shape[1]):
        order = np.argsort(p[:, j], kind="mergesort")  # stable tie handling
        ranks = np.empty(n, dtype=np.float32)
        ranks[order] = np.arange(n, dtype=np.float32) / float(denom)
        out[:, j] = np.clip(ranks, 1e-4, 1.0 - 1e-4)
    return out


proba = _rank_normalize_cols(proba)

pred_df = pd.DataFrame(proba, columns=target_cols)
pred_df.insert(0, "image_id", test["image_id"].values)

sub_out = sub[["image_id"]].merge(pred_df, on="image_id", how="left")

priors = train[target_cols].mean().astype(float)
for c in target_cols:
    sub_out[c] = sub_out[c].fillna(float(priors[c]))

sub_out[target_cols] = sub_out[target_cols].clip(0.0, 1.0)

out_path = "submission.csv"
sub_out.to_csv(out_path, index=False)
out_path, sub_out.head()
