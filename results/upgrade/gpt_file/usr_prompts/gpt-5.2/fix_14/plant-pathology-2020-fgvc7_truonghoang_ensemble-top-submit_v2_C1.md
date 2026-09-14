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
scipy==1.15.3
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

0.959855568552444

# 6. Current score

0.76637

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The current notebook fails because it tries to read eight external “plantpathology” submission CSVs that are not present in your Kaggle filesystem, so `sub1`…`sub8` are never created. To make it run end-to-end and still follow the same “use existing submissions / entropy selection” core idea, I load whatever prediction CSVs actually exist in the provided `../input` tree, validate/align them to `sample_submission.csv` by `image_id`, and then apply your entropy-based per-row selection across the available files. If none are found, it safely falls back to the sample submission probabilities (uniform baseline) and still writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.5) has done: 'Your current pipeline likely gets 0.5 because it often finds zero usable prediction CSVs (so it falls back to the uniform sample submission) and/or it may accidentally include non-model CSVs that happen to have the same columns. I make the discovery step stricter (only treat files as candidates if they match the test set exactly by `image_id` count and set) and remove the “average blend” overwrite so the final submission is the entropy-selected output when candidates exist. I also ensure we always align to `test.csv` order (not just `sample_submission.csv`) to avoid any silent ordering mismatch. These are minimal, logic-preserving changes that should increase the chance you actually use real predictions and move ROC AUC upward toward the target.'
- What this solution (achieved 0.64985) has done: 'Your current 0.5 score is consistent with falling back to the uniform sample submission because there are typically no valid external prediction CSVs available in this runtime, so the “entropy selection” never actually uses model outputs. To move toward the 0.9598 target while keeping the same overall approach (generate probabilities for the 4 labels and write `submission.csv`), I add a minimal, self-contained baseline model that creates real predictions from the provided data: scikit-learn multinomial logistic regression on simple per-image color features (mean/std RGB), which should substantially exceed 0.5 on this competition. I keep your existing “use discovered external submissions if present” logic intact; the new model only runs when no usable candidate prediction CSVs are found (or can optionally be blended very lightly). This preserves evaluation semantics (probabilities per class) and guarantees an end-to-end valid submission.'
- What this solution (achieved 0.69489) has done: 'Your current score (0.64985) is far below the target (0.95986), and since no external prediction CSVs are usually available, the only way to move toward the target is to make the fallback image-based model stronger while keeping the same overall approach (extract simple image features → multinomial logistic regression → class probabilities). I minimally enrich the feature vector (still hand-crafted, still very fast) by adding per-channel min/max, simple grayscale moments, and a coarse color histogram, and I standardize features (important for logistic regression). I also switch the target encoding to the actual class index derived from the one-hot row (same semantics) and keep the exact same submission-writing logic and paths. These changes should significantly improve ROC AUC while staying within the same core logic and runtime constraints.'
- What this solution (achieved 0.72295) has done: 'Your score is far below the target, so the safest way to move it upward while preserving your exact “simple image features → multinomial logistic regression → probabilities” core logic is to (1) add a couple of very cheap but informative handcrafted features (HSV mean/std + a lightweight edge-strength summary) and (2) tune the LogisticRegression regularization strength mildly. These are minimal extensions of your existing feature extraction and same sklearn model, and they typically help this competition noticeably without changing training semantics. I also make the class-index creation robust (in case of any non one-hot rows) and keep submission alignment strictly to `test.csv` order as you already do.'
- What this solution (achieved 0.73813) has done: 'Your current score (0.72295) is far below the target (0.95986), so we should nudge performance upward with minimal, core-logic-preserving changes to your existing “handcrafted image features → multinomial LogisticRegression → probabilities” fallback model. The safest gain here is to strengthen the same feature family by adding a tiny amount of spatial structure (very coarse block-wise RGB means) without changing the modeling approach, and to slightly adjust LogisticRegression regularization to better fit these richer features. I also add a deterministic in-memory cache for per-image features so we can afford these extra features without risking the 600s timeout. Everything else (paths, submission schema, entropy-selection behavior when external preds exist) remains the same.'
- What this solution (achieved 0.73958) has done: 'We keep your exact pipeline (discover external prediction CSVs → entropy select; otherwise fallback to handcrafted image features + multinomial LogisticRegression) and only make small, targeted changes to improve ROC AUC toward 0.9599. The main uplift come from (1) using a slightly higher-resolution resize in the same feature extractor and (2) modestly increasing the block-wise spatial summary (still the same “coarse block means” idea) to capture more lesion localization signal, while keeping the same model and training loop. I also switch `LogisticRegression` to `class_weight="balanced"` (same model, same loss family) to reduce bias against minority classes, which usually improves mean column-wise AUC for this dataset. All paths, submission schema, and the entropy-selection behavior remain unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.74905) has done: 'Your current score (0.73958) is far below the target (0.95986), so we should carefully increase performance without changing the core approach (handcrafted image features → multinomial LogisticRegression → probabilities). The smallest likely uplift is to add a tiny amount more spatial structure to the same feature family by including a second block-grid (4×4) in addition to your existing 3×3 block means; this often helps capture localized lesions with negligible extra complexity. I’m also slightly increasing `max_iter` to ensure convergence with the larger feature vector (same solver/model, just more stable fitting). Everything else (data discovery, entropy-selection behavior, submission format/paths) is kept identical.'
- What this solution (achieved 0.74066) has done: 'Your current score (0.74905) is far below the target (0.95986), so we should nudge performance upward while keeping the same core pipeline (handcrafted image features → multinomial LogisticRegression → probabilities → submission). The smallest likely gain without changing the model family is to add a bit more lesion-localization signal by including one more coarse spatial grid (5×5 block means), keeping all existing features intact. Because this increases feature dimensionality, I also slightly increase `max_iter` to ensure stable convergence (same solver/objective), which can improve probabilities and thus ROC AUC. All paths, CSV discovery/entropy-selection logic, and submission formatting remain unchanged.'
- What this solution (achieved 0.72341) has done: 'Your current gap to the target is large (0.74066 → 0.95986), so we need a small but meaningful uplift while keeping the same core pipeline (handcrafted image features → multinomial LogisticRegression → probabilities → submission). The minimal change with the best chance of improving mean ROC AUC here is to add a few more very cheap, still-handcrafted features that capture lesion texture/color distribution without changing the model family: (1) per-channel 25/50/75 percentiles, and (2) a slightly finer color histogram. These keep the same training approach and semantics (still just a feature vector into the same logistic regression), and remain fast under the timeout. I also slightly increase `max_iter` to ensure convergence with the slightly larger feature vector (no change in optimization objective).'
- What this solution (achieved 0.76487) has done: 'Your current score (0.72341) is far below the target (0.95986), so we should make a small, legitimate improvement to the same fallback pipeline (handcrafted image features → multinomial LogisticRegression → probabilities). The least invasive uplift is to strengthen the *same* feature extractor by adding a simple, rotation-invariant texture descriptor (uniform Local Binary Pattern histogram) computed on the grayscale image; this often helps leaf-disease separation without changing the model family or training semantics. To keep runtime under control, I compute LBP on the already-resized grayscale and add it as a small fixed-length histogram feature. Everything else (CSV discovery/entropy-selection behavior, submission alignment/format, and the LogisticRegression approach) stays the same.'
- What this solution (achieved 0.76321) has done: 'Your current score (0.76487) is far below the target (0.95986), so we should make a small uplift while keeping the same core approach: handcrafted image features → multinomial LogisticRegression → probabilities → submission. The most likely low-risk gain without changing the model family is to add a little more spatial structure (a 6×6 block-mean grid) and a slightly richer, still-cheap LBP texture summary (multi-scale LBP at radii 1 and 2, concatenated). These are direct extensions of your existing feature extractor (same semantics, same training loop), and should improve mean column-wise ROC AUC by capturing more localized lesion patterns. Everything else (external CSV discovery/entropy selection behavior, alignment to `test.csv`, and writing `submission.csv`) remains unchanged.'
- What this solution (achieved 0.76637) has done: 'We keep your exact pipeline (use discovered external prediction CSVs when available; otherwise fallback to handcrafted image features + multinomial LogisticRegression) and make one targeted, low-risk uplift to move mean ROC AUC upward toward the 0.9599 target. The main change is to add a simple “excess green/excess red” vegetation/lesion index feature family (computed from the same resized RGB array), which often separates healthy vs diseased leaf tissue better without changing the modeling approach. To preserve semantics and stability, we keep the same LogisticRegression setup and only slightly adjust regularization (C) to accommodate the richer but still small feature vector. Everything else (paths, CSV discovery/alignment, entropy selection, and writing `submission.csv`) remains unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from scipy.stats import entropy

INPUT_ROOT = "../input"
COMP_DIR_CANDIDATES = [
    "../input/plant-pathology-2020-fgvc7",
    "../input/plant-pathology-2020-fgvc7/plant-pathology-2020-fgvc7",
    "../input",
]


def find_first_existing(path_list, filename):
    for d in path_list:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return None


sample_path = find_first_existing(COMP_DIR_CANDIDATES, "sample_submission.csv")
test_path = find_first_existing(COMP_DIR_CANDIDATES, "test.csv")
train_path = find_first_existing(COMP_DIR_CANDIDATES, "train.csv")

if sample_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv under expected ../input paths."
    )
if test_path is None:
    raise FileNotFoundError("Could not locate test.csv under expected ../input paths.")
if train_path is None:
    raise FileNotFoundError("Could not locate train.csv under expected ../input paths.")

sample_sub = pd.read_csv(sample_path)
test_df = pd.read_csv(test_path)
train_df = pd.read_csv(train_path)

required_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
for c in required_cols:
    if c not in sample_sub.columns:
        raise ValueError(f"sample_submission.csv missing required column: {c}")

target_cols = required_cols[1:]

test_ids = test_df["image_id"].astype(str)
test_id_set = set(test_ids.tolist())


def discover_csvs(root="../input"):
    csvs = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".csv"):
                csvs.append(os.path.join(dirpath, fn))
    return csvs


all_csvs = discover_csvs(INPUT_ROOT)


def is_candidate_prediction_csv(df: pd.DataFrame) -> bool:
    cols = list(df.columns)
    if "image_id" not in cols:
        return False
    if not all(c in cols for c in target_cols):
        return False

    ids = df["image_id"].astype(str)
    if ids.nunique() != len(ids):
        return False
    if len(ids) != len(test_ids):
        return False
    if set(ids.tolist()) != test_id_set:
        return False

    return True


candidates = []
for p in all_csvs:
    base = os.path.basename(p)
    if base in {"train.csv", "test.csv", "sample_submission.csv"}:
        continue
    try:
        df = pd.read_csv(p)
    except Exception:
        continue
    if is_candidate_prediction_csv(df):
        candidates.append(p)

print(f"Found {len(candidates)} candidate prediction CSV(s).")
for p in candidates[:20]:
    print("  ", p)

aligned_subs = []
used_paths = []

for p in candidates:
    df = pd.read_csv(p)

    df = df[["image_id"] + target_cols].copy()
    df["image_id"] = df["image_id"].astype(str)
    df = df.drop_duplicates(subset=["image_id"], keep="first")

    merged = (
        test_df[["image_id"]]
        .assign(image_id=test_df["image_id"].astype(str))
        .merge(df, on="image_id", how="left")
    )

    if merged[target_cols].isna().any().any():
        continue

    preds = merged[target_cols].astype(float).clip(0.0, 1.0)

    aligned_subs.append(preds.to_numpy())
    used_paths.append(p)

print(f"Usable aligned prediction CSV(s): {len(aligned_subs)}")
for p in used_paths[:20]:
    print("  ", p)



## === cell 1
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def find_images_dir(path_list):
    for d in path_list:
        cand = os.path.join(d, "images")
        if os.path.isdir(cand):
            return cand
    return None


IMAGES_DIR = find_images_dir(COMP_DIR_CANDIDATES)
if IMAGES_DIR is None:
    raise FileNotFoundError(
        "Could not locate images/ directory under expected ../input paths."
    )


def _rgb_to_hsv_arr(rgb01: np.ndarray) -> np.ndarray:
    """
    Minimal, dependency-free RGB->HSV for arrays in [0,1], shape (H,W,3).
    Returns HSV in [0,1] (H,S,V).
    """
    r = rgb01[..., 0]
    g = rgb01[..., 1]
    b = rgb01[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(cmax, dtype=np.float32)
    mask = delta > 1e-8

    rc = ((g - b) / (delta + 1e-12)) % 6.0
    gc = ((b - r) / (delta + 1e-12)) + 2.0
    bc = ((r - g) / (delta + 1e-12)) + 4.0

    rmax = (cmax == r) & mask
    gmax = (cmax == g) & mask
    bmax = (cmax == b) & mask
    h[rmax] = rc[rmax]
    h[gmax] = gc[gmax]
    h[bmax] = bc[bmax]
    h = (h / 6.0).astype(np.float32)

    s = np.zeros_like(cmax, dtype=np.float32)
    nonzero = cmax > 1e-8
    s[nonzero] = (delta[nonzero] / (cmax[nonzero] + 1e-12)).astype(np.float32)

    v = cmax.astype(np.float32)

    return np.stack([h, s, v], axis=-1).astype(np.float32)


def _sobel_edge_strength(gray: np.ndarray) -> np.ndarray:
    """
    Very small edge summary: mean and std of Sobel gradient magnitude.
    gray: (H,W) float32 in [0,1]
    """
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    g = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
    gx = (
        kx[0, 0] * g[:-2, :-2]
        + kx[0, 1] * g[:-2, 1:-1]
        + kx[0, 2] * g[:-2, 2:]
        + kx[1, 0] * g[1:-1, :-2]
        + kx[1, 1] * g[1:-1, 1:-1]
        + kx[1, 2] * g[1:-1, 2:]
        + kx[2, 0] * g[2:, :-2]
        + kx[2, 1] * g[2:, 1:-1]
        + kx[2, 2] * g[2:, 2:]
    )
    gy = (
        ky[0, 0] * g[:-2, :-2]
        + ky[0, 1] * g[:-2, 1:-1]
        + ky[0, 2] * g[:-2, 2:]
        + ky[1, 0] * g[1:-1, :-2]
        + ky[1, 1] * g[1:-1, 1:-1]
        + ky[1, 2] * g[1:-1, 2:]
        + ky[2, 0] * g[2:, :-2]
        + ky[2, 1] * g[2:, 1:-1]
        + ky[2, 2] * g[2:, 2:]
    )
    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    return np.array([mag.mean(), mag.std()], dtype=np.float32)


def _block_means(arr: np.ndarray, grid=(2, 2)) -> np.ndarray:
    H, W, C = arr.shape
    gh, gw = grid
    feats = []
    for i in range(gh):
        hs, he = int(i * H / gh), int((i + 1) * H / gh)
        for j in range(gw):
            ws, we = int(j * W / gw), int((j + 1) * W / gw)
            block = arr[hs:he, ws:we, :]
            feats.append(block.mean(axis=(0, 1)).astype(np.float32))
    return np.concatenate(feats, axis=0).astype(np.float32)


def _lbp_uniform_riu2_hist(gray01: np.ndarray, radius: int = 1) -> np.ndarray:
    """
    Uniform LBP (P=8), rotation-invariant (riu2) histogram with 10 bins:
      - bins 0..8: number of 1s for uniform patterns
      - bin 9: all non-uniform patterns
    """
    r = int(radius)
    g = np.pad(gray01, ((r, r), (r, r)), mode="edge")
    c = g[r:-r, r:-r]

    n0 = (g[0 : -2 * r, 0 : -2 * r] >= c).astype(np.uint8)
    n1 = (g[0 : -2 * r, r:-r] >= c).astype(np.uint8)
    n2 = (g[0 : -2 * r, 2 * r :] >= c).astype(np.uint8)
    n3 = (g[r:-r, 2 * r :] >= c).astype(np.uint8)
    n4 = (g[2 * r :, 2 * r :] >= c).astype(np.uint8)
    n5 = (g[2 * r :, r:-r] >= c).astype(np.uint8)
    n6 = (g[2 * r :, 0 : -2 * r] >= c).astype(np.uint8)
    n7 = (g[r:-r, 0 : -2 * r] >= c).astype(np.uint8)

    neighbors = [n0, n1, n2, n3, n4, n5, n6, n7]
    ones = (n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7).astype(np.uint8)

    transitions = np.zeros_like(c, dtype=np.uint8)
    for a, b in zip(neighbors, neighbors[1:] + neighbors[:1]):
        transitions += (a != b).astype(np.uint8)

    uniform = transitions <= 2
    codes = np.where(uniform, ones, 9).astype(np.int32)

    hist = np.bincount(codes.ravel(), minlength=10).astype(np.float32)
    hist = hist / (hist.sum() + 1e-12)
    return hist


_FEATURE_CACHE = {}


def image_features(image_path, size=(128, 128), hist_bins=12):
    cached = _FEATURE_CACHE.get(image_path, None)
    if cached is not None:
        return cached

    with Image.open(image_path) as im:
        im = im.convert("RGB").resize(size)
        arr = np.asarray(im, dtype=np.float32) / 255.0  # (H,W,3)

    mean = arr.mean(axis=(0, 1))
    std = arr.std(axis=(0, 1))
    mn = arr.min(axis=(0, 1))
    mx = arr.max(axis=(0, 1))

    flat = arr.reshape(-1, 3)
    p25 = np.percentile(flat, 25, axis=0).astype(np.float32)
    p50 = np.percentile(flat, 50, axis=0).astype(np.float32)
    p75 = np.percentile(flat, 75, axis=0).astype(np.float32)

    gray = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]).astype(
        np.float32
    )
    g_mean = np.array([gray.mean()], dtype=np.float32)
    g_std = np.array([gray.std()], dtype=np.float32)

    hsv = _rgb_to_hsv_arr(arr)
    hsv_mean = hsv.mean(axis=(0, 1))
    hsv_std = hsv.std(axis=(0, 1))

    edge_stats = _sobel_edge_strength(gray)

    h_feats = []
    for c in range(3):
        h, _ = np.histogram(arr[..., c], bins=hist_bins, range=(0.0, 1.0), density=True)
        h_feats.append(h.astype(np.float32))
    h_feats = np.concatenate(h_feats, axis=0)

    block_3x3 = _block_means(arr, grid=(3, 3))
    block_4x4 = _block_means(arr, grid=(4, 4))
    block_5x5 = _block_means(arr, grid=(5, 5))
    block_6x6 = _block_means(arr, grid=(6, 6))

    lbp_hist_r1 = _lbp_uniform_riu2_hist(gray, radius=1)
    lbp_hist_r2 = _lbp_uniform_riu2_hist(gray, radius=2)
    lbp_hist = np.concatenate([lbp_hist_r1, lbp_hist_r2], axis=0).astype(np.float32)

    r = arr[..., 0].astype(np.float32)
    g = arr[..., 1].astype(np.float32)
    b = arr[..., 2].astype(np.float32)
    exg = (2.0 * g - r - b).astype(np.float32)  # Excess Green
    exr = (1.4 * r - g).astype(np.float32)  # Excess Red (variant)
    ndi = ((g - r) / (g + r + 1e-12)).astype(np.float32)  # normalized diff index

    veg_feats = np.array(
        [
            exg.mean(),
            exg.std(),
            exr.mean(),
            exr.std(),
            ndi.mean(),
            ndi.std(),
        ],
        dtype=np.float32,
    )

    feats = np.concatenate(
        [
            mean,
            std,
            mn,
            mx,
            p25,
            p50,
            p75,
            g_mean,
            g_std,
            hsv_mean,
            hsv_std,
            edge_stats,
            h_feats,
            block_3x3,
            block_4x4,
            block_5x5,
            block_6x6,
            lbp_hist,
            veg_feats,
        ],
        axis=0,
    ).astype(np.float32)

    _FEATURE_CACHE[image_path] = feats
    return feats


def build_feature_matrix(ids):
    first_existing = None
    for img_id in ids:
        p = os.path.join(IMAGES_DIR, f"{img_id}.jpg")
        if os.path.exists(p):
            first_existing = p
            break
    if first_existing is None:
        raise FileNotFoundError(f"No images found under {IMAGES_DIR} for provided ids.")

    d = image_features(first_existing).shape[0]
    X = np.zeros((len(ids), d), dtype=np.float32)

    missing = 0
    for i, img_id in enumerate(ids):
        p = os.path.join(IMAGES_DIR, f"{img_id}.jpg")
        if not os.path.exists(p):
            missing += 1
            continue
        X[i] = image_features(p)
    if missing:
        print(f"Warning: {missing} image(s) missing under {IMAGES_DIR}.")
    return X


y_mat = train_df[target_cols].astype(int).to_numpy()
row_sums = y_mat.sum(axis=1)
if not np.all(row_sums == 1):
    print("Warning: found non one-hot rows in train labels; using argmax class index.")
y_idx = np.argmax(y_mat, axis=1).astype(int)

X_train = build_feature_matrix(train_df["image_id"].astype(str).tolist())
X_test = build_feature_matrix(test_df["image_id"].astype(str).tolist())

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "lr",
            LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=6000,
                class_weight="balanced",
                C=12.0,
                n_jobs=1,
                random_state=0,
            ),
        ),
    ]
)

clf.fit(X_train, y_idx)
proba_test = clf.predict_proba(X_test)

model_pred = np.zeros((len(test_df), len(target_cols)), dtype=np.float32)
classes_ = clf.named_steps["lr"].classes_
for j, cls in enumerate(classes_):
    model_pred[:, int(cls)] = proba_test[:, j]

print("Built fallback image-feature model predictions:", model_pred.shape)



## === cell 2
sub = sample_sub.copy()

sub = (
    test_df[["image_id"]]
    .assign(image_id=test_df["image_id"].astype(str))
    .merge(
        sub.assign(image_id=sub["image_id"].astype(str)),
        on="image_id",
        how="left",
    )
)

if len(aligned_subs) == 0:
    sub = sub[["image_id"] + target_cols].copy()
    sub.loc[:, target_cols] = model_pred.astype(float).clip(0.0, 1.0)
    sub.to_csv("submission.csv", index=False)
    print(
        "No usable external prediction CSVs found; wrote model-based submission.csv. Shape:",
        sub.shape,
    )
else:
    ent_list = []
    for arr in aligned_subs:
        ent_list.append(entropy(arr, base=2, axis=1))
    entropies = np.vstack(ent_list).T  # (n_rows, n_models)
    selected = np.argmin(entropies, axis=1)

    stacked = np.stack(aligned_subs, axis=0)  # (n_models, n_rows, n_targets)
    chosen = stacked[selected, np.arange(stacked.shape[1]), :]  # (n_rows, n_targets)

    sub = sub[["image_id"] + target_cols].copy()
    sub.loc[:, target_cols] = chosen.astype(float).clip(0.0, 1.0)
    sub.to_csv("submission.csv", index=False)
    print("Wrote submission.csv (entropy selection). Shape:", sub.shape)
    print("Used models:", len(aligned_subs))
    if used_paths:
        print("First few used paths:")
        for p in used_paths[:10]:
            print("  ", p)
