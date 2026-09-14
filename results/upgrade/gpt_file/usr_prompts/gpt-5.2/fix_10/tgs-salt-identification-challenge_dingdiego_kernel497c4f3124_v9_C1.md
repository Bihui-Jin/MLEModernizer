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
Segment regions of salt in seismic images.

## Metric
Mean average precision at different intersection over union (IoU) thresholds. The IoU of a proposed set of object pixels and a set of true object pixels is calculated as:

$$\text{IoU}(A, B)=\frac{A \cap B}{A \cup B}$$

The metric sweeps over a range of IoU thresholds, at each point calculating an average precision value. The threshold values range from 0.5 to 0.95 with a step size of 0.05: `(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)`. In other words, at a threshold of 0.5, a predicted object is considered a "hit" if its intersection over union with a ground truth object is greater than 0.5.

At each threshold value 𝑡t, a precision value is calculated based on the number of true positives (TP), false negatives (FN), and false positives (FP) resulting from comparing the predicted object to all ground truth objects:

$$\frac{T P(t)}{T P(t)+F P(t)+F N(t)}$$

A true positive is counted when a single predicted object matches a ground truth object with an IoU above the threshold. A false positive indicates a predicted object had no associated ground truth object. A false negative indicates a ground truth object had no associated predicted object. The average precision of a single image is then calculated as the mean of the above precision values at each IoU threshold:

$$\frac{1}{\mid \text { thresholds } \mid} \sum_t \frac{T P(t)}{T P(t)+F P(t)+F N(t)}$$

## Submission Format
Use run-length encoding on the pixel values. Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The pixels are one-indexed\
and numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. It also checks that no two predicted masks for the same image are overlapping.

The file should contain a header and have the following format. Each row in your submission represents a single predicted salt segmentation for the given image.

```
id,rle_mask
3e06571ef3,1 1
a51b08d882,1 1
c32590b06f,1 1
etc.
```

## Dataset
The data is a set of images chosen at various locations chosen at random in the subsurface. The images are 101 x 101 pixels and each pixel is classified as either salt or sediment. In addition to the seismic images, the depth of the imaged location is provided for each image.

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            depths.csv (3001 lines)
            depths.csv.zip (25.0 kB)
            description.md (83 lines)
            sample_submission.csv (1001 lines)
            sample_submission.csv.zip (6.8 kB)
            test.zip (9.8 MB)
            train.csv (3001 lines)
            train.csv.zip (293.9 kB)
            train.zip (30.9 MB)
            test/
                images/
                    a05ae39815.png (10.5 kB)
                    9bf8a38b92.png (10.7 kB)
                    ... and 998 other files
                test/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
            train/
                images/
                    66cf41c563.png (7.7 kB)
                    429bf7c665.png (8.2 kB)
                    ... and 2998 other files
                masks/
                    6eeeda7f4a.png (230 Bytes)
                    5d600057f5.png (230 Bytes)
                    ... and 2998 other files
                train/
        input/
            depths.csv (3001 lines)
            depths.csv.zip (25.0 kB)
            description.md (83 lines)
            sample_submission.csv (1001 lines)
            sample_submission.csv.zip (6.8 kB)
            test.zip (9.8 MB)
            train.csv (3001 lines)
            train.csv.zip (293.9 kB)
            train.zip (30.9 MB)
            test/
                images/
                    a05ae39815.png (10.5 kB)
                    9bf8a38b92.png (10.7 kB)
                    ... and 998 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
            train/
                images/
                    66cf41c563.png (7.7 kB)
                    429bf7c665.png (8.2 kB)
                    ... and 2998 other files
                masks/
                    6eeeda7f4a.png (230 Bytes)
                    5d600057f5.png (230 Bytes)
                    ... and 2998 other files
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
        working/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
```

-> data/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> data/sample_submission.csv has 1000 rows and 2 columns.
The columns are: id, rle_mask

-> data/tgs-salt-identification-challenge/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> data/tgs-salt-identification-challenge/sample_submission.csv has 1000 rows and 2 columns.
The columns are: id, rle_mask

-> data/tgs-salt-identification-challenge/train.csv has 3000 rows and 2 columns.
The columns are: id, rle_mask

-> data/train.csv has 3000 rows and 2 columns.
The columns are: id, rle_mask

-> input/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> (stopped after 10 files for performance)

# 5. Target score

0.8144958892254429

# 6. Current score

0.5221

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the dependency on `pydensecrf` (it isn’t installed in this environment) and make the notebook/script fully runnable by replacing the CRF step with a safe, minimal post-processing fallback that preserves the rest of the pipeline. I also remove notebook-only magic (`%matplotlib inline`) and shell `ls` commands that break execution in a `.py` Kaggle run, while keeping visualization code optional and non-blocking. Finally, I fix the NaN checks and ensure we read an existing `sample_submission.csv` (since `../input/k-fold-merger/submission.csv` is not available) and always write a valid submission CSV with the required columns.'
- What this solution (achieved 0.5221) has done: 'Your current 0.0 score comes from submitting essentially the unmodified `sample_submission.csv` (all empty masks), so the smallest score-improving change is to actually generate predicted masks from the provided training labels. To keep core logic minimal and avoid introducing a new model, I build a simple per-pixel probability prior from `train.csv` masks (mean mask across all training images) and use a tuned global threshold to binarize it for every test image. I keep your CRF fallback (identity) and your corrected RLE encoding (Fortran-order) so submission semantics remain valid. This should move the score up from 0.0 toward your target, while staying lightweight and finishing fast.'
- What this solution (achieved 0.5221) has done: 'Your current approach uses a single global “mean mask prior” for every test image; the easiest legitimate way to move the score upward (without changing the modeling core) is to tune the binarization and add a tiny amount of post-processing that reduces obvious false positives. I (1) search a small set of thresholds on a held-out split of the training masks using the exact competition mAP@IoU metric, and (2) remove very small connected components from the predicted binary mask (a lightweight denoising step) before RLE encoding. This keeps the same core logic (mean-mask prior + threshold) but calibrates it to the metric and typically improves from the low 0.5s without introducing any new model/training. The script still run end-to-end in <600s and write a valid `crf_correction.csv`.'
- What this solution (achieved 0.5221) has done: 'Your current approach uses one global prior mask for every test image, so the only realistic way to move the score upward (without changing the modeling core) is to (1) tune the binarization/post-processing more directly for the leaderboard metric and (2) make the mask post-processing a bit less brittle. I keep the same “mean-mask prior + threshold + small-component removal + RLE” pipeline, but I (a) optimize the threshold and min-component size using a stratified holdout on mask coverage (reduces overfitting to the many-empty-mask subset) and (b) add one more tiny, safe post-processing knob: hole-filling by removing small background components (this often helps IoU without changing the core idea). These are minimal additions that preserve semantics, don’t add new dependencies, and should lift you from ~0.52 closer to the 0.81 target, while still finishing quickly and producing a valid submission CSV.'
- What this solution (achieved 0.5221) has done: 'The timeout is dominated by the hyperparameter grid-search in cell 6, which repeatedly runs Python-loop connected-components (remove_small_components/fill_small_holes) and recomputes the mAP metric per combination; this explodes runtime. I keep the exact same logic (depth-binned mean-mask prior + threshold + optional post-processing + exact mAP@IoU) but (1) precompute per-bin thresholded masks once per (bin, thr) and then only apply the optional morphology, (2) cache morphology results for identical inputs, and (3) vectorize the mAP computation over the whole validation set using precomputed intersections/unions. I also avoid per-row pandas .loc in the test loop and skip reading test PNGs since they are unused by the model (predictions depend only on depth prior), preserving identical outputs. These changes reduce repeated work without changing the algorithm or evaluation semantics.'
- What this solution (achieved 0.5221) has done: 'Your current pipeline is fundamentally limited because it predicts the same (or depth-binned) prior mask for every image, which caps performance well below the target; within the “minimal changes” constraint, the most direct way to move the score upward is to make the threshold/post-processing selection more faithful to the leaderboard objective and reduce overfitting in that tuning. I keep the exact same core logic (depth-binned mean-mask prior + threshold + optional small-component removal + optional hole-filling + RLE), but change the hyperparameter selection to cross-validated out-of-fold evaluation (same metric) instead of a single holdout, which typically yields a better-chosen operating point without changing the model. To stay within time, I also shrink the search space slightly around your currently good region and compute post-processing only once per (bin, thr, min_comp, hole_min) by caching, then reuse across folds. Finally, I ensure the depth file used for test depths is the correct one (same as competition: depths.csv includes both train+test), while preserving all I/O paths and submission format.'
- What this solution (achieved 0.5221) has done: 'The timeout is coming from the nested CV grid-search in cell 6: for every parameter combo you recompute depth-bin mean masks and then loop per validation sample doing per-pixel logical ops, plus expensive Python BFS morphology repeatedly. The optimized script keeps the exact same algorithm (depth-binned mean-mask prior + threshold(+delta) + optional component removal + optional hole filling + same CV folds and metric), but removes redundant work by (1) precomputing per-fold/per-bin mean masks once, (2) vectorizing IoU computation for the whole validation fold using matrix multiplication on boolean flats, and (3) caching post-processed (morphology) masks per (fold, bin, params) so morphology runs only a handful of times rather than thousands. These changes are provably equivalent (same masks/IoUs) and dramatically reduce Python-loop and repeated-work overhead while keeping determinism and output semantics unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import hashlib

from skimage.io import imread

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from tqdm import tqdm
except Exception:

    def tqdm(x, **kwargs):
        return x


HAS_DCRF = False

BASE_INPUT = "../input/tgs-salt-identification-challenge"
TEST_IMG_DIR = os.path.join(BASE_INPUT, "test", "images")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(BASE_INPUT, "train.csv")
DEPTHS_CSV_PATH = os.path.join(BASE_INPUT, "depths.csv")

assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission at: {SAMPLE_SUB_PATH}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing test image dir at: {TEST_IMG_DIR}"
assert os.path.exists(TRAIN_CSV_PATH), f"Missing train.csv at: {TRAIN_CSV_PATH}"
assert os.path.exists(DEPTHS_CSV_PATH), f"Missing depths.csv at: {DEPTHS_CSV_PATH}"

IMG_SHAPE = (101, 101)

np.random.seed(42)




## === cell 1
def rle_decode(rle_mask, shape=(101, 101)):
    """
    rle_mask: run-length as string formatted (start length)
    shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    if rle_mask is None:
        return np.zeros(shape, dtype=np.uint8)
    if isinstance(rle_mask, float) and np.isnan(rle_mask):
        return np.zeros(shape, dtype=np.uint8)

    rle_mask = str(rle_mask).strip()
    if rle_mask == "" or rle_mask.lower() == "nan":
        return np.zeros(shape, dtype=np.uint8)

    s = rle_mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)




## === cell 2
def rle_encode(im):
    """
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted.
    Kaggle Salt expects Fortran-order flattening (down then right).
    """
    im = (im > 0).astype(np.uint8)
    pixels = im.T.flatten()  # column-major equivalent
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)




## === cell 3
"""
Function which returns the labelled image after applying CRF.

pydensecrf is not available here, so we provide a safe fallback that
keeps the same call signature and returns a refined mask (identity).
"""


def crf(original_image, mask_img):
    if mask_img is None:
        return np.zeros(
            (original_image.shape[0], original_image.shape[1]), dtype=np.uint8
        )

    if len(mask_img.shape) == 3:
        mask2d = mask_img[:, :, 0]
    else:
        mask2d = mask_img

    mask2d = (mask2d > 0).astype(np.uint8)
    return mask2d




## === cell 4
def remove_small_components(mask, min_size=20):
    """
    Remove connected components smaller than min_size (4-connected).
    mask: uint8 {0,1}
    """
    mask = (mask > 0).astype(np.uint8)
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=np.uint8)
    out = mask.copy()

    for r in range(h):
        for c in range(w):
            if out[r, c] == 1 and visited[r, c] == 0:
                stack = [(r, c)]
                visited[r, c] = 1
                coords = [(r, c)]
                while stack:
                    rr, cc = stack.pop()
                    if rr > 0 and out[rr - 1, cc] == 1 and visited[rr - 1, cc] == 0:
                        visited[rr - 1, cc] = 1
                        stack.append((rr - 1, cc))
                        coords.append((rr - 1, cc))
                    if rr + 1 < h and out[rr + 1, cc] == 1 and visited[rr + 1, cc] == 0:
                        visited[rr + 1, cc] = 1
                        stack.append((rr + 1, cc))
                        coords.append((rr + 1, cc))
                    if cc > 0 and out[rr, cc - 1] == 1 and visited[rr, cc - 1] == 0:
                        visited[rr, cc - 1] = 1
                        stack.append((rr, cc - 1))
                        coords.append((rr, cc - 1))
                    if cc + 1 < w and out[rr, cc + 1] == 1 and visited[rr, cc + 1] == 0:
                        visited[rr, cc + 1] = 1
                        stack.append((rr, cc + 1))
                        coords.append((rr, cc + 1))

                if len(coords) < int(min_size):
                    for rr, cc in coords:
                        out[rr, cc] = 0
    return out


def fill_small_holes(mask, min_size=20):
    """
    Fill background connected components (holes) smaller than min_size by flipping them to 1.
    Implemented as: hole_fill(mask) = 1 - remove_small_components(1-mask, min_size)
    Uses 4-connected components consistent with remove_small_components.
    """
    mask = (mask > 0).astype(np.uint8)
    inv = (1 - mask).astype(np.uint8)
    inv_clean = remove_small_components(inv, min_size=min_size)
    return (1 - inv_clean).astype(np.uint8)




## === cell 5
def iou_numpy(y_true, y_pred):
    y_true = (y_true > 0).astype(np.uint8)
    y_pred = (y_pred > 0).astype(np.uint8)
    inter = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union


def map_iou_single(y_true, y_pred, thresholds=np.arange(0.5, 1.0, 0.05)):
    iou = iou_numpy(y_true, y_pred)
    return float(np.mean([iou > t for t in thresholds]))


def map_iou_mean(Y_true, Y_pred):
    scores = [map_iou_single(t, p) for t, p in zip(Y_true, Y_pred)]
    return float(np.mean(scores)) if len(scores) else 0.0




## === cell 6
"""
Speed optimizations (provably equivalent, same core logic/metric):
1) Precompute folds once (same as before).
2) For each (n_bins, fold) precompute:
   - depth edges from training split
   - boolean flat validation masks once
   - per-bin mean masks (and global mean) once
   This removes huge redundant recomputation inside the grid search.
3) Vectorize IoU for the entire validation fold:
   - Convert predicted prior masks to boolean flat
   - Compute intersections via matrix multiplication (val_bool @ pred_bool)
   - Compute unions from sums (|A|+|B|-|A∩B|)
   This matches per-sample logical_and/or sums exactly.
4) Cache morphology output per (fold, bin, thr_eff, min_comp, hole_min) so expensive BFS runs only
   a small number of times (bins * parameter combos), not per validation image.
"""
train_df = pd.read_csv(TRAIN_CSV_PATH)
depths_df = pd.read_csv(DEPTHS_CSV_PATH)

if "rle_mask" not in train_df.columns:
    raise ValueError(f"Unexpected train.csv format: {train_df.columns.tolist()}")
if (
    "id" not in train_df.columns
    or "id" not in depths_df.columns
    or "z" not in depths_df.columns
):
    raise ValueError("depths.csv/train.csv missing required columns.")

train_df = train_df.merge(depths_df, on="id", how="left")
if train_df["z"].isna().any():
    train_df["z"] = train_df["z"].fillna(train_df["z"].median())

Y = np.zeros((len(train_df), IMG_SHAPE[0], IMG_SHAPE[1]), dtype=np.uint8)
for i, rle in enumerate(tqdm(train_df["rle_mask"].values, desc="Decoding train masks")):
    Y[i] = rle_decode(rle, shape=IMG_SHAPE)

coverage = Y.reshape(Y.shape[0], -1).mean(axis=1)
bins_cov = np.zeros_like(coverage, dtype=np.int32)
bins_cov[coverage > 0.0] = 1
bins_cov[coverage > 0.10] = 2
bins_cov[coverage > 0.30] = 3

z_train = train_df["z"].values.astype(np.float64)
idx_all = np.arange(len(train_df))

_THRESHOLDS_METRIC = np.arange(0.5, 1.0, 0.05).astype(np.float64)


def map_from_iou_vec(iou_vec, thresholds=_THRESHOLDS_METRIC):
    return float((iou_vec[:, None] > thresholds[None, :]).mean(axis=1).mean())


def build_depth_bin_edges(z_values, n_bins):
    if n_bins <= 1:
        return np.array([-np.inf, np.inf], dtype=np.float64)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(z_values, qs).astype(np.float64)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


_morph_cache = {}


def _mask_sha1(mask_u8):
    return hashlib.sha1(mask_u8.tobytes()).hexdigest()


def postprocess_mask(pred2d_u8, min_comp, hole_min):
    if min_comp <= 0 and hole_min <= 0:
        return pred2d_u8
    key = (_mask_sha1(pred2d_u8), int(min_comp), int(hole_min))
    cached = _morph_cache.get(key, None)
    if cached is not None:
        return cached
    out = pred2d_u8
    if min_comp > 0:
        out = remove_small_components(out, min_size=min_comp)
    if hole_min > 0:
        out = fill_small_holes(out, min_size=hole_min)
    _morph_cache[key] = out
    return out


thr_candidates = np.array(
    [0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36], dtype=np.float64
)
delta_candidates = np.array([-0.02, 0.00, 0.02], dtype=np.float64)

min_comp_candidates = [0, 10, 20, 30]
hole_fill_candidates = [0, 10, 20]
depth_bin_candidates = [3, 5, 7]

K = 4
rng = np.random.RandomState(42)
folds = [[] for _ in range(K)]
for b in np.unique(bins_cov):
    idx_b = idx_all[bins_cov == b].copy()
    rng.shuffle(idx_b)
    for i, ix in enumerate(idx_b):
        folds[i % K].append(int(ix))
folds = [np.array(f, dtype=int) for f in folds]

Y_bool_flat = Y.reshape(Y.shape[0], -1).astype(np.bool_)
Y_sum_flat = Y_bool_flat.sum(axis=1).astype(np.int32)

fold_precomp = {}  # (n_bins, k) -> dict

for n_bins in depth_bin_candidates:
    for k in range(K):
        val_idx = folds[k]
        tr_idx = np.concatenate([folds[j] for j in range(K) if j != k])

        edges = build_depth_bin_edges(z_train[tr_idx], n_bins=n_bins)

        val_bins = np.searchsorted(edges, z_train[val_idx], side="right") - 1
        val_bins = np.clip(val_bins, 0, n_bins - 1).astype(np.int32)

        mean_masks = []
        for b in range(n_bins):
            zlo, zhi = edges[b], edges[b + 1]
            idx_b = tr_idx[(z_train[tr_idx] > zlo) & (z_train[tr_idx] <= zhi)]
            if len(idx_b) == 0:
                mean_masks.append(None)
            else:
                mean_masks.append(Y[idx_b].mean(axis=0).astype(np.float64))
        global_mean = Y[tr_idx].mean(axis=0).astype(np.float64)

        val_true_bool = Y_bool_flat[val_idx]  # (n_val, P) bool
        val_true_sum = Y_sum_flat[val_idx]  # (n_val,) int

        fold_precomp[(n_bins, k)] = {
            "val_idx": val_idx,
            "tr_idx": tr_idx,
            "edges": edges,
            "val_bins": val_bins,
            "mean_masks": mean_masks,
            "global_mean": global_mean,
            "val_true_bool": val_true_bool,
            "val_true_sum": val_true_sum,
        }

best = None
best_params = None

cv_prior_cache = {}

P = IMG_SHAPE[0] * IMG_SHAPE[1]

for n_bins in depth_bin_candidates:
    for thr in thr_candidates:
        for delta in delta_candidates:
            thr_eff = float(thr + delta)
            for min_comp in min_comp_candidates:
                for hole_min in hole_fill_candidates:
                    cv_scores = []
                    for k in range(K):
                        pc = fold_precomp[(n_bins, k)]
                        val_bins = pc["val_bins"]
                        mean_masks = pc["mean_masks"]
                        global_mean = pc["global_mean"]
                        val_true_bool = pc["val_true_bool"]
                        val_true_sum = pc["val_true_sum"]

                        pred_bin_flat = [None] * n_bins
                        pred_bin_sum = np.zeros((n_bins,), dtype=np.int32)

                        for b in range(n_bins):
                            cache_key = (
                                n_bins,
                                k,
                                b,
                                thr_eff,
                                int(min_comp),
                                int(hole_min),
                            )
                            cached = cv_prior_cache.get(cache_key, None)
                            if cached is not None:
                                pred_bool_flat = cached
                            else:
                                mm = mean_masks[b]
                                if mm is None:
                                    mm = global_mean
                                prior_u8 = (mm >= thr_eff).astype(np.uint8)
                                prior_u8 = postprocess_mask(
                                    prior_u8, min_comp=min_comp, hole_min=hole_min
                                )
                                pred_bool_flat = prior_u8.reshape(-1).astype(np.bool_)
                                cv_prior_cache[cache_key] = pred_bool_flat

                            pred_bin_flat[b] = pred_bool_flat
                            pred_bin_sum[b] = int(pred_bool_flat.sum())

                        inter = np.zeros((len(val_bins),), dtype=np.int32)
                        union = np.zeros((len(val_bins),), dtype=np.int32)

                        for b in range(n_bins):
                            idxs = np.nonzero(val_bins == b)[0]
                            if idxs.size == 0:
                                continue
                            pred_vec = pred_bin_flat[b].astype(np.uint8)  # (P,)
                            inter_b = val_true_bool[idxs].astype(np.uint8) @ pred_vec
                            pred_sum_b = pred_bin_sum[b]
                            union_b = val_true_sum[idxs] + pred_sum_b - inter_b

                            inter[idxs] = inter_b.astype(np.int32)
                            union[idxs] = union_b.astype(np.int32)

                        iou = inter.astype(np.float64) / np.maximum(union, 1).astype(
                            np.float64
                        )
                        both0 = union == 0
                        iou[both0] = 1.0

                        cv_scores.append(map_from_iou_vec(iou))

                    score = float(np.mean(cv_scores))
                    if best is None or score > best:
                        best = score
                        best_params = (
                            int(n_bins),
                            float(thr),
                            float(delta),
                            int(min_comp),
                            int(hole_min),
                        )

N_BINS, THRESH, THRESH_DELTA, MIN_COMP_SIZE, HOLE_MIN_SIZE = best_params
print(
    "Chosen params (CV):",
    "N_BINS=",
    N_BINS,
    "THRESH=",
    THRESH,
    "DELTA=",
    THRESH_DELTA,
    "MIN_COMP_SIZE=",
    MIN_COMP_SIZE,
    "HOLE_MIN_SIZE=",
    HOLE_MIN_SIZE,
    "CV mAP=",
    best,
)

DEPTH_EDGES = build_depth_bin_edges(z_train, n_bins=N_BINS)
bin_mean_masks = []
for b in range(N_BINS):
    zlo, zhi = DEPTH_EDGES[b], DEPTH_EDGES[b + 1]
    idx_b = idx_all[(z_train > zlo) & (z_train <= zhi)]
    if len(idx_b) == 0:
        bin_mean_masks.append(None)
    else:
        bin_mean_masks.append(Y[idx_b].mean(axis=0).astype(np.float64))
global_mean_mask = Y.mean(axis=0).astype(np.float64)

print("Global mean mask coverage:", float(global_mean_mask.mean()))



## === cell 7
"""
Prepare submission frame from official sample_submission to ensure correct ids/order,
then fill rle_mask with our generated predictions.

Note: depths.csv contains depths for both train and test in this competition,
so we read it once and map by id.
"""
df = pd.read_csv(SAMPLE_SUB_PATH)
if "id" not in df.columns or "rle_mask" not in df.columns:
    raise ValueError(
        f"Unexpected submission format in {SAMPLE_SUB_PATH}: {df.columns.tolist()}"
    )

all_depths = pd.read_csv(DEPTHS_CSV_PATH).set_index("id")["z"].to_dict()

_bin_final_cache = {}


def get_prior_mask_for_id(img_id):
    z = all_depths.get(img_id, None)
    if z is None or (isinstance(z, float) and np.isnan(z)):
        b = None
    else:
        b = int(np.searchsorted(DEPTH_EDGES, float(z), side="right") - 1)
        b = max(0, min(N_BINS - 1, b))

    cache_key = "global" if b is None else int(b)
    cached = _bin_final_cache.get(cache_key, None)
    if cached is not None:
        return cached

    if b is None:
        mm = global_mean_mask
    else:
        mm = bin_mean_masks[b]
        if mm is None:
            mm = global_mean_mask

    thr_eff = float(THRESH + THRESH_DELTA)
    prior = (mm >= thr_eff).astype(np.uint8)
    if MIN_COMP_SIZE > 0:
        prior = remove_small_components(prior, min_size=MIN_COMP_SIZE)
    if HOLE_MIN_SIZE > 0:
        prior = fill_small_holes(prior, min_size=HOLE_MIN_SIZE)

    _bin_final_cache[cache_key] = prior
    return prior


ids = df["id"].values
rles = np.empty((len(ids),), dtype=object)

for i in tqdm(range(len(ids)), desc="Generating test predictions"):
    img_id = ids[i]
    mask = get_prior_mask_for_id(img_id)

    mask = crf(np.zeros(mask.shape, dtype=np.uint8), mask)
    rles[i] = rle_encode(mask)

df["rle_mask"] = rles



## === cell 8
"""
(Optional) visualize the global mean prior and an example depth-bin prior if matplotlib is available.
Non-essential for submission.
"""
if plt is not None:
    if N_BINS > 1:
        mid_b = N_BINS // 2
        disp_mm = (
            bin_mean_masks[mid_b]
            if bin_mean_masks[mid_b] is not None
            else global_mean_mask
        )
    else:
        disp_mm = global_mean_mask

    disp_pred = (disp_mm >= float(THRESH + THRESH_DELTA)).astype(np.uint8)
    if MIN_COMP_SIZE > 0:
        disp_pred = remove_small_components(disp_pred, min_size=MIN_COMP_SIZE)
    if HOLE_MIN_SIZE > 0:
        disp_pred = fill_small_holes(disp_pred, min_size=HOLE_MIN_SIZE)

    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(global_mean_mask, cmap="viridis")
    plt.title("Global mean mask")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(disp_mm, cmap="viridis")
    plt.title(f"Depth-bin mean (N_BINS={N_BINS})")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(disp_pred, cmap="gray")
    plt.title(
        f"Pred (t={THRESH:+.2f}{THRESH_DELTA:+.2f}, min_comp={MIN_COMP_SIZE}, hole_min={HOLE_MIN_SIZE})"
    )
    plt.axis("off")
    plt.tight_layout()



## === cell 9
"""
Write submission.

Must be a .csv and include columns: id, rle_mask
"""
out_path = "crf_correction.csv"
df[["id", "rle_mask"]].to_csv(out_path, index=False)

print(f"Wrote submission to: {out_path}")
print(df.head())
