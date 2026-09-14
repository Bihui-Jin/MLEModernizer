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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

2.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8686914475672408

# 6. Current score

0.33371

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The immediate crash comes from importing TensorFlow in an environment where protobuf/TensorFlow are incompatible, plus the script also references three external `/kaggle/input/...` model datasets that are not present, so inference can never run. To make the notebook run end-to-end and still produce a valid `submission.csv`, I remove the TensorFlow dependency entirely and replace the missing-model ensemble with a deterministic, legitimate baseline that predicts the most frequent class from `train.csv` (this preserves correct submission semantics for an accuracy metric). I also update the cells to be Python 2.7 compatible (no `tf.random.set_seed`, no TF/Keras generators), keep the original cell order, and ensure the output file is written to `/kaggle/working/submission.csv` with the required columns.'
- What this solution (achieved 0.11883) has done: 'Your current score (0.61099) is far below the target (0.86869), so we should improve accuracy while keeping your “no-TensorFlow” baseline structure. The smallest legitimate upgrade that preserves your overall approach (train.csv-driven inference and producing `submission.csv`) is to replace the constant majority-class predictor with a simple image-driven heuristic: compute per-class mean RGB “prototype” vectors from the training images, then classify each test image by nearest prototype in RGB space. This avoids any new ML libraries, keeps runtime under the limit by downsampling images, and should move accuracy substantially upward compared with a constant guess while keeping the rest of the pipeline intact. I also keep your existing cells (bi-tempered/gambler placeholders) and only add minimal image reading logic using PIL (with a safe fallback if PIL is unavailable).'
- What this solution (achieved 0.23019) has done: 'Your current score (0.11883) is far below the target (0.86869), so we should increase accuracy with the smallest changes that keep your no-TensorFlow, prototype-based inference core intact. The main issue is that a single mean-RGB prototype per class is too weak; we can significantly improve it by computing multiple prototypes per class (K-means in RGB-mean space implemented with NumPy only) and classifying by nearest prototype, while still using the same “downsample image -> mean RGB feature -> nearest prototype” logic. To keep runtime safe under 600s, we (1) avoid recomputing train features repeatedly, (2) cap per-class samples, and (3) compute test features once then vectorize distance computation per chunk. We also keep the majority-label fallback to guarantee a valid submission if PIL isn’t available or images fail to load.'
- What this solution (achieved 0.27728) has done: 'Your current score (0.23019) is far below the target (0.86869), so we need a legitimate accuracy jump while keeping the same “downsample image → simple feature → nearest prototype” core logic. The smallest high-impact change is to enrich the feature from mean RGB (3D) to a tiny “color+texture” descriptor (mean+std per channel = 6D), and to make prototype matching more robust by whitening (z-scoring) features using train statistics before K-means and nearest-prototype search. This keeps the same pipeline structure (feature extraction + per-class K-means prototypes + nearest prototype inference), but usually improves separability a lot for cassava leaves. I also make the training subset selection deterministic but more representative via stratified sampling without changing caps or adding new dependencies, and keep the same submission writing logic.'
- What this solution (achieved 0.27354) has done: 'Your current score (0.27728) is far below the target (0.86869), so we should increase accuracy while keeping the same “downsample image → simple feature → per-class K-means prototypes → nearest-prototype” core logic. The biggest low-risk gain is to make the feature slightly more discriminative without changing the approach: add a tiny HSV mean+std block (6 more dims) and a simple “green-ness” ratio (1 dim), then keep the same z-scoring + K-means + nearest-prototype classifier. To avoid making things worse via class-imbalance effects, we also switch the subset selection to be balanced per class (same cap, but enforced) and use a few more prototypes per class (small change in the same prototype idea) while keeping runtime bounded. All changes are confined to feature extraction + prototype config; submission writing and overall pipeline stay the same.'
- What this solution (achieved 0.30531) has done: 'Your current score (0.27354) is far below the target (0.86869), so we need a meaningful accuracy increase while keeping the same core pipeline: global-stat image features → per-class K-means prototypes → nearest-prototype prediction → `submission.csv`. The minimal high-impact fix is to make the feature a bit more shape/texture-aware (still global, still cheap) by adding a small grayscale histogram + gradient magnitude statistics, while keeping your existing RGB/HSV/green-ratio block unchanged. To prevent the extra dimensions from hurting distances, we keep the same z-scoring but compute it after building the full feature vector, and we also make K-means initialization slightly more stable (k-means++ style) without changing the model family. All I/O paths and the submission-writing logic remain unchanged.'
- What this solution (achieved 0.29671) has done: 'Your current score (0.30531) is far below the target (0.86869), so we should improve accuracy while keeping the exact same overall pipeline: global image features → z-score → per-class K-means prototypes → nearest-prototype inference → submission.csv. The smallest high-impact change that stays within this core logic is to add a tiny “spatial/layout” signal without moving to a different model: append low-resolution grayscale block means (a 4×4 grid = 16D) to the existing 23D feature, yielding a 39D global descriptor that captures lesion placement/patchiness. To avoid the added dimensions dominating distances, we keep the same z-scoring but compute it over the new full feature vector (same as before, just extended). Everything else (paths, caps, K-means, distance computation, CSV writing) remains unchanged.'
- What this solution (achieved 0.33371) has done: 'Your current score (0.29671) is far below the target (0.86869), so we should increase accuracy while keeping the exact same pipeline (global handcrafted feature → z-score → per-class k-means prototypes → nearest-prototype). The smallest high-impact fix within this core logic is to add a coarse shape cue that is still “global stats”: a tiny low-resolution edge/gradient grid (4×4 block means) appended to the feature, which helps separate diseases with similar color but different lesion patterns. To make the added dimensions behave well under Euclidean distance, we keep the same z-scoring (now computed over the extended feature) and keep everything else (caps, k-means, inference, I/O) unchanged. This should move accuracy upward without introducing new dependencies or changing the modeling family.'

# 9. Code solution

## === cell 0
import os
import sys
import numpy as np
import pandas as pd

np.random.seed(42)

print("Python:", sys.version)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
if not os.path.isdir(DATA_DIR):
    alt = "/kaggle/data/cassava-leaf-disease-classification"
    if os.path.isdir(alt):
        DATA_DIR = alt

print("Using DATA_DIR:", DATA_DIR)



## === cell 1
"""
Bi-Tempered loss implementation (kept for minimal structural change from the original notebook).

Not used in this no-TensorFlow inference-only baseline because TensorFlow cannot be imported
in this environment (protobuf incompatibility). Kept to preserve original cell structure.
"""
pass



## === cell 2
"""
Gambler's loss helpers (kept for minimal structural change from the original notebook).

Not used in this no-TensorFlow inference-only baseline because TensorFlow cannot be imported.
"""
pass



## === cell 3
"""
Original notebook tried to load external SavedModels via TFSMLayer from datasets that are not present.

Change (score-improving, minimal core-logic impact):
- Keep the same image -> downsample -> simple feature -> prototype classifier approach.
- Keep existing global feature family (color/texture aggregates + coarse spatial cue).
- Add a tiny *edge-layout* cue WITHOUT changing the model family:
    * 4x4 grid of gradient-magnitude block means -> 16D
  This is still a cheap global-stat feature and still feeds the same
  z-scoring + per-class K-means + nearest-prototype classifier.
- Keep k-means++ initialization for stability (unchanged).
"""
train_csv_path = os.path.join(DATA_DIR, "train.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
train_img_dir = os.path.join(DATA_DIR, "train_images")
test_img_dir = os.path.join(DATA_DIR, "test_images")

for p in [train_csv_path, sample_path]:
    if not os.path.exists(p):
        raise IOError("Missing required file at: {}".format(p))

if not os.path.isdir(train_img_dir):
    alt_train = os.path.join(
        DATA_DIR, "cassava-leaf-disease-classification", "train_images"
    )
    if os.path.isdir(alt_train):
        train_img_dir = alt_train
if not os.path.isdir(test_img_dir):
    alt_test = os.path.join(
        DATA_DIR, "cassava-leaf-disease-classification", "test_images"
    )
    if os.path.isdir(alt_test):
        test_img_dir = alt_test

if not os.path.isdir(train_img_dir):
    raise IOError("Missing train_images directory at: {}".format(train_img_dir))
if not os.path.isdir(test_img_dir):
    raise IOError("Missing test_images directory at: {}".format(test_img_dir))

train_df = pd.read_csv(train_csv_path)
if "label" not in train_df.columns or "image_id" not in train_df.columns:
    raise ValueError("train.csv must contain columns: image_id, label")

label_counts = train_df["label"].value_counts()
majority_label = int(label_counts.idxmax())

print("Train rows:", train_df.shape[0])
print("Label distribution:\n{}".format(label_counts.to_string()))
print("Majority label (fallback):", majority_label)



## === cell 4
"""
Compute features and multiple prototypes (K-means) per class.

Why this should move score toward target (minimal change while keeping the same core logic):
- Add a 4x4 gradient-magnitude block-mean grid (edge-layout) to capture where lesions/edges occur.
- Still uses the same pipeline: global feature -> z-score -> per-class K-means -> nearest-prototype.
"""
try:
    from PIL import Image

    PIL_OK = True
except Exception as e:
    PIL_OK = False
    pil_err = str(e)


def _rgb_to_hsv_np01(rgb01):
    r = rgb01[:, 0]
    g = rgb01[:, 1]
    b = rgb01[:, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn

    v = mx

    s = np.zeros_like(mx)
    mask = mx > 1e-12
    s[mask] = diff[mask] / mx[mask]

    h = np.zeros_like(mx)
    nonzero = diff > 1e-12

    m = nonzero & (mx == r)
    h[m] = ((g[m] - b[m]) / diff[m]) % 6.0
    m = nonzero & (mx == g)
    h[m] = ((b[m] - r[m]) / diff[m]) + 2.0
    m = nonzero & (mx == b)
    h[m] = ((r[m] - g[m]) / diff[m]) + 4.0

    h = (h / 6.0) % 1.0
    hsv = np.stack([h, s, v], axis=1).astype(np.float32)
    return hsv


def _block_means_2d(mat, grid):
    H, W = mat.shape
    out = np.empty((grid, grid), dtype=np.float32)
    for i in range(grid):
        y0 = int(round(i * H / float(grid)))
        y1 = int(round((i + 1) * H / float(grid)))
        if y1 <= y0:
            y1 = min(H, y0 + 1)
        for j in range(grid):
            x0 = int(round(j * W / float(grid)))
            x1 = int(round((j + 1) * W / float(grid)))
            if x1 <= x0:
                x1 = min(W, x0 + 1)
            blk = mat[y0:y1, x0:x1]
            out[i, j] = float(blk.mean()) if blk.size else 0.0
    return out.reshape((-1,)).astype(np.float32)


def image_feat_global_stats(path, size=(96, 96), gray_bins=8, grid_gray=4, grid_gm=4):
    """
    Returns 55D feature:
      RGB mean(3), RGB std(3),
      HSV mean(3), HSV std(3),
      green_ratio(1),
      gray_hist(8),
      grad_mag mean/std (2),
      gray_block_means grid_gray*grid_gray (16 when grid_gray=4),
      gradmag_block_means grid_gm*grid_gm (16 when grid_gm=4)
    """
    if not PIL_OK:
        return None
    try:
        im = Image.open(path).convert("RGB")
        if size is not None:
            im = im.resize(size, resample=Image.BILINEAR)

        arr = np.asarray(im, dtype=np.float32)  # (H,W,3)
        flat = arr.reshape((-1, 3))

        mu_rgb = flat.mean(axis=0)
        sd_rgb = flat.std(axis=0)

        rgb01 = (flat / 255.0).astype(np.float32)
        hsv = _rgb_to_hsv_np01(rgb01)
        mu_hsv = hsv.mean(axis=0)
        sd_hsv = hsv.std(axis=0)

        denom = float(mu_rgb.sum()) + 1e-6
        green_ratio = np.array([mu_rgb[1] / denom], dtype=np.float32)

        gray = (
            0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
        ) / 255.0
        gray = gray.astype(np.float32)

        hist, _ = np.histogram(gray.ravel(), bins=gray_bins, range=(0.0, 1.0))
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-6)

        gx = gray[:, 1:] - gray[:, :-1]
        gy = gray[1:, :] - gray[:-1, :]
        gx = gx[:-1, :]
        gy = gy[:, :-1]
        gm = np.sqrt(gx * gx + gy * gy).astype(np.float32)
        gm_mu = np.array([gm.mean()], dtype=np.float32)
        gm_sd = np.array([gm.std()], dtype=np.float32)

        gimg = Image.fromarray((gray * 255.0).astype(np.uint8), mode="L")
        gimg = gimg.resize((grid_gray, grid_gray), resample=Image.BILINEAR)
        garr = (
            (np.asarray(gimg, dtype=np.float32) / 255.0)
            .reshape((-1,))
            .astype(np.float32)
        )

        gm_grid = _block_means_2d(gm, grid=grid_gm)

        feat = np.concatenate(
            [
                mu_rgb,
                sd_rgb,
                mu_hsv,
                sd_hsv,
                green_ratio,
                hist,
                gm_mu,
                gm_sd,
                garr,
                gm_grid,
            ],
            axis=0,
        )
        return feat.astype(np.float32)
    except Exception:
        return None


def _kmeanspp_init(X, k, rs):
    """
    NumPy-only k-means++ init (same K-means model family; more stable seeds).
    X: (N,D)
    """
    N = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=np.float32)
    idx0 = int(rs.randint(0, N))
    centers[0] = X[idx0]
    dist2 = ((X - centers[0][None, :]) ** 2).sum(axis=1)

    for i in range(1, k):
        probs = dist2 / (dist2.sum() + 1e-12)
        r = rs.rand()
        cdf = np.cumsum(probs)
        idx = int(np.searchsorted(cdf, r))
        if idx >= N:
            idx = N - 1
        centers[i] = X[idx]
        newd2 = ((X - centers[i][None, :]) ** 2).sum(axis=1)
        dist2 = np.minimum(dist2, newd2)
    return centers.astype(np.float32)


def kmeans_np(X, k, n_iter=25, seed=42):
    rs = np.random.RandomState(seed)
    N = X.shape[0]
    if N == 0:
        return None
    if N <= k:
        centers = X.copy()
        if centers.shape[0] < k:
            pad = centers[rs.randint(0, centers.shape[0], size=(k - centers.shape[0],))]
            centers = np.vstack([centers, pad])
        return centers.astype(np.float32)

    centers = _kmeanspp_init(X, k=k, rs=rs)

    for _ in range(n_iter):
        d = X[:, None, :] - centers[None, :, :]
        dist2 = (d * d).sum(axis=2)
        assign = dist2.argmin(axis=1)

        new_centers = np.empty_like(centers)
        for j in range(k):
            mask = assign == j
            if mask.any():
                new_centers[j] = X[mask].mean(axis=0)
            else:
                new_centers[j] = X[rs.randint(0, N)]
        centers = new_centers.astype(np.float32)

    return centers.astype(np.float32)


if not PIL_OK:
    print(
        "WARNING: PIL not available ({}). Falling back to majority-class predictions.".format(
            pil_err
        )
    )
    proto_centers = None
    proto_labels = None
    feat_mu = None
    feat_sd = None
    FEAT_DIM = 55
else:
    img_size = (96, 96)

    per_class_cap = 1600
    k_per_class = 8
    kmeans_iters = 30

    class_ids = sorted(train_df["label"].unique().tolist())

    parts = []
    for c in class_ids:
        dfc = train_df[train_df["label"] == c]
        n = int(min(per_class_cap, dfc.shape[0]))
        dfc = dfc.sample(frac=1.0, random_state=42).head(n)
        parts.append(dfc)
    sub_train = pd.concat(parts, axis=0).reset_index(drop=True)
    print("Prototype training subset size:", sub_train.shape[0])
    print("Per-class subset sizes:", sub_train["label"].value_counts().to_dict())

    feats = []
    labs = []
    missing = 0
    for i in range(sub_train.shape[0]):
        img_id = sub_train.at[i, "image_id"]
        c = int(sub_train.at[i, "label"])
        p = os.path.join(train_img_dir, img_id)
        f = image_feat_global_stats(
            p, size=img_size, gray_bins=8, grid_gray=4, grid_gm=4
        )
        if f is None:
            missing += 1
            continue
        feats.append(f)
        labs.append(c)

    if len(feats) == 0:
        print(
            "WARNING: no training features could be read; falling back to majority-class."
        )
        proto_centers = None
        proto_labels = None
        feat_mu = None
        feat_sd = None
        FEAT_DIM = 55
    else:
        X = np.stack(feats, axis=0).astype(np.float32)
        y = np.array(labs, dtype=np.int64)
        FEAT_DIM = int(X.shape[1])
        print("Training features shape:", X.shape, "Unreadable/missing:", missing)

        feat_mu = X.mean(axis=0).astype(np.float32)
        feat_sd = X.std(axis=0).astype(np.float32)
        feat_sd = np.maximum(feat_sd, 1e-6).astype(np.float32)

        Xn = (X - feat_mu[None, :]) / feat_sd[None, :]

        centers_list = []
        labels_list = []
        for c in class_ids:
            Xc = Xn[y == int(c)]
            if Xc.shape[0] == 0:
                continue
            centers = kmeans_np(
                Xc, k=k_per_class, n_iter=kmeans_iters, seed=42 + int(c)
            )
            if centers is None:
                continue
            centers_list.append(centers)
            labels_list.append(np.full((centers.shape[0],), int(c), dtype=np.int64))

        if len(centers_list) == 0:
            proto_centers = None
            proto_labels = None
        else:
            proto_centers = np.vstack(centers_list).astype(np.float32)
            proto_labels = np.concatenate(labels_list).astype(np.int64)
            print(
                "Total prototypes:",
                proto_centers.shape[0],
                "Feature dim:",
                FEAT_DIM,
                "Prototype labels:",
                np.unique(proto_labels),
            )



## === cell 5
"""
Load test list from sample_submission.csv (same as before).
"""
sample_sub = pd.read_csv(sample_path)
if "image_id" not in sample_sub.columns or "label" not in sample_sub.columns:
    raise ValueError("sample_submission.csv must contain columns: image_id, label")

test_df = sample_sub[["image_id"]].copy()
print("Test images:", len(test_df))



## === cell 6
"""
Generate predictions for each test image.

Why this should improve score (minimal change):
- Same nearest-prototype rule and chunked vectorized distance computation.
- Uses the slightly richer (still global-stat) feature vector with added edge-layout grid.
"""
if (
    proto_centers is None
    or proto_labels is None
    or feat_mu is None
    or feat_sd is None
    or (not PIL_OK)
):
    predicted_class_indices_new = np.full(
        (len(test_df),), majority_label, dtype=np.int64
    )
else:
    img_size = (96, 96)
    test_feats = np.empty((len(test_df), FEAT_DIM), dtype=np.float32)
    ok_mask = np.ones((len(test_df),), dtype=np.bool_)
    unreadable = 0

    for i in range(len(test_df)):
        img_id = test_df.at[i, "image_id"]
        p = os.path.join(test_img_dir, img_id)
        f = image_feat_global_stats(
            p, size=img_size, gray_bins=8, grid_gray=4, grid_gm=4
        )
        if f is None:
            ok_mask[i] = False
            test_feats[i] = 0.0
            unreadable += 1
        else:
            test_feats[i] = f

    test_feats = (test_feats - feat_mu[None, :]) / feat_sd[None, :]

    preds = np.full((len(test_df),), majority_label, dtype=np.int64)

    chunk = 512
    for s in range(0, len(test_df), chunk):
        e = min(len(test_df), s + chunk)
        Xb = test_feats[s:e]
        maskb = ok_mask[s:e]
        if not maskb.any():
            continue
        d = Xb[:, None, :] - proto_centers[None, :, :]
        dist2 = (d * d).sum(axis=2)
        nn = dist2.argmin(axis=1)
        pb = proto_labels[nn]
        out = preds[s:e]
        out[maskb] = pb[maskb]
        preds[s:e] = out

    predicted_class_indices_new = preds
    print(
        "Unreadable/missing test images (fell back to majority for those):", unreadable
    )

print(
    "Predictions generated. Unique predicted labels:",
    np.unique(predicted_class_indices_new),
)



## === cell 7
"""
Write submission.csv in the required format (unchanged).
"""
submission = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": predicted_class_indices_new}
)

assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["image_id", "label"]

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
