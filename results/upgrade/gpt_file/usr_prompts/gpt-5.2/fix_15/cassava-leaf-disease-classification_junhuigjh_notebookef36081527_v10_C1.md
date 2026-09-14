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

3.13

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

0.7893623451193714

# 6. Current score

0.29933

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the execution blockers by (1) removing the TFRecord reading path that triggers the protobuf `MessageFactory.GetPrototype` error in this environment and instead reading test images directly from `test_images/`, and (2) ensuring all constants/paths are defined in the first cell so later cells don’t crash with `NameError`. I also make the model loading robust by falling back to a simple majority-class submission if the referenced model file is not available, so a valid `submission.csv` is always produced. These changes preserve the core inference logic (Keras model predicts argmax over 5 classes) while making the pipeline run end-to-end and generate a correctly formatted submission.'
- What this solution (achieved 0.61099) has done: 'I fix the execution-blocking protobuf/Keras import issue by removing the unused `torch` import and delaying Keras imports until after basic setup, which avoids triggering the `MessageFactory.GetPrototype` error in this Kaggle environment. I also correct the cell numbering to start at 1 (your current script starts at cell 0), so it matches the expected notebook-like format and executes cleanly. To move the score toward your target (since 0.61099 is well below 0.7893), I keep the same “load Keras model → preprocess → predict argmax” core logic but add a small, score-improving test-time augmentation (original + horizontal flip averaged) and enable safe mixed-precision inference if available. Finally, I keep the same submission merge logic and always write a valid `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'I fix the execution blocker caused by the protobuf/Keras import crash (`MessageFactory.GetPrototype`) by removing the hard dependency on Keras in this environment and switching to a stable TensorFlow/Keras import path only if it can be imported successfully. If TensorFlow/Keras cannot be imported (likely under Python 3.13 here), the script still run end-to-end and generate a valid `submission.csv` using the majority-class fallback (so you always get a valid file). I also correct the cell numbering to start at 1 as required and keep the rest of the inference logic (preprocess → optional TTA → argmax over 5 classes) unchanged. This primarily fixes runtime stability; score only improve if the environment can actually load the provided `.keras` model.'
- What this solution (achieved 0.13789) has done: 'I fix the execution blocker caused by the protobuf/TensorFlow import crash by avoiding TensorFlow/Keras entirely in this Python 3.13 environment and using a stable, end-to-end classical image baseline instead (so the script always runs and writes `submission.csv`). To move the score upward from 0.61099 toward the 0.789 target, I replace the current majority-class fallback with a lightweight nearest-centroid classifier on simple color features computed from the provided `train_images/`, then apply it to `test_images/`. This keeps the overall “read images → compute features → predict class id → write submission.csv” inference semantics, but removes the broken dependency and should legitimately improve accuracy beyond the majority-class. I also fix the cell numbering to start at 1 and keep all paths consistent with the provided dataset layout.'
- What this solution (achieved 0.23991) has done: 'Your current score (0.13789) is far below the target (0.78936), so we should make small, legitimate improvements that keep the same “simple handcrafted features → nearest-centroid by cosine similarity → argmax” core logic. The main issue is that the current features are too weak; we add a few additional low-cost global color/texture statistics (still computed from the same images, same pipeline) and compute centroids in a more robust standardized feature space (z-score using train mean/std), which usually improves separability a lot for this kind of baseline. We also make inference/training deterministic by sorting file lists and using the same seed, and keep the exact same submission merge/format to avoid accidental misalignment. These are minimal changes that should move accuracy upward toward your target without changing the overall approach.'
- What this solution (achieved 0.31278) has done: 'The timeout is dominated by repeated PIL decode/resize/conversion work inside `_compute_features`, done once per training image and twice per test image (original + flip), plus Python-loop overhead in prototype fitting. I keep the same handcrafted feature vector, z-scoring, L2 normalization, cosine-sim prototype scoring, and the same caps/prototypes/iters, but eliminate redundant PIL conversions by computing all “small” representations from one resize and doing HSV conversion on the already-small image. I also make k-means fully vectorized (removing the per-cluster Python loop) while keeping identical assignment/update semantics, and parallelize feature extraction with a deterministic thread pool (PIL releases the GIL during image decode/resize), which preserves accuracy but cuts wall time substantially. Finally, I avoid repeated `os.path.exists` calls by precomputing available filenames sets for train/test directories.'
- What this solution (achieved 0.30269) has done: 'Your current score (0.31278) is far below the target (0.78936), so we should make small, legitimate accuracy improvements while keeping your exact core pipeline (handcrafted features → z-score → L2 normalize → cosine-sim prototypes via per-class k-means → argmax, with the same train cap and TTA). The biggest low-risk gain here is to use more prototypes per class (still the same prototype method), which better covers intra-class variation and usually improves accuracy substantially without changing evaluation semantics. I also add a deterministic tiny “sharpen” TTA (original + hflip + sharpened averaged) to help textures/lesions, and add a stable tie-break/temperature scaling on similarities to reduce overconfident wrong picks (still argmax over class scores). All paths and the submission format stay identical, and runtime should remain within limits due to small feature dimensionality and capped training set.'
- What this solution (achieved 0.32287) has done: 'Your current score (0.30269) is far below the target (0.78936), so we should make the smallest accuracy-oriented adjustments that keep the exact same core pipeline: handcrafted global features → z-score → L2 normalize → cosine similarity to per-class k-means prototypes → argmax → submission.csv. The most likely low-risk gain is that your “sharpen” TTA is mismatched to the training feature space (train prototypes are built from non-sharpened images), so we remove sharpen TTA and instead use only the already-aligned original + horizontal flip feature averaging. To better cover intra-class variation without changing the method, we modestly increase the number of prototypes per class (still k-means cosine prototypes) and keep everything deterministic (sorted file lists, fixed seeds) to stabilize the result. These changes preserve evaluation semantics and should move accuracy upward toward the target without rewriting the approach.'
- What this solution (achieved 0.31839) has done: 'Your current score (0.32287) is far below the target (0.78936), so we should make a small, legitimate accuracy improvement while keeping your exact pipeline (handcrafted features → z-score → L2 normalize → cosine k-means prototypes → argmax). The biggest low-risk issue is that you compute z-score statistics on the *full-image* feature distribution but then predict on a *TTA-averaged* feature distribution, which creates a train/test normalization mismatch; we fix this by fitting `mu, sd` on the same hflip-TTA features used at inference. To avoid changing the model logic, we reuse your existing feature extractor and just compute both train and test features consistently (orig+hflip average), keeping prototypes, temperature, and submission formatting unchanged. This should move accuracy upward toward your target without altering evaluation semantics or adding new dependencies.'
- What this solution (achieved 0.29933) has done: 'The timeout is dominated by per-image PIL work in `_compute_features_hflip_tta`, which currently performs multiple PIL conversions/resizes per region and repeats those operations again for horizontal-flip TTA. I keep the exact feature definitions and prototype-scoring logic, but remove redundant PIL work by computing small RGB/HSV/gray once per full image and once per flip, then slicing those precomputed arrays for each region (instead of resizing/converting each region separately). I also precompute the 2×2 region slices once and reuse them, and avoid unnecessary array allocations in normalization/application steps, while keeping all math identical (same bilinear resize sizes and HSV conversion). These changes preserve the core algorithm and evaluation semantics but significantly reduce constant factors so the pipeline fits within 600 seconds.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

SEED = 42
rng = np.random.default_rng(SEED)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV_PATH = f"{DATA_ROOT}/train.csv"
SAMPLE_SUB_PATH = f"{DATA_ROOT}/sample_submission.csv"
TRAIN_IMG_DIR = f"{DATA_ROOT}/train_images"
TEST_IMG_DIR = f"{DATA_ROOT}/test_images"

print("Python:", os.sys.version)
print("DATA_ROOT exists:", os.path.exists(DATA_ROOT))
print("TRAIN_CSV_PATH exists:", os.path.exists(TRAIN_CSV_PATH))
print("SAMPLE_SUB_PATH exists:", os.path.exists(SAMPLE_SUB_PATH))
print("TRAIN_IMG_DIR exists:", os.path.exists(TRAIN_IMG_DIR))
print("TEST_IMG_DIR exists:", os.path.exists(TEST_IMG_DIR))
print("Working dir:", os.getcwd())

MAX_WORKERS = min(8, (os.cpu_count() or 2))




## === cell 1
def _read_rgb(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        return np.asarray(im, dtype=np.uint8)


def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)


def _zscore_fit(X: np.ndarray, eps: float = 1e-6):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def _zscore_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((X.astype(np.float32, copy=False) - mu) / sd).astype(np.float32, copy=False)


def _kmeans_cosine(
    X_unit: np.ndarray, k: int, iters: int = 12, seed: int = 42
) -> np.ndarray:
    n, d = X_unit.shape
    if n == 0:
        raise ValueError("Empty X for kmeans.")
    if n <= k:
        centers = X_unit.copy()
        while centers.shape[0] < k:
            centers = np.vstack([centers, centers[: (k - centers.shape[0])]])
        return _l2_normalize(centers[:k]).astype(np.float32)

    local_rng = np.random.default_rng(seed)
    init_idx = local_rng.choice(n, size=k, replace=False)
    centers = X_unit[init_idx].copy()

    for _ in range(iters):
        sims = X_unit @ centers.T  # (n,k)
        assign = np.argmax(sims, axis=1).astype(np.int64)  # (n,)

        counts = np.bincount(assign, minlength=k).astype(np.float32)  # (k,)
        sums = np.zeros((k, d), dtype=np.float32)
        np.add.at(sums, assign, X_unit)  # exact accumulation per cluster

        new_centers = centers.copy()
        nonempty = counts > 0
        new_centers[nonempty] = sums[nonempty] / counts[nonempty, None]

        empty_idx = np.where(~nonempty)[0]
        if empty_idx.size:
            repl = local_rng.integers(0, n, size=empty_idx.size)
            new_centers[empty_idx] = X_unit[repl]

        centers = _l2_normalize(new_centers)
    return centers.astype(np.float32)


def _compute_features_hflip_tta(img_rgb: np.ndarray) -> np.ndarray:
    def _prep_small_views(rgb_u8: np.ndarray):
        im = Image.fromarray(rgb_u8, mode="RGB")
        small_rgb_u8 = np.asarray(
            im.resize((32, 32), resample=Image.BILINEAR), dtype=np.uint8
        )
        small_rgb = small_rgb_u8.astype(np.float32) / 255.0  # (32,32,3)

        hsv_small_u8 = np.asarray(
            Image.fromarray(small_rgb_u8, mode="RGB").convert("HSV"), dtype=np.uint8
        )
        hsv_small = hsv_small_u8.astype(np.float32) / 255.0  # (32,32,3)

        gray_full = rgb_u8.astype(np.float32) / 255.0
        gray_full = (
            gray_full[..., 0] * 0.299
            + gray_full[..., 1] * 0.587
            + gray_full[..., 2] * 0.114
        )
        g_im = Image.fromarray((gray_full * 255.0).astype(np.uint8), mode="L")
        gsmall = (
            np.asarray(g_im.resize((32, 32), resample=Image.BILINEAR), dtype=np.float32)
            / 255.0
        )  # (32,32)

        return small_rgb, hsv_small, gsmall

    def _region_feat(
        region_u8: np.ndarray,
        small_rgb_region: np.ndarray,  # (h,w,3) in [0,1]
        hsv_small_region: np.ndarray,  # (h,w,3) in [0,1]
        gsmall_region: np.ndarray,  # (h,w) in [0,1]
    ) -> np.ndarray:
        x = region_u8.astype(np.float32) / 255.0
        flat = x.reshape(-1, 3)

        mean = flat.mean(axis=0)
        std = flat.std(axis=0) + 1e-12

        s = small_rgb_region.reshape(-1, 3)
        s_mean = s.mean(axis=0)

        r, g, b = mean
        greenness = g - (r + b) / 2.0
        yellowness = (r + g) / 2.0 - b

        hsv = hsv_small_region.reshape(-1, 3)
        hsv_mean = hsv.mean(axis=0)
        hsv_std = hsv.std(axis=0)

        z = (flat - mean[None, :]) / std[None, :]
        skew = (z**3).mean(axis=0)

        gx = np.diff(gsmall_region, axis=1, append=gsmall_region[:, -1:])
        gy = np.diff(gsmall_region, axis=0, append=gsmall_region[-1:, :])
        grad = np.sqrt(gx * gx + gy * gy)
        grad_mean = float(grad.mean())
        grad_std = float(grad.std())

        feat = np.concatenate(
            [
                mean,
                std - 1e-12,
                s_mean,
                np.array([greenness, yellowness], dtype=np.float32),
                hsv_mean,
                hsv_std,
                skew,
                np.array([grad_mean, grad_std], dtype=np.float32),
            ],
            axis=0,
        )
        return feat.astype(np.float32)

    def _grid_slices(H, W):
        h2 = H // 2
        w2 = W // 2
        y0, y1, y2 = 0, max(1, h2), H
        x0, x1, x2 = 0, max(1, w2), W
        return (
            (slice(y0, y1), slice(x0, x1)),
            (slice(y0, y1), slice(x1, x2)),
            (slice(y1, y2), slice(x0, x1)),
            (slice(y1, y2), slice(x1, x2)),
        )

    def _grid_slices_32():
        return _grid_slices(32, 32)

    def _grid2x2_concat(rgb_u8: np.ndarray) -> np.ndarray:
        small_rgb, hsv_small, gsmall = _prep_small_views(rgb_u8)

        (tl, tr, bl, br) = _grid_slices(rgb_u8.shape[0], rgb_u8.shape[1])
        (stl, str_, sbl, sbr) = _grid_slices_32()

        f_global = _region_feat(rgb_u8, small_rgb, hsv_small, gsmall)

        f_tl = _region_feat(
            rgb_u8[tl[0], tl[1], :],
            small_rgb[stl[0], stl[1], :],
            hsv_small[stl[0], stl[1], :],
            gsmall[stl[0], stl[1]],
        )
        f_tr = _region_feat(
            rgb_u8[tr[0], tr[1], :],
            small_rgb[str_[0], str_[1], :],
            hsv_small[str_[0], str_[1], :],
            gsmall[str_[0], str_[1]],
        )
        f_bl = _region_feat(
            rgb_u8[bl[0], bl[1], :],
            small_rgb[sbl[0], sbl[1], :],
            hsv_small[sbl[0], sbl[1], :],
            gsmall[sbl[0], sbl[1]],
        )
        f_br = _region_feat(
            rgb_u8[br[0], br[1], :],
            small_rgb[sbr[0], sbr[1], :],
            hsv_small[sbr[0], sbr[1], :],
            gsmall[sbr[0], sbr[1]],
        )

        return np.concatenate([f_global, f_tl, f_tr, f_bl, f_br], axis=0).astype(
            np.float32
        )

    feat1 = _grid2x2_concat(img_rgb)
    feat2 = _grid2x2_concat(img_rgb[:, ::-1, :])
    return (0.5 * (feat1 + feat2)).astype(np.float32)




## === cell 2
train_df = pd.read_csv(TRAIN_CSV_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

test_image_ids = sample_sub["image_id"].astype(str).tolist()

PER_CLASS_CAP = 1200  # keep same cap to preserve runtime/behavior envelope

train_df = train_df.copy()
train_df["image_id"] = train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(int)

selected_idx = []
for c in sorted(train_df["label"].unique()):
    idx = train_df.index[train_df["label"] == c].to_numpy()
    if len(idx) > PER_CLASS_CAP:
        idx = rng.choice(idx, size=PER_CLASS_CAP, replace=False)
    selected_idx.append(idx)
selected_idx = np.concatenate(selected_idx, axis=0)
train_sub = train_df.loc[selected_idx].reset_index(drop=True)

print("Train total:", len(train_df), "Train used:", len(train_sub))
print("Train used per class:", train_sub["label"].value_counts().sort_index().to_dict())


def _feat_from_train_row(args):
    image_id, label = args
    p = os.path.join(TRAIN_IMG_DIR, image_id)
    if not os.path.isfile(p):
        return None
    img = _read_rgb(p)
    feat = _compute_features_hflip_tta(img)
    return feat, int(label)


train_args = list(zip(train_sub["image_id"].tolist(), train_sub["label"].tolist()))
X_list = []
y_list = []
missing_train = 0

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    for out in ex.map(_feat_from_train_row, train_args, chunksize=64):
        if out is None:
            missing_train += 1
        else:
            feat, label = out
            X_list.append(feat)
            y_list.append(label)

if missing_train:
    print("WARNING: missing train images:", missing_train)

X_train = np.stack(X_list, axis=0).astype(np.float32)
y_train = np.array(y_list, dtype=np.int64)

mu, sd = _zscore_fit(X_train)
X_train_z = _zscore_apply(X_train, mu, sd)
X_train_unit = _l2_normalize(X_train_z)

num_classes = 5

K_PROTOS = 12
protos = np.zeros((num_classes, K_PROTOS, X_train_unit.shape[1]), dtype=np.float32)
counts = np.zeros((num_classes,), dtype=np.int64)

for c in range(num_classes):
    mask = y_train == c
    Xc = X_train_unit[mask]
    counts[c] = int(Xc.shape[0])
    if Xc.shape[0] == 0:
        global_center = _l2_normalize(X_train_unit.mean(axis=0, keepdims=True))[0]
        protos[c, :, :] = global_center[None, :]
    elif Xc.shape[0] == 1:
        protos[c, :, :] = Xc[0][None, :]
    else:
        protos[c] = _kmeans_cosine(Xc, k=K_PROTOS, iters=12, seed=SEED + 1000 * c)

print("Prototype counts per class:", counts.tolist())

majority_label = int(train_df["label"].mode().iloc[0])
print("Majority label:", majority_label)

SIM_TEMPERATURE = 0.7  # keep as-is to preserve the current scoring behavior envelope




## === cell 3
image_ids_out = []
preds_out = []
missing_test = 0
unreadable_test = 0


def _predict_one(image_name: str):
    img_path = os.path.join(TEST_IMG_DIR, image_name)
    if not os.path.isfile(img_path):
        return image_name, majority_label, "missing"
    try:
        img = _read_rgb(img_path)

        feat = _compute_features_hflip_tta(img)
        feat = _zscore_apply(feat[None, :], mu, sd)[0]
        feat = _l2_normalize(feat[None, :])[0]

        sims = np.einsum("ckd,d->ck", protos, feat)  # (C,K)
        sims_class = sims.max(axis=1)  # (C,)

        sims_scaled = sims_class / SIM_TEMPERATURE
        pred = int(np.argmax(sims_scaled))
        return image_name, pred, None
    except Exception:
        return image_name, majority_label, "unreadable"


with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    for image_name, pred, err in ex.map(_predict_one, test_image_ids, chunksize=64):
        if err == "missing":
            missing_test += 1
        elif err == "unreadable":
            unreadable_test += 1
        image_ids_out.append(image_name)
        preds_out.append(pred)

if missing_test:
    print(
        f"WARNING: {missing_test} test images listed in sample_submission not found on disk."
    )
if unreadable_test:
    print(
        f"WARNING: {unreadable_test} test images could not be read; used majority fallback."
    )

pred_df = pd.DataFrame(
    {"image_id": image_ids_out, "label": np.array(preds_out, dtype=np.int64)}
)

submission = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

if submission["label"].isna().any():
    missing = int(submission["label"].isna().sum())
    print(
        f"WARNING: {missing} missing predictions after merge; filling with majority_label={majority_label}"
    )
    submission["label"] = submission["label"].fillna(majority_label)

submission["label"] = submission["label"].astype(int)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print("submission.csv saved at:", os.path.abspath("submission.csv"))
