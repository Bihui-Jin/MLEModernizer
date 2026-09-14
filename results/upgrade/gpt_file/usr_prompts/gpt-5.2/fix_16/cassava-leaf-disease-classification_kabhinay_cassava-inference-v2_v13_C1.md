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

0.8785131459655485

# 6. Current score

0.32511

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63901) has done: 'I remove the expensive in-memory `.cache()` of decoded/resized images (it can exceed RAM and/or thrash, causing the observed timeout) and keep fast streaming with parallel `map` + `prefetch`. I also ensure the input pipeline is fully static-shaped and uses deterministic, parallel decoding/resize to maximize throughput without changing any pixels, labels, shuffling, model, loss, or training loop semantics. Finally, I add `steps_per_execution` to reduce Python overhead inside `model.fit` while keeping identical optimization steps and results (up to negligible FP differences). Paths, architecture, epochs, batch size, and split logic remain unchanged.'
- What this solution (achieved 0.63453) has done: 'I remove the two environment settings that force the slow pure-Python protobuf implementation (they significantly slow TFRecord/data pipeline + graph construction) and instead keep TensorFlow on its default fast C++ protobuf backend. I also add tf.data optimizations that are provably semantics-preserving: set a fixed `deterministic=True` on `map`, use `cache()` for validation/test (no reshuffle, so correctness is identical), and avoid expensive full-dataset shuffle buffer by capping it without changing the shuffle semantics materially (still uniform-ish, but to preserve semantics strictly we keep the full buffer). Finally, I ensure the input pipeline uses `num_parallel_calls=AUTOTUNE` everywhere and keep determinism and seeds intact.'
- What this solution (achieved 0.61099) has done: 'The crash happens before any training because TensorFlow/protobuf is incompatible in this Kaggle Python 2.7 environment, triggering the `MessageFactory.GetPrototype` AttributeError during `import tensorflow`. The minimal fix is to avoid using TensorFlow entirely and switch to a pure-Python baseline that still produces a valid `submission.csv` with the correct columns and ordering. To move accuracy up from ~0.63 toward the 0.878 target without introducing external dependencies, the safest approach is to use a stratified frequency-based prior estimated from `train.csv` and predict the most likely class for all test images (a common strong baseline for this dataset due to class imbalance). All file paths remain the same resolution logic, and the script always write `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.13453) has done: 'Your current code always predicts the single majority class, which is a stable but low-accuracy baseline (~0.61). To move the accuracy upward toward the 0.8785 target without changing the “no-TensorFlow, pure-Python” core approach, I switch to a tiny image-based classifier that uses only the standard library + NumPy/Pandas: compute per-class average RGB color from a subset of training images, then predict each test image by nearest class mean in color space. This keeps runtime under the limit by sampling a fixed number of train images per class and using downscaled thumbnails, and it preserves submission format and file paths. If any image read fails, it falls back to the majority label to ensure a valid CSV is always produced.'
- What this solution (achieved 0.18946) has done: 'Your current score (0.13453) is far below the target (0.8785), and the main reason is that the “mean RGB” heuristic is too weak; the smallest legitimate way to push accuracy up without changing the overall non-TensorFlow, pure-Python approach is to compute a slightly richer but still cheap image feature. I keep the exact same workflow (compute per-class prototypes from sampled train images, then nearest-prototype prediction for test), but replace the 3D mean-RGB feature with a compact color-histogram feature (per-channel bins) which captures more discriminative information while staying fast. I also make sampling deterministic and ensure we read images from directories first (no zip scanning), keeping the same submission writing and paths. This should substantially increase accuracy toward the target while preserving the same training/prediction semantics (prototype-from-train, nearest-class at test).'
- What this solution (achieved 0.18012) has done: 'Your current gap to the target is large (0.18946 vs 0.8785), and the main issue is that the “single prototype per class from color histograms” is too weak. Keeping the same core logic (sample a fixed number of train images per class → compute per-class prototype feature → nearest-prototype classification for test), I make the feature slightly richer but still cheap by concatenating a coarse grayscale histogram and simple per-channel mean/std to the existing RGB histograms. I also L2-normalize features and use cosine distance (still nearest-prototype, just a more stable distance for normalized histograms), which typically improves accuracy materially without changing the overall approach. All paths and the submission writing remain the same, and the script still runs end-to-end in pure Python + numpy/pandas + PIL.'
- What this solution (achieved 0.43386) has done: 'Your current score (0.18012) is far below the target (0.8785), so we need a real accuracy lift while keeping the same “per-class prototype feature from sampled train images → nearest-prototype prediction” core logic. The smallest high-impact change is to add one more lightweight, discriminative feature that doesn’t alter the training/prediction approach: a coarse HOG-like gradient orientation histogram on the grayscale thumbnail, concatenated to your existing color histograms + mean/std. To keep runtime under 600s and avoid destabilizing behavior, I keep the same sampling scheme, determinism, and cosine-similarity nearest-prototype classifier, and only slightly increase thumbnail size to better capture texture. Submission writing, paths, and fallback behavior remain unchanged.'
- What this solution (achieved 0.35164) has done: 'Your current score (0.43386) is far below the target (0.8785), so we need a meaningful accuracy lift while preserving the same core “per-class prototype features from sampled train images → nearest-prototype cosine similarity prediction” logic. The smallest high-impact change within that framework is to move from a single prototype per class to a small set of prototypes per class (k-means-like, but implemented as deterministic mini-batch k-medoids/mean updates) and then classify by the best similarity to any prototype in each class. This keeps the same feature extractor, normalization, sampling, and cosine scoring semantics, but better captures intra-class variability (lighting/background/leaf pose) which is a major limitation of a single mean prototype. I also add a minimal, semantics-preserving improvement: class-balanced sampling uses a fixed per-class cap but ensures at least a minimum count per class when available, improving prototype stability without changing the overall approach.'
- What this solution (achieved 0.31278) has done: 'Your current score (0.35164) is far below the target (0.8785), so we should increase accuracy, but with minimal changes that keep your same “handcrafted feature → per-class multi-prototypes → cosine nearest-prototype” core logic. The biggest likely issue is that resizing without preserving aspect ratio distorts leaves and harms gradient/texture features; switching to a deterministic letterbox (“fit then pad”) keeps semantics and speed while improving feature stability. I also make PIL resizing explicitly use BILINEAR (deterministic across runs) and slightly increase HOG spatial resolution (more cells, fewer bins) to better capture disease texture without changing the overall method. Finally, I keep runtime bounded by reducing k-means iters a bit to offset the extra HOG cells, so it should still finish under 600s.'
- What this solution (achieved 0.32735) has done: 'Your current score (0.31278) is far below the target (0.8785), so we should push accuracy up with the smallest changes that keep the same “handcrafted feature → per-class multi-prototypes → cosine nearest-prototype” core logic. The most likely regression is the recent HOG parameter change (cells=8, bins=6) and smaller k-means iterations; I revert to the previously stronger/standard HOG setup (cells=6, bins=8) and restore clustering iterations to stabilize prototypes. I also make the cosine scoring numerically consistent by explicitly L2-normalizing the test feature vector (training prototypes are normalized already), which preserves the same classifier semantics but reduces misclassification from slight normalization drift. All paths, sampling approach, prototype method, and submission writing remain unchanged.'
- What this solution (achieved 0.32511) has done: 'Your current score (0.32735) is far below the target (0.8785), so we should improve accuracy, but keep your exact core approach: handcrafted features → per-class multi-prototypes → cosine nearest-prototype prediction. The biggest likely issue is that the feature is over-normalized (histograms are normalized by sum and then L2-normalized), which can suppress informative magnitude differences across channels/parts; I switch to a standard HOG-style block normalization: L2-normalize each feature block separately (RGB hists, gray hist, HOG, mean/std) and then concatenate, keeping the same feature families and classifier. I also add a tiny amount of robustness by increasing KMEANS_ITERS back up modestly (better prototype stability) and slightly increasing prototypes per class (captures intra-class variation) while keeping runtime bounded by reducing per-class samples a bit. Paths, determinism, submission format, and the overall training/prediction semantics remain unchanged.'

# 9. Code solution

## === cell 0
from __future__ import print_function

import os
import io
import zipfile
import numpy as np
import pandas as pd

np.random.seed(42)

print(
    "Python OK; running non-TensorFlow fallback (pure stdlib + numpy/pandas) to avoid TF/protobuf crash."
)




## === cell 1
def _resolve_data_dir():
    candidates = [
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification",
        "../input/cassava-leaf-disease-classification",
        "../data/cassava-leaf-disease-classification",
        "/kaggle/input",
        "/kaggle/data",
    ]
    for c in candidates[:4]:
        if os.path.exists(c) and os.path.exists(os.path.join(c, "train.csv")):
            return c
    for base in candidates[4:]:
        comp = os.path.join(base, "cassava-leaf-disease-classification")
        if os.path.exists(comp) and os.path.exists(os.path.join(comp, "train.csv")):
            return comp
    raise OSError(
        "Could not find cassava-leaf-disease-classification data directory in known locations."
    )


DATA_DIR = _resolve_data_dir()
train_csv_path = os.path.join(DATA_DIR, "train.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_path)

print("DATA_DIR:", DATA_DIR)
print("Train rows:", len(train_df), "Test rows:", len(sample_sub))
print("Train label counts:\n", train_df["label"].value_counts().sort_index())




## === cell 2
def _find_images_dir(split):
    candidates = [
        os.path.join(DATA_DIR, split),
        os.path.join(DATA_DIR, "cassava-leaf-disease-classification", split),
        os.path.join("/kaggle/input/cassava-leaf-disease-classification", split),
        os.path.join("/kaggle/data/cassava-leaf-disease-classification", split),
    ]
    for p in candidates:
        if os.path.exists(p) and os.path.isdir(p):
            return p
    return None


def _find_images_zip(split):
    candidates = [
        os.path.join(DATA_DIR, split),
        os.path.join(DATA_DIR, "cassava-leaf-disease-classification", split),
        os.path.join("/kaggle/input/cassava-leaf-disease-classification", split),
        os.path.join("/kaggle/data/cassava-leaf-disease-classification", split),
        os.path.join(os.path.dirname(DATA_DIR), split),
        os.path.join(
            os.path.dirname(DATA_DIR), "cassava-leaf-disease-classification", split
        ),
    ]
    for p in candidates:
        if os.path.exists(p) and os.path.isfile(p):
            return p
    return None


def _try_import_pil():
    try:
        from PIL import Image  # noqa: F401

        return True
    except Exception:
        return False


_HAS_PIL = _try_import_pil()
if _HAS_PIL:
    from PIL import Image


def _hog_like_gray_hist(gray2d, cells=6, bins=8):
    """
    Minimal, fast HOG-like feature:
    - compute simple central-difference gradients on grayscale
    - accumulate orientation histograms (0..pi) per cell weighted by magnitude
    """
    g = gray2d.astype(np.float32)
    h, w = g.shape

    dx = g[:, 2:] - g[:, :-2]
    dy = g[2:, :] - g[:-2, :]
    dx = dx[1:-1, :]
    dy = dy[:, 1:-1]

    mag = np.sqrt(dx * dx + dy * dy) + 1e-6
    ang = np.arctan2(np.abs(dy), np.abs(dx))
    ang = ang * 2.0  # 0..pi

    hh, ww = mag.shape
    cell_h = max(1, hh // cells)
    cell_w = max(1, ww // cells)

    feats = []
    for cy in range(cells):
        y0 = cy * cell_h
        y1 = (cy + 1) * cell_h if cy < cells - 1 else hh
        for cx in range(cells):
            x0 = cx * cell_w
            x1 = (cx + 1) * cell_w if cx < cells - 1 else ww

            m = mag[y0:y1, x0:x1].reshape(-1)
            a = ang[y0:y1, x0:x1].reshape(-1)

            hst, _ = np.histogram(a, bins=bins, range=(0.0, np.pi), weights=m)
            feats.append(hst.astype(np.float32))

    return np.concatenate(feats, axis=0).astype(np.float32)


def _l2_normalize_vec(v):
    n = float(np.sqrt((v * v).sum())) + 1e-12
    return (v / n).astype(np.float32)


def _l2_normalize_rows(x):
    n = np.sqrt((x * x).sum(axis=1, keepdims=True)) + 1e-12
    return x / n


def _feature_vector_from_image(im_rgb, bins=16, gray_bins=16, hog_cells=6, hog_bins=8):
    arr3 = np.asarray(im_rgb, dtype=np.uint8)
    arr = arr3.reshape(-1, 3)

    blocks = []

    rgb_blocks = []
    for c in range(3):
        hst, _ = np.histogram(arr[:, c], bins=bins, range=(0, 256))
        rgb_blocks.append(hst.astype(np.float32))
    rgb = np.concatenate(rgb_blocks, axis=0).astype(np.float32)
    blocks.append(_l2_normalize_vec(rgb))

    gray_flat = (0.2989 * arr[:, 0] + 0.5870 * arr[:, 1] + 0.1140 * arr[:, 2]).astype(
        np.float32
    )
    hg, _ = np.histogram(gray_flat, bins=gray_bins, range=(0.0, 255.0))
    hg = hg.astype(np.float32)
    blocks.append(_l2_normalize_vec(hg))

    gray2d = (
        0.2989 * arr3[:, :, 0] + 0.5870 * arr3[:, :, 1] + 0.1140 * arr3[:, :, 2]
    ).astype(np.float32)
    hog = _hog_like_gray_hist(gray2d, cells=hog_cells, bins=hog_bins)
    blocks.append(_l2_normalize_vec(hog))

    mu = arr.mean(axis=0).astype(np.float32) / 255.0
    sd = arr.std(axis=0).astype(np.float32) / 255.0
    ms = np.concatenate([mu, sd], axis=0).astype(np.float32)
    blocks.append(_l2_normalize_vec(ms))

    feat = np.concatenate(blocks, axis=0).astype(np.float32)
    feat = _l2_normalize_vec(feat)
    return feat.astype(np.float32)


def _fit_multi_prototypes_cosine(X, k=4, iters=6, seed=42):
    """
    Multi-prototype cosine clustering (deterministic).
    """
    n, d = X.shape
    if n == 0:
        return np.zeros((k, d), dtype=np.float32)
    if n <= k:
        return _l2_normalize_rows(X.astype(np.float32))

    rng = np.random.RandomState(seed)
    init_idx = rng.choice(n, size=k, replace=False)
    C = X[init_idx].astype(np.float32)

    Xn = _l2_normalize_rows(X.astype(np.float32))
    Cn = _l2_normalize_rows(C)

    for _ in range(iters):
        sims = np.dot(Xn, Cn.T)  # (n,k)
        assign = np.argmax(sims, axis=1)

        newC = np.zeros_like(Cn)
        for j in range(k):
            mask = assign == j
            cnt = int(mask.sum())
            if cnt > 0:
                m = Xn[mask].mean(axis=0).astype(np.float32)
                nm = float(np.sqrt((m * m).sum()))
                if nm > 0.0:
                    m /= nm
                newC[j] = m
            else:
                newC[j] = Cn[j]
        Cn = newC

    return Cn.astype(np.float32)


def _resize_letterbox_rgb(im, out_size, resample):
    w, h = im.size
    if w <= 0 or h <= 0:
        return im.resize((out_size, out_size), resample=resample)
    scale = float(out_size) / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    im2 = im.resize((new_w, new_h), resample=resample)
    canvas = Image.new("RGB", (out_size, out_size), (0, 0, 0))
    left = (out_size - new_w) // 2
    top = (out_size - new_h) // 2
    canvas.paste(im2, (left, top))
    return canvas


def _feature_from_path(path, thumb=112, bins=16, gray_bins=16, hog_cells=6, hog_bins=8):
    if not _HAS_PIL:
        return None
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im = _resize_letterbox_rgb(im, thumb, resample=Image.BILINEAR)
            return _feature_vector_from_image(
                im,
                bins=bins,
                gray_bins=gray_bins,
                hog_cells=hog_cells,
                hog_bins=hog_bins,
            )
    except Exception:
        return None


def _feature_from_zip(
    zf, member, thumb=112, bins=16, gray_bins=16, hog_cells=6, hog_bins=8
):
    if not _HAS_PIL:
        return None
    try:
        data = zf.read(member)
        with Image.open(io.BytesIO(data)) as im:
            im = im.convert("RGB")
            im = _resize_letterbox_rgb(im, thumb, resample=Image.BILINEAR)
            return _feature_vector_from_image(
                im,
                bins=bins,
                gray_bins=gray_bins,
                hog_cells=hog_cells,
                hog_bins=hog_bins,
            )
    except Exception:
        return None


label_counts = train_df["label"].value_counts()
majority_label = int(label_counts.idxmax())
print("Majority label:", majority_label)

train_images_dir = _find_images_dir("train_images")
test_images_dir = _find_images_dir("test_images")
train_images_zip = _find_images_zip("train_images.zip")
test_images_zip = _find_images_zip("test_images.zip")

print("PIL available:", _HAS_PIL)
print("train_images_dir:", train_images_dir)
print("test_images_dir:", test_images_dir)
print("train_images_zip:", train_images_zip)
print("test_images_zip:", test_images_zip)

if not _HAS_PIL:
    results_new = sample_sub[["image_id"]].copy()
    results_new["label"] = majority_label
    out_path = "/kaggle/working/submission.csv"
    results_new.to_csv(out_path, index=False)
    print(
        "PIL unavailable; wrote majority-label submission to:",
        out_path,
        "rows:",
        len(results_new),
    )
else:
    PER_CLASS_SAMPLES = 750
    MIN_PER_CLASS_SAMPLES = 400

    THUMB = 112
    BINS = 16
    GRAY_BINS = 16

    HOG_CELLS = 6
    HOG_BINS = 8

    K_PROTOS = 6
    KMEANS_ITERS = 8

    class_protos = {}
    labels = sorted(train_df["label"].unique().tolist())

    zf_train = None
    if train_images_dir is None and train_images_zip is not None:
        zf_train = zipfile.ZipFile(train_images_zip, "r")

    feat_dim = (3 * BINS) + GRAY_BINS + (HOG_CELLS * HOG_CELLS * HOG_BINS) + 6

    for lab in labels:
        df_lab = train_df[train_df["label"] == lab]
        n_cap = min(PER_CLASS_SAMPLES, len(df_lab))
        n = n_cap
        if len(df_lab) >= MIN_PER_CLASS_SAMPLES:
            n = max(min(n_cap, PER_CLASS_SAMPLES), MIN_PER_CLASS_SAMPLES)
        df_s = df_lab.sample(n=n, random_state=42)

        feats = []
        for img_id in df_s["image_id"].values:
            v = None
            if train_images_dir is not None:
                p = os.path.join(train_images_dir, img_id)
                v = _feature_from_path(
                    p,
                    thumb=THUMB,
                    bins=BINS,
                    gray_bins=GRAY_BINS,
                    hog_cells=HOG_CELLS,
                    hog_bins=HOG_BINS,
                )
            elif zf_train is not None:
                member = img_id
                if ("train_images/" + img_id) in zf_train.NameToInfo:
                    member = "train_images/" + img_id
                v = _feature_from_zip(
                    zf_train,
                    member,
                    thumb=THUMB,
                    bins=BINS,
                    gray_bins=GRAY_BINS,
                    hog_cells=HOG_CELLS,
                    hog_bins=HOG_BINS,
                )
            if v is not None:
                feats.append(v)

        if len(feats) == 0:
            proto = np.ones((1, feat_dim), dtype=np.float32) / float(feat_dim)
            proto = _l2_normalize_rows(proto)
            class_protos[lab] = proto
            print(
                "Warning: no readable images for label", lab, "- using uniform feature."
            )
        else:
            X = np.vstack(feats).astype(np.float32)
            protos = _fit_multi_prototypes_cosine(
                X, k=K_PROTOS, iters=KMEANS_ITERS, seed=42 + int(lab)
            )
            class_protos[lab] = protos

        print(
            "Label",
            lab,
            "prototypes:",
            class_protos[lab].shape,
            "from",
            len(feats),
            "images",
        )

    if zf_train is not None:
        zf_train.close()

    zf_test = None
    if test_images_dir is None and test_images_zip is not None:
        zf_test = zipfile.ZipFile(test_images_zip, "r")

    allP = []
    allL = []
    for lab in labels:
        P = class_protos[lab]
        allP.append(P)
        allL.extend([lab] * P.shape[0])
    allP = np.vstack(allP).astype(np.float32)  # (sum_k, F)
    allL = np.array(allL, dtype=np.int32)

    pred_labels = []
    unreadable = 0
    for img_id in sample_sub["image_id"].values:
        v = None
        if test_images_dir is not None:
            p = os.path.join(test_images_dir, img_id)
            v = _feature_from_path(
                p,
                thumb=THUMB,
                bins=BINS,
                gray_bins=GRAY_BINS,
                hog_cells=HOG_CELLS,
                hog_bins=HOG_BINS,
            )
        elif zf_test is not None:
            member = img_id
            if ("test_images/" + img_id) in zf_test.NameToInfo:
                member = "test_images/" + img_id
            v = _feature_from_zip(
                zf_test,
                member,
                thumb=THUMB,
                bins=BINS,
                gray_bins=GRAY_BINS,
                hog_cells=HOG_CELLS,
                hog_bins=HOG_BINS,
            )

        if v is None:
            pred_labels.append(majority_label)
            unreadable += 1
            continue

        v = _l2_normalize_vec(v)

        sims = np.dot(allP, v.reshape(-1, 1)).reshape(-1)
        best_idx = int(np.argmax(sims))
        pred_labels.append(int(allL[best_idx]))

    if zf_test is not None:
        zf_test.close()

    results_new = sample_sub[["image_id"]].copy()
    results_new["label"] = pred_labels

    out_path = "/kaggle/working/submission.csv"
    results_new.to_csv(out_path, index=False)

    print("Wrote:", out_path, "rows:", len(results_new))
    print("Unreadable test images (fell back to majority):", unreadable)
    print(results_new.head())
    print("Submission columns:", list(results_new.columns))
    print("Label value counts:\n", results_new["label"].value_counts().sort_index())
