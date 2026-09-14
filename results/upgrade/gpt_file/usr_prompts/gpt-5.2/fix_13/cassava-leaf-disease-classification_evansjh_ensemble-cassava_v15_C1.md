# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

np.random.seed(42)

BASE_INPUT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/input",  # fallback (some environments mount files directly here)
    "/kaggle/data",
]


def _resolve_base():
    for p in BASE_INPUT_CANDIDATES:
        if os.path.isdir(p):
            if os.path.isfile(os.path.join(p, "sample_submission.csv")):
                return p
            nested = os.path.join(p, "cassava-leaf-disease-classification")
            if os.path.isfile(os.path.join(nested, "sample_submission.csv")):
                return nested
    return None


BASE_INPUT = _resolve_base()
if BASE_INPUT is None:
    raise FileNotFoundError(
        "Could not find competition directory containing sample_submission.csv in: "
        + ", ".join(BASE_INPUT_CANDIDATES)
    )

train_image_dir = os.path.join(BASE_INPUT, "train_images")
test_image_dir = os.path.join(BASE_INPUT, "test_images")
sample = os.path.join(BASE_INPUT, "sample_submission.csv")
train_csv_path = os.path.join(BASE_INPUT, "train.csv")

if not os.path.isfile(sample):
    raise FileNotFoundError(f"Missing sample submission: {sample}")
if not os.path.isfile(train_csv_path):
    raise FileNotFoundError(f"Missing train.csv: {train_csv_path}")
if not os.path.isdir(train_image_dir):
    raise FileNotFoundError(f"Missing train_images directory: {train_image_dir}")
if not os.path.isdir(test_image_dir):
    raise FileNotFoundError(f"Missing test_images directory: {test_image_dir}")

print("Resolved paths:")
print("BASE_INPUT:", BASE_INPUT)
print("train_image_dir:", train_image_dir)
print("test_image_dir:", test_image_dir)
print("sample:", sample)
print("train_csv:", train_csv_path)



## === cell 1
sample_csv = pd.read_csv(sample)
assert list(sample_csv.columns) == [
    "image_id",
    "label",
], "Unexpected sample_submission.csv columns"
print(sample_csv.head())
print("Test rows:", len(sample_csv))

train_df = pd.read_csv(train_csv_path)
assert list(train_df.columns) == ["image_id", "label"], "Unexpected train.csv columns"
print(train_df.head())
print("Train rows:", len(train_df))

label_counts = train_df["label"].value_counts()
majority_label = int(label_counts.idxmax())
print("Train label distribution:")
print(label_counts.sort_index())
print("Majority label:", majority_label, "count:", int(label_counts.max()))




## === cell 2
def _try_import_pil():
    try:
        from PIL import Image, ImageOps  # type: ignore

        return Image, ImageOps
    except Exception:
        return None, None


Image, ImageOps = _try_import_pil()
if Image is None:
    print(
        "WARNING: Pillow (PIL) not available; falling back to majority-class predictions."
    )
    USE_IMAGES = False
else:
    USE_IMAGES = True
    print("Pillow available; using image-based nearest-centroid classifier.")

IMG_SIZE_1 = 96
IMG_SIZE_2 = 64

CENTER_CROP_FRAC = 0.90

MAX_TRAIN_IMAGES = 17000
N_CLASSES = 5

K_PER_CLASS = 8
KMEANS_ITERS = 5

TTA_CROPS = ("center", "tl", "tr")

AUG_FLIP_IN_TRAIN = True

USE_SOBEL_FEATURES = True

USE_LBP_FEATURES = True
LBP_BINS = 32
USE_COLOR_STATS = True

PER_BLOCK_L2_NORMALIZE = True

FINAL_FEATURE_STANDARDIZE = True

USE_DATASET_FEATURE_NORMALIZATION = True


def _crop_box(w, h, frac, mode="center"):
    side = int(min(w, h) * float(frac))
    if side <= 1:
        return (0, 0, w, h)
    if mode == "center":
        left = (w - side) // 2
        top = (h - side) // 2
    elif mode == "tl":
        left, top = 0, 0
    elif mode == "tr":
        left, top = w - side, 0
    elif mode == "bl":
        left, top = 0, h - side
    elif mode == "br":
        left, top = w - side, h - side
    else:
        left = (w - side) // 2
        top = (h - side) // 2
    left = int(max(0, min(left, w - side)))
    top = int(max(0, min(top, h - side)))
    return (left, top, left + side, top + side)


def _to_gray_standardized(arr_rgb):
    arr = np.asarray(arr_rgb, dtype=np.float32)
    gray = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
    gray = gray / 255.0
    m = float(gray.mean())
    s = float(gray.std())
    if s < 1e-6:
        s = 1e-6
    gray = (gray - m) / s
    return gray


def _sobel_features(gray2d: np.ndarray) -> np.ndarray:
    """
    Deterministic, non-learned feature to capture edges/texture.
    Uses simple Sobel kernels; output is standardized similarly to gray.
    """
    g = gray2d.astype(np.float32, copy=False)
    gp = np.pad(g, ((1, 1), (1, 1)), mode="edge")

    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    gx = (
        gp[:-2, :-2] * kx[0, 0]
        + gp[:-2, 1:-1] * kx[0, 1]
        + gp[:-2, 2:] * kx[0, 2]
        + gp[1:-1, :-2] * kx[1, 0]
        + gp[1:-1, 1:-1] * kx[1, 1]
        + gp[1:-1, 2:] * kx[1, 2]
        + gp[2:, :-2] * kx[2, 0]
        + gp[2:, 1:-1] * kx[2, 1]
        + gp[2:, 2:] * kx[2, 2]
    )
    gy = (
        gp[:-2, :-2] * ky[0, 0]
        + gp[:-2, 1:-1] * ky[0, 1]
        + gp[:-2, 2:] * ky[0, 2]
        + gp[1:-1, :-2] * ky[1, 0]
        + gp[1:-1, 1:-1] * ky[1, 1]
        + gp[1:-1, 2:] * ky[1, 2]
        + gp[2:, :-2] * ky[2, 0]
        + gp[2:, 1:-1] * ky[2, 1]
        + gp[2:, 2:] * ky[2, 2]
    )

    mag = np.sqrt(gx * gx + gy * gy + 1e-12).astype(np.float32)

    m = float(mag.mean())
    s = float(mag.std())
    if s < 1e-6:
        s = 1e-6
    mag = (mag - m) / s
    return np.concatenate([gx.reshape(-1), gy.reshape(-1), mag.reshape(-1)], axis=0)


def _lbp_hist(gray2d: np.ndarray, bins: int = LBP_BINS) -> np.ndarray:
    """
    Deterministic texture descriptor: basic 8-neighbor LBP code histogram.
    Uses the standardized gray input; only relative comparisons matter.
    Returns L1-normalized histogram.
    """
    g = gray2d.astype(np.float32, copy=False)
    gp = np.pad(g, ((1, 1), (1, 1)), mode="edge")
    c = gp[1:-1, 1:-1]

    code = np.zeros_like(c, dtype=np.uint8)
    code |= (gp[0:-2, 0:-2] > c).astype(np.uint8) << 0
    code |= (gp[0:-2, 1:-1] > c).astype(np.uint8) << 1
    code |= (gp[0:-2, 2:] > c).astype(np.uint8) << 2
    code |= (gp[1:-1, 2:] > c).astype(np.uint8) << 3
    code |= (gp[2:, 2:] > c).astype(np.uint8) << 4
    code |= (gp[2:, 1:-1] > c).astype(np.uint8) << 5
    code |= (gp[2:, 0:-2] > c).astype(np.uint8) << 6
    code |= (gp[1:-1, 0:-2] > c).astype(np.uint8) << 7

    bin_idx = (code.astype(np.int32) * bins) // 256
    hist = np.bincount(bin_idx.reshape(-1), minlength=bins).astype(np.float32)
    s = float(hist.sum())
    if s > 0:
        hist /= s
    return hist


def _color_stats(arr_rgb: np.ndarray) -> np.ndarray:
    """
    Simple global color stats (mean/std per channel) to separate healthy vs diseased discoloration.
    """
    x = np.asarray(arr_rgb, dtype=np.float32) / 255.0
    mu = x.reshape(-1, 3).mean(axis=0)
    sd = x.reshape(-1, 3).std(axis=0)
    return np.concatenate([mu, sd], axis=0).astype(np.float32)


def _final_standardize(v: np.ndarray) -> np.ndarray:
    m = float(v.mean())
    s = float(v.std())
    if s < 1e-6:
        s = 1e-6
    return ((v - m) / s).astype(np.float32, copy=False)


def _l2_normalize_block(v: np.ndarray) -> np.ndarray:
    n = float(np.sqrt((v * v).sum()))
    if n < 1e-12:
        return v.astype(np.float32, copy=False)
    return (v / n).astype(np.float32, copy=False)


def load_image_feature_from_pil(img, img_size1=IMG_SIZE_1, img_size2=IMG_SIZE_2):
    img1 = img.resize((img_size1, img_size1))
    img2 = img.resize((img_size2, img_size2))

    g1 = _to_gray_standardized(img1)
    g2 = _to_gray_standardized(img2)

    f_g1 = g1.reshape(-1).astype(np.float32, copy=False)
    f_g2 = g2.reshape(-1).astype(np.float32, copy=False)
    if PER_BLOCK_L2_NORMALIZE:
        f_g1 = _l2_normalize_block(f_g1)
        f_g2 = _l2_normalize_block(f_g2)

    feats = [f_g1, f_g2]

    if USE_SOBEL_FEATURES:
        s1 = _sobel_features(g1).astype(np.float32, copy=False)
        s2 = _sobel_features(g2).astype(np.float32, copy=False)
        if PER_BLOCK_L2_NORMALIZE:
            s1 = _l2_normalize_block(s1)
            s2 = _l2_normalize_block(s2)
        feats.extend([s1, s2])

    if USE_LBP_FEATURES:
        h1 = _lbp_hist(g1, bins=LBP_BINS).astype(np.float32, copy=False)
        h2 = _lbp_hist(g2, bins=LBP_BINS).astype(np.float32, copy=False)
        if PER_BLOCK_L2_NORMALIZE:
            h1 = _l2_normalize_block(h1)
            h2 = _l2_normalize_block(h2)
        feats.extend([h1, h2])

    if USE_COLOR_STATS:
        cs = _color_stats(np.asarray(img1)).astype(np.float32, copy=False)
        if PER_BLOCK_L2_NORMALIZE:
            cs = _l2_normalize_block(cs)
        feats.append(cs)

    v = np.concatenate(feats, axis=0).astype(np.float32, copy=False)
    if FINAL_FEATURE_STANDARDIZE:
        v = _final_standardize(v)
    return v


def load_image_feature(path, img_size1=IMG_SIZE_1, img_size2=IMG_SIZE_2):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    box = _crop_box(w, h, CENTER_CROP_FRAC, mode="center")
    img = img.crop(box)
    return load_image_feature_from_pil(img, img_size1=img_size1, img_size2=img_size2)


def load_image_feature_tta(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size

    feats = []
    for mode in TTA_CROPS:
        box = _crop_box(w, h, CENTER_CROP_FRAC, mode=mode)
        crop = img.crop(box)
        feats.append(load_image_feature_from_pil(crop))
    return np.mean(np.stack(feats, axis=0).astype(np.float32), axis=0)


def _stratified_subset(df, max_images, seed=42):
    rng = np.random.RandomState(seed)
    per_class = max(1, max_images // N_CLASSES)
    parts = []
    for cls in range(N_CLASSES):
        cls_df = df[df["label"] == cls]
        if len(cls_df) == 0:
            continue
        n = min(per_class, len(cls_df))
        idx = rng.choice(cls_df.index.values, size=n, replace=False)
        parts.append(df.loc[idx])
    if len(parts) == 0:
        return df.head(0)
    sub = pd.concat(parts, axis=0)
    if len(sub) < max_images:
        remaining = df.drop(index=sub.index)
        need = min(max_images - len(sub), len(remaining))
        if need > 0:
            perm = rng.permutation(remaining.index.values)
            fill_idx = perm[:need]
            sub = pd.concat([sub, df.loc[fill_idx]], axis=0)
    return sub.sort_values("image_id").reset_index(drop=True)


def _kmeans_lloyd(X, k, iters, seed=42):
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    if n == 0:
        return np.zeros((0, X.shape[1]), dtype=np.float32)

    k_eff = int(min(k, n))
    init_idx = rng.choice(n, size=k_eff, replace=False)
    C = X[init_idx].astype(np.float32, copy=True)

    for _ in range(int(iters)):
        C_norm = (C * C).sum(axis=1, keepdims=True)
        X_norm = (X * X).sum(axis=1, keepdims=True)
        d2 = X_norm + C_norm.T - 2.0 * (X @ C.T)
        a = np.argmin(d2, axis=1)

        newC = np.empty_like(C)
        for j in range(k_eff):
            mask = a == j
            if np.any(mask):
                newC[j] = X[mask].mean(axis=0)
            else:
                far_idx = int(np.argmax(d2.min(axis=1)))
                newC[j] = X[far_idx]
        C = newC
    return C.astype(np.float32)


def _compute_dataset_norm_stats(feature_list):
    X = np.stack(feature_list, axis=0).astype(np.float32, copy=False)
    mu = X.mean(axis=0).astype(np.float32, copy=False)
    sd = X.std(axis=0).astype(np.float32, copy=False)
    sd = np.where(sd < 1e-6, 1e-6, sd).astype(np.float32, copy=False)
    return mu, sd


def _apply_dataset_norm(x, mu, sd):
    return ((x - mu) / sd).astype(np.float32, copy=False)


def build_centroids(train_df, train_dir, max_images=MAX_TRAIN_IMAGES):
    sub = _stratified_subset(train_df, max_images=max_images, seed=42)

    feats_by_class = {c: [] for c in range(N_CLASSES)}
    failures = 0
    global_sum = None
    global_count = 0
    feat_dim = None

    all_train_feats_for_norm = []

    for img_id, y in zip(sub["image_id"].values, sub["label"].values):
        p = os.path.join(train_dir, img_id)
        try:
            img = Image.open(p)
            img = ImageOps.exif_transpose(img).convert("RGB")
            w, h = img.size
            box = _crop_box(w, h, CENTER_CROP_FRAC, mode="center")
            img_c = img.crop(box)

            x = load_image_feature_from_pil(img_c).astype(np.float32, copy=False)

            xs = [x]
            if AUG_FLIP_IN_TRAIN and ImageOps is not None:
                img_f = ImageOps.mirror(img_c)
                xf = load_image_feature_from_pil(img_f).astype(np.float32, copy=False)
                xs.append(xf)

            for x_i in xs:
                if feat_dim is None:
                    feat_dim = int(x_i.shape[0])
                    global_sum = np.zeros((feat_dim,), dtype=np.float64)

                all_train_feats_for_norm.append(x_i)
                cls = int(y)
                feats_by_class[cls].append(x_i)
                global_sum += x_i.astype(np.float64, copy=False)
                global_count += 1
        except Exception:
            failures += 1
            continue

    if global_count == 0 or global_sum is None or feat_dim is None:
        raise RuntimeError("Could not load any training images to build centroids.")

    global_mean = (global_sum / max(1, global_count)).astype(np.float32)

    if USE_DATASET_FEATURE_NORMALIZATION:
        feat_mu, feat_sd = _compute_dataset_norm_stats(all_train_feats_for_norm)
    else:
        feat_mu, feat_sd = None, None

    if USE_DATASET_FEATURE_NORMALIZATION:
        for cls in range(N_CLASSES):
            if len(feats_by_class[cls]) > 0:
                feats_by_class[cls] = [
                    _apply_dataset_norm(xi, feat_mu, feat_sd)
                    for xi in feats_by_class[cls]
                ]

        global_mean = _apply_dataset_norm(global_mean, feat_mu, feat_sd)

    centroids = {}
    per_class_counts = []
    for cls in range(N_CLASSES):
        X_list = feats_by_class.get(cls, [])
        per_class_counts.append(len(X_list))
        if len(X_list) == 0:
            centroids[int(cls)] = global_mean.reshape(1, -1)
            continue
        X = np.stack(X_list, axis=0)
        C = _kmeans_lloyd(X, k=K_PER_CLASS, iters=KMEANS_ITERS, seed=42 + cls)
        if C.shape[0] == 0:
            C = global_mean.reshape(1, -1)
        centroids[int(cls)] = C

    total_centroids = int(sum(centroids[c].shape[0] for c in centroids))
    print(
        f"Built centroids using {int(global_count)} augmented samples (failures={failures}). "
        f"Feature dim={feat_dim}. Total centroids={total_centroids} ({K_PER_CLASS}/class target)."
    )
    print("Per-class augmented samples used:", per_class_counts)
    print(
        "Per-class centroids:",
        {c: int(centroids[c].shape[0]) for c in sorted(centroids)},
    )

    if USE_DATASET_FEATURE_NORMALIZATION:
        print(
            "Using dataset-level feature normalization (mean/std) computed from training subset."
        )
    else:
        print("Dataset-level feature normalization disabled.")

    return centroids, feat_mu, feat_sd


def predict_nearest_centroid(
    centroids, test_ids, test_dir, feat_mu, feat_sd, fallback_label=majority_label
):
    classes = sorted(centroids.keys())
    C_list = []
    L_list = []
    for c in classes:
        Ci = centroids[c]
        C_list.append(Ci)
        L_list.extend([c] * Ci.shape[0])

    C = np.vstack(C_list).astype(np.float32)
    L = np.asarray(L_list, dtype=np.int64)
    C_norm = (C * C).sum(axis=1)

    preds = []
    failures = 0
    for img_id in test_ids:
        p = os.path.join(test_dir, img_id)
        try:
            x = load_image_feature_tta(p).astype(np.float32, copy=False)

            if (
                USE_DATASET_FEATURE_NORMALIZATION
                and feat_mu is not None
                and feat_sd is not None
            ):
                x = _apply_dataset_norm(x, feat_mu, feat_sd)

            x_norm = float((x * x).sum())
            dots = C @ x
            d2 = x_norm + C_norm - 2.0 * dots
            pred = int(L[int(np.argmin(d2))])
            preds.append(pred)
        except Exception:
            failures += 1
            preds.append(int(fallback_label))

    print(
        f"Predicted {len(preds)} test images (failures={failures}, used fallback={fallback_label} for those)."
    )
    return preds


if USE_IMAGES:
    centroids, feat_mu, feat_sd = build_centroids(
        train_df, train_image_dir, max_images=MAX_TRAIN_IMAGES
    )
    pred_labels = predict_nearest_centroid(
        centroids=centroids,
        test_ids=sample_csv["image_id"].values.tolist(),
        test_dir=test_image_dir,
        feat_mu=feat_mu,
        feat_sd=feat_sd,
        fallback_label=majority_label,
    )
else:
    pred_labels = np.full(
        shape=(len(sample_csv),), fill_value=majority_label, dtype=np.int64
    ).tolist()

print("Predictions:", len(pred_labels))



## === cell 3
submission_df = sample_csv.copy()
submission_df["label"] = pred_labels

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission_df.head())
print("Submission shape:", submission_df.shape)

assert submission_df.shape[0] == sample_csv.shape[0]
assert list(submission_df.columns) == ["image_id", "label"]
assert submission_path.endswith(".csv")
