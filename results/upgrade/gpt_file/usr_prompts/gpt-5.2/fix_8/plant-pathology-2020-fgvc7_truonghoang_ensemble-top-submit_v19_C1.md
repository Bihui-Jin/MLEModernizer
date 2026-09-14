# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import zipfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

np.random.seed(0)

CANDIDATE_ROOTS = [
    Path("../input/plant-pathology-2020-fgvc7"),
    Path("../input"),
    Path("/kaggle/input/plant-pathology-2020-fgvc7"),
    Path("/kaggle/input"),
    Path("../data/plant-pathology-2020-fgvc7"),
    Path("../data"),
    Path("/kaggle/data/plant-pathology-2020-fgvc7"),
    Path("/kaggle/data"),
]

DATA_ROOT = None
for r in CANDIDATE_ROOTS:
    if (r / "train.csv").exists() and (r / "test.csv").exists():
        DATA_ROOT = r
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate train.csv/test.csv under expected ../input or /kaggle paths."
    )

TRAIN_CSV = DATA_ROOT / "train.csv"
TEST_CSV = DATA_ROOT / "test.csv"
SAMPLE_SUB = DATA_ROOT / "sample_submission.csv"
IMAGES_DIR = DATA_ROOT / "images"
IMAGES_ZIP = DATA_ROOT / "images.zip"

print("Using DATA_ROOT:", DATA_ROOT)
print("Has images dir:", IMAGES_DIR.exists(), "Has images.zip:", IMAGES_ZIP.exists())

if not IMAGES_DIR.exists():
    if IMAGES_ZIP.exists():
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(IMAGES_ZIP, "r") as zf:
            zf.extractall(DATA_ROOT)
    else:
        raise FileNotFoundError("Neither images/ directory nor images.zip found.")


def _count_jpg_fast(d: Path):
    try:
        c = 0
        with os.scandir(d) as it:
            for e in it:
                if e.is_file() and e.name.lower().endswith(".jpg"):
                    c += 1
        return c
    except Exception:
        return len(list(d.glob("*.jpg")))


print("Images dir:", IMAGES_DIR, "num files:", _count_jpg_fast(IMAGES_DIR))



## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

TARGETS = [c for c in sample_sub.columns if c != "image_id"]
expected_targets = ["healthy", "multiple_diseases", "rust", "scab"]
if TARGETS != expected_targets:
    if all(t in sample_sub.columns for t in expected_targets):
        TARGETS = expected_targets
    else:
        raise ValueError(
            f"Unexpected target columns in sample_submission: {sample_sub.columns.tolist()}"
        )

assert "image_id" in train_df.columns and "image_id" in test_df.columns
for t in TARGETS:
    if t not in train_df.columns:
        raise ValueError(f"Train is missing target column: {t}")

train_df.head(), test_df.head(), sample_sub.head()



## === cell 2
_JPEG_SIZE_CACHE = {}
_FILESIZE_CACHE = {}
_IMAGE_PATH_CACHE = {}


def jpeg_size(path: Path):
    """
    Return (width, height) for a JPEG file by parsing markers.
    Fallback to (np.nan, np.nan) if parsing fails.
    Cached by path to avoid repeated parsing.
    """
    try:
        key = str(path)
        if key in _JPEG_SIZE_CACHE:
            return _JPEG_SIZE_CACHE[key]
        with open(path, "rb") as f:
            data = f.read(2048)  # usually enough to hit SOF marker
        if len(data) < 4 or data[0:2] != b"\xff\xd8":
            _JPEG_SIZE_CACHE[key] = (np.nan, np.nan)
            return _JPEG_SIZE_CACHE[key]
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            while i < len(data) and data[i] == 0xFF:
                i += 1
            if i >= len(data):
                break
            marker = data[i]
            i += 1
            if marker in (0xD8, 0xD9):
                continue
            if i + 2 > len(data):
                break
            seglen = int.from_bytes(data[i : i + 2], "big")
            if seglen < 2:
                break
            if marker in (
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            ):
                start = i + 2  # after seglen
                if start + 5 <= len(data):
                    height = int.from_bytes(data[start + 1 : start + 3], "big")
                    width = int.from_bytes(data[start + 3 : start + 5], "big")
                    _JPEG_SIZE_CACHE[key] = (width, height)
                    return _JPEG_SIZE_CACHE[key]
                break
            i += seglen
        _JPEG_SIZE_CACHE[key] = (np.nan, np.nan)
        return _JPEG_SIZE_CACHE[key]
    except Exception:
        return (np.nan, np.nan)


def _read_image_rgb_float01(path: Path):
    try:
        from PIL import Image

        with Image.open(str(path)) as im:
            im = im.convert("RGB")
            img = np.asarray(im, dtype=np.float32) / 255.0
        return np.clip(img, 0.0, 1.0)
    except Exception:
        try:
            import matplotlib.image as mpimg

            img = mpimg.imread(str(path))
            if img is None:
                return None
            img = np.asarray(img)
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            if img.shape[-1] == 4:
                img = img[..., :3]
            img = img.astype(np.float32)
            if img.max() > 1.5:
                img = img / 255.0
            img = np.clip(img, 0.0, 1.0)
            return img
        except Exception:
            return None


def _downsample_mean(img2d: np.ndarray, out_h: int, out_w: int):
    """Deterministic block-mean downsample without external deps (vectorized)."""
    H, W = img2d.shape
    if H < 1 or W < 1:
        return np.full((out_h, out_w), np.nan, dtype=np.float32)

    ys = np.linspace(0, H, out_h + 1, dtype=np.int32)
    xs = np.linspace(0, W, out_w + 1, dtype=np.int32)

    ys = ys.copy()
    xs = xs.copy()
    for i in range(out_h):
        if ys[i + 1] <= ys[i]:
            ys[i + 1] = min(H, ys[i] + 1)
    for j in range(out_w):
        if xs[j + 1] <= xs[j]:
            xs[j + 1] = min(W, xs[j] + 1)

    row_sums = np.add.reduceat(img2d, ys[:-1], axis=0)  # (out_h, W)
    block_sums = np.add.reduceat(row_sums, xs[:-1], axis=1)  # (out_h, out_w)

    row_counts = (ys[1:] - ys[:-1]).astype(np.float32)  # (out_h,)
    col_counts = (xs[1:] - xs[:-1]).astype(np.float32)  # (out_w,)
    denom = row_counts[:, None] * col_counts[None, :]
    out = (block_sums / denom).astype(np.float32)
    return out


def _rgb_to_hsv_np(img_rgb01: np.ndarray):
    """
    Minimal deterministic RGB->HSV for float RGB in [0,1], returns H,S,V in [0,1].
    """
    r = img_rgb01[..., 0]
    g = img_rgb01[..., 1]
    b = img_rgb01[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(cmax, dtype=np.float32)
    eps = 1e-6
    mask = delta > eps

    idx = mask & (cmax == r)
    h[idx] = ((g[idx] - b[idx]) / (delta[idx] + eps)) % 6.0
    idx = mask & (cmax == g)
    h[idx] = ((b[idx] - r[idx]) / (delta[idx] + eps)) + 2.0
    idx = mask & (cmax == b)
    h[idx] = ((r[idx] - g[idx]) / (delta[idx] + eps)) + 4.0
    h = (h / 6.0) % 1.0

    s = np.zeros_like(cmax, dtype=np.float32)
    s[cmax > eps] = delta[cmax > eps] / (cmax[cmax > eps] + eps)

    v = cmax.astype(np.float32)
    return h.astype(np.float32), s.astype(np.float32), v.astype(np.float32)


def _dct8_matrix_ortho():
    N = 8
    k = np.arange(N, dtype=np.float32).reshape(-1, 1)  # (8,1)
    n = np.arange(N, dtype=np.float32).reshape(1, -1)  # (1,8)
    M = np.cos(np.pi / N * (n + 0.5) * k).astype(np.float32)
    M[0, :] *= np.sqrt(1.0 / N)
    M[1:, :] *= np.sqrt(2.0 / N)
    return M


_DCT8 = _dct8_matrix_ortho()


def _dct2_8x8_ortho(a8: np.ndarray):
    """2D DCT-II (ortho) for 8x8 using precomputed transform matrix (separable)."""
    a8 = np.asarray(a8, dtype=np.float32)
    return (_DCT8 @ a8 @ _DCT8.T).astype(np.float32)


def _resolve_image_path(img_id: str):
    key = str(img_id)
    if key in _IMAGE_PATH_CACHE:
        return _IMAGE_PATH_CACHE[key]
    p = IMAGES_DIR / f"{key}.jpg"
    if p.exists():
        _IMAGE_PATH_CACHE[key] = p
        return p
    matches = list(IMAGES_DIR.glob(f"{key}.*"))
    out = matches[0] if matches else p
    _IMAGE_PATH_CACHE[key] = out
    return out


def _filesize_cached(p: Path):
    key = str(p)
    if key in _FILESIZE_CACHE:
        return _FILESIZE_CACHE[key]
    try:
        sz = p.stat().st_size
    except Exception:
        sz = np.nan
    _FILESIZE_CACHE[key] = sz
    return sz


def build_features(df: pd.DataFrame):
    img_ids = df["image_id"].astype(str).to_list()
    n = len(img_ids)

    widths = np.empty(n, dtype=np.float32)
    heights = np.empty(n, dtype=np.float32)
    sizes = np.empty(n, dtype=np.float32)

    F = 6 + 3 + 64 + 6 + 3 + 10 + 6 + 2 + 24
    pix_feats = np.empty((n, F), dtype=np.float32)

    bins = np.linspace(0.0, 1.0, 9, dtype=np.float32)  # 8 bins, as before
    coords = np.array(
        [
            (0, 1),
            (1, 0),
            (0, 2),
            (1, 1),
            (2, 0),
            (0, 3),
            (1, 2),
            (2, 1),
            (3, 0),
            (2, 2),
        ],
        dtype=np.int32,
    )

    eps = 1e-6
    inv_binw = 8.0  # since bins are uniform [0,1] split into 8

    for idx, img_id in enumerate(img_ids):
        p = _resolve_image_path(img_id)

        if p.exists():
            w, h = jpeg_size(p)
            widths[idx] = w
            heights[idx] = h
            sizes[idx] = _filesize_cached(p)
            img = _read_image_rgb_float01(p)
        else:
            widths[idx] = np.nan
            heights[idx] = np.nan
            sizes[idx] = np.nan
            img = None

        if img is None:
            pix_feats[idx, :] = np.nan
            continue

        flat = img.reshape(-1, 3)
        ch_mean = flat.mean(axis=0)
        ch_std = flat.std(axis=0)

        gray = (
            0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]
        ).astype(np.float32)

        base6 = np.array(
            [ch_mean[0], ch_mean[1], ch_mean[2], ch_std[0], ch_std[1], ch_std[2]],
            dtype=np.float32,
        )

        ratios3 = np.array(
            [
                float(ch_mean[0] / (ch_mean[1] + eps)),
                float(ch_mean[0] / (ch_mean[2] + eps)),
                float(ch_mean[1] / (ch_mean[2] + eps)),
            ],
            dtype=np.float32,
        )

        thumb8 = _downsample_mean(gray, 8, 8).astype(np.float32)  # (8,8)
        thumb = thumb8.reshape(-1)

        h_ch, s_ch, v_ch = _rgb_to_hsv_np(img)
        hsv_stats6 = np.array(
            [
                float(h_ch.mean()),
                float(s_ch.mean()),
                float(v_ch.mean()),
                float(h_ch.std()),
                float(s_ch.std()),
                float(v_ch.std()),
            ],
            dtype=np.float32,
        )

        r = img[..., 0]
        g = img[..., 1]
        b = img[..., 2]
        exg = (2.0 * g - r - b).astype(np.float32)
        exr = (1.4 * r - g).astype(np.float32)
        exb = (1.4 * b - g).astype(np.float32)
        veg3 = np.array(
            [float(exg.mean()), float(exr.mean()), float(exb.mean())], dtype=np.float32
        )

        dct8 = _dct2_8x8_ortho(thumb8)
        dct10 = dct8[coords[:, 0], coords[:, 1]].astype(np.float32)

        H, W = img.shape[0], img.shape[1]
        y0, y1 = int(H * 0.25), int(H * 0.75)
        x0, x1 = int(W * 0.25), int(W * 0.75)
        center = img[y0:y1, x0:x1, :]
        if center.size == 0:
            center = img
        center_mean = center.reshape(-1, 3).mean(axis=0).astype(np.float32)

        total_sum = img.reshape(-1, 3).sum(axis=0, dtype=np.float64)
        center_sum = center.reshape(-1, 3).sum(axis=0, dtype=np.float64)
        total_n = img.shape[0] * img.shape[1]
        center_n = center.shape[0] * center.shape[1]
        border_n = total_n - center_n
        if border_n <= 0:
            border_mean = (total_sum / max(total_n, 1)).astype(np.float32)
        else:
            border_mean = ((total_sum - center_sum) / border_n).astype(np.float32)

        center_border6 = np.concatenate(
            [center_mean, (center_mean - border_mean)], axis=0
        ).astype(np.float32)

        gx = np.abs(gray[:, 1:] - gray[:, :-1]).mean() if W > 1 else 0.0
        gy = np.abs(gray[1:, :] - gray[:-1, :]).mean() if H > 1 else 0.0
        grad2 = np.array([float(gx), float(gy)], dtype=np.float32)

        rgb_hist24 = np.empty(24, dtype=np.float32)
        for ch in range(3):
            x = img[..., ch].reshape(-1)
            bi = (x * inv_binw).astype(np.int32)
            bi = np.clip(bi, 0, 7)
            hst = np.bincount(bi, minlength=8).astype(np.float32)
            hst /= hst.sum() + 1e-6
            rgb_hist24[ch * 8 : (ch + 1) * 8] = hst

        pix_feats[idx, :] = np.concatenate(
            [
                base6,
                ratios3,
                thumb,
                hsv_stats6,
                veg3,
                dct10,
                center_border6,
                grad2,
                rgb_hist24,
            ],
            axis=0,
        )

    X_meta = pd.DataFrame({"w": widths, "h": heights, "filesize": sizes})
    X_meta["aspect"] = X_meta["w"] / (X_meta["h"] + 1e-6)

    colnames = []
    colnames += ["r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std"]
    colnames += ["rg_mean_ratio", "rb_mean_ratio", "gb_mean_ratio"]
    colnames += [f"thumb_{i}" for i in range(64)]
    colnames += ["h_mean", "s_mean", "v_mean", "h_std", "s_std", "v_std"]
    colnames += ["exg_mean", "exr_mean", "exb_mean"]
    colnames += [f"dct_{i}" for i in range(10)]
    colnames += [
        "center_r_mean",
        "center_g_mean",
        "center_b_mean",
        "center_minus_border_r",
        "center_minus_border_g",
        "center_minus_border_b",
    ]
    colnames += ["grad_x_meanabs", "grad_y_meanabs"]
    colnames += [f"rgb_hist_{i}" for i in range(24)]

    X_extra = pd.DataFrame(pix_feats.astype(np.float64), columns=colnames)
    X = pd.concat([X_meta, X_extra], axis=1)
    return X


X_train_df = build_features(train_df)
X_test_df = build_features(test_df)

med = X_train_df.median(numeric_only=True)
X_train_df = X_train_df.fillna(med)
X_test_df = X_test_df.fillna(med)

X_train_df.head(), X_test_df.head()




## === cell 3
def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def standardize_fit(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return mu, sigma


def standardize_transform(X, mu, sigma):
    return (X - mu) / sigma


def train_logreg_ovr(X, Y, lr=0.15, n_iter=1200, l2=1e-2):
    """
    Train independent logistic regressions for each column in Y.
    X: (n, d)
    Y: (n, k) binary
    Returns W: (d+1, k) including bias.
    """
    n, d = X.shape
    k = Y.shape[1]
    Xb = np.concatenate([np.ones((n, 1)), X], axis=1)  # bias
    W = np.zeros((d + 1, k), dtype=float)

    Yf = Y.astype(float, copy=False)

    for _ in range(n_iter):
        P = sigmoid(Xb @ W)  # (n,k)
        G = (Xb.T @ (P - Yf)) / n  # (d+1,k)
        G[1:, :] += l2 * W[1:, :]  # L2 on non-bias weights
        W -= lr * G
    return W


def predict_logreg_ovr(X, W):
    n = X.shape[0]
    Xb = np.concatenate([np.ones((n, 1)), X], axis=1)
    P = sigmoid(Xb @ W)
    return np.clip(P, 1e-6, 1 - 1e-6)


X_train = X_train_df.to_numpy(dtype=float)
X_test = X_test_df.to_numpy(dtype=float)
Y_train = train_df[TARGETS].to_numpy(dtype=float)

mu, sigma = standardize_fit(X_train)
X_train_s = standardize_transform(X_train, mu, sigma)
X_test_s = standardize_transform(X_test, mu, sigma)

W = train_logreg_ovr(X_train_s, Y_train, lr=0.15, n_iter=1200, l2=1e-2)
P_test = predict_logreg_ovr(X_test_s, W)

P_test.shape



## === cell 4
sub = sample_sub.copy()
sub = sub[["image_id"] + TARGETS].copy()

pred_df = pd.DataFrame(P_test, columns=TARGETS)
sub.loc[:, TARGETS] = pred_df[TARGETS].values

for c in TARGETS:
    sub[c] = pd.to_numeric(sub[c], errors="coerce").fillna(0.25).clip(0.0, 1.0)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", sub.shape)
sub.head()
