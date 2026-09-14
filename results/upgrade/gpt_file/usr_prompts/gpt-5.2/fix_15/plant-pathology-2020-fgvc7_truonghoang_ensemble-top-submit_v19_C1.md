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

0.97113

# 6. Current score

0.67137

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.52238) has done: 'I remove the dependency on missing external submission files under `../input/plantpathology/` (the cause of the FileNotFoundError) and instead generate predictions from the provided train/test/images in this environment. Because only NumPy/Pandas/OS are available, I implement a simple, deterministic image-feature + one-vs-rest logistic regression (trained via batch gradient descent) to produce valid probabilities for the 4 required columns. I also fix the dataset pathing to use the existing `../input/plant-pathology-2020-fgvc7/` (and fall back to `../input/`), and ensure the submission columns exactly match `sample_submission.csv`. The script run end-to-end and write `submission.csv` with the correct header and 183 rows.'
- What this solution (achieved 0.69751) has done: 'Your current 0.522 AUC is far below the 0.971 target, so we should improve score with minimal core-logic changes by making the existing logistic-regression-on-handcrafted-features actually learn from image content rather than filename/size metadata. I keep the same one-vs-rest logistic regression, gradient-descent training loop, sigmoid, and submission semantics, but replace/extend `build_features()` to extract simple, deterministic pixel statistics from each JPEG (downsampled) using only NumPy (no new packages). This adds real signal for disease patterns while staying lightweight and within time, and it should move AUC substantially upward toward the target band. I also make feature extraction deterministic and cached in-memory to avoid redundant file reads.'
- What this solution (achieved 0.65292) has done: 'You’re far below the target AUC, so the smallest score-relevant improvement while preserving your exact training loop and one-vs-rest logistic regression is to make the handcrafted features actually reflect leaf appearance instead of JPEG file-byte structure. I keep all model code identical, but change `build_features()` to extract deterministic, lightweight pixel-statistics from decoded images (RGB means/stds and a coarse 8×8 grayscale thumbnail) using only standard-library decoding via `matplotlib.image` (no new packages). I also add a tiny interaction feature set (channel ratios) that logistic regression can exploit without changing the learning approach. This should materially increase signal and move the score upward toward the 0.971 target while still running within the time limit and producing a valid `submission.csv`.'
- What this solution (achieved 0.65912) has done: 'Your current gap to the 0.97113 target is large, so the most direct improvement while preserving your one-vs-rest logistic regression and training loop is to make the handcrafted features more disease-informative. I keep the same optimizer/iterations/loss semantics, but extend `build_features()` with a few extra deterministic, cheap image statistics (HSV moments, simple vegetation indices like ExG, and low-frequency DCT of an 8×8 grayscale thumbnail) that logistic regression can exploit. I also ensure feature extraction is consistent and numeric (same NaN handling) while keeping output format identical. These are minimal changes focused solely on increasing ROC AUC without changing the model class or training approach.'
- What this solution (achieved 0.65807) has done: 'The timeout is dominated by per-image feature extraction: each image is decoded fully, flattened, and multiple full-resolution passes are done (mean/std, HSV, center/border, gradients, histograms), which is unnecessarily expensive because many features can be computed exactly from an 8×8 thumbnail and simple reductions. I keep the exact same features/logic/columns, but restructure `build_features()` to (1) compute global stats and most pixel-derived features on a cached 8×8 downsample per image, (2) avoid repeated reshapes/temporary arrays, and (3) compute center/border means and gradients with vectorized sums on the full image (still exact) while minimizing allocations. I also avoid DataFrame overhead by returning a preallocated NumPy array and constructing the DataFrame once, and I precompute small constants (gray weights) and reuse buffers. The logistic regression training loop is already vectorized; the main wins come from reducing image-decoding work and cutting full-resolution operations to the minimum needed while preserving identical semantics.'
- What this solution (achieved 0.65826) has done: 'We’re far below the 0.97113 target, so we should increase AUC with minimal risk while keeping your one-vs-rest logistic regression, training loop, and loss semantics unchanged. The biggest safe gain comes from fixing a subtle but important bug in your RGB histogram features: `x * inv_binw` is wrong (it compresses almost all pixels into the first bin), so the histogram carries little information; switching to `x * 8` yields a correct 8-bin histogram and typically improves ROC AUC. I also add a single, standard leaf-health signal feature (NDI = (G−R)/(G+R)) computed on the same 8×8 thumbnail (no new libraries), and keep everything else identical. Submission formatting and paths remain unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.65735) has done: 'Your current AUC (0.65826) is far below the target (0.97113), so we should push performance upward with the smallest score-relevant changes that keep your one-vs-rest logistic regression and training loop intact. The most likely bottleneck now is feature quality: we add a couple of very cheap, deterministic disease-sensitive color index features (normalized channel differences and simple chroma) computed on the same existing 8×8 thumbnail, without changing the model or optimization. These additions give the linear classifier more separable signals (e.g., rust/scab discoloration vs healthy) while keeping runtime within limits and preserving submission semantics. Everything else (paths, standardization, GD loop, loss, output CSV format) stays the same.'
- What this solution (achieved 0.67221) has done: 'Your current score is far below the 0.97113 target, so we should increase AUC with the smallest changes that keep your one-vs-rest logistic regression and training loop intact. The safest gain is to add a few more disease-informative but cheap image features derived from the same already-computed 8×8 thumbnail: simple per-channel quantiles (captures patchy lesions) and coarse hue/saturation histograms (captures rust/scab color shifts) without changing the model. This preserves evaluation semantics and runtime while giving the linear model more separable signals. Everything else (paths, standardization, GD loop, submission formatting) stays the same.'
- What this solution (achieved 0.67137) has done: 'We’re far below the 0.97113 target (0.67221 currently), so we should increase AUC with very small, score-relevant changes that keep your one-vs-rest logistic regression and training loop intact. The biggest likely issue now is feature scaling: your `filesize` (bytes) and `w/h` are on very different scales and can dominate standardization when combined with many pixel features; applying a safe `log1p` transform to filesize (and leaving everything else unchanged) typically improves linear separability and AUC without changing core logic. Additionally, the mean column-wise ROC AUC is insensitive to per-column monotonic transforms, so we can safely enforce per-row probability normalization (sum to 1) to better match the mutually-exclusive nature of the labels and reduce pathological probability allocations. These two changes are minimal, deterministic, and preserve the model class/optimizer while usually improving ROC AUC on this competition.'

# 9. Code solution

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

_THUMB8_CACHE = {}

_IMAGE_FILE_MAP = None
try:
    _IMAGE_FILE_MAP = {}
    with os.scandir(IMAGES_DIR) as it:
        for e in it:
            if e.is_file():
                name = e.name
                low = name.lower()
                if (
                    low.endswith(".jpg")
                    or low.endswith(".jpeg")
                    or low.endswith(".png")
                ):
                    stem = name.rsplit(".", 1)[0]
                    if stem not in _IMAGE_FILE_MAP:
                        _IMAGE_FILE_MAP[stem] = Path(e.path)
except Exception:
    _IMAGE_FILE_MAP = None


def jpeg_size(path: Path):
    """
    Return (width, height) for a JPEG file by parsing markers.
    Cached by path to avoid repeated parsing.
    """
    key = str(path)
    if key in _JPEG_SIZE_CACHE:
        return _JPEG_SIZE_CACHE[key]
    try:
        with open(path, "rb") as f:
            data = f.read(65536)
        if len(data) < 4 or data[0:2] != b"\xff\xd8":
            _JPEG_SIZE_CACHE[key] = (np.nan, np.nan)
            return _JPEG_SIZE_CACHE[key]
        i = 2
        n = len(data)
        while i + 9 < n:
            if data[i] != 0xFF:
                i += 1
                continue
            while i < n and data[i] == 0xFF:
                i += 1
            if i >= n:
                break
            marker = data[i]
            i += 1
            if marker in (0xD8, 0xD9):
                continue
            if i + 2 > n:
                break
            seglen = (data[i] << 8) + data[i + 1]
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
                start = i + 2
                if start + 5 <= n:
                    height = (data[start + 1] << 8) + data[start + 2]
                    width = (data[start + 3] << 8) + data[start + 4]
                    _JPEG_SIZE_CACHE[key] = (float(width), float(height))
                    return _JPEG_SIZE_CACHE[key]
                break
            i += seglen
        _JPEG_SIZE_CACHE[key] = (np.nan, np.nan)
        return _JPEG_SIZE_CACHE[key]
    except Exception:
        _JPEG_SIZE_CACHE[key] = (np.nan, np.nan)
        return _JPEG_SIZE_CACHE[key]


def _read_image_rgb_float01(path: Path):
    try:
        from PIL import Image

        with Image.open(str(path)) as im:
            try:
                im.draft("RGB", im.size)
            except Exception:
                pass
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.uint8)
        img = arr.astype(np.float32) * (1.0 / 255.0)
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
    H, W = img2d.shape
    if H < 1 or W < 1:
        return np.full((out_h, out_w), np.nan, dtype=np.float32)

    ys = np.linspace(0, H, out_h + 1, dtype=np.int32)
    xs = np.linspace(0, W, out_w + 1, dtype=np.int32)

    ys[1:] = np.maximum(ys[1:], ys[:-1] + 1)
    xs[1:] = np.maximum(xs[1:], xs[:-1] + 1)
    ys = np.minimum(ys, H)
    xs = np.minimum(xs, W)
    ys[-1] = H
    xs[-1] = W

    row_sums = np.add.reduceat(img2d, ys[:-1], axis=0)
    block_sums = np.add.reduceat(row_sums, xs[:-1], axis=1)
    row_counts = (ys[1:] - ys[:-1]).astype(np.float32)
    col_counts = (xs[1:] - xs[:-1]).astype(np.float32)
    denom = row_counts[:, None] * col_counts[None, :]
    return (block_sums / denom).astype(np.float32)


def _rgb_to_hsv_np(img_rgb01: np.ndarray):
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
    cm = cmax > eps
    s[cm] = delta[cm] / (cmax[cm] + eps)

    v = cmax.astype(np.float32)
    return h.astype(np.float32), s.astype(np.float32), v.astype(np.float32)


def _dct8_matrix_ortho():
    N = 8
    k = np.arange(N, dtype=np.float32).reshape(-1, 1)
    n = np.arange(N, dtype=np.float32).reshape(1, -1)
    M = np.cos(np.pi / N * (n + 0.5) * k).astype(np.float32)
    M[0, :] *= np.sqrt(1.0 / N)
    M[1:, :] *= np.sqrt(2.0 / N)
    return M


_DCT8 = _dct8_matrix_ortho()


def _dct2_8x8_ortho(a8: np.ndarray):
    a8 = np.asarray(a8, dtype=np.float32)
    return (_DCT8 @ a8 @ _DCT8.T).astype(np.float32)


def _resolve_image_path(img_id: str):
    key = str(img_id)
    if key in _IMAGE_PATH_CACHE:
        return _IMAGE_PATH_CACHE[key]

    if _IMAGE_FILE_MAP is not None:
        p = _IMAGE_FILE_MAP.get(key)
        if p is not None:
            _IMAGE_PATH_CACHE[key] = p
            return p

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


_GRAY_W = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)


def build_features(df: pd.DataFrame):
    img_ids = df["image_id"].astype(str).to_list()
    n = len(img_ids)

    widths = np.empty(n, dtype=np.float32)
    heights = np.empty(n, dtype=np.float32)
    sizes = np.empty(n, dtype=np.float32)

    EXTRA_COLOR_INDEX_FEATS = 7  # unchanged existing block
    EXTRA_QUANT_FEATS = 12  # 3 quantiles * 4 maps (r8,g8,b8,thumb8)
    EXTRA_HSV_HIST_FEATS = 16  # 8-bin hue hist + 8-bin sat hist on 8x8

    F = (
        6  # means/stds
        + 3  # mean ratios
        + 64  # thumb
        + 6  # hsv moments
        + 3  # exg/exr/exb means
        + 10  # low-freq dct
        + 6  # center/border
        + 2  # gradients
        + 24  # rgb hists
        + 1  # ndi
        + EXTRA_COLOR_INDEX_FEATS
        + EXTRA_QUANT_FEATS
        + EXTRA_HSV_HIST_FEATS
    )
    pix_feats = np.empty((n, F), dtype=np.float32)

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
    row = np.empty(F, dtype=np.float32)

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

        H, W = img.shape[0], img.shape[1]
        gray = (img @ _GRAY_W).astype(np.float32, copy=False)

        pkey = str(p)
        thumb8 = _THUMB8_CACHE.get(pkey)
        if thumb8 is None:
            thumb8 = _downsample_mean(gray, 8, 8).astype(np.float32, copy=False)
            _THUMB8_CACHE[pkey] = thumb8

        r8 = _downsample_mean(img[..., 0], 8, 8)
        g8 = _downsample_mean(img[..., 1], 8, 8)
        b8 = _downsample_mean(img[..., 2], 8, 8)

        r_flat = r8.reshape(-1)
        g_flat = g8.reshape(-1)
        b_flat = b8.reshape(-1)
        ch_mean = np.array(
            [r_flat.mean(), g_flat.mean(), b_flat.mean()], dtype=np.float32
        )
        ch_std = np.array([r_flat.std(), g_flat.std(), b_flat.std()], dtype=np.float32)

        pos = 0
        row[pos : pos + 6] = (
            ch_mean[0],
            ch_mean[1],
            ch_mean[2],
            ch_std[0],
            ch_std[1],
            ch_std[2],
        )
        pos += 6

        row[pos : pos + 3] = (
            float(ch_mean[0] / (ch_mean[1] + eps)),
            float(ch_mean[0] / (ch_mean[2] + eps)),
            float(ch_mean[1] / (ch_mean[2] + eps)),
        )
        pos += 3

        row[pos : pos + 64] = thumb8.reshape(-1)
        pos += 64

        rgb8 = np.stack([r8, g8, b8], axis=-1).astype(np.float32, copy=False)
        h_ch, s_ch, v_ch = _rgb_to_hsv_np(rgb8)
        row[pos : pos + 6] = (
            float(h_ch.mean()),
            float(s_ch.mean()),
            float(v_ch.mean()),
            float(h_ch.std()),
            float(s_ch.std()),
            float(v_ch.std()),
        )
        pos += 6

        exg = (2.0 * g8 - r8 - b8).astype(np.float32, copy=False)
        exr = (1.4 * r8 - g8).astype(np.float32, copy=False)
        exb = (1.4 * b8 - g8).astype(np.float32, copy=False)
        row[pos : pos + 3] = (float(exg.mean()), float(exr.mean()), float(exb.mean()))
        pos += 3

        dct8 = _dct2_8x8_ortho(thumb8)
        row[pos : pos + 10] = dct8[coords[:, 0], coords[:, 1]].astype(np.float32)
        pos += 10

        y0, y1 = int(H * 0.25), int(H * 0.75)
        x0, x1 = int(W * 0.25), int(W * 0.75)
        center = img[y0:y1, x0:x1, :]
        if center.size == 0:
            center = img

        center_sum = center.sum(axis=(0, 1), dtype=np.float64)
        center_n = center.shape[0] * center.shape[1]
        center_mean = (center_sum / max(center_n, 1)).astype(np.float32)

        total_sum = img.sum(axis=(0, 1), dtype=np.float64)
        total_n = H * W
        border_n = total_n - center_n
        if border_n <= 0:
            border_mean = (total_sum / max(total_n, 1)).astype(np.float32)
        else:
            border_mean = ((total_sum - center_sum) / border_n).astype(np.float32)

        row[pos : pos + 6] = np.concatenate(
            [center_mean, (center_mean - border_mean)], axis=0
        ).astype(np.float32, copy=False)
        pos += 6

        if W > 1:
            gx = np.abs(gray[:, 1:] - gray[:, :-1]).mean()
        else:
            gx = 0.0
        if H > 1:
            gy = np.abs(gray[1:, :] - gray[:-1, :]).mean()
        else:
            gy = 0.0
        row[pos : pos + 2] = (float(gx), float(gy))
        pos += 2

        for ch, x in enumerate((r_flat, g_flat, b_flat)):
            bi = (x * 8.0).astype(np.int32, copy=False)
            np.clip(bi, 0, 7, out=bi)
            hst = np.bincount(bi, minlength=8).astype(np.float32, copy=False)
            hst /= hst.sum() + 1e-6
            row[pos + ch * 8 : pos + (ch + 1) * 8] = hst
        pos += 24

        ndi = (g_flat.mean() - r_flat.mean()) / (g_flat.mean() + r_flat.mean() + eps)
        row[pos] = float(ndi)
        pos += 1

        mr = float(ch_mean[0])
        mg = float(ch_mean[1])
        mb = float(ch_mean[2])
        row[pos : pos + EXTRA_COLOR_INDEX_FEATS] = (
            (mr - mg) / (mr + mg + eps),  # NDRG
            (mr - mb) / (mr + mb + eps),  # NDRB
            (mg - mb) / (mg + mb + eps),  # NDGB
            float(
                np.sqrt((mr - mg) ** 2 + (mr - mb) ** 2 + (mg - mb) ** 2)
            ),  # chroma (mean-space)
            float(
                np.sqrt(ch_std[0] ** 2 + ch_std[1] ** 2 + ch_std[2] ** 2)
            ),  # chroma-ish variability
            float(thumb8.mean()),  # gray mean on same 8x8
            float(thumb8.std()),  # gray std on same 8x8
        )
        pos += EXTRA_COLOR_INDEX_FEATS

        qs = np.array([0.2, 0.5, 0.8], dtype=np.float32)
        row[pos : pos + 3] = np.quantile(r_flat, qs).astype(np.float32)
        pos += 3
        row[pos : pos + 3] = np.quantile(g_flat, qs).astype(np.float32)
        pos += 3
        row[pos : pos + 3] = np.quantile(b_flat, qs).astype(np.float32)
        pos += 3
        row[pos : pos + 3] = np.quantile(thumb8.reshape(-1), qs).astype(np.float32)
        pos += 3

        h_flat = h_ch.reshape(-1)
        s_flat = s_ch.reshape(-1)
        hb = (h_flat * 8.0).astype(np.int32, copy=False)
        np.clip(hb, 0, 7, out=hb)
        hh = np.bincount(hb, minlength=8).astype(np.float32, copy=False)
        hh /= hh.sum() + 1e-6
        sb = (s_flat * 8.0).astype(np.int32, copy=False)
        np.clip(sb, 0, 7, out=sb)
        sh = np.bincount(sb, minlength=8).astype(np.float32, copy=False)
        sh /= sh.sum() + 1e-6
        row[pos : pos + 8] = hh
        pos += 8
        row[pos : pos + 8] = sh
        pos += 8

        pix_feats[idx, :] = row

    sizes = np.log1p(sizes.astype(np.float32, copy=False))

    X_meta = np.empty((n, 4), dtype=np.float32)
    X_meta[:, 0] = widths
    X_meta[:, 1] = heights
    X_meta[:, 2] = sizes
    X_meta[:, 3] = widths / (heights + 1e-6)

    colnames = []
    colnames += ["w", "h", "filesize", "aspect"]
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
    colnames += ["ndi_gr"]
    colnames += [
        "nd_rg",
        "nd_rb",
        "nd_gb",
        "chroma_mean",
        "chroma_std",
        "gray8_mean",
        "gray8_std",
    ]
    colnames += [
        "r_q20",
        "r_q50",
        "r_q80",
        "g_q20",
        "g_q50",
        "g_q80",
        "b_q20",
        "b_q50",
        "b_q80",
        "gray_q20",
        "gray_q50",
        "gray_q80",
    ]
    colnames += [f"hue_hist_{i}" for i in range(8)]
    colnames += [f"sat_hist_{i}" for i in range(8)]

    X_all = np.concatenate(
        [X_meta.astype(np.float64), pix_feats.astype(np.float64)], axis=1
    )
    return pd.DataFrame(X_all, columns=colnames)


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

    Xb = np.empty((n, d + 1), dtype=float)
    Xb[:, 0] = 1.0
    Xb[:, 1:] = X

    W = np.zeros((d + 1, k), dtype=float)
    Yf = Y.astype(float, copy=False)

    P = np.empty((n, k), dtype=float)
    G = np.empty((d + 1, k), dtype=float)

    for _ in range(n_iter):
        np.matmul(Xb, W, out=P)
        P[:] = sigmoid(P)
        np.subtract(P, Yf, out=P)  # reuse P as (P - Y)
        np.matmul(Xb.T, P, out=G)
        G *= 1.0 / n
        G[1:, :] += l2 * W[1:, :]
        W -= lr * G
    return W


def predict_logreg_ovr(X, W):
    n = X.shape[0]
    Xb = np.empty((n, X.shape[1] + 1), dtype=float)
    Xb[:, 0] = 1.0
    Xb[:, 1:] = X
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

row_sum = P_test.sum(axis=1, keepdims=True)
P_test = P_test / (row_sum + 1e-12)
P_test = np.clip(P_test, 1e-6, 1 - 1e-6)

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
