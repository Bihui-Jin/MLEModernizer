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

0.9667180849324454

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your notebook fails because it tries to read several external “../input/plantpathology/*.csv” submission files that do not exist in this environment, so `dsub` is never created and later cells crash. The minimal safe fix is to remove that dependency and instead generate predictions directly from the provided training/test labels using a simple, deterministic baseline that outputs class priors (mean of each label in `train.csv`) for every test row. This run end-to-end on the given dataset, produce a correctly formatted `submission.csv`, and should yield a reasonable ROC AUC (better than an all-zeros submission) without changing any “model architecture” (there wasn’t one here). Paths are adjusted to the actual provided dataset location while keeping the same overall flow (load → build submission → write CSV).'
- What this solution (achieved 0.63456) has done: 'You’re currently submitting constant class-prior probabilities, which tends to score near random (≈0.5 AUC) because it cannot rank images. Since we must keep the core approach minimal and we don’t have deep-learning libraries available, the smallest legitimate improvement is to add a lightweight image-feature model using only installed packages: read each JPG, compute simple color statistics, and train one LogisticRegression per label. This preserves the “load → build features → fit → predict → write submission.csv” flow, produces a valid submission, and should move the score upward toward your target without changing evaluation semantics. I’m also keeping paths compatible with your provided directory layout and ensuring the submission column order matches `sample_submission.csv`.'
- What this solution (achieved 0.61925) has done: 'Your current pipeline is already a lightweight image-feature + per-label LogisticRegression approach, so we keep that core logic unchanged and only make small, score-relevant improvements. The main issue holding AUC back is that the current features are too weak to rank images well, so we add a few inexpensive, deterministic texture/color features (downsampled grayscale histogram + simple edge energy stats) while keeping the same model family and training loop. We also switch `class_weight` from `"balanced"` to `None` (still LogisticRegression) because for ROC AUC, probability ranking often improves when the model learns true priors rather than reweighted ones. Finally, we ensure strict column order/alignment to `sample_submission.csv` and keep paths unchanged.'
- What this solution (achieved 0.61365) has done: 'Your current score is far below the target, so we should improve ranking ability without changing the overall “simple image features → per-label LogisticRegression → submission.csv” approach. The main low-risk gain is to enrich the feature vector with a small, deterministic HOG-like descriptor computed from the same resized grayscale image; this tends to help ROC AUC because it adds shape/texture cues that separate diseases better than global color stats alone. I also add a very small amount of L2 regularization tuning (still LogisticRegression, same training loop) by setting `C` to a slightly higher value to reduce underfitting given the expanded feature set. Paths, submission formatting, and the rest of your pipeline remain unchanged.'
- What this solution (achieved 0.59365) has done: 'Your score is far below the target, so we should improve AUC by strengthening image ranking while keeping the same core approach (hand-crafted features → per-label LogisticRegression → submission.csv). The lowest-risk gain is to fix the feature extraction so it uses the correct filenames in this dataset (`Train_*.jpg` / `Test_*.jpg`) instead of assuming `{image_id}.jpg`, which likely caused many images to be read incorrectly or not at all. I also add a tiny caching layer so repeated image reads don’t cost time, and I increase feature resolution slightly (resize 160×160, smaller HOG cell) to capture more disease texture without changing the model family or training loop. Submission formatting and paths remain the same and the script still writes `submission.csv`.'
- What this solution (achieved 0.61039) has done: 'Your current approach is correct but the feature/model capacity is underpowered for ranking well, so the score is far below the target. I keep the exact same pipeline (hand-crafted features → per-label LogisticRegression → submission.csv) and make two minimal, score-relevant upgrades: add a compact multi-scale HOG-like descriptor plus simple spatial pooling of color/gray stats to capture lesion localization, and slightly increase `C` to reduce underfitting with the larger feature vector. I also make image reading more robust to the dataset’s `Train_*.jpg` / `Test_*.jpg` naming by adding a safe fallback search, without changing paths or the overall flow. Submission formatting and column order remain strictly aligned to `sample_submission.csv`.'
- What this solution (achieved 0.62508) has done: 'Your current score is far below the target, so we should improve the model’s ability to rank images while keeping the same core pipeline (hand-crafted image features → per-label LogisticRegression → submission.csv). The biggest low-risk issue is that LogisticRegression is currently fit independently per label without any calibration and with a relatively small feature representation of spatial structure; we can improve ranking by adding a compact, deterministic “downsampled grayscale patch” feature (a tiny bag of pixels) that often boosts separability for leaf lesions without changing the modeling family. We also set `n_jobs=-1` (speed, not semantics) and use a slightly stronger regularization (`C=6.0`) to reduce underfitting with the expanded feature vector (still the same LogisticRegression training loop). All paths, submission column order, and output filename remain unchanged, and this still runs end-to-end within the time limit.'
- What this solution (achieved 0.59521) has done: 'Your current score is far below the target, so we should improve ranking quality while keeping the exact same core pipeline (hand-crafted features → per-label LogisticRegression → submission.csv). The most likely bottleneck is that the features are not invariant to illumination and background, so I add a minimal “per-image color constancy” normalization (gray-world) before computing the same kinds of features; this often improves separability for disease patterns without changing model family/training semantics. I also switch the LogisticRegression to use `penalty="l2"` with `solver="liblinear"` (still LogisticRegression, same one-vs-rest-per-label loop) because on small/medium tabular feature sets it can yield better probability ranking than lbfgs here, while remaining deterministic. Finally, I keep submission formatting strictly aligned to `sample_submission.csv` as you already do.'
- What this solution (achieved 0.62485) has done: 'We keep your exact pipeline (hand-crafted image features → per-label LogisticRegression → submission.csv) but make two minimal, score-relevant adjustments aimed at improving ROC AUC ranking: (1) compute StandardScaler parameters on the training set once and reuse them for all four labels (same transformation, less per-label fitting noise), and (2) switch LogisticRegression back to `solver="lbfgs"` with `n_jobs=-1` (still L2 logistic regression, same training loop) which tends to yield smoother probability estimates on this kind of dense feature vector than `liblinear`. Everything else (feature set, image resolution, gray-world normalization, per-label independent models, output formatting/paths) remains unchanged, and the script still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
BASE_INPUT = "/kaggle/input"

CANDIDATES = [
    os.path.join("/kaggle", "data", "plant-pathology-2020-fgvc7"),
    os.path.join("/kaggle", "data"),
    os.path.join(BASE_INPUT, "plant-pathology-2020-fgvc7"),
    BASE_INPUT,
]

COMP_DIR = None
for c in CANDIDATES:
    if os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(
        os.path.join(c, "test.csv")
    ):
        COMP_DIR = c
        break

if COMP_DIR is None:
    raise FileNotFoundError(
        "Could not find train.csv/test.csv under expected Kaggle input/data directories."
    )

train_path = os.path.join(COMP_DIR, "train.csv")
test_path = os.path.join(COMP_DIR, "test.csv")
sample_path = os.path.join(COMP_DIR, "sample_submission.csv")

images_dir = os.path.join(COMP_DIR, "images")
if not os.path.isdir(images_dir):
    alt_images_dir = os.path.join("/kaggle", "data", "images")
    if os.path.isdir(alt_images_dir):
        images_dir = alt_images_dir

for p in [train_path, test_path, sample_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required file not found: {p}")

if not os.path.isdir(images_dir):
    raise FileNotFoundError(f"Images directory not found: {images_dir}")

print("Using COMP_DIR =", COMP_DIR)
print("Using images_dir =", images_dir)
print(
    "Files:",
    [
        os.path.basename(train_path),
        os.path.basename(test_path),
        os.path.basename(sample_path),
    ],
)



## === cell 2
from PIL import Image
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

missing_train = [c for c in (["image_id"] + target_cols) if c not in train.columns]
missing_test = [c for c in (["image_id"]) if c not in test.columns]
missing_sub = [c for c in (["image_id"] + target_cols) if c not in sample_sub.columns]
if missing_train:
    raise ValueError(f"train.csv missing columns: {missing_train}")
if missing_test:
    raise ValueError(f"test.csv missing columns: {missing_test}")
if missing_sub:
    raise ValueError(f"sample_submission.csv missing columns: {missing_sub}")

np.random.seed(42)

_IMAGE_ID_TO_PATH = None


def _build_image_index():
    idx = {}
    for de in os.scandir(images_dir):
        if not de.is_file():
            continue
        fn = de.name
        if not fn.lower().endswith(".jpg"):
            continue
        p = de.path
        idx[fn] = p
        idx[os.path.splitext(fn)[0]] = p
    return idx


def _resolve_image_path(image_id: str) -> str:
    global _IMAGE_ID_TO_PATH
    if _IMAGE_ID_TO_PATH is None:
        _IMAGE_ID_TO_PATH = _build_image_index()

    key = str(image_id)
    if key in _IMAGE_ID_TO_PATH:
        return _IMAGE_ID_TO_PATH[key]
    if f"{key}.jpg" in _IMAGE_ID_TO_PATH:
        return _IMAGE_ID_TO_PATH[f"{key}.jpg"]

    for pref in ("Train_", "Test_", "train_", "test_"):
        k2 = f"{pref}{key}"
        if k2 in _IMAGE_ID_TO_PATH:
            return _IMAGE_ID_TO_PATH[k2]
        if f"{k2}.jpg" in _IMAGE_ID_TO_PATH:
            return _IMAGE_ID_TO_PATH[f"{k2}.jpg"]

    raise FileNotFoundError(f"Image not found for image_id={image_id} in {images_dir}")


_IMG_CACHE_RESIZED_RGB_U8 = {}


def _open_resize_rgb_u8(path, resize):
    key = (path, resize)
    arr = _IMG_CACHE_RESIZED_RGB_U8.get(key)
    if arr is not None:
        return arr
    with Image.open(path) as im:
        im = im.convert("RGB")
        if resize is not None:
            im = im.resize(resize)  # preserve original resample behavior (PIL default)
        arr = np.asarray(im, dtype=np.uint8)
    _IMG_CACHE_RESIZED_RGB_U8[key] = arr
    return arr


_HOG_BIN_EDGES_9 = np.linspace(0.0, np.pi, 9 + 1, dtype=np.float32)


def _hog_like(gray, cell_size=12, num_bins=9):
    h, w = gray.shape
    if h < 3 or w < 3:
        return np.zeros((0,), dtype=np.float32)

    gx = gray[:, 2:] - gray[:, :-2]
    gy = gray[2:, :] - gray[:-2, :]
    gx = gx[1:-1, :]
    gy = gy[:, 1:-1]

    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32, copy=False)
    ang = (np.arctan2(gy, gx) % np.pi).astype(np.float32, copy=False)

    hh, ww = mag.shape
    ncy = hh // cell_size
    ncx = ww // cell_size
    if ncy == 0 or ncx == 0:
        return np.zeros((0,), dtype=np.float32)

    hh2 = ncy * cell_size
    ww2 = ncx * cell_size
    mag = mag[:hh2, :ww2]
    ang = ang[:hh2, :ww2]

    if num_bins == 9:
        bin_edges = _HOG_BIN_EDGES_9
    else:
        bin_edges = np.linspace(0.0, np.pi, num_bins + 1, dtype=np.float32)

    b = (np.searchsorted(bin_edges, ang, side="right") - 1).astype(np.int32, copy=False)
    np.clip(b, 0, num_bins - 1, out=b)

    cell_id = (np.arange(hh2, dtype=np.int32) // cell_size)[:, None] * ncx + (
        np.arange(ww2, dtype=np.int32) // cell_size
    )[None, :]
    idx = (cell_id.reshape(-1) * num_bins + b.reshape(-1)).astype(np.int32, copy=False)
    hist_flat = np.bincount(
        idx, weights=mag.reshape(-1), minlength=(ncy * ncx * num_bins)
    ).astype(np.float32, copy=False)
    hist = hist_flat.reshape(ncy, ncx, num_bins)

    norms = np.linalg.norm(hist, axis=2, keepdims=True)
    hist = hist / (norms + 1e-6)
    return hist.reshape(-1).astype(np.float32, copy=False)


def _spatial_pool_stats(arr01, grid=(2, 2)):
    h, w, _ = arr01.shape
    gy, gx = grid
    ys = np.linspace(0, h, gy + 1).astype(int)
    xs = np.linspace(0, w, gx + 1).astype(int)

    feats = []
    for i in range(gy):
        y0, y1 = ys[i], ys[i + 1]
        for j in range(gx):
            x0, x1 = xs[j], xs[j + 1]
            patch = arr01[y0:y1, x0:x1, :]
            if patch.size == 0:
                feats.append(np.zeros((6,), dtype=np.float32))
                continue
            flat = patch.reshape(-1, 3)
            m = flat.mean(axis=0).astype(np.float32, copy=False)
            s = flat.std(axis=0).astype(np.float32, copy=False)
            feats.append(np.concatenate([m, s], axis=0))
    return np.concatenate(feats, axis=0).astype(np.float32, copy=False)


def _tiny_gray_patch(gray01, out_hw=(24, 24)):
    im = Image.fromarray(np.clip(gray01 * 255.0, 0, 255).astype(np.uint8), mode="L")
    im = im.resize((out_hw[1], out_hw[0]), resample=Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = a.reshape(-1)
    a = (a - a.mean()) / (a.std() + 1e-6)
    return a.astype(np.float32, copy=False)


def _gray_world_normalize(arr01, eps=1e-6):
    ch_mean = arr01.reshape(-1, 3).mean(axis=0).astype(np.float32, copy=False)
    scale = (ch_mean.mean() / (ch_mean + eps)).astype(np.float32, copy=False)
    out = arr01 * scale[None, None, :]
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def extract_features(image_id, resize=(160, 160), hist_bins=16):
    img_path = _resolve_image_path(image_id)

    rgb_u8 = _open_resize_rgb_u8(img_path, resize)

    arr = rgb_u8.astype(np.float32, copy=False) / 255.0
    arr = _gray_world_normalize(arr)

    flat = arr.reshape(-1, 3)
    ch_mean = flat.mean(axis=0).astype(np.float32, copy=False)
    ch_std = flat.std(axis=0).astype(np.float32, copy=False)

    gray = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]).astype(
        np.float32,
        copy=False,
    )
    g_mean = np.float32(gray.mean())
    g_std = np.float32(gray.std())

    dh = (
        np.float32(np.abs(gray[:, 1:] - gray[:, :-1]).mean())
        if gray.shape[1] > 1
        else np.float32(0.0)
    )
    dv = (
        np.float32(np.abs(gray[1:, :] - gray[:-1, :]).mean())
        if gray.shape[0] > 1
        else np.float32(0.0)
    )

    hist, _ = np.histogram(gray, bins=hist_bins, range=(0.0, 1.0))
    hist = hist.astype(np.float32, copy=False)
    hist = hist / (hist.sum() + 1e-6)

    if gray.shape[0] > 2 and gray.shape[1] > 2:
        gx2 = gray[1:-1, 2:] - gray[1:-1, :-2]
        gy2 = gray[2:, 1:-1] - gray[:-2, 1:-1]
        mag = np.sqrt(gx2 * gx2 + gy2 * gy2).astype(np.float32, copy=False)
        edge_mean = np.float32(mag.mean())
        edge_std = np.float32(mag.std())
        edge_p90 = np.float32(np.quantile(mag.reshape(-1), 0.90))
    else:
        edge_mean, edge_std, edge_p90 = (
            np.float32(0.0),
            np.float32(0.0),
            np.float32(0.0),
        )

    hog_s1 = _hog_like(gray, cell_size=12, num_bins=9)
    hog_s2 = _hog_like(gray, cell_size=20, num_bins=9)

    pool = _spatial_pool_stats(arr, grid=(2, 2))
    tiny = _tiny_gray_patch(gray, out_hw=(24, 24))

    feat = np.concatenate(
        [
            ch_mean,
            ch_std,
            np.array(
                [g_mean, g_std, dh, dv, edge_mean, edge_std, edge_p90], dtype=np.float32
            ),
            hist,
            pool,
            hog_s1,
            hog_s2,
            tiny,
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    return feat


train_ids = train["image_id"].values
test_ids = test["image_id"].values


def _mp_init(image_index):
    global _IMAGE_ID_TO_PATH
    _IMAGE_ID_TO_PATH = image_index


def _featurize_ids(ids):
    ids = list(ids)
    global _IMAGE_ID_TO_PATH
    if _IMAGE_ID_TO_PATH is None:
        _IMAGE_ID_TO_PATH = _build_image_index()
    image_index = _IMAGE_ID_TO_PATH

    try:
        import multiprocessing as mp

        n_workers = min(8, (os.cpu_count() or 2))
        ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp

        chunksize = 32 if len(ids) >= 512 else 16

        with ctx.Pool(
            processes=n_workers, initializer=_mp_init, initargs=(image_index,)
        ) as pool:
            feats = list(pool.imap(extract_features, ids, chunksize=chunksize))
    except Exception:
        feats = list(map(extract_features, ids))

    D = feats[0].shape[0]
    X = np.empty((len(feats), D), dtype=np.float32)
    for i, f in enumerate(feats):
        X[i] = f
    return X


X_train = _featurize_ids(train_ids)
X_test = _featurize_ids(test_ids)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_sp = poly.fit_transform(X_train_s).astype(np.float32, copy=False)
X_test_sp = poly.transform(X_test_s).astype(np.float32, copy=False)

models = {}
test_pred = np.zeros((len(test), len(target_cols)), dtype=np.float32)

for j, c in enumerate(target_cols):
    y = train[c].astype(int).values

    clf = LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        max_iter=3000,
        class_weight=None,
        random_state=42,
        C=6.0,
        n_jobs=1,  # keep to avoid joblib temp-file issues
    )
    clf.fit(X_train_sp, y)
    test_pred[:, j] = clf.predict_proba(X_test_sp)[:, 1].astype(np.float32, copy=False)
    models[c] = clf

sub = pd.DataFrame({"image_id": test["image_id"].values})
for j, c in enumerate(target_cols):
    sub[c] = np.clip(test_pred[:, j], 0.0, 1.0)

sub = sub[sample_sub.columns.tolist()]
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
