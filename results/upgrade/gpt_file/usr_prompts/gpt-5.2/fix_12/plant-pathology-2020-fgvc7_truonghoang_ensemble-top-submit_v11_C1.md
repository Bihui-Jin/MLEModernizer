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

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



## === cell 1
BASE_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "../input/plant-pathology-2020-fgvc7",
    "../kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
    "../input",
]

BASE = None
for c in BASE_CANDIDATES:
    if os.path.exists(c):
        if os.path.isfile(os.path.join(c, "train.csv")) and os.path.isdir(
            os.path.join(c, "images")
        ):
            BASE = c
            break

if BASE is None:
    roots = ["/kaggle/input", "/kaggle/data", "../input", "../kaggle/data", "/"]
    found = []
    for r in roots:
        if os.path.exists(r):
            for dirpath, dirnames, filenames in os.walk(r):
                if (
                    "train.csv" in filenames
                    and "test.csv" in filenames
                    and "images" in dirnames
                ):
                    found.append(dirpath)
            if found:
                break
    if found:
        BASE = found[0]

if BASE is None:
    raise FileNotFoundError(
        "Could not locate competition dataset folder containing train.csv/test.csv/images."
    )

print("Using BASE:", BASE)
print(
    "Files:",
    [
        f
        for f in ["train.csv", "test.csv", "sample_submission.csv"]
        if os.path.exists(os.path.join(BASE, f))
    ],
)

TRAIN_CSV = os.path.join(BASE, "train.csv")
TEST_CSV = os.path.join(BASE, "test.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")
IMAGES_DIR = os.path.join(BASE, "images")



## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

target_cols = [c for c in sample_sub.columns if c != "image_id"]
assert "image_id" in sample_sub.columns
assert all(
    c in train_df.columns for c in ["image_id"] + target_cols
), "Train CSV missing expected target columns."
assert all(
    c in sample_sub.columns for c in ["image_id"] + target_cols
), "Sample submission missing expected columns."

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Targets:", target_cols)



## === cell 3
try:
    from PIL import Image
except Exception as e:
    raise ImportError(
        "PIL (Pillow) is required to read images but is not available in this environment."
    ) from e


def image_path(image_id: str) -> str:
    return (
        os.path.join(IMAGES_DIR, f"{image_id}.jpg")
        if not image_id.lower().endswith(".jpg")
        else os.path.join(IMAGES_DIR, image_id)
    )


def _lbp_hist(gray01: np.ndarray, bins: int = 16) -> np.ndarray:
    g = gray01
    c = g[1:-1, 1:-1]
    code = np.zeros_like(c, dtype=np.uint8)
    code |= (g[:-2, :-2] >= c).astype(np.uint8) << 7
    code |= (g[:-2, 1:-1] >= c).astype(np.uint8) << 6
    code |= (g[:-2, 2:] >= c).astype(np.uint8) << 5
    code |= (g[1:-1, 2:] >= c).astype(np.uint8) << 4
    code |= (g[2:, 2:] >= c).astype(np.uint8) << 3
    code |= (g[2:, 1:-1] >= c).astype(np.uint8) << 2
    code |= (g[2:, :-2] >= c).astype(np.uint8) << 1
    code |= (g[1:-1, :-2] >= c).astype(np.uint8) << 0

    grp = (code.astype(np.int32) // 16).reshape(-1)
    h, _ = np.histogram(grp, bins=bins, range=(0, bins), density=False)
    h = h.astype(np.float32)
    h /= h.sum() + 1e-6
    return h


def _glcm_stats_quant(gray01: np.ndarray, levels: int = 8) -> np.ndarray:
    q = np.clip((gray01 * (levels - 1) + 0.5).astype(np.int32), 0, levels - 1)
    a = q[:, :-1].reshape(-1)
    b = q[:, 1:].reshape(-1)

    M = np.zeros((levels, levels), dtype=np.float32)
    np.add.at(M, (a, b), 1.0)
    np.add.at(M, (b, a), 1.0)  # symmetric
    P = M / (M.sum() + 1e-6)

    i = np.arange(levels, dtype=np.float32)
    j = np.arange(levels, dtype=np.float32)
    I, J = np.meshgrid(i, j, indexing="ij")

    contrast = np.sum(P * (I - J) ** 2)
    homogeneity = np.sum(P / (1.0 + np.abs(I - J)))
    energy = np.sum(P * P)
    entropy = -np.sum(P * np.log(P + 1e-6))

    return np.array([contrast, homogeneity, energy, entropy], dtype=np.float32)


def extract_features_one(img_fp: str, size=(48, 48), hist_bins=16) -> np.ndarray:
    with Image.open(img_fp) as im:
        im_rgb = im.convert("RGB").resize(size, resample=Image.BILINEAR)
        arr = np.asarray(im_rgb, dtype=np.float32) / 255.0  # (H, W, 3)

        im_hsv = im_rgb.convert("HSV")
        hsv = np.asarray(im_hsv, dtype=np.float32) / 255.0

        im_ycc = im_rgb.convert("YCbCr")
        ycc = np.asarray(im_ycc, dtype=np.float32)  # ranges roughly [0..255]

    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]

    eps = 1e-6
    inten = r + g + b + eps
    r_n = r / inten
    g_n = g / inten
    b_n = b / inten

    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.float32)

    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    gmag = np.sqrt(gx * gx + gy * gy).astype(np.float32)

    ch_mean = arr.mean(axis=(0, 1)).astype(np.float32)
    ch_std = arr.std(axis=(0, 1)).astype(np.float32)

    chn_mean = np.array([r_n.mean(), g_n.mean(), b_n.mean()], dtype=np.float32)
    chn_std = np.array([r_n.std(), g_n.std(), b_n.std()], dtype=np.float32)

    gray_stats = np.array([gray.mean(), gray.std()], dtype=np.float32)
    grad_stats = np.array(
        [gmag.mean(), gmag.std(), np.percentile(gmag, 90)], dtype=np.float32
    )

    exg = (2.0 * g - r - b).astype(np.float32)  # excess green
    gr_ratio = (g + eps) / (r + eps)
    gb_ratio = (g + eps) / (b + eps)
    rg_diff = (r - g).astype(np.float32)

    ngrdi = ((g - r) / (g + r + eps)).astype(np.float32)
    vari = ((g - r) / (g + r - b + eps)).astype(np.float32)
    gli = ((2.0 * g - r - b) / (2.0 * g + r + b + eps)).astype(np.float32)

    def _zstats(x: np.ndarray) -> tuple:
        mu = float(x.mean())
        sd = float(x.std() + 1e-6)
        z = (x - mu) / sd
        return (
            np.float32(mu),
            np.float32(sd),
            np.float32(np.percentile(z, 10)),
            np.float32(np.percentile(z, 90)),
        )

    exg_mu, exg_sd, exg_z10, exg_z90 = _zstats(exg)
    rg_mu, rg_sd, rg_z10, rg_z90 = _zstats(rg_diff)

    idx_stats = np.array(
        [
            exg_mu,
            exg_sd,
            exg_z10,
            exg_z90,
            gr_ratio.mean(),
            gb_ratio.mean(),
            rg_mu,
            rg_sd,
            rg_z10,
            rg_z90,
            ngrdi.mean(),
            ngrdi.std(),
            vari.mean(),
            vari.std(),
            gli.mean(),
            gli.std(),
        ],
        dtype=np.float32,
    )

    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]
    hsv_stats = np.array(
        [h.mean(), h.std(), s.mean(), s.std(), v.mean(), v.std()],
        dtype=np.float32,
    )

    Y = ycc[..., 0] / 255.0
    Cb = (ycc[..., 1] - 128.0) / 128.0
    Cr = (ycc[..., 2] - 128.0) / 128.0
    ycc_stats = np.array(
        [Y.mean(), Y.std(), Cb.mean(), Cb.std(), Cr.mean(), Cr.std()],
        dtype=np.float32,
    )

    def _hist(x01: np.ndarray, bins: int) -> np.ndarray:
        h_, _ = np.histogram(x01.reshape(-1), bins=bins, range=(0.0, 1.0), density=True)
        h_ = np.clip(h_, 0.0, 10.0)
        return h_.astype(np.float32)

    hist_gray = _hist(gray, hist_bins)
    hist_r = _hist(r, hist_bins)
    hist_g = _hist(g, hist_bins)
    hist_b = _hist(b, hist_bins)
    hist_gmag = _hist(np.clip(gmag, 0.0, 1.0), hist_bins)

    gray01 = np.clip(gray, 0.0, 1.0)
    lbp16 = _lbp_hist(gray01, bins=16)
    glcm4 = _glcm_stats_quant(gray01, levels=8)

    flat_gray = gray.reshape(-1).astype(np.float32)
    flat_gmag = gmag.reshape(-1).astype(np.float32)

    feat = np.concatenate(
        [
            flat_gray,
            flat_gmag,
            ch_mean,
            ch_std,
            chn_mean,
            chn_std,
            gray_stats,
            grad_stats,
            idx_stats,
            hsv_stats,
            ycc_stats,
            hist_gray,
            hist_r,
            hist_g,
            hist_b,
            hist_gmag,
            lbp16,
            glcm4,
        ],
        axis=0,
    )
    return feat.astype(np.float32)


def extract_features(df: pd.DataFrame) -> np.ndarray:
    ids = df["image_id"].astype(str).tolist()

    def _resolve_fp(image_id: str) -> str:
        fp = image_path(image_id)
        if os.path.exists(fp):
            return fp
        alt = os.path.join(IMAGES_DIR, image_id)
        if os.path.exists(alt):
            return alt
        return ""

    fps = [_resolve_fp(i) for i in ids]
    missing = sum(1 for fp in fps if not fp)
    if missing:
        print(f"Warning: missing {missing} images; using zero features for them.")

    first_fp = next((fp for fp in fps if fp), "")
    if not first_fp:
        raise RuntimeError(
            "No images were found/read successfully; cannot build features."
        )
    first_feat = extract_features_one(first_fp)
    dim = int(first_feat.shape[0])

    X = np.zeros((len(fps), dim), dtype=np.float32)

    def _compute(fp: str):
        if not fp:
            return None
        return extract_features_one(fp)

    try:
        import concurrent.futures as _cf

        max_workers = min(8, (os.cpu_count() or 2))
        with _cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            for i, feat in enumerate(ex.map(_compute, fps, chunksize=16)):
                if feat is not None:
                    X[i] = feat
    except Exception:
        for i, fp in enumerate(fps):
            if fp:
                X[i] = extract_features_one(fp)

    return X


X_train = extract_features(train_df)
X_test = extract_features(test_df)

Y_train_multi = train_df[target_cols].astype(np.float32).values

print(
    "X_train:", X_train.shape, "Y_train:", Y_train_multi.shape, "X_test:", X_test.shape
)



## === cell 4
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression

y_class = np.argmax(Y_train_multi, axis=1).astype(np.int64)

row_sums = Y_train_multi.sum(axis=1)
if not np.all((row_sums == 1.0) | (row_sums == 0.0)):
    print(
        "Warning: train targets are not strictly one-hot; proceeding with argmax labels."
    )

PIX_DIM = 48 * 48
FLAT_DIM = 2 * PIX_DIM  # flat_gray + flat_gmag
assert (
    X_train.shape[1] > FLAT_DIM + 10
), "Unexpected feature dimensionality; cannot split tail safely."

Xtr_flat, Xtr_tail = X_train[:, :FLAT_DIM], X_train[:, FLAT_DIM:]
Xte_flat, Xte_tail = X_test[:, :FLAT_DIM], X_test[:, FLAT_DIM:]

poly = PolynomialFeatures(degree=2, include_bias=False)
Xtr_tail2 = poly.fit_transform(Xtr_tail)
Xte_tail2 = poly.transform(Xte_tail)

X_train2 = np.concatenate([Xtr_flat, Xtr_tail2], axis=1).astype(np.float32)
X_test2 = np.concatenate([Xte_flat, Xte_tail2], axis=1).astype(np.float32)

clf = LogisticRegression(
    solver="saga",
    penalty="l2",
    multi_class="multinomial",
    max_iter=12000,
    C=3.0,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=min(4, (os.cpu_count() or 2)),
)

model = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", clf),
    ]
)

model.fit(X_train2, y_class)

proba_raw = model.predict_proba(X_test2).astype(np.float32)

n = X_test2.shape[0]
proba = np.zeros((n, len(target_cols)), dtype=np.float32)
classes_seen = model.named_steps["clf"].classes_.astype(int)

if proba_raw.shape[1] != len(classes_seen):
    raise RuntimeError(
        f"Unexpected proba_raw shape {proba_raw.shape} vs classes {classes_seen}."
    )

for j, cls_idx in enumerate(classes_seen):
    if 0 <= cls_idx < len(target_cols):
        proba[:, cls_idx] = proba_raw[:, j]
    else:
        raise RuntimeError(
            f"Unexpected class index {cls_idx}; expected within [0, {len(target_cols)-1}]"
        )

proba = np.clip(proba, 0.0, 1.0)
print("Pred proba shape:", proba.shape)



## === cell 5
sub = pd.DataFrame({"image_id": test_df["image_id"].astype(str).values})
for j, c in enumerate(target_cols):
    sub[c] = proba[:, j].astype(np.float32)

sub = sub[["image_id"] + target_cols]

assert sub.shape[0] == test_df.shape[0]
assert list(sub.columns) == list(sample_sub.columns)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub.head())
