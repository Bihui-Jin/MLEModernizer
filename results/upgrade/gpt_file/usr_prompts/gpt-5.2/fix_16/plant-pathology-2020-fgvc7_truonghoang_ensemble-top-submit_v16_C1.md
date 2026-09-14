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



## === cell 1
BASE_PATH = "/kaggle/data/plant-pathology-2020-fgvc7"
if not os.path.exists(BASE_PATH):
    BASE_PATH = "/kaggle/input/plant-pathology-2020-fgvc7"

TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_CSV = os.path.join(BASE_PATH, "test.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_PATH, "sample_submission.csv")
IMAGES_DIR = os.path.join(BASE_PATH, "images")

print("BASE_PATH:", BASE_PATH)
print("Exists TRAIN_CSV:", os.path.exists(TRAIN_CSV))
print("Exists TEST_CSV:", os.path.exists(TEST_CSV))
print("Exists SAMPLE_SUB_CSV:", os.path.exists(SAMPLE_SUB_CSV))
print("Exists IMAGES_DIR:", os.path.exists(IMAGES_DIR))



## === cell 2
from PIL import Image

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

target_cols = [c for c in sample_sub.columns if c != "image_id"]
assert target_cols == [
    "healthy",
    "multiple_diseases",
    "rust",
    "scab",
], f"Unexpected target columns: {target_cols}"


def _resolve_image_path(image_id: str) -> str:
    return os.path.join(IMAGES_DIR, f"{image_id}.jpg")


def _hist_channel(ch_0_1: np.ndarray, bins: int = 16) -> np.ndarray:
    h, _ = np.histogram(ch_0_1, bins=bins, range=(0.0, 1.0), density=True)
    return h.astype(np.float32)


def _hist_range(x: np.ndarray, bins: int, vmin: float, vmax: float) -> np.ndarray:
    h, _ = np.histogram(x, bins=bins, range=(vmin, vmax), density=True)
    return h.astype(np.float32)


def _hist2d_hs(
    Hc_0_1: np.ndarray, Sc_0_1: np.ndarray, bins_h: int = 12, bins_s: int = 8
) -> np.ndarray:
    h2, _, _ = np.histogram2d(
        Hc_0_1.ravel(),
        Sc_0_1.ravel(),
        bins=(bins_h, bins_s),
        range=((0.0, 1.0), (0.0, 1.0)),
        density=True,
    )
    return h2.astype(np.float32).ravel()


def _lbp_uniform_riu2(gray_0_1: np.ndarray, radius: int = 1) -> np.ndarray:
    g = (gray_0_1 * 255.0).astype(np.float32)
    H, W = g.shape
    r = int(radius)
    if H < 2 * r + 1 or W < 2 * r + 1:
        out = np.zeros(10, dtype=np.float32)
        out[-1] = 1.0
        return out

    c = g[r : H - r, r : W - r]
    n0 = g[: H - 2 * r, : W - 2 * r]
    n1 = g[: H - 2 * r, r : W - r]
    n2 = g[: H - 2 * r, 2 * r :]
    n3 = g[r : H - r, 2 * r :]
    n4 = g[2 * r :, 2 * r :]
    n5 = g[2 * r :, r : W - r]
    n6 = g[2 * r :, : W - 2 * r]
    n7 = g[r : H - r, : W - 2 * r]

    bits = [
        (n0 >= c).astype(np.uint8),
        (n1 >= c).astype(np.uint8),
        (n2 >= c).astype(np.uint8),
        (n3 >= c).astype(np.uint8),
        (n4 >= c).astype(np.uint8),
        (n5 >= c).astype(np.uint8),
        (n6 >= c).astype(np.uint8),
        (n7 >= c).astype(np.uint8),
    ]
    ones = np.zeros_like(bits[0], dtype=np.uint8)
    for b in bits:
        ones += b

    transitions = np.zeros_like(bits[0], dtype=np.uint8)
    for i in range(8):
        transitions += (bits[i] ^ bits[(i + 1) % 8]).astype(np.uint8)

    code = np.where(transitions <= 2, ones, 9).astype(np.uint8)
    hist = np.bincount(code.ravel(), minlength=10).astype(np.float32)
    hist /= hist.sum() + 1e-12
    return hist


def _gaussian_blur_3x3(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    p = np.pad(g, ((1, 1), (1, 1)), mode="reflect")
    out = (
        1.0 * p[:-2, :-2]
        + 2.0 * p[:-2, 1:-1]
        + 1.0 * p[:-2, 2:]
        + 2.0 * p[1:-1, :-2]
        + 4.0 * p[1:-1, 1:-1]
        + 2.0 * p[1:-1, 2:]
        + 1.0 * p[2:, :-2]
        + 2.0 * p[2:, 1:-1]
        + 1.0 * p[2:, 2:]
    ) / 16.0
    return out.astype(np.float32)


def _hog_orientation_histograms(
    gray_0_1: np.ndarray,
    cell_size: int = 12,
    bins: int = 18,
    clip: float = 0.2,
) -> np.ndarray:
    g = gray_0_1.astype(np.float32)
    gx = np.zeros_like(g, dtype=np.float32)
    gy = np.zeros_like(g, dtype=np.float32)
    gx[:, 1:] = g[:, 1:] - g[:, :-1]
    gy[1:, :] = g[1:, :] - g[:-1, :]

    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    ang = np.arctan2(gy, gx).astype(np.float32)
    ang = np.where(ang < 0, ang + 2.0 * np.pi, ang)
    bin_f = (ang / (2.0 * np.pi)) * bins
    b0 = np.floor(bin_f).astype(np.int32)
    b0 = np.clip(b0, 0, bins - 1)

    H, W = g.shape
    ncy = H // cell_size
    ncx = W // cell_size
    if ncy == 0 or ncx == 0:
        return np.zeros(bins, dtype=np.float32)

    feats = []
    for cy in range(ncy):
        y0 = cy * cell_size
        y1 = y0 + cell_size
        for cx in range(ncx):
            x0 = cx * cell_size
            x1 = x0 + cell_size
            b = b0[y0:y1, x0:x1].ravel()
            m = mag[y0:y1, x0:x1].ravel()
            h = np.bincount(b, weights=m, minlength=bins).astype(np.float32)

            h /= np.sqrt((h * h).sum()) + 1e-12
            if clip is not None and clip > 0:
                h = np.minimum(h, np.float32(clip))
                h /= np.sqrt((h * h).sum()) + 1e-12

            feats.append(h)
    return np.concatenate(feats, axis=0).astype(np.float32)


def _rgb_to_hsv01(arr_rgb_0_1: np.ndarray):
    r = arr_rgb_0_1[..., 0].astype(np.float32)
    g = arr_rgb_0_1[..., 1].astype(np.float32)
    b = arr_rgb_0_1[..., 2].astype(np.float32)

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(cmax, dtype=np.float32)
    mask = delta > 1e-12

    m = mask & (cmax == r)
    h[m] = ((g[m] - b[m]) / (delta[m] + 1e-12)) % 6.0

    m = mask & (cmax == g)
    h[m] = ((b[m] - r[m]) / (delta[m] + 1e-12)) + 2.0

    m = mask & (cmax == b)
    h[m] = ((r[m] - g[m]) / (delta[m] + 1e-12)) + 4.0

    h = (h / 6.0).astype(np.float32)
    h = np.where(h < 0, h + 1.0, h)

    s = np.zeros_like(cmax, dtype=np.float32)
    m = cmax > 1e-12
    s[m] = (delta[m] / (cmax[m] + 1e-12)).astype(np.float32)

    v = cmax.astype(np.float32)
    return h, s, v


def extract_features(
    image_ids,
    size=(96, 96),
    hist_bins=16,
    hsv_bins=12,
    opp_bins=24,
    hs2d_bins_h=12,
    hs2d_bins_s=8,
    w_rgb=0.22,
    w_gray=0.22,
    w_edge=0.28,
    w_stats=3.0,
    w_hist=4.0,
    w_idx=3.5,
    w_hsv=3.0,
    w_lbp_r1=5.0,
    w_lbp_r2=4.0,
    w_hog=4.5,
    w_opp=4.0,
    edge_prop_thresholds=(0.8, 1.2, 1.6, 2.0),
    w_edge_props=6.0,
    w_hs2d=4.0,
    w_ratio_stats=3.0,
    pixel_log1p_scale=8.0,
):
    """
    Change (score-improving, minimal): compute *global* color/stat/hist features from the native-resolution
    RGB image (before resizing), while keeping the existing 96x96 pixel/HOG/LBP blocks unchanged.
    This preserves the same core approach (handcrafted features + linear model), but makes global cues
    less sensitive to downsampling artifacts, which typically improves mean ROC AUC.
    """
    n = len(image_ids)
    H, W = size

    d_rgb = 3 * H * W
    d_gray = H * W
    d_edge = H * W
    d_stats = 12 + 3
    d_hist = (3 + 1) * hist_bins
    d_idx_stats = 8
    d_hsv = 3 * hsv_bins
    d_lbp = 10

    cell_size = 12
    hog_bins = 18
    hog_cells_y = H // cell_size
    hog_cells_x = W // cell_size
    d_hog = hog_cells_y * hog_cells_x * hog_bins

    d_opp = opp_bins
    d_edge_props = len(edge_prop_thresholds)
    d_hs2d = hs2d_bins_h * hs2d_bins_s

    d = (
        d_rgb
        + d_gray
        + d_edge
        + d_stats
        + d_hist
        + d_idx_stats
        + d_hsv
        + d_lbp
        + d_lbp
        + d_hog
        + d_opp
        + d_edge_props
        + d_hs2d
    )
    X = np.empty((n, d), dtype=np.float32)

    s = np.float32(pixel_log1p_scale)
    log_denom = np.log1p(s).astype(np.float32)

    for i, img_id in enumerate(image_ids):
        p = _resolve_image_path(img_id)
        with Image.open(p) as im:
            im_rgb_native = im.convert("RGB")
            arr_rgb_native = np.asarray(im_rgb_native, dtype=np.float32) / 255.0

            im_rgb = im_rgb_native.resize((W, H), Image.BILINEAR)
            arr_rgb = np.asarray(im_rgb, dtype=np.float32) / 255.0

        rgb_comp = np.log1p(arr_rgb * s) / log_denom
        rgb_flat = rgb_comp.reshape(-1).astype(np.float32) * np.float32(w_rgb)

        gray = (
            0.299 * arr_rgb[..., 0] + 0.587 * arr_rgb[..., 1] + 0.114 * arr_rgb[..., 2]
        ).astype(np.float32)
        gray_blur = _gaussian_blur_3x3(gray)

        gray_comp = np.log1p(gray_blur * s) / log_denom
        gray_flat = gray_comp.reshape(-1).astype(np.float32) * np.float32(w_gray)

        gx = np.zeros_like(gray_blur, dtype=np.float32)
        gy = np.zeros_like(gray_blur, dtype=np.float32)
        gx[:, 1:] = gray_blur[:, 1:] - gray_blur[:, :-1]
        gy[1:, :] = gray_blur[1:, :] - gray_blur[:-1, :]
        edge = np.sqrt(gx * gx + gy * gy).astype(np.float32)

        edge_comp = np.log1p(edge * s) / log_denom
        edge_flat = edge_comp.reshape(-1).astype(np.float32) * np.float32(w_edge)

        e_mean = float(edge.mean())
        e_std = float(edge.std()) + 1e-6
        edge_z = (edge - e_mean) / e_std
        edge_props = np.asarray(
            [float((edge_z > t).mean()) for t in edge_prop_thresholds],
            dtype=np.float32,
        ) * np.float32(w_edge_props)

        Rn, Gn, Bn = (
            arr_rgb_native[..., 0].astype(np.float32),
            arr_rgb_native[..., 1].astype(np.float32),
            arr_rgb_native[..., 2].astype(np.float32),
        )

        stats = []
        for ch in (Rn, Gn, Bn):
            stats.extend(
                [float(ch.mean()), float(ch.std()), float(ch.min()), float(ch.max())]
            )

        sRGBn = (Rn + Gn + Bn + 1e-6).astype(np.float32)
        stats.extend(
            [
                float((Rn / sRGBn).mean()),
                float((Gn / sRGBn).mean()),
                float((Bn / sRGBn).mean()),
            ]
        )
        stats = np.asarray(stats, dtype=np.float32)
        stats[:12] *= np.float32(w_stats)
        stats[12:] *= np.float32(w_ratio_stats)

        gray_n = (0.299 * Rn + 0.587 * Gn + 0.114 * Bn).astype(np.float32)
        gray_n_blur = _gaussian_blur_3x3(gray_n)

        h_r = _hist_channel(Rn, bins=hist_bins)
        h_g = _hist_channel(Gn, bins=hist_bins)
        h_b = _hist_channel(Bn, bins=hist_bins)
        h_gray = _hist_channel(gray_n_blur, bins=hist_bins)
        hist = np.concatenate([h_r, h_g, h_b, h_gray], axis=0) * np.float32(w_hist)

        exg = (2.0 * Gn - Rn - Bn).astype(np.float32)
        exr = (1.4 * Rn - Gn).astype(np.float32)
        idx_stats = np.asarray(
            [
                exg.mean(),
                exg.std(),
                exg.min(),
                exg.max(),
                exr.mean(),
                exr.std(),
                exr.min(),
                exr.max(),
            ],
            dtype=np.float32,
        ) * np.float32(w_idx)

        denom = (Gn + Rn + 1e-6).astype(np.float32)
        opp = ((Gn - Rn) / denom).astype(np.float32)
        opp_hist = _hist_range(opp, bins=opp_bins, vmin=-1.0, vmax=1.0).astype(
            np.float32
        ) * np.float32(w_opp)

        Hc, Sc, Vc = _rgb_to_hsv01(arr_rgb_native)
        h_H = _hist_channel(Hc, bins=hsv_bins)
        h_S = _hist_channel(Sc, bins=hsv_bins)
        h_V = _hist_channel(Vc, bins=hsv_bins)
        hsv_hist = np.concatenate([h_H, h_S, h_V], axis=0).astype(
            np.float32
        ) * np.float32(w_hsv)

        hs2d = _hist2d_hs(Hc, Sc, bins_h=hs2d_bins_h, bins_s=hs2d_bins_s)
        hs2d = np.sqrt(np.maximum(hs2d, 0.0)).astype(np.float32)
        hs2d = hs2d * np.float32(w_hs2d)

        lbp_r1 = _lbp_uniform_riu2(gray_blur, radius=1).astype(np.float32) * np.float32(
            w_lbp_r1
        )
        lbp_r2 = _lbp_uniform_riu2(gray_blur, radius=2).astype(np.float32) * np.float32(
            w_lbp_r2
        )

        hog = _hog_orientation_histograms(
            gray_blur, cell_size=cell_size, bins=hog_bins, clip=0.2
        ).astype(np.float32) * np.float32(w_hog)

        X[i] = np.concatenate(
            [
                rgb_flat,
                gray_flat,
                edge_flat,
                stats,
                hist,
                idx_stats,
                hsv_hist,
                lbp_r1,
                lbp_r2,
                hog,
                opp_hist,
                edge_props,
                hs2d,
            ],
            axis=0,
        )

    return X


X_train = extract_features(train_df["image_id"].values)
y_train = train_df[target_cols].values.astype(int)
X_test = extract_features(test_df["image_id"].values)

print("X_train:", X_train.shape, "y_train:", y_train.shape, "X_test:", X_test.shape)



## === cell 3
base_clf = LogisticRegression(
    solver="lbfgs",
    C=1.5,
    max_iter=8000,
    random_state=42,
    class_weight="balanced",
    multi_class="ovr",
)

model = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", MultiOutputClassifier(base_clf, n_jobs=1)),
    ]
)

model.fit(X_train, y_train)

probas = model.predict_proba(X_test)
pred = np.column_stack([p[:, 1] for p in probas]).astype(np.float32)

eps = 1e-6
pred = np.clip(pred, eps, 1.0 - eps)

print("pred:", pred.shape, "min/max:", float(pred.min()), float(pred.max()))



## === cell 4
sub = test_df[["image_id"]].merge(sample_sub, on="image_id", how="left")
sub[target_cols] = pred
sub = sub[["image_id"] + target_cols]

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print(sub.head())
