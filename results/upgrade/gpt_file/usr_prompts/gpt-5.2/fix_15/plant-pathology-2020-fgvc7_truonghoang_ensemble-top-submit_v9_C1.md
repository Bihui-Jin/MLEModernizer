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
BASE_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]

BASE = None
for c in BASE_CANDIDATES:
    if os.path.exists(c):
        if c in ("/kaggle/input", "/kaggle/data"):
            cc = os.path.join(c, "plant-pathology-2020-fgvc7")
            if os.path.exists(cc):
                BASE = cc
                break
        BASE = c
        break

if BASE is None:
    raise FileNotFoundError(
        "Could not locate Kaggle data directory in expected locations."
    )

train_path = os.path.join(BASE, "train.csv")
test_path = os.path.join(BASE, "test.csv")
sample_path = os.path.join(BASE, "sample_submission.csv")
images_dir = os.path.join(BASE, "images")

print("Using BASE:", BASE)
print(
    "Files exist:",
    os.path.exists(train_path),
    os.path.exists(test_path),
    os.path.exists(sample_path),
    os.path.exists(images_dir),
)



## === cell 2
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

target_cols = [c for c in sample_sub.columns if c != "image_id"]
for c in ["image_id"] + target_cols:
    if c != "image_id" and c not in train_df.columns:
        raise ValueError(f"Train CSV missing expected target column: {c}")
if "image_id" not in test_df.columns:
    raise ValueError("Test CSV missing image_id column.")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Targets:", target_cols)
print("Images dir:", images_dir)



## === cell 3
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)


def _safe_image_path(image_id: str) -> str:
    return os.path.join(images_dir, f"{image_id}.jpg")


def _make_bin_lut(bins: int) -> np.ndarray:
    lut = (np.arange(256, dtype=np.int32) * bins) // 256
    lut[lut >= bins] = bins - 1
    return lut.astype(np.int32)


def _hist_from_uint8_channel(
    channel_u8: np.ndarray, bins: int, lut: np.ndarray
) -> np.ndarray:
    idx = lut[channel_u8]
    h = np.bincount(idx.ravel(), minlength=bins).astype(np.float32)
    s = float(h.sum())
    if s > 0.0:
        h /= s
    return h


def _circular_hue_stats(h_u8: np.ndarray) -> np.ndarray:
    """
    Hue is circular; use circular moments instead of a linear histogram for better separability.
    """
    h = h_u8.astype(np.float32)
    ang = (2.0 * np.pi) * (h / 255.0)
    c = np.cos(ang)
    s = np.sin(ang)
    mc = float(c.mean())
    ms = float(s.mean())
    R = float(np.sqrt(mc * mc + ms * ms))
    return np.array([mc, ms, R], dtype=np.float32)


def _block_features_from_arrays(
    rgb_arr: np.ndarray,
    hsv_arr: np.ndarray,
    bins_rgb: int,
    bins_s: int,
    bins_v: int,
    bins_gray: int,
    bins_edge: int,
    rgb_lut: np.ndarray,
    s_lut: np.ndarray,
    v_lut: np.ndarray,
    gray_lut: np.ndarray,
    edge_lut: np.ndarray,
) -> np.ndarray:
    feats = []

    for ch in range(3):
        feats.append(_hist_from_uint8_channel(rgb_arr[..., ch], bins_rgb, rgb_lut))

    arr_f = rgb_arr.astype(np.float32) / 255.0
    flat = arr_f.reshape(-1, 3)
    means = flat.mean(axis=0)
    stds = flat.std(axis=0)
    feats.append(means.astype(np.float32))
    feats.append(stds.astype(np.float32))

    feats.append(_circular_hue_stats(hsv_arr[..., 0]))

    feats.append(_hist_from_uint8_channel(hsv_arr[..., 1], bins_s, s_lut))
    feats.append(_hist_from_uint8_channel(hsv_arr[..., 2], bins_v, v_lut))

    q = np.array([0.25, 0.50, 0.75], dtype=np.float32)
    rgb_q = np.quantile(flat, q, axis=0).T.reshape(-1)  # (3*3,)
    feats.append(rgb_q.astype(np.float32))

    hsv_f = hsv_arr.astype(np.float32) / 255.0
    hs_flat = hsv_f[..., :2].reshape(-1, 2)
    hs_means = hs_flat.mean(axis=0)
    hs_stds = hs_flat.std(axis=0)
    feats.append(hs_means.astype(np.float32))
    feats.append(hs_stds.astype(np.float32))

    gray_u8 = (
        (0.299 * rgb_arr[..., 0].astype(np.float32))
        + (0.587 * rgb_arr[..., 1].astype(np.float32))
        + (0.114 * rgb_arr[..., 2].astype(np.float32))
    ).astype(np.uint8)
    feats.append(_hist_from_uint8_channel(gray_u8, bins_gray, gray_lut))
    gray_f = gray_u8.astype(np.float32) / 255.0
    feats.append(np.array([gray_f.mean(), gray_f.std()], dtype=np.float32))

    gx = np.diff(
        gray_u8.astype(np.int16), axis=1, append=gray_u8[:, -1:].astype(np.int16)
    )
    gy = np.diff(
        gray_u8.astype(np.int16), axis=0, append=gray_u8[-1:, :].astype(np.int16)
    )
    mag = np.abs(gx) + np.abs(gy)
    mag = np.clip(mag, 0, 255).astype(np.uint8)
    feats.append(_hist_from_uint8_channel(mag, bins_edge, edge_lut))
    mag_f = mag.astype(np.float32) / 255.0
    feats.append(np.array([mag_f.mean(), mag_f.std()], dtype=np.float32))

    H, W = gray_u8.shape
    y0, y1 = int(H * 0.25), int(H * 0.75)
    x0, x1 = int(W * 0.25), int(W * 0.75)
    center = gray_u8[y0:y1, x0:x1].astype(np.float32) / 255.0
    border_mask = np.ones((H, W), dtype=bool)
    border_mask[y0:y1, x0:x1] = False
    border = (gray_u8.astype(np.float32) / 255.0)[border_mask]
    if center.size == 0 or border.size == 0:
        feats.append(np.zeros((4,), dtype=np.float32))
    else:
        feats.append(
            np.array(
                [
                    center.mean(),
                    center.std(),
                    border.mean(),
                    border.std(),
                ],
                dtype=np.float32,
            )
        )

    return np.concatenate(feats, axis=0).astype(np.float32)


def extract_features(
    image_path: str,
    bins_rgb: int = 16,
    bins_s: int = 8,
    bins_v: int = 8,
    bins_gray: int = 16,
    bins_edge: int = 16,
    _rgb_lut=None,
    _s_lut=None,
    _v_lut=None,
    _gray_lut=None,
    _edge_lut=None,
    resize_hw=(256, 256),
    grid=(2, 2),
    tile_weight: float = 1.15,
    grid2=(3, 3),
    tile_weight2: float = 0.85,
) -> np.ndarray:
    gh, gw = grid
    gh2, gw2 = grid2

    extra_len_h_circ = 3
    extra_len_rgbq_hsv = (3 * 3) + (2 * 2)
    extra_len_gray = 2
    extra_len_edge = 2
    extra_len_center_border = 4
    block_len = (
        (3 * bins_rgb)
        + 6
        + extra_len_h_circ
        + (bins_s + bins_v)
        + extra_len_rgbq_hsv
        + bins_gray
        + extra_len_gray
        + bins_edge
        + extra_len_edge
        + extra_len_center_border
    )
    feat_len = (1 + gh * gw + gh2 * gw2) * block_len

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            if resize_hw is not None:
                img = img.resize((resize_hw[1], resize_hw[0]), resample=Image.BILINEAR)
            rgb = np.asarray(img, dtype=np.uint8)
            hsv = np.asarray(
                Image.fromarray(rgb, mode="RGB").convert("HSV"), dtype=np.uint8
            )
    except Exception:
        return np.zeros((feat_len,), dtype=np.float32)

    H, W = rgb.shape[0], rgb.shape[1]
    feats = []

    feats.append(
        _block_features_from_arrays(
            rgb,
            hsv,
            bins_rgb,
            bins_s,
            bins_v,
            bins_gray,
            bins_edge,
            _rgb_lut,
            _s_lut,
            _v_lut,
            _gray_lut,
            _edge_lut,
        )
    )

    ys = np.linspace(0, H, gh + 1, dtype=np.int32)
    xs = np.linspace(0, W, gw + 1, dtype=np.int32)
    for i in range(gh):
        y0, y1 = ys[i], ys[i + 1]
        for j in range(gw):
            x0, x1 = xs[j], xs[j + 1]
            rgb_t = rgb[y0:y1, x0:x1]
            hsv_t = hsv[y0:y1, x0:x1]
            feats.append(
                tile_weight
                * _block_features_from_arrays(
                    rgb_t,
                    hsv_t,
                    bins_rgb,
                    bins_s,
                    bins_v,
                    bins_gray,
                    bins_edge,
                    _rgb_lut,
                    _s_lut,
                    _v_lut,
                    _gray_lut,
                    _edge_lut,
                )
            )

    ys2 = np.linspace(0, H, gh2 + 1, dtype=np.int32)
    xs2 = np.linspace(0, W, gw2 + 1, dtype=np.int32)
    for i in range(gh2):
        y0, y1 = ys2[i], ys2[i + 1]
        for j in range(gw2):
            x0, x1 = xs2[j], xs2[j + 1]
            rgb_t = rgb[y0:y1, x0:x1]
            hsv_t = hsv[y0:y1, x0:x1]
            feats.append(
                tile_weight2
                * _block_features_from_arrays(
                    rgb_t,
                    hsv_t,
                    bins_rgb,
                    bins_s,
                    bins_v,
                    bins_gray,
                    bins_edge,
                    _rgb_lut,
                    _s_lut,
                    _v_lut,
                    _gray_lut,
                    _edge_lut,
                )
            )

    out = np.concatenate(feats, axis=0).astype(np.float32)
    if out.shape[0] != feat_len:
        return np.zeros((feat_len,), dtype=np.float32)
    return out


def build_feature_matrix(
    image_ids: pd.Series,
    bins_rgb: int = 16,
    bins_s: int = 8,
    bins_v: int = 8,
    bins_gray: int = 16,
    bins_edge: int = 16,
    resize_hw=(256, 256),
    grid=(2, 2),
    num_workers=None,
    tile_weight: float = 1.15,
    grid2=(3, 3),
    tile_weight2: float = 0.85,
) -> np.ndarray:
    rgb_lut = _make_bin_lut(bins_rgb)
    s_lut = _make_bin_lut(bins_s)
    v_lut = _make_bin_lut(bins_v)
    gray_lut = _make_bin_lut(bins_gray)
    edge_lut = _make_bin_lut(bins_edge)

    extra_len_h_circ = 3
    extra_len_rgbq_hsv = (3 * 3) + (2 * 2)
    extra_len_gray = 2
    extra_len_edge = 2
    extra_len_center_border = 4
    block_len = (
        (3 * bins_rgb)
        + 6
        + extra_len_h_circ
        + (bins_s + bins_v)
        + extra_len_rgbq_hsv
        + bins_gray
        + extra_len_gray
        + bins_edge
        + extra_len_edge
        + extra_len_center_border
    )
    gh, gw = grid
    gh2, gw2 = grid2
    feat_len = (1 + gh * gw + gh2 * gw2) * block_len

    ids = image_ids.tolist()
    paths = [_safe_image_path(iid) for iid in ids]

    X = np.zeros((len(ids), feat_len), dtype=np.float32)

    missing_count = sum(1 for p in paths if not os.path.exists(p))
    if missing_count:
        print(
            f"Warning: {missing_count} images not found under {images_dir}. Using zero-features for those."
        )

    if num_workers is None:
        cpu = os.cpu_count() or 4
        num_workers = min(8, cpu)

    def _one(p: str) -> np.ndarray:
        return extract_features(
            p,
            bins_rgb=bins_rgb,
            bins_s=bins_s,
            bins_v=bins_v,
            bins_gray=bins_gray,
            bins_edge=bins_edge,
            _rgb_lut=rgb_lut,
            _s_lut=s_lut,
            _v_lut=v_lut,
            _gray_lut=gray_lut,
            _edge_lut=edge_lut,
            resize_hw=resize_hw,
            grid=grid,
            tile_weight=tile_weight,
            grid2=grid2,
            tile_weight2=tile_weight2,
        )

    if len(paths) == 0:
        return X

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        for i, feat in enumerate(ex.map(_one, paths, chunksize=32)):
            X[i] = feat

    denom = np.abs(X).sum(axis=1, keepdims=True).astype(np.float32)
    denom = np.maximum(denom, 1e-6)
    X = X / denom

    return X




## === cell 4
BINS_RGB = 24
BINS_S = 12
BINS_V = 12
BINS_GRAY = 24
BINS_EDGE = 24

RESIZE_HW = (256, 256)

GRID = (2, 2)
GRID2 = (3, 3)

TILE_WEIGHT = 1.15
TILE_WEIGHT2 = 0.85

X_train = build_feature_matrix(
    train_df["image_id"],
    bins_rgb=BINS_RGB,
    bins_s=BINS_S,
    bins_v=BINS_V,
    bins_gray=BINS_GRAY,
    bins_edge=BINS_EDGE,
    resize_hw=RESIZE_HW,
    grid=GRID,
    tile_weight=TILE_WEIGHT,
    grid2=GRID2,
    tile_weight2=TILE_WEIGHT2,
)
X_test = build_feature_matrix(
    test_df["image_id"],
    bins_rgb=BINS_RGB,
    bins_s=BINS_S,
    bins_v=BINS_V,
    bins_gray=BINS_GRAY,
    bins_edge=BINS_EDGE,
    resize_hw=RESIZE_HW,
    grid=GRID,
    tile_weight=TILE_WEIGHT,
    grid2=GRID2,
    tile_weight2=TILE_WEIGHT2,
)

X_train = X_train.astype(np.float64, copy=False)
X_test = X_test.astype(np.float64, copy=False)

models = {}
test_pred = np.zeros((len(test_df), len(target_cols)), dtype=np.float64)

for j, col in enumerate(target_cols):
    y = train_df[col].astype(int).values

    if y.min() == y.max():
        prior = float(y.mean())
        test_pred[:, j] = prior
        models[col] = None
        continue

    lr_kwargs = dict(
        solver="saga",
        penalty="l2",
        max_iter=20000,
        C=2.0,
        random_state=0,
        n_jobs=1,
    )

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("lr", LogisticRegression(**lr_kwargs)),
        ]
    )
    clf.fit(X_train, y)
    proba = clf.predict_proba(X_test)[:, 1].astype(np.float64)
    test_pred[:, j] = proba
    models[col] = clf

test_pred = np.clip(test_pred, 0.0, 1.0)



## === cell 5
sub = sample_sub.copy()

pred_df = pd.DataFrame(test_pred, columns=target_cols)
pred_df.insert(0, "image_id", test_df["image_id"].values)

sub = sub[["image_id"] + target_cols].merge(
    pred_df, on="image_id", how="left", suffixes=("", "_pred")
)

priors = train_df[target_cols].mean(axis=0).astype(float)
for c in target_cols:
    pred_col = c + "_pred"
    if pred_col in sub.columns:
        sub[c] = sub[pred_col]
        sub.drop(columns=[pred_col], inplace=True)
    sub[c] = sub[c].astype(float)
    sub[c] = sub[c].fillna(float(priors[c])).clip(0.0, 1.0)

sub = sub[["image_id"] + target_cols]

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub.head())
print("Submission shape:", sub.shape)

assert out_path.endswith(".csv")
assert list(sub.columns) == list(sample_sub.columns)
assert len(sub) == len(sample_sub)
assert sub[target_cols].isna().sum().sum() == 0
assert ((sub[target_cols] >= 0.0) & (sub[target_cols] <= 1.0)).all().all()
