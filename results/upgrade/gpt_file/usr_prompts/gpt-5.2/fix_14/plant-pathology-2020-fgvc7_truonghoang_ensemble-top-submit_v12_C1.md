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

0.9697842817939653

# 6. Current score

0.6416

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.53266) has done: 'I remove the dependency on missing external “plantpathology” submission files (the current FileNotFoundError root cause) and instead generate predictions directly from the provided train/test data so the notebook runs end-to-end. Because only numpy/pandas/sklearn-related packages are available, I use a lightweight multi-label one-vs-rest logistic regression on simple image-derived features, keeping the approach minimal and fully CPU-safe. I also ensure the submission columns and row order exactly match `sample_submission.csv`/`test.csv`, and always write a `submission.csv` with the required header. These changes primarily fix runtime errors and produce a valid submission; they should also achieve a reasonable ROC AUC baseline compared to all-zeros.'
- What this solution (achieved 0.62141) has done: 'The timeout is dominated by repeatedly decoding images and computing features sequentially (each image is opened and converted multiple times). I keep the exact same feature definitions and model pipeline, but make feature extraction faster by (1) decoding each image only once and deriving RGB/HSV/gray from the same NumPy array, (2) caching features so train/test overlap (or reruns) don’t recompute, and (3) parallelizing feature extraction across CPU cores with a deterministic ordering. These changes are provably equivalent in outputs (same computations, just fewer redundant conversions and less Python overhead) and don’t alter the model, training, or evaluation semantics.'
- What this solution (achieved 0.62054) has done: 'Your current pipeline is a simple OVR logistic regression on handcrafted global color/grid thumbnail features, which tends to underfit this image task (hence the large gap to the target). To move the score upward with minimal, non-architectural changes, I keep the exact feature definitions and OVR LR core, but (1) add a second LR “head” trained on the same features with different regularization/solver behavior and (2) blend the two probability outputs; this often gives a noticeable AUC bump without changing the modeling family. I also switch the base LR to use `class_weight="balanced"` (still the same loss/solver) to better handle label imbalance and improve per-column ROC AUC. Submission formatting/ordering is kept identical and still writes `submission.csv`.'
- What this solution (achieved 0.61537) has done: 'Your current score (0.62054) is far below the target (0.96978), so we should increase performance while preserving the same core approach (handcrafted features + OVR logistic regression). The biggest likely issue is a distribution mismatch: the training labels in this competition are single-label (one-hot), so OneVsRest can be made more appropriate by using a true multinomial logistic regression over the 4 classes and then mapping its class probabilities back to the 4 submission columns. This keeps the same model family (logistic regression), same features, and same training semantics (cross-entropy on labels), but matches the data structure better and typically yields a large AUC gain versus independent OVR heads. We keep your existing blended OVR predictions as a fallback blend component to stay minimal-risk, and we keep submission ordering/columns identical while still writing `submission.csv`.'
- What this solution (achieved 0.62144) has done: 'Your current pipeline underfits, and the biggest low-risk gain (without changing the core “handcrafted features + logistic regression” logic) is to strengthen the same LR family via better regularization selection and a more appropriate loss for one-hot labels. I keep your feature extraction identical and keep the multinomial LR + OVR blend, but add a tiny internal CV on the **multinomial LR C** (only over a small grid) to pick a better regularization level for this dataset, which usually improves mean ROC AUC noticeably. I also make the mapping from multinomial class probabilities to the submission columns explicit using the known column order (healthy/multiple_diseases/rust/scab) to avoid any silent misalignment. Finally, I keep submission formatting identical and still write `submission.csv`.'
- What this solution (achieved 0.64801) has done: 'The timeout is driven by expensive per-image feature extraction (especially full-resolution quantiles and Sobel on large arrays) plus repeated CV training (12 multinomial fits + calibration + two OVR fits). To preserve core logic and accuracy, the main speedups are: compute features on a fixed downscaled image for all “global” statistics (RGB/HSV means/std/quantiles + Sobel), keep the thumbnail/grid parts as-is, and remove redundant array conversions/copies. Additionally, run the multinomial CV loop in parallel across folds (keeping identical models/metrics) and avoid oversubscription by limiting nested parallelism (outer fold-parallel, inner LR single-thread). All changes are deterministic and keep the same feature definitions (just evaluated on a standardized resolution) and the same modeling/blending logic.'
- What this solution (achieved 0.64224) has done: 'Your biggest score limiter is that `CalibratedClassifierCV` on top of multinomial LR is trained on a single-label target, so its `predict_proba` outputs sum to 1 and cannot represent multi-label probabilities; blending that into the final predictions drags AUC down. I keep your exact feature extraction and LR-based approach, but switch the “multinomial head” to be an OVR LR trained on the same one-hot targets (so each column is an independent probability as required by the metric), while keeping your existing two OVR models and the blend structure. I also fix a subtle bug where you fit `base_mult` but predict with the *unfitted* `clf_mult` (the calibrated wrapper), which currently makes the multinomial component invalid. These are minimal changes that preserve the handcrafted-features + logistic regression core and should move your score upward toward the target by making the main blended component metric-aligned and correctly trained.'
- What this solution (achieved 0.6416) has done: 'Your current score (0.64224) is far below the target (0.96978), so we should cautiously improve AUC while keeping the same handcrafted-feature + OVR logistic regression core. The most likely “minimal but impactful” improvement is to add a light probability calibration step (sigmoid/Platt scaling) per label using out-of-fold predictions, which typically improves ROC AUC without changing the underlying model family or features. To stay within runtime, we reuse the already-computed CV folds and calibrate only the main head (the best-C OVR LR), then blend as before. We also make feature extraction deterministic in parallel by collecting results with their indices to avoid any rare ordering issues.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

BASE_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/data",
    "/kaggle/input",
    "/kaggle/working/plant-pathology-2020-fgvc7",
    "/kaggle/working",
]
BASE_DIR = None
for p in BASE_CANDIDATES:
    if os.path.exists(p):
        if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
            os.path.join(p, "test.csv")
        ):
            BASE_DIR = p
            break

if BASE_DIR is None:
    for root in BASE_CANDIDATES:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if "train.csv" in filenames and "test.csv" in filenames:
                BASE_DIR = dirpath
                break
        if BASE_DIR is not None:
            break

if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not locate competition data folder containing train.csv and test.csv."
    )

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")

IMAGES_DIR = os.path.join(BASE_DIR, "images")
if not os.path.isdir(IMAGES_DIR):
    found = None
    for root in BASE_CANDIDATES:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if os.path.basename(dirpath) == "images":
                found = dirpath
                break
        if found is not None:
            break
    if found is None:
        raise FileNotFoundError("Could not locate images directory.")
    IMAGES_DIR = found

print("Using BASE_DIR:", BASE_DIR)
print("Using IMAGES_DIR:", IMAGES_DIR)



## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sub = pd.read_csv(SAMPLE_SUB)

test_df = test_df.merge(sub[["image_id"]], on="image_id", how="right")

target_cols = [c for c in sub.columns if c != "image_id"]
assert set(target_cols) == {
    "healthy",
    "multiple_diseases",
    "rust",
    "scab",
}, f"Unexpected target columns: {target_cols}"

train_df.head(), test_df.head(), sub.head()



## === cell 2
from PIL import Image


def _build_image_path_map(images_dir):
    m = {}
    for fn in os.listdir(images_dir):
        if fn.lower().endswith(".jpg"):
            m[fn[:-4]] = os.path.join(images_dir, fn)
    return m


_IMAGE_PATH_MAP = _build_image_path_map(IMAGES_DIR)


def _resolve_image_path_fast(image_id):
    s = str(image_id)
    p = _IMAGE_PATH_MAP.get(s)
    if p is not None:
        return p
    p = _IMAGE_PATH_MAP.get("Train_" + s)
    if p is not None:
        return p
    p = _IMAGE_PATH_MAP.get("Test_" + s)
    if p is not None:
        return p
    if s.lower().endswith(".jpg"):
        p = _IMAGE_PATH_MAP.get(s[:-4])
        if p is not None:
            return p
    return None


def _rgb_to_hsv_np(rgb_u8):
    rgb = rgb_u8.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    delta = maxc - minc

    s = np.zeros_like(maxc, dtype=np.float32)
    nonzero = maxc > 0.0
    s[nonzero] = delta[nonzero] / maxc[nonzero]

    h = np.zeros_like(maxc, dtype=np.float32)
    nz = delta > 0.0
    mask = nz & (maxc == r)
    h[mask] = (g[mask] - b[mask]) / delta[mask]
    mask = nz & (maxc == g)
    h[mask] = 2.0 + (b[mask] - r[mask]) / delta[mask]
    mask = nz & (maxc == b)
    h[mask] = 4.0 + (r[mask] - g[mask]) / delta[mask]
    h = (h / 6.0) % 1.0  # [0,1)

    hsv = np.stack([h, s, v], axis=-1).astype(np.float32)
    return hsv


def _sobel_edge_stats(gray_f32):
    gy = np.zeros_like(gray_f32, dtype=np.float32)
    gx = np.zeros_like(gray_f32, dtype=np.float32)

    gx[1:-1, 1:-1] = (
        -1.0 * gray_f32[:-2, :-2]
        + 1.0 * gray_f32[:-2, 2:]
        - 2.0 * gray_f32[1:-1, :-2]
        + 2.0 * gray_f32[1:-1, 2:]
        - 1.0 * gray_f32[2:, :-2]
        + 1.0 * gray_f32[2:, 2:]
    )
    gy[1:-1, 1:-1] = (
        -1.0 * gray_f32[:-2, :-2]
        - 2.0 * gray_f32[:-2, 1:-1]
        - 1.0 * gray_f32[:-2, 2:]
        + 1.0 * gray_f32[2:, :-2]
        + 2.0 * gray_f32[2:, 1:-1]
        + 1.0 * gray_f32[2:, 2:]
    )

    mag = np.sqrt(gx * gx + gy * gy, dtype=np.float32)
    return np.array(
        [
            float(mag.mean()),
            float(mag.std()),
            float(np.quantile(mag, 0.90)),
            float(np.quantile(mag, 0.99)),
        ],
        dtype=np.float32,
    )


_FEATURE_CACHE = {}


def image_features(image_path, rgb_small=(16, 16), grid=(4, 4), stats_size=(256, 256)):
    cached = _FEATURE_CACHE.get(image_path)
    if cached is not None:
        return cached

    with Image.open(image_path) as im0:
        im0 = im0.convert("RGB")

        im_stats = im0.resize(stats_size, resample=Image.BILINEAR)
        arr_u8_s = np.asarray(im_stats, dtype=np.uint8)  # hs x ws x 3
        arr_s = arr_u8_s.astype(np.float32) / 255.0
        flat_s = arr_s.reshape(-1, 3)

        means_rgb = flat_s.mean(axis=0)
        stds_rgb = flat_s.std(axis=0)

        hsv_arr = _rgb_to_hsv_np(arr_u8_s)
        hsv_flat = hsv_arr.reshape(-1, 3)
        means_hsv = hsv_flat.mean(axis=0)
        stds_hsv = hsv_flat.std(axis=0)

        gray = (
            0.2989 * arr_s[..., 0] + 0.5870 * arr_s[..., 1] + 0.1140 * arr_s[..., 2]
        ).astype(np.float32)
        gmean = float(gray.mean())
        gstd = float(gray.std())

        rgb_q = (
            np.quantile(flat_s, [0.1, 0.5, 0.9], axis=0).astype(np.float32).reshape(-1)
        )
        hsv_q = (
            np.quantile(hsv_flat, [0.1, 0.5, 0.9], axis=0)
            .astype(np.float32)
            .reshape(-1)
        )
        gray_q = np.array(
            [
                float(np.quantile(gray, 0.1)),
                float(np.quantile(gray, 0.5)),
                float(np.quantile(gray, 0.9)),
            ],
            dtype=np.float32,
        )

        edge_stats = _sobel_edge_stats(gray)

        im_s = im0.resize(rgb_small, resample=Image.BILINEAR)
        rgb_thumb = (
            (np.asarray(im_s, dtype=np.float32) / 255.0).reshape(-1).astype(np.float32)
        )

        gh, gw = grid
        H, W = arr_s.shape[0], arr_s.shape[1]
        Hc = (H // gh) * gh
        Wc = (W // gw) * gw
        if Hc == 0 or Wc == 0:
            grid_feats = np.zeros((gh * gw * 3,), dtype=np.float32)
        else:
            arr_c = arr_s[:Hc, :Wc, :]
            bh = Hc // gh
            bw = Wc // gw
            means_grid = arr_c.reshape(gh, bh, gw, bw, 3).mean(axis=(1, 3))
            grid_feats = means_grid.reshape(-1).astype(np.float32)

    feats = np.concatenate(
        [
            means_rgb.astype(np.float32),
            stds_rgb.astype(np.float32),
            means_hsv.astype(np.float32),
            stds_hsv.astype(np.float32),
            np.array([gmean, gstd], dtype=np.float32),
            rgb_q,
            hsv_q,
            gray_q,
            edge_stats,
            rgb_thumb,
            grid_feats,
        ],
        axis=0,
    ).astype(np.float32, copy=False)

    _FEATURE_CACHE[image_path] = feats
    return feats


def build_feature_matrix(df, images_dir):
    ids = df["image_id"].astype(str).values
    paths = [_resolve_image_path_fast(i) for i in ids]

    missing = [ids[i] for i, p in enumerate(paths) if p is None]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} images. Example: {missing[:5]}"
        )

    d0 = image_features(paths[0]).shape[0]
    X = np.empty((len(paths), d0), dtype=np.float32)

    from concurrent.futures import ThreadPoolExecutor

    max_workers = min(16, (os.cpu_count() or 4))

    def _feat_i(tup):
        i, p = tup
        return i, image_features(p)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, feats in ex.map(_feat_i, list(enumerate(paths)), chunksize=32):
            X[i] = feats

    return X




## === cell 3
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

X_train = build_feature_matrix(train_df, IMAGES_DIR)
y_train = train_df[target_cols].values.astype(np.int64)
X_test = build_feature_matrix(test_df, IMAGES_DIR)

y_single = y_train.argmax(axis=1).astype(np.int64)

C_grid = [0.3, 1.0, 3.0]
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)


def _eval_ovr_C(C):
    aucs = []
    for tr_idx, va_idx in skf.split(X_train, y_single):
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "ovr",
                    OneVsRestClassifier(
                        LogisticRegression(
                            max_iter=4000,
                            solver="lbfgs",
                            penalty="l2",
                            n_jobs=1,
                            C=C,
                        )
                    ),
                ),
            ]
        )
        clf.fit(X_train[tr_idx], y_train[tr_idx])
        proba_va = clf.predict_proba(X_train[va_idx])
        for j in range(4):
            aucs.append(roc_auc_score(y_train[va_idx, j], proba_va[:, j]))
    return float(np.mean(aucs))


best_C = None
best_cv = -1.0
for C in C_grid:
    cv_score = _eval_ovr_C(C)
    print(f"CV mean AUC for OVR LR (C={C}): {cv_score:.6f}")
    if cv_score > best_cv:
        best_cv = cv_score
        best_C = C

print("Selected best C for OVR LR:", best_C, "with CV mean AUC:", best_cv)

clf_head = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "ovr",
            OneVsRestClassifier(
                LogisticRegression(
                    max_iter=4000,
                    solver="lbfgs",
                    penalty="l2",
                    n_jobs=-1,
                    C=best_C,
                )
            ),
        ),
    ]
)

clf_balanced = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "ovr",
            OneVsRestClassifier(
                LogisticRegression(
                    max_iter=4000,
                    solver="saga",
                    penalty="l2",
                    n_jobs=-1,
                    class_weight="balanced",
                )
            ),
        ),
    ]
)

clf_alt = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "ovr",
            OneVsRestClassifier(
                LogisticRegression(
                    max_iter=4000,
                    solver="lbfgs",
                    penalty="l2",
                    n_jobs=-1,
                    C=0.7,
                    class_weight="balanced",
                )
            ),
        ),
    ]
)


def _sigmoid_calibrate_binary(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs", max_iter=2000)
    lr.fit(x, y.astype(int))
    a = float(lr.coef_.ravel()[0])
    b = float(lr.intercept_.ravel()[0])
    return a, b


def _apply_sigmoid_cal(p, a, b):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p))
    z = a * x + b
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


oof_head = np.zeros((X_train.shape[0], 4), dtype=np.float32)
for tr_idx, va_idx in skf.split(X_train, y_single):
    fold_clf = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "ovr",
                OneVsRestClassifier(
                    LogisticRegression(
                        max_iter=4000,
                        solver="lbfgs",
                        penalty="l2",
                        n_jobs=1,
                        C=best_C,
                    )
                ),
            ),
        ]
    )
    fold_clf.fit(X_train[tr_idx], y_train[tr_idx])
    oof_head[va_idx] = fold_clf.predict_proba(X_train[va_idx]).astype(np.float32)

cal_params = []
for j in range(4):
    a, b = _sigmoid_calibrate_binary(oof_head[:, j], y_train[:, j])
    cal_params.append((a, b))
print("Calibration params (a,b) per class:", cal_params)

clf_head.fit(X_train, y_train)
proba_head = clf_head.predict_proba(X_test).astype(np.float32)

proba_head_cal = np.empty_like(proba_head, dtype=np.float32)
for j, (a, b) in enumerate(cal_params):
    proba_head_cal[:, j] = _apply_sigmoid_cal(proba_head[:, j], a, b).astype(np.float32)

clf_balanced.fit(X_train, y_train)
proba1 = clf_balanced.predict_proba(X_test).astype(np.float32)

clf_alt.fit(X_train, y_train)
proba2 = clf_alt.predict_proba(X_test).astype(np.float32)

proba_ovr_blend = 0.6 * proba1 + 0.4 * proba2

proba = 0.80 * proba_head_cal + 0.20 * proba_ovr_blend
proba = np.clip(proba, 1e-6, 1 - 1e-6)

print("proba:", proba.shape, float(proba.min()), float(proba.max()))



## === cell 4
out = pd.DataFrame(proba, columns=target_cols)
out.insert(0, "image_id", test_df["image_id"].values)

out = sub[["image_id"]].merge(out, on="image_id", how="left")
assert out.shape[0] == sub.shape[0], "Submission row count mismatch."
assert list(out.columns) == list(sub.columns), "Submission column mismatch."

out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", out.shape)
print(out.head())
