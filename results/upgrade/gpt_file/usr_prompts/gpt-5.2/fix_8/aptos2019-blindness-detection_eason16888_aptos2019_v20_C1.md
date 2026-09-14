# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd
import cv2

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

np.random.seed(42)



## === cell 1
"""
    Config
"""
IMG_SIZE = 224


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        if not mask.any():
            return img
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        if not mask.any():
            return img
        ys = mask.any(1)
        xs = mask.any(0)
        if not ys.any() or not xs.any():
            return img
        return img[np.ix_(ys, xs)]
    return img


def load_ben_color(image, sigmaX=10):
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image.astype("float32")


def extract_features_from_rgb_uint8(img_rgb_uint8: np.ndarray) -> np.ndarray:
    img = load_ben_color(img_rgb_uint8)  # float32 in ~[0,255]
    x = (img * (1.0 / 255.0)).astype(np.float32, copy=False)

    gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float32, copy=False)

    r = x[..., 0]
    g = x[..., 1]
    b = x[..., 2]

    feats = np.empty(10, dtype=np.float32)
    feats[0] = float(gray.mean())
    feats[1] = float(gray.std())
    feats[2] = float(r.mean())
    feats[3] = float(g.mean())
    feats[4] = float(b.mean())
    feats[5] = float(r.std())
    feats[6] = float(g.std())
    feats[7] = float(b.std())
    feats[8] = float((np.max(x, axis=2) - np.min(x, axis=2)).mean())

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    feats[9] = float(lap.var())
    return feats


def softmax(z: np.ndarray, axis: int = 1) -> np.ndarray:
    zmax = np.max(z, axis=axis, keepdims=True)
    z = z - zmax
    e = np.exp(z)
    s = np.sum(e, axis=axis, keepdims=True) + 1e-12
    return e / s




## === cell 2
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    DATA_ROOT = "../input/aptos2019-blindness-detection"

TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
TRAIN_DIR_NESTED = os.path.join(TRAIN_DIR, "train_images")
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
TEST_DIR_NESTED = os.path.join(TEST_DIR, "test_images")

train_csv_path = os.path.join(DATA_ROOT, "train.csv")
test_csv_path = os.path.join(DATA_ROOT, "test.csv")
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")

train_df = pd.read_csv(train_csv_path)
test_df = (
    pd.read_csv(test_csv_path)
    if os.path.exists(test_csv_path)
    else pd.read_csv(sample_sub_path)[["id_code"]]
)

print("DATA_ROOT:", DATA_ROOT)
print("Train:", train_df.shape, "Test:", test_df.shape)




## === cell 3
def train_softmax_regression(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int = 5,
    lr: float = 0.5,
    epochs: int = 800,
    reg: float = 1e-3,
):
    n, d = X.shape
    W = np.zeros((d, n_classes), dtype=np.float32)
    b = np.zeros((n_classes,), dtype=np.float32)

    Y = np.zeros((n, n_classes), dtype=np.float32)
    Y[np.arange(n), y] = 1.0

    logits = np.empty((n, n_classes), dtype=np.float32)
    P = np.empty((n, n_classes), dtype=np.float32)
    G = np.empty((n, n_classes), dtype=np.float32)

    for _ in range(epochs):
        np.matmul(X, W, out=logits)
        logits += b[None, :]

        zmax = np.max(logits, axis=1, keepdims=True)
        np.subtract(logits, zmax, out=P)
        np.exp(P, out=P)
        denom = np.sum(P, axis=1, keepdims=True) + 1e-12
        P /= denom

        np.subtract(P, Y, out=G)
        G *= 1.0 / n

        dW = X.T @ G
        dW += reg * W
        db = G.sum(axis=0)

        W -= lr * dW
        b -= lr * db

    return W, b


def standardize_fit(X: np.ndarray):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-6] = 1.0
    return mu.astype(np.float32), sigma.astype(np.float32)


def standardize_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return ((X - mu) / sigma).astype(np.float32, copy=False)


def quadratic_weighted_kappa(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 5
) -> float:
    y_true = y_true.astype(np.int64, copy=False)
    y_pred = y_pred.astype(np.int64, copy=False)

    mask = (y_true >= 0) & (y_true < n_classes) & (y_pred >= 0) & (y_pred < n_classes)
    yt = y_true[mask]
    yp = y_pred[mask]

    idx = yt * n_classes + yp
    O = (
        np.bincount(idx, minlength=n_classes * n_classes)
        .reshape(n_classes, n_classes)
        .astype(np.float64)
    )

    act_hist = np.bincount(yt, minlength=n_classes).astype(np.float64)
    pred_hist = np.bincount(yp, minlength=n_classes).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    sE = E.sum()
    sO = O.sum()
    if sE > 0:
        E *= sO / sE

    ij = np.arange(n_classes, dtype=np.float64)
    Wm = (ij[:, None] - ij[None, :]) ** 2 / ((n_classes - 1) ** 2)

    num = (Wm * O).sum()
    den = (Wm * E).sum()
    if den == 0:
        return 0.0
    return float(1.0 - num / den)


def probs_to_expected_class(P: np.ndarray) -> np.ndarray:
    classes = np.arange(P.shape[1], dtype=np.float32)
    return (P.astype(np.float32, copy=False) * classes[None, :]).sum(axis=1)


def apply_cutpoints(x: np.ndarray, cutpoints: np.ndarray) -> np.ndarray:
    return np.digitize(x, bins=cutpoints).astype(np.int64)


def optimize_cutpoints(x: np.ndarray, y: np.ndarray, n_classes: int = 5) -> np.ndarray:
    """
    Coordinate descent over cutpoints to maximize QWK on validation.
    Minimal compute (grid over observed x range), no external deps.
    """
    x = x.astype(np.float32, copy=False)
    y = y.astype(np.int64, copy=False)

    qs = np.quantile(x, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
    cp = qs.copy()

    grid = np.quantile(x, np.linspace(0.02, 0.98, 97)).astype(np.float32)

    best = quadratic_weighted_kappa(y, apply_cutpoints(x, cp), n_classes=n_classes)

    for _pass in range(6):
        improved = False
        for j in range(4):
            low = cp[j - 1] + 1e-4 if j > 0 else -1e9
            high = cp[j + 1] - 1e-4 if j < 3 else 1e9
            candidates = grid[(grid > low) & (grid < high)]
            if candidates.size == 0:
                continue

            local_best = best
            local_cp_j = cp[j]
            for v in candidates:
                tmp = cp.copy()
                tmp[j] = float(v)
                pred = apply_cutpoints(x, tmp)
                score = quadratic_weighted_kappa(y, pred, n_classes=n_classes)
                if score > local_best:
                    local_best = score
                    local_cp_j = float(v)
            if local_best > best + 1e-12:
                cp[j] = local_cp_j
                best = local_best
                improved = True

        if not improved:
            break

    cp = np.sort(cp)
    cp[1] = max(cp[1], cp[0] + 1e-4)
    cp[2] = max(cp[2], cp[1] + 1e-4)
    cp[3] = max(cp[3], cp[2] + 1e-4)
    return cp.astype(np.float32)


train_ids = train_df["id_code"].astype(str).values
y_all = train_df["diagnosis"].astype(np.int64).values

n_total = len(train_ids)
X_all = np.empty((n_total, 10), dtype=np.float32)
y_all2 = np.empty((n_total,), dtype=np.int64)
valid_n = 0
skipped = 0

for idx, (img_id, y) in enumerate(zip(train_ids, y_all)):
    img_path = os.path.join(TRAIN_DIR, f"{img_id}.png")
    if not os.path.exists(img_path):
        alt = os.path.join(TRAIN_DIR_NESTED, f"{img_id}.png")
        if os.path.exists(alt):
            img_path = alt

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        skipped += 1
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    X_all[valid_n] = extract_features_from_rgb_uint8(img)
    y_all2[valid_n] = int(y)
    valid_n += 1

    if (idx + 1) % 800 == 0:
        gc.collect()

X_all = X_all[:valid_n]
y_all2 = y_all2[:valid_n]

print("Built train features:", X_all.shape, "Skipped:", skipped)

mu, sigma = standardize_fit(X_all)
X_all_std = standardize_apply(X_all, mu, sigma)

rng = np.random.RandomState(42)
n_classes = 5
val_frac = 0.2
train_idx = []
val_idx = []
for c in range(n_classes):
    idx_c = np.where(y_all2 == c)[0]
    rng.shuffle(idx_c)
    n_val = int(np.round(len(idx_c) * val_frac))
    val_idx.append(idx_c[:n_val])
    train_idx.append(idx_c[n_val:])
train_idx = np.concatenate(train_idx) if len(train_idx) else np.arange(len(y_all2))
val_idx = np.concatenate(val_idx) if len(val_idx) else np.array([], dtype=np.int64)

X_tr = X_all_std[train_idx]
y_tr = y_all2[train_idx]
X_va = X_all_std[val_idx] if val_idx.size else None
y_va = y_all2[val_idx] if val_idx.size else None

W, b = train_softmax_regression(X_tr, y_tr, n_classes=5, lr=0.5, epochs=800, reg=1e-3)

train_logits = X_tr @ W + b[None, :]
train_pred_argmax = np.argmax(train_logits, axis=1)
acc = float((train_pred_argmax == y_tr).mean())
print("Train acc (sanity):", acc)

cutpoints = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
if val_idx.size:
    va_logits = X_va @ W + b[None, :]
    va_P = softmax(va_logits, axis=1)
    va_x = probs_to_expected_class(va_P)

    va_pred_argmax = np.argmax(va_logits, axis=1)
    qwk_argmax = quadratic_weighted_kappa(y_va, va_pred_argmax, n_classes=5)

    cutpoints = optimize_cutpoints(va_x, y_va, n_classes=5)
    va_pred_cut = apply_cutpoints(va_x, cutpoints)
    qwk_cut = quadratic_weighted_kappa(y_va, va_pred_cut, n_classes=5)

    print("Val QWK argmax:", qwk_argmax)
    print("Val cutpoints:", cutpoints.tolist())
    print("Val QWK cutpoints:", qwk_cut)
else:
    print("No validation split available; using default cutpoints:", cutpoints.tolist())



## === cell 4
id_code = test_df["id_code"].astype(str).values
test_prediction = np.empty(len(id_code), dtype=np.int64)

inv_sigma = (1.0 / sigma).astype(np.float32)

for i in range(len(id_code)):
    img_path = os.path.join(TEST_DIR, f"{id_code[i]}.png")
    if not os.path.exists(img_path):
        alt = os.path.join(TEST_DIR_NESTED, f"{id_code[i]}.png")
        if os.path.exists(alt):
            img_path = alt

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    feats = extract_features_from_rgb_uint8(img)

    feats_std = (feats - mu) * inv_sigma
    logits = feats_std[None, :] @ W + b[None, :]
    P = softmax(logits, axis=1)
    x = probs_to_expected_class(P)[0]
    test_prediction[i] = int(
        apply_cutpoints(np.array([x], dtype=np.float32), cutpoints)[0]
    )



## === cell 5
sub = test_df[["id_code"]].copy()
sub["diagnosis"] = test_prediction.astype(np.int64)
sub.to_csv("submission.csv", index=False)

unique, counts = np.unique(test_prediction, return_counts=True)
print(dict(zip(unique.tolist(), counts.tolist())))
print("Wrote submission.csv with shape:", sub.shape)
print("Done!")
