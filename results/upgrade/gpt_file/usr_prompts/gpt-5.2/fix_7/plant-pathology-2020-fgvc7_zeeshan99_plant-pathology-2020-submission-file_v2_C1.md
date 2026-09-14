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

3.9

# 3. Installed packages

geopandas==0.14.4
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

0.8960507207161089

# 6. Current score

0.65201

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your notebook currently fails because it tries to read a non-existent `../input/notebook45bc751087/submission.csv`, so `df` is never created and later cells error as a consequence. To make this run end-to-end and produce a valid Kaggle submission, I instead load the provided competition `sample_submission.csv` from the actual dataset path and write it back out as `submission.csv`. I also keep the required `image_id` column and the four target columns intact (dropping `image_id` would make an invalid submission). This won’t hit the target score yet (it’s a baseline), but it produce a valid `.csv` submission file reliably.'
- What this solution (achieved 0.53742) has done: 'Your current 0.5 score comes from submitting the uniform probabilities in `sample_submission.csv`, so to move toward the 0.896 target we need real image-based predictions while keeping changes minimal. I keep the same overall flow (read CSVs → build `df` → write `submission.csv`) but replace the constant probabilities with a simple, fast baseline computed from the provided training labels: a 1-nearest-neighbor classifier on downsampled grayscale image pixels. This uses only standard libraries plus `pandas` and should run within the time limit on this small dataset, producing better-than-random AUC without changing any Kaggle I/O paths. The submission be aligned to `test.csv` `image_id` order and preserve the required column names.'
- What this solution (achieved 0.54657) has done: 'Your current 1-NN on raw grayscale pixels is a very low-capacity baseline, so the smallest reliable way to move AUC toward the 0.896 target (without changing the overall approach) is to make the distance metric less sensitive to lighting and add a tiny amount of neighborhood smoothing. I keep the same “extract downsampled image vectors → nearest-neighbor style prediction → write submission.csv” flow, but (1) standardize each image vector (zero-mean/unit-std) and L2-normalize it, and (2) switch from hard 1-NN to a small k-NN (k=5) with inverse-distance weighting, which is still the same nearest-neighbor method but typically boosts ROC AUC noticeably on this dataset. I also ensure predictions are clipped to valid probabilities and that test rows remain aligned to `test.csv` order with the required column names. These changes are minimal, fast on this dataset size, and should increase your score toward the target band.'
- What this solution (achieved 0.55198) has done: 'To move your AUC closer to the 0.896 target without changing the overall “downsample image vectors → kNN weighted prediction → write submission.csv” approach, I make the image vectors slightly more informative while keeping the same kNN core. Specifically, I switch from grayscale to small RGB (color often matters for rust/scab) and add a simple edge-like channel (Sobel magnitude) concatenated to the vector, then apply the same per-image standardization and L2 normalization as you already do. This keeps the model family identical (still just kNN in the same loop), but usually yields a meaningful bump over grayscale-only. I also set a fixed PIL resize interpolation for stability and keep the submission alignment/columns unchanged.'
- What this solution (achieved 0.65492) has done: 'Your current kNN pipeline is reasonable but it’s likely underperforming because k is small and the inverse-distance weights become extremely peaky, making predictions unstable and hurting mean ROC AUC. To move the score upward toward 0.896 with minimal logic changes, I keep the exact same feature extraction and weighted kNN approach, but (1) increase k moderately (more smoothing usually helps AUC here) and (2) switch weighting from `1/(d^2+eps)` to a softmax over negative distances with a temperature, which is still distance-weighted kNN but far better behaved. I also add a tiny prior-mixing (shrinkage) toward the training label means to improve calibration without changing the model family. The output format/path stays the same and remains aligned to `test.csv` order.'
- What this solution (achieved 0.65201) has done: 'Your current score (0.65492) is far below the target (0.89605), so we should make small, low-risk changes that increase mean ROC AUC while keeping the same kNN-on-image-vectors core. The biggest easy win inside your existing logic is to make the kNN weighting a bit less “global” by lowering the softmax temperature and slightly reducing the prior-mix shrinkage, which typically increases separability (and thus AUC) for this dataset without changing the model family. I also make the Sobel magnitude computation vectorized (same feature, same semantics) to avoid Python loops and keep runtime safely within limits, which reduces the risk of timeouts without changing what’s being modeled. Everything else (paths, feature extraction structure, kNN distance computation, submission columns/order) stays the same.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image

BASE_DIR_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]
base_dir = None
for p in BASE_DIR_CANDIDATES:
    if os.path.exists(p):
        base_dir = p
        break
if base_dir is None:
    raise FileNotFoundError(
        f"Could not find a Kaggle data directory. Tried: {BASE_DIR_CANDIDATES}"
    )

train_csv = os.path.join(base_dir, "train.csv")
test_csv = os.path.join(base_dir, "test.csv")
sample_csv = os.path.join(base_dir, "sample_submission.csv")
images_dir = os.path.join(base_dir, "images")

if not os.path.exists(train_csv):
    train_csv = os.path.join(base_dir, "plant-pathology-2020-fgvc7", "train.csv")
    test_csv = os.path.join(base_dir, "plant-pathology-2020-fgvc7", "test.csv")
    sample_csv = os.path.join(
        base_dir, "plant-pathology-2020-fgvc7", "sample_submission.csv"
    )
    images_dir = os.path.join(base_dir, "plant-pathology-2020-fgvc7", "images")

for pth in [train_csv, test_csv, sample_csv, images_dir]:
    if not os.path.exists(pth):
        raise FileNotFoundError(f"Missing required path: {pth}")

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)
df = pd.read_csv(sample_csv)

required_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"sample_submission.csv missing required columns: {missing}")

df = test_df.merge(df, on="image_id", how="left")
if df.shape[0] != test_df.shape[0]:
    raise ValueError("Submission rows do not match test.csv rows after merge.")

df.head()



## === cell 1
TARGETS = ["healthy", "multiple_diseases", "rust", "scab"]


def _sobel_mag(gray01: np.ndarray) -> np.ndarray:
    """Sobel magnitude on a (H,W) float32 array in [0,1], vectorized."""
    g = np.pad(gray01.astype(np.float32), ((1, 1), (1, 1)), mode="reflect")

    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    gx = (
        g[0:-2, 0:-2] * kx[0, 0]
        + g[0:-2, 1:-1] * kx[0, 1]
        + g[0:-2, 2:] * kx[0, 2]
        + g[1:-1, 0:-2] * kx[1, 0]
        + g[1:-1, 1:-1] * kx[1, 1]
        + g[1:-1, 2:] * kx[1, 2]
        + g[2:, 0:-2] * kx[2, 0]
        + g[2:, 1:-1] * kx[2, 1]
        + g[2:, 2:] * kx[2, 2]
    ).astype(np.float32)

    gy = (
        g[0:-2, 0:-2] * ky[0, 0]
        + g[0:-2, 1:-1] * ky[0, 1]
        + g[0:-2, 2:] * ky[0, 2]
        + g[1:-1, 0:-2] * ky[1, 0]
        + g[1:-1, 1:-1] * ky[1, 1]
        + g[1:-1, 2:] * ky[1, 2]
        + g[2:, 0:-2] * ky[2, 0]
        + g[2:, 1:-1] * ky[2, 1]
        + g[2:, 2:] * ky[2, 2]
    ).astype(np.float32)

    mag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    mmax = float(mag.max())
    if mmax > 1e-6:
        mag = mag / mmax
    else:
        mag = mag * 0.0
    return mag


def load_image_vector(image_id: str, size=(32, 32)) -> np.ndarray:
    img_path = os.path.join(images_dir, f"{image_id}.jpg")
    if not os.path.exists(img_path):
        img_path2 = os.path.join(images_dir, f"{image_id}.JPG")
        if os.path.exists(img_path2):
            img_path = img_path2
        else:
            raise FileNotFoundError(
                f"Image not found for {image_id}: tried {img_path} and {img_path2}"
            )

    im = Image.open(img_path).convert("RGB").resize(size, resample=Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0  # (H,W,3) in [0,1]

    gray = (
        0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
    ).astype(np.float32)
    edge = _sobel_mag(gray)  # (H,W)

    feat = np.concatenate([arr, edge[:, :, None]], axis=2)  # (H,W,4)
    v = feat.reshape(-1).astype(np.float32)

    m = float(v.mean())
    s = float(v.std())
    if s < 1e-6:
        s = 1e-6
    v = (v - m) / s

    n = float(np.sqrt((v * v).sum()))
    if n < 1e-6:
        n = 1e-6
    v = v / n
    return v.astype(np.float32)


X_train = np.vstack([load_image_vector(iid) for iid in train_df["image_id"].values])
Y_train = train_df[TARGETS].astype(np.float32).values
X_test = np.vstack([load_image_vector(iid) for iid in test_df["image_id"].values])


def _softmax_stable(z: np.ndarray, axis: int = 0) -> np.ndarray:
    zmax = np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z - zmax)
    return ez / np.sum(ez, axis=axis, keepdims=True)


def predict_knn_weighted(
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_te: np.ndarray,
    k: int = 25,
    block_size: int = 32,
    tau: float = 0.20,
    prior_mix: float = 0.03,
) -> np.ndarray:
    """
    Minimal score-oriented tweaks (same kNN core):
    - Slightly lower tau => weights focus more on closer neighbors (often improves AUC ranking).
    - Slightly lower prior_mix => less shrinkage toward class means, increasing separability for ROC AUC.
    """
    n_test = X_te.shape[0]
    n_classes = Y_tr.shape[1]
    preds = np.zeros((n_test, n_classes), dtype=np.float32)

    prior = Y_tr.mean(axis=0, keepdims=True).astype(np.float32)  # (1, c)

    tr_norm = (X_tr**2).sum(axis=1, keepdims=True)  # (n_train, 1)

    for start in range(0, n_test, block_size):
        end = min(start + block_size, n_test)
        Xb = X_te[start:end]  # (b, d)
        te_norm = (Xb**2).sum(axis=1, keepdims=True).T  # (1, b)
        dots = X_tr @ Xb.T  # (n_train, b)
        d2 = tr_norm + te_norm - 2.0 * dots  # (n_train, b)
        d2 = np.maximum(d2, 0.0)

        kk = min(k, d2.shape[0])
        idx = np.argpartition(d2, kth=kk - 1, axis=0)[:kk, :]  # (k, b)
        d2_k = np.take_along_axis(d2, idx, axis=0)  # (k, b)

        w = _softmax_stable(-d2_k / max(tau, 1e-6), axis=0).astype(np.float32)  # (k,b)

        Yk = Y_tr[idx]  # (k, b, c)
        pred_block = (w[:, :, None] * Yk).sum(axis=0).astype(np.float32)  # (b, c)

        if prior_mix > 0.0:
            pred_block = (1.0 - prior_mix) * pred_block + prior_mix * prior

        preds[start:end] = pred_block

    preds = np.clip(preds, 0.0, 1.0)
    return preds


test_pred = predict_knn_weighted(
    X_train, Y_train, X_test, k=25, block_size=32, tau=0.20, prior_mix=0.03
)

df_cropped = df[["image_id"]].copy()
for j, col in enumerate(TARGETS):
    df_cropped[col] = test_pred[:, j].astype(np.float32)

df_cropped.head()



## === cell 2
out_path = "./submission.csv"
df_cropped.to_csv(out_path, index=False)

print(f"Wrote submission to: {out_path}")
print(df_cropped.head())
print("Row count:", len(df_cropped), "Columns:", list(df_cropped.columns))
