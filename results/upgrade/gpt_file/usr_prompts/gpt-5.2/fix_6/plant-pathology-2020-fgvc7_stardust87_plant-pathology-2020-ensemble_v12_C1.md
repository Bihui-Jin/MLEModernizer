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

0.9711824920467708

# 6. Current score

0.55791

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your code fails because `SUBMISSIONS_PATH` points to a non-existent folder, so `submissions_all` is empty and indexing `[0,1,2]` crashes. I fix this by switching the input path to the competition’s provided `sample_submission.csv` and generating a valid baseline submission directly from it (this is score-improving vs “no submission” while keeping changes minimal). I also add small safety checks (existence/shape) and ensure the output is written as `submission.csv` with the exact required columns and row order aligned to `test.csv`.'
- What this solution (achieved 0.66867) has done: 'Your current pipeline just copies the sample submission’s constant probabilities, which typically scores around 0.5 AUC. To move toward the 0.971 target without changing the overall “no-training” approach, I generate image-based probabilities using a simple RGB feature extractor plus a multi-output logistic regression (same basic structure: read CSVs → compute features → fit → predict → write submission). I also add a stratified train/validation split and print the mean ROC AUC locally so you can verify the direction before submitting, while keeping the submission formatting logic and paths intact. This should substantially improve score toward the target while remaining lightweight and finishing within the time limit.'
- What this solution (achieved 0.59312) has done: 'Your current score (0.66867) is far below the target (0.97118), so we should improve model signal while keeping the same overall approach (hand-crafted image features → OneVsRest LogisticRegression → predict_proba → submission). The biggest low-risk gain is to enrich the feature extractor beyond global RGB mean/std: add simple color ratios, HSV stats, and a tiny downsampled grayscale thumbnail to capture texture/spotting patterns typical for scab/rust, while still staying in a lightweight “no deep learning” pipeline. I also set `class_weight="balanced"` in the same LogisticRegression to better handle label imbalance (especially `multiple_diseases`) without changing the training loop or metric semantics. Finally, I keep the submission-writing logic but ensure the output rows are aligned to `test.csv` order (no merge-induced reorder risk).'
- What this solution (achieved 0.54046) has done: 'Your current approach (hand-crafted image features + OneVsRest logistic regression) is fine but likely underperforming because the tiny grayscale thumbnail is too low-res for disease texture cues and because the LR regularization may be a bit too strong for these features. I keep the same pipeline and training loop, but slightly enrich the *existing* feature extractor by increasing the grayscale thumbnail resolution and adding a couple of simple distribution features (quantiles) to better capture spotting patterns. I also very mildly tune the LogisticRegression strength (C) and ensure probabilities are returned in the correct column order and aligned to `test.csv` without any merge reorder risk. These are minimal, low-risk changes intended to move AUC upward toward the 0.971 target without changing the core modeling approach.'
- What this solution (achieved 0.55791) has done: 'Your current score (0.54046) is far below the target (0.97118), so we should improve signal while keeping the same overall pipeline (hand-crafted image features → StandardScaler → OneVsRest LogisticRegression → predict_proba → submission). The biggest low-risk issue is that resizing with PIL defaults to a lower-quality resampler, which can blur disease textures; switching to a high-quality resampler preserves spots/lesions and typically improves AUC without changing the model. To better capture rust/scab texture at essentially the same computational cost, we also add a small set of gradient-energy features computed on the existing grayscale image (no new model/training changes). Finally, we keep the exact submission formatting but ensure feature length is inferred robustly (so it stays consistent if feature dims change) and keep row order aligned to `test.csv`.'

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

print("Exists TRAIN_CSV:", os.path.exists(TRAIN_CSV))
print("Exists TEST_CSV:", os.path.exists(TEST_CSV))
print("Exists SAMPLE_SUB_CSV:", os.path.exists(SAMPLE_SUB_CSV))
print("Exists IMAGES_DIR:", os.path.exists(IMAGES_DIR))




## === cell 2
def ensemble(submissions_all, sub_idx, weights=None):
    if weights is None:
        weights = [1.0] * len(sub_idx)

    if len(sub_idx) != len(weights):
        raise ValueError(
            f"sub_idx and weights must have the same length, got {len(sub_idx)} and {len(weights)}"
        )

    if len(submissions_all) == 0:
        raise ValueError(
            "submissions_all is empty. Provide valid submission file paths to ensemble."
        )

    submission_with_weight = []
    for i in range(len(sub_idx)):
        if sub_idx[i] >= len(submissions_all):
            raise IndexError(
                f"sub_idx[{i}]={sub_idx[i]} out of range for submissions_all of length {len(submissions_all)}"
            )
        print(
            f"I'm taking submission {submissions_all[sub_idx[i]]} with weight {weights[i]}"
        )
        submission = pd.read_csv(submissions_all[sub_idx[i]])
        submission = submission.loc[
            :, ["healthy", "multiple_diseases", "rust", "scab"]
        ].values
        submission_with_weight.append(submission * weights[i])

    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 3
def make_submission_file_from_df(pred_df, out_path="submission.csv"):
    sample = pd.read_csv(SAMPLE_SUB_CSV)
    required_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    if list(sample.columns) != required_cols:
        sample = sample[required_cols]

    if "image_id" not in pred_df.columns:
        raise ValueError("pred_df must contain an 'image_id' column")

    pred_df = pred_df.copy()
    pred_df = pred_df[required_cols]

    merged = sample[["image_id"]].merge(pred_df, on="image_id", how="left", sort=False)

    for c in required_cols[1:]:
        if merged[c].isna().any():
            merged[c] = merged[c].fillna(0.25)

    for c in required_cols[1:]:
        merged[c] = merged[c].clip(0.0, 1.0)

    merged.to_csv(out_path, index=False)
    print(
        f"Wrote {out_path} with shape {merged.shape} and columns {list(merged.columns)}"
    )




## === cell 4
import numpy as np

try:
    from PIL import Image
except Exception as e:
    raise ImportError(
        "PIL is required to read images. Please ensure Pillow is available in the environment."
    ) from e

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

_RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


def image_path(image_id: str) -> str:
    return os.path.join(IMAGES_DIR, f"{image_id}.jpg")


def _rgb_to_hsv_np(rgb01: np.ndarray) -> np.ndarray:
    """
    rgb01: float32 array (H,W,3) in [0,1]
    returns hsv float32 array (H,W,3) where H in [0,1], S,V in [0,1]
    """
    r = rgb01[..., 0]
    g = rgb01[..., 1]
    b = rgb01[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    delt = maxc - minc
    s = np.where(maxc > 1e-8, delt / (maxc + 1e-8), 0.0)

    h = np.zeros_like(maxc, dtype=np.float32)
    mask = delt > 1e-8

    rc = np.where(mask, (maxc - r) / (delt + 1e-8), 0.0)
    gc = np.where(mask, (maxc - g) / (delt + 1e-8), 0.0)
    bc = np.where(mask, (maxc - b) / (delt + 1e-8), 0.0)

    h = np.where((maxc == r) & mask, (bc - gc), h)
    h = np.where((maxc == g) & mask, 2.0 + (rc - bc), h)
    h = np.where((maxc == b) & mask, 4.0 + (gc - rc), h)
    h = (h / 6.0) % 1.0
    hsv = np.stack([h, s, v], axis=-1).astype(np.float32)
    return hsv


def extract_features(
    img_path: str, size_rgb=(96, 96), thumb_gray=(32, 32)
) -> np.ndarray:
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        if size_rgb is not None:
            im = im.resize(size_rgb, resample=_RESAMPLE)
        arr = np.asarray(im, dtype=np.float32) / 255.0  # (H,W,3)

    means = arr.mean(axis=(0, 1))  # 3
    stds = arr.std(axis=(0, 1))  # 3

    hsv = _rgb_to_hsv_np(arr)
    hsv_means = hsv.mean(axis=(0, 1))  # 3
    hsv_stds = hsv.std(axis=(0, 1))  # 3

    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    eps = 1e-6
    rg = (r / (g + eps)).mean()
    rb = (r / (b + eps)).mean()
    gb = (g / (b + eps)).mean()
    r_minus_g = (r - g).mean()
    r_minus_b = (r - b).mean()
    g_minus_b = (g - b).mean()

    overall_mean = np.array([arr.mean()], dtype=np.float32)
    overall_std = np.array([arr.std()], dtype=np.float32)

    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.float32)

    gray_q25 = np.array([np.quantile(gray, 0.25)], dtype=np.float32)
    gray_q50 = np.array([np.quantile(gray, 0.50)], dtype=np.float32)
    gray_q75 = np.array([np.quantile(gray, 0.75)], dtype=np.float32)

    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    grad_abs_mean = np.array(
        [np.mean(np.abs(dx)) + np.mean(np.abs(dy))], dtype=np.float32
    )
    grad_sq_mean = np.array([np.mean(dx * dx) + np.mean(dy * dy)], dtype=np.float32)

    gray_img = Image.fromarray(np.clip(gray * 255.0, 0, 255).astype(np.uint8), mode="L")
    gray_img = gray_img.resize(thumb_gray, resample=_RESAMPLE)
    gray_thumb = (np.asarray(gray_img, dtype=np.float32) / 255.0).reshape(-1)
    gray_thumb_mean = np.array([gray_thumb.mean()], dtype=np.float32)
    gray_thumb_std = np.array([gray_thumb.std()], dtype=np.float32)

    extra_scalars = np.array(
        [rg, rb, gb, r_minus_g, r_minus_b, g_minus_b],
        dtype=np.float32,
    )

    feat = np.concatenate(
        [
            means,
            stds,
            hsv_means,
            hsv_stds,
            overall_mean,
            overall_std,
            extra_scalars,
            gray_q25,
            gray_q50,
            gray_q75,
            grad_abs_mean,
            grad_sq_mean,
            gray_thumb_mean,
            gray_thumb_std,
            gray_thumb,
        ],
        axis=0,
    ).astype(np.float32)

    return feat


def build_features(image_ids: pd.Series) -> np.ndarray:
    first_feat = None
    for img_id in image_ids.tolist():
        p = image_path(img_id)
        if os.path.exists(p):
            first_feat = extract_features(p)
            break
    if first_feat is None:
        raise RuntimeError("No images found to infer feature dimension.")

    X = np.zeros((len(image_ids), first_feat.shape[0]), dtype=np.float32)
    for i, img_id in enumerate(image_ids.tolist()):
        p = image_path(img_id)
        if not os.path.exists(p):
            X[i] = 0.0
        else:
            X[i] = extract_features(p)
    return X


X = build_features(train_df["image_id"])
y = train_df[TARGET_COLS].values.astype(int)

strat = train_df[TARGET_COLS].astype(str).agg("_".join, axis=1)

X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=strat
)

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "ovr",
            OneVsRestClassifier(
                LogisticRegression(
                    max_iter=4000, solver="lbfgs", class_weight="balanced", C=3.0
                )
            ),
        ),
    ]
)

clf.fit(X_tr, y_tr)

va_pred = clf.predict_proba(X_va)
auc_cols = []
for j, c in enumerate(TARGET_COLS):
    if len(np.unique(y_va[:, j])) < 2:
        continue
    auc_cols.append(roc_auc_score(y_va[:, j], va_pred[:, j]))
if len(auc_cols) > 0:
    print("Local mean column-wise ROC AUC (valid):", float(np.mean(auc_cols)))
else:
    print("Local AUC could not be computed (constant labels in validation split).")

clf.fit(X, y)

X_test = build_features(test_df["image_id"])
test_pred = clf.predict_proba(X_test)

pred_df = pd.DataFrame(test_pred, columns=TARGET_COLS)
pred_df.insert(0, "image_id", test_df["image_id"].values)

make_submission_file_from_df(pred_df, out_path="submission.csv")
