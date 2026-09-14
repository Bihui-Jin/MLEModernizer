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
scipy==1.15.3
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

0.9669491813515626

# 6. Current score

0.65245

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your notebook fails because it tries to read four external “../input/...” ensemble submissions that are not present in this environment, so all downstream variables are undefined. I keep the same “pick the lowest-entropy model per row” ensemble core logic, but make it robust by automatically using any available submission files if they exist, otherwise falling back to a valid baseline built from `sample_submission.csv` (uniform probabilities). I also enforce correct column order (`image_id, healthy, multiple_diseases, rust, scab`) and align predictions by `image_id` to avoid silent misalignment bugs. This run end-to-end and always write a valid `submission.csv`.'
- What this solution (achieved 0.5776) has done: 'Your current 0.5 score is coming from the uniform-probability fallback (or from selecting an uninformative baseline) because no real model predictions exist in this environment. To move toward the 0.9669 target without changing the ensemble selection logic, I generate legitimate predictions from the provided train images using a minimal scikit-learn baseline: pixel-intensity features + One-vs-Rest Logistic Regression, then feed those predictions into your existing “lowest-entropy per row” selector. This keeps the core “choose lowest-entropy model per row” semantics intact, but ensures at least one non-uniform candidate model is available, which should substantially improve ROC AUC over 0.5. I also keep strict `image_id` alignment and correct submission columns/ordering, and still write `submission.csv` end-to-end.'
- What this solution (achieved 0.65757) has done: 'Your current score is far below the target, so we should improve the predictive signal while keeping your ensemble’s core “pick lowest-entropy model per row” logic unchanged. The biggest issue is the very weak image representation (raw 64×64 RGB pixels) and a too-restrictive probability renormalization that can distort one-vs-rest probabilities for a ROC-AUC metric. I keep the same scikit-learn OneVsRest+LogisticRegression approach, but switch to a stronger yet still lightweight handcrafted feature (color + gradient HOG-like summary) and remove per-row probability sum-to-1 normalization (keeping only clipping), which typically improves multi-label ROC AUC. I also fix a masking/indexing bug risk by using positional indexing consistently when writing back selected predictions.'
- What this solution (achieved 0.65245) has done: 'Your score gap to the target is large (0.6576 vs 0.9669), so we need more predictive signal while keeping your ensemble’s “pick lowest-entropy model per row” core logic unchanged. The biggest issue is that the entropy selector currently prefers overconfident models, but the entropy is computed on unnormalized one-vs-rest probabilities, which makes entropy comparisons inconsistent across models; we compute entropy on per-row normalized probabilities *only for selection*, while leaving the actual output probabilities untouched for ROC-AUC. We also strengthen the same scikit-learn OVR Logistic Regression baseline without changing the model family by standardizing features (crucial for LBFGS/LogReg) and enabling class balancing to help rare classes. Finally, we keep strict `image_id` alignment and still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import entropy

from PIL import Image
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler



## === cell 1
BASE_DIR = "/kaggle/data"
COMP_DIR = os.path.join(BASE_DIR, "plant-pathology-2020-fgvc7")

sample_path_candidates = [
    os.path.join(COMP_DIR, "sample_submission.csv"),
    os.path.join(BASE_DIR, "sample_submission.csv"),
]
test_path_candidates = [
    os.path.join(COMP_DIR, "test.csv"),
    os.path.join(BASE_DIR, "test.csv"),
]
train_path_candidates = [
    os.path.join(COMP_DIR, "train.csv"),
    os.path.join(BASE_DIR, "train.csv"),
]

images_dir_candidates = [
    os.path.join(COMP_DIR, "images"),
    os.path.join(BASE_DIR, "images"),
]


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


SAMPLE_PATH = first_existing(sample_path_candidates)
TEST_PATH = first_existing(test_path_candidates)
TRAIN_PATH = first_existing(train_path_candidates)
IMAGES_DIR = first_existing(images_dir_candidates)

if SAMPLE_PATH is None or TEST_PATH is None or TRAIN_PATH is None or IMAGES_DIR is None:
    raise FileNotFoundError(
        f"Could not find required files. SAMPLE_PATH={SAMPLE_PATH}, TEST_PATH={TEST_PATH}, "
        f"TRAIN_PATH={TRAIN_PATH}, IMAGES_DIR={IMAGES_DIR}"
    )

sub_template = pd.read_csv(SAMPLE_PATH)
test_df = pd.read_csv(TEST_PATH)
train_df = pd.read_csv(TRAIN_PATH)

target_cols = [c for c in sub_template.columns if c != "image_id"]
required_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]

if list(sub_template.columns) != required_cols:
    missing = set(required_cols) - set(sub_template.columns)
    if missing:
        raise ValueError(f"Sample submission missing required columns: {missing}")
    sub_template = sub_template[required_cols]
    target_cols = [c for c in sub_template.columns if c != "image_id"]

sub = sub_template.merge(test_df[["image_id"]], on="image_id", how="right")
sub = sub[required_cols]



## === cell 2
preferred_paths = [
    "../input/average-efficientnet/submission.csv",
    "../input/classification-densenet201-efficientnetb7/submission.csv",
    "../input/tf-zoo-models-on-tpu/submission.csv",
    "../input/fork-of-plant-2020-tpu-915e9c/submission.csv",
]

available_paths = [p for p in preferred_paths if os.path.exists(p)]

search_roots = [
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/working",
]
for root in search_roots:
    if os.path.exists(root):
        found = glob.glob(os.path.join(root, "**", "submission.csv"), recursive=True)
        available_paths.extend(found)

seen = set()
dedup_paths = []
for p in available_paths:
    if p not in seen:
        seen.add(p)
        dedup_paths.append(p)
available_paths = dedup_paths


def load_and_align_submission(path, template_image_ids, required_cols, target_cols):
    df = pd.read_csv(path)
    if "image_id" not in df.columns:
        return None
    if not set(target_cols).issubset(df.columns):
        return None
    df = df[["image_id"] + target_cols].copy()
    df = pd.DataFrame({"image_id": template_image_ids}).merge(
        df, on="image_id", how="left"
    )
    if df[target_cols].isna().any().any():
        return None
    df = df[required_cols]
    df[target_cols] = df[target_cols].astype(float).clip(1e-7, 1 - 1e-7)
    return df


template_image_ids = sub["image_id"].tolist()
subs = []
for p in available_paths:
    aligned = load_and_align_submission(
        p, template_image_ids, required_cols, target_cols
    )
    if aligned is not None:
        subs.append(aligned)

baseline = sub.copy()
baseline[target_cols] = 1.0 / len(target_cols)




## === cell 3
def _extract_features(image_id, images_dir, size=(128, 128), bins=9):
    img_path = os.path.join(images_dir, f"{image_id}.jpg")
    if not os.path.exists(img_path):
        return None

    with Image.open(img_path) as im:
        im = im.convert("RGB")
        im = im.resize(size)
        rgb = np.asarray(im, dtype=np.float32) / 255.0

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn

    h = np.zeros_like(mx)
    mask = diff > 1e-8
    idx = (mx == r) & mask
    h[idx] = ((g[idx] - b[idx]) / diff[idx]) % 6.0
    idx = (mx == g) & mask
    h[idx] = ((b[idx] - r[idx]) / diff[idx]) + 2.0
    idx = (mx == b) & mask
    h[idx] = ((r[idx] - g[idx]) / diff[idx]) + 4.0
    h = (h / 6.0).astype(np.float32)

    s = np.zeros_like(mx)
    s[mx > 1e-8] = (diff[mx > 1e-8] / mx[mx > 1e-8]).astype(np.float32)
    v = mx.astype(np.float32)

    def moments(ch):
        return np.array(
            [
                ch.mean(),
                ch.std(),
                np.percentile(ch, 10),
                np.percentile(ch, 50),
                np.percentile(ch, 90),
            ],
            dtype=np.float32,
        )

    hsv_feat = np.concatenate([moments(h), moments(s), moments(v)], axis=0)

    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.float32)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    mag = np.sqrt(gx * gx + gy * gy) + 1e-8
    ang = (np.arctan2(gy, gx) + np.pi) / (2.0 * np.pi)  # 0..1

    H, W = gray.shape
    hs = [slice(0, H // 2), slice(H // 2, H)]
    ws = [slice(0, W // 2), slice(W // 2, W)]
    hog_parts = []
    for si in hs:
        for sj in ws:
            a = ang[si, sj].ravel()
            m = mag[si, sj].ravel()
            hist = np.zeros(bins, dtype=np.float32)
            bi = np.minimum((a * bins).astype(int), bins - 1)
            for k in range(bins):
                mk = m[bi == k].sum()
                hist[k] = mk
            hist = hist / (hist.sum() + 1e-8)
            hog_parts.append(hist)
    hog_feat = np.concatenate(hog_parts, axis=0)

    feat = np.concatenate([hsv_feat, hog_feat], axis=0).astype(np.float32)
    return feat


def build_sklearn_submission(train_df, test_df, images_dir, target_cols, required_cols):
    X_list, y_list = [], []
    for _, row in train_df.iterrows():
        feat = _extract_features(row["image_id"], images_dir)
        if feat is None:
            continue
        X_list.append(feat)
        y_list.append(row[target_cols].to_numpy(dtype=int))
    if len(X_list) == 0:
        return None
    X_train = np.vstack(X_list)
    y_train = np.vstack(y_list)

    base_lr = LogisticRegression(
        solver="lbfgs",
        max_iter=800,
        C=3.0,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf = OneVsRestClassifier(
        make_pipeline(StandardScaler(with_mean=True, with_std=True), base_lr)
    )
    clf.fit(X_train, y_train)

    X_test_list, ok_ids = [], []
    for image_id in test_df["image_id"].tolist():
        feat = _extract_features(image_id, images_dir)
        if feat is None:
            return None
        X_test_list.append(feat)
        ok_ids.append(image_id)
    X_test = np.vstack(X_test_list)

    proba = clf.predict_proba(X_test).astype(float)
    proba = np.clip(proba, 1e-7, 1 - 1e-7)

    df = pd.DataFrame({"image_id": ok_ids})
    for j, c in enumerate(target_cols):
        df[c] = proba[:, j]
    df = df[required_cols]
    return df


sk_model_sub = build_sklearn_submission(
    train_df, test_df, IMAGES_DIR, target_cols, required_cols
)
if sk_model_sub is not None:
    subs.append(sk_model_sub)

if len(subs) == 0:
    subs = [baseline]
else:
    subs.append(baseline)

sub1 = subs[0]
sub2 = subs[1] if len(subs) > 1 else subs[0]
sub3 = subs[2] if len(subs) > 2 else subs[0]
sub4 = subs[3] if len(subs) > 3 else subs[0]



## === cell 4
pred_stack = np.stack(
    [s[target_cols].to_numpy(dtype=float) for s in subs], axis=0
)  # (M, N, C)
pred_stack = np.clip(pred_stack, 1e-12, 1.0)  # avoid log(0) in entropy

row_sums = pred_stack.sum(axis=2, keepdims=True)  # (M, N, 1)
row_sums = np.where(row_sums <= 0, 1.0, row_sums)
pred_stack_for_entropy = pred_stack / row_sums

entropies = entropy(pred_stack_for_entropy, base=2, axis=2)  # (M, N)
selected = np.argmin(entropies, axis=0)



## === cell 5
final_preds = np.zeros((len(sub), len(target_cols)), dtype=float)
for m_idx, s in enumerate(subs):
    mask = selected == m_idx
    if np.any(mask):
        idx = np.where(mask)[0]
        final_preds[idx] = s.iloc[idx][target_cols].to_numpy(dtype=float)

sub[target_cols] = final_preds
sub[target_cols] = sub[target_cols].astype(float).clip(1e-7, 1 - 1e-7)
sub = sub[required_cols]
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print("Number of candidate models in entropy stack:", len(subs))
print(sub.head())
