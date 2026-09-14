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

# 5. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.special import expit

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
def _extract_features(image_id, images_dir, size=(128, 128), bins=9, flip=False):
    img_path = os.path.join(images_dir, f"{image_id}.jpg")
    if not os.path.exists(img_path):
        return None

    with Image.open(img_path) as im:
        im = im.convert("RGB")
        im = im.resize(size)
        if flip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
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

    def grid_moments(ch, gh=4, gw=4):
        H, W = ch.shape
        feats = []
        for i in range(gh):
            i0 = (i * H) // gh
            i1 = ((i + 1) * H) // gh
            for j in range(gw):
                j0 = (j * W) // gw
                j1 = ((j + 1) * W) // gw
                patch = ch[i0:i1, j0:j1]
                feats.append(moments(patch))
        return np.concatenate(feats, axis=0).astype(np.float32)

    rgb_feat_global = np.concatenate([moments(r), moments(g), moments(b)], axis=0)
    hsv_feat_global = np.concatenate([moments(h), moments(s), moments(v)], axis=0)

    rgb_feat_grid = np.concatenate(
        [grid_moments(r), grid_moments(g), grid_moments(b)], axis=0
    )
    hsv_feat_grid = np.concatenate(
        [grid_moments(h), grid_moments(s), grid_moments(v)], axis=0
    )

    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.float32)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    mag = np.sqrt(gx * gx + gy * gy) + 1e-8
    ang = (np.arctan2(gy, gx) + np.pi) / (2.0 * np.pi)  # 0..1

    mag = mag / (np.sqrt((mag * mag).mean()) + 1e-8)

    H, W = gray.shape
    hs = [
        slice(0, H // 4),
        slice(H // 4, H // 2),
        slice(H // 2, (3 * H) // 4),
        slice((3 * H) // 4, H),
    ]
    ws = [slice(0, W // 2), slice(W // 2, W)]
    hog_parts = []
    for si in hs:
        for sj in ws:
            a = ang[si, sj].ravel()
            m = mag[si, sj].ravel()
            hist = np.zeros(bins, dtype=np.float32)
            bi = np.minimum((a * bins).astype(int), bins - 1)
            for k in range(bins):
                hist[k] = m[bi == k].sum()
            hist = hist / (hist.sum() + 1e-8)
            hog_parts.append(hist)
    hog_feat = np.concatenate(hog_parts, axis=0)

    feat = np.concatenate(
        [rgb_feat_global, hsv_feat_global, rgb_feat_grid, hsv_feat_grid, hog_feat],
        axis=0,
    ).astype(np.float32)
    return feat


def _extract_features_multiscale(
    image_id, images_dir, sizes=((96, 96), (160, 160)), bins=9, flip=False
):
    feats = []
    for sz in sizes:
        f = _extract_features(image_id, images_dir, size=sz, bins=bins, flip=flip)
        if f is None:
            return None
        feats.append(f)
    return np.concatenate(feats, axis=0).astype(np.float32)


def build_sklearn_submissions(
    train_df, test_df, images_dir, target_cols, required_cols
):
    X_list, y_list = [], []
    for _, row in train_df.iterrows():
        feat = _extract_features_multiscale(row["image_id"], images_dir, flip=False)
        feat_f = _extract_features_multiscale(row["image_id"], images_dir, flip=True)
        if feat is None or feat_f is None:
            continue
        y = row[target_cols].to_numpy(dtype=int)
        X_list.append(feat)
        y_list.append(y)
        X_list.append(feat_f)
        y_list.append(y)

    if len(X_list) == 0:
        return []

    X_train = np.vstack(X_list)
    y_train = np.vstack(y_list)

    X_test_list, ok_ids = [], []
    for image_id in test_df["image_id"].tolist():
        feat = _extract_features_multiscale(image_id, images_dir, flip=False)
        if feat is None:
            return []
        X_test_list.append(feat)
        ok_ids.append(image_id)
    X_test = np.vstack(X_test_list)

    submissions = []
    for C in (1.2, 2.0):
        base_lr = LogisticRegression(
            solver="lbfgs",
            max_iter=1600,
            C=C,
            class_weight="balanced",
            n_jobs=-1,
        )
        clf = OneVsRestClassifier(
            make_pipeline(StandardScaler(with_mean=True, with_std=True), base_lr)
        )
        clf.fit(X_train, y_train)

        scores = clf.decision_function(X_test)
        proba = expit(scores).astype(float)
        proba = np.clip(proba, 1e-7, 1 - 1e-7)

        df = pd.DataFrame({"image_id": ok_ids})
        for j, c in enumerate(target_cols):
            df[c] = proba[:, j]
        df = df[required_cols]
        submissions.append(df)

    return submissions


sk_model_subs = build_sklearn_submissions(
    train_df, test_df, IMAGES_DIR, target_cols, required_cols
)
for d in sk_model_subs:
    subs.append(d)

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

eps = 1e-4
pred_stack_for_entropy = (1.0 - eps) * pred_stack_for_entropy + eps * (
    1.0 / len(target_cols)
)

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
