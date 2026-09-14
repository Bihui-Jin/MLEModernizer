# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.9715261650067112

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.68107) has done: 'I remove the dependency on missing external “plant-best-sub” files that cause the FileNotFoundError, and instead generate predictions from the provided data only. To keep changes minimal and ensure an end-to-end run with the available packages, I implement a stable image-based baseline using simple color/statistical features extracted from the JPEGs and train one-vs-rest logistic regression models for the four labels. This preserves the evaluation semantics (probabilities per class) and produces a valid `submission.csv` with the exact required columns and test `image_id` order. I also make the input path resolution robust to your folder layout so it runs in this environment.'
- What this solution (achieved 0.73272) has done: 'We keep your “simple image stats → one-vs-rest LogisticRegression” core pipeline unchanged, but make two minimal adjustments that typically raise mean ROC AUC substantially for this competition: (1) compute features at a slightly higher resize resolution (keeps same feature family, just less information loss), and (2) add a very small set of additional, still-simple global color/texture statistics (HSV moments + a few gray-percentiles) to improve separability while preserving the same training loop and model. We also switch LogisticRegression to a slightly stronger default regularization (C) and increase max_iter to ensure convergence; this does not change the approach, only stabilizes fitting. Submission writing/order is kept identical and still uses `sample_submission.csv` columns and `test.csv` image_id alignment.'
- What this solution (achieved 0.78821) has done: 'Your current gap to the target is ~0.2388 (0.73272 vs 0.97153), so we need a meaningful but still “same core pipeline” uplift. I keep the exact approach (global image stats → one-vs-rest LogisticRegression) and only strengthen it by (1) extracting the same kinds of global statistics at multiple resize scales (more robust, still simple stats) and (2) adding a few additional global color/texture moments and edge summaries (still not changing the modeling family). I also set LogisticRegression to use a slightly larger C and more iterations to reduce underfitting/convergence issues, while keeping the same solver and overall training loop. Submission writing/order and column names remain exactly aligned to `sample_submission.csv` and `test.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from PIL import Image, ImageFilter
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(42)



## === cell 1
CANDIDATE_BASES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/data",
    "/kaggle/input",
    "../input/plant-pathology-2020-fgvc7",
    "../input",
]


def find_file(filename: str):
    for base in CANDIDATE_BASES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find {filename} under any candidate base paths: {CANDIDATE_BASES}"
    )


def find_images_dir():
    for base in CANDIDATE_BASES:
        path = os.path.join(base, "images")
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f"Could not find images/ directory under any candidate base paths: {CANDIDATE_BASES}"
    )


train_csv_path = find_file("train.csv")
test_csv_path = find_file("test.csv")
sample_sub_path = find_file("sample_submission.csv")
images_dir = find_images_dir()

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
assert all(
    c in train_df.columns for c in ["image_id"] + target_cols
), "Train columns mismatch."
assert (
    list(sample_sub.columns) == ["image_id"] + target_cols
), "Sample submission columns mismatch."

train_df.head(), test_df.head(), sample_sub.head()




## === cell 2
def _safe_open_rgb(image_path):
    try:
        return Image.open(image_path).convert("RGB")
    except Exception:
        return None


def _rgb_to_hsv_arr(rgb01: np.ndarray) -> np.ndarray:
    r = rgb01[..., 0]
    g = rgb01[..., 1]
    b = rgb01[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    delt = maxc - minc

    eps = 1e-12
    s = delt / (maxc + eps)

    h = np.zeros_like(maxc, dtype=np.float32)
    mask = delt > 0

    rc = (maxc - r) / (delt + eps)
    gc = (maxc - g) / (delt + eps)
    bc = (maxc - b) / (delt + eps)

    m = mask & (r == maxc)
    h[m] = bc[m] - gc[m]
    m = mask & (g == maxc)
    h[m] = 2.0 + (rc[m] - bc[m])
    m = mask & (b == maxc)
    h[m] = 4.0 + (gc[m] - rc[m])

    h = (h / 6.0) % 1.0
    hsv = np.stack([h, s, v], axis=-1).astype(np.float32)
    return hsv


def _features_single_scale_from_arr(
    arr_rgb_01: np.ndarray, hsv_arr_01: np.ndarray, edge_l_01: np.ndarray
):
    r, g, b = arr_rgb_01[:, :, 0], arr_rgb_01[:, :, 1], arr_rgb_01[:, :, 2]
    eps = 1e-6

    means = arr_rgb_01.mean(axis=(0, 1))  # 3
    stds = arr_rgb_01.std(axis=(0, 1))  # 3
    mins = arr_rgb_01.min(axis=(0, 1))  # 3
    maxs = arr_rgb_01.max(axis=(0, 1))  # 3

    exg = 2 * g - r - b
    exr = 1.4 * r - g
    exb = 1.4 * b - g
    exg_mean, exg_std = exg.mean(), exg.std()
    exr_mean, exr_std = exr.mean(), exr.std()
    exb_mean, exb_std = exb.mean(), exb.std()

    r_mean = r.mean()
    g_mean = g.mean()
    b_mean = b.mean()
    rg_ratio = (r_mean + eps) / (g_mean + eps)
    gb_ratio = (g_mean + eps) / (b_mean + eps)
    rb_ratio = (r_mean + eps) / (b_mean + eps)

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    gmag = np.sqrt(gx * gx + gy * gy)

    p5, p10, p25, p50, p75, p90, p95 = np.percentile(
        gray, [5, 10, 25, 50, 75, 90, 95]
    ).astype(np.float32)
    gray_mean = gray.mean()
    gray_std = gray.std()
    gray_skew = ((gray - gray_mean) ** 3).mean() / (gray_std**3 + eps)
    gray_kurt = ((gray - gray_mean) ** 4).mean() / (gray_std**4 + eps)

    hsv_means = hsv_arr_01.mean(axis=(0, 1)).astype(np.float32)  # 3
    hsv_stds = hsv_arr_01.std(axis=(0, 1)).astype(np.float32)  # 3

    edge_mean = edge_l_01.mean()
    edge_std = edge_l_01.std()
    flat = edge_l_01.reshape(-1)
    if flat.size == 0:
        edge_p90 = np.float32(0.0)
    else:
        k = int(np.ceil(0.90 * flat.size)) - 1
        k = 0 if k < 0 else (flat.size - 1 if k >= flat.size else k)
        edge_p90 = np.partition(flat, k)[k].astype(np.float32)

    feat = np.concatenate(
        [
            means.astype(np.float32),
            stds.astype(np.float32),
            mins.astype(np.float32),
            maxs.astype(np.float32),  # 12
            np.array(
                [
                    exg_mean,
                    exg_std,
                    exr_mean,
                    exr_std,
                    exb_mean,
                    exb_std,
                    rg_ratio,
                    gb_ratio,
                    rb_ratio,
                    gx,
                    gy,
                    gmag,
                ],
                dtype=np.float32,
            ),  # 12
            hsv_means,  # 3
            hsv_stds,  # 3
            np.array(
                [
                    p5,
                    p10,
                    p25,
                    p50,
                    p75,
                    p90,
                    p95,
                    gray_mean,
                    gray_std,
                    gray_skew,
                    gray_kurt,
                ],
                dtype=np.float32,
            ),  # 11
            np.array([edge_mean, edge_std, edge_p90], dtype=np.float32),  # 3
        ]
    )
    return feat.astype(np.float32)


def image_features(image_path, sizes=((96, 96), (160, 160), (224, 224))):
    img_rgb = _safe_open_rgb(image_path)
    if img_rgb is None:
        per_scale_len = 12 + 12 + 3 + 3 + 11 + 3  # 44
        return np.zeros(per_scale_len * len(sizes), dtype=np.float32)

    max_size = max(sizes, key=lambda x: x[0] * x[1])
    img_big = img_rgb.resize(max_size)

    edge_big = img_big.filter(ImageFilter.FIND_EDGES)

    arr_big = np.asarray(img_big, dtype=np.float32) / 255.0
    edge_big_l = np.asarray(edge_big.convert("L"), dtype=np.float32) / 255.0

    feats = []
    for sz in sizes:
        if sz == max_size:
            arr = arr_big
            edge_l = edge_big_l
        else:
            img_s = img_big.resize(sz)
            arr = np.asarray(img_s, dtype=np.float32) / 255.0
            edge_l = (
                np.asarray(edge_big.resize(sz).convert("L"), dtype=np.float32) / 255.0
            )

        hsv_arr = _rgb_to_hsv_arr(arr)
        feats.append(_features_single_scale_from_arr(arr, hsv_arr, edge_l))
    return np.concatenate(feats, axis=0).astype(np.float32)


def build_feature_matrix(image_ids):
    image_ids = list(image_ids)
    join = os.path.join
    idir = images_dir

    first_path = join(idir, f"{image_ids[0]}.jpg")
    first_feat = image_features(first_path)
    d = int(first_feat.shape[0])

    X = np.zeros((len(image_ids), d), dtype=np.float32)
    X[0] = first_feat

    for i in range(1, len(image_ids)):
        img_path = join(idir, f"{image_ids[i]}.jpg")
        X[i] = image_features(img_path)
    return X


X_train = build_feature_matrix(train_df["image_id"].values)
X_test = build_feature_matrix(test_df["image_id"].values)

y_train = train_df[target_cols].astype(np.int32).values

X_train.shape, X_test.shape, y_train.shape



## === cell 3
from joblib import Parallel, delayed

try:
    poly = PolynomialFeatures(
        degree=2, interaction_only=True, include_bias=False, sparse_output=True
    )
except TypeError:
    poly = PolynomialFeatures(
        degree=2, interaction_only=True, include_bias=False, sparse=True
    )

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

try:
    X_train_poly = X_train_poly.astype(np.float32, copy=False)
    X_test_poly = X_test_poly.astype(np.float32, copy=False)
except Exception:
    pass

scaler = StandardScaler(with_mean=False, with_std=True)
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)


def _fit_predict_one(j):
    clf = LogisticRegression(
        solver="saga",
        penalty="l2",
        max_iter=8000,
        C=5.0,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,  # deterministic inside each fit; parallelism handled outside
    )
    clf.fit(X_train_scaled, y_train[:, j])
    pred = clf.predict_proba(X_test_scaled)[:, 1].astype(np.float64, copy=False)
    return clf, pred


n_jobs = min(4, os.cpu_count() or 1)
results = Parallel(n_jobs=n_jobs, prefer="processes")(
    delayed(_fit_predict_one)(j) for j in range(len(target_cols))
)

models = []
test_pred = np.zeros((len(test_df), len(target_cols)), dtype=np.float64)
for j, (clf, pred) in enumerate(results):
    test_pred[:, j] = pred
    models.append(make_pipeline(scaler, clf))

test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)
test_pred[:3]



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3333550745.py in <cell line: 0>()
      5 try:
----> 6     poly = PolynomialFeatures(
      7         degree=2, interaction_only=True, include_bias=False, sparse_output=True

TypeError: PolynomialFeatures.__init__() got an unexpected keyword argument 'sparse_output'

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3333550745.py in <cell line: 0>()
      8     )
      9 except TypeError:
---> 10     poly = PolynomialFeatures(
     11         degree=2, interaction_only=True, include_bias=False, sparse=True
     12     )

TypeError: PolynomialFeatures.__init__() got an unexpected keyword argument 'sparse'

## === cell 4
sub = pd.DataFrame({"image_id": test_df["image_id"].values})
for j, c in enumerate(target_cols):
    sub[c] = test_pred[:, j]

sub = sub[["image_id"] + target_cols]
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
print("Saved to:", os.path.abspath("submission.csv"))

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1079794411.py in <cell line: 0>()
      2 sub = pd.DataFrame({"image_id": test_df["image_id"].values})
      3 for j, c in enumerate(target_cols):
----> 4     sub[c] = test_pred[:, j]
      5 
      6 sub = sub[["image_id"] + target_cols]

NameError: name 'test_pred' is not defined
