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
import numpy as np
import pandas as pd
import os

CANDIDATE_ROOTS = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/data",
    "/kaggle/input",
    "../input/plant-pathology-2020-fgvc7",
    "../input",
    "../kaggle/data/plant-pathology-2020-fgvc7",
    "../kaggle/data",
]


def _find_file(filename: str):
    for root in CANDIDATE_ROOTS:
        path = os.path.join(root, filename)
        if os.path.exists(path):
            return path
        nested = os.path.join(root, "plant-pathology-2020-fgvc7", filename)
        if os.path.exists(nested):
            return nested
    return None


train_path = _find_file("train.csv")
test_path = _find_file("test.csv")
sample_sub_path = _find_file("sample_submission.csv")

missing = [
    name
    for name, p in [
        ("train.csv", train_path),
        ("test.csv", test_path),
        ("sample_submission.csv", sample_sub_path),
    ]
    if p is None
]
if missing:
    raise FileNotFoundError(
        f"Could not locate required files: {missing}. Searched roots: {CANDIDATE_ROOTS}"
    )

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)

TARGETS = ["healthy", "multiple_diseases", "rust", "scab"]
for c in ["image_id"] + TARGETS:
    if c not in sample_sub.columns:
        raise ValueError(f"sample_submission.csv missing required column: {c}")



## === cell 1
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


def _iter_existing_image_dirs():
    seen = set()
    for root in CANDIDATE_ROOTS:
        for d in (
            os.path.join(root, "images"),
            os.path.join(root, "plant-pathology-2020-fgvc7", "images"),
        ):
            if d in seen:
                continue
            seen.add(d)
            if os.path.isdir(d):
                yield d


def _build_image_path_index():
    idx = {}
    for img_dir in _iter_existing_image_dirs():
        try:
            for fn in os.listdir(img_dir):
                if not fn.endswith(".jpg"):
                    continue
                iid = fn[:-4]
                if iid not in idx:
                    idx[iid] = os.path.join(img_dir, fn)
        except FileNotFoundError:
            pass
    return idx


_IMAGE_PATH_INDEX = _build_image_path_index()


def _find_image_path(image_id: str):
    return _IMAGE_PATH_INDEX.get(image_id, None)


def _extract_features(image_path: str):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        arr_u8 = np.asarray(img)  # uint8, (H,W,3)

    arr = arr_u8.astype(np.float32) * (1.0 / 255.0)  # (H,W,3)
    flat = arr.reshape(-1, 3)

    ch_mean = flat.mean(axis=0)
    ch_std = flat.std(axis=0)

    gray = flat.mean(axis=1)
    gray_mean = float(gray.mean())
    gray_std = float(gray.std())

    r, g, b = ch_mean
    ngrdi = float((g - r) / (g + r + 1e-6))

    mx = flat.max(axis=1)
    mn = flat.min(axis=1)
    sat = float((mx - mn).mean())

    return np.array(
        [
            ch_mean[0],
            ch_mean[1],
            ch_mean[2],
            ch_std[0],
            ch_std[1],
            ch_std[2],
            gray_mean,
            gray_std,
            ngrdi,
            sat,
        ],
        dtype=np.float32,
    )


def build_feature_matrix(image_ids):
    X = np.zeros((len(image_ids), 10), dtype=np.float32)
    missing_paths = []
    for i, iid in enumerate(image_ids):
        p = _find_image_path(iid)
        if p is None:
            missing_paths.append(iid)
            X[i, :] = 0.0
        else:
            X[i, :] = _extract_features(p)
    if missing_paths:
        print(
            f"Warning: {len(missing_paths)} images not found; features set to 0 for those ids. Example:",
            missing_paths[:5],
        )
    return X


X_train = build_feature_matrix(train_df["image_id"].values)
X_test = build_feature_matrix(test_df["image_id"].values)
Y_train = train_df[TARGETS].values.astype(int)

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "ovr",
            OneVsRestClassifier(
                LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0, random_state=0)
            ),
        ),
    ]
)
clf.fit(X_train, Y_train)
proba = clf.predict_proba(X_test).astype(np.float64)  # (n_test, 4)

d = pd.DataFrame({"image_id": test_df["image_id"].values})
for j, t in enumerate(TARGETS):
    d[t] = proba[:, j]
dsub = [d]
n = len(dsub)
print("Built", n, "model(s) for blending. Test preds shape:", d.shape)



## === cell 2
sub = sample_sub.copy()
sub = sub.merge(test_df[["image_id"]], on="image_id", how="right", sort=False)

for t in TARGETS:
    sub[t] = 0.0

for dm in dsub:
    dm2 = dm.set_index("image_id")[TARGETS]
    sub_vals = sub["image_id"].map(dm2.to_dict(orient="index"))
    add_df = pd.DataFrame(list(sub_vals), index=sub.index)
    for t in TARGETS:
        sub[t] += add_df[t].astype(float).to_numpy()

alpha = 0.01
for t in TARGETS:
    sub[t] = (1 - alpha) * (sub[t] / n) + alpha / 4.0
    sub[t] = sub[t].clip(0.0, 1.0)

sub = sub[["image_id"] + TARGETS]
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
