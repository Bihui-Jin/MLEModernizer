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

CANDIDATE_BASES = [
    "../input/plant-pathology-2020-fgvc7",
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "../input",
    "/kaggle/input",
    "/kaggle/data",
]


def find_file(relpath_options):
    for base in CANDIDATE_BASES:
        for rel in relpath_options:
            p = os.path.join(base, rel)
            if os.path.exists(p):
                return p
    for rel in relpath_options:
        if os.path.exists(rel):
            return rel
    raise FileNotFoundError(
        f"Could not find any of: {relpath_options} in bases: {CANDIDATE_BASES}"
    )


train_csv_path = find_file(["train.csv", "plant-pathology-2020-fgvc7/train.csv"])
test_csv_path = find_file(["test.csv", "plant-pathology-2020-fgvc7/test.csv"])
sample_sub_path = find_file(
    ["sample_submission.csv", "plant-pathology-2020-fgvc7/sample_submission.csv"]
)


def find_images_dir():
    candidates = []
    for base in CANDIDATE_BASES:
        candidates += [
            os.path.join(base, "images"),
            os.path.join(base, "plant-pathology-2020-fgvc7", "images"),
            os.path.join(
                base,
                "plant-pathology-2020-fgvc7",
                "plant-pathology-2020-fgvc7",
                "images",
            ),
        ]
    for d in candidates:
        if os.path.isdir(d):
            return d
    raise FileNotFoundError("Could not find images directory in expected locations.")


images_dir = find_images_dir()

print("train_csv:", train_csv_path)
print("test_csv:", test_csv_path)
print("sample_submission:", sample_sub_path)
print("images_dir:", images_dir)



## === cell 1
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

try:
    from PIL import Image  # type: ignore

    _READER = "PIL"
except Exception:
    Image = None
    try:
        import imageio.v2 as imageio  # type: ignore

        _READER = "IMAGEIO"
    except Exception:
        imageio = None
        import matplotlib.image as mpimg  # type: ignore

        _READER = "MPL"

try:
    from skimage.transform import resize as sk_resize  # type: ignore

    _HAS_SKIMAGE = True
except Exception:
    sk_resize = None
    _HAS_SKIMAGE = False

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
sub = pd.read_csv(sample_sub_path)

assert "image_id" in sub.columns, "sample_submission must have image_id"

TARGET_COLS = [c for c in sub.columns if c != "image_id"]
if len(TARGET_COLS) != 4:
    raise ValueError(
        f"Expected 4 target columns, got {len(TARGET_COLS)}: {TARGET_COLS}"
    )

missing_in_train = [c for c in TARGET_COLS if c not in train_df.columns]
if missing_in_train:
    raise ValueError(
        f"Train is missing target columns {missing_in_train}. "
        f"Train columns: {list(train_df.columns)}; sample targets: {TARGET_COLS}"
    )

_images_index = {}
for fn in os.listdir(images_dir):
    _images_index[fn] = os.path.join(images_dir, fn)
    _images_index[fn.lower()] = os.path.join(images_dir, fn)

_path_cache = {}


def _resolve_image_path(image_id):
    s = str(image_id)
    if s in _path_cache:
        return _path_cache[s]

    candidates = (s, s + ".jpg" if not s.lower().endswith(".jpg") else s)
    for key in candidates:
        p = _images_index.get(key) or _images_index.get(key.lower())
        if p is not None and os.path.exists(p):
            _path_cache[s] = p
            return p

    p = os.path.join(images_dir, s)
    if os.path.exists(p):
        _path_cache[s] = p
        return p
    if not s.lower().endswith(".jpg"):
        p2 = os.path.join(images_dir, s + ".jpg")
        if os.path.exists(p2):
            _path_cache[s] = p2
            return p2

    raise FileNotFoundError(
        f"Image not found for image_id={image_id} under {images_dir}"
    )


def _to_rgb_float01(arr):
    arr = arr.astype(np.float32, copy=False)
    if arr.max() > 1.5:
        arr = arr / 255.0

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[:, :, :3]
    else:
        arr = np.repeat(arr[:, :, :1], 3, axis=2)

    return np.clip(arr, 0.0, 1.0)


def load_image_features(image_id, size=(64, 64)):
    img_path = _resolve_image_path(image_id)

    if _READER == "PIL":
        im = Image.open(img_path).convert("RGB").resize(size, resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    else:
        if _READER == "IMAGEIO":
            arr = imageio.imread(img_path)
        else:
            arr = mpimg.imread(img_path)

        arr = _to_rgb_float01(arr)

        if _HAS_SKIMAGE:
            arr = sk_resize(
                arr,
                (size[0], size[1], 3),
                order=1,  # bilinear
                mode="reflect",
                anti_aliasing=True,
                preserve_range=True,
            ).astype(np.float32, copy=False)
            arr = np.clip(arr, 0.0, 1.0)
        else:
            h, w = arr.shape[:2]
            yy = (np.linspace(0, h - 1, size[0])).astype(int)
            xx = (np.linspace(0, w - 1, size[1])).astype(int)
            arr = arr[yy][:, xx]

    return arr.reshape(-1)


_feature_dim = 64 * 64 * 3
train_ids = train_df["image_id"].values
test_ids = test_df["image_id"].values

X_train = np.empty((train_ids.shape[0], _feature_dim), dtype=np.float32)
for i, iid in enumerate(train_ids):
    X_train[i, :] = load_image_features(iid)

y_train_onehot = train_df[TARGET_COLS].astype(int).values
row_sums = y_train_onehot.sum(axis=1)
if not np.all(row_sums == 1):
    raise ValueError(
        f"Expected one-hot labels summing to 1 per row; got min={row_sums.min()}, max={row_sums.max()}"
    )

y_train_class = (
    (y_train_onehot * np.arange(len(TARGET_COLS), dtype=int)).sum(axis=1).astype(int)
)

X_test = np.empty((test_ids.shape[0], _feature_dim), dtype=np.float32)
for i, iid in enumerate(test_ids):
    X_test[i, :] = load_image_features(iid)

print("Reader:", _READER, "| skimage_resize:", _HAS_SKIMAGE)
print("TARGET_COLS:", TARGET_COLS)
print("X_train shape:", X_train.shape, "y_train_class shape:", y_train_class.shape)
print("X_test shape:", X_test.shape)



## === cell 2
model = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=False, with_std=True)),
        (
            "clf",
            LogisticRegression(
                solver="lbfgs",
                max_iter=5000,
                multi_class="multinomial",
                n_jobs=None,
                random_state=0,
                class_weight="balanced",
                C=2.0,
            ),
        ),
    ]
)

X_tr, X_va, y_tr, y_va = train_test_split(
    X_train,
    y_train_class,
    test_size=0.2,
    random_state=0,
    stratify=y_train_class,
    shuffle=True,
)
model.fit(X_tr, y_tr)

clf = model.named_steps["clf"]
va_proba_raw = model.predict_proba(X_va).astype(np.float64)
va_classes = getattr(clf, "classes_", np.arange(va_proba_raw.shape[1]))

va_pred = np.zeros((va_proba_raw.shape[0], len(TARGET_COLS)), dtype=np.float64)
for j, cls in enumerate(va_classes):
    cls = int(cls)
    if 0 <= cls < va_pred.shape[1]:
        va_pred[:, cls] = va_proba_raw[:, j]

y_va_onehot = np.zeros((y_va.shape[0], len(TARGET_COLS)), dtype=int)
y_va_onehot[np.arange(y_va.shape[0]), y_va] = 1
try:
    local_auc = roc_auc_score(y_va_onehot, va_pred, average="macro")
    print("Local mean column-wise ROC AUC (macro over targets):", float(local_auc))
except Exception as e:
    print("Local AUC could not be computed:", repr(e))

model.fit(X_train, y_train_class)

proba_raw = model.predict_proba(X_test).astype(np.float64)
proba_raw = np.clip(proba_raw, 0.0, 1.0)

clf = model.named_steps["clf"]
classes = getattr(clf, "classes_", np.arange(proba_raw.shape[1]))

pred = np.zeros((proba_raw.shape[0], len(TARGET_COLS)), dtype=np.float64)
for j, cls in enumerate(classes):
    cls = int(cls)
    if 0 <= cls < pred.shape[1]:
        pred[:, cls] = proba_raw[:, j]



## === cell 3
pred_df = pd.DataFrame(pred, columns=TARGET_COLS)
pred_df["image_id"] = test_df["image_id"].values

out = sub[["image_id"] + TARGET_COLS].merge(
    pred_df, on="image_id", how="left", suffixes=("", "_pred")
)

for c in TARGET_COLS:
    pc = f"{c}_pred"
    if pc in out.columns:
        out[c] = out[pc].astype(float)
        out.drop(columns=[pc], inplace=True)
    out[c] = out[c].fillna(0.25)

assert (
    out.shape[0] == sub.shape[0]
), "Submission row count mismatch vs sample_submission"
assert list(out.columns) == ["image_id"] + TARGET_COLS, "Submission columns mismatch"
out.to_csv("submission.csv", index=False)

print(out.head())
print("Wrote submission.csv with shape:", out.shape)
