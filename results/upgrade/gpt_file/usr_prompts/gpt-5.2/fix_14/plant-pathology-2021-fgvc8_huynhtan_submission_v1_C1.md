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
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

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
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

print("Using non-TensorFlow fallback (scikit-learn) pipeline.")



## === cell 1
DATA_ROOT = "/kaggle/input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing: {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing: {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

all_labels = set()
for s in train_df["labels"].astype(str).values:
    for t in s.split():
        if t:
            all_labels.add(t)
class_name = sorted(all_labels)
label_to_idx = {s: i for i, s in enumerate(class_name)}
idx_to_label = {i: s for s, i in label_to_idx.items()}
num_classes = len(class_name)

print("Num atomic labels:", num_classes)
print("Labels:", class_name)



## === cell 2
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib


def load_image_feature(path, size=48):
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = im.resize((size, size), resample=Image.BICUBIC)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr.reshape(-1)


def _stable_cache_key(paths, size, resize_method_tag):
    h = hashlib.sha1()
    h.update(str(size).encode("utf-8"))
    h.update(b"|")
    h.update(str(resize_method_tag).encode("utf-8"))
    for p in paths:
        h.update(b"\0")
        h.update(p.encode("utf-8"))
    return h.hexdigest()


def extract_features_fast(
    paths,
    size=48,
    cache_dir="/kaggle/working/_feat_cache",
    cache_tag="",
    resize_method_tag="bicubic",
):
    """
    Speed change (correctness-preserving):
    - Uses as_completed to keep threads busy even if some images decode slower.
    - Uses a slightly larger default worker cap and chunk submission to reduce Python overhead.
    - Cache semantics unchanged (pure function of (paths,size,resize_tag)); same float outputs.
    """
    os.makedirs(cache_dir, exist_ok=True)
    paths = np.asarray(paths)
    n = len(paths)
    d = size * size * 3

    key = _stable_cache_key(paths.tolist(), size, resize_method_tag)
    cache_path = os.path.join(cache_dir, f"{cache_tag}_{key}_{n}x{d}.npy")

    if os.path.exists(cache_path):
        return np.load(cache_path, mmap_mode="r")

    X = np.empty((n, d), dtype=np.float32)

    max_workers = min(16, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for i, p in enumerate(paths):
            futures.append(ex.submit(load_image_feature, p, size))
        for i, fut in enumerate(as_completed(futures)):
            pass

    X = np.empty((n, d), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(load_image_feature, p, size): i for i, p in enumerate(paths)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            X[i] = fut.result()

    np.save(cache_path, X)
    return X


def labels_to_multihot(s, num_classes, label_to_idx):
    y = np.zeros((num_classes,), dtype=np.int8)
    for t in str(s).split():
        if t in label_to_idx:
            y[label_to_idx[t]] = 1
    return y


def apply_postprocess_rules(pred01_row, healthy_idx=None, complex_idx=None):
    active = np.where(pred01_row == 1)[0].tolist()

    if complex_idx is not None and complex_idx in active:
        active = [complex_idx]

    if healthy_idx is not None and len(active) > 1 and healthy_idx in active:
        active = [j for j in active if j != healthy_idx]

    if len(active) == 0:
        if healthy_idx is not None:
            active = [healthy_idx]
        else:
            active = [int(np.argmax(pred01_row))]  # safety fallback

    out = np.zeros_like(pred01_row, dtype=np.int8)
    out[active] = 1
    return out


train_df = train_df.copy()
train_df["filepath"] = [
    os.path.join(TRAIN_IMG_DIR, x) for x in train_df["image"].values
]

exists_mask = np.fromiter(
    (os.path.exists(p) for p in train_df["filepath"].values),
    dtype=bool,
    count=len(train_df),
)
train_df = train_df.loc[exists_mask].reset_index(drop=True)

Y = np.stack(
    [
        labels_to_multihot(s, num_classes, label_to_idx)
        for s in train_df["labels"].values
    ],
    axis=0,
)

print("Train rows after file-exists filter:", len(train_df))
print("Multi-hot target shape:", Y.shape)
print("Test rows (sample_submission):", len(sample_df))



## === cell 3
X_paths = train_df["filepath"].values

healthy_idx = label_to_idx.get("healthy", None)
if healthy_idx is not None:
    stratify_flag = (Y[:, healthy_idx] == 0).astype(int)
else:
    stratify_flag = (Y.sum(axis=1) > 0).astype(int)

tr_idx, va_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.1,
    random_state=SEED,
    shuffle=True,
    stratify=stratify_flag,
)

feat_size = 48

X_all = extract_features_fast(
    X_paths, size=feat_size, cache_tag="trainval_all", resize_method_tag="bicubic"
)
X_tr = X_all[tr_idx]
X_va = X_all[va_idx]

Y_tr = Y[tr_idx]
Y_va = Y[va_idx]

scaler = StandardScaler(with_mean=True, with_std=True)
X_tr_s = scaler.fit_transform(X_tr)
X_va_s = scaler.transform(X_va)

base_lr = LogisticRegression(
    solver="lbfgs",
    C=0.35,
    max_iter=200,
    n_jobs=1,  # keep per-estimator deterministic and avoid nested parallelism
    random_state=SEED,
    class_weight="balanced",
)

ovr = OneVsRestClassifier(base_lr, n_jobs=min(num_classes, (os.cpu_count() or 4)))
ovr.fit(X_tr_s, Y_tr)

va_proba = ovr.predict_proba(X_va_s).astype(np.float32)

present_labels = Y_va.sum(axis=0) > 0
if present_labels.sum() == 0:
    present_labels[:] = True

complex_idx = label_to_idx.get("complex", None)

threshold_grid = np.unique(
    np.round(
        np.concatenate(
            [
                np.linspace(0.02, 0.18, 33),
                np.linspace(0.20, 0.60, 41),
                np.linspace(0.62, 0.90, 15),
            ]
        ),
        3,
    )
).astype(np.float32)

label_thresholds = np.full((num_classes,), 0.30, dtype=np.float32)

for k in range(num_classes):
    if not present_labels[k]:
        label_thresholds[k] = 0.50
        continue

    y_true_k = Y_va[:, k].astype(np.int8)
    p_k = va_proba[:, k]

    pred_grid = (p_k[None, :] >= threshold_grid[:, None]).astype(np.int8)

    tp = (pred_grid & y_true_k[None, :]).sum(axis=1).astype(np.float32)
    fp = (pred_grid & (1 - y_true_k)[None, :]).sum(axis=1).astype(np.float32)
    fn = ((1 - pred_grid) & y_true_k[None, :]).sum(axis=1).astype(np.float32)

    denom = 2 * tp + fp + fn
    f1s = np.where(denom > 0, (2 * tp) / denom, 0.0).astype(np.float32)

    best_f1 = f1s.max()
    best_idxs = np.where(np.isclose(f1s, best_f1))[0]
    best_thr_k = float(threshold_grid[best_idxs[-1]])
    label_thresholds[k] = best_thr_k

va_pred = (va_proba >= label_thresholds[None, :]).astype(np.int8)
va_pred_pp = np.empty_like(va_pred, dtype=np.int8)
for i in range(va_pred.shape[0]):
    va_pred_pp[i] = apply_postprocess_rules(
        va_pred[i], healthy_idx=healthy_idx, complex_idx=complex_idx
    )

per_label_f1 = []
for k in np.where(present_labels)[0]:
    per_label_f1.append(
        f1_score(Y_va[:, k], va_pred_pp[:, k], average="binary", zero_division=0)
    )
mean_label_f1 = float(np.mean(per_label_f1)) if len(per_label_f1) else 0.0

macro_f1 = float(
    f1_score(
        Y_va[:, present_labels],
        va_pred_pp[:, present_labels],
        average="macro",
        zero_division=0,
    )
)
samples_f1 = float(f1_score(Y_va, va_pred_pp, average="samples", zero_division=0))

print(
    "Per-label thresholds tuned on val (min/mean/max):",
    float(label_thresholds.min()),
    float(label_thresholds.mean()),
    float(label_thresholds.max()),
)
print("Validation mean per-label F1 (proxy for Mean F1):", mean_label_f1)
print("Validation macro-F1 (diagnostic):", macro_f1)
print("Validation samples-F1 (diagnostic):", samples_f1)



## === cell 4
test_files = sorted(
    [
        os.path.join(TEST_IMG_DIR, f)
        for f in os.listdir(TEST_IMG_DIR)
        if f.lower().endswith(".jpg")
    ]
)
print("Discovered test image files:", len(test_files))

X_test = extract_features_fast(
    test_files, size=feat_size, cache_tag="test", resize_method_tag="bicubic"
)
X_test_s = scaler.transform(X_test)

test_proba = ovr.predict_proba(X_test_s).astype(np.float32)

test_pred = (test_proba >= label_thresholds[None, :]).astype(np.int8)

healthy_idx = label_to_idx.get("healthy", None)
complex_idx = label_to_idx.get("complex", None)

pred_labels = []
for i in range(len(test_files)):
    pred01_pp = apply_postprocess_rules(
        test_pred[i], healthy_idx=healthy_idx, complex_idx=complex_idx
    )
    active = np.where(pred01_pp == 1)[0].tolist()
    labs = [idx_to_label[j] for j in active]
    pred_labels.append(" ".join(labs))

pred_df = pd.DataFrame(
    {"image": [os.path.basename(p) for p in test_files], "labels": pred_labels}
)

print(pred_df.head())



## === cell 5
fallback_label = "healthy" if "healthy" in label_to_idx else class_name[0]

submission = sample_df[["image"]].merge(pred_df, on="image", how="left")
submission["labels"] = submission["labels"].fillna(fallback_label).astype(str)

submission = submission[["image", "labels"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
print("Unique predicted label strings (count):", submission["labels"].nunique())
