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

# 5. Target score

0.6415143120960296

# 6. Current score

0.32569

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.75058) has done: 'I remove `tensorflow_addons` (it fails to import in this environment due to protobuf incompatibility) and keep the rest of the TensorFlow pipeline intact. Since the referenced pre-trained model file does not exist in your input, I add a minimal “train-once then predict” fallback using a standard Keras ImageNet backbone so the notebook runs end-to-end and produces a valid `submission.csv`. I also fix path handling, ensure the labels are space-delimited strings, and make sure the submission uses only the basename for `image`. These changes are required for correctness and to yield a non-empty submission file.'
- What this solution (achieved 0.76653) has done: 'Your current code doesn’t yield a score mainly because it fail immediately in this environment: it tries to downgrade `protobuf` (to make `tensorflow_addons` happy), but TensorFlow 2.18 requires protobuf 5/6, so TensorFlow import break after the forced restart. I remove that protobuf-downgrade cell entirely so the notebook runs end-to-end. To move the score upward (without changing the overall “single-label softmax classifier” core logic), I also fix the label mapping bug: your `class_name` list contains combined-label strings inside one entry, which collapses multiple classes into one and discards most training data via filtering; instead we build `class_name` from the actual unique `train.csv` label strings. Finally, I keep the same model/training approach but increase epochs slightly (still lightweight with a frozen backbone) to improve F1 toward your target and ensure the submission aligns exactly to `sample_submission.csv`.'
- What this solution (achieved 0.36452) has done: 'Main bottlenecks are (1) slow Python-loop image loading/resize for ~15k train + ~1.5k val + ~3.7k test images and (2) very expensive multinomial LogisticRegression on ~20k-dimensional raw pixels. To stay within 600s without changing the algorithm, we (a) vectorize and parallelize image decoding/resizing with a thread pool (PIL releases the GIL during decode/resize), (b) preallocate the feature matrix to avoid huge temporary lists/stacks, and (c) cache extracted features to disk (memmap) so reruns and later cells don’t re-decode images. Core model, features (same pixels/normalization), split, scaler, and LR hyperparameters remain identical; only the data loading/extraction implementation is accelerated.'
- What this solution (achieved 0.41026) has done: 'Your current score (0.36452) is far below the target (0.6415), so we should improve performance without changing the core “raw pixels → StandardScaler → multinomial LogisticRegression” approach. The biggest score leak here is label handling: the competition is multi-label and scored by mean F1, but the code treats each unique space-delimited label string as a single class, which prevents predicting multiple labels and hurts F1. With minimal changes, we keep the same feature extraction and LR training loop, but train one-vs-rest logistic regression for each disease label (multi-output) and then output space-delimited labels using a fixed probability threshold tuned on the validation split for mean F1. This preserves the core model family and pipeline while aligning predictions to the metric and submission format.'
- What this solution (achieved 0.38199) has done: 'Your current pipeline is already aligned to multi-label mean F1, but it likely underperforms because each one-vs-rest classifier is heavily class-imbalanced and you use a single global threshold with no safeguard for always including `healthy` alongside diseases. To move the score upward toward 0.6415 with minimal risk and without changing the core “raw pixels → StandardScaler → LogisticRegression” approach, we (1) enable `class_weight="balanced"` in each per-label LogisticRegression to address imbalance, and (2) slightly expand and densify the validation threshold search range so the chosen global threshold is better calibrated to mean F1. Additionally, we apply a tiny post-processing rule: if any disease is predicted, we remove `healthy` (since co-occurrence often hurts F1), while keeping the existing fallback to `healthy` when nothing is predicted. These changes preserve the same architecture/training loop/features/loss, but improve calibration and label-set plausibility for the metric.'
- What this solution (achieved 0.32569) has done: 'We keep your exact raw-pixels → StandardScaler → one-vs-rest LogisticRegression pipeline, but tune two small, high-impact knobs that directly affect mean F1: (1) use `average="samples"` on the validation set when selecting the threshold (this matches the competition’s per-image multi-label F1 behavior much better than macro), and (2) allow a slightly wider, denser threshold grid to find a better-calibrated global cutoff without changing the model. Additionally, we apply a minimal post-processing rule: if `complex` is predicted, drop all other labels (since `complex` commonly stands alone and co-predicting tends to reduce F1), while keeping your existing “remove healthy when diseases exist” and “fallback to healthy if none predicted” logic. These changes are small, metric-aligned, and should move your score upward toward the 0.64 target while preserving core semantics and producing the same `submission.csv` format.'

# 9. Code solution

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

from concurrent.futures import ThreadPoolExecutor
import hashlib


def load_image_feature(path, size=48):
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = im.resize((size, size), resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr.reshape(-1)


def _stable_cache_key(paths, size):
    h = hashlib.sha1()
    h.update(str(size).encode("utf-8"))
    for p in paths:
        h.update(b"\0")
        h.update(p.encode("utf-8"))
    return h.hexdigest()


def extract_features_fast(
    paths, size=48, cache_dir="/kaggle/working/_feat_cache", cache_tag=""
):
    os.makedirs(cache_dir, exist_ok=True)
    paths = np.asarray(paths)
    n = len(paths)
    d = size * size * 3

    key = _stable_cache_key(paths.tolist(), size)
    cache_path = os.path.join(cache_dir, f"{cache_tag}_{key}_{n}x{d}.npy")

    if os.path.exists(cache_path):
        return np.load(cache_path, mmap_mode="r")

    X = np.empty((n, d), dtype=np.float32)

    def _read_one(i_p):
        i, p = i_p
        return i, load_image_feature(p, size=size)

    max_workers = min(8, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, feat in ex.map(_read_one, enumerate(paths), chunksize=64):
            X[i] = feat

    np.save(cache_path, X)
    return X


def labels_to_multihot(s, num_classes, label_to_idx):
    y = np.zeros((num_classes,), dtype=np.int8)
    for t in str(s).split():
        if t in label_to_idx:
            y[label_to_idx[t]] = 1
    return y


train_df = train_df.copy()
train_df["filepath"] = train_df["image"].apply(lambda x: os.path.join(TRAIN_IMG_DIR, x))

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

tr_idx, va_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.1,
    random_state=SEED,
    shuffle=True,
)

feat_size = 48

X_tr = extract_features_fast(X_paths[tr_idx], size=feat_size, cache_tag="train")
X_va = extract_features_fast(X_paths[va_idx], size=feat_size, cache_tag="val")

Y_tr = Y[tr_idx]
Y_va = Y[va_idx]

scaler = StandardScaler(with_mean=True, with_std=True)
X_tr_s = scaler.fit_transform(X_tr)
X_va_s = scaler.transform(X_va)

models = []
for k in range(num_classes):
    lr = LogisticRegression(
        solver="lbfgs",
        C=0.35,
        max_iter=200,
        n_jobs=None,
        random_state=SEED,
        class_weight="balanced",
    )
    lr.fit(X_tr_s, Y_tr[:, k])
    models.append(lr)

va_proba = np.zeros((len(va_idx), num_classes), dtype=np.float32)
for k, lr in enumerate(models):
    va_proba[:, k] = lr.predict_proba(X_va_s)[:, 1].astype(np.float32)

threshold_grid = np.round(np.linspace(0.02, 0.90, 45), 3)
best_thr = 0.30
best_f1 = -1.0
for thr in threshold_grid:
    va_pred = (va_proba >= thr).astype(np.int8)
    f1 = f1_score(Y_va, va_pred, average="samples", zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = float(thr)

print("Validation F1 (samples; proxy for mean F1):", best_f1)
print("Chosen global threshold:", best_thr)



## === cell 4
test_files = sorted(
    [
        os.path.join(TEST_IMG_DIR, f)
        for f in os.listdir(TEST_IMG_DIR)
        if f.lower().endswith(".jpg")
    ]
)
print("Discovered test image files:", len(test_files))

X_test = extract_features_fast(test_files, size=feat_size, cache_tag="test")
X_test_s = scaler.transform(X_test)

test_proba = np.zeros((len(test_files), num_classes), dtype=np.float32)
for k, lr in enumerate(models):
    test_proba[:, k] = lr.predict_proba(X_test_s)[:, 1].astype(np.float32)

test_pred = (test_proba >= best_thr).astype(np.int8)

healthy_idx = label_to_idx.get("healthy", None)
complex_idx = label_to_idx.get("complex", None)

pred_labels = []
for i in range(len(test_files)):
    active = np.where(test_pred[i] == 1)[0].tolist()

    if complex_idx is not None and complex_idx in active:
        active = [complex_idx]

    if healthy_idx is not None and len(active) > 1 and healthy_idx in active:
        active = [j for j in active if j != healthy_idx]

    if len(active) == 0:
        if healthy_idx is not None:
            active = [healthy_idx]
        else:
            active = [int(np.argmax(test_proba[i]))]

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
