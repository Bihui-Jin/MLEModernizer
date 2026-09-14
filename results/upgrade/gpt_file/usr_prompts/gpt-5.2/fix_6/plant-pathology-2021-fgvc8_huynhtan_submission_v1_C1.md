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

0.36452

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.75058) has done: 'I remove `tensorflow_addons` (it fails to import in this environment due to protobuf incompatibility) and keep the rest of the TensorFlow pipeline intact. Since the referenced pre-trained model file does not exist in your input, I add a minimal “train-once then predict” fallback using a standard Keras ImageNet backbone so the notebook runs end-to-end and produces a valid `submission.csv`. I also fix path handling, ensure the labels are space-delimited strings, and make sure the submission uses only the basename for `image`. These changes are required for correctness and to yield a non-empty submission file.'
- What this solution (achieved 0.76653) has done: 'Your current code doesn’t yield a score mainly because it fail immediately in this environment: it tries to downgrade `protobuf` (to make `tensorflow_addons` happy), but TensorFlow 2.18 requires protobuf 5/6, so TensorFlow import break after the forced restart. I remove that protobuf-downgrade cell entirely so the notebook runs end-to-end. To move the score upward (without changing the overall “single-label softmax classifier” core logic), I also fix the label mapping bug: your `class_name` list contains combined-label strings inside one entry, which collapses multiple classes into one and discards most training data via filtering; instead we build `class_name` from the actual unique `train.csv` label strings. Finally, I keep the same model/training approach but increase epochs slightly (still lightweight with a frozen backbone) to improve F1 toward your target and ensure the submission aligns exactly to `sample_submission.csv`.'
- What this solution (achieved 0.36452) has done: 'Main bottlenecks are (1) slow Python-loop image loading/resize for ~15k train + ~1.5k val + ~3.7k test images and (2) very expensive multinomial LogisticRegression on ~20k-dimensional raw pixels. To stay within 600s without changing the algorithm, we (a) vectorize and parallelize image decoding/resizing with a thread pool (PIL releases the GIL during decode/resize), (b) preallocate the feature matrix to avoid huge temporary lists/stacks, and (c) cache extracted features to disk (memmap) so reruns and later cells don’t re-decode images. Core model, features (same pixels/normalization), split, scaler, and LR hyperparameters remain identical; only the data loading/extraction implementation is accelerated.'

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

print(
    "Using non-TensorFlow fallback due to TF/protobuf incompatibility in this environment."
)



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

class_name = sorted(train_df["labels"].unique().tolist())
label_to_idx = {s: i for i, s in enumerate(class_name)}
idx_to_label = {i: s for s, i in label_to_idx.items()}
num_classes = len(class_name)

print("Num classes from train.csv:", num_classes)
print("First 10 classes:", class_name[:10])



## === cell 2
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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


train_df = train_df.copy()
train_df["y"] = train_df["labels"].map(label_to_idx).astype("int32")
train_df["filepath"] = train_df["image"].apply(lambda x: os.path.join(TRAIN_IMG_DIR, x))

exists_mask = np.fromiter(
    (os.path.exists(p) for p in train_df["filepath"].values),
    dtype=bool,
    count=len(train_df),
)
train_df = train_df.loc[exists_mask].reset_index(drop=True)

print("Train rows after file-exists filter:", len(train_df))
print("Test rows (sample_submission):", len(sample_df))



## === cell 3
X_paths = train_df["filepath"].values
y = train_df["y"].values

tr_paths, va_paths, y_tr, y_va = train_test_split(
    X_paths, y, test_size=0.1, random_state=SEED, shuffle=True, stratify=y
)

feat_size = 48

X_tr = extract_features_fast(tr_paths, size=feat_size, cache_tag="train")
X_va = extract_features_fast(va_paths, size=feat_size, cache_tag="val")

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "lr",
            LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                C=0.35,
                max_iter=200,
                n_jobs=None,
                random_state=SEED,
            ),
        ),
    ]
)

clf.fit(X_tr, y_tr)

va_pred = clf.predict(X_va)
va_acc = (va_pred == y_va).mean()
print("Validation accuracy (proxy):", va_acc)



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

test_pred_idx = clf.predict(X_test).astype(int)
test_pred_labels = [idx_to_label[i] for i in test_pred_idx]

pred_df = pd.DataFrame(
    {"image": [os.path.basename(p) for p in test_files], "labels": test_pred_labels}
)

print(pred_df.head())



## === cell 5
fallback_label = (
    "healthy" if "healthy" in label_to_idx else train_df["labels"].mode().iloc[0]
)

submission = sample_df[["image"]].merge(pred_df, on="image", how="left")
submission["labels"] = submission["labels"].fillna(fallback_label).astype(str)

submission = submission[["image", "labels"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
print("Unique predicted labels (count):", submission["labels"].nunique())
