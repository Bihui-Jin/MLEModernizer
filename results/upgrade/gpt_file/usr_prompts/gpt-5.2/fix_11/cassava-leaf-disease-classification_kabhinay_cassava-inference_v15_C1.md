# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

2.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
from __future__ import print_function
import os
import json
import numpy as np
import pandas as pd

np.random.seed(123)

print("Python:", os.sys.version)

BASE_INPUT_1 = "../input/cassava-leaf-disease-classification"
BASE_INPUT_2 = "/kaggle/input/cassava-leaf-disease-classification"

if os.path.isdir(BASE_INPUT_1):
    BASE_INPUT = BASE_INPUT_1
elif os.path.isdir(BASE_INPUT_2):
    BASE_INPUT = BASE_INPUT_2
else:
    BASE_INPUT = "/kaggle/data/cassava-leaf-disease-classification"

TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
SAMPLE_SUB = os.path.join(BASE_INPUT, "sample_submission.csv")
TEST_DIR = os.path.join(BASE_INPUT, "test_images")
TRAIN_IMG_DIR = os.path.join(BASE_INPUT, "train_images")

print("Using BASE_INPUT:", BASE_INPUT)
print("Exists train.csv:", os.path.exists(TRAIN_CSV))
print("Exists sample_submission.csv:", os.path.exists(SAMPLE_SUB))
print("Exists test_images dir:", os.path.isdir(TEST_DIR))
print("Exists train_images dir:", os.path.isdir(TRAIN_IMG_DIR))



## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
if "label" not in train_df.columns:
    raise ValueError("train.csv must contain 'label' column")

label_counts = train_df["label"].value_counts()
majority_label = int(label_counts.idxmax())
print("Train label distribution:\n", label_counts.sort_index())
print("Majority label:", majority_label)

sample_df = pd.read_csv(SAMPLE_SUB)
if not set(["image_id", "label"]).issubset(sample_df.columns):
    raise ValueError("sample_submission.csv must contain columns: image_id, label")
print("Sample submission rows:", len(sample_df))




## === cell 2
def _safe_imports():
    try:
        from PIL import Image
    except Exception as e:
        Image = None
        print(
            "WARNING: PIL import failed; will fall back to majority label. Error:",
            repr(e),
        )

    try:
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import accuracy_score

        sklearn_ok = True
    except Exception as e:
        sklearn_ok = False
        print(
            "WARNING: sklearn import failed; will fall back to majority label. Error:",
            repr(e),
        )

    return Image, sklearn_ok


Image, SKLEARN_OK = _safe_imports()


def extract_features(image_path, image_size=160):
    """
    Global deterministic image statistics (preserved core logic):
    - RGB mean/std/min/max
    - grayscale mean/std
    - HSV mean/std
    - simple gradient/edge proxies
    """
    img = Image.open(image_path).convert("RGB")
    if image_size is not None:
        img = img.resize((image_size, image_size))

    arr = np.asarray(img, dtype=np.float32) / 255.0  # H,W,3
    H, W = arr.shape[0], arr.shape[1]

    flat = arr.reshape(-1, 3)
    mean_rgb = flat.mean(axis=0)  # (3,)
    std_rgb = flat.std(axis=0)  # (3,)
    min_rgb = flat.min(axis=0)  # (3,)
    max_rgb = flat.max(axis=0)  # (3,)

    gray = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
    mean_g = float(gray.mean())
    std_g = float(gray.std())

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    cmax = np.maximum(r, np.maximum(g, b))
    cmin = np.minimum(r, np.minimum(g, b))
    delta = cmax - cmin

    eps = 1e-6
    delta_safe = np.where(delta < eps, 1.0, delta)

    h = np.zeros_like(cmax, dtype=np.float32)
    mask = delta >= eps

    mr = mask & (cmax == r)
    h[mr] = ((g[mr] - b[mr]) / delta_safe[mr]) % 6.0
    mg = mask & (cmax == g)
    h[mg] = ((b[mg] - r[mg]) / delta_safe[mg]) + 2.0
    mb = mask & (cmax == b)
    h[mb] = ((r[mb] - g[mb]) / delta_safe[mb]) + 4.0
    h = (h / 6.0).astype(np.float32)

    s = np.zeros_like(cmax, dtype=np.float32)
    s[mask] = (delta[mask] / (cmax[mask] + eps)).astype(np.float32)
    v = cmax.astype(np.float32)

    mean_hsv = np.array([h.mean(), s.mean(), v.mean()], dtype=np.float32)
    std_hsv = np.array([h.std(), s.std(), v.std()], dtype=np.float32)

    dx = gray[:, 1:] - gray[:, :-1]  # H, W-1
    dy = gray[1:, :] - gray[:-1, :]  # H-1, W
    edge_energy = float((dx * dx).mean() + (dy * dy).mean())
    if H > 1 and W > 1:
        gx = gray[:, 1:] - gray[:, :-1]
        gy = gray[1:, :] - gray[:-1, :]
        gmag = np.sqrt(gx[:-1, :] * gx[:-1, :] + gy[:, :-1] * gy[:, :-1])
        gmag_mean = float(gmag.mean())
        gmag_std = float(gmag.std())
    else:
        gmag_mean, gmag_std = 0.0, 0.0

    feats = np.concatenate(
        [
            mean_rgb,  # 3
            std_rgb,  # 3
            min_rgb,  # 3
            max_rgb,  # 3
            np.array([mean_g, std_g], dtype=np.float32),  # 2
            mean_hsv,  # 3
            std_hsv,  # 3
            np.array([edge_energy, gmag_mean, gmag_std], dtype=np.float32),  # 3
        ],
        axis=0,
    )
    return feats.astype(np.float32)


def build_train_matrix(df, img_dir, limit=None):
    feats = []
    labels = []
    used_ids = []
    n = len(df) if limit is None else min(len(df), int(limit))
    for i in range(n):
        image_id = str(df.iloc[i]["image_id"])
        label = int(df.iloc[i]["label"])
        path = os.path.join(img_dir, image_id)
        if not os.path.exists(path):
            continue
        feats.append(extract_features(path))
        labels.append(label)
        used_ids.append(image_id)
        if (i + 1) % 2000 == 0:
            print("Processed train images:", i + 1, "/", n)
    d = int(feats[0].shape[0]) if len(feats) else 0
    X = np.vstack(feats) if len(feats) else np.zeros((0, d), dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return X, y, used_ids


def build_test_matrix(image_ids, img_dir, d_expected):
    feats = []
    ok_ids = []
    for i, image_id in enumerate(image_ids):
        image_id = str(image_id)
        path = os.path.join(img_dir, image_id)
        if not os.path.exists(path):
            feats.append(np.zeros((d_expected,), dtype=np.float32))
            ok_ids.append(image_id)
            continue
        feats.append(extract_features(path))
        ok_ids.append(image_id)
        if (i + 1) % 500 == 0:
            print("Processed test images:", i + 1, "/", len(image_ids))
    X = np.vstack(feats) if len(feats) else np.zeros((0, d_expected), dtype=np.float32)
    return X, ok_ids


def _infer_source_group(image_id):
    """
    Minimal, score-relevant split robustness tweak:
    Cassava images often come from different capture sources; mixing sources unevenly in a random
    split can select a worse hyperparameter C. We infer a coarse source group from filename patterns.
    If no known pattern exists, we return None and fall back to the original split.
    """
    s = str(image_id)
    if s.startswith("train_"):
        return "trainprefix"
    if s.startswith("test_"):
        return "testprefix"
    digits = "".join([c for c in s if c.isdigit()])
    if len(digits) >= 2:
        return digits[:2]
    return None




## === cell 3

sub_df = None  # will be set in all branches

use_model = (
    (Image is not None)
    and SKLEARN_OK
    and os.path.isdir(TRAIN_IMG_DIR)
    and os.path.isdir(TEST_DIR)
)

if not use_model:
    print(
        "Falling back to majority-label baseline due to missing dependencies or image folders."
    )
    sub_df = sample_df.copy()
    sub_df["label"] = np.int64(majority_label)
else:
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import accuracy_score

        X, y, used_ids = build_train_matrix(train_df, TRAIN_IMG_DIR, limit=None)
        print("Feature matrix:", X.shape, "Labels:", y.shape)

        groups = np.array([_infer_source_group(iid) for iid in used_ids], dtype=object)

        use_group_strat = False
        if not np.all(pd.isnull(groups)):
            strat_key = np.array(
                [str(int(lbl)) + "_" + str(grp) for lbl, grp in zip(y, groups)]
            )
            uniq, counts = np.unique(strat_key, return_counts=True)
            if counts.min() >= 2:
                use_group_strat = True
            else:
                print(
                    "NOTE: label_x_group strata has min count",
                    int(counts.min()),
                    "-> falling back to stratify=y",
                )

        if use_group_strat:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X, y, test_size=0.15, random_state=123, stratify=strat_key
            )
            print("Split: stratify=label_x_group (robust)")
        else:
            X_tr, X_va, y_tr, y_va = train_test_split(
                X, y, test_size=0.15, random_state=123, stratify=y
            )
            print("Split: stratify=y (fallback)")

        Cs = [0.05, 0.1, 0.2, 0.3, 0.6, 1.0, 2.0, 3.0, 6.0, 10.0]

        best = None
        for C in Cs:
            clf = Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    (
                        "lr",
                        LogisticRegression(
                            C=C,
                            solver="lbfgs",
                            multi_class="multinomial",
                            class_weight=None,
                            max_iter=400,
                            n_jobs=1,
                            random_state=123,
                        ),
                    ),
                ]
            )
            clf.fit(X_tr, y_tr)
            va_pred = clf.predict(X_va)
            va_acc = accuracy_score(y_va, va_pred)
            print("C=", C, "val_acc=", float(va_acc))
            if (best is None) or (va_acc > best[0]):
                best = (va_acc, C, clf)

        best_acc, best_C, best_clf = best
        print("Selected C:", best_C, "with val_acc:", float(best_acc))

        final_clf = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "lr",
                    LogisticRegression(
                        C=best_C,
                        solver="lbfgs",
                        multi_class="multinomial",
                        class_weight=None,
                        max_iter=500,
                        n_jobs=1,
                        random_state=123,
                    ),
                ),
            ]
        )
        final_clf.fit(X, y)

        test_image_ids = sample_df["image_id"].astype(str).tolist()
        d_expected = int(X.shape[1]) if X.ndim == 2 else 0
        X_test, ok_ids = build_test_matrix(
            test_image_ids, TEST_DIR, d_expected=d_expected
        )
        preds = final_clf.predict(X_test).astype(np.int64)

        sub_df = pd.DataFrame({"image_id": ok_ids, "label": preds})
    except Exception as e:
        print(
            "WARNING: model path failed; falling back to majority label. Error:",
            repr(e),
        )
        sub_df = sample_df.copy()
        sub_df["label"] = np.int64(majority_label)



## === cell 4
sub_df["image_id"] = sub_df["image_id"].astype(str)
sub_df["label"] = sub_df["label"].astype(np.int64)

sub_df = sample_df[["image_id"]].merge(sub_df, on="image_id", how="left")
sub_df["label"] = sub_df["label"].fillna(majority_label).astype(np.int64)

out_path = "/kaggle/working/submission.csv"
sub_df.to_csv(out_path, index=False)

print("Wrote submission to:", out_path)
print(sub_df.head())
print("Rows:", len(sub_df), "Cols:", list(sub_df.columns))

if os.path.isdir(TEST_DIR):
    test_files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")])
    print("Found test jpgs:", len(test_files))
    if len(test_files) != len(sub_df):
        print(
            "WARNING: test image count != submission rows. (Submission follows sample_submission.csv.)"
        )
