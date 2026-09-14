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

# 5. Target score

0.870806890299184

# 6. Current score

0.31876

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The crash is happening before anything else because TensorFlow can’t import due to a protobuf binary mismatch; this prevents `tf/keras` from being defined and causes the downstream `NameError`s. The minimal reliable fix in this environment is to remove the TensorFlow dependency entirely and switch to a lightweight, deterministic baseline that still produces a valid `submission.csv` in the required format. To get a reasonable (and stable) accuracy without training a model, we predict the most frequent class from `train.csv` for every test image (a common safe baseline for this competition). The rest of the changes ensure paths resolve correctly and that the submission has exactly `image_id,label` with 2676 rows.'
- What this solution (achieved 0.61173) has done: 'Your current approach predicts the single most frequent class, which lands around ~0.61 accuracy; to move toward the 0.87 target without changing the “no-TF, no-training” core idea, we can make predictions depend on each image using simple, deterministic image statistics. The smallest meaningful upgrade is a lightweight per-image feature extractor (mean/std in RGB) plus a multinomial logistic regression trained on a train/validation split to pick a reasonable regularization strength. This keeps the pipeline fast (<600s), uses only common Python packages, and still writes a valid `submission.csv` with the required schema and row alignment to `sample_submission.csv`. If `sklearn` is unavailable in your environment, the code automatically falls back to your majority-class baseline to remain runnable.'
- What this solution (achieved 0.31876) has done: 'Your current pipeline is already a simple, dependency-light image-statistics + multinomial logistic regression approach; the fastest way to move accuracy up (toward 0.87) without changing the core modeling idea is to (1) extract slightly richer but still lightweight deterministic features, and (2) set `class_weight="balanced"` to counter the strong class imbalance in Cassava (which often hurts plain multinomial LR and can trap you near the majority-class accuracy). I keep the same training loop, same model family (LogisticRegression), same basic feature extraction approach (global image stats), and the same submission alignment logic. These two tweaks are minimal, stable, and commonly yield a meaningful lift over the ~0.61 region while staying well within the 600s runtime budget.'

# 9. Code solution

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


def extract_features(image_path, image_size=128):
    """
    Minimal score-improving tweak (keeps same core idea: global image stats):
    - slightly higher resize (128 vs 96) for more stable global statistics
    - add per-channel min/max (6) in addition to mean/std (6)
    - keep mean/std grayscale (2)
    Total features: 14
    """
    img = Image.open(image_path).convert("RGB")
    if image_size is not None:
        img = img.resize((image_size, image_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0  # H,W,3

    flat = arr.reshape(-1, 3)
    mean_rgb = flat.mean(axis=0)  # (3,)
    std_rgb = flat.std(axis=0)  # (3,)
    min_rgb = flat.min(axis=0)  # (3,)
    max_rgb = flat.max(axis=0)  # (3,)

    gray = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
    mean_g = float(gray.mean())
    std_g = float(gray.std())

    feats = np.concatenate(
        [
            mean_rgb,
            std_rgb,
            min_rgb,
            max_rgb,
            np.array([mean_g, std_g], dtype=np.float32),
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
    X = np.vstack(feats) if len(feats) else np.zeros((0, 14), dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return X, y, used_ids


def build_test_matrix(image_ids, img_dir):
    feats = []
    ok_ids = []
    for i, image_id in enumerate(image_ids):
        image_id = str(image_id)
        path = os.path.join(img_dir, image_id)
        if not os.path.exists(path):
            feats.append(np.zeros((14,), dtype=np.float32))
            ok_ids.append(image_id)
            continue
        feats.append(extract_features(path))
        ok_ids.append(image_id)
        if (i + 1) % 500 == 0:
            print("Processed test images:", i + 1, "/", len(image_ids))
    X = np.vstack(feats) if len(feats) else np.zeros((0, 14), dtype=np.float32)
    return X, ok_ids




## === cell 3
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
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score

    X, y, used_ids = build_train_matrix(train_df, TRAIN_IMG_DIR, limit=None)
    print("Feature matrix:", X.shape, "Labels:", y.shape)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.15, random_state=123, stratify=y
    )

    Cs = [0.3, 1.0, 3.0]
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
                        class_weight="balanced",
                        max_iter=250,
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
                    class_weight="balanced",
                    max_iter=300,
                    n_jobs=1,
                    random_state=123,
                ),
            ),
        ]
    )
    final_clf.fit(X, y)

    test_image_ids = sample_df["image_id"].astype(str).tolist()
    X_test, ok_ids = build_test_matrix(test_image_ids, TEST_DIR)
    preds = final_clf.predict(X_test).astype(np.int64)

    sub_df = pd.DataFrame({"image_id": ok_ids, "label": preds})




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
