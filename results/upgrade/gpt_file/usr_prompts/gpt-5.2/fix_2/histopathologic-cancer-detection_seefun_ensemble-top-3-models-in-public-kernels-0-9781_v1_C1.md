# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

# 2. Python version

3.7

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
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score



## === cell 1

CANDIDATE_ROOTS = [
    "/kaggle/input/histopathologic-cancer-detection",
    "/kaggle/data/histopathologic-cancer-detection",
    "../input/histopathologic-cancer-detection",
    "../kaggle/input/histopathologic-cancer-detection",
    "/kaggle/input",  # fallback; we'll search under it
    "../input",  # fallback; we'll search under it
]


def find_file(patterns, roots):
    for r in roots:
        if not os.path.exists(r):
            continue
        for pat in patterns:
            p1 = os.path.join(r, pat)
            if os.path.exists(p1):
                return p1
            hits = glob.glob(os.path.join(r, "**", pat), recursive=True)
            if hits:
                hits = sorted(hits, key=len)
                return hits[0]
    return None


train_csv = find_file(["train_labels.csv"], CANDIDATE_ROOTS)
sample_csv = find_file(["sample_submission.csv"], CANDIDATE_ROOTS)


def find_image_dir(dir_name):
    for r in CANDIDATE_ROOTS:
        if not os.path.exists(r):
            continue
        direct = os.path.join(r, dir_name)
        if os.path.isdir(direct) and glob.glob(os.path.join(direct, "*.tif")):
            return direct
        hits = [
            p
            for p in glob.glob(os.path.join(r, "**", dir_name), recursive=True)
            if os.path.isdir(p)
        ]
        hits = [p for p in hits if glob.glob(os.path.join(p, "*.tif"))]
        if hits:
            hits = sorted(hits, key=len)
            return hits[0]
    return None


train_dir = find_image_dir("train")
test_dir = find_image_dir("test")

print("train_csv:", train_csv)
print("sample_csv:", sample_csv)
print("train_dir:", train_dir)
print("test_dir:", test_dir)

if train_csv is None or sample_csv is None or train_dir is None or test_dir is None:
    raise FileNotFoundError(
        "Could not locate required competition files. "
        "Expected train_labels.csv, sample_submission.csv, and train/test image directories."
    )



## === cell 2
train_df = pd.read_csv(train_csv)
sub_df = pd.read_csv(sample_csv)

assert set(train_df.columns) >= {"id", "label"}
assert set(sub_df.columns) >= {"id", "label"}

print(train_df.shape, sub_df.shape)
train_df.head()



## === cell 3


def imread_any(path):
    try:
        import imageio.v2 as imageio

        return imageio.imread(path)
    except Exception:
        import matplotlib.pyplot as plt

        return plt.imread(path)


def extract_center_features(img):
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    h, w = img.shape[:2]
    y0 = (h - 32) // 2
    x0 = (w - 32) // 2
    patch = img[y0 : y0 + 32, x0 : x0 + 32]

    patch = patch.astype(np.float32)
    if patch.max() > 1.5:  # uint8-like
        patch = patch / 255.0

    ch_mean = patch.reshape(-1, 3).mean(axis=0)
    ch_std = patch.reshape(-1, 3).std(axis=0)
    overall_mean = float(patch.mean())
    overall_std = float(patch.std())
    eps = 1e-6
    rg = float(ch_mean[0] / (ch_mean[1] + eps))
    rb = float(ch_mean[0] / (ch_mean[2] + eps))
    gb = float(ch_mean[1] / (ch_mean[2] + eps))
    return np.array(
        [*ch_mean, *ch_std, overall_mean, overall_std, rg, rb, gb], dtype=np.float32
    )


def build_feature_matrix(ids, img_dir, verbose_every=5000):
    X = np.zeros((len(ids), 11), dtype=np.float32)
    missing = 0
    for i, img_id in enumerate(ids):
        fp = os.path.join(img_dir, f"{img_id}.tif")
        if not os.path.exists(fp):
            missing += 1
            continue
        img = imread_any(fp)
        X[i] = extract_center_features(img)
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"Processed {i+1}/{len(ids)} images...")
    if missing:
        print(f"Warning: {missing} images were missing in {img_dir}.")
    return X




## === cell 4
train_ids = train_df["id"].values
y = train_df["label"].values.astype(np.int32)

X = build_feature_matrix(train_ids, train_dir, verbose_every=20000)
print("X shape:", X.shape, "y shape:", y.shape)



## === cell 5
X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

clf = LogisticRegression(
    solver="lbfgs",
    max_iter=200,
    class_weight="balanced",
    n_jobs=None,
)
clf.fit(X_tr, y_tr)

va_pred = clf.predict_proba(X_va)[:, 1]
va_auc = roc_auc_score(y_va, va_pred)
print("Validation AUC:", va_auc)



## === cell 6
clf.fit(X, y)

test_ids = sub_df["id"].values
X_test = build_feature_matrix(test_ids, test_dir, verbose_every=20000)

test_pred = clf.predict_proba(X_test)[:, 1].astype(np.float32)

submission = pd.DataFrame({"id": test_ids, "label": test_pred})
assert submission.shape[0] == sub_df.shape[0]
assert submission["id"].iloc[0] == sub_df["id"].iloc[0]

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
submission.head()
