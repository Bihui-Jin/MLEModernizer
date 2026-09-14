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
import pandas as pd
import numpy as np
from PIL import Image  # moved import here to avoid repeated imports




## === cell 1
def locate(rel_path):
    candidates = [
        f"../input/plant-pathology-2020-fgvc7/{rel_path}",
        f"../input/plant-pathology/{rel_path}",
        f"../input/{rel_path}",
        f"./{rel_path}",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Unable to find {rel_path}")


print("Input folder contents (first 10):", os.listdir(locate(""))[:10])




## === cell 2
train_path = locate("train.csv")
train_df = pd.read_csv(train_path)

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]




## === cell 3
test_path = locate("test.csv")
test_df = pd.read_csv(test_path)




## === cell 4
def id_to_num(img_id):
    import re

    nums = re.findall(r"\d+", str(img_id))
    return int(nums[0]) if nums else 0


def id_features(img_id):
    """Return simple numeric features derived from the image_id."""
    s = str(img_id)
    num = id_to_num(s)
    length = len(s)
    digit_sum = sum(int(d) for d in s if d.isdigit())
    return [num, length, digit_sum]


def extract_image_features(img_path, bins=16):
    """
    Load an image once and compute:
        - per‑channel mean (3)
        - per‑channel std  (3)
        - color histogram (bins * 3)
        - file size (1)
        - width & height (2)
        - aspect ratio = width / height (1)
    Returns a 34‑element array (truncated/padded as before).
    """
    try:
        size_kb = os.path.getsize(img_path) / 1024.0
        img = Image.open(img_path).convert("RGB")
        width, height = img.width, img.height
        arr_uint8 = np.asarray(img, dtype=np.uint8)
        arr = arr_uint8.astype(np.float32) / 255.0

        mean = arr.mean(axis=(0, 1))
        std = arr.std(axis=(0, 1))

        hist = []
        for ch in range(3):
            h, _ = np.histogram(
                arr_uint8[:, :, ch], bins=bins, range=(0, 256), density=True
            )
            hist.append(h)
        hist = np.concatenate(hist)

        aspect_ratio = width / height if height != 0 else 0.0

        feat = np.concatenate(
            [mean, std, hist, [size_kb], [float(width), float(height), aspect_ratio]]
        )
        if feat.shape[0] > 34:
            feat = feat[:34]
        elif feat.shape[0] < 34:
            feat = np.pad(feat, (0, 34 - feat.shape[0]), constant_values=np.nan)
        return feat.astype(np.float32)
    except Exception:
        return np.full(34, np.nan, dtype=np.float32)


def _row_features(args):
    """Helper for parallel execution: returns (idx, feature_row)."""
    idx, img_id, img_dir = args
    id_feat = np.array(id_features(img_id), dtype=np.float32)
    img_path = os.path.join(img_dir, f"{img_id}.jpg")
    img_feat = extract_image_features(img_path)
    return idx, np.concatenate([id_feat, img_feat])


def build_features(df):
    """
    Build a numeric feature matrix for a dataframe containing an `image_id` column.
    Features (total 37):
        - ID‑based numeric descriptors (3)
        - Image features (34) defined in `extract_image_features`.
    Parallelised over available CPU cores.
    """
    import concurrent.futures

    img_dir = locate("images")
    n = len(df)
    X = np.empty((n, 37), dtype=np.float32)

    args_iter = ((i, img_id, img_dir) for i, img_id in enumerate(df["image_id"]))

    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for idx, row in executor.map(_row_features, args_iter, chunksize=32):
            X[idx, :] = row

    return X




## === cell 5
X_train = build_features(train_df)
X_test = build_features(test_df)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)




## === cell 6

from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
import concurrent.futures

hgb_params = dict(max_iter=4000, learning_rate=0.02, random_state=42)
gbc_params = dict(n_estimators=4000, learning_rate=0.03, max_depth=4, random_state=42)


def _train_column(col):
    """Fit HGB and GBC for a single target column and return them."""
    y = train_df[col].values
    n_samples = len(y)
    pos = y.sum()
    neg = n_samples - pos
    if pos == 0 or neg == 0:
        sample_weight = None
    else:
        w_pos = n_samples / (2.0 * pos)
        w_neg = n_samples / (2.0 * neg)
        sample_weight = np.where(y == 1, w_pos, w_neg)

    hgb = HistGradientBoostingClassifier(**hgb_params)
    gbc = GradientBoostingClassifier(**gbc_params)

    hgb.fit(X_train, y, sample_weight=sample_weight)
    gbc.fit(X_train, y, sample_weight=sample_weight)

    return (col, hgb, gbc)


models_hgb = {}
models_gbc = {}

with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    futures = [executor.submit(_train_column, col) for col in target_cols]
    for fut in concurrent.futures.as_completed(futures):
        col, hgb, gbc = fut.result()
        models_hgb[col] = hgb
        models_gbc[col] = gbc

preds = {}
for col in target_cols:
    proba_hgb = models_hgb[col].predict_proba(X_test)[:, 1]
    proba_gbc = models_gbc[col].predict_proba(X_test)[:, 1]
    preds[col] = (proba_hgb + proba_gbc) / 2.0

for col in target_cols:
    baseline = train_df[col].mean()
    preds[col] = 0.98 * preds[col] + 0.02 * baseline

submission = pd.DataFrame(
    {
        "image_id": test_df["image_id"],
        "healthy": preds["healthy"],
        "multiple_diseases": preds["multiple_diseases"],
        "rust": preds["rust"],
        "scab": preds["scab"],
    }
)

print("Submission preview:")
print(submission.head())




## === cell 7
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, rows: {len(submission)}")
