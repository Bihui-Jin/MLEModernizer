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
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression  # added for ensembling
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from joblib import Parallel, delayed  # parallelize image feature extraction




## === cell 1
base_dir = "../input/plant-pathology-2020-fgvc7"
if not os.path.isdir(base_dir):
    base_dir = "../input"
print("Using base directory:", base_dir)

train_path = os.path.join(base_dir, "train.csv")
train_df = pd.read_csv(train_path)

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]


def extract_num(img_id):
    import re

    m = re.search(r"\d+", str(img_id))
    return int(m.group()) if m else 0


def _process_image(img_id, images_root):
    img_path = os.path.join(images_root, f"{img_id}.jpg")
    try:
        size = os.path.getsize(img_path)
        with open(img_path, "rb") as f:
            data = f.read()
        if data:
            bytes_arr = np.frombuffer(data, dtype=np.uint8)
            mean = float(bytes_arr.mean())
            var = float(bytes_arr.var())
            mn = int(bytes_arr.min())
            mx = int(bytes_arr.max())
            hist = np.bincount(bytes_arr, minlength=256).astype(float)
            total = hist.sum()
            if total > 0:
                prob = hist / total
                hist_norm = prob  # already normalized
            else:
                prob = hist
                hist_norm = hist
            entropy = -float(np.sum(prob * np.log2(prob + 1e-12)))
            return (size, mean, var, mn, mx, entropy, hist_norm)
        else:
            return (0, 0.0, 0.0, 0, 0, 0.0, np.zeros(256, dtype=float))
    except Exception:
        return (0, 0.0, 0.0, 0, 0, 0.0, np.zeros(256, dtype=float))


def image_features(df, images_root):
    """
    Compute simple image‑based numeric features without external image libraries:
    - file size in bytes
    - mean byte value of the raw JPEG file
    - byte variance, min and max
    - entropy of byte distribution (captures information richness)
    - normalized 256‑bin byte histogram (adds richer visual signal)
    Missing or unreadable images are safely handled with fallback zeros.
    """
    img_ids = df["image_id"].tolist()
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(_process_image)(img_id, images_root) for img_id in img_ids
    )
    sizes, byte_means, byte_vars, byte_mins, byte_maxs, entropies, histograms = zip(
        *results
    )

    stats_df = pd.DataFrame(
        {
            "file_size": sizes,
            "file_byte_mean": byte_means,
            "file_byte_variance": byte_vars,
            "file_byte_min": byte_mins,
            "file_byte_max": byte_maxs,
            "file_entropy": entropies,
        }
    )
    hist_cols = [f"byte_hist_{i}" for i in range(256)]
    hist_df = pd.DataFrame(np.vstack(histograms), columns=hist_cols)

    return pd.concat([stats_df, hist_df], axis=1)


def engineer_features(df):
    """Create a richer set of numeric features from image_id and basic image stats."""
    img_num = df["image_id"].apply(extract_num)
    df_feat = pd.DataFrame(
        {
            "image_num": img_num,
            "image_num_mod10": img_num % 10,
            "id_len": df["image_id"].astype(str).apply(len),
            "is_train_prefix": df["image_id"]
            .astype(str)
            .str.startswith("Train")
            .astype(int),
            "is_test_prefix": df["image_id"]
            .astype(str)
            .str.startswith("Test")
            .astype(int),
            "digit_sum": img_num.astype(str).apply(
                lambda s: sum(int(ch) for ch in s) if s.isdigit() else 0
            ),
            "first_digit": img_num.astype(str).apply(lambda s: int(s[0]) if s else 0),
            "last_digit": img_num.astype(str).apply(lambda s: int(s[-1]) if s else 0),
            "num_div_10": img_num // 10,
            "num_div_100": img_num // 100,
            "log_image_num": np.log1p(img_num),
            "sqrt_image_num": np.sqrt(img_num),
            "square_image_num": img_num**2,
        }
    )
    img_stats = image_features(df, os.path.join(base_dir, "images"))
    df_feat = pd.concat([df_feat, img_stats], axis=1)
    return df_feat


X_full = engineer_features(train_df)
print("Engineered feature sample:")
print(X_full.head())




## === cell 2
models = {}
ensemble_weights = {}


def _train_target(col):
    gbc = GradientBoostingClassifier(
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.9,
        random_state=42,
    )
    gbc.fit(X_full, train_df[col])

    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        ),
    )
    lr.fit(X_full, train_df[col])

    if len(train_df[col].unique()) > 1:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_full,
            train_df[col],
            test_size=0.2,
            random_state=42,
            stratify=train_df[col],
        )
        gbc_tmp = GradientBoostingClassifier(
            n_estimators=2000,
            learning_rate=0.02,
            max_depth=6,
            subsample=0.9,
            random_state=42,
        )
        gbc_tmp.fit(X_tr, y_tr)

        lr_tmp = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
            ),
        )
        lr_tmp.fit(X_tr, y_tr)

        best_w = 0.5
        best_auc = 0.0
        for w in np.linspace(0, 1, 101):
            pred = (
                w * gbc_tmp.predict_proba(X_val)[:, 1]
                + (1 - w) * lr_tmp.predict_proba(X_val)[:, 1]
            )
            auc = roc_auc_score(y_val, pred)
            if auc > best_auc:
                best_auc = auc
                best_w = w
        print(
            f"Validation AUC for {col} (best gbc weight={best_w:.2f}): {best_auc:.4f}"
        )
        return col, {"gbc": gbc, "lr": lr}, best_w
    else:
        return col, {"gbc": gbc, "lr": lr}, 0.7


results = Parallel(n_jobs=-1, prefer="threads")(
    delayed(_train_target)(col) for col in target_cols
)

for col, model_dict, best_w in results:
    models[col] = model_dict
    ensemble_weights[col] = best_w




## === cell 3
test_path = os.path.join(base_dir, "test.csv")
test_df = pd.read_csv(test_path)
X_test = engineer_features(test_df)




## === cell 4
sub_path = os.path.join(base_dir, "sample_submission.csv")
sub = pd.read_csv(sub_path)

for col in target_cols:
    gbc = models[col]["gbc"]
    lr = models[col]["lr"]
    w = ensemble_weights.get(col, 0.7)  # weight for gbc
    probs = (
        w * gbc.predict_proba(X_test)[:, 1] + (1 - w) * lr.predict_proba(X_test)[:, 1]
    )
    sub[col] = probs

print("Submission preview:")
print(sub.head())




## === cell 5
output_path = "submission.csv"
sub.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}")
