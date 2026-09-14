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

# 5. Target score

0.9697366773573868

# 6. Current score

0.59012

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I replaced the failing loading of external CSVs with a safe fallback that computes the mean label values from the provided training data and fills the sample submission with those averages. This fixes the FileNotFoundError and NameError, ensures a proper `submission.csv` is written, and keeps the original “averaging” idea without altering the core modeling approach.'
- What this solution (achieved 0.54806) has done: 'I add a lightweight numeric feature extracted from the image filenames and fit a simple LogisticRegression for each target column. This keeps the pipeline small, avoids heavy image processing, and replaces the naive column‑mean filler with a model that can capture any pattern in the ID numbers, which should move the ROC‑AUC score upward toward the target while preserving the overall structure of the script. The code now writes a proper `submission.csv` using the model predictions.'
- What this solution (achieved 0.55413) has done: 'I added a lightweight feature‑engineering step that creates several numeric descriptors from the image filename (raw number, its modulo 10, length of the id string and a binary flag for the “Train” prefix). Both the training and test data now use these features, which gives the logistic models a richer signal while keeping the original per‑column LogisticRegression approach unchanged. This small augmentation is expected to raise the validation ROC‑AUC and thus move the Kaggle score nearer to the target.'
- What this solution (achieved 0.50986) has done: 'I expand the feature engineering to extract richer numeric signals from the image IDs (digit sum, first/last digit, higher‑order divisions, etc.) and increase the logistic regression iteration limit so the model can fit these extra features more fully. These minimal tweaks keep the original per‑column LogisticRegression setup while giving the models stronger inputs, which should raise the validation AUC and move the Kaggle score closer to the target.'
- What this solution (achieved 0.50312) has done: 'I add a few richer numeric transformations (log, sqrt, square) to the engineered features and replace the per‑column LogisticRegression with a GradientBoostingClassifier, which can capture non‑linear patterns in these simple ID‑based features while keeping the overall pipeline unchanged. This stronger model should raise the validation AUC and move the Kaggle score closer to the target. The script still writes a proper `submission.csv` using the same column names.'
- What this solution (achieved 0.47908) has done: 'I increased the model capacity and added a tiny extra identifier feature so the GradientBoosting classifiers can capture more patterns from the image IDs. The new `engineer_features` function now includes a binary flag for the “Test” prefix, and the classifier parameters are enlarged (more trees, deeper depth, lower learning rate, and subsampling). These modest adjustments should raise the validation AUC and move the Kaggle score closer to the target while keeping the original pipeline intact.'
- What this solution (achieved 0.51246) has done: 'Implemented lightweight image‑based features (file size and average pixel intensity) to give the model real visual signal, and modestly increased the GradientBoosting capacity (more trees and deeper depth). These changes keep the original pipeline intact while providing richer data, which should raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.59258) has done: 'I add richer image‑file based numeric features (binary‑byte mean and entropy) that work without Pillow, and slightly increase the GradientBoosting capacity so the model can exploit these new signals. These changes keep the overall pipeline and per‑column modeling untouched while providing more informative inputs, which should raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.61255) has done: 'I enrich the image‑based feature extractor by adding byte‑level statistics (variance, min, max) which give the model more signal from the raw JPEG data, and I slightly increase the GradientBoosting capacity (more trees, a a bit lower learning rate) to let the model exploit these extra features. These focused tweaks keep the overall pipeline unchanged while aiming to raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.59012) has done: 'I add a lightweight LogisticRegression model for each target and combine its predictions with the existing GradientBoosting ones (70 % GBC + 30 % LR). This keeps the original pipeline while giving the ensemble a bit more signal, which should raise the validation AUC and push the Kaggle score closer to the target.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression  # added for ensembling




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


def image_features(df, images_root):
    """
    Compute simple image‑based numeric features without external image libraries:
    - file size in bytes
    - mean byte value of the raw JPEG file
    - byte variance, min and max
    - entropy of byte distribution (captures information richness)
    Missing or unreadable images are safely handled with fallback zeros.
    """
    sizes = []
    byte_means = []
    byte_vars = []
    byte_mins = []
    byte_maxs = []
    entropies = []
    for img_id in df["image_id"]:
        img_path = os.path.join(images_root, f"{img_id}.jpg")
        try:
            size = os.path.getsize(img_path)
            sizes.append(size)
            with open(img_path, "rb") as f:
                data = f.read()
            if data:
                bytes_arr = np.frombuffer(data, dtype=np.uint8)
                byte_means.append(np.mean(bytes_arr))
                byte_vars.append(np.var(bytes_arr))
                byte_mins.append(int(bytes_arr.min()))
                byte_maxs.append(int(bytes_arr.max()))
                freq = np.array(list(Counter(data).values()))
                prob = freq / prob.sum() if (prob := freq).sum() != 0 else freq
                entropy = -np.sum(prob * np.log2(prob + 1e-12))
                entropies.append(entropy)
            else:
                byte_means.append(0.0)
                byte_vars.append(0.0)
                byte_mins.append(0)
                byte_maxs.append(0)
                entropies.append(0.0)
        except Exception:
            sizes.append(0)
            byte_means.append(0.0)
            byte_vars.append(0.0)
            byte_mins.append(0)
            byte_maxs.append(0)
            entropies.append(0.0)

    return pd.DataFrame(
        {
            "file_size": sizes,
            "file_byte_mean": byte_means,
            "file_byte_variance": byte_vars,
            "file_byte_min": byte_mins,
            "file_byte_max": byte_maxs,
            "file_entropy": entropies,
        }
    )


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
for col in target_cols:
    gbc = GradientBoostingClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.9,
        random_state=42,
    )
    gbc.fit(X_full, train_df[col])

    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1,
    )
    lr.fit(X_full, train_df[col])

    models[col] = {"gbc": gbc, "lr": lr}

    if len(train_df[col].unique()) > 1:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_full,
            train_df[col],
            test_size=0.2,
            random_state=42,
            stratify=train_df[col],
        )
        gbc_tmp = GradientBoostingClassifier(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            random_state=42,
        )
        gbc_tmp.fit(X_tr, y_tr)
        lr_tmp = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=-1,
        )
        lr_tmp.fit(X_tr, y_tr)

        val_pred = (
            0.7 * gbc_tmp.predict_proba(X_val)[:, 1]
            + 0.3 * lr_tmp.predict_proba(X_val)[:, 1]
        )
        auc = roc_auc_score(y_val, val_pred)
        print(f"Validation AUC for {col} (ensemble): {auc:.4f}")




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
    probs = 0.7 * gbc.predict_proba(X_test)[:, 1] + 0.3 * lr.predict_proba(X_test)[:, 1]
    sub[col] = probs

print("Submission preview:")
print(sub.head())




## === cell 5
output_path = "submission.csv"
sub.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}")
