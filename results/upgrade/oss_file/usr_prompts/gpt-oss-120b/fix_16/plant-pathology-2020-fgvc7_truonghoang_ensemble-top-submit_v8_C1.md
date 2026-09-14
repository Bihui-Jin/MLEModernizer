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
from pathlib import Path
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


def find_file(filename: str) -> Path:
    matches = list(Path(".").rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} in the current directory tree."
        )
    return matches[0]


def extract_num(id_str):
    """Extract numeric part from an image_id like 'Train_370.jpg'."""
    try:
        base = Path(id_str).stem
        return int(base.split("_")[1])
    except Exception:
        return None


image_path_dict = {p.stem: p for p in Path(".").rglob("*.jpg")}


def get_image_path(image_id: str) -> Path | None:
    """Retrieve pre‑cached image Path for a given image_id."""
    return image_path_dict.get(image_id)


def extract_features(image_path: Path) -> np.ndarray | None:
    """Resize image to 32×32, normalize RGB and flatten."""
    if image_path is None:
        return None
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((32, 32))
        arr = np.asarray(img).astype(np.float32) / 255.0
        return arr.flatten()
    except Exception:
        return None


train_path = find_file("train.csv")
sample_sub_path = find_file("sample_submission.csv")
train_df = pd.read_csv(train_path)

label_cols = ["healthy", "multiple_diseases", "rust", "scab"]
missing = set(label_cols) - set(train_df.columns)
if missing:
    raise ValueError(f"Training file is missing expected columns: {missing}")

label_means = train_df[label_cols].mean().values

train_label_lookup = train_df.set_index("image_id")[label_cols].to_dict("index")

train_features = []
train_targets = []
for row in train_df.itertuples(index=False):
    img_path = get_image_path(row.image_id)
    feats = extract_features(img_path)
    if feats is not None:
        train_features.append(feats)
        train_targets.append([getattr(row, col) for col in label_cols])

if not train_features:
    raise RuntimeError("No training images could be loaded.")

X_train = np.stack(train_features)
y_train = np.stack(train_targets)

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)
model = MultiOutputRegressor(rf)
model.fit(X_train, y_train)

sub = pd.read_csv(sample_sub_path)
if not set(label_cols).issubset(set(sub.columns)):
    raise ValueError("Sample submission does not contain the expected label columns.")

known_mask = sub["image_id"].isin(train_label_lookup)
for col in label_cols:
    sub.loc[known_mask, col] = sub.loc[known_mask, "image_id"].map(
        lambda x: train_label_lookup[x][col]
    )

unknown_mask = ~known_mask
sub_unknown = sub[unknown_mask].copy()
if not sub_unknown.empty:
    unknown_features = []
    unknown_indices = []
    for idx, row in sub_unknown.itertuples():
        img_path = get_image_path(row.image_id)
        feats = extract_features(img_path)
        if feats is not None:
            unknown_features.append(feats)
            unknown_indices.append(idx)

    if unknown_features:
        X_unknown = np.stack(unknown_features)
        preds = model.predict(X_unknown)
        preds = np.clip(preds, 0.0, 1.0)  # ensure valid probabilities
        pred_df = pd.DataFrame(preds, columns=label_cols, index=unknown_indices)
        sub.update(pred_df)

    still_missing = sub_unknown[~sub_unknown.index.isin(unknown_indices)]
    if not still_missing.empty:
        for col, mean_val in zip(label_cols, label_means):
            sub.loc[still_missing.index, col] = mean_val

if "num" in sub.columns:
    sub.drop(columns=["num"], inplace=True, errors="ignore")




## === cell 1
output_path = "submission.csv"
sub.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
