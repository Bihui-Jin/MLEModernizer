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

0.9684771380136814

# 6. Current score

0.63649

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the file-not-found bug by removing the dependency on external submission CSVs that aren’t present in your environment and instead generate predictions from the provided dataset. To keep changes minimal while ensuring a valid end-to-end run, I implement a simple, deterministic baseline that reads `train.csv` and outputs the per-class prevalence as constant probabilities for every test image (this preserves evaluation semantics: probabilistic multi-label outputs per class). I also make the paths robust to either `/kaggle/input/...` or the provided `/kaggle/data/...` layout, and ensure the submission columns exactly match `sample_submission.csv`. This always produce a valid `submission.csv` and should score meaningfully above random, moving toward the target from “no score”.'
- What this solution (achieved 0.63649) has done: 'Your current 0.5 score comes from predicting constant class prevalences, which is too weak for ROC AUC; to move toward the 0.968 target without changing the “core logic” of producing per-class probabilities, we replace constants with a simple deterministic image-based model using only allowed packages. Specifically, we extract lightweight RGB color histogram features from the provided JPGs, train one-vs-rest Logistic Regression per class on `train.csv`, and predict probabilities for `test.csv`. We keep paths and submission column order exactly aligned to `sample_submission.csv`, and add checks to ensure every test `image_id` gets a prediction even if an image is missing. This is a minimal, fast baseline that should materially improve AUC while staying within constraints and producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
BASE_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]

BASE = None
for c in BASE_CANDIDATES:
    if os.path.exists(c):
        if c in ("/kaggle/input", "/kaggle/data"):
            cc = os.path.join(c, "plant-pathology-2020-fgvc7")
            if os.path.exists(cc):
                BASE = cc
                break
        BASE = c
        break

if BASE is None:
    raise FileNotFoundError(
        "Could not locate Kaggle data directory in expected locations."
    )

train_path = os.path.join(BASE, "train.csv")
test_path = os.path.join(BASE, "test.csv")
sample_path = os.path.join(BASE, "sample_submission.csv")
images_dir = os.path.join(BASE, "images")

print("Using BASE:", BASE)
print(
    "Files exist:",
    os.path.exists(train_path),
    os.path.exists(test_path),
    os.path.exists(sample_path),
    os.path.exists(images_dir),
)



## === cell 2
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

target_cols = [c for c in sample_sub.columns if c != "image_id"]
for c in ["image_id"] + target_cols:
    if c != "image_id" and c not in train_df.columns:
        raise ValueError(f"Train CSV missing expected target column: {c}")
if "image_id" not in test_df.columns:
    raise ValueError("Test CSV missing image_id column.")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Targets:", target_cols)
print("Images dir:", images_dir)



## === cell 3
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def _safe_image_path(image_id: str) -> str:
    return os.path.join(images_dir, f"{image_id}.jpg")


def extract_rgb_hist_features(image_path: str, bins: int = 16) -> np.ndarray:
    """
    Lightweight feature: concatenated normalized RGB histograms.
    Deterministic and fast; avoids changing overall semantics (still probabilistic outputs).
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            arr = np.asarray(img, dtype=np.uint8)
    except Exception:
        return np.zeros((3 * bins,), dtype=np.float32)

    feats = []
    for ch in range(3):
        h, _ = np.histogram(arr[..., ch].ravel(), bins=bins, range=(0, 256))
        h = h.astype(np.float32)
        s = float(h.sum())
        if s > 0:
            h /= s
        feats.append(h)
    return np.concatenate(feats, axis=0).astype(np.float32)


def build_feature_matrix(image_ids: pd.Series, bins: int = 16) -> np.ndarray:
    X = np.zeros((len(image_ids), 3 * bins), dtype=np.float32)
    missing_count = 0
    for i, iid in enumerate(image_ids.tolist()):
        p = _safe_image_path(iid)
        if not os.path.exists(p):
            missing_count += 1
        X[i] = extract_rgb_hist_features(p, bins=bins)
    if missing_count:
        print(
            f"Warning: {missing_count} images not found under {images_dir}. Using zero-features for those."
        )
    return X




## === cell 4
BINS = 16
X_train = build_feature_matrix(train_df["image_id"], bins=BINS)
X_test = build_feature_matrix(test_df["image_id"], bins=BINS)

models = {}
test_pred = np.zeros((len(test_df), len(target_cols)), dtype=np.float64)

for j, col in enumerate(target_cols):
    y = train_df[col].astype(int).values

    if y.min() == y.max():
        prior = float(y.mean())
        test_pred[:, j] = prior
        models[col] = None
        continue

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "lr",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                    C=1.0,
                    random_state=0,
                ),
            ),
        ]
    )
    clf.fit(X_train, y)
    proba = clf.predict_proba(X_test)[:, 1].astype(np.float64)
    test_pred[:, j] = proba
    models[col] = clf

test_pred = np.clip(test_pred, 0.0, 1.0)



## === cell 5
sub = sample_sub.copy()

pred_df = pd.DataFrame(test_pred, columns=target_cols)
pred_df.insert(0, "image_id", test_df["image_id"].values)

sub = sub[["image_id"] + target_cols].merge(
    pred_df, on="image_id", how="left", suffixes=("", "_pred")
)

priors = train_df[target_cols].mean(axis=0).astype(float)
for c in target_cols:
    pred_col = c + "_pred"
    if pred_col in sub.columns:
        sub[c] = sub[pred_col]
        sub.drop(columns=[pred_col], inplace=True)
    sub[c] = sub[c].astype(float)
    sub[c] = sub[c].fillna(float(priors[c])).clip(0.0, 1.0)

sub = sub[["image_id"] + target_cols]

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub.head())
print("Submission shape:", sub.shape)

assert out_path.endswith(".csv")
assert list(sub.columns) == list(sample_sub.columns)
assert len(sub) == len(sample_sub)
assert sub[target_cols].isna().sum().sum() == 0
assert ((sub[target_cols] >= 0.0) & (sub[target_cols] <= 1.0)).all().all()
