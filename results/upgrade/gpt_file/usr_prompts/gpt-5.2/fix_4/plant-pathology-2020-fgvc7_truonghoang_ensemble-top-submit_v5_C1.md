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

0.9675421146508574

# 6. Current score

0.64308

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the pipeline by removing the dependency on missing external Kaggle dataset submission files (the `../input/plantpathology/*.csv` ensemble), which is what currently causes the `FileNotFoundError` and prevents `dsub` from being defined. To keep the solution valid end-to-end in this environment, I instead generate a safe baseline submission directly from `sample_submission.csv`, filling each class with a constant prior computed from the training label means (score-neutral vs. uniform and typically slightly better calibrated). I also make the input paths robust to both provided folder layouts (`/kaggle/input/...` and `/kaggle/data/...`). Finally, the script always write `submission.csv` with the exact required columns.'
- What this solution (achieved 0.65733) has done: 'Your current 0.5 score comes from predicting constants, which yields ~0.5 ROC AUC because the ranking contains no information. With the very limited installed packages (no deep learning / CV libs), the smallest legitimate improvement is to extract simple image-based features (RGB channel statistics) using only the Python standard library (`PIL`) and then train a lightweight multi-output classifier (one-vs-rest logistic regression) from scikit-learn to produce non-constant probabilities. This preserves the “simple baseline” spirit while creating meaningful per-image ranking signals that should move the score substantially upward toward your 0.9675 target without changing the evaluation semantics. The script still writes a valid `submission.csv` with the exact required columns and ordering.'
- What this solution (achieved 0.64308) has done: 'You’re far below the target (0.657 vs 0.9675), so we should improve ranking signal while keeping the same “simple image stats + OneVsRest LogisticRegression” core logic. The biggest low-risk gain is to expand the feature vector beyond global RGB mean/std to include coarse spatial information (same stats computed on a 2×2 grid) plus simple color ratios; this still uses PIL+NumPy only and keeps training identical. I also add stable, deterministic image preprocessing and class-balancing in LogisticRegression (same model family/solver) to reduce bias from class imbalance and typically improve ROC-AUC without changing evaluation semantics. The pipeline still run end-to-end and write a valid `submission.csv` with the required columns and ordering.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
CANDIDATE_BASES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]


def resolve_competition_dir():
    for b in CANDIDATE_BASES:
        if os.path.isdir(b):
            if os.path.isfile(os.path.join(b, "train.csv")):
                return b
            sub = os.path.join(b, "plant-pathology-2020-fgvc7")
            if os.path.isfile(os.path.join(sub, "train.csv")):
                return sub
    raise FileNotFoundError(
        "Could not locate competition directory containing train.csv"
    )


COMP_DIR = resolve_competition_dir()
COMP_DIR



## === cell 2
train_path = os.path.join(COMP_DIR, "train.csv")
test_path = os.path.join(COMP_DIR, "test.csv")
sample_path = os.path.join(COMP_DIR, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sub = pd.read_csv(sample_path)

train.shape, test.shape, sub.shape, train.columns.tolist()



## === cell 3
target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

missing_in_train = [c for c in target_cols if c not in train.columns]
missing_in_sub = [c for c in target_cols if c not in sub.columns]
if missing_in_train:
    raise ValueError(f"Missing target columns in train.csv: {missing_in_train}")
if missing_in_sub:
    raise ValueError(
        f"Missing target columns in sample_submission.csv: {missing_in_sub}"
    )
if "image_id" not in sub.columns:
    raise ValueError("sample_submission.csv must contain image_id")

IMG_DIR_CANDIDATES = [
    os.path.join(COMP_DIR, "images"),
    os.path.join(os.path.dirname(COMP_DIR), "images"),
]
IMG_DIR = None
for p in IMG_DIR_CANDIDATES:
    if os.path.isdir(p):
        IMG_DIR = p
        break
if IMG_DIR is None:
    raise FileNotFoundError(
        "Could not locate images/ directory next to competition CSVs."
    )

from PIL import Image, ImageOps  # noqa: E402


def img_path_from_id(image_id: str) -> str:
    return os.path.join(IMG_DIR, f"{image_id}.jpg")


def _safe_stats(arr: np.ndarray) -> np.ndarray:
    """Compute mean/std over H,W for each channel. arr is float32 in [0,1], shape (H,W,3)."""
    means = arr.mean(axis=(0, 1))
    stds = arr.std(axis=(0, 1))
    return np.concatenate([means, stds], axis=0)  # 6


def extract_basic_rgb_stats(image_id: str) -> np.ndarray:
    """
    Change rationale (score-improving but minimal): keep the same "simple image stats" core logic,
    but add coarse spatial stats (2x2 grid) + a few stable color-ratio features to improve ranking
    signal for ROC-AUC while still being lightweight and deterministic.
    """
    fp = img_path_from_id(image_id)
    with Image.open(fp) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im = im.resize((128, 128), resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0  # (H,W,3) in [0,1]

    global_stats = _safe_stats(arr)  # 6

    gray = arr.mean(axis=2)
    gmean = float(gray.mean())
    gstd = float(gray.std())

    H, W, _ = arr.shape
    h2, w2 = H // 2, W // 2
    quads = [
        arr[:h2, :w2, :],
        arr[:h2, w2:, :],
        arr[h2:, :w2, :],
        arr[h2:, w2:, :],
    ]
    quad_stats = np.concatenate([_safe_stats(q) for q in quads], axis=0)  # 24

    eps = 1e-6
    r = arr[:, :, 0].mean()
    g = arr[:, :, 1].mean()
    b = arr[:, :, 2].mean()
    rg = float(r / (g + eps))
    rb = float(r / (b + eps))
    gb = float(g / (b + eps))
    s = float((r + g + b) / 3.0)

    return np.concatenate(
        [
            global_stats,
            quad_stats,
            np.array([gmean, gstd, rg, rb, gb, s], dtype=np.float32),
        ],
        axis=0,
    )


train_ids = train["image_id"].astype(str).tolist()
test_ids = test["image_id"].astype(str).tolist()

X_train = np.vstack([extract_basic_rgb_stats(i) for i in train_ids])
X_test = np.vstack([extract_basic_rgb_stats(i) for i in test_ids])
y_train = train[target_cols].astype(int).values

from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.multiclass import OneVsRestClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

base_lr = LogisticRegression(
    max_iter=600,
    solver="lbfgs",
    class_weight="balanced",
)

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("ovr", OneVsRestClassifier(base_lr)),
    ]
)

clf.fit(X_train, y_train)

proba = clf.predict_proba(X_test)  # shape (n_test, 4)

pred_df = pd.DataFrame(proba, columns=target_cols)
pred_df.insert(0, "image_id", test["image_id"].values)

sub_out = sub[["image_id"]].merge(pred_df, on="image_id", how="left")

priors = train[target_cols].mean().astype(float)
for c in target_cols:
    sub_out[c] = sub_out[c].fillna(float(priors[c]))

sub_out[target_cols] = sub_out[target_cols].clip(0.0, 1.0)

out_path = "submission.csv"
sub_out.to_csv(out_path, index=False)
out_path, sub_out.head()
