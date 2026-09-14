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

0.971114781143264

# 6. Current score

0.60304

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your code errors because `/kaggle/input/submissions/` doesn’t exist in this environment, so `submissions_all` is empty and indexing `[0,1,2,4]` fails. I keep your ensemble logic intact, but add a safe fallback: if no external submission files are found, create a valid baseline submission directly from `sample_submission.csv` (uniform probabilities), ensuring a `.csv` is always produced. I also make the `ensemble()` function validate indices/weights and normalize weights so the output remains properly scaled, without changing the underlying averaging approach. Finally, I auto-detect the correct competition data directory (`/kaggle/input/plant-pathology-2020-fgvc7/` or `/kaggle/data/...`) so it runs reliably.'
- What this solution (achieved 0.56794) has done: 'Your current 0.5 score comes from writing a uniform 0.25/0.25/0.25/0.25 fallback when no external `/kaggle/input/submissions/` files exist; to move toward the 0.971 target, we should instead produce a meaningful prediction without changing your “ensemble submissions” core idea. I keep the ensemble logic intact, but add a minimal, legitimate model-based fallback using scikit-learn (LogisticRegression) on simple image features so the submission is no longer uniform. I also ensure strict alignment to `test.csv` ordering and the required columns, and keep probabilities well-formed via clipping. This should substantially increase AUC relative to 0.5 while remaining within Kaggle constraints and finishing quickly.'
- What this solution (achieved 0.60304) has done: 'Your current score is far below the 0.971 target, so we should legitimately improve the fallback model while keeping your pipeline structure (ensemble if external submissions exist; otherwise train a simple model and write `submission.csv`). The biggest issue is that the fallback uses very weak handcrafted features; we can substantially improve AUC with a minimal change: extract stronger but still lightweight image features using HOG (shape/texture) plus color statistics, then train the same OneVsRest LogisticRegression as you already do. I also make the fallback robust to `PIL`-missing environments by trying `skimage` only if available, and otherwise using your existing features, so the code always runs end-to-end. Finally, I keep strict `test.csv` ordering and required columns unchanged so the submission is always valid.'

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
DATA_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7/",
    "/kaggle/data/plant-pathology-2020-fgvc7/",
    "/kaggle/input/",
    "/kaggle/data/",
]
DATA_ROOT = None
for p in DATA_CANDIDATES:
    if os.path.exists(os.path.join(p, "sample_submission.csv")):
        DATA_ROOT = p
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in expected Kaggle data paths."
    )

print("Using DATA_ROOT =", DATA_ROOT)
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TEST_CSV_PATH = os.path.join(DATA_ROOT, "test.csv")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
IMAGES_DIR = os.path.join(DATA_ROOT, "images")



## === cell 2
SUBMISSIONS_PATH = "/kaggle/input/submissions/"



## === cell 3
submissions_all = []
if os.path.exists(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found submission files:", submissions_all)




## === cell 4
def ensemble(submissions_all, sub_idx, weights=[]):
    if len(submissions_all) == 0:
        raise ValueError("No submission files found to ensemble.")
    if len(sub_idx) == 0:
        raise ValueError("sub_idx is empty.")
    if len(weights) != len(sub_idx):
        raise ValueError(
            f"weights length ({len(weights)}) must match sub_idx length ({len(sub_idx)})."
        )

    for j in sub_idx:
        if j < 0 or j >= len(submissions_all):
            raise IndexError(
                f"sub_idx contains {j}, but submissions_all has length {len(submissions_all)}."
            )

    wsum = float(sum(weights))
    if wsum == 0:
        raise ValueError("Sum of weights is zero.")
    weights = [w / wsum for w in weights]

    submission_with_weight = []
    for i in range(len(sub_idx)):
        print(
            f"I'm taking submission {submissions_all[sub_idx[i]]} with weight {weights[i]}"
        )
        submission = pd.read_csv(submissions_all[sub_idx[i]])
        submission = submission.loc[
            :, ["healthy", "multiple_diseases", "rust", "scab"]
        ].values
        submission_with_weight.append(submission * weights[i])

    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 5
def make_submission_file(
    submission_avg, submissions_all=None, out_path="submission.csv"
):
    test_df = pd.read_csv(TEST_CSV_PATH)
    submission_df = pd.read_csv(SAMPLE_SUB_PATH)

    required_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    submission_df = submission_df.loc[:, required_cols].copy()

    submission_df = test_df.merge(submission_df, on="image_id", how="left")[
        required_cols
    ]

    submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = submission_avg
    submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = submission_df[
        ["healthy", "multiple_diseases", "rust", "scab"]
    ].clip(0.0, 1.0)

    submission_df.to_csv(out_path, index=False)
    print(
        f"Wrote {out_path} with shape {submission_df.shape} and columns {list(submission_df.columns)}"
    )




## === cell 6
def _extract_image_features_basic(image_path, size=(64, 64)):
    from PIL import Image
    import numpy as np

    img = Image.open(image_path).convert("RGB")
    img = img.resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3)

    mean_rgb = arr.reshape(-1, 3).mean(axis=0)
    std_rgb = arr.reshape(-1, 3).std(axis=0)
    gray = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
    mean_g = gray.mean()
    std_g = gray.std()

    patch = Image.fromarray((gray * 255).astype("uint8")).resize((16, 16))
    patch = (np.asarray(patch, dtype=np.float32) / 255.0).reshape(-1)

    feat = np.concatenate([mean_rgb, std_rgb, [mean_g, std_g], patch], axis=0)
    return feat


def _extract_image_features_hog(image_path, size=(128, 128)):
    import numpy as np
    from PIL import Image

    try:
        from skimage.feature import hog
    except Exception:
        return _extract_image_features_basic(image_path, size=(64, 64))

    img = Image.open(image_path).convert("RGB").resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0

    mean_rgb = arr.reshape(-1, 3).mean(axis=0)
    std_rgb = arr.reshape(-1, 3).std(axis=0)

    gray = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
    mean_g = float(gray.mean())
    std_g = float(gray.std())

    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)

    feat = np.concatenate(
        [mean_rgb, std_rgb, np.array([mean_g, std_g], np.float32), hog_feat], axis=0
    )
    return feat


def _build_features(df_ids):
    import numpy as np

    feats = []
    missing = 0
    for image_id in df_ids:
        img_path = os.path.join(IMAGES_DIR, f"{image_id}.jpg")
        if not os.path.exists(img_path):
            missing += 1
            feats.append(None)
            continue
        feats.append(_extract_image_features_hog(img_path))

    first = next((f for f in feats if f is not None), None)
    if first is None:
        raise RuntimeError("No images found to extract features from.")
    d = int(first.shape[0])

    for i in range(len(feats)):
        if feats[i] is None:
            feats[i] = np.zeros((d,), dtype=np.float32)

    if missing:
        print(f"Warning: {missing} images missing; using zero features for them.")
    return np.vstack(feats)


def train_and_predict_fallback():
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    train_df = pd.read_csv(TRAIN_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)

    target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

    X_train = _build_features(train_df["image_id"].values)
    X_test = _build_features(test_df["image_id"].values)
    y_train = train_df[target_cols].values.astype(int)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    base = LogisticRegression(
        C=2.0,
        solver="lbfgs",
        max_iter=600,  # Change: a bit more iteration budget for convergence on higher-dim HOG features
        n_jobs=1,
        class_weight="balanced",
        random_state=42,
    )
    clf = OneVsRestClassifier(base)
    clf.fit(X_train_s, y_train)

    proba = clf.predict_proba(X_test_s)
    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    return proba




## === cell 7
if len(submissions_all) == 0:
    submission_avg = train_and_predict_fallback()
    make_submission_file(
        submission_avg, submissions_all=None, out_path="submission.csv"
    )
else:
    requested_idx = [0, 1, 2, 4]
    requested_w = [0.1, 0.7, 0.15, 0.05]

    if max(requested_idx) < len(submissions_all):
        submission_avg = ensemble(submissions_all, requested_idx, requested_w)
    else:
        print(
            "Requested indices exceed available files; ensembling all found submissions equally."
        )
        all_idx = list(range(len(submissions_all)))
        all_w = [1.0] * len(submissions_all)
        submission_avg = ensemble(submissions_all, all_idx, all_w)

    make_submission_file(submission_avg, submissions_all, out_path="submission.csv")
