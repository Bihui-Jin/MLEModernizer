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

0.9684895161543868

# 6. Current score

0.50842

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the runtime error by making the code robust to the case where `/kaggle/input/submissions/` doesn’t exist or contains fewer files than the hard-coded indices. To still produce a valid `.csv` submission end-to-end, I fall back to using the provided `sample_submission.csv` (uniform probabilities) when no external submissions are available. I also add minimal validation to ensure required columns exist, weights match indices, and the output probabilities are clipped to `[0,1]`. These changes are execution- and format-focused (score-neutral except that a valid submission is produced instead of crashing).'
- What this solution (achieved 0.50842) has done: 'Your current 0.5 score comes from submitting uniform probabilities (fallback), so the smallest legitimate improvement is to generate non-uniform predictions from the provided images and labels instead of relying on external “submissions” that don’t exist in your environment. I keep the core “produce a submission.csv with the correct columns/order” semantics, but replace the empty-submissions fallback with a simple, deterministic image-feature + multi-output classifier pipeline trained on `train.csv` and the image folder you already have. This should move the score substantially upward toward the 0.968 target while staying within the installed packages (pandas + scikit-learn) and keeping runtime under 600s by using low-resolution grayscale features. If `/kaggle/input/submissions/` does contain valid submissions, the original ensembling path remains unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/"

FALLBACK_SAMPLE_SUB_1 = "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv"
FALLBACK_SAMPLE_SUB_2 = "/kaggle/data/sample_submission.csv"



## === cell 2
submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found submission candidates:", submissions_all)



## === cell 3
TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]


def _load_submission(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in (["image_id"] + TARGET_COLS) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Submission file {path} missing columns: {missing}. Has: {list(df.columns)}"
        )
    return df[["image_id"] + TARGET_COLS]


def ensemble(submissions_all, sub_idx, weights=None):
    """
    Weighted average over submission files by index in submissions_all.
    Returns numpy array of shape (n_rows, 4) aligned to the first selected submission's row order.
    """
    if weights is None:
        weights = [1.0] * len(sub_idx)
    if len(sub_idx) != len(weights):
        raise ValueError(
            f"sub_idx length ({len(sub_idx)}) must equal weights length ({len(weights)})"
        )

    if len(submissions_all) == 0:
        raise ValueError("No submission files found to ensemble.")

    if any((i < 0 or i >= len(submissions_all)) for i in sub_idx):
        raise IndexError(
            f"Requested indices {sub_idx} out of range for {len(submissions_all)} files."
        )

    submission_with_weight = []
    base_ids = None

    for i, idx in enumerate(sub_idx):
        path = submissions_all[idx]
        w = float(weights[i])
        print(f"I'm taking submission {path} with weight {w}")

        df = _load_submission(path)
        if base_ids is None:
            base_ids = df["image_id"].values
        else:
            if not (df["image_id"].values == base_ids).all():
                df = df.set_index("image_id").loc[base_ids].reset_index()

        submission_with_weight.append(df[TARGET_COLS].values * w)

    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 4
def make_submission_file(submission_avg, template_submission_path):
    """
    Writes submission.csv with correct columns and row count, using template_submission_path for image_id order.
    """
    submission_df = _load_submission(template_submission_path)

    submission_df.loc[:, TARGET_COLS] = submission_avg

    submission_df.loc[:, TARGET_COLS] = submission_df.loc[:, TARGET_COLS].clip(0.0, 1.0)

    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission_df.shape)
    print(submission_df.head())




## === cell 5
def _pick_fallback_sample():
    if os.path.exists(FALLBACK_SAMPLE_SUB_1):
        return FALLBACK_SAMPLE_SUB_1
    if os.path.exists(FALLBACK_SAMPLE_SUB_2):
        return FALLBACK_SAMPLE_SUB_2
    raise FileNotFoundError("Could not find any sample_submission.csv fallback path.")


def _find_dataset_root():
    candidates = [
        "/kaggle/input/plant-pathology-2020-fgvc7",
        "/kaggle/data/plant-pathology-2020-fgvc7",
        "/kaggle/input",
        "/kaggle/data",
    ]
    for root in candidates:
        train_csv = os.path.join(root, "train.csv")
        test_csv = os.path.join(root, "test.csv")
        images_dir = os.path.join(root, "images")
        if (
            os.path.exists(train_csv)
            and os.path.exists(test_csv)
            and os.path.isdir(images_dir)
        ):
            return root
    raise FileNotFoundError(
        "Could not locate train.csv/test.csv/images directory under expected Kaggle paths."
    )


def _load_image_as_feature_vector(image_path, size=(64, 64)):
    from PIL import Image
    import numpy as np

    img = (
        Image.open(image_path).convert("L").resize(size)
    )  # grayscale, low-res for speed
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(-1)


def _build_features(image_ids, images_dir, size=(64, 64)):
    import numpy as np

    X = np.zeros((len(image_ids), size[0] * size[1]), dtype=np.float32)
    missing = 0
    for i, img_id in enumerate(image_ids):
        p = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(p):
            missing += 1
            continue
        X[i] = _load_image_as_feature_vector(p, size=size)
    if missing:
        print(
            f"Warning: {missing} images not found in {images_dir}; using zero features for them."
        )
    return X


def _train_and_predict_fallback_submission():
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier

    root = _find_dataset_root()
    train_path = os.path.join(root, "train.csv")
    test_path = os.path.join(root, "test.csv")
    images_dir = os.path.join(root, "images")
    sample_sub_path = os.path.join(root, "sample_submission.csv")
    if not os.path.exists(sample_sub_path):
        sample_sub_path = _pick_fallback_sample()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    for c in ["image_id"] + TARGET_COLS:
        if c not in train_df.columns and c != "image_id":
            raise ValueError(f"train.csv missing required target column: {c}")
    if "image_id" not in test_df.columns:
        raise ValueError("test.csv missing image_id column")

    X_train = _build_features(train_df["image_id"].tolist(), images_dir, size=(64, 64))
    y_train = train_df[TARGET_COLS].astype(int).values
    X_test = _build_features(test_df["image_id"].tolist(), images_dir, size=(64, 64))

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "model",
                MultiOutputClassifier(
                    LogisticRegression(max_iter=300, solver="lbfgs", random_state=0)
                ),
            ),
        ]
    )

    clf.fit(X_train, y_train)

    probas = clf.predict_proba(X_test)
    pred = np.column_stack([p[:, 1] for p in probas]).astype(np.float32)

    template_df = _load_submission(sample_sub_path)
    pred_df = pd.DataFrame(pred, columns=TARGET_COLS)
    pred_df.insert(0, "image_id", test_df["image_id"].values)

    pred_df = (
        pred_df.set_index("image_id").loc[template_df["image_id"].values].reset_index()
    )
    return (
        pred_df[["image_id"] + TARGET_COLS].values[:, 1:].astype(np.float32),
        sample_sub_path,
    )


if len(submissions_all) >= 4:
    submission_avg = ensemble(submissions_all, [1, 2, 3], [0.3, 0.3, 0.4])
    template_path = submissions_all[1]  # template aligned with ensemble base ids
    make_submission_file(submission_avg, template_path)
elif len(submissions_all) > 0:
    sub_idx = list(range(len(submissions_all)))
    weights = [1.0 / len(sub_idx)] * len(sub_idx)
    submission_avg = ensemble(submissions_all, sub_idx, weights)
    template_path = submissions_all[0]
    make_submission_file(submission_avg, template_path)
else:
    submission_avg, template_path = _train_and_predict_fallback_submission()
    make_submission_file(submission_avg, template_path)
