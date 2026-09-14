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



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"

DATA_ROOT = "/kaggle/input/plant-pathology-2020-fgvc7"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")

IMAGES_DIR = os.path.join(DATA_ROOT, "images")

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]



## === cell 2
submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found external submissions:", submissions_all)




## === cell 3
def ensemble(submissions_all, sub_idx, weights=[]):
    """
    Weighted average ensemble of existing submission files.
    Bugfix: validate indices/weights and normalize by total weight so values remain probabilities.
    """
    if len(submissions_all) == 0:
        raise ValueError("No submissions found to ensemble.")
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
    total_w = float(sum(weights))
    if total_w <= 0:
        raise ValueError("Sum of weights must be > 0.")

    submission_with_weight = []
    for i in range(len(sub_idx)):
        print(
            f"I'm taking submission {submissions_all[sub_idx[i]]} with weight {weights[i]}"
        )
        submission = pd.read_csv(submissions_all[sub_idx[i]])
        submission = submission.loc[:, TARGET_COLS].values
        submission_with_weight.append(submission * float(weights[i]))

    submission_avg = sum(submission_with_weight) / total_w
    return submission_avg




## === cell 4
def make_submission_file_from_template(
    submission_avg, template_path, out_path="submission.csv"
):
    """
    Write submission using a provided template (sample_submission or an existing submission file).
    """
    submission_df = pd.read_csv(template_path)
    if "image_id" not in submission_df.columns:
        raise ValueError("Template must include image_id.")
    for c in TARGET_COLS:
        if c not in submission_df.columns:
            raise ValueError(f"Template missing required column: {c}")

    submission_df.loc[:, TARGET_COLS] = submission_avg
    submission_df.to_csv(out_path, index=False)
    print(
        f"Wrote {out_path} with shape {submission_df.shape} and columns {list(submission_df.columns)}"
    )




## === cell 5
def make_fallback_submission(out_path="submission.csv"):
    """
    Fallback (score-improving vs all-0.25): use class priors from train.csv as constant predictions.
    This preserves evaluation semantics (valid probabilities) and ensures end-to-end execution
    when no external submission files are available.
    """
    train = pd.read_csv(TRAIN_CSV)
    sample = pd.read_csv(SAMPLE_SUB_CSV)

    priors = train[TARGET_COLS].mean().values.astype(float)

    preds = pd.DataFrame([priors] * len(sample), columns=TARGET_COLS)

    sub = sample[["image_id"] + TARGET_COLS].copy()
    sub[TARGET_COLS] = preds[TARGET_COLS].values
    sub.to_csv(out_path, index=False)
    print("No external submissions found. Created fallback prior-based submission.")
    print("Priors used:", dict(zip(TARGET_COLS, priors)))
    print(f"Wrote {out_path} with shape {sub.shape}")




## === cell 6
def make_model_submission(out_path="submission.csv"):
    import numpy as np

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    from skimage.io import imread
    from skimage.transform import resize
    from skimage.feature import hog

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    sample = pd.read_csv(SAMPLE_SUB_CSV)

    test_ids = test_df["image_id"].tolist()

    img_size = (160, 160)
    hog_params = dict(
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )

    def _img_path(image_id):
        return os.path.join(IMAGES_DIR, f"{image_id}.jpg")

    def _zero_feats():
        dummy = np.zeros((img_size[0], img_size[1]), dtype=np.float32)
        l = hog(dummy, **hog_params).shape[0]
        return np.zeros(l * 3, dtype=np.float32)

    zero_vec = _zero_feats()

    def _featurize_ids(image_ids):
        X = []
        missing = 0
        for image_id in image_ids:
            path = _img_path(image_id)
            if not os.path.exists(path):
                missing += 1
                X.append(zero_vec.copy())
                continue

            img = imread(path)
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]

            img = resize(
                img, img_size, anti_aliasing=True, preserve_range=False
            ).astype(np.float32)

            feats = []
            for ch in range(3):
                feats.append(hog(img[:, :, ch], **hog_params).astype(np.float32))
            feats = np.concatenate(feats, axis=0).astype(np.float32)
            X.append(feats)

        X = np.vstack([x.reshape(1, -1) for x in X]).astype(np.float32)
        if missing:
            print(f"Warning: {missing} images missing; used zero features for them.")
        return X

    X_train = _featurize_ids(train_df["image_id"].tolist())
    y_train = train_df[TARGET_COLS].values.astype(int)
    X_test = _featurize_ids(test_ids)

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "ovr",
                OneVsRestClassifier(
                    LogisticRegression(
                        solver="lbfgs",
                        C=4.0,
                        max_iter=1000,
                        random_state=42,
                    )
                ),
            ),
        ]
    )

    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test).astype(float)

    proba = np.clip(proba, 1e-6, 1 - 1e-6)

    sub = pd.DataFrame({"image_id": test_ids})
    sub[TARGET_COLS] = proba
    sub = sub[["image_id"] + TARGET_COLS]
    sub.to_csv(out_path, index=False)
    print(f"Created model-based submission at {out_path} with shape {sub.shape}")




## === cell 7
if len(submissions_all) >= 2:
    submission_avg = ensemble(submissions_all, [0, 1], [0.8, 0.2])
    make_submission_file_from_template(
        submission_avg, submissions_all[0], out_path="submission.csv"
    )
elif len(submissions_all) == 1:
    single = pd.read_csv(submissions_all[0])
    missing = [c for c in (["image_id"] + TARGET_COLS) if c not in single.columns]
    if missing:
        raise ValueError(f"Single submission missing columns: {missing}")
    single[["image_id"] + TARGET_COLS].to_csv("submission.csv", index=False)
    print("Only one external submission found; copied it to submission.csv")
else:
    try:
        make_model_submission(out_path="submission.csv")
    except Exception as e:
        print("Model-based submission failed; falling back to prior-based submission.")
        print("Error was:", repr(e))
        make_fallback_submission(out_path="submission.csv")
