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

0.9699

# 6. Current score

0.68575

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the immediate crash by removing the hard dependency on a non-existent `/kaggle/input/submissions/` folder and instead default to the competition’s provided `sample_submission.csv`/`test.csv` when no external submissions are found. I keep the ensemble logic intact, but add validation so it only uses available submission indices and checks required columns, preventing index/column errors. Finally, I ensure a valid `submission.csv` is always written with the correct columns and row order aligned to `test.csv`, so you get a Kaggle-uploadable file end-to-end.'
- What this solution (achieved 0.58918) has done: 'Your current 0.5 score comes from outputting the sample-submission priors (all 0.25) or an ensemble of unknown external CSVs; without an actual model, ROC AUC stay around chance. To move toward the 0.9699 target while keeping changes minimal, I add a lightweight, self-contained classical ML image pipeline (HOG features + One-vs-Rest Logistic Regression) trained on the provided train images and labels, then predict probabilities for test images. This preserves the existing submission-writing logic and schema, but changes the source of `submission_avg` from a constant/unknown ensemble to learned predictions. If `/kaggle/input/submissions/` exists, the code still be able to ensemble those; otherwise it train and generate a proper model-based submission.'
- What this solution (achieved 0.68575) has done: 'Your current score is far below the 0.9699 target, so we need a small, legitimate lift without changing the overall approach (still HOG features + One-vs-Rest Logistic Regression + same submission writer). The biggest low-risk gains here come from (1) extracting HOG from color channels (more signal than grayscale) while keeping the same feature-extraction paradigm, and (2) making LogisticRegression closer to a “proper” probabilistic multi-label setup by using `class_weight="balanced"` and a slightly higher `max_iter` for convergence stability. I also add deterministic settings and a fallback to the competition’s sample_submission template even when ensembling external CSVs, ensuring the output format and row alignment are always correct. These changes should improve ROC AUC materially while remaining within the same core logic and running comfortably within the time limit.'

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/"  # may not exist
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7/",
    "/kaggle/data/plant-pathology-2020-fgvc7/",
    "/kaggle/input/",
    "/kaggle/data/",
]


def _first_existing(path_list, filename):
    for root in path_list:
        p = os.path.join(root, filename)
        if os.path.exists(p):
            return p
    return None


sample_sub_path = _first_existing(DATA_ROOT_CANDIDATES, "sample_submission.csv")
test_csv_path = _first_existing(DATA_ROOT_CANDIDATES, "test.csv")
train_csv_path = _first_existing(DATA_ROOT_CANDIDATES, "train.csv")

images_dir = None
for root in DATA_ROOT_CANDIDATES:
    cand = os.path.join(root, "images")
    if os.path.isdir(cand):
        images_dir = cand
        break

if (
    sample_sub_path is None
    or test_csv_path is None
    or train_csv_path is None
    or images_dir is None
):
    raise FileNotFoundError(
        "Could not locate sample_submission.csv, test.csv, train.csv and/or images/ in expected Kaggle paths."
    )

print("sample_submission.csv:", sample_sub_path)
print("test.csv:", test_csv_path)
print("train.csv:", train_csv_path)
print("images_dir:", images_dir)



## === cell 2
submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found submission files:", submissions_all)




## === cell 3
def ensemble(submissions_all, sub_idx, weights=[]):
    if len(submissions_all) == 0:
        raise ValueError("No submission files found to ensemble.")

    if len(sub_idx) != len(weights):
        raise ValueError("sub_idx and weights must have the same length.")

    submission_with_weight = []
    for i in range(len(sub_idx)):
        idx = sub_idx[i]
        if idx < 0 or idx >= len(submissions_all):
            raise IndexError(
                f"Requested submission index {idx} but only {len(submissions_all)} files exist."
            )

        print(f"I'm taking submission {submissions_all[idx]} with weight {weights[i]}")
        submission = pd.read_csv(submissions_all[idx])

        required_cols = ["healthy", "multiple_diseases", "rust", "scab"]
        missing = [c for c in required_cols if c not in submission.columns]
        if missing:
            raise ValueError(
                f"Submission file {submissions_all[idx]} missing columns: {missing}"
            )

        submission_vals = submission.loc[:, required_cols].values
        submission_with_weight.append(submission_vals * weights[i])

    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 4
def make_submission_file(submission_avg, template_csv_path, test_csv_path):
    test_df = pd.read_csv(test_csv_path)
    template_df = pd.read_csv(template_csv_path)

    required_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    missing = [c for c in required_cols if c not in template_df.columns]
    if missing:
        raise ValueError(f"Template submission missing columns: {missing}")

    submission_df = template_df.loc[:, required_cols].copy()
    submission_df = submission_df.iloc[: len(test_df)].copy()
    submission_df["image_id"] = test_df["image_id"].values

    if submission_avg.shape != (len(test_df), 4):
        raise ValueError(
            f"submission_avg has shape {submission_avg.shape}, expected {(len(test_df), 4)}"
        )

    submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = submission_avg
    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission_df.shape)




## === cell 5
def _build_image_path(image_id, images_dir):
    return os.path.join(images_dir, f"{image_id}.jpg")


def _load_rgb_image(path, target_size=(128, 128)):
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if target_size is not None:
        img = img.resize(target_size)
    return img


def _extract_hog_features_rgb(image_ids, images_dir, target_size=(128, 128)):
    import numpy as np
    from skimage.feature import hog

    feats = []
    missing_paths = 0
    for iid in image_ids:
        p = _build_image_path(iid, images_dir)
        if not os.path.exists(p):
            missing_paths += 1
            feats.append(None)
            continue

        img = _load_rgb_image(p, target_size=target_size)
        arr = np.asarray(img)  # HxWx3

        channel_feats = []
        for ch in range(3):
            f = hog(
                arr[:, :, ch],
                orientations=9,
                pixels_per_cell=(16, 16),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
                feature_vector=True,
            )
            channel_feats.append(f)
        feats.append(np.concatenate(channel_feats, axis=0))

    feat_len = None
    for f in feats:
        if f is not None:
            feat_len = len(f)
            break
    if feat_len is None:
        raise RuntimeError("No images could be loaded to extract features.")

    X = np.zeros((len(image_ids), feat_len), dtype=np.float32)
    for i, f in enumerate(feats):
        if f is None:
            continue
        X[i, :] = f.astype(np.float32, copy=False)

    if missing_paths:
        print(f"Warning: {missing_paths} images missing; filled with zeros.")
    return X


def train_and_predict_probs(train_csv_path, test_csv_path, images_dir):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import Pipeline

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    for c in ["image_id"] + target_cols:
        if c not in train_df.columns:
            raise ValueError(f"train.csv missing column: {c}")

    X_train = _extract_hog_features_rgb(
        train_df["image_id"].values.tolist(), images_dir
    )
    X_test = _extract_hog_features_rgb(test_df["image_id"].values.tolist(), images_dir)

    y = train_df[target_cols].values.astype(np.int32)

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "ovr",
                OneVsRestClassifier(
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=2000,
                        C=2.0,
                        class_weight="balanced",
                        n_jobs=1,
                    )
                ),
            ),
        ]
    )

    clf.fit(X_train, y)
    proba = clf.predict_proba(X_test)

    proba = np.clip(proba, 1e-6, 1 - 1e-6).astype("float32")
    return proba




## === cell 6
if len(submissions_all) >= 3:
    submission_avg = ensemble(submissions_all, [0, 1, 2], [0.15, 0.8, 0.05])
    template_for_output = sample_sub_path
elif len(submissions_all) >= 1:
    submission_avg = ensemble(submissions_all, [0], [1.0])
    template_for_output = sample_sub_path
else:
    submission_avg = train_and_predict_probs(train_csv_path, test_csv_path, images_dir)
    template_for_output = sample_sub_path
    print(
        "No external submissions found; trained a simple HOG(RGB)+LogReg model for predictions."
    )

make_submission_file(submission_avg, template_for_output, test_csv_path)
