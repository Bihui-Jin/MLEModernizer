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

0.969311782713818

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'The error happens because the notebook expects an external `/kaggle/input/submissions/submissions/` dataset that isn’t present, so `submissions_all` is empty and indexing `[0,1]` fails. I keep the “ensemble CSVs with weights” core logic, but add a fallback that uses the competition’s `sample_submission.csv` as a valid base when no external submissions are found. I also make the ensemble function validate indices/weights and align rows by `image_id` to avoid silent mis-ordering bugs. This run end-to-end and always write a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"

DATA_ROOT_CANDIDATES = [
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data",
    "/kaggle/input",
]


def _resolve_data_root():
    for root in DATA_ROOT_CANDIDATES:
        sample = os.path.join(root, "sample_submission.csv")
        test = os.path.join(root, "test.csv")
        train = os.path.join(root, "train.csv")
        images = os.path.join(root, "images")
        if all(os.path.exists(p) for p in [sample, test, train, images]):
            return root
    raise FileNotFoundError(
        "Could not resolve DATA_ROOT. Checked: "
        + ", ".join(DATA_ROOT_CANDIDATES)
        + ". Expected to find sample_submission.csv, test.csv, train.csv, images/."
    )


DATA_ROOT = _resolve_data_root()
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TEST_CSV_PATH = os.path.join(DATA_ROOT, "test.csv")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
IMAGES_DIR = os.path.join(DATA_ROOT, "images")

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

print("Using DATA_ROOT:", DATA_ROOT)
for p in [SAMPLE_SUB_PATH, TEST_CSV_PATH, TRAIN_CSV_PATH, IMAGES_DIR]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required path not found: {p}")



## === cell 2
submissions_all = []
if os.path.exists(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))

submissions_all.sort()
print("Found submission candidates:", submissions_all)




## === cell 3
def ensemble(submissions_all, sub_idx, weights=None, target_cols=None):
    """
    Weighted average ensemble of multiple submission CSVs.
    Fixes:
      - Guard against empty list / out-of-range indices
      - Ensure weights length matches sub_idx
      - Align rows by image_id to avoid ordering mismatches
    """
    if target_cols is None:
        target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    if weights is None:
        weights = [1.0] * len(sub_idx)

    if len(sub_idx) == 0:
        raise ValueError("sub_idx must contain at least one index.")
    if len(weights) != len(sub_idx):
        raise ValueError(
            f"weights length ({len(weights)}) must match sub_idx length ({len(sub_idx)})."
        )
    if len(submissions_all) == 0:
        raise ValueError(
            "submissions_all is empty; no external submissions to ensemble."
        )

    base_path = submissions_all[sub_idx[0]]
    base_df = pd.read_csv(base_path)
    if "image_id" not in base_df.columns:
        raise ValueError(f"'image_id' column missing from {base_path}")
    base_image_ids = base_df["image_id"].astype(str).values

    submission_with_weight = []
    for i in range(len(sub_idx)):
        si = sub_idx[i]
        if si < 0 or si >= len(submissions_all):
            raise IndexError(
                f"sub_idx[{i}]={si} is out of range for submissions_all of length {len(submissions_all)}"
            )

        path = submissions_all[si]
        w = float(weights[i])
        print(f"I'm taking submission {path} with weight {w}")

        df = pd.read_csv(path)
        missing = set(["image_id"] + target_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns {missing} in {path}")

        df["image_id"] = df["image_id"].astype(str)
        df = df.set_index("image_id").reindex(base_image_ids)

        if df[target_cols].isna().any().any():
            raise ValueError(
                f"NaNs introduced after aligning {path} to base image_id order. Check image_id consistency."
            )

        submission_with_weight.append(df[target_cols].to_numpy(dtype="float64") * w)

    submission_avg = sum(submission_with_weight)
    return submission_avg, base_image_ids




## === cell 4
def make_submission_file(
    submission_avg,
    image_ids,
    out_path="submission.csv",
    sample_sub_path=SAMPLE_SUB_PATH,
    target_cols=None,
):
    """
    Writes a valid Kaggle submission CSV.
    Fix:
      - Use official sample_submission.csv as template (ensures correct columns/order)
      - Enforce image_id alignment
    """
    if target_cols is None:
        target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

    sub_df = pd.read_csv(sample_sub_path)
    sub_df["image_id"] = sub_df["image_id"].astype(str)
    image_ids = pd.Series(image_ids, dtype=str)

    sub_df = sub_df.set_index("image_id").reindex(image_ids).reset_index()

    if len(sub_df) != len(submission_avg):
        raise ValueError(
            f"Length mismatch: template has {len(sub_df)} rows, predictions have {len(submission_avg)} rows"
        )

    sub_df.loc[:, target_cols] = submission_avg
    sub_df.to_csv(out_path, index=False)
    print(
        f"Wrote {out_path} with shape {sub_df.shape} and columns {list(sub_df.columns)}"
    )




## === cell 5
def _extract_mean_rgb_features(image_ids, images_dir, is_test=False):
    """
    Minimal, deterministic feature extraction: mean RGB per image (3 features).
    Uses PIL from the standard Kaggle environment via pillow (dependency of many stacks).
    """
    from PIL import Image
    import numpy as np

    feats = []
    ok_ids = []
    for img_id in image_ids:
        fname = f"{img_id}.jpg"
        path = os.path.join(images_dir, fname)
        if not os.path.exists(path):
            feats.append([float("nan"), float("nan"), float("nan")])
            ok_ids.append(img_id)
            continue
        im = Image.open(path).convert("RGB")
        arr = np.asarray(im, dtype=np.float32) / 255.0
        m = arr.reshape(-1, 3).mean(axis=0)
        feats.append(m.tolist())
        ok_ids.append(img_id)
    X = pd.DataFrame(feats, columns=["mean_r", "mean_g", "mean_b"])
    X.insert(0, "image_id", ok_ids)
    return X


def train_and_predict_baseline(
    train_csv_path=TRAIN_CSV_PATH,
    test_csv_path=TEST_CSV_PATH,
    images_dir=IMAGES_DIR,
    target_cols=TARGET_COLS,
):
    """
    Minimal supervised model: mean RGB features + per-label LogisticRegression.
    Score-oriented minimal changes (keeps same features/model family and training semantics):
      - Use MultiOutputClassifier(LogisticRegression) for true independent binary tasks,
        better aligned with mean column-wise ROC AUC.
      - Remove per-row probability normalization (hurts independent AUC).
    """
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    train_df["image_id"] = train_df["image_id"].astype(str)
    test_df["image_id"] = test_df["image_id"].astype(str)

    Xtr = _extract_mean_rgb_features(train_df["image_id"].tolist(), images_dir)
    Xte = _extract_mean_rgb_features(test_df["image_id"].tolist(), images_dir)

    Xtr = Xtr.set_index("image_id").reindex(train_df["image_id"]).reset_index()
    Xte = Xte.set_index("image_id").reindex(test_df["image_id"]).reset_index()

    y = train_df[target_cols].to_numpy(dtype=np.int32)
    Xtr_mat = Xtr[["mean_r", "mean_g", "mean_b"]].to_numpy(dtype=np.float32)
    Xte_mat = Xte[["mean_r", "mean_g", "mean_b"]].to_numpy(dtype=np.float32)

    base_lr = LogisticRegression(
        max_iter=500,
        solver="lbfgs",
        random_state=42,
        class_weight="balanced",
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", MultiOutputClassifier(base_lr, n_jobs=None)),
        ]
    )

    model.fit(Xtr_mat, y)

    proba_list = model.named_steps["clf"].predict_proba(
        model.named_steps["scaler"].transform(
            model.named_steps["imputer"].transform(Xte_mat)
        )
    )
    proba = np.vstack([p[:, 1] for p in proba_list]).T  # (n,4)

    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    return proba, test_df["image_id"].values




## === cell 6
if len(submissions_all) >= 2:
    submission_avg, image_ids = ensemble(
        submissions_all, [0, 1], [0.4, 0.6], target_cols=TARGET_COLS
    )
    make_submission_file(
        submission_avg, image_ids, out_path="submission.csv", target_cols=TARGET_COLS
    )
elif len(submissions_all) == 1:
    submission_avg, image_ids = ensemble(
        submissions_all, [0], [1.0], target_cols=TARGET_COLS
    )
    make_submission_file(
        submission_avg, image_ids, out_path="submission.csv", target_cols=TARGET_COLS
    )
else:
    submission_avg, image_ids = train_and_predict_baseline(
        train_csv_path=TRAIN_CSV_PATH,
        test_csv_path=TEST_CSV_PATH,
        images_dir=IMAGES_DIR,
        target_cols=TARGET_COLS,
    )
    make_submission_file(
        submission_avg, image_ids, out_path="submission.csv", target_cols=TARGET_COLS
    )
    print(
        "No external submissions found; trained baseline model and wrote submission.csv"
    )
