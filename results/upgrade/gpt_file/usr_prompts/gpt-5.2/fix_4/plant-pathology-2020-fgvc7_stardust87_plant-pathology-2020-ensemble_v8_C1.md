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

0.9697621918404656

# 6. Current score

0.70454

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The crash happens because the notebook assumes `/kaggle/input/submissions/submissions/` exists and contains at least two CSVs to ensemble; in this environment it doesn’t, so `submissions_all` is empty and indexing fails. I keep the ensembling logic, but add a safe fallback: if no external submissions are found, build a simple valid baseline submission from `sample_submission.csv` (uniform class probabilities), ensuring the pipeline always writes `submission.csv`. I also make the ensembling function validate indices/weights and normalize weights so the averaged probabilities stay in a sensible range. This run end-to-end and produce a correctly formatted `.csv` submission file.'
- What this solution (achieved 0.68088) has done: 'Your current 0.5 score comes from writing a uniform-probability fallback submission because no external submissions exist; to move toward the 0.9697 target, we need a real model prediction while keeping the “ensemble if available, else fallback” core flow intact. I add a minimal classical ML baseline that trains on the provided `train.csv` labels using simple image color statistics (no deep learning, no new dependencies) and predicts probabilities for `test.csv`. This should substantially lift ROC AUC above 0.5 while remaining fast (<600s) and producing a correctly formatted `submission.csv`. The existing ensembling logic stays; we just replace the uniform fallback with this learned baseline.'
- What this solution (achieved 0.70454) has done: 'Your current baseline is limited by very low-capacity features; to move the score upward toward 0.9697 without changing the overall approach (extract simple per-image features → OneVsRest LogisticRegression → predict_proba → submission), I minimally strengthen the feature extractor while keeping it lightweight and deterministic. Specifically, I (1) increase resize resolution modestly, (2) add a small set of additional robust features (HSV moments, coarse spatial grid means, and simple color histograms) that significantly improve separability for these leaf disease classes, and (3) switch the scaler to `RobustScaler` to reduce sensitivity to outliers, while keeping the same classifier family and training flow. The ensembling path remains unchanged; only the learned-fallback baseline is improved, and it still writes a valid `submission.csv` with the required columns and test ordering.'

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"

DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TEST_CSV_PATH = os.path.join(DATA_DIR, "test.csv")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "images")



## === cell 2
submissions_all = []
if os.path.exists(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print(submissions_all)




## === cell 3
def ensemble(submissions_all, sub_idx, weights=None):
    """
    Weighted average ensemble of submission files.
    Bugfixes:
      - Validate indices are in range.
      - Validate weights length.
      - Normalize weights to sum to 1 to keep probabilities calibrated.
    """
    if weights is None:
        weights = [1.0] * len(sub_idx)

    if len(sub_idx) != len(weights):
        raise ValueError(
            f"len(sub_idx)={len(sub_idx)} must equal len(weights)={len(weights)}"
        )

    if len(submissions_all) == 0:
        raise ValueError("No submissions found to ensemble.")

    for j in sub_idx:
        if j < 0 or j >= len(submissions_all):
            raise IndexError(
                f"Submission index {j} out of range for {len(submissions_all)} files."
            )

    wsum = float(sum(weights))
    if wsum <= 0:
        raise ValueError("Sum of weights must be > 0.")
    weights = [w / wsum for w in weights]

    submission_with_weight = []
    for i in range(len(sub_idx)):
        path = submissions_all[sub_idx[i]]
        print(f"I'm taking submission {path} with weight {weights[i]}")
        submission = pd.read_csv(path)

        required = ["healthy", "multiple_diseases", "rust", "scab"]
        missing = [c for c in required if c not in submission.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in {path}")

        arr = submission.loc[:, required].values
        submission_with_weight.append(arr * weights[i])

    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 4
def make_submission_file(submission_avg, template_path, out_path="submission.csv"):
    """
    Create final submission CSV.
    Bugfix: use a known-good template (sample_submission.csv) if external submissions are absent.
    """
    submission_df = pd.read_csv(template_path)

    required = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    missing = [c for c in required if c not in submission_df.columns]
    if missing:
        raise ValueError(f"Template {template_path} missing columns {missing}")

    submission_df = submission_df[required].copy()
    submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = submission_avg
    submission_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with shape {submission_df.shape}")




## === cell 5
def _extract_image_features(image_path):
    """
    Lightweight feature extractor: color/texture statistics.
    Change (score-improving, same core logic): add a few more robust, low-cost features
    (HSV moments, coarse spatial grid means, small RGB histograms) while keeping
    deterministic hand-crafted features + linear model training.
    """
    from PIL import Image
    import numpy as np

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return np.zeros(70, dtype=np.float32)

    img = img.resize((128, 128))
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    eps = 1e-6

    feats = []

    for ch in range(3):
        x = arr[:, :, ch]
        feats.append(float(x.mean()))
        feats.append(float(x.std()))

    feats.append(float((r.mean() + eps) / (g.mean() + eps)))
    feats.append(float((r.mean() + eps) / (b.mean() + eps)))
    feats.append(float((g.mean() + eps) / (b.mean() + eps)))

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    feats.append(float(gray.mean()))
    feats.append(float(gray.std()))
    h, w = gray.shape
    c0, c1 = h // 4, 3 * h // 4
    d0, d1 = w // 4, 3 * w // 4
    center = gray[c0:c1, d0:d1]
    feats.append(float(center.mean()))
    feats.append(float(center.std()))

    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx * gx + gy * gy)
    feats.append(float(grad.mean()))
    feats.append(float(grad.std()))

    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    for ch in range(3):
        x = hsv[:, :, ch]
        feats.append(float(x.mean()))
        feats.append(float(x.std()))

    h2, w2 = h // 2, w // 2
    q1 = gray[:h2, :w2].mean()
    q2 = gray[:h2, w2:].mean()
    q3 = gray[h2:, :w2].mean()
    q4 = gray[h2:, w2:].mean()
    feats.extend([float(q1), float(q2), float(q3), float(q4)])

    bins = 8
    for x in (r, g, b):
        hist, _ = np.histogram(x, bins=bins, range=(0.0, 1.0), density=True)
        feats.extend([float(v) for v in hist])

    feats.append(
        float(hsv[:, :, 1].mean())
    )  # saturation mean (redundant but stabilizes)
    feats.append(float(hsv[:, :, 2].mean()))  # value mean

    return np.asarray(feats, dtype=np.float32)


def train_and_predict_baseline(train_csv_path, test_csv_path, images_dir):
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    targets = ["healthy", "multiple_diseases", "rust", "scab"]

    X_train = np.vstack(
        [
            _extract_image_features(os.path.join(images_dir, f"{img_id}.jpg"))
            for img_id in train_df["image_id"].values
        ]
    )
    y_train = train_df[targets].values.astype(int)

    X_test = np.vstack(
        [
            _extract_image_features(os.path.join(images_dir, f"{img_id}.jpg"))
            for img_id in test_df["image_id"].values
        ]
    )

    clf = Pipeline(
        steps=[
            ("scaler", RobustScaler(quantile_range=(10.0, 90.0))),
            (
                "ovr",
                OneVsRestClassifier(
                    LogisticRegression(
                        solver="lbfgs",
                        C=2.0,
                        max_iter=600,
                        random_state=42,
                    ),
                    n_jobs=1,
                ),
            ),
        ]
    )

    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)

    if isinstance(proba, list):
        proba = np.vstack([p[:, 1] if p.ndim == 2 else p for p in proba]).T

    proba = np.clip(proba, 1e-6, 1 - 1e-6).astype(np.float32)
    return test_df["image_id"].values, proba




## === cell 6
if len(submissions_all) >= 2:
    submission_avg = ensemble(submissions_all, [0, 1], [0.3, 0.7])
    template_path = submissions_all[0]
    make_submission_file(submission_avg, template_path, out_path="submission.csv")
else:
    image_ids, proba = train_and_predict_baseline(
        TRAIN_CSV_PATH, TEST_CSV_PATH, IMAGES_DIR
    )

    sub = pd.read_csv(SAMPLE_SUB_PATH)
    required = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    sub = sub[required].copy()

    pred_df = pd.DataFrame(
        {
            "image_id": image_ids,
            "healthy": proba[:, 0],
            "multiple_diseases": proba[:, 1],
            "rust": proba[:, 2],
            "scab": proba[:, 3],
        }
    )

    test = pd.read_csv(TEST_CSV_PATH)
    sub = test.merge(pred_df, on="image_id", how="left")
    for c in ["healthy", "multiple_diseases", "rust", "scab"]:
        sub[c] = sub[c].astype("float32").fillna(0.25)

    sub.to_csv("submission.csv", index=False)
    print(f"No external submissions found under {SUBMISSIONS_PATH}.")
    print(f"Wrote learned baseline submission.csv with shape {sub.shape}")
