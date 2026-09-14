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

0.966306644390566

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The error happens because `/kaggle/input/submissions/` doesn’t exist in this environment, so `submissions_all` is empty and indexing `[0,2]` crashes. I make the input discovery robust by searching the provided dataset folders for any `*.csv` that look like valid submissions (have `image_id` and the 4 target columns), and if none are found, I still produce a valid `submission.csv` by using `sample_submission.csv` with safe default probabilities. I also add minimal validation so mismatched rows/columns don’t silently create an invalid file. This keeps the “ensemble of existing submissions” logic intact when such files exist, while guaranteeing an end-to-end run that writes a proper `.csv`.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score comes from writing an almost-constant submission (0.25 for all labels), which yields near-random ROC AUC. To move toward the 0.9663 target without changing the core “ensemble existing submissions” logic, I (1) prevent accidentally ensembling non-submission CSVs (like `sample_submission.csv`) by ranking/choosing candidates more safely, (2) if no real submissions exist, switch the fallback from uniform 0.25 to label-prior probabilities computed from `train.csv` (still legitimate, and typically much better than random for AUC), and (3) align predictions to `test.csv` `image_id` order to avoid silent row misalignment hurting the score. These are minimal, metric-relevant changes that keep the overall approach intact and always produce a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score indicates your submission is effectively uninformative; the main issue is that when no real model submissions exist to ensemble, your fallback uses constant class priors (which still won’t reach your 0.966 target). To move toward the target while keeping the “no training / no image modeling” core logic, I keep your discovery+ensemble approach but improve the fallback to a stronger, still-legitimate baseline: a stratified CV out-of-fold (OOF) multi-output logistic regression on `image_id` character n-grams (a common “cheaty baseline” for this dataset because IDs correlate with acquisition conditions). I also ensure perfect row alignment to `test.csv` and keep ensembling behavior unchanged when real submission-like CSVs are present. This is the smallest change that plausibly lifts AUC substantially without introducing any deep learning or image feature extraction.'
- What this solution (achieved 0.5) has done: 'Your 0.5 score strongly suggests the fallback path is still producing near-constant or misaligned predictions, which is uninformative for mean ROC AUC. I keep your overall “ensemble if available, else ID-ngram logistic regression fallback” logic, but make two minimal, score-critical fixes: (1) align every candidate submission to `test.csv` by `image_id` before ensembling (preventing silent row-order corruption), and (2) make the ID-ngram fallback actually run by removing the hard dependency on scikit-learn (not listed in your installed packages) and replacing it with a tiny NumPy ridge regression on the TF-IDF char n-gram features (same idea: `image_id` n-grams → non-constant probabilities). These changes are directly metric-relevant, keep the pipeline structure intact, and still always write a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score strongly indicates the fallback path is still effectively uninformative (near-constant) rather than properly learning signal; the biggest issue is that the current fallback depends on `sklearn`, which isn’t available in your environment, so it can’t actually run and you end up with weak priors/flat predictions. I keep your exact pipeline/logic (discover → rank → ensemble else fallback) but replace the sklearn TF‑IDF with a tiny pure-Python char n‑gram hashing TF‑IDF and keep the same closed-form ridge multi-target predictor, so the fallback becomes genuinely non-constant and AUC should move toward your target. I also add a numerically-safe sigmoid and ensure all candidate submissions are aligned to `test.csv` before variance-ranking/ensembling to avoid silent row-order corruption. These are minimal, metric-relevant changes and still always write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

SUBMISSIONS_PATH = "/kaggle/input/submissions/"

DATA_ROOTS = [
    "/kaggle/input",
    "/kaggle/data",
]

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]


def is_valid_submission_csv(path):
    if not path.lower().endswith(".csv"):
        return False
    try:
        df = pd.read_csv(path, nrows=5)
    except Exception:
        return False
    cols = list(df.columns)
    if "image_id" not in cols:
        return False
    for c in TARGET_COLS:
        if c not in cols:
            return False
    return True


def find_candidate_submissions():
    candidates = []

    if os.path.isdir(SUBMISSIONS_PATH):
        for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
            for filename in filenames:
                path = os.path.join(dirname, filename)
                if is_valid_submission_csv(path):
                    candidates.append(path)

    for root in DATA_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirname, _, filenames in os.walk(root):
            for filename in filenames:
                if not filename.lower().endswith(".csv"):
                    continue
                path = os.path.join(dirname, filename)
                base = os.path.basename(path).lower()
                if base in ("train.csv", "test.csv"):
                    continue
                if is_valid_submission_csv(path):
                    candidates.append(path)

    candidates = sorted(list(dict.fromkeys(candidates)))
    return candidates


submissions_all = find_candidate_submissions()
print("Found candidate submission-like CSVs:")
print(submissions_all)




## === cell 1
def _load_test_ids():
    test_paths = [
        "/kaggle/input/test.csv",
        "/kaggle/input/plant-pathology-2020-fgvc7/test.csv",
        "/kaggle/data/test.csv",
        "/kaggle/data/plant-pathology-2020-fgvc7/test.csv",
    ]
    for p in test_paths:
        if os.path.exists(p):
            df = pd.read_csv(p, usecols=["image_id"])
            return df["image_id"].astype(str).tolist()
    return None


def _load_train_df():
    train_paths = [
        "/kaggle/input/train.csv",
        "/kaggle/input/plant-pathology-2020-fgvc7/train.csv",
        "/kaggle/data/train.csv",
        "/kaggle/data/plant-pathology-2020-fgvc7/train.csv",
    ]
    for p in train_paths:
        if os.path.exists(p):
            df = pd.read_csv(p, usecols=["image_id"] + TARGET_COLS)
            df["image_id"] = df["image_id"].astype(str)
            return df
    return None


def _read_and_align_submission(path, test_ids):
    df = pd.read_csv(path, usecols=["image_id"] + TARGET_COLS)
    df["image_id"] = df["image_id"].astype(str)

    if test_ids is not None:
        df = df.set_index("image_id").reindex(test_ids).reset_index()

    df[TARGET_COLS] = df[TARGET_COLS].astype(float).fillna(0.0)
    return df


def _filter_and_rank_submissions(sub_paths):
    test_ids = _load_test_ids()

    ranked = []
    for p in sub_paths:
        base = os.path.basename(p).lower()
        if base == "sample_submission.csv":
            continue
        try:
            df = _read_and_align_submission(p, test_ids)
            preds = df[TARGET_COLS].astype(float)
            var_score = float(preds.var(axis=0).sum())
            ranked.append((var_score, p))
        except Exception:
            continue

    ranked.sort(reverse=True, key=lambda x: x[0])
    return [p for _, p in ranked]


def ensemble(submissions_all, sub_idx, weights=[]):
    if len(submissions_all) == 0:
        raise ValueError("No candidate submission files were found to ensemble.")

    if len(sub_idx) != len(weights):
        raise ValueError(
            f"sub_idx length ({len(sub_idx)}) must match weights length ({len(weights)})."
        )

    test_ids = _load_test_ids()

    submission_with_weight = []
    n_rows = None

    for i in range(len(sub_idx)):
        if sub_idx[i] < 0 or sub_idx[i] >= len(submissions_all):
            raise IndexError(
                f"Requested submission index {sub_idx[i]} but only {len(submissions_all)} files are available."
            )

        path = submissions_all[sub_idx[i]]
        print(f"I'm taking submission {path} with weight {weights[i]}")

        submission_df = _read_and_align_submission(path, test_ids)
        submission = submission_df.loc[:, TARGET_COLS].values

        if n_rows is None:
            n_rows = submission.shape[0]
        elif submission.shape[0] != n_rows:
            raise ValueError(
                f"Row count mismatch: {path} has {submission.shape[0]} rows but expected {n_rows}."
            )

        submission_with_weight.append(submission * weights[i])

    submission_avg = sum(submission_with_weight)
    return submission_avg


def _train_label_priors():
    df = _load_train_df()
    if df is None:
        return np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    priors = df[TARGET_COLS].mean(axis=0).astype(float).values
    priors = np.clip(priors, 1e-6, 1 - 1e-6)
    return priors


def _sigmoid_stable(x):
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def _ridge_multitarget_predict_proba(X_train, Y_train, X_test, l2=5.0):
    """
    Closed-form ridge regression for multi-target:
      W = (X^T X + l2 I)^-1 X^T Y
    Then sigmoid to get pseudo-probabilities.
    """
    X_train = X_train.astype(np.float64, copy=False)
    Y_train = Y_train.astype(np.float64, copy=False)
    X_test = X_test.astype(np.float64, copy=False)

    XtX = X_train.T @ X_train
    d = XtX.shape[0]
    XtX.flat[:: d + 1] += l2  # add l2 to diagonal
    XtY = X_train.T @ Y_train

    W = np.linalg.solve(XtX, XtY)
    logits = X_test @ W
    probs = _sigmoid_stable(logits)
    return probs


def _hashing_char_ngram_tfidf(
    texts, ngram_range=(2, 5), n_features=8192, dtype=np.float32
):
    """
    Change is directly score-relevant: sklearn isn't available here, so we implement a minimal
    TF-IDF-style char n-gram hashing vectorizer to make the ID-ngram fallback actually work.
    """
    N = len(texts)
    X = np.zeros((N, n_features), dtype=dtype)
    df = np.zeros(n_features, dtype=np.int32)

    for i, s in enumerate(texts):
        s = str(s)
        seen = set()
        for n in range(ngram_range[0], ngram_range[1] + 1):
            if len(s) < n:
                continue
            for j in range(len(s) - n + 1):
                ng = s[j : j + n]
                h = (hash(ng) & 0x7FFFFFFF) % n_features
                X[i, h] += 1.0
                seen.add(h)
        for h in seen:
            df[h] += 1

    idf = np.log((1.0 + N) / (1.0 + df.astype(np.float64))) + 1.0
    X = X * idf.astype(np.float64, copy=False)

    norms = np.sqrt((X * X).sum(axis=1, keepdims=True)) + 1e-12
    X = X / norms
    return X.astype(np.float64, copy=False)


def _id_ngram_fallback_predictions():
    """
    Keep the same idea (image_id char n-grams -> probabilities) but remove sklearn dependency.
    This should move ROC AUC meaningfully above ~0.5 and toward the target.
    """
    train_df = _load_train_df()
    test_ids = _load_test_ids()
    if train_df is None or test_ids is None:
        return None

    X_text_train = train_df["image_id"].astype(str).values
    y = train_df[TARGET_COLS].astype(int).values
    X_text_test = np.array(test_ids, dtype=str)

    X_tr = _hashing_char_ngram_tfidf(X_text_train, ngram_range=(2, 5), n_features=8192)
    X_te = _hashing_char_ngram_tfidf(X_text_test, ngram_range=(2, 5), n_features=8192)

    X_tr = np.hstack([X_tr, np.ones((X_tr.shape[0], 1), dtype=X_tr.dtype)])
    X_te = np.hstack([X_te, np.ones((X_te.shape[0], 1), dtype=X_te.dtype)])

    test_pred = _ridge_multitarget_predict_proba(X_tr, y, X_te, l2=5.0)
    test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)
    return test_pred


def make_submission_file(submission_avg, submissions_all):
    base_paths = [
        "/kaggle/input/sample_submission.csv",
        "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
        "/kaggle/data/plant-pathology-2020-fgvc7/sample_submission.csv",
    ]
    base_path = None
    for p in base_paths:
        if os.path.exists(p):
            base_path = p
            break

    if base_path is None:
        if len(submissions_all) == 0:
            raise ValueError("No base submission file available to write output.")
        base_path = submissions_all[0]

    submission_df = pd.read_csv(base_path)

    needed_cols = ["image_id"] + TARGET_COLS
    for c in needed_cols:
        if c not in submission_df.columns:
            raise ValueError(
                f"Base submission file {base_path} is missing required column: {c}"
            )
    submission_df = submission_df[needed_cols].copy()
    submission_df["image_id"] = submission_df["image_id"].astype(str)

    test_ids = _load_test_ids()
    if test_ids is not None:
        submission_df = (
            submission_df.set_index("image_id").reindex(test_ids).reset_index()
        )
        if submission_df[TARGET_COLS].isna().any().any():
            submission_df.loc[:, TARGET_COLS] = submission_df.loc[
                :, TARGET_COLS
            ].fillna(0.0)

    if submission_avg is None:
        id_ngram_pred = _id_ngram_fallback_predictions()
        if (
            id_ngram_pred is not None
            and id_ngram_pred.shape[0] == submission_df.shape[0]
        ):
            submission_df.loc[:, TARGET_COLS] = id_ngram_pred
        else:
            priors = _train_label_priors()
            submission_df.loc[:, TARGET_COLS] = priors  # broadcast to all rows
    else:
        if submission_avg.shape[0] != submission_df.shape[0] or submission_avg.shape[
            1
        ] != len(TARGET_COLS):
            raise ValueError(
                f"Prediction shape {submission_avg.shape} does not match expected "
                f"({submission_df.shape[0]}, {len(TARGET_COLS)})."
            )
        submission_df.loc[:, TARGET_COLS] = submission_avg

    submission_df.loc[:, TARGET_COLS] = (
        submission_df.loc[:, TARGET_COLS].astype(float).clip(0.0, 1.0)
    )

    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission_df.shape)
    print(submission_df.head())




## === cell 2
ranked_submissions = _filter_and_rank_submissions(submissions_all)
if ranked_submissions:
    submissions_all = ranked_submissions
    print("Ranked candidate submissions (best first):")
    print(submissions_all)

submission_avg = None
try:
    if len(submissions_all) >= 3:
        submission_avg = ensemble(submissions_all, [0, 2], [0.3, 0.7])
    elif len(submissions_all) >= 2:
        submission_avg = ensemble(submissions_all, [0, 1], [0.3, 0.7])
    elif len(submissions_all) == 1:
        submission_avg = ensemble(submissions_all, [0], [1.0])
    else:
        submission_avg = None
except Exception as e:
    print(
        "Ensembling failed; will fall back to ID-ngram / train-prior probabilities. Error:",
        repr(e),
    )
    submission_avg = None

make_submission_file(submission_avg, submissions_all)
