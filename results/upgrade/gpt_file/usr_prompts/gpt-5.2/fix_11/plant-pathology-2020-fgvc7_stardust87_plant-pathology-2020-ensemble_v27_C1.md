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

0.49516

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The error happens because `/kaggle/input/submissions/` doesn’t exist in this environment, so `submissions_all` is empty and indexing `[0,2]` crashes. I make the input discovery robust by searching the provided dataset folders for any `*.csv` that look like valid submissions (have `image_id` and the 4 target columns), and if none are found, I still produce a valid `submission.csv` by using `sample_submission.csv` with safe default probabilities. I also add minimal validation so mismatched rows/columns don’t silently create an invalid file. This keeps the “ensemble of existing submissions” logic intact when such files exist, while guaranteeing an end-to-end run that writes a proper `.csv`.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score comes from writing an almost-constant submission (0.25 for all labels), which yields near-random ROC AUC. To move toward the 0.9663 target without changing the core “ensemble existing submissions” logic, I (1) prevent accidentally ensembling non-submission CSVs (like `sample_submission.csv`) by ranking/choosing candidates more safely, (2) if no real submissions exist, switch the fallback from uniform 0.25 to label-prior probabilities computed from `train.csv` (still legitimate, and typically much better than random for AUC), and (3) align predictions to `test.csv` `image_id` order to avoid silent row misalignment hurting the score. These are minimal, metric-relevant changes that keep the overall approach intact and always produce a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score indicates your submission is effectively uninformative; the main issue is that when no real model submissions exist to ensemble, your fallback uses constant class priors (which still won’t reach your 0.966 target). To move toward the target while keeping the “no training / no image modeling” core logic, I keep your discovery+ensemble approach but improve the fallback to a stronger, still-legitimate baseline: a stratified CV out-of-fold (OOF) multi-output logistic regression on `image_id` character n-grams (a common “cheaty baseline” for this dataset because IDs correlate with acquisition conditions). I also ensure perfect row alignment to `test.csv` and keep ensembling behavior unchanged when real submission-like CSVs are present. This is the smallest change that plausibly lifts AUC substantially without introducing any deep learning or image feature extraction.'
- What this solution (achieved 0.5) has done: 'Your 0.5 score strongly suggests the fallback path is still producing near-constant or misaligned predictions, which is uninformative for mean ROC AUC. I keep your overall “ensemble if available, else ID-ngram logistic regression fallback” logic, but make two minimal, score-critical fixes: (1) align every candidate submission to `test.csv` by `image_id` before ensembling (preventing silent row-order corruption), and (2) make the ID-ngram fallback actually run by removing the hard dependency on scikit-learn (not listed in your installed packages) and replacing it with a tiny NumPy ridge regression on the TF-IDF char n-gram features (same idea: `image_id` n-grams → non-constant probabilities). These changes are directly metric-relevant, keep the pipeline structure intact, and still always write a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score strongly indicates the fallback path is still effectively uninformative (near-constant) rather than properly learning signal; the biggest issue is that the current fallback depends on `sklearn`, which isn’t available in your environment, so it can’t actually run and you end up with weak priors/flat predictions. I keep your exact pipeline/logic (discover → rank → ensemble else fallback) but replace the sklearn TF‑IDF with a tiny pure-Python char n‑gram hashing TF‑IDF and keep the same closed-form ridge multi-target predictor, so the fallback becomes genuinely non-constant and AUC should move toward your target. I also add a numerically-safe sigmoid and ensure all candidate submissions are aligned to `test.csv` before variance-ranking/ensembling to avoid silent row-order corruption. These are minimal, metric-relevant changes and still always write a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'Your 0.5 score is consistent with the fallback producing almost-constant or effectively random predictions; the biggest culprit here is that Python’s built-in `hash()` is randomized per process, so your char n-gram hashing features change each run and can collapse signal, plus you accidentally build different TF‑IDF spaces for train vs test by computing IDF separately. I keep your overall “ensemble if available else ID‑ngram fallback” logic intact, but make the hashing deterministic (stable md5-based hash) and compute a single shared IDF on the combined train+test texts, then split back—this preserves the same approach while making the fallback much more informative (and thus moving AUC toward your target). I also ensure the ridge model uses the same feature dimension and that probabilities are clipped safely. No model architecture/training loop changes are introduced beyond these minimal, metric-relevant fixes.'
- What this solution (achieved 0.5) has done: 'Your 0.5 score suggests the fallback path is still producing almost-constant/uninformative predictions. To move toward the 0.9663 target without changing the overall pipeline (discover → rank → ensemble else fallback), I strengthen only the fallback by adding a tiny K-fold out-of-fold calibration step (still the same ridge-on-ID-ngrams model) and then fitting once on full data for test predictions. I also fix a subtle feature bug by using a signed hashing trick (so collisions don’t only increase counts) and make the ridge solution more numerically stable. These changes keep the core idea intact (no images, no deep model), but should produce much more rank-informative probabilities and increase mean ROC AUC versus near-constant outputs.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 AUC indicates the fallback is still effectively uninformative; the most likely cause is that the ID‑ngram ridge model is learning almost nothing because it uses only signed counts (no TF) and then applies a potentially harmful OOF “calibration” via least-squares (can invert ranking and hurt AUC). I keep the same overall pipeline (discover → rank → ensemble else fallback) and the same ridge-on-hashed-char-ngram idea, but make the fallback strictly more rank-informative by (1) using proper hashed TF + global IDF + log(TF) TF‑IDF and (2) removing the affine calibration step (AUC is rank-based and this step can degrade ranking). I also ensure the ensemble output is a true weighted average (divide by sum of weights), which avoids accidental scaling/clipping that can flatten predictions. These are minimal, metric-relevant changes and still always write a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'Your 0.5 score strongly suggests the fallback is still too close to uninformative (flat-ish) predictions for a rank-based metric like mean ROC AUC. I keep your exact pipeline (discover→rank→ensemble else fallback) and the same ID‑ngram hashed TF‑IDF + ridge + sigmoid core, but make two minimal, score-relevant fixes: (1) add per-class centering of targets and an intercept handled in closed-form (so ridge isn’t forced to explain label means through sparse hashed features), and (2) add a small monotone “temperature” scaling (power transform) on the final probabilities to increase spread without changing rankings much, which often improves AUC when outputs are overly squashed around 0.5. I also ensure the ensemble path uses the same post-processing so both paths behave consistently. These changes are lightweight, deterministic, and keep the evaluation semantics intact while plausibly lifting AUC toward your target.'
- What this solution (achieved 0.49516) has done: 'Your current 0.5 AUC indicates the output is still essentially uninformative; the most likely cause is that the ID‑ngram fallback’s ridge is producing logits near the mean and the extra probability “temperature” is not helping ranking. I keep your exact pipeline (discover → rank → ensemble else fallback) and the same hashed char n‑gram TF‑IDF + ridge + sigmoid core, but make two minimal, metric-relevant fixes: (1) switch the ridge to a numerically safer primal form (solve in sample space) when features >> samples, which improves the learned ranking signal without changing the model family, and (2) remove temperature scaling (monotone but can amplify ties/saturation) and instead apply a tiny per-class epsilon jitter based on a stable hash of `image_id` to break exact probability ties (helps ROC AUC). I also ensure we never accidentally include `sample_submission.csv` as an ensemble candidate and keep strict alignment to `test.csv` order.'

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

    candidates = [
        p for p in candidates if os.path.basename(p).lower() != "sample_submission.csv"
    ]

    candidates = sorted(list(dict.fromkeys(candidates)))
    return candidates


submissions_all = find_candidate_submissions()
print("Found candidate submission-like CSVs:")
print(submissions_all)



## === cell 1
import hashlib


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

    wsum = float(np.sum(weights))
    if not np.isfinite(wsum) or wsum <= 0:
        raise ValueError(f"Invalid weights: sum(weights) must be > 0, got {wsum}")

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

        submission_with_weight.append(submission * float(weights[i]))

    submission_avg = sum(submission_with_weight) / wsum
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


def _stable_hash_bucket_and_sign(s, n_features):
    h = hashlib.md5(s.encode("utf-8")).digest()
    v = int.from_bytes(h[:4], byteorder="little", signed=False)
    bucket = v % n_features
    sign = 1.0 if (h[4] & 1) == 0 else -1.0
    return bucket, sign


def _hashing_char_ngram_tfidf_fit_transform(texts, ngram_range=(2, 5), n_features=8192):
    """
    Deterministic hashed TF-IDF with signed hashing + log1p(TF) and global IDF.
    """
    N = len(texts)
    X_tf = np.zeros((N, n_features), dtype=np.float32)
    df = np.zeros(n_features, dtype=np.int32)

    for i, s in enumerate(texts):
        s = str(s)
        seen = set()
        for n in range(ngram_range[0], ngram_range[1] + 1):
            if len(s) < n:
                continue
            for j in range(len(s) - n + 1):
                ng = s[j : j + n]
                b, sign = _stable_hash_bucket_and_sign(ng, n_features)
                X_tf[i, b] += sign
                seen.add(b)
        for b in seen:
            df[b] += 1

    X = np.sign(X_tf) * np.log1p(np.abs(X_tf))

    idf = np.log((1.0 + N) / (1.0 + df.astype(np.float64))) + 1.0
    X = X.astype(np.float64, copy=False) * idf.astype(np.float64, copy=False)

    norms = np.sqrt((X * X).sum(axis=1, keepdims=True)) + 1e-12
    X = X / norms
    return X.astype(np.float64, copy=False)


def _ridge_multitarget_predict_proba(X_train, Y_train, X_test, l2=5.0):
    """
    Change rationale (score-relevant, minimal): with ~1638 samples and ~8193 features,
    solving (X^T X) can be ill-conditioned; solving in sample space (XX^T) is numerically
    safer and yields more informative rankings for AUC while keeping the same ridge model.
    Uses explicit centering (intercept) and returns sigmoid probabilities.
    """
    X_train = X_train.astype(np.float64, copy=False)
    Y_train = Y_train.astype(np.float64, copy=False)
    X_test = X_test.astype(np.float64, copy=False)

    y_mean = Y_train.mean(axis=0, keepdims=True)
    Yc = Y_train - y_mean

    n = X_train.shape[0]
    K = X_train @ X_train.T
    K = (K + K.T) * 0.5
    K.flat[:: n + 1] += float(l2)

    alpha = np.linalg.solve(K, Yc)  # (n, C)
    W = X_train.T @ alpha  # (d, C)

    logits = (X_test @ W) + y_mean
    probs = _sigmoid_stable(logits)
    return probs


def _tiny_stable_jitter_probs(image_ids, probs, eps=2e-4):
    """
    Change rationale (score-relevant, minimal): ROC AUC suffers when many predictions are
    exactly tied (common with hashed linear models + sigmoid). Add a deterministic, very
    small per-row/per-class jitter based on image_id hash to break ties without changing
    overall calibration materially.
    """
    image_ids = [str(x) for x in image_ids]
    P = np.asarray(probs, dtype=np.float64)
    J = np.zeros_like(P, dtype=np.float64)
    for i, s in enumerate(image_ids):
        h = hashlib.md5(s.encode("utf-8")).digest()
        for k in range(P.shape[1]):
            v = int.from_bytes(h[k : k + 2], "little", signed=False)
            u = (v / 65535.0) - 0.5
            J[i, k] = u
    P = np.clip(P + eps * J, 1e-6, 1.0 - 1e-6)
    return P


def _id_ngram_fallback_predictions():
    train_df = _load_train_df()
    test_ids = _load_test_ids()
    if train_df is None or test_ids is None:
        return None

    X_text_train = train_df["image_id"].astype(str).values
    y = train_df[TARGET_COLS].astype(int).values
    X_text_test = np.array(test_ids, dtype=str)

    all_text = np.concatenate([X_text_train, X_text_test], axis=0)
    X_all = _hashing_char_ngram_tfidf_fit_transform(
        all_text, ngram_range=(2, 5), n_features=8192
    )
    X_tr = X_all[: len(X_text_train)]
    X_te = X_all[len(X_text_train) :]

    X_tr = np.hstack([X_tr, np.ones((X_tr.shape[0], 1), dtype=X_tr.dtype)])
    X_te = np.hstack([X_te, np.ones((X_te.shape[0], 1), dtype=X_te.dtype)])

    test_pred = _ridge_multitarget_predict_proba(X_tr, y, X_te, l2=5.0)

    test_pred = _tiny_stable_jitter_probs(test_ids, test_pred, eps=2e-4)

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
            submission_df.loc[:, TARGET_COLS] = priors
    else:
        if submission_avg.shape[0] != submission_df.shape[0] or submission_avg.shape[
            1
        ] != len(TARGET_COLS):
            raise ValueError(
                f"Prediction shape {submission_avg.shape} does not match expected "
                f"({submission_df.shape[0]}, {len(TARGET_COLS)})."
            )

        if test_ids is not None:
            submission_avg = _tiny_stable_jitter_probs(
                test_ids, submission_avg, eps=1e-4
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
