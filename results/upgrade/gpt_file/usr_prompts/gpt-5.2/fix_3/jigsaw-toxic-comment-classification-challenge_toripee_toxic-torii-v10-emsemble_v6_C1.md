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
Given a dataset of comments from Wikipedia's talk page edits, predict the probability of each comment being toxic.

## Metric
Mean column-wise ROC AUC; the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict a probability for each of the six possible types of comment toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate). The columns must be in the same order as shown below. The file should contain a header and have the following format:

```
id,toxic,severe_toxic,obscene,threat,insult,identity_hate
00001cee341fdb12,0.5,0.5,0.5,0.5,0.5,0.5
0000247867823ef7,0.5,0.5,0.5,0.5,0.5,0.5
etc.
```

## Dataset 
- **train.csv** - the training set, contains comments with their binary labels
- **test.csv** - the test set, you must predict the toxicity probabilities for these comments.
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.14

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
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        input/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        working/
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
```

-> data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/jigsaw-toxic-comment-classification-challenge/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-toxic-comment-classification-challenge/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> (stopped after 10 files for performance)

# 5. Target score

0.9856548304084284

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your code currently can’t yield a Kaggle score because it depends on external `/kaggle/input/...` submissions that are not present in the provided environment, so no valid ensemble file is produced. I keep the same “blend multiple model submissions” core idea, but make it robust by (1) automatically searching the available `/kaggle/input` and `/kaggle/data` trees for any `submission*.csv` files, (2) validating/aligning them to the official `sample_submission.csv` by `id`, and (3) blending whatever valid submissions are found (or falling back to a safe baseline using the sample submission if none are found) so a valid `.csv` is always written. This unblock submission generation and give you a legitimate score (likely improved vs. no-submission), while preserving the ensemble semantics and not changing any ML training logic.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score indicates you’re effectively submitting constant 0.5 predictions (i.e., no valid external submissions were found to ensemble). To move toward the 0.9856 target while keeping the same “blend submissions” core logic, I add a minimal fallback model that trains a simple TF‑IDF + LogisticRegression multi-output classifier on `train.csv` and generates real probabilities for `test.csv` only when no valid `submission*.csv` files are available. This preserves your primary ensembling behavior and only activates training when needed, producing a legitimate, much higher AUC submission. I also make path discovery use the provided `/kaggle/data/...` tree reliably and keep the output schema aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import pandas as pd

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

SAMPLE_PATHS = [
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv",
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip",
    "/kaggle/data/sample_submission.csv",
    "/kaggle/data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv",
    "/kaggle/data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip",
]


def read_sample_submission():
    for p in SAMPLE_PATHS:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(
        f"sample_submission.csv(.zip) not found in expected locations: {SAMPLE_PATHS}"
    )


sample_sub = read_sample_submission()

expected_cols = ["id"] + label_cols
missing = [c for c in expected_cols if c not in sample_sub.columns]
if missing:
    raise ValueError(
        f"sample_submission missing columns: {missing}. Found columns: {list(sample_sub.columns)}"
    )

print(
    f"Loaded sample_submission: shape={sample_sub.shape}, columns={list(sample_sub.columns)}"
)



## === cell 1
print("Searching for available submission CSVs in /kaggle/input and /kaggle/data ...")

search_roots = ["/kaggle/input", "/kaggle/data"]
candidate_paths = []
for root in search_roots:
    if os.path.exists(root):
        candidate_paths.extend(
            glob.glob(os.path.join(root, "**", "submission*.csv"), recursive=True)
        )
        candidate_paths.extend(
            glob.glob(os.path.join(root, "**", "*submission*.csv"), recursive=True)
        )

seen = set()
candidate_paths_unique = []
for p in candidate_paths:
    if p not in seen:
        seen.add(p)
        candidate_paths_unique.append(p)

print(f"Found {len(candidate_paths_unique)} candidate CSV(s).")


def load_and_align_submission(path, sample_ids, label_cols):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f" - Skipped (read error): {path} -> {e}")
        return None

    if "id" not in df.columns:
        return None

    if not all(c in df.columns for c in label_cols):
        return None

    df_small = df[["id"] + label_cols].copy()
    df_small = df_small.drop_duplicates("id", keep="last")
    df_small = df_small.set_index("id")

    coverage = df_small.index.isin(sample_ids).mean()
    if coverage < 0.95:
        return None

    aligned = df_small.reindex(sample_ids)
    if aligned.isna().any().any():
        return None

    aligned[label_cols] = aligned[label_cols].clip(0.0, 1.0)

    return aligned[label_cols]


sample_ids = sample_sub["id"].tolist()

valid_submissions = []
valid_paths = []
for p in candidate_paths_unique:
    aligned = load_and_align_submission(p, sample_ids, label_cols)
    if aligned is not None:
        valid_submissions.append(aligned)
        valid_paths.append(p)

print(f"Valid submissions found for ensembling: {len(valid_submissions)}")
for i, p in enumerate(valid_paths[:20]):
    print(f" {i+1:02d}: {p}")
if len(valid_paths) > 20:
    print(f" ... and {len(valid_paths) - 20} more")



## === cell 2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

TRAIN_PATHS = [
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv",
    "/kaggle/data/train.csv",
    "/kaggle/data/jigsaw-toxic-comment-classification-challenge/train.csv",
]
TEST_PATHS = [
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv",
    "/kaggle/data/test.csv",
    "/kaggle/data/jigsaw-toxic-comment-classification-challenge/test.csv",
]


def read_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(f"File not found in expected locations: {paths}")


def train_tfidf_lr_and_predict(sample_sub, label_cols):
    train_df = read_first_existing(TRAIN_PATHS)
    test_df = read_first_existing(TEST_PATHS)

    if "comment_text" not in train_df.columns or "comment_text" not in test_df.columns:
        raise ValueError("Expected column 'comment_text' missing in train/test.")

    X_train_text = train_df["comment_text"].fillna("").astype(str)
    y_train = train_df[label_cols].astype(int)
    X_test_text = test_df["comment_text"].fillna("").astype(str)

    vect = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        strip_accents="unicode",
        lowercase=True,
        sublinear_tf=True,
        max_features=200000,
    )
    Xtr = vect.fit_transform(X_train_text)
    Xte = vect.transform(X_test_text)

    base_lr = LogisticRegression(
        solver="liblinear",
        C=4.0,
        max_iter=100,
        random_state=42,
    )
    clf = MultiOutputClassifier(base_lr, n_jobs=1)
    clf.fit(Xtr, y_train)

    probs = []
    for est in clf.estimators_:
        p = est.predict_proba(Xte)[:, 1]
        probs.append(p)

    pred = pd.DataFrame({c: probs[i] for i, c in enumerate(label_cols)})
    pred.insert(0, "id", test_df["id"].values)

    pred = pred.drop_duplicates("id", keep="last").set_index("id")
    pred = pred.reindex(sample_sub["id"].values)
    if pred.isna().any().any():
        raise ValueError(
            "Fallback model prediction alignment produced NaNs (id mismatch)."
        )

    pred[label_cols] = pred[label_cols].clip(0.0, 1.0)
    return pred.reset_index()




## === cell 3
out = sample_sub.copy()

if len(valid_submissions) >= 1:
    blend = sum(valid_submissions) / float(len(valid_submissions))

    weights = None
    lower_paths = [p.lower() for p in valid_paths]
    key_to_w = {
        "bart": 0.55,
        "tfidf": 0.20,
        "gru": 0.10,
        "lstm": 0.10,
        "feature": 0.05,
        "features": 0.05,
    }

    chosen = []
    chosen_w = []
    for key, w in key_to_w.items():
        for idx, lp in enumerate(lower_paths):
            if key in lp:
                chosen.append(valid_submissions[idx])
                chosen_w.append(w)
                break

    if len(chosen) >= 2:
        s = sum(chosen_w)
        if s > 0:
            weights = [w / s for w in chosen_w]

    if weights is not None:
        blend = sum(m * w for m, w in zip(chosen, weights))

    out[label_cols] = blend.values
else:
    pred_df = train_tfidf_lr_and_predict(sample_sub, label_cols)
    out = pred_df[["id"] + label_cols].copy()

out[label_cols] = out[label_cols].clip(0.0, 1.0)

out_path = "submission.csv"
out.to_csv(out_path, index=False)

print(f"Wrote submission to: {out_path}")
print(out.head())
print(out.describe(include="all"))
