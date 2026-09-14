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

0.983552631199519

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your notebook fails because it tries to ensemble four external Kaggle Dataset submissions that are not present in your environment (`../input/improved-lstm-baseline-...`). I keep the same “blend multiple submissions” core logic, but make it robust by (1) auto-discovering any available `submission.csv` files under the provided `/kaggle/input` and `/kaggle/data` trees, and (2) falling back to a safe baseline using `sample_submission.csv` (0.5s) if none are found so a valid CSV is always produced. I also enforce correct column order, numeric types, id alignment, and fill any missing label columns to avoid silent format bugs. This run end-to-end and always write `submission.csv`.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score happens because the notebook is (correctly) falling back to the 0.5 baseline when it can’t find compatible external submission files to blend. To move the score toward your target, the smallest legitimate change is to keep the same “blend multiple submissions” core logic but add a lightweight in-notebook baseline model trained from `train.csv` and used only when no usable external submissions are found. This keeps the submission semantics the same (probabilities per class) while producing a much stronger fallback than constant 0.5. I also keep strict id alignment with `sample_submission.csv` and clip probabilities to [0,1] to avoid format/metric issues.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score indicates the pipeline is producing a valid file but the fallback is likely not being used effectively (or is too weak due to solver/regularization), so we keep the same TF‑IDF + per-label LogisticRegression fallback core logic but make two minimal, high-impact, metric-aligned improvements. First, we switch to `solver="saga"` with `n_jobs=-1` and a slightly stronger/steadier `C`, which typically improves ROC AUC for sparse TF‑IDF without changing the modeling approach. Second, we add a small amount of word-level preprocessing directly in the vectorizer (`lowercase=True` is default, plus `stop_words="english"`) and increase `max_features` modestly to capture more signal while staying within time. Everything else (file discovery/blending behavior, alignment to `sample_submission.csv`, and output format) remains the same and still always writes `submission.csv`.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score strongly suggests you’re still effectively submitting near-constant probabilities (either the fallback didn’t run on Kaggle, or the fallback model underfit due to missing/incorrect imports or suboptimal regularization). To move the score toward the 0.98355 target with minimal core-logic change, I keep the same TF‑IDF + per-label LogisticRegression fallback, but (1) add a character-level TF‑IDF in parallel and stack it with the word-level features, and (2) use class_weight="balanced" to better fit rare labels like `threat` and `identity_hate` (both are standard, metric-aligned improvements for this competition). I also make the fallback reliably available by ensuring scikit-learn imports are present and by keeping strict id alignment to `sample_submission.csv` so the submission is valid. These changes typically lift AUC substantially without changing the overall approach (still linear models on TF‑IDF features, one model per label).'
- What this solution (achieved 0.5) has done: 'Your 0.5 score indicates you’re still effectively submitting near-constant probabilities, most likely because the fallback model never runs successfully in your environment due to missing scikit-learn (your installed package list does not include `scikit-learn`). To move the score up toward the 0.98355 target while keeping the same core “fallback baseline when no external submissions exist” logic, I replace the scikit-learn TF‑IDF+LogReg fallback with a pure-pandas/NumPy Naive Bayes word model (still trained only on `train.csv` comment_text) that outputs probabilities per label. This is a minimal architectural change forced by package availability, and it generally yields a large AUC jump over 0.5 for this competition. I also keep the exact same alignment to `sample_submission.csv`, enforce column order/types, and always write a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score suggests the model is still effectively near-constant or too weak, so we keep your exact “fallback NB from train.csv when no external submissions exist” core logic but make the NB computation correct and stronger. Specifically, we compute proper per-label positive and negative word-count models (instead of deriving negatives from global counts, which is inconsistent and can flatten predictions), and we add a small log-odds prior calibration term so outputs aren’t stuck near the base rate for rare labels. We also make the text model slightly richer at essentially no conceptual cost by adding character n-grams alongside word tokens within the same NB framework (still Naive Bayes on counts), which usually improves ROC AUC on this competition. Everything else—file discovery/blending behavior, strict id alignment to `sample_submission.csv`, column order, and writing `submission.csv`—remains unchanged.'

# 9. Code solution

## === cell 0
import os
import glob
import re
import numpy as np
import pandas as pd

RANDOM_SEED = 42

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
sub_cols = ["id"] + label_cols

SEARCH_DIRS = [
    "/kaggle/input",
    "/kaggle/data",
    "../input",  # keep original relative pattern, in case it exists
    "../data",
]


def find_existing_file(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def list_submission_candidates(max_files=50):
    patterns = []
    for d in SEARCH_DIRS:
        patterns.extend(
            [
                os.path.join(d, "**", "submission.csv"),
                os.path.join(d, "**", "*submission*.csv"),
            ]
        )
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    seen = set()
    uniq = []
    for f in files:
        if f not in seen and os.path.isfile(f):
            seen.add(f)
            uniq.append(f)
    return uniq[:max_files]


def load_submission(path):
    df = pd.read_csv(path)
    if "id" not in df.columns:
        raise ValueError(f"{path} has no 'id' column")
    for c in label_cols:
        if c not in df.columns:
            df[c] = 0.5
    df = df[sub_cols].copy()
    for c in label_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        df[c] = df[c].fillna(0.5).clip(0.0, 1.0)
    df["id"] = df["id"].astype(str)
    return df




## === cell 1
sample_submission_path = find_existing_file(
    [
        "/kaggle/data/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv",
        "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv",
        "../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv",
        "../input/sample_submission.csv",
    ]
)

if sample_submission_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected Kaggle directories."
    )

sample = pd.read_csv(sample_submission_path)
if "id" not in sample.columns:
    raise ValueError("sample_submission.csv missing 'id' column")

sample["id"] = sample["id"].astype(str)
for c in label_cols:
    if c not in sample.columns:
        sample[c] = 0.5
sample = sample[sub_cols].copy()

print("Using sample_submission:", sample_submission_path)
print("Sample shape:", sample.shape)



## === cell 2
candidates = list_submission_candidates()

candidates = [
    p
    for p in candidates
    if os.path.abspath(p) != os.path.abspath(sample_submission_path)
]

print(f"Found {len(candidates)} submission candidate file(s).")
for p in candidates[:10]:
    print(" -", p)



## === cell 3
loaded = []
for p in candidates:
    try:
        df = load_submission(p)
        loaded.append((p, df))
    except Exception:
        continue

print(f"Loaded {len(loaded)} valid submission(s).")




## === cell 4
def build_fallback_from_training(sample_ids_df):
    train_path = find_existing_file(
        [
            "/kaggle/data/train.csv",
            "/kaggle/input/train.csv",
            "/kaggle/data/jigsaw-toxic-comment-classification-challenge/train.csv",
            "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv",
            "../input/jigsaw-toxic-comment-classification-challenge/train.csv",
            "../input/train.csv",
        ]
    )
    test_path = find_existing_file(
        [
            "/kaggle/data/test.csv",
            "/kaggle/input/test.csv",
            "/kaggle/data/jigsaw-toxic-comment-classification-challenge/test.csv",
            "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv",
            "../input/jigsaw-toxic-comment-classification-challenge/test.csv",
            "../input/test.csv",
        ]
    )
    if train_path is None or test_path is None:
        p = sample_ids_df.copy()
        for c in label_cols:
            p[c] = 0.5
        return p[sub_cols]

    train = pd.read_csv(train_path, usecols=["comment_text"] + label_cols)
    test = pd.read_csv(test_path, usecols=["id", "comment_text"])
    test["id"] = test["id"].astype(str)

    word_re = re.compile(r"[a-z]{2,}", re.IGNORECASE)

    def word_tokens(text):
        if not isinstance(text, str):
            return []
        return word_re.findall(text.lower())

    def char_tokens(text, n_min=3, n_max=5):
        if not isinstance(text, str):
            return []
        s = text.lower()
        s = re.sub(r"[^a-z0-9 ]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        if len(s) < n_min:
            return []
        toks = []
        L = len(s)
        for n in range(n_min, n_max + 1):
            if L >= n:
                for i in range(L - n + 1):
                    toks.append(s[i : i + n])
        return toks

    def tokenize(text):
        wt = word_tokens(text)
        ct = char_tokens(text)
        if len(ct) > 400:
            ct = ct[:400]
        return wt + ct

    max_vocab = 250000
    min_df = 3
    batch = 50000

    freq = {}
    n_train = len(train)
    for i in range(0, n_train, batch):
        texts = train["comment_text"].iloc[i : i + batch].tolist()
        for t in texts:
            for w in tokenize(t):
                freq[w] = freq.get(w, 0) + 1

    items = [(w, c) for w, c in freq.items() if c >= min_df]
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:max_vocab]
    vocab = {w: idx for idx, (w, _) in enumerate(items)}
    V = len(vocab)
    if V == 0:
        p = sample_ids_df.copy()
        for c in label_cols:
            p[c] = 0.5
        return p[sub_cols]

    alpha = 1.0  # Laplace smoothing

    pos_counts = np.zeros((len(label_cols), V), dtype=np.float64)
    neg_counts = np.zeros((len(label_cols), V), dtype=np.float64)
    pos_totals = np.zeros(len(label_cols), dtype=np.float64)
    neg_totals = np.zeros(len(label_cols), dtype=np.float64)
    pos_docs = np.zeros(len(label_cols), dtype=np.float64)

    for i in range(n_train):
        toks = tokenize(train.at[i, "comment_text"])
        if not toks:
            continue
        tc = {}
        for w in toks:
            j = vocab.get(w)
            if j is not None:
                tc[j] = tc.get(j, 0) + 1
        if not tc:
            continue

        for li, col in enumerate(label_cols):
            y = int(train.at[i, col])
            if y == 1:
                pos_docs[li] += 1.0
                for j, cnt in tc.items():
                    pos_counts[li, j] += cnt
                    pos_totals[li] += cnt
            else:
                for j, cnt in tc.items():
                    neg_counts[li, j] += cnt
                    neg_totals[li] += cnt

    prior = (pos_docs + 1.0) / (n_train + 2.0)
    log_prior_odds = np.log(prior) - np.log(1.0 - prior)

    log_pw_pos = np.log((pos_counts + alpha) / (pos_totals[:, None] + alpha * V))
    log_pw_neg = np.log((neg_counts + alpha) / (neg_totals[:, None] + alpha * V))

    preds = np.zeros((len(test), len(label_cols)), dtype=np.float64)
    for i in range(len(test)):
        toks = tokenize(test.at[i, "comment_text"])
        if toks:
            tc = {}
            for w in toks:
                j = vocab.get(w)
                if j is not None:
                    tc[j] = tc.get(j, 0) + 1
        else:
            tc = {}

        if not tc:
            preds[i, :] = prior
            continue

        js = np.fromiter(tc.keys(), dtype=np.int64)
        cnts = np.fromiter(tc.values(), dtype=np.float64)

        logit = log_prior_odds + (log_pw_pos[:, js] - log_pw_neg[:, js]) @ cnts
        preds[i, :] = 1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50)))

    p = pd.DataFrame(preds, columns=label_cols)
    p.insert(0, "id", test["id"].values)

    p = sample_ids_df[["id"]].merge(p, on="id", how="left", sort=False)
    for c in label_cols:
        p[c] = pd.to_numeric(p[c], errors="coerce").astype("float64")
        p[c] = p[c].fillna(0.5).clip(0.0, 1.0)
    return p[sub_cols]


if len(loaded) == 0:
    p_res = build_fallback_from_training(sample)
    print(
        "No valid external submissions found; using Naive Bayes fallback from train.csv."
    )
else:
    aligned = []
    for path, df in loaded:
        merged = sample[["id"]].merge(df, on="id", how="left", sort=False)
        missing = merged[label_cols].isna().mean().mean()
        if missing > 0.05:
            continue
        for c in label_cols:
            merged[c] = merged[c].fillna(0.5).clip(0.0, 1.0).astype("float64")
        aligned.append(merged[sub_cols])

    if len(aligned) == 0:
        p_res = build_fallback_from_training(sample)
        print(
            "All loaded submissions incompatible; using Naive Bayes fallback from train.csv."
        )
    else:
        p_res = sample[["id"]].copy()
        stack = np.stack(
            [a[label_cols].to_numpy(dtype=np.float64) for a in aligned], axis=0
        )
        avg = stack.mean(axis=0)
        p_res[label_cols] = avg
        print(f"Averaged {len(aligned)} submission(s).")



## === cell 5
p_res = p_res[sub_cols].copy()
p_res["id"] = p_res["id"].astype(str)

for c in label_cols:
    p_res[c] = pd.to_numeric(p_res[c], errors="coerce").astype("float64")
    p_res[c] = p_res[c].fillna(0.5).clip(0.0, 1.0)

if len(p_res) != len(sample):
    raise ValueError(
        f"Submission row count mismatch: got {len(p_res)} expected {len(sample)}"
    )

if not p_res["id"].equals(sample["id"]):
    p_res = sample[["id"]].merge(p_res, on="id", how="left", sort=False)
    for c in label_cols:
        p_res[c] = p_res[c].fillna(0.5).clip(0.0, 1.0)
    p_res = p_res[sub_cols]

print("Submission ready. Shape:", p_res.shape)
print(p_res.head())



## === cell 6
out_path = "submission.csv"
p_res.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("Columns:", list(p_res.columns))
