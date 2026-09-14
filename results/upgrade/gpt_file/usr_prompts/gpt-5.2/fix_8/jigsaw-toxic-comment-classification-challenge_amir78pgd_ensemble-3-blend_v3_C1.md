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

0.9863197044121592

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The current notebook fails because it tries to ensemble many external submission files from `../input/...` that don’t exist in this environment, so no predictions are ever created. I keep the “ensemble submissions” core idea but make it robust: automatically discover any available submission-like CSVs under the provided `/kaggle/input` and `/kaggle/data` trees, validate they contain the required columns, align them to the sample submission ids, and then average them. If none are found (likely here), I fall back to producing a valid submission by copying `sample_submission.csv` (all 0.5s), ensuring a `.csv` is always written end-to-end without errors.'
- What this solution (achieved 0.5) has done: 'Your current score is 0.5 because the notebook is effectively outputting the sample submission (all 0.5s) since it can’t reliably find any valid external submissions to ensemble. To move the score toward the 0.986 target while preserving your “simple pipeline” spirit, I keep the submission-writing flow but replace the unreliable external-CSV ensembling fallback with a minimal, self-contained baseline model trained from the provided `train.csv` and applied to `test.csv`. This uses a standard scikit-learn TF-IDF + one-vs-rest Logistic Regression setup, which matches the ROC-AUC metric well and should substantially increase score with minimal conceptual change (still “average of independent label probability models”). The code still writes a valid `submission.csv` with the required columns and correct row alignment.'
- What this solution (achieved 0.5) has done: 'Your 0.5 score indicates you’re effectively submitting near-constant predictions; the most likely cause here is a silent misalignment/format issue between `p_res` and the required `sample_submission` ids (the current `merge(..., how="right")` can create NaNs/row-order differences that collapse performance). I keep your exact core approach (TF‑IDF + per-label LogisticRegression; or mean-ensemble if external submissions exist) but make the id alignment deterministic by reindexing to `sample_sub["id"]` and filling any missing predictions with safe priors (train label means). I also ensure the test-set length matches the sample submission (this competition’s sample has 153,164 rows, while your `test.csv` is 552,888 in this environment), by always predicting only for the sample ids subset. These are minimal changes that should move your score substantially upward toward the 0.986 target while preserving semantics.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score is consistent with near-constant predictions, so the main goal is to ensure the TF‑IDF + per-label LogisticRegression path is reliably used and produces well-calibrated probabilities aligned to the sample submission ids. I keep the exact core model (TF‑IDF + LogisticRegression per label) but fix two score-limiting issues: (1) the solver/regularization choice and iterations to better fit sparse TF‑IDF without changing the modeling approach, and (2) add a minimal, metric-aligned blend with label priors to stabilize rare labels (especially “threat”) without changing semantics. I also make the “external submission ensemble” detection stricter to avoid accidentally averaging in irrelevant CSVs that can degrade score. The output remains a valid `submission.csv` with the exact required columns and row alignment to `sample_submission.csv`.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score strongly suggests the submission is either constant-like or misaligned; the biggest low-risk improvement is to make the TF‑IDF + per-label LogisticRegression predictions stronger and more stable without changing the overall approach. I keep your exact pipeline (TF‑IDF → 6 independent LogisticRegression probability models → write `submission.csv`) and only adjust hyperparameters that materially impact ROC-AUC on this task: use a better-optimized `max_df`, slightly higher `max_iter`, and `class_weight="balanced"` to help rare labels like `threat`. I also make the prior-blend label-dependent (smaller blend for common labels, larger for rare labels) to stabilize extremes while staying within the same “probability blending” semantics. Output schema and alignment to `sample_submission.csv` remain deterministic and unchanged.'
- What this solution (achieved 0.5) has done: 'Your score is stuck at 0.5 because the external-submission ensemble path almost never finds valid files here, so you fall back to a baseline that likely doesn’t train correctly in this environment (missing `scikit-learn` in the installed packages list) or produces weak/misaligned predictions. To move toward the 0.986 target with minimal conceptual change, I keep the same “TF‑IDF → 6 independent linear models → probabilities” core logic but implement it using only `numpy/pandas` (a simple, efficient multinomial Naive Bayes on TF‑IDF-like log-count ratios), which avoids dependency issues and is a standard strong baseline for this competition. I also keep your strict id alignment to `sample_submission.csv` and preserve the prior-blending idea to stabilize rare labels. The script still writes a valid `submission.csv` with the required columns and exact row count/order.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 suggests near-constant or misaligned predictions; the simplest way toward the 0.986 target is to make the self-contained text model path stronger while keeping the same “independent per-label linear NB-on-words probabilities” core logic. I keep your Naive Bayes log-count-ratio model, but (1) ensure we always train/predict on the exact `sample_submission` ids (already done), (2) improve tokenization slightly to include common toxic markers (e.g., `!`, `?`, repeated punctuation) without changing the model family, and (3) add a very small, metric-friendly calibration blend using out-of-fold (OOF) predictions to choose a per-label blending weight toward priors (reduces over/under-confidence, especially for rare labels) while remaining within the same probability-blending semantics. These are minimal changes that should materially increase ROC-AUC from 0.5 toward the target without changing the overall approach or requiring external files. The script still always write a valid `submission.csv` with the required columns and row alignment.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

CANDIDATE_ROOTS = [
    "../input",  # typical Kaggle notebooks
    "/kaggle/input",  # typical Kaggle notebooks (absolute)
    "/kaggle/data",  # provided in this environment
    "/kaggle/data/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge",
]

existing_roots = [p for p in CANDIDATE_ROOTS if os.path.exists(p)]
existing_roots



## === cell 1
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def find_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


sample_path = find_first_existing(
    [
        "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv",
        "/kaggle/data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
)

test_path = find_first_existing(
    [
        "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv",
        "/kaggle/data/jigsaw-toxic-comment-classification-challenge/test.csv",
        "/kaggle/input/test.csv",
        "/kaggle/data/test.csv",
    ]
)

train_path = find_first_existing(
    [
        "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv",
        "/kaggle/data/jigsaw-toxic-comment-classification-challenge/train.csv",
        "/kaggle/input/train.csv",
        "/kaggle/data/train.csv",
    ]
)

sample_path, test_path, train_path



## === cell 2
if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected locations."
    )

sample_sub = pd.read_csv(sample_path)
required_cols = ["id"] + label_cols

missing = [c for c in required_cols if c not in sample_sub.columns]
if missing:
    raise ValueError(f"sample_submission.csv missing columns: {missing}")

sample_sub = sample_sub[required_cols].copy()
sample_sub.head()




## === cell 3
def iter_csv_files(root, max_files=2000):
    n = 0
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".csv"):
                yield os.path.join(dirpath, fn)
                n += 1
                if n >= max_files:
                    return


def try_load_submission_csv(path, sample_ids):
    base = os.path.basename(path).lower()
    if "submission" not in base:
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if not all(c in df.columns for c in required_cols):
        return None

    if len(df) != len(sample_ids):
        return None

    try:
        df = df[required_cols].copy()
        df = df.set_index("id").reindex(sample_ids)
    except Exception:
        return None

    if df.isna().any().any():
        return None

    for c in label_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[label_cols].isna().any().any():
        return None

    df[label_cols] = df[label_cols].clip(0.0, 1.0)
    df = df.reset_index()
    return df


sample_ids = sample_sub["id"].tolist()

found_paths = []
models = []

for root in existing_roots:
    for csv_path in iter_csv_files(root):
        df_sub = try_load_submission_csv(csv_path, sample_ids)
        if df_sub is not None:
            models.append(df_sub)
            found_paths.append(csv_path)

len(models), (found_paths[:5] if found_paths else found_paths)



## === cell 4
import re
from collections import Counter


_token_re = re.compile(r"[a-zA-Z]{2,}|[!?.]{2,}|[!?.]")


def simple_tokenize(text):
    return _token_re.findall(text.lower())


def build_vocab(texts, max_features=200000, min_df=2):
    df = Counter()
    for t in texts:
        toks = set(simple_tokenize(t))
        df.update(toks)
    items = [(w, c) for w, c in df.items() if c >= min_df]
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:max_features]
    vocab = {w: i for i, (w, _) in enumerate(items)}
    return vocab


def texts_to_counts(texts, vocab):
    rows = []
    for t in texts:
        cnt = Counter()
        for w in simple_tokenize(t):
            j = vocab.get(w)
            if j is not None:
                cnt[j] += 1
        rows.append(cnt)
    return rows


def fit_nb_logcountratio(count_rows, y, n_features, alpha=1.0):
    y = np.asarray(y).astype(np.int64)
    pos = np.zeros(n_features, dtype=np.float64)
    neg = np.zeros(n_features, dtype=np.float64)

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)

    for i, cnt in enumerate(count_rows):
        if y[i] == 1:
            for j, v in cnt.items():
                pos[j] += v
        else:
            for j, v in cnt.items():
                neg[j] += v

    pos += alpha
    neg += alpha
    r = np.log(pos / pos.sum()) - np.log(neg / neg.sum())

    p = (n_pos + 1.0) / (n_pos + n_neg + 2.0)
    b = np.log(p / (1.0 - p))
    return r, b, p


def predict_nb_logcountratio(count_rows, r, b):
    z = np.empty(len(count_rows), dtype=np.float64)
    for i, cnt in enumerate(count_rows):
        s = b
        for j, v in cnt.items():
            s += v * r[j]
        z[i] = s
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def _rankdata_average(a):
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    s = a[order]
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and s[j] == s[i]:
            j += 1
        if j - i > 1:
            avg = 0.5 * (i + 1 + j)
            ranks[order[i:j]] = avg
        i = j
    return ranks


def fast_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _rankdata_average(y_score)
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def choose_alpha_blend_oof(
    count_rows, y, n_features, prior, alpha_nb=1.0, n_folds=3, seed=1337
):
    y = np.asarray(y).astype(np.int64)
    n = len(y)
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    folds = np.array_split(idx, n_folds)
    oof = np.zeros(n, dtype=np.float64)

    for k in range(n_folds):
        va = folds[k]
        tr = np.concatenate([folds[i] for i in range(n_folds) if i != k])
        r, b, _ = fit_nb_logcountratio(
            [count_rows[i] for i in tr], y[tr], n_features, alpha=alpha_nb
        )
        oof[va] = predict_nb_logcountratio([count_rows[i] for i in va], r, b)

    grid = np.array(
        [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20], dtype=np.float64
    )
    best_a = 0.0
    best_auc = -1.0
    for a in grid:
        blended = (1.0 - a) * oof + a * prior
        auc = fast_auc(y, blended)
        if auc > best_auc:
            best_auc = auc
            best_a = float(a)
    return best_a, best_auc


if models:
    preds = np.stack([m[label_cols].to_numpy(dtype=np.float64) for m in models], axis=0)
    mean_preds = preds.mean(axis=0)

    p_res = sample_sub.copy()
    p_res[label_cols] = mean_preds
    p_res[label_cols] = p_res[label_cols].clip(0.0, 1.0)
    p_res = p_res[required_cols]
else:
    if train_path is None or test_path is None:
        p_res = sample_sub.copy()
    else:
        train_df = pd.read_csv(train_path, usecols=["comment_text"] + label_cols)
        test_df = pd.read_csv(test_path, usecols=["id", "comment_text"])

        train_df["comment_text"] = train_df["comment_text"].fillna("").astype(str)
        test_df["comment_text"] = test_df["comment_text"].fillna("").astype(str)

        test_df_idx = test_df.set_index("id")
        test_aligned = test_df_idx.reindex(sample_ids)
        test_aligned["comment_text"] = test_aligned["comment_text"].fillna("")

        X_train_text = train_df["comment_text"].values.tolist()
        X_test_text = test_aligned["comment_text"].values.tolist()

        vocab = build_vocab(X_train_text, max_features=200000, min_df=2)
        n_features = len(vocab)

        Xtr_counts = texts_to_counts(X_train_text, vocab)
        Xte_counts = texts_to_counts(X_test_text, vocab)

        test_pred = np.zeros((len(sample_ids), len(label_cols)), dtype=np.float64)
        label_priors = train_df[label_cols].mean(axis=0).to_dict()

        for j, col in enumerate(label_cols):
            y = train_df[col].values
            prior = float(label_priors[col])

            alpha_blend, _ = choose_alpha_blend_oof(
                Xtr_counts,
                y,
                n_features,
                prior=prior,
                alpha_nb=1.0,
                n_folds=3,
                seed=1337 + j,
            )

            r, b, _ = fit_nb_logcountratio(Xtr_counts, y, n_features, alpha=1.0)
            proba = predict_nb_logcountratio(Xte_counts, r, b)
            test_pred[:, j] = (1.0 - alpha_blend) * proba + alpha_blend * prior

        p_res = pd.DataFrame({"id": sample_ids})
        for j, col in enumerate(label_cols):
            p_res[col] = np.clip(test_pred[:, j], 0.0, 1.0)

        for col in label_cols:
            if p_res[col].isna().any():
                p_res[col] = p_res[col].fillna(float(label_priors[col]))

        p_res = p_res[required_cols]

p_res.head(), p_res.shape



## === cell 5
out_path = "submission.csv"
p_res.to_csv(out_path, index=False)

chk = pd.read_csv(out_path, nrows=5)
assert (
    list(chk.columns) == required_cols
), f"Bad submission columns: {chk.columns.tolist()}"
assert len(p_res) == len(
    sample_sub
), f"Row mismatch: submission {len(p_res)} vs sample {len(sample_sub)}"

chk.columns.tolist(), out_path
