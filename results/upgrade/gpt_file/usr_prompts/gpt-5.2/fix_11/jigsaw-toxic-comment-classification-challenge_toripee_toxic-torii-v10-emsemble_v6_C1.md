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
- What this solution (achieved 0.5) has done: 'Your 0.5 score means the fallback is effectively outputting non-informative probabilities; the simplest way to move toward the 0.9857 target while keeping the same TF‑IDF + LogisticRegression core is to switch from a `MultiOutputClassifier` wrapper to training one calibrated LogisticRegression per label with a solver that performs better on large sparse text (saga). I also slightly adjust vectorizer settings to the well-known strong baseline for this competition (word 1–2 grams, char 3–5 grams concatenated) while still staying within the same “TF‑IDF features + linear models” approach and keeping runtime reasonable. Finally, I make the ensemble blending deterministic and ensure the fallback always aligns to `sample_submission` ids to produce a valid CSV.'
- What this solution (achieved 0.5) has done: 'Your 0.5 score strongly suggests the fallback model wasn’t actually used effectively (or produced near-constant outputs), so we make the fallback more reliably predictive while keeping the same “TF‑IDF features + per-label LogisticRegression + optional blending of found submissions” core logic. The smallest high-impact fix is to stop using `class_weight="balanced"` (it often hurts ROC AUC calibration for this task) and to slightly increase solver iterations so each label model converges on this large sparse problem. We also ensure the fallback predictions are always aligned to the official `sample_submission` ids (already done) and keep ensembling behavior unchanged when valid external submissions exist. This should move the score substantially upward toward your 0.9856 target while preserving the pipeline and output format.'
- What this solution (achieved 0.5) has done: 'Your 0.5 score indicates you’re effectively submitting near-constant probabilities (either the fallback didn’t run as expected or the linear models are underfitting/converging poorly). To move the score upward toward 0.9857 while keeping the same TF‑IDF + per-label LogisticRegression fallback and the same “blend any found submissions else fallback” logic, I make two minimal, high-impact baseline tweaks: use the standard word/char TF‑IDF feature sizes for this competition (without changing the approach) and switch LogisticRegression to a better default configuration for sparse text AUC (solver/liblinear per label, higher max_iter). I also add a tiny safety check to ensure the fallback predictions are not degenerate (std ~ 0), and if they are, retry once with slightly different regularization; this keeps semantics the same (still TF‑IDF + LR) but prevents a repeat 0.5 submission. Output format/paths remain unchanged and a valid `submission.csv` is always written.'
- What this solution (achieved 0.5) has done: 'Your 0.5 score means you’re still effectively submitting non-informative predictions, so the smallest reliable way to move toward the 0.9857 target is to make the TF‑IDF + per-label LogisticRegression fallback stronger and less likely to collapse to near-constant outputs. I keep the same overall core logic (optionally blend found submissions else train TF‑IDF+LR) but (1) make the vectorizer closer to the classic strong baseline by using word(1,2)+char(3,5) without hard `max_features` caps, (2) switch LR to `saga` with a bit more iterations for better convergence on large sparse text, and (3) add a single, metric-safe post-processing step: per-label rank (CDF) normalization to match the test-time score distribution, which often boosts ROC AUC without changing the model family. Submission alignment to `sample_submission` ids and the output CSV schema remain unchanged.'
- What this solution (achieved 0.5) has done: 'Your 0.5 score means you’re still ending up with essentially non-informative predictions; given this code, the most likely cause is that the TF‑IDF+LR fallback isn’t actually running in your environment due to missing scikit-learn (your package list doesn’t include it), so you either error out or fall back to constant-like outputs elsewhere. To move the score sharply upward toward the 0.9857 target while preserving the “fallback when no external submissions exist” core logic, I add a minimal pure-pandas/numpy Naive Bayes (binary bag-of-words) fallback that does not require scikit-learn and produces meaningful probabilities. I keep the existing ensemble-discovery/blending logic unchanged when valid external submissions are found, and I keep the submission aligned to `sample_submission.csv` ids/columns. I also add a tiny safeguard: if scikit-learn is available, your existing TF‑IDF+LR fallback is still used; otherwise, the NB fallback activates automatically to avoid another 0.5 submission.'
- What this solution (achieved 0.5) has done: 'Your 0.5 score indicates your submission is still effectively non-informative; in this environment that’s most plausibly because the TF‑IDF+LogReg fallback never actually runs (scikit-learn isn’t installed) and the pure-Python NB fallback is too slow/weak and/or produces near-constant outputs. To move the score sharply upward toward the 0.9857 target while preserving the existing “ensemble found submissions else train a text fallback and submit probabilities” core logic, I keep the same fallback choice structure but replace the slow NB implementation with a fast, competition-standard Multinomial Naive Bayes on hashed char/word n-grams implemented in pure numpy (no scikit-learn). I also add a minimal degeneracy check (if predictions are ~constant, adjust smoothing once) and keep your metric-safe per-column rank(CDF) normalization. Output remains aligned to `sample_submission.csv` by `id` and always writes a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'Your 0.5 score implies you’re still submitting essentially non-informative probabilities; the most likely cause is that neither scikit-learn training nor a robust fallback is actually producing meaningful predictions in this environment. To move toward the 0.9856 target with minimal semantic change, I keep your “ensemble found submissions else fallback model” logic, but replace the slow/weak pure-Python hashing NB with a fast, higher-quality pure-numpy hashed Multinomial NB that is vectorized and uses a stable hash, plus a light word+char feature setup similar to your TF‑IDF intent. I also add a strict sanity check that refuses to output near-constant predictions (the 0.5 trap) by retrying once with a different smoothing strength, and I keep your rank(CDF) post-processing (AUC-safe) and exact submission alignment to `sample_submission.csv`. Paths and output filename remain unchanged, and the script still always writes a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'Your 0.5 score indicates you’re still effectively submitting non-informative predictions; given the provided package list, the most likely cause is that scikit-learn isn’t available so you always fall back to the pure-numpy hashed NB, but that implementation is too slow/weak and/or collapses toward near-constant outputs at this dataset scale. To move the score upward toward the 0.9857 target with minimal semantic change, I keep the same “blend found submissions else train a fallback text model” logic, but make the NB fallback substantially more predictive by (1) training on a deterministic subsample of the training data (still legitimate training, same model family), and (2) switching from raw counts to TF-like row-normalized counts (NB-compatible and typically improves AUC for long texts), while keeping the same hashed word(1,2)+char(3,5) feature approach and the same rank(CDF) post-processing. I also fix a silent numeric edge case in `_mnb_train_predict_all_labels` where rows with zero features can produce invalid reduceat behavior, which can lead to degenerate predictions. These are the smallest changes that directly target escaping the 0.5 “constant prediction” trap and should move your score materially closer to the target.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
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


def _rank_cdf_per_column(df, cols):
    out = df.copy()
    n = out.shape[0]
    for c in cols:
        r = out[c].rank(method="average").to_numpy(dtype=np.float64)
        out[c] = (r - 0.5) / float(n)
    return out


def _try_import_sklearn():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
        from sklearn.linear_model import LogisticRegression  # noqa: F401
        from sklearn.pipeline import FeatureUnion  # noqa: F401

        return True
    except Exception as e:
        print(f"scikit-learn not available; will use NB fallback. Import error: {e}")
        return False


_SKLEARN_OK = _try_import_sklearn()


def _fit_predict_tfidf_lr(train_df, test_df, sample_sub, label_cols, *, C):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion

    X_train_text = train_df["comment_text"].fillna("").astype(str)
    y_train = train_df[label_cols].astype(int)
    X_test_text = test_df["comment_text"].fillna("").astype(str)

    word_vect = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        strip_accents="unicode",
        lowercase=True,
        sublinear_tf=True,
        dtype=np.float32,
    )
    char_vect = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.9,
        strip_accents="unicode",
        lowercase=True,
        sublinear_tf=True,
        dtype=np.float32,
    )
    vect = FeatureUnion([("word", word_vect), ("char", char_vect)])

    Xtr = vect.fit_transform(X_train_text)
    Xte = vect.transform(X_test_text)

    preds = np.zeros((Xte.shape[0], len(label_cols)), dtype=np.float64)
    for j, col in enumerate(label_cols):
        lr = LogisticRegression(
            solver="saga",
            penalty="l2",
            C=float(C),
            max_iter=4000,
            random_state=42,
            n_jobs=-1,
        )
        lr.fit(Xtr, y_train[col].values)
        preds[:, j] = lr.predict_proba(Xte)[:, 1]

    pred = pd.DataFrame(preds, columns=label_cols)
    pred.insert(0, "id", test_df["id"].values)

    pred = pred.drop_duplicates("id", keep="last").set_index("id")
    pred = pred.reindex(sample_sub["id"].values)
    if pred.isna().any().any():
        raise ValueError(
            "Fallback TF-IDF+LR prediction alignment produced NaNs (id mismatch)."
        )

    pred[label_cols] = pred[label_cols].clip(0.0, 1.0)
    pred = pred.reset_index()

    pred[label_cols] = _rank_cdf_per_column(pred[label_cols], label_cols)[label_cols]
    return pred


def _stable_hash32_bytes(b):
    h = np.uint32(2166136261)
    for x in b:
        h ^= np.uint32(x)
        h *= np.uint32(16777619)
    return int(h)


def _iter_hashed_features(text, *, num_buckets):
    s = (text or "").lower()

    toks = []
    cur = []
    for ch in s:
        if ("a" <= ch <= "z") or ("0" <= ch <= "9"):
            cur.append(ch)
        else:
            if cur:
                toks.append("".join(cur))
                cur = []
    if cur:
        toks.append("".join(cur))

    L = len(toks)
    for n in (1, 2):
        if L < n:
            continue
        for i in range(L - n + 1):
            ng = " ".join(toks[i : i + n]).encode("utf-8", errors="ignore")
            yield _stable_hash32_bytes(ng) % num_buckets, 1

    chars = []
    last_space = False
    for ch in s:
        if ch.isspace():
            if not last_space:
                chars.append(" ")
            last_space = True
        else:
            chars.append(ch)
            last_space = False
    s2 = "".join(chars)
    L2 = len(s2)
    for n in (3, 4, 5):
        if L2 < n:
            continue
        for i in range(L2 - n + 1):
            ng = s2[i : i + n].encode("utf-8", errors="ignore")
            yield _stable_hash32_bytes(ng) % num_buckets, 1


def _hashed_count_matrix(texts, *, num_buckets, dtype=np.uint16):
    indptr = [0]
    indices = []
    data = []
    for t in texts:
        d = {}
        for idx, v in _iter_hashed_features(t, num_buckets=num_buckets):
            d[idx] = d.get(idx, 0) + v
        if d:
            ks = sorted(d.keys())
            indices.extend(ks)
            data.extend([d[k] for k in ks])
        indptr.append(len(indices))
    return (
        np.asarray(indptr, dtype=np.int64),
        np.asarray(indices, dtype=np.int32),
        np.asarray(data, dtype=dtype),
    )


def _row_sums_from_csr(indptr, data):
    n = indptr.size - 1
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        a, b = int(indptr[i]), int(indptr[i + 1])
        if b > a:
            out[i] = float(data[a:b].astype(np.float64).sum())
    return out


def _mnb_train_predict_all_labels(
    indptr_tr, idx_tr, dat_tr, y_tr, indptr_te, idx_te, dat_te, num_buckets, alpha
):
    n_test = indptr_te.size - 1
    n_labels = y_tr.shape[1]
    preds = np.zeros((n_test, n_labels), dtype=np.float64)

    tr_totals = _row_sums_from_csr(indptr_tr, dat_tr)
    te_totals = _row_sums_from_csr(indptr_te, dat_te)

    for j in range(n_labels):
        y = y_tr[:, j].astype(np.int8)
        pos = y == 1
        neg = ~pos

        n_pos = int(pos.sum())
        n_neg = int(neg.sum())
        prior = (n_pos + 1.0) / (n_pos + n_neg + 2.0)
        log_prior_odds = np.log(prior) - np.log(1.0 - prior)

        pos_counts = np.zeros(num_buckets, dtype=np.float64)
        neg_counts = np.zeros(num_buckets, dtype=np.float64)

        for i in np.where(pos)[0]:
            a, b = indptr_tr[i], indptr_tr[i + 1]
            if b > a:
                pos_counts[idx_tr[a:b]] += dat_tr[a:b].astype(np.float64)
        for i in np.where(neg)[0]:
            a, b = indptr_tr[i], indptr_tr[i + 1]
            if b > a:
                neg_counts[idx_tr[a:b]] += dat_tr[a:b].astype(np.float64)

        pos_total = float(tr_totals[pos].sum())
        neg_total = float(tr_totals[neg].sum())

        denom_pos = pos_total + alpha * num_buckets
        denom_neg = neg_total + alpha * num_buckets

        llr = np.log((pos_counts + alpha) / denom_pos) - np.log(
            (neg_counts + alpha) / denom_neg
        )

        scores = np.empty(n_test, dtype=np.float64)
        for i in range(n_test):
            a, b = indptr_te[i], indptr_te[i + 1]
            s = log_prior_odds
            if b > a:
                row_total = te_totals[i]
                if row_total > 0:
                    s += float(
                        (
                            llr[idx_te[a:b]]
                            * (dat_te[a:b].astype(np.float64) / row_total)
                        ).sum()
                    )
            scores[i] = s

        scores = np.clip(scores, -50.0, 50.0)
        preds[:, j] = 1.0 / (1.0 + np.exp(-scores))

    return preds


def _fit_predict_nb_fallback(train_df, test_df, sample_sub, label_cols):
    rs = np.random.RandomState(42)
    n_train = train_df.shape[0]
    max_train_rows = 120000  # chosen to fit under 600s in pure-Python loops
    if n_train > max_train_rows:
        idx = rs.choice(n_train, size=max_train_rows, replace=False)
        train_df = train_df.iloc[idx].reset_index(drop=True)
        print(
            f"NB fallback: subsampled train from {n_train} -> {train_df.shape[0]} rows"
        )
    else:
        print(f"NB fallback: using full train rows = {n_train}")

    X_train_text = train_df["comment_text"].fillna("").astype(str).tolist()
    X_test_text = test_df["comment_text"].fillna("").astype(str).tolist()
    y_train = train_df[label_cols].astype(int).to_numpy()

    num_buckets = 2**18
    print(
        f"NB fallback using hashed features: buckets={num_buckets}, word(1,2)+char(3,5)"
    )

    indptr_tr, idx_tr, dat_tr = _hashed_count_matrix(
        X_train_text, num_buckets=num_buckets, dtype=np.uint16
    )
    indptr_te, idx_te, dat_te = _hashed_count_matrix(
        X_test_text, num_buckets=num_buckets, dtype=np.uint16
    )

    def run_alpha(alpha):
        preds = _mnb_train_predict_all_labels(
            indptr_tr,
            idx_tr,
            dat_tr,
            y_train,
            indptr_te,
            idx_te,
            dat_te,
            num_buckets,
            float(alpha),
        )
        pred = pd.DataFrame(preds, columns=label_cols)
        pred.insert(0, "id", test_df["id"].values)

        pred = pred.drop_duplicates("id", keep="last").set_index("id")
        pred = pred.reindex(sample_sub["id"].values)
        if pred.isna().any().any():
            raise ValueError(
                "NB fallback prediction alignment produced NaNs (id mismatch)."
            )

        pred[label_cols] = pred[label_cols].clip(0.0, 1.0)
        pred = pred.reset_index()

        pred[label_cols] = _rank_cdf_per_column(pred[label_cols], label_cols)[
            label_cols
        ]
        return pred

    pred_df = run_alpha(alpha=0.5)
    std_mean = float(pred_df[label_cols].to_numpy().std(axis=0).mean())
    print(f"NB fallback prediction mean(std per label) = {std_mean:.6f}")

    if std_mean < 1e-4:
        print("Degenerate NB predictions detected; retrying with alpha=0.1 ...")
        pred_df = run_alpha(alpha=0.1)

    return pred_df


def train_fallback_and_predict(sample_sub, label_cols):
    train_df = read_first_existing(TRAIN_PATHS)
    test_df = read_first_existing(TEST_PATHS)

    if "comment_text" not in train_df.columns or "comment_text" not in test_df.columns:
        raise ValueError("Expected column 'comment_text' missing in train/test.")

    if _SKLEARN_OK:
        pred_df = _fit_predict_tfidf_lr(
            train_df, test_df, sample_sub, label_cols, C=4.0
        )
        std_mean = float(pred_df[label_cols].to_numpy().std(axis=0).mean())
        print(f"TF-IDF+LR fallback prediction mean(std per label) = {std_mean:.6f}")
        if std_mean < 1e-4:
            print("Degenerate TF-IDF+LR predictions detected; retrying with C=2.0 ...")
            pred_df = _fit_predict_tfidf_lr(
                train_df, test_df, sample_sub, label_cols, C=2.0
            )
        return pred_df

    return _fit_predict_nb_fallback(train_df, test_df, sample_sub, label_cols)




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
    pred_df = train_fallback_and_predict(sample_sub, label_cols)
    out = pred_df[["id"] + label_cols].copy()

out[label_cols] = out[label_cols].clip(0.0, 1.0)

out_path = "submission.csv"
out.to_csv(out_path, index=False)

print(f"Wrote submission to: {out_path}")
print(out.head())
print(out.describe(include="all"))
