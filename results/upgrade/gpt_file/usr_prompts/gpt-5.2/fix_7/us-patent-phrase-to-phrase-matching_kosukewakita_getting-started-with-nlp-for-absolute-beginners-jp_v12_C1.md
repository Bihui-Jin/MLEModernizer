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
Given pairs of phrases (an `anchor` and a `target` phrase), build a model to rate how similar they are.  

## Metric
Pearson correlation coefficient.

## Submission Format
For each `id` (representing a pair of phrases) in the test set, you must predict the similarity `score`. The file should contain a header and have the following format:

```
id,score
4112d61851461f60,0
09e418c93a776564,0.25
36baf228038e314b,1
etc.

```

## Dataset
The scores are in the 0-1 range with increments of 0.25 with the following meanings:

- **1.0** - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. "the", "and", "or").
- **0.75** - Close synonym, e.g. "mobile phone" vs. "cellphone". This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
- **0.5** - Synonyms which don't have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
- **0.25** - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
- **0.0** - Unrelated.

Files
-----

- **train.csv** - the training set, containing phrases, contexts, and their similarity scores
- **test.csv** - the test set set, identical in structure to the training set but without the score
- **sample_submission.csv** - a sample submission file in the correct format

Columns
-------

- `id` - a unique identifier for a pair of phrases
- `anchor` - the first phrase
- `target` - the second phrase
- `context` - the [CPC classification (version 2021.05)](https://en.wikipedia.org/wiki/Cooperative_Patent_Classification), which indicates the subject within which the similarity is to be scored
- `score` - the similarity. This is sourced from a combination of one or more manual expert ratings.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        input/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        working/
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
```

-> data/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> data/us-patent-phrase-to-phrase-matching/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/us-patent-phrase-to-phrase-matching/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/us-patent-phrase-to-phrase-matching/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> input/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> (stopped after 10 files for performance)

# 5. Target score

0.8069452756275215

# 6. Current score

0.49256

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.40238) has done: 'I fix the pipeline so it runs end-to-end on Kaggle without requiring any internet or extra uploaded model datasets. The main bug is that the notebook assumes a local DeBERTa model exists under `../input`, which it doesn’t, so I switch to a lightweight, fully local baseline using only the provided CSVs (no external packages beyond what’s already used). I also remove the failing `transformers`/`Trainer` import path that triggers a protobuf-related crash in this environment, and ensure we always write a valid `submission.csv` with the correct `id,score` columns. The resulting approach keeps the overall “pairwise text similarity scoring” semantics and should yield a reasonable Pearson score (though likely below transformer-based targets) while being robust and fast.'
- What this solution (achieved 0.40894) has done: 'Your current score (0.40238) is far below the target (0.8069), so we should improve performance while keeping the same “handcrafted token overlap features + ridge regression” core logic. The biggest avoidable score killer here is the post-processing that pulls predictions toward the discrete label levels; Pearson correlation generally prefers continuous predictions, so we remove that quantization blending. We also make the token features a bit more semantically aligned to the task by comparing anchor↔target in the *same context* via simple context-prefix expansion (still pure string/token logic, no new models/packages), and we lightly retune ridge regularization to stabilize weights. These are minimal changes that preserve your approach and should move the score upward toward the target.'
- What this solution (achieved 0.49256) has done: 'Your current score (0.40894) is far below the target (0.80695), so we should improve while preserving your handcrafted-overlap-features + closed-form ridge core logic. The biggest low-risk gain is to add a few more *still simple* similarity features that capture character-level matching and phrase-level proximity (e.g., normalized edit distance proxy, char 3-gram Jaccard, and token containment flags), which often helps when surface forms differ (plurals/abbreviations). We also standardize features before ridge solving (same ridge approach, just better conditioned) and tune alpha slightly to reduce under/over-regularization. These are minimal, local changes that keep the same training approach and still produce the same `submission.csv` format.'
- What this solution (achieved 0.49256) has done: 'We keep your handcrafted feature extraction and closed-form ridge regression exactly as-is, but make two minimal changes that typically improve Pearson correlation without changing the modeling approach. First, we tune the ridge `alpha` using a small deterministic CV over groups defined by `context` (to better match the competition’s context-dependent similarity), then retrain on all data with the best `alpha`. Second, we apply a simple post-hoc linear calibration (fit `y ≈ a*pred + b` on out-of-fold predictions) to correct scale/offset mismatch; this preserves ranking behavior while improving correlation. Both changes are lightweight, stay within the same ridge framework, and still write a valid `submission.csv` with `id,score`.'
- What this solution (achieved 0.49256) has done: 'We keep your handcrafted similarity features and closed-form ridge regression intact, but fix two small score-limiters that typically depress Pearson correlation. First, the ridge is currently regularizing the bias term (intercept) the same as all weights, which can unnecessarily shrink predictions toward the mean; we exclude the intercept from L2 regularization (same ridge, more standard). Second, we fit the post-hoc calibration (a,b) in a leakage-safe way using out-of-fold (OOF) predictions and then apply that calibration to test predictions (same linear calibration idea, but more reliable). These minimal changes should increase correlation from your current ~0.49 toward the target without changing the overall approach or output format.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

iskaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != ""
print("iskaggle:", iskaggle)



## === cell 1
creds = ""



## === cell 2
cred_path = Path("~/.kaggle/kaggle.json").expanduser()
if not cred_path.exists() and creds:
    cred_path.parent.mkdir(exist_ok=True)
    cred_path.write_text(creds)
    cred_path.chmod(0o600)



## === cell 3
path = Path("us-patent-phrase-to-phrase-matching")



## === cell 4
from zipfile import ZipFile

if iskaggle:
    path = Path("../input/us-patent-phrase-to-phrase-matching")
else:
    if not path.exists():
        import zipfile, kaggle  # type: ignore

        kaggle.api.competition_download_cli(str(path))
        zipfile.ZipFile(f"{path}.zip").extractall(path)

print("Using data path:", path)
print("Train exists:", (path / "train.csv").exists())
print("Test exists:", (path / "test.csv").exists())



## === cell 5
import re
import numpy as np
import pandas as pd

train_df = pd.read_csv(path / "train.csv")
test_df = pd.read_csv(path / "test.csv")

print("train_df:", train_df.shape, "test_df:", test_df.shape)
train_df.head()




## === cell 6
def make_input(df: pd.DataFrame) -> pd.Series:
    return (
        "TEXT1: "
        + df["context"].astype(str)
        + "; TEXT2: "
        + df["target"].astype(str)
        + "; ANC1: "
        + df["anchor"].astype(str)
    )


train_df["input"] = make_input(train_df)
test_df["input"] = make_input(test_df)

train_df["input"].head()



## === cell 7
_token_re = re.compile(r"[a-z0-9]+")
_space_re = re.compile(r"\s+")


def toks(s: str):
    return _token_re.findall(str(s).lower())


def norm_text(s: str) -> str:
    s = str(s).lower().strip()
    s = _space_re.sub(" ", s)
    return s


def char_ngrams(s: str, n: int = 3):
    s = norm_text(s)
    s = s.replace(" ", "")
    if len(s) < n:
        return []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def jaccard(a_tokens, b_tokens):
    a = set(a_tokens)
    b = set(b_tokens)
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def overlap_ratio(a_tokens, b_tokens):
    a = set(a_tokens)
    b = set(b_tokens)
    if not a:
        return 0.0
    return len(a & b) / max(1, len(a))


def len_ratio(a_tokens, b_tokens):
    la = len(a_tokens)
    lb = len(b_tokens)
    if la == 0 and lb == 0:
        return 1.0
    return min(la, lb) / max(1, max(la, lb))


def containment_flag(a_tokens, b_tokens) -> float:
    a = set(a_tokens)
    b = set(b_tokens)
    if not a:
        return 0.0
    return 1.0 if a.issubset(b) else 0.0


def normalized_lcs_ratio(a: str, b: str) -> float:
    a = norm_text(a)
    b = norm_text(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    a = a.replace(" ", "")
    b = b.replace(" ", "")
    na, nb = len(a), len(b)
    if na == 0 and nb == 0:
        return 1.0
    prev = np.zeros(nb + 1, dtype=np.int32)
    curr = np.zeros(nb + 1, dtype=np.int32)
    for i in range(1, na + 1):
        ai = a[i - 1]
        curr[0] = 0
        for j in range(1, nb + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = prev[j] if prev[j] >= curr[j - 1] else curr[j - 1]
        prev, curr = curr, prev
    lcs = prev[nb]
    return float(lcs) / float(max(1, max(na, nb)))


def featurize(df: pd.DataFrame) -> np.ndarray:
    feats = np.zeros((len(df), 14), dtype=np.float32)
    for i, row in enumerate(df.itertuples(index=False)):
        anchor = str(getattr(row, "anchor"))
        target = str(getattr(row, "target"))
        context = str(getattr(row, "context"))

        a_toks = toks(anchor)
        t_toks = toks(target)
        c_toks = toks(context)

        a_ctoks = c_toks + a_toks
        t_ctoks = c_toks + t_toks

        feats[i, 0] = jaccard(a_toks, t_toks)
        feats[i, 1] = overlap_ratio(a_toks, t_toks)
        feats[i, 2] = overlap_ratio(t_toks, a_toks)
        feats[i, 3] = len_ratio(a_toks, t_toks)
        feats[i, 4] = jaccard(t_toks, c_toks)  # target-context relatedness
        feats[i, 5] = jaccard(a_toks, c_toks)  # anchor-context relatedness
        feats[i, 6] = jaccard(a_ctoks, t_ctoks)
        feats[i, 7] = overlap_ratio(a_ctoks, t_ctoks)

        a3 = char_ngrams(anchor, 3)
        t3 = char_ngrams(target, 3)
        feats[i, 8] = jaccard(a3, t3)  # robust to minor morphology/spacing differences

        feats[i, 9] = normalized_lcs_ratio(
            anchor, target
        )  # robust to abbreviations/plurals

        feats[i, 10] = containment_flag(a_toks, t_toks)
        feats[i, 11] = containment_flag(t_toks, a_toks)

        a3c = char_ngrams(context + " " + anchor, 3)
        t3c = char_ngrams(context + " " + target, 3)
        feats[i, 12] = jaccard(a3c, t3c)

        feats[i, 13] = float(abs(len(a_toks) - len(t_toks))) / 10.0

    return feats


X = featurize(train_df)
y = train_df["score"].astype(np.float32).values
X_test = featurize(test_df)

X.shape, y.shape, X_test.shape




## === cell 8
def standardize_fit_transform(X: np.ndarray):
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    return (X - mu) / sigma, mu, sigma


def standardize_transform(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return (X - mu) / sigma


def ridge_fit_predict(X, y, X_test, alpha=1e-2):
    Xs, mu, sigma = standardize_fit_transform(X)
    Xts = standardize_transform(X_test, mu, sigma)

    X_ = np.concatenate([np.ones((Xs.shape[0], 1), dtype=Xs.dtype), Xs], axis=1)
    Xt_ = np.concatenate([np.ones((Xts.shape[0], 1), dtype=Xts.dtype), Xts], axis=1)

    reg = alpha * np.eye(X_.shape[1], dtype=X_.dtype)
    reg[0, 0] = 0.0  # exclude intercept from L2
    A = X_.T @ X_ + reg
    b = X_.T @ y
    w = np.linalg.solve(A, b)
    return Xt_ @ w


def pearsonr_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())
    if denom <= 0:
        return 0.0
    return float((a * b).sum() / denom)


def context_group_folds(contexts: np.ndarray, n_splits: int = 5, seed: int = 0):
    ctx = np.asarray(contexts)
    uniq = np.unique(ctx)
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, n_splits)
    for k in range(n_splits):
        val_ctx = set(folds[k].tolist())
        val_idx = np.array([c in val_ctx for c in ctx], dtype=bool)
        tr_idx = ~val_idx
        yield tr_idx, val_idx


contexts = train_df["context"].astype(str).values

alphas = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1], dtype=np.float64)
cv_scores = []
oof_pred_by_alpha = {}

for alpha in alphas:
    oof = np.zeros(len(train_df), dtype=np.float32)
    for tr_idx, val_idx in context_group_folds(contexts, n_splits=5, seed=42):
        pred_val = ridge_fit_predict(
            X[tr_idx], y[tr_idx], X[val_idx], alpha=float(alpha)
        )
        oof[val_idx] = pred_val.astype(np.float32)
    score = pearsonr_np(oof, y)
    cv_scores.append(score)
    oof_pred_by_alpha[float(alpha)] = oof
    print(f"alpha={alpha:g}  OOF pearson={score:.6f}")

best_alpha = float(alphas[int(np.argmax(cv_scores))])
best_oof = oof_pred_by_alpha[best_alpha]
print("Best alpha:", best_alpha, "Best OOF pearson:", float(np.max(cv_scores)))


A = np.vstack([best_oof.astype(np.float64), np.ones_like(best_oof, dtype=np.float64)]).T
coef, _, _, _ = np.linalg.lstsq(A, y.astype(np.float64), rcond=None)
cal_a, cal_b = float(coef[0]), float(coef[1])
print("Calibration (OOF): y ≈ a*pred + b with a=", cal_a, "b=", cal_b)

preds = ridge_fit_predict(X, y, X_test, alpha=best_alpha).astype(np.float32)
preds = (preds * cal_a + cal_b).astype(np.float32)
preds = np.clip(preds, 0.0, 1.0)

preds[:10], preds.shape



## === cell 9
submission = pd.DataFrame({"id": test_df["id"].values, "score": preds})
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv:", submission.shape)
submission.head()
