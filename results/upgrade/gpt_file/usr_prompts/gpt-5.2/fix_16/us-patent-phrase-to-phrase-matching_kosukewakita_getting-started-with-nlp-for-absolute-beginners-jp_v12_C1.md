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

0.50785

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.40238) has done: 'I fix the pipeline so it runs end-to-end on Kaggle without requiring any internet or extra uploaded model datasets. The main bug is that the notebook assumes a local DeBERTa model exists under `../input`, which it doesn’t, so I switch to a lightweight, fully local baseline using only the provided CSVs (no external packages beyond what’s already used). I also remove the failing `transformers`/`Trainer` import path that triggers a protobuf-related crash in this environment, and ensure we always write a valid `submission.csv` with the correct `id,score` columns. The resulting approach keeps the overall “pairwise text similarity scoring” semantics and should yield a reasonable Pearson score (though likely below transformer-based targets) while being robust and fast.'
- What this solution (achieved 0.40894) has done: 'Your current score (0.40238) is far below the target (0.8069), so we should improve performance while keeping the same “handcrafted token overlap features + ridge regression” core logic. The biggest avoidable score killer here is the post-processing that pulls predictions toward the discrete label levels; Pearson correlation generally prefers continuous predictions, so we remove that quantization blending. We also make the token features a bit more semantically aligned to the task by comparing anchor↔target in the *same context* via simple context-prefix expansion (still pure string/token logic, no new models/packages), and we lightly retune ridge regularization to stabilize weights. These are minimal changes that preserve your approach and should move the score upward toward the target.'
- What this solution (achieved 0.49256) has done: 'Your current score (0.40894) is far below the target (0.80695), so we should improve while preserving your handcrafted-overlap-features + closed-form ridge core logic. The biggest low-risk gain is to add a few more *still simple* similarity features that capture character-level matching and phrase-level proximity (e.g., normalized edit distance proxy, char 3-gram Jaccard, and token containment flags), which often helps when surface forms differ (plurals/abbreviations). We also standardize features before ridge solving (same ridge approach, just better conditioned) and tune alpha slightly to reduce under/over-regularization. These are minimal, local changes that keep the same training approach and still produce the same `submission.csv` format.'
- What this solution (achieved 0.49256) has done: 'We keep your handcrafted feature extraction and closed-form ridge regression exactly as-is, but make two minimal changes that typically improve Pearson correlation without changing the modeling approach. First, we tune the ridge `alpha` using a small deterministic CV over groups defined by `context` (to better match the competition’s context-dependent similarity), then retrain on all data with the best `alpha`. Second, we apply a simple post-hoc linear calibration (fit `y ≈ a*pred + b` on out-of-fold predictions) to correct scale/offset mismatch; this preserves ranking behavior while improving correlation. Both changes are lightweight, stay within the same ridge framework, and still write a valid `submission.csv` with `id,score`.'
- What this solution (achieved 0.49256) has done: 'We keep your handcrafted similarity features and closed-form ridge regression intact, but fix two small score-limiters that typically depress Pearson correlation. First, the ridge is currently regularizing the bias term (intercept) the same as all weights, which can unnecessarily shrink predictions toward the mean; we exclude the intercept from L2 regularization (same ridge, more standard). Second, we fit the post-hoc calibration (a,b) in a leakage-safe way using out-of-fold (OOF) predictions and then apply that calibration to test predictions (same linear calibration idea, but more reliable). These minimal changes should increase correlation from your current ~0.49 toward the target without changing the overall approach or output format.'
- What this solution (achieved 0.49256) has done: 'Your current score (0.49256) is far below the target (0.80695), so we should make small, low-risk changes that improve correlation while keeping your handcrafted-features + closed-form ridge + linear calibration core logic intact. The main score limiter is that the calibration is fit only once on raw OOF predictions; using a leakage-safe *fold-wise* calibration (still linear) and then averaging calibrated test predictions across folds typically improves Pearson by correcting per-fold scale drift. We also make the ridge solver a bit more numerically stable (float64 solve) without changing the model, and keep the final clip to [0,1] for valid outputs. The output remains a valid `submission.csv` with `id,score`.'
- What this solution (achieved 0.49256) has done: 'We keep your handcrafted feature set and closed-form ridge + fold-wise linear calibration intact, but adjust the CV selection criterion to match the final (calibrated) model behavior. Right now `best_alpha` is chosen using *uncalibrated* OOF predictions, which can pick an alpha that looks good pre-calibration but is suboptimal after calibration; we select `alpha` by the *post fold-wise calibrated* OOF Pearson instead. This is a minimal, leakage-safe change (calibration is fit only on each fold’s training split) and should move your score upward toward the target without changing architecture, features, or training paradigm. The script still runs end-to-end offline and writes a valid `submission.csv` with `id,score`.'
- What this solution (achieved 0.49256) has done: 'We keep your handcrafted feature extraction and closed-form ridge + fold-wise linear calibration exactly the same, but fix a subtle score limiter: the calibration currently uses in-fold training predictions (`pred_tr`) which are overly optimistic and can make the learned `(a,b)` too aggressive, hurting generalization (and Pearson) on test. We instead fit the fold calibration on a leakage-safe *inner OOF* prediction within each training fold (still linear calibration, just using less biased inputs), then apply it to that fold’s validation and test predictions. This is a minimal, targeted change that preserves your model/training paradigm and should improve correlation from ~0.49 toward the target. The script still run end-to-end offline and write a valid `submission.csv` with `id,score`.'
- What this solution (achieved 0.49257) has done: 'Your current score (0.49256) is far below the target (0.80695), so we should improve correlation while keeping your handcrafted features + closed-form ridge + calibration approach unchanged. The most likely score limiter now is that the inner calibration OOF uses random splits that ignore `context`, even though your outer CV is context-grouped; this mismatch can distort the learned (a,b) scaling and hurt generalization. I make the inner OOF calibration also split by `context` (within each training fold) so calibration learns under the same distributional constraints, and I keep everything else (features, ridge solving, alpha grid, fold-wise averaging, clipping, submission format) identical. This is a minimal, leakage-safe change expected to move Pearson upward toward the target without altering core modeling semantics.'
- What this solution (achieved 0.49431) has done: 'Your current score (0.49257) is far below the target (0.80695), so we should improve Pearson correlation while keeping your handcrafted features + closed-form ridge + fold-wise linear calibration core logic unchanged. The biggest low-risk gain is to make the ridge model context-aware without changing the learning approach: we add simple one-hot context features (and context group buckets) so the same similarity features can map to different score baselines/slopes per context family. This keeps the same featurization style (purely from provided CSVs, no external models) and the same ridge solver, but typically boosts correlation substantially on this competition. We keep the existing CV alpha selection and inner-OOF calibration logic, just applied to the expanded feature matrix, and still write a valid `submission.csv`.'
- What this solution (achieved 0.49484) has done: 'We need to move your score up toward the target (0.8069) from 0.4943 while keeping the same handcrafted-features + closed-form ridge + fold-wise linear calibration core pipeline. The smallest high-impact fix is to correct a semantic mismatch in `make_input`: it currently builds strings in the order `context; target; anchor`, which conflicts with the rest of your feature extraction that consistently treats anchor↔target as the main pair; although `input` isn’t used directly now, keeping fields consistent helps prevent accidental leakage/bugs and lets us safely add one minimal, metric-aligned feature. Then, without changing the model type or training loop, we add two very lightweight, strictly similarity-based features (exact normalized match and numeric-token agreement) that often improve Pearson for this competition due to frequent near-duplicate and number-heavy phrases. Finally, we slightly lower the context one-hot `min_count` threshold so more contexts get explicit offsets (still the same one-hot idea), which typically lifts correlation with minimal risk and runtime impact.'
- What this solution (achieved 0.49496) has done: 'Your current score (0.49484) is far below the target (0.80695), so we should make small, safe improvements without changing the overall “handcrafted similarity features → standardized ridge → fold-wise linear calibration → average test preds” pipeline. The biggest likely limiter now is that your many context one-hot features are being standardized like continuous features, which turns sparse 0/1 indicators into large ± values and can destabilize ridge; we standardize only the base similarity features and keep context one-hots unscaled. Next, we add just one minimal, highly relevant feature: a context-aware “anchor↔target token Jaccard after removing tokens that also appear in context”, which often helps when context words otherwise inflate similarity. These changes preserve your training approach, solver, calibration scheme, and output semantics, and should move Pearson upward.'
- What this solution (achieved 0.50916) has done: 'Your current score (0.49496) is far below the target (0.80695), so we should make small, low-risk changes that improve Pearson while keeping your handcrafted-features → ridge → fold-wise calibration pipeline intact. The biggest likely limiter is that the context one-hot block is only providing per-context intercept shifts; adding a minimal set of “context × base-feature” interaction features lets the same similarity signals map to different slopes per context (still linear ridge, same training loop/solver). To keep changes stable and avoid overfitting, we only add interactions for the coarse context group (A–H/OTHER), not every fine context code, and we keep the same CV/alpha selection/calibration logic. Everything still runs offline and writes a valid `submission.csv` with `id,score`.'
- What this solution (achieved 0.50785) has done: 'We need to increase your Pearson score toward the target (0.8069) from 0.50916, so we make a very small, metric-aligned improvement without changing the core “handcrafted similarity features → (standardize base) ridge → fold-wise linear calibration” pipeline. The main low-risk lever is to tune the strength of the new context×feature interaction block: right now it’s implicitly weight=1, which can easily over/under-contribute; we add a tiny CV over a single scalar multiplier for interactions and pick the best by the same calibrated OOF Pearson you already use. This keeps the model linear ridge with the same solver, same folds, same calibration scheme—only the feature matrix is scaled in a controlled way. We also make the alpha grid slightly denser around the likely optimum to improve selection resolution while staying within the same ridge approach and runtime budget.'

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
        "CONTEXT: "
        + df["context"].astype(str)
        + "; ANCHOR: "
        + df["anchor"].astype(str)
        + "; TARGET: "
        + df["target"].astype(str)
    )


train_df["input"] = make_input(train_df)
test_df["input"] = make_input(test_df)

train_df["input"].head()



## === cell 7
_token_re = re.compile(r"[a-z0-9]+")
_space_re = re.compile(r"\s+")
_num_re = re.compile(r"\d+")


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


def numeric_jaccard(a: str, b: str) -> float:
    aa = set(_num_re.findall(norm_text(a)))
    bb = set(_num_re.findall(norm_text(b)))
    if not aa and not bb:
        return 1.0
    return len(aa & bb) / max(1, len(aa | bb))


def featurize(df: pd.DataFrame) -> np.ndarray:
    feats = np.zeros((len(df), 17), dtype=np.float32)
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
        feats[i, 8] = jaccard(a3, t3)

        feats[i, 9] = normalized_lcs_ratio(anchor, target)

        feats[i, 10] = containment_flag(a_toks, t_toks)
        feats[i, 11] = containment_flag(t_toks, a_toks)

        a3c = char_ngrams(context + " " + anchor, 3)
        t3c = char_ngrams(context + " " + target, 3)
        feats[i, 12] = jaccard(a3c, t3c)

        feats[i, 13] = float(abs(len(a_toks) - len(t_toks))) / 10.0

        feats[i, 14] = 1.0 if norm_text(anchor) == norm_text(target) else 0.0
        feats[i, 15] = numeric_jaccard(anchor, target)

        cset = set(c_toks)
        a_wo = [z for z in a_toks if z not in cset]
        t_wo = [z for z in t_toks if z not in cset]
        feats[i, 16] = jaccard(a_wo, t_wo)

    return feats


X_base = featurize(train_df)
y = train_df["score"].astype(np.float32).values
X_test_base = featurize(test_df)

print("Base feature shapes:", X_base.shape, y.shape, X_test_base.shape)




## === cell 8
def build_context_features(
    train_context: np.ndarray,
    test_context: np.ndarray,
    min_count: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    tr = np.asarray(train_context, dtype=str)
    te = np.asarray(test_context, dtype=str)

    vc = pd.Series(tr).value_counts()
    kept = vc[vc >= min_count].index.tolist()
    cols = kept + ["__OTHER__"]
    col_index = {c: j for j, c in enumerate(cols)}

    def encode(ctx_arr: np.ndarray) -> np.ndarray:
        out = np.zeros((len(ctx_arr), len(cols)), dtype=np.float32)
        for i, c in enumerate(ctx_arr.tolist()):
            j = col_index.get(c, col_index["__OTHER__"])
            out[i, j] = 1.0
        return out

    tr_full = encode(tr)
    te_full = encode(te)

    def encode_group(ctx_arr: np.ndarray) -> np.ndarray:
        out = np.zeros((len(ctx_arr), 9), dtype=np.float32)  # A-H plus OTHER
        for i, c in enumerate(ctx_arr.tolist()):
            g = c[:1].upper() if c else ""
            if "A" <= g <= "H":
                out[i, ord(g) - ord("A")] = 1.0
            else:
                out[i, 8] = 1.0
        return out

    tr_grp = encode_group(tr)
    te_grp = encode_group(te)

    return np.concatenate([tr_full, tr_grp], axis=1), np.concatenate(
        [te_full, te_grp], axis=1
    )


ctx_tr = train_df["context"].astype(str).values
ctx_te = test_df["context"].astype(str).values
X_ctx_tr, X_ctx_te = build_context_features(ctx_tr, ctx_te, min_count=30)

X_grp_tr = X_ctx_tr[:, -9:]
X_grp_te = X_ctx_te[:, -9:]
X_int_tr = (
    (X_grp_tr[:, :, None] * X_base[:, None, :])
    .reshape(len(train_df), -1)
    .astype(np.float32, copy=False)
)
X_int_te = (
    (X_grp_te[:, :, None] * X_test_base[:, None, :])
    .reshape(len(test_df), -1)
    .astype(np.float32, copy=False)
)

print(
    "Blocks:",
    "X_base",
    X_base.shape,
    "X_ctx_tr",
    X_ctx_tr.shape,
    "X_int_tr",
    X_int_tr.shape,
)




## === cell 9
def standardize_fit_transform(X: np.ndarray):
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    return (X - mu) / sigma, mu, sigma


def standardize_transform(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return (X - mu) / sigma


def ridge_fit_predict(X, y, X_test, alpha=1e-2, n_scale: int | None = None):
    if n_scale is None:
        n_scale = X.shape[1]

    Xs = X.astype(np.float32, copy=False)
    Xts = X_test.astype(np.float32, copy=False)

    Xs_cont, mu, sigma = standardize_fit_transform(Xs[:, :n_scale])
    Xts_cont = standardize_transform(Xts[:, :n_scale], mu, sigma)

    if n_scale < Xs.shape[1]:
        Xs_final = np.concatenate([Xs_cont, Xs[:, n_scale:]], axis=1)
        Xts_final = np.concatenate([Xts_cont, Xts[:, n_scale:]], axis=1)
    else:
        Xs_final = Xs_cont
        Xts_final = Xts_cont

    X_ = np.concatenate(
        [np.ones((Xs_final.shape[0], 1), dtype=Xs_final.dtype), Xs_final], axis=1
    )
    Xt_ = np.concatenate(
        [np.ones((Xts_final.shape[0], 1), dtype=Xts_final.dtype), Xts_final], axis=1
    )

    X64 = X_.astype(np.float64, copy=False)
    y64 = np.asarray(y, dtype=np.float64)
    reg = alpha * np.eye(X64.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0  # exclude intercept from L2
    A = X64.T @ X64 + reg
    b = X64.T @ y64
    w = np.linalg.solve(A, b)
    return (Xt_.astype(np.float64, copy=False) @ w).astype(np.float32)


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


def fit_calibration_inner_oof(
    X_tr,
    y_tr,
    alpha: float,
    n_splits: int = 5,
    seed: int = 123,
    contexts_tr: np.ndarray | None = None,
    n_scale: int | None = None,
):
    n = len(y_tr)
    pred_oof = np.zeros(n, dtype=np.float32)

    if contexts_tr is None:
        rng = np.random.RandomState(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        parts = np.array_split(idx, n_splits)
        for k in range(n_splits):
            val_i = parts[k]
            tr_i = np.setdiff1d(idx, val_i, assume_unique=False)
            pred_oof[val_i] = ridge_fit_predict(
                X_tr[tr_i], y_tr[tr_i], X_tr[val_i], alpha=alpha, n_scale=n_scale
            )
    else:
        inner_folds = list(
            context_group_folds(np.asarray(contexts_tr), n_splits=n_splits, seed=seed)
        )
        for tr_i_mask, val_i_mask in inner_folds:
            pred_oof[val_i_mask] = ridge_fit_predict(
                X_tr[tr_i_mask],
                y_tr[tr_i_mask],
                X_tr[val_i_mask],
                alpha=alpha,
                n_scale=n_scale,
            )

    A = np.vstack([pred_oof.astype(np.float64), np.ones(n, dtype=np.float64)]).T
    coef, _, _, _ = np.linalg.lstsq(A, y_tr.astype(np.float64), rcond=None)
    a, b = float(coef[0]), float(coef[1])
    return a, b


N_SCALE = int(X_base.shape[1])

contexts = train_df["context"].astype(str).values
folds = list(context_group_folds(contexts, n_splits=5, seed=42))

alphas = np.array(
    [3e-5, 1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 6e-3, 1e-2, 2e-2, 3e-2, 6e-2, 1e-1],
    dtype=np.float64,
)

int_scales = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25], dtype=np.float64)


def make_design(int_scale: float) -> tuple[np.ndarray, np.ndarray]:
    X_tr_full = np.concatenate(
        [X_base, X_ctx_tr, (X_int_tr * float(int_scale))], axis=1
    ).astype(np.float32, copy=False)
    X_te_full = np.concatenate(
        [X_test_base, X_ctx_te, (X_int_te * float(int_scale))], axis=1
    ).astype(np.float32, copy=False)
    return X_tr_full, X_te_full


best = None  # (cal_score, alpha, int_scale)
for int_scale in int_scales:
    X_full, _ = make_design(float(int_scale))

    cv_cal_scores = []
    for alpha in alphas:
        oof_cal = np.zeros(len(train_df), dtype=np.float32)

        for fold_id, (tr_idx, val_idx) in enumerate(folds):
            pred_val = ridge_fit_predict(
                X_full[tr_idx],
                y[tr_idx],
                X_full[val_idx],
                alpha=float(alpha),
                n_scale=N_SCALE,
            )

            a_fold, b_fold = fit_calibration_inner_oof(
                X_full[tr_idx],
                y[tr_idx],
                alpha=float(alpha),
                n_splits=5,
                seed=1000 + fold_id,
                contexts_tr=contexts[tr_idx],
                n_scale=N_SCALE,
            )
            oof_cal[val_idx] = (pred_val * a_fold + b_fold).astype(np.float32)

        cal_score = pearsonr_np(oof_cal, y)
        cv_cal_scores.append(cal_score)

        if (best is None) or (cal_score > best[0]):
            best = (cal_score, float(alpha), float(int_scale))

        print(
            f"int_scale={int_scale:>4g}  alpha={alpha:g}  OOF pearson cal={cal_score:.6f}"
        )

best_cal, best_alpha, best_int_scale = best
print(
    "Best (by calibrated OOF pearson):",
    "int_scale=",
    best_int_scale,
    "alpha=",
    best_alpha,
    "OOF_cal=",
    best_cal,
)

X_full, X_test_full = make_design(best_int_scale)

oof_cal = np.zeros(len(train_df), dtype=np.float32)
test_pred_sum = np.zeros(len(test_df), dtype=np.float32)

for fold_id, (tr_idx, val_idx) in enumerate(folds):
    pred_val = ridge_fit_predict(
        X_full[tr_idx], y[tr_idx], X_full[val_idx], alpha=best_alpha, n_scale=N_SCALE
    )

    a_fold, b_fold = fit_calibration_inner_oof(
        X_full[tr_idx],
        y[tr_idx],
        alpha=best_alpha,
        n_splits=5,
        seed=2000 + fold_id,
        contexts_tr=contexts[tr_idx],
        n_scale=N_SCALE,
    )
    oof_cal[val_idx] = (pred_val * a_fold + b_fold).astype(np.float32)

    pred_test_fold = ridge_fit_predict(
        X_full[tr_idx], y[tr_idx], X_test_full, alpha=best_alpha, n_scale=N_SCALE
    )
    test_pred_sum += (pred_test_fold * a_fold + b_fold).astype(np.float32)

oof_score = pearsonr_np(oof_cal, y)
print("OOF pearson after fold-wise (inner-OOF) calibration (best params):", oof_score)

preds = (test_pred_sum / len(folds)).astype(np.float32)
preds = np.clip(preds, 0.0, 1.0)

preds[:10], preds.shape



## === cell 10
submission = pd.DataFrame({"id": test_df["id"].values, "score": preds})
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv:", submission.shape)
submission.head()
