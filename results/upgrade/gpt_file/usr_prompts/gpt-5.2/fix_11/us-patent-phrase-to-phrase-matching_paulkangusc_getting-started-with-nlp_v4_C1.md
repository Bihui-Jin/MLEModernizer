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

3.12

# 3. Installed packages

datasets==4.4.1
geopandas==0.14.4
kaggle==1.7.4.5
kaggle-environments==1.18.0
kagglehub==0.3.13
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
transformers==4.53.3
vega-datasets==0.9.0

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

0.7564637808094372

# 6. Current score

0.54688

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.47523) has done: 'I fix the end-to-end runtime blockers by (1) making the data path resolution work with your provided `/kaggle/input/...` layout, (2) removing the hard dependency on an offline Hugging Face BERT checkpoint (which is not present) and switching to a locally-available sentence-transformers model for embeddings, and (3) replacing the broken Transformers `Trainer` section (also failing due to a protobuf-related issue) with a minimal regression head trained on those embeddings. This preserves the core “text → transformer-based representation → regression → clip to [0,1] → submission.csv” semantics while ensuring the notebook runs fully offline within Kaggle. Finally, it always write a valid `submission.csv` with the required `id,score` columns.'
- What this solution (achieved 0.44869) has done: 'I fix the runtime error in the SentenceTransformer load path by pinning protobuf to the Python implementation via an environment variable and adding a safe fallback that uses a local Transformers encoder if SentenceTransformer still fails. This preserves the existing “encode text → normalized embeddings → Ridge regression → clip to [0,1] → submission.csv” core semantics while unblocking end-to-end execution offline. To improve the score toward your target (current 0.47523 vs target ~0.756), I keep the same model family (Ridge) but minimally change the embedding input construction to a more standard pairwise format (anchor/context/target) and add a tiny set of similarity-derived features (dot product and abs-diff) concatenated to the embeddings, which is a common, lightweight boost for this competition without changing the training loop. The output format and file path remain unchanged and a valid `submission.csv` is always written.'
- What this solution (achieved 0.52752) has done: 'I fix the two runtime blockers preventing any submission from being produced: (1) the protobuf-related `MessageFactory.GetPrototype` crash by safely importing/using SentenceTransformer and immediately falling back to TF‑IDF when that stack is broken, and (2) the scikit-learn Ridge solver path that crashes due to an incompatible SciPy `cg(tol=...)` signature by forcing a deterministic direct solver that does not use `cg`. These changes keep the same core pipeline (text pair → embedding/features → Ridge regression → clip to [0,1] → submission.csv) while making it run end-to-end offline. I also ensure the code always defines `preds` and writes a valid `submission.csv` with the required `id,score` columns.'
- What this solution (achieved 0.52752) has done: 'I fix the protobuf crash that currently stops execution by ensuring the problematic SentenceTransformer import/usage is fully contained and never triggers a hard failure; instead we reliably fall back to TF‑IDF when that stack is broken. I also correct the Ridge solver selection (it’s currently inconsistent with the comment and can choose an unstable/unsupported path), making it deterministic and compatible with sparse TF‑IDF features. These changes are minimal, preserve the existing “text → embeddings/features → Ridge regression → clip to [0,1] → submission.csv” pipeline, and should improve the score toward your target because TF‑IDF is actually used end-to-end instead of crashing mid-cell.'
- What this solution (achieved 0.54688) has done: 'I fix the hard crash in the SentenceTransformer stack by preventing the protobuf-triggering import from happening at all (so it cannot raise an uncaught AttributeError) and using the existing TF‑IDF fallback reliably end-to-end. Then I make a minimal, score-improving adjustment that preserves the same “TF‑IDF → Ridge regression → clip → submission.csv” core pipeline by adding a tiny set of simple numeric similarity features (lengths/overlap) concatenated to the TF‑IDF matrix. Finally, I keep the same training loop and submission writing, just ensuring everything runs deterministically and produces a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

iskaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != ""
print("iskaggle:", iskaggle)



## === cell 1
from pathlib import Path

candidate_paths = [
    Path("/kaggle/input/us-patent-phrase-to-phrase-matching"),
    Path("../input/us-patent-phrase-to-phrase-matching"),
    Path("/kaggle/data/us-patent-phrase-to-phrase-matching"),
    Path("us-patent-phrase-to-phrase-matching"),
]

path = next(
    (p for p in candidate_paths if (p.exists() and (p / "train.csv").exists())), None
)
if path is None:
    candidate_paths2 = [
        Path("/kaggle/input"),
        Path("/kaggle/data"),
    ]
    path2 = next(
        (p for p in candidate_paths2 if (p.exists() and (p / "train.csv").exists())),
        None,
    )
    if path2 is None:
        raise FileNotFoundError(
            f"Could not find train.csv under any of: {candidate_paths + candidate_paths2}"
        )
    path = path2

print("Using data path:", path)
print("Has train.csv:", (path / "train.csv").exists())
print("Has test.csv:", (path / "test.csv").exists())
print("Has sample_submission.csv:", (path / "sample_submission.csv").exists())



## === cell 2
import pandas as pd

df = pd.read_csv(path / "train.csv")
df.head()



## === cell 3
df.describe(include="object")



## === cell 4
eval_df = pd.read_csv(path / "test.csv")
eval_df.head()



## === cell 5
eval_df.describe(include="object")



## === cell 6
df["input"] = (
    "ANCHOR: "
    + df["anchor"].astype(str)
    + " [SEP] TARGET: "
    + df["target"].astype(str)
    + " [SEP] CONTEXT: "
    + df["context"].astype(str)
)
eval_df["input"] = (
    "ANCHOR: "
    + eval_df["anchor"].astype(str)
    + " [SEP] TARGET: "
    + eval_df["target"].astype(str)
    + " [SEP] CONTEXT: "
    + eval_df["context"].astype(str)
)



## === cell 7
import numpy as np
from typing import Optional


def _l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / (norms + eps)


train_texts = df["input"].astype(str).tolist()
test_texts = eval_df["input"].astype(str).tolist()


def encode_with_sentence_transformers(texts, batch_size=256) -> Optional[np.ndarray]:
    """
    Fix: In this Kaggle/Python 3.12 environment, importing/using sentence-transformers
    can hard-crash due to a protobuf AttributeError ('MessageFactory' has no 'GetPrototype')
    that may bypass local try/except during import. To guarantee end-to-end execution,
    we skip SentenceTransformer entirely and use the existing TF-IDF fallback.
    """
    return None


def encode_with_tfidf_char_ngrams(
    train_texts, test_texts, ngram_range=(3, 5), max_features=200000
):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=2,
        max_features=max_features,
        strip_accents="unicode",
        lowercase=True,
    )
    Xtr = vec.fit_transform(train_texts)
    Xte = vec.transform(test_texts)
    print("TF-IDF shapes:", Xtr.shape, Xte.shape)
    return Xtr, Xte, vec


X = encode_with_sentence_transformers(train_texts, batch_size=256)
X_test = encode_with_sentence_transformers(test_texts, batch_size=256)

y = df["score"].astype(float).values

use_tfidf = (X is None) or (X_test is None)
print("use_tfidf:", use_tfidf)



## === cell 8

from numpy.typing import NDArray


def add_similarity_features_dense(
    E: NDArray[np.float32], E_test: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    mean_abs = np.mean(np.abs(E), axis=1, keepdims=True)
    mean_sq = np.mean(E * E, axis=1, keepdims=True)
    feats = np.hstack([E, mean_abs, mean_sq])

    mean_abs_t = np.mean(np.abs(E_test), axis=1, keepdims=True)
    mean_sq_t = np.mean(E_test * E_test, axis=1, keepdims=True)
    feats_test = np.hstack([E_test, mean_abs_t, mean_sq_t])

    return feats.astype(np.float32, copy=False), feats_test.astype(
        np.float32, copy=False
    )


def _basic_overlap_features(anchor: pd.Series, target: pd.Series) -> np.ndarray:
    a = anchor.fillna("").astype(str).str.lower()
    t = target.fillna("").astype(str).str.lower()

    a_tokens = a.str.split()
    t_tokens = t.str.split()

    feats = np.zeros((len(a), 6), dtype=np.float32)
    for i, (at, tt) in enumerate(zip(a_tokens, t_tokens)):
        aset = set(at)
        tset = set(tt)
        inter = len(aset & tset)
        union = len(aset | tset) if (aset or tset) else 1
        feats[i, 0] = len(at)  # anchor token count
        feats[i, 1] = len(tt)  # target token count
        feats[i, 2] = float(inter)  # intersection size
        feats[i, 3] = float(union)  # union size
        feats[i, 4] = float(inter) / float(union)  # Jaccard
        feats[i, 5] = abs(len(at) - len(tt))  # abs token length diff
    return feats


if not use_tfidf:
    X2, X2_test = add_similarity_features_dense(
        X.astype(np.float32, copy=False), X_test.astype(np.float32, copy=False)
    )
    print("Dense embedding shapes:", X.shape, X_test.shape)
    print("X2 shape:", X2.shape, "X2_test shape:", X2_test.shape)
else:
    X_tfidf, X_tfidf_test, vec = encode_with_tfidf_char_ngrams(train_texts, test_texts)

    from scipy import sparse

    num_train = _basic_overlap_features(df["anchor"], df["target"])
    num_test = _basic_overlap_features(eval_df["anchor"], eval_df["target"])

    X_num = sparse.csr_matrix(num_train)
    X_num_test = sparse.csr_matrix(num_test)

    X2 = sparse.hstack([X_tfidf, X_num], format="csr")
    X2_test = sparse.hstack([X_tfidf_test, X_num_test], format="csr")

    print("Using TF-IDF + numeric overlap features.")
    print("Final shapes:", X2.shape, X2_test.shape)



## === cell 9
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

X_tr, X_va, y_tr, y_va = train_test_split(X2, y, test_size=0.25, random_state=42)

solver = "lsqr" if use_tfidf else "svd"

reg = Ridge(alpha=1.0, random_state=42, solver=solver)
reg.fit(X_tr, y_tr)

va_pred = reg.predict(X_va)
va_pred = np.asarray(va_pred).reshape(-1)
y_va_ = np.asarray(y_va).reshape(-1)

va_corr = np.corrcoef(va_pred, y_va_)[0, 1]
va_rmse = mean_squared_error(y_va_, va_pred, squared=False)

print("Validation pearson:", float(va_corr))
print("Validation RMSE:", float(va_rmse))



## === cell 10
solver_full = "lsqr" if use_tfidf else "svd"
reg_full = Ridge(alpha=1.0, random_state=42, solver=solver_full)
reg_full.fit(X2, y)



## === cell 11
preds = reg_full.predict(X2_test)
preds = np.asarray(preds, dtype=float).reshape(-1)
preds = np.clip(preds, 0.0, 1.0)
print(preds[:10], len(preds))



## === cell 12
submission_df = pd.DataFrame({"id": eval_df["id"].values, "score": preds})
submission_df.to_csv("submission.csv", index=False)

print(submission_df.head())
print("Wrote submission.csv with shape:", submission_df.shape)
print("Columns:", submission_df.columns.tolist())
assert submission_df.shape[0] == eval_df.shape[0]
assert submission_df.columns.tolist() == ["id", "score"]
assert str(Path("submission.csv").name).endswith(".csv")
