# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.47523) has done: 'I fix the end-to-end runtime blockers by (1) making the data path resolution work with your provided `/kaggle/input/...` layout, (2) removing the hard dependency on an offline Hugging Face BERT checkpoint (which is not present) and switching to a locally-available sentence-transformers model for embeddings, and (3) replacing the broken Transformers `Trainer` section (also failing due to a protobuf-related issue) with a minimal regression head trained on those embeddings. This preserves the core “text → transformer-based representation → regression → clip to [0,1] → submission.csv” semantics while ensuring the notebook runs fully offline within Kaggle. Finally, it always write a valid `submission.csv` with the required `id,score` columns.'
- What this solution (achieved 0.44869) has done: 'I fix the runtime error in the SentenceTransformer load path by pinning protobuf to the Python implementation via an environment variable and adding a safe fallback that uses a local Transformers encoder if SentenceTransformer still fails. This preserves the existing “encode text → normalized embeddings → Ridge regression → clip to [0,1] → submission.csv” core semantics while unblocking end-to-end execution offline. To improve the score toward your target (current 0.47523 vs target ~0.756), I keep the same model family (Ridge) but minimally change the embedding input construction to a more standard pairwise format (anchor/context/target) and add a tiny set of similarity-derived features (dot product and abs-diff) concatenated to the embeddings, which is a common, lightweight boost for this competition without changing the training loop. The output format and file path remain unchanged and a valid `submission.csv` is always written.'

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


def _l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / (norms + eps)




## === cell 8
from typing import Optional

train_texts = df["input"].astype(str).tolist()
test_texts = eval_df["input"].astype(str).tolist()


def encode_with_sentence_transformers(texts, batch_size=256) -> Optional[np.ndarray]:
    return None


def encode_with_transformers_mean_pool(
    texts, batch_size=64, max_length=128
) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer, AutoModel

    model_name_candidates = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "microsoft/MiniLM-L12-H384-uncased",
        "distilbert-base-uncased",
    ]

    tok = None
    mdl = None
    last_err = None
    for name in model_name_candidates:
        try:
            tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
            mdl = AutoModel.from_pretrained(name, local_files_only=True)
            print("Loaded Transformers encoder:", name)
            break
        except Exception as e:
            last_err = e
            tok, mdl = None, None
            continue
    if tok is None or mdl is None:
        raise RuntimeError(
            "Could not load any local Transformers model for fallback. "
            f"Tried: {model_name_candidates}. Last error: {last_err}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(device)
    mdl.eval()

    all_emb = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = mdl(**enc)
            token_embeddings = out.last_hidden_state  # [B, T, H]
            attn_mask = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]
            masked = token_embeddings * attn_mask
            summed = masked.sum(dim=1)  # [B, H]
            counts = attn_mask.sum(dim=1).clamp(min=1)
            mean_pooled = summed / counts
            emb = mean_pooled.detach().cpu().numpy()
            all_emb.append(emb)

    emb = np.vstack(all_emb)
    emb = _l2_normalize(emb)
    return emb


X = encode_with_sentence_transformers(train_texts, batch_size=256)
X_test = encode_with_sentence_transformers(test_texts, batch_size=256)

if X is None or X_test is None:
    print("Using Transformers mean-pooling encoder...")
    X = encode_with_transformers_mean_pool(train_texts, batch_size=64, max_length=128)
    X_test = encode_with_transformers_mean_pool(
        test_texts, batch_size=64, max_length=128
    )

y = df["score"].astype(float).values

print("X shape:", X.shape, "X_test shape:", X_test.shape, "y shape:", y.shape)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3576324544.py in <cell line: 0>()
     81 if X is None or X_test is None:
     82     print("Using Transformers mean-pooling encoder...")
---> 83     X = encode_with_transformers_mean_pool(train_texts, batch_size=64, max_length=128)
     84     X_test = encode_with_transformers_mean_pool(
     85         test_texts, batch_size=64, max_length=128

/tmp/ipykernel_11/3576324544.py in encode_with_transformers_mean_pool(texts, batch_size, max_length)
     40             continue
     41     if tok is None or mdl is None:
---> 42         raise RuntimeError(
     43             "Could not load any local Transformers model for fallback. "
     44             f"Tried: {model_name_candidates}. Last error: {last_err}"

RuntimeError: Could not load any local Transformers model for fallback. Tried: ['sentence-transformers/all-MiniLM-L6-v2', 'microsoft/MiniLM-L12-H384-uncased', 'distilbert-base-uncased']. Last error: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

## === cell 9
from numpy.typing import NDArray


def add_similarity_features(
    E: NDArray[np.float64], E_test: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    mean_abs = np.mean(np.abs(E), axis=1, keepdims=True)
    mean_sq = np.mean(E * E, axis=1, keepdims=True)
    feats = np.hstack([E, mean_abs, mean_sq])

    mean_abs_t = np.mean(np.abs(E_test), axis=1, keepdims=True)
    mean_sq_t = np.mean(E_test * E_test, axis=1, keepdims=True)
    feats_test = np.hstack([E_test, mean_abs_t, mean_sq_t])

    return feats.astype(np.float32, copy=False), feats_test.astype(
        np.float32, copy=False
    )


X2, X2_test = add_similarity_features(X, X_test)
print("X2 shape:", X2.shape, "X2_test shape:", X2_test.shape)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3338555852.py in <cell line: 0>()
     18 
     19 
---> 20 X2, X2_test = add_similarity_features(X, X_test)
     21 print("X2 shape:", X2.shape, "X2_test shape:", X2_test.shape)
     22 

/tmp/ipykernel_11/3338555852.py in add_similarity_features(E, E_test)
      5     E: NDArray[np.float64], E_test: NDArray[np.float64]
      6 ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
----> 7     mean_abs = np.mean(np.abs(E), axis=1, keepdims=True)
      8     mean_sq = np.mean(E * E, axis=1, keepdims=True)
      9     feats = np.hstack([E, mean_abs, mean_sq])

TypeError: bad operand type for abs(): 'NoneType'

## === cell 10
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

X_tr, X_va, y_tr, y_va = train_test_split(X2, y, test_size=0.25, random_state=42)

reg = Ridge(alpha=1.0, random_state=42)
reg.fit(X_tr, y_tr)

va_pred = reg.predict(X_va)
va_corr = np.corrcoef(va_pred.reshape(-1), y_va.reshape(-1))[0, 1]
va_rmse = mean_squared_error(y_va, va_pred, squared=False)

print("Validation pearson:", float(va_corr))
print("Validation RMSE:", float(va_rmse))



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/453619001.py in <cell line: 0>()
      3 from sklearn.metrics import mean_squared_error
      4 
----> 5 X_tr, X_va, y_tr, y_va = train_test_split(X2, y, test_size=0.25, random_state=42)
      6 
      7 reg = Ridge(alpha=1.0, random_state=42)

NameError: name 'X2' is not defined

## === cell 11
reg_full = Ridge(alpha=1.0, random_state=42)
reg_full.fit(X2, y)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1954777866.py in <cell line: 0>()
      1 reg_full = Ridge(alpha=1.0, random_state=42)
----> 2 reg_full.fit(X2, y)
      3 

NameError: name 'X2' is not defined

## === cell 12
preds = reg_full.predict(X2_test).astype(float).reshape(-1)
preds = np.clip(preds, 0.0, 1.0)
print(preds[:10], len(preds))



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3233586458.py in <cell line: 0>()
----> 1 preds = reg_full.predict(X2_test).astype(float).reshape(-1)
      2 preds = np.clip(preds, 0.0, 1.0)
      3 print(preds[:10], len(preds))
      4 

NameError: name 'X2_test' is not defined

## === cell 13
submission_df = pd.DataFrame({"id": eval_df["id"].values, "score": preds})
submission_df.to_csv("submission.csv", index=False)

print(submission_df.head())
print("Wrote submission.csv with shape:", submission_df.shape)
print("Columns:", submission_df.columns.tolist())
assert submission_df.shape[0] == eval_df.shape[0]
assert submission_df.columns.tolist() == ["id", "score"]

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2612253854.py in <cell line: 0>()
----> 1 submission_df = pd.DataFrame({"id": eval_df["id"].values, "score": preds})
      2 submission_df.to_csv("submission.csv", index=False)
      3 
      4 print(submission_df.head())
      5 print("Wrote submission.csv with shape:", submission_df.shape)

NameError: name 'preds' is not defined
