# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    from google.protobuf.message_factory import MessageFactory

    if not hasattr(MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass  # If protobuf is not installed, the later import will raise its own error

from pathlib import Path

iskaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "")




## === cell 1
path = Path("us-patent-phrase-to-phrase-matching")
if not iskaggle and not path.exists():
    import zipfile, kaggle

    kaggle.api.competition_download_cli(str(path))
    zipfile.ZipFile(f"{path}.zip").extractall(path)

if iskaggle:
    path = Path("../input/us-patent-phrase-to-phrase-matching")

print("Data path:", path)




## === cell 2
import pandas as pd

train_df = pd.read_csv(path / "train.csv")
test_df = pd.read_csv(path / "test.csv")

print("Train shape:", train_df.shape, "Test shape:", test_df.shape)




## === cell 3
from sentence_transformers import SentenceTransformer
import numpy as np

embedder = SentenceTransformer("all-MiniLM-L12-v2")


def encode_series(series):
    texts = series.astype(str).tolist()
    return embedder.encode(texts, batch_size=64, show_progress_bar=False)


anchor_emb = encode_series(train_df["anchor"])
target_emb = encode_series(train_df["target"])
context_emb = encode_series(train_df["context"])

anchor_emb_test = encode_series(test_df["anchor"])
target_emb_test = encode_series(test_df["target"])
context_emb_test = encode_series(test_df["context"])


def cosine_sim(a, b):
    dot = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    return dot / (norm_a * norm_b + 1e-8)


def build_features(a, t, c):
    cos_at = cosine_sim(a, t)[:, None]
    cos_ac = cosine_sim(a, c)[:, None]
    cos_tc = cosine_sim(t, c)[:, None]

    diff_at = np.abs(a - t)
    diff_ac = np.abs(a - c)
    diff_tc = np.abs(t - c)

    prod_at = a * t
    prod_ac = a * c
    prod_tc = t * c

    norm_a = np.linalg.norm(a, axis=1)[:, None]
    norm_t = np.linalg.norm(t, axis=1)[:, None]
    norm_c = np.linalg.norm(c, axis=1)[:, None]

    diff_norm_at = np.abs(norm_a - norm_t)
    diff_norm_ac = np.abs(norm_a - norm_c)
    diff_norm_tc = np.abs(norm_t - norm_c)

    return np.hstack(
        [
            a,
            t,
            c,  # raw embeddings
            cos_at,
            cos_ac,
            cos_tc,  # cosine features
            diff_at,
            diff_ac,
            diff_tc,  # absolute differences
            prod_at,
            prod_ac,
            prod_tc,  # products
            norm_a,
            norm_t,
            norm_c,  # embedding norms
            diff_norm_at,
            diff_norm_ac,
            diff_norm_tc,  # **new norm‑difference features**
        ]
    )


X_train = build_features(anchor_emb, target_emb, context_emb)
X_test = build_features(anchor_emb_test, target_emb_test, context_emb_test)

y_train = train_df["score"].values




## === cell 4
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)

candidate_alphas = np.logspace(-3, 3, 80)  # 0.001 … 1000
best_alpha = candidate_alphas[0]
best_pearson = -1.0
best_model = None

for a in candidate_alphas:
    ridge = Ridge(alpha=a, random_state=42)
    ridge.fit(X_tr, y_tr)
    val_pred = ridge.predict(X_val)
    pearson = np.corrcoef(val_pred, y_val)[0, 1]
    if pearson > best_pearson:
        best_pearson = pearson
        best_alpha = a
        best_model = ridge

linreg = LinearRegression()
linreg.fit(X_tr, y_tr)
val_pred_lr = linreg.predict(X_val)
pearson_lr = np.corrcoef(val_pred_lr, y_val)[0, 1]
if pearson_lr > best_pearson:
    best_pearson = pearson_lr
    best_alpha = None  # indicate plain linear model
    best_model = linreg

print(
    f"Chosen model: {'LinearRegression' if best_alpha is None else f'Ridge(alpha={best_alpha:.6f})'} "
    f"with validation Pearson {best_pearson:.4f}"
)

if best_alpha is None:
    final_model = LinearRegression()
    final_model.fit(X_train_scaled, y_train)
else:
    final_model = Ridge(alpha=best_alpha, random_state=42)
    final_model.fit(X_train_scaled, y_train)

train_pred_full = final_model.predict(X_train_scaled)
train_pearson = np.corrcoef(train_pred_full, y_train)[0, 1]
print("Training Pearson correlation on full data:", train_pearson)




## === cell 5
test_preds = final_model.predict(X_test_scaled)
test_preds = np.clip(test_preds, 0.0, 1.0)




## === cell 6
submission = pd.DataFrame({"id": test_df["id"], "score": test_preds})
submission = submission[["id", "score"]]
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, rows:", submission.shape[0])
