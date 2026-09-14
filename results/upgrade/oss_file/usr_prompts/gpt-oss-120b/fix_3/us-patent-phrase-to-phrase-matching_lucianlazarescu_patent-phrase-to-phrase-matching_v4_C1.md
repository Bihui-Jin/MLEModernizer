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

0.7837437211583778

# 6. Current score

0.56407

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.56297) has done: 'I replace the failing HuggingFace datasets and Trainer pipeline with a lightweight embedding‑based regression that avoids the protobuf import error, computes cosine similarity between the anchor and target sentences using a pretrained Sentence‑Transformer, fits a simple linear model on the training data, and writes a correctly formatted `submission.csv` file. This fixes all runtime errors and yields a valid submission while keeping the overall approach (using transformer embeddings) close to the original intent.'
- What this solution (achieved 0.56407) has done: 'The fix adds a protobuf‑compatibility hint, wraps the SentenceTransformer import in a safe try/except and falls back to a TF‑IDF based embedding if needed. It also enriches the model with several similarity features (anchor‑target, anchor‑context, target‑context cosine similarities) and fits a Ridge regression, which improves correlation while keeping the original lightweight regression approach. The script now reliably writes a correctly formatted `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer

    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    USING_TRANSFORMER = True
    print("Using SentenceTransformer embeddings.")
except Exception as e:
    print("SentenceTransformer import failed – falling back to TF‑IDF embeddings.")
    USING_TRANSFORMER = False
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf_vectorizer = TfidfVectorizer(max_features=300)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
base_path = "/kaggle/input/us-patent-phrase-to-phrase-matching/"
train_path = os.path.join(base_path, "train.csv")
test_path = os.path.join(base_path, "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")




## === cell 2
def compute_embeddings(texts, batch_size=64):
    """Encode a list of texts to normalized numpy embeddings."""
    if USING_TRANSFORMER:
        embeddings = embed_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings
    else:
        tfidf = tfidf_vectorizer.fit(texts)
        vecs = tfidf.transform(texts).toarray()
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs / norms




## === cell 3
train_anchor = train_df["anchor"].astype(str).tolist()
train_target = train_df["target"].astype(str).tolist()
train_context = train_df["context"].astype(str).tolist()

print("Encoding training anchors...")
emb_anchor = compute_embeddings(train_anchor)
print("Encoding training targets...")
emb_target = compute_embeddings(train_target)
print("Encoding training contexts...")
emb_context = compute_embeddings(train_context)

cos_at = np.sum(emb_anchor * emb_target, axis=1)
cos_ac = np.sum(emb_anchor * emb_context, axis=1)
cos_tc = np.sum(emb_target * emb_context, axis=1)

X_train = np.vstack([cos_at, cos_ac, cos_tc, np.ones_like(cos_at)]).T  # intercept

y_train = train_df["score"].values.astype(float)

from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1e-3, fit_intercept=False)
ridge.fit(X_train, y_train)
print(f"Fitted ridge coefficients: {ridge.coef_}")



## === cell 4
test_anchor = test_df["anchor"].astype(str).tolist()
test_target = test_df["target"].astype(str).tolist()
test_context = test_df["context"].astype(str).tolist()

print("Encoding test anchors...")
emb_anchor_test = compute_embeddings(test_anchor)
print("Encoding test targets...")
emb_target_test = compute_embeddings(test_target)
print("Encoding test contexts...")
emb_context_test = compute_embeddings(test_context)

cos_at_test = np.sum(emb_anchor_test * emb_target_test, axis=1)
cos_ac_test = np.sum(emb_anchor_test * emb_context_test, axis=1)
cos_tc_test = np.sum(emb_target_test * emb_context_test, axis=1)

X_test = np.vstack([cos_at_test, cos_ac_test, cos_tc_test, np.ones_like(cos_at_test)]).T

preds = ridge.predict(X_test)
preds = np.clip(preds, 0.0, 1.0)  # keep within valid score range



## === cell 5
submission = pd.DataFrame({"id": test_df["id"], "score": preds})

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
