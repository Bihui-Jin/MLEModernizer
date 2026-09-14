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

3.10

# 3. Installed packages

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

0.4879785933288825

# 6. Current score

0.56328

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.56328) has done: 'I remove the failing pip-install cell (it’s pulling an incompatible protobuf build) and instead rely on the already-installed `sentence-transformers==4.1.0` in your Kaggle environment. Then I make the model path robust by trying a few standard Kaggle input locations and falling back to a small default SentenceTransformer model if the custom path isn’t present. Finally, I keep your core inference logic (cosine similarity of embeddings) but vectorize encoding in batches to avoid timeouts and ensure `submission.csv` is written with the correct `id,score` columns.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from numpy.linalg import norm

from sentence_transformers import SentenceTransformer

np.random.seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
TEST_PATH_CANDIDATES = [
    "/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv",
    "/kaggle/data/us-patent-phrase-to-phrase-matching/test.csv",
    "/kaggle/input/test.csv",
    "/kaggle/data/test.csv",
]
test_path = next((p for p in TEST_PATH_CANDIDATES if os.path.exists(p)), None)
if test_path is None:
    raise FileNotFoundError(
        f"Could not find test.csv in any of: {TEST_PATH_CANDIDATES}"
    )

csv_data = pd.read_csv(test_path)

MODEL_PATH_CANDIDATES = [
    "../input/notebook129e094f5a/models/bert-base-nli-mean-tokens",
    "/kaggle/input/notebook129e094f5a/models/bert-base-nli-mean-tokens",
]
model_path = next((p for p in MODEL_PATH_CANDIDATES if os.path.exists(p)), None)

if model_path is not None:
    model = SentenceTransformer(model_path)
else:
    model = SentenceTransformer("all-MiniLM-L6-v2")

result = pd.DataFrame(
    {
        "id": csv_data["id"].astype(str).values,
        "score": np.zeros(len(csv_data), dtype=float),
    }
)



## === cell 2

anchors = csv_data["anchor"].fillna("").astype(str).tolist()
targets = csv_data["target"].fillna("").astype(str).tolist()

batch_size = 256

anchor_embs = model.encode(
    anchors,
    batch_size=batch_size,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=False,
)
target_embs = model.encode(
    targets,
    batch_size=batch_size,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=False,
)

num = np.sum(anchor_embs * target_embs, axis=1)
den = np.linalg.norm(anchor_embs, axis=1) * np.linalg.norm(target_embs, axis=1)
den = np.where(den == 0, 1e-12, den)
scores = num / den

scores = np.clip(scores, 0.0, 1.0)

result["score"] = scores.astype(float)

result.to_csv("submission.csv", index=False)

print(result.head())
print("Wrote submission.csv with shape:", result.shape)
