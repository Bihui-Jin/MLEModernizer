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

0.38855

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.38855) has done: 'I remove the failing SentenceTransformer loading (which caused a protobuf AttributeError) and replace it with a simple Jaccard similarity based on token overlap—this avoids heavy dependencies and still provides a reasonable baseline that should reach the target Pearson score. The script now reads the test CSV, computes the Jaccard score for each pair, stores the results, and writes a correctly‑named `submission.csv` file. No other logic is altered.'
- What this solution (achieved 0.38855) has done: 'I keep the overall structure but improve the similarity calculation: lower‑case the text, strip punctuation, and combine the Jaccard overlap of the anchor‑target pair with the overlap between each phrase and the provided context. A weighted average of these three overlaps (0.5 for anchor‑target, 0.25 each for the other two) gives a richer similarity estimate that should raise the Pearson score toward the target while preserving the original workflow.'
- What this solution (achieved 0.5441) has done: 'I add an optional SentenceTransformer‑based similarity that computes cosine similarity between the anchor, target and context embeddings, using the same 0.5/0.25/0.25 weighting as the original Jaccard version. If the model cannot be loaded (e.g., protobuf issues) the code gracefully fall back to the existing Jaccard implementation, ensuring the script always runs and writes a valid `submission.csv`. This richer semantic similarity is expected to raise the Pearson score toward the target while preserving the overall pipeline.'
- What this solution (achieved 0.38855) has done: 'The update adds robust file‑path handling that checks several common Kaggle directories for the test CSV, preventing the FileNotFoundError. It also refines the Jaccard‑based similarity by incorporating an additional overlap between the combined anchor‑target tokens and the context, slightly re‑weighting the components to improve the Pearson correlation while keeping the original logic intact. The script now reliably reads the data, computes scores, and writes a proper `submission.csv`.'

# 9. Code solution

## === cell 0
_use_transformer = False
_model = None




## === cell 1
import pandas as pd
import numpy as np
import string
import os
from numpy.linalg import norm

possible_paths = [
    os.path.join("input", "us-patent-phrase-to-phrase-matching", "test.csv"),
    os.path.join("kaggle", "input", "us-patent-phrase-to-phrase-matching", "test.csv"),
    os.path.join("data", "us-patent-phrase-to-phrase-matching", "test.csv"),
    os.path.join("kaggle", "input", "us-patent-phrase-to-phrase-matching", "test.csv"),
    os.path.join("..", "input", "us-patent-phrase-to-phrase-matching", "test.csv"),
]
test_path = None
for p in possible_paths:
    if os.path.exists(p):
        test_path = p
        break
if test_path is None:
    raise FileNotFoundError("test.csv not found in expected locations.")

csv_data = pd.read_csv(test_path)


def tokenize(text):
    """Lower‑case, remove basic punctuation and split on whitespace."""
    if pd.isna(text):
        return set()
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    clean = text.translate(translator).lower()
    return set(clean.split())


def cos_sim(a, b):
    """Cosine similarity between two 1‑D numpy arrays."""
    if a.ndim != 1 or b.ndim != 1:
        return 0.0
    denom = norm(a) * norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


preds = []

if _use_transformer:
    anchors = csv_data["anchor"].fillna("").tolist()
    targets = csv_data["target"].fillna("").tolist()
    contexts = csv_data["context"].fillna("").tolist()

    anchor_emb = _model.encode(
        anchors, convert_to_numpy=True, show_progress_bar=False, batch_size=64
    )
    target_emb = _model.encode(
        targets, convert_to_numpy=True, show_progress_bar=False, batch_size=64
    )
    context_emb = _model.encode(
        contexts, convert_to_numpy=True, show_progress_bar=False, batch_size=64
    )

    for idx, row in csv_data.iterrows():
        at = cos_sim(anchor_emb[idx], target_emb[idx])
        ac = cos_sim(anchor_emb[idx], context_emb[idx])
        tc = cos_sim(target_emb[idx], context_emb[idx])

        score = 0.5 * at + 0.25 * ac + 0.25 * tc
        preds.append([row.id, score])
else:
    for _, row in csv_data.iterrows():
        t_anchor = tokenize(row.anchor)
        t_target = tokenize(row.target)
        t_context = tokenize(row.context)

        jt = (
            len(t_anchor & t_target) / len(t_anchor | t_target)
            if (t_anchor or t_target)
            else 0.0
        )
        ja = (
            len(t_anchor & t_context) / len(t_anchor | t_context)
            if (t_anchor or t_context)
            else 0.0
        )
        jc = (
            len(t_target & t_context) / len(t_target | t_context)
            if (t_target or t_context)
            else 0.0
        )
        combined = t_anchor | t_target
        jac = (
            len(combined & t_context) / len(combined | t_context)
            if (combined or t_context)
            else 0.0
        )

        score = 0.45 * jt + 0.225 * ja + 0.225 * jc + 0.10 * jac
        preds.append([row.id, score])

result = pd.DataFrame(preds, columns=["id", "score"])
result.to_csv("submission.csv", index=False)
