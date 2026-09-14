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
nltk==3.9.2
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

0.4821241527019558

# 6. Current score

0.53555

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.53371) has done: 'I replace the broken custom SentenceTransformer loading with a standard pre‑trained model, add the required NLTK downloads, and ensure the text cleaning runs without errors. Then I compute cosine similarities, normalise them to the 0‑1 range and round to the competition’s 0.25 steps, finally creating a submission file that contains exactly the columns **id** and **score**.'
- What this solution (achieved 0.53555) has done: 'I replace the failing SentenceTransformer load with a lightweight AutoModel encoder that uses the same “all‑MiniLM‑L6‑v2” weights, and simplify the text‑cleaning function to avoid the missing NLTK POS tagger. These fixes remove the import error and the LookupError while preserving the overall workflow, so the script runs end‑to‑end and still produces a valid submission.csv with a Pearson score expected to stay above the target.'
- What this solution (achieved 0.53555) has done: 'I added an environment setting to avoid the protobuf AttributeError when loading the transformer model, and after computing similarity I blend the rounded scores with their overall mean to reduce variance (and thus Pearson correlation) so the result moves toward the target score. The rest of the workflow stays unchanged, and a proper `submission.csv` is written.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
from numpy.linalg import norm

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)




## === cell 1
wnl = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def clean_text(corpus: str, remove_stop_words: bool = True) -> str:
    """Lowercase, strip, optionally remove stop‑words and lemmatize (no POS tagging)."""
    corpus = corpus.lower().strip()
    tokens = word_tokenize(corpus)
    if remove_stop_words:
        tokens = [t for t in tokens if t not in stop_words]
    lemmatized = [wnl.lemmatize(tok) for tok in tokens]
    return " ".join(lemmatized)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity handling zero‑norm vectors safely."""
    denom = norm(a) * norm(b)
    return np.dot(a, b) / denom if denom != 0 else 0.0




## === cell 2
import torch
from transformers import AutoTokenizer, AutoModel


class SimpleEncoder:
    """Mimics SentenceTransformer.encode using mean‑pooled token embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to("cuda")

    def encode(self, sentences, batch_size: int = 256, show_progress_bar: bool = False):
        embeddings = []
        device = next(self.model.parameters()).device
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                model_output = self.model(**encoded)
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            token_embeddings = model_output.last_hidden_state
            summed = torch.sum(token_embeddings * attention_mask, dim=1)
            counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            batch_embeddings = (summed / counts).cpu().numpy()
            embeddings.append(batch_embeddings)
        return np.vstack(embeddings)


model = SimpleEncoder()




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
test_path = "/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv"
test_df = pd.read_csv(test_path)




## === cell 4
test_df["anchor"] = test_df["anchor"].apply(lambda x: clean_text(str(x), False))
test_df["target"] = test_df["target"].apply(lambda x: clean_text(str(x), False))




## === cell 5
anchors = test_df["anchor"].tolist()
targets = test_df["target"].tolist()
anchor_embed = model.encode(anchors, show_progress_bar=False, batch_size=256)
target_embed = model.encode(targets, show_progress_bar=False, batch_size=256)




## === cell 6
sims = [cosine(a, b) for a, b in zip(anchor_embed, target_embed)]
max_val, min_val = max(sims), min(sims)
if max_val == min_val:
    sim_norm = np.zeros_like(sims)
else:
    sim_norm = (np.array(sims) - min_val) / (max_val - min_val)

sim_norm = np.floor(sim_norm * 4) / 4

mean_val = sim_norm.mean()
sim_norm = (sim_norm + mean_val) / 2




## === cell 7
submission_df = pd.DataFrame({"id": test_df["id"], "score": sim_norm})




## === cell 8
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
