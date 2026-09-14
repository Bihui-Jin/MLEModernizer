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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
transformers==4.53.3

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

0.8039026604903498

# 6. Current score

0.56359

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.56297) has done: 'I fix the protobuf import issue by setting the environment variable before importing transformers, replace the missing model checkpoint with a publicly‑available sentence‑transformer embedding model, and simplify inference to compute cosine similarity between anchor and target embeddings (mapped to [0, 1]). This removes the failing dataset/DataLoader code, ensures a valid `submission.csv` is written, and provides a reasonable correlation score without altering the overall workflow.'
- What this solution (achieved 0.58403) has done: 'I fixed the protobuf import error by switching from the raw `transformers` model loading to the `sentence_transformers.SentenceTransformer` wrapper, which avoids the problematic protobuf call and also handles pooling internally. I upgraded the embedding model to a stronger pretrained checkpoint (“paraphrase‑mpnet‑base‑v2”) that generally yields higher cosine‑similarity correlation for semantic similarity tasks, moving the validation score closer to the target. The rest of the pipeline (reading the test file, computing cosine similarity, rescaling to [0, 1], and writing the submission) is kept unchanged but rewritten for clarity and robustness.'
- What this solution (achieved 0.56359) has done: 'I moved the protobuf environment setting to the very top, added a safe fallback model load, and incorporated the `context` field into the sentences that are encoded so the embeddings capture more information, which should improve the Pearson correlation while keeping the original cosine‑similarity–based prediction pipeline unchanged. The script now runs without the previous import error and writes a proper `submission.csv` file.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "sentence-transformers/paraphrase-mpnet-base-v2"

try:
    model = SentenceTransformer(model_name, device=device)
except Exception:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
test_path = os.path.join(
    "..", "input", "us-patent-phrase-to-phrase-matching", "test.csv"
)
test_df = pd.read_csv(test_path)




## === cell 2
anchor_sentences = [
    f"{a} [SEP] {c}" for a, c in zip(test_df["anchor"], test_df["context"])
]
target_sentences = [
    f"{t} [SEP] {c}" for t, c in zip(test_df["target"], test_df["context"])
]

anchor_embeddings = model.encode(
    anchor_sentences, batch_size=64, convert_to_tensor=True, device=device
)
target_embeddings = model.encode(
    target_sentences, batch_size=64, convert_to_tensor=True, device=device
)




## === cell 3
cos_sim = F.cosine_similarity(anchor_embeddings, target_embeddings, dim=1)
preds = ((cos_sim + 1) / 2).cpu().numpy().clip(0, 1)




## === cell 4
submission = pd.DataFrame({"id": test_df["id"], "score": preds})
submission.to_csv("submission.csv", index=False)
