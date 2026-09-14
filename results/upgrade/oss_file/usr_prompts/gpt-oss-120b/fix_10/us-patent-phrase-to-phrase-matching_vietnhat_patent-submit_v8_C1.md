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

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import torch
import multiprocessing
from sklearn.linear_model import Ridge
from sentence_transformers import SentenceTransformer

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

torch.set_num_threads(1)

torch.set_num_threads(multiprocessing.cpu_count())

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
try:
    from google.protobuf import message_factory

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass




## === cell 1
TRAIN_PATH = "../input/us-patent-phrase-to-phrase-matching/train.csv"
TEST_PATH = "../input/us-patent-phrase-to-phrase-matching/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)




## === cell 2
train_texts = (
    train_df["context"]
    .str.cat(train_df["anchor"], sep=" ")
    .str.cat(train_df["target"], sep=" ")
    .tolist()
)
test_texts = (
    test_df["context"]
    .str.cat(test_df["anchor"], sep=" ")
    .str.cat(test_df["target"], sep=" ")
    .tolist()
)

train_anchor_texts = train_df["anchor"].tolist()
train_target_texts = train_df["target"].tolist()
test_anchor_texts = test_df["anchor"].tolist()
test_target_texts = test_df["target"].tolist()

all_texts = (
    train_texts
    + test_texts
    + train_anchor_texts
    + train_target_texts
    + test_anchor_texts
    + test_target_texts
)

codes, uniques = pd.Series(all_texts).factorize(sort=False)

model_name = "sentence-transformers/paraphrase-mpnet-base-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(model_name, device=device)

max_workers = min(8, multiprocessing.cpu_count())

unique_emb = embedder.encode(
    uniques.tolist(),
    batch_size=512,
    show_progress_bar=False,
    convert_to_numpy=True,
    normalize_embeddings=True,
    num_workers=max_workers,
)

all_emb = unique_emb[codes]

offset = 0
n_train = len(train_texts)
train_comb_emb = all_emb[offset : offset + n_train]
offset += n_train

n_test = len(test_texts)
test_comb_emb = all_emb[offset : offset + n_test]
offset += n_test

n_train_anchor = len(train_anchor_texts)
train_anchor_emb = all_emb[offset : offset + n_train_anchor]
offset += n_train_anchor

n_train_target = len(train_target_texts)
train_target_emb = all_emb[offset : offset + n_train_target]
offset += n_train_target

n_test_anchor = len(test_anchor_texts)
test_anchor_emb = all_emb[offset : offset + n_test_anchor]
offset += n_test_anchor

n_test_target = len(test_target_texts)
test_target_emb = all_emb[offset : offset + n_test_target]

train_cosine = np.sum(train_anchor_emb * train_target_emb, axis=1, keepdims=True)
test_cosine = np.sum(test_anchor_emb * test_target_emb, axis=1, keepdims=True)

train_features = np.hstack([train_comb_emb, train_cosine])
test_features = np.hstack([test_comb_emb, test_cosine])

train_targets = train_df["score"].values.astype(np.float32)




## === cell 3
regressor = Ridge(alpha=0.01)
regressor.fit(train_features, train_targets)




## === cell 4
test_pred = regressor.predict(test_features)
test_pred = np.clip(test_pred, 0.0, 1.0)  # keep predictions in the valid range




## === cell 5
submission = pd.DataFrame({"id": test_df["id"], "score": test_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape: {submission.shape}")
