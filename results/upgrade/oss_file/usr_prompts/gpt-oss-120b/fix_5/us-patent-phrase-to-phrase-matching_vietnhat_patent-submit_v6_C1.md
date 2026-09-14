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

0.8210748332613725

# 6. Current score

0.38267

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.38428) has done: 'The timeout was caused by encoding each row individually with `SentenceTransformer.encode`, which runs the model many thousands of times.  
We replace the per‑row `apply` calls with a single batched encode for all “context + anchor” strings and all “target” strings, then concatenate the two matrices. This uses the same model, yields identical embeddings, and reduces runtime dramatically while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.55296) has done: 'I fix the protobuf import error by setting the compatibility environment variable before importing any libraries, and I replace the high‑dimensional concatenated embeddings with a single cosine‑similarity feature (which better captures phrase similarity). This keeps the overall pipeline unchanged while providing a more informative feature for the Ridge regressor, helping the Pearson correlation move toward the target score. The script now writes a proper `submission.csv` file.'
- What this solution (achieved 0.38267) has done: 'I fixed the protobuf import error by removing the `SentenceTransformer` import, which caused the incompatibility, and replaced it with a lightweight embedding pipeline using HuggingFace `AutoTokenizer` and `AutoModel`. The new pipeline batches encoding for efficiency, builds concatenated embeddings of “context + anchor” and “target”, and feeds them to a Ridge regressor (instead of just cosine similarity). This richer feature set markedly improves the Pearson correlation, moving the score toward the target while keeping the overall workflow unchanged and ensuring a proper `submission.csv` is written.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"



## === cell 1
TRAIN_PATH = "../input/us-patent-phrase-to-phrase-matching/train.csv"
TEST_PATH = "../input/us-patent-phrase-to-phrase-matching/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)



## === cell 2
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def embed_texts(texts, batch_size=64):
    """Return NumPy array of mean‑pooled embeddings for a list of strings."""
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            outputs = model(**encoded)
            last_hidden = outputs.last_hidden_state  # (bs, seq_len, dim)
            mask = encoded["attention_mask"].unsqueeze(-1).expand(last_hidden.shape)
            masked_embeddings = last_hidden * mask
            summed = masked_embeddings.sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            batch_emb = (summed / counts).cpu().numpy()
            embeddings.append(batch_emb)
    return np.vstack(embeddings)


train_txt1 = (train_df["context"] + " " + train_df["anchor"]).tolist()
train_txt2 = train_df["target"].tolist()

vec1_train = embed_texts(train_txt1)
vec2_train = embed_texts(train_txt2)

X = np.concatenate([vec1_train, vec2_train], axis=1)

y = train_df["score"].values.astype(np.float32)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

regressor = Ridge(alpha=1.0, random_state=42)
regressor.fit(X_train, y_train)

val_pred = regressor.predict(X_val)
val_rmse = mean_squared_error(y_val, val_pred, squared=False)
print(f"Validation RMSE: {val_rmse:.4f}")



## === cell 4
test_txt1 = (test_df["context"] + " " + test_df["anchor"]).tolist()
test_txt2 = test_df["target"].tolist()

vec1_test = embed_texts(test_txt1)
vec2_test = embed_texts(test_txt2)

X_test = np.concatenate([vec1_test, vec2_test], axis=1)

test_pred = regressor.predict(X_test)
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = pd.DataFrame({"id": test_df["id"], "score": test_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
