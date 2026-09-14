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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True  # faster convolution kernels on GPU



## === cell 1
TRAIN_PATH = "../input/us-patent-phrase-to-phrase-matching/train.csv"
TEST_PATH = "../input/us-patent-phrase-to-phrase-matching/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)



## === cell 2
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedder = SentenceTransformer(MODEL_NAME, device=device.type)


def embed_texts(texts, batch_size=4096):
    """
    Encode a list of strings into a NumPy array of embeddings using SentenceTransformer.
    Larger batch_size reduces the number of Python‑level calls.
    """
    with torch.no_grad():
        embeddings = embedder.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
            device=device.type,
        )
    return embeddings.astype(np.float32)  # keep dtype consistent and memory‑efficient


train_txt1 = (train_df["context"] + " " + train_df["anchor"]).tolist()
train_txt2 = train_df["target"].tolist()

vec1_train = embed_texts(train_txt1)
vec2_train = embed_texts(train_txt2)

n_train, dim = vec1_train.shape
X = np.empty((n_train, 1 + 4 * dim), dtype=np.float32)

dot_prod = np.sum(vec1_train * vec2_train, axis=1)
norm1 = np.linalg.norm(vec1_train, axis=1)
norm2 = np.linalg.norm(vec2_train, axis=1)
cosine_sim = dot_prod / (norm1 * norm2 + 1e-8)
X[:, 0] = cosine_sim

X[:, 1 : 1 + dim] = vec1_train
X[:, 1 + dim : 1 + 2 * dim] = vec2_train
X[:, 1 + 2 * dim : 1 + 3 * dim] = np.abs(vec1_train - vec2_train)
X[:, 1 + 3 * dim :] = vec1_train * vec2_train

y = train_df["score"].values.astype(np.float32)



## === cell 3
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.1, random_state=42
)

regressor = Ridge(alpha=1.0, solver="sag", random_state=42, max_iter=1000)
regressor.fit(X_train, y_train)

val_pred = regressor.predict(X_val)
val_rmse = mean_squared_error(y_val, val_pred, squared=False)
print(f"Validation RMSE: {val_rmse:.4f}")



## === cell 4
test_txt1 = (test_df["context"] + " " + test_df["anchor"]).tolist()
test_txt2 = test_df["target"].tolist()

vec1_test = embed_texts(test_txt1)
vec2_test = embed_texts(test_txt2)

n_test, dim = vec1_test.shape
X_test = np.empty((n_test, 1 + 4 * dim), dtype=np.float32)

dot_prod_test = np.sum(vec1_test * vec2_test, axis=1)
norm1_test = np.linalg.norm(vec1_test, axis=1)
norm2_test = np.linalg.norm(vec2_test, axis=1)
cosine_sim_test = dot_prod_test / (norm1_test * norm2_test + 1e-8)
X_test[:, 0] = cosine_sim_test

X_test[:, 1 : 1 + dim] = vec1_test
X_test[:, 1 + dim : 1 + 2 * dim] = vec2_test
X_test[:, 1 + 2 * dim : 1 + 3 * dim] = np.abs(vec1_test - vec2_test)
X_test[:, 1 + 3 * dim :] = vec1_test * vec2_test

X_test_scaled = scaler.transform(X_test)

test_pred = regressor.predict(X_test_scaled)
test_pred = np.clip(test_pred, 0.0, 1.0)

submission = pd.DataFrame({"id": test_df["id"], "score": test_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
