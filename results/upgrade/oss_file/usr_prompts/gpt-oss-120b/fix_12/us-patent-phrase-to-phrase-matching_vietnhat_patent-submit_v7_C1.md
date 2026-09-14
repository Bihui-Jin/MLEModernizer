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

0.8275192399122357

# 6. Current score

0.40304

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.52676) has done: 'The fix changes the Ridge regression to use a solver that works with the SciPy version bundled in the environment (`solver='lsqr'`). This avoids the `cg()` TypeError caused by the default `sparse_cg` solver. No other logic is altered, so the model, feature handling, and submission creation remain the same, yielding a valid `submission.csv` and moving the score toward the target.'
- What this solution (achieved 0.45432) has done: 'I keep the overall pipeline structure (TF‑IDF → linear model) but make small, targeted adjustments that are known to improve text‑regression performance: use a richer n‑gram range, enable sub‑linear term frequency scaling, and lower the Ridge regularisation strength. Adding a `StandardScaler` (compatible with sparse data) normalises the TF‑IDF features before regression, which often raises Pearson correlation without altering the core modelling approach.'
- What this solution (achieved 0.26617) has done: 'I enhance the text representation by adding character‑level TF‑IDF features alongside the existing word‑level TF‑IDF and reduce the Ridge regularisation (alpha = 0.01). These changes keep the overall TF‑IDF + linear‑model pipeline while providing richer features that typically raise Pearson correlation, moving the score closer to the target.'
- What this solution (achieved 0.25479) has done: 'I keep the overall TF‑IDF + Ridge pipeline but give the `context` field extra weight (by repeating it) and broaden the character‑level n‑grams, which usually captures more subtle lexical patterns and can raise the Pearson correlation without altering the core modelling approach. The rest of the code stays the same, ensuring the script still writes a correct `submission.csv`.'
- What this solution (achieved 0.40304) has done: 'We replace the TF‑IDF + Ridge pipeline with a sentence‑transformer embedding based ridge regression. Sentence embeddings capture semantic similarity far better than simple n‑grams, so this change is expected to raise the Pearson correlation toward the target while keeping the overall linear‑model approach and the same data handling and submission logic.'
- What this solution (achieved 0.40304) has done: 'We replace the failing `sentence_transformers` import with a lightweight embedding routine built from the HuggingFace `transformers` library, which avoids the protobuf error while keeping the same TF‑IDF → Ridge pipeline logic. The new `encode_texts` function tokenizes, runs the `"all-MiniLM-L6-v2"` model, and mean‑pools token embeddings to produce sentence vectors. The rest of the script (data loading, field combination, Ridge regression, clipping, and CSV export) stays unchanged, ensuring a valid `submission.csv` and a higher Pearson correlation than before.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["WANDB_DISABLED"] = "true"

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import torch
from transformers import AutoTokenizer, AutoModel




## === cell 1
import pathlib


def locate_data_dir():
    """
    Walk the current directory tree and return the first folder that
    contains both train.csv and test.csv.
    """
    for root, dirs, files in os.walk("."):
        if "train.csv" in files and "test.csv" in files:
            return pathlib.Path(root)
    raise FileNotFoundError(
        "Could not locate the competition data directory containing train.csv and test.csv."
    )


base_path = locate_data_dir()

train_path = base_path / "train.csv"
test_path = base_path / "test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


def combine_fields(df):
    """Concatenate context (repeated for extra weight), anchor, and target into a single text field."""
    context = df["context"].fillna("")
    anchor = df["anchor"].fillna("")
    target = df["target"].fillna("")
    return context + " " + context + " " + anchor + " " + target


X_train_raw = combine_fields(train_df)
y_train = train_df["score"].astype(float)  # scores are already in 0‑1 range
X_test_raw = combine_fields(test_df)




## === cell 2
TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
MODEL = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
MODEL.eval()
DEVICE = torch.device("cpu")
MODEL.to(DEVICE)


def mean_pooling(model_output, attention_mask):
    """
    Perform mean pooling on token embeddings.
    """
    token_embeddings = model_output[
        0
    ]  # First element of output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def encode_texts(texts, batch_size=64, show_progress_bar=False):
    """
    Encode a list of strings into sentence embeddings using the
    all‑MiniLM‑L6‑v2 model.
    """
    all_embeddings = []
    total = len(texts)
    for start_idx in range(0, total, batch_size):
        batch_texts = texts[start_idx : start_idx + batch_size]
        encoded = TOKENIZER.batch_encode_plus(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)

        with torch.no_grad():
            model_output = MODEL(input_ids=input_ids, attention_mask=attention_mask)

        batch_embeddings = mean_pooling(model_output, attention_mask)
        batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
        all_embeddings.append(batch_embeddings.cpu().numpy())

        if show_progress_bar:
            progress = min(start_idx + batch_size, total)
            print(f"\rEncoding batch {progress}/{total}", end="")

    if show_progress_bar:
        print()  # newline after progress

    return np.vstack(all_embeddings)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
X_train_emb = encode_texts(
    X_train_raw.tolist(),
    batch_size=64,
    show_progress_bar=True,
)

X_test_emb = encode_texts(
    X_test_raw.tolist(),
    batch_size=64,
    show_progress_bar=True,
)

ridge = Ridge(alpha=0.5, random_state=42, solver="auto")
ridge.fit(X_train_emb, y_train)

pred = ridge.predict(X_test_emb)
pred = np.clip(pred, 0.0, 1.0)

submission = pd.DataFrame({"id": test_df["id"], "score": pred})
submission.to_csv("submission.csv", index=False)
