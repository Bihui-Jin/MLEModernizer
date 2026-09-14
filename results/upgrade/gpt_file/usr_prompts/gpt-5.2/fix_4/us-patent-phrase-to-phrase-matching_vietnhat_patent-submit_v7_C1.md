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

-0.21385

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.01921) has done: 'I fix the import-time `protobuf` crash by forcing a compatible protobuf version behavior and importing `transformers` only after setting those environment variables. Then I fix the dataset length bug (your `__len__` returned number of keys instead of number of rows), which caused `Trainer.predict()` to run on only a few items and produced a length mismatch when creating the submission. Finally, I make the encoding use `return_tensors="np"` and implement a stable softmax so predictions are correctly shaped and the script always writes a valid `submission.csv` with `id,score`.'
- What this solution (achieved -0.21385) has done: 'We fix the protobuf/transformers import crash by explicitly forcing the pure-Python protobuf backend and setting `protobuf`’s internal implementation choice before importing `transformers` (this resolves the `MessageFactory.GetPrototype` AttributeError seen at import time). Then we keep your existing inference-only pipeline intact, but make the model/tokenizer loading robust for Kaggle’s offline environment by preferring local model directories and falling back to a widely-available base model only if needed. Finally, we keep the same prediction-to-score mapping and ensure the submission is always written as `submission.csv` with exactly `id,score` and correct row alignment.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import warnings

warnings.filterwarnings("ignore")

try:
    from google.protobuf import internal as _pb_internal

    if hasattr(_pb_internal, "_api_implementation"):
        _pb_internal._api_implementation._default_implementation_type = "python"
        _pb_internal._api_implementation._implementation_type = "python"
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import Trainer, AutoModelForSequenceClassification, AutoTokenizer

print("torch:", torch.__version__)
print("transformers imported OK")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def load_model_and_tokenizer():
    candidate_paths = [
        "../input/patent-phrase-matching/patent_phrase/checkpoint-2052",
        "../input/patent-phrase-matching/patent_phrase",
        "../input/us-patent-phrase-to-phrase-matching",
        "../input",
        "../kaggle/input/patent-phrase-matching/patent_phrase/checkpoint-2052",
        "../kaggle/input/patent-phrase-matching/patent_phrase",
        "../kaggle/input/us-patent-phrase-to-phrase-matching",
        "../kaggle/input",
    ]

    def looks_like_model_dir(p: str) -> bool:
        return os.path.isdir(p) and (
            os.path.isfile(os.path.join(p, "config.json"))
            or os.path.isfile(os.path.join(p, "pytorch_model.bin"))
            or os.path.isfile(os.path.join(p, "model.safetensors"))
        )

    for p in candidate_paths:
        if looks_like_model_dir(p):
            tok = AutoTokenizer.from_pretrained(p, local_files_only=True)
            mdl = AutoModelForSequenceClassification.from_pretrained(
                p, num_labels=5, local_files_only=True
            )
            return mdl, tok

    fallback_model_name = "microsoft/deberta-v3-small"
    tok = AutoTokenizer.from_pretrained(fallback_model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        fallback_model_name, num_labels=5
    )
    return mdl, tok


model, tokenizer = load_model_and_tokenizer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("device:", device)




## === cell 2
class MyDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        self._n = len(next(iter(encodings.values())))

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        return item




## === cell 3
def build_encodings(df, test=False, max_length=128):
    text_a = (
        df["context"].astype(str).str[0] + " " + df["anchor"].astype(str)
    ).tolist()
    text_b = df["target"].astype(str).tolist()

    enc = tokenizer(
        text_a,
        text_b,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="np",
    )

    if not test:
        labels = (
            np.digitize(df["score"].to_numpy(), bins=np.linspace(0, 1, 5)) - 1
        ).astype(np.int64)
        enc["labels"] = labels
    return enc


test_path = "../input/us-patent-phrase-to-phrase-matching/test.csv"
if not os.path.exists(test_path):
    test_path = "../input/test.csv"

if not os.path.exists(test_path):
    test_path = "../kaggle/input/us-patent-phrase-to-phrase-matching/test.csv"
if not os.path.exists(test_path):
    test_path = "../kaggle/input/test.csv"

test_df = pd.read_csv(test_path)
test_encodings = build_encodings(test_df, test=True, max_length=128)
testset = MyDataset(test_encodings)

print("test rows:", len(test_df), "dataset len:", len(testset))




## === cell 4
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
)


def softmax_stable(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


with torch.no_grad():
    outputs = trainer.predict(testset)

logits = outputs.predictions
if logits.ndim == 3:
    logits = np.squeeze(logits, axis=1)

prob = softmax_stable(logits, axis=1)

score_values = np.linspace(0, 1, 5)
pred = (prob * score_values).sum(axis=1)

pred = np.asarray(pred).reshape(-1)
assert len(pred) == len(
    test_df
), f"Prediction length {len(pred)} != test length {len(test_df)}"

submit = pd.DataFrame({"id": test_df["id"].values, "score": pred})
submit.to_csv("submission.csv", index=False)

print(submit.head())
print("Wrote submission.csv with shape:", submit.shape)
