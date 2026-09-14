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

0.7674398724521859

# 6. Current score

0.06756

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.21114) has done: 'I fix the runtime error by ensuring the model outputs a 1D prediction array; the current `.squeeze()` can still leave a 2D array (e.g., if logits have shape `[N, 2]` or `[N, 1]` in an unexpected way), which breaks `pd.DataFrame`. To keep core logic intact, I only add robust post-processing: if logits are multi-class I map them to a single score via softmax on the “positive” class; if single-logit regression, I keep it as-is. I also remove the unnecessary pip install (can cause instability/timeouts) but won’t change the model or batching logic. The script then reliably write a valid `submission.csv` with the correct columns and row count.'
- What this solution (achieved 0.08744) has done: 'The import-time `MessageFactory`/`GetPrototype` error is coming from an incompatible protobuf runtime used indirectly by `transformers` in this Kaggle image, so the first fix is to force the pure-Python protobuf implementation before importing `transformers` (this is a standard workaround and is score-neutral). Next, your very low score is consistent with using a 2-class classifier head and then taking softmax(class=1), which is usually the *wrong mapping* for this competition’s 0–1 regression target; the minimal fix (without changing the model/training) is to treat logits as a single regression signal: if 2-class, use the logit difference and pass through sigmoid to get a calibrated 0–1 score. Finally, keep the rest identical and ensure `submission.csv` is written with correct columns and row alignment.'
- What this solution (achieved -0.16788) has done: 'I fix the protobuf/transformers import crash by forcing a compatible protobuf runtime mode *before* any protobuf-related imports and by restarting the import order accordingly; this is score-neutral but unblocks execution. Then I correct the prediction post-processing to better match this competition’s 0–1 regression target by mapping model logits to a continuous score in a way that’s consistent with how SequenceClassification heads commonly represent regression vs. 2-class outputs (this is a minimal change and should raise Pearson substantially from the current 0.087). Finally, I keep the data loading, batching, and submission writing the same, ensuring `submission.csv` is produced with the correct columns and row alignment.'
- What this solution (achieved 0.06756) has done: 'I fix the `MessageFactory.GetPrototype` import crash by forcing a compatible protobuf stack *before* importing `transformers` (unsetting the C++/upb implementation and using the pure-Python one), which is score-neutral but unblocks execution. Then I minimally adjust the 2-logit post-processing: instead of softmax(class=1) (which is often poorly aligned for this regression-style 0–1 task), I convert `[N,2]` logits to a single continuous score via `sigmoid(logit1 - logit0)`, keeping the rest of the inference pipeline identical. I also add a tiny safety fallback so if protobuf is still broken, the script run using a SentenceTransformer embedding cosine baseline and still write a valid `submission.csv` (this fallback only triggers when transformers cannot import/load). The output remains a properly formatted `submission.csv` with `id,score` and the correct row count.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_CPP", None)

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader



## === cell 1
USE_TRANSFORMERS = True
_transformers_import_error = None
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
    )
except Exception as e:
    USE_TRANSFORMERS = False
    _transformers_import_error = repr(e)

USE_SENTENCE_TRANSFORMERS_FALLBACK = not USE_TRANSFORMERS

if USE_SENTENCE_TRANSFORMERS_FALLBACK:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim

print("USE_TRANSFORMERS:", USE_TRANSFORMERS)
if _transformers_import_error is not None:
    print("Transformers import error (will fallback):", _transformers_import_error)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

local_model_dir = "/kaggle/input/patentbert-simple/fold_0"
fallback_model_id = "microsoft/deberta-v3-small"



## === cell 3
tokenizer = None
model = None
st_model = None

if USE_TRANSFORMERS:
    if os.path.isdir(local_model_dir):
        model_name_or_path = local_model_dir
        local_only = True
    else:
        model_name_or_path = fallback_model_id
        local_only = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, local_files_only=local_only
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path, local_files_only=local_only
    ).to(device)
else:
    st_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device=str(device)
    )



## === cell 4
test_df = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv")
test_df.head()




## === cell 5
class TextDataset(Dataset):
    def __init__(self, df, tokenizer):
        texts = (
            "Category: "
            + df.context.astype(str)
            + " Text 1: "
            + df.anchor.astype(str)
            + " Text 2: "
            + df.target.astype(str)
        ).tolist()
        self.data = [tokenizer(t, truncation=True) for t in texts]

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)




## === cell 6
preds = None

if USE_TRANSFORMERS:
    dataset = TextDataset(test_df, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    test_dl = DataLoader(
        dataset, batch_size=32, shuffle=False, collate_fn=data_collator
    )

    model.eval()
    all_logits = []

    with torch.no_grad():
        for batch in test_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            all_logits.append(out.logits.detach().cpu())

    logits = torch.cat(all_logits, dim=0)  # [N], [N,1], or [N,2] typical

    if logits.ndim == 1:
        preds = logits.numpy()
    else:
        if logits.size(-1) == 1:
            preds = logits[:, 0].numpy()
        elif logits.size(-1) == 2:
            diff = logits[:, 1] - logits[:, 0]
            preds = torch.sigmoid(diff).numpy()
        else:
            preds = torch.sigmoid(logits.mean(dim=-1)).numpy()

    preds = np.asarray(preds, dtype=np.float32).reshape(-1)
    preds = np.clip(preds, 0.0, 1.0)

    assert preds.shape[0] == test_df.shape[0], (preds.shape, test_df.shape)
else:
    left_texts = (
        "Category: "
        + test_df["context"].astype(str)
        + " Text: "
        + test_df["anchor"].astype(str)
    ).tolist()
    right_texts = (
        "Category: "
        + test_df["context"].astype(str)
        + " Text: "
        + test_df["target"].astype(str)
    ).tolist()

    with torch.no_grad():
        emb_left = st_model.encode(
            left_texts,
            batch_size=64,
            convert_to_tensor=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        emb_right = st_model.encode(
            right_texts,
            batch_size=64,
            convert_to_tensor=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        sim = (emb_left * emb_right).sum(dim=1)  # cosine since normalized

    preds = sim.detach().cpu().numpy().astype(np.float32)
    preds = np.clip((preds + 1.0) / 2.0, 0.0, 1.0)

    assert preds.shape[0] == test_df.shape[0], (preds.shape, test_df.shape)



## === cell 7
submission = pd.DataFrame({"id": test_df["id"].values, "score": preds})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote:", os.path.abspath("submission.csv"), "rows:", len(submission))
assert submission.shape[0] == test_df.shape[0]
assert list(submission.columns) == ["id", "score"]
assert os.path.basename("submission.csv").endswith(".csv")
