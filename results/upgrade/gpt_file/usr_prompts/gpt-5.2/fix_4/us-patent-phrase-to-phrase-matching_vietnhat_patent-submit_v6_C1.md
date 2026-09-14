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

0.5527

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.45531) has done: 'I fix the root cause preventing execution: the notebook forces offline Hugging Face loading (`local_files_only=True` and `TRANSFORMERS_OFFLINE=1`) but the DeBERTa model isn’t available in the local cache, so tokenization/model creation fails and cascades into later NameErrors. To keep the core inference logic intact while ensuring the code runs end-to-end in Kaggle without internet, I add a safe offline fallback using `sentence-transformers` (which ships with cached models on Kaggle more often) to generate similarity scores, and I preserve your original 5-level scoring semantics by snapping predictions to the allowed {0,0.25,0.5,0.75,1.0} levels. I also make paths robust (try `/kaggle/input/...` then `/kaggle/data/...`) and ensure `submission.csv` is always written with the required columns and row alignment.'
- What this solution (achieved 0.5527) has done: 'I fix the runtime error coming from `sentence-transformers` by pinning a compatible `protobuf` implementation at runtime (via the pure-Python backend) before importing it, which avoids the `MessageFactory.GetPrototype` crash in Kaggle’s environment. I also make the fallback model choice deterministic and improve score toward your target by switching from hard “snap-to-levels” outputs (which tends to hurt Pearson correlation) to a calibrated continuous prediction mapped from cosine similarity into [0,1] and then lightly shrunk toward the dataset mean (calibration only; no architecture/training changes). The HF/DeBERTa offline path remains unchanged and still be used if it’s actually available in the local cache. Finally, I keep the submission-writing logic but ensure `pred` is always a float array aligned to `test_df` and the CSV is written as `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 1
MODEL_NAME = "microsoft/deberta-v3-base"

tokenizer = None
model = None
use_fallback_st = False

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=5,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
except Exception as e:
    print(
        "HF offline load failed; will use sentence-transformers fallback.\nReason:",
        repr(e),
    )
    use_fallback_st = True

use_fallback_st




## === cell 2
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]




## === cell 3
def _read_csv_robust(rel_path_under_competition: str) -> pd.DataFrame:
    candidates = [
        f"/kaggle/input/us-patent-phrase-to-phrase-matching/{rel_path_under_competition}",
        f"/kaggle/data/us-patent-phrase-to-phrase-matching/{rel_path_under_competition}",
        f"/kaggle/input/{rel_path_under_competition}",
        f"/kaggle/data/{rel_path_under_competition}",
    ]
    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(
        f"Could not find {rel_path_under_competition} in known Kaggle paths: {candidates}"
    )


train_df = _read_csv_robust("train.csv")
test_df = _read_csv_robust("test.csv")
sample_sub = _read_csv_robust("sample_submission.csv")

train_df.shape, test_df.shape, sample_sub.shape




## === cell 4
def encode_row(row, test=False):
    text_a = row["context"][0] + " " + row["anchor"]
    text_b = row["target"]
    ret = tokenizer(text_a, text_b, truncation=True)
    if not test:
        ret["label"] = np.digitize(row["score"], bins=np.linspace(0, 1, 5)) - 1
    return ret


if not use_fallback_st:
    test_data = [encode_row(row, test=True) for _, row in test_df.iterrows()]
    testset = MyDataset(test_data)
    print("HF testset ready:", len(testset))
else:
    testset = None
    print("Skipping HF tokenization because fallback is enabled.")




## === cell 5
def collate_fn(batch):
    out = {}
    keys = batch[0].keys()
    for k in keys:
        if k == "label":
            continue
        out[k] = torch.tensor([b[k] for b in batch], dtype=torch.long)
    return out


logits = None

if not use_fallback_st:
    loader = DataLoader(
        testset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    all_logits = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            all_logits.append(out.logits.detach().cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    print("Logits shape:", logits.shape)
else:
    print("Skipping HF forward pass because fallback is enabled.")



## === cell 6
pred = None

levels = np.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1.0]

if logits is not None:
    x = logits - np.max(logits, axis=1, keepdims=True)
    prob = np.exp(x)
    prob = prob / np.sum(prob, axis=1, keepdims=True)
    pred = np.sum(prob * levels, axis=1).astype(np.float32)
else:
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

    from sentence_transformers import SentenceTransformer

    st_candidates = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
    ]
    st_model = None
    st_errs = []
    for name in st_candidates:
        try:
            st_model = SentenceTransformer(name, device=str(device))
            break
        except Exception as e:
            st_errs.append((name, repr(e)))

    if st_model is None:
        raise RuntimeError(
            f"Could not load any sentence-transformers model. Errors: {st_errs}"
        )

    a_texts = (
        test_df["context"].astype(str).str[0] + " " + test_df["anchor"].astype(str)
    ).tolist()
    b_texts = test_df["target"].astype(str).tolist()

    with torch.no_grad():
        emb_a = st_model.encode(
            a_texts, batch_size=64, convert_to_tensor=True, normalize_embeddings=True
        )
        emb_b = st_model.encode(
            b_texts, batch_size=64, convert_to_tensor=True, normalize_embeddings=True
        )
        cos = (emb_a * emb_b).sum(dim=1).detach().cpu().numpy()  # approx [-1,1]

    raw = np.clip((cos + 1.0) / 2.0, 0.0, 1.0)

    train_mean = float(train_df["score"].mean())
    alpha = 0.90
    pred = (alpha * raw + (1.0 - alpha) * train_mean).astype(np.float32)
    pred = np.clip(pred, 0.0, 1.0)

pred[:5], pred.shape



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 7
submit = pd.DataFrame({"id": test_df["id"].values, "score": pred.astype(float)})

if "id" in sample_sub.columns and len(sample_sub) == len(submit):
    submit = sample_sub[["id"]].merge(submit, on="id", how="left")
    if submit["score"].isna().any():
        raise ValueError(
            "Some test ids missing scores after merge; check id alignment."
        )

submit.to_csv("submission.csv", index=False)
submit.head(), submit.shape
