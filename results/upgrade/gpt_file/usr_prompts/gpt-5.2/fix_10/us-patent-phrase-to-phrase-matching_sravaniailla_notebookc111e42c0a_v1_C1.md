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

3.13

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

0.8275192406827523

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.2128) has done: 'I fix the environment/runtime break caused by the protobuf/transformers incompatibility by forcing the pure-Python protobuf implementation before importing `transformers`. Then I fix model/tokenizer loading by switching to an available local pretrained checkpoint (or a safe default) under the Kaggle input directory, avoiding the invalid path that triggers `HFValidationError`. Finally, I make tokenization return proper padded/truncated tensors for `Trainer.predict`, and keep the same prediction post-processing to produce a valid `submission.csv` with the required `id,score` columns.'
- What this solution (achieved 0.11523) has done: 'I fix the protobuf-related import crash by switching to the correct/available pure-Python protobuf setting (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` plus a safe minimum `protobuf` version guard) and by avoiding importing `transformers` before the environment is set. Then I correct the label binning logic (your current `np.digitize` mapping is off-by-one and can produce wrong classes), which should significantly improve correlation while keeping the same 5-class formulation and expected-value post-processing. Finally, I make the dataset return PyTorch tensors (not Python lists) to ensure `Trainer.predict` batches/pads consistently and the script always writes a valid `submission.csv` with `id,score`.'
- What this solution (achieved -0.08142) has done: 'I remove the biggest avoidable overhead: per-sample tensor conversion inside `__getitem__`, by switching the dataset to use already-tokenized NumPy arrays and a fast collator that batch-converts to torch tensors (same values, less Python work). I also enable `torch.compile` for the model (when available) to speed up the training/eval forward pass without changing architecture or training semantics. Finally, I reduce Trainer-side overhead by disabling unnecessary columns pruning work and keeping dataloader settings efficient; tokenization and label logic remain identical.'
- What this solution (achieved -0.08142) has done: 'I fix the two runtime blockers: (1) the protobuf/transformers crash by forcing the pure-Python protobuf implementation *before* importing `transformers`, and (2) the `num_items_in_batch` incompatibility by disabling that Trainer feature via `TrainingArguments(average_tokens_across_devices=False)` (keeps the same training semantics). I keep your model, 5-class formulation, binning, and expected-value post-processing unchanged to avoid score surprises while getting you back to a sane positive correlation and producing a valid `submission.csv`. I also make the local-checkpoint finder more robust but still default to the same public model if none exists locally.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from transformers import (
    Trainer,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)

print("Torch:", torch.__version__)
try:
    import transformers  # noqa: F401

    print("Transformers:", transformers.__version__)
except Exception as e:
    print("Transformers import issue:", repr(e))
try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as pbv

    print("Protobuf:", pbv)
except Exception as e:
    print("Protobuf import issue:", repr(e))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def find_local_checkpoint():
    candidates = [
        "/kaggle/input/us-patent-phrase-to-phrase-matching/deberta-v3-small",
        "/kaggle/input/us-patent-phrase-to-phrase-matching/deberta-v3-base",
        "/kaggle/input/patent-phrase-matching/patent_phrase/checkpoint-2052",
        "/kaggle/input/patent-phrase-matching/patent_phrase",
        "/kaggle/input/patent-phrase-matching",
    ]
    for p in candidates:
        if os.path.isdir(p):
            if os.path.isfile(os.path.join(p, "config.json")):
                return p
            for ck in ("checkpoint-1", "checkpoint-100", "checkpoint-1000"):
                if os.path.isdir(os.path.join(p, ck)) and os.path.isfile(
                    os.path.join(p, ck, "config.json")
                ):
                    return os.path.join(p, ck)
    return "microsoft/deberta-v3-small"


model_name_or_path = find_local_checkpoint()
print("Using model:", model_name_or_path)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path,
    num_labels=5,
)

USE_TORCH_COMPILE = False
if USE_TORCH_COMPILE and hasattr(torch, "compile"):
    try:
        model = torch.compile(model)  # PyTorch 2.x optimization
        print("torch.compile enabled")
    except Exception as e:
        print("torch.compile unavailable, continuing without it:", repr(e))

device = "cuda" if torch.cuda.is_available() else "cpu"
_ = model.to(device)
print("Device:", device)




## === cell 2
class MyDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        self._keys = tuple(encodings.keys())
        self._length = int(encodings["input_ids"].shape[0])

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        return {k: self.encodings[k][idx] for k in self._keys}


def fast_collate(features):
    batch = {}
    keys = features[0].keys()
    for k in keys:
        v0 = features[0][k]
        if torch.is_tensor(v0):
            batch[k] = torch.stack([f[k] for f in features], dim=0)
        else:
            batch[k] = torch.as_tensor([f[k] for f in features])
    return batch




## === cell 3
def _df_fingerprint(df, cols):
    h = pd.util.hash_pandas_object(df[cols], index=False).to_numpy(dtype=np.uint64)
    return int(h.sum(dtype=np.uint64))


def encode_df(df, test=False, max_length=128, cache_dir="token_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    use_cols = ["context", "anchor", "target"] + ([] if test else ["score"])
    fp = _df_fingerprint(df, use_cols)
    cache_path = os.path.join(
        cache_dir, f"enc_fp{fp}_ml{max_length}_{'test' if test else 'train'}.npz"
    )

    if os.path.isfile(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        enc = {k: data[k] for k in data.files}
    else:
        ctx = df["context"].to_numpy(dtype=object)
        ctx0 = np.char.array(ctx.astype(str)).str[:1]
        anchor = df["anchor"].to_numpy(dtype=object).astype(str)
        text_a = (ctx0 + " " + anchor).tolist()

        text_b = df["target"].to_numpy(dtype=object).astype(str).tolist()

        enc = tokenizer(
            text_a,
            text_b,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="np",  # keep as numpy for compact caching
        )
        enc = dict(enc)

        if not test:
            labels = (
                (df["score"].astype(float) * 4)
                .round()
                .astype(int)
                .clip(0, 4)
                .to_numpy()
            )
            if labels.dtype != np.int64:
                labels = labels.astype(np.int64, copy=False)
            enc["labels"] = labels

        np.savez_compressed(cache_path, **enc)

    out = {}
    for k, v in enc.items():
        if k == "labels":
            out[k] = torch.from_numpy(v.astype(np.int64, copy=False))
        else:
            if v.dtype == np.int32:
                v = v.astype(np.int64, copy=False)
            out[k] = torch.from_numpy(v)
    return out




## === cell 4
train_df = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv")

perm = np.random.RandomState(SEED).permutation(len(train_df))
val_size = int(0.1 * len(train_df))
val_idx = perm[:val_size]
trn_idx = perm[val_size:]

trn_df = train_df.iloc[trn_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

max_length = 128
trn_enc = encode_df(trn_df, test=False, max_length=max_length)
val_enc = encode_df(val_df, test=False, max_length=max_length)

trainset = MyDataset(trn_enc)
valset = MyDataset(val_enc)

print("Train/val sizes:", len(trainset), len(valset))




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1431289932.py in <cell line: 0>()
     10 
     11 max_length = 128
---> 12 trn_enc = encode_df(trn_df, test=False, max_length=max_length)
     13 val_enc = encode_df(val_df, test=False, max_length=max_length)
     14 

/tmp/ipykernel_11/1494944207.py in encode_df(df, test, max_length, cache_dir)
     24         # Faster construction of text pairs than multiple pandas .astype/.str ops
     25         ctx = df["context"].to_numpy(dtype=object)
---> 26         ctx0 = np.char.array(ctx.astype(str)).str[:1]
     27         anchor = df["anchor"].to_numpy(dtype=object).astype(str)
     28         text_a = (ctx0 + " " + anchor).tolist()

AttributeError: 'chararray' object has no attribute 'str'

## === cell 5
num_workers = min(2, (os.cpu_count() or 2))

training_args = TrainingArguments(
    output_dir="outputs",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    eval_strategy="epoch",
    save_strategy="no",
    logging_strategy="steps",
    logging_steps=200,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.06,
    fp16=torch.cuda.is_available(),
    report_to=[],
    dataloader_num_workers=num_workers,
    dataloader_pin_memory=torch.cuda.is_available(),
    seed=SEED,
    remove_unused_columns=False,
)


class PatchedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


trainer = PatchedTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=trainset,
    eval_dataset=valset,
    data_collator=fast_collate,
)

trainer.train()




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4285418862.py in <cell line: 0>()
     38     args=training_args,
     39     tokenizer=tokenizer,
---> 40     train_dataset=trainset,
     41     eval_dataset=valset,
     42     data_collator=fast_collate,

NameError: name 'trainset' is not defined

## === cell 6
test_df = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv")
test_enc = encode_df(test_df, test=True, max_length=128)
testset = MyDataset(test_enc)

outputs = trainer.predict(testset)

logits = np.asarray(outputs.predictions)

logits = logits - logits.max(axis=1, keepdims=True)
exp_logits = np.exp(logits)
prob = exp_logits / exp_logits.sum(axis=1, keepdims=True)

score_values = np.linspace(0, 1, 5)
pred = (prob * score_values).sum(axis=1)

submit = pd.DataFrame({"id": test_df["id"].values, "score": pred})
submit.to_csv("submission.csv", index=False)

print(submit.head())
print("Wrote submission.csv with shape:", submit.shape)
print("submission.csv path:", os.path.abspath("submission.csv"))

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4235500240.py in <cell line: 0>()
      1 test_df = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv")
----> 2 test_enc = encode_df(test_df, test=True, max_length=128)
      3 testset = MyDataset(test_enc)
      4 
      5 outputs = trainer.predict(testset)

/tmp/ipykernel_11/1494944207.py in encode_df(df, test, max_length, cache_dir)
     24         # Faster construction of text pairs than multiple pandas .astype/.str ops
     25         ctx = df["context"].to_numpy(dtype=object)
---> 26         ctx0 = np.char.array(ctx.astype(str)).str[:1]
     27         anchor = df["anchor"].to_numpy(dtype=object).astype(str)
     28         text_a = (ctx0 + " " + anchor).tolist()

AttributeError: 'chararray' object has no attribute 'str'
