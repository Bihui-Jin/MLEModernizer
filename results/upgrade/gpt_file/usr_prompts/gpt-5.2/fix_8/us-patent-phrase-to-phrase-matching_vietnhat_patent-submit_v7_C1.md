# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

nan

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.01921) has done: 'I fix the import-time `protobuf` crash by forcing a compatible protobuf version behavior and importing `transformers` only after setting those environment variables. Then I fix the dataset length bug (your `__len__` returned number of keys instead of number of rows), which caused `Trainer.predict()` to run on only a few items and produced a length mismatch when creating the submission. Finally, I make the encoding use `return_tensors="np"` and implement a stable softmax so predictions are correctly shaped and the script always writes a valid `submission.csv` with `id,score`.'
- What this solution (achieved -0.21385) has done: 'We fix the protobuf/transformers import crash by explicitly forcing the pure-Python protobuf backend and setting `protobuf`’s internal implementation choice before importing `transformers` (this resolves the `MessageFactory.GetPrototype` AttributeError seen at import time). Then we keep your existing inference-only pipeline intact, but make the model/tokenizer loading robust for Kaggle’s offline environment by preferring local model directories and falling back to a widely-available base model only if needed. Finally, we keep the same prediction-to-score mapping and ensure the submission is always written as `submission.csv` with exactly `id,score` and correct row alignment.'
- What this solution (achieved nan) has done: 'We fix the root cause of the crash by changing model/tokenizer loading to use a model that is guaranteed to exist in this offline Kaggle dataset (the competition’s provided baseline model files), instead of searching for a non-existent fine-tuned checkpoint. Then we ensure `model`, `tokenizer`, `testset`, and `device` are always defined so later cells don’t fail with `NameError`. Finally, we keep your exact inference core (softmax over 5 labels → expected value in [0,1]) and only add a safe fallback that still produces a valid `submission.csv` even if a model can’t be loaded for any reason.'
- What this solution (achieved nan) has done: 'Your current score is `nan` because the code is producing a valid CSV but is almost certainly using the heuristic fallback (`model is None`), which yields an effectively non-competitive constant prediction. The smallest change that legitimately improves toward your target is to reliably load an actually-available offline model from the Kaggle environment by searching common local Hugging Face cache locations and typical competition-provided model folders, without changing your inference core (softmax → expected value). I also ensure the model is loaded with its own config (so `num_labels` isn’t forced incorrectly), while preserving the same 5-label expected-score mapping you already use. If no model is found even after expanded search, the script still produce a valid `submission.csv`.'
- What this solution (achieved nan) has done: 'Your `nan` score is coming from the constant-prediction fallback (`model is None`), so the smallest legitimate improvement toward your target is to reliably load an actually-present offline model directory in this Kaggle environment. I keep your exact inference core (softmax over logits → expected value mapped into [0,1]) and only strengthen the local model discovery to include the standard Kaggle/Transformers cache locations (including `~/.cache/huggingface/transformers` and snapshot subfolders). I also make model-dir detection robust to snapshot layouts (where weights/config live one level deeper), without changing architecture or any training logic (you have none). If no model is still found, the code behave exactly as before and still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

print("torch:", torch.__version__)



## === cell 1
from transformers import AutoModelForSequenceClassification, AutoTokenizer

print("transformers imported OK")




## === cell 2
def load_model_and_tokenizer():
    candidate_paths = [
        "../input/us-patent-phrase-to-phrase-matching",
        "../kaggle/input/us-patent-phrase-to-phrase-matching",
        "../input/patent-phrase-matching/patent_phrase/checkpoint-2052",
        "../input/patent-phrase-matching/patent_phrase",
        "../input",
        "../kaggle/input/patent-phrase-matching/patent_phrase/checkpoint-2052",
        "../kaggle/input/patent-phrase-matching/patent_phrase",
        "../kaggle/input",
        "/kaggle/input",
        "/kaggle/working",
        os.path.expanduser("~/.cache/huggingface/hub"),
        "/root/.cache/huggingface/hub",
        os.path.expanduser("~/.cache/huggingface/transformers"),
        "/root/.cache/huggingface/transformers",
        os.path.expanduser("~/.cache/transformers"),
        "/root/.cache/transformers",
        "/kaggle/working/.cache/huggingface/hub",
        "/kaggle/working/.cache/huggingface/transformers",
    ]

    def is_file(p):  # small helper
        try:
            return os.path.isfile(p)
        except Exception:
            return False

    def is_dir(p):
        try:
            return os.path.isdir(p)
        except Exception:
            return False

    def looks_like_model_dir(p: str) -> bool:
        if not is_dir(p):
            return False
        return is_file(os.path.join(p, "config.json")) and (
            is_file(os.path.join(p, "pytorch_model.bin"))
            or is_file(os.path.join(p, "model.safetensors"))
            or is_file(os.path.join(p, "tf_model.h5"))
        )

    def iter_subdirs(base: str):
        if not is_dir(base):
            return
        try:
            for name in os.listdir(base):
                full = os.path.join(base, name)
                if is_dir(full):
                    yield full
        except Exception:
            return

    def iter_snapshot_like_dirs(base: str):
        """
        HF hub layout:
          models--org--name/snapshots/<hash>/{config.json, model.safetensors,...}
        Also some caches have files directly under a hash directory.
        """
        if not is_dir(base):
            return
        for d1 in iter_subdirs(base):
            yield d1
            snap = os.path.join(d1, "snapshots")
            if is_dir(snap):
                for d2 in iter_subdirs(snap):
                    yield d2
            for d2 in iter_subdirs(d1):
                yield d2

    expanded = []
    seen = set()

    def add_path(p):
        if p and p not in seen:
            expanded.append(p)
            seen.add(p)

    for p in candidate_paths:
        add_path(p)
        if is_dir(p):
            for sub in iter_subdirs(p):
                add_path(sub)
        if is_dir(p) and ("huggingface" in p or "transformers" in p):
            for sub in iter_snapshot_like_dirs(p):
                add_path(sub)

    for p in expanded:
        if looks_like_model_dir(p):
            try:
                tok = AutoTokenizer.from_pretrained(p, local_files_only=True)
                mdl = AutoModelForSequenceClassification.from_pretrained(
                    p, local_files_only=True
                )
                print(f"Loaded local model from: {p}")
                return mdl, tok
            except Exception as e:
                print(
                    f"Found model-like dir but failed to load: {p} | err: {type(e).__name__}: {e}"
                )

    print(
        "WARNING: No local HF model directory found; will fall back to simple heuristic predictions."
    )
    return None, None


model, tokenizer = load_model_and_tokenizer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if model is not None:
    model.to(device)
    model.eval()
print("device:", device, "| model_loaded:", model is not None)




## === cell 3
class MyDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        self._n = len(next(iter(encodings.values())))

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        return item




## === cell 4
def build_encodings(df, test=False, max_length=128):
    if tokenizer is None:
        raise RuntimeError(
            "Tokenizer is not available because no local model was found."
        )

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

testset = None
if tokenizer is not None:
    test_encodings = build_encodings(test_df, test=True, max_length=128)
    testset = MyDataset(test_encodings)
    print("test rows:", len(test_df), "dataset len:", len(testset))
else:
    print(
        "test rows:", len(test_df), "| dataset not built because tokenizer is missing"
    )




## === cell 5
def softmax_stable(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


if model is None or testset is None:
    pred = np.full(len(test_df), 0.5, dtype=np.float32)
else:
    loader = DataLoader(testset, batch_size=64, shuffle=False)

    all_logits = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device)
                for k, v in batch.items()
                if k in ("input_ids", "attention_mask", "token_type_ids")
            }
            out = model(**batch)
            logits = out.logits.detach().cpu().numpy()
            all_logits.append(logits)

    logits = np.concatenate(all_logits, axis=0)
    if logits.ndim == 3:
        logits = np.squeeze(logits, axis=1)

    prob = softmax_stable(logits, axis=1)

    if prob.shape[1] == 5:
        score_values = np.linspace(0, 1, 5)
        pred = (prob * score_values).sum(axis=1)
    else:
        score_values = np.linspace(0, 1, prob.shape[1])
        pred = (prob * score_values).sum(axis=1)

    pred = np.asarray(pred).reshape(-1)

assert len(pred) == len(
    test_df
), f"Prediction length {len(pred)} != test length {len(test_df)}"

submit = pd.DataFrame({"id": test_df["id"].values, "score": pred})
submit.to_csv("submission.csv", index=False)

print(submit.head())
print("Wrote submission.csv with shape:", submit.shape)
