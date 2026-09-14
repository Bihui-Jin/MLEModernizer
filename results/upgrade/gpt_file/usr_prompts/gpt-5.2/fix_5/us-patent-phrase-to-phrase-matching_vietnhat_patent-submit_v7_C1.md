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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.01921) has done: 'I fix the import-time `protobuf` crash by forcing a compatible protobuf version behavior and importing `transformers` only after setting those environment variables. Then I fix the dataset length bug (your `__len__` returned number of keys instead of number of rows), which caused `Trainer.predict()` to run on only a few items and produced a length mismatch when creating the submission. Finally, I make the encoding use `return_tensors="np"` and implement a stable softmax so predictions are correctly shaped and the script always writes a valid `submission.csv` with `id,score`.'
- What this solution (achieved -0.21385) has done: 'We fix the protobuf/transformers import crash by explicitly forcing the pure-Python protobuf backend and setting `protobuf`’s internal implementation choice before importing `transformers` (this resolves the `MessageFactory.GetPrototype` AttributeError seen at import time). Then we keep your existing inference-only pipeline intact, but make the model/tokenizer loading robust for Kaggle’s offline environment by preferring local model directories and falling back to a widely-available base model only if needed. Finally, we keep the same prediction-to-score mapping and ensure the submission is always written as `submission.csv` with exactly `id,score` and correct row alignment.'

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
        if not os.path.isdir(p):
            return False
        return any(
            os.path.isfile(os.path.join(p, fn))
            for fn in ("config.json", "pytorch_model.bin", "model.safetensors")
        )

    for p in candidate_paths:
        if looks_like_model_dir(p):
            tok = AutoTokenizer.from_pretrained(p, local_files_only=True)
            mdl = AutoModelForSequenceClassification.from_pretrained(
                p, num_labels=5, local_files_only=True
            )
            print(f"Loaded local model from: {p}")
            return mdl, tok

    fallback_model_name = "microsoft/deberta-v3-small"
    try:
        tok = AutoTokenizer.from_pretrained(fallback_model_name, local_files_only=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(
            fallback_model_name, num_labels=5, local_files_only=True
        )
        print(f"Loaded fallback local model: {fallback_model_name}")
        return mdl, tok
    except Exception as e:
        raise RuntimeError(
            "Could not find any local model directory and fallback model is not available offline. "
            "Please attach a dataset containing the fine-tuned checkpoint."
        ) from e


model, tokenizer = load_model_and_tokenizer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("device:", device)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
LocalEntryNotFoundError                   Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    113 
--> 114         return fn(*args, **kwargs)
    115 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in hf_hub_download(repo_id, filename, subfolder, repo_type, revision, library_name, library_version, cache_dir, local_dir, user_agent, force_download, proxies, etag_timeout, token, local_files_only, headers, endpoint, resume_download, force_filename, local_dir_use_symlinks)
   1006     else:
-> 1007         return _hf_hub_download_to_cache_dir(
   1008             # Destination

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _hf_hub_download_to_cache_dir(cache_dir, repo_id, filename, repo_type, revision, endpoint, etag_timeout, headers, proxies, token, local_files_only, force_download)
   1113         # Otherwise, raise appropriate error
-> 1114         _raise_on_head_call_error(head_call_error, force_download, local_files_only)
   1115 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _raise_on_head_call_error(head_call_error, force_download, local_files_only)
   1645     if local_files_only:
-> 1646         raise LocalEntryNotFoundError(
   1647             "Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable"

LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable hf.co look-ups and downloads online, set 'local_files_only' to False.

The above exception was the direct cause of the following exception:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/537975834.py in load_model_and_tokenizer()
     34     try:
---> 35         tok = AutoTokenizer.from_pretrained(fallback_model_name, local_files_only=True)
     36         mdl = AutoModelForSequenceClassification.from_pretrained(

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/tokenization_auto.py in from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs)
   1002                 else:
-> 1003                     config = AutoConfig.from_pretrained(
   1004                         pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/configuration_auto.py in from_pretrained(cls, pretrained_model_name_or_path, **kwargs)
   1196 
-> 1197         config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
   1198         has_remote_code = "auto_map" in config_dict and "AutoConfig" in config_dict["auto_map"]

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    607         # Get config dict associated with the base config file
--> 608         config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
    609         if config_dict is None:

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in _get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    666                 # Load from local folder or from cache or download from model Hub and cache
--> 667                 resolved_config_file = cached_file(
    668                     pretrained_model_name_or_path,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    542             elif _raise_exceptions_for_missing_entries:
--> 543                 raise OSError(
    544                     f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load the files, and couldn't find them in the"

OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

The above exception was the direct cause of the following exception:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/537975834.py in <cell line: 0>()
     46 
     47 
---> 48 model, tokenizer = load_model_and_tokenizer()
     49 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     50 model.to(device)

/tmp/ipykernel_11/537975834.py in load_model_and_tokenizer()
     40         return mdl, tok
     41     except Exception as e:
---> 42         raise RuntimeError(
     43             "Could not find any local model directory and fallback model is not available offline. "
     44             "Please attach a dataset containing the fine-tuned checkpoint."

RuntimeError: Could not find any local model directory and fallback model is not available offline. Please attach a dataset containing the fine-tuned checkpoint.

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




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4103348484.py in <cell line: 0>()
     33 test_df = pd.read_csv(test_path)
     34 
---> 35 test_encodings = build_encodings(test_df, test=True, max_length=128)
     36 testset = MyDataset(test_encodings)
     37 

/tmp/ipykernel_11/4103348484.py in build_encodings(df, test, max_length)
      5     text_b = df["target"].astype(str).tolist()
      6 
----> 7     enc = tokenizer(
      8         text_a,
      9         text_b,

NameError: name 'tokenizer' is not defined

## === cell 5
def softmax_stable(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


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

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/199211153.py in <cell line: 0>()
      7 
      8 
----> 9 loader = DataLoader(testset, batch_size=64, shuffle=False)
     10 
     11 all_logits = []

NameError: name 'testset' is not defined
