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
Predict the score of student essays.

## Metric
Quadratic weighted kappa.

## Submission Format
For each `essay_id` in the test set, you must predict the corresponding `score` (between 1-6, see [rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf) for more details). The file should contain a header and have the following format:

```
essay_id,score
000d118,3
000fe60,3
001ab80,4
...
```

## Dataset
- **train.csv** - Essays and scores to be used as training data.
    - `essay_id` - The unique ID of the essay
    - `full_text` - The full essay response
    - `score` - Holistic score of the essay on a 1-6 scale
- **test.csv** - The essays to be used as test data. Contains the same fields as `train.csv`, aside from exclusion of `score`.
- **sample_submission.csv** - A submission file in the correct format.
    - `essay_id` - The unique ID of the essay
    - `score` - The predicted holistic score of the essay on a 1-6 scale

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
text-unidecode==1.3
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
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        input/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        working/
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
```

-> data/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/learning-agency-lab-automated-essay-scoring-2/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/learning-agency-lab-automated-essay-scoring-2/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> data/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> (stopped after 10 files for performance)

# 5. Target score

0.786623184549083

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.00556) has done: 'I fix the immediate crash by making model loading robust to local Kaggle dataset folder structures (including nested `snapshots/` layouts) and by forcing Transformers to treat the path as a local directory rather than a Hub repo id. Then I unblock the downstream cells by ensuring `df`, `tokenizer`, and `model` are always defined (with a safe fallback model if the provided local model isn’t present), so the script runs end-to-end and writes `submission.csv` with the required columns. I also add missing tokenizer settings needed for some decoder-only tokenizers (pad token) to prevent runtime padding errors during batching. These changes keep the core inference logic (argmax over logits, +1, clip 1–6) the same while ensuring a valid submission is produced.'

# 9. Code solution

## === cell 0
import os
import re
import codecs
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from text_unidecode import unidecode


class PredictionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]["full_text"]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end


codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)


def resolve_encodings_and_normalize(text: str) -> str:
    """Resolve encoding problems and normalize abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text


def preprocess_essay_text(text: str) -> str:
    """
    Prepares essay text for scoring by cleaning non-essential issues without altering quality indicators.
    - Resolves encoding issues
    - Normalizes whitespace
    - Preserves original spelling, grammar, and casing
    """
    text = resolve_encodings_and_normalize(text)
    text = re.sub(r"\s+", " ", text.strip())  # Normalize whitespace
    text = re.sub(r'\s+([?.!,"])', r"\1", text)  # Remove spaces before punctuation
    text = re.sub(r",([^\s])", r", \1", text)  # Add space after commas
    return text


def _find_hf_model_dir(base_path: str) -> str:
    """
    Find a directory containing a HF model config.json under base_path (supports nested layouts).
    Returns the first match found, else returns base_path unchanged.
    """
    if base_path and os.path.isdir(base_path):
        if os.path.isfile(os.path.join(base_path, "config.json")):
            return base_path
        for root, _, files in os.walk(base_path):
            if "config.json" in files:
                return root
    return base_path


def _load_local_model_and_tokenizer(model_base: str):
    model_path = _find_hf_model_dir(model_base)

    if not (
        model_path
        and os.path.isdir(model_path)
        and os.path.isfile(os.path.join(model_path, "config.json"))
    ):
        raise FileNotFoundError(
            f"Could not find a local HF model with config.json under: {model_base}"
        )

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, config=config, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, use_fast=True
    )
    return model, tokenizer, model_path


def _load_offline_pretrained_fallback():
    """
    Bug fix: the provided notebook expects a local model directory that is not present.
    Minimal fallback: use a standard HF model that is typically available in Kaggle's
    Transformers cache, with local_files_only=True to avoid internet.
    Core inference logic remains identical (argmax over logits, +1, clip).
    """
    candidates = [
        "distilbert-base-uncased-finetuned-sst-2-english",
        "distilbert-base-uncased",
        "bert-base-uncased",
        "roberta-base",
    ]
    last_err = None
    for name in candidates:
        try:
            config = AutoConfig.from_pretrained(name, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                name, config=config, local_files_only=True
            )
            tokenizer = AutoTokenizer.from_pretrained(
                name, local_files_only=True, use_fast=True
            )
            return model, tokenizer, f"hf-cache:{name}"
        except Exception as e:
            last_err = e
    raise RuntimeError(
        "Could not load any fallback pretrained model from local cache. "
        "This environment appears to have no cached HF models and no provided local model directory."
    ) from last_err


MODEL_BASE = "/kaggle/input/smollm2-360m-essay-scoring-model"

model = None
tokenizer = None
MODEL_PATH = None

try:
    model, tokenizer, MODEL_PATH = _load_local_model_and_tokenizer(MODEL_BASE)
except Exception as e:
    print("WARNING: Failed to load local model from:", MODEL_BASE)
    print("Reason:", repr(e))

    local_fallback_candidates = [
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2",
        "/kaggle/input",
    ]

    last_err = None
    for cand in local_fallback_candidates:
        try:
            model, tokenizer, MODEL_PATH = _load_local_model_and_tokenizer(cand)
            print("Falling back to local model found under:", cand)
            break
        except Exception as e2:
            last_err = e2

    if model is None or tokenizer is None:
        print(
            "WARNING: No local HF model found under /kaggle/input; using cached pretrained fallback."
        )
        model, tokenizer, MODEL_PATH = _load_offline_pretrained_fallback()

if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

if (
    getattr(model.config, "pad_token_id", None) is None
    and tokenizer.pad_token_id is not None
):
    model.config.pad_token_id = tokenizer.pad_token_id

TEST_CANDIDATES = [
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv",
    "/kaggle/input/test.csv",
]
test_path = None
for p in TEST_CANDIDATES:
    if os.path.exists(p):
        test_path = p
        break
if test_path is None:
    raise FileNotFoundError(f"Could not find test.csv in candidates: {TEST_CANDIDATES}")

df = pd.read_csv(test_path)
df["full_text"] = df["full_text"].astype(str).apply(preprocess_essay_text)

print("Loaded model from:", MODEL_PATH)
print("Loaded test from:", test_path)
print("Test rows:", len(df))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3928346465.py in <cell line: 0>()
    155 try:
--> 156     model, tokenizer, MODEL_PATH = _load_local_model_and_tokenizer(MODEL_BASE)
    157 except Exception as e:

/tmp/ipykernel_55/3928346465.py in _load_local_model_and_tokenizer(model_base)
     99     ):
--> 100         raise FileNotFoundError(
    101             f"Could not find a local HF model with config.json under: {model_base}"

FileNotFoundError: Could not find a local HF model with config.json under: /kaggle/input/smollm2-360m-essay-scoring-model

During handling of the above exception, another exception occurred:

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
/tmp/ipykernel_55/3928346465.py in _load_offline_pretrained_fallback()
    131         try:
--> 132             config = AutoConfig.from_pretrained(name, local_files_only=True)
    133             model = AutoModelForSequenceClassification.from_pretrained(

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
/tmp/ipykernel_55/3928346465.py in <cell line: 0>()
    179             "WARNING: No local HF model found under /kaggle/input; using cached pretrained fallback."
    180         )
--> 181         model, tokenizer, MODEL_PATH = _load_offline_pretrained_fallback()
    182 
    183 # Ensure padding works for all tokenizers (decoder-only or missing pad token)

/tmp/ipykernel_55/3928346465.py in _load_offline_pretrained_fallback()
    140         except Exception as e:
    141             last_err = e
--> 142     raise RuntimeError(
    143         "Could not load any fallback pretrained model from local cache. "
    144         "This environment appears to have no cached HF models and no provided local model directory."

RuntimeError: Could not load any fallback pretrained model from local cache. This environment appears to have no cached HF models and no provided local model directory.

## === cell 1
test_dataset = PredictionDataset(df, tokenizer)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4016130188.py in <cell line: 0>()
----> 1 test_dataset = PredictionDataset(df, tokenizer)
      2 

NameError: name 'df' is not defined

## === cell 2
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

predicted_labels = []
with torch.no_grad():
    for batch in test_dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        predicted_labels.extend(predictions)

predicted_labels = (np.array(predicted_labels) + 1).astype(int)
predicted_labels = np.clip(predicted_labels, 1, 6)

df["score"] = predicted_labels
df[["essay_id", "score"]].to_csv("submission.csv", index=False)

print("Predictions saved to 'submission.csv'")
print(df[["essay_id", "score"]].head())

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/58422017.py in <cell line: 0>()
----> 1 test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
      2 
      3 model.eval()
      4 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      5 model.to(device)

NameError: name 'test_dataset' is not defined
