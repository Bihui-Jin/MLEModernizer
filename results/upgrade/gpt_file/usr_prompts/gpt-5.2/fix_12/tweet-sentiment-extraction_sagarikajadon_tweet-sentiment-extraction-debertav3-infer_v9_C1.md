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
Predict the word or phrase from tweets that exemplifies the labelled sentiment.

## Metric
Word-level Jaccard score.

## Submission Format
For each ID in the test set, you must predict the string that best supports the sentiment for the tweet in question. Note that the selected text _needs_ to be **quoted** and **complete** (include punctuation, etc. - the above code splits ONLY on whitespace) to work correctly. The file should contain a header and have the following format:
```
textID,selected_text
2,"very good"
5,"I don't care"
6,"bad"
8,"it was, yes"
etc.
```

## Dataset
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

- `textID` - unique ID for each piece of text
- `text` - the text of the tweet
- `sentiment` - the general sentiment of the tweet
- `selected_text` - [train only] the text that supports the tweet's sentiment

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
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
tqdm==4.67.1
transformers==4.53.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        input/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        working/
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
```

-> data/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> data/tweet-sentiment-extraction/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/tweet-sentiment-extraction/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/tweet-sentiment-extraction/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> input/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> (stopped after 10 files for performance)

# 5. Target score

0.5516448616981506

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.14602) has done: 'I fix the environment/runtime failures by (1) removing the protobuf-triggering import stack that breaks at startup, (2) ensuring `DataCollatorWithPadding` is imported, and (3) replacing the invalid private Kaggle input paths for tokenizer/config/weights with public Hugging Face model loading so the notebook can run without missing files. To preserve the core inference logic (start/end span logits over tokens), I keep the same `QAModel` architecture and decoding approach, but load a compatible backbone and run a single deterministic forward pass. Finally, I guarantee we always generate exactly 2749 predictions and write a valid `submission.csv` with the required columns and quoting handled by pandas.'
- What this solution (achieved 0.14602) has done: 'I fix the startup crash caused by an incompatible protobuf/transformers import stack by forcing the pure-Python protobuf implementation before importing `transformers`. Then I correct a key decoding logic bug: applying `Softmax(dim=1)` to a 1D logit vector is wrong (it normalizes across samples when batched and errors/behaves badly per-row); instead I use `dim=-1` so probabilities are computed across token positions. These two changes keep your model/inference approach the same (start/end span logits over tokens) but make it run end-to-end and materially improve span selection toward the target Jaccard score. Finally, I keep the submission writing as-is to ensure a valid `submission.csv` is always produced.'
- What this solution (achieved 0.16526) has done: 'I fix the tensor stacking crash by padding the variable-length logits/masks to a common sequence length before `vstack`, which preserves your core inference logic but makes batching robust. Then I ensure downstream cells run by producing `fin_output_start/end/mask` deterministically even when batches have different padded lengths. Finally, I keep your existing softmax-over-tokens and span decoding, and guarantee a correctly formatted `submission.csv` is written with exactly 2749 rows.'
- What this solution (achieved 0.06911) has done: 'Your current inference is using a randomly initialized DeBERTa QA head (because there is no fine-tuning), so the span predictions are essentially noise and the Jaccard score stays very low. To move the score substantially toward the target while preserving the same core span-logit decoding semantics, the minimal legitimate improvement is to switch the backbone to a sentiment-extraction model that already has the same start/end span head trained for this competition. I keep your dataset/offset mapping, start/end softmax over tokens, and argmax span decoding unchanged; the main change is loading `AutoModelForQuestionAnswering` weights and reading `start_logits/end_logits` directly. This should move the score much closer to the target band without changing the evaluation semantics or adding new training loops.'
- What this solution (achieved 0.20219) has done: 'I fix the immediate runtime blocker: the local path `/kaggle/input/tweet-sentiment-extraction` is a dataset directory (not a Transformers model repo) so `AutoTokenizer/AutoModelForQuestionAnswering.from_pretrained` fails; I switch to a public pretrained QA backbone that loads correctly in this environment. Then I restore end-to-end execution by ensuring `QADataset` is defined (it wasn’t due to the earlier crash) and by keeping dynamic padding + padding-to-global-max stacking so logits can be concatenated safely. Finally, I keep your core “start/end token logits → softmax over tokens → argmax span → offsets to text” decoding unchanged, and write a valid `submission.csv` with exactly 2749 rows and the required columns.'
- What this solution (achieved 0.26862) has done: 'Your current score is far below the target, so we should legitimately improve span selection while keeping the same QA start/end-logit decoding pipeline. The biggest issue is that the offset mappings returned by the tokenizer are for the concatenated (question, context) pair, but your decoding treats them as if they index only into the tweet text; this misaligns character spans and destroys Jaccard. I keep the exact same model/backbone and inference approach, but (1) request `sequence_ids` so we can mask to context tokens only, (2) convert offsets to be relative to the tweet text, and (3) use a standard constrained span decode (maximize start_prob*end_prob with a small max answer length) restricted to context tokens. These are minimal, metric-aligned fixes that typically move this baseline much closer to the target without changing architecture or adding training.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import subprocess


def _ensure_protobuf_compat():
    """
    Keep runtime stability in Kaggle-like images where protobuf>=5 can break some stacks.
    This is not a scoring change; it prevents import/runtime failures.
    """
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
        if major >= 5:
            raise RuntimeError(f"Incompatible protobuf version: {pb_ver}")
    except Exception:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf>=4.21.0,<5"]
            )
            import importlib

            importlib.invalidate_caches()
        except Exception:
            pass


_ensure_protobuf_compat()

import gc
import random
import math
import re
import string

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForQuestionAnswering,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 1
class CFG:
    DEBUG = False
    TRAIN = False  # inference-only
    N_FOLDS = 5
    TRAIN_FOLDS = [i for i in range(N_FOLDS)]
    SEED = 42
    TEST_BATCHSIZE = 100
    MAX_LENGTH = 128

    FINETUNED_QA_MODEL = "mrm8488/bert-tweet-sentiment-extraction"

    FC_DROPOUT = [0.1, 0.2, 0.3, 0.4, 0.5]




## === cell 2
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(seed=CFG.SEED)




## === cell 3
test_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
test_df.head()




## === cell 4
tokenizer = AutoTokenizer.from_pretrained(CFG.FINETUNED_QA_MODEL, use_fast=True)
CFG.TOKENIZER = tokenizer

collate_fn = DataCollatorWithPadding(
    CFG.TOKENIZER, padding="longest", return_tensors="pt"
)


class QADataset:
    """
    DataCollatorWithPadding can only pad/tokenize numeric fields.
    Return only tokenizer outputs + meta fields collated manually.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        orig_text = str(self.df.text.iloc[item])
        text = " ".join(orig_text.split())
        sentiment_str = self.df.sentiment.iloc[item]

        question = f"extract {sentiment_str}"

        enc = CFG.TOKENIZER(
            question,
            text,
            add_special_tokens=True,
            max_length=CFG.MAX_LENGTH,
            padding=False,  # let collator pad dynamically
            truncation=True,
            return_offsets_mapping=True,
            return_attention_mask=True,
        )

        seq_ids = enc.sequence_ids()

        sentiment = [1, 0, 0]
        if sentiment_str == "positive":
            sentiment = [0, 0, 1]
        if sentiment_str == "negative":
            sentiment = [0, 1, 0]

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "offset_mapping": enc["offset_mapping"],
            "sequence_ids": seq_ids,
            "orig_text": orig_text,
            "context_text": text,
            "orig_sentiment": sentiment_str,
            "sentiment": sentiment,  # unused, but kept
        }


def qa_collate(batch):
    token_features = [
        {"input_ids": x["input_ids"], "attention_mask": x["attention_mask"]}
        for x in batch
    ]
    padded = collate_fn(token_features)

    padded["offset_mapping"] = [x["offset_mapping"] for x in batch]
    padded["sequence_ids"] = [x["sequence_ids"] for x in batch]
    padded["orig_text"] = [x["orig_text"] for x in batch]
    padded["context_text"] = [x["context_text"] for x in batch]
    padded["orig_sentiment"] = [x["orig_sentiment"] for x in batch]
    padded["sentiment"] = torch.tensor(
        [x["sentiment"] for x in batch], dtype=torch.long
    )
    return padded




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
HTTPError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_http.py in hf_raise_for_status(response, endpoint_name)
    401     try:
--> 402         response.raise_for_status()
    403     except HTTPError as e:

/usr/local/lib/python3.11/dist-packages/requests/models.py in raise_for_status(self)
   1025         if http_error_msg:
-> 1026             raise HTTPError(http_error_msg, response=self)
   1027 

HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/mrm8488/bert-tweet-sentiment-extraction/resolve/main/tokenizer_config.json

The above exception was the direct cause of the following exception:

RepositoryNotFoundError                   Traceback (most recent call last)
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
   1654         # Unauthorized => likely a token issue => let's raise the actual error
-> 1655         raise head_call_error
   1656     else:

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _get_metadata_or_catch_error(repo_id, filename, repo_type, revision, endpoint, proxies, etag_timeout, headers, token, local_files_only, relative_filename, storage_folder)
   1542             try:
-> 1543                 metadata = get_hf_file_metadata(
   1544                     url=url, proxies=proxies, timeout=etag_timeout, headers=headers, token=token, endpoint=endpoint

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    113 
--> 114         return fn(*args, **kwargs)
    115 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in get_hf_file_metadata(url, token, proxies, timeout, library_name, library_version, user_agent, headers, endpoint)
   1459     # Retrieve metadata
-> 1460     r = _request_wrapper(
   1461         method="HEAD",

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _request_wrapper(method, url, follow_relative_redirects, **params)
    282     if follow_relative_redirects:
--> 283         response = _request_wrapper(
    284             method=method,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _request_wrapper(method, url, follow_relative_redirects, **params)
    306     response = http_backoff(method=method, url=url, **params)
--> 307     hf_raise_for_status(response)
    308     return response

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_http.py in hf_raise_for_status(response, endpoint_name)
    451             )
--> 452             raise _format(RepositoryNotFoundError, message, response) from e
    453 

RepositoryNotFoundError: 401 Client Error. (Request ID: Root=1-6a086264-721b90cf5568778f061cc727;fcf3a492-c6b2-49a2-83d2-f5bd2c630a77)

Repository Not Found for url: https://huggingface.co/mrm8488/bert-tweet-sentiment-extraction/resolve/main/tokenizer_config.json.
Please make sure you specified the correct `repo_id` and `repo_type`.
If you are trying to access a private or gated repo, make sure you are authenticated. For more details, see https://huggingface.co/docs/huggingface_hub/authentication
Invalid username or password.

The above exception was the direct cause of the following exception:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/3626056843.py in <cell line: 0>()
----> 1 tokenizer = AutoTokenizer.from_pretrained(CFG.FINETUNED_QA_MODEL, use_fast=True)
      2 CFG.TOKENIZER = tokenizer
      3 
      4 collate_fn = DataCollatorWithPadding(
      5     CFG.TOKENIZER, padding="longest", return_tensors="pt"

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/tokenization_auto.py in from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs)
    981 
    982         # Next, let's try to use the tokenizer_config file to get the tokenizer class.
--> 983         tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
    984         if "_commit_hash" in tokenizer_config:
    985             kwargs["_commit_hash"] = tokenizer_config["_commit_hash"]

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/tokenization_auto.py in get_tokenizer_config(pretrained_model_name_or_path, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, **kwargs)
    813 
    814     commit_hash = kwargs.get("_commit_hash", None)
--> 815     resolved_config_file = cached_file(
    816         pretrained_model_name_or_path,
    817         TOKENIZER_CONFIG_FILE,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    310     ```
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file
    314     return file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    500         # We cannot recover from them
    501         if isinstance(e, RepositoryNotFoundError) and not isinstance(e, GatedRepoError):
--> 502             raise OSError(
    503                 f"{path_or_repo_id} is not a local folder and is not a valid model identifier "
    504                 "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a token "

OSError: mrm8488/bert-tweet-sentiment-extraction is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo either by logging in with `huggingface-cli login` or by passing `token=<your_token>`

## === cell 5
class QAModel(nn.Module):
    def __init__(self, config_path=None, pretrained=False):
        super().__init__()
        self.qa = AutoModelForQuestionAnswering.from_pretrained(CFG.FINETUNED_QA_MODEL)

    def forward(self, input_ids, mask):
        out = self.qa(input_ids=input_ids, attention_mask=mask)
        return out.start_logits, out.end_logits




## === cell 6
def _pad_to_length(x: torch.Tensor, length: int, value: float = 0.0) -> torch.Tensor:
    """
    Different batches can have different sequence lengths due to dynamic padding.
    Pad each batch output to the global max length before stacking.
    """
    if x.size(1) == length:
        return x
    pad_len = length - x.size(1)
    if pad_len < 0:
        return x[:, :length]
    return torch.nn.functional.pad(x, (0, pad_len), value=value)


def test_fn(dataloader, model):
    model.eval()

    fin_output_start = []
    fin_output_end = []
    fin_mask = []
    fin_offsets = []
    fin_seq_ids = []
    fin_orig_text = []
    fin_context_text = []
    fin_orig_sentiment = []

    max_len = 0

    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader)):
            input_ids = data["input_ids"].to(device)
            mask = data["attention_mask"].to(device)

            start_logits, end_logits = model(input_ids, mask)

            max_len = max(
                max_len, start_logits.size(1), end_logits.size(1), mask.size(1)
            )

            fin_output_start.append(start_logits.cpu())
            fin_output_end.append(end_logits.cpu())
            fin_mask.append(mask.cpu())

            fin_offsets.extend(data["offset_mapping"])
            fin_seq_ids.extend(data["sequence_ids"])
            fin_orig_text.extend(data["orig_text"])
            fin_context_text.extend(data["context_text"])
            fin_orig_sentiment.extend(data["orig_sentiment"])

    fin_output_start = torch.vstack(
        [_pad_to_length(t, max_len, value=0.0) for t in fin_output_start]
    )
    fin_output_end = torch.vstack(
        [_pad_to_length(t, max_len, value=0.0) for t in fin_output_end]
    )
    fin_mask = torch.vstack([_pad_to_length(t, max_len, value=0.0) for t in fin_mask])

    return (
        fin_output_start,
        fin_output_end,
        fin_mask,
        fin_offsets,
        fin_seq_ids,
        fin_orig_text,
        fin_context_text,
        fin_orig_sentiment,
    )




## === cell 7
test_dataset = QADataset(test_df)
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.TEST_BATCHSIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    collate_fn=qa_collate,
)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/709919954.py in <cell line: 0>()
----> 1 test_dataset = QADataset(test_df)
      2 test_loader = DataLoader(
      3     test_dataset,
      4     batch_size=CFG.TEST_BATCHSIZE,
      5     shuffle=False,

NameError: name 'QADataset' is not defined

## === cell 8
model = QAModel(config_path=None, pretrained=True).to(device)

(
    fin_output_start,
    fin_output_end,
    fin_mask,
    fin_offsets,
    fin_seq_ids,
    fin_orig_text,
    fin_context_text,
    fin_orig_sentiment,
) = test_fn(test_loader, model)

(
    fin_output_start.shape,
    fin_output_end.shape,
    fin_mask.shape,
    len(fin_offsets),
    len(fin_orig_text),
)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
HTTPError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_http.py in hf_raise_for_status(response, endpoint_name)
    401     try:
--> 402         response.raise_for_status()
    403     except HTTPError as e:

/usr/local/lib/python3.11/dist-packages/requests/models.py in raise_for_status(self)
   1025         if http_error_msg:
-> 1026             raise HTTPError(http_error_msg, response=self)
   1027 

HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/mrm8488/bert-tweet-sentiment-extraction/resolve/main/config.json

The above exception was the direct cause of the following exception:

RepositoryNotFoundError                   Traceback (most recent call last)
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
   1654         # Unauthorized => likely a token issue => let's raise the actual error
-> 1655         raise head_call_error
   1656     else:

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _get_metadata_or_catch_error(repo_id, filename, repo_type, revision, endpoint, proxies, etag_timeout, headers, token, local_files_only, relative_filename, storage_folder)
   1542             try:
-> 1543                 metadata = get_hf_file_metadata(
   1544                     url=url, proxies=proxies, timeout=etag_timeout, headers=headers, token=token, endpoint=endpoint

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    113 
--> 114         return fn(*args, **kwargs)
    115 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in get_hf_file_metadata(url, token, proxies, timeout, library_name, library_version, user_agent, headers, endpoint)
   1459     # Retrieve metadata
-> 1460     r = _request_wrapper(
   1461         method="HEAD",

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _request_wrapper(method, url, follow_relative_redirects, **params)
    282     if follow_relative_redirects:
--> 283         response = _request_wrapper(
    284             method=method,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _request_wrapper(method, url, follow_relative_redirects, **params)
    306     response = http_backoff(method=method, url=url, **params)
--> 307     hf_raise_for_status(response)
    308     return response

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_http.py in hf_raise_for_status(response, endpoint_name)
    451             )
--> 452             raise _format(RepositoryNotFoundError, message, response) from e
    453 

RepositoryNotFoundError: 401 Client Error. (Request ID: Root=1-6a086264-65f377402cd2cd524d8bd1e9;8a4b6c44-5317-476d-8f0a-9d1c980b9c26)

Repository Not Found for url: https://huggingface.co/mrm8488/bert-tweet-sentiment-extraction/resolve/main/config.json.
Please make sure you specified the correct `repo_id` and `repo_type`.
If you are trying to access a private or gated repo, make sure you are authenticated. For more details, see https://huggingface.co/docs/huggingface_hub/authentication
Invalid username or password.

The above exception was the direct cause of the following exception:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/295054776.py in <cell line: 0>()
----> 1 model = QAModel(config_path=None, pretrained=True).to(device)
      2 
      3 (
      4     fin_output_start,
      5     fin_output_end,

/tmp/ipykernel_55/657272437.py in __init__(self, config_path, pretrained)
      3         super().__init__()
      4         # Score-moving change: load task-finetuned QA weights (same head outputs).
----> 5         self.qa = AutoModelForQuestionAnswering.from_pretrained(CFG.FINETUNED_QA_MODEL)
      6 
      7     def forward(self, input_ids, mask):

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/auto_factory.py in from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)
    506             if not isinstance(config, PretrainedConfig):
    507                 # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
--> 508                 resolved_config_file = cached_file(
    509                     pretrained_model_name_or_path,
    510                     CONFIG_NAME,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    310     ```
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file
    314     return file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    500         # We cannot recover from them
    501         if isinstance(e, RepositoryNotFoundError) and not isinstance(e, GatedRepoError):
--> 502             raise OSError(
    503                 f"{path_or_repo_id} is not a local folder and is not a valid model identifier "
    504                 "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a token "

OSError: mrm8488/bert-tweet-sentiment-extraction is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo either by logging in with `huggingface-cli login` or by passing `token=<your_token>`

## === cell 9
s = torch.nn.Softmax(dim=-1)
fin_output_start = s(fin_output_start)
fin_output_end = s(fin_output_end)
fin_mask = fin_mask.float()

final_outputs = []

MAX_ANSWER_TOKENS = 40

for j in range(len(fin_context_text)):
    context_text = str(fin_context_text[j])  # normalized (" ".join)
    orig_text = str(fin_orig_text[j])  # original, used only as fallback
    offsets = fin_offsets[j]
    seq_ids = fin_seq_ids[j]  # None (special), 0 (question), 1 (context)
    mask = fin_mask[j]
    sentiment_str = str(fin_orig_sentiment[j])

    if sentiment_str == "neutral":
        final_outputs.append(context_text if context_text.strip() else orig_text)
        continue

    n_tok = min(len(offsets), int(mask.numel()))
    start_p = fin_output_start[j, :n_tok]
    end_p = fin_output_end[j, :n_tok]

    valid = torch.zeros(n_tok, dtype=torch.float32)
    for i in range(n_tok):
        if mask[i].item() <= 0:
            continue
        if seq_ids[i] != 1:
            continue
        off = offsets[i]
        if off is None:
            continue
        if isinstance(off, (list, tuple)) and len(off) == 2 and off[1] > off[0]:
            valid[i] = 1.0

    if valid.sum().item() == 0:
        final_outputs.append(context_text if context_text.strip() else orig_text)
        continue

    start_p = start_p * valid
    end_p = end_p * valid

    best_score = -1.0
    best_i = int(torch.argmax(start_p).item())
    best_k = best_i

    valid_idx = torch.where(valid > 0)[0]
    if valid_idx.numel() == 0:
        final_outputs.append(context_text if context_text.strip() else orig_text)
        continue

    topk = min(30, int(valid_idx.numel()))
    start_top = valid_idx[torch.topk(start_p[valid_idx], k=topk).indices].tolist()

    for i in start_top:
        k_max = min(n_tok - 1, i + MAX_ANSWER_TOKENS)
        window_end = end_p[i : k_max + 1]
        if window_end.numel() == 0:
            continue
        rel_k = int(torch.argmax(window_end).item())
        k = i + rel_k
        score = float(start_p[i].item() * end_p[k].item())
        if score > best_score:
            best_score = score
            best_i, best_k = i, k

    start_char = offsets[best_i][0]
    end_char = offsets[best_k][1]

    if end_char <= start_char:
        pred = context_text if context_text.strip() else orig_text
    else:
        pred = context_text[start_char:end_char].strip()
        if pred == "":
            pred = context_text if context_text.strip() else orig_text

    final_outputs.append(pred)

len(final_outputs), final_outputs[0]




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/808255170.py in <cell line: 0>()
      1 s = torch.nn.Softmax(dim=-1)
----> 2 fin_output_start = s(fin_output_start)
      3 fin_output_end = s(fin_output_end)
      4 fin_mask = fin_mask.float()
      5 

NameError: name 'fin_output_start' is not defined

## === cell 10
assert len(final_outputs) == len(
    test_df
), f"Pred length {len(final_outputs)} != test length {len(test_df)}"

test_ids = test_df["textID"].values
sub = pd.DataFrame({"textID": test_ids, "selected_text": final_outputs})
sub.head()




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1881837104.py in <cell line: 0>()
----> 1 assert len(final_outputs) == len(
      2     test_df
      3 ), f"Pred length {len(final_outputs)} != test length {len(test_df)}"
      4 
      5 test_ids = test_df["textID"].values

NameError: name 'final_outputs' is not defined

## === cell 11
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.isna().sum())
print(sub.head(3).to_string(index=False))

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3116703084.py in <cell line: 0>()
----> 1 sub.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", sub.shape)
      3 print(sub.isna().sum())
      4 print(sub.head(3).to_string(index=False))

NameError: name 'sub' is not defined
