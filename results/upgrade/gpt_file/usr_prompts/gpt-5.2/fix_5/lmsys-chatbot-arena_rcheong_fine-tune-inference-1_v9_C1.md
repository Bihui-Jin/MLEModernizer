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
Predict which chatbot response a user will prefer in a competition between two chatbots.

## Metric
Log loss with "eps=auto"

## Submission Format
For each id in the test set, you must predict the probability for each target class. The file should contain a header and have the following format:

```
 id,winner_model_a,winner_model_b,winner_tie
 136060,0.33,0,33,0.33
 211333,0.33,0,33,0.33
 1233961,0.33,0,33,0.33
 etc
```

## Dataset
**train.csv**

- `id` - A unique identifier for the row.
- `model_[a/b]` - The identity of model_[a/b]. Included in train.csv but not test.csv.
- `prompt` - The prompt that was given as an input (to both models).
- `response_[a/b]` - The response from model_[a/b] to the given prompt.
- `winner_model_[a/b/tie]` - Binary columns marking the judge's selection. The ground truth target column.

**test.csv**

- `id`
- `prompt`
- `response_[a/b]`

**sample_submission.csv** A submission file in the correct format.

- `id`
- `winner_model_[a/b/tie]` - This is what is predicted from the test set.

# 2. Python version

3.14

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
peft==0.16.0
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
tqdm==4.67.1
transformers==4.53.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (98 lines)
            sample_submission.csv (5749 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (5749 lines)
            test.csv.zip (5.9 MB)
            train.csv (51730 lines)
            train.csv.zip (53.4 MB)
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
        input/
            description.md (98 lines)
            sample_submission.csv (5749 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (5749 lines)
            test.csv.zip (5.9 MB)
            train.csv (51730 lines)
            train.csv.zip (53.4 MB)
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
        working/
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
```

-> data/lmsys-chatbot-arena/sample_submission.csv has 5748 rows and 4 columns.
The columns are: id, winner_model_a, winner_model_b, winner_tie

-> data/lmsys-chatbot-arena/test.csv has 5748 rows and 4 columns.
The columns are: id, prompt, response_a, response_b

-> data/lmsys-chatbot-arena/train.csv has 51729 rows and 9 columns.
The columns are: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie

-> data/sample_submission.csv has 5748 rows and 4 columns.
The columns are: id, winner_model_a, winner_model_b, winner_tie

-> data/test.csv has 5748 rows and 4 columns.
The columns are: id, prompt, response_a, response_b

-> data/train.csv has 51729 rows and 9 columns.
The columns are: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie

-> (stopped after 10 files for performance)

# 5. Target score

1.0372911976442458

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 3.11778) has done: 'I fix the crash caused by forcing the pure-Python protobuf implementation, which is incompatible with the installed protobuf stack in this environment and triggers the `MessageFactory.GetPrototype` error when importing/using Transformers. Then I ensure the submission is always valid by explicitly normalizing probabilities and clipping away exact zeros/ones (log-loss “eps=auto” can still reject numerically bad rows, and Kaggle also checks each row sums to 1). Finally, since your current run did not yield a score, I keep the model/inference logic the same and only add these stability fixes plus a safe fallback to match the sample submission column order.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print("Environment ready.")


## === cell 1
CANDIDATE_LOCAL_PATHS = [
    "/kaggle/input/mistral-checkpt-50-fp16-final/transformers/default/1",
    "/kaggle/input/mistral-checkpt-50-fp16-final",
]

FALLBACK_MODEL_ID = "distilbert-base-uncased-finetuned-mnli"


def pick_model_path():
    for p in CANDIDATE_LOCAL_PATHS:
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json")):
            return p
    return None


MODEL_PATH = pick_model_path()
if MODEL_PATH is None:
    print("WARNING: Custom checkpoint not found. Falling back to:", FALLBACK_MODEL_ID)
    MODEL_PATH = FALLBACK_MODEL_ID
else:
    print("Loading local model from:", MODEL_PATH)

local_only = os.path.isdir(MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, use_fast=True, local_files_only=local_only
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    )
tokenizer.padding_side = "right"

use_cuda = torch.cuda.is_available()
dtype = torch.float16 if use_cuda else torch.float32

quant_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    torch_dtype=dtype,
    device_map="auto" if use_cuda else None,
    local_files_only=local_only,
)
quant_model.eval()

num_labels = getattr(quant_model.config, "num_labels", None)
device = next(quant_model.parameters()).device
print("Loaded model. num_labels =", num_labels, "| device =", device)

model_max_len = getattr(quant_model.config, "max_position_embeddings", None)
if model_max_len is None:
    tmax = getattr(tokenizer, "model_max_length", 512)
    model_max_len = (
        int(tmax) if isinstance(tmax, (int, np.integer)) and tmax < 100000 else 512
    )

MAX_LENGTH = int(max(8, min(1536, model_max_len)))
print(
    "Using MAX_LENGTH =",
    MAX_LENGTH,
    "(model max_position_embeddings =",
    model_max_len,
    ")",
)


## --- ERROR in cell 1, traceback:
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

HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/distilbert-base-uncased-finetuned-mnli/resolve/main/tokenizer_config.json

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

RepositoryNotFoundError: 401 Client Error. (Request ID: Root=1-6a08b845-60c6fb8a65a2701a557bfec6;ad0f74c8-0a0b-4737-afcf-e509f0bf8c45)

Repository Not Found for url: https://huggingface.co/distilbert-base-uncased-finetuned-mnli/resolve/main/tokenizer_config.json.
Please make sure you specified the correct `repo_id` and `repo_type`.
If you are trying to access a private or gated repo, make sure you are authenticated. For more details, see https://huggingface.co/docs/huggingface_hub/authentication
Invalid username or password.

The above exception was the direct cause of the following exception:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/873675456.py in <cell line: 0>()
     25 local_only = os.path.isdir(MODEL_PATH)
     26 
---> 27 tokenizer = AutoTokenizer.from_pretrained(
     28     MODEL_PATH, use_fast=True, local_files_only=local_only
     29 )

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

OSError: distilbert-base-uncased-finetuned-mnli is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo either by logging in with `huggingface-cli login` or by passing `token=<your_token>`

## === cell 2
test_path = "/kaggle/input/lmsys-chatbot-arena/test.csv"
if not os.path.exists(test_path):
    test_path = "/kaggle/data/lmsys-chatbot-arena/test.csv"

test_df = pd.read_csv(test_path)

test_df["combined_text"] = (
    test_df["prompt"].fillna("").astype(str)
    + "\n\nResponse A:\n"
    + test_df["response_a"].fillna("").astype(str)
    + "\n\nResponse B:\n"
    + test_df["response_b"].fillna("").astype(str)
)

print("Test loaded:", test_df.shape)


## === cell 3
temperature = 1.8
batch_size = 16
preds = []

device = next(quant_model.parameters()).device

for i in tqdm(range(0, len(test_df), batch_size), desc="Predict"):
    batch = test_df.iloc[i : i + batch_size]["combined_text"].tolist()
    inputs = tokenizer(
        batch,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = quant_model(**inputs)
        logits = out.logits
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)

    preds.append(probs.detach().cpu().numpy())

preds = np.concatenate(preds, axis=0)

if preds.shape[1] == 3:
    p3 = preds
elif preds.shape[1] == 2:
    tie = np.full((preds.shape[0], 1), 1e-3, dtype=preds.dtype)
    p3 = np.concatenate([preds[:, :1], preds[:, 1:2], tie], axis=1)
else:
    if preds.shape[1] > 3:
        p3 = preds[:, :3]
    else:
        pad = np.full((preds.shape[0], 3 - preds.shape[1]), 1e-3, dtype=preds.dtype)
        p3 = np.concatenate([preds, pad], axis=1)

p3 = np.asarray(p3, dtype=np.float64)
p3 = np.clip(p3, 1e-8, 1.0)
p3 = p3 / p3.sum(axis=1, keepdims=True)

submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "winner_model_a": p3[:, 0],
        "winner_model_b": p3[:, 1],
        "winner_tie": p3[:, 2],
    }
)

sample_path = "/kaggle/input/lmsys-chatbot-arena/sample_submission.csv"
if not os.path.exists(sample_path):
    sample_path = "/kaggle/data/lmsys-chatbot-arena/sample_submission.csv"
if os.path.exists(sample_path):
    sample_cols = pd.read_csv(sample_path, nrows=1).columns.tolist()
    submission = submission[sample_cols]

submission.to_csv("submission.csv", index=False)
print("submission.csv ready:", submission.shape)
print(
    "Row-sum check:",
    float(
        np.max(
            np.abs(
                submission[["winner_model_a", "winner_model_b", "winner_tie"]]
                .sum(axis=1)
                .values
                - 1.0
            )
        )
    ),
)
print(submission.head())

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1924120431.py in <cell line: 0>()
      3 preds = []
      4 
----> 5 device = next(quant_model.parameters()).device
      6 
      7 for i in tqdm(range(0, len(test_df), batch_size), desc="Predict"):

NameError: name 'quant_model' is not defined
