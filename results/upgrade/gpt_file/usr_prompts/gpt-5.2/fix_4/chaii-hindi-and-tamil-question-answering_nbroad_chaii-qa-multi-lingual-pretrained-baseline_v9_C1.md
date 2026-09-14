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
Predicting the answers to questions in Hindi and Tamil.

## Metric
Word-level Jaccard score.

A Python implementation is provided below.

```
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
```

The formula for the overall metric is:
\text{score} = \frac{1}{n} \sum_{i=1}^n \text{jaccard}(gt_i, dt_i)

where:
$n$ = number of documents

$\text{jaccard}$ = the function provided above

$gt_i$ = the ith ground truth

$dt_i$ = the ith prediction

## Submission Format
For each ID in the test set, you must predict the string that best answers the provided question based on the context. Note that the selected text needs to be quoted and complete to work correctly. Include punctuation, etc. The file should contain a header and have the following format:

```
id,PredictionString
8c8ee6504,"1"
3163c22d0,"2 string"
66aae423b,"4 word 6"
722085a7b,"1"
etc.
```

## Dataset 
**All files should be encoded as UTF-8.**

- **train.csv** - the training set, containing context, questions, and answers. Also includes the start character of the answer for disambiguation.
- **test.csv** - the test set, containing context and questions.
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `id` - a unique identifier
- `context` - the text of the Hindi/Tamil sample from which answers should be derived
- `question` - the question, in Hindi/Tamil
- `answer_text` (train only) - the answer to the question (manual annotation) (note: for test, this is what you are attempting to predict)
- `answer_start` (train only) - the starting character in `context` for the answer (determined using substring match during data preparation)
- `language` - whether the text in question is in Tamil or Hindi

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
transformers==4.53.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (138 lines)
            sample_submission.csv (113 lines)
            sample_submission.csv.zip (950 Bytes)
            test.csv (7173 lines)
            test.csv.zip (648.4 kB)
            train.csv (67723 lines)
            train.csv.zip (5.9 MB)
            chaii-hindi-and-tamil-question-answering/
                description.md (138 lines)
                sample_submission.csv (113 lines)
                ... and 5 other files
                chaii-hindi-and-tamil-question-answering/
        input/
            description.md (138 lines)
            sample_submission.csv (113 lines)
            sample_submission.csv.zip (950 Bytes)
            test.csv (7173 lines)
            test.csv.zip (648.4 kB)
            train.csv (67723 lines)
            train.csv.zip (5.9 MB)
            chaii-hindi-and-tamil-question-answering/
                description.md (138 lines)
                sample_submission.csv (113 lines)
                ... and 5 other files
                chaii-hindi-and-tamil-question-answering/
        working/
            chaii-hindi-and-tamil-question-answering/
                description.md (138 lines)
                sample_submission.csv (113 lines)
                ... and 5 other files
                chaii-hindi-and-tamil-question-answering/
```

-> data/chaii-hindi-and-tamil-question-answering/sample_submission.csv has 112 rows and 2 columns.
The columns are: id, PredictionString

-> data/chaii-hindi-and-tamil-question-answering/test.csv has 7172 rows and 4 columns.
The columns are: id, context, question, language

-> data/chaii-hindi-and-tamil-question-answering/train.csv has 67722 rows and 6 columns.
The columns are: id, context, question, answer_text, answer_start, language

-> data/sample_submission.csv has 112 rows and 2 columns.
The columns are: id, PredictionString

-> data/test.csv has 7172 rows and 4 columns.
The columns are: id, context, question, language

-> data/train.csv has 67722 rows and 6 columns.
The columns are: id, context, question, answer_text, answer_start, language

-> input/chaii-hindi-and-tamil-question-answering/sample_submission.csv has 112 rows and 2 columns.
The columns are: id, PredictionString

-> (stopped after 10 files for performance)

# 5. Target score

0.3464473485946655

# 6. Current score

0.53065

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.53065) has done: 'I fix the crash by resolving the Hugging Face path-vs-repo-id issue: your `from_pretrained` call is treating a non-existent local directory as a Hub repo ID. The minimal robust fix is to search for an actually-existing local model directory and, if none is present, fall back to a public model ID while forcing `local_files_only=True` first and then allowing online download only if needed. This unblock inference so `predictions` is created, and then the submission-writing cell run and produce a valid `submission.csv` with the required columns. I also add a tiny safeguard to ensure predictions length matches the test set and no `None` values end up in the CSV.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import pandas as pd

TEST_PATH_CANDIDATES = [
    "../input/chaii-hindi-and-tamil-question-answering/test.csv",
    "/kaggle/input/chaii-hindi-and-tamil-question-answering/test.csv",
    "../kaggle/input/chaii-hindi-and-tamil-question-answering/test.csv",
    "../kaggle/data/chaii-hindi-and-tamil-question-answering/test.csv",
]
test_path = next((p for p in TEST_PATH_CANDIDATES if os.path.exists(p)), None)
if test_path is None:
    raise FileNotFoundError(
        f"Could not find test.csv in any known locations: {TEST_PATH_CANDIDATES}"
    )

test_df = pd.read_csv(test_path)
test_df.head()



## === cell 1
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    device = torch.device("cpu")

MODEL_CANDIDATES = [
    "../input/pretrained-xlm-models-for-squad/seongju/squadv2-xlm-roberta-base",
    "/kaggle/input/pretrained-xlm-models-for-squad/seongju/squadv2-xlm-roberta-base",
    "../kaggle/input/pretrained-xlm-models-for-squad/seongju/squadv2-xlm-roberta-base",
    "../kaggle/data/pretrained-xlm-models-for-squad/seongju/squadv2-xlm-roberta-base",
]

local_model_dir = next((p for p in MODEL_CANDIDATES if os.path.isdir(p)), None)
fallback_hf_model_id = "deepset/xlm-roberta-base-squad2"

model_name_or_path = (
    local_model_dir if local_model_dir is not None else fallback_hf_model_id
)


def load_tokenizer_and_model(name_or_path: str):
    try:
        tok = AutoTokenizer.from_pretrained(
            name_or_path, use_fast=True, local_files_only=True
        )
        mdl = AutoModelForQuestionAnswering.from_pretrained(
            name_or_path, local_files_only=True
        )
        return tok, mdl
    except Exception:
        tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
        mdl = AutoModelForQuestionAnswering.from_pretrained(name_or_path)
        return tok, mdl


tokenizer, model = load_tokenizer_and_model(model_name_or_path)
model.to(device)
model.eval()

max_length = 384
doc_stride = 128
max_answer_tokens = 30
batch_size = 8


def predict_batch(contexts, questions):
    enc = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=False,  # keep single window per example (minimal + fast)
        return_offsets_mapping=True,
        padding=True,
        return_tensors="pt",
    )

    offset_mapping = enc.pop("offset_mapping").cpu().numpy()
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
        start_logits = out.start_logits.detach().cpu().numpy()
        end_logits = out.end_logits.detach().cpu().numpy()

    answers = []
    for i in range(len(contexts)):
        offsets = offset_mapping[i]
        s_logits = start_logits[i]
        e_logits = end_logits[i]

        valid = offsets[:, 1] > offsets[:, 0]

        s_logits_masked = np.where(valid, s_logits, -1e9)
        e_logits_masked = np.where(valid, e_logits, -1e9)

        start_indexes = np.argsort(s_logits_masked)[-20:][::-1]
        end_indexes = np.argsort(e_logits_masked)[-20:][::-1]

        best_score = -1e18
        best_span = (0, 0)
        for s_idx in start_indexes:
            for e_idx in end_indexes:
                if e_idx < s_idx:
                    continue
                if (e_idx - s_idx + 1) > max_answer_tokens:
                    continue
                score = s_logits_masked[s_idx] + e_logits_masked[e_idx]
                if score > best_score:
                    best_score = score
                    best_span = (s_idx, e_idx)

        s_idx, e_idx = best_span
        start_char, _ = offsets[s_idx]
        _, end_char = offsets[e_idx]
        if end_char <= start_char:
            answers.append("")
        else:
            answers.append(contexts[i][int(start_char) : int(end_char)].strip())

    return answers


predictions = []
contexts = test_df["context"].astype(str).tolist()
questions = test_df["question"].astype(str).tolist()

for i in range(0, len(test_df), batch_size):
    batch_ctx = contexts[i : i + batch_size]
    batch_q = questions[i : i + batch_size]
    predictions.extend(predict_batch(batch_ctx, batch_q))

if len(predictions) != len(test_df):
    raise RuntimeError(
        f"Predictions length {len(predictions)} != test length {len(test_df)}"
    )

len(predictions), predictions[:3]



## --- ERROR in cell 1, traceback:
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
/tmp/ipykernel_55/3827851222.py in load_tokenizer_and_model(name_or_path)
     32     try:
---> 33         tok = AutoTokenizer.from_pretrained(
     34             name_or_path, use_fast=True, local_files_only=True

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

During handling of the above exception, another exception occurred:

AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
submission_df = pd.DataFrame(
    {
        "id": test_df["id"].astype(str),
        "PredictionString": pd.Series(predictions, dtype="string").fillna(""),
    }
)
submission_df.to_csv("submission.csv", index=False)

submission_df.head()
