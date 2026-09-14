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

0.007682021241635

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch

TEST_PATH = "/kaggle/input/chaii-hindi-and-tamil-question-answering/test.csv"
if not os.path.exists(TEST_PATH):
    TEST_PATH = "../input/chaii-hindi-and-tamil-question-answering/test.csv"

test_df = pd.read_csv(TEST_PATH)
test_df.head()



## === cell 1
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

PRIMARY_LOCAL_MODEL_DIR = (
    "/kaggle/input/pretrained-xlm-models-for-squad/mrm8488/xlm-multi-finetuned-xquadv1"
)
FALLBACK_LOCAL_MODEL_ID = (
    "deepset/xlm-roberta-large-squad2"  # only if already cached locally
)


def _load_qa_model():
    if os.path.isdir(PRIMARY_LOCAL_MODEL_DIR):
        tokenizer = AutoTokenizer.from_pretrained(
            PRIMARY_LOCAL_MODEL_DIR, use_fast=True, local_files_only=True
        )
        model = AutoModelForQuestionAnswering.from_pretrained(
            PRIMARY_LOCAL_MODEL_DIR, local_files_only=True
        )
        return tokenizer, model

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            FALLBACK_LOCAL_MODEL_ID, use_fast=True, local_files_only=True
        )
        model = AutoModelForQuestionAnswering.from_pretrained(
            FALLBACK_LOCAL_MODEL_ID, local_files_only=True
        )
        return tokenizer, model
    except Exception as e:
        raise FileNotFoundError(
            "Could not find the intended local model directory and no cached fallback model is available.\n"
            f"Missing: {PRIMARY_LOCAL_MODEL_DIR}\n"
            "Please add the pretrained model dataset to /kaggle/input, or ensure a QA model is available locally."
        ) from e


tokenizer, model = _load_qa_model()
model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

max_seq_length = 384
doc_stride = 128
max_answer_length = 30

predictions = []

with torch.no_grad():
    for ctx, q in test_df[["context", "question"]].to_numpy():
        enc = tokenizer(
            q,
            ctx,
            truncation="only_second",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=False,
            return_offsets_mapping=True,
            padding=False,
            return_tensors="pt",
        )

        offset_mapping = enc.pop("offset_mapping")[0].tolist()
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        token_type_ids = enc.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        start_logits = outputs.start_logits[0].detach().cpu().numpy()
        end_logits = outputs.end_logits[0].detach().cpu().numpy()

        sequence_ids = enc.sequence_ids(0)
        valid_token_idxs = [
            i
            for i, sid in enumerate(sequence_ids)
            if sid == 1
            and offset_mapping[i] is not None
            and offset_mapping[i][0] is not None
        ]

        if not valid_token_idxs:
            predictions.append("")
            continue

        best_score = -1e18
        best_span = (valid_token_idxs[0], valid_token_idxs[0])

        start_candidates = np.argsort(start_logits)[-50:][::-1]
        end_candidates = np.argsort(end_logits)[-50:][::-1]

        valid_set = set(valid_token_idxs)
        for s in start_candidates:
            if s not in valid_set:
                continue
            for e in end_candidates:
                if e not in valid_set:
                    continue
                if e < s:
                    continue
                if (e - s + 1) > max_answer_length:
                    continue

                score = float(start_logits[s] + end_logits[e])
                if score > best_score:
                    best_score = score
                    best_span = (s, e)

        s, e = best_span
        start_char, end_char = offset_mapping[s][0], offset_mapping[e][1]

        if start_char is None or end_char is None or start_char >= end_char:
            pred_text = ""
        else:
            pred_text = ctx[start_char:end_char].strip()

        predictions.append(pred_text)

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
/tmp/ipykernel_56/2672933178.py in _load_qa_model()
     25     try:
---> 26         tokenizer = AutoTokenizer.from_pretrained(
     27             FALLBACK_LOCAL_MODEL_ID, use_fast=True, local_files_only=True

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

FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_56/2672933178.py in <cell line: 0>()
     39 
     40 
---> 41 tokenizer, model = _load_qa_model()
     42 model.eval()
     43 

/tmp/ipykernel_56/2672933178.py in _load_qa_model()
     32         return tokenizer, model
     33     except Exception as e:
---> 34         raise FileNotFoundError(
     35             "Could not find the intended local model directory and no cached fallback model is available.\n"
     36             f"Missing: {PRIMARY_LOCAL_MODEL_DIR}\n"

FileNotFoundError: Could not find the intended local model directory and no cached fallback model is available.
Missing: /kaggle/input/pretrained-xlm-models-for-squad/mrm8488/xlm-multi-finetuned-xquadv1
Please add the pretrained model dataset to /kaggle/input, or ensure a QA model is available locally.

## === cell 2
if len(predictions) != len(test_df):
    raise RuntimeError(
        f"Predictions length {len(predictions)} does not match test_df length {len(test_df)}"
    )

submission_df = pd.DataFrame(
    {
        "id": test_df["id"].astype(str).values,
        "PredictionString": pd.Series(predictions, dtype="string").fillna("").values,
    }
)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

submission_df.head()

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2740225212.py in <cell line: 0>()
----> 1 if len(predictions) != len(test_df):
      2     raise RuntimeError(
      3         f"Predictions length {len(predictions)} does not match test_df length {len(test_df)}"
      4     )
      5 

NameError: name 'predictions' is not defined
