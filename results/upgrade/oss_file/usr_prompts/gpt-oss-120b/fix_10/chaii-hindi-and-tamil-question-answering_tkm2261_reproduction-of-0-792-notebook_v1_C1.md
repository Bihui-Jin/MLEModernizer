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

0.727562665939331

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00773) has done: 'The changes speed up data preparation by using a faster iterator and pre‑allocating the feature list, shrink DataLoader overhead by using a single worker, and move tensors to the GPU only once per batch. Inference now reuses the same device tensors and avoids repeated `.cuda()` calls, while still loading each fold’s checkpoint and averaging logits exactly as before, preserving full model logic and prediction accuracy.'
- What this solution (achieved 0.0) has done: 'I added the missing imports, defined the simple `APEX_INSTALLED` flag, and replaced the undefined functions with a straightforward `transformers` question‑answering pipeline that runs the pretrained XLM‑RoBERTa model on each test example. The pipeline directly produces answer strings, which are then cleaned and written to a CSV file named `submission.csv` with the required column names. This fixes the runtime errors and provides a functional baseline that should achieve a reasonable Jaccard score toward the target.'
- What this solution (achieved 0.00955) has done: 'I added the missing `torch` import, removed the failing custom model loader, and created the QA pipeline directly from the model name/path. This avoids the protobuf error and the undefined `torch` reference, letting the script run end‑to‑end and correctly write a `submission.csv` with the required columns.'
- What this solution (achieved 0.00409) has done: 'I remove the imports that trigger the protobuf `MessageFactory` error and keep only what is needed for the QA pipeline. The `Config` class stay unchanged (it only stores paths), and the unused `make_model` function remain but won’t be called. This fixes the runtime error and still produces a proper `submission.csv` with the required columns, moving the solution from a failing state toward a usable baseline.'
- What this solution (achieved 0.0) has done: 'I adjust the configuration to use a publicly‑available multilingual QA model that is actually fine‑tuned for question answering (instead of the generic XLM‑RoBERTa checkpoint that caused almost empty answers). This small change keeps the overall pipeline unchanged while providing much more meaningful predictions, which should raise the Jaccard score toward the target. The rest of the code remains the same, and the script still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import string
import torch  # for device handling
import pandas as pd
from tqdm.auto import tqdm
from transformers import pipeline

APEX_INSTALLED = False


class Config:
    model_type = "xlm_roberta"
    model_name_or_path = "valhalla/xlm-roberta-large-qa-multilingual"
    config_name = "valhalla/xlm-roberta-large-qa-multilingual"
    fp16 = True if APEX_INSTALLED else False
    fp16_opt_level = "O1"
    gradient_accumulation_steps = 2

    tokenizer_name = "valhalla/xlm-roberta-large-qa-multilingual"
    max_seq_length = 400
    doc_stride = 135

    epochs = 1
    train_batch_size = 4
    eval_batch_size = 128

    optimizer_type = "AdamW"
    learning_rate = 1e-5
    weight_decay = 1e-2
    epsilon = 1e-8
    max_grad_norm = 1.0

    decay_name = "linear-warmup"
    warmup_ratio = 0.1

    logging_steps = 10

    output_dir = "output"
    seed = 2021




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def make_model(args):
    """
    Load config, tokenizer and model, falling back to the hub model if a local path is missing.
    This function is retained for compatibility but is not used in the current pipeline.
    """
    from transformers import AutoConfig, AutoTokenizer, AutoModelForQuestionAnswering

    try:
        config = AutoConfig.from_pretrained(args.config_name)
    except Exception:
        config = AutoConfig.from_pretrained("xlm-roberta-large")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large", use_fast=False)
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name_or_path, config=config
    )
    return config, tokenizer, model




## === cell 2
test_path = "../input/chaii-hindi-and-tamil-question-answering/test.csv"
test = pd.read_csv(test_path)

test["context"] = test["context"].apply(lambda x: " ".join(str(x).split()))
test["question"] = test["question"].apply(lambda x: " ".join(str(x).split()))

qa_pipeline = pipeline(
    "question-answering",
    model=Config().model_name_or_path,
    tokenizer=Config().tokenizer_name,
    device=0 if torch.cuda.is_available() else -1,
    framework="pt",
    max_seq_length=Config.max_seq_length,
    doc_stride=Config.doc_stride,
)




## --- ERROR in cell 2, traceback:
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

HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/valhalla/xlm-roberta-large-qa-multilingual/resolve/main/config.json

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

RepositoryNotFoundError: 401 Client Error. (Request ID: Root=1-69fab56d-5319a47063b2dfe63dc6a3a6;eeac4cb3-604d-4e01-a9e8-f423dbf5593b)

Repository Not Found for url: https://huggingface.co/valhalla/xlm-roberta-large-qa-multilingual/resolve/main/config.json.
Please make sure you specified the correct `repo_id` and `repo_type`.
If you are trying to access a private or gated repo, make sure you are authenticated. For more details, see https://huggingface.co/docs/huggingface_hub/authentication
Invalid username or password.

The above exception was the direct cause of the following exception:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/497165756.py in <cell line: 0>()
      8 
      9 # Initialise QA pipeline with the chosen multilingual model
---> 10 qa_pipeline = pipeline(
     11     "question-answering",
     12     model=Config().model_name_or_path,

/usr/local/lib/python3.11/dist-packages/transformers/pipelines/__init__.py in pipeline(task, model, config, tokenizer, feature_extractor, image_processor, processor, framework, revision, use_fast, token, device, device_map, torch_dtype, trust_remote_code, model_kwargs, pipeline_class, **kwargs)
    890         if not isinstance(config, PretrainedConfig) and pretrained_model_name_or_path is not None:
    891             # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
--> 892             resolved_config_file = cached_file(
    893                 pretrained_model_name_or_path,
    894                 CONFIG_NAME,

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

OSError: valhalla/xlm-roberta-large-qa-multilingual is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo either by logging in with `huggingface-cli login` or by passing `token=<your_token>`

## === cell 3
predictions = {}
for _, row in tqdm(test.iterrows(), total=len(test), desc="Running QA pipeline"):
    context = row["context"]
    question = row["question"]
    try:
        result = qa_pipeline(question=question, context=context, top_k=1)
        answer = result["answer"] if isinstance(result, dict) else result[0]["answer"]
    except Exception:
        answer = ""
    answer = " ".join(answer.split()).strip(string.punctuation)
    predictions[row["id"]] = answer

submission = pd.DataFrame(list(predictions.items()), columns=["id", "PredictionString"])

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission file written to {submission_path}")
