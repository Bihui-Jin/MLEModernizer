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

No external packages required in the script and installed.

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

0.8069452756275215

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

base_path = Path("input/us-patent-phrase-to-phrase-matching")
if not base_path.exists():
    base_path = Path("us-patent-phrase-to-phrase-matching")
assert base_path.exists(), f"Data path not found: {base_path}"



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv(base_path / "train.csv")
test_df = pd.read_csv(base_path / "test.csv")

train_df["input"] = (
    "TEXT1: "
    + train_df.context
    + "; TEXT2: "
    + train_df.target
    + "; ANC1: "
    + train_df.anchor
)
test_df["input"] = (
    "TEXT1: "
    + test_df.context
    + "; TEXT2: "
    + test_df.target
    + "; ANC1: "
    + test_df.anchor
)



## === cell 2
train_ds = Dataset.from_pandas(train_df[["id", "input", "score"]])
test_ds = Dataset.from_pandas(test_df[["id", "input"]])



## === cell 3
model_name = "google/deberta-v3-small"
tokz = AutoTokenizer.from_pretrained(model_name)


def tok_func(examples):
    return tokz(
        examples["input"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_attention_mask=True,
    )


train_ds = train_ds.map(tok_func, batched=True, remove_columns=["input"])
test_ds = test_ds.map(tok_func, batched=True, remove_columns=["input"])

train_ds = train_ds.rename_column("score", "labels")
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])



## --- ERROR in cell 3, traceback:
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

HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/google/deberta-v3-small/resolve/main/tokenizer_config.json

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

RepositoryNotFoundError: 401 Client Error. (Request ID: Root=1-69f769f3-102a4e8a167811eb06269be8;f7e43a21-2a47-4b87-8d54-c6d315b6cd98)

Repository Not Found for url: https://huggingface.co/google/deberta-v3-small/resolve/main/tokenizer_config.json.
Please make sure you specified the correct `repo_id` and `repo_type`.
If you are trying to access a private or gated repo, make sure you are authenticated. For more details, see https://huggingface.co/docs/huggingface_hub/authentication
Invalid username or password.

The above exception was the direct cause of the following exception:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/4140291236.py in <cell line: 0>()
      1 # Use a publicly available model – DeBERTa‑v3‑small
      2 model_name = "google/deberta-v3-small"
----> 3 tokz = AutoTokenizer.from_pretrained(model_name)
      4 
      5 

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

OSError: google/deberta-v3-small is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo either by logging in with `huggingface-cli login` or by passing `token=<your_token>`

## === cell 4
splits = train_ds.train_test_split(test_size=0.25, seed=42)
train_split = splits["train"]
val_split = splits["test"]



## === cell 5
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)



## --- ERROR in cell 5, traceback:
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

HTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/google/deberta-v3-small/resolve/main/config.json

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

RepositoryNotFoundError: 401 Client Error. (Request ID: Root=1-69f769f4-53f2f5961f6b3703541dbae9;c69509c8-c89e-4419-a601-850c978b3d64)

Repository Not Found for url: https://huggingface.co/google/deberta-v3-small/resolve/main/config.json.
Please make sure you specified the correct `repo_id` and `repo_type`.
If you are trying to access a private or gated repo, make sure you are authenticated. For more details, see https://huggingface.co/docs/huggingface_hub/authentication
Invalid username or password.

The above exception was the direct cause of the following exception:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/4234892754.py in <cell line: 0>()
      1 # Load the model for regression (single continuous label)
----> 2 model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
      3 

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

OSError: google/deberta-v3-small is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'
If this is a private repository, make sure to pass a token having permission to this repo either by logging in with `huggingface-cli login` or by passing `token=<your_token>`

## === cell 6
training_args = TrainingArguments(
    output_dir="outputs",
    learning_rate=8e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=True,
    eval_strategy="epoch",  # corrected from “evaluation_strategy”
    per_device_train_batch_size=128,
    per_device_eval_batch_size=256,
    num_train_epochs=2,  # keep training short for the 600 s limit
    weight_decay=0.01,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="pearson",
    greater_is_better=True,
)


def compute_pearson(eval_pred):
    preds = eval_pred.predictions.squeeze()
    labels = eval_pred.label_ids
    return {"pearson": np.corrcoef(preds, labels)[0, 1]}


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_split,
    eval_dataset=val_split,
    tokenizer=tokz,
    compute_metrics=compute_pearson,
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2206883019.py in <cell line: 0>()
      1 # Training arguments – use the correct argument name for the evaluation strategy
----> 2 training_args = TrainingArguments(
      3     output_dir="outputs",
      4     learning_rate=8e-5,
      5     warmup_ratio=0.1,

/usr/local/lib/python3.11/dist-packages/transformers/training_args.py in __init__(self, output_dir, overwrite_output_dir, do_train, do_eval, do_predict, eval_strategy, prediction_loss_only, per_device_train_batch_size, per_device_eval_batch_size, per_gpu_train_batch_size, per_gpu_eval_batch_size, gradient_accumulation_steps, eval_accumulation_steps, eval_delay, torch_empty_cache_steps, learning_rate, weight_decay, adam_beta1, adam_beta2, adam_epsilon, max_grad_norm, num_train_epochs, max_steps, lr_scheduler_type, lr_scheduler_kwargs, warmup_ratio, warmup_steps, log_level, log_level_replica, log_on_each_node, logging_dir, logging_strategy, logging_first_step, logging_steps, logging_nan_inf_filter, save_strategy, save_steps, save_total_limit, save_safetensors, save_on_each_node, save_only_model, restore_callback_states_from_checkpoint, no_cuda, use_cpu, use_mps_device, seed, data_seed, jit_mode_eval, use_ipex, bf16, fp16, fp16_opt_level, half_precision_backend, bf16_full_eval, fp16_full_eval, tf32, local_rank, ddp_backend, tpu_num_cores, tpu_metrics_debug, debug, dataloader_drop_last, eval_steps, dataloader_num_workers, dataloader_prefetch_factor, past_index, run_name, disable_tqdm, remove_unused_columns, label_names, load_best_model_at_end, metric_for_best_model, greater_is_better, ignore_data_skip, fsdp, fsdp_min_num_params, fsdp_config, fsdp_transformer_layer_cls_to_wrap, accelerator_config, deepspeed, label_smoothing_factor, optim, optim_args, adafactor, group_by_length, length_column_name, report_to, ddp_find_unused_parameters, ddp_bucket_cap_mb, ddp_broadcast_buffers, dataloader_pin_memory, dataloader_persistent_workers, skip_memory_metrics, use_legacy_prediction_loop, push_to_hub, resume_from_checkpoint, hub_model_id, hub_strategy, hub_token, hub_private_repo, hub_always_push, hub_revision, gradient_checkpointing, gradient_checkpointing_kwargs, include_inputs_for_metrics, include_for_metrics, eval_do_concat_batches, fp16_backend, push_to_hub_model_id, push_to_hub_organization, push_to_hub_token, mp_parameters, auto_find_batch_size, full_determinism, torchdynamo, ray_scope, ddp_timeout, torch_compile, torch_compile_backend, torch_compile_mode, include_tokens_per_second, include_num_input_tokens_seen, neftune_noise_alpha, optim_target_modules, batch_eval_metrics, eval_on_start, use_liger_kernel, liger_kernel_config, eval_use_gather_object, average_tokens_across_devices)

/usr/local/lib/python3.11/dist-packages/transformers/training_args.py in __post_init__(self)
   1661         if self.load_best_model_at_end and self.save_strategy != SaveStrategy.BEST:
   1662             if self.eval_strategy != self.save_strategy:
-> 1663                 raise ValueError(
   1664                     "--load_best_model_at_end requires the save and eval strategy to match, but found\n- Evaluation "
   1665                     f"strategy: {self.eval_strategy}\n- Save strategy: {self.save_strategy}"

ValueError: --load_best_model_at_end requires the save and eval strategy to match, but found
- Evaluation strategy: IntervalStrategy.EPOCH
- Save strategy: SaveStrategy.STEPS

## === cell 7
trainer.train()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1589621210.py in <cell line: 0>()
      1 # Train the model (short run)
----> 2 trainer.train()
      3 

NameError: name 'trainer' is not defined

## === cell 8
preds = trainer.predict(test_ds).predictions.squeeze()
preds = np.clip(preds, 0.0, 1.0)  # ensure values are within the required range



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2817601719.py in <cell line: 0>()
      1 # Predict on the test set
----> 2 preds = trainer.predict(test_ds).predictions.squeeze()
      3 preds = np.clip(preds, 0.0, 1.0)  # ensure values are within the required range
      4 

NameError: name 'trainer' is not defined

## === cell 9
submission = pd.DataFrame({"id": test_df["id"], "score": preds})
submission_path = Path("submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path.resolve()}")

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2191570667.py in <cell line: 0>()
      1 # Build submission DataFrame and write CSV
----> 2 submission = pd.DataFrame({"id": test_df["id"], "score": preds})
      3 submission_path = Path("submission.csv")
      4 submission.to_csv(submission_path, index=False)
      5 print(f"Submission written to {submission_path.resolve()}")

NameError: name 'preds' is not defined
