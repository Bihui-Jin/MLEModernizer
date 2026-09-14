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

3.11

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

0.8066012823938029

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from pathlib import Path
path = Path('../input/us-patent-phrase-to-phrase-matching')

## === cell 1
! ls {path}

## === cell 2
import pandas as pd

## === cell 3
df = pd.read_csv(path/"train.csv")
df.head()

## === cell 4
df.tail()

## === cell 5
df.describe(include=object)

## === cell 6
df['input'] = 'TEXT1: ' + df.context + '; TEXT2: ' + df.target + '; ANC: ' + df.anchor
df.input.head()

## === cell 7
from datasets import Dataset, DatasetDict
ds = Dataset.from_pandas(df)

## === cell 8
ds

## === cell 9
model_name = "../input/deberta-v3-small"

## === cell 10
from transformers import AutoModelForSequenceClassification, AutoTokenizer
tokz = AutoTokenizer.from_pretrained(model_name)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/deberta-v3-small'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/tmp/ipykernel_11/1030904020.py in <cell line: 0>()
      1 from transformers import AutoModelForSequenceClassification, AutoTokenizer
----> 2 tokz = AutoTokenizer.from_pretrained(model_name)

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
    520 
    521         # Now we try to recover if we can find all files correctly in the cache
--> 522         resolved_files = [
    523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in <listcomp>(.0)
    521         # Now we try to recover if we can find all files correctly in the cache
    522         resolved_files = [
--> 523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]
    525         if all(file is not None for file in resolved_files):

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
    138 ):
    139     # We try to see if we have a cached version (not up to date):
--> 140     resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision)
    141     if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:
    142         return resolved_file

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    104         ):
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 
    108             elif arg_name == "token" and arg_value is not None:

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    152 
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"
    156             f" '{repo_id}'. Use `repo_type` argument if needed."

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/deberta-v3-small'. Use `repo_type` argument if needed.

## === cell 11
type(tokz)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/983864178.py in <cell line: 0>()
----> 1 type(tokz)

NameError: name 'tokz' is not defined

## === cell 12
tokz.tokenize("This is a piece of text that the \"tokenizer\" is going to tokenize!")

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2943371933.py in <cell line: 0>()
----> 1 tokz.tokenize("This is a piece of text that the \"tokenizer\" is going to tokenize!")

NameError: name 'tokz' is not defined

## === cell 13
def tok_func(x): return tokz(x['input'])

## === cell 14
tok_ds = ds.map(tok_func, batched=True)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4163748473.py in <cell line: 0>()
----> 1 tok_ds = ds.map(tok_func, batched=True)

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in wrapper(*args, **kwargs)
    560         }
    561         # apply actual function
--> 562         out: Union["Dataset", "DatasetDict"] = func(self, *args, **kwargs)
    563         datasets: list["Dataset"] = list(out.values()) if isinstance(out, dict) else [out]
    564         # re-apply format to the output

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in map(self, function, with_indices, with_rank, input_columns, batched, batch_size, drop_last_batch, remove_columns, keep_in_memory, load_from_cache_file, cache_file_name, writer_batch_size, features, disable_nullable, fn_kwargs, num_proc, suffix_template, new_fingerprint, desc, try_original_type)
   3339                 else:
   3340                     for unprocessed_kwargs in unprocessed_kwargs_per_job:
-> 3341                         for rank, done, content in Dataset._map_single(**unprocessed_kwargs):
   3342                             check_if_shard_done(rank, done, content)
   3343 

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in _map_single(shard, function, with_indices, with_rank, input_columns, batched, batch_size, drop_last_batch, remove_columns, keep_in_memory, cache_file_name, writer_batch_size, features, disable_nullable, fn_kwargs, new_fingerprint, rank, offset, try_original_type)
   3695                 else:
   3696                     _time = time.time()
-> 3697                     for i, batch in iter_outputs(shard_iterable):
   3698                         num_examples_in_batch = len(i)
   3699                         if update_data:

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in iter_outputs(shard_iterable)
   3645             else:
   3646                 for i, example in shard_iterable:
-> 3647                     yield i, apply_function(example, i, offset=offset)
   3648 
   3649         num_examples_progress_update = 0

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in apply_function(pa_inputs, indices, offset)
   3568             """Utility to apply the function on a selection of columns."""
   3569             inputs, fn_args, additional_args, fn_kwargs = prepare_inputs(pa_inputs, indices, offset=offset)
-> 3570             processed_inputs = function(*fn_args, *additional_args, **fn_kwargs)
   3571             return prepare_outputs(pa_inputs, inputs, processed_inputs)
   3572 

/tmp/ipykernel_11/3123274832.py in tok_func(x)
----> 1 def tok_func(x): return tokz(x['input'])

NameError: name 'tokz' is not defined

## === cell 15
tok_ds[0]['input_ids']

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1299466251.py in <cell line: 0>()
----> 1 tok_ds[0]['input_ids']

NameError: name 'tok_ds' is not defined

## === cell 16
tokz.vocab['▁going']

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2693237368.py in <cell line: 0>()
----> 1 tokz.vocab['▁going']

NameError: name 'tokz' is not defined

## === cell 17
tok_ds = tok_ds.rename_columns({"score": "labels"})

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2892929822.py in <cell line: 0>()
----> 1 tok_ds = tok_ds.rename_columns({"score": "labels"})

NameError: name 'tok_ds' is not defined

## === cell 18
tok_ds[0]

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/760174941.py in <cell line: 0>()
----> 1 tok_ds[0]

NameError: name 'tok_ds' is not defined

## === cell 19
eval_df = pd.read_csv(path/"test.csv")
eval_df.describe()

## === cell 20
dds = tok_ds.train_test_split(0.25, seed=42)
dds

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3363613736.py in <cell line: 0>()
----> 1 dds = tok_ds.train_test_split(0.25, seed=42)
      2 dds

NameError: name 'tok_ds' is not defined

## === cell 21
eval_df['input'] = "TEXT1: " + df.context + "; TEXT2: " + eval_df.target + "; ANC: " + eval_df.anchor
eval_df.input.head()

## === cell 22
eval_ds = Dataset.from_pandas(eval_df).map(tok_func, batched=True)

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3110274974.py in <cell line: 0>()
----> 1 eval_ds = Dataset.from_pandas(eval_df).map(tok_func, batched=True)

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in wrapper(*args, **kwargs)
    560         }
    561         # apply actual function
--> 562         out: Union["Dataset", "DatasetDict"] = func(self, *args, **kwargs)
    563         datasets: list["Dataset"] = list(out.values()) if isinstance(out, dict) else [out]
    564         # re-apply format to the output

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in map(self, function, with_indices, with_rank, input_columns, batched, batch_size, drop_last_batch, remove_columns, keep_in_memory, load_from_cache_file, cache_file_name, writer_batch_size, features, disable_nullable, fn_kwargs, num_proc, suffix_template, new_fingerprint, desc, try_original_type)
   3339                 else:
   3340                     for unprocessed_kwargs in unprocessed_kwargs_per_job:
-> 3341                         for rank, done, content in Dataset._map_single(**unprocessed_kwargs):
   3342                             check_if_shard_done(rank, done, content)
   3343 

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in _map_single(shard, function, with_indices, with_rank, input_columns, batched, batch_size, drop_last_batch, remove_columns, keep_in_memory, cache_file_name, writer_batch_size, features, disable_nullable, fn_kwargs, new_fingerprint, rank, offset, try_original_type)
   3695                 else:
   3696                     _time = time.time()
-> 3697                     for i, batch in iter_outputs(shard_iterable):
   3698                         num_examples_in_batch = len(i)
   3699                         if update_data:

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in iter_outputs(shard_iterable)
   3645             else:
   3646                 for i, example in shard_iterable:
-> 3647                     yield i, apply_function(example, i, offset=offset)
   3648 
   3649         num_examples_progress_update = 0

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in apply_function(pa_inputs, indices, offset)
   3568             """Utility to apply the function on a selection of columns."""
   3569             inputs, fn_args, additional_args, fn_kwargs = prepare_inputs(pa_inputs, indices, offset=offset)
-> 3570             processed_inputs = function(*fn_args, *additional_args, **fn_kwargs)
   3571             return prepare_outputs(pa_inputs, inputs, processed_inputs)
   3572 

/tmp/ipykernel_11/3123274832.py in tok_func(x)
----> 1 def tok_func(x): return tokz(x['input'])

NameError: name 'tokz' is not defined

## === cell 23
import numpy as np

## === cell 24
def corr(x, y): return np.corrcoef(x, y)[0][1]

## === cell 25
def corr_d(eval_pred): return {'pearson': corr(*eval_pred)}

## === cell 26
from transformers import TrainingArguments, Trainer

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 27
bs = 128
epochs = 4
lr = 8e-5

## === cell 28
args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
                        evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
                        num_train_epochs=epochs, weight_decay=0.01, report_to='none')

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4128642071.py in <cell line: 0>()
----> 1 args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
      2                         evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
      3                         num_train_epochs=epochs, weight_decay=0.01, report_to='none')

TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'

## === cell 29
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
trainer = Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
                 tokenizer=tokz, compute_metrics=corr_d)

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/deberta-v3-small'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/tmp/ipykernel_11/3470072817.py in <cell line: 0>()
----> 1 model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
      2 trainer = Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
      3                  tokenizer=tokz, compute_metrics=corr_d)

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
    520 
    521         # Now we try to recover if we can find all files correctly in the cache
--> 522         resolved_files = [
    523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in <listcomp>(.0)
    521         # Now we try to recover if we can find all files correctly in the cache
    522         resolved_files = [
--> 523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]
    525         if all(file is not None for file in resolved_files):

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
    138 ):
    139     # We try to see if we have a cached version (not up to date):
--> 140     resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision)
    141     if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:
    142         return resolved_file

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    104         ):
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 
    108             elif arg_name == "token" and arg_value is not None:

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    152 
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"
    156             f" '{repo_id}'. Use `repo_type` argument if needed."

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/deberta-v3-small'. Use `repo_type` argument if needed.

## === cell 30
trainer.train();

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2912127331.py in <cell line: 0>()
----> 1 trainer.train();

NameError: name 'trainer' is not defined

## === cell 31
preds = trainer.predict(eval_ds).predictions.astype(float)
preds

## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3682598947.py in <cell line: 0>()
----> 1 preds = trainer.predict(eval_ds).predictions.astype(float)
      2 preds

NameError: name 'trainer' is not defined

## === cell 32
preds = np.clip(preds, 0, 1)

## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/498934129.py in <cell line: 0>()
----> 1 preds = np.clip(preds, 0, 1)

NameError: name 'preds' is not defined

## === cell 33
preds.min(), preds.max()

## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2322045482.py in <cell line: 0>()
----> 1 preds.min(), preds.max()

NameError: name 'preds' is not defined

## === cell 34
preds

## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/222146027.py in <cell line: 0>()
----> 1 preds

NameError: name 'preds' is not defined

## === cell 35
import datasets

submission = datasets.Dataset.from_dict({
    'id': eval_ds['id'],
    'score': [preds[o][0] for o in range(len(preds))]
})

submission.to_csv('submission.csv', index=False)

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1932532920.py in <cell line: 0>()
      2 
      3 submission = datasets.Dataset.from_dict({
----> 4     'id': eval_ds['id'],
      5     'score': [preds[o][0] for o in range(len(preds))]
      6 })

NameError: name 'eval_ds' is not defined
