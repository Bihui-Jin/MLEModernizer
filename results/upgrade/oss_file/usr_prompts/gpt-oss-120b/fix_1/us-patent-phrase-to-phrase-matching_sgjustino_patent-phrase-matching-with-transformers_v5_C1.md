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

3.12

# 3. Installed packages

datasets==4.4.1
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
seaborn==0.12.2
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
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
vega-datasets==0.9.0

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

0.7805410791891036

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 5

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings,transformers,logging,torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments,Trainer
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
import time
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset
import gc
from torch.utils.data import DataLoader
from datasets import DatasetDict
import datasets

warnings.simplefilter('ignore')
logging.disable(logging.WARNING)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
train_df = pd.read_csv('/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv')

test_df = pd.read_csv('/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv')

print(train_df.head())

print(train_df.info())

## === cell 7
missing_values = train_df.isnull().sum()
dataset_summary = train_df.describe(include='all')

missing_values, dataset_summary

## === cell 9
sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.histplot(train_df['score'], bins=20, kde=True)
plt.title('Distribution of Similarity Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()

## === cell 11
common_anchors = train_df['anchor'].value_counts().head(10)
common_targets = train_df['target'].value_counts().head(10)

common_anchors, common_targets

## === cell 13
common_contexts = train_df['context'].value_counts().head(10)

common_contexts

## === cell 15
train_df['anchor_length'] = train_df['anchor'].apply(len)
train_df['target_length'] = train_df['target'].apply(len)

plt.figure(figsize=(14, 6))

max_length = max(train_df['anchor_length'].max(), train_df['target_length'].max())

plt.subplot(1, 2, 1)
sns.histplot(train_df['anchor_length'], bins=30, kde=True)
plt.title('Distribution of Anchor Phrase Lengths')
plt.xlabel('Length of Anchor Phrase')
plt.ylabel('Frequency')
plt.xlim(0, max_length)

plt.subplot(1, 2, 2)
sns.histplot(train_df['target_length'], bins=30, kde=True)
plt.title('Distribution of Target Phrase Lengths')
plt.xlabel('Length of Target Phrase')
plt.xlim(0, max_length)

plt.tight_layout()
plt.show()

## === cell 17
high_similarity_examples = train_df[train_df['score'] >= 0.9].sample(5, random_state=1)
low_similarity_examples = train_df[train_df['score'] <= 0.1].sample(5, random_state=1)

high_similarity_examples, low_similarity_examples

## === cell 19
model_path = '/kaggle/input/deberta-v3-small/deberta-v3-small'
tokenizer_deberta = AutoTokenizer.from_pretrained(model_path)

## --- ERROR in cell 19, traceback:
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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/deberta-v3-small/deberta-v3-small'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/tmp/ipykernel_11/2009993438.py in <cell line: 0>()
      1 model_path = '/kaggle/input/deberta-v3-small/deberta-v3-small'
----> 2 tokenizer_deberta = AutoTokenizer.from_pretrained(model_path)

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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/deberta-v3-small/deberta-v3-small'. Use `repo_type` argument if needed.

## === cell 21
sep = tokenizer_deberta.sep_token
train_df['inputs'] = train_df['context'] + sep + train_df['anchor'] + sep + train_df['target']
test_df['inputs'] = test_df['context'] + sep + test_df['anchor'] + sep + test_df['target']

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/394840975.py in <cell line: 0>()
----> 1 sep = tokenizer_deberta.sep_token
      2 train_df['inputs'] = train_df['context'] + sep + train_df['anchor'] + sep + train_df['target']
      3 test_df['inputs'] = test_df['context'] + sep + test_df['anchor'] + sep + test_df['target']

NameError: name 'tokenizer_deberta' is not defined

## === cell 23
train_ds = Dataset.from_pandas(train_df.rename(columns={"score": "label"}))
test_ds = Dataset.from_pandas(test_df)

## === cell 25
def token_func(examples): 
    return tokenizer_deberta(examples['inputs'], padding='max_length', truncation=True, max_length=48)

## === cell 27
tokenized_train_ds = train_ds.map(token_func, batched=True)
tokenized_test_ds = test_ds.map(token_func, batched=True)

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/493326115.py in <cell line: 0>()
----> 1 tokenized_train_ds = train_ds.map(token_func, batched=True)
      2 tokenized_test_ds = test_ds.map(token_func, batched=True)

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

/tmp/ipykernel_11/870320420.py in token_func(examples)
      1 def token_func(examples):
----> 2     return tokenizer_deberta(examples['inputs'], padding='max_length', truncation=True, max_length=48)

NameError: name 'tokenizer_deberta' is not defined

## === cell 28
tokenized_train_ds[0]

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3972606386.py in <cell line: 0>()
----> 1 tokenized_train_ds[0]

NameError: name 'tokenized_train_ds' is not defined

## === cell 30
columns_to_remove = ["anchor", "target", "anchor_length", "target_length", "context", "inputs", "id"]
tokenized_train_ds = tokenized_train_ds.remove_columns(columns_to_remove)

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/652966873.py in <cell line: 0>()
      1 columns_to_remove = ["anchor", "target", "anchor_length", "target_length", "context", "inputs", "id"]
----> 2 tokenized_train_ds = tokenized_train_ds.remove_columns(columns_to_remove)

NameError: name 'tokenized_train_ds' is not defined

## === cell 31
tokenized_train_ds[0]

## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3972606386.py in <cell line: 0>()
----> 1 tokenized_train_ds[0]

NameError: name 'tokenized_train_ds' is not defined

## === cell 33
train_indices, val_indices = train_test_split(range(len(tokenized_train_ds)), test_size=0.2, random_state=42)

dataset_split = DatasetDict({"train":tokenized_train_ds.select(train_indices),
             "test": tokenized_train_ds.select(val_indices)})

## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2864868309.py in <cell line: 0>()
----> 1 train_indices, val_indices = train_test_split(range(len(tokenized_train_ds)), test_size=0.2, random_state=42)
      2 
      3 dataset_split = DatasetDict({"train":tokenized_train_ds.select(train_indices),
      4              "test": tokenized_train_ds.select(val_indices)})

NameError: name 'tokenized_train_ds' is not defined

## === cell 38
def corr(eval_pred): return {'pearson': np.corrcoef(*eval_pred)[0][1]}

## === cell 40
args = TrainingArguments('outputs', learning_rate=8e-5, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=False,
    evaluation_strategy="epoch", per_device_train_batch_size=256, per_device_eval_batch_size=256,
    num_train_epochs=5, weight_decay=0.01, report_to='none')

deberta_model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)
deberta_trainer = Trainer(deberta_model, args, train_dataset=dataset_split['train'], eval_dataset=dataset_split['test'],
               tokenizer=tokenizer_deberta, compute_metrics=corr)

## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1775959056.py in <cell line: 0>()
----> 1 args = TrainingArguments('outputs', learning_rate=8e-5, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=False,
      2     evaluation_strategy="epoch", per_device_train_batch_size=256, per_device_eval_batch_size=256,
      3     num_train_epochs=5, weight_decay=0.01, report_to='none')
      4 
      5 deberta_model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)

TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'

## === cell 42
training_outcome = deberta_trainer.train()

## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4045374572.py in <cell line: 0>()
----> 1 training_outcome = deberta_trainer.train()

NameError: name 'deberta_trainer' is not defined

## === cell 43
test_prediction = deberta_trainer.predict(tokenized_test_ds).predictions.astype(float)
test_prediction

## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3641982051.py in <cell line: 0>()
----> 1 test_prediction = deberta_trainer.predict(tokenized_test_ds).predictions.astype(float)
      2 test_prediction

NameError: name 'deberta_trainer' is not defined

## === cell 44
rounded_test_prediction = np.round(test_prediction * 4) / 4
rounded_test_prediction = np.clip(rounded_test_prediction, 0, 1)
rounded_test_prediction = rounded_test_prediction.flatten()
rounded_test_prediction

## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/632381621.py in <cell line: 0>()
----> 1 rounded_test_prediction = np.round(test_prediction * 4) / 4
      2 rounded_test_prediction = np.clip(rounded_test_prediction, 0, 1)
      3 rounded_test_prediction = rounded_test_prediction.flatten()
      4 rounded_test_prediction

NameError: name 'test_prediction' is not defined

## === cell 45
submission = pd.DataFrame({
    'id': test_df['id'],
    'score': rounded_test_prediction,
})

submission.head(14)

## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/924050103.py in <cell line: 0>()
      1 submission = pd.DataFrame({
      2     'id': test_df['id'],
----> 3     'score': rounded_test_prediction,
      4 })
      5 

NameError: name 'rounded_test_prediction' is not defined

## === cell 46
submission.to_csv('submission.csv', index=False)

## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1690294540.py in <cell line: 0>()
----> 1 submission.to_csv('submission.csv', index=False)

NameError: name 'submission' is not defined
