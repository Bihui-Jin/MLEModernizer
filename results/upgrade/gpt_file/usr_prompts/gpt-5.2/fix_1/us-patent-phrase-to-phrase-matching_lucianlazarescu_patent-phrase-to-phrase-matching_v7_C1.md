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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
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

0.7853653726936117

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import pandas as pd
import numpy as np
import datasets

from datasets import Dataset,DatasetDict
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from transformers import TrainingArguments,Trainer


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def corr(x,y): return np.corrcoef(x,y)[0][1]


## === cell 3
def corr_d(eval_pred): return {'pearson': corr(*eval_pred)}


## === cell 4
path = '/kaggle/input/us-patent-phrase-to-phrase-matching/'
train_data = pd.read_csv(path + 'train.csv')
print(train_data)


## === cell 5
train_data.target.value_counts()

## === cell 6
train_data.anchor.value_counts()

## === cell 7
train_data.score.hist()

## === cell 8
train_data['section'] = train_data.context.str[0]
train_data.section.value_counts()

## === cell 9
train_data['sectok'] = '[' + train_data.section + ']'
sectoks = list(train_data.sectok.unique())
sectoks

## === cell 10
test_data = pd.read_csv(path + 'test.csv')
print(test_data)


## === cell 11
test_data['section'] = test_data.context.str[0]
test_data.section.value_counts()

## === cell 12
model_nm = '/kaggle/input/deberta-v3-small'
tokenizer  = AutoTokenizer.from_pretrained(model_nm, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)


## --- ERROR in cell 12, traceback:
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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/deberta-v3-small'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/tmp/ipykernel_11/2567972751.py in <cell line: 0>()
      2 #model_nm = 'microsoft/deberta-v3-large'
      3 model_nm = '/kaggle/input/deberta-v3-small'
----> 4 tokenizer  = AutoTokenizer.from_pretrained(model_nm, use_fast=False)
      5 model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)

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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/deberta-v3-small'. Use `repo_type` argument if needed.

## === cell 13
sep = tokenizer.sep_token

sep = " [s] "

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3930358809.py in <cell line: 0>()
----> 1 sep = tokenizer.sep_token
      2 
      3 # change separator
      4 sep = " [s] "

NameError: name 'tokenizer' is not defined

## === cell 14
def prepare_data(df):

    
    out = df.sectok + sep + df.context + sep + df.anchor.str.lower() + sep + df.target 

    return out

## === cell 15

tokenizer.add_special_tokens({'additional_special_tokens': sectoks})
train_data['input']=prepare_data(train_data)
ds = Dataset.from_pandas(train_data)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3621352806.py in <cell line: 0>()
      5 ## Version 2
      6 ##train_data['input'] = train_data.context + sep +  train_data.anchor + sep + train_data.target
----> 7 tokenizer.add_special_tokens({'additional_special_tokens': sectoks})
      8 train_data['input']=prepare_data(train_data)
      9 ds = Dataset.from_pandas(train_data)

NameError: name 'tokenizer' is not defined

## === cell 16
model

model.resize_token_embeddings(len(tokenizer))

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/841421142.py in <cell line: 0>()
----> 1 model
      2 
      3 #model = get_model()
      4 model.resize_token_embeddings(len(tokenizer))

NameError: name 'model' is not defined

## === cell 17
def tok_func(x):
    return tokenizer(x["input"])


## === cell 18
tok_ds = ds.map(tok_func, batched=True)
tok_ds = tok_ds.rename_columns({'score':'labels'}) # Rename target column to label


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/488615246.py in <cell line: 0>()
      1 # Adds input_ids column with the numericalized input
----> 2 tok_ds = ds.map(tok_func, batched=True)
      3 tok_ds = tok_ds.rename_columns({'score':'labels'}) # Rename target column to label

NameError: name 'ds' is not defined

## === cell 19
i = 0
for line in tok_ds:
    print(line)
    i = i + 1
    if i == 5:
        break

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4270463723.py in <cell line: 0>()
      1 i = 0
----> 2 for line in tok_ds:
      3     print(line)
      4     i = i + 1
      5     if i == 5:

NameError: name 'tok_ds' is not defined

## === cell 20
dds = tok_ds.train_test_split(0.25, seed=42)
dds

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2603786897.py in <cell line: 0>()
      1 # Create DataSetDict by splitting the training data into train/validation sets
----> 2 dds = tok_ds.train_test_split(0.25, seed=42)
      3 dds

NameError: name 'tok_ds' is not defined

## === cell 21
eval_df = pd.read_csv(path+'test.csv')


eval_df['section'] = eval_df.context.str[0]
eval_df['sectok'] = '[' + eval_df.section + ']'
eval_df['input'] = prepare_data(eval_df)
eval_ds = Dataset.from_pandas(eval_df).map(tok_func, batched=True)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1736143057.py in <cell line: 0>()
      9 eval_df['section'] = eval_df.context.str[0]
     10 eval_df['sectok'] = '[' + eval_df.section + ']'
---> 11 eval_df['input'] = prepare_data(eval_df)
     12 eval_ds = Dataset.from_pandas(eval_df).map(tok_func, batched=True)

/tmp/ipykernel_11/730810207.py in prepare_data(df)
      7 
      8     ## Version 3
----> 9     out = df.sectok + sep + df.context + sep + df.anchor.str.lower() + sep + df.target
     10 
     11 #    out = out.str.lower()

NameError: name 'sep' is not defined

## === cell 22
bs = 32
epochs = 4
lr = 8e-5


## === cell 23
args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=True,
    eval_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
    num_train_epochs=epochs, weight_decay=0.01, report_to='none', save_strategy = 'no')

trainer = Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
                  tokenizer=tokenizer, compute_metrics=corr_d)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3595071899.py in <cell line: 0>()
      4     num_train_epochs=epochs, weight_decay=0.01, report_to='none', save_strategy = 'no')
      5 
----> 6 trainer = Trainer(model, args, train_dataset=dds['train'], eval_dataset=dds['test'],
      7                   tokenizer=tokenizer, compute_metrics=corr_d)
      8 

NameError: name 'model' is not defined

## === cell 24
trainer.train()


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/408984990.py in <cell line: 0>()
      1 # Train the model
----> 2 trainer.train()

NameError: name 'trainer' is not defined

## === cell 25
preds = trainer.predict(eval_ds).predictions.astype(float)
preds = np.clip(preds, 0, 1) # Clip all predicitons to 0 or 1


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1650141799.py in <cell line: 0>()
      1 # Make predictions on the eval_ds
----> 2 preds = trainer.predict(eval_ds).predictions.astype(float)
      3 preds = np.clip(preds, 0, 1) # Clip all predicitons to 0 or 1

NameError: name 'trainer' is not defined

## === cell 26
score=[]
for value in preds:
    if value >= 0.875:
        score.append(1)
        continue
    if value >= 0.625:
        score.append(0.75)
        continue
    if value >= 0.375:
        score.append(0.5)
        continue
    if value >= 0.125:
        score.append(0.25)
        continue
    score.append(0)
    

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2081179793.py in <cell line: 0>()
      1 score=[]
----> 2 for value in preds:
      3     if value >= 0.875:
      4         score.append(1)
      5         continue

NameError: name 'preds' is not defined

## === cell 27
submission = Dataset.from_dict({
    'id': eval_ds['id'],
    'score': score # preds.flatten()
})

submission.to_csv('submission.csv', index=False)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1364247575.py in <cell line: 0>()
      1 submission = Dataset.from_dict({
----> 2     'id': eval_ds['id'],
      3     'score': score # preds.flatten()
      4 })
      5 

NameError: name 'eval_ds' is not defined
