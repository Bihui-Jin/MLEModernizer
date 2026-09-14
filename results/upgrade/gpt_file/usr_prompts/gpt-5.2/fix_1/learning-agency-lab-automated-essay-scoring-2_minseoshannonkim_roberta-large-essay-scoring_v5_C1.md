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

3.12

# 3. Installed packages

No external packages required in the script and installed.

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

0.8077619023103844

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



## === cell 2
train_file = '/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv'


train_df = pd.read_csv(train_file)
train_df.head()

## === cell 3
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

data_path = '/kaggle/input/learning-agency-lab-automated-essay-scoring-2'

data_files = {'train': 'train.csv', 'test': 'test.csv'}

train_file_path = f'{data_path}/train.csv'

dataset = load_dataset('csv', data_files=train_file_path)

dataset = dataset.rename_columns({"full_text":"text", "score":"labels"})

def adjust_labels(example):
    example['labels'] -= 1
    return example

dataset = dataset.map(adjust_labels)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
id2label = {0:1, 1: 2, 2:3, 3:4, 4:5, 5:6}
label2id = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5}
is_submission = True
output_dir = '/kaggle/input/roberta-large'

## === cell 7
if not is_submission:
    from peft import LoraConfig, PeftModel
    
    model_name = 'roberta-large'  # This can be any model suitable for sequence classification
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6, id2label=id2label, label2id=label2id)
    
    model.save_pretrained(save_directory=f"/kaggle/working/results/roberta-large-base")
    
    adapter_config= LoraConfig(
        **{
            "lora_alpha":16,
            "lora_dropout":0.1,
            "r":1,
            "bias":"none",
            "task_type":"SEQ_CLS"
        }
    )
    
    model.add_adapter(adapter_config)# 6 labels (0 to 5)
    
else:
    model_name = f"{output_dir}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6, id2label=id2label, label2id=label2id)


## --- ERROR in cell 7, traceback:
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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/roberta-large'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/tmp/ipykernel_11/2319441573.py in <cell line: 0>()
     23     model_name = f"{output_dir}"
     24 #     adapter_name = f"{output_dir}/roberta-large-adapter/checkpoint-2598"
---> 25     tokenizer = AutoTokenizer.from_pretrained(model_name)
     26     model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6, id2label=id2label, label2id=label2id)
     27 #     model.load_adapter(adapter_name)

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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/roberta-large'. Use `repo_type` argument if needed.

## === cell 9
model

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2355735933.py in <cell line: 0>()
----> 1 model

NameError: name 'model' is not defined

## === cell 10
from transformers import DataCollatorWithPadding

if not is_submission:
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=True)

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    train_set, valid_set = train_test_split(tokenized_datasets['train'], train_size=0.8, random_state=42, stratify=tokenized_datasets['train']['labels'])
    train_data, valid_data = Dataset.from_dict(train_set), Dataset.from_dict(valid_set)

## === cell 11
def sft_trainer(model, tokenizer, train_data, valid_data, data_collator):
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=4,
        weight_decay=0.01,
        report_to='none',
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    
    return trainer

## === cell 12
if not is_submission:
    trainer = sft_trainer(model, tokenizer, train_data, valid_data, data_collator)
    trainer.train()

## === cell 13
if not is_submission:
    trainer.model = trainer.model.merge_and_unload()
    trainer.save_model()

## === cell 14
%%bash
cd ./results
zip -r roberta-large-adapter.zip checkpoint-2598
cd ../

## === cell 23
from transformers import pipeline


text_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, max_length=512, truncation=True)  # device=0 for GPU

def load_test_dataset(test_file_path):
    test_dataset = load_dataset('csv', data_files=test_file_path)
    test_dataset = test_dataset.rename_columns({"full_text": "text"})
    return test_dataset

test_file_path = f"{data_path}/test.csv"

test_dataset = load_test_dataset(test_file_path)

predictions = text_classifier(test_dataset['train']['text'])

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3493857816.py in <cell line: 0>()
      4 
      5 # Initialize a text classification pipeline with your trained model
----> 6 text_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, max_length=512, truncation=True)  # device=0 for GPU
      7 
      8 # Function to load the test dataset without labels

NameError: name 'model' is not defined

## === cell 24
adjusted_predictions = [pred['label'] for pred in predictions]

result_df = pd.DataFrame({
    'essay_id': test_dataset['train']['essay_id'],  # Ensure 'essay_id' is present in the original dataset
    'score': adjusted_predictions
})

result_df.to_csv('submission.csv',index=False)

result_df

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3054581050.py in <cell line: 0>()
      1 # Convert predictions to the required format
----> 2 adjusted_predictions = [pred['label'] for pred in predictions]
      3 
      4 # Attach predictions to the essay_ids
      5 result_df = pd.DataFrame({

NameError: name 'predictions' is not defined
