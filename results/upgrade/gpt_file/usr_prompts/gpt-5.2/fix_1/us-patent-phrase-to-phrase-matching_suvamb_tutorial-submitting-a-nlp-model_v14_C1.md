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

3.13

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

0.552832324180412

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

iskaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')

if not iskaggle:
    !pip install -q transformers datasets scikit-learn fsspec 


## === cell 1
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

path = Path('../input/us-patent-phrase-to-phrase-matching')
df = pd.read_csv(path/'train.csv')
test_df = pd.read_csv(path/'test.csv')




## === cell 2
df = pd.read_csv(path/'train.csv')  # contains anchor, target, context, score

anchors = df['anchor'].unique()
np.random.seed(42)
np.random.shuffle(anchors)

val_prop = 0.25
val_sz = int(len(anchors) * val_prop)
val_anchors = anchors[:val_sz]

is_val = df['anchor'].isin(val_anchors)
train_df = df[~is_val].reset_index(drop=True)
val_df   = df[is_val].reset_index(drop=True)


## === cell 4
model_name = '/kaggle/input/cached/mv1'
tokz = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
model = model.cuda()
sep = tokz.sep_token



df['inputs'] = df.context + sep + df.anchor + sep + df.target

df['label'] = df['score'].astype(np.float32)

anchors = df.anchor.unique()
np.random.seed(42)
np.random.shuffle(anchors)

val_prop = 0.25
val_sz = int(len(anchors) * val_prop)
val_anchors = anchors[:val_sz]

is_val = df.anchor.isin(val_anchors)
train_df = df[~is_val].reset_index(drop=True)
val_df = df[is_val].reset_index(drop=True)

print("Train/Val size:", len(train_df), len(val_df))
print("Train/Val score means:", train_df.label.mean(), val_df.label.mean())

## --- ERROR in cell 4, traceback:
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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/cached/mv1'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/tmp/ipykernel_11/2154602005.py in <cell line: 0>()
      1 # Load model/tokenizer once to cache
      2 model_name = '/kaggle/input/cached/mv1'
----> 3 tokz = AutoTokenizer.from_pretrained(model_name)
      4 model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
      5 model = model.cuda()

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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/cached/mv1'. Use `repo_type` argument if needed.

## === cell 5
class TextPairDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        inputs = self.tokenizer(
            row['inputs'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item['label'] = torch.tensor(row['label'], dtype=torch.float)
        return item


## === cell 6
BATCH_SIZE = 16

train_ds = TextPairDataset(train_df, tokz)
val_ds = TextPairDataset(val_df, tokz)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)


train_ds = TextPairDataset(train_df, tokz)
val_ds = TextPairDataset(val_df, tokz)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2377296022.py in <cell line: 0>()
      1 BATCH_SIZE = 16
      2 
----> 3 train_ds = TextPairDataset(train_df, tokz)
      4 val_ds = TextPairDataset(val_df, tokz)
      5 

NameError: name 'tokz' is not defined

## === cell 7
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import tqdm

## === cell 8
lr = 8e-5
bs = 32
wd = 0.01
epochs = 1
warmup_ratio = 0.1


## === cell 9
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)

num_training_steps = len(train_dl) * epochs
num_warmup_steps = int(num_training_steps * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)

loss_fn = nn.MSELoss()


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3904394312.py in <cell line: 0>()
----> 1 optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
      2 
      3 num_training_steps = len(train_dl) * epochs
      4 num_warmup_steps = int(num_training_steps * warmup_ratio)
      5 

NameError: name 'model' is not defined

## === cell 10
def pearsonr(x, y):
    return np.corrcoef(x, y)[0, 1]


## === cell 11
for epoch in range(epochs):
    model.train()
    train_loss = 0

    for batch in tqdm(train_dl, desc=f"Epoch {epoch+1} Training"):
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        labels = batch["label"].unsqueeze(1).cuda()

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits

        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_dl)

    model.eval()
    val_loss = 0
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for batch in tqdm(val_dl, desc=f"Epoch {epoch+1} Validation"):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["label"].unsqueeze(1).cuda()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits

            loss = loss_fn(preds, labels)
            val_loss += loss.item()

            preds_all.append(preds.cpu().numpy())
            labels_all.append(labels.cpu().numpy())

    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    pearson = pearsonr(preds_all.flatten(), labels_all.flatten())
    avg_val_loss = val_loss / len(val_dl)

    print(f"\nEpoch {epoch+1}:")
    print(f"Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f} | Pearson = {pearson:.4f}")


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3746749529.py in <cell line: 0>()
      1 for epoch in range(epochs):
----> 2     model.train()
      3     train_loss = 0
      4 
      5     for batch in tqdm(train_dl, desc=f"Epoch {epoch+1} Training"):

NameError: name 'model' is not defined

## === cell 12
test_df['inputs'] = test_df.context + sep + test_df.anchor + sep + test_df.target


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4014906175.py in <cell line: 0>()
----> 1 test_df['inputs'] = test_df.context + sep + test_df.anchor + sep + test_df.target

NameError: name 'sep' is not defined

## === cell 13
class TestDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        inputs = self.tokenizer(
            row['inputs'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item['id'] = row['id']
        return item

test_ds = TestDataset(test_df, tokz)
test_dl = DataLoader(test_ds, batch_size=128)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2154513.py in <cell line: 0>()
     21         return item
     22 
---> 23 test_ds = TestDataset(test_df, tokz)
     24 test_dl = DataLoader(test_ds, batch_size=128)

NameError: name 'tokz' is not defined

## === cell 14
model.eval()
pred_ids = []
pred_scores = []

with torch.no_grad():
    for batch in tqdm(test_dl, desc="Predicting"):
        ids = batch.pop('id')
        inputs = {k: v.cuda() for k, v in batch.items()}
        outputs = model(**inputs)
        preds = outputs.logits.squeeze(-1).cpu().numpy()

        pred_ids.extend(ids)
        pred_scores.extend(preds)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1276489439.py in <cell line: 0>()
----> 1 model.eval()
      2 pred_ids = []
      3 pred_scores = []
      4 
      5 with torch.no_grad():

NameError: name 'model' is not defined

## === cell 15
submission = pd.DataFrame({'id': pred_ids, 'score': pred_scores})
submission.to_csv("submission.csv", index=False)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2697301216.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({'id': pred_ids, 'score': pred_scores})
      2 submission.to_csv("submission.csv", index=False)

NameError: name 'pred_ids' is not defined

## === cell 16
!cat submission.csv

## === cell 17
!cat /kaggle/input/us-patent-phrase-to-phrase-matching/sample_submission.csv
