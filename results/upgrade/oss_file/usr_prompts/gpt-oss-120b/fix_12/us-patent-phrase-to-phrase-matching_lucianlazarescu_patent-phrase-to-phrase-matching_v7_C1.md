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
import numpy as np
import pandas as pd
import random, os, torch, multiprocessing
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer

cpu_cnt = multiprocessing.cpu_count()
torch.set_num_threads(cpu_cnt)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

torch.set_float32_matmul_precision("high")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def corr(x, y):
    return np.corrcoef(x, y)[0, 1]


def corr_d(eval_pred):
    return {"pearson": corr(eval_pred.predictions.squeeze(), eval_pred.label_ids)}




## === cell 2
path = "/kaggle/input/us-patent-phrase-to-phrase-matching/"

train_data = pd.read_csv(path + "train.csv")
test_data = pd.read_csv(path + "test.csv")




## === cell 3
train_data["section"] = train_data.context.str[0]
train_data["sectok"] = "[" + train_data.section + "]"

test_data["section"] = test_data.context.str[0]
test_data["sectok"] = "[" + test_data.section + "]"




## === cell 4
sectoks = list(train_data.sectok.unique())




## === cell 5
model_nm = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(model_nm, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)

if not torch.cuda.is_available():
    model = torch.compile(model)




## === cell 6
sep = " [s] "




## === cell 7
def prepare_data(df):
    return df.sectok + sep + df.context + sep + df.anchor.str.lower() + sep + df.target




## === cell 8
tokenizer.add_special_tokens({"additional_special_tokens": sectoks})
model.resize_token_embeddings(len(tokenizer))




## === cell 9
train_data["input"] = prepare_data(train_data)
test_data["input"] = prepare_data(test_data)

train_ds = Dataset.from_pandas(train_data[["input", "score"]])
test_ds = Dataset.from_pandas(test_data[["id", "input"]])




## === cell 10
def tokenize_batch(batch):
    return tokenizer(
        batch["input"], truncation=True, padding="max_length", max_length=64
    )


train_ds = train_ds.map(
    tokenize_batch,
    batched=True,
    batch_size=2000,
    num_proc=cpu_cnt,
    remove_columns=["input"],
)
test_ds = test_ds.map(
    tokenize_batch,
    batched=True,
    batch_size=2000,
    num_proc=cpu_cnt,
    remove_columns=["input"],
)

train_ds = train_ds.rename_column("score", "labels")
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])




## === cell 11
split_ds = train_ds.train_test_split(test_size=0.25, seed=42)
train_split = split_ds["train"]
val_split = split_ds["test"]




## === cell 12
bs = 128
epochs = 4
lr = 8e-5

args = TrainingArguments(
    output_dir="outputs",
    learning_rate=lr,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=True,
    eval_strategy="epoch",
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs * 2,
    num_train_epochs=epochs,
    weight_decay=0.01,
    report_to="none",
    save_strategy="no",
    dataloader_num_workers=cpu_cnt,  # faster batch loading with more workers
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_split,
    eval_dataset=val_split,
    tokenizer=tokenizer,
    compute_metrics=corr_d,
)




## === cell 13
trainer.train()




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/134302066.py in <cell line: 0>()
----> 1 trainer.train()
      2 
      3 

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in train(self, resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
   2204                 hf_hub_utils.enable_progress_bars()
   2205         else:
-> 2206             return inner_training_loop(
   2207                 args=args,
   2208                 resume_from_checkpoint=resume_from_checkpoint,

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in _inner_training_loop(self, batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)
   2254         logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
   2255         # Data loader and number of training steps
-> 2256         train_dataloader = self.get_train_dataloader()
   2257         if self.is_fsdp_xla_v2_enabled:
   2258             train_dataloader = tpu_spmd_dataloader(train_dataloader)

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in get_train_dataloader(self)
   1051             raise ValueError("Trainer: training requires a train_dataset.")
   1052 
-> 1053         return self._get_dataloader(
   1054             dataset=self.train_dataset,
   1055             description="Training",

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in _get_dataloader(self, dataset, description, batch_size, sampler_fn, is_training, dataloader_key)
   1005         data_collator = self.data_collator
   1006         if is_datasets_available() and isinstance(dataset, datasets.Dataset):
-> 1007             dataset = self._remove_unused_columns(dataset, description=description)
   1008         else:
   1009             data_collator = self._get_collator_with_removed_columns(self.data_collator, description=description)

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in _remove_unused_columns(self, dataset, description)
    931         columns = [k for k in signature_columns if k in dataset.column_names]
    932         if len(columns) == 0:
--> 933             raise ValueError(
    934                 f"No columns in the dataset match the model's forward method signature: ({', '.join(signature_columns)}). "
    935                 f"The following columns have been ignored: [{', '.join(ignored_columns)}]. "

ValueError: No columns in the dataset match the model's forward method signature: (args, kwargs, label, label_ids). The following columns have been ignored: [attention_mask, token_type_ids, labels, input_ids]. Please check the dataset and model. You may need to set `remove_unused_columns=False` in `TrainingArguments`.

## === cell 14
preds = trainer.predict(test_ds).predictions.squeeze()
preds = np.clip(preds, 0, 1)




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4083054166.py in <cell line: 0>()
----> 1 preds = trainer.predict(test_ds).predictions.squeeze()
      2 preds = np.clip(preds, 0, 1)
      3 
      4 

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in predict(self, test_dataset, ignore_keys, metric_key_prefix)
   4271         self._memory_tracker.start()
   4272 
-> 4273         test_dataloader = self.get_test_dataloader(test_dataset)
   4274         start_time = time.time()
   4275 

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in get_test_dataloader(self, test_dataset)
   1154                 `model.forward()` method are automatically removed. It must implement `__len__`.
   1155         """
-> 1156         return self._get_dataloader(
   1157             dataset=test_dataset,
   1158             description="test",

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in _get_dataloader(self, dataset, description, batch_size, sampler_fn, is_training, dataloader_key)
   1005         data_collator = self.data_collator
   1006         if is_datasets_available() and isinstance(dataset, datasets.Dataset):
-> 1007             dataset = self._remove_unused_columns(dataset, description=description)
   1008         else:
   1009             data_collator = self._get_collator_with_removed_columns(self.data_collator, description=description)

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in _remove_unused_columns(self, dataset, description)
    931         columns = [k for k in signature_columns if k in dataset.column_names]
    932         if len(columns) == 0:
--> 933             raise ValueError(
    934                 f"No columns in the dataset match the model's forward method signature: ({', '.join(signature_columns)}). "
    935                 f"The following columns have been ignored: [{', '.join(ignored_columns)}]. "

ValueError: No columns in the dataset match the model's forward method signature: (args, kwargs, label, label_ids). The following columns have been ignored: [attention_mask, token_type_ids, input_ids, id]. Please check the dataset and model. You may need to set `remove_unused_columns=False` in `TrainingArguments`.

## === cell 15
bins = np.array([0.125, 0.375, 0.625, 0.875])
scores = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
indices = np.digitize(preds, bins, right=False)
score = scores[indices].tolist()




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/772921302.py in <cell line: 0>()
      1 bins = np.array([0.125, 0.375, 0.625, 0.875])
      2 scores = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
----> 3 indices = np.digitize(preds, bins, right=False)
      4 score = scores[indices].tolist()
      5 

NameError: name 'preds' is not defined

## === cell 16
submission = pd.DataFrame({"id": test_data["id"], "score": score})
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3977904776.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"id": test_data["id"], "score": score})
      2 submission.to_csv("submission.csv", index=False)

NameError: name 'score' is not defined
