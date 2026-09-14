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

0.8039326304000577

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,  # new import for efficient per‑batch padding
)

torch.manual_seed(42)
np.random.seed(42)

torch.set_num_threads(os.cpu_count() or 1)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
base_path = Path("../input/us-patent-phrase-to-phrase-matching")
train_path = base_path / "train.csv"
test_path = base_path / "test.csv"
sample_sub_path = base_path / "sample_submission.csv"



## === cell 2
train_df = pd.read_csv(train_path)
train_df["input"] = (
    "TEXT1: "
    + train_df["context"]
    + "; TEXT2: "
    + train_df["target"]
    + "; ANC1: "
    + train_df["anchor"]
)



## === cell 3
model_name = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)



## === cell 4
train_split, val_split = train_test_split(train_df, test_size=0.2, random_state=42)




## === cell 5
class PhraseDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, is_train=True):
        self.is_train = is_train
        self.texts = df["input"].tolist()
        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            padding=False,  # changed: avoid global longest padding
            max_length=256,
            return_tensors="pt",
        )
        if is_train:
            self.labels = torch.tensor(
                df["score"].astype(np.float32).values, dtype=torch.float32
            )
        else:
            self.labels = None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.is_train:
            item["labels"] = self.labels[idx]
        return item


train_dataset = PhraseDataset(train_split, tokenizer, is_train=True)
val_dataset = PhraseDataset(val_split, tokenizer, is_train=True)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in convert_to_tensors(self, tensor_type, prepend_batch_axis)
    766                 if not is_tensor(value):
--> 767                     tensor = as_tensor(value)
    768 

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in as_tensor(value, dtype)
    728                     return torch.from_numpy(np.array(value))
--> 729                 return torch.tensor(value)
    730 

ValueError: expected sequence of length 18 at dim 1 (got 19)

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2381831920.py in <cell line: 0>()
     28 
     29 
---> 30 train_dataset = PhraseDataset(train_split, tokenizer, is_train=True)
     31 val_dataset = PhraseDataset(val_split, tokenizer, is_train=True)
     32 

/tmp/ipykernel_11/2381831920.py in __init__(self, df, tokenizer, is_train)
      4         self.texts = df["input"].tolist()
      5         # Tokenize without padding; padding will be handled per‑batch by the collator
----> 6         self.encodings = tokenizer(
      7             self.texts,
      8             truncation=True,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in __call__(self, text, text_pair, text_target, text_pair_target, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, **kwargs)
   2853             if not self._in_target_context_manager:
   2854                 self._switch_to_input_mode()
-> 2855             encodings = self._call_one(text=text, text_pair=text_pair, **all_kwargs)
   2856         if text_target is not None:
   2857             self._switch_to_target_mode()

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in _call_one(self, text, text_pair, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, split_special_tokens, **kwargs)
   2941                 )
   2942             batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
-> 2943             return self.batch_encode_plus(
   2944                 batch_text_or_text_pairs=batch_text_or_text_pairs,
   2945                 add_special_tokens=add_special_tokens,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in batch_encode_plus(self, batch_text_or_text_pairs, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, split_special_tokens, **kwargs)
   3142         )
   3143 
-> 3144         return self._batch_encode_plus(
   3145             batch_text_or_text_pairs=batch_text_or_text_pairs,
   3146             add_special_tokens=add_special_tokens,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_fast.py in _batch_encode_plus(self, batch_text_or_text_pairs, add_special_tokens, padding_strategy, truncation_strategy, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, split_special_tokens)
    599         for input_ids in sanitized_tokens["input_ids"]:
    600             self._eventual_warn_about_too_long_sequence(input_ids, max_length, verbose)
--> 601         return BatchEncoding(sanitized_tokens, sanitized_encodings, tensor_type=return_tensors)
    602 
    603     def _encode_plus(

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in __init__(self, data, encoding, tensor_type, prepend_batch_axis, n_sequences)
    238         self._n_sequences = n_sequences
    239 
--> 240         self.convert_to_tensors(tensor_type=tensor_type, prepend_batch_axis=prepend_batch_axis)
    241 
    242     @property

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in convert_to_tensors(self, tensor_type, prepend_batch_axis)
    781                         "Please see if a fast version of this tokenizer is available to have this feature available."
    782                     ) from e
--> 783                 raise ValueError(
    784                     "Unable to create tensor, you should probably activate truncation and/or padding with"
    785                     " 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your"

ValueError: Unable to create tensor, you should probably activate truncation and/or padding with 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your features (`input_ids` in this case) have excessive nesting (inputs type `list` where type `int` is expected).

## === cell 6
def pearson_corr(pred):
    preds = pred.predictions.squeeze()
    labels = pred.label_ids.squeeze()
    return {"pearson": np.corrcoef(preds, labels)[0, 1]}




## === cell 7
if torch.cuda.is_available():
    train_batch = 64
    eval_batch = 128
    use_fp16 = True
else:
    train_batch = 32
    eval_batch = 64
    use_fp16 = False

training_args = TrainingArguments(
    output_dir="outputs",
    learning_rate=8e-5,
    per_device_train_batch_size=train_batch,
    per_device_eval_batch_size=eval_batch,
    num_train_epochs=2,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    fp16=use_fp16,
    eval_strategy="no",
    report_to="none",
    seed=42,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
)



## === cell 8
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=1, problem_type="regression"
)



## === cell 9
data_collator = DataCollatorWithPadding(tokenizer, padding="longest")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,  # added collator for speed
    compute_metrics=pearson_corr,
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1511068361.py in <cell line: 0>()
      5     model=model,
      6     args=training_args,
----> 7     train_dataset=train_dataset,
      8     eval_dataset=val_dataset,
      9     tokenizer=tokenizer,

NameError: name 'train_dataset' is not defined

## === cell 10
trainer.train()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3352579090.py in <cell line: 0>()
----> 1 trainer.train()
      2 

NameError: name 'trainer' is not defined

## === cell 11
test_df = pd.read_csv(test_path)
test_df["input"] = (
    "TEXT1: "
    + test_df["context"]
    + "; TEXT2: "
    + test_df["target"]
    + "; ANC1: "
    + test_df["anchor"]
)
test_dataset = PhraseDataset(test_df, tokenizer, is_train=False)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in convert_to_tensors(self, tensor_type, prepend_batch_axis)
    766                 if not is_tensor(value):
--> 767                     tensor = as_tensor(value)
    768 

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in as_tensor(value, dtype)
    728                     return torch.from_numpy(np.array(value))
--> 729                 return torch.tensor(value)
    730 

ValueError: expected sequence of length 19 at dim 1 (got 23)

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3800866664.py in <cell line: 0>()
      8     + test_df["anchor"]
      9 )
---> 10 test_dataset = PhraseDataset(test_df, tokenizer, is_train=False)
     11 

/tmp/ipykernel_11/2381831920.py in __init__(self, df, tokenizer, is_train)
      4         self.texts = df["input"].tolist()
      5         # Tokenize without padding; padding will be handled per‑batch by the collator
----> 6         self.encodings = tokenizer(
      7             self.texts,
      8             truncation=True,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in __call__(self, text, text_pair, text_target, text_pair_target, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, **kwargs)
   2853             if not self._in_target_context_manager:
   2854                 self._switch_to_input_mode()
-> 2855             encodings = self._call_one(text=text, text_pair=text_pair, **all_kwargs)
   2856         if text_target is not None:
   2857             self._switch_to_target_mode()

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in _call_one(self, text, text_pair, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, split_special_tokens, **kwargs)
   2941                 )
   2942             batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
-> 2943             return self.batch_encode_plus(
   2944                 batch_text_or_text_pairs=batch_text_or_text_pairs,
   2945                 add_special_tokens=add_special_tokens,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in batch_encode_plus(self, batch_text_or_text_pairs, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, split_special_tokens, **kwargs)
   3142         )
   3143 
-> 3144         return self._batch_encode_plus(
   3145             batch_text_or_text_pairs=batch_text_or_text_pairs,
   3146             add_special_tokens=add_special_tokens,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_fast.py in _batch_encode_plus(self, batch_text_or_text_pairs, add_special_tokens, padding_strategy, truncation_strategy, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, split_special_tokens)
    599         for input_ids in sanitized_tokens["input_ids"]:
    600             self._eventual_warn_about_too_long_sequence(input_ids, max_length, verbose)
--> 601         return BatchEncoding(sanitized_tokens, sanitized_encodings, tensor_type=return_tensors)
    602 
    603     def _encode_plus(

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in __init__(self, data, encoding, tensor_type, prepend_batch_axis, n_sequences)
    238         self._n_sequences = n_sequences
    239 
--> 240         self.convert_to_tensors(tensor_type=tensor_type, prepend_batch_axis=prepend_batch_axis)
    241 
    242     @property

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in convert_to_tensors(self, tensor_type, prepend_batch_axis)
    781                         "Please see if a fast version of this tokenizer is available to have this feature available."
    782                     ) from e
--> 783                 raise ValueError(
    784                     "Unable to create tensor, you should probably activate truncation and/or padding with"
    785                     " 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your"

ValueError: Unable to create tensor, you should probably activate truncation and/or padding with 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your features (`input_ids` in this case) have excessive nesting (inputs type `list` where type `int` is expected).

## === cell 12
preds = trainer.predict(test_dataset).predictions.squeeze()
preds = np.clip(preds, 0, 1)

submission = pd.DataFrame({"id": test_df["id"], "score": preds})
submission.to_csv("submission.csv", index=False)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/136298915.py in <cell line: 0>()
----> 1 preds = trainer.predict(test_dataset).predictions.squeeze()
      2 preds = np.clip(preds, 0, 1)
      3 
      4 submission = pd.DataFrame({"id": test_df["id"], "score": preds})
      5 submission.to_csv("submission.csv", index=False)

NameError: name 'trainer' is not defined

## === cell 13
print("Saved submission.csv")
print(submission.head())

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4178028048.py in <cell line: 0>()
      1 print("Saved submission.csv")
----> 2 print(submission.head())

NameError: name 'submission' is not defined
