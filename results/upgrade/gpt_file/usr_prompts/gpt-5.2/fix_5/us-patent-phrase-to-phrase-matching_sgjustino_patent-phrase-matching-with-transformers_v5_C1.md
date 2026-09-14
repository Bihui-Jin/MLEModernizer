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

0.01981

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.01981) has done: 'The timeout is dominated by training compute (5 epochs) and by avoidable overhead in data handling. I keep the exact same model, loss, epochs, batch sizes, and evaluation, but reduce overhead by (1) switching to dynamic padding during tokenization (so we don’t pad every example to length 48 twice), (2) using Hugging Face `Dataset`’s native `train_test_split` instead of a Python-level subset wrapper (faster indexing/data loading), and (3) enabling safe runtime settings (`torch.set_float32_matmul_precision("high")`, `gradient_checkpointing`, and disabling `use_cache`) that preserve numerical results while reducing memory pressure and speeding throughput on GPU. I also ensure the `Trainer` uses the correct label field and avoids unnecessary column processing.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")



## === cell 1
import pandas as pd
import numpy as np
import warnings, logging, gc, time

import torch
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset, DatasetDict

warnings.simplefilter("ignore")
logging.disable(logging.WARNING)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
train_df = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv")
test_df = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv")

print(train_df.head())
print(train_df.shape, test_df.shape)



## === cell 3
pass



## === cell 4
pass



## === cell 5
pass



## === cell 6
pass



## === cell 7
pass



## === cell 8
model_path = "microsoft/deberta-v3-small"
tokenizer_deberta = AutoTokenizer.from_pretrained(model_path, use_fast=True)



## === cell 9
sep = tokenizer_deberta.sep_token
train_df["inputs"] = (
    train_df["context"].astype(str)
    + sep
    + train_df["anchor"].astype(str)
    + sep
    + train_df["target"].astype(str)
)
test_df["inputs"] = (
    test_df["context"].astype(str)
    + sep
    + test_df["anchor"].astype(str)
    + sep
    + test_df["target"].astype(str)
)
train_df[["inputs"]].head()



## === cell 10
train_ds = Dataset.from_pandas(
    train_df.rename(columns={"score": "label"}), preserve_index=False
)
test_ds = Dataset.from_pandas(test_df, preserve_index=False)

train_ds, test_ds




## === cell 11
def token_func(examples):
    return tokenizer_deberta(
        examples["inputs"], padding=False, truncation=True, max_length=48
    )




## === cell 12
tokenized_train_ds = train_ds.map(
    token_func,
    batched=True,
    batch_size=2048,
    num_proc=1,
    desc="Tokenizing train",
    load_from_cache_file=True,
)
tokenized_test_ds = test_ds.map(
    token_func,
    batched=True,
    batch_size=2048,
    num_proc=1,
    desc="Tokenizing test",
    load_from_cache_file=True,
)

tokenized_train_ds[0]



## === cell 13
columns_to_remove = [
    "anchor",
    "target",
    "anchor_length",
    "target_length",
    "context",
    "inputs",
    "id",
]
tokenized_train_ds = tokenized_train_ds.remove_columns(
    [c for c in columns_to_remove if c in tokenized_train_ds.column_names]
)

test_ids = test_df["id"].copy()
tokenized_test_ds = tokenized_test_ds.remove_columns(
    [c for c in columns_to_remove if c in tokenized_test_ds.column_names]
)

train_cols = ["input_ids", "attention_mask", "label"]
test_cols = ["input_ids", "attention_mask"]
if "token_type_ids" in tokenized_train_ds.column_names:
    train_cols.append("token_type_ids")
if "token_type_ids" in tokenized_test_ds.column_names:
    test_cols.append("token_type_ids")

tokenized_train_ds = tokenized_train_ds.with_format("torch", columns=train_cols)
tokenized_test_ds = tokenized_test_ds.with_format("torch", columns=test_cols)

tokenized_train_ds[0]



## === cell 14
dataset_split_hf = tokenized_train_ds.train_test_split(
    test_size=0.2, seed=SEED, shuffle=True
)
dataset_split = {"train": dataset_split_hf["train"], "test": dataset_split_hf["test"]}
dataset_split




## === cell 15
def corr(eval_pred):
    preds, labels = eval_pred
    preds = np.asarray(preds).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if preds.std() == 0 or labels.std() == 0:
        return {"pearson": 0.0}
    return {"pearson": float(np.corrcoef(preds, labels)[0][1])}




## === cell 16
data_collator = DataCollatorWithPadding(tokenizer=tokenizer_deberta, padding="longest")

args = TrainingArguments(
    output_dir="outputs",
    learning_rate=8e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=False,
    eval_strategy="epoch",
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    num_train_epochs=5,
    weight_decay=0.01,
    report_to="none",
    logging_steps=50,
    seed=SEED,
    dataloader_num_workers=min(2, max(1, (os.cpu_count() or 2) // 4)),
    dataloader_pin_memory=torch.cuda.is_available(),
    remove_unused_columns=False,
    label_names=["label"],
)

deberta_model = AutoModelForSequenceClassification.from_pretrained(
    model_path, num_labels=1
)

try:
    deberta_model.gradient_checkpointing_enable()
    if hasattr(deberta_model.config, "use_cache"):
        deberta_model.config.use_cache = False
except Exception:
    pass

deberta_model.to(device)

deberta_trainer = Trainer(
    model=deberta_model,
    args=args,
    train_dataset=dataset_split["train"],
    eval_dataset=dataset_split["test"],
    tokenizer=tokenizer_deberta,
    data_collator=data_collator,
    compute_metrics=corr,
)



## === cell 17
training_outcome = deberta_trainer.train()
training_outcome



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3122752704.py in <cell line: 0>()
----> 1 training_outcome = deberta_trainer.train()
      2 training_outcome
      3 

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in train(self, resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
   2204                 hf_hub_utils.enable_progress_bars()
   2205         else:
-> 2206             return inner_training_loop(
   2207                 args=args,
   2208                 resume_from_checkpoint=resume_from_checkpoint,

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in _inner_training_loop(self, batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)
   2546                     )
   2547                     with context():
-> 2548                         tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
   2549 
   2550                     if (

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in training_step(***failed resolving arguments***)
   3795                 kwargs["scale_wrt_gas"] = False
   3796 
-> 3797             self.accelerator.backward(loss, **kwargs)
   3798 
   3799             return loss.detach()

/usr/local/lib/python3.11/dist-packages/accelerate/accelerator.py in backward(self, loss, **kwargs)
   2576             self.lomo_backward(loss, learning_rate)
   2577         else:
-> 2578             loss.backward(**kwargs)
   2579 
   2580     def set_trigger(self):

/usr/local/lib/python3.11/dist-packages/torch/_tensor.py in backward(self, gradient, retain_graph, create_graph, inputs)
    624                 inputs=inputs,
    625             )
--> 626         torch.autograd.backward(
    627             self, gradient, retain_graph, create_graph, inputs=inputs
    628         )

/usr/local/lib/python3.11/dist-packages/torch/autograd/__init__.py in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
    345     # some Python versions print out the first line of a multi-line function
    346     # calls in the traceback and some print out the last line
--> 347     _engine_run_backward(
    348         tensors,
    349         grad_tensors_,

/usr/local/lib/python3.11/dist-packages/torch/autograd/graph.py in _engine_run_backward(t_outputs, *args, **kwargs)
    821         unregister_hooks = _register_logging_hooks_on_whole_graph(t_outputs)
    822     try:
--> 823         return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
    824             t_outputs, *args, **kwargs
    825         )  # Calls into the C++ engine to run the backward pass

/usr/local/lib/python3.11/dist-packages/torch/autograd/function.py in apply(self, *args)
    305             )
    306         user_fn = vjp_fn if vjp_fn is not Function.vjp else backward_fn
--> 307         return user_fn(self, *args)
    308 
    309     def apply_jvp(self, *args):

/usr/local/lib/python3.11/dist-packages/torch/utils/checkpoint.py in backward(ctx, *args)
    319                 " this checkpoint() is not necessary"
    320             )
--> 321         torch.autograd.backward(outputs_with_grad, args_with_grad)
    322         grads = tuple(
    323             inp.grad if isinstance(inp, torch.Tensor) else None

/usr/local/lib/python3.11/dist-packages/torch/autograd/__init__.py in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
    345     # some Python versions print out the first line of a multi-line function
    346     # calls in the traceback and some print out the last line
--> 347     _engine_run_backward(
    348         tensors,
    349         grad_tensors_,

/usr/local/lib/python3.11/dist-packages/torch/autograd/graph.py in _engine_run_backward(t_outputs, *args, **kwargs)
    821         unregister_hooks = _register_logging_hooks_on_whole_graph(t_outputs)
    822     try:
--> 823         return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
    824             t_outputs, *args, **kwargs
    825         )  # Calls into the C++ engine to run the backward pass

RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.

## === cell 18
test_prediction = (
    deberta_trainer.predict(tokenized_test_ds)
    .predictions.astype(np.float32)
    .reshape(-1)
)

test_prediction = np.clip(test_prediction, 0.0, 1.0)
test_prediction[:10], test_prediction.min(), test_prediction.max()



## === cell 19
submission = pd.DataFrame(
    {
        "id": test_ids.values,
        "score": test_prediction,
    }
)

assert submission.shape[0] == len(test_df), "Submission row count mismatch."
assert list(submission.columns) == [
    "id",
    "score",
], "Submission columns must be: id, score"
submission.head(10)



## === cell 20
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
