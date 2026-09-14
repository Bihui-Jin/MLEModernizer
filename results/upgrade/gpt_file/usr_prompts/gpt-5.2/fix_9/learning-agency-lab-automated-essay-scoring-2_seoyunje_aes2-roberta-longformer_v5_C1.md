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

cudf-polars-cu12==25.6.0
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
polars==1.25.0
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

0.7968195210060658

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import sys
import gc
import re
import ctypes
import random
from tqdm import tqdm
import polars as pl

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

import torch

from transformers import (
    AutoTokenizer,
    LongformerForSequenceClassification,
    DataCollatorWithPadding,
    AutoConfig,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import warnings

warnings.filterwarnings("ignore")




## === cell 1
class CFG:
    SEED = 2024
    VER = 1
    INFERENCE = True
    BASE_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"

    _LOCAL_PRESET_CANDIDATES = [
        "/kaggle/input/aes2-longformer/DeBerta_BASE_v1",  # original (often missing)
        "/kaggle/input/aes2-longformer",  # common alternative
        "/kaggle/input/longformer",  # generic dataset name
        "/kaggle/input/longformer-base-4096",  # common model dataset name
    ]
    preset = None  # set below

    MAX_LEN = 1024
    TRAIN_BATCH = 4
    EVAL_BATCH = 4
    EPOCHS = 4


for p in CFG._LOCAL_PRESET_CANDIDATES:
    if os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json")):
        CFG.preset = p
        break
if CFG.preset is None:
    CFG.preset = "allenai/longformer-base-4096"

print("Using preset:", CFG.preset)



## === cell 2
Clean = True


def clean_memory():
    if Clean:
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        gc.collect()


clean_memory()




## === cell 3
def seed_everything():
    random.seed(CFG.SEED)
    os.environ["PYTHONHASHSEED"] = str(CFG.SEED)
    np.random.seed(CFG.SEED)
    torch.manual_seed(CFG.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CFG.SEED)
        torch.cuda.manual_seed_all(CFG.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.use_deterministic_algorithms(
            False
        )  # preserve original semantics (not strictly forcing)
    except Exception:
        pass


seed_everything()




## === cell 4
def _maybe_display(x, n=5):
    try:
        from IPython.display import display

        return display(x)
    except Exception:
        print(x.head(n))




## === cell 5
df_train = pd.read_csv(CFG.BASE_PATH + "train.csv")
df_train = df_train.sort_values(by="essay_id")
df_train["label"] = df_train["score"] - 1

skf = StratifiedKFold(n_splits=5, random_state=CFG.SEED, shuffle=True)
for i, (_, val_index) in enumerate(skf.split(df_train, df_train["score"])):
    df_train.loc[val_index, "fold"] = i

print("Shape of Train: ", df_train.shape)
_maybe_display(df_train)



## === cell 6
df_test = pd.read_csv(CFG.BASE_PATH + "test.csv")
df_test = df_test.sort_values(by="essay_id")
print("Shape of Test: ", df_test.shape)
_maybe_display(df_test)



## === cell 7
cList = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'alls": "you alls",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you you will",
    "you'll've": "you you will have",
    "you're": "you are",
    "you've": "you have",
}



## === cell 8
_c_re = re.compile("(%s)" % "|".join(map(re.escape, cList.keys())))
_html_re = re.compile(r"<.*?>")
_re_at = re.compile(r"@\w+")
_re_apost_num = re.compile(r"'\d+")
_re_num = re.compile(r"\d+")
_re_http = re.compile(r"http\w+")
_re_space = re.compile(r"\s+")
_re_dots = re.compile(r"\.+")
_re_commas = re.compile(r"\,+")
_re_keep = re.compile(r'[^\w\s.,;:""\'\'?!]')


def expandContractions(text, c_re=_c_re, _cList=cList):
    def replace(match):
        return _cList[match.group(0)]

    return c_re.sub(replace, text)


def removeHTML(x, _html_re=_html_re):
    return _html_re.sub(r"", x)


def dataPreprocessing(
    x,
    _removeHTML=removeHTML,
    _re_at=_re_at,
    _re_apost_num=_re_apost_num,
    _re_num=_re_num,
    _re_http=_re_http,
    _re_space=_re_space,
    _expand=expandContractions,
    _re_dots=_re_dots,
    _re_commas=_re_commas,
    _re_keep=_re_keep,
):
    x = x.lower()
    x = _removeHTML(x)
    x = _re_at.sub("", x)
    x = _re_apost_num.sub("", x)
    x = _re_num.sub("", x)
    x = _re_http.sub("", x)
    x = _re_space.sub(" ", x)
    x = _expand(x)
    x = _re_dots.sub(".", x)
    x = _re_commas.sub(",", x)
    x = _re_keep.sub("", x)
    x = x.strip()
    return x




## === cell 9
_NUM_PROC_TEXT = max(1, min(4, os.cpu_count() or 2))


def _clean_batch(batch):
    return {"full_text": [dataPreprocessing(str(t)) for t in batch["full_text"]]}


_ds_train_clean = Dataset.from_pandas(
    df_train[["essay_id", "full_text", "score", "label", "fold"]], preserve_index=False
)
_ds_train_clean = _ds_train_clean.map(
    _clean_batch,
    batched=True,
    batch_size=256,
    num_proc=_NUM_PROC_TEXT,
    load_from_cache_file=False,
    desc="Cleaning train text",
)
df_train["full_text"] = pd.Series(_ds_train_clean["full_text"])

_ds_test_clean = Dataset.from_pandas(
    df_test[["essay_id", "full_text"]], preserve_index=False
)
_ds_test_clean = _ds_test_clean.map(
    _clean_batch,
    batched=True,
    batch_size=256,
    num_proc=_NUM_PROC_TEXT,
    load_from_cache_file=False,
    desc="Cleaning test text",
)
df_test["full_text"] = pd.Series(_ds_test_clean["full_text"])



## === cell 10
df_train["label"] = df_train["label"].astype("float32")



## === cell 11
tokenizer = AutoTokenizer.from_pretrained(
    CFG.preset,
    truncation=True,
    max_length=CFG.MAX_LEN,
    clean_up_tokenization_spaces=False,
    use_fast=True,
)
print(tokenizer.__class__.__name__)




## === cell 12
def preprocess_batch(batch):
    toks = tokenizer(batch["full_text"], truncation=True, max_length=CFG.MAX_LEN)
    toks["length"] = [len(x) for x in toks["input_ids"]]
    return toks




## === cell 13
_NUM_PROC = max(1, min(4, os.cpu_count() or 2))
_MAP_KW = dict(
    batched=True,
    batch_size=1024,
    num_proc=_NUM_PROC,
    load_from_cache_file=True,
    keep_in_memory=True,
    desc="Tokenizing",
)

if CFG.INFERENCE is None:
    train_df = df_train[df_train["fold"] != 0]
    valid_df = df_train[df_train["fold"] == 0]

    dataset_v = Dataset.from_pandas(valid_df, preserve_index=False)
    dataset_t = Dataset.from_pandas(train_df, preserve_index=False)
    tokenized_dataset_v = dataset_v.map(
        preprocess_batch,
        remove_columns=["essay_id", "full_text", "score", "fold"],
        **_MAP_KW,
    )
    tokenized_dataset_t = dataset_t.map(
        preprocess_batch,
        remove_columns=["essay_id", "full_text", "score", "fold"],
        **_MAP_KW,
    )
else:
    dataset_train = Dataset.from_pandas(df_train, preserve_index=False)
    tokenized_dataset_train = dataset_train.map(
        preprocess_batch,
        remove_columns=["essay_id", "full_text", "score", "fold"],
        **_MAP_KW,
    )



## === cell 14
if CFG.INFERENCE is None:
    _eval_strategy = "epoch"
    _save_strategy = "epoch"
    _load_best = True
    _metric_for_best_model = "qwk"
else:
    _eval_strategy = "no"
    _save_strategy = "no"
    _load_best = False
    _metric_for_best_model = None

_group_by_length = True
_length_col = "length"
try:
    _probe_ds = (
        tokenized_dataset_train if CFG.INFERENCE is not None else tokenized_dataset_t
    )
    if _length_col not in _probe_ds.column_names:
        _group_by_length = False
        _length_col = None
except Exception:
    _group_by_length = False
    _length_col = None

training_args = TrainingArguments(
    output_dir=f"/kaggle/working/output_v{CFG.VER}",
    per_device_train_batch_size=CFG.TRAIN_BATCH,
    per_device_eval_batch_size=CFG.EVAL_BATCH,
    num_train_epochs=CFG.EPOCHS,
    eval_strategy=_eval_strategy,
    save_strategy=_save_strategy,
    load_best_model_at_end=_load_best,
    weight_decay=1e-3,
    fp16=bool(torch.cuda.is_available()),
    learning_rate=1e-5,
    lr_scheduler_type="linear",
    optim="adamw_torch",
    metric_for_best_model=_metric_for_best_model,
    save_total_limit=1,
    report_to="none",
    gradient_checkpointing=True,
    gradient_accumulation_steps=2,
    remove_unused_columns=False,
    dataloader_num_workers=4,
    dataloader_pin_memory=bool(torch.cuda.is_available()),
    group_by_length=_group_by_length,
    length_column_name=_length_col,
)

print("group_by_length:", _group_by_length, "| length_column_name:", _length_col)




## === cell 15
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.asarray(predictions).reshape(-1)
    qwk = cohen_kappa_score(labels, preds.clip(0, 5).round(), weights="quadratic")
    return {"qwk": qwk}




## === cell 16
model_config = AutoConfig.from_pretrained(CFG.preset)
model_config.attention_probs_dropout_prob = 0.0
model_config.hidden_dropout_prob = 0.0
model_config.num_labels = 1  # regression head

model = LongformerForSequenceClassification.from_pretrained(
    CFG.preset, config=model_config
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    train_dataset=(
        tokenized_dataset_t if CFG.INFERENCE is None else tokenized_dataset_train
    ),
    eval_dataset=(tokenized_dataset_v if CFG.INFERENCE is None else None),
)

saved_dir = f"/kaggle/working/DeBerta_BASE_v{CFG.VER}"
if (
    CFG.INFERENCE is not None
    and os.path.isdir(saved_dir)
    and os.path.exists(os.path.join(saved_dir, "config.json"))
):
    print(f"Loading saved model from: {saved_dir}")
    model = LongformerForSequenceClassification.from_pretrained(saved_dir)
    trainer.model = model
else:
    print("Training model (required to produce predictions)...")
    trainer.train()
    try:
        trainer.save_model(saved_dir)
        print(f"Saved model to: {saved_dir}")
    except Exception as e:
        print("Warning: could not save model:", repr(e))



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_54/3334574550.py in <cell line: 0>()
     31 else:
     32     print("Training model (required to produce predictions)...")
---> 33     trainer.train()
     34     try:
     35         trainer.save_model(saved_dir)

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

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in training_step(self, model, inputs, num_items_in_batch)
   3747 
   3748         with self.compute_loss_context_manager():
-> 3749             loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
   3750 
   3751         del inputs

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in compute_loss(self, model, inputs, return_outputs, num_items_in_batch)
   3834                 loss_kwargs["num_items_in_batch"] = num_items_in_batch
   3835             inputs = {**inputs, **loss_kwargs}
-> 3836         outputs = model(**inputs)
   3837         # Save past state if it exists
   3838         # TODO: this needs to be fixed and made cleaner later.

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/accelerate/utils/operations.py in forward(*args, **kwargs)
    816 
    817     def forward(*args, **kwargs):
--> 818         return model_forward(*args, **kwargs)
    819 
    820     # To act like a decorator so that it can be popped when doing `extract_model_from_parallel`

/usr/local/lib/python3.11/dist-packages/accelerate/utils/operations.py in __call__(self, *args, **kwargs)
    804 
    805     def __call__(self, *args, **kwargs):
--> 806         return convert_to_fp32(self.model_forward(*args, **kwargs))
    807 
    808     def __getstate__(self):

/usr/local/lib/python3.11/dist-packages/torch/amp/autocast_mode.py in decorate_autocast(*args, **kwargs)
     42     def decorate_autocast(*args, **kwargs):
     43         with autocast_instance:
---> 44             return func(*args, **kwargs)
     45 
     46     decorate_autocast.__script_unsupported = "@autocast() decorator is not supported in script mode"  # type: ignore[attr-defined]

TypeError: LongformerForSequenceClassification.forward() got an unexpected keyword argument 'length'

## === cell 17
if CFG.INFERENCE is None:
    y_true = valid_df["score"].values
    predictions = trainer.predict(tokenized_dataset_v).predictions
    predictions = np.asarray(predictions).reshape(-1) + 1
    cm = confusion_matrix(
        y_true, predictions.clip(1, 6).round(), labels=[x for x in range(1, 7)]
    )
    draw_cm = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=[x for x in range(1, 7)]
    )
    draw_cm.plot()
    plt.show()



## === cell 18
dataset_test = Dataset.from_pandas(df_test, preserve_index=False)
tokenized_test = dataset_test.map(
    preprocess_batch,
    remove_columns=["essay_id", "full_text"],
    **{**_MAP_KW, "desc": "Tokenizing test"},
)



## === cell 19
preds_test = trainer.predict(tokenized_test).predictions
preds_test = np.asarray(preds_test).reshape(-1) + 1



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_54/2747570959.py in <cell line: 0>()
----> 1 preds_test = trainer.predict(tokenized_test).predictions
      2 preds_test = np.asarray(preds_test).reshape(-1) + 1
      3 

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in predict(self, test_dataset, ignore_keys, metric_key_prefix)
   4275 
   4276         eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
-> 4277         output = eval_loop(
   4278             test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
   4279         )

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in evaluation_loop(self, dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
   4392 
   4393             # Prediction step
-> 4394             losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
   4395             main_input_name = getattr(self.model, "main_input_name", "input_ids")
   4396             inputs_decode = (

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in prediction_step(self, model, inputs, prediction_loss_only, ignore_keys)
   4618                     loss = None
   4619                     with self.compute_loss_context_manager():
-> 4620                         outputs = model(**inputs)
   4621                     if isinstance(outputs, dict):
   4622                         logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/accelerate/utils/operations.py in forward(*args, **kwargs)
    816 
    817     def forward(*args, **kwargs):
--> 818         return model_forward(*args, **kwargs)
    819 
    820     # To act like a decorator so that it can be popped when doing `extract_model_from_parallel`

/usr/local/lib/python3.11/dist-packages/accelerate/utils/operations.py in __call__(self, *args, **kwargs)
    804 
    805     def __call__(self, *args, **kwargs):
--> 806         return convert_to_fp32(self.model_forward(*args, **kwargs))
    807 
    808     def __getstate__(self):

/usr/local/lib/python3.11/dist-packages/torch/amp/autocast_mode.py in decorate_autocast(*args, **kwargs)
     42     def decorate_autocast(*args, **kwargs):
     43         with autocast_instance:
---> 44             return func(*args, **kwargs)
     45 
     46     decorate_autocast.__script_unsupported = "@autocast() decorator is not supported in script mode"  # type: ignore[attr-defined]

TypeError: LongformerForSequenceClassification.forward() got an unexpected keyword argument 'length'

## === cell 20
sub = pd.DataFrame({"essay_id": df_test.essay_id.values})
sub["score"] = np.clip(np.rint(preds_test), 1, 6).astype(int)
sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
print(sub.head())
print("Wrote submission.csv")

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/912509651.py in <cell line: 0>()
      1 sub = pd.DataFrame({"essay_id": df_test.essay_id.values})
----> 2 sub["score"] = np.clip(np.rint(preds_test), 1, 6).astype(int)
      3 sub.to_csv("submission.csv", index=False)
      4 print("Submission shape", sub.shape)
      5 print(sub.head())

NameError: name 'preds_test' is not defined
