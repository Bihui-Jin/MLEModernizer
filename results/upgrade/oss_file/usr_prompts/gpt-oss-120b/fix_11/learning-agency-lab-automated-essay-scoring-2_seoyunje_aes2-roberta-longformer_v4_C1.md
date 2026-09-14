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

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import sys
import gc
import re
import ctypes
import random
from tqdm import tqdm

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

import torch


class CFG:
    BASE_PATH = "./data/learning-agency-lab-automated-essay-scoring-2/"
    SEED = 42
    preset = "allenai/longformer-base-4096"
    MAX_LEN = 1024
    INFERENCE = True  # skip training, only inference
    VER = "v1"
    TRAIN_BATCH = 4
    EVAL_BATCH = 4
    EPOCHS = 1



torch.manual_seed(CFG.SEED)
np.random.seed(CFG.SEED)
random.seed(CFG.SEED)
torch.backends.cudnn.benchmark = True

try:
    import google.protobuf.message_factory as _mf

    if not hasattr(_mf.MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _mf.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

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
df_train = pd.read_csv(CFG.BASE_PATH + "train.csv")
df_train = df_train.sort_values(by="essay_id")
df_train["label"] = df_train["score"] - 1

skf = StratifiedKFold(n_splits=5, random_state=CFG.SEED, shuffle=True)
for i, (_, val_index) in enumerate(skf.split(df_train, df_train["score"])):
    df_train.loc[val_index, "fold"] = i

print("Shape of Train:", df_train.shape)
display(df_train.head())




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3427340841.py in <cell line: 0>()
----> 1 df_train = pd.read_csv(CFG.BASE_PATH + "train.csv")
      2 df_train = df_train.sort_values(by="essay_id")
      3 df_train["label"] = df_train["score"] - 1
      4 
      5 skf = StratifiedKFold(n_splits=5, random_state=CFG.SEED, shuffle=True)

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: './data/learning-agency-lab-automated-essay-scoring-2/train.csv'

## === cell 2
df_test = pd.read_csv(CFG.BASE_PATH + "test.csv")
df_test = df_test.sort_values(by="essay_id")
print("Shape of Test:", df_test.shape)
display(df_test.head())




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1453516165.py in <cell line: 0>()
----> 1 df_test = pd.read_csv(CFG.BASE_PATH + "test.csv")
      2 df_test = df_test.sort_values(by="essay_id")
      3 print("Shape of Test:", df_test.shape)
      4 display(df_test.head())
      5 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: './data/learning-agency-lab-automated-essay-scoring-2/test.csv'

## === cell 3
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
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "so've": "so have",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we had",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "when's": "when is",
    "where'd": "where did",
    "where's": "where is",
    "who's": "who is",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "y'all": "you all",
    "you'd": "you had",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}




## === cell 4
c_re = re.compile("(%s)" % "|".join(cList.keys()))


def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]

    return c_re.sub(replace, text)


def removeHTML(x):
    html = re.compile(r"<.*?>")
    return html.sub(r"", x)


def dataPreprocessing(x):
    x = x.lower()
    x = removeHTML(x)
    x = re.sub("@\w+", "", x)
    x = re.sub("'\d+", "", x)
    x = re.sub("\d+", "", x)
    x = re.sub("http\w+", "", x)
    x = re.sub(r"\s+", " ", x)
    x = expandContractions(x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ",", x)
    x = re.sub(r'[^\w\s.,;:"\'?!]', "", x)
    x = x.strip()
    return x




## === cell 5
from multiprocessing import Pool, cpu_count


def _process_batch(texts):
    return [dataPreprocessing(t) for t in texts]


def parallel_preprocess(series, n_jobs=4, chunksize=1000):
    """Apply dataPreprocessing to a pandas Series in parallel."""
    with Pool(min(n_jobs, cpu_count())) as pool:
        results = pool.map(
            _process_batch,
            (series[i : i + chunksize] for i in range(0, len(series), chunksize)),
        )
    return [item for sublist in results for item in sublist]


df_train["full_text"] = parallel_preprocess(df_train["full_text"].tolist(), n_jobs=4)
df_test["full_text"] = parallel_preprocess(df_test["full_text"].tolist(), n_jobs=4)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/641456935.py in <cell line: 0>()
     16 
     17 
---> 18 df_train["full_text"] = parallel_preprocess(df_train["full_text"].tolist(), n_jobs=4)
     19 df_test["full_text"] = parallel_preprocess(df_test["full_text"].tolist(), n_jobs=4)
     20 

NameError: name 'df_train' is not defined

## === cell 6
df_train["label"] = df_train["label"].astype("float32")




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2259194860.py in <cell line: 0>()
----> 1 df_train["label"] = df_train["label"].astype("float32")
      2 
      3 

NameError: name 'df_train' is not defined

## === cell 7
tokenizer = AutoTokenizer.from_pretrained(
    CFG.preset,
    truncation=True,
    max_length=CFG.MAX_LEN,
    clean_up_tokenization_spaces=False,
)
print("Tokenizer loaded:", tokenizer.__class__.__name__)




## === cell 8
def preprocess(samples):
    return tokenizer(samples["full_text"], truncation=True, max_length=CFG.MAX_LEN)


if CFG.INFERENCE is None:
    train_split = df_train[df_train["fold"] != 0]
    valid_split = df_train[df_train["fold"] == 0]

    dataset_v = Dataset.from_pandas(valid_split, preserve_index=False)
    dataset_t = Dataset.from_pandas(train_split, preserve_index=False)

    tokenized_dataset_v = dataset_v.map(
        preprocess,
        batched=True,
        batch_size=1000,
        num_proc=4,
        remove_columns=["essay_id", "full_text", "score", "fold"],
    )
    tokenized_dataset_t = dataset_t.map(
        preprocess,
        batched=True,
        batch_size=1000,
        num_proc=4,
        remove_columns=["essay_id", "full_text", "score", "fold"],
    )
    tokenized_dataset_v = tokenized_dataset_v.rename_column("label", "labels")
    tokenized_dataset_t = tokenized_dataset_t.rename_column("label", "labels")
else:
    dataset_train = Dataset.from_pandas(df_train, preserve_index=False)
    tokenized_dataset_train = dataset_train.map(
        preprocess,
        batched=True,
        batch_size=1000,
        num_proc=4,
        remove_columns=["essay_id", "full_text", "score", "fold"],
    )
    tokenized_dataset_train = tokenized_dataset_train.rename_column("label", "labels")




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4074390029.py in <cell line: 0>()
     27     tokenized_dataset_t = tokenized_dataset_t.rename_column("label", "labels")
     28 else:
---> 29     dataset_train = Dataset.from_pandas(df_train, preserve_index=False)
     30     tokenized_dataset_train = dataset_train.map(
     31         preprocess,

NameError: name 'df_train' is not defined

## === cell 9
training_args = TrainingArguments(
    output_dir=f"./output_v{CFG.VER}",
    per_device_train_batch_size=CFG.TRAIN_BATCH,
    per_device_eval_batch_size=CFG.EVAL_BATCH,
    num_train_epochs=CFG.EPOCHS,
    load_best_model_at_end=False,
    weight_decay=1e-3,
    fp16=True,
    learning_rate=1e-5,
    lr_scheduler_type="linear",
    optim="adamw_torch",
    metric_for_best_model="qwk",
    save_total_limit=1,
    report_to="none",
    gradient_checkpointing=True,
    gradient_accumulation_steps=1,
    eval_strategy="no",
    save_strategy="steps",
    dataloader_num_workers=4,
    remove_unused_columns=False,
)




## === cell 10
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.squeeze()
    qwk = cohen_kappa_score(labels, preds.clip(0, 5).round(), weights="quadratic")
    return {"qwk": qwk}




## === cell 11
model_config = AutoConfig.from_pretrained(CFG.preset)
model_config.attention_probs_dropout_prob = 0.0
model_config.hidden_dropout_prob = 0.0
model_config.num_labels = 1
model = LongformerForSequenceClassification.from_pretrained(
    CFG.preset, config=model_config
)

model = torch.compile(model, mode="reduce-overhead")

if CFG.INFERENCE:
    train_ds = tokenized_dataset_train
    eval_ds = None
else:
    train_ds = tokenized_dataset_t
    eval_ds = tokenized_dataset_v

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/712922666.py in <cell line: 0>()
     12 # Choose appropriate datasets based on mode
     13 if CFG.INFERENCE:
---> 14     train_ds = tokenized_dataset_train
     15     eval_ds = None
     16 else:

NameError: name 'tokenized_dataset_train' is not defined

## === cell 12
if CFG.INFERENCE is None:
    trainer.train()
else:
    pass  # inference mode – skip training




## === cell 13
if CFG.INFERENCE is None:
    y_true = valid_split["score"].values
    predictions = trainer.predict(tokenized_dataset_v).predictions.squeeze()
    predictions = predictions + 1  # shift back to 1‑6 scale
    cm = confusion_matrix(
        y_true, predictions.clip(1, 6).round(), labels=[x for x in range(1, 7)]
    )
    draw_cm = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=[x for x in range(1, 7)]
    )
    draw_cm.plot()
    plt.show()
    trainer.save_model(f"Longformer_BASE_v{CFG.VER}")
else:
    pass




## === cell 14
dataset_test = Dataset.from_pandas(df_test, preserve_index=False)
tokenized_test = dataset_test.map(
    preprocess,
    batched=True,
    batch_size=1000,
    num_proc=4,
    remove_columns=["essay_id", "full_text"],
)




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1584137430.py in <cell line: 0>()
----> 1 dataset_test = Dataset.from_pandas(df_test, preserve_index=False)
      2 tokenized_test = dataset_test.map(
      3     preprocess,
      4     batched=True,
      5     batch_size=1000,

NameError: name 'df_test' is not defined

## === cell 15
preds_test = trainer.predict(tokenized_test).predictions.squeeze()
preds_test = preds_test + 1  # back to original score range




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/534905621.py in <cell line: 0>()
----> 1 preds_test = trainer.predict(tokenized_test).predictions.squeeze()
      2 preds_test = preds_test + 1  # back to original score range
      3 
      4 

NameError: name 'trainer' is not defined

## === cell 16
sub = pd.DataFrame({"essay_id": df_test["essay_id"].values})
sub["score"] = preds_test.clip(1, 6).round().astype(int)
sub.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv, shape:", sub.shape)
sub.head()

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4007873774.py in <cell line: 0>()
----> 1 sub = pd.DataFrame({"essay_id": df_test["essay_id"].values})
      2 sub["score"] = preds_test.clip(1, 6).round().astype(int)
      3 sub.to_csv("submission.csv", index=False)
      4 print("Submission saved to submission.csv, shape:", sub.shape)
      5 sub.head()

NameError: name 'df_test' is not defined
