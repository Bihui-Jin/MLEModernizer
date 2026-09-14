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
)

from transformers import Trainer, TrainingArguments
from datasets import Dataset
import warnings

warnings.filterwarnings("ignore")


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1613509688.py in <cell line: 0>()
     23 import torch
     24 
---> 25 torch.manual_seed(CFG.SEED)
     26 np.random.seed(CFG.SEED)
     27 random.seed(CFG.SEED)

NameError: name 'CFG' is not defined

## === cell 1
df_train = pd.read_csv(CFG.BASE_PATH + "train.csv")
df_train = df_train.sort_values(by="essay_id")
df_train["label"] = df_train["score"] - 1

skf = StratifiedKFold(n_splits=5, random_state=CFG.SEED, shuffle=True)
for i, (_, val_index) in enumerate(skf.split(df_train, df_train["score"])):
    df_train.loc[val_index, "fold"] = i

print("Shape of Train: ", df_train.shape)
print(display(df_train.head()))


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3130578913.py in <cell line: 0>()
----> 1 df_train = pd.read_csv(CFG.BASE_PATH + "train.csv")
      2 df_train = df_train.sort_values(by="essay_id")
      3 df_train["label"] = df_train["score"] - 1
      4 
      5 skf = StratifiedKFold(n_splits=5, random_state=CFG.SEED, shuffle=True)

NameError: name 'CFG' is not defined

## === cell 2
df_test = pd.read_csv(CFG.BASE_PATH + "test.csv")
df_test = df_test.sort_values(by="essay_id")
print("Shape of Test: ", df_test.shape)
print(display(df_test.head()))


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2768748615.py in <cell line: 0>()
----> 1 df_test = pd.read_csv(CFG.BASE_PATH + "test.csv")
      2 df_test = df_test.sort_values(by="essay_id")
      3 print("Shape of Test: ", df_test.shape)
      4 print(display(df_test.head()))

NameError: name 'CFG' is not defined

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
    "you'll": "you will",
    "you'll've": "you will have",
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
    return html.sub(r"", x)  # html -> ''


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
/tmp/ipykernel_55/2720102397.py in <cell line: 0>()
     16 
     17 
---> 18 df_train["full_text"] = parallel_preprocess(df_train["full_text"].tolist(), n_jobs=4)
     19 df_test["full_text"] = parallel_preprocess(df_test["full_text"].tolist(), n_jobs=4)

NameError: name 'df_train' is not defined

## === cell 6
df_train["label"] = df_train["label"].astype("float32")


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1572196487.py in <cell line: 0>()
----> 1 df_train["label"] = df_train["label"].astype("float32")

NameError: name 'df_train' is not defined

## === cell 7
tokenizer = AutoTokenizer.from_pretrained(
    CFG.preset,
    truncation=True,
    max_length=CFG.MAX_LEN,
    clean_up_tokenization_spaces=False,
)
print(tokenizer)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3857871888.py in <cell line: 0>()
----> 1 tokenizer = AutoTokenizer.from_pretrained(
      2     CFG.preset,
      3     truncation=True,
      4     max_length=CFG.MAX_LEN,
      5     clean_up_tokenization_spaces=False,

NameError: name 'AutoTokenizer' is not defined

## === cell 8
def preprocess(samples):
    return tokenizer(samples["full_text"], truncation=True, max_length=CFG.MAX_LEN)




## === cell 9
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


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/292682174.py in <cell line: 0>()
----> 1 if CFG.INFERENCE is None:
      2     train_split = df_train[df_train["fold"] != 0]
      3     valid_split = df_train[df_train["fold"] == 0]
      4 
      5     # Preserve_index=False drops the pandas index column that later causes '__index_level_0__' errors

NameError: name 'CFG' is not defined

## === cell 10
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




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/730014036.py in <cell line: 0>()
----> 1 training_args = TrainingArguments(
      2     output_dir=f"./output_v{CFG.VER}",
      3     per_device_train_batch_size=CFG.TRAIN_BATCH,
      4     per_device_eval_batch_size=CFG.EVAL_BATCH,
      5     num_train_epochs=CFG.EPOCHS,

NameError: name 'TrainingArguments' is not defined

## === cell 11
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.squeeze()
    qwk = cohen_kappa_score(labels, preds.clip(0, 5).round(), weights="quadratic")
    return {"qwk": qwk}




## === cell 12
model_config = AutoConfig.from_pretrained(CFG.preset)
model_config.attention_probs_dropout_prob = 0.0
model_config.hidden_dropout_prob = 0.0
model_config.num_labels = 1
model = LongformerForSequenceClassification.from_pretrained(
    CFG.preset, config=model_config
)

model = torch.compile(model, mode="reduce-overhead")

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    train_dataset=tokenized_dataset_t,
    eval_dataset=tokenized_dataset_v,
)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4151750709.py in <cell line: 0>()
----> 1 model_config = AutoConfig.from_pretrained(CFG.preset)
      2 model_config.attention_probs_dropout_prob = 0.0
      3 model_config.hidden_dropout_prob = 0.0
      4 model_config.num_labels = 1
      5 model = LongformerForSequenceClassification.from_pretrained(

NameError: name 'AutoConfig' is not defined

## === cell 13
if CFG.INFERENCE is None:
    trainer.train()
else:
    pass


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2737322282.py in <cell line: 0>()
----> 1 if CFG.INFERENCE is None:
      2     trainer.train()
      3 else:
      4     pass

NameError: name 'CFG' is not defined

## === cell 14
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


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1535241188.py in <cell line: 0>()
----> 1 if CFG.INFERENCE is None:
      2     y_true = valid_split["score"].values
      3     predictions = trainer.predict(tokenized_dataset_v).predictions.squeeze()
      4     predictions = predictions + 1  # shift back to 1‑6 scale
      5     cm = confusion_matrix(

NameError: name 'CFG' is not defined

## === cell 15
dataset_test = Dataset.from_pandas(df_test, preserve_index=False)
tokenized_test = dataset_test.map(
    preprocess,
    batched=True,
    batch_size=1000,
    num_proc=4,
    remove_columns=["essay_id", "full_text"],
)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1298983977.py in <cell line: 0>()
----> 1 dataset_test = Dataset.from_pandas(df_test, preserve_index=False)
      2 tokenized_test = dataset_test.map(
      3     preprocess,
      4     batched=True,
      5     batch_size=1000,

NameError: name 'Dataset' is not defined

## === cell 16
preds_test = trainer.predict(tokenized_test).predictions.squeeze()
preds_test = preds_test + 1  # back to original score range


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1529708742.py in <cell line: 0>()
----> 1 preds_test = trainer.predict(tokenized_test).predictions.squeeze()
      2 preds_test = preds_test + 1  # back to original score range

NameError: name 'trainer' is not defined

## === cell 17
sub = pd.DataFrame({"essay_id": df_test["essay_id"].values})
sub["score"] = preds_test.clip(1, 6).round().astype(int)
sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
sub.head()

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2214556691.py in <cell line: 0>()
----> 1 sub = pd.DataFrame({"essay_id": df_test["essay_id"].values})
      2 sub["score"] = preds_test.clip(1, 6).round().astype(int)
      3 sub.to_csv("submission.csv", index=False)
      4 print("Submission shape", sub.shape)
      5 sub.head()

NameError: name 'df_test' is not defined
