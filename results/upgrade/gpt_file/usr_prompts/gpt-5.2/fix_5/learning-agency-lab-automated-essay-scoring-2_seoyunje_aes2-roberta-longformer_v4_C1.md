# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the protobuf-related crash by forcing the compatible Python protobuf implementation before importing `transformers/datasets`, and I also disable evaluation/save strategies when running inference-only so `Trainer` can be instantiated without an `eval_dataset`. Then I ensure a `trainer` object always exists by loading the model from `CFG.preset` and using it directly for test-time prediction. Finally, I keep the same regression-to-1..6 rounding logic and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'I fix the protobuf crash that prevents the notebook from running by pinning a compatible `protobuf` major version at runtime and forcing the pure-Python implementation before importing `transformers/datasets`. Then I make sure inference mode creates a usable `trainer` by attaching a minimal `TrainingArguments` and a dummy dataset (so `Trainer.predict()` works reliably across HF versions), without changing your model, preprocessing, or rounding logic. Finally, I ensure the submission is always written as `submission.csv` with the exact required columns and correct row alignment.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the submission rows not matching Kaggle’s expected `essay_id` order/contents (row count mismatch vs sample_submission is a red flag), or with predictions being produced by an effectively “untrained/random” head because the loaded checkpoint isn’t actually being used as intended for regression. I keep your exact model and inference approach, but (1) force the model to run in pure inference mode (`model.eval()` + `torch.no_grad()` via `Trainer` settings), (2) ensure we load a sequence-classification checkpoint robustly even if the preset folder is nested, and (3) write the submission by left-joining onto `sample_submission.csv` so the output has exactly the expected `essay_id` rows and order. These are minimal changes that typically move a broken/invalid-alignment submission from ~0.0 toward a reasonable QWK without changing core modeling logic.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import sys
import subprocess


def _ensure_protobuf_compatible():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
        if major >= 5:
            raise RuntimeError(f"Incompatible protobuf version detected: {pb_ver}")
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        import importlib

        importlib.invalidate_caches()


_ensure_protobuf_compatible()

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
)
from transformers import Trainer, TrainingArguments
from datasets import Dataset

import warnings

warnings.filterwarnings("ignore")

print("Python:", sys.version)
import google.protobuf

print("protobuf:", google.protobuf.__version__)




## === cell 1
class CFG:
    SEED = 2024
    VER = 1
    INFERENCE = True
    BASE_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"

    preset = "/kaggle/input/aes2-longformer/DeBerta_BASE_v1"

    MAX_LEN = 1024
    TRAIN_BATCH = 4
    EVAL_BATCH = 4
    EPOCHS = 4




## === cell 2
Clean = True


def clean_memory():
    if Clean:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
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


seed_everything()




## === cell 4
def resolve_model_path(preset_path: str) -> str:
    if os.path.isdir(preset_path) and os.path.exists(
        os.path.join(preset_path, "config.json")
    ):
        return preset_path

    parent = os.path.dirname(preset_path.rstrip("/"))
    if os.path.isdir(parent):
        for root, dirs, files in os.walk(parent):
            if "config.json" in files:
                return root

    return "allenai/longformer-base-4096"


CFG.preset = resolve_model_path(CFG.preset)
print("Using preset:", CFG.preset)



## === cell 5
df_train = pd.read_csv(CFG.BASE_PATH + "train.csv")
df_train = df_train.sort_values(by="essay_id")
df_train["label"] = df_train["score"] - 1

skf = StratifiedKFold(n_splits=5, random_state=CFG.SEED, shuffle=True)
for i, (_, val_index) in enumerate(skf.split(df_train, df_train["score"])):
    df_train.loc[val_index, "fold"] = i

print("Shape of Train: ", df_train.shape)
print(df_train.head())



## === cell 6
df_test = pd.read_csv(CFG.BASE_PATH + "test.csv")
df_test = df_test.sort_values(by="essay_id")
print("Shape of Test: ", df_test.shape)
print(df_test.head())



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
    "y'all'd": "you all would have",
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
    x = re.sub(r"@\w+", "", x)
    x = re.sub(r"'\d+", "", x)
    x = re.sub(r"\d+", "", x)
    x = re.sub(r"http\w+", "", x)
    x = re.sub(r"\s+", " ", x)
    x = expandContractions(x)
    x = re.sub(r"\.+", ".", x)
    x = re.sub(r"\,+", ",", x)
    x = re.sub(r'[^\w\s.,;:""\'\'?!]', "", x)
    x = x.strip()
    return x




## === cell 9
train = pl.from_pandas(df_train)
test = pl.from_pandas(df_test)

train = train.with_columns(
    pl.col("full_text").map_elements(
        lambda x: dataPreprocessing(x), return_dtype=pl.Utf8
    )
)
df_train = train.to_pandas()

test = test.with_columns(
    pl.col("full_text").map_elements(
        lambda x: dataPreprocessing(x), return_dtype=pl.Utf8
    )
)
df_test = test.to_pandas()



## === cell 10
df_train["label"] = df_train["label"].astype("float32")



## === cell 11
tokenizer = AutoTokenizer.from_pretrained(
    CFG.preset,
    clean_up_tokenization_spaces=False,
    use_fast=True,
)
print(tokenizer.__class__.__name__)




## === cell 12
def preprocess(sample):
    return tokenizer(
        sample["full_text"],
        truncation=True,
        max_length=CFG.MAX_LEN,
    )




## === cell 13
if CFG.INFERENCE is None:
    train_df = df_train[df_train["fold"] != 0]
    valid_df = df_train[df_train["fold"] == 0]

    dataset_v = Dataset.from_pandas(valid_df, preserve_index=False)
    dataset_t = Dataset.from_pandas(train_df, preserve_index=False)

    tokenized_dataset_v = dataset_v.map(
        preprocess,
        num_proc=None,
        remove_columns=["essay_id", "full_text", "score", "fold"],
    )
    tokenized_dataset_t = dataset_t.map(
        preprocess,
        num_proc=None,
        remove_columns=["essay_id", "full_text", "score", "fold"],
    )
else:
    dataset_train = Dataset.from_pandas(df_train, preserve_index=False)
    tokenized_dataset_train = dataset_train.map(
        preprocess,
        num_proc=None,
        remove_columns=["essay_id", "full_text", "score", "fold"],
    )



## === cell 14
if CFG.INFERENCE is None:
    eval_strategy = "epoch"
    save_strategy = "epoch"
    load_best_model_at_end = True
else:
    eval_strategy = "no"
    save_strategy = "no"
    load_best_model_at_end = False

training_args = TrainingArguments(
    output_dir=f"/kaggle/working/output_v{CFG.VER}",
    per_device_train_batch_size=CFG.TRAIN_BATCH,
    per_device_eval_batch_size=CFG.EVAL_BATCH,
    num_train_epochs=CFG.EPOCHS,
    eval_strategy=eval_strategy,
    save_strategy=save_strategy,
    load_best_model_at_end=load_best_model_at_end,
    weight_decay=1e-3,
    fp16=torch.cuda.is_available(),
    learning_rate=1e-5,
    lr_scheduler_type="linear",
    optim="adamw_torch",
    metric_for_best_model="qwk",
    save_total_limit=1,
    report_to="none",
    gradient_checkpointing=True,
    gradient_accumulation_steps=2,
    logging_steps=50,
    do_train=(CFG.INFERENCE is None),
    do_eval=(CFG.INFERENCE is None),
)




## === cell 15
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.asarray(predictions).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    qwk = cohen_kappa_score(
        labels, np.clip(predictions, 0, 5).round(), weights="quadratic"
    )
    return {"qwk": qwk}




## === cell 16
model_config = AutoConfig.from_pretrained(CFG.preset)
model_config.attention_probs_dropout_prob = 0.0
model_config.hidden_dropout_prob = 0.0
model_config.num_labels = 1  # regression

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
        tokenized_dataset_t
        if CFG.INFERENCE is None
        else tokenized_dataset_train.select(range(1))
    ),
    eval_dataset=tokenized_dataset_v if CFG.INFERENCE is None else None,
)

if CFG.INFERENCE is not None:
    trainer.model.eval()

if CFG.INFERENCE is None:
    trainer.train()



## === cell 17
if CFG.INFERENCE is None:
    y_true = valid_df["score"].values
    predictions = trainer.predict(tokenized_dataset_v).predictions
    predictions = np.asarray(predictions).reshape(-1) + 1
    cm = confusion_matrix(
        y_true, np.clip(predictions, 1, 6).round(), labels=[x for x in range(1, 7)]
    )
    draw_cm = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=[x for x in range(1, 7)]
    )
    draw_cm.plot()
    plt.show()

    trainer.save_model(f"/kaggle/working/DeBerta_BASE_v{CFG.VER}")



## === cell 18
dataset_test = Dataset.from_pandas(df_test, preserve_index=False)
tokenized_test = dataset_test.map(
    preprocess,
    num_proc=None,
    remove_columns=["essay_id", "full_text"],
)



## === cell 19
preds_test = trainer.predict(tokenized_test).predictions
preds_test = np.asarray(preds_test).reshape(-1) + 1



## === cell 20
sample_sub = pd.read_csv(CFG.BASE_PATH + "sample_submission.csv")

pred_df = pd.DataFrame({"essay_id": df_test["essay_id"].values})
pred_df["score"] = np.clip(preds_test, 1, 6).round().astype(int)

sub = sample_sub[["essay_id"]].merge(pred_df, on="essay_id", how="left")

sub["score"] = sub["score"].fillna(3).astype(int)

sub.to_csv("submission.csv", index=False)
print("Submission shape", sub.shape)
print(sub.head())
print("Saved to:", os.path.abspath("submission.csv"))
print("Missing scores after merge:", int(sub["score"].isna().sum()))
