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
import os, sys, gc, re, ctypes, random

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from tqdm import tqdm
import polars as pl
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score
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




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class CFG:
    SEED = 2024
    VER = 1
    INFERENCE = False  # set to False so training is performed
    BASE_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/"
    preset = "allenai/longformer-base-4096"
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
    torch.cuda.manual_seed(CFG.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything()



## === cell 4
df_train = pd.read_csv(CFG.BASE_PATH + "train.csv")
df_train = df_train.sort_values("essay_id")
df_train["label"] = df_train["score"] - 1

skf = StratifiedKFold(n_splits=5, random_state=CFG.SEED, shuffle=True)
for i, (_, val_idx) in enumerate(skf.split(df_train, df_train["score"])):
    df_train.loc[val_idx, "fold"] = i

df_test = pd.read_csv(CFG.BASE_PATH + "test.csv")
df_test = df_test.sort_values("essay_id")

print("Train shape:", df_train.shape)
print("Test shape :", df_test.shape)



## === cell 5
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
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "when's": "when is",
    "where's": "where is",
    "who's": "who is",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "you'd": "you had",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}
c_re = re.compile("(%s)" % "|".join(cList.keys()))


def expandContractions(text, c_re=c_re):
    return c_re.sub(lambda m: cList[m.group(0)], text)


def removeHTML(x):
    return re.sub(r"<.*?>", "", x)


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
    x = re.sub(r'[^\w\s.,;:"""' "?!]", "", x)
    return x.strip()




## === cell 6
train_pl = pl.from_pandas(df_train)
test_pl = pl.from_pandas(df_test)

train_pl = train_pl.with_columns(
    pl.col("full_text").map_elements(lambda x: dataPreprocessing(x))
)
test_pl = test_pl.with_columns(
    pl.col("full_text").map_elements(lambda x: dataPreprocessing(x))
)

df_train = train_pl.to_pandas()
df_test = test_pl.to_pandas()
df_train["label"] = df_train["label"].astype("float32")



## === cell 7
tokenizer = AutoTokenizer.from_pretrained(
    CFG.preset,
    truncation=True,
    max_length=CFG.MAX_LEN,
    clean_up_tokenization_spaces=False,
)
print("Tokenizer loaded:", tokenizer.__class__.__name__)


def preprocess(sample):
    return tokenizer(sample["full_text"], truncation=True, max_length=CFG.MAX_LEN)




## === cell 8
if not CFG.INFERENCE:
    train_df = df_train[df_train["fold"] != 0].reset_index(drop=True)
    valid_df = df_train[df_train["fold"] == 0].reset_index(drop=True)

    dataset_train = Dataset.from_pandas(train_df)
    dataset_valid = Dataset.from_pandas(valid_df)

    tokenized_dataset_train = dataset_train.map(
        preprocess,
        num_proc=2,
        remove_columns=["essay_id", "full_text", "score", "fold"],
    )
    tokenized_dataset_valid = dataset_valid.map(
        preprocess,
        num_proc=2,
        remove_columns=["essay_id", "full_text", "score", "fold"],
    )
else:
    dataset_train = Dataset.from_pandas(df_train)
    tokenized_dataset_train = dataset_train.map(
        preprocess,
        num_proc=2,
        remove_columns=["essay_id", "full_text", "score", "fold"],
    )



## === cell 9
training_args = TrainingArguments(
    output_dir=f"/output_v{CFG.VER}",
    per_device_train_batch_size=CFG.TRAIN_BATCH,
    per_device_eval_batch_size=CFG.EVAL_BATCH,
    num_train_epochs=CFG.EPOCHS,
    eval_strategy="epoch",  # correct keyword
    save_strategy="epoch",
    load_best_model_at_end=True,
    weight_decay=1e-3,
    fp16=True,
    learning_rate=1e-5,
    lr_scheduler_type="linear",
    optim="adamw_torch",
    metric_for_best_model="qwk",
    save_total_limit=1,
    report_to="none",
    gradient_checkpointing=True,
    gradient_accumulation_steps=2,
)




## === cell 10
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.clip(0, 5).round()
    qwk = cohen_kappa_score(labels, preds, weights="quadratic")
    return {"qwk": qwk}




## === cell 11
model_config = AutoConfig.from_pretrained(CFG.preset)
model_config.attention_probs_dropout_prob = 0.0
model_config.hidden_dropout_prob = 0.0
model_config.num_labels = 1  # regression output

model = LongformerForSequenceClassification.from_pretrained(
    CFG.preset, config=model_config
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3385645427.py in <cell line: 0>()
      8 )
      9 
---> 10 trainer = Trainer(
     11     model=model,
     12     args=training_args,

/usr/local/lib/python3.11/dist-packages/transformers/utils/deprecation.py in wrapped_func(*args, **kwargs)
    170                 warnings.warn(message, FutureWarning, stacklevel=2)
    171 
--> 172             return func(*args, **kwargs)
    173 
    174         return wrapped_func

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in __init__(self, model, args, data_collator, train_dataset, eval_dataset, processing_class, model_init, compute_loss_func, compute_metrics, callbacks, optimizers, optimizer_cls_and_kwargs, preprocess_logits_for_metrics)
    445                 )
    446         if args.eval_strategy is not None and args.eval_strategy != "no" and eval_dataset is None:
--> 447             raise ValueError(
    448                 f"You have set `args.eval_strategy` to {args.eval_strategy} but you didn't pass an `eval_dataset` to `Trainer`. Either set `args.eval_strategy` to `no` or pass an `eval_dataset`. "
    449             )

ValueError: You have set `args.eval_strategy` to IntervalStrategy.EPOCH but you didn't pass an `eval_dataset` to `Trainer`. Either set `args.eval_strategy` to `no` or pass an `eval_dataset`. 

## === cell 12
if not CFG.INFERENCE:
    trainer.train()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1666196812.py in <cell line: 0>()
      1 if not CFG.INFERENCE:
----> 2     trainer.train()
      3 

NameError: name 'trainer' is not defined

## === cell 13
if not CFG.INFERENCE:
    y_true = valid_df["score"].values
    preds = trainer.predict(tokenized_dataset_valid).predictions.squeeze()
    preds = preds + 1  # shift back to 1‑6 scale
    cm = confusion_matrix(y_true, preds.clip(1, 6).round(), labels=range(1, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(1, 7))
    disp.plot()
    plt.show()
    trainer.save_model(f"DeBerta_BASE_v{CFG.VER}")
else:
    pass



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4259543807.py in <cell line: 0>()
      2     # evaluation on validation split
      3     y_true = valid_df["score"].values
----> 4     preds = trainer.predict(tokenized_dataset_valid).predictions.squeeze()
      5     preds = preds + 1  # shift back to 1‑6 scale
      6     cm = confusion_matrix(y_true, preds.clip(1, 6).round(), labels=range(1, 7))

NameError: name 'trainer' is not defined

## === cell 14
dataset_test = Dataset.from_pandas(df_test)
tokenized_test = dataset_test.map(
    preprocess, num_proc=3, remove_columns=["essay_id", "full_text"]
)

test_preds = trainer.predict(tokenized_test).predictions.squeeze()
test_preds = test_preds + 1  # convert regression output to original score range
sub = pd.DataFrame(
    {
        "essay_id": df_test["essay_id"],
        "score": test_preds.clip(1, 6).round().astype(int),
    }
)
sub.to_csv("submission.csv", index=False)
print("Submission saved, shape:", sub.shape)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2451638612.py in <cell line: 0>()
      5 )
      6 
----> 7 test_preds = trainer.predict(tokenized_test).predictions.squeeze()
      8 test_preds = test_preds + 1  # convert regression output to original score range
      9 sub = pd.DataFrame(

NameError: name 'trainer' is not defined
