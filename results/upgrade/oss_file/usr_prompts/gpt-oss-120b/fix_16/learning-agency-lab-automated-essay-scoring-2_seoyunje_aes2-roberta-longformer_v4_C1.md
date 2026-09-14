# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os, re, random, warnings, pathlib
from tqdm import tqdm
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
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

warnings.filterwarnings("ignore")


class CFG:
    SEED = 42
    preset = "allenai/longformer-base-4096"
    MAX_LEN = 1024
    INFERENCE = False  # perform a short fine‑tuning run (1 epoch)
    VER = "v1"
    TRAIN_BATCH = 4
    EVAL_BATCH = 4
    EPOCHS = 1


torch.manual_seed(CFG.SEED)
np.random.seed(CFG.SEED)
random.seed(CFG.SEED)
torch.backends.cudnn.benchmark = True

possible_paths = [
    "data/learning-agency-lab-automated-essay-scoring-2/",
    "kaggle/input/learning-agency-lab-automated-essay-scoring-2/",
    "input/learning-agency-lab-automated-essay-scoring-2/",
    "./learning-agency-lab-automated-essay-scoring-2/",
]
for p in possible_paths:
    if pathlib.Path(p, "train.csv").exists():
        CFG.BASE_PATH = p
        break
else:
    raise FileNotFoundError("train.csv not found in any expected location")




## === cell 1
df_train = pd.read_csv(CFG.BASE_PATH + "train.csv")
df_train = df_train.sort_values(by="essay_id")
df_train["label"] = df_train["score"] - 1

df_test = pd.read_csv(CFG.BASE_PATH + "test.csv")
df_test = df_test.sort_values(by="essay_id")

skf = StratifiedKFold(n_splits=5, random_state=CFG.SEED, shuffle=True)
for i, (_, val_index) in enumerate(skf.split(df_train, df_train["score"])):
    df_train.loc[val_index, "fold"] = i

print("Shape of Train:", df_train.shape)
print("Shape of Test :", df_test.shape)




## === cell 2
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
c_re = re.compile("(%s)" % "|".join(cList.keys()))




## === cell 3
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




## === cell 4
from multiprocessing import Pool, cpu_count


def _process_batch(texts):
    return [dataPreprocessing(t) for t in texts]


def parallel_preprocess(series, n_jobs=None, chunksize=5000):
    """Apply dataPreprocessing to a pandas Series in parallel, using all cores."""
    if n_jobs is None:
        n_jobs = cpu_count()
    with Pool(min(n_jobs, cpu_count())) as pool:
        results = pool.map(
            _process_batch,
            (series[i : i + chunksize] for i in range(0, len(series), chunksize)),
        )
    return [item for sublist in results for item in sublist]


df_train["full_text"] = parallel_preprocess(df_train["full_text"].tolist())
df_test["full_text"] = parallel_preprocess(df_test["full_text"].tolist())




## === cell 5
df_train["label"] = df_train["label"].astype("float32")




## === cell 6
tokenizer = AutoTokenizer.from_pretrained(
    CFG.preset,
    truncation=True,
    max_length=CFG.MAX_LEN,
    clean_up_tokenization_spaces=False,
    use_fast=True,
)
print("Tokenizer loaded:", tokenizer.__class__.__name__)




## === cell 7
def preprocess(samples):
    return tokenizer(samples["full_text"], truncation=True, max_length=CFG.MAX_LEN)


if not CFG.INFERENCE:
    train_split = df_train[df_train["fold"] != 0]
    valid_split = df_train[df_train["fold"] == 0]

    dataset_v = Dataset.from_pandas(valid_split, preserve_index=False)
    dataset_t = Dataset.from_pandas(train_split, preserve_index=False)

    tokenized_dataset_v = dataset_v.map(
        preprocess,
        batched=True,
        batch_size=1000,
        num_proc=cpu_count(),
        remove_columns=["essay_id", "full_text", "score", "fold"],
    )
    tokenized_dataset_t = dataset_t.map(
        preprocess,
        batched=True,
        batch_size=1000,
        num_proc=cpu_count(),
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
        num_proc=cpu_count(),
        remove_columns=["essay_id", "full_text", "score", "fold"],
    )
    tokenized_dataset_train = tokenized_dataset_train.rename_column("label", "labels")




## === cell 8
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
    dataloader_num_workers=cpu_count(),
    remove_unused_columns=False,
    dataloader_pin_memory=True,
)




## === cell 9
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.squeeze()
    qwk = cohen_kappa_score(labels, preds.clip(0, 5).round(), weights="quadratic")
    return {"qwk": qwk}




## === cell 10
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




## === cell 11
if CFG.INFERENCE:
    pass
else:
    trainer.train()




## === cell 12
dataset_test = Dataset.from_pandas(df_test, preserve_index=False)
tokenized_test = dataset_test.map(
    preprocess,
    batched=True,
    batch_size=1000,
    num_proc=cpu_count(),
    remove_columns=["essay_id", "full_text"],
)




## === cell 13
preds_test = trainer.predict(tokenized_test).predictions.squeeze()
preds_test = preds_test + 1  # back to original score range




## === cell 14
sub = pd.DataFrame({"essay_id": df_test["essay_id"].values})
sub["score"] = preds_test.clip(1, 6).round().astype(int)
sub.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv, shape:", sub.shape)
sub.head()
