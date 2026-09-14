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

datasets==4.4.1
geopandas==0.14.4
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
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
tokenizers==0.21.2
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
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

0.81698905205311

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the runtime error by ensuring that inference `Trainer` instances do not require an `eval_dataset` (set `eval_strategy="no"` only for inference), while keeping the original training/CV `TrainingArguments` unchanged. I also make sure we always create `preds` so the submission cell can run, and keep the prediction post-processing identical to your current logic (regression + clip/round +1). Finally, I remove the unsupported pip install step (it can fail/offline) since it’s not needed for this pipeline in Kaggle, improving stability without changing model behavior.'
- What this solution (achieved 0.0) has done: 'The crash happens before any modeling code runs: `transformers` imports protobuf internals, and with the Kaggle image’s protobuf version this triggers `MessageFactory.GetPrototype` errors. I fix this by forcing the pure-Python protobuf implementation *before* importing `transformers/datasets`, which is a standard compatibility workaround and is score-neutral. I also make the inference `Trainer` use `eval_dataset` (not `train_dataset`) to match expected semantics and avoid edge-case Trainer behaviors, without changing the prediction logic. Finally, I add a small fallback so a valid `submission.csv` is always written even if fold checkpoints are missing.'

# 9. Code solution

## === cell 0
import os
import sys
import random
import warnings
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from tokenizers import AddedToken
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

warnings.simplefilter("ignore")

IS_KAGGLE = True
IS_DEV = False

PERFORM_TRAINING = False

if IS_KAGGLE:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    base_data_dir = Path("/kaggle/input/learning-agency-lab-automated-essay-scoring-2")

    class PATHS:
        train_path = base_data_dir / "train.csv"
        test_path = base_data_dir / "test.csv"
        sub_path = base_data_dir / "sample_submission.csv"
        model_path = "microsoft/deberta-v3-xsmall"
        output_dir = Path(".")

else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    base_data_dir = Path("../data/competition_data")

    class PATHS:
        train_path = base_data_dir / "train.csv"
        test_path = base_data_dir / "test.csv"
        sub_path = base_data_dir / "sample_submission.csv"
        model_path = "microsoft/deberta-v3-xsmall"
        output_dir = Path("for_stacking/model_1")


BASE_MODEL_NAME = PATHS.model_path.split("/")[-1]
USE_REGRESSION = True

if PERFORM_TRAINING:
    COMPUTE_CV = True
    LOAD_FROM = None
else:
    COMPUTE_CV = True
    candidate = (
        Path("/kaggle/input/model-collection-1/model_1")
        if IS_KAGGLE
        else PATHS.output_dir
    )
    LOAD_FROM = candidate if candidate.exists() else None


class CFG:
    n_splits = 5
    seed = 42
    max_length = 1024
    lr = 1e-5
    train_batch_size = 4
    eval_batch_size = 8
    train_epochs = 4
    weight_decay = 0.01
    warmup_ratio = 0.0
    num_labels = 6


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(CFG.seed)


class Tokenize(object):
    def __init__(self, train, valid, tokenizer):
        self.tokenizer = tokenizer
        self.train = train
        self.valid = valid

    def get_dataset(self, df):
        return Dataset.from_dict(
            {
                "essay_id": [e for e in df["essay_id"]],
                "full_text": [ft for ft in df["full_text"]],
                "label": [s for s in df["label"]],
            }
        )

    def tokenize_function(self, example):
        return self.tokenizer(
            example["full_text"], truncation=True, max_length=CFG.max_length
        )

    def __call__(self):
        train_ds = self.get_dataset(self.train)
        valid_ds = self.get_dataset(self.valid)
        tokenized_train = train_ds.map(self.tokenize_function, batched=True)
        tokenized_valid = valid_ds.map(self.tokenize_function, batched=True)
        return tokenized_train, tokenized_valid, self.tokenizer


def compute_metrics_for_regression(eval_pred):
    predictions, labels = eval_pred
    predictions = np.asarray(predictions).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    qwk = cohen_kappa_score(
        labels, predictions.clip(0, 5).round(0), weights="quadratic"
    )
    return {"qwk": qwk}


def compute_metrics_for_classification(eval_pred):
    predictions, labels = eval_pred
    qwk = cohen_kappa_score(labels, predictions.argmax(-1), weights="quadratic")
    return {"qwk": qwk}




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
if IS_DEV:
    nrows = 2000
else:
    nrows = None

data = pd.read_csv(PATHS.train_path, nrows=nrows)
data["label"] = data["score"].apply(lambda x: x - 1)

if USE_REGRESSION:
    data["label"] = data["label"].astype("float32")
else:
    data["label"] = data["label"].astype("int32")

skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
for i, (_, val_index) in enumerate(skf.split(data, data["score"])):
    data.loc[val_index, "fold"] = i
data["fold"] = data["fold"].astype(int)

training_args = TrainingArguments(
    output_dir=f"tmp/{BASE_MODEL_NAME}",
    fp16=True,
    learning_rate=CFG.lr,
    per_device_train_batch_size=CFG.train_batch_size,
    per_device_eval_batch_size=CFG.eval_batch_size,
    num_train_epochs=CFG.train_epochs,
    weight_decay=CFG.weight_decay,
    eval_strategy="epoch",  # requires eval_dataset; OK for CV block
    metric_for_best_model="qwk",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    report_to="none",
    warmup_ratio=CFG.warmup_ratio,
    lr_scheduler_type="linear",
    optim="adamw_torch",
    logging_first_step=True,
)

inference_args = TrainingArguments(
    output_dir=f"tmp/{BASE_MODEL_NAME}_infer",
    fp16=True,
    per_device_eval_batch_size=CFG.eval_batch_size,
    eval_strategy="no",
    report_to="none",
    dataloader_drop_last=False,
)



## === cell 2
if LOAD_FROM is not None:
    for fold in range(len(data["fold"].unique())):
        train = data[data["fold"] != fold]
        valid = data[data["fold"] == fold].copy()

        tokenizer = AutoTokenizer.from_pretrained(LOAD_FROM / f"fold_{fold}")
        tokenizer.add_tokens([AddedToken("\n", normalized=False)])
        tokenizer.add_tokens([AddedToken(" " * 2, normalized=False)])

        tokenize = Tokenize(train, valid, tokenizer)
        tokenized_train, tokenized_valid, _ = tokenize()

        config = AutoConfig.from_pretrained(LOAD_FROM / f"fold_{fold}")
        if USE_REGRESSION:
            config.attention_probs_dropout_prob = 0.0
            config.hidden_dropout_prob = 0.0
            config.num_labels = 1
        else:
            config.num_labels = CFG.num_labels

        print(f'Loading model from {LOAD_FROM / f"fold_{fold}"}')
        model = AutoModelForSequenceClassification.from_pretrained(
            LOAD_FROM / f"fold_{fold}"
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        compute_metrics = (
            compute_metrics_for_regression
            if USE_REGRESSION
            else compute_metrics_for_classification
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_valid,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        predictions0 = trainer.predict(tokenized_valid).predictions

        if USE_REGRESSION:
            valid["pred"] = np.asarray(predictions0).reshape(-1) + 1
        else:
            COLS = [f"p{x}" for x in range(CFG.num_labels)]
            valid[COLS] = predictions0

        valid.to_csv(f"valid_df_fold_{fold}.csv", index=False)

    if COMPUTE_CV:
        dfs = []
        for k in range(CFG.n_splits):
            dfs.append(pd.read_csv(f"valid_df_fold_{k}.csv"))
            os.system(f"rm valid_df_fold_{k}.csv")

        dfs = pd.concat(dfs, ignore_index=True)
        columns = ["essay_id", "score", "label", "fold", "pred"]
        dfs[columns].to_csv("valid_df_m1.csv", index=False)

        print("Valid OOF shape:", dfs.shape)
        if USE_REGRESSION:
            for k in range(CFG.n_splits):
                m = cohen_kappa_score(
                    dfs[dfs.fold == k].score.values,
                    dfs[dfs.fold == k].pred.values.clip(1, 6).round(0),
                    weights="quadratic",
                )
                print(f"Fold {k} QWK =", round(m, 5))
            m = cohen_kappa_score(
                dfs.score.values,
                dfs.pred.values.clip(1, 6).round(0),
                weights="quadratic",
            )
        else:
            m = cohen_kappa_score(
                dfs.score.values,
                dfs.iloc[:, -6:].values.argmax(axis=1) + 1,
                weights="quadratic",
            )

        print("Overall QWK CV =", round(m, 5))
else:
    print(
        "Pretrained fold checkpoints not found. Skipping CV/OFF evaluation and running direct inference from base model."
    )



## === cell 3
test = pd.read_csv(PATHS.test_path)
print("Test shape:", test.shape)

all_pred = []
test["label"] = 0.0

for fold in range(CFG.n_splits):
    if LOAD_FROM is None:
        tokenizer = AutoTokenizer.from_pretrained(PATHS.model_path)
        tokenize = Tokenize(test, test, tokenizer)
        tokenized_test, _, _ = tokenize()

        config = AutoConfig.from_pretrained(PATHS.model_path)
        if USE_REGRESSION:
            config.num_labels = 1
        else:
            config.num_labels = CFG.num_labels

        model = AutoModelForSequenceClassification.from_pretrained(
            PATHS.model_path, config=config
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        trainer = Trainer(
            model=model,
            args=inference_args,
            eval_dataset=tokenized_test,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        predictions = trainer.predict(tokenized_test).predictions
        all_pred.append(predictions)
        break

    tokenizer = AutoTokenizer.from_pretrained(LOAD_FROM / f"fold_{fold}")
    tokenize = Tokenize(test, test, tokenizer)
    tokenized_test, _, _ = tokenize()

    model = AutoModelForSequenceClassification.from_pretrained(
        LOAD_FROM / f"fold_{fold}"
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=inference_args,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    predictions = trainer.predict(tokenized_test).predictions
    all_pred.append(predictions)

preds = (
    np.mean(all_pred, axis=0)
    if len(all_pred) > 0
    else np.zeros((len(test), 1), dtype=np.float32)
)
print("Predictions shape:", np.asarray(preds).shape)



## === cell 4
sub = pd.read_csv(PATHS.sub_path)

if USE_REGRESSION:
    preds_1d = np.asarray(preds).reshape(-1)
    sub["score"] = preds_1d.clip(0, 5).round(0) + 1
else:
    sub["score"] = np.asarray(preds).argmax(axis=1) + 1

sub["score"] = sub["score"].astype("int32")
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv")
print("Submission shape:", sub.shape)
print(sub.head())
