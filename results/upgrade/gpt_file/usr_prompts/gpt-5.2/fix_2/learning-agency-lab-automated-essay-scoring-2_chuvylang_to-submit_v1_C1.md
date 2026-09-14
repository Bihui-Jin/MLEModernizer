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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
tokenizers==0.21.2
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

0.7993434084034929

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I fix the environment-breaking import error by forcing the pure-Python protobuf implementation before importing `transformers/datasets`, which avoids the `MessageFactory.GetPrototype` crash. Then I make model loading robust by detecting when the provided `/kaggle/input/...` checkpoint path doesn’t exist in this notebook environment and falling back to a public base model with the same `AutoModelForSequenceClassification` core logic, so inference can run end-to-end. I update `TrainingArguments` to the current Transformers API (`eval_strategy` instead of `evaluation_strategy`) to remove the init error. Finally, I prevent NaNs/infs from reaching the submission by handling prediction shapes correctly (regression vs logits) and sanitizing non-finite values before casting to int, ensuring a valid `submission.csv` is written.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

LOAD_FROM = (
    "/kaggle/input/vuxvuxregression/transformers/fuckyou/1/output_v1/checkpoint-27692"
)



## === cell 1
import warnings
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding

from datasets import Dataset

warnings.simplefilter("ignore")




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
class PATHS:
    train_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
    test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
    sub_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
    model_path = "/kaggle/input/vuxvuxregression/transformers/fuckyou/1/output_v1/checkpoint-27692"


USE_REGRESSION = True


def pick_model_path(
    preferred_path: str, fallback_repo: str = "microsoft/deberta-v3-small"
) -> str:
    if preferred_path and os.path.isdir(preferred_path):
        return preferred_path
    if LOAD_FROM and os.path.isdir(LOAD_FROM):
        return LOAD_FROM
    return fallback_repo


MODEL_PATH = pick_model_path(PATHS.model_path)

print("Using MODEL_PATH:", MODEL_PATH)




## === cell 3
class Tokenize(object):
    def __init__(self, train, valid, tokenizer):
        self.tokenizer = tokenizer
        self.train = train
        self.valid = valid

    def get_dataset(self, df):
        ds = Dataset.from_dict(
            {
                "essay_id": [e for e in df["essay_id"]],
                "full_text": [ft for ft in df["full_text"]],
                "label": [s for s in df["label"]],
            }
        )
        return ds

    def tokenize_function(self, example):
        tokenized_inputs = self.tokenizer(
            example["full_text"], truncation=True, max_length=CFG.max_length
        )
        return tokenized_inputs

    def __call__(self):
        train_ds = self.get_dataset(self.train)
        valid_ds = self.get_dataset(self.valid)

        tokenized_train = train_ds.map(self.tokenize_function, batched=True)
        tokenized_valid = valid_ds.map(self.tokenize_function, batched=True)

        return tokenized_train, tokenized_valid, self.tokenizer




## === cell 4
class CFG:
    n_splits = 5
    seed = 42
    max_length = 4096
    lr = 1e-5
    train_batch_size = 2
    eval_batch_size = 2
    train_epochs = 30
    weight_decay = 0.01
    warmup_ratio = 0.0
    num_labels = 6




## === cell 5
training_args = TrainingArguments(
    output_dir="output_v",
    fp16=True,
    learning_rate=CFG.lr,
    per_device_train_batch_size=CFG.train_batch_size,
    per_device_eval_batch_size=CFG.eval_batch_size,
    num_train_epochs=CFG.train_epochs,
    weight_decay=CFG.weight_decay,
    eval_strategy="epoch",
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



## === cell 6
test = pd.read_csv(PATHS.test_path)
print("Test shape:", test.shape)
test.head()



## === cell 7
test = test.copy()
test["label"] = 0.0

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

if USE_REGRESSION:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=1,
        problem_type="regression",
        ignore_mismatched_sizes=True,
    )
else:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=CFG.num_labels,
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True,
    )



## === cell 8
all_pred = []

tokenize = Tokenize(test, test, tokenizer)
tokenized_test, _, _ = tokenize()

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

for fold in range(CFG.n_splits):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_test,  # kept as in original code (not used for training here)
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    predictions = trainer.predict(tokenized_test).predictions
    all_pred.append(predictions)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1810043215.py in <cell line: 0>()
     10 # predictions will be repeated but still valid and deterministic.
     11 for fold in range(CFG.n_splits):
---> 12     trainer = Trainer(
     13         model=model,
     14         args=training_args,

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

## === cell 9
preds = np.mean(all_pred, axis=0)
print("Predictions shape:", preds.shape)

if USE_REGRESSION and preds.ndim == 2 and preds.shape[1] == 1:
    preds = preds[:, 0]



## === cell 10
sub = pd.read_csv(PATHS.sub_path)

preds = np.asarray(preds)
preds = np.nan_to_num(preds, nan=0.0, posinf=5.0, neginf=0.0)

if USE_REGRESSION:
    sub["score"] = (np.clip(preds, 0, 5).round(0) + 1).astype(np.int32)
else:
    if preds.ndim == 1:
        cls = np.clip(np.round(preds).astype(int), 0, 5)
    else:
        cls = np.argmax(preds, axis=1)
    sub["score"] = (cls + 1).astype(np.int32)

sub = sub[["essay_id", "score"]].copy()
sub["score"] = sub["score"].clip(1, 6).astype(np.int32)

sub.to_csv("submission.csv", index=False)
print("Submission shape:", sub.shape)
sub.head()
