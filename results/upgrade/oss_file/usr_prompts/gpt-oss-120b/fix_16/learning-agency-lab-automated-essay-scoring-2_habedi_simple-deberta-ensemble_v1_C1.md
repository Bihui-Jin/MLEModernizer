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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I adjust the Trainer configuration so that inference (validation and test) does not require an evaluation dataset. After creating `training_args`, I set `eval_strategy` to `"no"` when the script is not performing full training (`PERFORM_TRAINING=False`). This prevents the ValueError while keeping the original training logic unchanged.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
import os
import random
import warnings
from pathlib import Path

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    from google.protobuf import message_factory as _message_factory

    if not hasattr(_message_factory.MessageFactory, "GetPrototype"):

        def _get_prototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _get_prototype
except Exception:
    pass

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
PERFORM_TRAINING = True

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
    if IS_KAGGLE:
        COMPUTE_CV = True
        LOAD_FROM = Path("/kaggle/input/model-collection-1/model_1")
    else:
        COMPUTE_CV = True
        LOAD_FROM = PATHS.output_dir  # / f'{BASE_MODEL_NAME}'


class CFG:
    n_splits = 3
    seed = 42
    max_length = 1024
    lr = 1e-5
    train_batch_size = 64
    eval_batch_size = 32
    train_epochs = 4
    weight_decay = 0.01
    warmup_ratio = 0.0
    num_labels = 6


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def compute_metrics_for_regression(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.squeeze()
    qwk = cohen_kappa_score(labels, preds.clip(0, 5).round(0), weights="quadratic")
    return {"qwk": qwk}


def compute_metrics_for_classification(eval_pred):
    predictions, labels = eval_pred
    qwk = cohen_kappa_score(labels, predictions.argmax(-1), weights="quadratic")
    return {"qwk": qwk}


seed_everything(CFG.seed)

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

NUM_PROC = os.cpu_count() or 8  # fallback to 8 if cpu_count() returns None

training_args = TrainingArguments(
    output_dir=f"tmp/{BASE_MODEL_NAME}",
    fp16=True,
    learning_rate=CFG.lr,
    per_device_train_batch_size=CFG.train_batch_size,
    per_device_eval_batch_size=CFG.eval_batch_size,
    gradient_accumulation_steps=1,  # removed accumulation
    num_train_epochs=CFG.train_epochs,
    weight_decay=CFG.weight_decay,
    eval_strategy="no",  # no validation during training
    save_strategy="no",  # skip checkpoint writes each epoch
    logging_strategy="no",  # suppress step‑level logging
    metric_for_best_model="qwk",
    load_best_model_at_end=False,
    report_to="none",
    warmup_ratio=CFG.warmup_ratio,
    lr_scheduler_type="linear",
    optim="adamw_torch",
    dataloader_num_workers=NUM_PROC,  # parallel data loading
    dataloader_pin_memory=True,
)

if not PERFORM_TRAINING:
    training_args.eval_strategy = "no"

data["fold"] = data["fold"].astype(int)


def get_pretrained_path(fold):
    """
    Return a path usable by `from_pretrained`. Prefer a locally fine‑tuned checkpoint
    if it exists; otherwise fall back to the original Hugging Face model identifier.
    """
    if LOAD_FROM is not None:
        candidate = LOAD_FROM / f"fold_{fold}"
        if candidate.exists():
            return str(candidate)
    return PATHS.model_path


base_tokenizer = AutoTokenizer.from_pretrained(PATHS.model_path)
base_tokenizer.add_tokens([AddedToken("\n", normalized=False)])
base_tokenizer.add_tokens([AddedToken(" " * 2, normalized=False)])


def tokenize_function(example):
    return base_tokenizer(
        example["full_text"], truncation=True, max_length=CFG.max_length
    )


full_hf_dataset = Dataset.from_dict(
    {
        "essay_id": data["essay_id"].tolist(),
        "full_text": data["full_text"].tolist(),
        "label": data["label"].tolist(),
    }
)

tokenized_full = full_hf_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    num_proc=NUM_PROC,
)

test_df = pd.read_csv(PATHS.test_path)
test_hf_dataset = Dataset.from_dict(
    {
        "essay_id": test_df["essay_id"].tolist(),
        "full_text": test_df["full_text"].tolist(),
        "label": [0.0] * len(test_df),  # placeholder
    }
)
tokenized_test = test_hf_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    num_proc=NUM_PROC,
)


trained_models = []

for fold in range(len(data["fold"].unique())):
    train_idx = data[data["fold"] != fold].index.tolist()
    valid_idx = data[data["fold"] == fold].index.tolist()
    tokenized_train = tokenized_full.select(train_idx)
    tokenized_valid = tokenized_full.select(valid_idx)

    tokenizer = base_tokenizer  # reuse the already‑prepared tokenizer

    config = AutoConfig.from_pretrained(get_pretrained_path(fold))
    if USE_REGRESSION:
        config.attention_probs_dropout_prob = 0.0
        config.hidden_dropout_prob = 0.0
        config.num_labels = 1
        config.problem_type = "regression"
    else:
        config.num_labels = CFG.num_labels

    print(f"Loading model from {get_pretrained_path(fold)}")
    model = AutoModelForSequenceClassification.from_pretrained(
        get_pretrained_path(fold), config=config
    )
    model.resize_token_embeddings(len(tokenizer))

    model = torch.compile(model, mode="max-autotune")

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
        eval_dataset=tokenized_valid,  # required field but not used because eval_strategy="no"
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    if PERFORM_TRAINING:
        trainer.train()
        trained_models.append(trainer.model)

    y_true = data.loc[valid_idx, "score"].values
    predictions0 = trainer.predict(tokenized_valid).predictions

    valid_df = data.loc[valid_idx].copy()
    if USE_REGRESSION:
        valid_df["pred"] = predictions0.squeeze() + 1
    else:
        COLS = [f"p{x}" for x in range(CFG.num_labels)]
        valid_df[COLS] = predictions0

    valid_df.to_csv(f"valid_df_fold_{fold}.csv", index=False)


if COMPUTE_CV:
    dfs = []

    for k in range(CFG.n_splits):
        dfs.append(pd.read_csv(f"valid_df_fold_{k}.csv"))
        os.system(f"rm valid_df_fold_{k}.csv")

    dfs = pd.concat(dfs)
    columns = ["essay_id", "score", "label", "fold", "pred"]
    dfs[columns].to_csv("valid_df_m1.csv", index=False)

    print("Valid OOF shape:", dfs.shape)
    print("-" * 50)
    print(dfs[columns].head())

    if USE_REGRESSION:
        print("-" * 50)

        for k in range(CFG.n_splits):
            m = cohen_kappa_score(
                dfs[dfs.fold == k].score.values,
                dfs[dfs.fold == k].pred.values.clip(1, 6).round(0),
                weights="quadratic",
            )
            print(f"Fold {k} QWK =", round(m, 5))

        print("-" * 50)

        m = cohen_kappa_score(
            dfs.score.values, dfs.pred.values.clip(1, 6).round(0), weights="quadratic"
        )

        print("-" * 50)
    else:
        m = cohen_kappa_score(
            dfs.score.values,
            dfs.iloc[:, -6:].values.argmax(axis=1) + 1,
            weights="quadratic",
        )

    print("Overall QWK CV =", round(m, 5))

print("Test shape:", test_df.shape)
test_df.head()

all_pred = []

for fold in range(CFG.n_splits):
    tokenizer = base_tokenizer  # reuse the same tokenizer

    if PERFORM_TRAINING and len(trained_models) == CFG.n_splits:
        model = trained_models[fold]
    else:
        config = AutoConfig.from_pretrained(get_pretrained_path(fold))
        if USE_REGRESSION:
            config.num_labels = 1
            config.problem_type = "regression"
        else:
            config.num_labels = CFG.num_labels
        model = AutoModelForSequenceClassification.from_pretrained(
            get_pretrained_path(fold), config=config
        )
        model.resize_token_embeddings(len(tokenizer))
        model = torch.compile(model, mode="max-autotune")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_test,  # dummy placeholder required by Trainer
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    predictions = trainer.predict(tokenized_test).predictions
    all_pred.append(predictions)

preds = np.mean(all_pred, axis=0)
print("Predictions shape:", preds.shape)

sub = pd.read_csv(PATHS.sub_path)

if USE_REGRESSION:
    sub["score"] = preds.squeeze().clip(0, 5).round(0) + 1
else:
    sub["score"] = preds.argmax(axis=1) + 1

sub.score = sub.score.astype("int32")
sub.to_csv("submission.csv", index=False)
print("Submission shape:", sub.shape)
sub.head()

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_56/4003817591.py in <cell line: 0>()
    264 
    265     if PERFORM_TRAINING:
--> 266         trainer.train()
    267         trained_models.append(trainer.model)
    268 

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in train(self, resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
   2204                 hf_hub_utils.enable_progress_bars()
   2205         else:
-> 2206             return inner_training_loop(
   2207                 args=args,
   2208                 resume_from_checkpoint=resume_from_checkpoint,

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in _inner_training_loop(self, batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)
   2500                 update_step += 1
   2501                 num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
-> 2502                 batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
   2503                 for i, inputs in enumerate(batch_samples):
   2504                     step += 1

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in get_batch_samples(self, epoch_iterator, num_batches, device)
   5298         for _ in range(num_batches):
   5299             try:
-> 5300                 batch_samples.append(next(epoch_iterator))
   5301             except StopIteration:
   5302                 break

/usr/local/lib/python3.11/dist-packages/accelerate/data_loader.py in __iter__(self)
    565         # We iterate one batch ahead to check when we are at the end
    566         try:
--> 567             current_batch = next(dataloader_iter)
    568         except StopIteration:
    569             self.end()

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

ValueError: Caught ValueError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 55, in fetch
    return self.collate_fn(data)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/data/data_collator.py", line 272, in __call__
    batch = pad_without_fast_tokenizer_warning(
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/data/data_collator.py", line 67, in pad_without_fast_tokenizer_warning
    padded = tokenizer.pad(*pad_args, **pad_kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py", line 3291, in pad
    raise ValueError(
ValueError: You should supply an encoding or a list of encodings to this method that includes input_ids, but you provided ['label']
