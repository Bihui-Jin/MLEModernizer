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
scipy==1.15.3
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

0.8030895862887903

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the environment/import crash in the first cell by forcing the pure-Python protobuf implementation (this avoids the `MessageFactory.GetPrototype` error that happens with some package combinations). Then I fix model/tokenizer loading so a local Kaggle dataset path like `/kaggle/input/...` is treated as a filesystem directory (not a Hub repo id), which resolves the `HFValidationError`. Finally, I make the inference-to-submission step correct for a sequence-classification model by converting logits to class predictions (1–6) rather than clipping raw logits; this preserves the intended evaluation semantics and produces a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I (1) fix the protobuf-related crash by setting the env var before any protobuf/transformers-related imports and forcing a safe protobuf runtime choice, (2) make the model path robust by automatically resolving your intended local model directory under `/kaggle/input` (and fall back to a simple baseline if it truly doesn’t exist, so a valid submission is always produced), and (3) keep inference semantics correct by converting sequence-classification logits to class labels 1–6. These changes are minimal and directly address the current failures that prevent any valid submission (hence score 0.0). The fallback baseline is only used if the model files are unavailable in the runtime; otherwise your original model inference is used unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import random
import numpy as np
import pandas as pd

import torch
from datasets import Dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
MODEL_NAME = "/kaggle/input/f-tfm-small/funnel-small-ft_ver4"
MAX_LENGTH = 3072
BATCH_SIZE = 2
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)



## === cell 2
df_train = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
)
df_test = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)
sample_submission = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)

assert {"essay_id", "full_text", "score"}.issubset(df_train.columns)
assert {"essay_id", "full_text"}.issubset(df_test.columns)
assert {"essay_id", "score"}.issubset(sample_submission.columns)




## === cell 3
def resolve_model_dir(model_path: str) -> str | None:
    """
    Fix: MODEL_NAME may point to a non-existent directory in this environment.
    Try common Kaggle input locations; return a valid directory or None.
    """
    candidates = []

    candidates.append(model_path)

    if not model_path.startswith("/kaggle/input/"):
        candidates.append(os.path.join("/kaggle/input", model_path))

    leaf = os.path.basename(model_path.rstrip("/"))
    if leaf:
        for base in ["/kaggle/input", "/kaggle/data", "/kaggle/working"]:
            if os.path.isdir(base):
                for root, dirs, _files in os.walk(base):
                    if leaf in dirs:
                        candidates.append(os.path.join(root, leaf))

    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    for c in uniq:
        if os.path.isdir(c):
            return c
    return None


model_dir = resolve_model_dir(MODEL_NAME)

use_fallback_baseline = model_dir is None
if use_fallback_baseline:
    print(f"WARNING: Could not find MODEL_NAME directory: {MODEL_NAME}")
    print("WARNING: Falling back to a simple baseline to produce a valid submission.")
else:
    print(f"Using local model directory: {model_dir}")



## === cell 4
if not use_fallback_baseline:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True
    )

    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    def tokenize(batch):
        return tokenizer(batch["full_text"], max_length=MAX_LENGTH, truncation=True)

    test_ds = (
        Dataset.from_pandas(df_test[["essay_id", "full_text"]])
        .map(tokenize, batched=True)
        .remove_columns(["essay_id", "full_text"])
    )



## === cell 5
preds = None

if not use_fallback_baseline:
    predictions = []

    args = TrainingArguments(
        output_dir=".",
        per_device_eval_batch_size=BATCH_SIZE,
        report_to="none",
        dataloader_drop_last=False,
        seed=SEED,
    )

    trainer = Trainer(
        args=args,
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    logits = trainer.predict(test_ds).predictions  # (n_examples, n_labels)
    predictions.append(logits)

    preds = np.mean(predictions, axis=0)



## === cell 6
if use_fallback_baseline:
    mean_score = float(df_train["score"].mean())
    const_pred = int(np.clip(np.rint(mean_score), 1, 6))
    df_test["score"] = const_pred
else:
    if preds is None:
        raise RuntimeError(
            "Internal error: preds is None despite model inference path."
        )
    if preds.ndim == 2:
        pred_labels = np.argmax(preds, axis=1)  # 0..num_labels-1
        df_test["score"] = (pred_labels + 1).astype(int)
    else:
        df_test["score"] = (np.clip(preds, 0, 5).round(0) + 1).astype(int)



## === cell 7
submission = df_test[["essay_id", "score"]].copy()
submission = sample_submission[["essay_id"]].merge(
    submission, on="essay_id", how="left"
)

if submission["score"].isna().any():
    missing = submission.loc[submission["score"].isna(), "essay_id"].head(5).tolist()
    raise ValueError(
        f"Some test essay_ids are missing predictions after merge; examples: {missing}"
    )

submission["score"] = submission["score"].astype(int)
submission["score"] = submission["score"].clip(1, 6)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print(f"Wrote submission.csv with shape: {submission.shape}")
print(f"Score value counts:\n{submission['score'].value_counts().sort_index()}")
