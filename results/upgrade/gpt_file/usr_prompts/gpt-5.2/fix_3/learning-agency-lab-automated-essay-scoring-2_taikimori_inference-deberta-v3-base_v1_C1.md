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

datasets==4.4.1
geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
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
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import numpy as np
import pandas as pd

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
)



## === cell 1
MAX_LENGTH = 512


def tokenize(example, tokenizer, max_length=None):
    return tokenizer(
        example["full_text"],
        padding=False,
        truncation=True,
        max_length=max_length,
    )


def build_dataset(data_path, tokenizer, is_train: bool):
    df = pd.read_csv(data_path)
    ds = Dataset.from_pandas(df, preserve_index=False)

    if is_train and "score" in ds.column_names:
        ds = ds.rename_column("score", "labels")
        ds = ds.map(lambda x: {"labels": int(x["labels"]) - 1})

    remove_cols = ["full_text"]
    ds = ds.map(
        tokenize,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": MAX_LENGTH},
        remove_columns=remove_cols,
    )
    return ds


def _find_local_model_dir(base_dir: str) -> str:
    """
    If a Kaggle dataset with a HF model exists under /kaggle/input, locate a folder containing config.json.
    """
    if not base_dir or not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base dir does not exist: {base_dir}")

    candidates = [
        base_dir,
        os.path.join(base_dir, "transformers", "v1", "1"),
        os.path.join(base_dir, "transformers", "default", "1"),
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "config.json")):
            return c

    for root, _, files in os.walk(base_dir):
        if "config.json" in files:
            return root

    raise FileNotFoundError(
        f"Could not find a local HuggingFace model directory with config.json under: {base_dir}"
    )


def run_inference():
    test_data_path = (
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
    )
    train_data_path = (
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
    )

    model_base_dir = "/kaggle/input/aes-baseline-deberta-v3-base"

    fallback_model_id = "microsoft/deberta-v3-base"  # standard checkpoint if local model isn't available
    num_labels = 6

    local_model_dir = None
    if os.path.isdir(model_base_dir):
        try:
            local_model_dir = _find_local_model_dir(model_base_dir)
        except FileNotFoundError:
            local_model_dir = None

    if local_model_dir is not None:
        model_source = local_model_dir
        tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_source, local_files_only=True
        )
    else:
        model_source = fallback_model_id
        tokenizer = AutoTokenizer.from_pretrained(model_source)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            num_labels=num_labels,
        )

        train_ds = build_dataset(train_data_path, tokenizer, is_train=True)

        collator = DataCollatorWithPadding(tokenizer)
        args = TrainingArguments(
            output_dir="./model_out",
            per_device_train_batch_size=8,
            learning_rate=2e-5,
            num_train_epochs=1,
            weight_decay=0.01,
            logging_steps=50,
            save_strategy="no",
            report_to="none",
            fp16=False,  # Py3.12 CPU-only environments often don't support fp16 well
        )
        trainer = Trainer(
            model=model,
            args=args,
            data_collator=collator,
            tokenizer=tokenizer,
            train_dataset=train_ds,
        )
        trainer.train()

    collator = DataCollatorWithPadding(tokenizer)
    test_ds = build_dataset(test_data_path, tokenizer, is_train=False)

    args = TrainingArguments(
        output_dir=".",
        per_device_eval_batch_size=16,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    logits = trainer.predict(test_ds).predictions

    if logits.ndim == 2 and logits.shape[1] > 1:
        pred_class = np.argmax(logits, axis=1)
        preds = pred_class + 1
    else:
        preds = np.rint(np.squeeze(logits)).astype(int)

    preds = np.clip(preds, 1, 6).astype(int)

    submission_df = pd.DataFrame({"essay_id": test_ds["essay_id"], "score": preds})
    submission_df.to_csv("submission.csv", index=False)
    return submission_df




## === cell 2
run_inference()
