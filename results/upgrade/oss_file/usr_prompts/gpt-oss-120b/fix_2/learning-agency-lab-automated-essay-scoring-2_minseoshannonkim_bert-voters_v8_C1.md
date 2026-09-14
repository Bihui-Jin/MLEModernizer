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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.7811435150492567

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

data_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
train_path = os.path.join(data_path, "train.csv")
test_path = os.path.join(data_path, "test.csv")
output_dir = "./model_output"  # where the fine‑tuned model will be saved
os.makedirs(output_dir, exist_ok=True)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
raw_datasets = load_dataset("csv", data_files={"train": train_path, "test": test_path})

raw_datasets["train"] = raw_datasets["train"].rename_columns(
    {"full_text": "text", "score": "labels"}
)
raw_datasets["test"] = raw_datasets["test"].rename_columns({"full_text": "text"})


def shift_labels(example):
    example["labels"] = example["labels"] - 1
    return example


raw_datasets["train"] = raw_datasets["train"].map(shift_labels)



## === cell 2
model_name = "bert-base-uncased"  # use the public HuggingFace model
tokenizer = AutoTokenizer.from_pretrained(model_name)


def preprocess_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding=False)


tokenized_datasets = raw_datasets.map(
    preprocess_fn, batched=True, remove_columns=["text"]
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



## === cell 3
train_indices, val_indices = train_test_split(
    list(range(len(tokenized_datasets["train"]))),
    test_size=0.2,
    stratify=tokenized_datasets["train"]["labels"],
    random_state=42,
)

train_dataset = tokenized_datasets["train"].select(train_indices)
val_dataset = tokenized_datasets["train"].select(val_indices)



## === cell 4
id2label = {i: str(i + 1) for i in range(6)}  # model will output strings "1"‑"6"
label2id = {str(i + 1): i for i in range(6)}

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=6, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=os.path.join(output_dir, "logs"),
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3421932512.py in <cell line: 0>()
      7 )
      8 
----> 9 training_args = TrainingArguments(
     10     output_dir=output_dir,
     11     evaluation_strategy="epoch",

TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'

## === cell 5
trainer.train()

trainer.save_model(output_dir)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2362577056.py in <cell line: 0>()
      1 # Train the model
----> 2 trainer.train()
      3 
      4 # Save the final model
      5 trainer.save_model(output_dir)

NameError: name 'trainer' is not defined

## === cell 6
test_dataset = tokenized_datasets["test"]
predictions = trainer.predict(test_dataset)
pred_labels = np.argmax(predictions.predictions, axis=1)  # 0‑5

final_scores = pred_labels + 1



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4145397051.py in <cell line: 0>()
      1 # Predict on the test set
      2 test_dataset = tokenized_datasets["test"]
----> 3 predictions = trainer.predict(test_dataset)
      4 pred_labels = np.argmax(predictions.predictions, axis=1)  # 0‑5
      5 

NameError: name 'trainer' is not defined

## === cell 7
submission = pd.DataFrame(
    {"essay_id": raw_datasets["test"]["essay_id"], "score": final_scores}
)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2813439408.py in <cell line: 0>()
      1 # Build the submission DataFrame
      2 submission = pd.DataFrame(
----> 3     {"essay_id": raw_datasets["test"]["essay_id"], "score": final_scores}
      4 )
      5 

NameError: name 'final_scores' is not defined
