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

3.13

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

0.7600545627569122

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.05213) has done: 'I set the protobuf implementation to the pure‑Python version before importing transformers to avoid the `MessageFactory` error, and I reduce `MAX_LENGTH` from 1024 to 512 so the position‑embedding size matches BERT’s 512‑token limit. These minimal fixes resolve the import crash and the runtime size mismatch, allowing the script to run end‑to‑end and generate a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    from google.protobuf import message_factory as _mf

    if not hasattr(_mf.MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _mf.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import torch
import transformers
import datasets
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

print(f"PyTorch version: {torch.__version__}")
print(f"Hugging Face Transformers version: {transformers.__version__}")
print(f"Hugging Face Datasets version: {datasets.__version__}")



## === cell 1
MODEL_NAME = "bert-base-cased"
INPUT_DIR = "/kaggle/input/"
MAX_LENGTH = 512
RANDOM_SEED = 42
TRAIN_SUBSET_SIZE = 20000  # small subset to keep runtime reasonable
NUM_EPOCHS = 2
BATCH_SIZE = 8

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")


def load_tokenizer_and_model():
    checkpoint_path = Path(
        "/kaggle/input/aes2-persuade-bertbase-2ep-results/checkpoint-7000"
    )
    if checkpoint_path.is_dir():
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(checkpoint_path), local_files_only=True
            )
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                str(checkpoint_path), local_files_only=True
            )
            print("Loaded tokenizer and model from local checkpoint.")
            return tokenizer, model
        except Exception as e:
            print(f"Failed to load from local checkpoint: {e}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=6
    )
    print("Loaded tokenizer and model from pretrained bert-base-cased.")
    return tokenizer, model


tokenizer, model = load_tokenizer_and_model()
model.to(device)

train_path = os.path.join(
    INPUT_DIR, "learning-agency-lab-automated-essay-scoring-2/train.csv"
)
df_train = pd.read_csv(train_path)

df_train = df_train.sample(
    n=min(TRAIN_SUBSET_SIZE, len(df_train)),
    random_state=RANDOM_SEED,
    stratify=df_train["score"],
).reset_index(drop=True)

df_train["label"] = df_train["score"] - 1  # convert to 0‑5

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_train["full_text"].tolist(),
    df_train["label"].tolist(),
    test_size=0.1,
    random_state=RANDOM_SEED,
    stratify=df_train["label"],
)


def tokenize_batch(texts):
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="np",
    )


train_enc = tokenize_batch(train_texts)
val_enc = tokenize_batch(val_texts)

train_dataset = datasets.Dataset.from_dict(
    {**{k: v.tolist() for k, v in train_enc.items()}, "labels": train_labels}
)
val_dataset = datasets.Dataset.from_dict(
    {**{k: v.tolist() for k, v in val_enc.items()}, "labels": val_labels}
)


def compute_kappa(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    kappa = cohen_kappa_score(labels, preds, weights="quadratic")
    return {"quadratic_kappa": kappa}


training_args = transformers.TrainingArguments(
    output_dir="./model_tmp",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    logging_steps=50,
    seed=RANDOM_SEED,
    fp16=False,  # disable mixed precision unless GPU is present
    report_to="none",
)

trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_kappa,
)

print("Starting fine‑tuning...")
trainer.train()
print("Fine‑tuning completed.")

test_path = os.path.join(
    INPUT_DIR, "learning-agency-lab-automated-essay-scoring-2/test.csv"
)
df_test = pd.read_csv(test_path)
test_texts = df_test["full_text"].tolist()

test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)
test_dataset = datasets.Dataset.from_dict(test_encodings)

prediction_output = trainer.predict(test_dataset)
logits = prediction_output.predictions  # shape (N, 6)
preds = np.argmax(logits, axis=1) + 1  # back to 1‑6 scale

sample_sub_path = os.path.join(
    INPUT_DIR,
    "learning-agency-lab-automated-essay-scoring-2/sample_submission.csv",
)
submission_df = pd.read_csv(sample_sub_path)
submission_df["score"] = preds.astype(int)

submission_df.to_csv("submission.csv", index=False)
print("Submission file created (first rows):")
print(submission_df.head())

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1273728510.py in <cell line: 0>()
     47 
     48 # Sample a subset for quick fine‑tuning
---> 49 df_train = df_train.sample(
     50     n=min(TRAIN_SUBSET_SIZE, len(df_train)),
     51     random_state=RANDOM_SEED,

TypeError: NDFrame.sample() got an unexpected keyword argument 'stratify'
