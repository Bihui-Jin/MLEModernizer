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

0.781539931926555

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I fix the environment/runtime error coming from an incompatible protobuf stack by forcing the pure-Python protobuf implementation before importing `transformers`/`datasets`. Then I fix the inference crash by aligning `MAX_LENGTH` with BERT’s maximum positional embeddings (512), which removes the 1024-vs-512 shape mismatch while keeping the same argmax+1 prediction logic. Finally, I make the dataset compatible with `Trainer.predict()` by setting torch format and moving the model to the detected device, ensuring a `submission.csv` with the required columns is always written.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch

print(f"PyTorch version: {torch.__version__}")

import transformers

print(f"Hugging Face Transformers version: {transformers.__version__}")

import datasets

print(f"Hugging Face Datasets version: {datasets.__version__}")




## === cell 1
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_NAME = "bert-base-cased"
INPUT_DIR = "/kaggle/input/"
MODEL_DIR = "/kaggle/input/aes2-bertbase-5ep-results/aes2_bertbase_5ep_results/"
CHECKPOINT_SUBDIR = "checkpoint-1000"

MAX_LENGTH = 512

RANDOM_SEED = 42
EVAL_USE_PRETRAIN = 1
SUBMISSION = 1

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device:", device)
if torch.cuda.is_available():
    print("cuda current_device:", torch.cuda.current_device())
else:
    print("cuda not available; running on CPU")

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)




## === cell 2
checkpoint_path = os.path.join(MODEL_DIR, CHECKPOINT_SUBDIR)
use_path = checkpoint_path if os.path.isdir(checkpoint_path) else MODEL_NAME

if os.path.isdir(checkpoint_path):
    print("Loading local checkpoint from:", checkpoint_path)
else:
    print("WARNING: checkpoint not found at:", checkpoint_path)
    print("Falling back to base model:", MODEL_NAME)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    use_path,
    local_files_only=os.path.isdir(checkpoint_path),
    use_fast=True,
)
print("Loaded tokenizer successfully")

model = transformers.AutoModelForSequenceClassification.from_pretrained(
    use_path,
    local_files_only=os.path.isdir(checkpoint_path),
)
model.to(device)
model.eval()
print("Loaded model successfully")

use_fp16 = bool(torch.cuda.is_available())

training_args = transformers.TrainingArguments(
    output_dir=".",
    per_device_eval_batch_size=32,
    report_to="none",
    fp16=use_fp16,
)

trainer = transformers.Trainer(
    model=model,
    args=training_args,
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
df = pd.read_csv(
    os.path.join(INPUT_DIR, "learning-agency-lab-automated-essay-scoring-2", "test.csv")
)
texts = df["full_text"].astype(str).values.tolist()
print("Read test.csv successfully:", df.shape)

encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
)
print("Encoded texts successfully; keys:", list(encodings.keys()))

dataset = datasets.Dataset.from_dict(encodings)
dataset = dataset.with_format("torch", columns=list(encodings.keys()))
print("Built Dataset successfully:", dataset)

submission_logits = trainer.predict(dataset).predictions
submission_preds = torch.argmax(torch.tensor(submission_logits), dim=1) + 1
submission_preds = submission_preds.cpu().numpy().astype(int)
print("Predicted successfully; preds shape:", submission_preds.shape)

submission_csv = pd.read_csv(
    os.path.join(
        INPUT_DIR,
        "learning-agency-lab-automated-essay-scoring-2",
        "sample_submission.csv",
    )
)
if len(submission_csv) != len(submission_preds):
    raise ValueError(
        f"Prediction length mismatch: sample_submission={len(submission_csv)} vs preds={len(submission_preds)}"
    )

submission_csv["score"] = submission_preds
submission_csv["score"] = submission_csv["score"].astype(int)

submission_csv["score"] = submission_csv["score"].clip(1, 6)

submission_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv:", submission_csv.shape)
print(submission_csv.head())
