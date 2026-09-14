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

0.7846490997226996

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.01381) has done: 'I remove the unsafe CUDA call, make the model loading robust by falling back to the HuggingFace hub if the local checkpoint isn’t present, and ensure the tokenizer variable is defined before it’s used. These minimal fixes let the script run end‑to‑end and produce a valid `submission.csv` without altering the core modelling logic.'
- What this solution (achieved 0.09637) has done: 'I lower the maximum sequence length to BERT’s 512 limit (preventing the positional‑embedding size error) and replace the `Trainer` with a lightweight manual inference loop that runs the model directly on batches. This also avoids the protobuf‑related `MessageFactory` error triggered by the `Trainer` setup. The core model and tokenization remain unchanged, and the script now reliably creates a proper `submission.csv` file.'
- What this solution (achieved -0.00062) has done: 'The changes load a fine‑tuned checkpoint if it exists (otherwise fall back to the HuggingFace hub) using the explicit `BertForSequenceClassification` class to avoid the protobuf `MessageFactory` error, and build the submission directly from the test IDs to ensure correct ordering. This fixes the runtime crash and should improve the Quadratic Weighted Kappa score while keeping the core modeling logic unchanged.'
- What this solution (achieved 0.0) has done: 'I wrapped the tokenizer loading in a try/except that falls back to the base model tokenizer when the fine‑tuned checkpoint cannot be read (preventing the protobuf `MessageFactory` error), and I changed the prediction step to use the soft‑max weighted expected score (rounded to the 1‑6 range) instead of a plain argmax, which gives a modest improvement toward the target QWK while keeping the original model and training logic unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import transformers

MODEL_NAME = "bert-base-cased"
INPUT_DIR = "/kaggle/input/"
MODEL_DIR = "/kaggle/input/aes2-persuade-bertbase-5ep-results/"
MAX_LENGTH = 512  # BERT’s positional embedding limit
RANDOM_SEED = 42
EVAL_USE_PRETRAIN = 1
SUBMISSION = 1

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

try:
    if os.path.isdir(MODEL_DIR):
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_DIR)
    else:
        raise FileNotFoundError
except Exception as e:
    print(f"Failed to load tokenizer from {MODEL_DIR}: {e}")
    print("Falling back to base tokenizer.")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

try:
    if os.path.isdir(MODEL_DIR):
        model = transformers.BertForSequenceClassification.from_pretrained(
            MODEL_DIR,
            num_labels=6,
        )
    else:
        raise FileNotFoundError
except Exception as e:
    print(f"Failed to load fine‑tuned model from {MODEL_DIR}: {e}")
    print("Falling back to base model from HuggingFace hub.")
    model = transformers.BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=6,
    )

model.to(device)
model.eval()
print("Model and tokenizer loaded successfully.")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1555820974.py in <cell line: 0>()
     38     else:
---> 39         raise FileNotFoundError
     40 except Exception as e:

FileNotFoundError: 

During handling of the above exception, another exception occurred:

AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df = pd.read_csv(
    os.path.join(INPUT_DIR, "learning-agency-lab-automated-essay-scoring-2/test.csv")
)
texts = df["full_text"].tolist()
print("Read test CSV successfully.")

encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)
print("Sample encoding keys:", list(encodings.keys())[:3])


class EssayDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


dataset = EssayDataset(encodings)
loader = torch.utils.data.DataLoader(dataset, batch_size=32)

all_logits = []
with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits  # (batch, 6)
        all_logits.append(logits.cpu())
all_logits = torch.cat(all_logits, dim=0)  # (num_examples, 6)

probs = torch.nn.functional.softmax(all_logits, dim=1)  # (n, 6)
labels = torch.arange(1, 7, dtype=probs.dtype)  # 1‑6
expected_scores = (probs * labels).sum(dim=1)  # continuous scores
pred_labels = torch.round(expected_scores).clamp(1, 6).int()  # round & clip to 1‑6

print("Prediction completed.")

submission_df = pd.DataFrame(
    {"essay_id": df["essay_id"], "score": pred_labels.numpy().astype(int)}
)

submission_df.to_csv("submission.csv", index=False)
print("Submission file written: submission.csv")
print(submission_df.head())
