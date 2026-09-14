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

4e-05

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.00339) has done: 'I reduced the maximum sequence length to BERT’s limit (512 tokens) and removed the `Trainer`‑based inference, which caused protobuf errors. Instead, I perform manual batched forward passes with a simple `DataLoader`, collect logits, and build the submission CSV directly. This fixes the shape mismatch and eliminates the `MessageFactory` error while keeping the original model and tokenizer logic untouched.'
- What this solution (achieved -0.00528) has done: 'Implemented a robust model loading routine that avoids the protobuf `MessageFactory` error by always initializing the tokenizer and a base BERT model from the HuggingFace hub, then manually loading the fine‑tuned weights from the local checkpoint if they exist. This preserves the original architecture while fixing the crash. The rest of the pipeline (encoding, DataLoader, inference, and submission writing) remains unchanged, ensuring a valid `submission.csv` is produced and improving the model’s performance toward the target score.'
- What this solution (achieved -0.12036) has done: 'The fix loads the fine‑tuned checkpoint safely by using a CPU map location and allowing mismatched layer sizes, which avoids the protobuf `MessageFactory` error. The device handling is also corrected. All other logic stays unchanged, so the model now uses the intended weights and produces a proper `submission.csv` with predictions in the required format.'
- What this solution (achieved 4e-05) has done: 'I replace the manual checkpoint loading that triggers the protobuf `MessageFactory` error with a safer `from_pretrained` call that directly loads the fine‑tuned model if the checkpoint directory exists, otherwise it falls back to the base BERT model. This eliminates the AttributeError while keeping the original architecture unchanged, ensuring the script runs end‑to‑end and produces a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch
import transformers
from pathlib import Path

MODEL_NAME = "bert-base-cased"
INPUT_DIR = "/kaggle/input/"
MODEL_DIR = "/kaggle/input/aes2-bertbase-5ep-results/aes2_bertbase_5ep_results/"
MAX_LENGTH = 512  # BERT’s positional embedding limit
RANDOM_SEED = 42

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

checkpoint_path = os.path.join(MODEL_DIR, "checkpoint-1000")
if os.path.isdir(checkpoint_path):
    try:
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path,
            num_labels=6,
            ignore_mismatched_sizes=True,
        )
        print("Loaded fine‑tuned model from checkpoint.")
    except Exception as e:
        print(f"Failed to load fine‑tuned model from checkpoint: {e}")
        print("Falling back to base pretrained model.")
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=6,
            ignore_mismatched_sizes=True,
        )
else:
    print("Checkpoint not found; using base pretrained model.")
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=6,
        ignore_mismatched_sizes=True,
    )

model.to(device)
model.eval()




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df = pd.read_csv(
    os.path.join(INPUT_DIR, "learning-agency-lab-automated-essay-scoring-2/test.csv")
)
texts = df["full_text"].tolist()
print("Read test CSV successfully:", len(texts), "records")

encodings = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors="pt",
)
print("First encoding example:", {k: v[0][:5].tolist() for k, v in encodings.items()})


class EssayDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings["input_ids"].size(0)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


test_dataset = EssayDataset(encodings)
loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

all_logits = []
model.to(device)
with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits  # shape (batch, num_labels)
        all_logits.append(logits.cpu())

logits_tensor = torch.cat(all_logits, dim=0).numpy()
pred_ids = np.argmax(logits_tensor, axis=1) + 1  # convert to 1‑6 range
print("Prediction completed, sample predictions:", pred_ids[:5])

sample_sub_path = os.path.join(
    INPUT_DIR, "learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)
submission_df = pd.read_csv(sample_sub_path)
submission_df["score"] = pred_ids.astype(int)
submission_df.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")
print(submission_df.head())
