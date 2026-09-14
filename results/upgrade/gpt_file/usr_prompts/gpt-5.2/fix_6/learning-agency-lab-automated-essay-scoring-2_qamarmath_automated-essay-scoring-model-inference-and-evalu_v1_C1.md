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

0.7577191694774467

# 6. Current score

0.34869

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.15209) has done: 'I fix the two root causes preventing end-to-end execution: (1) the `datasets` mapping crash caused by an incompatible protobuf runtime, and (2) the model path failing HF validation because the referenced Kaggle input model directory doesn’t exist in your provided file tree. To keep the core inference logic (tokenize → DeBERTa sequence classification → argmax+1) intact, I load a compatible public DeBERTa-v3 model from the local Transformers cache when available, otherwise download it (no training added). I also avoid `datasets` entirely (use a plain PyTorch `Dataset` + DataLoader) to sidestep the protobuf issue while keeping identical tokenization and padding semantics. Finally, I write `submission.csv` with the required `essay_id,score` columns and correct row alignment.'
- What this solution (achieved 0.00791) has done: 'I fix the protobuf-related crash (`MessageFactory.GetPrototype`) that happens when importing/using `transformers` in this Kaggle image by forcing a compatible pure-Python protobuf implementation before `transformers` is imported. This is a minimal environment fix and keeps your core logic identical (tokenize → DeBERTa sequence classification → argmax+1 → write submission). I also add safe fallbacks for the test file path and ensure the submission is always written with the required `essay_id,score` columns. No training, architecture, or inference semantics are changed—this should both run end-to-end and substantially improve score vs. the broken/degenerate run.'
- What this solution (achieved 0.24088) has done: 'We fix the protobuf runtime error by forcing the pure-Python protobuf implementation *and* ensuring it takes effect before any `protobuf/transformers` internals load, plus set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` which is the most compatible mode for this Kaggle image. This is an execution-stability fix that preserves your exact inference semantics (same model, same tokenization, same argmax+1 postprocess) and should restore the intended score (your current 0.00791 is consistent with a broken runtime). I also add a small safety fallback to CPU if CUDA is present but unstable, without changing outputs otherwise, and keep the submission format unchanged. No training, architecture, or thresholding changes are introduced.'
- What this solution (achieved 0.06337) has done: 'The crash is coming from an incompatible compiled `protobuf` backend being used at runtime despite your env vars; `transformers` ends up importing `google.protobuf` before the pure-Python implementation is enforced. I fix this by forcing the env vars *and* reloading the protobuf modules before importing anything from `transformers`, which is the minimal stability change that keeps your inference logic identical. I also add a small fallback to disable fast tokenizers only if the same protobuf error persists (score-neutral but makes the notebook robust). The rest (model choice, max_length, argmax+1 postprocess, submission format) stays unchanged.'
- What this solution (achieved 0.34869) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by enforcing the pure-Python protobuf implementation *before* any `google.protobuf` / `transformers` imports and by hard-blocking the compiled `google._upb` module if it was loaded. This is an execution-stability fix that keeps your core inference logic unchanged (same tokenizer/model family, same argmax+1 postprocess) and should restore a non-degenerate score. I also make the exception handling catch both `AttributeError` and `ImportError` for this specific protobuf failure mode so the fallback actually triggers. Finally, I keep paths unchanged and ensure `submission.csv` is always written with the required `essay_id,score` columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import sys
import importlib

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf") or m.startswith("google._upb"):
        del sys.modules[m]
importlib.invalidate_caches()

sys.modules["google._upb"] = None

import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification

TEST_DATA_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
if not os.path.exists(TEST_DATA_PATH):
    TEST_DATA_PATH = "/kaggle/input/test.csv"

MAX_LENGTH = 1024
EVAL_BATCH_SIZE = 1

MODEL_PATH = "microsoft/deberta-v3-large"

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    device = torch.device("cpu")

df_test = pd.read_csv(TEST_DATA_PATH)

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
except (AttributeError, ImportError) as e:
    if "GetPrototype" in str(e) or "google._upb" in str(e):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    else:
        raise


class EssayDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts = df["full_text"].astype(str).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"full_text": self.texts[idx]}


class DataCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        texts = [f["full_text"] for f in features]
        batch = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        return batch


ds = EssayDataset(df_test, tokenizer, MAX_LENGTH)
collator = DataCollator(tokenizer, MAX_LENGTH)
dl = DataLoader(ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collator)




## === cell 1
def load_model_safely():
    try:
        return AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH, num_labels=6
        )
    except (AttributeError, ImportError) as e:
        if "GetPrototype" in str(e) or "google._upb" in str(e):
            for m in list(sys.modules.keys()):
                if m.startswith("google.protobuf") or m.startswith("google._upb"):
                    del sys.modules[m]
            importlib.invalidate_caches()
            sys.modules["google._upb"] = None
            return AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH, num_labels=6
            )
        raise


model = load_model_safely()
model.to(device)
model.eval()

all_logits = []
with torch.no_grad():
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        all_logits.append(out.logits.detach().cpu())

logits = torch.cat(all_logits, dim=0).numpy()

del model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
preds = logits.argmax(axis=-1) + 1  # labels 1..6
preds = np.clip(preds, 1, 6).astype(int)

df_sub = pd.DataFrame({"essay_id": df_test["essay_id"].values, "score": preds})
df_sub.to_csv("submission.csv", index=False)

print(df_sub.head())
print("Wrote submission.csv with shape:", df_sub.shape)
print("Submission columns:", df_sub.columns.tolist())
