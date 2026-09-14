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
Predict which chatbot response a user will prefer in a competition between two chatbots.

## Metric
Log loss with "eps=auto"

## Submission Format
For each id in the test set, you must predict the probability for each target class. The file should contain a header and have the following format:

```
 id,winner_model_a,winner_model_b,winner_tie
 136060,0.33,0,33,0.33
 211333,0.33,0,33,0.33
 1233961,0.33,0,33,0.33
 etc
```

## Dataset
**train.csv**

- `id` - A unique identifier for the row.
- `model_[a/b]` - The identity of model_[a/b]. Included in train.csv but not test.csv.
- `prompt` - The prompt that was given as an input (to both models).
- `response_[a/b]` - The response from model_[a/b] to the given prompt.
- `winner_model_[a/b/tie]` - Binary columns marking the judge's selection. The ground truth target column.

**test.csv**

- `id`
- `prompt`
- `response_[a/b]`

**sample_submission.csv** A submission file in the correct format.

- `id`
- `winner_model_[a/b/tie]` - This is what is predicted from the test set.

# 2. Python version

3.14

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
peft==0.16.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (98 lines)
            sample_submission.csv (5749 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (5749 lines)
            test.csv.zip (5.9 MB)
            train.csv (51730 lines)
            train.csv.zip (53.4 MB)
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
        input/
            description.md (98 lines)
            sample_submission.csv (5749 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (5749 lines)
            test.csv.zip (5.9 MB)
            train.csv (51730 lines)
            train.csv.zip (53.4 MB)
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
        working/
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
```

-> data/lmsys-chatbot-arena/sample_submission.csv has 5748 rows and 4 columns.
The columns are: id, winner_model_a, winner_model_b, winner_tie

-> data/lmsys-chatbot-arena/test.csv has 5748 rows and 4 columns.
The columns are: id, prompt, response_a, response_b

-> data/lmsys-chatbot-arena/train.csv has 51729 rows and 9 columns.
The columns are: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie

-> data/sample_submission.csv has 5748 rows and 4 columns.
The columns are: id, winner_model_a, winner_model_b, winner_tie

-> data/test.csv has 5748 rows and 4 columns.
The columns are: id, prompt, response_a, response_b

-> data/train.csv has 51729 rows and 9 columns.
The columns are: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie

-> (stopped after 10 files for performance)

# 5. Target score

1.0372911976442458

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print("Environment ready.")


## === cell 1
CANDIDATE_LOCAL_PATHS = [
    "/kaggle/input/mistral-checkpt-50-fp16-final/transformers/default/1",
    "/kaggle/input/mistral-checkpt-50-fp16-final",
]

FALLBACK_MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"


def pick_model_path():
    for p in CANDIDATE_LOCAL_PATHS:
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json")):
            return p
    return None


MODEL_PATH = pick_model_path()
if MODEL_PATH is None:
    print("WARNING: Custom checkpoint not found. Falling back to:", FALLBACK_MODEL_ID)
    MODEL_PATH = FALLBACK_MODEL_ID
else:
    print("Loading local model from:", MODEL_PATH)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, use_fast=True, local_files_only=os.path.isdir(MODEL_PATH)
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = (
        tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    )
tokenizer.padding_side = "right"

use_cuda = torch.cuda.is_available()
dtype = torch.float16 if use_cuda else torch.float32

quant_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    torch_dtype=dtype,
    device_map="auto" if use_cuda else None,
    local_files_only=os.path.isdir(MODEL_PATH),
)
quant_model.eval()

num_labels = getattr(quant_model.config, "num_labels", None)
device = next(quant_model.parameters()).device
print("Loaded model. num_labels =", num_labels, "| device =", device)

model_max_len = getattr(quant_model.config, "max_position_embeddings", None)
if model_max_len is None:
    tmax = getattr(tokenizer, "model_max_length", 512)
    model_max_len = (
        int(tmax) if isinstance(tmax, (int, np.integer)) and tmax < 100000 else 512
    )

MAX_LENGTH = int(max(8, min(1536, model_max_len)))
print(
    "Using MAX_LENGTH =",
    MAX_LENGTH,
    "(model max_position_embeddings =",
    model_max_len,
    ")",
)


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
test_path = "/kaggle/input/lmsys-chatbot-arena/test.csv"
if not os.path.exists(test_path):
    test_path = "/kaggle/data/lmsys-chatbot-arena/test.csv"

test_df = pd.read_csv(test_path)

test_df["combined_text"] = (
    test_df["prompt"].fillna("").astype(str)
    + "\n\nResponse A:\n"
    + test_df["response_a"].fillna("").astype(str)
    + "\n\nResponse B:\n"
    + test_df["response_b"].fillna("").astype(str)
)

print("Test loaded:", test_df.shape)


## === cell 3
temperature = 1.8
batch_size = 16
preds = []

device = next(quant_model.parameters()).device

for i in tqdm(range(0, len(test_df), batch_size), desc="Predict"):
    batch = test_df.iloc[i : i + batch_size]["combined_text"].tolist()
    inputs = tokenizer(
        batch,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,  # Fix: clamp to model-supported maximum length
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = quant_model(**inputs)
        logits = out.logits

        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)

    preds.append(probs.detach().cpu().numpy())

preds = np.concatenate(preds, axis=0)

if preds.shape[1] == 3:
    p3 = preds
elif preds.shape[1] == 2:
    tie = np.full((preds.shape[0], 1), 1e-3, dtype=preds.dtype)
    p3 = np.concatenate([preds[:, :1], preds[:, 1:2], tie], axis=1)
    p3 = p3 / p3.sum(axis=1, keepdims=True)
else:
    if preds.shape[1] > 3:
        p3 = preds[:, :3]
    else:
        pad = np.full((preds.shape[0], 3 - preds.shape[1]), 1e-3, dtype=preds.dtype)
        p3 = np.concatenate([preds, pad], axis=1)
    p3 = p3 / p3.sum(axis=1, keepdims=True)

submission = pd.DataFrame(
    {
        "id": test_df["id"].values,
        "winner_model_a": p3[:, 0],
        "winner_model_b": p3[:, 1],
        "winner_tie": p3[:, 2],
    }
)

submission.to_csv("submission.csv", index=False)
print("submission.csv ready:", submission.shape)
print(submission.head())

## --- ERROR in outputing the csv:
Invalid submission: Each row in submission DataFrame must sum to 1.
