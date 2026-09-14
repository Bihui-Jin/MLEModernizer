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

1.091960670764198

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import re
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("Imports complete")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)



## === cell 1
print("Loading test.csv...")
test_path = "/kaggle/input/lmsys-chatbot-arena/test.csv"
test = pd.read_csv(test_path)
print(f"Loaded {len(test)} rows from test.csv")
print("Columns:", list(test.columns))




## === cell 2
def clean_text(text):
    """Remove invalid UTF-16 surrogates and coerce to string."""
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"[\ud800-\udfff]", "", text)


def _maybe_parse_first(x):
    """train/test sometimes contain stringified lists; if so, take the first element."""
    try:
        v = ast.literal_eval(x) if isinstance(x, str) else x
        if isinstance(v, list):
            return v[0] if len(v) else ""
        return v
    except Exception:
        return x


def build_text_from_row(row):
    prompt = clean_text(_maybe_parse_first(row["prompt"]))
    ra = clean_text(_maybe_parse_first(row["response_a"]))
    rb = clean_text(_maybe_parse_first(row["response_b"]))
    return f"Prompt: {prompt}\n\nResponse A: {ra}\n\nResponse B: {rb}"


print("Preprocessing test data...")
test_texts = [build_text_from_row(r) for _, r in test.iterrows()]
print("Built texts:", len(test_texts))



## === cell 3
model_name = "distilbert-base-uncased"
print(f"Loading tokenizer/model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
model.to(device)
model.eval()
print("Model and tokenizer loaded successfully")




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
class TextInferenceDataset(TorchDataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        return item


print("Creating dataloader...")
infer_ds = TextInferenceDataset(test_texts, tokenizer, max_length=512)
loader = DataLoader(
    infer_ds,
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)
print("Dataloader ready:", len(loader), "batches")



## === cell 5
print("Running inference on test set...")
all_probs = []

use_amp = torch.cuda.is_available()
autocast_ctx = torch.cuda.amp.autocast if use_amp else torch.cpu.amp.autocast

with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(**batch).logits
        else:
            logits = model(**batch).logits

        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        all_probs.append(probs)

probabilities = np.vstack(all_probs)
print("Inference finished. probs shape:", probabilities.shape)

if probabilities.shape[0] != len(test) or probabilities.shape[1] != 3:
    raise RuntimeError(
        f"Unexpected probabilities shape {probabilities.shape}; expected ({len(test)}, 3)"
    )



## === cell 6
submission = pd.DataFrame(
    {
        "id": test["id"].values,
        "winner_model_a": probabilities[:, 0],
        "winner_model_b": probabilities[:, 1],
        "winner_tie": probabilities[:, 2],
    }
)

row_sums = (
    submission[["winner_model_a", "winner_model_b", "winner_tie"]].sum(axis=1).values
)
submission[["winner_model_a", "winner_model_b", "winner_tie"]] = submission[
    ["winner_model_a", "winner_model_b", "winner_tie"]
].div(row_sums, axis=0)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print(f"{out_path} created successfully!")
print(submission.head())
print("Submission shape:", submission.shape)
print("Row sums (min/max):", row_sums.min(), row_sums.max())

## --- ERROR in outputing the csv:
Invalid submission: Each row in submission DataFrame must sum to 1.
