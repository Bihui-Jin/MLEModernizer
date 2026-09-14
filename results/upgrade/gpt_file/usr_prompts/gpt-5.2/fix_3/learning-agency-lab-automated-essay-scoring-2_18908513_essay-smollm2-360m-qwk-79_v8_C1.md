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
text-unidecode==1.3
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

0.786623184549083

# 6. Current score

0.00556

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.00556) has done: 'I fix the immediate crash by making model loading robust to local Kaggle dataset folder structures (including nested `snapshots/` layouts) and by forcing Transformers to treat the path as a local directory rather than a Hub repo id. Then I unblock the downstream cells by ensuring `df`, `tokenizer`, and `model` are always defined (with a safe fallback model if the provided local model isn’t present), so the script runs end-to-end and writes `submission.csv` with the required columns. I also add missing tokenizer settings needed for some decoder-only tokenizers (pad token) to prevent runtime padding errors during batching. These changes keep the core inference logic (argmax over logits, +1, clip 1–6) the same while ensuring a valid submission is produced.'

# 9. Code solution

## === cell 0
import os
import re
import codecs
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from text_unidecode import unidecode


class PredictionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = self.dataframe.iloc[idx]["full_text"]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start : error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start : error.end].decode("cp1252"), error.end


codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)


def resolve_encodings_and_normalize(text: str) -> str:
    """Resolve encoding problems and normalize abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text


def preprocess_essay_text(text: str) -> str:
    """
    Prepares essay text for scoring by cleaning non-essential issues without altering quality indicators.
    - Resolves encoding issues
    - Normalizes whitespace
    - Preserves original spelling, grammar, and casing
    """
    text = resolve_encodings_and_normalize(text)
    text = re.sub(r"\s+", " ", text.strip())  # Normalize whitespace
    text = re.sub(r'\s+([?.!,"])', r"\1", text)  # Remove spaces before punctuation
    text = re.sub(r",([^\s])", r", \1", text)  # Add space after commas
    return text


def _find_hf_model_dir(base_path: str) -> str:
    candidates = []
    if base_path and os.path.isdir(base_path):
        candidates.append(base_path)
        for root, _, files in os.walk(base_path):
            if "config.json" in files:
                return root
    return base_path


def _load_local_model_and_tokenizer(model_base: str):
    model_path = _find_hf_model_dir(model_base)

    if not (
        model_path
        and os.path.isdir(model_path)
        and os.path.isfile(os.path.join(model_path, "config.json"))
    ):
        raise FileNotFoundError(
            f"Could not find a local HF model with config.json under: {model_base}"
        )

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, config=config, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, use_fast=True
    )
    return model, tokenizer, model_path


MODEL_BASE = "/kaggle/input/smollm2-360m-essay-scoring-model"

try:
    model, tokenizer, MODEL_PATH = _load_local_model_and_tokenizer(MODEL_BASE)
except Exception as e:
    fallback = "distilbert-base-uncased"
    MODEL_PATH = fallback
    model = AutoModelForSequenceClassification.from_pretrained(
        fallback, num_labels=6, local_files_only=False
    )
    tokenizer = AutoTokenizer.from_pretrained(
        fallback, use_fast=True, local_files_only=False
    )
    print("WARNING: Failed to load local model from:", MODEL_BASE)
    print("Reason:", repr(e))
    print("Falling back to:", fallback)

if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

df = pd.read_csv("/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv")
df["full_text"] = df["full_text"].astype(str).apply(preprocess_essay_text)

print("Loaded model from:", MODEL_PATH)
print("Test rows:", len(df))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_56/1090136799.py in <cell line: 0>()
    117 try:
--> 118     model, tokenizer, MODEL_PATH = _load_local_model_and_tokenizer(MODEL_BASE)
    119 except Exception as e:

/tmp/ipykernel_56/1090136799.py in _load_local_model_and_tokenizer(model_base)
     98     ):
---> 99         raise FileNotFoundError(
    100             f"Could not find a local HF model with config.json under: {model_base}"

FileNotFoundError: Could not find a local HF model with config.json under: /kaggle/input/smollm2-360m-essay-scoring-model

During handling of the above exception, another exception occurred:

AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
test_dataset = PredictionDataset(df, tokenizer)



## === cell 2
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

predicted_labels = []
with torch.no_grad():
    for batch in test_dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        predicted_labels.extend(predictions)

predicted_labels = (np.array(predicted_labels) + 1).astype(int)
predicted_labels = np.clip(predicted_labels, 1, 6)

df["score"] = predicted_labels
df[["essay_id", "score"]].to_csv("submission.csv", index=False)

print("Predictions saved to 'submission.csv'")
print(df[["essay_id", "score"]].head())
