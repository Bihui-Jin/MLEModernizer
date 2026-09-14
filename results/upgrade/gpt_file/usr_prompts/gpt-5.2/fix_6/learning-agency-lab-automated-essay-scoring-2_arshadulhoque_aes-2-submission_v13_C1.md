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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification

torch.manual_seed(42)
np.random.seed(42)



## === cell 1
test_data = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)
test_data.head()



## === cell 2
PRIMARY_MODEL_NAME = "microsoft/deberta-v3-xsmall"
FALLBACK_MODEL_NAME = "microsoft/deberta-v3-base"  # broadly available; will run with random classification head if needed


def load_tokenizer_and_model(model_name: str):
    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    mdl = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)
    return tok, mdl


try:
    tokenizer, model = load_tokenizer_and_model(PRIMARY_MODEL_NAME)
except Exception as e_primary:
    tokenizer, model = load_tokenizer_and_model(FALLBACK_MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()



## === cell 3
model_max_len = getattr(tokenizer, "model_max_length", 512)
if model_max_len is None or model_max_len > 1024:
    model_max_len = 512

max_length = min(512, model_max_len)

enc = tokenizer(
    test_data["full_text"].tolist(),
    truncation=True,
    padding=True,
    max_length=max_length,
    return_tensors="np",
)

test_encodings = {k: v for k, v in enc.items()}
len(test_encodings["input_ids"]), len(test_data)




## === cell 4
class EncodedTextDataset(TorchDataset):
    def __init__(self, encodings_np):
        self.input_ids = torch.from_numpy(encodings_np["input_ids"]).long()
        self.attention_mask = torch.from_numpy(encodings_np["attention_mask"]).long()
        self.token_type_ids = None
        if "token_type_ids" in encodings_np:
            self.token_type_ids = torch.from_numpy(
                encodings_np["token_type_ids"]
            ).long()

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[idx]
        return item


test_dataset = EncodedTextDataset(test_encodings)



## === cell 5
batch_size = 64 if torch.cuda.is_available() else 16

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,  # dataset is already tensor-backed; extra workers often add overhead in Kaggle
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False,
)

all_logits = []
with torch.no_grad():
    for batch in test_loader:
        if torch.cuda.is_available():
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        else:
            batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        all_logits.append(outputs.logits.detach().cpu().numpy())

logits = np.concatenate(all_logits, axis=0)
logits.shape



## === cell 6
pred_class = np.argmax(logits, axis=-1)

num_labels = logits.shape[-1]
if num_labels == 6:
    predicted_scores = (pred_class + 1).astype(np.int32)
else:
    predicted_scores = np.rint(1 + (pred_class / max(1, num_labels - 1)) * 5).astype(
        np.int32
    )

predicted_scores = np.clip(predicted_scores, 1, 6)

predicted_scores[:10], predicted_scores.min(), predicted_scores.max(), num_labels



## === cell 7
submission = pd.DataFrame(
    {
        "essay_id": test_data["essay_id"].values,
        "score": predicted_scores,
    }
)

submission = submission[["essay_id", "score"]]
submission.to_csv("submission.csv", index=False)
submission.head()



## === cell 8
submission
