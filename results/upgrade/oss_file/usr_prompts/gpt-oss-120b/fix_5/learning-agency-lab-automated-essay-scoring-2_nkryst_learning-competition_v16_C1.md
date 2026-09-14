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

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer as BertTokenizer, AutoModel as BertModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
class EssayDataset(Dataset):
    """Custom Dataset for essay texts."""

    def __init__(self, tokenizer, texts, max_length):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_len = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
        }


class BertRegressor(nn.Module):
    """BERT based regressor (used only for loading fine‑tuned weights)."""

    def __init__(self, pretrained_name):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_name)
        self.out = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        return self.out(pooled)


LOCAL_MODEL_PATH = "/kaggle/input/bert-model/pytorch/bertmodel/2/bert_regressor (2)/local_bert_base_uncased_model"
MODEL_WEIGHTS_PATH = (
    "/kaggle/input/bert-model/pytorch/bertmodel/2/bert_regressor/bert_regressor.pth"
)
TOKENIZER_PATH = "/kaggle/input/bert-model/pytorch/bertmodel/2/bert_regressor (2)/local_bert_base_uncased_tokenizer"

BATCH_SIZE = 16
MAX_LEN = 256

try:
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
    logging.info("Loaded tokenizer from local path.")
except Exception as e:
    logging.warning(f"Local tokenizer load failed ({e}); using 'bert-base-uncased'.")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

fine_tuned_available = False
try:
    model = BertRegressor(LOCAL_MODEL_PATH)
    model.load_state_dict(
        torch.load(MODEL_WEIGHTS_PATH, map_location=torch.device("cpu"))
    )
    logging.info("Loaded fine‑tuned model weights.")
    fine_tuned_available = True
except Exception as e:
    logging.warning(f"Fine‑tuned model load failed ({e}); using default BERT weights.")
    model = BertRegressor("bert-base-uncased")
model.eval()




## === cell 2
test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
train_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"

test_df = pd.read_csv(test_path)
train_df = pd.read_csv(train_path)

baseline_score = int(round(train_df["score"].mean()))
baseline_score = np.clip(baseline_score, 1, 6)


def get_embeddings(loader):
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            out = model.bert(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            pooled = out.pooler_output  # shape: (batch, hidden)
            embeddings.append(pooled.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


test_dataset = EssayDataset(tokenizer, test_df["full_text"].tolist(), MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

if not fine_tuned_available:
    SAMPLE_SIZE = 20000
    if len(train_df) > SAMPLE_SIZE:
        train_subset = train_df.sample(n=SAMPLE_SIZE, random_state=42)
    else:
        train_subset = train_df

    train_dataset = EssayDataset(tokenizer, train_subset["full_text"].tolist(), MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    logging.info("Computing BERT embeddings for the training subset...")
    train_embeddings = get_embeddings(train_loader)
    train_scores = train_subset["score"].values.astype(np.float32)

    from sklearn.linear_model import Ridge

    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(train_embeddings, train_scores)

    logging.info("Computing BERT embeddings for the test set...")
    test_embeddings = get_embeddings(test_loader)
    preds = ridge.predict(test_embeddings)
    preds = np.round(preds).astype(int)
    preds = np.clip(preds, 1, 6)
else:
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            out = model(**batch)  # shape: (batch, 1)
            preds.extend(out.squeeze().tolist())
    preds = np.round(preds).astype(int)
    preds = np.clip(preds, 1, 6)

test_df["score"] = preds
submission = test_df[["essay_id", "score"]]

output_path = "/kaggle/working/submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
