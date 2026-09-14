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

3.14

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.metrics import (
    cohen_kappa_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

TRAIN_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
TEST_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
LOCAL_MODEL_PATH = "/kaggle/input/aes2-debertav3-large"
FALLBACK_MODEL_NAME = "microsoft/deberta-v3-large"
OUTPUT_PATH = "/kaggle/working/"

MAX_LEN = 512
OVERLAP = 64
MIN_CHUNK_RATIO = 0.3
TRAIN_BATCH_SIZE = 8  # keep modest for fine‑tuning
PRED_BATCH_SIZE = 32  # larger batch for faster inference
NUM_LABELS = 6
NUM_WORKERS = 4  # increased workers for parallel data loading


def load_tokenizer_and_model(local_path: str, fallback_name: str, num_labels: int):
    """
    Load tokenizer and model, preferring a local directory.
    Falls back to HuggingFace hub if needed.
    Uses generic Auto classes to avoid protobuf issues.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            local_path, num_labels=num_labels
        )
        print(f"Loaded tokenizer & model from local path: {local_path}")
    except Exception as e:
        print(f"Local load failed ({e}); loading from hub '{fallback_name}'.")
        tokenizer = AutoTokenizer.from_pretrained(fallback_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            fallback_name, num_labels=num_labels
        )
    return tokenizer, model


tokenizer, model = load_tokenizer_and_model(
    LOCAL_MODEL_PATH, FALLBACK_MODEL_NAME, NUM_LABELS
)




## === cell 1
def plot_confusion_matrix(preds, labels, title="Confusion Matrix"):
    preds = np.array(preds)
    labels = np.array(labels)

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Greens", values_format="d")
    plt.title(title)
    plt.show()


def compute_metrics(preds, labels):
    if len(preds.shape) > 1:
        preds = np.argmax(preds, axis=-1)
    qwk_score = cohen_kappa_score(labels, preds, weights="quadratic")
    acc_score = accuracy_score(labels, preds)
    return {"quadratic_weighted_kappa": qwk_score, "accuracy": acc_score}




## === cell 2
_TAG_TOKENS = tokenizer("[B]", add_special_tokens=False)["input_ids"]


class ChunkedEssayDataset(Dataset):
    """
    Base class that tokenises essays into overlapping chunks.
    Used for both training (with labels) and test (without labels).
    """

    def __init__(
        self,
        df,
        tokenizer,
        max_len: int = 512,
        overlap: int = 64,
        min_chunk_ratio: float = 0.3,
        with_labels: bool = False,
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.overlap = overlap
        self.min_chunk_ratio = min_chunk_ratio
        self.with_labels = with_labels

        self.num_special = tokenizer.num_special_tokens_to_add(pair=False)
        len_tag = len(_TAG_TOKENS)
        self.max_content_len = max_len - self.num_special - len_tag

        for row in df.itertuples(index=False):
            essay_id = row.essay_id
            base_tokens = tokenizer(row.full_text, add_special_tokens=False)[
                "input_ids"
            ]
            step = self.max_content_len - overlap
            start = 0
            while start < len(base_tokens):
                end = start + self.max_content_len
                chunk = base_tokens[start:end]

                if (
                    len(chunk) < self.max_content_len * self.min_chunk_ratio
                    and start != 0
                ):
                    break

                chunk_with_tag = _TAG_TOKENS + chunk
                processed = tokenizer.build_inputs_with_special_tokens(chunk_with_tag)

                if len(processed) > self.max_len:
                    processed = processed[: self.max_len]

                pad_len = self.max_len - len(processed)
                if pad_len > 0:
                    processed += [tokenizer.pad_token_id] * pad_len

                if self.with_labels:
                    self.samples.append((processed, essay_id, int(row.score) - 1))
                else:
                    self.samples.append((processed, essay_id))
                start += step
                if end >= len(base_tokens):
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.with_labels:
            input_ids, essay_id, label = self.samples[idx]
        else:
            input_ids, essay_id = self.samples[idx]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "essay_id": essay_id,
        }
        if self.with_labels:
            out["label"] = torch.tensor(label, dtype=torch.long)
        return out




## === cell 3
df_train_full = pd.read_csv(TRAIN_PATH)
df_train = df_train_full.sample(n=5000, random_state=42).reset_index(drop=True)

train_dataset = ChunkedEssayDataset(
    df_train, tokenizer, MAX_LEN, OVERLAP, MIN_CHUNK_RATIO, with_labels=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

torch.backends.cudnn.benchmark = True

train_loader = DataLoader(
    train_dataset,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,  # more workers for faster data loading
    pin_memory=True,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
model.train()

print("Starting fine‑tuning on a 5k‑sample subset...")
for epoch in range(1):  # single epoch to stay within time limits
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

torch.cuda.empty_cache()  # free unused memory before inference




## === cell 4
def predict_essay_score(model, dataset, num_labels, batch_size, device):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,  # parallel loading of pre‑tokenised chunks
        pin_memory=True,
    )
    model.eval()
    preds, essay_ids_all = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            essay_ids = batch["essay_id"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            pred_labels = torch.argmax(logits, dim=1).cpu().numpy() + 1  # back to 1‑6
            preds.extend(pred_labels)
            essay_ids_all.extend(essay_ids)

    chunk_results = pd.DataFrame({"essay_id": essay_ids_all, "raw_prediction": preds})
    aggregated = chunk_results.groupby("essay_id")["raw_prediction"].mean()
    final_scores = np.rint(aggregated.values)
    final_scores = np.clip(final_scores, 1, num_labels).astype(int)

    final_results_df = pd.DataFrame(
        {"essay_id": aggregated.index.values, "score": final_scores}
    )
    return final_results_df, np.array(preds)




## === cell 5
df_test = pd.read_csv(TEST_PATH)
test_dataset = ChunkedEssayDataset(
    df_test, tokenizer, MAX_LEN, OVERLAP, MIN_CHUNK_RATIO, with_labels=False
)



## === cell 6
test_results, _ = predict_essay_score(
    model=model,
    dataset=test_dataset,
    num_labels=NUM_LABELS,
    batch_size=PRED_BATCH_SIZE,  # larger batch for faster inference
    device=device,
)

print("Submission preview:")
print(test_results.head())



## === cell 7
submission_path = os.path.join(OUTPUT_PATH, "submission.csv")
test_results.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
