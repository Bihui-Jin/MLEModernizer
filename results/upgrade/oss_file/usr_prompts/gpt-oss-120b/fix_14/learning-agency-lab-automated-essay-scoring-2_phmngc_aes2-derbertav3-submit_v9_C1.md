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
import collections  # efficient aggregation
from contextlib import nullcontext  # moved to top level

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
TRAIN_BATCH_SIZE = 8  # modest for fine‑tuning
PRED_BATCH_SIZE = 256  # larger batch for faster inference, still safe on GPU
NUM_LABELS = 6
NUM_WORKERS = min(4, os.cpu_count() or 0)  # a few more workers for loading


def load_tokenizer_and_model(local_path: str, fallback_name: str, num_labels: int):
    """
    Load tokenizer and model. If a local directory exists, load from it;
    otherwise, fall back to the HuggingFace hub model.
    """
    if os.path.isdir(local_path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                local_path, num_labels=num_labels, local_files_only=True
            )
            print(f"Loaded tokenizer & model from local path: {local_path}")
            return tokenizer, model
        except Exception as e:
            print(f"Failed loading from local path ({e}); falling back to hub.")
    tokenizer = AutoTokenizer.from_pretrained(fallback_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        fallback_name, num_labels=num_labels
    )
    print(f"Loaded tokenizer & model from hub: {fallback_name}")
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
    Tokenises essays into overlapping chunks using fast tokenizer overflow handling.
    Works for both training (with labels) and inference (without labels).
    Pre‑computes tensors to avoid per‑sample Python overhead.
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

        essay_ids_arr = df["essay_id"].astype(str).values
        scores_arr = df["score"].values if with_labels else None
        texts = df["full_text"].astype(str).tolist()

        tokenised = tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            truncation=True,
            max_length=self.max_content_len,
            stride=self.overlap,
            return_overflowing_tokens=True,
        )
        input_ids_chunks = tokenised["input_ids"]
        overflow_to_sample = tokenised["overflow_to_sample_mapping"]

        pad_id = tokenizer.pad_token_id
        build_fn = tokenizer.build_inputs_with_special_tokens
        min_len_allowed = self.max_content_len * self.min_chunk_ratio

        first_chunk_seen = {}

        for chunk_idx, chunk_ids in enumerate(input_ids_chunks):
            essay_idx = overflow_to_sample[chunk_idx]
            essay_id = essay_ids_arr[essay_idx]

            if len(chunk_ids) < min_len_allowed:
                if first_chunk_seen.get(essay_id, False):
                    continue

            first_chunk_seen[essay_id] = True

            chunk_with_tag = _TAG_TOKENS + chunk_ids
            processed = build_fn(chunk_with_tag)

            if len(processed) > self.max_len:
                processed = processed[: self.max_len]
            else:
                processed += [pad_id] * (self.max_len - len(processed))

            input_ids_tensor = torch.tensor(processed, dtype=torch.long)
            attention_mask_tensor = (input_ids_tensor != pad_id).long()

            if self.with_labels:
                label = int(scores_arr[essay_idx]) - 1
                self.samples.append(
                    (input_ids_tensor, attention_mask_tensor, essay_id, label)
                )
            else:
                self.samples.append((input_ids_tensor, attention_mask_tensor, essay_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.with_labels:
            input_ids, attention_mask, essay_id, label = self.samples[idx]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "essay_id": essay_id,
                "label": torch.tensor(label, dtype=torch.long),
            }
        else:
            input_ids, attention_mask, essay_id = self.samples[idx]
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "essay_id": essay_id,
            }




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
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
model.train()

print("Starting fine‑tuning on a 5k‑sample subset...")
for epoch in range(2):  # two epochs for a modest boost
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.autocast(device.type):
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
    """
    Perform inference on chunked essays and aggregate chunk predictions
    by simple averaging per essay (identical to original logic).
    Uses lightweight Python dicts for aggregation to avoid heavy pandas ops.
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    model.eval()
    sum_preds = collections.defaultdict(float)
    count_preds = collections.defaultdict(int)

    autocast_ctx = (
        torch.autocast(device.type) if device.type == "cuda" else nullcontext()
    )

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            essay_ids = batch["essay_id"]

            with autocast_ctx:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            pred_labels = torch.argmax(logits, dim=1).cpu().numpy() + 1  # back to 1‑6

            for eid, pred in zip(essay_ids, pred_labels):
                sum_preds[eid] += pred
                count_preds[eid] += 1

    final_essay_ids = []
    final_scores = []
    for eid in sum_preds:
        mean_score = sum_preds[eid] / count_preds[eid]
        rounded = int(np.rint(mean_score))
        clipped = int(np.clip(rounded, 1, num_labels))
        final_essay_ids.append(eid)
        final_scores.append(clipped)

    final_results_df = pd.DataFrame(
        {"essay_id": final_essay_ids, "score": final_scores}
    )
    raw_preds = np.array([int(v) for v in sum_preds.values()])  # placeholder, not used
    return final_results_df, raw_preds




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
