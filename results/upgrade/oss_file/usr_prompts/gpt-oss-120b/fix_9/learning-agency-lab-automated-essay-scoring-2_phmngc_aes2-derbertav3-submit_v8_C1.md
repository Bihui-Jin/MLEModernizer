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
import random
import numpy as np
import torch

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True  # enable fast kernel selection

MODEL_NAME_OR_PATH = "microsoft/deberta-v3-large"  # public pretrained model
NUM_LABELS = 6  # scores 1‑6
MAX_LEN = 512
OVERLAP = 128
MIN_CHUNK_RATIO = 0.3
BATCH_SIZE = 4  # GPU‑friendly batch size

possible_base_dirs = [
    os.path.join("data", "learning-agency-lab-automated-essay-scoring-2"),
    os.path.join("data", "input", "learning-agency-lab-automated-essay-scoring-2"),
    os.path.join("/kaggle", "input", "learning-agency-lab-automated-essay-scoring-2"),
    os.path.join("/kaggle", "working", "learning-agency-lab-automated-essay-scoring-2"),
]
BASE_DIR = None
for d in possible_base_dirs:
    if os.path.isdir(d):
        BASE_DIR = d
        break
if BASE_DIR is None:
    raise FileNotFoundError("Could not locate the dataset directory.")

TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")
OUTPUT_PATH = os.path.join("output")
os.makedirs(OUTPUT_PATH, exist_ok=True)




## === cell 1
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    cohen_kappa_score,
    accuracy_score,
)

import matplotlib.pyplot as plt

use_transformer = True
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME_OR_PATH, num_labels=NUM_LABELS
    )
except Exception as e:
    print(
        f"Transformer load failed ({e}); falling back to TF‑IDF + LogisticRegression."
    )
    use_transformer = False
    tokenizer = None
    model = None

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression




## === cell 2
def plot_confusion_matrix(preds, labels, title="Confusion Matrix"):
    preds = np.array(preds)
    labels = np.array(labels)

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Greens", values_format="d")
    plt.title(title)
    plt.show()




## === cell 3
def compute_metrics(preds, labels):
    if len(preds.shape) > 1:
        preds = np.argmax(preds, axis=-1)
    qwk_score = cohen_kappa_score(labels, preds, weights="quadratic")
    acc_score = accuracy_score(labels, preds)
    return {"quadratic_weighted_kappa": qwk_score, "accuracy": acc_score}




## === cell 4
def predict_essay_score(model, dataset, num_labels, batch_size, device):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = model.to(device)
    model.eval()

    preds, essay_ids_all = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", unit="batch"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            essay_ids = batch["essay_id"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            pred_labels = torch.argmax(logits, dim=1).cpu().numpy() + 1  # shift to 1‑6

            preds.extend(pred_labels)
            essay_ids_all.extend(essay_ids)

    chunk_results = pd.DataFrame({"essay_id": essay_ids_all, "raw_prediction": preds})
    aggregated = chunk_results.groupby("essay_id")["raw_prediction"].mean()
    y_pred = np.rint(aggregated.values)
    final_scores = np.clip(y_pred, 1, num_labels).astype(int)

    final_results_df = pd.DataFrame(
        {"essay_id": aggregated.index, "score": final_scores}
    )
    return final_results_df, np.array(preds)




## === cell 5
class EssayDataset(Dataset):
    """
    Shared dataset for both training and testing.
    For training, `labels` are required (0‑based).
    """

    def __init__(
        self,
        df,
        tokenizer,
        max_len=512,
        overlap=128,
        min_chunk_ratio=0.3,  # kept for API compatibility
        is_train=False,
    ):
        self.is_train = is_train
        self.tokenizer = tokenizer

        texts = ("[A] " + df["full_text"].astype(str)).tolist()

        encodings = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
            stride=overlap,
            padding="max_length",
            return_overflowing_tokens=True,
            return_attention_mask=True,
        )

        self.input_ids = torch.tensor(encodings["input_ids"], dtype=torch.long)
        self.attention_mask = torch.tensor(
            encodings["attention_mask"], dtype=torch.long
        )

        mappings = encodings["overflow_to_sample_mapping"]
        self.essay_ids = [df.iloc[idx]["essay_id"] for idx in mappings]

        if self.is_train:
            raw_labels = df["score"].astype(int).values - 1
            self.labels = torch.tensor(
                [raw_labels[idx] for idx in mappings], dtype=torch.long
            )

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        out = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "essay_id": self.essay_ids[idx],
        }
        if self.is_train:
            out["labels"] = self.labels[idx]
        return out




## === cell 6
df_train = pd.read_csv(TRAIN_PATH)

train_df, val_df = train_test_split(
    df_train,
    test_size=0.1,
    random_state=42,
    stratify=df_train["score"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"

if use_transformer:
    train_dataset = EssayDataset(
        train_df.reset_index(drop=True),
        tokenizer,
        MAX_LEN,
        OVERLAP,
        MIN_CHUNK_RATIO,
        is_train=True,
    )
    val_dataset = EssayDataset(
        val_df.reset_index(drop=True),
        tokenizer,
        MAX_LEN,
        OVERLAP,
        MIN_CHUNK_RATIO,
        is_train=True,
    )

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    scaler = torch.cuda.amp.GradScaler()

    model.train()
    for epoch in range(1):  # 1 epoch – lightweight fine‑tuning
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        epoch_losses = []
        for batch in tqdm(train_loader, desc=f"Training epoch {epoch+1}", unit="batch"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(loss.item())
        print(f"Epoch {epoch+1} average loss: {np.mean(epoch_losses):.4f}")

        model.eval()
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].cpu().numpy()
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = (
                        outputs.logits if hasattr(outputs, "logits") else outputs[0]
                    )
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
        val_metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
        print(
            f"Validation QWK: {val_metrics['quadratic_weighted_kappa']:.4f}, "
            f"Acc: {val_metrics['accuracy']:.4f}"
        )
        model.train()
else:
    vectorizer = TfidfVectorizer(
        max_features=50000, ngram_range=(1, 2), stop_words="english"
    )
    X_train = vectorizer.fit_transform(df_train["full_text"].astype(str))
    y_train = df_train["score"].astype(int)  # keep 1‑6 labels
    clf = LogisticRegression(
        multi_class="multinomial", solver="lbfgs", max_iter=200, n_jobs=-1
    )
    clf.fit(X_train, y_train)




## === cell 7
df_test = pd.read_csv(TEST_PATH)

if use_transformer:
    test_dataset = EssayDataset(
        df_test, tokenizer, MAX_LEN, OVERLAP, MIN_CHUNK_RATIO, is_train=False
    )
else:
    test_texts = df_test["full_text"].astype(str)




## === cell 8
if use_transformer:
    test_results, _ = predict_essay_score(
        model=model,
        dataset=test_dataset,
        num_labels=NUM_LABELS,
        batch_size=BATCH_SIZE,
        device=device,
    )
else:
    X_test = vectorizer.transform(test_texts)
    test_preds = clf.predict(X_test)
    test_results = pd.DataFrame(
        {"essay_id": df_test["essay_id"], "score": test_preds.astype(int)}
    )

print("Submission preview:")
print(test_results.head())




## === cell 9
submission_path = os.path.join(OUTPUT_PATH, "submission.csv")
test_results.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
