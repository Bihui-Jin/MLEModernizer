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

# 5. Target score

0.8019899540067545

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I set the protobuf implementation before importing transformers, replace the invalid local model path with a public DeBERTa‑v3‑large model, add a safe device selection, and ensure the script creates the test dataset only after the tokenizer is defined. These fixes remove the import errors, allow the model to load correctly, and guarantee that a “submission.csv” with the required columns is written, enabling a valid end‑to‑end run.'
- What this solution (achieved 0.0) has done: 'Implemented a robust fix for the protobuf import error by resetting the environment variable right before importing `transformers`. Added a lightweight fine‑tuning loop (1 epoch) on the training data using the same chunking logic as the test dataset, which raises the model from a raw pretrained state to a task‑specific model and therefore improves the quadratic weighted kappa score toward the target. All other logic (prediction, aggregation, CSV output) remains unchanged, and the script now reliably writes a valid `submission.csv` file.'

# 9. Code solution

## === cell 0
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
tokenizer = None
model = None

from transformers import AutoTokenizer, AutoModelForSequenceClassification

if use_transformer:
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME_OR_PATH, num_labels=NUM_LABELS
        )
    except Exception as e:
        print(f"Transformer loading failed ({e}), falling back to TF‑IDF baseline.")
        use_transformer = False

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression



## === cell 1
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

    model.train()
    for epoch in range(1):  # 1 epoch – lightweight fine‑tuning
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
        )
        epoch_losses = []
        for batch in tqdm(train_loader, desc=f"Training epoch {epoch+1}", unit="batch"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        print(f"Epoch {epoch+1} average loss: {np.mean(epoch_losses):.4f}")

        model.eval()
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
        )
        all_preds, all_labels = [], []
        with torch.inference_mode():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].cpu().numpy()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
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
        max_features=200000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(df_train["full_text"].astype(str))
    y_train = df_train["score"].astype(int)  # keep 1‑6 labels
    clf = LogisticRegression(
        multi_class="multinomial", solver="lbfgs", max_iter=500, n_jobs=-1
    )
    clf.fit(X_train, y_train)

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/276398512.py in <cell line: 0>()
----> 1 df_train = pd.read_csv(TRAIN_PATH)
      2 
      3 train_df, val_df = train_test_split(
      4     df_train,
      5     test_size=0.1,

NameError: name 'TRAIN_PATH' is not defined
