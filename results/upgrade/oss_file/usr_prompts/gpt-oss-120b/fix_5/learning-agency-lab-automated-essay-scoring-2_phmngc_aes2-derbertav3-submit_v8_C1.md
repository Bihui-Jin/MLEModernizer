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
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

MODEL_NAME_OR_PATH = "microsoft/deberta-v3-large"  # public pretrained model
NUM_LABELS = 6  # scores 1‑6
MAX_LEN = 512
OVERLAP = 128
MIN_CHUNK_RATIO = 0.3
BATCH_SIZE = 4  # GPU‑friendly batch size

BASE_DIR = os.path.join("data", "learning-agency-lab-automated-essay-scoring-2")
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

from transformers import AutoTokenizer, AutoModelForSequenceClassification



## === cell 2
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME_OR_PATH, num_labels=NUM_LABELS
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
def plot_confusion_matrix(preds, labels, title="Confusion Matrix"):
    preds = np.array(preds)
    labels = np.array(labels)

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Greens", values_format="d")
    plt.title(title)
    plt.show()




## === cell 4
def compute_metrics(preds, labels):
    if len(preds.shape) > 1:
        preds = np.argmax(preds, axis=-1)
    qwk_score = cohen_kappa_score(labels, preds, weights="quadratic")
    acc_score = accuracy_score(labels, preds)
    return {"quadratic_weighted_kappa": qwk_score, "accuracy": acc_score}




## === cell 5
def predict_essay_score(model, dataset, num_labels, batch_size, device):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    model.eval()

    preds, essay_ids_all = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
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




## === cell 6
class EssayDataset(Dataset):
    """
    Shared dataset for both training and testing.
    For training, `labels` must be supplied (0‑based).
    """

    def __init__(
        self,
        df,
        tokenizer,
        max_len=512,
        overlap=128,
        min_chunk_ratio=0.3,
        is_train=False,
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.overlap = overlap
        self.min_chunk_ratio = min_chunk_ratio
        self.is_train = is_train

        num_special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
        self.max_content_len = max_len - num_special_tokens

        for _, row in df.iterrows():
            essay_id = row["essay_id"]
            text = "[A] " + row["full_text"]
            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]

            step = self.max_content_len - overlap
            start = 0
            while start < len(tokens):
                end = start + self.max_content_len
                chunk = tokens[start:end]
                if len(chunk) < self.max_content_len * self.min_chunk_ratio:
                    break

                processed = tokenizer.build_inputs_with_special_tokens(chunk)
                padding_len = self.max_len - len(processed)
                if padding_len > 0:
                    processed += [tokenizer.pad_token_id] * padding_len

                sample = {"input_ids": processed, "essay_id": essay_id}
                if self.is_train:
                    sample["label"] = int(row["score"]) - 1
                self.samples.append(sample)

                start += step
                if end >= len(tokens):
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_ids = torch.tensor(sample["input_ids"], dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "essay_id": sample["essay_id"],
        }
        if self.is_train:
            out["labels"] = torch.tensor(sample["label"], dtype=torch.long)
        return out




## === cell 7
from sklearn.model_selection import train_test_split

df_train = pd.read_csv(TRAIN_PATH)

train_df, val_df = train_test_split(
    df_train,
    test_size=0.1,
    random_state=42,
    stratify=df_train["score"],
)

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

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

model.train()
for epoch in range(1):  # 1 epoch – lightweight fine‑tuning
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    epoch_losses = []
    for batch in tqdm(train_loader, desc=f"Training epoch {epoch+1}", unit="batch"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    print(f"Epoch {epoch+1} average loss: {np.mean(epoch_losses):.4f}")

    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
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



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/134112798.py in <cell line: 0>()
      1 from sklearn.model_selection import train_test_split
      2 
----> 3 df_train = pd.read_csv(TRAIN_PATH)
      4 
      5 train_df, val_df = train_test_split(

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'data/learning-agency-lab-automated-essay-scoring-2/train.csv'

## === cell 8
df_test = pd.read_csv(TEST_PATH)
test_dataset = EssayDataset(
    df_test, tokenizer, MAX_LEN, OVERLAP, MIN_CHUNK_RATIO, is_train=False
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2337779078.py in <cell line: 0>()
----> 1 df_test = pd.read_csv(TEST_PATH)
      2 test_dataset = EssayDataset(
      3     df_test, tokenizer, MAX_LEN, OVERLAP, MIN_CHUNK_RATIO, is_train=False
      4 )
      5 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'data/learning-agency-lab-automated-essay-scoring-2/test.csv'

## === cell 9
device = "cuda" if torch.cuda.is_available() else "cpu"

test_results, _ = predict_essay_score(
    model=model,
    dataset=test_dataset,
    num_labels=NUM_LABELS,
    batch_size=BATCH_SIZE,
    device=device,
)

print("Submission preview:")
print(test_results.head())



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4275788552.py in <cell line: 0>()
      3 test_results, _ = predict_essay_score(
      4     model=model,
----> 5     dataset=test_dataset,
      6     num_labels=NUM_LABELS,
      7     batch_size=BATCH_SIZE,

NameError: name 'test_dataset' is not defined

## === cell 10
submission_path = os.path.join(OUTPUT_PATH, "submission.csv")
test_results.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3321823371.py in <cell line: 0>()
      1 submission_path = os.path.join(OUTPUT_PATH, "submission.csv")
----> 2 test_results.to_csv(submission_path, index=False)
      3 print(f"Submission saved to {submission_path}")

NameError: name 'test_results' is not defined
