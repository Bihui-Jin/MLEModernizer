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

0.8017657349561782

# 6. Current score

0.06679

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.03116) has done: 'The main blocker is that `MODEL_PATH` points to a Kaggle dataset that is not available in your environment, so the tokenizer/model never load and downstream cells fail (`tokenizer`/`test_dataset` undefined). I fix this by making model resolution robust: if the local directory is missing, it fall back to a small, commonly-cached Hugging Face model name (and only then error with a clear message), keeping the rest of your inference logic unchanged. I also renumber the cells to start at 1 (your original started at cell 0) and ensure the model is moved to the selected device. Finally, the script always write `/kaggle/working/submission.csv` with the required `essay_id,score` columns.'
- What this solution (achieved -0.0365) has done: 'I fix the crash in model loading caused by an incompatible protobuf runtime by forcing Transformers to avoid protobuf-based tokenizers (SentencePiece) and by removing the protobuf environment override that triggers the `MessageFactory.GetPrototype` error. I also make the model fallback choose a DeBERTa-style checkpoint when available (instead of `distilbert-base-uncased`), which should move your QWK score up substantially toward the target while preserving your inference/chunking/aggregation logic. Finally, I add a small safety guard to ensure the model has a valid `pad_token_id` and the output submission is always aligned to `sample_submission.csv` with correct columns and integer scores 1–6.'
- What this solution (achieved 0.0) has done: 'You’re crashing in `AutoTokenizer.from_pretrained()` due to a protobuf API mismatch (`MessageFactory.GetPrototype`) that can be triggered by fast tokenizers / sentencepiece plumbing even when you try to disable it. I fix this by forcing the safe, pure-Python (slow) tokenizer path (`use_fast=False`) and by ensuring we pick a fallback checkpoint that is DeBERTa-based but *doesn’t* require SentencePiece, which should also improve QWK toward your target compared to DistilBERT. I also add a small defensive try/except to automatically retry with the slow tokenizer if any tokenizer load still fails, without changing your chunking/aggregation or prediction logic. Finally, I renumber cells to start at 1 and keep writing `/kaggle/working/submission.csv` with correct `essay_id,score` columns aligned to `sample_submission.csv`.'
- What this solution (achieved 0.0) has done: 'The crash happens before any inference because `transformers` is importing an incompatible protobuf runtime (the `MessageFactory.GetPrototype` AttributeError). The smallest robust fix is to proactively avoid protobuf usage by forcing the Python protobuf implementation and (if needed) uninstalling/neutralizing protobuf at runtime, then loading the tokenizer/model in a way that doesn’t require SentencePiece (keeping your model/inference/chunking logic unchanged). I also add a safe fallback model choice that is WordPiece-based (no SentencePiece) if your intended local model can’t be loaded, so the notebook always runs end-to-end and writes `/kaggle/working/submission.csv`. No changes are made to your chunking, aggregation, label mapping (0..5 -> 1..6), or submission formatting beyond ensuring the pipeline completes.'
- What this solution (achieved -0.19853) has done: 'I fix the protobuf-related crash by removing the forced protobuf environment overrides (they can trigger the `MessageFactory.GetPrototype` mismatch) and by making model/tokenizer loading avoid SentencePiece-dependent checkpoints. To keep your core inference/chunking/aggregation logic intact, I only change the model resolution and loading robustness, preferring a locally-available DeBERTa *AES* checkpoint when present, otherwise falling back to a WordPiece model that doesn’t need SentencePiece. I also renumber cells to start at 1 (Kaggle-friendly) and ensure the script always writes `/kaggle/working/submission.csv` with the required `essay_id,score` columns aligned to `sample_submission.csv`. These changes are aimed at unblocking execution and improving score from 0.0 (no valid run) toward your target.'
- What this solution (achieved -0.02205) has done: 'The crash happens before inference because `transformers` triggers a protobuf API mismatch (`MessageFactory.GetPrototype`) during import/tokenizer/model loading. The minimal robust fix is to force the pure-Python protobuf implementation *before* importing `transformers`, and to defensively set `TRANSFORMERS_NO_TF/NO_FLAX` to avoid extra imports that can indirectly pull problematic protobuf code paths. I’m keeping your exact chunking + mean-aggregation + rounding logic unchanged, only making model/tokenizer loading reliably work in this environment and ensuring the submission is written as `/kaggle/working/submission.csv` with `essay_id,score`. This should unblock execution and move performance up from the current broken/degenerate run toward the target by allowing the intended local DeBERTa AES checkpoint to load when present.'
- What this solution (achieved 0.01958) has done: 'The crash is coming from an incompatible protobuf runtime being imported indirectly when `transformers` loads tokenizer/model; setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` alone isn’t sufficient if `google.protobuf` has already been imported, or if the C++ implementation is still used. I force the pure-Python protobuf backend before any `transformers` import and proactively remove already-imported `google.protobuf` modules so the correct implementation is used, which should eliminate the `MessageFactory.GetPrototype` error and let inference run end-to-end. I also keep your model/tokenizer fallback logic intact, but make the tokenizer/model loaders retry with `local_files_only=True/False` in a safer order without changing your chunking/aggregation/scoring logic. Finally, I ensure a valid `/kaggle/working/submission.csv` is always written with `essay_id,score` aligned to `sample_submission.csv`.'
- What this solution (achieved -0.03642) has done: 'We need to fix the protobuf `MessageFactory.GetPrototype` crash that happens during/after importing `transformers` by ensuring the pure-Python protobuf backend is selected *before* any protobuf-dependent imports and by forcing the python implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` plus `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2`, then reloading any already-imported protobuf modules. This is a minimal, execution-unblocking change that preserves your model/inference/chunking logic and should substantially improve score versus the current near-random run by allowing the intended DeBERTa AES checkpoint to actually load. I also add a small hardening in model/tokenizer loading: if the initially resolved model dir still triggers protobuf issues, we fall back to a WordPiece-only checkpoint (BERT) without changing downstream logic. Finally, the script still always writes `/kaggle/working/submission.csv` with correct `essay_id,score` columns aligned to `sample_submission.csv`.'
- What this solution (achieved 0.06679) has done: 'The crash is coming from an incompatible protobuf runtime being imported before/while `transformers` loads tokenizers/models, and your current try/except won’t catch it because the `AttributeError` is raised during the `transformers` import itself. I force the pure-Python protobuf implementation earlier and proactively remove already-imported protobuf modules before importing `transformers`, which unblocks model/tokenizer loading and allows the intended local AES checkpoint to be used (big score improvement vs the current broken/degenerate run). I also make the fallback ranking prefer DeBERTa-v3 AES directories more strongly (still same inference/chunking/aggregation), and keep the submission writing aligned to `sample_submission.csv` with integer scores 1–6. No changes are made to your chunking scheme, aggregation (mean + round), or label mapping beyond ensuring the pipeline can actually run end-to-end.'

# 9. Code solution

## === cell 0
TRAIN_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
TEST_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
MODEL_PATH = "/kaggle/input/aes2-debertav3large/transformers/default/1"
OUTPUT_PATH = "/kaggle/working/"

MAX_LEN = 512
OVERLAP = 64
MIN_CHUNK_RATIO = 0.3  # Skip if this chunk < RATIO * MAX_LEN
BATCH_SIZE = 2
NUM_LABELS = 6




## === cell 1
import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

for k in list(sys.modules.keys()):
    if k.startswith("google.protobuf"):
        sys.modules.pop(k, None)

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())




## === cell 2
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def _resolve_model_source(model_path: str) -> str:
    """
    Prefer the user-provided MODEL_PATH if it exists.
    Otherwise, search /kaggle/input for a local HF model directory containing config.json.
    Prefer DeBERTa-v3 + AES checkpoints if multiple are found (likely best for this task).
    Finally, fall back to a HF model name that does NOT require SentencePiece.
    """
    if (
        isinstance(model_path, str)
        and os.path.isdir(model_path)
        and os.path.exists(os.path.join(model_path, "config.json"))
    ):
        return model_path

    base = "/kaggle/input"
    candidates = []
    for root, _, files in os.walk(base):
        if "config.json" in files:
            candidates.append(root)

    if candidates:

        def _rank(p: str) -> tuple:
            pl = p.lower()
            return (
                0 if "aes" in pl else 1,
                (
                    0
                    if ("deberta" in pl or "debertav3" in pl or "deberta-v3" in pl)
                    else 1
                ),
                0 if ("v3" in pl) else 1,
                0 if "large" in pl else 1,
                len(p),
                p,
            )

        candidates = sorted(candidates, key=_rank)
        print(f"WARNING: MODEL_PATH not found or missing config.json: {model_path}")
        print(f"WARNING: Falling back to detected HF model dir: {candidates[0]}")
        return candidates[0]

    fallback_model_name = "bert-base-uncased"
    print(
        f"WARNING: MODEL_PATH not found and no local HF model dir discovered under {base}."
    )
    print(f"WARNING: Falling back to HF model name: {fallback_model_name}")
    return fallback_model_name


def _safe_load_tokenizer(model_source: str, local_only: bool):
    try:
        return AutoTokenizer.from_pretrained(
            model_source, local_files_only=local_only, use_fast=False
        )
    except Exception as e1:
        print("WARNING: Tokenizer load failed for source:", model_source)
        print("Original error:", repr(e1))
        try:
            return AutoTokenizer.from_pretrained(
                model_source, local_files_only=False, use_fast=False
            )
        except Exception as e2:
            print("WARNING: Tokenizer retry (local_files_only=False) failed.")
            print("Retry error:", repr(e2))

        fallback_no_sp = "bert-base-uncased"
        print("WARNING: Retrying tokenizer load with fallback:", fallback_no_sp)
        return AutoTokenizer.from_pretrained(
            fallback_no_sp, local_files_only=False, use_fast=False
        )


def _safe_load_model(model_source: str, local_only: bool, num_labels: int):
    try:
        return AutoModelForSequenceClassification.from_pretrained(
            model_source,
            local_files_only=local_only,
            num_labels=num_labels,
        )
    except Exception as e1:
        print("WARNING: Model load failed for source:", model_source)
        print("Original error:", repr(e1))
        try:
            return AutoModelForSequenceClassification.from_pretrained(
                model_source,
                local_files_only=False,
                num_labels=num_labels,
            )
        except Exception as e2:
            print("WARNING: Model retry (local_files_only=False) failed.")
            print("Retry error:", repr(e2))

        fallback_no_sp = "bert-base-uncased"
        print("WARNING: Retrying model load with fallback:", fallback_no_sp)
        return AutoModelForSequenceClassification.from_pretrained(
            fallback_no_sp,
            local_files_only=False,
            num_labels=num_labels,
        )


resolved_model_source = _resolve_model_source(MODEL_PATH)
local_only = os.path.isdir(resolved_model_source)

try:
    tokenizer = _safe_load_tokenizer(resolved_model_source, local_only=local_only)
except Exception as e:
    if "GetPrototype" in str(e) or "protobuf" in str(e).lower():
        print(
            "WARNING: Protobuf-related error during tokenizer load; falling back to bert-base-uncased"
        )
        resolved_model_source = "bert-base-uncased"
        local_only = False
        tokenizer = _safe_load_tokenizer(resolved_model_source, local_only=local_only)
    else:
        raise

if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

try:
    model = _safe_load_model(
        resolved_model_source,
        local_only=local_only,
        num_labels=NUM_LABELS,
    )
except Exception as e:
    if "GetPrototype" in str(e) or "protobuf" in str(e).lower():
        print(
            "WARNING: Protobuf-related error during model load; falling back to bert-base-uncased"
        )
        resolved_model_source = "bert-base-uncased"
        local_only = False
        model = _safe_load_model(
            resolved_model_source,
            local_only=local_only,
            num_labels=NUM_LABELS,
        )
    else:
        raise

if hasattr(model, "resize_token_embeddings"):
    model.resize_token_embeddings(len(tokenizer))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print("Using device:", device)
print("Resolved model source:", resolved_model_source)
print("Local files only:", local_only)
print("Tokenizer name_or_path:", getattr(tokenizer, "name_or_path", "unknown"))
print("Tokenizer pad_token_id:", tokenizer.pad_token_id)




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
def compute_metrics(preds, labels, device="cuda"):
    if len(preds.shape) > 1:
        preds = np.argmax(preds, axis=-1)
    qwk_score = cohen_kappa_score(labels, preds, weights="quadratic")
    acc_score = accuracy_score(labels, preds)
    return {
        "quadratic_weighted_kappa": qwk_score,
        "accuracy": acc_score,
    }




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
            pred_labels = (
                torch.argmax(logits, axis=1).cpu().numpy() + 1
            )  # map 0..5 -> 1..6

            preds.extend(pred_labels)
            essay_ids_all.extend(essay_ids)

    preds = np.array(preds)
    essay_ids_all = np.array(essay_ids_all)

    chunk_results = pd.DataFrame({"essay_id": essay_ids_all, "raw_prediction": preds})

    aggregated_predictions = chunk_results.groupby("essay_id")["raw_prediction"].mean()
    y_pred_aggregated = np.rint(aggregated_predictions.values)
    final_scores = np.clip(y_pred_aggregated, 1, num_labels).astype(int)

    final_results_df = pd.DataFrame(
        {"essay_id": aggregated_predictions.index.values, "score": final_scores}
    )

    return final_results_df, preds




## === cell 6
class TestEssayDataset(Dataset):
    def __init__(
        self,
        df,
        tokenizer,
        max_len: int = 512,
        overlap: int = 128,
        min_chunk_ratio: float = 0.3,
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.overlap = overlap
        self.min_chunk_ratio = min_chunk_ratio

        num_special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
        self.max_content_len = max_len - num_special_tokens

        for _, row in df.iterrows():
            essay_id = row["essay_id"]
            text = "[B] " + row["full_text"]
            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]

            step = self.max_content_len - overlap
            start = 0

            while start < len(tokens):
                end = start + self.max_content_len
                chunk_content = tokens[start:end]
                if len(chunk_content) < self.max_content_len * self.min_chunk_ratio:
                    break

                processed_tokens = tokenizer.build_inputs_with_special_tokens(
                    chunk_content
                )
                padding_len = self.max_len - len(processed_tokens)
                if padding_len > 0:
                    processed_tokens += [tokenizer.pad_token_id] * padding_len

                self.samples.append((processed_tokens, essay_id))
                start += step

                if end >= len(tokens):
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids, essay_id = self.samples[idx]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "essay_id": essay_id,
        }




## === cell 7
df_test = pd.read_csv(TEST_PATH)
test_dataset = TestEssayDataset(df_test, tokenizer, MAX_LEN, OVERLAP, MIN_CHUNK_RATIO)
print("Test rows:", len(df_test), "Chunks:", len(test_dataset))




## === cell 8
test_results, _ = predict_essay_score(
    dataset=test_dataset,
    model=model,
    num_labels=NUM_LABELS,
    batch_size=BATCH_SIZE,
    device=device,
)

print("Submission preview:")
print(test_results.head())
print("Submission shape:", test_results.shape)

sample_path = (
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    test_results = sample[["essay_id"]].merge(test_results, on="essay_id", how="left")
    test_results["score"] = test_results["score"].fillna(3).astype(int)

test_results["score"] = np.clip(test_results["score"].astype(int), 1, NUM_LABELS)

out_file = os.path.join(OUTPUT_PATH, "submission.csv")
test_results.to_csv(out_file, index=False)
print("Wrote:", out_file)
print(test_results.describe(include="all"))
