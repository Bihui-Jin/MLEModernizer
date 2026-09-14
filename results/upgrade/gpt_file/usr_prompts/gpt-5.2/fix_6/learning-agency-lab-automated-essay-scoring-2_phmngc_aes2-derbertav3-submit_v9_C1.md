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

0.7855532689009783

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the import-time crash caused by an incompatible protobuf runtime by forcing the pure-Python protobuf implementation before importing `transformers`. Then I make model loading robust to Kaggle dataset directory layouts by auto-detecting the actual local model folder (falling back to a known public DeBERTa checkpoint if the provided path doesn’t exist), and I ensure `local_files_only=True` when loading from disk. Finally, I make device selection safe (CPU fallback if no CUDA) and write the submission to `/kaggle/working/submission.csv` with the required columns and row alignment to the test `essay_id`s.'
- What this solution (achieved 0.0) has done: 'I fix the import-time crash coming from an incompatible protobuf C++ runtime by forcing the pure-Python protobuf implementation and (crucially) restarting the protobuf module state before importing `transformers`. I also make the `transformers` import itself more robust by setting the env var earlier and avoiding optional imports that can trigger protobuf usage. After that, I keep your model/dataset/prediction logic unchanged so the score behavior stays consistent, and ensure the submission is always written to `/kaggle/working/submission.csv` with the required `essay_id,score` columns and correct row alignment.'
- What this solution (achieved 0.0) has done: 'You’re crashing during `transformers` import due to a protobuf API mismatch (`MessageFactory.GetPrototype`), so I prevent the protobuf-backed fast tokenizer and any protobuf-dependent paths from being used by setting safe environment flags before importing `transformers`. Then I switch to `AutoTokenizer/AutoModelForSequenceClassification` with `use_fast=False` and keep `local_files_only=True` for the local checkpoint to preserve your exact inference logic while avoiding the protobuf failure. Finally, I keep the chunking/aggregation and submission writing unchanged, only adding small guards so the code reliably produces `/kaggle/working/submission.csv` aligned to `test.csv` essay_ids.'
- What this solution (achieved 0.0) has done: 'You’re failing during `transformers` import/model loading because the protobuf runtime in the environment is incompatible with what `transformers` expects (`MessageFactory.GetPrototype`). I fix this by forcing the pure-Python protobuf implementation **and** disabling optional protobuf-backed code paths in `transformers` (notably the new protobuf conversion utilities) before importing `transformers`. I also make model loading more robust by preferring a local model directory when available, otherwise falling back to a public checkpoint with `local_files_only=False` only for the fallback case. Finally, I keep your chunking/aggregation and submission writing logic the same, ensuring `/kaggle/working/submission.csv` is always produced with `essay_id,score`.'
- What this solution (achieved 0.0) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by ensuring the protobuf Python implementation is forced *before* any protobuf/transformers-dependent imports and by preventing `transformers` from importing protobuf-backed optional modules (via a safe `google.protobuf` stub only if needed). This keeps your core inference logic (tokenization, chunking, DeBERTa sequence classification logits → argmax → mean aggregation → rounding/clipping) unchanged, but makes the notebook run end-to-end in the Kaggle environment. I also make model-path resolution more robust and ensure `submission.csv` is always written with the required `essay_id,score` columns aligned to `test.csv`. These changes are execution/stability fixes; they should move the score from 0.0 (broken submission) toward a valid QWK score.'

# 9. Code solution

## === cell 0
TRAIN_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
TEST_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
MODEL_PATH = "/kaggle/input/aes2-debertav3-large/transformers/default/1"
OUTPUT_PATH = "/kaggle/working/"

MAX_LEN = 512
OVERLAP = 64
MIN_CHUNK_RATIO = 0.3  # Skip if this chunk < RATIO * MAX_LEN
BATCH_SIZE = 2
NUM_LABELS = 6




## === cell 1
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_FAST_TOKENIZER", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import sys
import importlib
import types

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]


def _safe_import_transformers():
    try:
        import transformers  # noqa: F401

        return
    except AttributeError as e:
        if "GetPrototype" not in str(e):
            raise

        google_mod = sys.modules.get("google") or types.ModuleType("google")
        protobuf_mod = types.ModuleType("google.protobuf")
        message_mod = types.ModuleType("google.protobuf.message")
        descriptor_mod = types.ModuleType("google.protobuf.descriptor")
        symbol_database_mod = types.ModuleType("google.protobuf.symbol_database")
        json_format_mod = types.ModuleType("google.protobuf.json_format")

        class _DummyMessage:  # minimal placeholder
            pass

        message_mod.Message = _DummyMessage

        class _DummySymbolDatabase:
            def GetPrototype(self, *args, **kwargs):
                raise AttributeError("GetPrototype not available in stub")

        def Default():
            return _DummySymbolDatabase()

        symbol_database_mod.Default = Default

        def MessageToDict(*args, **kwargs):
            raise RuntimeError(
                "protobuf json_format is not available in this environment"
            )

        def ParseDict(*args, **kwargs):
            raise RuntimeError(
                "protobuf json_format is not available in this environment"
            )

        json_format_mod.MessageToDict = MessageToDict
        json_format_mod.ParseDict = ParseDict

        protobuf_mod.message = message_mod
        protobuf_mod.descriptor = descriptor_mod
        protobuf_mod.symbol_database = symbol_database_mod
        protobuf_mod.json_format = json_format_mod

        google_mod.protobuf = protobuf_mod

        sys.modules["google"] = google_mod
        sys.modules["google.protobuf"] = protobuf_mod
        sys.modules["google.protobuf.message"] = message_mod
        sys.modules["google.protobuf.descriptor"] = descriptor_mod
        sys.modules["google.protobuf.symbol_database"] = symbol_database_mod
        sys.modules["google.protobuf.json_format"] = json_format_mod

        import transformers  # noqa: F401

        return


_safe_import_transformers()

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

torch.manual_seed(42)
np.random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)




## === cell 2
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def _find_local_hf_model_dir(preferred_path: str) -> str | None:
    if preferred_path and os.path.isdir(preferred_path):
        if os.path.exists(os.path.join(preferred_path, "config.json")) and any(
            os.path.exists(os.path.join(preferred_path, w))
            for w in ("pytorch_model.bin", "model.safetensors")
        ):
            return preferred_path

        for root, dirs, files in os.walk(preferred_path):
            if "config.json" in files and any(
                f in files for f in ("pytorch_model.bin", "model.safetensors")
            ):
                return root

    base = "/kaggle/input"
    if not os.path.isdir(base):
        return None

    candidates = []
    for root, dirs, files in os.walk(base):
        if "config.json" in files and any(
            f in files for f in ("pytorch_model.bin", "model.safetensors")
        ):
            candidates.append(root)

    if candidates:

        def score(p: str) -> int:
            s = 0
            pl = p.lower()
            if "transformers" in pl:
                s += 2
            if "deberta" in pl:
                s += 2
            if "aes" in pl:
                s += 1
            if "large" in pl:
                s += 1
            return s

        candidates = sorted(candidates, key=score, reverse=True)
        return candidates[0]

    return None


local_model_dir = _find_local_hf_model_dir(MODEL_PATH)

if local_model_dir is not None:
    print("Loading model from local path:", local_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_dir, local_files_only=True, use_fast=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        local_model_dir, num_labels=NUM_LABELS, local_files_only=True
    )
else:
    fallback_ckpt = "microsoft/deberta-v3-large"
    print("Local model not found; falling back to:", fallback_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(fallback_ckpt, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        fallback_ckpt, num_labels=NUM_LABELS
    )

model.to(device)
model.eval()




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
            pred_labels = torch.argmax(logits, axis=1).cpu().numpy() + 1  # labels 1..6

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

        self.num_special = tokenizer.num_special_tokens_to_add(pair=False)

        tag_tokens = tokenizer("[B]", add_special_tokens=False)["input_ids"]
        len_tag = len(tag_tokens)

        self.max_content_len = max_len - self.num_special - len_tag
        if self.max_content_len <= 8:
            raise ValueError(
                f"max_content_len too small ({self.max_content_len}); check MAX_LEN/tag/special tokens"
            )

        for _, row in df.iterrows():
            essay_id = row["essay_id"]
            base_tokens = tokenizer(row["full_text"], add_special_tokens=False)[
                "input_ids"
            ]

            step = self.max_content_len - overlap
            if step <= 0:
                step = self.max_content_len

            start = 0
            while start < len(base_tokens):
                end = start + self.max_content_len
                chunk = base_tokens[start:end]

                if (
                    len(chunk) < self.max_content_len * self.min_chunk_ratio
                    and start != 0
                ):
                    break

                chunk_with_tag = tag_tokens + chunk
                processed = tokenizer.build_inputs_with_special_tokens(chunk_with_tag)

                if len(processed) > self.max_len:
                    processed = processed[: self.max_len]

                pad_len = self.max_len - len(processed)
                if pad_len > 0:
                    processed += [tokenizer.pad_token_id] * pad_len

                self.samples.append((processed, essay_id))

                start += step
                if end >= len(base_tokens):
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
print("Num test essays:", len(df_test), "Num chunks:", len(test_dataset))




## === cell 8
test_results, _ = predict_essay_score(
    dataset=test_dataset,
    model=model,
    num_labels=NUM_LABELS,
    batch_size=BATCH_SIZE,
    device=device,
)

test_results = df_test[["essay_id"]].merge(test_results, on="essay_id", how="left")
test_results["score"] = test_results["score"].fillna(3).astype(int)

print("Submission head:")
print(test_results.head())
print("Rows:", len(test_results), "Cols:", list(test_results.columns))

sub_path = os.path.join(OUTPUT_PATH, "submission.csv")
test_results.to_csv(sub_path, index=False)
print("Saved submission to:", sub_path)
print(pd.read_csv(sub_path).head())
