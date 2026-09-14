# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.7925148136739251

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I fix the immediate runtime blocker by making model loading robust when the provided `MODEL_PATH` (and the assumed fallback DeBERTa dirs) are not actually present in the Kaggle input tree. The minimal safe fix is to (1) auto-discover any local Transformers checkpoint directory under `/kaggle/input` that contains both `config.json` and some weights file, and (2) fall back to a deterministic constant-score submission if no checkpoint exists, so a valid `submission.csv` is always produced. This preserves your core inference/chunking/aggregation logic whenever a model is available, and only uses the fallback path to avoid “no submission yielded”. Finally, I ensure the submission is aligned to `sample_submission.csv` and written with the required columns and `.csv` suffix.'

# 9. Code solution

## === cell 0
TRAIN_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
TEST_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
MODEL_PATH = "/kaggle/input/aes2-debertav3large/transformers/default/1"
OUTPUT_PATH = "/kaggle/working/"

MAX_LEN = 512
OVERLAP = 128
MIN_CHUNK_RATIO = 0.375  # Skip if this chunk < RATIO * MAX_LEN
BATCH_SIZE = 2
NUM_LABELS = 6




## === cell 1
import os

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())




## === cell 2
def _resolve_local_model_dir(path: str) -> str:
    """
    Resolve a directory that contains config.json (or return original path if not found).
    """
    path = os.path.expanduser(path)
    if os.path.isfile(path):
        return os.path.dirname(path)
    if os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json")):
        return path

    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            if "config.json" in files:
                return root

    return path


def _is_transformers_checkpoint_dir(d: str) -> bool:
    if not os.path.isdir(d):
        return False
    if not os.path.isfile(os.path.join(d, "config.json")):
        return False
    weight_files = [
        "pytorch_model.bin",
        "model.safetensors",
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
    ]
    return any(os.path.isfile(os.path.join(d, wf)) for wf in weight_files)


def _find_any_checkpoint_under(base_dir: str = "/kaggle/input", max_hits: int = 20):
    """
    Bugfix: the provided MODEL_PATH may not exist in this environment.
    As a robust fallback, search for any local HF checkpoint dir that has config.json + weights.
    """
    hits = []
    if not os.path.isdir(base_dir):
        return hits
    for root, dirs, files in os.walk(base_dir):
        if "config.json" in files:
            if _is_transformers_checkpoint_dir(root):
                hits.append(root)
                if len(hits) >= max_hits:
                    break
    return hits


def _pick_model_dir(primary_path: str) -> str | None:
    """
    Pick a usable local transformers checkpoint directory.
    Returns None if nothing usable is found.
    """
    candidates = [
        primary_path,
        _resolve_local_model_dir(primary_path),
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/deberta-v3-base",
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/deberta-v3-large",
    ]

    for cand in candidates:
        resolved = _resolve_local_model_dir(cand)
        if _is_transformers_checkpoint_dir(resolved):
            return resolved

    hits = _find_any_checkpoint_under("/kaggle/input")
    if hits:
        preferred = None
        for h in hits:
            hl = h.lower()
            if "deberta" in hl or "essay" in hl or "aes" in hl:
                preferred = h
                break
        return preferred or hits[0]

    return None


MODEL_DIR = _pick_model_dir(MODEL_PATH)
print("Resolved MODEL_DIR:", MODEL_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)




## === cell 3
tokenizer = None
model = None
MODEL_AVAILABLE = False

if MODEL_DIR is not None:
    load_errors = []
    try:
        cfg = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)
        cfg.num_labels = NUM_LABELS
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR, local_files_only=True, use_fast=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR,
            config=cfg,
            local_files_only=True,
        )
        MODEL_AVAILABLE = True
    except Exception as e:
        load_errors.append(("Auto*", repr(e)))
        MODEL_AVAILABLE = False

    if not MODEL_AVAILABLE:
        print("Model/tokenizer loading failed with errors:")
        for tag, err in load_errors:
            print(f" - {tag}: {err}")
        print("\nFiles in MODEL_DIR (top-level):")
        try:
            print(sorted(os.listdir(MODEL_DIR))[:120])
        except Exception as e:
            print("Could not list MODEL_DIR:", repr(e))
else:
    print(
        "No local Transformers checkpoint directory was found under MODEL_PATH or /kaggle/input."
    )

if MODEL_AVAILABLE:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))

    model = model.to(device)
    model.eval()

    print("Loaded tokenizer:", type(tokenizer).__name__)
    print("Loaded model:", type(model).__name__)




## === cell 4
def plot_confusion_matrix(preds, labels, title="Confusion Matrix"):
    preds = np.array(preds)
    labels = np.array(labels)

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Greens", values_format="d")
    plt.title(title)
    plt.show()




## === cell 5
def compute_metrics(preds, labels, device="cuda"):
    if len(preds.shape) > 1:
        preds = np.argmax(preds, axis=-1)
    qwk_score = cohen_kappa_score(labels, preds, weights="quadratic")
    acc_score = accuracy_score(labels, preds)
    return {"quadratic_weighted_kappa": qwk_score, "accuracy": acc_score}




## === cell 6
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
            pred_labels = torch.argmax(logits, axis=1).cpu().numpy() + 1

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




## === cell 7
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

        for _, row in df.iterrows():
            essay_id = row["essay_id"]
            base_tokens = tokenizer(row["full_text"], add_special_tokens=False)[
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




## === cell 8
df_test = pd.read_csv(TEST_PATH)

if MODEL_AVAILABLE:
    test_dataset = TestEssayDataset(
        df_test, tokenizer, MAX_LEN, OVERLAP, MIN_CHUNK_RATIO
    )
    print("Num test essays:", len(df_test), "Num chunks:", len(test_dataset))
else:
    test_dataset = None
    print(
        "MODEL_AVAILABLE=False -> will create fallback submission without running inference."
    )
    print("Num test essays:", len(df_test))




## === cell 9
sample_path = os.path.join(os.path.dirname(TEST_PATH), "sample_submission.csv")
sample_sub = (
    pd.read_csv(sample_path)
    if os.path.exists(sample_path)
    else pd.DataFrame({"essay_id": df_test["essay_id"].astype(str)})
)

if MODEL_AVAILABLE:
    test_results, _ = predict_essay_score(
        dataset=test_dataset,
        model=model,
        num_labels=NUM_LABELS,
        batch_size=BATCH_SIZE,
        device=device,
    )

    print("Submission preview (raw model):")
    print(test_results.head())
    print("Submission shape (raw model):", test_results.shape)

    test_results = sample_sub[["essay_id"]].merge(
        test_results, on="essay_id", how="left"
    )
    test_results["score"] = test_results["score"].fillna(3).astype(int)
else:
    test_results = sample_sub.copy()
    test_results["score"] = 3

test_results = test_results[["essay_id", "score"]]
test_results["essay_id"] = test_results["essay_id"].astype(str)
test_results["score"] = test_results["score"].astype(int)




## === cell 10
out_file = os.path.join(OUTPUT_PATH, "submission.csv")
test_results.to_csv(out_file, index=False)

print("Wrote:", out_file)
print(pd.read_csv(out_file).head())
print("Shape:", pd.read_csv(out_file).shape)
print("File exists:", os.path.exists(out_file), "Size:", os.path.getsize(out_file))
