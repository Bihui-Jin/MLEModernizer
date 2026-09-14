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

- What this solution (achieved -0.12599) has done: 'I fix the two root causes preventing an end-to-end run: (1) the protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation before importing `transformers`, and (2) the Hugging Face loader failing because the provided `MODEL_PATH` isn’t a valid local `from_pretrained` directory in this environment. To keep core logic intact while ensuring a submission is produced, I add a robust local-path resolver and a safe fallback to a public DeBERTa model if the Kaggle input model folder is missing/unloadable. I also make sure the prediction code always returns a correctly formatted `submission.csv` with `essay_id,score` aligned to the test set order. No changes are made to the chunking/prediction aggregation logic beyond making it run reliably.'
- What this solution (achieved 0.0) has done: 'You’re currently crashing at import time due to an incompatibility between `transformers` and the installed protobuf runtime (`MessageFactory.GetPrototype`), so the first fix is to force a protobuf version that works in Kaggle by uninstalling `protobuf`/`googleapis-common-protos` and installing a compatible protobuf (3.20.x) before importing `transformers`. Next, your predictions are likely badly calibrated for QWK because you’re using a 6-class classifier head with `argmax+1` on a base model that isn’t your intended fine-tuned checkpoint; to move the score toward the target, we keep the same inference logic but make the local model-path resolver actually find and load the Kaggle-provided checkpoint reliably (and avoid falling back to an unfine-tuned public model). Finally, we ensure the written file is exactly `submission.csv` with `essay_id,score` aligned to the test order and all scores clipped to 1–6.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with an “invalid submission” or a badly misaligned prediction scale, so the smallest safe improvement is to guarantee the model you load is the intended fine-tuned local checkpoint and that the written `submission.csv` exactly matches the required schema and row order. I keep your chunking + mean-aggregation + rounding core logic unchanged, but I (1) strengthen the local model directory resolver to correctly handle Kaggle dataset layouts (including `snapshots/...`), and (2) enforce `id2label/label2id`-aware mapping so we don’t accidentally shift labels with `+1` when the checkpoint already encodes labels 1–6. Finally, I add a lightweight sanity check that the submission has exactly `len(test)` rows, the right columns, and scores are in [1,6], so you don’t end up with a 0.0 due to formatting/alignment issues.'

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

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    DebertaV2Tokenizer,
    DebertaV2ForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def _resolve_local_model_dir(model_path: str) -> str | None:
    """
    Ensure we reliably find the actual local fine-tuned HF checkpoint directory.
    Handles common Kaggle layouts including:
      - transformers/default/1
      - snapshots/<hash> (from HF cache-like exports)
    """
    if not model_path:
        return None

    candidates = []
    for p in [
        model_path,
        os.path.dirname(model_path),
        os.path.dirname(os.path.dirname(model_path)),
    ]:
        if p and os.path.isdir(p):
            candidates.append(p)

    probes = [
        model_path,
        os.path.join(model_path, "transformers", "default", "1"),
        os.path.join(model_path, "transformers", "default"),
        os.path.join(model_path, "transformers"),
        os.path.join(model_path, "1"),
    ]
    for p in probes:
        if p and os.path.isdir(p):
            candidates.append(p)

    seen = set()
    uniq = []
    for c in candidates:
        c = os.path.abspath(c)
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    def looks_like_hf_dir(d: str) -> bool:
        has_config = os.path.isfile(os.path.join(d, "config.json"))
        has_weights = any(
            os.path.isfile(os.path.join(d, fn))
            for fn in ["pytorch_model.bin", "model.safetensors"]
        )
        return has_config and has_weights

    for c in uniq:
        if looks_like_hf_dir(c):
            return c

    for c in uniq:
        snap_root = os.path.join(c, "snapshots")
        if os.path.isdir(snap_root):
            try:
                for sub in sorted(os.listdir(snap_root)):
                    d = os.path.join(snap_root, sub)
                    if os.path.isdir(d) and looks_like_hf_dir(d):
                        return d
            except Exception:
                pass

        try:
            for root, dirs, files in os.walk(c):
                rel_depth = os.path.relpath(root, c).count(os.sep)
                if rel_depth > 6:
                    dirs[:] = []
                    continue
                if "config.json" in files and (
                    "pytorch_model.bin" in files or "model.safetensors" in files
                ):
                    return root
        except Exception:
            continue

    return None


def _load_tokenizer_and_model_strict(model_path: str):
    """
    Fix (score): load ONLY the intended local fine-tuned checkpoint.
    If we silently fall back to a public base model, QWK typically collapses; failing fast is safer than scoring 0.0.
    """
    resolved = _resolve_local_model_dir(model_path)
    if resolved is None:
        raise FileNotFoundError(
            f"Could not resolve a local HF checkpoint dir from MODEL_PATH={model_path}"
        )

    try:
        tok = AutoTokenizer.from_pretrained(
            resolved, local_files_only=True, use_fast=True
        )
        mdl = AutoModelForSequenceClassification.from_pretrained(
            resolved, local_files_only=True
        )
        print(f"Loaded local model via Auto* from: {resolved}")
        return tok, mdl
    except Exception as e_auto:
        print(f"[WARN] Local Auto* load failed ({resolved}): {repr(e_auto)}")

    tok = DebertaV2Tokenizer.from_pretrained(resolved, local_files_only=True)
    mdl = DebertaV2ForSequenceClassification.from_pretrained(
        resolved, local_files_only=True
    )
    print(f"Loaded local model via DebertaV2* from: {resolved}")
    return tok, mdl


tokenizer, model = _load_tokenizer_and_model_strict(MODEL_PATH)

if hasattr(model, "config") and getattr(model.config, "num_labels", None) is not None:
    NUM_LABELS = int(model.config.num_labels)
print(f"Using NUM_LABELS={NUM_LABELS}")




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1454424695.py in <cell line: 0>()
    109 
    110 
--> 111 tokenizer, model = _load_tokenizer_and_model_strict(MODEL_PATH)
    112 
    113 # Fix (score): align NUM_LABELS to the checkpoint head when available to avoid incorrect clipping/mapping.

/tmp/ipykernel_55/1454424695.py in _load_tokenizer_and_model_strict(model_path)
     83     resolved = _resolve_local_model_dir(model_path)
     84     if resolved is None:
---> 85         raise FileNotFoundError(
     86             f"Could not resolve a local HF checkpoint dir from MODEL_PATH={model_path}"
     87         )

FileNotFoundError: Could not resolve a local HF checkpoint dir from MODEL_PATH=/kaggle/input/aes2-debertav3large/transformers/default/1

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
def _infer_label_mapping(mdl, num_labels: int):
    """
    Avoid incorrect '+1' shifting when the checkpoint's id2label already maps
    indices to 1..6 or otherwise non-0-based labels. If mapping is unclear, fall back to +1.
    Returns a function: idx_array -> score_array (int).
    """
    cfg = getattr(mdl, "config", None)
    id2label = getattr(cfg, "id2label", None) if cfg is not None else None

    def default_map(idxs: np.ndarray) -> np.ndarray:
        return idxs.astype(int) + 1

    if not isinstance(id2label, dict) or len(id2label) != num_labels:
        return default_map

    try:
        keys = sorted(int(k) for k in id2label.keys())
        labels = [id2label[str(k)] if str(k) in id2label else id2label[k] for k in keys]

        numeric = []
        for lab in labels:
            if isinstance(lab, (int, np.integer)):
                numeric.append(int(lab))
            elif isinstance(lab, str) and lab.strip().isdigit():
                numeric.append(int(lab.strip()))
            else:
                numeric = []
                break

        if len(numeric) == num_labels and sorted(numeric) == list(
            range(1, num_labels + 1)
        ):
            mapping = {k: v for k, v in zip(keys, numeric)}

            def mapped(idxs: np.ndarray) -> np.ndarray:
                return np.vectorize(lambda z: mapping.get(int(z), int(z) + 1))(
                    idxs
                ).astype(int)

            print("[INFO] Using config.id2label-based mapping (no forced +1 shift).")
            return mapped
    except Exception:
        pass

    return default_map


def predict_essay_score(model, dataset, num_labels, batch_size, device):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    model.eval()

    map_fn = _infer_label_mapping(model, num_labels)

    preds, essay_ids_all = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            essay_ids = batch["essay_id"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            pred_class = torch.argmax(logits, axis=1).cpu().numpy()
            pred_labels = map_fn(pred_class)

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
            text = "[A] " + row["full_text"]
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

test_results, _ = predict_essay_score(
    dataset=test_dataset,
    model=model,
    num_labels=NUM_LABELS,
    batch_size=BATCH_SIZE,
    device=device,
)

test_results = df_test[["essay_id"]].merge(test_results, on="essay_id", how="left")
test_results["score"] = test_results["score"].fillna(3).astype(int)
test_results["score"] = test_results["score"].clip(1, NUM_LABELS)

test_results = test_results[["essay_id", "score"]]
assert len(test_results) == len(df_test), "Submission row count mismatch with test.csv"
assert list(test_results.columns) == [
    "essay_id",
    "score",
], "Submission columns incorrect"

print("Submission head:")
print(test_results.head())
print(f"Rows: {len(test_results)} (expected {len(df_test)})")
print("Score value counts:")
print(test_results["score"].value_counts().sort_index())



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1922158759.py in <cell line: 0>()
      1 df_test = pd.read_csv(TEST_PATH)
      2 
----> 3 test_dataset = TestEssayDataset(df_test, tokenizer, MAX_LEN, OVERLAP, MIN_CHUNK_RATIO)
      4 
      5 test_results, _ = predict_essay_score(

NameError: name 'tokenizer' is not defined

## === cell 8
sub_path = os.path.join(OUTPUT_PATH, "submission.csv")
test_results.to_csv(sub_path, index=False)

check = pd.read_csv(sub_path)
assert len(check) == len(df_test), "Written CSV row count mismatch"
assert list(check.columns) == ["essay_id", "score"], "Written CSV columns mismatch"
assert (
    check["score"].between(1, NUM_LABELS).all()
), "Written CSV has out-of-range scores"

print(f"Wrote submission to: {sub_path}")
print(check.head())

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1966351938.py in <cell line: 0>()
      1 sub_path = os.path.join(OUTPUT_PATH, "submission.csv")
----> 2 test_results.to_csv(sub_path, index=False)
      3 
      4 check = pd.read_csv(sub_path)
      5 assert len(check) == len(df_test), "Written CSV row count mismatch"

NameError: name 'test_results' is not defined
