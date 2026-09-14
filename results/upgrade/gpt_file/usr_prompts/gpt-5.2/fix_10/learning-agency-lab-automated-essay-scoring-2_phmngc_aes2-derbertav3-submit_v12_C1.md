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

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the immediate runtime blocker by making model loading robust when the provided `MODEL_PATH` (and the assumed fallback DeBERTa dirs) are not actually present in the Kaggle input tree. The minimal safe fix is to (1) auto-discover any local Transformers checkpoint directory under `/kaggle/input` that contains both `config.json` and some weights file, and (2) fall back to a deterministic constant-score submission if no checkpoint exists, so a valid `submission.csv` is always produced. This preserves your core inference/chunking/aggregation logic whenever a model is available, and only uses the fallback path to avoid “no submission yielded”. Finally, I ensure the submission is aligned to `sample_submission.csv` and written with the required columns and `.csv` suffix.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is almost certainly because the model checkpoint path doesn’t exist in this runtime, so you fall back to a constant “3” submission (which scores very poorly on QWK). The smallest improvement toward the target is to keep your exact inference/chunking/aggregation logic, but make the fallback load a real pretrained DeBERTa checkpoint that is already cached inside the `transformers`/`sentence-transformers` installs (so it works offline without extra files). This produce non-constant predictions and should move the score upward substantially without changing your model architecture or prediction semantics. If that fallback model is also unavailable locally, we keep your deterministic constant submission so a valid `submission.csv` is always produced.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from the constant-score fallback when no fine-tuned checkpoint is found; to move toward the target, we need non-constant, task-relevant predictions while preserving your exact chunking + mean aggregation + rounding semantics. The smallest safe improvement is to (1) correctly detect and load any *fine-tuned AES* checkpoint that exists anywhere under `/kaggle/input` (your current search is too generic and can miss or pick irrelevant checkpoints), and (2) if none exists, still run inference using an offline-cached DeBERTa backbone but add a minimal, deterministic per-essay calibration using the **train set class prior** (so outputs aren’t nearly random/constant). This keeps your model, inference loop, and post-processing intact, and only changes how we obtain logits/predictions in the fallback path to increase QWK from ~0 toward the target. The submission writing remains aligned to `sample_submission.csv` and always produces `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from generating a constant-score fallback when no fine-tuned checkpoint is found/loaded; to move toward the target, we need to actually load a local AES model if it exists and, if not, still produce a non-constant prediction using an offline-cached backbone. I make the model discovery prefer checkpoints whose `config.json` indicates a 6-label sequence classification head (much more likely to be the competition model), and I also enable loading from common Kaggle cache locations (including `/kaggle/working` and HF cache dirs) without changing your inference/chunking/aggregation logic. Finally, if we must fall back to a generic pretrained model, I keep your exact ranking→prior binning calibration but apply it globally per essay (after aggregation) to avoid chunk-level rank noise that can depress QWK, while still preserving the same overall semantics (ordinal mapping + rounding/clipping). The script still always write `/kaggle/working/submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from producing a constant-score submission when no fine-tuned checkpoint is found/loaded. The smallest change that should move you toward the 0.79 target (without changing your chunking/aggregation logic) is to (1) make model discovery *prefer and actually find* checkpoints under `/kaggle/input` that look like AES fine-tunes (6-label head, DeBERTa, and common Kaggle “transformers/default/1” layout), and (2) if only a base pretrained model is available, wrap it with a tiny deterministic 6-label classification head and run a short, fixed-epoch training on `train.csv` using the same tokenization/chunking scheme, so predictions are non-constant and task-aligned. This preserves your core inference semantics (chunk → model → aggregate → round/clip) and keeps the output format aligned to `sample_submission.csv`. The script still always writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from producing essentially non-task predictions when the fine-tuned checkpoint path is missing; to move toward the 0.7925 target with minimal logic changes, the smallest reliable fix is to ensure we *actually* load a real fine-tuned AES checkpoint from `/kaggle/input` (most Kaggle models live under their own dataset folder, and your current generic search can pick irrelevant cached checkpoints). I tighten checkpoint discovery to (1) preferentially search within the specific competition input tree and (2) strongly require `num_labels==6` (or clearly prefer it) so we don’t accidentally select unrelated models. If no 6-label checkpoint is found, we keep your existing fallback behavior, but we also prevent the “base model without 6 labels” path from silently producing weak scalar logits by always attempting the existing `ignore_mismatched_sizes=True` 6-label head construction first (still the same model class/approach). This keeps your chunking → model → mean aggregation → round/clip semantics intact while making it much more likely you generate a meaningful (non-constant) submission and thus improve QWK substantially from 0.0 toward the target.'

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


def _safe_read_num_labels(d: str):
    """
    Prefer checkpoints that actually have a 6-label head to avoid loading unrelated cached models.
    """
    try:
        cfg = AutoConfig.from_pretrained(d, local_files_only=True)
        return getattr(cfg, "num_labels", None)
    except Exception:
        return None


def _find_any_checkpoint_under(base_dir: str, max_hits: int = 200):
    """
    Robustly search for local HF checkpoint dirs that have config.json + weights.
    """
    hits = []
    if not os.path.isdir(base_dir):
        return hits
    for root, dirs, files in os.walk(base_dir):
        if "config.json" in files and _is_transformers_checkpoint_dir(root):
            hits.append(root)
            if len(hits) >= max_hits:
                break
    return hits


def _score_checkpoint_path(p: str, desired_num_labels: int = 6) -> int:
    """
    Change (relevance to score): make discovery more likely to pick the actual AES finetune.
    Heavily prefer num_labels==6 and common Kaggle transformers export layouts.
    """
    pl = p.lower()
    score = 0

    nl = _safe_read_num_labels(p)
    if nl == desired_num_labels:
        score += 300
    elif nl is None:
        score += 0
    else:
        score -= 200

    for key, w in [
        ("aes2", 30),
        ("automated-essay-scoring", 25),
        ("essay", 15),
        ("aes", 15),
        ("deberta", 10),
        ("transformers/default", 20),
        ("/transformers/default/1", 40),
        (os.path.sep + "transformers" + os.path.sep, 8),
        (os.path.sep + "default" + os.path.sep, 8),
        (os.path.sep + "1", 3),
        ("checkpoint", 2),
    ]:
        if key in pl:
            score += w

    for bad, w in [
        ("tokenizer", -5),
        ("vocab", -5),
        ("onnx", -8),
        ("logs", -5),
        ("wandb", -5),
    ]:
        if bad in pl:
            score += w

    return score


def _pick_model_dir(primary_path: str) -> str | None:
    """
    Change (relevance to score): prioritize finding a *6-label* AES finetune under /kaggle/input,
    especially within competition-related folders, to avoid falling back to constant/weak predictions.
    Returns None if nothing usable is found.
    """
    primary_path_resolved = _resolve_local_model_dir(primary_path)
    if _is_transformers_checkpoint_dir(primary_path_resolved):
        return primary_path_resolved

    preferred_roots = [
        "/kaggle/input",
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2",
    ]

    fallback_roots = [
        "/kaggle/working",
        os.path.expanduser("~/.cache/huggingface"),
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expanduser("~/.cache/transformers"),
    ]

    candidates = [
        primary_path,
        primary_path_resolved,
        "/kaggle/input/aes2-debertav3large/transformers/default/1",
    ]
    for cand in candidates:
        resolved = _resolve_local_model_dir(cand)
        if _is_transformers_checkpoint_dir(resolved):
            return resolved

    def collect_hits(roots):
        hits = []
        for base in roots:
            hits.extend(_find_any_checkpoint_under(base_dir=base, max_hits=400))
            if len(hits) >= 400:
                break
        return list(dict.fromkeys(hits))

    hits_pref = collect_hits(preferred_roots)
    hits_fallback = collect_hits(fallback_roots)

    all_hits = hits_pref + [h for h in hits_fallback if h not in set(hits_pref)]
    if not all_hits:
        return None

    all_hits_sorted = sorted(
        all_hits,
        key=lambda p: (_score_checkpoint_path(p, desired_num_labels=NUM_LABELS), p),
        reverse=True,
    )

    six_label_hits = [
        p for p in all_hits_sorted if _safe_read_num_labels(p) == NUM_LABELS
    ]
    if six_label_hits:
        return six_label_hits[0]

    return all_hits_sorted[0]


MODEL_DIR = _pick_model_dir(MODEL_PATH)
print("Resolved MODEL_DIR (auto-discovered):", MODEL_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)



## === cell 3
FALLBACK_MODEL_CANDIDATES = [
    "microsoft/deberta-v3-base",
    "microsoft/deberta-v3-large",
    "microsoft/deberta-base",
    "microsoft/deberta-large",
]

tokenizer = None
model = None
MODEL_AVAILABLE = False
MODEL_SOURCE = None

load_errors = []

if MODEL_DIR is not None:
    try:
        cfg = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR, local_files_only=True, use_fast=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR,
            config=cfg,
            local_files_only=True,
        )

        MODEL_AVAILABLE = True
        MODEL_SOURCE = f"local_dir:{MODEL_DIR}"
        print("Loaded model from local directory:", MODEL_DIR)
        print("Checkpoint num_labels:", getattr(model.config, "num_labels", None))
    except Exception as e:
        load_errors.append((f"local_dir:{MODEL_DIR}", repr(e)))
        MODEL_AVAILABLE = False
        MODEL_SOURCE = None

if not MODEL_AVAILABLE:
    for name in FALLBACK_MODEL_CANDIDATES:
        try:
            cfg = AutoConfig.from_pretrained(name, local_files_only=True)

            tokenizer = AutoTokenizer.from_pretrained(
                name, local_files_only=True, use_fast=True
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                name, config=cfg, local_files_only=True
            )
            MODEL_AVAILABLE = True
            MODEL_SOURCE = f"fallback_name:{name}"
            print("Loaded fallback pretrained model (offline cache):", name)
            print("Fallback num_labels:", getattr(model.config, "num_labels", None))
            break
        except Exception as e:
            load_errors.append((f"fallback_name:{name}", repr(e)))
            MODEL_AVAILABLE = False
            MODEL_SOURCE = None

if not MODEL_AVAILABLE:
    print("Model/tokenizer loading failed. Errors tried (most recent last):")
    for tag, err in load_errors[-10:]:
        print(f" - {tag}: {err}")

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
    print("MODEL_SOURCE:", MODEL_SOURCE)




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
def _load_train_prior(train_path: str, num_labels: int) -> np.ndarray:
    """
    Use train label distribution as deterministic prior for calibration in the fallback case.
    """
    df_tr = pd.read_csv(train_path, usecols=["score"])
    counts = (
        df_tr["score"]
        .value_counts()
        .reindex(range(1, num_labels + 1), fill_value=0)
        .values.astype(np.float64)
    )
    prior = counts / max(counts.sum(), 1.0)
    prior = np.clip(prior, 1e-12, 1.0)
    prior = prior / prior.sum()
    return prior


TRAIN_PRIOR = None
try:
    TRAIN_PRIOR = _load_train_prior(TRAIN_PATH, NUM_LABELS)
    print("Loaded train prior:", TRAIN_PRIOR.round(4), "sum=", float(TRAIN_PRIOR.sum()))
except Exception as e:
    print("Warning: could not load train prior:", repr(e))
    TRAIN_PRIOR = None




## === cell 7
def predict_essay_score(
    model, dataset, num_labels, batch_size, device, train_prior=None
):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    model.eval()

    preds, essay_ids_all = [], []

    model_num_labels = getattr(getattr(model, "config", None), "num_labels", None)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            essay_ids = batch["essay_id"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            if model_num_labels == num_labels:
                pred_labels = torch.argmax(logits, axis=1).cpu().numpy() + 1
            else:
                if logits.shape[1] >= 2:
                    scalar = (logits[:, 1] - logits[:, 0]).float()
                else:
                    scalar = logits[:, 0].float()
                s = scalar.detach().cpu().numpy()
                preds.extend(s.tolist())
                essay_ids_all.extend(essay_ids)
                continue

            preds.extend(pred_labels.tolist())
            essay_ids_all.extend(essay_ids)

    preds = np.array(preds)
    essay_ids_all = np.array(essay_ids_all)

    chunk_results = pd.DataFrame({"essay_id": essay_ids_all, "raw_prediction": preds})

    aggregated_predictions = chunk_results.groupby("essay_id")["raw_prediction"].mean()
    agg_vals = aggregated_predictions.values

    if model_num_labels == num_labels:
        y_pred_aggregated = np.rint(agg_vals)
        final_scores = np.clip(y_pred_aggregated, 1, num_labels).astype(int)
    else:
        s = agg_vals.astype(np.float64)
        ranks = s.argsort().argsort().astype(np.float64)
        u = (ranks + 0.5) / max(len(s), 1.0)

        if train_prior is None:
            edges = np.linspace(0.0, 1.0, num_labels + 1)
        else:
            edges = np.concatenate([[0.0], np.cumsum(train_prior)])
            edges[-1] = 1.0

        final_scores = np.searchsorted(edges, u, side="right")
        final_scores = np.clip(final_scores, 1, num_labels).astype(int)

    final_results_df = pd.DataFrame(
        {"essay_id": aggregated_predictions.index.values, "score": final_scores}
    )

    return final_results_df, preds




## === cell 8
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




## === cell 9
class TrainEssayDataset(Dataset):
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
            label = int(row["score"])  # 1..6
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
                ) and start != 0:
                    break

                chunk_with_tag = tag_tokens + chunk
                processed = tokenizer.build_inputs_with_special_tokens(chunk_with_tag)
                if len(processed) > self.max_len:
                    processed = processed[: self.max_len]

                pad_len = self.max_len - len(processed)
                if pad_len > 0:
                    processed += [tokenizer.pad_token_id] * pad_len

                self.samples.append((processed, essay_id, label - 1))  # 0..5

                start += step
                if end >= len(base_tokens):
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_ids, essay_id, label = self.samples[idx]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
            "essay_id": essay_id,
        }


def _maybe_fit_head_on_train(model, tokenizer, train_path: str, device: str):
    model_num_labels = getattr(getattr(model, "config", None), "num_labels", None)
    if model_num_labels == NUM_LABELS:
        return model, False

    base_name = getattr(getattr(model, "config", None), "_name_or_path", None)
    if base_name is None:
        return model, False

    try:
        cfg6 = AutoConfig.from_pretrained(
            base_name, local_files_only=True, num_labels=NUM_LABELS
        )
        model6 = AutoModelForSequenceClassification.from_pretrained(
            base_name, config=cfg6, local_files_only=True, ignore_mismatched_sizes=True
        )
    except Exception as e:
        print(
            "Could not construct 6-label model from base; will keep calibration fallback. Error:",
            repr(e),
        )
        return model, False

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model6.resize_token_embeddings(len(tokenizer))

    torch.manual_seed(42)
    np.random.seed(42)

    df_tr = pd.read_csv(train_path, usecols=["essay_id", "full_text", "score"])
    max_train_essays = 6000
    if len(df_tr) > max_train_essays:
        df_tr = df_tr.iloc[:max_train_essays].copy()

    train_ds = TrainEssayDataset(df_tr, tokenizer, MAX_LEN, OVERLAP, MIN_CHUNK_RATIO)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model6 = model6.to(device)
    model6.train()

    optimizer = torch.optim.AdamW(model6.parameters(), lr=2e-5, weight_decay=0.01)

    epochs = 1
    print(
        f"Fitting 6-label head on {len(df_tr)} essays ({len(train_ds)} chunks), epochs={epochs} ..."
    )
    for ep in range(epochs):
        pbar = tqdm(train_loader, desc=f"Train ep {ep+1}/{epochs}", unit="batch")
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model6(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = out.loss
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=float(loss.detach().cpu().item()))
    model6.eval()
    return model6, True


if MODEL_AVAILABLE:
    model, DID_FIT = _maybe_fit_head_on_train(model, tokenizer, TRAIN_PATH, device)
    if DID_FIT:
        MODEL_SOURCE = f"{MODEL_SOURCE} + fitted_6label_head"
        print("Updated MODEL_SOURCE:", MODEL_SOURCE)



## === cell 10
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



## === cell 11
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
        train_prior=TRAIN_PRIOR,
    )

    print("Submission preview (model/calibrated):")
    print(test_results.head())
    print("Submission shape (model/calibrated):", test_results.shape)

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

out_file = os.path.join(OUTPUT_PATH, "submission.csv")
test_results.to_csv(out_file, index=False)

print("Wrote:", out_file)
print(pd.read_csv(out_file).head())
print("Shape:", pd.read_csv(out_file).shape)
print("File exists:", os.path.exists(out_file), "Size:", os.path.getsize(out_file))
