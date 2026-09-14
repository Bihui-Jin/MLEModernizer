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
Given questions and answers from various StackExchange properties, predict target values of 30 labels for each question-answer pair.

## Metric
Mean column-wise Spearman's correlation coefficient. The Spearman's rank correlation is computed for each target column, and the mean of these values is calculated for the submission score.

## Submission Format
For each qa_id in the test set, you must predict a probability for each target variable. The predictions should be in the range [0,1]. The file should contain a header and have the following format:

```
qa_id,question_asker_intent_understanding,...,answer_well_written
6,0.0,...,0.5
8,0.5,...,0.1
18,1.0,...,0.0
etc.
```

## Dataset
The list of 30 target labels are the same as the column names in the `sample_submission.csv` file. Target labels with the prefix `question_` relate to the `question_title` and/or `question_body` features in the data. Target labels with the prefix `answer_` relate to the `answer` feature.

Target labels are aggregated from multiple raters, and can have continuous values in the range `[0,1]`. Therefore, predictions must also be in that range.

- **train.csv** - the training data (target labels are the last 30 columns)
- **test.csv** - the test set (you must predict 30 labels for each test set row)
- **sample_submission.csv** - a sample submission file in the correct format; column names are the 30 target labels

# 2. Python version

3.14

# 3. Installed packages

geopandas==0.14.4
huggingface-hub==0.36.0
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scipy==1.15.3
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
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        input/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        working/
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
```

-> data/google-quest-challenge/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/google-quest-challenge/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/google-quest-challenge/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> data/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.358533902324924

# 6. Current score

nan

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved nan) has done: 'I make the code robust to Kaggle’s offline/no-internet environment by resolving the tokenizer/model directory locally instead of falling back to a hub id that cannot be downloaded. I also fix a hard bug where `qa_id` is incorrectly cast to `torch.long` even though it’s a string in this competition, by keeping it as a Python string and carrying it through the DataLoader. Finally, since the provided checkpoint path doesn’t exist, I implement a safe fallback that still produces a valid submission end-to-end using a lightweight, deterministic baseline prediction (column-wise training means) so you always get a `submission.csv` file with correct columns and `[0,1]` values.'
- What this solution (achieved nan) has done: 'I fix the hard failure where no local HuggingFace model/tokenizer directory is found (common in offline Kaggle runs) by making the pipeline gracefully fall back to a no-transformer baseline instead of raising. To preserve core semantics, the model code stays unchanged; we only gate tokenizer/dataloader/model loading behind an availability check. I also ensure the submission is always created with the exact sample_submission column order and valid `[0,1]` values, so you no longer get `nan` due to missing/invalid output. This should run end-to-end within the time limit and produce `submission.csv`.'
- What this solution (achieved nan) has done: 'Your current “nan” is most likely coming from an invalid submission schema (row count mismatch vs the competition test set) and/or misaligned paths (you’re reading `/kaggle/input/google-quest-challenge/test.csv`, which in your environment is 19,550 rows, while `sample_submission.csv` is 608 rows). I make the input path resolution robust by auto-selecting the dataset directory that contains a consistent trio of `train.csv/test.csv/sample_submission.csv` where test rows match the sample submission rows, so the produced CSV is valid and scores instead of returning `nan`. I also force the submission column order to exactly match `sample_submission.csv` and add a safety assertion that `len(submission)` matches `len(sample_submission)` before writing. The model/architecture and inference logic remain unchanged; this is purely fixing dataset/format alignment so you can get a real score and move toward the target.'
- What this solution (achieved nan) has done: 'Your “nan” is coming from producing an invalid submission for this environment: `test.csv` has 19,550 rows while `sample_submission.csv` has 608 rows, so Kaggle can’t score it. I make `pick_data_dir()` choose the dataset directory where `test.csv` row count matches `sample_submission.csv`, and if none match, I hard-align to the `qa_id` list in `sample_submission.csv` by filtering/reordering `test_df` (minimal change, preserves your model/inference logic). I also ensure predictions are reordered to exactly match `sample_submission.csv` `qa_id` order before writing, which prevents silent misalignment and improves score validity. These changes are purely data/format alignment fixes so you get a real score (and thus can move toward the target) without changing the model architecture or inference semantics.'

# 9. Code solution

## === cell 0
import os
import html
import re
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoConfig
from scipy.stats import spearmanr
from tqdm.auto import tqdm

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")




## === cell 1
def pick_data_dir(candidates: List[str]) -> str:
    valid: List[Tuple[str, int, int]] = []
    for d in candidates:
        train_p = os.path.join(d, "train.csv")
        test_p = os.path.join(d, "test.csv")
        sample_p = os.path.join(d, "sample_submission.csv")
        if not (
            os.path.isfile(train_p)
            and os.path.isfile(test_p)
            and os.path.isfile(sample_p)
        ):
            continue
        try:
            n_test = pd.read_csv(test_p, usecols=["qa_id"]).shape[0]
            n_sub = pd.read_csv(sample_p, usecols=["qa_id"]).shape[0]
            valid.append((d, n_test, n_sub))
        except Exception:
            continue

    if not valid:
        raise FileNotFoundError(
            "Could not find any directory containing train.csv/test.csv/sample_submission.csv."
        )

    exact = [t for t in valid if t[1] == t[2]]
    chosen = (exact[0] if exact else valid[0])[0]

    print("Data dir candidates (dir, n_test, n_sample_submission):")
    for d, n_test, n_sub in valid:
        print(f"  {d} -> test={n_test}, sample_submission={n_sub}")
    print(f"Chosen DATA_DIR: {chosen}")
    if not exact:
        print(
            "Warning: No candidate has test.csv rowcount == sample_submission.csv rowcount. "
            "Will later align test_df to sample_submission qa_id order to ensure a valid submission."
        )
    return chosen


DATA_DIR = pick_data_dir(
    candidates=[
        "/kaggle/input/google-quest-challenge",
        "/kaggle/input/google-quest-challenge/google-quest-challenge",
        "/kaggle/data/google-quest-challenge",
        "/kaggle/data/google-quest-challenge/google-quest-challenge",
    ]
)

TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
SAMPLE_FILE = os.path.join(DATA_DIR, "sample_submission.csv")

MODEL_NAME = "/kaggle/input/deberta-v3-large-fold3-val-spearman0-4392/pytorch/default/2/deberta-v3-large-model/deberta-v3-large-model"
MAX_LENGTH = 512
BATCH_SIZE = 8
SEED = 42

CHECKPOINT_PATH = "/kaggle/input/deberta-v3-large-fold3-val-spearman0-4392/pytorch/default/2/deberta_v3_large_fold3_val_spearman0.4392.ckpt"

QUESTION_TARGET_COLS = [
    "question_asker_intent_understanding",
    "question_body_critical",
    "question_conversational",
    "question_expect_short_answer",
    "question_fact_seeking",
    "question_has_commonly_accepted_answer",
    "question_interestingness_others",
    "question_interestingness_self",
    "question_multi_intent",
    "question_not_really_a_question",
    "question_opinion_seeking",
    "question_type_choice",
    "question_type_compare",
    "question_type_consequence",
    "question_type_definition",
    "question_type_entity",
    "question_type_instructions",
    "question_type_procedure",
    "question_type_reason_explanation",
    "question_type_spelling",
    "question_well_written",
]

ANSWER_TARGET_COLS = [
    "answer_helpful",
    "answer_level_of_information",
    "answer_plausible",
    "answer_relevance",
    "answer_satisfaction",
    "answer_type_instructions",
    "answer_type_procedure",
    "answer_type_reason_explanation",
    "answer_well_written",
]

TARGET_COLS = QUESTION_TARGET_COLS + ANSWER_TARGET_COLS
NUM_QUESTION_TARGETS = len(QUESTION_TARGET_COLS)
NUM_ANSWER_TARGETS = len(ANSWER_TARGET_COLS)
NUM_TARGETS = len(TARGET_COLS)

DROPOUT_RATES = [0.1, 0.15, 0.2, 0.25, 0.3]

print(f"Number of targets: {NUM_TARGETS}")
print(f"Question targets: {NUM_QUESTION_TARGETS}")
print(f"Answer targets: {NUM_ANSWER_TARGETS}")
print(f"Checkpoint (requested): {CHECKPOINT_PATH}")




## === cell 2
def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_text(text: str) -> str:
    """Clean and preprocess text."""
    if pd.isna(text):
        return ""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_spearman(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute mean Spearman correlation (for offline validation if needed)."""
    scores = []
    for i in range(predictions.shape[1]):
        score, _ = spearmanr(predictions[:, i], targets[:, i])
        if not np.isnan(score):
            scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


set_seed(SEED)




## === cell 3
class QuestDataset(Dataset):
    """
    Dataset for Google QUEST Q&A Labeling.
    Creates two inputs for Siamese architecture:
    - Question input: [CLS] question_title [SEP] question_body [SEP]
    - Answer input: [CLS] question_title + body [SEP] answer [SEP]
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = 512,
        is_test: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        self.question_titles = [clean_text(t) for t in self.df["question_title"].values]
        self.question_bodies = [clean_text(t) for t in self.df["question_body"].values]
        self.answers = [clean_text(t) for t in self.df["answer"].values]

        if not is_test:
            self.targets = self.df[TARGET_COLS].values.astype(np.float32)

        self.qa_ids = self.df["qa_id"].astype(str).values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        title = self.question_titles[idx]
        body = self.question_bodies[idx]
        answer = self.answers[idx]

        q_encoding = self.tokenizer(
            title,
            body,
            max_length=self.max_length,
            padding="max_length",
            truncation="longest_first",
            return_tensors="pt",
        )

        question_text = f"{title} {body}"
        a_encoding = self.tokenizer(
            question_text,
            answer,
            max_length=self.max_length,
            padding="max_length",
            truncation="longest_first",
            return_tensors="pt",
        )

        item = {
            "q_input_ids": q_encoding["input_ids"].squeeze(0),
            "q_attention_mask": q_encoding["attention_mask"].squeeze(0),
            "a_input_ids": a_encoding["input_ids"].squeeze(0),
            "a_attention_mask": a_encoding["attention_mask"].squeeze(0),
            "qa_id": self.qa_ids[idx],  # keep as string
        }

        if not self.is_test:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate to keep qa_id as list[str] while stacking tensors."""
    out: Dict[str, object] = {}
    for k in ["q_input_ids", "q_attention_mask", "a_input_ids", "a_attention_mask"]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    if "targets" in batch[0]:
        out["targets"] = torch.stack([b["targets"] for b in batch], dim=0)
    out["qa_id"] = [b["qa_id"] for b in batch]
    return out  # type: ignore[return-value]




## === cell 4
class AttentionPooling(nn.Module):
    """Attention-weighted pooling over sequence dimension."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        weights = self.attention(hidden_states).squeeze(-1)
        weights = weights.masked_fill(~attention_mask.bool(), float("-inf"))
        weights = F.softmax(weights, dim=1)
        pooled = (hidden_states * weights.unsqueeze(-1)).sum(dim=1)
        return pooled


class MultiSampleDropout(nn.Module):
    """Multi-sample dropout for better generalization."""

    def __init__(self, dropout_rates: List[float] = None):
        super().__init__()
        if dropout_rates is None:
            dropout_rates = DROPOUT_RATES
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_rates])

    def forward(self, x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
        outputs = torch.stack([linear(dropout(x)) for dropout in self.dropouts])
        return outputs.mean(dim=0)


class SiameseQuestModel(nn.Module):
    """
    Siamese Dual-Transformer for Google QUEST Q&A Labeling.
    - Shared transformer encoder
    - Two branches: Question and Answer
    - Weighted layer aggregation
    - Attention pooling
    - Multi-sample dropout
    """

    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()

        if pretrained:
            self.transformer = AutoModel.from_pretrained(
                model_name,
                output_hidden_states=True,
                local_files_only=True,
            )
        else:
            config = AutoConfig.from_pretrained(model_name, local_files_only=True)
            config.output_hidden_states = True
            self.transformer = AutoModel.from_config(config)

        hidden_size = self.transformer.config.hidden_size
        num_layers = self.transformer.config.num_hidden_layers

        self.layer_weights = nn.Parameter(torch.ones(num_layers + 1))

        self.q_attention = AttentionPooling(hidden_size)
        self.a_attention = AttentionPooling(hidden_size)

        self.multi_dropout = MultiSampleDropout()

        self.question_head = nn.Linear(hidden_size, NUM_QUESTION_TARGETS)
        self.answer_head = nn.Linear(hidden_size, NUM_ANSWER_TARGETS)
        self.combined_head = nn.Linear(hidden_size * 2, NUM_TARGETS)

    def weighted_layer_pooling(
        self, hidden_states: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        stacked = torch.stack(hidden_states, dim=0)
        weights = F.softmax(self.layer_weights, dim=0)
        weighted = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)
        return weighted

    def encode_branch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        attention_pooling: AttentionPooling,
    ) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden = self.weighted_layer_pooling(outputs.hidden_states)
        pooled = attention_pooling(hidden, attention_mask)
        return pooled

    def forward(
        self,
        q_input_ids: torch.Tensor,
        q_attention_mask: torch.Tensor,
        a_input_ids: torch.Tensor,
        a_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        q_pooled = self.encode_branch(q_input_ids, q_attention_mask, self.q_attention)
        a_pooled = self.encode_branch(a_input_ids, a_attention_mask, self.a_attention)

        q_preds = self.multi_dropout(q_pooled, self.question_head)
        a_preds = self.multi_dropout(a_pooled, self.answer_head)

        combined = torch.cat([q_pooled, a_pooled], dim=-1)
        combined_preds = self.multi_dropout(combined, self.combined_head)

        specialized_preds = torch.cat([q_preds, a_preds], dim=-1)
        final_logits = specialized_preds * 0.5 + combined_preds * 0.5
        return final_logits




## === cell 5
def resolve_model_path(path_or_id: str) -> str:
    if os.path.isdir(path_or_id):
        return os.path.abspath(path_or_id)
    return path_or_id


def find_local_transformer_dir(preferred_path: str) -> Optional[str]:
    """
    Bugfix: avoid hard-failing when no local HF model exists.
    We search /kaggle/input for a directory that looks like a HF model folder.
    """

    def looks_like_hf_model_dir(d: str) -> bool:
        return os.path.isfile(os.path.join(d, "config.json")) and (
            os.path.isfile(os.path.join(d, "tokenizer.json"))
            or os.path.isfile(os.path.join(d, "tokenizer_config.json"))
            or os.path.isfile(os.path.join(d, "vocab.json"))
            or os.path.isfile(os.path.join(d, "spiece.model"))
            or os.path.isfile(os.path.join(d, "vocab.txt"))
        )

    if os.path.isdir(preferred_path) and looks_like_hf_model_dir(preferred_path):
        return preferred_path

    base = "/kaggle/input"
    candidates: List[str] = []
    for root, dirs, _files in os.walk(base):
        for d in dirs:
            full = os.path.join(root, d)
            if looks_like_hf_model_dir(full):
                candidates.append(full)

    if not candidates:
        return None

    prio = [
        c
        for c in candidates
        if "deberta" in os.path.basename(c).lower() or "deberta" in c.lower()
    ]
    return sorted(prio or candidates)[0]


def find_existing_checkpoint(requested_path: str) -> Optional[str]:
    if os.path.isfile(requested_path):
        return requested_path

    base = "/kaggle/input"
    candidates: List[str] = []
    for root, _, files in os.walk(base):
        for fn in files:
            lfn = fn.lower()
            if lfn.endswith(".ckpt") or lfn.endswith(".pth") or lfn.endswith(".bin"):
                candidates.append(os.path.join(root, fn))

    req_name = os.path.basename(requested_path).lower()
    for p in candidates:
        if os.path.basename(p).lower() == req_name:
            return p

    priority: List[str] = []
    for p in candidates:
        bn = os.path.basename(p).lower()
        if "spearman" in bn or "best" in bn or "fold" in bn or "quest" in bn:
            priority.append(p)
    if priority:
        return sorted(priority)[0]

    if candidates:
        return sorted(candidates)[0]

    return None


def adapt_state_dict_keys(
    state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    sample_key = next(iter(state_dict.keys()))
    if (
        sample_key.startswith("transformer.")
        or sample_key.startswith("layer_weights")
        or sample_key.startswith("q_attention")
    ):
        return state_dict
    if sample_key.startswith("model."):
        return {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def load_model_from_checkpoint(
    checkpoint_path: str, model_name: str, device: torch.device
) -> Optional[nn.Module]:
    """
    If no checkpoint is available, return None and we will fall back to baseline
    predictions to ensure a valid submission is produced.
    """
    print(f"Loading checkpoint (requested): {checkpoint_path}")
    resolved_ckpt = find_existing_checkpoint(checkpoint_path)
    if resolved_ckpt is None:
        print(
            "Warning: No checkpoint found under /kaggle/input; will use baseline predictions."
        )
        return None

    print(f"Loading checkpoint (resolved):  {resolved_ckpt}")
    checkpoint = torch.load(resolved_ckpt, map_location=device, weights_only=False)

    model = SiameseQuestModel(model_name, pretrained=False)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format.")

    state_dict = adapt_state_dict_keys(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"Loaded state_dict with missing={len(missing)}, unexpected={len(unexpected)}"
    )

    model = model.to(device)
    model.eval()
    return model




## === cell 6
print("Loading data...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

sample_sub = pd.read_csv(SAMPLE_FILE)
sample_cols = [c for c in sample_sub.columns if c != "qa_id"]
if set(sample_cols) != set(TARGET_COLS):
    raise ValueError(
        "TARGET_COLS mismatch vs sample_submission.csv; cannot proceed safely."
    )
if sample_cols != TARGET_COLS:
    print("Aligning TARGET_COLS ordering to sample_submission.csv")
    TARGET_COLS = sample_cols

sample_sub_ids = sample_sub[["qa_id"]].copy()
sample_sub_ids["qa_id"] = sample_sub_ids["qa_id"].astype(str)
test_df["qa_id"] = test_df["qa_id"].astype(str)

if len(sample_sub_ids) != len(test_df):
    print(
        f"Warning: Row-count mismatch detected (test={len(test_df)}, sample_submission={len(sample_sub_ids)}). "
        "Aligning test_df to sample_submission qa_id order to produce a valid, scorable submission."
    )
    test_idx = test_df.set_index("qa_id", drop=False)
    aligned = test_idx.reindex(sample_sub_ids["qa_id"].values)
    missing_n = int(aligned["qa_id"].isna().sum())
    if missing_n > 0:
        raise ValueError(
            f"Cannot align test_df to sample_submission: {missing_n} qa_id values from sample_submission are missing in test.csv."
        )
    test_df = aligned.reset_index(drop=True)

print(
    f"Aligned Test shape: {test_df.shape} (should match sample_submission rows={len(sample_sub_ids)})"
)



## === cell 7
resolved_model_name = resolve_model_path(MODEL_NAME)
print(f"Requested MODEL_NAME: {resolved_model_name}")

local_model_dir = find_local_transformer_dir(resolved_model_name)
if local_model_dir is None:
    print(
        "Warning: Could not locate any local HuggingFace model/tokenizer directory under /kaggle/input.\n"
        "Will skip transformer inference and use baseline predictions from train target means."
    )
    tokenizer = None
    resolved_model_name = None
else:
    resolved_model_name = local_model_dir
    print(f"Using local transformer dir: {resolved_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_model_name, local_files_only=True
    )
    print(f"Tokenizer loaded. Vocab size: {getattr(tokenizer, 'vocab_size', 'N/A')}")



## === cell 8
if tokenizer is not None:
    test_dataset = QuestDataset(
        df=test_df,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        is_test=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,  # keep qa_id strings
    )

    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Test batches: {len(test_dataloader)}")
else:
    test_dataloader = None



## === cell 9
print(f"Checkpoint exists at requested path? {os.path.exists(CHECKPOINT_PATH)}")



## === cell 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = None
if resolved_model_name is not None:
    model = load_model_from_checkpoint(CHECKPOINT_PATH, resolved_model_name, device)

if model is None:
    train_means = (
        train_df[TARGET_COLS].mean(axis=0).astype(np.float32).clip(0.0, 1.0).values
    )
    predictions = np.tile(train_means[None, :], (len(test_df), 1))
    qa_ids = test_df["qa_id"].astype(str).values
    print("Generated baseline predictions from train target means.")
else:
    assert test_dataloader is not None
    all_predictions = []
    all_qa_ids: List[str] = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Predicting"):
            q_input_ids = batch["q_input_ids"].to(device)
            q_attention_mask = batch["q_attention_mask"].to(device)
            a_input_ids = batch["a_input_ids"].to(device)
            a_attention_mask = batch["a_attention_mask"].to(device)

            logits = model(
                q_input_ids=q_input_ids,
                q_attention_mask=q_attention_mask,
                a_input_ids=a_input_ids,
                a_attention_mask=a_attention_mask,
            )

            preds = torch.sigmoid(logits)
            all_predictions.append(preds.cpu().numpy())
            all_qa_ids.extend(batch["qa_id"])

    predictions = np.concatenate(all_predictions, axis=0)
    qa_ids = np.array(all_qa_ids, dtype=str)
    print(f"Predictions shape: {predictions.shape}, qa_ids shape: {qa_ids.shape}")

qa_ids = qa_ids.astype(str)
if len(qa_ids) != len(sample_sub_ids):
    raise ValueError(
        f"Internal mismatch after alignment: predicted rows={len(qa_ids)} vs sample_submission rows={len(sample_sub_ids)}"
    )
if not np.array_equal(qa_ids, sample_sub_ids["qa_id"].values):
    order = pd.Index(qa_ids)
    take = order.get_indexer(sample_sub_ids["qa_id"].values)
    if (take < 0).any():
        raise ValueError(
            "Could not reorder predictions to sample_submission qa_id order (missing ids)."
        )
    predictions = predictions[take]
    qa_ids = sample_sub_ids["qa_id"].values



## === cell 11
submission = pd.DataFrame(predictions, columns=TARGET_COLS)
submission.insert(0, "qa_id", qa_ids.astype(str))

for col in TARGET_COLS:
    submission[col] = submission[col].clip(0.0, 1.0)

assert len(submission) == len(
    sample_sub_ids
), f"Submission rows ({len(submission)}) != sample_submission rows ({len(sample_sub_ids)})."
assert list(submission.columns) == list(
    sample_sub.columns
), "Submission columns/order must match sample_submission.csv exactly."
assert (
    submission["qa_id"].astype(str).values == sample_sub_ids["qa_id"].values
).all(), "qa_id order must match sample_submission.csv."

print(f"Submission shape: {submission.shape}")
out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print(f"{out_path} created!")
print(submission.head())
