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

0.3649142456650344

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import html
import re
import random
from typing import Dict, List, Optional, Tuple

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
DATA_DIR = "/kaggle/input/google-quest-challenge"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

MODEL_NAME = "roberta-large"

MAX_LENGTH = 512
BATCH_SIZE = 8
SEED = 42

CHECKPOINT_PATH = "/kaggle/input/deberta-v3-large-fold3-val-spearman0-4392/pytorch/default/3/roberta_large_fold2_val_spearman0.4418.ckpt"

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
WORKING_DIR = "/kaggle/working"
os.makedirs(WORKING_DIR, exist_ok=True)
print(f"Working dir: {WORKING_DIR} (exists={os.path.exists(WORKING_DIR)})")




## === cell 3
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
    text = text.strip()
    return text


def compute_spearman(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute mean Spearman correlation."""
    scores = []
    for i in range(predictions.shape[1]):
        score, _ = spearmanr(predictions[:, i], targets[:, i])
        if not np.isnan(score):
            scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


set_seed(SEED)




## === cell 4
def find_any_ckpt(
    preferred_path: str, search_root: str = "/kaggle/input"
) -> Optional[str]:
    """
    The provided checkpoint path may not exist. Search /kaggle/input for a .ckpt file.
    Returns the preferred_path if it exists, else the first discovered ckpt (stable sorted).
    """
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    ckpts = []
    for root, _, files in os.walk(search_root):
        for fn in files:
            if fn.endswith(".ckpt"):
                ckpts.append(os.path.join(root, fn))
    ckpts = sorted(ckpts)
    return ckpts[0] if ckpts else None


def strip_prefix_from_state_dict(
    state_dict: Dict[str, torch.Tensor], prefixes: Tuple[str, ...]
) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        new_k = k
        for p in prefixes:
            if new_k.startswith(p):
                new_k = new_k[len(p) :]
        out[new_k] = v
    return out


def find_local_hf_model_dir(
    preferred_id: str,
    search_roots: Tuple[str, ...] = (
        "/kaggle/input",
        "/kaggle/working",
        "/kaggle/data",
    ),
) -> Optional[str]:
    """
    BUGFIX: Kaggle internet is disabled; AutoTokenizer/AutoModel must load from local files.
    Try (1) a direct local directory match, (2) known HF cache directories, (3) any folder containing config.json.
    """
    if (
        preferred_id
        and os.path.isdir(preferred_id)
        and os.path.exists(os.path.join(preferred_id, "config.json"))
    ):
        return preferred_id

    candidate_dirs = []
    hf_cache_roots = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        "/root/.cache/huggingface/hub",
        "/kaggle/working/.cache/huggingface/hub",
    ]
    for cache_root in hf_cache_roots:
        if not os.path.isdir(cache_root):
            continue
        for root, dirs, files in os.walk(cache_root):
            if "config.json" in files:
                candidate_dirs.append(root)

    for sroot in search_roots:
        if not os.path.isdir(sroot):
            continue
        for root, dirs, files in os.walk(sroot):
            if "config.json" in files:
                candidate_dirs.append(root)

    candidate_dirs = sorted(set(candidate_dirs))
    if not candidate_dirs:
        return None

    preferred_lower = (preferred_id or "").lower().replace("/", "_")
    for d in candidate_dirs:
        if preferred_lower and preferred_lower in d.lower().replace("/", "_"):
            return d

    return candidate_dirs[0]




## === cell 5
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

        self.qa_ids = self.df["qa_id"].values

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
            "qa_id": self.qa_ids[idx],
        }

        if not self.is_test:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item




## === cell 6
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

    Architecture:
    - Shared transformer encoder (RoBERTa-Large or local equivalent)
    - Two branches: Question and Answer
    - Weighted layer aggregation
    - Attention pooling
    - Multi-sample dropout
    """

    def __init__(self, model_name_or_path: str, pretrained: bool = True):
        super().__init__()

        if pretrained:
            self.transformer = AutoModel.from_pretrained(
                model_name_or_path,
                output_hidden_states=True,
                local_files_only=True,
            )
        else:
            config = AutoConfig.from_pretrained(
                model_name_or_path, local_files_only=True
            )
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




## === cell 7
def load_model_from_checkpoint(
    checkpoint_path: Optional[str], model_name_or_path: str, device: torch.device
) -> nn.Module:
    """
    Load model from PyTorch Lightning checkpoint (if available); otherwise fall back to pretrained.
    Also strips common Lightning prefixes so weights actually load.
    """
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print(
            "No checkpoint found. Falling back to base pretrained transformer weights."
        )
        model = SiameseQuestModel(model_name_or_path, pretrained=True).to(device)
        model.eval()
        return model

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = SiameseQuestModel(model_name_or_path, pretrained=False)

    state_dict = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint
        else checkpoint
    )
    state_dict = strip_prefix_from_state_dict(state_dict, prefixes=("model.",))

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"Checkpoint loaded with strict=False. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
    )

    model = model.to(device)
    model.eval()
    return model




## === cell 8
print("Loading data...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_path)
sample_target_cols = [c for c in sample_sub.columns if c != "qa_id"]
assert set(sample_target_cols) == set(
    TARGET_COLS
), "TARGET_COLS mismatch vs sample_submission.csv"

TARGET_COLS_ORDERED = sample_target_cols


## === cell 9
local_model_path = find_local_hf_model_dir(MODEL_NAME)
if local_model_path is None:
    raise RuntimeError(
        f"Could not find any local Hugging Face model files (config.json) for '{MODEL_NAME}'. "
        "Please add a dataset containing the transformer model/tokenizer files."
    )

print(f"Resolved local model path for offline loading: {local_model_path}")

print(f"Loading tokenizer from: {local_model_path}")
tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
print(f"Tokenizer loaded. Vocab size: {getattr(tokenizer, 'vocab_size', 'N/A')}")


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3417025402.py in <cell line: 0>()
      2 local_model_path = find_local_hf_model_dir(MODEL_NAME)
      3 if local_model_path is None:
----> 4     raise RuntimeError(
      5         f"Could not find any local Hugging Face model files (config.json) for '{MODEL_NAME}'. "
      6         "Please add a dataset containing the transformer model/tokenizer files."

RuntimeError: Could not find any local Hugging Face model files (config.json) for 'roberta-large'. Please add a dataset containing the transformer model/tokenizer files.

## === cell 10
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
)

print(f"Test dataset size: {len(test_dataset)}")
print(f"Test batches: {len(test_dataloader)}")


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2388807693.py in <cell line: 0>()
      1 test_dataset = QuestDataset(
      2     df=test_df,
----> 3     tokenizer=tokenizer,
      4     max_length=MAX_LENGTH,
      5     is_test=True,

NameError: name 'tokenizer' is not defined

## === cell 11
resolved_ckpt = find_any_ckpt(CHECKPOINT_PATH, search_root="/kaggle/input")
print(f"Checkpoint resolved: {resolved_ckpt if resolved_ckpt else 'NONE FOUND'}")


## === cell 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = load_model_from_checkpoint(resolved_ckpt, local_model_path, device)

all_predictions = []
all_qa_ids = []

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
print(f"Predictions shape: {predictions.shape}")
assert predictions.shape[0] == len(test_df)
assert predictions.shape[1] == len(TARGET_COLS)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
LocalEntryNotFoundError                   Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    113 
--> 114         return fn(*args, **kwargs)
    115 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in hf_hub_download(repo_id, filename, subfolder, repo_type, revision, library_name, library_version, cache_dir, local_dir, user_agent, force_download, proxies, etag_timeout, token, local_files_only, headers, endpoint, resume_download, force_filename, local_dir_use_symlinks)
   1006     else:
-> 1007         return _hf_hub_download_to_cache_dir(
   1008             # Destination

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _hf_hub_download_to_cache_dir(cache_dir, repo_id, filename, repo_type, revision, endpoint, etag_timeout, headers, proxies, token, local_files_only, force_download)
   1113         # Otherwise, raise appropriate error
-> 1114         _raise_on_head_call_error(head_call_error, force_download, local_files_only)
   1115 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _raise_on_head_call_error(head_call_error, force_download, local_files_only)
   1645     if local_files_only:
-> 1646         raise LocalEntryNotFoundError(
   1647             "Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable"

LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable hf.co look-ups and downloads online, set 'local_files_only' to False.

The above exception was the direct cause of the following exception:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/1582842875.py in <cell line: 0>()
      3 
      4 # Use the resolved local model path for offline model config/weights loading
----> 5 model = load_model_from_checkpoint(resolved_ckpt, local_model_path, device)
      6 
      7 all_predictions = []

/tmp/ipykernel_55/1701365670.py in load_model_from_checkpoint(checkpoint_path, model_name_or_path, device)
     10             "No checkpoint found. Falling back to base pretrained transformer weights."
     11         )
---> 12         model = SiameseQuestModel(model_name_or_path, pretrained=True).to(device)
     13         model.eval()
     14         return model

/tmp/ipykernel_55/255564835.py in __init__(self, model_name_or_path, pretrained)
     50 
     51         if pretrained:
---> 52             self.transformer = AutoModel.from_pretrained(
     53                 model_name_or_path,
     54                 output_hidden_states=True,

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/auto_factory.py in from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs)
    545                 _ = kwargs.pop("quantization_config")
    546 
--> 547             config, kwargs = AutoConfig.from_pretrained(
    548                 pretrained_model_name_or_path,
    549                 return_unused_kwargs=True,

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/configuration_auto.py in from_pretrained(cls, pretrained_model_name_or_path, **kwargs)
   1195         code_revision = kwargs.pop("code_revision", None)
   1196 
-> 1197         config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
   1198         has_remote_code = "auto_map" in config_dict and "AutoConfig" in config_dict["auto_map"]
   1199         has_local_code = "model_type" in config_dict and config_dict["model_type"] in CONFIG_MAPPING

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    606         original_kwargs = copy.deepcopy(kwargs)
    607         # Get config dict associated with the base config file
--> 608         config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
    609         if config_dict is None:
    610             return {}, kwargs

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in _get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    665             try:
    666                 # Load from local folder or from cache or download from model Hub and cache
--> 667                 resolved_config_file = cached_file(
    668                     pretrained_model_name_or_path,
    669                     configuration_file,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    310     ```
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file
    314     return file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    541             # even when `local_files_only` is True, in which case raising for connections errors only would not make sense)
    542             elif _raise_exceptions_for_missing_entries:
--> 543                 raise OSError(
    544                     f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load the files, and couldn't find them in the"
    545                     f" cached files.\nCheck your internet connection or see how to run the library in offline mode at"

OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

## === cell 13
submission = pd.DataFrame(predictions, columns=TARGET_COLS)

submission.insert(0, "qa_id", [str(x) for x in all_qa_ids])

submission = submission[["qa_id"] + TARGET_COLS_ORDERED]

for col in TARGET_COLS_ORDERED:
    submission[col] = submission[col].clip(0, 1)

out_path = os.path.join(WORKING_DIR, "submission.csv")
print(f"Submission shape: {submission.shape}")
submission.to_csv(out_path, index=False)
print(f"{out_path} created!")
print(submission.head())

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3552932041.py in <cell line: 0>()
      1 # Ensure submission uses the exact column order from sample_submission.csv
----> 2 submission = pd.DataFrame(predictions, columns=TARGET_COLS)
      3 
      4 # qa_id items can be numpy scalars/strings; robust conversion
      5 submission.insert(0, "qa_id", [str(x) for x in all_qa_ids])

NameError: name 'predictions' is not defined
