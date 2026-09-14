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

## === cell 1
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

from huggingface_hub import snapshot_download

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

## === cell 2
DATA_DIR = "/kaggle/input/google-quest-challenge"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

MODEL_NAME = "/kaggle/input/deberta-v3-large-fold3-val-spearman0-4392/pytorch/default/3/roberta-large/roberta-large"
MAX_LENGTH = 512
BATCH_SIZE = 8  # Adjust based on your GPU memory
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
print(f"Checkpoint: {CHECKPOINT_PATH}")

## === cell 4
def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clean_text(text: str) -> str:
    """Clean and preprocess text."""
    if pd.isna(text):
        return ""
    text = html.unescape(str(text))
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def compute_spearman(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute mean Spearman correlation."""
    scores = []
    for i in range(predictions.shape[1]):
        score, _ = spearmanr(predictions[:, i], targets[:, i])
        if not np.isnan(score):
            scores.append(score)
    return np.mean(scores) if scores else 0.0

set_seed(SEED)

## === cell 6
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
            title, body,
            max_length=self.max_length,
            padding="max_length",
            truncation="longest_first",
            return_tensors="pt",
        )
        
        question_text = f"{title} {body}"
        a_encoding = self.tokenizer(
            question_text, answer,
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

## === cell 8
class AttentionPooling(nn.Module):
    """Attention-weighted pooling over sequence dimension."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1),
        )
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        weights = self.attention(hidden_states).squeeze(-1)
        weights = weights.masked_fill(~attention_mask.bool(), float('-inf'))
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
    - Shared transformer encoder (RoBERTa-Large)
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
            )
        else:
            config = AutoConfig.from_pretrained(model_name)
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
    
    def weighted_layer_pooling(self, hidden_states: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        stacked = torch.stack(hidden_states, dim=0)
        weights = F.softmax(self.layer_weights, dim=0)
        weighted = (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)
        return weighted
    
    def encode_branch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                      attention_pooling: AttentionPooling) -> torch.Tensor:
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

## === cell 10
def load_model_from_checkpoint(checkpoint_path: str, model_name: str, device: torch.device) -> nn.Module:
    """
    Load model from PyTorch Lightning checkpoint.
    
    Args:
        checkpoint_path: Path to .ckpt file
        model_name: HuggingFace model name
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = SiameseQuestModel(model_name, pretrained=False)
    
    state_dict = checkpoint['state_dict']
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
 
    return model

## === cell 12
print("Loading data...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

## === cell 13
print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Vocab size: {tokenizer.vocab_size}")

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/deberta-v3-large-fold3-val-spearman0-4392/pytorch/default/3/roberta-large/roberta-large'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/tmp/ipykernel_11/3283950897.py in <cell line: 0>()
      1 # Initialize tokenizer
      2 print(f"Loading tokenizer: {MODEL_NAME}")
----> 3 tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
      4 print(f"Vocab size: {tokenizer.vocab_size}")

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/tokenization_auto.py in from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs)
    981 
    982         # Next, let's try to use the tokenizer_config file to get the tokenizer class.
--> 983         tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
    984         if "_commit_hash" in tokenizer_config:
    985             kwargs["_commit_hash"] = tokenizer_config["_commit_hash"]

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/tokenization_auto.py in get_tokenizer_config(pretrained_model_name_or_path, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, **kwargs)
    813 
    814     commit_hash = kwargs.get("_commit_hash", None)
--> 815     resolved_config_file = cached_file(
    816         pretrained_model_name_or_path,
    817         TOKENIZER_CONFIG_FILE,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    310     ```
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file
    314     return file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    520 
    521         # Now we try to recover if we can find all files correctly in the cache
--> 522         resolved_files = [
    523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in <listcomp>(.0)
    521         # Now we try to recover if we can find all files correctly in the cache
    522         resolved_files = [
--> 523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]
    525         if all(file is not None for file in resolved_files):

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
    138 ):
    139     # We try to see if we have a cached version (not up to date):
--> 140     resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision)
    141     if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:
    142         return resolved_file

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    104         ):
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 
    108             elif arg_name == "token" and arg_value is not None:

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    152 
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"
    156             f" '{repo_id}'. Use `repo_type` argument if needed."

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/deberta-v3-large-fold3-val-spearman0-4392/pytorch/default/3/roberta-large/roberta-large'. Use `repo_type` argument if needed.

## === cell 14
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
    pin_memory=True,
)

print(f"Test dataset size: {len(test_dataset)}")
print(f"Test batches: {len(test_dataloader)}")

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2973487849.py in <cell line: 0>()
      2 test_dataset = QuestDataset(
      3     df=test_df,
----> 4     tokenizer=tokenizer,
      5     max_length=MAX_LENGTH,
      6     is_test=True,

NameError: name 'tokenizer' is not defined

## === cell 16
exists = os.path.exists(CHECKPOINT_PATH)
status = "OK" if exists else "NOT FOUND"
print(f"Checkpoint: {CHECKPOINT_PATH} [{status}]")

## === cell 19
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = load_model_from_checkpoint(CHECKPOINT_PATH, MODEL_NAME, device)

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

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3527534063.py in <cell line: 0>()
      4 
      5 # Load model from checkpoint
----> 6 model = load_model_from_checkpoint(CHECKPOINT_PATH, MODEL_NAME, device)
      7 
      8 # Generate predictions

/tmp/ipykernel_11/1934152250.py in load_model_from_checkpoint(checkpoint_path, model_name, device)
     14 
     15     # Load checkpoint
---> 16     checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
     17 
     18     # Create model

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    749 def _open_file_like(name_or_buffer, mode):
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:
    753         if "w" in mode:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    730 class _open_file(_opener):
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 
    734     def __exit__(self, *args):

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/deberta-v3-large-fold3-val-spearman0-4392/pytorch/default/3/roberta_large_fold2_val_spearman0.4418.ckpt'

## === cell 21
submission = pd.DataFrame(predictions, columns=TARGET_COLS)
submission.insert(0, "qa_id", [x.item() for x in all_qa_ids])

for col in TARGET_COLS:
    submission[col] = submission[col].clip(0, 1)

print(f"Submission shape: {submission.shape}")
submission.to_csv('submission.csv', index=False)
print('submission.csv created!')
submission.head()

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1986601062.py in <cell line: 0>()
      1 ## Create submission DataFrame
----> 2 submission = pd.DataFrame(predictions, columns=TARGET_COLS)
      3 submission.insert(0, "qa_id", [x.item() for x in all_qa_ids])
      4 
      5 # Clip predictions to [0, 1]

NameError: name 'predictions' is not defined
