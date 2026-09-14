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

0.00442

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00442) has done: 'I fix the HuggingFace `HFValidationError` by making the local model path resolution robust: the current `MODEL_NAME` points to a folder name that doesn’t contain the HF config/tokenizer files, so we search upward/downward for the actual folder containing `config.json`. I also make checkpoint/model loading and tokenizer loading use the same resolved directory, and add a safe fallback to a public backbone (local-only first, then allow download only if necessary) so the notebook can always run end-to-end in Kaggle. Finally, I ensure the dataloader and submission creation always execute (no undefined variables), and that the submission CSV is written with correct columns and row alignment.'
- What this solution (achieved 0.00442) has done: 'The crash comes from an upstream protobuf/transformers interaction when importing/using Hugging Face objects; in Kaggle this is commonly fixed by forcing the pure-Python protobuf implementation before importing `transformers`. Your current very low score (0.00442 vs target 0.3649) is also consistent with not actually loading the intended fine-tuned checkpoint/backbone, so I also make the model/tokenizer resolution strictly local-first and ensure we only fall back to downloading if absolutely necessary (keeping the same architecture/inference). Finally, I make the DataLoader more Kaggle-safe (num_workers=0) and ensure `qa_id` alignment stays exact so the submission is valid.'
- What this solution (achieved 0.00442) has done: 'The crash in cell 10 is caused by an incompatibility between `transformers` and the installed protobuf runtime; forcing the pure-Python protobuf implementation must be done before *any* protobuf/transformers import, and we also need to ensure the Python fallback is actually used (not the C++ one). To move the score from 0.00442 toward the 0.3649 target without changing core modeling logic, I also prevent the code from silently falling back to an unfine-tuned backbone when the checkpoint/model directory don’t match: we resolve the HF model directory from the checkpoint folder first (local-only), then load the Lightning checkpoint with `strict=True` (fail fast if mismatched) and only then fall back to other options. Finally, I keep the submission formatting/alignment intact and guarantee a valid `submission.csv` is written.'
- What this solution (achieved 0.00442) has done: 'I remove the failing protobuf/pip auto-downgrade logic (it cannot work offline with `--no-index`) while keeping the environment variable workaround, so the notebook starts reliably. I also fix the cascading `NameError`s by ensuring all required imports and symbols are defined before any later cells run, without changing the model architecture or inference logic. To avoid a silent bad fallback that tanks score, I keep the local-first resolution for the HF model folder and only fall back to downloading if the local checkpoint/model truly cannot be loaded. Finally, I guarantee that `submission.csv` is written with the exact sample submission columns and correct `qa_id` alignment.'
- What this solution (achieved 0.00442) has done: 'We fix the protobuf/transformers crash by forcing the pure-Python protobuf implementation as early as possible and additionally disabling the compiled implementation via `google.protobuf.internal.api_implementation._implementation_type` when available; this directly addresses the `MessageFactory.GetPrototype` AttributeError without changing your model logic. We also ensure Hugging Face doesn’t try to use tokenizers/transformers features that can trigger protobuf internals by setting `TRANSFORMERS_NO_ADVISORY_WARNINGS` and keeping `use_fast=True` but with a safe fallback to `use_fast=False` if needed. Finally, we keep your exact architecture/inference pipeline, but make model/tokenizer resolution consistent and deterministic so you actually load the fine-tuned checkpoint when present—this should move the score materially toward the target instead of silently using a mismatched fallback.'
- What this solution (achieved 0.00442) has done: 'We fix the protobuf crash that occurs during `transformers` model/tokenizer loading by pinning protobuf to the pure-Python implementation *and* (critically) forcing the runtime to use it before any `transformers` import, plus adding a safe compatibility patch for the missing `MessageFactory.GetPrototype` API expected by some stacks. Then we make model loading consistently use the resolved local HF directory derived from the checkpoint location (so we actually load the intended fine-tuned weights rather than silently falling back), which should move the score substantially toward your 0.3649 target without changing the architecture or inference semantics. Finally, we keep the dataloader/inference/submission alignment intact and ensure a valid `submission.csv` is always written.'
- What this solution (achieved 0.00442) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by applying a compatibility patch earlier and more robustly (covering both class and instance access patterns) before any `transformers` model/tokenizer usage. Then I ensure we actually load the intended fine-tuned checkpoint + matching HF backbone locally (fail-fast if the local checkpoint can’t be applied, and only then fall back), because the current very low score is consistent with silently running an un-fine-tuned fallback. Finally, I keep the inference and submission formatting the same but make `qa_id` collection deterministic and type-safe so the saved `submission.csv` is always valid and aligned.'
- What this solution (achieved 0.00442) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by applying a compatibility patch earlier and more robustly, covering both the class and the concrete default factory instance that `transformers` may touch. I also ensure no `transformers` import happens before the protobuf environment variables and patch are applied, which is required for the pure-Python protobuf fallback to actually take effect. These changes are execution/stability fixes and do not change your model architecture, checkpoint usage, or inference semantics; they should also allow the intended fine-tuned checkpoint path to be used (which should materially improve the score from the current near-random 0.00442 toward the target). Finally, I keep submission formatting identical and guaranteed to write `submission.csv` with correct column order and row alignment.'
- What this solution (achieved 0.00442) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by applying the compatibility patch to the actual class used at runtime (`google.protobuf.message_factory.MessageFactory`) and to the default factory instance (`message_factory._DEFAULT`), and I do it before importing `transformers`. This is a minimal execution fix that keeps your model/inference logic identical, but it should allow the intended checkpoint/model to load instead of failing (which is the main reason your score is near-random). I also make `qa_id` collection deterministic by forcing it to be returned as a plain Python string from the Dataset so the final submission alignment is guaranteed. No architecture, training loop, feature extraction, or post-processing changes are introduced beyond these stability fixes.'
- What this solution (achieved 0.00442) has done: 'We fix the protobuf `MessageFactory.GetPrototype` crash by applying the compatibility patch to the actual runtime module (`google.protobuf.message_factory`) and its `MessageFactory` class/instance before any `transformers` import, since your current patch targets a different module path and doesn’t take effect. Then we keep the model/tokenizer and checkpoint loading logic the same, but ensure the environment variables and patch execute first so the fine-tuned checkpoint can actually load (your current 0.00442 score is consistent with failing before real inference). Finally, we make the DataLoader collate for `qa_id` deterministic (string list) to avoid any alignment issues and always write a valid `submission.csv` with the exact sample submission column order.'
- What this solution (achieved 0.00442) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by applying a more direct compatibility patch to both `google.protobuf.message_factory.MessageFactory` and the default factory instance *before* any `transformers` import, because your current patch targets a different module path and doesn’t reliably affect the runtime class being used. This is an execution blocker; once fixed, the model/tokenizer should load and inference run end-to-end to produce `submission.csv`. I keep your model architecture/inference identical, but I also ensure the checkpoint-driven HF directory resolution is used consistently so you actually load the intended fine-tuned weights (the current very low score is consistent with failing before real inference or loading the wrong backbone). No training, loss, or feature logic is changed.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

try:
    import google.protobuf.internal.api_implementation as _api_impl  # type: ignore

    if hasattr(_api_impl, "_implementation_type"):
        _api_impl._implementation_type = "python"  # type: ignore[attr-defined]
except Exception:
    pass


def _ensure_getprototype(obj) -> None:
    """
    Some stacks (via transformers / sentencepiece / protobuf) expect MessageFactory.GetPrototype,
    but newer protobuf may not expose it. Provide compatibility mapping to GetMessageClass when possible.
    """
    if obj is None or hasattr(obj, "GetPrototype"):
        return

    if hasattr(obj, "GetMessageClass"):

        def GetPrototype(descriptor, _obj=obj):  # type: ignore
            return _obj.GetMessageClass(descriptor)

        try:
            setattr(obj, "GetPrototype", GetPrototype)
            return
        except Exception:
            pass

    if hasattr(obj, "_InternalCreateMessageClass"):

        def GetPrototype(descriptor, _obj=obj):  # type: ignore
            return _obj._InternalCreateMessageClass(descriptor)  # type: ignore[attr-defined]

        try:
            setattr(obj, "GetPrototype", GetPrototype)
            return
        except Exception:
            pass


try:
    from google.protobuf import message_factory as _mf  # type: ignore

    _mf_cls = getattr(_mf, "MessageFactory", None)
    if (
        _mf_cls is not None
        and not hasattr(_mf_cls, "GetPrototype")
        and hasattr(_mf_cls, "GetMessageClass")
    ):

        def _cls_GetPrototype(self, descriptor):  # type: ignore
            return self.GetMessageClass(descriptor)

        try:
            setattr(_mf_cls, "GetPrototype", _cls_GetPrototype)
        except Exception:
            pass

    _ensure_getprototype(getattr(_mf, "_DEFAULT", None))
    _default_callable = getattr(_mf, "Default", None)
    if callable(_default_callable):
        try:
            _ensure_getprototype(_default_callable())
        except Exception:
            pass
except Exception:
    pass

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

print(f"Python: {os.sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")



## === cell 1
DATA_DIR = "/kaggle/input/google-quest-challenge"
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"DATA_DIR not found: {DATA_DIR}")

TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

if not (os.path.exists(TRAIN_FILE) and os.path.exists(TEST_FILE)):
    nested = os.path.join(DATA_DIR, "google-quest-challenge")
    if os.path.exists(os.path.join(nested, "train.csv")):
        DATA_DIR = nested
        TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
        TEST_FILE = os.path.join(DATA_DIR, "test.csv")

MODEL_NAME = "/kaggle/input/deberta-v3-large-fold3-val-spearman0-4392/pytorch/default/3/roberta-large"

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

print(f"DATA_DIR: {DATA_DIR}")
print(f"TRAIN_FILE exists: {os.path.exists(TRAIN_FILE)}")
print(f"TEST_FILE exists: {os.path.exists(TEST_FILE)}")
print(f"Number of targets: {NUM_TARGETS}")
print(f"Question targets: {NUM_QUESTION_TARGETS}")
print(f"Answer targets: {NUM_ANSWER_TARGETS}")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Model dir (raw): {MODEL_NAME}")




## === cell 2
def set_seed(seed: int = 42) -> None:
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
    if pd.isna(text):
        return ""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_spearman(predictions: np.ndarray, targets: np.ndarray) -> float:
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
    - Question input: title vs body
    - Answer input: (title+body) vs answer
    """

    def __init__(
        self, df: pd.DataFrame, tokenizer, max_length: int = 512, is_test: bool = False
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

        qa_id = self.qa_ids[idx]
        if isinstance(qa_id, (np.generic,)):
            qa_id = qa_id.item()
        qa_id = str(qa_id)

        item = {
            "q_input_ids": q_encoding["input_ids"].squeeze(0),
            "q_attention_mask": q_encoding["attention_mask"].squeeze(0),
            "a_input_ids": a_encoding["input_ids"].squeeze(0),
            "a_attention_mask": a_encoding["attention_mask"].squeeze(0),
            "qa_id": qa_id,
        }

        if not self.is_test:
            item["targets"] = torch.tensor(self.targets[idx], dtype=torch.float32)

        return item




## === cell 4
class AttentionPooling(nn.Module):
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
        return (hidden_states * weights.unsqueeze(-1)).sum(dim=1)


class MultiSampleDropout(nn.Module):
    def __init__(self, dropout_rates: List[float] = None):
        super().__init__()
        if dropout_rates is None:
            dropout_rates = DROPOUT_RATES
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_rates])

    def forward(self, x: torch.Tensor, linear: nn.Linear) -> torch.Tensor:
        outputs = torch.stack([linear(dropout(x)) for dropout in self.dropouts])
        return outputs.mean(dim=0)


class SiameseQuestModel(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        if pretrained:
            self.transformer = AutoModel.from_pretrained(
                model_name, output_hidden_states=True, local_files_only=True
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
        return (stacked * weights.view(-1, 1, 1, 1)).sum(dim=0)

    def encode_branch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        attention_pooling: AttentionPooling,
    ) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden = self.weighted_layer_pooling(outputs.hidden_states)
        return attention_pooling(hidden, attention_mask)

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
def _strip_pl_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        out[nk] = v
    return out


def _find_hf_folder(path: str) -> Optional[str]:
    """
    Find a usable on-disk folder containing config.json (robust to nested Kaggle dataset layouts).
    """
    if not path:
        return None
    if os.path.isfile(path):
        path = os.path.dirname(path)
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
        return path

    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            if "config.json" in files:
                return root

        cur = os.path.abspath(path)
        for _ in range(6):
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            if os.path.exists(os.path.join(parent, "config.json")):
                return parent
            for root, _, files in os.walk(parent):
                if "config.json" in files:
                    return root
            cur = parent

    return None


def _resolve_local_hf_dir(path: str) -> str:
    resolved = _find_hf_folder(path)
    return resolved if resolved is not None else path


def _resolve_model_dir_from_checkpoint(
    checkpoint_path: str, fallback_model_dir: str
) -> str:
    """
    Score fix (minimal): prefer HF files located near the .ckpt to ensure we load the exact backbone.
    """
    cand = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        cand = _find_hf_folder(os.path.dirname(checkpoint_path))
    if cand is not None:
        return cand
    return _resolve_local_hf_dir(fallback_model_dir)


def load_model_from_checkpoint(
    checkpoint_path: str, model_name: str, device: torch.device
) -> nn.Module:
    """
    Load Lightning checkpoint if present; otherwise load transformer weights.
    Core architecture unchanged.
    """
    model_name_resolved = _resolve_local_hf_dir(model_name)
    print(f"Resolved HF model dir for model loading: {model_name_resolved}")

    if os.path.exists(checkpoint_path):
        print(f"Loading Lightning checkpoint: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model = SiameseQuestModel(model_name_resolved, pretrained=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = _strip_pl_prefix(state_dict)
        model.load_state_dict(state_dict, strict=True)
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        print(
            f"Falling back to AutoModel weights from local directory: {model_name_resolved}"
        )
        model = SiameseQuestModel(model_name_resolved, pretrained=True)

    model = model.to(device)
    model.eval()
    return model




## === cell 6
print("Loading data...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")



## === cell 7
resolved_model_dir = _resolve_model_dir_from_checkpoint(CHECKPOINT_PATH, MODEL_NAME)
print(f"Resolved model directory (tokenizer/model): {resolved_model_dir}")

tokenizer = None
tokenizer_load_errors = []

tokenizer_candidates = [
    (resolved_model_dir, True, True),  # (path, local_only, use_fast)
    (resolved_model_dir, True, False),
    ("roberta-large", False, True),
    ("roberta-large", False, False),
]

for candidate, local_only, use_fast in tokenizer_candidates:
    try:
        print(
            f"Trying tokenizer from: {candidate} (local_files_only={local_only}, use_fast={use_fast})"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            candidate,
            local_files_only=local_only,
            use_fast=use_fast,
        )
        print(f"Tokenizer loaded from: {candidate} (use_fast={use_fast})")
        break
    except Exception as e:
        tokenizer_load_errors.append(
            (f"{candidate}|local={local_only}|fast={use_fast}", repr(e))
        )
        tokenizer = None

if tokenizer is None:
    msg = "Failed to load tokenizer. Errors:\n" + "\n".join(
        [f"- {c}: {err}" for c, err in tokenizer_load_errors]
    )
    raise RuntimeError(msg)

print(f"Tokenizer vocab size: {getattr(tokenizer, 'vocab_size', 'n/a')}")



## === cell 8
test_dataset = QuestDataset(
    df=test_df,
    tokenizer=tokenizer,
    max_length=MAX_LENGTH,
    is_test=True,
)


def _collate_fn(batch):
    out = {}
    for k in batch[0].keys():
        if k == "qa_id":
            out[k] = [str(x["qa_id"]) for x in batch]
        else:
            out[k] = torch.stack([x[k] for x in batch], dim=0)
    return out


test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    collate_fn=_collate_fn,
)

print(f"Test dataset size: {len(test_dataset)}")
print(f"Test batches: {len(test_dataloader)}")



## === cell 9
exists = os.path.exists(CHECKPOINT_PATH)
status = "OK" if exists else "NOT FOUND"
print(f"Checkpoint: {CHECKPOINT_PATH} [{status}]")
print(f"Model directory exists (raw): {os.path.isdir(MODEL_NAME)}")
print(
    f"Resolved model directory: {resolved_model_dir} [exists={os.path.isdir(resolved_model_dir)}]"
)

if os.path.isdir(resolved_model_dir):
    for fn in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "pytorch_model.bin",
        "model.safetensors",
    ]:
        p = os.path.join(resolved_model_dir, fn)
        if os.path.exists(p):
            print(f"Found in model dir: {fn}")



## === cell 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_load_errors = []
model = None

candidates = [
    ("CHECKPOINT+LOCAL_HF", True),
    ("roberta-large", False),
]

for candidate, is_primary in candidates:
    try:
        if candidate == "CHECKPOINT+LOCAL_HF":
            if not os.path.exists(CHECKPOINT_PATH):
                raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
            model = load_model_from_checkpoint(
                CHECKPOINT_PATH, resolved_model_dir, device
            )
        else:
            model = SiameseQuestModel(candidate, pretrained=False).to(device).eval()
            model.transformer = (
                AutoModel.from_pretrained(
                    candidate, output_hidden_states=True, local_files_only=False
                )
                .to(device)
                .eval()
            )

        print(f"Model loaded using: {candidate}")
        break
    except Exception as e:
        model_load_errors.append((candidate, repr(e)))
        model = None
        if is_primary:
            print(f"Primary model load failed, will try fallback. Error: {repr(e)}")

if model is None:
    msg = "Failed to load model. Errors:\n" + "\n".join(
        [f"- {c}: {err}" for c, err in model_load_errors]
    )
    raise RuntimeError(msg)

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
        all_predictions.append(preds.detach().cpu().numpy())

        all_qa_ids.extend([str(x) for x in batch["qa_id"]])

predictions = np.concatenate(all_predictions, axis=0)
print(f"Predictions shape: {predictions.shape}")
assert predictions.shape[0] == len(test_df), "Prediction rows must match test rows"
assert (
    predictions.shape[1] == NUM_TARGETS
), "Prediction columns must match number of targets"



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 11
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_path)
required_cols = list(sample_sub.columns)
assert required_cols[0] == "qa_id", "Unexpected sample submission format"

submission = pd.DataFrame(predictions, columns=TARGET_COLS)

qa_id_series = pd.Series(all_qa_ids)
try:
    qa_id_series = qa_id_series.astype(test_df["qa_id"].dtype)
except Exception:
    pass

submission.insert(0, "qa_id", qa_id_series)

for col in TARGET_COLS:
    submission[col] = submission[col].clip(0, 1)

submission = submission[required_cols]
assert submission.shape[0] == test_df.shape[0], "Row count mismatch for submission"
assert (
    list(submission.columns) == required_cols
), "Column mismatch vs sample_submission.csv"

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"{submission_path} created! shape={submission.shape}")
print(submission.head())
